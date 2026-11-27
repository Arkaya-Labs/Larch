cmake_minimum_required(VERSION 3.24)

if(LARCH_HIPIFY)
  # Script mode (cmake -P), one file: hipify IN to OUT and fail on any call hipify cannot convert
  # (architecture 6.3: zero unconverted calls). hipify-perl reports each one as a "warning:" line.
  get_filename_component(out_dir "${OUT}" DIRECTORY)
  file(MAKE_DIRECTORY "${out_dir}")
  # Windows runs hipify-perl through perl (PERL set); elsewhere it runs on its own.
  execute_process(COMMAND ${PERL} "${HIPIFY}" -print-stats "-o=${OUT}" "${IN}"
    RESULT_VARIABLE failed OUTPUT_VARIABLE log ERROR_VARIABLE log)
  if(failed OR log MATCHES "warning:")
    file(REMOVE "${OUT}")
    message(FATAL_ERROR "hipify-perl ${IN}:\n${log}")
  endif()
  message(STATUS "hipify-perl ${IN}: ${log}")
  return()
endif()

# The Windows HIP SDK sets HIP_PATH (C:/Program Files/AMD/ROCm/<version>) instead of ROCM_PATH. Its clang
# takes only GNU-style flags and CMake will not mix that with cl.exe, so C and C++ use it too.
if(WIN32 AND LARCH_BACKEND STREQUAL "HIP" AND NOT CMAKE_HIP_COMPILER AND NOT DEFINED ENV{HIPCXX}
   AND DEFINED ENV{HIP_PATH})
  file(TO_CMAKE_PATH "$ENV{HIP_PATH}" hip_path)
  set(CMAKE_C_COMPILER "${hip_path}/bin/clang.exe")
  set(CMAKE_CXX_COMPILER "${hip_path}/bin/clang++.exe")
  set(CMAKE_HIP_COMPILER "${hip_path}/bin/clang++.exe")
  list(APPEND CMAKE_PREFIX_PATH "${hip_path}") # hipblas-config finds hip and rocblas without hints
endif()

project(larch_kernels LANGUAGES C CXX)

set(LARCH_BACKEND "AUTO" CACHE STRING "AUTO, CUDA, HIP, METAL or CPU")
set_property(CACHE LARCH_BACKEND PROPERTY STRINGS AUTO CUDA HIP METAL CPU)
option(LARCH_BUILD_TESTS "Build tests/*.cpp" ${PROJECT_IS_TOP_LEVEL})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# larch-sys passes Rust's runtime; a plain CMake build takes the static one that ships (nvcc's host
# compiler follows it too).
if(NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# AUTO picks Metal on Apple, else CUDA if nvcc is found, else CPU.
set(BACKEND ${LARCH_BACKEND})
if(BACKEND STREQUAL "AUTO")
  if(APPLE)
    set(BACKEND METAL)
  else()
    include(CheckLanguage)
    if(NOT CMAKE_CUDA_COMPILER)
      unset(CMAKE_CUDA_COMPILER CACHE) # retry if an earlier configure found no nvcc
    endif()
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      set(BACKEND CUDA)
    else()
      set(BACKEND CPU)
    endif()
  endif()
elseif(NOT BACKEND MATCHES "^(CUDA|HIP|METAL|CPU)$")
  message(FATAL_ERROR "LARCH_BACKEND=${LARCH_BACKEND}; expected AUTO, CUDA, HIP, METAL or CPU")
endif()
message(STATUS "larch backend: ${BACKEND}")

# The fast CPU backend (lane c1) names each source by the ISA it needs: fast_*_avx2.cpp and
# fast_*_avx512.cpp get those flags on x86-64, fast_*_neon.cpp builds on ARM64 (its baseline), and the
# other ISAs' files are left out. The compiler's target decides, not CMAKE_SYSTEM_PROCESSOR, which names
# the host in a cross build. Every other file, the CPU reference included, keeps the default flags, so an
# ISA file must share no inline code with them: the linker keeps one copy of an inline function or
# template instance, and an AVX2 copy would reach the reference on a CPU without AVX2. So everything but
# its entry points goes in an anonymous namespace, it includes no repo header with inline functions, and
# callers check the CPU at run time before they enter it.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("#if !defined(__x86_64__) && !defined(_M_X64)\n#error\n#endif\nint main() {}"
  LARCH_X86_64)
check_cxx_source_compiles("#if !defined(__aarch64__) && !defined(_M_ARM64)\n#error\n#endif\nint main() {}"
  LARCH_ARM64)
if(MSVC)
  set(LARCH_ISA_avx2 /arch:AVX2)
  set(LARCH_ISA_avx512 /arch:AVX512) # F, CD, BW, DQ and VL
else()
  set(LARCH_ISA_avx2 -mavx2 -mfma -mf16c)
  set(LARCH_ISA_avx512 -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mfma -mf16c)
endif()
function(larch_isa_sources var)
  set(files ${${var}})
  if(LARCH_X86_64)
    list(FILTER files EXCLUDE REGEX "/fast_[^/]*_neon[.]cpp$")
    foreach(isa avx2 avx512)
      set(isa_files ${files})
      list(FILTER isa_files INCLUDE REGEX "/fast_[^/]*_${isa}[.]cpp$")
      set_source_files_properties(${isa_files} PROPERTIES COMPILE_OPTIONS "${LARCH_ISA_${isa}}")
    endforeach()
  elseif(LARCH_ARM64)
    list(FILTER files EXCLUDE REGEX "/fast_[^/]*_avx(2|512)[.]cpp$")
  else()
    list(FILTER files EXCLUDE REGEX "/fast_[^/]*_(avx2|avx512|neon)[.]cpp$")
  endif()
  set(${var} ${files} PARENT_SCOPE)
endfunction()

# The dispatch layer (src/common) and the CPU reference (src/cpu) are always compiled, plus at most
# one GPU backend.
add_library(larch_kernels STATIC)
target_include_directories(larch_kernels PUBLIC include PRIVATE src)
file(GLOB COMMON_SRC CONFIGURE_DEPENDS src/*.cpp src/common/*.cpp src/cpu/*.cpp)
larch_isa_sources(COMMON_SRC)
target_sources(larch_kernels PRIVATE ${COMMON_SRC})
set(LINK_INFO "")

if(BACKEND STREQUAL "CUDA")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86-real;89-real;120-real;120-virtual") # RTX 3050, 4060, 5070 Ti + PTX
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  # Target properties: the target exists already, so the CMAKE_CUDA_* defaults would not reach it.
  set_target_properties(larch_kernels PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CUDA_STANDARD 20 CUDA_RUNTIME_LIBRARY Static)
  file(GLOB CUDA_SRC CONFIGURE_DEPENDS src/cuda/*.cu)
  target_sources(larch_kernels PRIVATE ${CUDA_SRC})
  # nvcc builds the arches one after another on one core; quant.cu took 19 min that way, 7.7 with all at once.
  # Low-memory machines set NVCC_APPEND_FLAGS=--threads=1 (the last --threads wins), as the hosted CUDA build in backend-builds.yml does.
  target_compile_options(larch_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--threads=0>)
  target_link_libraries(larch_kernels PUBLIC CUDA::cudart_static CUDA::cublas)
  target_compile_definitions(larch_kernels PRIVATE LARCH_BACKEND_IS_CUDA=1)
  set(LINK_INFO "cuda_lib_dir=${CUDAToolkit_LIBRARY_DIR}\n")
elseif(BACKEND STREQUAL "HIP")
  # One source tree (architecture 6.3): src/cuda/*.cu and the GPU headers of src/common are hipified
  # into build/hip, which mirrors src and comes first on the include path. Never committed.
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201") # RDNA 2, 3, 4
    if(WIN32)
      # Windows has no HSA_OVERRIDE_GFX_VERSION, so the RX 6600/6700 family (gfx1031, gfx1032) runs only
      # code built for it. AMD lists neither for Windows: best effort, not a supported target.
      list(APPEND CMAKE_HIP_ARCHITECTURES gfx1031 gfx1032)
    endif()
  endif()
  enable_language(HIP)
  find_package(hipblas CONFIG REQUIRED HINTS "${CMAKE_HIP_COMPILER_ROCM_ROOT}" "$ENV{ROCM_PATH}" "$ENV{HIP_PATH}" /opt/rocm)
  set_target_properties(larch_kernels PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}" HIP_STANDARD 20)
  # backend.cu counts only the GPUs this list has code for (installer E2).
  string(REPLACE ";" "," hip_archs "${CMAKE_HIP_ARCHITECTURES}")
  target_compile_definitions(larch_kernels PRIVATE "LARCH_HIP_ARCHS=\"${hip_archs}\"")
  # The Windows HIP SDK (7.x) ships hipify-perl in libexec/hipify, not bin.
  find_program(HIPIFY_PERL hipify-perl HINTS "$ENV{ROCM_PATH}/bin" "$ENV{HIP_PATH}/bin"
    "$ENV{HIP_PATH}/libexec/hipify" /opt/rocm/bin REQUIRED)
  set(hipify_perl "")
  if(WIN32)
    # hipify-perl is a script with no extension, which Windows cannot run without perl.
    find_package(Perl REQUIRED)
    set(hipify_perl "-DPERL=${PERL_EXECUTABLE}")
  endif()
  file(GLOB GPU_SRC CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    src/cuda/*.cu src/cuda/*.cuh src/common/*.cuh src/common/blas.h src/common/graph.h)
  set(hip_dir "${CMAKE_CURRENT_BINARY_DIR}/hip")
  set(hip_out "")
  foreach(rel IN LISTS GPU_SRC)
    string(REGEX REPLACE "[.]cu$" ".hip" out "${hip_dir}/${rel}")
    list(APPEND hip_out "${out}")
    add_custom_command(OUTPUT "${out}"
      COMMAND "${CMAKE_COMMAND}" -DLARCH_HIPIFY=ON "-DHIPIFY=${HIPIFY_PERL}" ${hipify_perl}
              "-DIN=${CMAKE_CURRENT_SOURCE_DIR}/src/${rel}" "-DOUT=${out}" -P "${CMAKE_CURRENT_LIST_FILE}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/${rel}" VERBATIM)
    target_sources(larch_kernels PRIVATE "${out}")
  endforeach()
  # A renamed or deleted source must not leave a stale copy. Only those go: removing every copy
  # would rehipify and recompile all kernels on each configure, and the CI runners keep this build.
  file(GLOB_RECURSE stale "${hip_dir}/*")
  list(REMOVE_ITEM stale ${hip_out})
  if(stale)
    file(REMOVE ${stale})
  endif()
  get_filename_component(rocm "${hipblas_DIR}/../../.." ABSOLUTE) # <rocm>/lib/cmake/hipblas
  # src/common resolves a hipified header's bare include of a sibling that is not hipified
  # (backend.hpp). hipify writes <hipblas.h>; ROCm installs it as <hipblas/hipblas.h>.
  target_include_directories(larch_kernels BEFORE PRIVATE "${hip_dir}" src/common "${rocm}/include/hipblas")
  target_link_libraries(larch_kernels PUBLIC roc::hipblas hip::host)
  if(hipblas_VERSION VERSION_GREATER_EQUAL 3)
    # hipify-perl still writes hipBLAS 2's _v2 names; hipBLAS 3 dropped them and its plain names are the v2 API.
    target_compile_definitions(larch_kernels PRIVATE hipblasGemmEx_v2=hipblasGemmEx)
  endif()
  target_compile_definitions(larch_kernels PRIVATE LARCH_BACKEND_IS_HIP=1)
  set(LINK_INFO "hip_lib_dir=${rocm}/lib\n")
  if(WIN32)
    # clang's __builtin_cpu_supports needs its builtins library, which link.exe (Rust's linker) does not add.
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" --rtlib=compiler-rt --print-libgcc-file-name
      OUTPUT_VARIABLE clang_rt OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    string(APPEND LINK_INFO "clang_rt=${clang_rt}\n")
  endif()
elseif(BACKEND STREQUAL "METAL")
  include(metal.cmake) # owned by the Metal lane
  target_compile_definitions(larch_kernels PRIVATE LARCH_BACKEND_IS_METAL=1)
endif()
string(TOLOWER "${BACKEND}" backend_name)

if(MSVC)
  # /EHsc: cmake-rs replaces CMAKE_CXX_FLAGS, which drops MSVC's default exception model.
  target_compile_options(larch_kernels PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX /EHsc>)
  # larch-sys builds RelWithDebInfo, whose MSVC /Ob1 inlines only what is marked inline: the fast CPU
  # kernels' helpers stay calls, and fast_mmvq_id's q4_k decode step ran 2.1x slower (Core Ultra 7 155U).
  string(REPLACE "/Ob1" "/Ob2" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
else()
  target_compile_options(larch_kernels PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>)
endif()

# Read by crates/larch-sys/build.rs.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/larch_link_info.txt" "backend=${backend_name}\n${LINK_INFO}")
install(TARGETS larch_kernels ARCHIVE DESTINATION lib)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/larch_link_info.txt" DESTINATION share/larch)

# Each tests/*.cpp becomes its own ctest, and so does contract.cpp, which sits outside tests/ so the
# owner of larch.h, not a backend lane, owns it. Every test links the CPU reference.
if(LARCH_BUILD_TESTS)
  enable_testing()
  file(GLOB TESTS CONFIGURE_DEPENDS contract.cpp tests/*.cpp)
  larch_isa_sources(TESTS)
  foreach(src IN LISTS TESTS)
    get_filename_component(name "${src}" NAME_WE)
    add_executable(${name} "${src}")
    target_link_libraries(${name} PRIVATE larch_kernels)
    add_test(NAME ${name} COMMAND ${name})
  endforeach()
  # fast_mmvq_id runs in ~10 s on the CPU build; a lost pool wake hangs it, which must fail, not stall CI.
  set_tests_properties(fast_mmvq_id PROPERTIES TIMEOUT 120)
  # verify skips (77) where GPU 0 is absent or neither CUDA nor Metal with the staged tile: HIP has no mmvq_verify.
  set_tests_properties(verify PROPERTIES SKIP_RETURN_CODE 77)
  # quant.cu routes Q4_K, Q6_K and Q3_K mmvq (one row, routed rows) to half2 on sm_86/89 unless LARCH_DECODE_H2=0;
  # verify pins the half2 decoder's bytes, verify_f32 the f32 decoder's (Q3_K's row prints its hash and pins nothing
  # yet), and both run the whole gate on their path.
  add_test(NAME verify_f32 COMMAND verify)
  set_tests_properties(verify_f32 PROPERTIES ENVIRONMENT LARCH_DECODE_H2=0 SKIP_RETURN_CODE 77)
  # ops again on the CPU device the engine computes on (-2), whose fast slots replace the reference's.
  add_test(NAME ops_cpu_device COMMAND ops)
  set_tests_properties(ops_cpu_device PROPERTIES ENVIRONMENT LARCH_TEST_DEVICE=-2)
  # fused on the CPU device too, whose fast mmvq_id the fused sequence must run as well.
  add_test(NAME fused_cpu_device COMMAND fused)
  set_tests_properties(fused_cpu_device PROPERTIES ENVIRONMENT LARCH_TEST_DEVICE=-2)
  # gdn again with its half on GPU 0 (77 where that is not CUDA, so Metal's entries go untested here).
  add_test(NAME gdn_gpu COMMAND gdn)
  set_tests_properties(gdn_gpu PROPERTIES ENVIRONMENT LARCH_TEST_GDN_GPU=1 SKIP_RETURN_CODE 77)
  # inkling again with its GPU half (77 without a GPU or where GPU 0 lacks an entry); inkling_shapes checks GPU 0
  # against the CPU reference at Inkling 2 Mini's shapes (77 likewise).
  add_test(NAME inkling_gpu COMMAND inkling)
  set_tests_properties(inkling_gpu PROPERTIES ENVIRONMENT LARCH_TEST_INKLING_GPU=1 SKIP_RETURN_CODE 77)
  set_tests_properties(inkling_shapes PROPERTIES SKIP_RETURN_CODE 77)
  if(BACKEND STREQUAL "CUDA")
    # ops.cu picks the tensor-core prefill and decode from LARCH_PREFILL_TC and LARCH_DECODE_TC once, at library
    # load; unset they are the default on sm_120, so `attention` pins both CUDA-core dispatches, attention_prefill_tc
    # and attention_decode_tc force one tensor-core kernel on and the other off (77 without a GPU) and
    # attention_default runs what ships, whatever the shell sets (77 where that is the CUDA-core dispatch `attention`
    # already runs).
    set_tests_properties(attention PROPERTIES ENVIRONMENT "LARCH_PREFILL_TC=0;LARCH_DECODE_TC=0")
    add_test(NAME attention_prefill_tc COMMAND attention)
    set_tests_properties(attention_prefill_tc PROPERTIES
      ENVIRONMENT "LARCH_PREFILL_TC=1;LARCH_DECODE_TC=0" SKIP_RETURN_CODE 77)
    add_test(NAME attention_decode_tc COMMAND attention)
    set_tests_properties(attention_decode_tc PROPERTIES
      ENVIRONMENT "LARCH_DECODE_TC=1;LARCH_PREFILL_TC=0" SKIP_RETURN_CODE 77)
    add_test(NAME attention_default COMMAND attention)
    set_tests_properties(attention_default PROPERTIES
      ENVIRONMENT_MODIFICATION "LARCH_PREFILL_TC=unset:;LARCH_DECODE_TC=unset:" SKIP_RETURN_CODE 77)
    # ops runs attn_full on its default route, the tensor-core kernel from sm_80; ops_attn_full_scalar
    # forces the CUDA-core kernel (HIP's route, and the fallback for rows not 16-byte aligned).
    add_test(NAME ops_attn_full_scalar COMMAND ops)
    set_tests_properties(ops_attn_full_scalar PROPERTIES ENVIRONMENT LARCH_ATTN_FULL_TC=0)
  endif()
endif()
