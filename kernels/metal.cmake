# Metal backend (architecture 6.4), included by CMakeLists.txt when the backend is METAL.
# src/metal/*.mm is Objective-C++ with ARC; src/metal/*.metal is compiled into one larch.metallib
# that is embedded in the library as the byte array larch_metallib / larch_metallib_len.

if(LARCH_EMBED_METALLIB)
  # Script mode (cmake -P): write the metallib as a C++ byte array.
  file(READ "${IN}" hex HEX)
  string(LENGTH "${hex}" hex_len)
  math(EXPR len "${hex_len} / 2")
  string(REGEX REPLACE "(..)" "0x\\1," bytes "${hex}")
  file(WRITE "${OUT}" "#include <cstddef>\n"
    "extern \"C\" const unsigned char ${SYM}[] = {${bytes}};\n"
    "extern \"C\" const std::size_t ${SYM}_len = ${len};\n")
  return()
endif()

# The macOS floor (Metal 4, architecture 6.4) when nothing sets it: standalone builds such as ctest.
# Cargo builds get it from MACOSX_DEPLOYMENT_TARGET in .cargo/config.toml, which CMake reads; keep the
# two in sync.
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET 26.0)
elseif(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 26.0)
  message(FATAL_ERROR "The Metal backend needs macOS 26 (Metal 4), but the deployment target is "
    "${CMAKE_OSX_DEPLOYMENT_TARGET} (MACOSX_DEPLOYMENT_TARGET in the environment overrides .cargo/config.toml).")
endif()
# Xcode 26+ ships the Metal compiler as a separate download, and Command Line Tools lack it.
execute_process(COMMAND xcrun -sdk macosx metal --version RESULT_VARIABLE metal_missing OUTPUT_QUIET ERROR_QUIET)
if(metal_missing)
  message(FATAL_ERROR "The Metal backend needs the Metal toolchain: run `xcodebuild -downloadComponent "
    "MetalToolchain` (needs Xcode), or build with LARCH_BACKEND=CPU.")
endif()

enable_language(OBJCXX)
set(METAL_PERF_SRC "${CMAKE_CURRENT_LIST_DIR}/src/metal/perf.mm")
file(GLOB METAL_MM CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/src/metal/*.mm)
list(REMOVE_ITEM METAL_MM "${METAL_PERF_SRC}")
target_sources(larch_kernels PRIVATE ${METAL_MM})
set_source_files_properties(${METAL_MM} "${METAL_PERF_SRC}" PROPERTIES COMPILE_OPTIONS "-fobjc-arc;-Wall;-Wextra;-Werror")
target_link_libraries(larch_kernels PUBLIC
  "-framework Metal" "-framework Foundation" "-framework IOKit")

file(GLOB METAL_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/src/metal/*.metal)
# MLX's NAX attention hardcodes macOS 26.2's cooperative-tensor layout (wrong output built for 26.0, M5, 11 Nov),
# so it is its own metallib for 26.2, which the backend loads only there.
set(METAL_NAX_SRC "${CMAKE_CURRENT_LIST_DIR}/src/metal/attn_full_nax.metal")
list(REMOVE_ITEM METAL_SRC "${METAL_NAX_SRC}")
set(nax_target ${CMAKE_OSX_DEPLOYMENT_TARGET})
if(nax_target VERSION_LESS 26.2)
  set(nax_target 26.2)
endif()
file(GLOB_RECURSE METAL_HDR CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/src/metal/*.h)  # with the vendored mlx/
set(metal_dir "${CMAKE_CURRENT_BINARY_DIR}/metal")
set(air_files "")
foreach(src IN LISTS METAL_SRC METAL_NAX_SRC)
  get_filename_component(name "${src}" NAME_WE)
  set(target ${CMAKE_OSX_DEPLOYMENT_TARGET})
  if(src STREQUAL METAL_NAX_SRC)
    set(target ${nax_target})
  endif()
  set(air "${metal_dir}/${name}.air")
  add_custom_command(OUTPUT "${air}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${metal_dir}"
    # safe math: the kernels compare and exp -inf (attention, top-k), which fast math assumes away.
    # ponytail: safe for every kernel. relaxed keeps -inf, but its x / repeat_penalty is 1 ulp off where the
    # op table wants bitwise; measure relaxed (with precise::divide there) for mmvq and gemm in the perf loop.
    COMMAND xcrun -sdk macosx metal -O3 -std=metal4.0 -mtargetos=macos${target} -Wall -Werror
            -fmetal-math-mode=safe -I "${CMAKE_CURRENT_LIST_DIR}/src/metal" -c "${src}" -o "${air}"
    DEPENDS "${src}" ${METAL_HDR} VERBATIM)
  if(NOT src STREQUAL METAL_NAX_SRC)
    list(APPEND air_files "${air}")
  endif()
endforeach()
foreach(lib larch larch_nax)
  set(airs ${air_files})
  if(lib STREQUAL larch_nax)
    set(airs "${metal_dir}/attn_full_nax.air")
  endif()
  add_custom_command(OUTPUT "${metal_dir}/${lib}.metallib"
    COMMAND xcrun -sdk macosx metallib ${airs} -o "${metal_dir}/${lib}.metallib"
    DEPENDS ${airs} VERBATIM)
  add_custom_command(OUTPUT "${metal_dir}/${lib}_metallib.cpp"
    COMMAND "${CMAKE_COMMAND}" -DLARCH_EMBED_METALLIB=ON "-DIN=${metal_dir}/${lib}.metallib"
            "-DOUT=${metal_dir}/${lib}_metallib.cpp" "-DSYM=${lib}_metallib" -P "${CMAKE_CURRENT_LIST_FILE}"
    DEPENDS "${metal_dir}/${lib}.metallib" VERBATIM)
  target_sources(larch_kernels PRIVATE "${metal_dir}/${lib}_metallib.cpp")
endforeach()

# The plumbing smoke test and every op against the CPU reference, under the Metal API validation layer;
# the op comparison also under shader validation, which faults an out-of-bounds access.
if(LARCH_BUILD_TESTS)
  enable_testing()
  foreach(name smoke compare)
    add_executable(metal_${name} ${CMAKE_CURRENT_LIST_DIR}/src/metal/${name}.cpp)
    target_link_libraries(metal_${name} PRIVATE larch_kernels)
  endforeach()
  add_executable(metal_perf "${METAL_PERF_SRC}")  # manual timing tool, run under flock ~/src/gpu.lock
  target_link_libraries(metal_perf PRIVATE larch_kernels)
  add_test(NAME metal_smoke COMMAND "${CMAKE_COMMAND}" -E env
    "MTL_DEBUG_LAYER=1" "$<TARGET_FILE:metal_smoke>")
  # Keep the unforced comparison independent of the invoking shell's tensor override.
  add_test(NAME metal_compare COMMAND "${CMAKE_COMMAND}" -E env
    --unset=LARCH_METAL_TENSOR "MTL_DEBUG_LAYER=1" "MTL_SHADER_VALIDATION=1" "$<TARGET_FILE:metal_compare>")
  # The staged decode tile (metal_perf st): each format's rows at 2 to 32 rows against each row alone, across a K split
  # (4 parts) and a ragged column tile (n 80), and each row against larch_mmvq to 1e-2. 77 without Apple GPU family 10.
  add_test(NAME metal_staged COMMAND "${CMAKE_COMMAND}" -E env "MTL_DEBUG_LAYER=1" "MTL_SHADER_VALIDATION=1"
    "$<TARGET_FILE:metal_perf>" st --device 0 --format all --n 80 --k 2048 --iterations 1)
  set_tests_properties(metal_staged PROPERTIES SKIP_RETURN_CODE 77)
  # ops on the plain 64 x 32 Q3_K expert tile, which the default's register-prefetch tile replaces.
  add_test(NAME ops_moe_plain_tile COMMAND "${CMAKE_COMMAND}" -E env "LARCH_METAL_MOE_PREFETCH=0" "$<TARGET_FILE:ops>")

  # Separate directory scopes deliberately retain the existing exact test name, so CI's Metal exclusion
  # continues to cover all four GPU comparisons while each forced process receives its real environment.
  # TENSOR=0 runs the simdgroup-matrix kernels; SIMD_KERNELS=0 too keeps the paths a GPU falls back to when they
  # fail to build: dequant + vv_gemm, mmvq_id_grouped and vv_attn prefill.
  set(LARCH_METAL_TENSOR_TEST_VALUE 0)
  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/src/metal" "${CMAKE_CURRENT_BINARY_DIR}/metal-compare-simd")
  set(LARCH_METAL_SIMD_KERNELS_TEST_VALUE 0)
  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/src/metal" "${CMAKE_CURRENT_BINARY_DIR}/metal-compare-plain")
  unset(LARCH_METAL_SIMD_KERNELS_TEST_VALUE)
  set(LARCH_METAL_TENSOR_TEST_VALUE 1)
  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/src/metal" "${CMAKE_CURRENT_BINARY_DIR}/metal-compare-tensor")
  unset(LARCH_METAL_TENSOR_TEST_VALUE)
endif()
