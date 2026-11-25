# The llama.cpp baseline (architecture 16, D13): llama-serve-bench plus llama.cpp's own llama-bench,
# llama-batched-bench and llama-server, all from one llama.cpp pin, the same tag as tools/llama-logits.
#   cmake -S tools/llama-serve-bench -B target/llama-serve-bench -DCMAKE_BUILD_TYPE=Release [-DGGML_CUDA=ON | -DGGML_METAL=ON]
#   cmake --build target/llama-serve-bench --config Release --target llama-serve-bench llama-bench llama-batched-bench llama-server
# scripts/baseline.py runs exactly this; -DFETCHCONTENT_SOURCE_DIR_LLAMA=<checkout of the same tag> skips the download.
cmake_minimum_required(VERSION 3.21)
project(larch_llama_serve_bench CXX)

include(FetchContent)
set(LLAMA_TAG b11458)
FetchContent_Declare(llama GIT_REPOSITORY https://github.com/ggml-org/llama.cpp GIT_TAG ${LLAMA_TAG} GIT_SHALLOW TRUE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(llama)

add_executable(llama-serve-bench main.cpp)
target_link_libraries(llama-serve-bench PRIVATE llama llama-common vendor::hash)
# greedy.h's scan is an int max over every logit. Baseline x86-64 has no packed int32 max (SSE4.1), so the scan is
# built for this CPU as ggml is under GGML_NATIVE: 12 us per 152k-token row on wsl5070, 20 without. ARM has it in NEON.
# MSVC is also set for clang-cl.
# ponytail: /arch:AVX2 without ggml's per-CPU check (FindSIMD); add that check when a Windows host without AVX2 runs this.
if(GGML_NATIVE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(MSVC)
        set(SCAN_ARCH /arch:AVX2)
    else()
        set(SCAN_ARCH -march=native)
    endif()
endif()
target_compile_options(llama-serve-bench PRIVATE ${SCAN_ARCH})
# greedy.h against a naive reference, with the driver's options:
#   cmake --build <dir> --target greedy-test && ctest --test-dir <dir> -R greedy
enable_testing()
add_executable(greedy-test greedy_test.cpp)
target_compile_options(greedy-test PRIVATE ${SCAN_ARCH})
add_test(NAME greedy COMMAND greedy-test)
# A shallow checkout has no build number, so the result records the tag itself. `build` is the sha256 of the driver's
# sources, this file and GGML_NATIVE (its compile options on this machine), so a change to its loop, sampling or flags
# shows even under the same llama.cpp pin; editing them reconfigures.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS main.cpp greedy.h)
file(READ main.cpp DRIVER_MAIN)
file(READ greedy.h DRIVER_GREEDY)
file(READ CMakeLists.txt DRIVER_CMAKE)
string(SHA256 DRIVER_BUILD "${DRIVER_MAIN}${DRIVER_GREEDY}${DRIVER_CMAKE}GGML_NATIVE=${GGML_NATIVE}")
target_compile_definitions(llama-serve-bench PRIVATE LLAMA_TAG="${LLAMA_TAG}" DRIVER_BUILD="${DRIVER_BUILD}")
target_compile_features(llama-serve-bench PRIVATE cxx_std_17)
set_target_properties(llama-serve-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
