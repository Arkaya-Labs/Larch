# The llama.cpp baseline (architecture 16, D13): llama-serve-bench plus llama.cpp's own llama-bench,
# llama-batched-bench and llama-server, all from one llama.cpp tag: the pin below (tools/llama-logits' tag) by default.
#   cmake -S tools/llama-serve-bench -B target/llama-serve-bench -DCMAKE_BUILD_TYPE=Release [-DGGML_CUDA=ON | -DGGML_METAL=ON]
#   cmake --build target/llama-serve-bench --config Release --target llama-serve-bench llama-bench llama-batched-bench llama-server
# scripts/baseline.py runs exactly this; -DFETCHCONTENT_SOURCE_DIR_LLAMA=<checkout on a tag> builds that tag instead.
cmake_minimum_required(VERSION 3.21)
project(larch_llama_serve_bench CXX)

include(FetchContent)
set(LLAMA_TAG b11458)
FetchContent_Declare(llama GIT_REPOSITORY https://github.com/ggml-org/llama.cpp GIT_TAG ${LLAMA_TAG} GIT_SHALLOW TRUE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(llama)
# A local checkout (-DFETCHCONTENT_SOURCE_DIR_LLAMA) may be any tag; results record that tag (with "-dirty" for local
# edits), never the pin above. An untagged commit is refused: a ledger row names a release others can rebuild. Revisit
# if a baseline ever has to come from an unreleased commit.
if(FETCHCONTENT_SOURCE_DIR_LLAMA)
    execute_process(COMMAND git -C "${FETCHCONTENT_SOURCE_DIR_LLAMA}" describe --tags --exact-match --dirty
                    OUTPUT_VARIABLE LLAMA_TAG OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    # A worktree's or submodule's .git is a file; its HEAD lives in the git dir git names.
    execute_process(COMMAND git -C "${FETCHCONTENT_SOURCE_DIR_LLAMA}" rev-parse --absolute-git-dir
                    OUTPUT_VARIABLE LLAMA_GIT_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${LLAMA_GIT_DIR}/HEAD")
    # On a branch, a commit moves the branch but not HEAD, so nothing would reconfigure and the tag would go stale.
    execute_process(COMMAND git -C "${FETCHCONTENT_SOURCE_DIR_LLAMA}" symbolic-ref -q HEAD
                    RESULT_VARIABLE LLAMA_ON_BRANCH OUTPUT_QUIET ERROR_QUIET)
    if(LLAMA_ON_BRANCH EQUAL 0)
        message(FATAL_ERROR "${FETCHCONTENT_SOURCE_DIR_LLAMA} is on a branch: check out the tag itself (detached)")
    endif()
endif()

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
