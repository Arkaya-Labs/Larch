# llama.cpp reference tools (architecture 15), against llama.cpp pinned to a release tag. CPU-only is
# enough for tokenisation, templates and the CPU oracle dump:
#   cmake -S tools/llama-logits -B target/llama-tools -DGGML_NATIVE=OFF && cmake --build target/llama-tools
# -DFETCHCONTENT_SOURCE_DIR_LLAMA=<checkout of the same tag> skips the download; -DGGML_CUDA=ON (Metal is
# on by default on macOS) builds the GPU dumps.
cmake_minimum_required(VERSION 3.21)
project(larch_llama_tools CXX)

# -DLLAMA_TAG=<tag> builds against another release (the Inkling 2 dumps take b12723, whose llama.cpp runs
# inkling2_loop); the dump header records the tag and commit that ran.
set(LLAMA_TAG b11458 CACHE STRING "llama.cpp release tag")
include(FetchContent)
FetchContent_Declare(llama GIT_REPOSITORY https://github.com/ggml-org/llama.cpp GIT_TAG ${LLAMA_TAG} GIT_SHALLOW TRUE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
# llama.cpp adds a BLAS backend whenever one is built (the macOS default), and it would take a "CPU"
# dump's matmuls; off, a CPU dump is ggml's CPU backend alone. That still differs by machine (SIMD,
# Accelerate's vector ops), so the oracle's CPU reference is one machine's dump.
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
# libmtmd, for llama-logits --mmproj (the image oracle).
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(llama)

add_executable(tokdump tokdump.cpp)
target_link_libraries(tokdump PRIVATE llama llama-common)

add_executable(llama-logits logits.cpp)
target_link_libraries(llama-logits PRIVATE llama llama-common mtmd vendor::hash)
target_compile_definitions(llama-logits PRIVATE LLAMA_TAG="${LLAMA_TAG}")
