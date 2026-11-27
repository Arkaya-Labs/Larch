//! Inkling 2 Mini (`inkling2_loop`) on the CPU reference, against llama.cpp on the generated model.

use std::path::PathBuf;

use larch_model::tiny;

/// Writes the generated model for a llama.cpp dump into LARCH_INKLING_TINY_OUT:
///   LARCH_INKLING_TINY_OUT=<dir> cargo test -p larch-model --test inkling -- --ignored write_tiny
#[test]
#[ignore = "writes the generated GGUF for a llama.cpp dump: LARCH_INKLING_TINY_OUT"]
fn write_tiny_gguf() {
    let dir = PathBuf::from(std::env::var("LARCH_INKLING_TINY_OUT").unwrap());
    std::fs::write(dir.join("tiny-inkling2.gguf"), tiny::inkling_gguf(7)).unwrap();
}
