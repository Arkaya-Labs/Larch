//! The window and taskbar icon on Windows; other platforms take it from the bundle.
fn main() {
    println!("cargo:rerun-if-changed=icons/app.rc");
    println!("cargo:rerun-if-changed=icons/icon.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        embed_resource::compile("icons/app.rc", embed_resource::NONE)
            .manifest_optional()
            .expect("the icon resource compiles");
    }
}
