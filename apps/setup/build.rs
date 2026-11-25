//! The window and taskbar icon on Windows, the app's own; other platforms take it from the bundle.
fn main() {
    println!("cargo:rerun-if-changed=../desktop/icons/app.rc");
    println!("cargo:rerun-if-changed=../desktop/icons/icon.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        embed_resource::compile("../desktop/icons/app.rc", embed_resource::NONE)
            .manifest_optional()
            .expect("the icon resource compiles");
    }
}
