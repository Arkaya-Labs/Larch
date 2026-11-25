//! Larch Setup: the installer window, in the app's own look (its theme and widget pieces are
//! shared file for file with apps/desktop). It runs this same program as larch-setup in a child
//! (engine.rs), so one download installs, and the lock, journal and recovery stay in larch-setup.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod engine;
mod screens;
// The installer uses part of each; the rest is the app's.
#[allow(dead_code)]
#[path = "../../desktop/src/kit.rs"]
mod kit;
#[allow(dead_code)]
#[path = "../../desktop/src/theme.rs"]
mod theme;

use std::borrow::Cow;

use gpui::{
    App, AppContext, Application, AssetSource, Bounds, SharedString, TitlebarOptions, WindowBounds,
    WindowOptions, px, size,
};

/// The fonts and icons the window uses, from the app's assets, built into the binary.
struct Assets;

macro_rules! assets {
    ($($path:literal),* $(,)?) => {
        fn asset(path: &str) -> Option<&'static [u8]> {
            match path {
                $($path => Some(include_bytes!(concat!("../../desktop/assets/", $path))),)*
                _ => None,
            }
        }
        const FILES: &[&str] = &[$($path),*];
    };
}

assets!(
    "fonts/Figtree-Regular.ttf",
    "fonts/Figtree-SemiBold.ttf",
    "fonts/JetBrainsMono-Regular.ttf",
    "fonts/Fraunces-Medium.ttf",
    "icons/check.svg",
    "icons/down.svg",
    "icons/gpu-card-block.svg",
    "icons/gpu-card.svg",
    "icons/gpu-chip-block.svg",
    "icons/gpu-chip.svg",
    "icons/larch-block.svg",
    "icons/larch.svg",
    "icons/plus.svg",
    "icons/settings.svg",
    "icons/trash.svg",
    "icons/x.svg",
);

impl AssetSource for Assets {
    fn load(&self, path: &str) -> gpui::Result<Option<Cow<'static, [u8]>>> {
        Ok(asset(path).map(Cow::Borrowed))
    }

    fn list(&self, path: &str) -> gpui::Result<Vec<SharedString>> {
        Ok(FILES
            .iter()
            .filter(|p| p.starts_with(path))
            .map(|p| (*p).into())
            .collect())
    }
}

/// The temporary copy's name in %TEMP%, so only this copy deletes itself.
#[cfg(windows)]
const COPY: &str = "larch-setup-window-";

/// Windows cannot remove a running program's file, so the window installed in setup's folder
/// hands over to a copy in %TEMP%; true when it did and this process should exit.
#[cfg(windows)]
fn relaunched() -> std::io::Result<bool> {
    let exe = std::fs::canonicalize(std::env::current_exe()?)?;
    let installed = larch_fetch::install_root()
        .ok()
        .and_then(|root| std::fs::canonicalize(root).ok())
        .is_some_and(|root| exe.starts_with(root));
    if !installed {
        return Ok(false);
    }
    let copy = std::env::temp_dir().join(format!("{COPY}{}.exe", std::process::id()));
    std::fs::copy(&exe, &copy)?;
    std::process::Command::new(copy)
        .args(std::env::args_os().skip(1))
        .spawn()?;
    Ok(true)
}

/// The folder `larch-setup window` unpacks this window into, in the temporary directory.
const WINDOW_DIR: &str = "larch-installer-";

/// Deletes what this window left in the temporary directory once it exits: the copy it handed over
/// to on Windows, or the folder `larch-setup window` (the install command) unpacked it into.
fn remove_temporary() -> std::io::Result<()> {
    let exe = std::fs::canonicalize(std::env::current_exe()?)?;
    // Canonical on both sides: %TEMP% can be an 8.3 short path.
    let temp = std::fs::canonicalize(std::env::temp_dir())?;
    // From the folder up: the program itself is never the folder to remove.
    let folder = (exe.parent().into_iter().flat_map(|p| p.ancestors()))
        .find(|a| a.parent() == Some(temp.as_path()) && named(a, WINDOW_DIR));
    #[cfg(windows)]
    let target = folder
        .or((exe.parent() == Some(temp.as_path()) && named(&exe, COPY)).then_some(exe.as_path()));
    #[cfg(not(windows))]
    let target = folder;
    target.map_or(Ok(()), remove_after_exit)
}

/// Whether a path's last part starts with `prefix`.
fn named(path: &std::path::Path, prefix: &str) -> bool {
    path.file_name()
        .is_some_and(|n| n.to_string_lossy().starts_with(prefix))
}

/// A running program's files cannot be removed on Windows, so a detached shell removes them after
/// this process has gone, as larch-setup does with its own run copy.
#[cfg(windows)]
fn remove_after_exit(path: &std::path::Path) -> std::io::Result<()> {
    use std::os::windows::process::CommandExt;
    use std::process::{Command, Stdio};
    let script = if path.is_dir() {
        "ping -n 3 127.0.0.1 >nul & rmdir /s /q \"%LARCH_RUN_FILE%\""
    } else {
        "ping -n 3 127.0.0.1 >nul & del /q \"%LARCH_RUN_FILE%\""
    };
    Command::new("cmd.exe")
        .args(["/d", "/v:off", "/c", script])
        .env("LARCH_RUN_FILE", path)
        // DETACHED_PROCESS | CREATE_NO_WINDOW
        .creation_flags(0x0000_0008 | 0x0800_0000)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    Ok(())
}

/// Elsewhere a running program's files can be removed while it runs.
#[cfg(not(windows))]
fn remove_after_exit(path: &std::path::Path) -> std::io::Result<()> {
    std::fs::remove_dir_all(path)
}

fn main() {
    if std::env::var_os(engine::ENGINE).is_some() {
        return larch_setup::cli::main();
    }
    // A failed copy runs in place, where removing setup then reports the window as in use.
    #[cfg(windows)]
    if matches!(relaunched(), Ok(true)) {
        return;
    }
    // The system's Uninstall entry opens the window with --uninstall: it then starts there.
    let uninstalling = std::env::args().any(|a| a == "--uninstall");
    Application::new()
        .with_assets(Assets)
        .run(move |cx: &mut App| {
            for f in FILES.iter().filter(|f| f.starts_with("fonts/")) {
                if let Err(e) = cx
                    .text_system()
                    .add_fonts(vec![Cow::Borrowed(asset(f).unwrap_or_default())])
                {
                    eprintln!("cannot load {f}: {e}");
                }
            }
            let bounds = Bounds::centered(None, size(px(880.), px(680.)), cx);
            let opened = cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    window_min_size: Some(size(px(720.), px(560.))),
                    titlebar: Some(TitlebarOptions {
                        title: Some("Larch Setup".into()),
                        ..Default::default()
                    }),
                    app_id: Some("com.arkaya-labs.larch.setup".into()),
                    ..Default::default()
                },
                move |window, cx| cx.new(|cx| screens::Setup::new(uninstalling, window, cx)),
            );
            if let Err(e) = opened {
                eprintln!("cannot open the window: {e}");
                cx.quit();
            }
            cx.on_window_closed(|cx| cx.quit()).detach();
            cx.on_app_quit(|_| {
                engine::stop_all();
                // Best effort: what is left behind is only temporary files.
                let _ = remove_temporary();
                async {}
            })
            .detach();
            cx.activate(true);
        });
}
