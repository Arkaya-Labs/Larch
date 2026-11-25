//! Larch's desktop app: a native GPUI window over larch-app. Each command is one call into
//! larch-app off the UI thread; its events come back on a channel and redraw the window.
//! `--demo` opens the design's sample data with no engine; `--self-test` checks the core starts.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod context_menu;
mod crew;
mod demo;
mod images;
mod input;
mod kit;
mod logic;
mod mcp;
mod pages;
mod picker;
mod prefs;
mod select;
#[cfg(test)]
mod task_tests;
mod tasks;
mod tasks_model;
mod theme;
mod tour;
mod tree;
mod ui;
mod update;

use std::borrow::Cow;
use std::sync::Arc;

use gpui::{
    App, AppContext, Application, AssetSource, Bounds, Menu, MenuItem, SharedString,
    TitlebarOptions, WindowBounds, WindowOptions, px, size,
};

/// The fonts, brand marks and icons, built into the binary: the release ships one file.
struct Assets;

macro_rules! assets {
    ($($path:literal),* $(,)?) => {
        fn asset(path: &str) -> Option<&'static [u8]> {
            match path {
                $($path => Some(include_bytes!(concat!("../assets/", $path))),)*
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
    "brand/arkaya-mark-small.svg",
    "icons/archive.svg",
    "icons/attach.svg",
    "icons/chat-block.svg",
    "icons/chat.svg",
    "icons/check.svg",
    "icons/clock-block.svg",
    "icons/clock.svg",
    "icons/connect.svg",
    "icons/copy.svg",
    "icons/down.svg",
    "icons/engine.svg",
    "icons/folder.svg",
    "icons/home-block.svg",
    "icons/home.svg",
    "icons/larch-block.svg",
    "icons/larch.svg",
    "icons/memory.svg",
    "icons/models-block.svg",
    "icons/models.svg",
    "icons/more.svg",
    "icons/new-chat-block.svg",
    "icons/new-chat.svg",
    "icons/page-block.svg",
    "icons/page.svg",
    "icons/panel.svg",
    "icons/pin.svg",
    "icons/plus.svg",
    "icons/rename.svg",
    "icons/search.svg",
    "icons/settings-block.svg",
    "icons/settings.svg",
    "icons/shrink.svg",
    "icons/stop.svg",
    "icons/tasks-block.svg",
    "icons/tasks.svg",
    "icons/tools-block.svg",
    "icons/tools.svg",
    "icons/trash.svg",
    "icons/up.svg",
    "icons/x.svg",
);

impl AssetSource for Assets {
    fn load(&self, path: &str) -> gpui::Result<Option<Cow<'static, [u8]>>> {
        // The lotus at the sidebar's foot is knocked out in the sidebar colour: `brand/lotus-RRGGBB.svg`.
        if let Some(hex) = path
            .strip_prefix("brand/lotus-")
            .and_then(|p| p.strip_suffix(".svg"))
        {
            let svg =
                std::str::from_utf8(asset("brand/arkaya-mark-small.svg").unwrap_or_default())?;
            return Ok(Some(Cow::Owned(
                svg.replace("#FAFAF7", &format!("#{hex}")).into_bytes(),
            )));
        }
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

fn core(emit: impl Fn(larch_app::Event) + Send + Sync + 'static) -> Result<larch_app::App, String> {
    let data = larch_app::handoff::data_dir().map_err(|e| e.to_string())?;
    larch_app::App::new(data, larch_app::default_device()?, emit)
}

fn self_test() -> Result<(), String> {
    let app = core(|_| {})?;
    let device = app.app_state().device;
    println!(
        "larch-desktop self-test ok {} device={}",
        env!("CARGO_PKG_VERSION"),
        if device.is_empty() { "cpu" } else { &device }
    );
    Ok(())
}

/// Whether this is Linux under WSL, whose window system is WSLg.
#[cfg(target_os = "linux")]
fn wsl() -> bool {
    std::env::var_os("WSL_DISTRO_NAME").is_some()
        || std::path::Path::new("/proc/sys/fs/binfmt_misc/WSLInterop").exists()
}

/// Under WSL the window goes through WSLg's X server: gpui's Wayland client panics on WSLg's
/// compositor (UnsupportedVersion). That X server's root has no cursor, and gpui never sets the
/// plain arrow on its windows (it takes the arrow as already showing), so the pointer would be
/// invisible over the window wherever no other cursor is asked for: the root gets the arrow, as
/// `xsetroot -cursor_name left_ptr` does on a bare X server. Must run before any thread starts.
#[cfg(target_os = "linux")]
fn wsl_display() {
    if !wsl() || std::env::var_os("DISPLAY").is_none() {
        return;
    }
    // SAFETY: called first in main, before any thread is started.
    unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
    if let Err(e) = root_arrow() {
        eprintln!("cannot give the X root a cursor: {e}");
    }
}

#[cfg(target_os = "linux")]
fn root_arrow() -> Result<(), Box<dyn std::error::Error>> {
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::{ChangeWindowAttributesAux, ConnectionExt as _};
    let (conn, screen) = x11rb::connect(None)?;
    let db = x11rb::resource_manager::new_from_default(&conn)?;
    let cursor = x11rb::cursor::Handle::new(&conn, screen, &db)?
        .reply()?
        .load_cursor(&conn, "left_ptr")?;
    let root = conn.setup().roots[screen].root;
    (conn.change_window_attributes(root, &ChangeWindowAttributesAux::new().cursor(cursor))?)
        .check()?;
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args == ["--self-test"] {
        if let Err(error) = self_test() {
            eprintln!("{}", error.replace(['\r', '\n'], " "));
            std::process::exit(1);
        }
        return;
    }
    #[cfg(target_os = "linux")]
    wsl_display();
    let demo = args.iter().any(|a| a == "--demo");
    // Larch Setup reopens the app with `--setup-error <transaction>` after it undid an update.
    let update_failed = args.iter().any(|a| a == "--setup-error");
    let _install_lock = if demo {
        None
    } else {
        match larch_fetch::hold_install_lock() {
            Ok(lock) => Some(lock),
            Err(error) => {
                eprintln!("cannot hold the install lock: {error}");
                std::process::exit(1);
            }
        }
    };
    // Only a copy setup installed (it holds the install lock) is offered setup's updates.
    let installed = matches!(_install_lock, Some(Some(_)));
    let (tx, rx) = futures::channel::mpsc::unbounded();
    let started = if demo {
        Ok(None)
    } else {
        // The window may be closing; an event nobody receives is not an error.
        core(move |e| drop(tx.unbounded_send(e))).map(|c| Some(Arc::new(c)))
    };
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
            cx.bind_keys(input::bindings());
            cx.bind_keys(ui::bindings());
            cx.bind_keys(tour::bindings());
            // Cmd-Q and the app menu's Quit: a quit runs the quit observers (ui.rs stops the
            // tasks), then the process exits and the engine's GPU memory goes with it.
            cx.on_action(|_: &ui::Quit, cx| cx.quit());
            cx.set_menus(vec![Menu {
                name: "Larch".into(),
                items: vec![MenuItem::action("Quit Larch", ui::Quit)],
            }]);
            let bounds = Bounds::centered(None, size(px(1280.), px(820.)), cx);
            let opened = cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    window_min_size: Some(size(px(880.), px(560.))),
                    titlebar: Some(TitlebarOptions {
                        title: Some("Larch".into()),
                        ..Default::default()
                    }),
                    app_id: Some("com.arkaya-labs.larch".into()),
                    ..Default::default()
                },
                move |window, cx| {
                    cx.new(|cx| ui::Ecl::new(started, installed, update_failed, rx, window, cx))
                },
            );
            if let Err(e) = opened {
                eprintln!("cannot open the window: {e}");
                cx.quit();
            }
            cx.on_window_closed(|cx| cx.quit()).detach();
            cx.activate(true);
        });
}
