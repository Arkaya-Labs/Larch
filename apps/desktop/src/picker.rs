//! The system file picker, for "+ Attach", "Work here" and Library's "Add".
//!
//! gpui's Linux picker is xdg-desktop-portal's FileChooser, which WSL and many plain window managers
//! do not run. There gpui answers with an error, which the buttons used to drop, so they did
//! nothing at all. Now zenity or kdialog stands in when one is installed, and with neither a toast
//! says what to install.

use std::path::PathBuf;
use std::process::Command;

use gpui::{Context, PathPromptOptions};

use crate::ui::Ecl;

/// What the toast adds to gpui's error when no picker opened.
const INSTALL: &str = "Install xdg-desktop-portal-gtk or zenity to choose files.";

/// Asks for paths with the system's picker and hands a non-empty pick to `then`. A cancel does
/// nothing; a picker that cannot open says why in a toast, then hands `then` `None`.
pub fn pick(
    cx: &mut Context<Ecl>,
    options: PathPromptOptions,
    then: impl FnOnce(&mut Ecl, Option<Vec<PathBuf>>, &mut Context<Ecl>) + 'static,
) {
    let options = honoured(options, cx.can_select_mixed_files_and_dirs());
    let rx = cx.prompt_for_paths(options.clone());
    cx.spawn(async move |this, cx| {
        let picked = match rx.await {
            Ok(Ok(paths)) => Ok(paths),
            Ok(Err(e)) if cfg!(target_os = "linux") => {
                let stand_in = cx
                    .background_executor()
                    .spawn(async move { fallback(&options) });
                stand_in
                    .await
                    .unwrap_or_else(|| Err(format!("{e} {INSTALL}")))
            }
            Ok(Err(e)) => Err(e.to_string()),
            // The platform dropped the request without an answer, as on a closed window.
            Err(_) => Ok(None),
        };
        this.update(cx, |s, cx| {
            match picked {
                Ok(Some(paths)) if !paths.is_empty() => then(s, Some(paths), cx),
                Ok(_) => return,
                Err(e) => {
                    s.say(format!("The system file picker did not open: {e}"));
                    then(s, None, cx);
                }
            }
            cx.notify()
        })
        .ok();
    })
    .detach();
}

/// The options the platform can honour. A portal picks files or folders, never both, and gpui asks
/// it for folders when both are wanted, which hid every PDF and image "+ Attach" is for; so where
/// both cannot be mixed (`mixed`), files win.
fn honoured(mut options: PathPromptOptions, mixed: bool) -> PathPromptOptions {
    if options.files && options.directories && !mixed {
        options.directories = false;
    }
    options
}

/// Picks with zenity, else kdialog: `None` when neither is installed, `Ok(None)` on a cancel.
fn fallback(options: &PathPromptOptions) -> Option<Result<Option<Vec<PathBuf>>, String>> {
    for (program, args) in commands(options) {
        let Ok(out) = Command::new(program).args(&args).output() else {
            continue; // not installed
        };
        return Some(match out.status.code() {
            Some(0) => Ok(Some(paths(&out.stdout))),
            Some(1) => Ok(None), // both exit 1 on Cancel or a closed dialog
            _ => Err(format!(
                "{program} could not open: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )),
        });
    }
    None
}

/// The zenity and kdialog command lines for `options`, one path per output line.
fn commands(options: &PathPromptOptions) -> [(&'static str, Vec<String>); 2] {
    let folders = options.directories && !options.files;
    let title = options
        .prompt
        .as_ref()
        .map_or("Choose", |p| p.as_ref())
        .to_owned();
    let mut zenity = vec!["--file-selection".into(), format!("--title={title}")];
    if folders {
        zenity.push("--directory".into());
    }
    if options.multiple {
        zenity.extend(["--multiple".into(), "--separator=\n".into()]);
    }
    let start = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let mut kdialog = vec!["--title".into(), title];
    if folders {
        kdialog.extend(["--getexistingdirectory".into(), start]);
    } else {
        kdialog.extend(["--getopenfilename".into(), start]);
        if options.multiple {
            kdialog.extend(["--multiple".into(), "--separate-output".into()]);
        }
    }
    [("zenity", zenity), ("kdialog", kdialog)]
}

/// The paths a stand-in printed, one per line.
fn paths(stdout: &[u8]) -> Vec<PathBuf> {
    String::from_utf8_lossy(stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .map(PathBuf::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(files: bool, directories: bool, multiple: bool) -> PathPromptOptions {
        PathPromptOptions {
            files,
            directories,
            multiple,
            prompt: Some("Attach".into()),
        }
    }

    /// Asked for files and folders where a picker shows one kind, "+ Attach" gets files: the
    /// portal's folder picker hid the PDFs and images it attaches.
    #[test]
    fn files_win_where_files_and_folders_cannot_mix() {
        let o = honoured(opts(true, true, true), false);
        assert!(o.files && !o.directories && o.multiple);
        let o = honoured(opts(true, true, true), true);
        assert!(o.files && o.directories);
        let o = honoured(opts(false, true, false), false);
        assert!(!o.files && o.directories);
    }

    #[test]
    fn the_stand_ins_pick_what_was_asked() {
        let [(z, za), (k, ka)] = commands(&opts(true, false, true));
        assert_eq!((z, k), ("zenity", "kdialog"));
        assert!(za.contains(&"--title=Attach".into()) && za.contains(&"--multiple".into()));
        assert!(!za.contains(&"--directory".into()));
        assert!(
            ka.contains(&"--getopenfilename".into()) && ka.contains(&"--separate-output".into())
        );

        let [(_, za), (_, ka)] = commands(&opts(false, true, false));
        assert!(za.contains(&"--directory".into()) && !za.contains(&"--multiple".into()));
        assert!(ka.contains(&"--getexistingdirectory".into()));
    }

    #[test]
    fn a_stand_in_prints_one_path_per_line() {
        let got = paths(b"/home/a/paper one.pdf\n/home/a/b.png\n\n");
        assert_eq!(
            got,
            [
                PathBuf::from("/home/a/paper one.pdf"),
                PathBuf::from("/home/a/b.png")
            ]
        );
        assert!(paths(b"").is_empty());
    }
}
