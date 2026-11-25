//! The window's own settings, kept in `<data>/desktop.json`. The engine's live in larch-app.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::theme::{Dark, Light};

/// Light, dark, or as the OS is.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, Debug)]
pub enum Mode {
    #[default]
    System,
    Light,
    Dark,
}

/// How much things move.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, Debug)]
pub enum Motion {
    #[default]
    System,
    Full,
    Reduced,
    Off,
}

/// Every window setting.
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Prefs {
    pub light: Light,
    pub dark: Dark,
    pub mode: Mode,
    /// Engine, Memory and the numbers: off is the plain app.
    pub advanced: bool,
    /// With Advanced on, a chat shows its side panel (the reply's numbers, sources and memory).
    pub side_panel: bool,
    pub compact: bool,
    pub text_size: u8,
    pub motion: Motion,
    /// The assistant tree shows at all.
    pub sun: bool,
    /// The chat's tree stays a one-line banner at the top instead of the full tree above the chat.
    pub sun_small: bool,
    /// The tree moves; off holds each state's pose.
    pub animate: bool,
    pub bubbles: bool,
    pub notify: bool,
    pub sound: bool,
    /// The name the greeting uses.
    pub name: String,
    /// The library downloads go to.
    pub download_to: Option<String>,
    /// The last tok/s each model reached here, by path, for the Library's cards.
    pub tok_s: HashMap<String, f64>,
    /// The model last loaded, which the next launch loads again.
    pub last_model: Option<String>,
    /// How long each model's last load took, in ms, for the launch screen's bar.
    pub load_ms: HashMap<String, u64>,
    /// Finished task runs the user filed as done; None until the first task list seeds it.
    pub reviewed: Option<std::collections::BTreeSet<String>>,
    /// Chats kept at the top of Recent and Chats, in the order they were pinned.
    pub pinned: Vec<String>,
    /// Chats kept but out of Recent, listed apart under Chats.
    pub archived: Vec<String>,
    /// The tour version this user finished or skipped (`tour::TOUR`); 0 until the first launch's
    /// welcome is answered. Settings saved before the tour existed read as seen, so an update does
    /// not greet a returning user as new.
    #[serde(default = "tour_seen")]
    pub welcomed: u32,
    /// Ask Larch Setup at launch and once a day whether a newer release is out. It only asks;
    /// installing waits for a click.
    pub updates: bool,
}

fn tour_seen() -> u32 {
    crate::tour::TOUR
}

impl Default for Prefs {
    fn default() -> Self {
        let name = std::env::var("USER")
            .or_else(|_| std::env::var("USERNAME"))
            .map(|n| {
                let mut c = n.chars();
                c.next()
                    .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
                    .unwrap_or_default()
            })
            .unwrap_or_default();
        Prefs {
            light: Light::Paper,
            dark: Dark::Night,
            mode: Mode::System,
            advanced: false,
            side_panel: true,
            compact: false,
            text_size: 14,
            motion: Motion::System,
            sun: true,
            sun_small: false,
            animate: true,
            bubbles: true,
            notify: true,
            sound: false,
            name,
            download_to: None,
            tok_s: HashMap::new(),
            last_model: None,
            load_ms: HashMap::new(),
            reviewed: None,
            pinned: Vec::new(),
            archived: Vec::new(),
            welcomed: 0,
            updates: true,
        }
    }
}

/// Where the settings live.
pub fn path(data: &Path) -> PathBuf {
    data.join("desktop.json")
}

impl Prefs {
    /// The saved settings; defaults for a missing file. A file that does not parse is reported and
    /// kept aside, never overwritten in silence; its user is not new, so the welcome stays away.
    pub fn load(data: &Path) -> (Prefs, Option<String>) {
        let p = path(data);
        match std::fs::read(&p) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (Prefs::default(), None),
            Err(e) => (
                Prefs::returning(),
                Some(format!("Could not read {}: {e}", p.display())),
            ),
            Ok(b) => match serde_json::from_slice(&b) {
                Ok(v) => (v, None),
                Err(e) => {
                    let aside = p.with_extension("json.bad");
                    let kept = std::fs::rename(&p, &aside).is_ok();
                    (
                        Prefs::returning(),
                        Some(format!(
                            "Your window settings did not parse ({e}), so they are back to defaults{}.",
                            if kept {
                                format!("; the old file is {}", aside.display())
                            } else {
                                String::new()
                            }
                        )),
                    )
                }
            },
        }
    }

    /// Defaults for someone whose settings file exists but could not be used.
    fn returning() -> Prefs {
        Prefs {
            welcomed: tour_seen(),
            ..Prefs::default()
        }
    }

    /// Saves the settings, through a temporary file so a crash never leaves half a file.
    pub fn save(&self, data: &Path) -> Result<(), String> {
        let p = path(data);
        let tmp = p.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        // Synced before the rename: a power loss must not leave an empty file under the real name.
        std::fs::File::create(&tmp)
            .and_then(|mut f| {
                std::io::Write::write_all(&mut f, &bytes)?;
                f.sync_all()
            })
            .and_then(|()| std::fs::rename(&tmp, &p))
            .map_err(|e| {
                format!(
                    "Could not save your window settings to {}: {e}",
                    p.display()
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A save reads back the same, and a broken file is set aside with a message, not lost.
    #[test]
    fn saves_round_trip_and_a_bad_file_is_kept() {
        let d = std::env::temp_dir().join(format!("ecl-prefs-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let mut p = Prefs {
            text_size: 16,
            ..Default::default()
        };
        p.tok_s.insert("/m/a.gguf".into(), 47.);
        p.save(&d).unwrap();
        let (back, err) = Prefs::load(&d);
        assert_eq!(
            (back.text_size, back.tok_s.get("/m/a.gguf").copied(), err),
            (16, Some(47.), None)
        );
        std::fs::write(path(&d), "{").unwrap();
        let (fresh, err) = Prefs::load(&d);
        assert_eq!(fresh.text_size, 14);
        assert!(err.unwrap().contains("did not parse"));
        assert!(d.join("desktop.json.bad").exists());
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// Settings saved before the side panel could be hidden keep showing it; a hidden one stays
    /// hidden across a save.
    #[test]
    fn the_side_panel_shows_unless_hidden() {
        let old: Prefs = serde_json::from_str(r#"{"advanced":true}"#).unwrap();
        assert!(old.advanced && old.side_panel);
        let hidden = Prefs {
            side_panel: false,
            ..Default::default()
        };
        let back: Prefs = serde_json::from_str(&serde_json::to_string(&hidden).unwrap()).unwrap();
        assert!(!back.side_panel);
    }

    /// Only a new install has the welcome to see: settings saved before the tour existed, and a
    /// file that could not be used, read as seen; a saved answer stays saved.
    #[test]
    fn only_a_new_install_is_welcomed() {
        let d = std::env::temp_dir().join(format!("ecl-prefs-tour-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        assert_eq!(Prefs::load(&d).0.welcomed, 0);
        let old: Prefs = serde_json::from_str(r#"{"name":"Sam"}"#).unwrap();
        assert_eq!(old.welcomed, crate::tour::TOUR);
        let seen = Prefs {
            welcomed: crate::tour::TOUR,
            ..Default::default()
        };
        seen.save(&d).unwrap();
        assert_eq!(Prefs::load(&d).0.welcomed, crate::tour::TOUR);
        std::fs::write(path(&d), "{").unwrap();
        let (broken, err) = Prefs::load(&d);
        assert!(err.is_some());
        assert_eq!(broken.welcomed, crate::tour::TOUR);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
