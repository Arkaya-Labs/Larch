//! Idempotent per-user OS integrations, with explicit test roots and no global environment writes.
mod windows;
use crate::{
    command::{self, Output},
    detect::Os,
    manifest::{Error, Result},
    txn::{atomic_write, create_dir_all, sync_parent},
};
use serde::Deserialize;
use std::{
    env,
    ffi::{OsStr, OsString},
    fs, io,
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};
#[cfg(windows)]
pub(crate) use windows::normal_path;
pub use windows::{path_add, path_remove};
/// The desktop application's existing icon, shared with the installer without an engine dependency.
pub const ICON: &[u8] = include_bytes!("../../../apps/desktop/icons/icon.png");
const START: &str = "# >>> larch-setup >>>";
const END: &str = "# <<< larch-setup <<<";
/// Platform directories and a test-only registry namespace.
#[derive(Clone, Debug)]
pub struct Roots {
    /// User home directory.
    pub home: PathBuf,
    /// User data base, before the larch product directory.
    pub data: PathBuf,
    /// User configuration base.
    pub config: PathBuf,
    /// User cache base.
    pub cache: PathBuf,
    /// Windows roaming application data.
    pub appdata: PathBuf,
    /// Windows local application data.
    pub localappdata: PathBuf,
    /// Windows Start-menu programs directory.
    pub start_menu: PathBuf,
    /// Separate HKCU prefix used only by tests.
    pub registry_prefix: Option<String>,
}
impl Roots {
    /// Reads user directories, ignoring the test override in production builds.
    pub fn from_env() -> Result<Self> {
        #[cfg(any(test, feature = "test-roots"))]
        if let Some(root) = env::var_os("LARCH_SETUP_TEST_ROOTS") {
            return Self::in_test_root(Path::new(&root));
        }
        let home = env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .ok_or_else(|| Error::Refused("an absolute user home is required".into()))?;
        let absolute = |name: &str, fallback: PathBuf| {
            env::var_os(name)
                .map(PathBuf::from)
                .filter(|p| p.is_absolute())
                .unwrap_or(fallback)
        };
        let localappdata = absolute("LOCALAPPDATA", home.join("AppData/Local"));
        let appdata = absolute("APPDATA", home.join("AppData/Roaming"));
        let data = if cfg!(windows) {
            localappdata.clone()
        } else if cfg!(target_os = "macos") {
            home.join("Library/Application Support")
        } else {
            absolute("XDG_DATA_HOME", home.join(".local/share"))
        };
        Ok(Self {
            config: absolute("XDG_CONFIG_HOME", home.join(".config")),
            cache: absolute("XDG_CACHE_HOME", home.join(".cache")),
            start_menu: appdata.join("Microsoft/Windows/Start Menu/Programs"),
            home,
            data,
            appdata,
            localappdata,
            registry_prefix: None,
        })
    }
    /// Constructs isolated paths without changing the process environment.
    #[cfg(any(test, feature = "test-roots"))]
    pub fn in_test_root(root: &Path) -> Result<Self> {
        use std::hash::{Hash, Hasher};
        if !root.is_absolute() {
            return Err(Error::Refused("test roots must be absolute".into()));
        }
        let mut hash = std::hash::DefaultHasher::new();
        root.hash(&mut hash);
        let home = root.join("home");
        let data = if cfg!(windows) {
            root.join("local")
        } else if cfg!(target_os = "macos") {
            home.join("Library/Application Support")
        } else {
            root.join("data")
        };
        Ok(Self {
            home,
            data,
            config: root.join("config"),
            cache: root.join("cache"),
            appdata: root.join("roaming"),
            localappdata: root.join("local"),
            start_menu: root.join("start-menu"),
            registry_prefix: Some(format!("Software\\LarchSetupTests\\{:016x}", hash.finish())),
        })
    }
    /// Redirects child tool and self-test directories only when test roots are active.
    pub fn isolate(&self, command: &mut Command) {
        if self.registry_prefix.is_some() {
            for (name, path) in [
                ("HOME", &self.home),
                ("USERPROFILE", &self.home),
                ("XDG_DATA_HOME", &self.data),
                ("XDG_CONFIG_HOME", &self.config),
                ("XDG_CACHE_HOME", &self.cache),
                ("APPDATA", &self.appdata),
                ("LOCALAPPDATA", &self.localappdata),
            ] {
                command.env(name, path);
            }
        }
    }
    /// The same product data directory used by the shared downloader.
    pub fn data_dir(&self) -> PathBuf {
        self.data.join("larch")
    }
    /// The shell-independent user command directory.
    pub fn bin(&self) -> PathBuf {
        self.home.join(".local/bin")
    }
    /// The Linux desktop launcher.
    pub fn desktop_entry(&self) -> PathBuf {
        self.data.join("applications/com.arkaya-labs.larch.desktop")
    }
    /// The Linux launcher for the installer window, which every install gets.
    pub fn setup_entry(&self) -> PathBuf {
        self.data
            .join("applications/com.arkaya-labs.larch.setup.desktop")
    }
    /// The matching icon, at its source resolution.
    pub fn desktop_icon(&self) -> PathBuf {
        self.data
            .join("icons/hicolor/1024x1024/apps/com.arkaya-labs.larch.png")
    }
}
#[derive(Deserialize)]
struct Value {
    root: PathBuf,
    #[serde(default)]
    name: String,
    #[serde(default)]
    file: String,
    #[serde(default)]
    version: String,
    #[serde(default)]
    size: u64,
}
fn value(text: &str) -> Result<Value> {
    let v: Value = serde_json::from_str(text)?;
    if !v.root.is_absolute()
        || v.root
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(Error::Refused(
            "effect root must be absolute and normalized".into(),
        ));
    }
    Ok(v)
}
/// OS integrations over captured shell/PATH observations and explicit user directories.
pub struct Effects {
    roots: Roots,
    os: Os,
    shell: String,
    path: OsString,
    messages: Vec<String>,
}
impl Effects {
    /// Captures the current platform and environment without modifying either.
    pub fn from_env() -> Result<Self> {
        Ok(Self::new(
            Roots::from_env()?,
            Os::current()?,
            &env::var("SHELL").unwrap_or_default(),
            &env::var_os("PATH").unwrap_or_default(),
        ))
    }
    /// Uses explicit observations so tests can never modify real user settings.
    pub fn new(roots: Roots, os: Os, shell: &str, path: &OsStr) -> Self {
        Self {
            roots,
            os,
            shell: shell.into(),
            path: path.into(),
            messages: Vec::new(),
        }
    }
    /// Returns warnings that the caller prints and appends to setup.log.
    pub fn take_messages(&mut self) -> Vec<String> {
        std::mem::take(&mut self.messages)
    }
    fn warn(&mut self, message: String) {
        self.messages.push(message.replace(['\r', '\n'], " "));
    }
    /// Checks the exact integration without creating files or registry keys.
    pub fn present(&mut self, kind: &str, text: &str) -> Result<bool> {
        let v = value(text)?;
        match kind {
            "path" => {
                if self.os == Os::Windows {
                    self.windows_path_present(&v)
                } else {
                    let (link, target) = self.link(&v)?;
                    ours_link(&link, &target)
                }
            }
            "shell" => {
                let (file, block, _) = self.shell_file(&v)?;
                Ok(read(&file)?.is_some_and(|b| exact_block(&b, &block)))
            }
            "launcher" => {
                if self.os == Os::Windows {
                    self.windows_launcher_present(&v)
                } else if self.os == Os::Linux {
                    let (path, bytes) = self.launcher(&v)?;
                    Ok(read(&path)?.is_some_and(|old| old == bytes))
                } else {
                    let (link, target) = self.mac_launcher(&v)?;
                    ours_link(&link, &target)
                }
            }
            "uninstall" => {
                if self.os == Os::Windows {
                    self.windows_uninstall_present(&v)
                } else {
                    Ok(false)
                }
            }
            _ => Err(Error::Refused(format!("unknown OS effect {kind:?}"))),
        }
    }
    /// Saves raw Windows PATH bytes so rollback preserves original ordering and duplicates.
    pub fn snapshot(&self, kind: &str, text: &str) -> Result<Option<String>> {
        let _ = value(text)?;
        if self.os == Os::Windows && kind == "path" {
            Ok(Some(self.windows_path_snapshot()?))
        } else {
            Ok(None)
        }
    }
    /// Restores owned state while retaining unrelated concurrent PATH edits.
    pub fn restore(&mut self, kind: &str, text: &str, state: &str) -> Result<()> {
        if self.os == Os::Windows && kind == "path" {
            self.windows_path_restore(&value(text)?, state)
        } else {
            Err(Error::Refused(format!(
                "effect {kind:?} does not use captured state"
            )))
        }
    }
    /// Applies one owned integration idempotently, preserving foreign files and edits.
    pub fn apply(&mut self, kind: &str, text: &str) -> Result<()> {
        let v = value(text)?;
        match kind {
            "path" => {
                if self.os == Os::Windows {
                    self.windows_path(&v, true)
                } else {
                    let (link, target) = self.link(&v)?;
                    self.add_link(&link, &target, "add the command to PATH manually")
                }
            }
            "shell" => self.apply_shell(&v),
            "launcher" => {
                if self.os == Os::Windows {
                    self.windows_launcher(&v, true)
                } else if self.os == Os::Linux {
                    self.linux_launcher(&v, true)
                } else {
                    let (link, target) = self.mac_launcher(&v)?;
                    self.add_link(&link, &target, "open Larch Setup from its install folder")
                }
            }
            "uninstall" => {
                if self.os == Os::Windows {
                    self.windows_uninstall(&v, true)
                } else {
                    Ok(())
                }
            }
            _ => Err(Error::Refused(format!("unknown OS effect {kind:?}"))),
        }
    }
    /// Removes only unchanged content or links owned by this installation.
    pub fn remove(&mut self, kind: &str, text: &str) -> Result<()> {
        let v = value(text)?;
        match kind {
            "path" => {
                if self.os == Os::Windows {
                    self.windows_path(&v, false)
                } else {
                    let (link, target) = self.link(&v)?;
                    self.drop_link(&link, &target)
                }
            }
            "shell" => self.remove_shell(&v),
            "launcher" => {
                if self.os == Os::Windows {
                    self.windows_launcher(&v, false)
                } else if self.os == Os::Linux {
                    self.linux_launcher(&v, false)
                } else {
                    let (link, target) = self.mac_launcher(&v)?;
                    self.drop_link(&link, &target)
                }
            }
            "uninstall" => {
                if self.os == Os::Windows {
                    self.windows_uninstall(&v, false)
                } else {
                    Ok(())
                }
            }
            _ => Err(Error::Refused(format!("unknown OS effect {kind:?}"))),
        }
    }
    fn link(&self, v: &Value) -> Result<(PathBuf, PathBuf)> {
        let component = match v.name.as_str() {
            "larch" | "larch-agent" => "cli",
            "larch-setup" => "setup",
            _ => return Err(Error::Refused("unknown user command link".into())),
        };
        Ok((
            self.roots.bin().join(&v.name),
            v.root.join(component).join(&v.name),
        ))
    }
    /// The installer window in Applications: Finder opens a symlinked bundle as the bundle.
    fn mac_launcher(&self, v: &Value) -> Result<(PathBuf, PathBuf)> {
        if v.file != "setup" {
            return Err(Error::Refused("unknown launcher integration file".into()));
        }
        Ok((
            self.roots.home.join("Applications/Larch Setup.app"),
            v.root.join("setup/Larch Setup.app"),
        ))
    }
    fn add_link(&mut self, link: &Path, target: &Path, otherwise: &str) -> Result<()> {
        if ours_link(link, target)? {
            return Ok(());
        }
        if exists(link)? {
            self.warn(format!(
                "left {} untouched: it is not our symlink; {otherwise}",
                link.display()
            ));
            return Ok(());
        }
        create_dir_all(link.parent().unwrap())?;
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(target, link)?;
            sync_parent(link)?;
            Ok(())
        }
        #[cfg(not(unix))]
        {
            Err(Error::Refused(
                "Unix symlinks are unavailable on this platform".into(),
            ))
        }
    }
    fn drop_link(&mut self, link: &Path, target: &Path) -> Result<()> {
        if ours_link(link, target)? {
            fs::remove_file(link)?;
            sync_parent(link)?;
        } else if exists(link)? {
            self.warn(format!(
                "left {} untouched: it is not our symlink; remove it manually if unwanted",
                link.display()
            ));
        }
        Ok(())
    }
    fn shell_file(&self, v: &Value) -> Result<(PathBuf, Vec<u8>, &'static str)> {
        let (file, shell, line) = match v.file.as_str() {
            "bashrc" => (
                self.roots.home.join(".bashrc"),
                "bash",
                "export PATH=\"$HOME/.local/bin:$PATH\"",
            ),
            "profile" => (
                self.roots.home.join(".profile"),
                "bash",
                "export PATH=\"$HOME/.local/bin:$PATH\"",
            ),
            "zshenv" => (
                self.roots.home.join(".zshenv"),
                "zsh",
                "export PATH=\"$HOME/.local/bin:$PATH\"",
            ),
            "fish" => (
                self.roots.config.join("fish/conf.d/larch.fish"),
                "fish",
                "set -gx PATH \"$HOME/.local/bin\" $PATH",
            ),
            _ => return Err(Error::Refused("unknown shell integration file".into())),
        };
        Ok((
            file,
            format!("\n{START}\n{line}\n{END}\n").into_bytes(),
            shell,
        ))
    }
    fn apply_shell(&mut self, v: &Value) -> Result<()> {
        if self.os == Os::Windows {
            return Ok(());
        }
        let (file, block, shell) = self.shell_file(v)?;
        let current = Path::new(&self.shell)
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or_default();
        if !matches!(current, "bash" | "zsh" | "fish") {
            if v.file == "bashrc" {
                self.warn(
                    "shell not recognized; add export PATH=\"$HOME/.local/bin:$PATH\" manually"
                        .into(),
                );
            }
            return Ok(());
        }
        if shell != current
            || (self.os != Os::Macos && env::split_paths(&self.path).any(|p| p == self.roots.bin()))
        {
            return Ok(());
        }
        let old = read(&file)?;
        if old.is_none() && exists(&file)? {
            self.warn(format!(
                "left non-file shell target {} untouched; update PATH manually",
                file.display()
            ));
            return Ok(());
        }
        let old = old.unwrap_or_default();
        if exact_block(&old, &block) {
            return Ok(());
        }
        if has_marker(&old) {
            self.warn(format!(
                "left edited shell block in {}; update PATH there manually",
                file.display()
            ));
            return Ok(());
        }
        create_dir_all(file.parent().unwrap())?;
        let mut bytes = old;
        bytes.extend_from_slice(&block);
        atomic_write(&file, &bytes)
    }
    fn remove_shell(&mut self, v: &Value) -> Result<()> {
        if self.os == Os::Windows {
            return Ok(());
        }
        let (file, block, _) = self.shell_file(v)?;
        let Some(old) = read(&file)? else {
            if exists(&file)? {
                self.warn(format!("left non-file shell target {} untouched; remove its block manually if unwanted",file.display()));
            }
            return Ok(());
        };
        if exact_block(&old, &block) {
            let start = old.windows(block.len()).position(|s| s == block).unwrap();
            let mut bytes = old[..start].to_vec();
            bytes.extend_from_slice(&old[start + block.len()..]);
            if v.file == "fish" && bytes.is_empty() {
                fs::remove_file(&file)?;
                sync_parent(&file)?;
            } else {
                atomic_write(&file, &bytes)?;
            }
        } else if has_marker(&old) {
            self.warn(format!(
                "left edited shell block in {}; remove it manually if unwanted",
                file.display()
            ));
        }
        Ok(())
    }
    fn launcher(&self, v: &Value) -> Result<(PathBuf, Vec<u8>)> {
        match v.file.as_str() {
            "icon" => Ok((self.roots.desktop_icon(), ICON.to_vec())),
            "entry" | "setup" => {
                let (name, exe, path) = if v.file == "entry" {
                    ("Larch", "app/larch-desktop", self.roots.desktop_entry())
                } else {
                    (
                        "Larch Setup",
                        "setup/larch-installer",
                        self.roots.setup_entry(),
                    )
                };
                let content = format!(
                    "[Desktop Entry]\nType=Application\nName={name}\nExec={}\nIcon={}\nTerminal=false\nCategories=Development;\n",
                    desktop_quote(&v.root.join(exe).to_string_lossy()),
                    desktop_string(&self.roots.desktop_icon().to_string_lossy())
                );
                Ok((path, content.into_bytes()))
            }
            _ => Err(Error::Refused("unknown launcher integration file".into())),
        }
    }
    fn linux_launcher(&mut self, v: &Value, apply: bool) -> Result<()> {
        let (path, bytes) = self.launcher(v)?;
        let old = read(&path)?;
        if apply {
            if old.as_ref().is_some_and(|b| b == &bytes) {
                return Ok(());
            }
            if old.is_some() || exists(&path)? {
                self.warn(format!("left foreign launcher file {} untouched; launch the app from its install directory",path.display()));
                return Ok(());
            }
            create_dir_all(path.parent().unwrap())?;
            atomic_write(&path, &bytes)?;
        } else if old.as_ref().is_some_and(|b| b == &bytes) {
            fs::remove_file(&path)?;
            sync_parent(&path)?;
        } else if old.is_some() || exists(&path)? {
            self.warn(format!(
                "left modified launcher file {} untouched; remove it manually if unwanted",
                path.display()
            ));
        }
        if v.file != "icon" {
            match command::run(Command::new("update-desktop-database").arg(self.roots.desktop_entry().parent().unwrap()),Duration::from_secs(10))? {Output::Missing=>{},Output::Finished{success:true,..}=>{},_=>self.warn("launcher database refresh did not finish; installed files are safe; run larch-setup repair".into())}
        }
        Ok(())
    }
}
fn exists(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}
fn read(path: &Path) -> Result<Option<Vec<u8>>> {
    match fs::symlink_metadata(path) {
        Ok(meta) if !meta.is_file() => Ok(None),
        Ok(_) => Ok(Some(fs::read(path)?)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}
fn ours_link(link: &Path, target: &Path) -> Result<bool> {
    match fs::read_link(link) {
        Ok(old) => Ok(old == target
            || fs::canonicalize(link)
                .ok()
                .zip(fs::canonicalize(target).ok())
                .is_some_and(|(a, b)| a == b)),
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::InvalidInput
            ) =>
        {
            Ok(false)
        }
        Err(e) => Err(e.into()),
    }
}
fn has_marker(bytes: &[u8]) -> bool {
    [START, END]
        .iter()
        .any(|m| bytes.windows(m.len()).any(|s| s == m.as_bytes()))
}
fn exact_block(bytes: &[u8], block: &[u8]) -> bool {
    bytes.windows(block.len()).filter(|s| *s == block).count() == 1
        && [START, END].iter().all(|m| {
            bytes
                .windows(m.len())
                .filter(|s| *s == m.as_bytes())
                .count()
                == 1
        })
}
fn desktop_string(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}
/// Quotes one Exec argument through both desktop-entry escaping layers and field expansion.
pub fn desktop_quote(text: &str) -> String {
    let mut q = String::from("\"");
    for c in text.chars() {
        if matches!(c, '"' | '`' | '$' | '\\') {
            q.push('\\');
        }
        if c == '%' {
            q.push('%');
        }
        q.push(c);
    }
    q.push('"');
    desktop_string(&q)
}
/// One running image under the installation, used only for Busy diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Holder {
    /// Process identifier.
    pub pid: u32,
    /// Executable filename without user directory information.
    pub name: String,
}
/// Lists installed images without terminating or signalling any process.
pub fn holders(root: &Path, os: Os) -> Result<Vec<Holder>> {
    let mut result = Vec::new();
    match os {
        Os::Linux => {
            let entries = match fs::read_dir("/proc") {
                Ok(v) => v,
                Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(result),
                Err(e) => return Err(e.into()),
            };
            for entry in entries {
                let entry = entry?;
                let Some(pid) = entry.file_name().to_str().and_then(|s| s.parse().ok()) else {
                    continue;
                };
                if let Ok(path) = fs::read_link(entry.path().join("exe"))
                    && path.starts_with(root)
                {
                    result.push(Holder {
                        pid,
                        name: path
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned(),
                    });
                }
            }
        }
        Os::Macos => {
            if let Some(text) = command::text("ps", &["-axo", "pid=,comm="])? {
                for line in text.lines() {
                    let line = line.trim();
                    if let Some((pid, path)) = line.split_once(char::is_whitespace)
                        && let Ok(pid) = pid.parse()
                        && Path::new(path.trim()).starts_with(root)
                    {
                        result.push(Holder {
                            pid,
                            name: Path::new(path.trim())
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .into_owned(),
                        });
                    }
                }
            }
        }
        Os::Windows => return windows::holders(root),
    }
    result.sort_by_key(|h| h.pid);
    Ok(result)
}
