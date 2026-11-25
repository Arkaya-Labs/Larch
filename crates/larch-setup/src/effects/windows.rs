use super::*;
/// Adds unexpanded Windows PATH entries without changing any pre-existing bytes.
pub fn path_add(raw: &str, entries: &[String]) -> String {
    let mut result = raw.to_owned();
    for entry in entries {
        if !result.split(';').any(|p| p.eq_ignore_ascii_case(entry)) {
            if !result.is_empty() {
                result.push(';');
            }
            result.push_str(entry);
        }
    }
    result
}
/// Removes only matching entries, preserving all other entries and empty segments.
pub fn path_remove(raw: &str, entries: &[String]) -> String {
    raw.split(';')
        .filter(|p| !entries.iter().any(|e| p.eq_ignore_ascii_case(e)))
        .collect::<Vec<_>>()
        .join(";")
}
fn restored_path(current: &str, state: &str, entry: &str) -> (String, bool) {
    if current == state
        || current == path_add(state, &[entry.into()])
        || current == path_remove(state, &[entry.into()])
    {
        return (state.into(), false);
    }
    let mut kept: Vec<_> = current
        .split(';')
        .filter(|p| !p.eq_ignore_ascii_case(entry))
        .map(str::to_owned)
        .collect();
    for (i, part) in state.split(';').enumerate() {
        if part.eq_ignore_ascii_case(entry) {
            kept.insert(i.min(kept.len()), part.into());
        }
    }
    (kept.join(";"), true)
}
/// The path without its verbatim prefix, as PowerShell and the shell expect it.
pub(crate) fn normal_path(path: &Path) -> Result<String> {
    let text = path
        .to_str()
        .ok_or_else(|| Error::Refused("Windows integration path is not Unicode".into()))?;
    Ok(if let Some(unc) = text.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{unc}")
    } else {
        text.strip_prefix(r"\\?\").unwrap_or(text).into()
    })
}
const READ_PATH: &str = r#"$ErrorActionPreference='Stop'
$key=[Microsoft.Win32.Registry]::CurrentUser.OpenSubKey($env:LARCH_ENV_KEY)
$raw=''
if($null -ne $key){$raw=$key.GetValue('Path','','DoNotExpandEnvironmentNames');$key.Dispose()}
if($raw -isnot [string]){throw 'User PATH is not a string'}
[Console]::Write($raw)"#;
const WRITE_PATH: &str = r#"$ErrorActionPreference='Stop'
$key=[Microsoft.Win32.Registry]::CurrentUser.CreateSubKey($env:LARCH_ENV_KEY)
$raw=$key.GetValue('Path','','DoNotExpandEnvironmentNames')
if($raw -cne $env:LARCH_OLD_PATH){$key.Dispose();throw 'User PATH changed concurrently; retry setup'}
$key.SetValue('Path',$env:LARCH_NEW_PATH,[Microsoft.Win32.RegistryValueKind]::ExpandString)
if($env:LARCH_TEST_ROOTS -ne '1'){
[Environment]::SetEnvironmentVariable($env:LARCH_BROADCAST,'1','User')
[Environment]::SetEnvironmentVariable($env:LARCH_BROADCAST,$null,'User')
}
$key.Flush()
$key.Dispose()"#;
const SHORTCUT: &str = r#"$ErrorActionPreference='Stop'
$shell=New-Object -ComObject WScript.Shell
if($env:LARCH_MODE -eq 'present'){
  if(!(Test-Path -LiteralPath $env:LARCH_LINK)){[Console]::Write('false');exit 0}
  try{$link=$shell.CreateShortcut($env:LARCH_LINK);[Console]::Write($link.TargetPath.Equals($env:LARCH_APP,[StringComparison]::OrdinalIgnoreCase).ToString().ToLowerInvariant())}catch{[Console]::Write('false')}
}else{
  $link=$shell.CreateShortcut($env:LARCH_LINK)
  $link.TargetPath=$env:LARCH_APP
  $link.WorkingDirectory=$env:LARCH_APP_DIR
  $link.Arguments=''
  $link.IconLocation=$env:LARCH_APP
  $link.Save()
}"#;
const UNINSTALL: &str = r#"$ErrorActionPreference='Stop'
$parent=[Microsoft.Win32.Registry]::CurrentUser
$key=$parent.OpenSubKey($env:LARCH_UNINSTALL_KEY)
$ours=$false
if($null -ne $key){$location=$key.GetValue('InstallLocation','','DoNotExpandEnvironmentNames');$ours=([string]$location).Equals($env:LARCH_ROOT,[StringComparison]::OrdinalIgnoreCase);$key.Dispose()}
if($env:LARCH_MODE -eq 'present'){[Console]::Write($ours.ToString().ToLowerInvariant());exit 0}
if($env:LARCH_MODE -eq 'remove'){if($ours){$parent.DeleteSubKeyTree($env:LARCH_UNINSTALL_KEY,$false);$parent.Flush()};exit 0}
if($null -ne $key -and !$ours){throw 'Uninstall entry belongs to another installation; it was preserved; uninstall it before changing roots'}
$key=$parent.CreateSubKey($env:LARCH_UNINSTALL_KEY)
$key.SetValue('DisplayName','Larch')
$key.SetValue('DisplayVersion',$env:LARCH_VERSION)
$key.SetValue('Publisher','Arkaya Labs')
$key.SetValue('DisplayIcon',$env:LARCH_ICON)
$key.SetValue('InstallLocation',$env:LARCH_ROOT)
$key.SetValue('EstimatedSize',[int]$env:LARCH_SIZE,[Microsoft.Win32.RegistryValueKind]::DWord)
$key.SetValue('NoModify',1,[Microsoft.Win32.RegistryValueKind]::DWord)
$key.SetValue('NoRepair',1,[Microsoft.Win32.RegistryValueKind]::DWord)
$key.SetValue('UninstallString',$env:LARCH_UNINSTALL)
$key.Flush()
$key.Dispose()"#;
fn ps(script: &str, variables: &[(&str, OsString)], roots: &Roots) -> Result<String> {
    let mut command = Command::new("powershell.exe");
    let script = format!("[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);\n{script}");
    command.args(["-NoProfile", "-NonInteractive", "-Command", &script]);
    roots.isolate(&mut command);
    for (name, value) in variables {
        command.env(name, value);
    }
    match command::run(&mut command,Duration::from_secs(10))? {Output::Finished{success:true,stdout,..}=>Ok(stdout),Output::Finished{stderr,..}=>Err(Error::Refused(format!("Windows integration failed: {}",stderr.trim().replace(['\r','\n']," ")))),Output::Missing=>Err(Error::Refused("PowerShell is unavailable; user settings were preserved; retry setup with Windows PowerShell installed".into())),Output::TimedOut=>Err(Error::Refused("PowerShell timed out; run larch-setup repair to check user settings".into()))}
}
impl Effects {
    fn ps(&self, script: &str, variables: &[(&str, OsString)]) -> Result<String> {
        ps(script, variables, &self.roots)
    }
    fn env_key(&self) -> String {
        self.roots
            .registry_prefix
            .as_ref()
            .map(|p| format!("{p}\\Environment"))
            .unwrap_or_else(|| "Environment".into())
    }
    fn uninstall_key(&self) -> String {
        self.roots
            .registry_prefix
            .as_ref()
            .map(|p| format!("{p}\\Uninstall\\larch"))
            .unwrap_or_else(|| {
                "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\larch".into()
            })
    }
    fn entry(&self, v: &Value) -> Result<String> {
        if !matches!(v.name.as_str(), "cli" | "setup") {
            return Err(Error::Refused("unknown Windows PATH entry".into()));
        }
        let entry = normal_path(&v.root.join(&v.name))?;
        if entry.contains(';') {
            return Err(Error::Refused("installation path contains the Windows PATH separator; choose a different install root".into()));
        }
        Ok(entry)
    }
    pub(super) fn windows_path_present(&self, v: &Value) -> Result<bool> {
        let raw = self.ps(READ_PATH, &[("LARCH_ENV_KEY", self.env_key().into())])?;
        let entry = self.entry(v)?;
        Ok(raw.split(';').any(|s| s.eq_ignore_ascii_case(&entry)))
    }
    pub(super) fn windows_path(&self, v: &Value, add: bool) -> Result<()> {
        let key = self.env_key();
        let old = self.ps(READ_PATH, &[("LARCH_ENV_KEY", key.clone().into())])?;
        let entry = self.entry(v)?;
        let new = if add {
            path_add(&old, &[entry])
        } else {
            path_remove(&old, &[entry])
        };
        if new != old {
            let broadcast = if self.roots.registry_prefix.is_some() {
                "LARCH_SETUP_TEST_BROADCAST"
            } else {
                "LARCH_SETUP_BROADCAST"
            };
            self.ps(
                WRITE_PATH,
                &[
                    ("LARCH_ENV_KEY", key.into()),
                    ("LARCH_OLD_PATH", old.into()),
                    ("LARCH_NEW_PATH", new.into()),
                    ("LARCH_BROADCAST", broadcast.into()),
                    (
                        "LARCH_TEST_ROOTS",
                        if self.roots.registry_prefix.is_some() {
                            "1"
                        } else {
                            "0"
                        }
                        .into(),
                    ),
                ],
            )?;
        }
        Ok(())
    }
    pub(super) fn windows_path_snapshot(&self) -> Result<String> {
        self.ps(READ_PATH, &[("LARCH_ENV_KEY", self.env_key().into())])
    }
    pub(super) fn windows_path_restore(&mut self, v: &Value, state: &str) -> Result<()> {
        let current = self.windows_path_snapshot()?;
        if current == state {
            return Ok(());
        }
        let entry = self.entry(v)?;
        let (next, changed) = restored_path(&current, state, &entry);
        if changed {
            self.warn("PATH changed outside setup; restored Larch entries and preserved other changes; check your PATH".into());
        }
        self.ps(
            WRITE_PATH,
            &[
                ("LARCH_ENV_KEY", self.env_key().into()),
                ("LARCH_OLD_PATH", current.into()),
                ("LARCH_NEW_PATH", next.into()),
                ("LARCH_BROADCAST", "LARCH_SETUP_BROADCAST".into()),
                (
                    "LARCH_TEST_ROOTS",
                    if self.roots.registry_prefix.is_some() {
                        "1"
                    } else {
                        "0"
                    }
                    .into(),
                ),
            ],
        )?;
        Ok(())
    }
    fn shortcut_vars(&self, v: &Value, mode: &str) -> Result<Vec<(&'static str, OsString)>> {
        if v.file != "entry" {
            return Err(Error::Refused("unknown Windows launcher file".into()));
        }
        Ok(vec![
            ("LARCH_MODE", mode.into()),
            (
                "LARCH_LINK",
                self.roots.start_menu.join("Larch.lnk").into_os_string(),
            ),
            (
                "LARCH_APP",
                normal_path(&v.root.join("app").join("larch-desktop.exe"))?.into(),
            ),
            ("LARCH_APP_DIR", normal_path(&v.root.join("app"))?.into()),
        ])
    }
    pub(super) fn windows_launcher_present(&self, v: &Value) -> Result<bool> {
        let link = self.roots.start_menu.join("Larch.lnk");
        if fs::symlink_metadata(&link).is_ok_and(|m| !m.is_file()) {
            return Ok(false);
        }
        Ok(self.ps(SHORTCUT, &self.shortcut_vars(v, "present")?)? == "true")
    }
    pub(super) fn windows_launcher(&mut self, v: &Value, apply: bool) -> Result<()> {
        let link = self.roots.start_menu.join("Larch.lnk");
        let present = self.windows_launcher_present(v)?;
        if apply && !present {
            if exists(&link)? {
                self.warn(format!(
                    "left foreign shortcut {} untouched; launch the installed app directly",
                    link.display()
                ));
                return Ok(());
            }
            fs::create_dir_all(&self.roots.start_menu)?;
            use std::sync::atomic::{AtomicU64, Ordering};
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let cache = self.roots.cache.join("larch-setup");
            create_dir_all(&cache)?;
            let temp = loop {
                let p = cache.join(format!(
                    "shortcut-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                match fs::create_dir(&p) {
                    Ok(()) => break p,
                    Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(e) => return Err(e.into()),
                }
            };
            let result = (|| -> Result<()> {
                let temporary = temp.join("Larch.lnk");
                let mut vars = self.shortcut_vars(v, "apply")?;
                vars.iter_mut()
                    .find(|(name, _)| *name == "LARCH_LINK")
                    .unwrap()
                    .1 = temporary.clone().into_os_string();
                self.ps(SHORTCUT, &vars)?;
                let bytes = fs::read(temporary)?;
                if exists(&link)? {
                    return Err(Error::Refused(
                        "shortcut appeared outside setup; it was preserved; retry setup".into(),
                    ));
                }
                atomic_write(&link, &bytes)
            })();
            if let Err(cleanup) = fs::remove_dir_all(temp) {
                return Err(Error::Refused(format!(
                    "shortcut temporary-file cleanup failed: {cleanup}"
                )));
            }
            result?;
        } else if !apply && present {
            fs::remove_file(link)?;
        } else if !apply && exists(&link)? {
            self.warn(format!(
                "left modified shortcut {} untouched; remove it manually if unwanted",
                link.display()
            ));
        }
        Ok(())
    }
    fn uninstall_vars(&self, v: &Value, mode: &str) -> Result<Vec<(&'static str, OsString)>> {
        let root = normal_path(&v.root)?;
        // The installer window: Settings shows its logo and its Uninstall button opens it there.
        // A package without it (the e2e fixture's) keeps the terminal program.
        let window = v.root.join("setup").join("larch-installer.exe");
        let (exe, uninstall) = if window.is_file() {
            (normal_path(&window)?, "--uninstall")
        } else {
            (
                normal_path(&v.root.join("setup").join("larch-setup.exe"))?,
                "uninstall",
            )
        };
        let kib = v.size.div_ceil(1024);
        if kib > i32::MAX as u64 {
            return Err(Error::Refused(
                "uninstall entry size exceeds the Windows registry field".into(),
            ));
        }
        Ok(vec![
            ("LARCH_MODE", mode.into()),
            ("LARCH_UNINSTALL_KEY", self.uninstall_key().into()),
            ("LARCH_ROOT", root.into()),
            ("LARCH_VERSION", v.version.clone().into()),
            ("LARCH_ICON", exe.clone().into()),
            ("LARCH_SIZE", kib.to_string().into()),
            ("LARCH_UNINSTALL", format!("\"{exe}\" {uninstall}").into()),
        ])
    }
    pub(super) fn windows_uninstall_present(&self, v: &Value) -> Result<bool> {
        Ok(self.ps(UNINSTALL, &self.uninstall_vars(v, "present")?)? == "true")
    }
    pub(super) fn windows_uninstall(&self, v: &Value, apply: bool) -> Result<()> {
        self.ps(
            UNINSTALL,
            &self.uninstall_vars(v, if apply { "apply" } else { "remove" })?,
        )?;
        Ok(())
    }
}
pub(super) fn holders(root: &Path) -> Result<Vec<Holder>> {
    let text = ps(
        "$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Select-Object ProcessId,ExecutablePath) | ConvertTo-Json -Compress",
        &[],
        &Roots::from_env()?,
    )?;
    let values: serde_json::Value = serde_json::from_str(&text)?;
    let values = match values {
        serde_json::Value::Array(v) => v,
        serde_json::Value::Object(_) => vec![values],
        serde_json::Value::Null => Vec::new(),
        _ => return Err(Error::Refused("invalid Windows process inventory".into())),
    };
    let root = normal_path(root)?.to_ascii_lowercase();
    let mut result = Vec::new();
    for v in values {
        if let (Some(pid), Some(image)) = (v["ProcessId"].as_u64(), v["ExecutablePath"].as_str())
            && Path::new(&image.to_ascii_lowercase()).starts_with(&root)
        {
            result.push(Holder {
                pid: u32::try_from(pid)
                    .map_err(|_| Error::Refused("invalid Windows process identifier".into()))?,
                name: Path::new(image)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
            });
        }
    }
    result.sort_by_key(|h| h.pid);
    Ok(result)
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    #[test]
    fn windows_registry_and_shortcut_in_test_roots() {
        let root = env::temp_dir().join(format!(
            "larch-windows-effects-{} O'Brien",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        // Long names, as setup's canonical root has: TEMP can be an 8.3 short path
        // (C:\Users\LONGNA~1), which a shortcut's target never reads back as.
        let root = PathBuf::from(normal_path(&fs::canonicalize(root).unwrap()).unwrap());
        let roots = Roots::in_test_root(&root).unwrap();
        let install = root.join("install");
        let mut e = Effects::new(roots.clone(), Os::Windows, "", OsStr::new(""));
        let initial = r"%USERPROFILE%\bin;;C:\keep;";
        e.ps(
            WRITE_PATH,
            &[
                ("LARCH_ENV_KEY", e.env_key().into()),
                ("LARCH_OLD_PATH", "".into()),
                ("LARCH_NEW_PATH", initial.into()),
                ("LARCH_TEST_ROOTS", "1".into()),
            ],
        )
        .unwrap();
        for name in ["cli", "setup"] {
            let v = serde_json::json!({"root":install,"name":name}).to_string();
            e.apply("path", &v).unwrap();
            e.apply("path", &v).unwrap();
            assert!(e.present("path", &v).unwrap());
            e.remove("path", &v).unwrap();
            e.remove("path", &v).unwrap();
        }
        assert_eq!(
            e.ps(READ_PATH, &[("LARCH_ENV_KEY", e.env_key().into())])
                .unwrap(),
            initial
        );
        let kind=e.ps("$k=[Microsoft.Win32.Registry]::CurrentUser.OpenSubKey($env:LARCH_ENV_KEY);[Console]::Write($k.GetValueKind('Path').ToString());$k.Dispose()",&[("LARCH_ENV_KEY",e.env_key().into())]).unwrap();
        assert_eq!(kind, "ExpandString");
        fs::create_dir_all(install.join("app")).unwrap();
        fs::write(install.join("app/larch-desktop.exe"), "fake").unwrap();
        let launcher = serde_json::json!({"root":install,"file":"entry"}).to_string();
        e.apply("launcher", &launcher).unwrap();
        assert!(e.present("launcher", &launcher).unwrap());
        e.remove("launcher", &launcher).unwrap();
        assert!(!e.present("launcher", &launcher).unwrap());
        let entry = serde_json::json!({"root":install,"version":"1.2.3","size":2049}).to_string();
        e.apply("uninstall", &entry).unwrap();
        assert!(e.present("uninstall", &entry).unwrap());
        e.remove("uninstall", &entry).unwrap();
        assert!(!e.present("uninstall", &entry).unwrap());
        e.ps("$p=$env:LARCH_PREFIX;if(!$p.StartsWith('Software\\LarchSetupTests\\')){throw 'unsafe test prefix'};[Microsoft.Win32.Registry]::CurrentUser.DeleteSubKeyTree($p,$false)",&[("LARCH_PREFIX",roots.registry_prefix.clone().unwrap().into())]).unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod path_tests {
    use super::*;
    #[test]
    fn rollback_restores_raw_order_duplicates_and_preserves_concurrent_other_entries() {
        let entry = r"C:\larch\cli";
        let old = format!("pre;{entry};mid;{};post;", entry.to_ascii_uppercase());
        let removed = path_remove(&old, &[entry.into()]);
        assert_eq!(restored_path(&removed, &old, entry), (old.clone(), false));
        let current = "pre;mid;post;;new";
        let before = format!("pre;{entry};mid;post;");
        assert_eq!(
            restored_path(current, &before, entry),
            (format!("pre;{entry};mid;post;;new"), true)
        );
    }
}
