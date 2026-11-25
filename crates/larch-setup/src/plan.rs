//! Pure component selection and reversible filesystem plans over an explicit snapshot.
mod steps;
use steps::{finish_effects, relink, swap, sync_runtimes};

use crate::{
    detect::{Choice, Os},
    manifest::{self, Artifact, Error, Manifest, Result},
    txn::{Action, Component, Detection, Effect, Model, Receipt, Step},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};
/// A requested change; versions are explicit only on install and rollback.
#[derive(Clone, Debug)]
pub enum Request {
    /// Install or replace the selected user-facing components.
    Install {
        /// Include the command-line binaries.
        cli: bool,
        /// Include the desktop app.
        app: bool,
        /// Release channel selected by the user.
        channel: String,
        /// An explicit release, permitting a downgrade.
        version: Option<String>,
    },
    /// Add and remove components without implicitly replacing the others.
    Modify {
        /// Components to add.
        add: Vec<String>,
        /// Components to remove.
        remove: Vec<String>,
    },
    /// Update installed components, refusing a pin.
    Update,
    /// Replace installed components and reselect their backend.
    Repair,
    /// Restore one previous executable component with current runtimes.
    Rollback {
        /// One of cli, app or setup.
        component: String,
    },
    /// Remove a component or the complete installation.
    Uninstall {
        /// A selected user-facing component, or all components.
        component: Option<String>,
        /// Offer only recorded model files for data purging.
        purge: bool,
    },
    /// Prevent updates until explicitly unpinned.
    Pin(String),
    /// Clear an existing pin.
    Unpin,
}
/// Previous executable metadata travels in the same journal as its directory.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Previous {
    /// The previous version and its owned archive files.
    pub component: Component,
    /// Runtime links carried by that previous directory.
    pub links: Vec<(PathBuf, PathBuf)>,
}
/// Caller-observed paths; constructing a plan performs no filesystem access.
#[derive(Clone, Debug)]
pub struct Paths {
    /// Canonical installation root.
    pub root: PathBuf,
    /// User home, including the fixed macOS app destination.
    pub home: PathBuf,
    /// Artifact target triple.
    pub target: String,
    /// Unique transaction identity.
    pub txn: String,
    /// Existing paths, including previous directories and metadata.
    pub existing: BTreeSet<PathBuf>,
    /// Validated previous component descriptors.
    pub previous: BTreeMap<String, Previous>,
}
impl Paths {
    /// Installed component destination, including the macOS app exception.
    pub fn dest(&self, name: &str, os: Os) -> PathBuf {
        if name == "app" && os == Os::Macos {
            self.home.join("Applications/Larch.app")
        } else {
            self.root.join(name)
        }
    }
    /// Component staging destination on the filesystem where it will be installed.
    pub fn stage(&self, name: &str, os: Os) -> PathBuf {
        if name == "app" && os == Os::Macos {
            self.home
                .join("Applications/.larch-staging")
                .join(&self.txn)
        } else {
            self.root.join(".staging").join(&self.txn).join(name)
        }
    }
    /// Persistent previous-version descriptor, outside the signed archive inventory.
    pub fn previous_meta(&self, name: &str) -> PathBuf {
        self.root.join(".previous").join(format!("{name}.json"))
    }
    fn previous_dir(&self, name: &str) -> PathBuf {
        self.root.join(".previous").join(name)
    }
    fn trash(&self, name: &str) -> PathBuf {
        self.root.join(".trash").join(&self.txn).join(name)
    }
    fn metadata(&self, name: &str) -> PathBuf {
        self.root
            .join(".staging")
            .join(&self.txn)
            .join("metadata")
            .join(format!("{name}.json"))
    }
}
/// A fully reviewable change, including all staged bytes and owned data candidates.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Component archives to fetch and verify.
    pub downloads: Vec<(String, Artifact)>,
    /// Ordered reversible journal operations.
    pub steps: Vec<Step>,
    /// Committed target state.
    pub to: Receipt,
    /// User-facing decisions and byte totals.
    pub lines: Vec<String>,
    /// Compressed bytes to fetch.
    pub download_size: u64,
    /// Unpacked bytes to stage.
    pub unpacked_size: u64,
    /// Metadata staged before journaling, then moved by journal steps.
    pub metadata: Vec<(PathBuf, Vec<u8>)>,
    /// Recorded models eligible for the caller's explicit post-commit purge.
    pub purge_models: Vec<Model>,
}
impl Plan {
    /// Download plus unpacked bytes with ten percent rounded-up free-space headroom.
    pub fn required_space(&self) -> Result<u64> {
        let total = self
            .download_size
            .checked_add(self.unpacked_size)
            .ok_or_else(|| Error::Refused("plan byte total overflow".into()))?;
        total
            .checked_add(total.div_ceil(10))
            .ok_or_else(|| Error::Refused("plan headroom overflow".into()))
    }
    fn summary(&mut self) -> Result<()> {
        self.lines.push(format!(
            "download {} bytes; unpack {} bytes; required free space {} bytes",
            self.download_size,
            self.unpacked_size,
            self.required_space()?
        ));
        Ok(())
    }
    fn delete(&mut self, path: PathBuf, paths: &Paths) {
        if paths.existing.contains(&path)
            && !self
                .steps
                .iter()
                .any(|s| matches!(s,Step::Delete{path:p} if p==&path))
        {
            self.steps.push(Step::Delete { path });
        }
    }
    fn download(&mut self, name: &str, artifact: &Artifact) -> Result<()> {
        self.download_size = self
            .download_size
            .checked_add(artifact.size)
            .ok_or_else(|| Error::Refused("download size overflow".into()))?;
        for (_, (_, n)) in manifest::inventory(&artifact.files)? {
            self.unpacked_size = self
                .unpacked_size
                .checked_add(n)
                .ok_or_else(|| Error::Refused("unpacked size overflow".into()))?;
        }
        self.downloads.push((name.into(), artifact.clone()));
        Ok(())
    }
    fn remember(
        &mut self,
        name: &str,
        old: &Component,
        links: &[(PathBuf, PathBuf)],
        paths: &Paths,
    ) -> Result<()> {
        let before = Previous {
            component: old.clone(),
            links: links
                .iter()
                .filter(|(p, _)| p.starts_with(&old.dest))
                .cloned()
                .collect(),
        };
        let staged = paths.metadata(name);
        self.metadata
            .push((staged.clone(), serde_json::to_vec(&before)?));
        self.steps.push(Step::Rename {
            from: staged,
            to: paths.previous_meta(name),
        });
        Ok(())
    }
    fn remove_files(&mut self, c: &Component, paths: &Paths) -> Result<()> {
        for name in manifest::inventory(&c.files)?.keys() {
            self.delete(c.dest.join(manifest::relative_path(name)?), paths);
        }
        Ok(())
    }
}
fn executable(name: &str) -> bool {
    matches!(name, "cli" | "app" | "setup")
}
fn user_component(name: &str) -> Result<()> {
    if !matches!(name, "cli" | "app") {
        return Err(Error::Refused(format!(
            "component {name:?} must be cli or app"
        )));
    }
    Ok(())
}
fn identifier(value: &str) -> Result<()> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b))
    {
        return Err(Error::Refused(format!(
            "unsafe component or transaction identifier: {value:?}"
        )));
    }
    Ok(())
}
/// Keeps reused component archives at their own version when a newer manifest references them.
pub fn artifact_version(
    artifact: &Artifact,
    name: &str,
    target: &str,
    flavour: &str,
    release: &str,
) -> Result<String> {
    let file = artifact
        .url
        .split(['?', '#'])
        .next()
        .unwrap_or_default()
        .rsplit('/')
        .next()
        .unwrap_or_default();
    let prefix = format!("{name}-{target}-{flavour}-");
    let version = file
        .strip_prefix(&prefix)
        .and_then(|s| s.strip_suffix(".tar.zst"))
        .unwrap_or(release);
    manifest::accept_version(None, version, false)?;
    Ok(version.into())
}
/// Artifact ordering does not make a component change.
pub fn artifact_changed(
    current: &Component,
    artifact: &Artifact,
    version: &str,
    flavour: &str,
) -> Result<bool> {
    Ok(current.version != version
        || current.flavour != flavour
        || manifest::inventory(&current.files)? != manifest::inventory(&artifact.files)?)
}
fn offer<'a>(
    release: &'a Manifest,
    name: &str,
    target: &str,
    flavour: &str,
) -> Result<&'a Artifact> {
    let key = format!("{target}-{flavour}");
    let artifact = if let Some(runtime) = release.runtimes.get(name) {
        runtime.get(&key).map(|r| &r.artifact)
    } else {
        release.components.get(name).and_then(|m| m.get(&key))
    };
    artifact.ok_or_else(|| Error::Refused(format!("release has no {name} artifact for {key}")))
}
fn component(
    artifact: &Artifact,
    name: &str,
    flavour: &str,
    release: &Manifest,
    paths: &Paths,
    os: Os,
) -> Result<Component> {
    Ok(Component {
        version: artifact_version(artifact, name, &paths.target, flavour, &release.version)?,
        flavour: flavour.into(),
        dest: paths.dest(name, os),
        files: manifest::inventory(&artifact.files)?
            .into_iter()
            .map(|(n, (h, s))| (n, h, s))
            .collect(),
    })
}
fn owned_path(path: &std::path::Path, dest: &std::path::Path) -> bool {
    path.strip_prefix(dest)
        .ok()
        .and_then(|p| p.to_str())
        .is_some_and(|p| manifest::relative_path(p).is_ok())
}
fn validate_state(current: &Receipt, paths: &Paths, os: Os) -> Result<()> {
    if current.root != paths.root || current.schema != 1 {
        return Err(Error::StateChangedOutside);
    }
    for (name, c) in &current.components {
        identifier(name)?;
        if c.dest != paths.dest(name, os) {
            return Err(Error::Refused(format!(
                "component {name} destination does not match this installation"
            )));
        }
        manifest::inventory(&c.files)?;
    }
    for (link, target) in &current.links {
        let link_ok = current
            .components
            .iter()
            .any(|(name, c)| matches!(name.as_str(), "cli" | "app") && owned_path(link, &c.dest));
        let target_ok = current.components.iter().any(|(name, c)| {
            !executable(name)
                && owned_path(target, &c.dest)
                && target
                    .strip_prefix(&c.dest)
                    .ok()
                    .and_then(|p| p.to_str())
                    .is_some_and(|rel| c.files.iter().any(|(n, _, _)| n == &rel.replace('\\', "/")))
        });
        if !link_ok || !target_ok {
            return Err(Error::Refused(
                "recorded runtime link escapes its owned component inventory".into(),
            ));
        }
    }
    Ok(())
}
/// Plans a change from committed state using only signed metadata and caller observations.
pub fn plan(
    current: Option<&Receipt>,
    release: &Manifest,
    request: &Request,
    choice: &Choice,
    paths: &Paths,
    os: Os,
) -> Result<Plan> {
    identifier(&paths.txn)?;
    if paths.root.to_str().is_none()
        || paths.home.to_str().is_none()
        || !paths.root.is_absolute()
        || !paths.home.is_absolute()
        || paths
            .root
            .components()
            .chain(paths.home.components())
            .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(Error::Refused(
            "plan requires absolute normalized Unicode root and home".into(),
        ));
    }
    let old = current
        .cloned()
        .unwrap_or_else(|| Receipt::empty(paths.root.clone(), release.channel.clone()));
    validate_state(&old, paths, os)?;
    for (name, prev) in &paths.previous {
        if !executable(name) || prev.component.dest != paths.dest(name, os) {
            return Err(Error::Refused(
                "invalid previous component descriptor".into(),
            ));
        }
        manifest::inventory(&prev.component.files)?;
        for (p, _) in &prev.links {
            if !p.starts_with(&prev.component.dest)
                || p.components()
                    .any(|p| matches!(p, std::path::Component::ParentDir))
            {
                return Err(Error::Refused("previous link escapes its component".into()));
            }
        }
    }
    let mut plan = Plan {
        downloads: Vec::new(),
        steps: Vec::new(),
        to: old.clone(),
        lines: vec![choice.why.clone()],
        download_size: 0,
        unpacked_size: 0,
        metadata: Vec::new(),
        purge_models: Vec::new(),
    };
    plan.to.last_txn = paths.txn.clone();
    if let Request::Pin(version) = request {
        if current.is_none() {
            return Err(Error::Refused("nothing is installed to pin".into()));
        }
        manifest::accept_version(None, version, true)?;
        plan.to.pinned = Some(version.clone());
        plan.lines.push(format!("pin updates at {version}"));
        return Ok(plan);
    }
    if matches!(request, Request::Unpin) {
        if current.is_none() {
            return Err(Error::Refused("nothing is installed to unpin".into()));
        }
        plan.to.pinned = None;
        plan.lines.push("unpin updates".into());
        return Ok(plan);
    }
    if matches!(request, Request::Update) && old.pinned.is_some() {
        return Err(Error::Refused(
            "updates are pinned; run larch-setup unpin first".into(),
        ));
    }
    if let Request::Rollback { component: name } = request {
        if !executable(name) {
            return Err(Error::Refused(
                "rollback component must be cli, app or setup".into(),
            ));
        }
        let before = paths
            .previous
            .get(name)
            .ok_or_else(|| Error::Refused(format!("no previous {name} version is recorded")))?;
        let live = old
            .components
            .get(name)
            .ok_or_else(|| Error::Refused(format!("{name} is not installed")))?;
        if !paths.existing.contains(&paths.previous_dir(name))
            || !paths.existing.contains(&live.dest)
        {
            return Err(Error::Refused(
                "rollback directories are missing; run repair".into(),
            ));
        }
        let scratch = paths.stage(&format!("rollback-{name}"), os);
        plan.steps.extend([
            Step::Rename {
                from: live.dest.clone(),
                to: scratch.clone(),
            },
            Step::Rename {
                from: paths.previous_dir(name),
                to: live.dest.clone(),
            },
            Step::Rename {
                from: scratch,
                to: paths.previous_dir(name),
            },
            Step::Rename {
                from: paths.previous_meta(name),
                to: paths.trash(&format!("{name}.json")),
            },
        ]);
        plan.remember(name, live, &old.links, paths)?;
        plan.to
            .components
            .insert(name.clone(), before.component.clone());
        plan.to.detect.flavour = before.component.flavour.clone();
        plan.to.detect.why = if before.component.flavour == "cpu" {
            format!("rolled back {name}; CPU mode is slow")
        } else {
            format!("rolled back {name}; checked runtime dependencies")
        };
        let (mut changed, removed) =
            sync_runtimes(&mut plan, &old, release, choice, paths, os, request)?;
        for removed in removed {
            if let Some(c) = old.components.get(&removed) {
                plan.remove_files(c, paths)?;
            }
        }
        changed.insert(name.clone());
        relink(&mut plan, &old, paths, os, &changed, Some((name, before)))?;
        if matches!(name.as_str(), "cli" | "app") {
            plan.steps.push(Step::SelfTest {
                component: name.clone(),
            });
        }
        finish_effects(&mut plan, &old, os, false)?;
        plan.lines
            .push(format!("rollback {name} to {}", before.component.version));
        plan.summary()?;
        return Ok(plan);
    }
    let mut selected = BTreeSet::new();
    let mut removed = BTreeSet::new();
    let full = matches!(
        request,
        Request::Uninstall {
            component: None,
            ..
        }
    );
    match request {
        Request::Install {
            cli,
            app,
            channel,
            version,
        } => {
            if !cli && !app {
                return Err(Error::Refused("select --cli, --app, or both".into()));
            }
            if channel != &release.channel
                || version.as_ref().is_some_and(|v| v != &release.version)
            {
                return Err(Error::Refused(
                    "release does not match the requested channel or version".into(),
                ));
            }
            if *cli {
                selected.insert("cli".to_owned());
            }
            if *app {
                selected.insert("app".to_owned());
            }
            plan.to.channel = channel.clone();
        }
        Request::Modify { add, remove } => {
            if add.is_empty() && remove.is_empty() {
                return Err(Error::Refused("modify requires --add or --remove".into()));
            }
            if current.is_none() {
                return Err(Error::Refused("nothing is installed to modify".into()));
            }
            for name in add {
                user_component(name)?;
                if remove.contains(name) {
                    return Err(Error::Refused(format!(
                        "{name} cannot be both added and removed"
                    )));
                }
                selected.insert(name.clone());
            }
            for name in remove {
                user_component(name)?;
                removed.insert(name.clone());
            }
        }
        Request::Update | Request::Repair => {
            if current.is_none() {
                return Err(Error::Refused("nothing is installed; run install".into()));
            }
            selected.extend(
                old.components
                    .keys()
                    .filter(|n| matches!(n.as_str(), "cli" | "app"))
                    .cloned(),
            );
        }
        Request::Uninstall { component, purge } => {
            if let Some(name) = component {
                user_component(name)?;
                removed.insert(name.clone());
            } else {
                removed.extend(old.components.keys().cloned());
            }
            if *purge {
                plan.purge_models = old.models.clone();
            }
            if full {
                plan.to.models.clear();
                plan.to.pinned = None;
            }
        }
        _ => unreachable!(),
    }
    if !matches!(request, Request::Uninstall { .. }) {
        let explicit = matches!(
            request,
            Request::Install {
                version: Some(_),
                ..
            }
        );
        for c in old.components.values() {
            manifest::accept_version(Some(&c.version), &release.version, explicit)?;
        }
        if !release
            .flavours
            .get(&paths.target)
            .is_some_and(|f| f.contains(&choice.flavour))
            || !release.gated.contains(&choice.flavour)
        {
            return Err(Error::Refused(format!(
                "flavour {} is not built and gated",
                choice.flavour
            )));
        }
        selected.insert("setup".into());
        plan.to.channel = release.channel.clone();
        plan.to.detect = Detection {
            flavour: choice.flavour.clone(),
            gpu: choice.gpu.as_ref().map(|g| g.name.clone()),
            why: choice.why.clone(),
        };
    }
    for name in &removed {
        plan.to.components.remove(name);
    }
    let mut changed = BTreeSet::new();
    for name in &selected {
        let flavour = if name == "setup" {
            "cpu"
        } else {
            &choice.flavour
        };
        let artifact = offer(release, name, &paths.target, flavour)?;
        let new = component(artifact, name, flavour, release, paths, os)?;
        let replace = matches!(request, Request::Repair)
            || old
                .components
                .get(name)
                .map(|c| artifact_changed(c, artifact, &new.version, flavour))
                .transpose()?
                .unwrap_or(true);
        plan.to.components.insert(name.clone(), new.clone());
        if replace {
            plan.download(name, artifact)?;
            swap(
                &mut plan,
                name,
                old.components.get(name),
                &new,
                paths,
                os,
                &old.links,
            )?;
            changed.insert(name.clone());
            plan.lines
                .push(format!("install {name} {} ({flavour})", new.version));
        }
    }
    let (runtime_changes, runtime_removals) =
        sync_runtimes(&mut plan, &old, release, choice, paths, os, request)?;
    changed.extend(runtime_changes);
    removed.extend(runtime_removals);
    for name in &removed {
        if let Some(c) = old.components.get(name) {
            plan.remove_files(c, paths)?;
            plan.lines
                .push(format!("remove {name}: {} recorded files", c.files.len()));
        }
        if let Some(prev) = paths.previous.get(name) {
            let mut c = prev.component.clone();
            c.dest = paths.previous_dir(name);
            plan.remove_files(&c, paths)?;
            for (link, _) in &prev.links {
                let rel = link
                    .strip_prefix(&prev.component.dest)
                    .map_err(|_| Error::Refused("invalid previous link".into()))?;
                plan.delete(c.dest.join(rel), paths);
            }
            plan.delete(paths.previous_meta(name), paths);
        }
    }
    for (link, _) in &old.links {
        if removed
            .iter()
            .filter_map(|name| old.components.get(name))
            .any(|c| link.starts_with(&c.dest))
        {
            plan.delete(link.clone(), paths);
        }
    }
    relink(&mut plan, &old, paths, os, &changed, None)?;
    let runtime_changed = changed.iter().any(|n| !executable(n));
    for (name, c) in &plan.to.components {
        if matches!(name.as_str(), "cli" | "app")
            && (changed.contains(name)
                || (runtime_changed && matches!(c.flavour.as_str(), "cuda" | "hip")))
        {
            plan.steps.push(Step::SelfTest {
                component: name.clone(),
            });
        }
    }
    finish_effects(&mut plan, &old, os, matches!(request, Request::Repair))?;
    plan.summary()?;
    Ok(plan)
}
