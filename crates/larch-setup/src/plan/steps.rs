use super::*;

pub(super) fn swap(
    plan: &mut Plan,
    name: &str,
    old: Option<&Component>,
    new: &Component,
    paths: &Paths,
    os: Os,
    links: &[(PathBuf, PathBuf)],
) -> Result<()> {
    if let Some(old) = old {
        if executable(name) {
            let previous = paths.previous_dir(name);
            if paths.existing.contains(&previous) {
                if !paths.previous.contains_key(name) {
                    return Err(Error::Refused(format!(
                        "unrecorded previous directory {}; preserve it before repair",
                        previous.display()
                    )));
                }
                plan.steps.push(Step::Rename {
                    from: previous.clone(),
                    to: paths.trash(&format!("{name}.prev")),
                });
            }
            if paths.existing.contains(&paths.previous_meta(name)) {
                plan.steps.push(Step::Rename {
                    from: paths.previous_meta(name),
                    to: paths.trash(&format!("{name}.json")),
                });
            }
            if paths.existing.contains(&old.dest) {
                plan.steps.push(Step::Rename {
                    from: old.dest.clone(),
                    to: previous,
                });
                plan.remember(name, old, links, paths)?;
            }
        } else if paths.existing.contains(&old.dest) {
            plan.steps.push(Step::Rename {
                from: old.dest.clone(),
                to: paths.trash(name),
            });
        }
    } else if paths.existing.contains(&new.dest) {
        return Err(Error::Refused(format!(
            "unrecorded destination {}; it has been preserved",
            new.dest.display()
        )));
    }
    plan.steps.push(Step::Rename {
        from: paths.stage(name, os),
        to: new.dest.clone(),
    });
    Ok(())
}
fn effects(to: &Receipt, os: Os) -> Result<Vec<Effect>> {
    if to.components.is_empty() {
        return Ok(Vec::new());
    }
    let mut effects = Vec::new();
    let names: &[&str] = if os == Os::Windows {
        &["cli", "setup"]
    } else if to.components.contains_key("cli") {
        &["larch", "larch-agent", "larch-setup"]
    } else {
        &["larch-setup"]
    };
    // One effect per entry keeps partial existing integrations reversible.
    for name in names {
        effects.push(Effect {
            kind: "path".into(),
            value: serde_json::json!({"root":to.root,"name":name}).to_string(),
            applied: true,
        });
    }
    if os != Os::Windows {
        for file in ["bashrc", "profile", "zshenv", "fish"] {
            effects.push(Effect {
                kind: "shell".into(),
                value: serde_json::json!({"root":to.root,"file":file}).to_string(),
                applied: true,
            });
        }
    }
    // `setup` opens the installer window, the GUI uninstaller; Windows reaches it from Settings.
    let app = to.components.contains_key("app");
    let launchers: &[&str] = match os {
        Os::Windows if app => &["entry"],
        Os::Windows => &[],
        Os::Linux if app => &["entry", "icon", "setup"],
        Os::Linux => &["icon", "setup"],
        Os::Macos => &["setup"],
    };
    for file in launchers {
        effects.push(Effect {
            kind: "launcher".into(),
            value: serde_json::json!({"root":to.root,"file":file}).to_string(),
            applied: true,
        });
    }
    if os == Os::Windows {
        let size =
            to.components
                .values()
                .flat_map(|c| &c.files)
                .try_fold(0u64, |n, (_, _, s)| {
                    n.checked_add(*s)
                        .ok_or_else(|| Error::Refused("installed size overflow".into()))
                })?;
        let version = to
            .components
            .get("setup")
            .map(|c| c.version.as_str())
            .unwrap_or("0.0.0");
        let value = serde_json::json!({"root":to.root,"version":version,"size":size}).to_string();
        effects.push(Effect {
            kind: "uninstall".into(),
            value,
            applied: true,
        });
    }
    Ok(effects)
}
pub(super) fn finish_effects(plan: &mut Plan, old: &Receipt, os: Os, force: bool) -> Result<()> {
    let desired = effects(&plan.to, os)?;
    let removed: Vec<_> = old
        .effects
        .iter()
        .filter(|e| e.applied && !desired.contains(e))
        .map(|e| Step::Effect {
            kind: e.kind.clone(),
            value: e.value.clone(),
            action: Action::Remove,
        })
        .collect();
    plan.steps.splice(0..0, removed);
    for effect in &desired {
        if force || !old.effects.contains(effect) {
            plan.steps.push(Step::Effect {
                kind: effect.kind.clone(),
                value: effect.value.clone(),
                action: Action::Apply,
            });
        }
    }
    plan.to.effects = desired;
    Ok(())
}
pub(super) fn relink(
    plan: &mut Plan,
    old: &Receipt,
    paths: &Paths,
    os: Os,
    changed: &BTreeSet<String>,
    rollback: Option<(&str, &Previous)>,
) -> Result<()> {
    let mut links = Vec::new();
    if os == Os::Windows {
        for (name, c) in &plan.to.components {
            if !matches!(name.as_str(), "cli" | "app") {
                continue;
            }
            for (runtime, r) in &plan.to.components {
                if (c.flavour == "cuda" && runtime == "cuda")
                    || (c.flavour == "hip" && runtime.starts_with("rocm-"))
                {
                    for path in manifest::inventory(&r.files)?
                        .keys()
                        .filter(|p| p.to_ascii_lowercase().ends_with(".dll"))
                    {
                        let path = manifest::relative_path(path)?;
                        let file = path
                            .file_name()
                            .ok_or_else(|| Error::Refused("runtime DLL has no filename".into()))?;
                        let pair = (c.dest.join(file), r.dest.join(&path));
                        if manifest::inventory(&c.files)?
                            .keys()
                            .any(|p| p.eq_ignore_ascii_case(&file.to_string_lossy()))
                        {
                            return Err(Error::Refused(format!(
                                "runtime DLL collides with a component file: {}",
                                pair.0.display()
                            )));
                        }
                        if links.iter().any(|(dest, _): &(PathBuf, PathBuf)| {
                            dest.to_string_lossy()
                                .eq_ignore_ascii_case(&pair.0.to_string_lossy())
                        }) {
                            return Err(Error::Refused(format!(
                                "runtime DLL name collision: {}",
                                pair.0.display()
                            )));
                        }
                        links.push(pair);
                    }
                }
            }
        }
    }
    let mut carried: Vec<_> = old
        .links
        .iter()
        .filter(|(link, _)| {
            !plan
                .steps
                .iter()
                .any(|s| matches!(s,Step::Delete{path} if path==link))
        })
        .cloned()
        .collect();
    if let Some((_, prev)) = rollback {
        carried.retain(|(p, _)| !p.starts_with(&prev.component.dest));
        carried.extend(prev.links.clone());
    }
    for (i, (link, target)) in carried.iter().enumerate() {
        let owner = plan
            .to
            .components
            .iter()
            .find(|(n, c)| matches!(n.as_str(), "cli" | "app") && link.starts_with(&c.dest))
            .map(|(n, _)| n);
        let carried_by_swap = owner.is_some_and(|n| changed.contains(n)) && rollback.is_none();
        let needs_move = (!links.contains(&(link.clone(), target.clone()))
            || owner.is_some_and(|n| changed.contains(n))
            || plan.to.components.iter().any(|(n, c)| {
                !executable(n) && changed.contains(n) && target.starts_with(&c.dest)
            }))
            && !carried_by_swap;
        let existed_before_rollback = rollback.is_some_and(|(name, prev)| {
            link.strip_prefix(&prev.component.dest)
                .is_ok_and(|rel| paths.existing.contains(&paths.previous_dir(name).join(rel)))
        });
        if needs_move && (paths.existing.contains(link) || existed_before_rollback) {
            plan.steps.push(Step::Rename {
                from: link.clone(),
                to: paths.trash(&format!("link-{i}")),
            });
        }
    }
    for (link, target) in &links {
        let owner_changed = plan
            .to
            .components
            .iter()
            .any(|(n, c)| changed.contains(n) && link.starts_with(&c.dest));
        let runtime_changed = plan
            .to
            .components
            .iter()
            .any(|(n, c)| changed.contains(n) && !executable(n) && target.starts_with(&c.dest));
        if !old.links.contains(&(link.clone(), target.clone())) || owner_changed || runtime_changed
        {
            plan.steps.push(Step::Link {
                from: target.clone(),
                to: link.clone(),
            });
        }
    }
    plan.to.links = links;
    Ok(())
}

pub(super) fn sync_runtimes(
    plan: &mut Plan,
    old: &Receipt,
    release: &Manifest,
    choice: &Choice,
    paths: &Paths,
    os: Os,
    request: &Request,
) -> Result<(BTreeSet<String>, BTreeSet<String>)> {
    let mut changed = BTreeSet::new();
    let mut removed = BTreeSet::new();
    let cuda = plan
        .to
        .components
        .iter()
        .any(|(n, c)| matches!(n.as_str(), "cli" | "app") && c.flavour == "cuda");
    let hip = plan
        .to
        .components
        .iter()
        .any(|(n, c)| matches!(n.as_str(), "cli" | "app") && c.flavour == "hip");
    let mut runtimes = BTreeSet::new();
    if cuda {
        runtimes.insert("cuda".to_owned());
    }
    if hip {
        if choice.flavour == "hip" {
            let gfx = choice
                .gpu
                .as_ref()
                .and_then(|g| g.gfx.as_ref())
                .ok_or_else(|| Error::Refused("hip requires a graphics target".into()))?;
            runtimes.insert(format!("rocm-{gfx}"));
        } else {
            runtimes.extend(
                old.components
                    .keys()
                    .filter(|n| n.starts_with("rocm-"))
                    .cloned(),
            );
        }
        if !runtimes.iter().any(|n| n.starts_with("rocm-")) {
            return Err(Error::Refused(
                "installed hip component has no recorded runtime".into(),
            ));
        }
    }
    for name in old
        .components
        .keys()
        .filter(|n| !executable(n) && !runtimes.contains(*n))
    {
        plan.to.components.remove(name);
        removed.insert(name.clone());
    }
    if !matches!(request, Request::Uninstall { .. }) {
        for name in &runtimes {
            identifier(name)?;
            let flavour = if name == "cuda" { "cuda" } else { "hip" };
            let old_runtime = old.components.get(name);
            if matches!(request, Request::Rollback { .. }) && old_runtime.is_some() {
                continue;
            }
            if !release.gated.iter().any(|f| f == flavour) {
                if old_runtime.is_some() {
                    continue;
                }
                return Err(Error::Refused(format!(
                    "rollback or install needs a gated {flavour} runtime"
                )));
            }
            let artifact = offer(release, name, &paths.target, flavour)?;
            let new = component(artifact, name, flavour, release, paths, os)?;
            let replace = matches!(request, Request::Repair)
                || old_runtime
                    .map(|c| artifact_changed(c, artifact, &new.version, flavour))
                    .transpose()?
                    .unwrap_or(true);
            plan.to.components.insert(name.clone(), new.clone());
            if replace {
                plan.download(name, artifact)?;
                swap(plan, name, old_runtime, &new, paths, os, &old.links)?;
                changed.insert(name.clone());
                plan.lines
                    .push(format!("install runtime {name} {}", new.version));
            }
        }
    }
    Ok((changed, removed))
}
