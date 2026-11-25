//! Plans cover every component/backend combination and preserve ownership on uninstall.
mod support;
use larch_setup::{
    detect::{Choice, Os},
    manifest::{self, Artifact, Manifest, RuntimeArtifact},
    plan::*,
    txn::{Action, Receipt, Step},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};
fn paths(os: Os) -> Paths {
    // The planner refuses relative roots, and "/install" is relative on Windows.
    let base = PathBuf::from(if cfg!(windows) { r"C:\" } else { "/" });
    Paths {
        root: base.join("install"),
        home: base.join("home").join("test"),
        target: match os {
            Os::Linux => "x86_64-unknown-linux-gnu",
            Os::Windows => "x86_64-pc-windows-msvc",
            Os::Macos => "aarch64-apple-darwin",
        }
        .into(),
        txn: "txn".into(),
        existing: BTreeSet::new(),
        previous: BTreeMap::new(),
    }
}
fn choice(flavour: &str) -> Choice {
    Choice {
        flavour: flavour.into(),
        gpu: if flavour == "hip" {
            larch_setup::detect::kfd_node("gfx_target_version 110000", &[]).unwrap()
        } else {
            None
        },
        why: if flavour == "cpu" {
            "unsupported GPU; CPU mode is slow".into()
        } else {
            format!("selected {flavour}")
        },
    }
}
fn artifact(name: &str, target: &str, flavour: &str, version: &str) -> Artifact {
    let exe = if target.contains("windows") {
        ".exe"
    } else {
        ""
    };
    let files = if name == "cli" {
        vec![
            (format!("larch{exe}"), "1".repeat(64), 10),
            (format!("larch-agent{exe}"), "2".repeat(64), 20),
        ]
    } else if name == "setup" {
        vec![(format!("larch-setup{exe}"), "3".repeat(64), 30)]
    } else if name == "app" {
        vec![(
            if target.contains("apple") {
                "Contents/MacOS/larch-desktop".into()
            } else {
                format!("larch-desktop{exe}")
            },
            "4".repeat(64),
            40,
        )]
    } else {
        vec![(
            if target.contains("windows") {
                "runtime.dll"
            } else {
                "lib/libruntime.so"
            }
            .into(),
            "5".repeat(64),
            50,
        )]
    };
    Artifact {
        url: format!("https://releases.invalid/{name}-{target}-{flavour}-{version}.tar.zst"),
        sha256: "a".repeat(64),
        size: 100,
        files,
    }
}
fn release(p: &Paths) -> Manifest {
    let mut m = support::manifest();
    for name in ["cli", "app", "setup"] {
        for flavour in ["cpu", "cuda", "hip", "metal"] {
            m.components.entry(name.into()).or_default().insert(
                format!("{}-{flavour}", p.target),
                artifact(name, &p.target, flavour, &m.version),
            );
        }
    }
    for (name, flavour) in [("cuda", "cuda"), ("rocm-gfx1100", "hip")] {
        m.runtimes.entry(name.into()).or_default().insert(
            format!("{}-{flavour}", p.target),
            RuntimeArtifact {
                artifact: artifact(name, &p.target, flavour, &m.version),
                provenance: "test upstream".into(),
            },
        );
    }
    m
}
fn install(cli: bool, app: bool) -> Request {
    Request::Install {
        cli,
        app,
        channel: "stable".into(),
        version: None,
    }
}
fn observed(p: &mut Paths, r: &Receipt) {
    p.existing.clear();
    for c in r.components.values() {
        p.existing.insert(c.dest.clone());
        for (n, _, _) in &c.files {
            p.existing.insert(c.dest.join(n));
        }
    }
    for (link, _) in &r.links {
        p.existing.insert(link.clone());
    }
}
fn downloads(p: &Plan) -> BTreeSet<&str> {
    p.downloads.iter().map(|(n, _)| n.as_str()).collect()
}
#[test]
fn installs_cli_app_both_for_every_backend_and_checks_space_and_order() {
    for flavour in ["cpu", "cuda", "hip", "metal"] {
        let os = if flavour == "metal" {
            Os::Macos
        } else {
            Os::Linux
        };
        let p = paths(os);
        let m = release(&p);
        for (cli, app) in [(true, false), (false, true), (true, true)] {
            let plan = plan(None, &m, &install(cli, app), &choice(flavour), &p, os).unwrap();
            assert!(plan.to.components.contains_key("setup"));
            assert_eq!(plan.to.components.contains_key("cli"), cli);
            assert_eq!(plan.to.components.contains_key("app"), app);
            assert_eq!(plan.to.components.contains_key("cuda"), flavour == "cuda");
            assert_eq!(
                plan.to.components.contains_key("rocm-gfx1100"),
                flavour == "hip"
            );
            assert_eq!(plan.download_size, plan.downloads.len() as u64 * 100);
            assert_eq!(
                plan.required_space().unwrap(),
                (plan.download_size + plan.unpacked_size) * 11 / 10
            );
            assert!(plan.lines.iter().any(|s| s.contains(&choice(flavour).why)));
            let first_effect = plan
                .steps
                .iter()
                .position(|s| matches!(s, Step::Effect { .. }))
                .unwrap();
            assert!(
                plan.steps[..first_effect]
                    .iter()
                    .filter(|s| matches!(s, Step::SelfTest { .. }))
                    .count()
                    == usize::from(cli) + usize::from(app)
            );
            if flavour == "metal" && app {
                assert_eq!(
                    plan.to.components["app"].dest,
                    p.home.join("Applications/Larch.app")
                );
                assert!(plan.steps.iter().any(|s|matches!(s,Step::Rename{from,to} if from==&p.home.join("Applications/.larch-staging/txn")&&to==&p.home.join("Applications/Larch.app"))));
            }
            assert!(!plan.steps.iter().any(|s| matches!(s, Step::Link { .. })));
        }
    }
}
#[test]
fn modify_refcounts_both_ways_and_flavour_switches_are_one_plan() {
    let os = Os::Linux;
    let mut p = paths(os);
    let m = release(&p);
    let r = plan(None, &m, &install(true, false), &choice("cuda"), &p, os)
        .unwrap()
        .to;
    observed(&mut p, &r);
    let added = plan(
        Some(&r),
        &m,
        &Request::Modify {
            add: vec!["app".into()],
            remove: vec![],
        },
        &choice("cpu"),
        &p,
        os,
    )
    .unwrap();
    assert_eq!(downloads(&added), BTreeSet::from(["app"]));
    assert!(added.to.components.contains_key("cuda"));
    assert_eq!(added.to.components["cli"].flavour, "cuda");
    assert_eq!(added.to.components["app"].flavour, "cpu");
    observed(&mut p, &added.to);
    let removed = plan(
        Some(&added.to),
        &m,
        &Request::Modify {
            add: vec![],
            remove: vec!["cli".into()],
        },
        &choice("cpu"),
        &p,
        os,
    )
    .unwrap();
    assert!(!removed.to.components.contains_key("cuda"));
    assert!(removed.to.components.contains_key("app"));
    observed(&mut p, &removed.to);
    let back = plan(
        Some(&removed.to),
        &m,
        &Request::Modify {
            add: vec!["cli".into()],
            remove: vec!["app".into()],
        },
        &choice("hip"),
        &p,
        os,
    )
    .unwrap();
    assert!(back.to.components.contains_key("rocm-gfx1100"));
    assert!(!back.to.components.contains_key("app"));
    let mut p = paths(os);
    let both = plan(None, &m, &install(true, true), &choice("cuda"), &p, os)
        .unwrap()
        .to;
    observed(&mut p, &both);
    let switched = plan(Some(&both), &m, &Request::Repair, &choice("cpu"), &p, os).unwrap();
    assert_eq!(
        switched
            .steps
            .iter()
            .filter(|s| matches!(
                s,
                Step::Effect {
                    action: Action::Apply,
                    ..
                }
            ))
            .count(),
        switched.to.effects.len()
    );
    assert!(!switched.to.components.contains_key("cuda"));
    assert_eq!(switched.to.components["cli"].flavour, "cpu");
    assert_eq!(switched.to.components["app"].flavour, "cpu");
    assert!(downloads(&switched).is_superset(&BTreeSet::from(["cli", "app"])));
}
#[test]
fn only_changed_components_update_pins_refuse_and_downgrades_require_explicit_version() {
    let os = Os::Linux;
    let mut p = paths(os);
    let m = release(&p);
    let r = plan(None, &m, &install(true, true), &choice("cpu"), &p, os)
        .unwrap()
        .to;
    observed(&mut p, &r);
    let mut next = m.clone();
    next.version = "1.2.4".into();
    let cli = next
        .components
        .get_mut("cli")
        .unwrap()
        .get_mut(&format!("{}-cpu", p.target))
        .unwrap();
    cli.url = cli.url.replace("1.2.3", "1.2.4");
    cli.files[0].1 = "b".repeat(64);
    let changed = plan(Some(&r), &next, &Request::Update, &choice("cpu"), &p, os).unwrap();
    assert_eq!(downloads(&changed), BTreeSet::from(["cli"]));
    assert_eq!(changed.to.components["app"].version, "1.2.3");
    assert_eq!(changed.to.components["cli"].version, "1.2.4");
    assert_eq!(changed.metadata.len(), 1);
    let desc: Previous = serde_json::from_slice(&changed.metadata[0].1).unwrap();
    assert_eq!(desc.component.version, "1.2.3");
    let pinned = plan(
        Some(&r),
        &m,
        &Request::Pin("1.2.3".into()),
        &choice("cpu"),
        &p,
        os,
    )
    .unwrap()
    .to;
    assert!(
        plan(
            Some(&pinned),
            &next,
            &Request::Update,
            &choice("cpu"),
            &p,
            os
        )
        .unwrap_err()
        .to_string()
        .contains("pinned")
    );
    let unpinned = plan(Some(&pinned), &m, &Request::Unpin, &choice("cpu"), &p, os)
        .unwrap()
        .to;
    assert!(unpinned.pinned.is_none());
    let mut older = m.clone();
    older.version = "1.2.2".into();
    for map in older.components.values_mut() {
        for a in map.values_mut() {
            a.url = a.url.replace("1.2.3", "1.2.2");
        }
    }
    assert!(
        plan(Some(&r), &older, &Request::Update, &choice("cpu"), &p, os)
            .unwrap_err()
            .to_string()
            .contains("downgrade")
    );
    let explicit = Request::Install {
        cli: true,
        app: false,
        channel: "stable".into(),
        version: Some("1.2.2".into()),
    };
    assert!(plan(Some(&r), &older, &explicit, &choice("cpu"), &p, os).is_ok());
    let mut reordered = m.clone();
    reordered
        .components
        .get_mut("cli")
        .unwrap()
        .get_mut(&format!("{}-cpu", p.target))
        .unwrap()
        .files
        .reverse();
    assert!(
        plan(
            Some(&r),
            &reordered,
            &Request::Update,
            &choice("cpu"),
            &p,
            os
        )
        .unwrap()
        .downloads
        .is_empty()
    );
}
#[test]
fn windows_links_install_runtime_update_and_rollback_use_current_runtime() {
    let os = Os::Windows;
    let mut p = paths(os);
    let m = release(&p);
    let first = plan(None, &m, &install(true, true), &choice("cuda"), &p, os).unwrap();
    assert_eq!(
        first
            .steps
            .iter()
            .filter(|s| matches!(s, Step::Link { .. }))
            .count(),
        2
    );
    let r = first.to;
    observed(&mut p, &r);
    let mut next = m.clone();
    let runtime = &mut next
        .runtimes
        .get_mut("cuda")
        .unwrap()
        .get_mut(&format!("{}-cuda", p.target))
        .unwrap()
        .artifact;
    runtime.files[0].1 = "b".repeat(64);
    let changed = plan(Some(&r), &next, &Request::Update, &choice("cuda"), &p, os).unwrap();
    assert_eq!(downloads(&changed), BTreeSet::from(["cuda"]));
    for c in ["cli", "app"] {
        let link = p.root.join(c).join("runtime.dll");
        let moved = changed
            .steps
            .iter()
            .position(|s| matches!(s,Step::Rename{from,..} if from==&link))
            .unwrap();
        let linked = changed
            .steps
            .iter()
            .position(|s| matches!(s,Step::Link{to,..} if to==&link))
            .unwrap();
        assert!(moved < linked);
    }
    assert_eq!(
        changed
            .steps
            .iter()
            .filter(|s| matches!(s, Step::SelfTest { .. }))
            .count(),
        2
    );
    let mut previous = r.components["cli"].clone();
    previous.version = "1.2.2".into();
    let prev = Previous {
        component: previous,
        links: vec![(p.root.join("cli/old.dll"), p.root.join("cuda/old.dll"))],
    };
    p.previous.insert("cli".into(), prev);
    p.existing.insert(p.root.join(".previous/cli"));
    p.existing.insert(p.root.join(".previous/cli/old.dll"));
    p.existing.insert(p.previous_meta("cli"));
    let rolled = plan(
        Some(&r),
        &m,
        &Request::Rollback {
            component: "cli".into(),
        },
        &choice("cuda"),
        &p,
        os,
    )
    .unwrap();
    assert_eq!(rolled.to.components["cli"].version, "1.2.2");
    assert!(rolled.downloads.is_empty());
    assert_eq!(rolled.to.components["cuda"], r.components["cuda"]);
    assert!(
        rolled
            .steps
            .iter()
            .any(|s| matches!(s,Step::Rename{from,..} if from==&p.root.join("cli/old.dll")))
    );
    assert!(rolled.steps.iter().any(|s|matches!(s,Step::Link{from,to} if from==&p.root.join("cuda/runtime.dll")&&to==&p.root.join("cli/runtime.dll"))));
}
#[test]
fn uninstall_lists_only_owned_files_links_and_effects_and_never_downloads() {
    let os = Os::Windows;
    let mut p = paths(os);
    let m = release(&p);
    let r = plan(None, &m, &install(true, true), &choice("cuda"), &p, os)
        .unwrap()
        .to;
    observed(&mut p, &r);
    p.existing.insert(p.root.join("cli/user.txt"));
    p.existing.insert(p.root.join("user.txt"));
    let removed = plan(
        Some(&r),
        &m,
        &Request::Uninstall {
            component: None,
            purge: false,
        },
        &choice("cpu"),
        &p,
        os,
    )
    .unwrap();
    assert!(removed.to.components.is_empty());
    assert!(removed.downloads.is_empty());
    let mut saw_delete = false;
    let mut deletes = BTreeSet::new();
    for step in &removed.steps {
        match step {
            Step::Effect {
                action: Action::Remove,
                ..
            } => assert!(!saw_delete),
            Step::Delete { path } => {
                saw_delete = true;
                deletes.insert(path.clone());
            }
            _ => panic!("unexpected uninstall operation {step:?}"),
        }
    }
    let expected: BTreeSet<_> = r
        .components
        .values()
        .flat_map(|c| c.files.iter().map(|(n, _, _)| c.dest.join(n)))
        .chain(r.links.iter().map(|(p, _)| p.clone()))
        .collect();
    assert_eq!(deletes, expected);
    assert!(!deletes.contains(&p.root.join("cli/user.txt")));
    let mut newer = m.clone();
    newer.version = "2.0.0".into();
    let runtime = &mut newer
        .runtimes
        .get_mut("cuda")
        .unwrap()
        .get_mut(&format!("{}-cuda", p.target))
        .unwrap()
        .artifact;
    runtime.files[0].1 = "c".repeat(64);
    let partial = plan(
        Some(&r),
        &newer,
        &Request::Uninstall {
            component: Some("app".into()),
            purge: false,
        },
        &choice("cpu"),
        &p,
        os,
    )
    .unwrap();
    assert!(partial.downloads.is_empty());
    assert!(partial.to.components.contains_key("cuda"));
    assert!(partial.to.components.contains_key("cli"));
}
#[test]
fn path_and_inventory_trust_boundary_refuses_aliases_and_overflow() {
    for bad in [
        "/abs",
        "../x",
        "a/../x",
        "C:/x",
        "a\\x",
        "a//x",
        "a/./x",
        "NUL",
        "con.txt",
        "COM1",
        "LPT¹.txt",
        "a.",
        "a ",
        "x:y",
        "x?y",
        "x\0y",
    ] {
        assert!(manifest::relative_path(bad).is_err(), "{bad:?}");
    }
    assert_eq!(
        manifest::relative_path("./a/b").unwrap(),
        PathBuf::from("a/b")
    );
    let h = "a".repeat(64);
    assert!(
        manifest::inventory(&[("a".into(), h.clone(), 1), ("A".into(), h.clone(), 1)]).is_err()
    );
    assert!(
        manifest::inventory(&[("A".into(), h.clone(), 1), ("a/b".into(), h.clone(), 1)]).is_err()
    );
    assert!(manifest::inventory(&[("a".into(), "bad".into(), 1)]).is_err());
    let p = paths(Os::Linux);
    let mut m = release(&p);
    m.components
        .get_mut("cli")
        .unwrap()
        .get_mut(&format!("{}-cpu", p.target))
        .unwrap()
        .size = u64::MAX;
    assert!(
        plan(
            None,
            &m,
            &install(true, false),
            &choice("cpu"),
            &p,
            Os::Linux
        )
        .is_err()
    );
    let mut p = paths(Os::Linux);
    p.existing.insert(p.root.join("cli"));
    assert!(
        plan(
            None,
            &release(&p),
            &install(true, false),
            &choice("cpu"),
            &p,
            Os::Linux
        )
        .unwrap_err()
        .to_string()
        .contains("unrecorded destination")
    );
}

#[test]
fn planned_metadata_and_directories_survive_update_rollback_and_uninstall() {
    use larch_setup::manifest::{Error, Result};
    use larch_setup::txn::{Hooks, Journal, Transaction};
    use std::fs;
    struct Fake {
        root: PathBuf,
        to: Receipt,
    }
    impl Hooks for Fake {
        fn present(&mut self, kind: &str, value: &str) -> Result<bool> {
            match fs::read_to_string(self.root.join(format!(".fake-{kind}"))) {
                Ok(v) => Ok(v == value),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
                Err(e) => Err(e.into()),
            }
        }
        fn apply(&mut self, kind: &str, value: &str) -> Result<()> {
            fs::write(self.root.join(format!(".fake-{kind}")), value)?;
            Ok(())
        }
        fn remove(&mut self, kind: &str, value: &str) -> Result<()> {
            if self.present(kind, value)? {
                fs::remove_file(self.root.join(format!(".fake-{kind}")))?;
            }
            Ok(())
        }
        fn self_test(&mut self, name: &str) -> Result<()> {
            let c = &self.to.components[name];
            let text = fs::read_to_string(c.dest.join(&c.files[0].0))?;
            if text != c.version {
                return Err(Error::Refused("fake executable version mismatch".into()));
            }
            Ok(())
        }
    }
    fn scan(root: &std::path::Path, p: &mut Paths) {
        fn walk(dir: &std::path::Path, set: &mut BTreeSet<PathBuf>) {
            for e in fs::read_dir(dir).unwrap() {
                let path = e.unwrap().path();
                set.insert(path.clone());
                if path.is_dir() {
                    walk(&path, set);
                }
            }
        }
        p.existing.clear();
        walk(root, &mut p.existing);
        p.previous.clear();
        for name in ["cli", "app", "setup"] {
            let meta = p.previous_meta(name);
            if meta.exists() {
                p.previous.insert(
                    name.into(),
                    serde_json::from_slice(&fs::read(meta).unwrap()).unwrap(),
                );
            }
        }
    }
    fn execute(t: &Transaction, p: &Paths, plan: Plan) -> Receipt {
        for (name, _) in &plan.downloads {
            let c = &plan.to.components[name];
            for (file, _, _) in &c.files {
                let target = p.stage(name, Os::Linux).join(file);
                fs::create_dir_all(target.parent().unwrap()).unwrap();
                fs::write(target, &c.version).unwrap();
            }
        }
        for (path, bytes) in &plan.metadata {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, bytes).unwrap();
        }
        let mut hooks = Fake {
            root: p.root.clone(),
            to: plan.to.clone(),
        };
        t.run(
            Journal::new(
                p.txn.clone(),
                t.receipt_digest().unwrap(),
                plan.to,
                plan.steps,
            ),
            &mut hooks,
        )
        .unwrap()
    }
    let root = std::env::temp_dir().join(format!("larch-plan-roundtrip-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let t = Transaction::acquire(&root).unwrap();
    let mut p = paths(Os::Linux);
    p.root = t.root().into();
    let mut m = release(&p);
    let first = plan(
        None,
        &m,
        &install(true, true),
        &choice("cpu"),
        &p,
        Os::Linux,
    )
    .unwrap();
    let r = execute(&t, &p, first);
    fs::write(p.root.join("cli/user-file"), "preserve").unwrap();
    fs::write(p.root.join("user-file"), "preserve").unwrap();
    scan(t.root(), &mut p);
    p.txn = "update".into();
    m.version = "1.2.4".into();
    let a = m
        .components
        .get_mut("cli")
        .unwrap()
        .get_mut(&format!("{}-cpu", p.target))
        .unwrap();
    a.url = a.url.replace("1.2.3", "1.2.4");
    a.files[0].1 = "b".repeat(64);
    let update = plan(
        Some(&r),
        &m,
        &Request::Update,
        &choice("cpu"),
        &p,
        Os::Linux,
    )
    .unwrap();
    let r = execute(&t, &p, update);
    assert_eq!(r.components["cli"].version, "1.2.4");
    assert_eq!(
        fs::read_to_string(p.root.join(".previous/cli/user-file")).unwrap(),
        "preserve"
    );
    scan(t.root(), &mut p);
    p.txn = "rollback".into();
    let rollback = plan(
        Some(&r),
        &m,
        &Request::Rollback {
            component: "cli".into(),
        },
        &choice("cpu"),
        &p,
        Os::Linux,
    )
    .unwrap();
    let r = execute(&t, &p, rollback);
    assert_eq!(r.components["cli"].version, "1.2.3");
    assert_eq!(
        fs::read_to_string(p.root.join("cli/user-file")).unwrap(),
        "preserve"
    );
    scan(t.root(), &mut p);
    p.txn = "uninstall".into();
    let uninstall = plan(
        Some(&r),
        &m,
        &Request::Uninstall {
            component: None,
            purge: false,
        },
        &choice("cpu"),
        &p,
        Os::Linux,
    )
    .unwrap();
    let r = execute(&t, &p, uninstall);
    assert!(r.components.is_empty());
    assert_eq!(
        fs::read_to_string(p.root.join("cli/user-file")).unwrap(),
        "preserve"
    );
    assert!(p.root.join("user-file").exists());
    assert!(!p.root.join(".previous").exists());
    drop(t);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn rollback_across_flavours_reconciles_runtime_references() {
    let os = Os::Windows;
    let p = paths(os);
    let m = release(&p);
    for (current_flavour, previous_flavour) in [("cpu", "cuda"), ("cuda", "cpu")] {
        let mut p = paths(os);
        let r = plan(
            None,
            &m,
            &install(true, false),
            &choice(current_flavour),
            &p,
            os,
        )
        .unwrap()
        .to;
        observed(&mut p, &r);
        p.previous.clear();
        let mut previous = r.components["cli"].clone();
        previous.flavour = previous_flavour.into();
        previous.version = "1.2.2".into();
        p.previous.insert(
            "cli".into(),
            Previous {
                component: previous,
                links: Vec::new(),
            },
        );
        p.existing.insert(p.root.join(".previous/cli"));
        p.existing.insert(p.previous_meta("cli"));
        let rolled = plan(
            Some(&r),
            &m,
            &Request::Rollback {
                component: "cli".into(),
            },
            &choice("cpu"),
            &p,
            os,
        )
        .unwrap();
        assert_eq!(rolled.to.components["cli"].flavour, previous_flavour);
        assert_eq!(
            rolled.to.components.contains_key("cuda"),
            previous_flavour == "cuda"
        );
        assert_eq!(
            downloads(&rolled),
            if previous_flavour == "cuda" {
                BTreeSet::from(["cuda"])
            } else {
                BTreeSet::new()
            }
        );
    }
}

#[test]
fn model_purge_is_explicit_and_space_headroom_rounds_up() {
    let os = Os::Linux;
    let mut p = paths(os);
    let m = release(&p);
    let mut r = plan(None, &m, &install(true, false), &choice("cpu"), &p, os)
        .unwrap()
        .to;
    r.models.push(larch_setup::txn::Model {
        file: "/data/models/m1.gguf".into(),
        repo: "test/repo".into(),
        revision: "a".repeat(40),
        sha256: "b".repeat(64),
    });
    observed(&mut p, &r);
    let kept = plan(
        Some(&r),
        &m,
        &Request::Uninstall {
            component: None,
            purge: false,
        },
        &choice("cpu"),
        &p,
        os,
    )
    .unwrap();
    assert!(kept.purge_models.is_empty());
    let mut purged = plan(
        Some(&r),
        &m,
        &Request::Uninstall {
            component: None,
            purge: true,
        },
        &choice("cpu"),
        &p,
        os,
    )
    .unwrap();
    assert_eq!(purged.purge_models.len(), 1);
    assert!(
        !purged
            .steps
            .iter()
            .any(|s| matches!(s,Step::Delete{path} if path.starts_with("/data")))
    );
    purged.download_size = 1;
    purged.unpacked_size = 0;
    assert_eq!(purged.required_space().unwrap(), 2);
}
