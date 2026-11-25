//! Detection fixtures exercise each parser and the release selection policy without a GPU.
mod support;
use larch_setup::{
    command::{self, Output},
    detect::*,
};
use std::{collections::BTreeMap, io, process::Command, time::Duration};
macro_rules! fixture {
    ($name:literal) => {
        include_str!(concat!("fixtures/detect/", $name, ".txt"))
    };
}
fn machine() -> Machine {
    Machine {
        os: Os::Linux,
        target: "x86_64-unknown-linux-gnu".into(),
        os_version: String::new(),
        arm64: false,
        glibc: Some("2.39".into()),
        glibcxx: Some("3.4.30".into()),
        memory: 32 * 1024 * 1024 * 1024,
        gpus: Vec::new(),
        kfd: KfdAccess::Ready,
    }
}
#[test]
fn every_parser_accepts_its_fixture_and_rejects_a_bad_line() {
    for csv in [
        fixture!("nvidia-3050"),
        fixture!("nvidia-4060"),
        fixture!("nvidia-5070-ti"),
        fixture!("nvidia-2080"),
        fixture!("nvidia-two"),
    ] {
        assert!(!nvidia_smi(csv).unwrap().is_empty());
    }
    for bad in [
        "x,name,8192,580.65,8.6",
        "0,name,-1,580.65,8.6",
        "0,name,1,580.bad,8.6",
        "0,name,1,580.65,8.x",
        "0,name,1,580.65,8.6,extra",
    ] {
        assert!(nvidia_smi(bad).unwrap_err().to_string().contains(bad));
    }
    let gpu = proc_nvidia(fixture!("proc-version"), fixture!("proc-information")).unwrap();
    assert!(!gpu.verified && gpu.cc.is_none());
    assert!(proc_nvidia("nonsense", fixture!("proc-information")).is_err());
    assert!(proc_nvidia(fixture!("proc-version"), "bad line").is_err());
    for (n, gfx) in [
        (100300, "gfx1030"),
        (110000, "gfx1100"),
        (120001, "gfx1201"),
        (100302, "gfx1032"),
    ] {
        assert_eq!(
            kfd_node(
                &format!("gfx_target_version {n}"),
                &[fixture!("kfd-bank"), "heap_type 0\nsize_in_bytes 100"]
            )
            .unwrap()
            .unwrap()
            .gfx
            .as_deref(),
            Some(gfx)
        );
    }
    let gpu = kfd_node(
        fixture!("kfd-node"),
        &[
            fixture!("kfd-bank"),
            "heap_type 2\nsize_in_bytes 1024",
            "heap_type 3\nsize_in_bytes 1000",
        ],
    )
    .unwrap()
    .unwrap();
    assert_eq!(gpu.memory, 12884901888 + 1024);
    assert!(kfd_node("gfx_target_version 0", &[]).unwrap().is_none());
    for bad in ["gfx_target_version bad", "gfx_target_version 111600", "bad"] {
        assert!(kfd_node(bad, &[]).is_err());
    }
    assert!(kfd_node(fixture!("kfd-node"), &["heap_type 1"]).is_err());
    assert_eq!(kfd_access(None), KfdAccess::Ready);
    assert_eq!(
        kfd_access(Some(io::ErrorKind::NotFound)),
        KfdAccess::NoDriver
    );
    assert_eq!(
        kfd_access(Some(io::ErrorKind::PermissionDenied)),
        KfdAccess::NeedRenderGroup
    );
    let pnp = "PCI\\VEN_1002&DEV_744C";
    let gpu = windows_amd(
        fixture!("windows-amd"),
        &BTreeMap::from([(pnp.into(), 12884901888)]),
    )
    .unwrap()
    .remove(0);
    assert_eq!(gpu.memory, 12884901888);
    assert_eq!(
        windows_amd(fixture!("windows-amd"), &BTreeMap::new()).unwrap()[0].memory,
        0
    );
    assert!(windows_amd("{bad}", &BTreeMap::new()).is_err());
    assert!(windows_amd("{}", &BTreeMap::new()).is_err());
    assert_eq!(
        macos(
            fixture!("macos-version"),
            fixture!("macos-memory"),
            fixture!("macos-arm64")
        )
        .unwrap(),
        ("26.0".into(), 34359738368, true)
    );
    for args in [
        ("bad", "100", "1"),
        ("26.0", "bad", "1"),
        ("26.0", "100", "2"),
    ] {
        assert!(macos(args.0, args.1, args.2).is_err());
    }
    assert_eq!(glibc(fixture!("glibc")).unwrap(), "2.39");
    assert!(glibc("libc 2.39").is_err());
    assert_eq!(glibcxx(fixture!("glibcxx")).unwrap(), "3.4.30");
    assert!(glibcxx("GLIBCXX_bad").is_err());
    assert_eq!(meminfo(fixture!("meminfo")).unwrap(), 32768000 * 1024);
    assert!(meminfo("MemTotal: 1 MB").is_err());
    assert_eq!(df_pk(fixture!("df")).unwrap(), 700000 * 1024);
    assert_eq!(
        df_pk("server:/disk with spaces 1000 300 700 30% /mount with spaces").unwrap(),
        700 * 1024
    );
    assert!(df_pk("malformed df").is_err());
    assert_eq!(ps_free(fixture!("ps-free")).unwrap(), 123456789012);
    assert!(ps_free("bad").is_err());
    assert!(version_at_least("580.65.06", "580.65").unwrap());
    assert!(!version_at_least("580.64.99", "580.65").unwrap());
    assert!(version_at_least("580.bad", "580.65").is_err());
}
#[test]
fn chooses_gated_supported_gpus_and_explains_cpu_fallbacks() {
    let mut m = machine();
    let mut release = support::manifest();
    for csv in [
        fixture!("nvidia-3050"),
        fixture!("nvidia-4060"),
        fixture!("nvidia-5070-ti"),
    ] {
        m.gpus = nvidia_smi(csv).unwrap();
        assert_eq!(choose(&m, &release, None).unwrap().flavour, "cuda");
    }
    m.gpus = nvidia_smi(fixture!("nvidia-2080")).unwrap();
    let c = choose(&m, &release, None).unwrap();
    assert_eq!(c.flavour, "cpu");
    assert!(c.why.contains("7.5") && c.why.contains("slow"));
    m.gpus = nvidia_smi(fixture!("nvidia-3050")).unwrap();
    m.gpus[0].driver = Some("579.99".into());
    assert_eq!(choose(&m, &release, None).unwrap().flavour, "cpu");
    m.gpus = vec![proc_nvidia(fixture!("proc-version"), fixture!("proc-information")).unwrap()];
    assert!(
        choose(&m, &release, None)
            .unwrap()
            .why
            .contains("unverified")
    );
    m.gpus = vec![
        kfd_node(fixture!("kfd-node"), &[fixture!("kfd-bank")])
            .unwrap()
            .unwrap(),
    ];
    assert_eq!(choose(&m, &release, None).unwrap().flavour, "hip");
    release.gated.retain(|f| f != "hip");
    assert_eq!(choose(&m, &release, None).unwrap().flavour, "cpu");
    assert!(choose(&m, &release, Some("hip")).is_err());
    release.gated.push("hip".into());
    m.gpus[0].gfx = Some("gfx1031".into());
    assert_eq!(choose(&m, &release, None).unwrap().flavour, "cpu");
    m.gpus[0].gfx = Some("gfx1100".into());
    m.kfd = KfdAccess::NeedRenderGroup;
    assert!(
        choose(&m, &release, None)
            .unwrap()
            .why
            .contains("render group")
    );
    m.kfd = KfdAccess::Ready;
    m.gpus.extend(nvidia_smi(fixture!("nvidia-two")).unwrap());
    let c = choose(&m, &release, None).unwrap();
    assert_eq!(c.flavour, "cuda");
    assert!(c.gpu.unwrap().name.contains("5070"));
    assert_eq!(choose(&m, &release, Some("cpu")).unwrap().flavour, "cpu");
    assert!(choose(&m, &release, Some("unbuilt")).is_err());
}
#[test]
fn os_floors_precede_overrides_and_windows_amd_uses_manifest_ids() {
    let release = support::manifest();
    let mut m = machine();
    m.glibc = Some("2.27".into());
    assert!(
        choose(&m, &release, Some("cpu"))
            .unwrap_err()
            .to_string()
            .contains("glibc floor 2.28")
    );
    m = machine();
    m.glibcxx = None;
    assert!(choose(&m, &release, None).is_err());
    m = machine();
    m.os = Os::Macos;
    m.target = "aarch64-apple-darwin".into();
    m.os_version = "26.0".into();
    m.arm64 = true;
    assert_eq!(choose(&m, &release, None).unwrap().flavour, "metal");
    m.os_version = "25.9".into();
    assert!(choose(&m, &release, None).is_err());
    m.os_version = "26.0".into();
    m.arm64 = false;
    assert!(
        choose(&m, &release, None)
            .unwrap_err()
            .to_string()
            .contains("Intel Mac")
    );
    m = machine();
    m.os = Os::Windows;
    m.target = "x86_64-pc-windows-msvc".into();
    m.os_version = "22631".into();
    m.gpus = windows_amd(fixture!("windows-amd"), &BTreeMap::new()).unwrap();
    let c = choose(&m, &release, None).unwrap();
    assert_eq!(c.flavour, "hip");
    assert_eq!(c.gpu.unwrap().gfx.as_deref(), Some("gfx1100"));
    m.gpus.extend(nvidia_smi(fixture!("nvidia-4060")).unwrap());
    assert_eq!(
        choose(&m, &release, Some("hip"))
            .unwrap()
            .gpu
            .unwrap()
            .gfx
            .as_deref(),
        Some("gfx1100")
    );
    m.os_version = "19045".into();
    assert!(
        choose(&m, &release, None)
            .unwrap_err()
            .to_string()
            .contains("Windows build floor")
    );
}
#[test]
fn bounded_runner_distinguishes_missing_exit_and_timeout() {
    assert!(matches!(
        command::run(
            &mut Command::new("larch-no-such-tool-987654"),
            Duration::from_millis(100)
        )
        .unwrap(),
        Output::Missing
    ));
    #[cfg(unix)]
    {
        assert!(
            matches!(command::run(Command::new("sh").args(["-c","printf ok; printf err >&2; exit 3"]),Duration::from_secs(1)).unwrap(),Output::Finished{success:false,stdout,stderr} if stdout=="ok"&&stderr=="err")
        );
        assert!(matches!(
            command::run(
                Command::new("sh").args(["-c", "exec sleep 10"]),
                Duration::from_millis(40)
            )
            .unwrap(),
            Output::TimedOut
        ));
    }
}
#[cfg(target_os = "linux")]
#[test]
fn current_cpu_host_observation_and_free_space_are_readable() {
    let m = detect_machine().unwrap();
    assert_eq!(m.os, Os::Linux);
    assert!(m.memory > 0);
    assert!(free_space(&std::env::temp_dir()).unwrap() > 0);
}
