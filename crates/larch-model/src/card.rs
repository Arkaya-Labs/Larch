//! The card profile: the gated routes `larch routes-check` proved on one card, per model (each op
//! within the CPU reference's tolerance and more than 5 % faster than the route it replaces at that
//! model's shapes, decode graphs replaying, then `larch parity` passing with them on). One small
//! file in the data folder, keyed by the card's name, architecture, SM count, CUDA API level and
//! driver build, and by the larch build that wrote it; another card, driver or build ignores it,
//! saying why on stderr. A profile only adds routes, and only for the model they were proved on:
//! with none, routing is the measured arch defaults. The environment switches (`LARCH_QUANT_I8`)
//! still win, and `LARCH_CARD_PROFILE=0` turns the profile off.

use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once, OnceLock};

use larch_kernels::{Backend, DeviceProps};
use serde::{Deserialize, Serialize};

/// The profile's file name in the data folder.
pub const FILE: &str = "card-routes.json";

/// The build a profile was proved under: the source commit (`unknown` in a source archive without
/// Git metadata).
// ponytail: the commit misses uncommitted kernel edits; key on a kernel-source hash if local
// builds start sharing a data folder with releases.
pub const BUILD: &str = env!("LARCH_COMMIT");

/// A route the code gates by architecture (on by default only where it was measured: sm_120, or
/// [`Route::DecodeGraphs`]'s arch list in the engine).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Route {
    /// The direct quantized GEMM's f16 tiles, in their measured row windows.
    QuantGemm,
    /// INT8's plain tile on full grids up to 32 rows (Q6_K too; elsewhere only Q4_K's).
    I8FullGrid,
    /// INT8's Stream-K on short grids up to 32 rows.
    I8StreamK,
    /// INT8's plain tile past 32 rows on grids over 8 tiles.
    I8Long,
    /// Captured decode graphs on by default (still fit-gated).
    DecodeGraphs,
}

impl Route {
    /// Every gated route, in the order `routes-check` reports them.
    pub const ALL: [Route; 5] = [
        Route::QuantGemm,
        Route::I8FullGrid,
        Route::I8StreamK,
        Route::I8Long,
        Route::DecodeGraphs,
    ];

    /// The route's name in the profile and in `routes-check`'s report.
    pub fn name(self) -> &'static str {
        match self {
            Route::QuantGemm => "quant_gemm",
            Route::I8FullGrid => "i8_full_grid",
            Route::I8StreamK => "i8_stream_k",
            Route::I8Long => "i8_long",
            Route::DecodeGraphs => "decode_graphs",
        }
    }
}

/// What a profile must match to apply: the card, its driver and the larch build.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Card {
    /// The device's name.
    pub name: String,
    /// The SM architecture, such as 86.
    pub arch_code: u32,
    /// The SM count.
    pub compute_units: u32,
    /// The CUDA API level the driver supports (`cudaDriverGetVersion`, such as 13000).
    pub driver_version: u32,
    /// The installed driver build (NVML, such as "581.57"); None where NVML cannot say.
    pub driver_build: Option<String>,
    /// [`BUILD`] when proved.
    pub build: String,
}

impl Card {
    /// This process's card, driver and build.
    pub fn of(props: &DeviceProps) -> Card {
        Card {
            name: props.name.clone(),
            arch_code: props.arch_code,
            compute_units: props.compute_units,
            driver_version: props.driver_version,
            driver_build: larch_sys::nvml::Nvml::open(&props.name).and_then(|n| n.driver_version()),
            build: BUILD.into(),
        }
    }
}

/// The routes proved on one model ([`model_key`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Proved {
    /// The model's [`model_key`].
    pub model: String,
    /// The routes on for it.
    pub on: Vec<Route>,
}

/// The routes proved on one card, per model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    /// The card it was proved on.
    pub card: Card,
    /// The models it was proved on.
    pub models: Vec<Proved>,
}

impl Profile {
    /// `earlier` (kept only when proved on `card`) with `model`'s routes set to `on`.
    pub fn with(earlier: Option<Profile>, card: Card, model: String, on: Vec<Route>) -> Profile {
        let mut models = earlier
            .filter(|p| p.card == card)
            .map_or_else(Vec::new, |p| p.models);
        models.retain(|m| m.model != model);
        models.push(Proved { model, on });
        Profile { card, models }
    }
}

/// A model's identity in a profile: its file name and size.
// ponytail: name and size, not a hash of a many-GiB file; hash the header if two quantizations
// of one name and size ever meet.
pub fn model_key(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_string_lossy();
    Some(format!("{name}:{}", std::fs::metadata(path).ok()?.len()))
}

/// Shapes a route never takes off sm_120, whatever a profile says: the route, its output width,
/// its rows, and why.
pub const EXCLUDED: [(Route, i64, RangeInclusive<i64>, &str); 1] = [(
    Route::I8StreamK,
    512,
    28..=32,
    "sm_86 passes its op test but misses m3 `larch parity --mode decode --parallel 32` (decisive \
     top-1 0.984); lift only after a model-level parity run passes",
)];

/// Why `route` may not take an output `n` wide at `rows` on `arch_code`, if it may not.
pub fn excluded(route: Route, n: i64, rows: i64, arch_code: u32) -> Option<&'static str> {
    if arch_code == 120 {
        return None;
    }
    EXCLUDED
        .iter()
        .find(|e| e.0 == route && e.1 == n && e.2.contains(&rows))
        .map(|e| e.3)
}

/// The profile at `path`: Ok(None) when there is none, Err(why) when it cannot be read.
pub fn read(path: &Path) -> Result<Option<Profile>, String> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|e| format!("{} is not a profile: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{} cannot be read: {e}", path.display())),
    }
}

/// Writes `profile` to `path`, creating its folder: a temporary file beside it, then a rename, so
/// a reader never sees half a file.
pub fn write(path: &Path, profile: &Profile) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)?;
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    let written = std::fs::write(&tmp, serde_json::to_vec_pretty(profile)?)
        .and_then(|()| std::fs::rename(&tmp, path));
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written
}

/// The routes `profile` turns on for `model` on `card`, or why it turns none on.
pub fn for_card(profile: &Profile, card: &Card, model: &str) -> Result<Vec<Route>, String> {
    let had = &profile.card;
    let why = if had.name != card.name
        || had.arch_code != card.arch_code
        || had.compute_units != card.compute_units
    {
        format!("it was proved on a {} (sm_{})", had.name, had.arch_code)
    } else if had.driver_version != card.driver_version || had.driver_build != card.driver_build {
        let build = |c: &Card| c.driver_build.clone().unwrap_or_else(|| "unknown".into());
        format!(
            "it was proved under driver {} (CUDA {}), not {} (CUDA {})",
            build(had),
            had.driver_version,
            build(card),
            card.driver_version
        )
    } else if had.build != card.build {
        format!("it was proved by build {}, not {}", had.build, card.build)
    } else {
        return Ok((profile.models.iter())
            .find(|m| m.model == model)
            .map_or_else(Vec::new, |m| m.on.clone()));
    };
    Err(format!("{why}: run `larch routes-check` again"))
}

/// The profile's path: the per-user data folder, by `larch_fetch::data_dir`'s rule (this crate
/// links no network code).
pub fn path() -> Option<PathBuf> {
    #[cfg(windows)]
    let base = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    #[cfg(target_os = "macos")]
    let base =
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"));
    #[cfg(all(unix, not(target_os = "macos")))]
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|dir| dir.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")));
    Some(base?.join("larch").join(FILE))
}

/// The routes [`routes`] returns in place of the file's while set: `routes-check` sets none while
/// it measures and the candidates while it proves them.
static CHECKING: Mutex<Option<Vec<Route>>> = Mutex::new(None);

/// While `on` is Some, every load in this process takes exactly its routes, whatever the file says.
pub fn set_checking(on: Option<Vec<Route>>) {
    *CHECKING.lock().unwrap_or_else(|e| e.into_inner()) = on;
}

/// This card's profiled routes for `model`. The file is read once per process, only for a CUDA
/// device, and not with `LARCH_CARD_PROFILE=0`; one stderr line names the routes a profile turns
/// on, or why it turns none on.
// ponytail: the line prints once per process; once per model if a long-lived process swaps
// models and someone needs each named.
pub fn routes(props: &DeviceProps, model: &Path) -> Vec<Route> {
    static PROFILE: OnceLock<Result<Option<(Profile, Card)>, String>> = OnceLock::new();
    static SAID: Once = Once::new();
    if props.backend != Backend::Cuda {
        return Vec::new();
    }
    if let Some(on) = CHECKING.lock().unwrap_or_else(|e| e.into_inner()).clone() {
        return on;
    }
    let profile = PROFILE.get_or_init(|| {
        if std::env::var_os("LARCH_CARD_PROFILE").is_some_and(|v| v == "0") {
            return Ok(None);
        }
        let Some(path) = path() else { return Ok(None) };
        Ok(read(&path)?.map(|p| (p, Card::of(props))))
    });
    let on = match profile {
        Ok(Some((p, card))) => {
            for_card(p, card, &model_key(model).unwrap_or_default()).map_err(|why| why.clone())
        }
        Ok(None) => Ok(Vec::new()),
        Err(why) => Err(why.clone()),
    };
    match on {
        Ok(on) => {
            if !on.is_empty() {
                let names: Vec<_> = on.iter().map(|r| r.name()).collect();
                SAID.call_once(|| eprintln!("card profile: {} on", names.join(", ")));
            }
            on
        }
        Err(why) => {
            SAID.call_once(|| eprintln!("card profile ignored: {why}"));
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(name: &str, arch_code: u32, units: u32, api: u32, driver: &str, build: &str) -> Card {
        Card {
            name: name.into(),
            arch_code,
            compute_units: units,
            driver_version: api,
            driver_build: Some(driver.into()),
            build: build.into(),
        }
    }

    fn temp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("larch-card-{name}-{}", std::process::id()))
    }

    #[test]
    fn a_profile_round_trips_for_its_own_card_and_model() {
        let dir = temp("round-trip");
        let path = dir.join(FILE);
        let c = card("RTX 3050", 86, 16, 13000, "581.57", "abc");
        let on = vec![Route::QuantGemm, Route::I8StreamK];
        let p = Profile::with(None, c.clone(), "m.gguf:1".into(), on.clone());
        write(&path, &p).unwrap();
        let back = read(&path).unwrap().unwrap();
        assert_eq!(for_card(&back, &c, "m.gguf:1"), Ok(on));
        // Another model of the same card takes none of them.
        assert_eq!(for_card(&back, &c, "n.gguf:1"), Ok(Vec::new()));
        // No temporary file is left beside it.
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn another_card_driver_or_build_ignores_the_profile_and_says_why() {
        let c = card("RTX 3050", 86, 16, 13000, "581.57", "abc");
        let p = Profile::with(None, c.clone(), "m".into(), vec![Route::QuantGemm]);
        for other in [
            card("RTX 4060", 86, 16, 13000, "581.57", "abc"),
            card("RTX 3050", 89, 16, 13000, "581.57", "abc"),
            card("RTX 3050", 86, 20, 13000, "581.57", "abc"),
            card("RTX 3050", 86, 16, 13010, "581.57", "abc"),
            card("RTX 3050", 86, 16, 13000, "581.80", "abc"),
            card("RTX 3050", 86, 16, 13000, "581.57", "def"),
        ] {
            let why = for_card(&p, &other, "m").unwrap_err();
            assert!(why.ends_with("run `larch routes-check` again"), "{why}");
        }
        // A missing file is no profile; an unreadable one says why.
        let dir = temp("mismatch");
        assert_eq!(read(&dir.join("absent.json")), Ok(None));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(FILE);
        std::fs::write(&path, "{not json").unwrap();
        assert!(read(&path).unwrap_err().contains("is not a profile"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A second model's check keeps the first's routes on the same card, and a check on another
    /// card or build starts over.
    #[test]
    fn a_check_replaces_only_its_own_models_routes() {
        let c = card("RTX 3050", 86, 16, 13000, "581.57", "abc");
        let first = Profile::with(None, c.clone(), "a".into(), vec![Route::QuantGemm]);
        let both = Profile::with(Some(first), c.clone(), "b".into(), vec![Route::I8Long]);
        assert_eq!(for_card(&both, &c, "a"), Ok(vec![Route::QuantGemm]));
        assert_eq!(for_card(&both, &c, "b"), Ok(vec![Route::I8Long]));
        let again = Profile::with(Some(both.clone()), c.clone(), "a".into(), Vec::new());
        assert_eq!(for_card(&again, &c, "a"), Ok(Vec::new()));
        assert_eq!(for_card(&again, &c, "b"), Ok(vec![Route::I8Long]));
        let rebuilt = card("RTX 3050", 86, 16, 13000, "581.57", "def");
        let fresh = Profile::with(Some(both), rebuilt.clone(), "b".into(), Vec::new());
        assert_eq!(fresh.models.len(), 1);
        assert_eq!(for_card(&fresh, &rebuilt, "a"), Ok(Vec::new()));
    }

    #[test]
    fn the_exclusion_table_holds_the_512_wide_stream_k_off_sm120() {
        for arch in [86, 89, 121] {
            for rows in 28..=32 {
                assert!(excluded(Route::I8StreamK, 512, rows, arch).is_some());
            }
            assert!(excluded(Route::I8StreamK, 512, 33, arch).is_none());
            assert!(excluded(Route::I8StreamK, 1024, 28, arch).is_none());
            assert!(excluded(Route::I8FullGrid, 512, 28, arch).is_none());
        }
        assert!(excluded(Route::I8StreamK, 512, 28, 120).is_none());
    }
}
