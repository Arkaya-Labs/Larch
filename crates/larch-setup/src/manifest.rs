//! Signed release metadata and the primary/backup trust policy.

use minisign_verify::{PublicKey, Signature};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::BTreeMap};

/// The newest release schema understood by this installer.
pub const SUPPORTED_SCHEMA: u32 = 1;
/// The actionable error for releases requiring a newer installer.
pub const NEWER_SETUP: &str = "a newer larch-setup is needed: run the install command again";

/// All installer refusals retain their cause and an actionable description.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Another installer or installed process holds the installation lock.
    #[error("another larch-setup is running")]
    Busy,
    /// Installed programs hold the shared installation lock.
    #[error("installation is in use by {0}; close these programs and retry")]
    InUse(String),
    /// The committed state no longer matches the transaction's starting point.
    #[error("state was changed outside setup; run `larch-setup repair`")]
    StateChangedOutside,
    /// A filesystem operation failed.
    #[error("filesystem operation failed: {0}")]
    Io(#[from] std::io::Error),
    /// Metadata is not valid JSON for its schema.
    #[error("invalid release JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// A trust or compatibility rule refused the operation.
    #[error("{0}")]
    Refused(String),
}
/// Installer results share one error type.
pub type Result<T> = std::result::Result<T, Error>;

/// One setup executable advertised by a channel pointer.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SetupArtifact {
    /// Its authenticated download location.
    pub url: String,
    /// Its SHA-256 in hexadecimal.
    pub sha256: String,
    /// Its exact byte count.
    pub size: u64,
}

/// The signed channel head and the installer needed to consume it.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Pointer {
    /// The pointer wire format, currently one.
    pub schema: u32,
    /// The signed release channel.
    pub channel: String,
    /// The offered semantic version.
    pub version: String,
    /// The manifest's download location.
    pub manifest_url: String,
    /// The manifest's exact SHA-256.
    pub manifest_sha256: String,
    /// Setup executables indexed by target triple.
    pub setup: BTreeMap<String, SetupArtifact>,
}

/// A signed component archive and its complete installed file inventory.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Artifact {
    /// Its download location.
    pub url: String,
    /// Its archive SHA-256.
    pub sha256: String,
    /// Its compressed size in bytes.
    pub size: u64,
    /// Relative file paths with content hashes and unpacked byte counts.
    pub files: Vec<(String, String, u64)>,
}

/// A runtime archive also identifies the upstream material it was repackaged from.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RuntimeArtifact {
    /// The archive's authenticated contents.
    #[serde(flatten)]
    pub artifact: Artifact,
    /// Upstream URL, hash or pinned image digest.
    pub provenance: String,
}

/// A release owner's hardware and driver support rule.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GpuRule {
    /// The GPU vendor, independent of its display name.
    pub vendor: String,
    /// Inclusive minimum compute capability, as a dotted version.
    pub cc_min: String,
    /// Exclusive upper compute capability, when bounded.
    pub cc_max_excl: Option<String>,
    /// Minimum supported driver version.
    pub driver_min: String,
    /// The backend this rule enables.
    pub flavour: String,
    /// Release-owner guidance, including PTX JIT expectations.
    pub note: String,
}

/// The operating-system and ABI floors measured or gated for a release.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Floors {
    /// Minimum macOS product version.
    pub macos_min: String,
    /// Minimum GNU libc version.
    pub glibc_min: String,
    /// Minimum libstdc++ symbol version.
    pub glibcxx_min: String,
    /// Minimum Windows build number.
    pub windows_min_build: u32,
}

/// A revision-pinned optional model download shown after installation.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CatalogModel {
    /// A short display name for the selection menu.
    pub display: String,
    /// The source repository.
    pub repo: String,
    /// The file within the repository.
    pub file: String,
    /// The immutable source revision.
    pub revision: String,
    /// The expected file SHA-256.
    pub sha256: String,
    /// The exact file size.
    pub size: u64,
    /// The minimum machine memory recommended by the catalog.
    pub min_mem_gb: u64,
    /// The license the user must acknowledge before downloading.
    pub license_url: String,
}

/// A signed release's artifacts, gates, compatibility rules and optional models.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Manifest {
    /// The release wire format.
    pub schema: u32,
    /// The oldest installer able to consume this release.
    pub min_setup: String,
    /// The release's semantic version.
    pub version: String,
    /// The release's channel.
    pub channel: String,
    /// Component archives indexed by component and target-flavour pair.
    pub components: BTreeMap<String, BTreeMap<String, Artifact>>,
    /// Runtime archives indexed by runtime and target-flavour pair.
    pub runtimes: BTreeMap<String, BTreeMap<String, RuntimeArtifact>>,
    /// Ordered GPU selection rules.
    pub gpu_rules: Vec<GpuRule>,
    /// AMD device IDs mapped to graphics targets.
    pub amd_ids: BTreeMap<String, String>,
    /// OS and ABI compatibility requirements.
    pub floors: Floors,
    /// Backend flavours built for each target.
    pub flavours: BTreeMap<String, Vec<String>>,
    /// Only these human-gated flavours may be installed.
    pub gated: Vec<String>,
    /// AMD graphics targets packaged for this release.
    pub hip_targets: Vec<String>,
    /// Optional, licensed model downloads.
    pub models: Vec<CatalogModel>,
}

/// Independently authenticated primary and recovery public keys.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct KeySet {
    /// A monotonically increasing keyset revision.
    pub version: u64,
    /// The only key allowed to sign releases.
    pub primary: String,
    /// The recovery authority allowed to rotate both keys.
    pub backup: String,
}

impl Default for KeySet {
    fn default() -> Self {
        Self {
            version: 1,
            primary: crate::keys::PRIMARY.into(),
            backup: crate::keys::BACKUP.into(),
        }
    }
}

fn public_key(text: &str) -> Result<PublicKey> {
    let result = if text.contains('\n') {
        PublicKey::decode(text)
    } else {
        PublicKey::from_base64(text)
    };
    result.map_err(|e| Error::Refused(format!("invalid minisign public key: {e}")))
}

fn key_material(text: &str) -> &str {
    if text.contains('\n') {
        text.lines().nth(1).unwrap_or_default()
    } else {
        text
    }
}

fn verify(bytes: &[u8], sig: &str, key: &str, role: &str) -> Result<Signature> {
    if sig.trim().is_empty() {
        return Err(Error::Refused(format!("missing {role} signature")));
    }
    let sig = Signature::decode(sig)
        .map_err(|e| Error::Refused(format!("invalid {role} signature: {e}")))?;
    public_key(key)?
        .verify(bytes, &sig, false)
        .map_err(|e| Error::Refused(format!("{role} signature refused: {e}")))?;
    Ok(sig)
}

/// Authenticates a release and binds its signed version/channel comment to its JSON body.
pub fn verify_release(bytes: &[u8], sig: &str, keys: &KeySet) -> Result<()> {
    let signature = verify(bytes, sig, &keys.primary, "primary release")?;
    let body: serde_json::Value = serde_json::from_slice(bytes)?;
    let mut comment = BTreeMap::new();
    for (position, pair) in signature.trusted_comment().split_whitespace().enumerate() {
        let (key, value) = pair
            .split_once('=')
            .filter(|(k, v)| !k.is_empty() && !v.is_empty())
            .ok_or_else(|| {
                Error::Refused("invalid trusted comment: expected key=value pairs".into())
            })?;
        if position < 2 && key != ["version", "channel"][position] {
            return Err(Error::Refused(
                "trusted comment must begin with version=<v> channel=<c>".into(),
            ));
        }
        if comment.insert(key, value).is_some() {
            return Err(Error::Refused(format!(
                "duplicate {key} in trusted comment"
            )));
        }
    }
    for key in ["version", "channel"] {
        if body
            .get(key)
            .and_then(|v| v.as_str())
            .is_none_or(|v| comment.get(key).copied() != Some(v))
        {
            return Err(Error::Refused(format!(
                "trusted comment {key} does not match release body"
            )));
        }
    }
    Ok(())
}

/// Adopts a strictly newer keyset only with the current backup's authorization.
pub fn verify_keyset(
    bytes: &[u8],
    sig_backup: &str,
    sig_primary: Option<&str>,
    current: &KeySet,
) -> Result<KeySet> {
    verify(bytes, sig_backup, &current.backup, "backup keyset")?;
    if let Some(signature) = sig_primary {
        verify(bytes, signature, &current.primary, "primary keyset")?;
    }
    let new: KeySet = serde_json::from_slice(bytes)?;
    if new.version <= current.version {
        return Err(Error::Refused("keyset version must increase".into()));
    }
    public_key(&new.primary)?;
    public_key(&new.backup)?;
    if key_material(&new.primary) == key_material(&new.backup) {
        return Err(Error::Refused(
            "primary and backup keys must be distinct".into(),
        ));
    }
    Ok(new)
}

// The root lockfile had no semver crate; the supported syntax is MAJOR.MINOR.PATCH[-pre].
fn version(value: &str) -> Result<([u64; 3], Option<Vec<&str>>)> {
    let invalid = || {
        Error::Refused(format!(
            "invalid version {value:?}: expected MAJOR.MINOR.PATCH[-pre]"
        ))
    };
    let (core, pre) = value
        .split_once('-')
        .map_or((value, None), |(c, p)| (c, Some(p)));
    let parts: Vec<_> = core.split('.').collect();
    if parts.len() != 3 {
        return Err(invalid());
    }
    let mut numbers = [0; 3];
    for (out, part) in numbers.iter_mut().zip(parts) {
        if part.is_empty()
            || (part.len() > 1 && part.starts_with('0'))
            || !part.bytes().all(|b| b.is_ascii_digit())
        {
            return Err(invalid());
        }
        *out = part.parse().map_err(|_| invalid())?;
    }
    let pre = pre.map(|p| p.split('.').collect::<Vec<_>>());
    if pre.as_ref().is_some_and(|p| {
        p.iter().any(|s| {
            s.is_empty()
                || !s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
                || (s.len() > 1 && s.starts_with('0') && s.bytes().all(|b| b.is_ascii_digit()))
        })
    }) {
        return Err(invalid());
    }
    Ok((numbers, pre))
}

/// Compares the supported semantic versions, including numeric prerelease identifiers.
pub fn compare_versions(a: &str, b: &str) -> Result<Ordering> {
    let (a, ap) = version(a)?;
    let (b, bp) = version(b)?;
    if a != b {
        return Ok(a.cmp(&b));
    }
    Ok(match (ap, bp) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(a), Some(b)) => {
            let mut order = Ordering::Equal;
            for (a, b) in a.iter().zip(&b) {
                let numeric = |s: &str| s.bytes().all(|b| b.is_ascii_digit());
                order = match (numeric(a), numeric(b)) {
                    (true, true) => a.len().cmp(&b.len()).then(a.cmp(b)),
                    (true, false) => Ordering::Less,
                    (false, true) => Ordering::Greater,
                    _ => a.cmp(b),
                };
                if order != Ordering::Equal {
                    break;
                }
            }
            if order == Ordering::Equal {
                a.len().cmp(&b.len())
            } else {
                order
            }
        }
    })
}

/// Refuses an implicit downgrade while accepting a deliberately requested older version.
pub fn accept_version(installed: Option<&str>, offered: &str, explicit: bool) -> Result<()> {
    version(offered)?;
    if let Some(installed) = installed
        && compare_versions(offered, installed)? == Ordering::Less
        && !explicit
    {
        return Err(Error::Refused(format!(
            "refusing downgrade from {installed} to {offered}; type an explicit version or use rollback"
        )));
    }
    Ok(())
}

/// Parses a channel pointer only in its supported wire format.
pub fn parse_pointer(bytes: &[u8]) -> Result<Pointer> {
    let wire: serde_json::Value = serde_json::from_slice(bytes)?;
    if wire
        .get("schema")
        .and_then(|v| v.as_u64())
        .is_some_and(|v| v > u64::from(SUPPORTED_SCHEMA))
    {
        return Err(Error::Refused(NEWER_SETUP.into()));
    }
    let pointer: Pointer = serde_json::from_slice(bytes)?;
    if pointer.schema != SUPPORTED_SCHEMA {
        return Err(Error::Refused(NEWER_SETUP.into()));
    }
    version(&pointer.version)?;
    Ok(pointer)
}

/// Parses a release only when this installer understands its schema and minimum version.
pub fn parse_manifest(bytes: &[u8]) -> Result<Manifest> {
    let wire: serde_json::Value = serde_json::from_slice(bytes)?;
    if wire
        .get("schema")
        .and_then(|v| v.as_u64())
        .is_some_and(|v| v > u64::from(SUPPORTED_SCHEMA))
    {
        return Err(Error::Refused(NEWER_SETUP.into()));
    }
    let manifest: Manifest = serde_json::from_slice(bytes)?;
    if manifest.schema != SUPPORTED_SCHEMA
        || compare_versions(&manifest.min_setup, env!("CARGO_PKG_VERSION"))? == Ordering::Greater
    {
        return Err(Error::Refused(NEWER_SETUP.into()));
    }
    version(&manifest.version)?;
    Ok(manifest)
}

/// Normalizes an archive-relative path and rejects platform aliases or traversal.
pub fn relative_path(text: &str) -> Result<std::path::PathBuf> {
    let mut text = text;
    while let Some(rest) = text.strip_prefix("./") {
        text = rest;
    }
    if text.is_empty()
        || text.starts_with('/')
        || text.contains(['\\', ':', '*', '?', '"', '<', '>', '|'])
        || text.chars().any(char::is_control)
    {
        return Err(Error::Refused(format!("unsafe relative path: {text:?}")));
    }
    for part in text.split('/') {
        let base = part
            .split('.')
            .next()
            .unwrap_or_default()
            .trim_end_matches(['.', ' '])
            .to_ascii_uppercase();
        let numbered = base
            .strip_prefix("COM")
            .or_else(|| base.strip_prefix("LPT"))
            .is_some_and(|n| {
                matches!(
                    n,
                    "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
                )
            });
        if part.is_empty()
            || matches!(part, "." | "..")
            || part.ends_with(['.', ' '])
            || matches!(
                base.as_str(),
                "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$" | "CLOCK$"
            )
            || numbered
        {
            return Err(Error::Refused(format!("unsafe relative path: {text:?}")));
        }
    }
    Ok(text.into())
}

/// Validates an inventory once, normalizing paths and hashes and refusing aliases.
pub fn inventory(files: &[(String, String, u64)]) -> Result<BTreeMap<String, (String, u64)>> {
    let mut seen = std::collections::BTreeSet::new();
    let mut result = BTreeMap::new();
    for (name, hash, size) in files {
        let path = relative_path(name)?.to_string_lossy().replace('\\', "/");
        if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(Error::Refused(format!("invalid SHA-256 for {name:?}")));
        }
        if !seen.insert(path.to_ascii_lowercase()) {
            return Err(Error::Refused(format!(
                "duplicate inventory path: {name:?}"
            )));
        }
        result.insert(path, (hash.to_ascii_lowercase(), *size));
    }
    for path in result.keys() {
        for parent in std::path::Path::new(path).ancestors().skip(1) {
            if seen.contains(
                &parent
                    .to_string_lossy()
                    .replace('\\', "/")
                    .to_ascii_lowercase(),
            ) {
                return Err(Error::Refused(format!(
                    "inventory file is also a parent: {path:?}"
                )));
            }
        }
    }
    Ok(result)
}
