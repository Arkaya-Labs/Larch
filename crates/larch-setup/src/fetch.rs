//! Signed release discovery and archive transport through the shared downloader only.
use crate::{
    log::clean,
    manifest::{self, Artifact, Error, KeySet, Manifest, Pointer, Result},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, env, fs, io::Read, path::Path};
const REPO: &str = "Arkaya-Labs/Larch";
const RAW: &str = "https://raw.githubusercontent.com/Arkaya-Labs/Larch/channels/";
const DOWNLOAD: &str = "https://github.com/Arkaya-Labs/Larch/releases/download/";
const API: &str = "https://api.github.com/repos/Arkaya-Labs/Larch/";
const MAX_METADATA: u64 = 8 * 1024 * 1024;
/// Authenticated pointer bytes and their validator, suitable for a read-only check cache.
#[derive(Clone, Serialize, Deserialize)]
pub struct PointerDocument {
    /// Parsed and verified channel head.
    pub pointer: Pointer,
    /// Exact bytes covered by the signature.
    pub body: Vec<u8>,
    /// Primary-key minisign signature.
    pub signature: String,
    /// HTTP validator used only after verifying cached bytes again.
    pub etag: Option<String>,
}
/// A release verified against the channel head or a deliberately requested older version.
pub struct Release {
    /// Current signed channel head, including bootstrap setup hashes.
    pub head: PointerDocument,
    /// The selected authenticated manifest.
    pub manifest: Manifest,
    /// Exact manifest bytes retained for verification and offline recovery.
    pub body: Vec<u8>,
    /// The manifest's primary-key signature.
    pub signature: String,
}
struct Response {
    body: Vec<u8>,
    etag: Option<String>,
    status: u16,
}
/// Release credentials are held only in memory and never included in diagnostics or URLs.
pub struct Client {
    base: Option<String>,
    token: Option<String>,
    keys: KeySet,
    assets: BTreeMap<String, BTreeMap<String, String>>,
}
fn identifier(value: &str) -> Result<()> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b))
    {
        return Err(Error::Refused("invalid release channel".into()));
    }
    Ok(())
}
fn local_http(base: &str) -> bool {
    let Some(rest) = base.strip_prefix("http://") else {
        return false;
    };
    let authority = rest.split('/').next().unwrap_or_default();
    ["localhost", "127.0.0.1", "[::1]"].iter().any(|host| {
        authority == *host
            || authority
                .strip_prefix(host)
                .and_then(|s| s.strip_prefix(':'))
                .is_some_and(|port| !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()))
    })
}
impl Client {
    /// Reads the documented source and token overrides, using the compiled public keys.
    pub fn from_env() -> Result<Self> {
        let keys = KeySet::default();
        #[cfg(feature = "test-roots")]
        let keys = if env::var_os("LARCH_SETUP_TEST_ROOTS").is_some() {
            match env::var_os("LARCH_SETUP_TEST_KEYS") {
                Some(path) => serde_json::from_slice(&fs::read(path)?)?,
                None => keys,
            }
        } else {
            keys
        };
        Self::new(
            env::var("LARCH_SETUP_RELEASES_URL").ok(),
            env::var("LARCH_RELEASES_TOKEN").ok(),
            keys,
        )
    }
    /// Uses explicit source and keys so generated-key local-server tests need no global environment.
    pub fn new(base: Option<String>, token: Option<String>, keys: KeySet) -> Result<Self> {
        let base = base.map(|s| s.trim_end_matches('/').to_owned());
        if base.as_ref().is_some_and(|s| {
            (!s.starts_with("https://") && !local_http(s))
                || s.contains(['@', '?', '#'])
                || s.chars().any(char::is_control)
        }) {
            return Err(Error::Refused(
                "release source must use HTTPS, or loopback HTTP for tests".into(),
            ));
        }
        let token = token.filter(|s| !s.is_empty());
        if token.as_ref().is_some_and(|s| {
            s.len() > 8192 || s.chars().any(|c| c.is_control() || c.is_whitespace())
        }) {
            return Err(Error::Refused(
                "release token is not a single HTTP header value".into(),
            ));
        }
        Ok(Self {
            base,
            token,
            keys,
            assets: BTreeMap::new(),
        })
    }
    fn error(&self, text: impl AsRef<str>) -> Error {
        Error::Refused(clean(text.as_ref(), self.token.as_deref()))
    }
    fn mapped(&self, url: &str) -> Result<String> {
        if let Some(base) = &self.base {
            if url
                .strip_prefix(base)
                .is_some_and(|s| s.is_empty() || s.starts_with('/'))
            {
                return Ok(url.into());
            }
            for (prefix, replacement) in [
                (RAW, "channels/"),
                (DOWNLOAD, "releases/download/"),
                (API, "api/"),
            ] {
                if let Some(rest) = url.strip_prefix(prefix) {
                    return Ok(format!("{base}/{replacement}{rest}"));
                }
            }
            if url.starts_with('/') && !url.starts_with("//") {
                return Ok(format!("{base}{url}"));
            }
        }
        if url.starts_with(RAW) || url.starts_with(DOWNLOAD) || url.starts_with(API) {
            Ok(url.into())
        } else {
            Err(self.error("release asset URL is outside the configured repository"))
        }
    }
    fn response(&self, url: &str, accept: &str, etag: Option<&str>) -> Result<Response> {
        let url = self.mapped(url)?;
        let agent = larch_fetch::agent(&url);
        let mut headers = vec![("Accept", accept)];
        if let Some(etag) = etag.filter(|s| s.len() <= 8192 && !s.chars().any(char::is_control)) {
            headers.push(("If-None-Match", etag));
        }
        let response =
            larch_fetch::get(&agent, &url, self.token.as_deref(), &headers).map_err(|e| {
                let code = (*e).into_response().map(|r| r.status());
                self.error(match code {
                    // ponytail: drop this hint when larch-releases goes public (architecture 9.5).
                    Some(404) if self.token.is_none() => "release request failed: HTTP 404; while \
                        the release repository is private, set LARCH_RELEASES_TOKEN to a read token"
                        .into(),
                    Some(code) => format!("release request failed: HTTP {code}"),
                    None => {
                        "release request failed: transport error; check the connection and proxy"
                            .into()
                    }
                })
            })?;
        let status = response.status();
        let etag = response
            .header("etag")
            .filter(|s| s.len() <= 8192 && !s.chars().any(char::is_control))
            .map(str::to_owned);
        if status == 304 {
            return Ok(Response {
                body: Vec::new(),
                etag,
                status,
            });
        }
        if status != 200 {
            return Err(self.error(format!("release metadata returned HTTP {status}")));
        }
        if response
            .header("content-length")
            .and_then(|s| s.parse::<u64>().ok())
            .is_some_and(|n| n > MAX_METADATA)
        {
            return Err(self.error("release metadata exceeds 8 MiB"));
        }
        let mut body = Vec::new();
        response
            .into_reader()
            .take(MAX_METADATA + 1)
            .read_to_end(&mut body)?;
        if body.len() as u64 > MAX_METADATA {
            return Err(self.error("release metadata exceeds 8 MiB"));
        }
        Ok(Response { body, etag, status })
    }
    fn pointer_url(&self, channel: &str, signature: bool) -> String {
        let suffix = if signature { ".json.minisig" } else { ".json" };
        if self.token.is_some() {
            format!("{API}contents/{channel}{suffix}?ref=channels")
        } else {
            format!("{RAW}{channel}{suffix}")
        }
    }
    fn verified_pointer(
        &self,
        body: Vec<u8>,
        signature: String,
        etag: Option<String>,
        channel: &str,
    ) -> Result<PointerDocument> {
        manifest::verify_release(&body, &signature, &self.keys)?;
        let pointer = manifest::parse_pointer(&body)?;
        if pointer.channel != channel {
            return Err(self.error("signed pointer channel does not match the requested channel"));
        }
        Ok(PointerDocument {
            pointer,
            body,
            signature,
            etag,
        })
    }
    /// Fetches the pointer conditionally and authenticates cached bytes before accepting a 304.
    pub fn pointer(
        &self,
        channel: &str,
        cached: Option<&PointerDocument>,
    ) -> Result<PointerDocument> {
        identifier(channel)?;
        let accept = "application/vnd.github.raw+json";
        let response = self.response(
            &self.pointer_url(channel, false),
            accept,
            cached.and_then(|c| c.etag.as_deref()),
        )?;
        if response.status == 304 {
            let cached = cached.ok_or_else(|| {
                self.error("server returned not-modified without a cached pointer")
            })?;
            return self.verified_pointer(
                cached.body.clone(),
                cached.signature.clone(),
                response.etag.or_else(|| cached.etag.clone()),
                channel,
            );
        }
        let signature = self
            .response(&self.pointer_url(channel, true), accept, None)?
            .body;
        let signature = String::from_utf8(signature)
            .map_err(|_| self.error("pointer signature is not UTF-8"))?;
        self.verified_pointer(response.body, signature, response.etag, channel)
    }
    fn asset_url(&mut self, url: &str) -> Result<String> {
        if self.token.is_none() {
            return self.mapped(url);
        }
        if self
            .base
            .as_ref()
            .is_some_and(|base| url.strip_prefix(base).is_some_and(|s| s.starts_with('/')))
        {
            return self.mapped(url);
        }
        if url.starts_with(&format!("{API}releases/assets/")) {
            return self.mapped(url);
        }
        let rest = url.strip_prefix(DOWNLOAD).ok_or_else(|| {
            self.error("private release asset URL is outside the configured repository")
        })?;
        let (tag, name) = rest
            .split_once('/')
            .ok_or_else(|| self.error("release asset URL has no tag and filename"))?;
        if tag.contains(['?', '#']) || name.contains(['/', '?', '#']) {
            return Err(self.error("invalid release asset path"));
        }
        if !self.assets.contains_key(tag) {
            let response = self.response(
                &format!("{API}releases/tags/{tag}"),
                "application/vnd.github+json",
                None,
            )?;
            #[derive(Deserialize)]
            struct Assets {
                assets: Vec<Asset>,
            }
            #[derive(Deserialize)]
            struct Asset {
                name: String,
                url: String,
            }
            let assets: Assets = serde_json::from_slice(&response.body)?;
            let mut map = BTreeMap::new();
            for a in assets.assets {
                if !a.url.starts_with(&format!("{API}releases/assets/")) {
                    return Err(
                        self.error("GitHub returned an asset URL outside the release repository")
                    );
                }
                map.insert(a.name, a.url);
            }
            self.assets.insert(tag.into(), map);
        }
        let asset = self
            .assets
            .get(tag)
            .and_then(|m| m.get(name))
            .ok_or_else(|| self.error(format!("release asset {name} is missing")))?;
        self.mapped(asset)
    }
    fn asset(&mut self, url: &str) -> Result<Vec<u8>> {
        let url = self.asset_url(url)?;
        Ok(self.response(&url, "application/octet-stream", None)?.body)
    }
    /// Verifies pointer, manifest hash and primary signatures in that order.
    pub fn release(&mut self, channel: &str, version: Option<&str>) -> Result<Release> {
        let head = self.pointer(channel, None)?;
        self.release_from_head(channel, version, head)
    }
    /// Authenticates stored manifest bytes before using them for offline maintenance.
    pub fn saved(&self, body: &[u8], signature: &str) -> Result<Manifest> {
        manifest::verify_release(body, signature, &self.keys)?;
        manifest::parse_manifest(body)
    }
    /// Resolves an already fetched channel head without discarding its conditional request.
    pub fn release_from_head(
        &mut self,
        channel: &str,
        version: Option<&str>,
        head: PointerDocument,
    ) -> Result<Release> {
        identifier(channel)?;
        let head = self.verified_pointer(head.body, head.signature, head.etag, channel)?;
        if let Some(version) = version {
            manifest::accept_version(None, version, true)?;
        }
        let explicit = version.is_some_and(|v| v != head.pointer.version);
        let selected = version.unwrap_or(&head.pointer.version);
        let url = if explicit {
            format!("{DOWNLOAD}v{selected}/manifest.json")
        } else {
            head.pointer.manifest_url.clone()
        };
        let body = self.asset(&url)?;
        if !explicit
            && !larch_fetch::sha256_bytes(&body).eq_ignore_ascii_case(&head.pointer.manifest_sha256)
        {
            return Err(self.error("manifest SHA-256 does not match the signed pointer"));
        }
        let signature = self.asset(&format!("{url}.minisig"))?;
        let signature = String::from_utf8(signature)
            .map_err(|_| self.error("manifest signature is not UTF-8"))?;
        manifest::verify_release(&body, &signature, &self.keys)?;
        let manifest = manifest::parse_manifest(&body)?;
        if manifest.channel != channel || manifest.version != selected {
            return Err(self
                .error("signed manifest version or channel does not match the requested release"));
        }
        Ok(Release {
            head,
            manifest,
            body,
            signature,
        })
    }
    /// Authenticates the running bootstrap executable against its target-specific pointer entry.
    pub fn bootstrap(&self, head: &PointerDocument, target: &str, exe: &Path) -> Result<()> {
        manifest::verify_release(&head.body, &head.signature, &self.keys)?;
        let pointer = manifest::parse_pointer(&head.body)?;
        let artifact = pointer
            .setup
            .get(target)
            .ok_or_else(|| self.error("channel has no setup executable for this target"))?;
        if fs::metadata(exe)?.len() != artifact.size
            || !larch_fetch::sha256_file(exe)?.eq_ignore_ascii_case(&artifact.sha256)
        {
            return Err(
                self.error("bootstrap setup size or SHA-256 mismatch; rerun the install command")
            );
        }
        Ok(())
    }
    /// Resumes an archive with the same shared range/ETag/hash implementation as the app.
    pub fn download(
        &mut self,
        artifact: &Artifact,
        dest: &Path,
        progress: impl FnMut(u64, u64) -> bool,
    ) -> Result<()> {
        let url = self.asset_url(&artifact.url)?;
        let agent = larch_fetch::agent(&url);
        let result = larch_fetch::download_with(
            dest,
            Some(&artifact.sha256),
            artifact.size,
            |start, etag| {
                let range = format!("bytes={start}-");
                let mut headers = vec![
                    ("Accept", "application/octet-stream"),
                    ("Range", range.as_str()),
                ];
                if let Some(etag) = etag {
                    headers.push(("If-Range", etag));
                }
                larch_fetch::get(&agent, &url, self.token.as_deref(), &headers)
            },
            progress,
        );
        result
            .map(|_| ())
            .map_err(|e| self.error(format!("archive download failed: {e}")))
    }
    /// The repository used by source diagnostics without exposing authentication headers.
    pub fn repository(&self) -> &'static str {
        REPO
    }
}
