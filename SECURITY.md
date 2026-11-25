# Security policy

## Development status

larch is early development with no stable product release or supported release series. Interfaces,
packaging and security controls may change. Do not treat the current source or a CI artifact as a
qualified production release.

## Reporting a vulnerability

Report security problems privately: open the repository's **Security** tab and choose **Report a
vulnerability**. Only maintainers see the report. There is no monitored inbox and no response-time
commitment.

Do not post an exploit, sensitive reproduction, credentials or personal data in a public issue, pull
request, discussion or commit. If the button is missing, open a public issue that asks for a private
channel and contains no vulnerability details.

Include the affected commit/version, platform and backend, impact,
and a minimal reproduction with secrets and personal data removed. Coordinate disclosure privately
before publishing details.

## `larch serve` boundaries

These describe the current [implementation](crates/larch/src/serve.rs), not a deployment guarantee:

- The default listener is `127.0.0.1:8080`. Startup rejects a `--host` that resolves to any non-loopback
  address unless `LARCH_API_KEY` is non-empty. Keep hostnames and network configuration under your
  control; this startup check is not a firewall.
- With a key configured, data APIs accept that shared key in `Authorization: Bearer <key>` or
  `x-api-key: <key>`. Without a key, a loopback server permits unauthenticated requests. Local processes
  that can reach it share that access. A keyless server refuses a request whose `Host` is not a loopback
  name or whose `Origin` is any other site, so a web page or a DNS-rebinding name cannot use it. A page served from this machine on any
  port (a local dev server, a notebook) counts as local and can use it. The
  key is a server-wide credential, not per-user authorization or tenant isolation.
- `GET /health` is unauthenticated. `/v1/*` data APIs and `/api/status` are subject to the configured key. Do not expose telemetry to
  untrusted clients.
- The listener uses plain TCP/HTTP and has no built-in TLS. Keys and prompts need transport protection
  outside a trusted local environment. Use a TLS/authentication proxy and restrict network access if
  remote access is required; preserve protection on the proxy-to-server connection too.
- Each accepted connection gets a thread, without a connection cap. Request size limits and socket
  timeouts do not provide complete denial-of-service protection. A proxy or network boundary needs
  appropriate connection limits before accepting untrusted traffic.

## Files, models and local tools

Download GGUF files only from sources you trust. The benchmark model helper verifies hashes from
[bench/models.toml](bench/models.toml); a matching hash identifies expected bytes, not model safety or a
license grant. Review model licenses separately from the [source license](LICENSE) and
[third-party notices](THIRD_PARTY_NOTICES).

Chat writes transcripts under its data directory. Treat these files and
any application/agent state, tool output and logs as potentially sensitive. Protect their filesystem
access and redact them before sharing. When using agent tools, review their configured permissions and
the resources they can access; the inference server's API key does not sandbox another process's tools.

Do not commit credentials, API keys, private model files or personal data. Report any accidental
exposure through the verified private reporting route once it is established.
