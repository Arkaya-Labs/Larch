//! Streamable HTTP: one endpoint, `POST /mcp`, on 127.0.0.1 only, behind the bearer token in
//! `<data_dir>/agent/mcp-token`. Each POST carries one message; a request's answer, and the
//! progress before it, come back as a server-sent event stream that ends with the answer.

use std::{
    io::{self, BufRead, BufReader, Read, Write},
    net::{Ipv4Addr, TcpListener, TcpStream},
    path::Path,
    sync::{Arc, mpsc},
    thread,
    time::Duration,
};

use super::{Out, Server, VERSIONS};

const MAX_HEAD: u64 = 16 * 1024;
const MAX_BODY: usize = 4 << 20;
/// How long a client may take to send its request.
const TIMEOUT: Duration = Duration::from_secs(30);

/// The bearer token HTTP clients send: made at first run from the system's random source and kept
/// in `<data_dir>/agent/mcp-token`, which only its owner may read; a token file others may read,
/// or one that is not a whole token, is refused.
// ponytail: Unix only (/dev/urandom, mode 0600); add BCryptGenRandom and an owner-only ACL in
// larch-sys when Windows needs the HTTP transport, which stdio does not
pub fn token(data_dir: &Path) -> io::Result<String> {
    #[cfg(unix)]
    {
        use std::{
            fs,
            os::unix::fs::{OpenOptionsExt, PermissionsExt},
        };
        let dir = data_dir.join("agent");
        let path = dir.join("mcp-token");
        let refuse = |why: String| {
            let fix = "delete it for a new token";
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("{}: {why}; {fix}", path.display()),
            )
        };
        match fs::read_to_string(&path) {
            Ok(token) => {
                let mode = fs::metadata(&path)?.permissions().mode() & 0o777;
                if mode & 0o077 != 0 {
                    return Err(refuse(format!("other users may read it (mode {mode:o})")));
                }
                // An empty file, from a crash before its write, would let an empty key in.
                let token = token.trim();
                if token.len() != 64 {
                    return Err(refuse("it holds no whole token".into()));
                }
                return Ok(token.into());
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        let mut bytes = [0; 32];
        fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
        let token = larch::bench::hex(&bytes);
        fs::create_dir_all(&dir)?;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)?;
        file.write_all(token.as_bytes())?;
        file.sync_all()?;
        Ok(token)
    }
    #[cfg(not(unix))]
    {
        let _ = data_dir;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "the MCP HTTP transport needs Unix here; use stdio",
        ))
    }
}

impl Server {
    /// Serves Streamable HTTP on `listener` until the process ends. The listener must be bound to
    /// 127.0.0.1: any other address is refused, so Iris never opens a port to the network.
    pub fn http(&self, listener: TcpListener) -> io::Result<()> {
        let addr = listener.local_addr()?;
        if addr.ip() != Ipv4Addr::LOCALHOST {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Iris's MCP server listens on 127.0.0.1 only, not {addr}"),
            ));
        }
        let token = &token(&self.shared.data_dir)?;
        // ponytail: one thread per connection, uncapped, as serve's (only local processes reach
        // it); cap them if a local process trickling heads ever holds threads
        thread::scope(|s| {
            for conn in listener.incoming() {
                // A failed accept is that client's connection gone; the next one is unaffected.
                let Ok(conn) = conn else { continue };
                // An I/O error is the client gone: there is no one left to answer.
                s.spawn(move || {
                    let _ = self.connection(&conn, token);
                });
            }
        });
        Ok(())
    }

    fn connection(&self, conn: &TcpStream, token: &str) -> io::Result<()> {
        conn.set_read_timeout(Some(TIMEOUT))?;
        // A client that stops reading its stream must not hold the thread for ever.
        conn.set_write_timeout(Some(TIMEOUT))?;
        // Each event goes out at once: Nagle would hold progress back.
        conn.set_nodelay(true)?;
        let mut out = conn;
        let mut reader = BufReader::new(conn);
        let Some(head) = read_head(&mut reader)? else {
            return bare(&mut out, 431);
        };
        if head.chunked {
            return bare(&mut out, 411);
        }
        if head.length > MAX_BODY {
            return bare(&mut out, 413);
        }
        // A page in a browser may reach 127.0.0.1 too (DNS rebinding): its Origin gives it away.
        let status = match () {
            _ if !head.origin.as_deref().is_none_or(local_origin) => 403,
            _ if !head.token.as_deref().is_some_and(|t| same(t, token)) => 401,
            _ if head.path != "/mcp" => 404,
            _ if head.method != "POST" => 405,
            _ if head
                .version
                .as_deref()
                .is_some_and(|v| !VERSIONS.contains(&v)) =>
            {
                400
            }
            _ => 200,
        };
        if status != 200 {
            // Closing on an unread body resets the connection, and the client would lose the
            // answer.
            io::copy(&mut (&mut reader).take(head.length as u64), &mut io::sink())?;
            return bare(&mut out, status);
        }
        let mut body = vec![0; head.length];
        reader.read_exact(&mut body)?;
        let Ok(body) = String::from_utf8(body) else {
            return bare(&mut out, 400);
        };
        let (tx, rx) = mpsc::channel();
        let sink: Out = Arc::new(move |message| {
            // The stream ended with its answer; a task's later progress has nowhere to go.
            let _ = tx.send(message);
        });
        self.handle(&body, &sink);
        drop(sink);
        // Nothing to send, and no call holding the sink: a notification or a response.
        let Ok(first) = rx.recv() else {
            return bare(&mut out, 202);
        };
        out.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
              Connection: close\r\n\r\n",
        )?;
        let mut next = Some(first);
        while let Some(message) = next {
            out.write_all(format!("event: message\ndata: {message}\n\n").as_bytes())?;
            // The answer ends the stream; a cancelled call never answers, and drops the sink.
            next = message.get("method").and_then(|_| rx.recv().ok());
        }
        Ok(())
    }
}

/// The request line and the headers this server reads.
#[derive(Default)]
pub(super) struct Head {
    pub(super) method: String,
    pub(super) path: String,
    length: usize,
    chunked: bool,
    token: Option<String>,
    origin: Option<String>,
    version: Option<String>,
}

/// Reads the head; `None` when it is over [`MAX_HEAD`] or the client ended it early.
pub(super) fn read_head(r: &mut impl BufRead) -> io::Result<Option<Head>> {
    let mut r = r.take(MAX_HEAD);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let mut head = Head {
        method: parts.next().unwrap_or_default().into(),
        path: parts.next().unwrap_or_default().into(),
        ..Head::default()
    };
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        let header = line.trim();
        if header.is_empty() {
            return Ok(Some(head));
        }
        let (name, value) = header.split_once(':').unwrap_or((header, ""));
        let value = value.trim();
        match name.to_ascii_lowercase().as_str() {
            "content-length" => head.length = value.parse().unwrap_or(usize::MAX),
            "transfer-encoding" => head.chunked = value.to_ascii_lowercase().contains("chunked"),
            // The scheme's case is the client's choice (RFC 9110).
            "authorization" => {
                head.token = (value.split_once(' '))
                    .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
                    .map(|(_, token)| token.trim().into());
            }
            "origin" => head.origin = Some(value.into()),
            "mcp-protocol-version" => head.version = Some(value.into()),
            _ => {}
        }
    }
}

/// Whether an `Origin` is a page served from this computer.
fn local_origin(origin: &str) -> bool {
    let host = origin.strip_prefix("http://").unwrap_or_default();
    let host = (host.rsplit_once(':'))
        .filter(|(_, port)| port.bytes().all(|b| b.is_ascii_digit()))
        .map_or(host, |(host, _)| host);
    matches!(host, "127.0.0.1" | "localhost")
}

/// Equal tokens, compared in time that depends only on their lengths.
fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0, |d, (x, y)| d | (x ^ y)) == 0
}

/// A response with no body; every connection carries one request.
fn bare(out: &mut impl Write, status: u16) -> io::Result<()> {
    let reason = match status {
        202 => "Accepted",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        411 => "Length Required",
        413 => "Payload Too Large",
        _ => "Request Header Fields Too Large",
    };
    write!(
        out,
        "HTTP/1.1 {status} {reason}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )
}
