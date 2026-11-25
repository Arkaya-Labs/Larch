//! Read-only checks for the local model API, without browser CORS or additional dependencies.

use std::{io::Read, time::Duration};

use serde_json::Value;

const MAX_STATUS_BYTES: u64 = 2 * 1024 * 1024;

/// The HTTP status and parsed body returned to the connection guide.
#[derive(Debug, serde::Serialize)]
pub struct ServerStatus {
    /// Status from the local HTTP server, including authentication and readiness errors.
    pub status: u16,
    /// The server's JSON, displayed as text by the UI.
    pub body: Value,
}

/// Checks only the local status route; tokens never follow redirects or environment proxies.
pub fn local_server_status(port: u16, api_key: Option<&str>) -> Result<ServerStatus, String> {
    if port == 0 {
        return Err("Use a port from 1 to 65535.".into());
    }
    let key = api_key.unwrap_or_default();
    if key.len() > 8192 || key.chars().any(char::is_control) {
        return Err("The API key contains an invalid character or is too long.".into());
    }
    let agent = ureq::AgentBuilder::new()
        .redirects(0)
        .try_proxy_from_env(false)
        .timeout(Duration::from_secs(5))
        .build();
    let mut request = agent.get(&format!("http://127.0.0.1:{port}/api/status"));
    if !key.is_empty() {
        request = request.set("Authorization", &format!("Bearer {key}"));
    }
    let response = match request.call() {
        Ok(response) | Err(ureq::Error::Status(_, response)) => response,
        // Nothing listening is refused at once: it told of a 5-second timeout that never ran.
        Err(ureq::Error::Transport(t)) if t.kind() == ureq::ErrorKind::ConnectionFailed => {
            return Err(format!(
                "Nothing answers on port {port}. Start the server, or check its port."
            ));
        }
        Err(ureq::Error::Transport(_)) => {
            return Err("Cannot reach the local server (5-second timeout). Check its port.".into());
        }
    };
    let status = response.status();
    if (300..400).contains(&status) {
        return Err(
            "The local status route redirected. Check that this port runs larch serve.".into(),
        );
    }
    let mut bytes = Vec::new();
    response
        .into_reader()
        .take(MAX_STATUS_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "The local status response could not be read.".to_owned())?;
    if bytes.len() as u64 > MAX_STATUS_BYTES {
        return Err("The local status response is larger than 2 MB.".into());
    }
    let body: Value = serde_json::from_slice(&bytes)
        .map_err(|_| "The port did not return JSON. Check that it runs larch serve.".to_owned())?;
    Ok(ServerStatus { status, body })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader, Write},
        net::TcpListener,
        thread,
    };

    fn serve(status: &str, body: &str) -> (u16, thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let reply = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let worker = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut reader = BufReader::new(&stream);
            let mut headers = String::new();
            loop {
                let mut line = String::new();
                assert!(reader.read_line(&mut line).unwrap() > 0);
                headers.push_str(&line);
                if line == "\r\n" {
                    break;
                }
            }
            stream.write_all(reply.as_bytes()).unwrap();
            headers
        });
        (port, worker)
    }

    #[test]
    fn status_keeps_http_errors_and_uses_only_the_status_route() {
        let (port, worker) = serve("401 Unauthorized", r#"{"error":{"message":"missing key"}}"#);
        let value = local_server_status(port, Some("test-key")).unwrap();
        assert_eq!(value.status, 401);
        assert_eq!(value.body["error"]["message"], "missing key");
        let headers = worker.join().unwrap();
        assert!(headers.starts_with("GET /api/status HTTP/1.1\r\n"));
        assert!(
            headers
                .to_lowercase()
                .contains("authorization: bearer test-key\r\n")
        );
    }

    #[test]
    fn redirects_invalid_json_and_oversized_responses_are_refused() {
        for (status, body, reason) in [
            ("302 Found", "{}".to_owned(), "redirected"),
            ("200 OK", "<html>other service</html>".to_owned(), "JSON"),
            (
                "200 OK",
                " ".repeat(MAX_STATUS_BYTES as usize + 1),
                "larger",
            ),
        ] {
            let (port, worker) = serve(status, &body);
            assert!(
                local_server_status(port, None)
                    .unwrap_err()
                    .contains(reason)
            );
            worker.join().unwrap();
        }
        assert!(local_server_status(0, None).is_err());
        // A port nothing listens on: refused at once, and said so.
        let free = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = free.local_addr().unwrap().port();
        drop(free);
        let e = local_server_status(port, None).unwrap_err();
        assert!(
            e.starts_with(&format!("Nothing answers on port {port}")),
            "{e}"
        );
        assert!(local_server_status(8080, Some("key\r\nInjected: header")).is_err());
    }
}
