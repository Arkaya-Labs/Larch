//! fetch_page through the environment's proxy: a name that does not resolve on this computer
//! goes to the proxy, which resolves it itself; an address of this machine or its networks, and
//! `localhost`, never do. The fetches run in a child of this test with the proxy in its
//! environment, as the agents read it once.

use std::{
    io::{BufRead, BufReader},
    net::TcpListener,
    process::Command,
    sync::{Arc, Mutex},
    thread,
};

use larch_agent::web::fetch::{self, MAX_BYTES};

/// Set in the child, which fetches.
const CHILD: &str = "LARCH_FETCH_PROXY_CHILD";
const NAME: &str = "through_a_proxy_unresolved_names_are_tried_and_local_addresses_refused";

#[test]
fn through_a_proxy_unresolved_names_are_tried_and_local_addresses_refused() {
    if std::env::var_os(CHILD).is_some() {
        return fetch_through_the_proxy();
    }
    // A proxy that logs each request line and closes the connection.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let proxy = format!("http://{}", listener.local_addr().unwrap());
    let asked = Arc::new(Mutex::new(Vec::<String>::new()));
    let log = asked.clone();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let mut line = String::new();
            let _ = BufReader::new(stream.unwrap()).read_line(&mut line);
            log.lock().unwrap().push(line.trim_end().to_string());
        }
    });
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", NAME, "--nocapture"])
        .env(CHILD, "1")
        .env("ALL_PROXY", &proxy)
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{said}");
    assert_eq!(
        *asked.lock().unwrap(),
        ["CONNECT iris-proxy-test.invalid:443 HTTP/1.1"],
        "{said}"
    );
}

fn fetch_through_the_proxy() {
    let dir = tempfile::tempdir().unwrap();
    let urls = [
        "https://iris-proxy-test.invalid/page",
        "https://10.0.0.1/",
        "https://100.64.1.1/",
        "https://localhost/",
        "https://a.localhost/",
    ];
    let got = fetch::fetch_many(&urls, MAX_BYTES, dir.path());
    // The proxy closes the connection, so the first fails too, after reaching it.
    println!("{:?}", got[0]);
    for (url, got) in urls.iter().zip(&got) {
        let e = got.as_ref().expect_err(url).to_string();
        assert_eq!(e.contains("never opens"), *url != urls[0], "{url}: {e}");
    }
}
