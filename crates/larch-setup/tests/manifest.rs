//! Release signatures bind both bytes and trusted comments; backup authority is separate.

use larch_setup::manifest::*;
use minisign::KeyPair;
use serde_json::{Value, json};
use std::io::Cursor;

fn keys() -> (KeyPair, KeyPair, KeySet) {
    let primary = KeyPair::generate_unencrypted_keypair().unwrap();
    let backup = KeyPair::generate_unencrypted_keypair().unwrap();
    let keys = KeySet {
        version: 1,
        primary: primary.pk.to_base64(),
        backup: backup.pk.to_base64(),
    };
    (primary, backup, keys)
}

fn sign(pair: &KeyPair, bytes: &[u8], comment: &str) -> String {
    minisign::sign(
        Some(&pair.pk),
        &pair.sk,
        Cursor::new(bytes),
        Some(comment),
        Some("test signature"),
    )
    .unwrap()
    .to_string()
}

fn body() -> Value {
    json!({"schema":1,"min_setup":"0.0.0","version":"1.2.3","channel":"stable",
        "components":{},"runtimes":{},"gpu_rules":[],"amd_ids":{},
        "floors":{"macos_min":"26.0","glibc_min":"2.28","glibcxx_min":"3.4.29","windows_min_build":22000},
        "flavours":{},"gated":[],"hip_targets":[],"models":[]})
}

#[test]
fn release_requires_primary_signature_over_bytes_and_trusted_comment() {
    let (primary, backup, keys) = keys();
    let bytes = serde_json::to_vec(&body()).unwrap();
    let sig = sign(&primary, &bytes, "version=1.2.3 channel=stable gates=test");
    verify_release(&bytes, &sig, &keys).unwrap();
    let mut changed = bytes.clone();
    changed.push(b' ');
    assert!(verify_release(&changed, &sig, &keys).is_err());
    assert!(
        verify_release(&bytes, "", &keys)
            .unwrap_err()
            .to_string()
            .contains("missing")
    );
    assert!(
        verify_release(
            &bytes,
            &sign(&backup, &bytes, "version=1.2.3 channel=stable"),
            &keys
        )
        .is_err()
    );
    let wrong = KeyPair::generate_unencrypted_keypair().unwrap();
    assert!(
        verify_release(
            &bytes,
            &sign(&wrong, &bytes, "version=1.2.3 channel=stable"),
            &keys
        )
        .is_err()
    );
    for comment in [
        "version=1.2.2 channel=stable",
        "version=1.2.3 channel=beta",
        "channel=stable",
        "channel=stable version=1.2.3",
        "version=1.2.3 channel=stable version=1.2.3",
    ] {
        assert!(
            verify_release(&bytes, &sign(&primary, &bytes, comment), &keys).is_err(),
            "{comment}"
        );
    }
    let changed_comment = sig.replace("channel=stable", "channel=beta");
    assert!(verify_release(&bytes, &changed_comment, &keys).is_err());
}

#[test]
fn key_rotation_requires_backup_and_every_supplied_primary_signature() {
    let (primary, backup, current) = keys();
    let (_, _, mut next) = keys();
    next.version = current.version + 1;
    let bytes = serde_json::to_vec(&next).unwrap();
    let b = sign(&backup, &bytes, "keyset=2");
    let p = sign(&primary, &bytes, "keyset=2");
    assert!(verify_keyset(&bytes, "", Some(&p), &current).is_err());
    assert!(verify_keyset(&bytes, &p, None, &current).is_err());
    assert_eq!(verify_keyset(&bytes, &b, None, &current).unwrap(), next);
    assert_eq!(verify_keyset(&bytes, &b, Some(&p), &current).unwrap(), next);
    assert!(verify_keyset(&bytes, &b, Some(&b), &current).is_err());
    let mut same_key = next.clone();
    same_key.backup = format!("untrusted comment: another label\n{}", same_key.primary);
    let bytes_same = serde_json::to_vec(&same_key).unwrap();
    assert!(
        verify_keyset(
            &bytes_same,
            &sign(&backup, &bytes_same, "keyset=2"),
            None,
            &current
        )
        .unwrap_err()
        .to_string()
        .contains("distinct")
    );
    for v in [0, 1] {
        next.version = v;
        let bytes = serde_json::to_vec(&next).unwrap();
        assert!(
            verify_keyset(&bytes, &sign(&backup, &bytes, "keyset=old"), None, &current)
                .unwrap_err()
                .to_string()
                .contains("increase")
        );
    }
}

#[test]
fn versions_obey_numeric_prerelease_order_and_explicit_downgrades() {
    accept_version(Some("1.2.3"), "1.2.2", true).unwrap();
    assert!(accept_version(Some("1.2.3"), "1.2.2", false).is_err());
    let ordered = [
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-alpha.beta",
        "1.0.0-beta",
        "1.0.0-beta.2",
        "1.0.0-beta.11",
        "1.0.0-rc.1",
        "1.0.0",
        "1.0.1",
        "2.0.0",
    ];
    for pair in ordered.windows(2) {
        assert_eq!(
            compare_versions(pair[0], pair[1]).unwrap(),
            std::cmp::Ordering::Less
        );
    }
    for invalid in [
        "1.2",
        "01.2.3",
        "1.2.3-",
        "1.2.3-01",
        "1.2.3+x",
        "1.2.3-a_1",
    ] {
        assert!(accept_version(None, invalid, false).is_err(), "{invalid}");
    }
}

#[test]
fn future_schema_and_minimum_setup_refuse_with_the_upgrade_command() {
    let mut manifest = body();
    manifest["schema"] = json!(2);
    assert_eq!(
        parse_manifest(&serde_json::to_vec(&manifest).unwrap())
            .unwrap_err()
            .to_string(),
        NEWER_SETUP
    );
    assert_eq!(
        parse_manifest(br#"{"schema":2}"#).unwrap_err().to_string(),
        NEWER_SETUP
    );
    manifest["schema"] = json!(1);
    manifest["min_setup"] = json!("99.0.0");
    assert_eq!(
        parse_manifest(&serde_json::to_vec(&manifest).unwrap())
            .unwrap_err()
            .to_string(),
        NEWER_SETUP
    );
    manifest["min_setup"] = json!("0.0.0");
    manifest["future_field"] = json!({"anything":true});
    assert_eq!(
        parse_manifest(&serde_json::to_vec(&manifest).unwrap())
            .unwrap()
            .version,
        "1.2.3"
    );
    let pointer = json!({"schema":1,"channel":"stable","version":"1.2.3","manifest_url":"https://example.invalid/manifest.json","manifest_sha256":"a".repeat(64),"setup":{},"extra":true});
    parse_pointer(&serde_json::to_vec(&pointer).unwrap()).unwrap();
    assert_eq!(
        parse_pointer(br#"{"schema":2}"#).unwrap_err().to_string(),
        NEWER_SETUP
    );
}

#[test]
fn compiled_anchors_are_the_release_owners_public_keys() {
    let keys = KeySet::default();
    for key in [&keys.primary, &keys.backup] {
        assert!(!key.contains("TEST ONLY"));
        minisign_verify::PublicKey::decode(key).unwrap();
    }
    assert_ne!(keys.primary, keys.backup);
}
