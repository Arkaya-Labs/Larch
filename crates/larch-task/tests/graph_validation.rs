use larch_task::graph::{Graph, Input, Node, NodeKind, validate_graph, validate_revision};
use std::collections::{BTreeMap, BTreeSet};

fn node(id: &str, deps: &[&str]) -> Node {
    Node {
        when: None,
        optional: Vec::new(),
        id: id.into(),
        depends_on: deps.iter().map(|s| s.to_string()).collect(),
        kind: NodeKind::Activity,
        inputs: BTreeMap::new(),
        outputs: BTreeMap::from([("result".into(), "text".into())]),
    }
}

#[test]
fn checks_topology_and_typed_references() {
    let a = node("a", &[]);
    let mut b = node("b", &["a"]);
    b.inputs.insert(
        "in".into(),
        Input {
            node: "a".into(),
            output: "result".into(),
            schema: "text".into(),
        },
    );
    let mut g = Graph { nodes: vec![a, b] };
    assert!(validate_graph(&g).is_ok());
    g.nodes[1].inputs.get_mut("in").unwrap().schema = "number".into();
    assert!(validate_graph(&g).is_err());
    g.nodes[1].inputs.clear();
    g.nodes[0].depends_on.push("b".into());
    assert!(validate_graph(&g).is_err());
    g.nodes[0].depends_on.clear();
    g.nodes[1].kind = NodeKind::Loop { max_iterations: 0 };
    assert!(validate_graph(&g).is_err());
}

#[test]
fn revisions_preserve_history_authority_and_acceptance() {
    let old = Graph {
        nodes: vec![node("a", &[])],
    };
    let mut new = old.clone();
    let started = BTreeSet::from(["a".to_string()]);
    let grants = BTreeSet::from(["read".to_string()]);
    let checks = BTreeSet::from(["check".to_string()]);
    assert!(
        validate_revision(
            &old, &new, 1, 1, &started, &grants, &grants, &checks, &checks
        )
        .is_ok()
    );
    assert!(
        validate_revision(
            &old, &new, 0, 1, &started, &grants, &grants, &checks, &checks
        )
        .is_err()
    );
    assert!(
        validate_revision(
            &old,
            &new,
            1,
            1,
            &started,
            &grants,
            &BTreeSet::from(["write".into()]),
            &checks,
            &checks,
        )
        .is_err()
    );
    assert!(
        validate_revision(
            &old,
            &new,
            1,
            1,
            &started,
            &grants,
            &grants,
            &checks,
            &BTreeSet::new(),
        )
        .is_err()
    );
    new.nodes[0].outputs.clear();
    assert!(
        validate_revision(
            &old, &new, 1, 1, &started, &grants, &grants, &checks, &checks
        )
        .is_err()
    );
}
