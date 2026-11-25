//! Capability intersections and revocation checks for runtime effects.
//!
//! Capabilities are opaque, canonical resource identifiers supplied by the trusted host. A
//! child receives only the exact intersection; model-provided strings confer no authority.
use std::collections::BTreeSet;
use std::sync::{Arc, RwLock, RwLockReadGuard};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Scope(pub BTreeSet<String>);

impl Scope {
    pub fn intersect(&self, requested: &Self) -> Self {
        Self(self.0.intersection(&requested.0).cloned().collect())
    }

    pub fn contains(&self, capability: &str) -> bool {
        self.0.contains(capability)
    }
}

/// Updates take an exclusive lock. A dispatch keeps the read guard through its final check and
/// tool call, so a completed revocation cannot race a newly starting effect.
#[derive(Clone, Default)]
pub struct LiveGrants(Arc<RwLock<Scope>>);

impl LiveGrants {
    pub fn new(scope: Scope) -> Self {
        Self(Arc::new(RwLock::new(scope)))
    }
    pub fn replace(&self, scope: Scope) -> Result<(), &'static str> {
        *self.0.write().map_err(|_| "grant lock poisoned")? = scope;
        Ok(())
    }
    pub fn check<'a>(
        &'a self,
        inherited: &Scope,
        capability: &str,
    ) -> Result<RwLockReadGuard<'a, Scope>, &'static str> {
        let guard = self.0.read().map_err(|_| "grant lock poisoned")?;
        if !inherited.contains(capability) || !guard.contains(capability) {
            return Err("capability denied or revoked");
        }
        Ok(guard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scope(values: &[&str]) -> Scope {
        Scope(values.iter().map(|value| (*value).to_owned()).collect())
    }
    #[test]
    fn child_scope_only_narrows_and_revocation_blocks_future_dispatch() {
        let parent = scope(&["read:a", "write:a"]);
        let child = parent.intersect(&scope(&["read:a", "write:b"]));
        assert!(child.contains("read:a"));
        assert!(!child.contains("write:b"));
        let live = LiveGrants::new(parent);
        assert!(live.check(&child, "read:a").is_ok());
        live.replace(Scope::default()).unwrap();
        assert!(live.check(&child, "read:a").is_err());
    }
}
