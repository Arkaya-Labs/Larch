use super::filesystem::*;
use super::*;

impl Transaction {
    /// Takes the existing lock exclusively without waiting or changing installed files.
    pub fn acquire(root: &Path) -> Result<Self> {
        mkdir(root)?;
        let root = fs::canonicalize(root)?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(root.join(".lock"))?;
        lock.sync_all()?;
        sync_parent(&root.join(".lock"))?;
        match lock.try_lock() {
            Ok(()) => Ok(Self { root, _lock: lock }),
            Err(fs::TryLockError::WouldBlock) => Err(Error::Busy),
            Err(fs::TryLockError::Error(e)) => Err(e.into()),
        }
    }
    /// The canonical root protected by this lock.
    pub fn root(&self) -> &Path {
        &self.root
    }
    /// Reads only the atomically committed receipt.
    pub fn receipt(&self) -> Result<Option<Receipt>> {
        match fs::read(self.root.join("receipt.json")) {
            Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
    /// The exact digest used to detect out-of-band receipt changes.
    pub fn receipt_digest(&self) -> Result<String> {
        match larch_fetch::sha256_file(&self.root.join("receipt.json")) {
            Ok(hash) => Ok(hash),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                Ok("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".into())
            }
            Err(e) => Err(e.into()),
        }
    }
    fn journal(&self, journal: &Journal) -> Result<()> {
        atomic_write(
            &self.root.join("journal.json"),
            &serde_json::to_vec(journal)?,
        )
    }
    fn trash(&self, journal: &Journal, index: usize) -> PathBuf {
        self.root
            .join(".trash")
            .join(&journal.txn)
            .join(index.to_string())
    }
    fn paths(&self, step: &Step, journal: &Journal, index: usize) -> Option<(PathBuf, PathBuf)> {
        match step {
            Step::Rename { from, to } | Step::Link { from, to } => Some((from.clone(), to.clone())),
            Step::Delete { path } => Some((path.clone(), self.trash(journal, index))),
            _ => None,
        }
    }
    /// Checks staging and metadata paths before any pre-journal filesystem write.
    pub fn check_path(&self, path: &Path) -> Result<()> {
        if !path.is_absolute()
            || path
                .components()
                .any(|p| matches!(p, std::path::Component::ParentDir))
        {
            return Err(Error::Refused(format!(
                "unsafe transaction path: {}",
                path.display()
            )));
        }
        let roots = [self.root.clone()];
        #[cfg(target_os = "macos")]
        let mut roots = roots.to_vec();
        #[cfg(target_os = "macos")]
        if let Ok(user) = crate::effects::Roots::from_env() {
            roots.push(user.home.join("Applications/Larch.app"));
            roots.push(user.home.join("Applications/.larch-staging"));
        }
        let parent = existing_ancestor(
            path.parent()
                .ok_or_else(|| Error::Refused("transaction path has no parent".into()))?,
        )?;
        if !roots.iter().any(|root| {
            let boundary = fs::symlink_metadata(root)
                .ok()
                .filter(|m| m.is_dir())
                .and_then(|_| fs::canonicalize(root).ok())
                .unwrap_or_else(|| root.clone());
            path != self.root
                && path.starts_with(root)
                && (parent.starts_with(&boundary)
                    || (root != &self.root
                        && (path == root || !root.exists())
                        && root.parent().is_some_and(|p| {
                            existing_ancestor(p).is_ok_and(|ancestor| ancestor == parent)
                        })))
        }) {
            return Err(Error::Refused(format!(
                "transaction path escapes its installation: {}",
                path.display()
            )));
        }
        Ok(())
    }
    fn validate(&self, journal: &Journal) -> Result<()> {
        if journal.txn.is_empty()
            || !journal
                .txn
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b))
            || journal.done > journal.steps.len()
            || journal.to.root != self.root
            || journal.to.schema != 1
            || journal
                .prepared
                .is_some_and(|i| i != journal.done || i >= journal.steps.len())
        {
            return Err(Error::Refused(
                "invalid transaction journal identity, receipt root or done counter".into(),
            ));
        }
        for dir in &journal.created_dirs {
            self.check_path(dir)?;
        }
        for (i, step) in journal.steps.iter().enumerate() {
            if let Some((from, to)) = self.paths(step, journal, i) {
                self.check_path(&from)?;
                self.check_path(&to)?;
                if !matches!(step, Step::Link { .. }) && !same_volume(&from, &to)? {
                    return Err(Error::Refused(format!(
                        "cross-volume rename refused before transaction: {} -> {}",
                        from.display(),
                        to.display()
                    )));
                }
            }
        }
        Ok(())
    }
    fn done(&self, journal: &Journal, index: usize, hooks: &mut impl Hooks) -> Result<bool> {
        Ok(match &journal.steps[index] {
            Step::Rename { from, to } => exists(to)? && !exists(from)?,
            Step::Delete { path } => exists(&self.trash(journal, index))? && !exists(path)?,
            Step::Link { from, to } => same_file(from, to)?,
            Step::Effect {
                kind,
                value,
                action,
            } => hooks.present(kind, value)? == (*action == Action::Apply),
            Step::SelfTest { .. } => false,
        })
    }
    fn apply(&self, journal: &Journal, index: usize, hooks: &mut impl Hooks) -> Result<()> {
        match &journal.steps[index] {
            Step::Rename { from, to } => {
                if exists(to)? {
                    return Err(Error::Refused(format!(
                        "rename destination already exists: {}",
                        to.display()
                    )));
                }
                mkdir(to.parent().unwrap())?;
                rename_retry(from, to)?;
                sync_parent(from)?;
                sync_parent(to)?;
            }
            Step::Delete { path } => {
                if fs::symlink_metadata(path)?.is_dir() {
                    return Err(Error::Refused(format!(
                        "refusing to delete a directory: {}",
                        path.display()
                    )));
                }
                let to = self.trash(journal, index);
                if exists(&to)? {
                    return Err(Error::Refused(format!(
                        "trash destination already exists: {}",
                        to.display()
                    )));
                }
                mkdir(to.parent().unwrap())?;
                rename_retry(path, &to)?;
                sync_parent(path)?;
                sync_parent(&to)?;
            }
            Step::Link { from, to } => {
                if same_file(from, to)? {
                    return Ok(());
                }
                if exists(to)? && journal.before.get(&index).is_some_and(|b| b.link_existed) {
                    return Err(Error::Refused(format!(
                        "refusing to replace an existing link destination: {}",
                        to.display()
                    )));
                }
                mkdir(to.parent().unwrap())?;
                if let Err(error) = fs::hard_link(from, to) {
                    self.log(&format!(
                        "hard link failed ({error}); copying {}",
                        to.display()
                    ));
                    let temp = to.with_file_name(format!(".larch-copy-{}-{index}", journal.txn));
                    match fs::remove_file(&temp) {
                        Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e.into()),
                        _ => {}
                    }
                    let mut copy = File::options().write(true).create_new(true).open(&temp)?;
                    io::copy(&mut File::open(from)?, &mut copy)?;
                    copy.set_permissions(fs::metadata(from)?.permissions())?;
                    copy.sync_all()?;
                    drop(copy);
                    rename_retry(&temp, to)?;
                }
                sync_parent(to)?;
            }
            Step::Effect {
                kind,
                value,
                action,
            } => match action {
                Action::Apply => hooks.apply(kind, value)?,
                Action::Remove => hooks.remove(kind, value)?,
            },
            Step::SelfTest { component } => hooks.self_test(component)?,
        }
        Ok(())
    }
    fn undo(&self, journal: &Journal, index: usize, hooks: &mut impl Hooks) -> Result<()> {
        match &journal.steps[index] {
            Step::Rename { .. } | Step::Delete { .. } => {
                let (from, to) = self.paths(&journal.steps[index], journal, index).unwrap();
                if journal
                    .before
                    .get(&index)
                    .is_some_and(|b| !b.source_existed || b.destination_existed)
                {
                    return Ok(());
                }
                if exists(&from)? && !exists(&to)? {
                    return Ok(());
                }
                if exists(&to)? && !exists(&from)? {
                    mkdir(from.parent().unwrap())?;
                    rename_retry(&to, &from)?;
                    sync_parent(&to)?;
                    sync_parent(&from)?;
                } else {
                    return Err(Error::StateChangedOutside);
                }
            }
            Step::Link { to, .. } => {
                if !journal.before.get(&index).is_some_and(|b| b.link_existed) {
                    match fs::remove_file(to) {
                        Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e.into()),
                        _ => {}
                    }
                    let temp = to.with_file_name(format!(".larch-copy-{}-{index}", journal.txn));
                    match fs::remove_file(temp) {
                        Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e.into()),
                        _ => {}
                    }
                    if to.parent().is_some_and(Path::exists) {
                        sync_parent(to)?;
                    }
                }
            }
            Step::Effect { kind, value, .. } => {
                if let Some(state) = journal
                    .before
                    .get(&index)
                    .and_then(|b| b.effect_state.as_deref())
                {
                    return hooks.restore(kind, value, state);
                }
                let before = journal
                    .before
                    .get(&index)
                    .and_then(|b| b.effect_present)
                    .ok_or_else(|| {
                        Error::Refused("journal is missing the effect's before-state".into())
                    })?;
                if before {
                    hooks.apply(kind, value)?;
                } else {
                    hooks.remove(kind, value)?;
                }
            }
            Step::SelfTest { .. } => {}
        }
        Ok(())
    }
    fn rollback(&self, journal: &mut Journal, hooks: &mut impl Hooks) -> Result<()> {
        journal.rolling_back = true;
        self.journal(journal)?;
        if let Some(index) = journal.prepared {
            self.undo(journal, index, hooks)?;
            journal.prepared = None;
            self.journal(journal)?;
        }
        while journal.done > 0 {
            self.undo(journal, journal.done - 1, hooks)?;
            journal.done -= 1;
            self.journal(journal)?;
        }
        fs::remove_file(self.root.join("journal.json"))?;
        sync_parent(&self.root.join("journal.json"))?;
        for dir in journal.created_dirs.iter().rev() {
            match fs::remove_dir(dir) {
                Err(e)
                    if !matches!(
                        e.kind(),
                        io::ErrorKind::NotFound | io::ErrorKind::DirectoryNotEmpty
                    ) =>
                {
                    self.log(&format!(
                        "rollback directory cleanup {}: {e}",
                        dir.display()
                    ))
                }
                _ => {}
            }
        }
        Ok(())
    }
    fn forward(&self, journal: &mut Journal, hooks: &mut impl Hooks) -> Result<()> {
        while journal.done < journal.steps.len() {
            let index = journal.done;
            if journal.prepared == Some(index) && self.done(journal, index, hooks)? {
                journal.done += 1;
                journal.prepared = None;
                self.journal(journal)?;
                continue;
            }
            if journal.prepared.is_none() {
                let before = match &journal.steps[index] {
                    Step::Effect { kind, value, .. } => Before {
                        effect_present: Some(hooks.present(kind, value)?),
                        effect_state: hooks.snapshot(kind, value)?,
                        ..Before::default()
                    },
                    Step::Link { to, .. } => Before {
                        link_existed: exists(to)?,
                        ..Before::default()
                    },
                    Step::Rename { .. } | Step::Delete { .. } => {
                        let (from, to) = self.paths(&journal.steps[index], journal, index).unwrap();
                        Before {
                            source_existed: exists(&from)?,
                            destination_existed: exists(&to)?,
                            ..Before::default()
                        }
                    }
                    _ => Before::default(),
                };
                journal.before.insert(index, before);
                journal.prepared = Some(index);
                self.journal(journal)?;
            }
            self.apply(journal, index, hooks).map_err(|e| {
                Error::Refused(format!(
                    "step {index} {:?} failed: {e}",
                    journal.steps[index]
                ))
            })?;
            crash(index, true);
            journal.done += 1;
            journal.prepared = None;
            self.journal(journal)?;
            crash(index, false);
        }
        self.commit(journal)
    }
    /// Appends `text` to the setup log; a log that cannot be written is reported on stderr only.
    pub(crate) fn log(&self, text: &str) {
        if let Err(e) = crate::log::Log::new(&self.root).line(text) {
            eprintln!("setup log write failed: {e}");
        }
    }
    fn commit(&self, journal: &Journal) -> Result<()> {
        let receipt = self.root.join("receipt.json");
        let bytes = serde_json::to_vec(&journal.to)?;
        // The receipt's rename is the commit point: a failure past it (here the directory sync)
        // must not reach the caller's rollback, which would undo files the receipt now names. The
        // proof of the rename is `recover`'s: a receipt changed since the journal began that holds
        // the journal's exact bytes. A journal left behind is finished by `recover` the same way.
        if let Err(e) = atomic_write(&receipt, &bytes) {
            let renamed = self
                .receipt_digest()
                .is_ok_and(|d| d != journal.from_receipt)
                && fs::read(&receipt).is_ok_and(|b| b == bytes);
            if !renamed {
                return Err(e);
            }
            self.log(&format!("committed; receipt sync failed: {e}"));
        }
        let journal_path = self.root.join("journal.json");
        if let Err(e) = fs::remove_file(&journal_path).and_then(|()| sync_parent(&journal_path)) {
            self.log(&format!("committed; journal cleanup failed: {e}"));
        }
        let trash = self.root.join(".trash");
        if trash.exists()
            && let Err(e) = fs::remove_dir_all(&trash)
        {
            self.log(&format!("committed; trash cleanup failed: {e}"));
        }
        for (i, step) in journal.steps.iter().enumerate() {
            if let Some((from, _)) = self.paths(step, journal, i) {
                for dir in from.ancestors().skip(1) {
                    if dir == self.root || !dir.starts_with(&self.root) {
                        break;
                    }
                    match fs::remove_dir(dir) {
                        Ok(()) => {}
                        Err(e)
                            if matches!(
                                e.kind(),
                                io::ErrorKind::NotFound | io::ErrorKind::DirectoryNotEmpty
                            ) =>
                        {
                            break;
                        }
                        Err(e) => {
                            self.log(&format!("committed; directory cleanup failed: {e}"));
                            break;
                        }
                    }
                }
            }
        }
        Ok(())
    }
    /// Recovers before a new mutation, retaining the old state when staged data is invalid.
    pub fn recover(
        &self,
        hooks: &mut impl Hooks,
        staged_ok: impl FnOnce(&Journal) -> Result<bool>,
    ) -> Result<Recovery> {
        let bytes = match fs::read(self.root.join("journal.json")) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Recovery::None),
            Err(e) => return Err(e.into()),
        };
        let mut journal: Journal = serde_json::from_slice(&bytes)?;
        let digest = self.receipt_digest()?;
        if digest != journal.from_receipt {
            let committed_bytes = serde_json::to_vec(&journal.to)?;
            // Receipt replacement can survive a crash before journal deletion; its exact bytes prove commit.
            if journal.done == journal.steps.len()
                && !journal.rolling_back
                && fs::read(self.root.join("receipt.json"))
                    .is_ok_and(|bytes| bytes == committed_bytes)
            {
                self.validate(&journal)?;
                self.commit(&journal)?;
                return Ok(Recovery::Forward);
            }
            return Err(Error::StateChangedOutside);
        }
        self.validate(&journal)?;
        if !journal.rolling_back && staged_ok(&journal)? {
            match self.forward(&mut journal, hooks) {
                Ok(()) => return Ok(Recovery::Forward),
                Err(error) => {
                    if let Err(undo) = self.rollback(&mut journal, hooks) {
                        return Err(Error::Refused(format!(
                            "{error}; rollback incomplete: {undo}; run larch-setup repair"
                        )));
                    }
                    return Err(error);
                }
            }
        }
        self.rollback(&mut journal, hooks)?;
        Ok(Recovery::Back)
    }
    /// Executes an unstarted plan, rolling back every prepared or completed step on failure.
    pub fn run(&self, mut journal: Journal, hooks: &mut impl Hooks) -> Result<Receipt> {
        if exists(&self.root.join("journal.json"))? {
            return Err(Error::Refused(
                "a transaction needs recovery before another can start".into(),
            ));
        }
        self.validate(&journal)?;
        if journal.done != 0
            || journal.prepared.is_some()
            || journal.from_receipt != self.receipt_digest()?
        {
            return Err(Error::StateChangedOutside);
        }
        for (i, step) in journal.steps.iter().enumerate() {
            if let Some((_, to)) = self.paths(step, &journal, i) {
                for dir in to.ancestors().skip(1) {
                    if dir == self.root || dir.exists() {
                        break;
                    }
                    if !journal.created_dirs.iter().any(|d| d == dir) {
                        journal.created_dirs.push(dir.into());
                    }
                }
            }
        }
        journal.created_dirs.sort_by_key(|p| p.components().count());
        self.journal(&journal)?;
        match self.forward(&mut journal, hooks) {
            Ok(()) => Ok(journal.to),
            Err(error) => {
                if let Err(undo) = self.rollback(&mut journal, hooks) {
                    return Err(Error::Refused(format!(
                        "{error}; rollback incomplete: {undo}; run larch-setup repair"
                    )));
                }
                Err(error)
            }
        }
    }
}
