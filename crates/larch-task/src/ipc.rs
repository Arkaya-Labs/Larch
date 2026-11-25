//! Bounded framing for the authenticated local service transport.
//! Framing is not authentication; the host must establish peer identity before reading commands.
use std::io::{self, Read, Write};

pub fn read_frame(reader: &mut impl Read, max: usize) -> io::Result<Vec<u8>> {
    let mut header = [0u8; 4];
    reader.read_exact(&mut header)?;
    let len = u32::from_be_bytes(header) as usize;
    if len > max {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "frame exceeds limit",
        ));
    }
    let mut body = vec![0; len];
    reader.read_exact(&mut body)?;
    Ok(body)
}

pub fn write_frame(writer: &mut impl Write, body: &[u8], max: usize) -> io::Result<()> {
    if body.len() > max || body.len() > u32::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "frame exceeds limit",
        ));
    }
    writer.write_all(&(body.len() as u32).to_be_bytes())?;
    writer.write_all(body)
}

/// The start of every pipe name the service creates; 64 lowercase hex characters follow.
#[cfg(any(windows, test))]
pub(crate) const PIPE_PREFIX: &str = r"\\.\pipe\larch-runtime-";
/// `service.pipe` is only as trusted as the data directory: anything but a local service pipe
/// (a UNC path that would send the user's credentials and the token to another machine, a file)
/// is refused before it is opened.
#[cfg(any(windows, test))]
fn pipe_name(text: &str) -> Result<&str, String> {
    text.strip_prefix(PIPE_PREFIX)
        .is_some_and(|hex| {
            hex.len() == 64 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        })
        .then_some(text)
        .ok_or_else(|| "runtime service endpoint is not a local service pipe".into())
}
#[cfg(test)]
#[test]
fn only_a_local_service_pipe_name_is_accepted() {
    let hex = "0123456789abcdef".repeat(4);
    assert!(pipe_name(&format!("{PIPE_PREFIX}{hex}")).is_ok());
    for bad in [
        format!(r"\\attacker\pipe\larch-runtime-{hex}"),
        format!("{PIPE_PREFIX}{}", hex.to_uppercase()),
        format!("{PIPE_PREFIX}{hex}0"),
        format!("{PIPE_PREFIX}{}", &hex[..63]),
        String::new(),
        r"C:\Users\me\token".into(),
    ] {
        assert!(pipe_name(&bad).is_err(), "{bad}");
    }
}

/// Authenticated client for the local task service. Calls are synchronous and each uses a new
/// bounded connection, so a disconnect cannot consume a durable decision or event.
pub struct Client {
    socket: std::path::PathBuf,
    token: String,
    capabilities: crate::host::Capabilities,
}
impl Client {
    pub fn connect(data_dir: &std::path::Path) -> Result<Self, String> {
        let directory = data_dir.join("runtime-service");
        let token = std::fs::read_to_string(directory.join("token"))
            .map_err(|e| format!("runtime service unavailable: {e}"))?;
        #[cfg(unix)]
        let socket = directory.join("service.sock");
        // The pipe's name is random per service start; the service writes it once it listens.
        #[cfg(windows)]
        let socket = pipe_name(
            &std::fs::read_to_string(directory.join("service.pipe"))
                .map_err(|e| format!("runtime service unavailable: {e}"))?,
        )?
        .into();
        let mut client = Self {
            socket,
            token,
            capabilities: crate::host::Capabilities::default(),
        };
        client.capabilities = client.capabilities()?;
        Ok(client)
    }
    /// What the service offered when this client connected, without another round trip.
    pub fn capabilities_at_connect(&self) -> &crate::host::Capabilities {
        &self.capabilities
    }
    fn request<T: serde::de::DeserializeOwned>(
        &self,
        action: crate::host::Action,
    ) -> Result<T, String> {
        let request = crate::host::Request {
            version: crate::host::PROTOCOL_VERSION,
            token: self.token.clone(),
            action,
        };
        let bytes = serde_json::to_vec(&request).map_err(|e| e.to_string())?;
        let body = self.exchange(&bytes)?;
        let response: crate::host::Response =
            serde_json::from_slice(&body).map_err(|e| format!("invalid service response: {e}"))?;
        match (response.result, response.error) {
            (_, Some(error)) => Err(error),
            (Some(value), None) => {
                serde_json::from_value(value).map_err(|e| format!("invalid service result: {e}"))
            }
            (None, None) => Err("invalid service response envelope".into()),
        }
    }
    #[cfg(unix)]
    fn exchange(&self, bytes: &[u8]) -> Result<Vec<u8>, String> {
        use std::{os::unix::net::UnixStream, time::Duration};
        let mut stream = UnixStream::connect(&self.socket)
            .map_err(|e| format!("runtime service unavailable: {e}"))?;
        crate::peer::authenticate_peer(&stream)?;
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .map_err(|e| e.to_string())?;
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .map_err(|e| e.to_string())?;
        write_frame(&mut stream, bytes, crate::host::MAX_FRAME).map_err(|e| e.to_string())?;
        read_frame(&mut stream, crate::host::MAX_FRAME).map_err(|e| e.to_string())
    }
    #[cfg(windows)]
    fn exchange(&self, bytes: &[u8]) -> Result<Vec<u8>, String> {
        let mut stream = larch_sys::pipe::PipeStream::connect(
            &self.socket.to_string_lossy(),
            std::time::Duration::from_secs(5),
        )
        .map_err(|e| format!("runtime service unavailable: {e}"))?;
        // As on Unix: a server running as another user never sees the token.
        if !stream
            .server_is_current_user()
            .map_err(|e| format!("runtime service unavailable: {e}"))?
        {
            return Err("runtime service endpoint belongs to another user".into());
        }
        write_frame(&mut stream, bytes, crate::host::MAX_FRAME).map_err(|e| e.to_string())?;
        read_frame(&mut stream, crate::host::MAX_FRAME).map_err(|e| e.to_string())
    }
    pub fn capabilities(&self) -> Result<crate::host::Capabilities, String> {
        self.request(crate::host::Action::Capabilities)
    }
    pub fn create(
        &self,
        command_id: &str,
        state: &crate::types::RunState,
    ) -> Result<crate::types::CommandReply, String> {
        self.request(crate::host::Action::Create {
            command_id: command_id.into(),
            state: state.clone(),
        })
    }
    /// Persist a plan and make the run eligible for the configured bounded executor.
    pub fn start_plan(
        &self,
        run_id: &str,
        plan: &crate::execution::ExecutionPlan,
        command_id: &str,
        expected_revision: u64,
    ) -> Result<crate::types::CommandReply, String> {
        self.request(crate::host::Action::Start {
            run_id: run_id.into(),
            plan: plan.clone(),
            command_id: command_id.into(),
            expected_revision,
        })
    }
    pub fn start_goal(
        &self,
        run_id: &str,
        command_id: &str,
        expected_revision: u64,
    ) -> Result<crate::types::CommandReply, String> {
        self.request(crate::host::Action::StartGoal {
            run_id: run_id.into(),
            command_id: command_id.into(),
            expected_revision,
        })
    }
    /// Save a schedule with its immutable activity template; enabling is a separate action.
    #[allow(clippy::too_many_arguments)]
    pub fn create_schedule(
        &self,
        id: &str,
        schedule: crate::schedule::Schedule,
        policy: crate::schedule::Policy,
        definition: crate::types::WorkflowDefinition,
        plan: crate::execution::ExecutionPlan,
        from: chrono::DateTime<chrono::Utc>,
    ) -> Result<(), String> {
        let _: serde_json::Value = self.request(crate::host::Action::CreateSchedule {
            id: id.into(),
            schedule,
            policy,
            definition,
            plan,
            from,
        })?;
        Ok(())
    }
    pub fn list_schedules(&self) -> Result<Vec<crate::schedule::ScheduleRecord>, String> {
        self.request(crate::host::Action::ListSchedules)
    }
    pub fn enable_schedule(
        &self,
        id: &str,
        expected_revision: u64,
        enabled: bool,
    ) -> Result<u64, String> {
        self.request(crate::host::Action::EnableSchedule {
            id: id.into(),
            expected_revision,
            enabled,
        })
    }
    /// Deletes a schedule whose runs are all settled, fenced by the revision the user saw.
    pub fn delete_schedule(&self, id: &str, expected_revision: u64) -> Result<(), String> {
        self.request(crate::host::Action::DeleteSchedule {
            id: id.into(),
            expected_revision,
        })
    }
    /// Deletes a settled run that nothing refers to.
    pub fn delete_run(&self, run_id: &str) -> Result<(), String> {
        self.request(crate::host::Action::DeleteRun {
            run_id: run_id.into(),
        })
    }
    /// Replaces a saved schedule's rule, policy, goal and steps; returns the new revision.
    pub fn update_schedule(
        &self,
        id: &str,
        expected_revision: u64,
        schedule: crate::schedule::Schedule,
        policy: crate::schedule::Policy,
        definition: crate::types::WorkflowDefinition,
        plan: crate::execution::ExecutionPlan,
    ) -> Result<u64, String> {
        self.request(crate::host::Action::UpdateSchedule {
            id: id.into(),
            expected_revision,
            schedule,
            policy,
            definition,
            plan,
        })
    }
    /// Each attempt's wall time and tokens, oldest first, read a page at a time so a long loop
    /// never outgrows a frame.
    pub fn step_stats(&self, run_id: &str) -> Result<Vec<crate::execution::StepStat>, String> {
        let mut all = Vec::new();
        loop {
            let page: Vec<crate::execution::StepStat> =
                self.request(crate::host::Action::StepStats {
                    run_id: run_id.into(),
                    offset: all.len(),
                })?;
            let more = page.len() == crate::host::STATS_PAGE;
            all.extend(page);
            if !more {
                return Ok(all);
            }
        }
    }
    /// The run's execution plan: what each step does.
    pub fn plan(&self, run_id: &str) -> Result<Option<crate::execution::ExecutionPlan>, String> {
        self.request(crate::host::Action::Plan {
            run_id: run_id.into(),
        })
    }
    /// Undelivered notifications, uncertain ones included.
    pub fn notifications(
        &self,
        limit: u32,
    ) -> Result<Vec<crate::watch::PendingNotification>, String> {
        self.request(crate::host::Action::Notifications { limit })
    }
    /// Marks a notification in flight for `transport`; call before showing it, and show it as a
    /// possible repeat when the claim says so.
    pub fn claim_notification(
        &self,
        id: &str,
        transport: &str,
    ) -> Result<crate::watch::NotificationClaim, String> {
        self.request(crate::host::Action::ClaimNotification {
            id: id.into(),
            transport: transport.into(),
        })
    }
    /// Commits delivery with a receipt, or returns it to pending with none.
    pub fn resolve_notification(
        &self,
        id: &str,
        transport: &str,
        receipt: Option<&str>,
    ) -> Result<(), String> {
        self.request(crate::host::Action::ResolveNotification {
            id: id.into(),
            transport: transport.into(),
            receipt: receipt.map(Into::into),
        })
    }
    /// Every run, a page at a time where the host pages (each step's result cut to a card's
    /// preview there), so a long history never outgrows a frame.
    pub fn list(&self) -> Result<Vec<crate::types::RunState>, String> {
        if !self.capabilities.list_pages {
            return self.request(crate::host::Action::List);
        }
        let mut all: Vec<crate::types::RunState> = Vec::new();
        loop {
            let page: Vec<crate::types::RunState> =
                self.request(crate::host::Action::ListPage { offset: all.len() })?;
            if page.is_empty() {
                return Ok(unique_runs(all));
            }
            all.extend(page);
        }
    }
    pub fn get(&self, run_id: &str) -> Result<Option<crate::types::RunState>, String> {
        self.request(crate::host::Action::Get {
            run_id: run_id.into(),
        })
    }
    pub fn events(
        &self,
        run_id: &str,
        cursor: u64,
        limit: u32,
    ) -> Result<Vec<crate::types::Event>, String> {
        self.request(crate::host::Action::Events {
            run_id: run_id.into(),
            cursor,
            limit,
        })
    }
    pub fn control(
        &self,
        run_id: &str,
        command: &crate::types::CommandEnvelope,
    ) -> Result<crate::types::CommandReply, String> {
        self.request(crate::host::Action::Control {
            run_id: run_id.into(),
            command: command.clone(),
        })
    }
    /// Answer one pending decision for the exact action digest the user was shown.
    pub fn respond_decision(
        &self,
        run_id: &str,
        command_id: &str,
        expected_revision: u64,
        request_id: &str,
        action_digest: &str,
        approve: bool,
    ) -> Result<crate::types::CommandReply, String> {
        self.request(crate::host::Action::RespondDecision {
            run_id: run_id.into(),
            command_id: command_id.into(),
            expected_revision,
            request_id: request_id.into(),
            action_digest: action_digest.into(),
            approve,
        })
    }
    pub fn shutdown(&self) -> Result<(), String> {
        let _: serde_json::Value = self.request(crate::host::Action::Shutdown)?;
        Ok(())
    }
}

/// Runs read a page at a time, each once, by id as the store keeps them. Runs come by id and
/// none is deleted, so runs created between pages shift the rest forward and a later page can
/// repeat a block of earlier ones (`[b,c,d]` then `[c,d,e]`); the later copy is the newer read.
fn unique_runs(pages: Vec<crate::types::RunState>) -> Vec<crate::types::RunState> {
    let mut by_id = std::collections::BTreeMap::new();
    for run in pages {
        by_id.insert(run.id.clone(), run);
    }
    by_id.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_oversized_and_truncated_frames() {
        assert!(read_frame(&mut &u32::MAX.to_be_bytes()[..], 100).is_err());
        assert!(read_frame(&mut &[0, 0, 0, 2, 1][..], 100).is_err());
        assert!(write_frame(&mut Vec::new(), &[1, 2], 1).is_err());
        let mut frame = Vec::new();
        write_frame(&mut frame, b"ok", 10).unwrap();
        assert_eq!(read_frame(&mut &frame[..], 10).unwrap(), b"ok");
    }

    /// Two runs created between pages repeat a block of the last page; each run is kept once,
    /// as its later read.
    #[test]
    fn pages_that_repeat_a_block_keep_each_run_once() {
        let def = crate::types::WorkflowDefinition {
            version: crate::types::SCHEMA_VERSION,
            goal: "goal".into(),
            initial_graph: None,
            planning_policy: crate::types::PlanningPolicy::Initial,
            grants: vec![],
            budgets: Default::default(),
            acceptance: vec![],
        };
        let run = |id: &str, result: &str| {
            let mut r = crate::types::RunState::new(id.into(), def.clone()).unwrap();
            r.result = Some(result.into());
            r
        };
        let pages = vec![
            run("b", "old"),
            run("c", "old"),
            run("d", "old"),
            run("c", "new"),
            run("d", "new"),
            run("e", "new"),
        ];
        let got = unique_runs(pages);
        let ids: Vec<_> = got.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["b", "c", "d", "e"]);
        assert_eq!(got[1].result.as_deref(), Some("new"));
    }
}
