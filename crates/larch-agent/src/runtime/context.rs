//! Context packets reconstructed from committed data, never from cached KV state.
use larch_task::types::WorkflowDefinition;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextPacket {
    pub goal: String,
    pub acceptance: Vec<String>,
    pub artifacts: Vec<String>,
    pub history: Vec<String>,
}

impl ContextPacket {
    /// Keep the authoritative goal and acceptance verbatim. Discard oldest history first, then
    /// least-recent artifacts, measuring the entire rendered packet after each change.
    pub fn fit(
        definition: &WorkflowDefinition,
        artifacts: &[String],
        history: &[String],
        max_tokens: usize,
        count: impl Fn(&str) -> usize,
    ) -> Result<Self, &'static str> {
        let mut packet = Self {
            goal: definition.goal.clone(),
            acceptance: definition.acceptance.clone(),
            artifacts: artifacts.to_vec(),
            history: history.to_vec(),
        };
        while count(&packet.render()) > max_tokens {
            if !packet.history.is_empty() {
                packet.history.remove(0);
            } else if !packet.artifacts.is_empty() {
                packet.artifacts.remove(0);
            } else {
                return Err("goal and acceptance exceed context budget");
            }
        }
        Ok(packet)
    }
    pub fn render(&self) -> String {
        // JSON encoding preserves boundaries even when user-controlled text contains labels. An
        // artifact that is JSON already (a step's activity and inputs, a helper's result) goes in
        // as its value: as a string it was escaped again, every quote and newline of its text
        // twice over, which the model read as `\\\"` and paid for in context.
        let artifacts: Vec<serde_json::Value> = (self.artifacts.iter())
            .map(|a| serde_json::from_str(a).unwrap_or_else(|_| a.as_str().into()))
            .collect();
        serde_json::json!({"goal":self.goal,"acceptance":self.acceptance,
            "untrusted_artifacts":artifacts,"untrusted_history":self.history})
        .to_string()
    }
}

impl ContextPacket {
    /// Measure the full model chat template, tools and thinking setting, reserving room for the
    /// answer. No KV cache or process-local state is required to reconstruct the packet.
    pub fn fit_for_session(
        larch: &larch::Larch,
        definition: &WorkflowDefinition,
        system: &str,
        artifacts: &[String],
        history: &[String],
        context_tokens: usize,
        params: &larch::GenParams,
    ) -> Result<Self, String> {
        if params.max_tokens == 0 {
            return Err("a finite response budget is required".into());
        }
        let available = context_tokens
            .checked_sub(params.max_tokens as usize)
            .ok_or("response budget exceeds context")?;
        let mut packet = Self {
            goal: definition.goal.clone(),
            acceptance: definition.acceptance.clone(),
            artifacts: artifacts.to_vec(),
            history: history.to_vec(),
        };
        loop {
            let prompt =
                larch::Prompt::Messages(crate::run::messages(system, &packet.render(), false));
            let size = larch
                .prompt_len(prompt, &params.tools, params.thinking)
                .map_err(|e| e.to_string())?;
            if size <= available {
                return Ok(packet);
            }
            if !packet.history.is_empty() {
                packet.history.remove(0);
            } else if !packet.artifacts.is_empty() {
                packet.artifacts.remove(0);
            } else {
                return Err("goal and acceptance exceed context budget".into());
            }
        }
    }
}
