//! Agent-specific model registration for the shared local task host.

pub use larch_task::host::*;

/// Host a pre-owned inference handle through the same durable task executor. The caller must
/// hold the device-ownership protocol for the entire call and supply admission shared with every
/// other user of that handle. `ready` is told the service's token once it owns the folder and
/// listens, so a caller tells its own service from one another process started on the same
/// folder. Model steps are offered `step_tools`' read tools, whose tool steps
/// call its writes on the user's approval, those their run's grants hold, and whose delegate
/// steps hand a goal to a helper that cites what it read (`ActivityRegistry::offer_step_tools`,
/// `register_tool_steps`, `register_delegate`).
#[allow(clippy::too_many_arguments)]
pub fn serve_with_tools(
    data_dir: &std::path::Path,
    larch: larch::Larch,
    admission: super::inference::SeatAdmission,
    params: larch::GenParams,
    context_tokens: usize,
    capacity: usize,
    step_tools: Option<(super::driver::StepToolSource, crate::tool::ToolCtx)>,
    ready: Ready,
) -> Result<(), String> {
    serve_with_components_notify(
        data_dir,
        move |actor, stop| {
            let mut registry = super::driver::ActivityRegistry::new();
            registry.register_checks(
                actor.clone(),
                larch.clone(),
                admission.clone(),
                params.clone(),
                context_tokens,
                stop.clone(),
            )?;
            // A helper must cite the tool results it read, so delegation comes with step tools.
            let delegates = step_tools.is_some();
            if let Some((tools, context)) = step_tools {
                registry.register_tool_steps(actor.clone(), tools.clone(), context.clone())?;
                registry.offer_step_tools(tools, context)?;
            }
            registry.register_model(
                "model",
                actor.clone(),
                larch.clone(),
                admission.clone(),
                params.clone(),
                context_tokens,
                capacity,
                stop.clone(),
            )?;
            if delegates {
                let child = ["model".to_string()].into();
                registry.register_delegate(super::driver::DELEGATE_STEP, actor, child)?;
            }
            let planner = super::planner::LarchPlanner {
                larch,
                seats: admission,
                params,
                context_tokens,
                capacity,
                max_nodes: 32,
                allowed_kinds: registry.activity_kinds(),
                step_tools: registry.step_tools(),
                stop,
            };
            Ok((
                std::sync::Arc::new(registry),
                Some(std::sync::Arc::new(planner)),
            ))
        },
        ready,
    )
}
