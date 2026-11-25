//! One-shot injection, once per engine thread. Allocation counts span startup's K/V and three
//! serving buffers, observation growth's three replacement buffers, and reload's five buffers.
//! Model/kernel-owned allocations are outside this seam. DeviceLost and hang are synthetic host
//! faults until the kernel lane supplies a debug device op; neither proves CUDA stickiness or TDR.
use larch_kernels::KError;
use std::cell::RefCell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Trigger {
    Panic,
    DeviceLost,
    Hang,
    Oom,
}
#[derive(Default)]
struct Injection {
    target: Option<(Trigger, u64)>,
    allocations: u64,
}
impl Injection {
    fn parse(value: &str) -> Self {
        let target = value.split_once(':').and_then(|(kind, count)| {
            let kind = match kind {
                "panic@step" => Trigger::Panic,
                "device_lost@step" => Trigger::DeviceLost,
                "hang@step" => Trigger::Hang,
                "oom@alloc" => Trigger::Oom,
                _ => return None,
            };
            if count.is_empty() || !count.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            let count = count.parse::<u64>().ok().filter(|&n| n > 0)?;
            Some((kind, count))
        });
        Self {
            target,
            allocations: 0,
        }
    }
    fn take(&mut self, allocation: bool, count: u64) -> Option<Trigger> {
        let (kind, target) = self.target?;
        if (kind == Trigger::Oom) == allocation && count == target {
            self.target = None;
            Some(kind)
        } else {
            None
        }
    }
}
thread_local! {
    static INJECTION: RefCell<Injection> = RefCell::new(Injection::parse(&std::env::var("LARCH_FAULT").unwrap_or_default()));
}
/// Counts an engine allocation attempt, returning OOM only at the selected attempt.
pub(crate) fn allocation() -> Result<(), KError> {
    INJECTION.with(|state| {
        let mut state = state.borrow_mut();
        state.allocations = state.allocations.saturating_add(1);
        let count = state.allocations;
        if state.take(true, count).is_some() {
            Err(KError::Oom)
        } else {
            Ok(())
        }
    })
}
/// Injects at an executing step, before any device operation or host wait.
pub(crate) fn step(count: u64) -> Result<(), KError> {
    let trigger = INJECTION.with(|state| state.borrow_mut().take(false, count));
    match trigger {
        Some(Trigger::Panic) => panic!("injected engine step panic"),
        Some(Trigger::DeviceLost) => Err(KError::DeviceLost),
        Some(Trigger::Hang) => loop {
            std::thread::park();
        },
        _ => Ok(()),
    }
}
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    /// Installs a thread-local fixture without mutating the process environment.
    pub(crate) fn set(value: &str) {
        INJECTION.with(|state| *state.borrow_mut() = Injection::parse(value));
    }
    #[test]
    fn exact_positive_targets_are_one_shot_and_counter_specific() {
        for (value, kind, allocation) in [
            ("panic@step:2", Trigger::Panic, false),
            ("device_lost@step:2", Trigger::DeviceLost, false),
            ("hang@step:2", Trigger::Hang, false),
            ("oom@alloc:2", Trigger::Oom, true),
        ] {
            let mut injection = Injection::parse(value);
            assert_eq!(injection.take(!allocation, 2), None);
            assert_eq!(injection.take(allocation, 1), None);
            assert_eq!(injection.take(allocation, 2), Some(kind));
            assert_eq!(injection.take(allocation, 2), None);
            assert_eq!(injection.take(allocation, 3), None);
        }
    }
    #[test]
    fn malformed_zero_and_overflow_targets_are_inert() {
        for value in [
            "",
            "panic",
            "panic@step:0",
            "oom@alloc:0",
            "panic@alloc:1",
            "oom@step:1",
            "hang@step:-1",
            "device_lost@step:+1",
            "panic@step:1:2",
            "panic@step:18446744073709551616",
            "panic@step: 1",
            "panic@step:1 ",
        ] {
            let mut injection = Injection::parse(value);
            assert_eq!(injection.take(false, 0), None, "{value}");
            assert_eq!(injection.take(false, 1), None, "{value}");
            assert_eq!(injection.take(true, 1), None, "{value}");
        }
    }
}
