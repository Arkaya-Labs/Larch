use chrono::{DateTime, Duration, TimeZone, Utc};
use larch_task::{
    schedule::{
        CatchUp, MISSED_OVERLAP, MISSED_STALE, MISSED_SUPERSEDED, Missed, Overlap, Policy, Schedule,
    },
    store::SqliteStore,
    types::{
        Attempt, Command, CommandEnvelope, PlanningPolicy, RunStatus, SCHEMA_VERSION,
        WorkflowDefinition,
    },
};
use std::collections::BTreeMap;

fn path() -> std::path::PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("larch-overlap-{}-{n}.sqlite", std::process::id()))
}
fn anchor() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
}
/// An hourly schedule with this overlap, already enabled.
fn store(overlap: Overlap) -> (SqliteStore, std::path::PathBuf) {
    store_with(
        Schedule::Interval {
            anchor: anchor(),
            seconds: 3600,
        },
        Policy {
            overlap,
            catch_up: CatchUp::All { max: 10 },
            freshness_seconds: 86400,
        },
    )
}
fn definition() -> WorkflowDefinition {
    WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "hourly".into(),
        initial_graph: None,
        planning_policy: PlanningPolicy::Initial,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    }
}
fn store_with(schedule: Schedule, policy: Policy) -> (SqliteStore, std::path::PathBuf) {
    let file = path();
    let mut s = SqliteStore::open(&file).unwrap();
    s.create_schedule("s", &schedule, policy, &definition(), anchor())
        .unwrap();
    s.enable_schedule("s", 0, true, anchor()).unwrap();
    (s, file)
}
fn span(first: DateTime<Utc>, last: DateTime<Utc>, count: u64, reason: &str) -> Missed {
    Missed {
        first,
        last,
        count,
        reason: reason.into(),
    }
}
fn missed(s: &SqliteStore) -> Vec<Missed> {
    s.missed_occurrences("s", 100).unwrap()
}
fn control(s: &mut SqliteStore, at: DateTime<Utc>, command: Command) {
    let id = run(at);
    let revision = s.get(&id).unwrap().unwrap().revision;
    s.execute(
        &id,
        &CommandEnvelope {
            id: format!("control-{revision}-{id}"),
            expected_revision: revision,
            command,
        },
    )
    .unwrap();
}
fn hour(n: i64) -> DateTime<Utc> {
    anchor() + Duration::hours(n)
}
fn run(at: DateTime<Utc>) -> String {
    let stamp = at.to_rfc3339_opts(chrono::SecondsFormat::Nanos, true);
    format!("schedule:1:s:{stamp}")
}
fn start(s: &mut SqliteStore, at: DateTime<Utc>) -> Result<(), String> {
    let id = run(at);
    let command = CommandEnvelope {
        id: format!("start-{id}"),
        expected_revision: 0,
        command: Command::StartAttempt {
            attempt: Attempt {
                iteration: None,
                id: format!("attempt-{id}"),
                node_id: "__planning__".into(),
                epoch: 1,
            },
        },
    };
    s.claim_scheduled_start_once(&format!("start:{id}"), &command)
        .map(|_| ())
}
fn finish(s: &mut SqliteStore, at: DateTime<Utc>) {
    let id = run(at);
    let revision = s.get(&id).unwrap().unwrap().revision;
    s.execute(
        &id,
        &CommandEnvelope {
            id: format!("cancel-{id}"),
            expected_revision: revision,
            command: Command::Cancel,
        },
    )
    .unwrap();
}
fn pumped(s: &mut SqliteStore, now: DateTime<Utc>) -> Vec<DateTime<Utc>> {
    let revision = s.schedule_revision("s").unwrap().unwrap();
    let result = s.pump_schedule("s", revision, now, 100).unwrap();
    result.into_iter().map(|o| o.scheduled_at).collect()
}
fn minutely(overlap: Overlap, catch_up: CatchUp) -> (SqliteStore, std::path::PathBuf) {
    store_with(
        Schedule::Interval {
            anchor: anchor(),
            seconds: 60,
        },
        Policy {
            overlap,
            catch_up,
            freshness_seconds: 86400,
        },
    )
}
fn minute(n: i64) -> DateTime<Utc> {
    anchor() + Duration::minutes(n)
}

#[test]
fn queue_starts_occurrences_one_at_a_time_oldest_first() {
    let (mut s, file) = store(Overlap::Queue { max_queued: 1 });
    assert_eq!(pumped(&mut s, hour(0)), vec![hour(0)]);
    assert_eq!(pumped(&mut s, hour(1)), vec![hour(1)]);
    // The queue is full: the third stays due for a later tick, not yet missed.
    let revision = s.schedule_revision("s").unwrap();
    assert!(pumped(&mut s, hour(2)).is_empty());
    assert_eq!(
        s.schedule_revision("s").unwrap(),
        revision,
        "nothing changed"
    );
    assert!(missed(&s).is_empty());
    assert!(start(&mut s, hour(1)).is_err(), "out of turn");
    start(&mut s, hour(0)).unwrap();
    assert!(start(&mut s, hour(1)).is_err(), "first still unfinished");
    finish(&mut s, hour(0));
    start(&mut s, hour(1)).unwrap();
    drop(s);
    let _ = std::fs::remove_file(file);
}

#[test]
fn buffer_latest_replaces_the_unstarted_occurrence_with_the_newest() {
    let (mut s, file) = store(Overlap::BufferLatest);
    assert_eq!(pumped(&mut s, hour(0)), vec![hour(0)]);
    start(&mut s, hour(0)).unwrap();
    assert_eq!(pumped(&mut s, hour(1)), vec![hour(1)]);
    // Two more fall due while the first runs: only the newest is buffered.
    assert_eq!(pumped(&mut s, hour(3)), vec![hour(3)]);
    let superseded = s.get(&run(hour(1))).unwrap().unwrap();
    assert_eq!(superseded.status, RunStatus::Cancelled);
    assert_eq!(superseded.result.as_deref(), Some(MISSED_SUPERSEDED));
    let commands: u32 = rusqlite::Connection::open(&file)
        .unwrap()
        .query_row(
            "SELECT count(*) FROM commands WHERE run_id=?1 AND id LIKE 'superseded:%'",
            [run(hour(1))],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(commands, 1, "the cancel has its command record");
    assert_eq!(
        missed(&s),
        vec![span(hour(1), hour(2), 2, MISSED_SUPERSEDED)]
    );
    let outbox: Vec<String> = s
        .pending_outbox(10)
        .unwrap()
        .into_iter()
        .map(|e| e.payload)
        .collect();
    assert_eq!(
        outbox,
        vec![run(hour(3))],
        "the superseded start is retired"
    );
    assert!(start(&mut s, hour(3)).is_err(), "waits for the running one");
    finish(&mut s, hour(0));
    start(&mut s, hour(3)).unwrap();
    drop(s);
    let _ = std::fs::remove_file(file);
}

#[test]
fn pausing_and_resuming_the_buffered_run_keeps_buffer_latest_working() {
    let (mut s, file) = store(Overlap::BufferLatest);
    assert_eq!(pumped(&mut s, hour(0)), vec![hour(0)]);
    start(&mut s, hour(0)).unwrap();
    assert_eq!(pumped(&mut s, hour(1)), vec![hour(1)]);
    control(&mut s, hour(1), Command::Pause);
    control(&mut s, hour(1), Command::Resume);
    assert_eq!(s.get(&run(hour(1))).unwrap().unwrap().revision, 2);
    // Planning is a start too: it waits its turn like the outbox claim.
    let planning = CommandEnvelope {
        id: "plan-early".into(),
        expected_revision: 2,
        command: Command::StartAttempt {
            attempt: Attempt {
                iteration: None,
                id: "planner-early".into(),
                node_id: "__planning__".into(),
                epoch: 1,
            },
        },
    };
    assert!(s.execute(&run(hour(1)), &planning).is_err(), "out of turn");
    assert_eq!(pumped(&mut s, hour(2)), vec![hour(2)]);
    assert_eq!(
        s.get(&run(hour(1))).unwrap().unwrap().status,
        RunStatus::Cancelled
    );
    assert_eq!(
        missed(&s),
        vec![span(hour(1), hour(1), 1, MISSED_SUPERSEDED)]
    );
    assert_eq!(pumped(&mut s, hour(3)), vec![hour(3)], "still advancing");
    drop(s);
    let _ = std::fs::remove_file(file);
}

#[test]
fn buffer_latest_supersedes_a_buffered_run_left_after_the_running_one_finished() {
    let (mut s, file) = store(Overlap::BufferLatest);
    assert_eq!(pumped(&mut s, hour(0)), vec![hour(0)]);
    start(&mut s, hour(0)).unwrap();
    assert_eq!(pumped(&mut s, hour(1)), vec![hour(1)]);
    // The running one finishes before the worker claims the buffered one.
    finish(&mut s, hour(0));
    assert_eq!(pumped(&mut s, hour(2)), vec![hour(2)]);
    assert_eq!(
        s.get(&run(hour(1))).unwrap().unwrap().status,
        RunStatus::Cancelled,
        "only one occurrence stays buffered"
    );
    start(&mut s, hour(2)).unwrap();
    drop(s);
    let _ = std::fs::remove_file(file);
}

#[test]
fn parallel_starts_beside_others_up_to_its_bound() {
    let (mut s, file) = store(Overlap::Parallel { max_active: 2 });
    assert_eq!(pumped(&mut s, hour(1)), vec![hour(0), hour(1)]);
    start(&mut s, hour(1)).unwrap();
    start(&mut s, hour(0)).unwrap();
    assert!(pumped(&mut s, hour(2)).is_empty());
    assert_eq!(missed(&s), vec![span(hour(2), hour(2), 1, MISSED_OVERLAP)]);
    finish(&mut s, hour(0));
    assert_eq!(pumped(&mut s, hour(3)), vec![hour(3)]);
    drop(s);
    let _ = std::fs::remove_file(file);
}

#[test]
fn a_late_wake_starts_the_newest_and_records_the_rest_once() {
    let (mut s, file) = store(Overlap::Skip);
    // Two days late: everything before the cutoff is one stale span, the fresh ones a newer
    // occurrence superseded, and the newest starts.
    assert_eq!(pumped(&mut s, hour(48)), vec![hour(48)]);
    let expected = vec![
        span(hour(24), hour(47), 24, MISSED_SUPERSEDED),
        span(hour(0), hour(23), 24, MISSED_STALE),
    ];
    assert_eq!(missed(&s), expected);
    start(&mut s, hour(48)).unwrap();
    assert!(pumped(&mut s, hour(48)).is_empty());
    drop(s);
    let mut s = SqliteStore::open(&file).unwrap();
    assert_eq!(missed(&s), expected);
    // Only an occurrence that comes due while the run is unfinished is an overlap.
    assert!(pumped(&mut s, hour(49)).is_empty());
    assert_eq!(missed(&s)[0], span(hour(49), hour(49), 1, MISSED_OVERLAP));
    drop(s);
    let _ = std::fs::remove_file(file);
}

#[test]
fn a_queued_occurrence_that_later_starts_is_never_reported_missed() {
    let (mut s, file) = store(Overlap::Queue { max_queued: 1 });
    assert_eq!(pumped(&mut s, hour(1)), vec![hour(0), hour(1)]);
    // Full queue: hour 2 is deferred, hour 3 is beyond it.
    assert!(pumped(&mut s, hour(3)).is_empty());
    assert!(missed(&s).is_empty());
    start(&mut s, hour(0)).unwrap();
    finish(&mut s, hour(0));
    // Capacity frees: the deferred occurrences start in order.
    assert_eq!(pumped(&mut s, hour(3)), vec![hour(2)]);
    start(&mut s, hour(1)).unwrap();
    finish(&mut s, hour(1));
    start(&mut s, hour(2)).unwrap();
    assert_eq!(
        s.get(&run(hour(2))).unwrap().unwrap().status,
        RunStatus::Running
    );
    assert_eq!(pumped(&mut s, hour(3)), vec![hour(3)]);
    assert!(
        missed(&s).is_empty(),
        "an executed occurrence is not missed"
    );
    drop(s);
    let _ = std::fs::remove_file(file);
}

#[test]
fn a_sleep_of_two_thousand_periods_jumps_to_the_latest_occurrence() {
    let (mut s, file) = minutely(Overlap::Skip, CatchUp::Latest);
    assert_eq!(pumped(&mut s, minute(0)), vec![minute(0)]);
    finish(&mut s, minute(0));
    // Far past the host's 1024-occurrence scan bound, with a scan limit of one.
    let revision = s.schedule_revision("s").unwrap().unwrap();
    let started = s.pump_schedule("s", revision, minute(2000), 1).unwrap();
    assert_eq!(started.len(), 1);
    assert_eq!(started[0].scheduled_at, minute(2000));
    assert_eq!(
        missed(&s),
        vec![
            span(minute(560), minute(1999), 1440, MISSED_SUPERSEDED),
            span(minute(1), minute(559), 559, MISSED_STALE),
        ]
    );
    finish(&mut s, minute(2000));
    assert_eq!(pumped(&mut s, minute(2001)), vec![minute(2001)]);
    drop(s);
    let _ = std::fs::remove_file(file);
}

#[test]
fn a_queue_full_for_two_thousand_periods_resumes_when_a_slot_frees() {
    let (mut s, file) = minutely(Overlap::Queue { max_queued: 1 }, CatchUp::All { max: 4 });
    assert_eq!(pumped(&mut s, minute(1)), vec![minute(0), minute(1)]);
    start(&mut s, minute(0)).unwrap();
    assert!(pumped(&mut s, minute(2000)).is_empty());
    let stale = vec![span(minute(2), minute(559), 558, MISSED_STALE)];
    assert_eq!(missed(&s), stale, "deferred ones that went stale, once");
    finish(&mut s, minute(0));
    // The oldest fresh occurrence that fits starts; the rest stay due instead of failing the tick.
    assert_eq!(pumped(&mut s, minute(2000)), vec![minute(560)]);
    assert_eq!(missed(&s), stale);
    drop(s);
    let _ = std::fs::remove_file(file);
}

#[test]
fn every_occurrence_passed_by_a_busy_schedule_is_counted_in_one_growing_span() {
    let (mut s, file) = store_with(
        Schedule::Interval {
            anchor: anchor(),
            seconds: 1,
        },
        Policy::default(),
    );
    let second = |n| anchor() + Duration::seconds(n);
    assert_eq!(pumped(&mut s, second(0)), vec![second(0)]);
    start(&mut s, second(0)).unwrap();
    assert!(pumped(&mut s, second(2000)).is_empty());
    assert_eq!(
        missed(&s),
        vec![span(second(1), second(2000), 2000, MISSED_OVERLAP)]
    );
    assert!(pumped(&mut s, second(2001)).is_empty());
    assert_eq!(
        missed(&s),
        vec![span(second(1), second(2001), 2001, MISSED_OVERLAP)],
        "a stuck schedule keeps one row"
    );
    drop(s);
    let _ = std::fs::remove_file(file);
}

#[test]
fn queue_and_parallel_with_latest_catch_up_start_only_the_newest() {
    let latest = |overlap| Policy {
        overlap,
        catch_up: CatchUp::Latest,
        freshness_seconds: 86400,
    };
    let hourly = || Schedule::Interval {
        anchor: anchor(),
        seconds: 3600,
    };
    let (mut s, file) = store_with(hourly(), latest(Overlap::Queue { max_queued: 1 }));
    assert_eq!(pumped(&mut s, hour(2)), vec![hour(2)]);
    assert_eq!(
        pumped(&mut s, hour(5)),
        vec![hour(5)],
        "queued behind hour 2"
    );
    assert!(pumped(&mut s, hour(6)).is_empty(), "queue full: deferred");
    assert_eq!(
        missed(&s),
        vec![
            span(hour(3), hour(4), 2, MISSED_SUPERSEDED),
            span(hour(0), hour(1), 2, MISSED_SUPERSEDED)
        ]
    );
    drop(s);
    let _ = std::fs::remove_file(file);
    let (mut s, file) = store_with(hourly(), latest(Overlap::Parallel { max_active: 2 }));
    assert_eq!(pumped(&mut s, hour(2)), vec![hour(2)]);
    assert_eq!(pumped(&mut s, hour(4)), vec![hour(4)]);
    assert!(pumped(&mut s, hour(5)).is_empty());
    assert_eq!(missed(&s)[0], span(hour(5), hour(5), 1, MISSED_OVERLAP));
    drop(s);
    let _ = std::fs::remove_file(file);
}

#[test]
fn policy_bounds_from_clients_are_checked() {
    let file = path();
    let mut s = SqliteStore::open(&file).unwrap();
    let schedule = Schedule::Once { at: anchor() };
    for (overlap, catch_up) in [
        (Overlap::Parallel { max_active: 0 }, CatchUp::Latest),
        (Overlap::Queue { max_queued: 0 }, CatchUp::Latest),
        (Overlap::Parallel { max_active: 33 }, CatchUp::Latest),
        (Overlap::Skip, CatchUp::All { max: 0 }),
        (Overlap::Skip, CatchUp::All { max: 33 }),
    ] {
        let policy = Policy {
            overlap,
            catch_up,
            freshness_seconds: 60,
        };
        assert!(
            s.create_schedule("bad", &schedule, policy, &definition(), anchor())
                .is_err(),
            "{policy:?}"
        );
    }
    drop(s);
    let _ = std::fs::remove_file(file);
}

#[test]
fn a_clock_that_jumps_ahead_and_back_resumes_in_real_time() {
    let (mut s, file) = store(Overlap::Skip);
    assert_eq!(pumped(&mut s, hour(0)), vec![hour(0)]);
    finish(&mut s, hour(0));
    // The wall clock jumps 100 hours ahead: indistinguishable from a sleep, so it runs.
    assert_eq!(pumped(&mut s, hour(100)), vec![hour(100)]);
    finish(&mut s, hour(100));
    assert!(!missed(&s).is_empty());
    // Back to real time: the cursor would wait for hour 101; the pump notices and rewinds.
    let results = s.pump_due(hour(1), 100).unwrap();
    assert_eq!(results.len(), 1, "a rewound schedule is pumped");
    assert_eq!(
        results[0].outcome.as_ref().unwrap()[0].scheduled_at,
        hour(1)
    );
    assert!(
        missed(&s).iter().all(|span| span.last < hour(1)),
        "no row names an instant that has not happened"
    );
    finish(&mut s, hour(1));
    assert_eq!(pumped(&mut s, hour(2)), vec![hour(2)]);
    drop(s);
    let _ = std::fs::remove_file(file);
}

#[test]
fn a_run_that_happened_before_a_rewind_is_never_recorded_missed() {
    let (mut s, file) = store_with(
        Schedule::Interval {
            anchor: anchor(),
            seconds: 3600,
        },
        Policy::default(),
    );
    assert_eq!(pumped(&mut s, hour(0)), vec![hour(0)]);
    finish(&mut s, hour(0));
    assert_eq!(pumped(&mut s, hour(1)), vec![hour(1)]);
    finish(&mut s, hour(1));
    // Back to hour 0: its occurrence exists, so nothing starts and the cursor moves to hour 1.
    assert!(pumped(&mut s, hour(0)).is_empty());
    // Late at hour 2, only the newest starts; hour 1 already ran, so it is not missed.
    assert_eq!(pumped(&mut s, hour(2)), vec![hour(2)]);
    assert!(
        missed(&s)
            .iter()
            .all(|m| m.last < hour(1) || m.first > hour(1)),
        "{:?}",
        missed(&s)
    );
    drop(s);
    let _ = std::fs::remove_file(file);
}

#[test]
fn a_daily_time_inside_a_dst_gap_still_runs_once() {
    let day = |d| chrono::NaiveDate::from_ymd_opt(2024, 3, d).unwrap();
    let (mut s, file) = store_with(
        Schedule::Calendar {
            start: day(9),
            time: chrono::NaiveTime::from_hms_opt(2, 30, 0).unwrap(),
            timezone: chrono_tz::America::New_York,
            rule_revision: chrono_tz::IANA_TZDB_VERSION.into(),
            days: larch_task::schedule::EVERY_DAY,
        },
        Policy::default(),
    );
    let utc = |d, h, m| Utc.with_ymd_and_hms(2024, 3, d, h, m, 0).unwrap();
    assert_eq!(pumped(&mut s, utc(9, 8, 0)), vec![utc(9, 7, 30)]);
    finish(&mut s, utc(9, 7, 30));
    // 02:30 does not exist on 10 March; it runs at 03:30 EDT instead of vanishing.
    assert_eq!(pumped(&mut s, utc(10, 8, 0)), vec![utc(10, 7, 30)]);
    finish(&mut s, utc(10, 7, 30));
    assert_eq!(pumped(&mut s, utc(12, 7, 0)), vec![utc(12, 6, 30)]);
    // 11 March fell outside the one-day freshness window by the time the schedule was pumped.
    assert_eq!(
        missed(&s),
        vec![span(utc(11, 6, 30), utc(11, 6, 30), 1, MISSED_STALE)]
    );
    drop(s);
    let _ = std::fs::remove_file(file);
}

#[test]
fn a_scheduled_run_names_its_schedule_and_occurrence() {
    use larch_task::schedule::{occurrence_run_id, run_origin};
    assert_eq!(occurrence_run_id("s", hour(3)), run(hour(3)));
    // A colon in the schedule id stays inside the length-prefixed part.
    let id = occurrence_run_id("digest:am", hour(2));
    assert_eq!(run_origin(&id), Some(("digest:am".into(), hour(2))));
    for other in ["desktop-1-0", "schedule:9:s:x", "schedule:x:s:2024", ""] {
        assert_eq!(run_origin(other), None, "{other}");
    }
}
