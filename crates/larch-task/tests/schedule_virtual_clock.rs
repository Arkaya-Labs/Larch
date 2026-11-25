use chrono::{NaiveDate, NaiveTime, TimeZone, Utc};
use larch_task::schedule::{CatchUp, Overlap, Policy, Schedule, due};
#[test]
fn dst_gap_runs_under_the_earlier_offset_and_fold_uses_earlier_instant() {
    let date = NaiveDate::from_ymd_opt(2024, 3, 10).unwrap();
    let mut s = Schedule::Calendar {
        start: date,
        time: NaiveTime::from_hms_opt(2, 30, 0).unwrap(),
        timezone: chrono_tz::America::New_York,
        rule_revision: chrono_tz::IANA_TZDB_VERSION.into(),
        days: larch_task::schedule::EVERY_DAY,
    };
    assert_eq!(
        s.next_at_or_after(Utc.with_ymd_and_hms(2024, 3, 10, 0, 0, 0).unwrap())
            .unwrap()
            .unwrap(),
        // 02:30 does not exist that day; it runs at 02:30 EST, which the clock shows as 03:30 EDT.
        Utc.with_ymd_and_hms(2024, 3, 10, 7, 30, 0).unwrap()
    );
    assert_eq!(
        s.next_at_or_after(Utc.with_ymd_and_hms(2024, 3, 10, 7, 30, 1).unwrap())
            .unwrap()
            .unwrap(),
        Utc.with_ymd_and_hms(2024, 3, 11, 6, 30, 0).unwrap()
    );
    if let Schedule::Calendar { start, time, .. } = &mut s {
        *start = NaiveDate::from_ymd_opt(2024, 11, 3).unwrap();
        *time = NaiveTime::from_hms_opt(1, 30, 0).unwrap();
    }
    assert_eq!(
        s.next_at_or_after(Utc.with_ymd_and_hms(2024, 11, 3, 0, 0, 0).unwrap())
            .unwrap()
            .unwrap(),
        Utc.with_ymd_and_hms(2024, 11, 3, 5, 30, 0).unwrap()
    );
    assert_eq!(
        s.next_at_or_after(Utc.with_ymd_and_hms(2024, 11, 3, 5, 30, 1).unwrap())
            .unwrap()
            .unwrap(),
        Utc.with_ymd_and_hms(2024, 11, 4, 6, 30, 0).unwrap()
    );
}
#[test]
fn interval_catchup_is_bounded_and_overlap_skips() {
    let anchor = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let now = anchor + chrono::Duration::seconds(30);
    let s = Schedule::Interval {
        anchor,
        seconds: 10,
    };
    let p = Policy {
        overlap: Overlap::Skip,
        catch_up: CatchUp::Latest,
        freshness_seconds: 25,
    };
    let selected = due(&s, "s", anchor, now, p, false, 0, 10).unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].scheduled_at, now);
    assert!(
        due(&s, "s", anchor, now, p, true, 0, 10)
            .unwrap()
            .is_empty()
    );
    // Latest-style catch-up computes the newest occurrence: no scan, so no scan limit to exhaust.
    assert_eq!(due(&s, "s", anchor, now, p, false, 0, 1).unwrap(), selected);
    assert_eq!(
        selected,
        due(&s, "s", anchor, now, p, false, 0, 10).unwrap()
    );
}
#[test]
fn rejects_unavailable_timezone_rules() {
    let s = Schedule::Calendar {
        start: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        time: NaiveTime::from_hms_opt(0, 0, 0).unwrap(),
        timezone: chrono_tz::UTC,
        rule_revision: "unavailable".into(),
        days: larch_task::schedule::EVERY_DAY,
    };
    assert!(
        s.next_at_or_after(Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap())
            .is_err()
    );
}
#[test]
fn a_weekday_calendar_skips_the_days_it_does_not_name_both_ways() {
    // Monday, Wednesday and Friday at 07:45 in New York; 15 Nov 2024 is a Friday.
    let s = Schedule::Calendar {
        start: NaiveDate::from_ymd_opt(2024, 11, 1).unwrap(),
        time: NaiveTime::from_hms_opt(7, 45, 0).unwrap(),
        timezone: chrono_tz::America::New_York,
        rule_revision: chrono_tz::IANA_TZDB_VERSION.into(),
        days: 0b10101,
    };
    let friday_after = Utc.with_ymd_and_hms(2024, 11, 15, 13, 0, 0).unwrap();
    assert_eq!(
        s.next_at_or_after(friday_after).unwrap(),
        Some(Utc.with_ymd_and_hms(2024, 11, 18, 12, 45, 0).unwrap())
    );
    let sunday = Utc.with_ymd_and_hms(2024, 11, 17, 20, 0, 0).unwrap();
    assert_eq!(
        s.latest_at_or_before(sunday).unwrap(),
        Some(Utc.with_ymd_and_hms(2024, 11, 15, 12, 45, 0).unwrap())
    );
    let mut never = s.clone();
    if let Schedule::Calendar { days, .. } = &mut never {
        *days = 0;
    }
    assert!(never.next_at_or_after(sunday).is_err());
    // A schedule saved before weekdays existed runs every day.
    let mut old = serde_json::to_value(&s).unwrap();
    old["Calendar"].as_object_mut().unwrap().remove("days");
    let old: Schedule = serde_json::from_value(old).unwrap();
    assert!(matches!(
        old,
        Schedule::Calendar {
            days: larch_task::schedule::EVERY_DAY,
            ..
        }
    ));
}
