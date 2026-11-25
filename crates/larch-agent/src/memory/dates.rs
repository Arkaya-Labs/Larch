//! The date phrases `recall` reads at the start or end of a query, turned into a range of days in
//! Rust: an 8B model that rewrites temporal queries recalls worse than one that never sees them.
//! Days count from 1970-01-01 (UTC, as the daily notes are), both ends of a range included.

/// The first and last day a phrase names.
pub(crate) type Range = (i64, i64);

const MONTHS: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

const WEEKDAYS: [&str; 7] = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
];

/// Monday 0 to Sunday 6 of a weekday's name, or of its possessive ("saturday's").
fn weekday(word: &str) -> Option<i64> {
    let word = word.strip_suffix("'s").unwrap_or(word);
    WEEKDAYS.iter().position(|d| *d == word).map(|i| i as i64)
}

/// The latest day before `today` that is `weekday`: on a Saturday, last Saturday is a week back.
fn latest(weekday: i64, today: i64) -> i64 {
    // 1970-01-01 was a Thursday.
    today - 1 - ((today + 3).rem_euclid(7) - weekday - 1).rem_euclid(7)
}

/// Days since 1970 of a calendar date that may not exist (Howard Hinnant's `days_from_civil`).
fn civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

/// The day of `y-m-d`, when that date exists and its year has four digits.
pub(crate) fn day(y: i64, m: i64, d: i64) -> Option<i64> {
    let month_len = civil(y, m + 1, 1) - civil(y, m, 1);
    ((1000..=9999).contains(&y) && (1..=12).contains(&m) && (1..=month_len).contains(&d))
        .then(|| civil(y, m, d))
}

/// The day of a `YYYY-MM-DD` date, or of the date that starts `text` (a fact's `created`).
pub(crate) fn iso(text: &str) -> Option<i64> {
    let date = text.get(..10)?;
    let mut parts = date.split('-');
    let mut num = |len| {
        parts
            .next()
            .filter(|p: &&str| p.len() == len && p.bytes().all(|b| b.is_ascii_digit()))?
            .parse()
            .ok()
    };
    let (y, m, d) = (num(4)?, num(2)?, num(2)?);
    if text.len() > 10 && text.as_bytes()[10].is_ascii_digit() {
        return None;
    }
    day(y, m, d)
}

/// The year and month of `day`, `civil`'s exact inverse (Hinnant's `civil_from_days`), before
/// 1970 as after.
pub(crate) fn year_month(day: i64) -> (i64, i64) {
    let z = day + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (era * 400 + yoe + i64::from(m <= 2), m)
}

/// The days of month `m` of year `y`.
fn month(y: i64, m: i64) -> Range {
    (civil(y, m, 1), civil(y, m + 1, 1) - 1)
}

/// A month named without a year is the latest one not after `today`'s month, so "in September"
/// in October is this September and in August last year's.
fn year_for(m: i64, today: i64) -> i64 {
    let (y, now) = year_month(today);
    if m <= now { y } else { y - 1 }
}

pub(crate) fn month_of(word: &str) -> Option<i64> {
    MONTHS
        .iter()
        .position(|name| word == *name || word.len() >= 3 && name.starts_with(word))
        .map(|i| i as i64 + 1)
}

pub(crate) fn year_of(word: &str) -> Option<i64> {
    (word.len() == 4)
        .then(|| word.parse().ok())
        .flatten()
        .filter(|y| (1000..=9999).contains(y))
}

pub(crate) fn day_of_month(word: &str) -> Option<i64> {
    let digits = ["st", "nd", "rd", "th"]
        .iter()
        .find_map(|s| word.strip_suffix(s))
        .unwrap_or(word);
    (digits.len() <= 2).then(|| digits.parse().ok()).flatten()
}

/// One day or month named by all of `words`: `2026-10-03`, `3 October [2026]`, `October 3[rd]
/// [2026]`, `October [2026]` or `2026`; a year left out is `year_for`'s.
fn term(words: &[&str], today: i64) -> Option<Range> {
    let single = |d| Some((d, d));
    match *words {
        [w] if iso(w).is_some() && w.len() == 10 => iso(w).and_then(single),
        // A number alone is a year only from 1900 to this one: "the code 4821" names no days.
        [w] if year_of(w).is_some_and(|y| (1900..=year_month(today).0).contains(&y)) => {
            let y = year_of(w)?;
            Some((civil(y, 1, 1), civil(y + 1, 1, 1) - 1))
        }
        [w] => month_of(w).map(|m| month(year_for(m, today), m)),
        [a, b] if month_of(a).is_some() && year_of(b).is_some() => {
            Some(month(year_of(b)?, month_of(a)?))
        }
        [a, b] | [b, a] if day_of_month(a).is_some() && month_of(b).is_some() => {
            let m = month_of(b)?;
            day(year_for(m, today), m, day_of_month(a)?).and_then(single)
        }
        [a, b, c] | [b, a, c] if day_of_month(a).is_some() && month_of(b).is_some() => {
            day(year_of(c)?, month_of(b)?, day_of_month(a)?).and_then(single)
        }
        _ => None,
    }
}

/// The first single day written in `text`, a fact's words: `2026-10-03`, `3 October 2026` or
/// `October 3`, a year left out being `year_for`'s from `today`.
pub(crate) fn first_day(text: &str, today: i64) -> Option<i64> {
    let words = words(text);
    let words: Vec<&str> = words.iter().map(String::as_str).collect();
    (0..words.len()).find_map(|i| {
        [3, 2, 1].into_iter().find_map(|n| {
            let (from, to) = term(words.get(i..i + n)?, today)?;
            (from == to && (n > 1 || words[i].contains('-'))).then_some(from)
        })
    })
}

/// A whole date phrase: `today` or `tonight`, `yesterday` or `last night`, `last` or `this past`
/// a weekday or the weekend (the latest before today), `this week` (from Monday), `last week` (the
/// Monday to Sunday before this week's Monday), `this` or `last` month or year, `in` a month or year, `on` a day,
/// `between` two days or months, `before` or `after` a day, a month or an event that a fact dates
/// (`event` gives that fact's day).
fn phrase<E>(
    words: &[&str],
    today: i64,
    event: &mut impl FnMut(&str) -> Result<Option<i64>, E>,
) -> Result<Option<Range>, E> {
    let (y, m) = year_month(today);
    let last_month = if m == 1 { (y - 1, 12) } else { (y, m - 1) };
    // 1970-01-01 was a Thursday.
    let monday = today - (today + 3).rem_euclid(7);
    Ok(match words {
        ["today" | "tonight"] => Some((today, today)),
        ["yesterday"] | ["last", "night"] => Some((today - 1, today - 1)),
        ["last" | "past", w] | ["this", "past", w] if weekday(w).is_some() => {
            weekday(w).map(|d| (latest(d, today), latest(d, today)))
        }
        ["last" | "past", "weekend"] | ["this", "past", "weekend"] => {
            let saturday = latest(5, today);
            Some((saturday, saturday + 1))
        }
        ["this", "year"] => Some((civil(y, 1, 1), today)),
        ["last", "year"] => Some((civil(y - 1, 1, 1), civil(y, 1, 1) - 1)),
        ["this", "week"] => Some((monday, today)),
        ["last", "week"] => Some((monday - 7, monday - 1)),
        ["this", "month"] => Some((civil(y, m, 1), today)),
        ["last", "month"] => Some(month(last_month.0, last_month.1)),
        ["in", rest @ ..] | ["on", rest @ ..] => term(rest, today),
        ["between", rest @ ..] => {
            let and = rest.iter().position(|w| *w == "and");
            and.and_then(|i| {
                let (a, b) = (term(&rest[..i], today)?, term(&rest[i + 1..], today)?);
                Some((a.0.min(b.0), a.1.max(b.1)))
            })
        }
        [edge @ ("before" | "after"), rest @ ..] if !rest.is_empty() => {
            let at = match term(rest, today) {
                Some(range) => Some(range),
                None => event(&rest.join(" "))?.map(|d| (d, d)),
            };
            at.map(|(from, to)| match *edge {
                "before" => (i64::MIN, from - 1),
                _ => (to + 1, i64::MAX),
            })
        }
        _ => None,
    })
}

/// `text` lowercased, as words without the punctuation around them.
fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// The range a date phrase at the end of `query`, or else at its start, names, and the query's
/// other words; the whole query when no phrase is there. The longest phrase wins.
pub(crate) fn split<E>(
    query: &str,
    today: i64,
    mut event: impl FnMut(&str) -> Result<Option<i64>, E>,
) -> Result<(Option<Range>, String), E> {
    let words = words(query);
    let words: Vec<&str> = words.iter().map(String::as_str).collect();
    for i in 0..words.len() {
        if let Some(range) = phrase(&words[i..], today, &mut event)? {
            return Ok((Some(range), words[..i].join(" ")));
        }
    }
    for i in (1..words.len()).rev() {
        if let Some(range) = phrase(&words[..i], today, &mut event)? {
            return Ok((Some(range), words[i..].join(" ")));
        }
    }
    Ok((None, query.to_owned()))
}

/// The days the first relative date phrase anywhere in `text` names ("this past Saturday",
/// "yesterday", "last week"), for a question to the web: a date written out is the search's to read.
// ponytail: today is UTC's, so from 7 PM CDT "yesterday" is the user's today; take the user's
// zone from the OS when a bench row shows it
pub(crate) fn relative(text: &str, today: i64) -> Option<Range> {
    let words = words(text);
    let words: Vec<&str> = words.iter().map(String::as_str).collect();
    (0..words.len()).find_map(|i| {
        [3, 2, 1].into_iter().find_map(|n| {
            let w = words.get(i..i + n)?;
            let written = matches!(w[0], "in" | "on" | "between" | "before" | "after");
            let range = phrase(w, today, &mut |_| Ok::<_, ()>(None));
            range.ok().flatten().filter(|_| !written)
        })
    })
}

/// `day` as a search and an answer write it: "October 24 2026".
pub(crate) fn written(day: i64) -> String {
    let (y, m) = year_month(day);
    let name = MONTHS[m as usize - 1];
    let d = day - civil(y, m, 1) + 1;
    format!("{}{} {d} {y}", name[..1].to_uppercase(), &name[1..])
}

#[cfg(test)]
mod tests {
    use super::{Range, iso, relative, split, written};

    fn range(from: &str, to: &str) -> Range {
        (iso(from).unwrap(), iso(to).unwrap())
    }

    #[test]
    fn each_date_phrase_names_its_days_on_a_fixed_today() {
        // A Saturday.
        let today = iso("2026-10-17").unwrap();
        // The Lisbon trip, as a fact dates it.
        let lisbon = iso("2026-10-03").unwrap();
        let event = |words: &str| Ok::<_, ()>((words == "the lisbon trip").then_some(lisbon));
        let before = |day| (i64::MIN, iso(day).unwrap());
        let cases = [
            (
                "what did I do today",
                range("2026-10-17", "2026-10-17"),
                "what did i do",
            ),
            (
                "what did I do yesterday?",
                range("2026-10-16", "2026-10-16"),
                "what did i do",
            ),
            (
                "the boiler this week",
                range("2026-10-12", "2026-10-17"),
                "the boiler",
            ),
            (
                "the boiler last week",
                range("2026-10-05", "2026-10-11"),
                "the boiler",
            ),
            ("rent this month", range("2026-10-01", "2026-10-17"), "rent"),
            ("rent last month", range("2026-09-01", "2026-09-30"), "rent"),
            (
                "rent in September",
                range("2026-09-01", "2026-09-30"),
                "rent",
            ),
            // A month after this one is last year's.
            (
                "rent in November",
                range("2025-11-01", "2025-11-30"),
                "rent",
            ),
            (
                "rent in Feb 2024",
                range("2024-02-01", "2024-02-29"),
                "rent",
            ),
            ("rent in 2025", range("2025-01-01", "2025-12-31"), "rent"),
            (
                "dentist on 2026-10-03",
                range("2026-10-03", "2026-10-03"),
                "dentist",
            ),
            (
                "dentist on 3 October",
                range("2026-10-03", "2026-10-03"),
                "dentist",
            ),
            (
                "dentist on October 3rd, 2025",
                range("2025-10-03", "2025-10-03"),
                "dentist",
            ),
            (
                "rent between 2026-09-03 and 2026-10-01",
                range("2026-09-03", "2026-10-01"),
                "rent",
            ),
            (
                "rent between July and August",
                range("2026-07-01", "2026-08-31"),
                "rent",
            ),
            ("rent before 2026-10-03", before("2026-10-02"), "rent"),
            (
                "rent after September",
                (iso("2026-10-01").unwrap(), i64::MAX),
                "rent",
            ),
            (
                "what did I pack before the Lisbon trip",
                before("2026-10-02"),
                "what did i pack",
            ),
            (
                "In June, what did Tomasz say",
                range("2026-06-01", "2026-06-30"),
                "what did tomasz say",
            ),
        ];
        for (query, want, rest) in cases {
            let (got, words) = split(query, today, event).unwrap();
            assert_eq!(got, Some(want), "{query}");
            assert_eq!(words, rest, "{query}");
        }
        // No phrase, a phrase in the middle, a day that does not exist, an event no fact dates, a
        // number that is no year up to this one:
        // the query as it is, with no range.
        for query in [
            "what is my address",
            "what did I do last week at the gym",
            "rent on 2026-02-30",
            "rent before the Porto trip",
            "the may pole",
            "the door code 4821",
            "rent in 2027",
        ] {
            let got = split(query, today, event).unwrap();
            assert_eq!(got, (None, query.to_owned()), "{query}");
        }
    }

    #[test]
    fn a_relative_phrase_anywhere_in_a_question_names_its_days() {
        // Monday 26 October 2026.
        let today = iso("2026-10-26").unwrap();
        let one = |d| Some(range(d, d));
        let q = "Search the internet to fin teh UT Austin Football score from this past saturday";
        assert_eq!(relative(q, today), one("2026-10-24"));
        assert_eq!(relative("last Saturday's game?", today), one("2026-10-24"));
        assert_eq!(
            relative("what happened last monday", today),
            one("2026-10-19")
        );
        assert_eq!(
            relative("the news from last night", today),
            one("2026-10-25")
        );
        assert_eq!(relative("who plays tonight", today), one("2026-10-26"));
        let weekend = Some(range("2026-10-24", "2026-10-25"));
        assert_eq!(relative("scores this past weekend", today), weekend);
        let week = Some(range("2026-10-19", "2026-10-25"));
        assert_eq!(
            relative("what did the Fed do last week at its meeting", today),
            week
        );
        let year = Some(range("2025-01-01", "2025-12-31"));
        assert_eq!(relative("the best films of last year", today), year);
        // On a Saturday, last Saturday is a week back.
        let saturday = iso("2026-10-24").unwrap();
        assert_eq!(relative("last saturday", saturday), one("2026-10-17"));
        // A date written out, or none: no span.
        for q in [
            "how do I make chicken stew?",
            "the game on 24 October",
            "in 2025",
        ] {
            assert_eq!(relative(q, today), None, "{q}");
        }
        assert_eq!(written(today), "October 26 2026");
    }

    fn days(query: &str, today: &str) -> Option<Range> {
        split(query, iso(today).unwrap(), |_| Ok::<_, ()>(None))
            .unwrap()
            .0
    }

    #[test]
    fn last_week_is_the_monday_to_sunday_before_this_weeks_monday() {
        // From this week's Monday to its Sunday, last week is the same seven days.
        for today in ["2026-10-12", "2026-10-17", "2026-10-18"] {
            let want = range("2026-10-05", "2026-10-11");
            assert_eq!(days("last week", today), Some(want), "{today}");
        }
        // The next Monday moves it on, and this week starts that day.
        let monday = "2026-10-19";
        let want = range("2026-10-12", "2026-10-18");
        assert_eq!(days("last week", monday), Some(want));
        assert_eq!(days("this week", monday), Some(range(monday, monday)));
    }

    #[test]
    fn year_month_inverts_civil_before_and_after_1970() {
        // Every day from 1599-12-01 to 2401-01-31 falls in the month it names.
        for day in super::civil(1599, 12, 1)..=super::civil(2401, 1, 31) {
            let (y, m) = super::year_month(day);
            assert!(
                super::civil(y, m, 1) <= day && day < super::civil(y, m + 1, 1),
                "{day}: {y}-{m}"
            );
        }
        // A month named from a day before 1970: December into January, and February in a year
        // that is a leap year (1904, 1600) and one that is not (1900). `this month` ends today,
        // the month's last day in each case.
        let cases = [
            ("this month", "1900-12-31", ("1900-12-01", "1900-12-31")),
            ("last month", "1900-01-01", ("1899-12-01", "1899-12-31")),
            ("this month", "1969-12-31", ("1969-12-01", "1969-12-31")),
            ("last month", "1970-01-01", ("1969-12-01", "1969-12-31")),
            ("last month", "1900-03-01", ("1900-02-01", "1900-02-28")),
            ("last month", "1904-03-01", ("1904-02-01", "1904-02-29")),
            ("last month", "1600-03-31", ("1600-02-01", "1600-02-29")),
            ("this month", "1904-02-29", ("1904-02-01", "1904-02-29")),
        ];
        for (query, today, (from, to)) in cases {
            let want = range(from, to);
            assert_eq!(days(query, today), Some(want), "{query} on {today}");
        }
    }
}
