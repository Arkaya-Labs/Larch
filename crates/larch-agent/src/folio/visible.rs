//! What a reader can see now: the ranges its session holds, so a repeated read gets only what is
//! new. (From Research v2's record.rs; the rest of that file is research's.)

use super::Range;

/// The ranges a reader's session holds now. It is cleared with the session.
#[derive(Clone, Debug, Default)]
pub struct Visible {
    ranges: Vec<Range>,
}

impl Visible {
    /// `range` is now in the session.
    pub fn seen(&mut self, range: Range) {
        if range.start >= range.end {
            return;
        }
        // Stored ranges never overlap or touch, so one pass merges every one `range` reaches.
        let mut r = range;
        self.ranges.retain(|v| {
            let apart = v.doc != r.doc || v.end < r.start || r.end < v.start;
            if !apart {
                (r.start, r.end) = (r.start.min(v.start), r.end.max(v.end));
            }
            apart
        });
        self.ranges.push(r);
        self.ranges.sort();
    }

    /// The parts of `range` the session already holds, merged; empty when none.
    pub fn already(&self, range: Range) -> Vec<Range> {
        (self.ranges.iter())
            .filter(|v| v.doc == range.doc)
            .map(|v| Range {
                doc: range.doc,
                start: v.start.max(range.start),
                end: v.end.min(range.end),
            })
            .filter(|r| r.start < r.end)
            .collect()
    }

    /// Whether the session holds all of `range`; an empty one is held when it sits inside or at
    /// the edge of a held range.
    pub fn holds(&self, range: Range) -> bool {
        match range.start == range.end {
            true => (self.ranges.iter())
                .any(|v| v.doc == range.doc && v.start <= range.start && range.start <= v.end),
            false => self.already(range) == [range],
        }
    }

    /// The session was reset: it holds nothing.
    pub fn clear(&mut self) {
        self.ranges.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(doc: usize, start: usize, end: usize) -> Range {
        Range { doc, start, end }
    }

    #[test]
    fn visible_merges_ranges_and_gives_the_covered_parts() {
        let mut v = Visible::default();
        v.seen(r(0, 10, 20));
        v.seen(r(0, 30, 40));
        v.seen(r(1, 0, 100));
        assert_eq!(v.already(r(0, 0, 50)), [r(0, 10, 20), r(0, 30, 40)]);
        v.seen(r(0, 15, 30));
        assert_eq!(v.already(r(0, 0, 50)), [r(0, 10, 40)]);
        assert_eq!(v.already(r(0, 40, 50)), []);
        assert_eq!(v.already(r(1, 5, 6)), [r(1, 5, 6)]);
    }
}
