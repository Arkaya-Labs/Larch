//! Code mode on a real model: small repositories in Python, Rust, JavaScript and HTML with a
//! request each (a bug to fix, a function to write, a feature, a new project, a follow-up), each
//! run as `larch-agent code` runs it. A task is solved when its hidden check, which the agent
//! never sees, passes after the run; one JSON line per run says so, then the totals.
//!
//!   LARCH_MODEL=<gguf> cargo test --release -p larch-agent --test code_eval -- --ignored --nocapture
//!
//! LARCH_CPU=1 runs it on the CPU backend instead of GPU 0; LARCH_TASKS=py-add,rs-fib runs
//! only those; LARCH_RUNS=n runs each n times; LARCH_KEEP=<dir> copies each run's folder to
//! <dir>/<task>-<run> before it is removed. On Windows js-clamp's `node --test` hangs in the sandbox (`sandbox::windows::run`).

mod common;

use std::{
    fs, io,
    path::Path,
    process::{Command, Output},
    time::Instant,
};

use larch_agent::{
    AgentEvent,
    code::{
        allow::Decision,
        r#loop::{BUDGET, Outcome, decide},
        task::{Budget, CodeTask, Overrides},
    },
};
use serde_json::json;

/// Python's unittest, which every Python has; Windows' Python has no python3 outside the
/// Store's aliases, which the sandbox cannot start.
const UNITTEST: &str = if cfg!(windows) {
    "python -m unittest"
} else {
    "python3 -m unittest"
};

/// web-cafe's hidden check.
const PY_SITE: &str = if cfg!(windows) {
    "python hidden_site.py"
} else {
    "python3 hidden_site.py"
};

/// A repository's files: each path and its text.
type Files = &'static [(&'static str, &'static str)];

/// A task: the repository the agent starts from, what it is asked, and the hidden check that
/// says whether it solved it.
struct Task {
    /// The task's name, which LARCH_TASKS filters on.
    name: &'static str,
    /// The request, then each follow-up, each run on the change the one before kept.
    requests: &'static [&'static str],
    /// The test command; None: the manifest's.
    test_cmd: Option<&'static str>,
    /// Text the last run's diff must hold besides the hidden check passing.
    want: &'static str,
    /// The repository at the start.
    files: Files,
    /// Files copied in after the run, which the agent never sees.
    hidden: Files,
    /// The command whose exit 0 means solved, run once `hidden` is copied in.
    check: &'static str,
}

const TASKS: &[Task] = &[
    Task {
        name: "py-add",
        requests: &["Fix the failing test"],
        test_cmd: Some(UNITTEST),
        want: "",
        files: &[
            (
                "pyproject.toml",
                "[project]\nname = 'calc'\nversion = '0.1.0'\n",
            ),
            (
                "calc.py",
                "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n",
            ),
            (
                "test_calc.py",
                "import unittest\nfrom calc import add, mul\n\n\nclass T(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\n    def test_mul(self):\n        self.assertEqual(mul(2, 3), 6)\n",
            ),
        ],
        hidden: &[],
        check: UNITTEST,
    },
    Task {
        name: "py-slug",
        requests: &["implement slugify in text.py so the tests pass"],
        test_cmd: Some(UNITTEST),
        want: "",
        files: &[
            (
                "pyproject.toml",
                "[project]\nname = 'text'\nversion = '0.1.0'\n",
            ),
            (
                "text.py",
                "def slugify(title):\n    \"\"\"Lower case, words joined by '-', anything but letters and digits dropped.\"\"\"\n    raise NotImplementedError\n",
            ),
            (
                "test_text.py",
                "import unittest\nfrom text import slugify\n\n\nclass T(unittest.TestCase):\n    def test_simple(self):\n        self.assertEqual(slugify('Hello World'), 'hello-world')\n\n    def test_punct(self):\n        self.assertEqual(slugify('  Rust, C++ & Go!  '), 'rust-c-go')\n",
            ),
        ],
        hidden: &[],
        check: UNITTEST,
    },
    Task {
        name: "py-discount",
        requests: &["the order total is wrong when a discount code applies; fix it"],
        test_cmd: Some(UNITTEST),
        want: "",
        files: &[
            (
                "pyproject.toml",
                "[project]\nname = 'shop'\nversion = '0.1.0'\n",
            ),
            ("shop/__init__.py", ""),
            (
                "shop/pricing.py",
                "DISCOUNTS = {'SAVE10': 10, 'HALF': 50}\n\n\ndef discounted(amount, code):\n    pct = DISCOUNTS.get(code, 0)\n    return amount - amount * pct\n",
            ),
            (
                "shop/order.py",
                "from shop.pricing import discounted\n\n\nclass Order:\n    def __init__(self):\n        self.lines = []\n\n    def add(self, price, qty=1):\n        self.lines.append((price, qty))\n\n    def total(self, code=None):\n        subtotal = sum(p * q for p, q in self.lines)\n        return discounted(subtotal, code)\n",
            ),
            ("tests/__init__.py", ""),
            (
                "tests/test_order.py",
                "import unittest\nfrom shop.order import Order\n\n\nclass T(unittest.TestCase):\n    def test_plain(self):\n        o = Order()\n        o.add(10, 2)\n        self.assertEqual(o.total(), 20)\n\n    def test_code(self):\n        o = Order()\n        o.add(100)\n        self.assertAlmostEqual(o.total('SAVE10'), 90)\n\n    def test_half(self):\n        o = Order()\n        o.add(30, 2)\n        self.assertAlmostEqual(o.total('HALF'), 30)\n",
            ),
        ],
        hidden: &[],
        check: UNITTEST,
    },
    Task {
        name: "py-feature",
        requests: &["add a median function to stats.py, with a test for it in test_stats.py"],
        test_cmd: Some(UNITTEST),
        want: "def median",
        files: &[
            (
                "pyproject.toml",
                "[project]\nname = 'stats'\nversion = '0.1.0'\n",
            ),
            ("stats.py", "def mean(xs):\n    return sum(xs) / len(xs)\n"),
            (
                "test_stats.py",
                "import unittest\nfrom stats import mean\n\n\nclass T(unittest.TestCase):\n    def test_mean(self):\n        self.assertEqual(mean([1, 2, 3]), 2)\n",
            ),
        ],
        hidden: &[(
            "test_hidden_median.py",
            r#"import unittest
from stats import median


class Hidden(unittest.TestCase):
    def test_odd(self):
        self.assertEqual(median([3, 1, 2]), 2)

    def test_even(self):
        self.assertEqual(median([4, 1, 3, 2]), 2.5)
"#,
        )],
        check: UNITTEST,
    },
    Task {
        name: "rs-fib",
        requests: &["fix fib so the tests pass"],
        test_cmd: None,
        want: "",
        files: &[
            (
                "Cargo.toml",
                "[package]\nname = \"fib\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (".gitignore", "/target\n"),
            (
                "src/lib.rs",
                "/// The n-th Fibonacci number, fib(0) = 0 and fib(1) = 1.\npub fn fib(n: u32) -> u64 {\n    let (mut a, mut b) = (0u64, 1u64);\n    for _ in 1..n {\n        (a, b) = (b, a + b);\n    }\n    a\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n\n    #[test]\n    fn small() {\n        assert_eq!(fib(0), 0);\n        assert_eq!(fib(1), 1);\n        assert_eq!(fib(2), 1);\n        assert_eq!(fib(10), 55);\n    }\n}\n",
            ),
        ],
        hidden: &[],
        check: "cargo test",
    },
    Task {
        name: "rs-parse",
        requests: &["implement parse_kv in src/lib.rs so the tests pass"],
        test_cmd: None,
        want: "",
        files: &[
            (
                "Cargo.toml",
                "[package]\nname = \"kv\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (".gitignore", "/target\n"),
            (
                "src/lib.rs",
                "/// Parses `key=value` pairs separated by `;`, trimming spaces around keys and values; an\n/// entry without `=` is skipped.\npub fn parse_kv(s: &str) -> Vec<(String, String)> {\n    todo!()\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n\n    #[test]\n    fn pairs() {\n        let got = parse_kv(\"a=1; b = two ;junk; c=\");\n        let want = vec![\n            (\"a\".to_string(), \"1\".to_string()),\n            (\"b\".to_string(), \"two\".to_string()),\n            (\"c\".to_string(), \"\".to_string()),\n        ];\n        assert_eq!(got, want);\n    }\n\n    #[test]\n    fn empty() {\n        assert!(parse_kv(\"\").is_empty());\n    }\n}\n",
            ),
        ],
        hidden: &[],
        check: "cargo test",
    },
    Task {
        name: "js-clamp",
        requests: &["Fix the failing test"],
        test_cmd: None,
        want: "",
        files: &[
            (
                "package.json",
                "{\n  \"name\": \"clamp\",\n  \"version\": \"1.0.0\",\n  \"scripts\": { \"test\": \"node --test\" }\n}\n",
            ),
            (
                "clamp.js",
                "function clamp(x, lo, hi) {\n  if (x < lo) return hi;\n  if (x > hi) return lo;\n  return x;\n}\n\nmodule.exports = { clamp };\n",
            ),
            (
                "clamp.test.js",
                "const test = require('node:test');\nconst assert = require('node:assert');\nconst { clamp } = require('./clamp');\n\ntest('below', () => assert.strictEqual(clamp(-5, 0, 10), 0));\ntest('above', () => assert.strictEqual(clamp(50, 0, 10), 10));\ntest('inside', () => assert.strictEqual(clamp(5, 0, 10), 5));\n",
            ),
        ],
        hidden: &[],
        check: "node --test",
    },
    Task {
        name: "py-caller",
        requests: &["add route_length(points) to route.py that sums the distances"],
        test_cmd: Some(UNITTEST),
        want: "def route_length",
        files: &[
            (
                "pyproject.toml",
                "[project]\nname = 'geo'\nversion = '0.1.0'\n",
            ),
            (
                "geo.py",
                r#"import math
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float


def distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)
"#,
            ),
            (
                "route.py",
                "\"\"\"Routes: lists of geo.Point, visited in order.\"\"\"\n",
            ),
            (
                "test_geo.py",
                r#"import unittest
from geo import Point, distance


class T(unittest.TestCase):
    def test_distance(self):
        self.assertEqual(distance(Point(0, 0), Point(3, 4)), 5)
"#,
            ),
        ],
        hidden: &[(
            "test_hidden_route.py",
            r#"import unittest
from geo import Point
from route import route_length


class Hidden(unittest.TestCase):
    def test_three_points(self):
        self.assertAlmostEqual(route_length([Point(0, 0), Point(3, 4), Point(3, 10)]), 11)
"#,
        )],
        check: UNITTEST,
    },
    Task {
        name: "py-testguided",
        requests: &["make the inventory tests pass"],
        test_cmd: Some(UNITTEST),
        want: "",
        files: &[
            (
                "pyproject.toml",
                "[project]\nname = 'inventory'\nversion = '0.1.0'\n",
            ),
            (
                "inventory.py",
                r#"class Inventory:
    def __init__(self):
        self.stock = {}

    def add(self, item, qty):
        self.stock[item] = qty

    def remove(self, item, qty):
        if self.stock.get(item, 0) < qty:
            raise ValueError(f"not enough {item}")
        self.stock[item] -= qty

    def count(self, item):
        return self.stock.get(item, 0)
"#,
            ),
            (
                "test_inventory.py",
                r#"import unittest
from inventory import Inventory


class T(unittest.TestCase):
    def test_adding_twice_adds_up(self):
        inv = Inventory()
        inv.add("apple", 3)
        inv.add("apple", 2)
        self.assertEqual(inv.count("apple"), 5)

    def test_remove_more_than_there_is(self):
        inv = Inventory()
        inv.add("apple", 1)
        with self.assertRaises(ValueError):
            inv.remove("apple", 2)
"#,
            ),
        ],
        hidden: &[(
            "test_hidden_inventory.py",
            r#"import unittest
from inventory import Inventory


class Hidden(unittest.TestCase):
    def test_three_adds(self):
        inv = Inventory()
        for _ in range(3):
            inv.add("pear", 1)
        self.assertEqual(inv.count("pear"), 3)

    def test_add_after_remove(self):
        inv = Inventory()
        inv.add("fig", 5)
        inv.remove("fig", 2)
        inv.add("fig", 4)
        self.assertEqual(inv.count("fig"), 7)
"#,
        )],
        check: UNITTEST,
    },
    Task {
        name: "js-feature3",
        requests: &[
            "add a coupon to the cart: cart.applyCoupon(code) with 'SAVE10' = 10% off, and show it in index.js's summary line",
        ],
        test_cmd: None,
        want: "applyCoupon",
        files: &[
            (
                "package.json",
                "{\n  \"name\": \"cart\",\n  \"version\": \"1.0.0\",\n  \"scripts\": { \"test\": \"node --test\" }\n}\n",
            ),
            (
                "cart.js",
                r#"class Cart {
  constructor() {
    this.items = [];
  }

  add(name, price, qty = 1) {
    this.items.push({ name, price, qty });
  }

  subtotal() {
    return this.items.reduce((sum, i) => sum + i.price * i.qty, 0);
  }

  total() {
    return this.subtotal();
  }
}

module.exports = { Cart };
"#,
            ),
            (
                "format.js",
                "function money(x) {\n  return '$' + x.toFixed(2);\n}\n\nmodule.exports = { money };\n",
            ),
            (
                "index.js",
                r#"const { Cart } = require('./cart');
const { money } = require('./format');

function summary(cart) {
  return `${cart.items.length} items, total ${money(cart.total())}`;
}

module.exports = { summary };

if (require.main === module) {
  const cart = new Cart();
  cart.add('tea', 5, 2);
  console.log(summary(cart));
}
"#,
            ),
            (
                "cart.test.js",
                r#"const test = require('node:test');
const assert = require('node:assert');
const { Cart } = require('./cart');

test('total', () => {
  const cart = new Cart();
  cart.add('tea', 5, 2);
  assert.strictEqual(cart.total(), 10);
});
"#,
            ),
        ],
        hidden: &[(
            "hidden_coupon.test.js",
            r#"const test = require('node:test');
const assert = require('node:assert');
const { Cart } = require('./cart');
const { money } = require('./format');
const { summary } = require('./index');

function cart() {
  const c = new Cart();
  c.add('tea', 5, 2);
  c.add('cake', 10);
  return c;
}

test('no coupon, full price', () => assert.strictEqual(cart().total(), 20));
test('SAVE10 takes 10% off', () => {
  const c = cart();
  c.applyCoupon('SAVE10');
  assert.ok(Math.abs(c.total() - 18) < 1e-9, `total ${c.total()}`);
});
test('the summary shows the coupon and the total after it', () => {
  const c = cart();
  c.applyCoupon('SAVE10');
  const line = summary(c);
  assert.match(line, /SAVE10/);
  assert.ok(line.includes(money(18)), line);
});
"#,
        )],
        check: "node --test hidden_coupon.test.js",
    },
    Task {
        name: "rs-followup",
        requests: &[
            "add a `Stack<T>` with push/pop in src/lib.rs",
            "also add peek and is_empty, with tests",
        ],
        test_cmd: None,
        want: "fn peek",
        files: &[
            (
                "Cargo.toml",
                "[package]\nname = \"stack\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (".gitignore", "/target\n"),
            ("src/lib.rs", "//! Collections.\n"),
        ],
        hidden: &[(
            "tests/hidden.rs",
            r#"use stack::Stack;

#[test]
fn push_pop_peek_is_empty() {
    let mut s = Stack::new();
    assert!(s.is_empty());
    s.push(1);
    s.push(2);
    assert_eq!(s.peek(), Some(&2));
    assert_eq!(s.pop(), Some(2));
    assert_eq!(s.pop(), Some(1));
    assert!(s.is_empty());
    assert_eq!(s.pop(), None);
}
"#,
        )],
        check: "cargo test",
    },
    Task {
        name: "js-newproject",
        requests: &[
            "make a small node project with a `slug(text)` function in slug.js and a test for it, run with npm test",
        ],
        test_cmd: None,
        want: "slug",
        files: &[],
        hidden: &[(
            "hidden_slug.test.mjs",
            r#"import test from 'node:test';
import assert from 'node:assert';

const m = await import('./slug.js');
const slug = m.slug ?? m.default?.slug ?? m.default;

test('words', () => assert.strictEqual(slug('Hello World'), 'hello-world'));
test('spaces', () => assert.strictEqual(slug('  Many   Spaces  '), 'many-spaces'));
"#,
        )],
        check: "npm test && node --test hidden_slug.test.mjs",
    },
    Task {
        name: "web-cafe",
        requests: &[
            "make a one-page website for my cafe, Bean There, open 7-3 daily at 12 Main St",
        ],
        test_cmd: None,
        want: "Bean There",
        files: &[],
        // Code mode's broken_pages, for pages and links: each page whole, each local link there.
        hidden: &[(
            "hidden_site.py",
            r#"import pathlib
import re
import sys

pages = [p for p in pathlib.Path(".").rglob("*.html") if ".git" not in p.parts]
wrong = [] if pages else ["no page"]
text = ""
for page in pages:
    t = page.read_text(errors="replace")
    text += t
    if "<body" not in t.lower():
        wrong.append(f"{page} has no <body>")
    for target in re.findall(r"""(?:href|src)\s*=\s*["']([^"'#?]+)""", t):
        local = ":" not in target and not target.startswith("//")
        if local and not (page.parent / target).exists():
            wrong.append(f"{page} links {target}, which does not exist")
for want in ("Bean There", "12 Main St", "7"):
    if want not in text:
        wrong.append(f"no {want!r} on the page")
if re.search(r"\(\d{3}\)", text):
    wrong.append("an invented phone number")
sys.exit("\n".join(wrong) or None)
"#,
        )],
        check: PY_SITE,
    },
];

/// Each task's known-good end state: files written over its start, on which its hidden check
/// must pass.
const SOLUTIONS: &[(&str, Files)] = &[
    (
        "py-add",
        &[(
            "calc.py",
            "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n",
        )],
    ),
    (
        "py-slug",
        &[(
            "text.py",
            "import re\n\n\ndef slugify(title):\n    return '-'.join(re.findall(r'[a-z0-9]+', title.lower()))\n",
        )],
    ),
    (
        "py-discount",
        &[(
            "shop/pricing.py",
            "DISCOUNTS = {'SAVE10': 10, 'HALF': 50}\n\n\ndef discounted(amount, code):\n    pct = DISCOUNTS.get(code, 0)\n    return amount - amount * pct / 100\n",
        )],
    ),
    (
        "py-feature",
        &[(
            "stats.py",
            r#"def mean(xs):
    return sum(xs) / len(xs)


def median(xs):
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
"#,
        )],
    ),
    (
        "rs-fib",
        &[(
            "src/lib.rs",
            "pub fn fib(n: u32) -> u64 {\n    let (mut a, mut b) = (0u64, 1u64);\n    for _ in 0..n {\n        (a, b) = (b, a + b);\n    }\n    a\n}\n",
        )],
    ),
    (
        "rs-parse",
        &[(
            "src/lib.rs",
            r#"pub fn parse_kv(s: &str) -> Vec<(String, String)> {
    s.split(';')
        .filter_map(|e| e.split_once('='))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect()
}
"#,
        )],
    ),
    (
        "js-clamp",
        &[(
            "clamp.js",
            "function clamp(x, lo, hi) {\n  return Math.min(Math.max(x, lo), hi);\n}\n\nmodule.exports = { clamp };\n",
        )],
    ),
    (
        "py-caller",
        &[(
            "route.py",
            "from geo import distance\n\n\ndef route_length(points):\n    return sum(distance(a, b) for a, b in zip(points, points[1:]))\n",
        )],
    ),
    (
        "py-testguided",
        &[(
            "inventory.py",
            r#"class Inventory:
    def __init__(self):
        self.stock = {}

    def add(self, item, qty):
        self.stock[item] = self.stock.get(item, 0) + qty

    def remove(self, item, qty):
        if self.stock.get(item, 0) < qty:
            raise ValueError(f"not enough {item}")
        self.stock[item] -= qty

    def count(self, item):
        return self.stock.get(item, 0)
"#,
        )],
    ),
    (
        "js-feature3",
        &[
            (
                "cart.js",
                r#"const COUPONS = { SAVE10: 10 };

class Cart {
  constructor() {
    this.items = [];
    this.coupon = null;
  }

  add(name, price, qty = 1) {
    this.items.push({ name, price, qty });
  }

  applyCoupon(code) {
    if (!(code in COUPONS)) throw new Error(`unknown coupon ${code}`);
    this.coupon = code;
  }

  subtotal() {
    return this.items.reduce((sum, i) => sum + i.price * i.qty, 0);
  }

  total() {
    const pct = this.coupon ? COUPONS[this.coupon] : 0;
    return this.subtotal() * (1 - pct / 100);
  }
}

module.exports = { Cart };
"#,
            ),
            (
                "index.js",
                r#"const { money } = require('./format');

function summary(cart) {
  const coupon = cart.coupon ? ` with ${cart.coupon}` : '';
  return `${cart.items.length} items, total ${money(cart.total())}${coupon}`;
}

module.exports = { summary };
"#,
            ),
        ],
    ),
    (
        "rs-followup",
        &[(
            "src/lib.rs",
            r#"/// A last-in, first-out stack.
#[derive(Default)]
pub struct Stack<T>(Vec<T>);

impl<T> Stack<T> {
    pub fn new() -> Self {
        Stack(Vec::new())
    }
    pub fn push(&mut self, x: T) {
        self.0.push(x)
    }
    pub fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }
    pub fn peek(&self) -> Option<&T> {
        self.0.last()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
"#,
        )],
    ),
    (
        "js-newproject",
        &[
            (
                "package.json",
                "{\n  \"name\": \"slug\",\n  \"version\": \"1.0.0\",\n  \"scripts\": { \"test\": \"node --test\" }\n}\n",
            ),
            (
                "slug.js",
                "function slug(text) {\n  return text.toLowerCase().trim().split(/[^a-z0-9]+/).filter(Boolean).join('-');\n}\n\nmodule.exports = { slug };\n",
            ),
            (
                "slug.test.js",
                "const test = require('node:test');\nconst assert = require('node:assert');\nconst { slug } = require('./slug');\n\ntest('slug', () => assert.strictEqual(slug('A b'), 'a-b'));\n",
            ),
        ],
    ),
    (
        "web-cafe",
        &[
            (
                "index.html",
                r#"<!doctype html>
<html>
<head><title>Bean There</title><link rel="stylesheet" href="style.css"></head>
<body><h1>Bean There</h1><p>Open 7am to 3pm daily, 12 Main St.</p></body>
</html>
"#,
            ),
            ("style.css", "body { font-family: sans-serif; }\n"),
        ],
    ),
];

/// Writes `files` into `dir`, over what is there.
fn write(dir: &Path, files: Files) {
    for (path, text) in files {
        let path = dir.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }
}

/// The task's repository in `dir`, its files committed; an empty one has an empty commit.
fn repo(dir: &Path, files: Files) {
    fs::create_dir_all(dir).unwrap();
    write(dir, files);
    common::commit(dir);
}

/// `task`'s hidden check in `dir`: its hidden files copied in, then its command, whose exit 0
/// means solved. The command builds in `dir`, never in the target folder of the cargo that runs
/// this, which holds that folder's lock.
// ponytail: runs unsandboxed and without a timeout on what the model wrote; run it through
// code mode's sandbox when the eval runs models or tasks we do not watch
fn hidden(task: &Task, dir: &Path) -> Output {
    write(dir, task.hidden);
    let (shell, flag) = if cfg!(windows) {
        ("cmd", "/C")
    } else {
        ("sh", "-c")
    };
    Command::new(shell)
        .args([flag, task.check])
        .current_dir(dir)
        .env_remove("CARGO_TARGET_DIR")
        .output()
        .unwrap()
}

#[test]
fn hidden_checks_fail_at_the_start_and_pass_on_a_solution() {
    let node = Command::new("node")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !node {
        println!("node is not on PATH: the js- tasks' hidden checks are not run");
    }
    // Windows CI has no Python; a missing program anywhere else fails its task.
    let python = !cfg!(windows)
        || (Command::new("python").arg("--version").output()).is_ok_and(|o| o.status.success());
    if !python {
        println!("python is not on PATH: the py- and web- tasks' hidden checks are not run");
    }
    for task in TASKS {
        let (_, solution) = (SOLUTIONS.iter().find(|s| s.0 == task.name))
            .unwrap_or_else(|| panic!("{} has no solution", task.name));
        assert!(
            solution.iter().any(|(_, text)| text.contains(task.want)),
            "{}: the solution lacks {:?}",
            task.name,
            task.want
        );
        if (!node && task.name.starts_with("js-")) || (!python && task.check.starts_with("python"))
        {
            continue;
        }
        let tmp = tempfile::tempdir().unwrap();
        let shown = |out: &Output| {
            String::from_utf8_lossy(&out.stdout).into_owned()
                + &String::from_utf8_lossy(&out.stderr)
        };
        let start = tmp.path().join("start");
        repo(&start, task.files);
        let out = hidden(task, &start);
        assert!(
            !out.status.success(),
            "{}: the hidden check passes on the start: {}",
            task.name,
            shown(&out)
        );
        let solved = tmp.path().join("solved");
        repo(&solved, task.files);
        write(&solved, solution);
        let out = hidden(task, &solved);
        assert!(
            out.status.success(),
            "{}: the hidden check fails on the solution: {}",
            task.name,
            shown(&out)
        );
    }
}

#[test]
#[ignore = "needs a model: LARCH_MODEL=<gguf>"]
fn code_tasks_on_a_real_model() {
    let path = std::env::var("LARCH_MODEL").expect("LARCH_MODEL=<gguf>");
    let device = match std::env::var("LARCH_CPU").is_ok_and(|v| v == "1") {
        true => larch_kernels::CPU,
        false => 0,
    };
    let (larch, code) = common::real(&path, device);
    let only = std::env::var("LARCH_TASKS").unwrap_or_default();
    let runs: u32 = std::env::var("LARCH_RUNS").map_or(1, |n| n.parse().expect("LARCH_RUNS=<n>"));
    let keep = std::env::var_os("LARCH_KEEP").map(std::path::PathBuf::from);
    let (mut solved, mut false_done, mut ran) = (0, 0, 0);
    for task in TASKS {
        if !only.is_empty() && !only.split(',').any(|t| t == task.name) {
            continue;
        }
        for run in 1..=runs {
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().join(task.name);
            repo(&dir, task.files);
            let started = Instant::now();
            let (mut calls, mut steps, mut tokens) = (0, 0, 0);
            let mut last: Option<Outcome> = None;
            let mut stopped = String::new();
            for request in task.requests {
                // A follow-up runs on the change the run before kept, as approving it does.
                if let Some(mut before) = last.take() {
                    decide(&mut before, &mut "approve\n".as_bytes(), &mut io::sink()).unwrap();
                }
                let overrides = Overrides {
                    test_cmd: task.test_cmd.map(String::from),
                    allowed_paths: None,
                    check_cmds: vec![],
                    // One attempt, so each phase's gate compares like with like; best-of-k is
                    // measured on its own.
                    budget: Budget { k: 1, ..BUDGET },
                };
                let code_task = CodeTask::new(request, &dir, overrides).unwrap();
                let got = code.run_with(
                    &larch,
                    &code_task,
                    &tmp.path().join("data"),
                    &mut |e| {
                        if let AgentEvent::ToolCall { name, args } = &e {
                            calls += 1;
                            println!("  -> {name} {args}");
                        }
                    },
                    &mut |_| Decision::No,
                );
                match got {
                    Ok(outcome) => {
                        steps += outcome.steps;
                        tokens += outcome.tokens;
                        last = Some(outcome);
                    }
                    Err(e) => {
                        stopped = format!("error: {e}");
                        break;
                    }
                }
            }
            let (mut ok, mut wrong_done) = (false, false);
            if let Some(mut outcome) = last {
                let passed = hidden(task, outcome.task.dir().unwrap()).status.success();
                ok = passed && outcome.packet.diff.contains(task.want);
                stopped = outcome.packet.stopped.take().unwrap_or_default();
                wrong_done = !passed && stopped.contains("said it is done");
                if let Some(keep) = &keep {
                    let to = keep.join(format!("{}-{run}", task.name));
                    fs::create_dir_all(keep).unwrap();
                    let from = outcome.task.dir().unwrap();
                    let copied = Command::new("cp").arg("-R").args([from, &to]).status();
                    assert!(copied.unwrap().success(), "cp -R {from:?} {to:?}");
                }
                outcome.task.discard().unwrap();
            }
            ran += 1;
            solved += usize::from(ok);
            false_done += usize::from(wrong_done);
            let line = json!({
                "task": task.name,
                "run": run,
                "solved": ok,
                "false_done": wrong_done,
                "steps": steps,
                "calls": calls,
                "secs": started.elapsed().as_secs_f64().round(),
                "tokens": tokens,
                "stopped": stopped,
            });
            println!("{line}");
        }
    }
    println!("{solved} of {ran} solved, {false_done} false done");
}
