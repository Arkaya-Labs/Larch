# Renders screenshot-like eval pages (and a few EXIF-rotated photo-like JPEGs) with exact ground truth.
# Extends ~/larch-image-paste-bench/gen_images.py.  All content is invented.
# Run: uv run --with pillow python -I gen_eval.py <outdir>
# Writes <outdir>/*.png|jpg and <outdir>/gen_items.jsonl.  Deterministic (seeded RNGs, no clock).
import datetime, json, os, random, re, sys
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

OUT = os.path.abspath(sys.argv[1])
os.makedirs(OUT, exist_ok=True)
FD = "/mnt/c/Windows/Fonts/"
FAM = {
    "Consolas": ("consola.ttf", "consolab.ttf"),
    "Segoe UI": ("segoeui.ttf", "segoeuib.ttf"),
    "Arial": ("arial.ttf", "arialbd.ttf"),
    "Times New Roman": ("times.ttf", "timesbd.ttf"),
    "Calibri": ("calibri.ttf", "calibrib.ttf"),
    "Segoe Print": ("segoepr.ttf", "segoeprb.ttf"),
}
_fc = {}


def F(fam, px, bold=False):
    k = (fam, int(round(px)), bool(bold))
    if k not in _fc:
        _fc[k] = ImageFont.truetype(FD + FAM[fam][1 if bold else 0], k[1])
    return _fc[k]


TRANSCRIBE = "Transcribe all the text in this image exactly, line by line. Output only the text."
ABSENT_SUFFIX = " If it is not shown, answer 'not shown'."
ITEMS = []

# ---------------------------------------------------------------- drawing helpers
DARK = dict(bg=(30, 31, 34), panel=(43, 45, 49), fg=(220, 221, 222), dim=(140, 143, 148), line=(70, 72, 77),
            accent=(88, 166, 255), red=(240, 90, 90), green=(110, 200, 120), yellow=(230, 190, 90),
            kw=(198, 120, 221), st=(152, 195, 121), num=(209, 154, 102), com=(110, 118, 129))
LIGHT = dict(bg=(255, 255, 255), panel=(243, 243, 243), fg=(25, 25, 25), dim=(110, 110, 110), line=(210, 210, 210),
             accent=(0, 95, 184), red=(196, 43, 28), green=(16, 124, 16), yellow=(150, 110, 0),
             kw=(0, 0, 200), st=(163, 21, 21), num=(9, 134, 88), com=(0, 128, 0))


class Page:
    def __init__(self, w, h, bg, scale=1):
        self.w, self.h, self.scale = w, h, scale
        self.im = Image.new("RGB", (w * scale, h * scale), bg)
        self.d = ImageDraw.Draw(self.im)
        self.lines = []

    def S(self, v):
        return int(round(v * self.scale))

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1, r=0):
        box = [self.S(x0), self.S(y0), self.S(x1), self.S(y1)]
        if r:
            self.d.rounded_rectangle(box, self.S(r), fill=fill, outline=outline, width=self.S(width))
        else:
            self.d.rectangle(box, fill=fill, outline=outline, width=self.S(width))

    def ln(self, x0, y0, x1, y1, fill, width=1):
        self.d.line([self.S(x0), self.S(y0), self.S(x1), self.S(y1)], fill=fill, width=max(1, self.S(width)))

    def ellipse(self, x0, y0, x1, y1, fill=None, outline=None):
        self.d.ellipse([self.S(x0), self.S(y0), self.S(x1), self.S(y1)], fill=fill, outline=outline)

    @staticmethod
    def tw(txt, fam, px, bold=False):  # logical width
        return F(fam, px, bold).getlength(txt)

    def _rec(self, txt, nl, sep):
        if not txt.strip():
            return
        if nl or not self.lines:
            self.lines.append(txt.rstrip())
        else:
            self.lines[-1] += sep + txt.rstrip()

    def text(self, x, y, txt, fam, px, fill, bold=False, nl=True, sep=" ", rec=True, anchor="la"):
        f = F(fam, px * self.scale, bold)
        xy = (self.S(x), self.S(y))
        bb = self.d.textbbox(xy, txt, font=f, anchor=anchor)
        if bb[0] < 0 or bb[1] < 0 or bb[2] > self.im.width or bb[3] > self.im.height:
            raise ValueError(f"text out of bounds: {txt!r} {bb} {self.im.size}")
        self.d.text(xy, txt, font=f, fill=fill, anchor=anchor)
        if rec:
            self._rec(txt, nl, sep)

    def right(self, xr, y, txt, fam, px, fill, **kw):
        self.text(xr, y, txt, fam, px, fill, anchor="ra", **kw)

    def center(self, xc, y, txt, fam, px, fill, **kw):
        self.text(xc, y, txt, fam, px, fill, anchor="ma", **kw)

    def rich(self, x, y, segs, fam, px, bold=False):
        """Draw coloured segments on one line; recorded as one line (no separators)."""
        xx = x
        full = "".join(t for t, _ in segs)
        for t, c in segs:
            if t:
                self.text(xx, y, t, fam, px, c, bold=bold, rec=False)
                xx += self.tw(t, fam, px, bold)
        self._rec(full, True, "")

    def mono(self, x, y, lines, fam, px, fill, gap=1.45, colors=None):
        for i, t in enumerate(lines):
            if t:
                self.text(x, y + i * px * gap, t, fam, px, (colors or {}).get(i, fill))
        return y + len(lines) * px * gap


def wrap(txt, fam, px, maxw):
    out, cur = [], ""
    for w in txt.split():
        t = (cur + " " + w).strip()
        if Page.tw(t, fam, px) <= maxw:
            cur = t
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


KW = set("""fn pub struct impl let mut self Self if else return use as for in while match true false const async await
export function throw new import from def class with SELECT FROM JOIN ON WHERE AND GROUP BY HAVING ORDER DESC AS DATE
COUNT SUM""".split())


def hl(line, th):
    segs = []
    for m in re.finditer(r"(#.*$|//.*$|\"[^\"]*\"|'[^']*'|`[^`]*`|\b\d[\d_.]*\b|\b[A-Za-z_]+\b|\s+|.)", line):
        t = m.group(0)
        if t.startswith(("#", "//")) and not t.startswith("#["):
            c = th["com"]
        elif t[0] in "\"'`":
            c = th["st"]
        elif t[0].isdigit():
            c = th["num"]
        elif t in KW:
            c = th["kw"]
        else:
            c = th["fg"]
        segs.append((t, c))
    return segs


def titlebar(P, w, title, th, h=30, fam="Segoe UI", px=12, x0=0, y0=0):
    P.rect(x0, y0, x0 + w, y0 + h, fill=th["panel"])
    P.text(x0 + 12, y0 + (h - px) / 2 - 2, title, fam, px, th["fg"])
    for i, c in enumerate([(230, 80, 70), (230, 190, 60), (90, 190, 80)]):
        P.ellipse(x0 + w - 70 + i * 20, y0 + h / 2 - 5, x0 + w - 60 + i * 20, y0 + h / 2 + 5, fill=c)


def button(P, x, y, label, fam, px, th, primary=False, nl=True):
    w = P.tw(label, fam, px) + 28
    h = px * 2.2
    P.rect(x, y, x + w, y + h, fill=th["accent"] if primary else th["panel"], outline=th["line"], r=4)
    P.text(x + 14, y + (h - px) / 2 - px * 0.15, label, fam, px, (255, 255, 255) if primary else th["fg"], nl=nl)
    return x + w + 10


def table(P, x, y, cols, rows, fam, px, th, row_h=None, header_bold=True, zebra=None, grid=True, aligns=None,
          header_fill=None):
    """cols: list of (title, width). rows: list of lists. Records each row as one line."""
    row_h = row_h or px * 1.9
    tot = sum(w for _, w in cols)
    if header_fill:
        P.rect(x, y, x + tot, y + row_h, fill=header_fill)
    allrows = [[c for c, _ in cols]] + rows
    for r, row in enumerate(allrows):
        yy = y + r * row_h
        if zebra and r > 0 and r % 2 == 0:
            P.rect(x, yy, x + tot, yy + row_h, fill=zebra)
        xx = x
        for c, cell in enumerate(row):
            w = cols[c][1]
            al = (aligns or "l" * len(cols))[c]
            ty = yy + (row_h - px) / 2 - px * 0.12
            bold = header_bold and r == 0
            if al == "r":
                P.right(xx + w - 6, ty, cell, fam, px, th["fg"], bold=bold, nl=(c == 0))
            else:
                P.text(xx + 6, ty, cell, fam, px, th["fg"], bold=bold, nl=(c == 0))
            xx += w
        if grid:
            P.ln(x, yy + row_h, x + tot, yy + row_h, th["line"])
    if grid:
        xx = x
        for _, w in cols + [("", 0)]:
            P.ln(xx, y, xx, y + len(allrows) * row_h, th["line"])
            xx += w
        P.ln(x, y, x + tot, y, th["line"])
    return y + len(allrows) * row_h


# ---------------------------------------------------------------- page builders
# Each builder draws into P (logical coordinates) and returns {"qa": [(q, [answers])...], "absent": [q...]}.

def terminal_cargo(P, th):
    P.text(12, 8, "sam@devbox: ~/work/ledger", "Segoe UI", 12, th["dim"])
    lines = [
        "sam@devbox:~/work/ledger$ cargo build --release",
        "   Compiling serde v1.0.214",
        "   Compiling ledger-core v0.9.3 (/home/sam/work/ledger/core)",
        "warning: unused variable: `rate`",
        "  --> core/src/fx.rs:88:9",
        "error[E0308]: mismatched types",
        "   --> core/src/post.rs:142:23",
        "    |",
        "142 |     let total: u64 = amount * qty;",
        "    |                ---   ^^^^^^^^^^^^ expected `u64`, found `i64`",
        "error: could not compile `ledger-core` (lib) due to 1 previous error; 1 warning emitted",
        "sam@devbox:~/work/ledger$",
    ]
    cols = {3: th["yellow"], 5: th["red"], 10: th["red"]}
    P.mono(14, 36, lines, "Consolas", 12, th["fg"], colors=cols)
    return dict(qa=[("What error code is reported for the mismatched types error?", ["E0308", "error[E0308]"]),
                    ("On which line of post.rs is the error reported?", ["142"])],
                absent=["What exit status code is printed after the build?"])


def terminal_npm(P, th):
    lines = [
        "PS C:\\src\\portal> npm run build",
        "",
        "> portal@3.12.0 build",
        "> vite build --mode production",
        "",
        "vite v5.4.10 building for production...",
        "1284 modules transformed.",
        "dist/index.html                    0.46 kB | gzip:   0.30 kB",
        "dist/assets/index-4f9c2a1b.css    38.72 kB | gzip:   7.14 kB",
        "dist/assets/vendor-b81e07d3.js   412.05 kB | gzip: 131.88 kB",
        "dist/assets/index-0c5d9e44.js     96.33 kB | gzip:  29.61 kB",
        "(!) Some chunks are larger than 400 kB after minification.",
        "built in 7.82s",
        "PS C:\\src\\portal>",
    ]
    P.mono(10, 10, lines, "Consolas", 10, th["fg"], colors={6: th["green"], 11: th["yellow"], 12: th["green"]})
    return dict(qa=[("How many modules were transformed?", ["1284"]),
                    ("What is the gzip size of the vendor JS file?", ["131.88 kB", "131.88"])],
                absent=["What is the size of the file dist/assets/worker.js?"])


def terminal_git(P, th):
    lines = [
        "$ git log --oneline -6",
        "a3f9c21 (HEAD -> main, origin/main) Fix rounding in VAT export",
        "7be0d44 Add retry to webhook sender",
        "19c6e8f Bump tokio to 1.41.1",
        "e02a7b3 Merge pull request #318 from feature/csv-import",
        "c4d81f0 Handle empty CSV rows",
        "5a6b29e Update README install steps",
        "$ git status -sb",
        "## main...origin/main",
        " M src/export/vat.rs",
        "?? notes/todo.md",
        "$",
    ]
    titlebar(P, P.w, "Terminal - billing-svc", th, h=28, px=12)
    P.mono(14, 40, lines, "Consolas", 14, th["fg"], colors={9: th["red"], 10: th["red"]})
    return dict(qa=[("What is the short hash of the commit that bumps tokio?", ["19c6e8f"]),
                    ("Which pull request number was merged?", ["318", "#318"])],
                absent=["Which commit is tagged v2.0?"])


def terminal_pytest(P, th):
    lines = [
        "============================= test session starts ==============================",
        "platform linux -- Python 3.12.4, pytest-8.3.2, pluggy-1.5.0",
        "rootdir: /srv/app",
        "collected 57 items",
        "",
        "tests/test_api.py ........F......                                        [ 26%]",
        "tests/test_models.py ....................                                [ 61%]",
        "tests/test_tasks.py ........s.........s...                               [100%]",
        "",
        "FAILED tests/test_api.py::test_refund_partial - AssertionError: 1250 != 1205",
        "=================== 1 failed, 54 passed, 2 skipped in 8.41s ====================",
    ]
    P.mono(10, 10, lines, "Consolas", 9, th["fg"], colors={9: th["red"], 10: th["red"]})
    return dict(qa=[("How many tests were skipped?", ["2"]), ("What Python version is shown?", ["3.12.4"])],
                absent=["How many tests were deselected?"])


def buildlog_ci(P, th):
    P.rect(0, 0, P.w, 32, fill=th["panel"])
    P.text(12, 9, "test (ubuntu-24.04)  /  Run unit tests", "Segoe UI", 11, th["fg"], bold=True)
    lines = [
        "2026-10-30T14:02:11.418Z  Run actions/checkout@v4",
        "2026-10-30T14:02:12.903Z  Syncing repository: northwind/billing-svc",
        "2026-10-30T14:02:19.551Z  Run ./gradlew test --no-daemon",
        "2026-10-30T14:03:47.206Z  > Task :core:test",
        "2026-10-30T14:04:02.774Z  InvoiceServiceTest > appliesCreditNote() FAILED",
        "2026-10-30T14:04:02.775Z      java.lang.AssertionError: expected:<118.40> but was:<121.90>",
        "2026-10-30T14:04:02.776Z          at InvoiceServiceTest.java:213",
        "2026-10-30T14:04:09.330Z  86 tests completed, 1 failed, 3 skipped",
        "2026-10-30T14:04:09.512Z  BUILD FAILED in 1m 57s",
        "2026-10-30T14:04:09.940Z  Error: Process completed with exit code 1.",
    ]
    for i, t in enumerate(lines):
        y = 44 + i * 11 * 1.6
        P.text(14, y, str(i + 1), "Consolas", 11, th["dim"])
        c = th["red"] if i in (4, 5, 8, 9) else th["fg"]
        P.text(44, y, t, "Consolas", 11, c, nl=False)
    return dict(qa=[("What was the expected value in the failed assertion?", ["118.40"]),
                    ("How many tests completed?", ["86"])],
                absent=["Which Java version did the job use?"])


def code_editor(P, th, tabs, fname, start, code, px, status):
    gut = px * 3.6
    P.rect(0, 0, P.w, 30, fill=th["panel"])
    x = 10
    for t in tabs:
        w = P.tw(t, "Segoe UI", 12) + 24
        if t == fname:
            P.rect(x, 2, x + w, 30, fill=th["bg"])
        P.text(x + 12, 8, t, "Segoe UI", 12, th["fg"] if t == fname else th["dim"], nl=(t == tabs[0]), sep="  ")
        x += w + 2
    P.rect(0, 30, gut, P.h - 24, fill=th["panel"])
    y = 40
    for i, c in enumerate(code):
        n = str(start + i)
        P.right(gut - 8, y, n, "Consolas", px, th["dim"])
        if c:
            # record "N code" as one line
            P.lines[-1] += " "
            xx = gut + 10
            for t, col in hl(c, th):
                if t:
                    P.text(xx, y, t, "Consolas", px, col, rec=False)
                    xx += P.tw(t, "Consolas", px)
            P.lines[-1] += c
        y += px * 1.5
    assert y <= P.h - 24, f"code overflows editor: {y} > {P.h - 24}"
    P.rect(0, P.h - 24, P.w, P.h, fill=th["accent"])
    P.text(10, P.h - 19, status, "Segoe UI", 11, (255, 255, 255))


def code_rust(P, th):
    code = [
        "pub struct RateLimiter {",
        "    capacity: u32,",
        "    refill_per_sec: f64,",
        "    tokens: f64,",
        "    last: Instant,",
        "}",
        "",
        "impl RateLimiter {",
        "    pub fn new(capacity: u32, refill_per_sec: f64) -> Self {",
        "        Self { capacity, refill_per_sec, tokens: capacity as f64, last: Instant::now() }",
        "    }",
        "",
        "    pub fn try_acquire(&mut self) -> bool {",
        "        let now = Instant::now();",
        "        let dt = now.duration_since(self.last).as_secs_f64();",
        "        self.tokens = (self.tokens + dt * self.refill_per_sec).min(self.capacity as f64);",
        "        self.last = now;",
        "        if self.tokens >= 1.0 { self.tokens -= 1.0; true } else { false }",
        "    }",
        "}",
    ]
    start = 37
    code_editor(P, th, ["main.rs", "rate_limit.rs", "config.rs"], "rate_limit.rs", start, code, 12,
                "Ln 52, Col 9    Spaces: 4    UTF-8    LF    Rust")
    ln = start + next(i for i, c in enumerate(code) if "fn try_acquire" in c)
    return dict(qa=[("On which line number is the function try_acquire declared?", [str(ln)]),
                    ("What is the type of the field refill_per_sec?", ["f64"])],
                absent=["What is the value of the DEFAULT_CAPACITY constant?"])


def code_ts(P, th):
    code = [
        "export async function fetchInvoices(customerId: string, page = 1) {",
        "  const url = `${API_BASE}/customers/${customerId}/invoices?page=${page}&limit=25`;",
        "  const res = await fetch(url, { headers: { Authorization: `Bearer ${token()}` } });",
        "  if (res.status === 429) {",
        "    await sleep(1500);",
        "    return fetchInvoices(customerId, page);",
        "  }",
        "  if (!res.ok) throw new Error(`Invoice fetch failed: ${res.status}`);",
        "  const body: InvoicePage = await res.json();",
        "  return body.items.filter((i) => i.state !== \"void\");",
        "}",
    ]
    code_editor(P, th, ["api.ts", "useInvoices.ts", "types.ts"], "useInvoices.ts", 112, code, 14,
                "Ln 116, Col 22    Spaces: 2    UTF-8    LF    TypeScript")
    return dict(qa=[("How many milliseconds does the code sleep after a 429 response?", ["1500"]),
                    ("What page size (limit) is used in the URL?", ["25"])],
                absent=["What timeout in milliseconds is passed to fetch?"])


def code_py(P, th):
    code = [
        "import csv",
        "from dataclasses import dataclass",
        "from decimal import Decimal",
        "",
        "VAT_RATE = Decimal(\"0.21\")",
        "MAX_ROWS = 50_000",
        "",
        "@dataclass",
        "class Line:",
        "    sku: str",
        "    qty: int",
        "    unit_price: Decimal",
        "",
        "def load(path):",
        "    with open(path, newline=\"\") as fh:",
        "        rows = list(csv.DictReader(fh))[:MAX_ROWS]",
        "    return [Line(r[\"sku\"], int(r[\"qty\"]), Decimal(r[\"price\"])) for r in rows]",
    ]
    code_editor(P, th, ["lines.py", "report.py"], "lines.py", 1, code, 10,
                "Ln 6, Col 18    Spaces: 4    UTF-8    Python 3.12")
    return dict(qa=[("What is the value of VAT_RATE?", ["0.21"]),
                    ("What is MAX_ROWS set to?", ["50_000", "50000", "50,000"])],
                absent=["What is the value of the DEFAULT_CURRENCY constant?"])


def code_sql(P, th):
    code = [
        "SELECT c.region,",
        "       COUNT(*)          AS orders,",
        "       SUM(o.total_eur)  AS revenue",
        "FROM orders o",
        "JOIN customers c ON c.id = o.customer_id",
        "WHERE o.created_at >= DATE '2026-07-01'",
        "  AND o.status <> 'cancelled'",
        "GROUP BY c.region",
        "HAVING SUM(o.total_eur) > 25000",
        "ORDER BY revenue DESC;",
    ]
    code_editor(P, th, ["regional_revenue.sql"], "regional_revenue.sql", 8, code, 16,
                "Connected: analytics@warehouse    Ln 17, Col 23")
    return dict(qa=[("What is the start date in the WHERE clause?", ["2026-07-01"]),
                    ("What revenue threshold is used in the HAVING clause?", ["25000", "25,000"])],
                absent=["What LIMIT is applied to the query?"])


def desktop(P, th):
    for y in range(0, P.h, 4):
        t = y / P.h
        P.rect(0, y, P.w, y + 4, fill=(int(40 + 30 * t), int(70 + 40 * t), int(120 + 50 * t)))


def dialog_error_win(P, th):
    desktop(P, th)
    x0, y0, x1, y1 = 90, 60, 670, 330
    P.rect(x0, y0, x1, y1, fill=th["bg"], outline=th["line"])
    titlebar(P, x1 - x0, "Harbor Sync", th, h=30, px=12, x0=x0, y0=y0)
    P.ellipse(x0 + 22, y0 + 54, x0 + 54, y0 + 86, fill=th["red"])
    P.text(x0 + 33, y0 + 60, "!", "Segoe UI", 16, (255, 255, 255), bold=True, rec=False)
    P.text(x0 + 70, y0 + 52, "Sync failed", "Segoe UI", 15, th["fg"], bold=True)
    body = ["The file could not be uploaded because it is locked by another process.",
            "File: D:\\Projects\\Q4\\forecast_v3.xlsx",
            "Error code: 0x8007016A",
            "Attempt 3 of 5. Next retry in 45 seconds."]
    for i, t in enumerate(body):
        P.text(x0 + 70, y0 + 84 + i * 22, t, "Segoe UI", 12, th["fg"])
    x = x0 + 250
    x = button(P, x, y1 - 46, "Retry now", "Segoe UI", 12, th, primary=True)
    x = button(P, x, y1 - 46, "Skip file", "Segoe UI", 12, th, nl=False)
    button(P, x, y1 - 46, "Cancel", "Segoe UI", 12, th, nl=False)
    return dict(qa=[("What is the error code?", ["0x8007016A"]),
                    ("How many seconds until the next retry?", ["45"])],
                absent=["What is the file size of the locked file?"])


def dialog_error_dark(P, th):
    P.rect(0, 0, P.w, P.h, fill=(18, 18, 20))
    x0, y0, x1, y1 = 60, 50, 640, 320
    P.rect(x0, y0, x1, y1, fill=th["bg"], outline=th["line"], r=6)
    titlebar(P, x1 - x0, "Atlas IDE", th, h=28, px=11, x0=x0, y0=y0)
    P.text(x0 + 24, y0 + 46, "Debugger disconnected", "Segoe UI", 14, th["fg"], bold=True)
    body = ["Lost connection to the debug adapter on port 9229.",
            "Process: node (PID 18342)",
            "Last breakpoint: src/server/routes/orders.js:77"]
    for i, t in enumerate(body):
        P.text(x0 + 24, y0 + 80 + i * 20, t, "Segoe UI", 11, th["fg"])
    P.rect(x0 + 24, y0 + 152, x0 + 36, y0 + 164, fill=th["accent"])
    P.text(x0 + 44, y0 + 150, "Restart debugging automatically", "Segoe UI", 11, th["fg"])
    x = x1 - 190
    x = button(P, x, y1 - 44, "Reconnect", "Segoe UI", 11, th, primary=True)
    button(P, x, y1 - 44, "Close", "Segoe UI", 11, th, nl=False)
    return dict(qa=[("What port was the debug adapter on?", ["9229"]),
                    ("What is the PID of the process?", ["18342"])],
                absent=["What Node.js version is shown?"])


def dialog_confirm(P, th):
    P.rect(0, 0, P.w, P.h, fill=(200, 205, 212))
    x0, y0, x1, y1 = 60, 50, 760, 360
    P.rect(x0, y0, x1, y1, fill=th["bg"], outline=th["line"])
    titlebar(P, x1 - x0, "Delete Multiple Items", th, h=32, fam="Arial", px=14, x0=x0, y0=y0)
    P.text(x0 + 28, y0 + 56, "Delete 14 items?", "Arial", 18, th["fg"], bold=True)
    body = ["These items will be moved to the Recycle Bin. Items larger than 2 GB",
            "will be deleted permanently.",
            "Largest item: render_cache_0412.bin (3.7 GB)"]
    for i, t in enumerate(body):
        P.text(x0 + 28, y0 + 96 + i * 24, t, "Arial", 14, th["fg"])
    P.rect(x0 + 28, y0 + 182, x0 + 42, y0 + 196, outline=th["dim"])
    P.text(x0 + 52, y0 + 181, "Do this for all current items", "Arial", 14, th["fg"])
    x = x1 - 210
    x = button(P, x, y1 - 54, "Delete", "Arial", 14, th, primary=True)
    button(P, x, y1 - 54, "Cancel", "Arial", 14, th, nl=False)
    return dict(qa=[("How many items are being deleted?", ["14"]),
                    ("What is the size of the largest item?", ["3.7 GB"])],
                absent=["When were the items last modified?"])


def settings_page(P, th, fam, px, sidebar, active, title, rows, foot=None, side_w=190):
    P.rect(0, 0, side_w, P.h, fill=th["panel"])
    P.text(20, 18, "Settings", fam, px + 6, th["fg"], bold=True)
    for i, s in enumerate(sidebar):
        y = 64 + i * px * 2.6
        if s == active:
            P.rect(10, y - px * 0.6, side_w - 10, y + px * 1.6, fill=th["line"], r=4)
        P.text(24, y, s, fam, px, th["fg"])
    x = side_w + 32
    P.text(x, 18, title, fam, px + 8, th["fg"], bold=True)
    y = 70
    for k, v in rows:
        P.rect(x, y, P.w - 30, y + px * 3.2, fill=th["panel"], r=4)
        P.text(x + 14, y + px * 1.0, k, fam, px, th["fg"])
        P.right(P.w - 46, y + px * 1.0, v, fam, px, th["accent"], nl=False)
        y += px * 3.2 + 6
    if foot:
        for f in foot:
            y += px * 0.6
            if f.startswith("[btn]"):
                button(P, x, y, f[5:], fam, px, th)
                y += px * 2.4
            else:
                P.text(x, y, f, fam, px, th["dim"])
                y += px * 1.6


def settings_app(P, th):
    settings_page(P, th, "Segoe UI", 12,
                  ["General", "Appearance", "Notifications", "Privacy", "Storage", "Updates", "About"], "Storage",
                  "Storage", [("Cache location", "C:\\Users\\Public\\AppCache"), ("Maximum cache size", "2048 MB"),
                              ("Clear cache on exit", "Off"), ("Offline files", "Keep 30 days"),
                              ("Download folder", "D:\\Downloads\\Harbor"), ("Compression", "zstd (level 6)"),
                              ("Last cleanup", "2026-10-02 03:15")],
                  foot=["[btn]Clear cache now", "Used: 1.38 GB of 2.00 GB"])
    return dict(qa=[("What is the maximum cache size?", ["2048 MB", "2048"]),
                    ("What compression level is configured?", ["6", "level 6"])],
                absent=["What is the proxy server address?"])


def settings_dark(P, th):
    settings_page(P, th, "Segoe UI", 11,
                  ["System", "Display", "Sound", "Notifications", "Focus", "Power", "Accounts"], "Notifications",
                  "Notifications", [("Do not disturb", "On"), ("Schedule", "22:00 - 07:30"),
                                    ("Allow calls from", "Favorites"), ("Repeat callers", "Off"),
                                    ("Badge count limit", "99"), ("Sound", "Chime 2"),
                                    ("Show previews", "When unlocked")],
                  foot=["Changes apply to all signed-in devices."])
    return dict(qa=[("What is the Do not disturb schedule?", ["22:00 - 07:30", "22:00-07:30", "22:00 to 07:30"]),
                    ("What is the badge count limit?", ["99"])],
                absent=["What is the vibration pattern setting?"])


def settings_network(P, th):
    settings_page(P, th, "Calibri", 14, ["Status", "Ethernet", "Wi-Fi", "VPN", "Proxy"], "Ethernet",
                  "Ethernet 2 - IP settings",
                  [("IPv4 address", "192.168.40.27"), ("Subnet mask", "255.255.255.0"),
                   ("Default gateway", "192.168.40.1"), ("Preferred DNS", "192.168.40.2"),
                   ("Alternate DNS", "192.168.40.3"), ("MTU", "1492"), ("DHCP", "Disabled"),
                   ("MAC address", "3C-52-82-1F-A9-0E")], side_w=170)
    return dict(qa=[("What is the default gateway?", ["192.168.40.1"]), ("What is the MTU?", ["1492"])],
                absent=["What is the IPv6 address?"])


CUSTOMERS = ["Bluefin Labs", "Orchard Supply", "Kestrel Media", "Tallow & Pine", "Northgate Foods", "Quarry Bikes",
             "Mosaic Dental", "Ridge Analytics", "Saltmarsh Co.", "Pemberly Tools"]


def money(v, sym="$"):
    return f"{sym}{v:,.2f}"


def sheet_sales(P, th):
    rng = random.Random(1601)
    rows, d0 = [], datetime.date(2026, 9, 1)
    offs = sorted(rng.sample(range(0, 60), 14))
    tot_u, tot_a = 0, 0.0
    for i in range(14):
        u = rng.randint(3, 240)
        a = round(u * rng.uniform(12, 95), 2)
        tot_u += u
        tot_a += a
        rows.append([f"SO-{10431 + i}", (d0 + datetime.timedelta(days=offs[i])).isoformat(), rng.choice(CUSTOMERS),
                     rng.choice(["North", "South", "East", "West"]), str(u), money(a)])
    # chrome
    P.rect(0, 0, P.w, 28, fill=(33, 115, 70))
    P.text(10, 7, "Q3-orders.xlsx - Sheet1", "Calibri", 12, (255, 255, 255))
    P.rect(0, 28, P.w, 54, fill=(245, 245, 245))
    P.text(10, 34, "F16", "Calibri", 11, th["fg"])
    P.text(60, 34, "fx", "Calibri", 11, th["dim"], nl=False)
    P.text(84, 34, "=SUM(F2:F15)", "Calibri", 11, th["fg"], nl=False)
    cols = [("Order", 80), ("Date", 84), ("Customer", 130), ("Region", 66), ("Units", 54), ("Amount", 100)]
    x0, y0, rh = 34, 60, 21
    letters = "ABCDEF"
    xx = x0
    for i, (_, w) in enumerate(cols):
        P.center(xx + w / 2, y0 + 4, letters[i], "Calibri", 11, th["dim"], nl=(i == 0))
        xx += w
    data = rows + [["Total", "", "", "", str(tot_u), money(tot_a)]]
    for r in range(len(data) + 1):
        P.right(x0 - 6, y0 + rh * (r + 1) + 4, str(r + 1), "Calibri", 11, th["dim"])  # row number starts the line
        row = [c for c, _ in cols] if r == 0 else data[r - 1]
        xx = x0
        for c, cell in enumerate(row):
            w = cols[c][1]
            if cell:
                if c >= 4 and r > 0:
                    P.right(xx + w - 5, y0 + rh * (r + 1) + 4, cell, "Calibri", 11, th["fg"], nl=False,
                            bold=(r == len(data)))
                else:
                    P.text(xx + 5, y0 + rh * (r + 1) + 4, cell, "Calibri", 11, th["fg"], nl=False,
                           bold=(r == 0 or r == len(data)))
            xx += w
    for r in range(len(data) + 3):
        P.ln(0, y0 + rh * r, P.w, y0 + rh * r, th["line"])
    xx = x0
    for _, w in cols + [("", 0)]:
        P.ln(xx, y0, xx, y0 + rh * (len(data) + 2), th["line"])
        xx += w
    q1, q2 = rows[6], rows[11]
    return dict(qa=[(f"What is the Amount for order {q1[0]}?", [q1[5], q1[5][1:]]),
                    (f"On what date was order {q2[0]} placed?", [q2[1]])],
                absent=["What is the Amount for order SO-10458?"])


def sheet_inventory(P, th):
    rng = random.Random(1702)
    descs = ["Hex bolt M8x40", "Hex nut M8", "Washer M8 zinc", "Cable tie 200mm", "Hose clamp 25mm",
             "Wood screw 4x40", "Anchor plug 6mm", "Hinge 75mm brass", "Drawer slide 450mm", "Shelf bracket 200mm",
             "Gasket 32mm", "O-ring 18x2", "Spring pin 4x30", "Rivet 4.8x12", "Threaded rod M10 1m",
             "Eye bolt M6", "Wing nut M6", "Caster 50mm"]
    rows = []
    for i, ds in enumerate(descs):
        sku = f"HX-{2201 + i * 3 + rng.randint(0, 2)}"
        rows.append([sku, ds, f"{rng.choice('ABCD')}-{rng.randint(1, 24):02d}-{rng.randint(1, 6)}",
                     str(rng.randint(0, 2400)), str(rng.choice([50, 100, 200, 250, 500])),
                     money(rng.uniform(0.04, 18.0)),
                     f"{rng.randint(9, 11)}/{rng.randint(1, 28):02d}/2026"])
    P.text(12, 10, "Inventory - Warehouse 2 (Fasteners & Hardware)", "Arial", 13, th["fg"], bold=True)
    P.text(12, 30, "Showing 18 of 412 items  |  Filter: Category = Hardware", "Arial", 10, th["dim"])
    cols = [("SKU", 70), ("Description", 170), ("Bin", 70), ("On hand", 70), ("Reorder", 70), ("Unit cost", 80),
            ("Last count", 90)]
    table(P, 12, 52, cols, rows, "Arial", 10, th, row_h=20, zebra=(246, 248, 250), aligns="lllrrrl",
          header_fill=(230, 234, 240))
    a, b = rows[4], rows[13]
    return dict(qa=[(f"Which bin is SKU {a[0]} stored in?", [a[2]]),
                    (f"What is the unit cost of the item '{b[1]}'?", [b[5], b[5][1:]])],
                absent=[f"Who is the supplier of SKU {rows[9][0]}?"])


def sheet_dark(P, th):
    rng = random.Random(1803)
    names = ["api-eu-01", "api-eu-02", "api-us-01", "api-us-02", "worker-01", "worker-02", "worker-03",
             "cache-01", "db-replica-1", "gateway-01"]
    deg = rng.randrange(len(names))
    rows = []
    for i, n in enumerate(names):
        rows.append([n, f"{rng.uniform(4, 96):.1f}", f"{rng.uniform(2, 15):.1f} / 16", f"{rng.randint(0, 60)}d {rng.randint(0, 23)}h",
                     f"2026-11-{rng.randint(1, 10):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}",
                     "Degraded" if i == deg else "Healthy"])
    P.text(16, 12, "Fleet overview", "Segoe UI", 16, th["fg"], bold=True)
    P.text(16, 38, "10 hosts  |  refreshed every 30 s", "Segoe UI", 12, th["dim"])
    cols = [("Server", 140), ("CPU %", 80), ("Mem GB", 110), ("Uptime", 90), ("Last deploy", 150), ("Status", 100)]
    table(P, 16, 66, cols, rows, "Segoe UI", 12, th, row_h=28, aligns="lrrrll", header_fill=th["panel"])
    a = rows[3]
    return dict(qa=[(f"What is the CPU % of {a[0]}?", [a[1]]),
                    ("Which server has the status Degraded?", [rows[deg][0]])],
                absent=[f"What is the disk usage of {rows[1][0]}?"])


def table_fx(P, th):
    rng = random.Random(1904)
    cur = [("Euro", "EUR", 0.93), ("British pound", "GBP", 0.79), ("Swiss franc", "CHF", 0.88),
           ("Canadian dollar", "CAD", 1.39), ("Australian dollar", "AUD", 1.52), ("Swedish krona", "SEK", 10.71),
           ("Norwegian krone", "NOK", 10.95), ("Danish krone", "DKK", 6.94), ("Polish zloty", "PLN", 4.01),
           ("Czech koruna", "CZK", 23.38), ("Singapore dollar", "SGD", 1.33), ("New Zealand dollar", "NZD", 1.68)]
    rows = []
    for n, c, r in cur:
        r = r * rng.uniform(0.97, 1.03)
        rows.append([n, c, f"{r * 0.99:.4f}", f"{r * 1.01:.4f}", f"{rng.uniform(-0.9, 0.9):+.2f}%"])
    P.text(10, 8, "Indicative rates vs USD - 11 Nov 2026 10:00 UTC", "Arial", 9, th["fg"], bold=True)
    cols = [("Currency", 140), ("Code", 50), ("Buy", 80), ("Sell", 80), ("Change", 70)]
    y = table(P, 10, 26, cols, rows, "Arial", 9, th, row_h=17, aligns="llrrr", zebra=(245, 245, 245))
    P.text(10, y + 8, "Rates are indicative only and exclude fees. Source: internal treasury feed.", "Arial", 9,
           th["dim"])
    a, b = rows[2], rows[6]
    return dict(qa=[(f"What is the Sell rate for {a[1]}?", [a[3]]), (f"What is the Change for {b[1]}?", [b[4]])],
                absent=["What is the Buy rate for JPY?"])


def receipt_cafe(P, th):
    lines = [
        "      THE COPPER KETTLE",
        "    41 Mill Lane, Ashford",
        "      Tel 01632 960 418",
        "Order #4471          Table 6",
        "2026-10-18 12:47  Server: Dana",
        "----------------------------",
        "2 x Flat white          7.00",
        "1 x Avocado toast       9.50",
        "1 x Tomato soup         6.25",
        "1 x Lemon tart          4.80",
        "----------------------------",
        "Subtotal               27.55",
        "Service 10%             2.76",
        "TOTAL                  30.31",
        "",
        "Paid: CARD ****2291",
        "Thank you - see you soon!",
    ]
    P.mono(22, 24, lines, "Consolas", 12, (30, 30, 30), gap=1.6)
    return dict(qa=[("What is the TOTAL on the receipt?", ["30.31"]), ("What is the order number?", ["4471", "#4471"])],
                absent=["How much VAT is shown on the receipt?"])


def receipt_hardware(P, th):
    f, px = "Arial", 10
    y = 18
    def lr(l, r, bold=False):
        nonlocal y
        P.text(18, y, l, f, px, (20, 20, 20), bold=bold)
        if r:
            P.right(P.w - 18, y, r, f, px, (20, 20, 20), nl=False, bold=bold)
        y += px * 1.7
    P.center(P.w / 2, y, "BRANDT HARDWARE", f, 14, (20, 20, 20), bold=True); y += 24
    P.center(P.w / 2, y, "1900 Quarry Road, Millbrook", f, px, (20, 20, 20)); y += 17
    lr("Store 0213  Reg 04  Txn 88120", "")
    lr("10/27/2026  16:05", "")
    y += 6
    items = [("WOOD SCREWS 4X40 200PK", "7.49"), ("WALL ANCHORS 6MM 50PK", "4.29"), ("STUD FINDER BASIC", "18.99"),
             ("MASKING TAPE 24MM x2", "6.58"), ("SANDPAPER ASSORTED", "5.99"), ("PAINT TRAY LINER 3PK", "3.79")]
    for n, p in items:
        lr(n, p)
    sub = sum(float(p) for _, p in items)
    tax = round(sub * 0.0725, 2)
    tot = round(sub + tax, 2)
    y += 6
    lr("SUBTOTAL", f"{sub:.2f}")
    lr("SALES TAX 7.25%", f"{tax:.2f}")
    lr("TOTAL", f"{tot:.2f}", bold=True)
    lr("CASH", "60.00")
    lr("CHANGE", f"{60 - tot:.2f}")
    y += 8
    P.center(P.w / 2, y, "RETURNS WITHIN 30 DAYS WITH RECEIPT", f, 9, (20, 20, 20))
    return dict(qa=[("What is the transaction number?", ["88120"]),
                    ("How much change was given?", [f"{60 - tot:.2f}"])],
                absent=["What is the cashier's name?"])


def invoice(P, th):
    f = "Arial"
    P.text(50, 40, "INVOICE", f, 26, (30, 60, 110), bold=True)
    P.text(50, 80, "Lumen Print Studio", f, 13, th["fg"], bold=True)
    P.text(50, 100, "Unit 4, 220 Harbour Road, Portsmouth PO2 8LX", f, 11, th["fg"])
    P.text(50, 118, "hello@lumenprint.example", f, 11, th["fg"])
    for i, (k, v) in enumerate([("Invoice no:", "INV-2026-0917"), ("Issue date:", "03 Nov 2026"),
                                ("Due date:", "03 Dec 2026")]):
        P.text(520, 82 + i * 18, k, f, 11, th["dim"])
        P.right(770, 82 + i * 18, v, f, 11, th["fg"], nl=False)
    P.text(50, 170, "Bill to:", f, 11, th["dim"])
    for i, t in enumerate(["Fernhill Primary School", "Accounts Payable", "12 Orchard Way, Fernhill FH3 1QA"]):
        P.text(50, 188 + i * 17, t, f, 11, th["fg"])
    rows = [["A3 posters, gloss 170gsm", "40", "£1.85", "£74.00"],
            ["Tri-fold leaflets, 1000", "2", "£68.00", "£136.00"],
            ["Roller banner 850x2000", "1", "£94.50", "£94.50"],
            ["Design time (hours)", "3", "£45.00", "£135.00"]]
    cols = [("Description", 380), ("Qty", 70), ("Unit price", 120), ("Amount", 150)]
    y = table(P, 50, 270, cols, rows, f, 11, th, row_h=26, aligns="lrrr", header_fill=(232, 238, 246))
    for i, (k, v) in enumerate([("Subtotal", "£439.50"), ("VAT 20%", "£87.90"), ("Total due", "£527.40")]):
        P.text(520, y + 16 + i * 22, k, f, 11, th["fg"], bold=(i == 2))
        P.right(764, y + 16 + i * 22, v, f, 11, th["fg"], nl=False, bold=(i == 2))
    P.text(50, y + 110, "Payment terms: 30 days. Please quote the invoice number as your payment reference.", f, 11,
           th["dim"])
    P.text(50, y + 128, "Thank you for your business.", f, 11, th["dim"])
    return dict(qa=[("What is the total due on the invoice?", ["£527.40", "527.40"]),
                    ("What is the due date?", ["03 Dec 2026", "3 Dec 2026", "2026-12-03", "December 3, 2026"])],
                absent=["What is the purchase order number?"])


def chat_team(P, th):
    P.rect(0, 0, 170, P.h, fill=(63, 14, 64))
    P.text(16, 14, "Northwind Eng", "Segoe UI", 14, (255, 255, 255), bold=True)
    for i, c in enumerate(["# general", "# release-train", "# payments", "# random"]):
        if c == "# release-train":
            P.rect(8, 50 + i * 26, 162, 72 + i * 26, fill=(17, 100, 163), r=4)
        P.text(16, 54 + i * 26, c, "Segoe UI", 12, (230, 220, 230))
    P.text(190, 14, "# release-train", "Segoe UI", 14, th["fg"], bold=True)
    P.ln(170, 44, P.w, 44, th["line"])
    msgs = [("Mira Okafor", "9:14 AM", ["Cutting the 4.18 release branch at 11:00, please merge anything",
                                        "urgent before then."]),
            ("Tomas Reyes", "9:16 AM", ["Payments fix is in review, PR #2291. Should land by 10:30."]),
            ("Mira Okafor", "9:17 AM", ["Great. QA sign-off is booked for 2 PM in room Birch."]),
            ("Lena Vogt", "9:22 AM", ["Heads up: staging DB migration 0047 takes about 12 minutes."])]
    y = 58
    for who, t, body in msgs:
        P.rect(190, y + 2, 222, y + 34, fill=(120 + len(who) * 7 % 100, 140, 180), r=4)
        P.text(232, y, who, "Segoe UI", 12, th["fg"], bold=True)
        P.text(232 + P.tw(who, "Segoe UI", 12, True) + 8, y + 1, t, "Segoe UI", 11, th["dim"], nl=False)
        for b in body:
            y += 19
            P.text(232, y, b, "Segoe UI", 12, th["fg"])
        y += 32
    P.rect(190, P.h - 52, P.w - 20, P.h - 16, outline=th["line"], r=6)
    P.text(202, P.h - 42, "Message #release-train", "Segoe UI", 12, th["dim"])
    return dict(qa=[("At what time will the release branch be cut?", ["11:00"]),
                    ("What is the PR number for the payments fix?", ["2291", "#2291"])],
                absent=["What is the rollback deadline?"])


def chat_dark(P, th):
    P.rect(0, 0, P.w, 64, fill=th["panel"])
    P.text(70, 20, "Jonah", "Segoe UI", 18, th["fg"], bold=True)
    P.ellipse(16, 12, 56, 52, fill=(90, 120, 200))
    msgs = [(0, "Landed! Gate B22, baggage claim 7"), (1, "Nice, I'm in the short-stay car park,"),
            (1, "level 3"), (0, "Which row?"), (1, "Row K, silver estate"), (0, "See you in 10"),
            (1, "Parking was 6.80 so be quick :)")]
    y = 90
    for me, m in msgs:
        w = P.tw(m, "Segoe UI", 14)
        if me:
            P.rect(P.w - w - 44, y, P.w - 16, y + 36, fill=(0, 120, 230), r=16)
            P.text(P.w - w - 30, y + 8, m, "Segoe UI", 14, (255, 255, 255))
        else:
            P.rect(16, y, w + 44, y + 36, fill=(58, 59, 64), r=16)
            P.text(30, y + 8, m, "Segoe UI", 14, th["fg"])
        y += 50
    P.text(P.w - 70, y - 6, "Read 18:42", "Segoe UI", 10, th["dim"])
    P.rect(16, P.h - 56, P.w - 16, P.h - 16, fill=th["panel"], r=20)
    P.text(34, P.h - 46, "Message", "Segoe UI", 14, th["dim"])
    return dict(qa=[("Which baggage claim number is mentioned?", ["7"]),
                    ("Which level of the car park is mentioned?", ["3"])],
                absent=["What is the flight number?"])


def banner_wide(P, th):
    for x in range(0, P.w, 8):
        t = x / P.w
        P.rect(x, 0, x + 8, P.h, fill=(int(250 - 20 * t), int(214 - 30 * t), int(80 + 20 * t)))
    P.text(80, 60, "WINTER SERVICE NOTICE", "Arial", 64, (60, 20, 0), bold=True)
    P.text(80, 170, "Line 14 trains run every 12 minutes on weekends from 6 December.", "Arial", 40, (60, 20, 0))
    P.text(80, 240, "Replacement buses between Hollow Park and Eastmere use stop E4.", "Arial", 40, (60, 20, 0))
    P.text(80, 320, "Plan ahead at transit.example/winter", "Arial", 28, (90, 40, 10))
    return dict(qa=[("Every how many minutes do Line 14 trains run on weekends?", ["12"]),
                    ("Which stop do the replacement buses use?", ["E4"])],
                absent=["What is the fare for the replacement bus?"])


def wide_statusbar(P, th):
    P.rect(0, 0, P.w, 34, fill=th["panel"])
    tabs = ["orders.py", "pricing.py", "test_pricing.py", "settings.toml", "Dockerfile", "README.md",
            "migrations/0047_add_index.sql", "ci.yml"]
    x = 8
    for i, t in enumerate(tabs):
        w = P.tw(t, "Segoe UI", 12) + 30
        P.rect(x, 4, x + w, 34, fill=th["bg"] if i == 1 else th["panel"])
        P.text(x + 15, 12, t, "Segoe UI", 12, th["fg"] if i == 1 else th["dim"], nl=(i == 0), sep="  ")
        x += w + 2
    P.rect(0, 40, P.w, 210, fill=th["bg"])
    P.text(20, 60, "PROBLEMS  OUTPUT  DEBUG CONSOLE  TERMINAL  PORTS", "Segoe UI", 12, th["dim"])
    P.text(20, 90, "pricing.py:88  W0612 Unused variable 'tier'", "Consolas", 12, th["yellow"])
    P.text(20, 112, "pricing.py:131  E1120 No value for argument 'currency' in function call", "Consolas", 12,
           th["red"])
    P.text(20, 134, "test_pricing.py:42  E0602 Undefined variable 'FIXTURE_DIR'", "Consolas", 12, th["red"])
    P.text(20, 156, "orders.py:17  E0401 Unable to import 'legacy.tax'", "Consolas", 12, th["red"])
    P.rect(0, P.h - 26, P.w, P.h, fill=th["accent"])
    items = ["main*", "Errors 3", "Warnings 17", "Ln 214, Col 33", "Spaces: 4", "UTF-8", "CRLF",
             "Python 3.11.9 ('.venv')", "Prettier", "Server: eu-west-2", "Latency 84 ms", "Last sync 14:52"]
    x = 14
    for i, t in enumerate(items):
        P.text(x, P.h - 20, t, "Segoe UI", 12, (255, 255, 255), nl=(i == 0), sep="  ")
        x += P.tw(t, "Segoe UI", 12) + (190 if i in (0, 2, 9) else 40)
    return dict(qa=[("How many warnings are shown in the status bar?", ["17"]),
                    ("What latency is shown in the status bar?", ["84 ms", "84"])],
                absent=["How many tests are passing?"])


def tall_doc(P, th):
    f, px, maxw = "Times New Roman", 14, P.w - 140
    secs = [
        ("Remote Work Equipment Policy", None),
        ("Version 3.2  -  Effective 1 January 2027  -  Owner: People Operations", None),
        ("1. Purpose", "This policy sets out what equipment the company provides to employees who work remotely "
         "for two or more days per week, how to request it, and how it is returned at the end of employment."),
        ("2. Eligibility", "Permanent employees and fixed-term employees with contracts of six months or longer are "
         "eligible after completing their probation period. Contractors are not covered by this policy and should "
         "agree equipment terms with their engagement manager."),
        ("3. Standard equipment", "Each eligible employee receives one laptop, one external monitor of up to 27 "
         "inches, a keyboard, a mouse and a headset. Replacement laptops are issued every four years, or sooner if "
         "the IT service desk confirms that a device cannot be repaired."),
        ("4. Furniture allowance", "Employees may claim up to $650 every 36 months towards an ergonomic chair and "
         "desk. Claims above this amount require written approval from a department head before purchase."),
        ("5. Claims", "Receipts must be submitted through the expenses portal within 60 days of purchase. Claims "
         "submitted after this period will not be reimbursed. Reimbursement is paid with the next monthly payroll "
         "run after approval."),
        ("6. Care of equipment", "Employees are responsible for keeping equipment secure and in good condition. "
         "Loss or theft must be reported to the IT service desk within 24 hours, and a police reference number "
         "should be provided where theft is suspected."),
        ("7. Return of equipment", "On leaving the company, all equipment except furniture purchased under the "
         "allowance must be returned within 10 working days. Prepaid return labels are available from the IT "
         "service desk."),
        ("8. Exceptions", "Requests for equipment outside this policy, including accessibility adjustments, should "
         "be sent to People Operations. Accessibility requests are reviewed within five working days."),
        ("9. Review", "This policy is reviewed every two years. The next review is due in December 2028."),
    ]
    y = 60
    for i, (h, body) in enumerate(secs):
        if body is None:
            P.text(70, y, h, f, 22 if i == 0 else 13, th["fg"], bold=(i == 0))
            y += 34 if i == 0 else 40
            continue
        P.text(70, y, h, f, 16, th["fg"], bold=True)
        y += 28
        for ln in wrap(body, f, px, maxw):
            P.text(70, y, ln, f, px, th["fg"])
            y += px * 1.55
        y += 22
    P.center(P.w / 2, P.h - 50, "Page 1 of 1", f, 11, th["dim"])
    return dict(qa=[("How much can employees claim towards an ergonomic chair and desk?", ["$650", "650"]),
                    ("Within how many days of purchase must receipts be submitted?", ["60"])],
                absent=["What is the monthly internet allowance?"])


def tall_release_notes(P, th):
    f, px = "Segoe UI", 12
    rel = [
        ("v4.18.0 - 2026-11-04", ["Added bulk export of invoices to CSV and XLSX.",
                                  "Added keyboard shortcut Ctrl+Shift+E for export.",
                                  "Fixed a crash when exporting more than 65,535 rows.",
                                  "Minimum supported macOS version is now 13.5."]),
        ("v4.17.2 - 2026-10-21", ["Fixed currency rounding for JPY and KRW.",
                                  "Fixed login loop when the session cookie expired during sync."]),
        ("v4.17.1 - 2026-10-09", ["Security: updated bundled OpenSSL to 3.0.15.",
                                  "Reduced memory use of the dashboard by about 30%."]),
        ("v4.17.0 - 2026-09-30", ["New: recurring invoice templates.",
                                  "New: per-customer payment terms (7, 14, 30 or 60 days).",
                                  "Changed: reports now default to the current fiscal quarter.",
                                  "Removed: legacy PDF engine (deprecated since v4.12)."]),
        ("v4.16.3 - 2026-09-12", ["Fixed the CSV import ignoring the first row when it had no header.",
                                  "Fixed time zone shown on audit log entries."]),
        ("v4.16.2 - 2026-08-28", ["Fixed duplicate reminders being sent for partially paid invoices."]),
        ("v4.16.0 - 2026-08-14", ["New: two-factor authentication with hardware keys.",
                                  "New: dark theme for the web dashboard.",
                                  "Changed: API rate limit raised to 600 requests per minute."]),
    ]
    P.text(32, 28, "Release notes", f, 24, th["fg"], bold=True)
    y = 80
    for h, items in rel:
        P.text(32, y, h, f, 16, th["accent"], bold=True)
        y += 30
        for it in items:
            P.ellipse(40, y + 6, 45, y + 11, fill=th["dim"])
            P.text(56, y, it, f, px, th["fg"])
            y += 22
        y += 22
    return dict(qa=[("In which version was the crash when exporting more than 65,535 rows fixed?",
                     ["v4.18.0", "4.18.0"]),
                    ("What is the new API rate limit per minute?", ["600"])],
                absent=["Which version added support for Linux ARM64?"])


def doc_letter(P, th):
    f, px = "Times New Roman", 12
    P.text(60, 50, "Ashgrove Water Services", f, 20, (20, 60, 120), bold=True)
    P.text(60, 78, "PO Box 4021, Ashgrove AG1 9ZZ", f, px, th["fg"])
    P.right(P.w - 60, 50, "Account number: 7702 1846 33", f, px, th["fg"])
    P.right(P.w - 60, 68, "Date: 6 November 2026", f, px, th["fg"])
    y = 130
    for t in ["The Occupier", "Flat 2, 18 Linden Terrace", "Ashgrove AG4 2PL"]:
        P.text(60, y, t, f, px, th["fg"])
        y += 17
    y += 20
    P.text(60, y, "Your estimated bill for August to October 2026", f, 14, th["fg"], bold=True)
    y += 30
    paras = ["Dear Customer,",
             "We were unable to read your meter on 14 October 2026, so this bill is based on an estimate of your "
             "usage. Your previous actual reading was 04821 cubic metres, taken on 12 July 2026.",
             "The estimated charge for this period is 96.40 GBP, made up of a standing charge of 21.15 GBP and "
             "usage charges of 75.25 GBP. This amount will be collected by Direct Debit on or after 20 November "
             "2026.",
             "If you would like a bill based on your actual usage, please submit a reading through your online "
             "account before 18 November 2026 and we will issue a revised bill.",
             "Yours faithfully,", "Customer Billing Team"]
    for p in paras:
        for l in wrap(p, f, px, P.w - 120):
            P.text(60, y, l, f, px, th["fg"])
            y += px * 1.5
        y += 10
    return dict(qa=[("What is the estimated charge for the period?", ["96.40 GBP", "96.40", "£96.40"]),
                    ("What was the previous actual meter reading?", ["04821", "4821"])],
                absent=["What is the customer's phone number?"])


def doc_memo(P, th):
    f, px = "Calibri", 16
    P.rect(0, 0, P.w, 8, fill=(46, 117, 182))
    P.text(50, 34, "MEMO", f, 28, (46, 117, 182), bold=True)
    for i, (k, v) in enumerate([("To:", "All warehouse shift leads"), ("From:", "Operations Planning"),
                                ("Date:", "10 November 2026"), ("Re:", "Peak season staffing")]):
        P.text(50, 84 + i * 26, k, f, px, th["fg"], bold=True)
        P.text(130, 84 + i * 26, v, f, px, th["fg"], nl=False)
    P.ln(50, 196, P.w - 50, 196, th["line"])
    y = 214
    paras = ["From 24 November to 4 January the warehouse will run three shifts, seven days a week. Each shift "
             "needs a minimum of 38 pickers and 9 forklift operators.",
             "Overtime is capped at 12 hours per person per week. Requests above the cap must be approved by the "
             "site manager by Thursday noon for the following week.",
             "Agency staff will start induction on 17 November in Training Room 3. Please confirm your shift "
             "rosters in the planning sheet by 14 November."]
    for p in paras:
        for l in wrap(p, f, px, P.w - 100):
            P.text(50, y, l, f, px, th["fg"])
            y += px * 1.45
        y += 12
    return dict(qa=[("What is the minimum number of pickers per shift?", ["38"]),
                    ("In which room will agency staff induction take place?", ["Training Room 3", "Room 3", "3"])],
                absent=["What is the hourly overtime rate?"])


def small_props(P, th):
    f, px = "Segoe UI", 9
    P.rect(0, 0, P.w, 22, fill=th["panel"])
    P.text(8, 5, "quarterly_export_2026Q3.parquet Properties", f, px, th["fg"])
    tabs = ["General", "Security", "Details", "Previous Versions"]
    x = 8
    for i, t in enumerate(tabs):
        P.text(x, 30, t, f, px, th["fg"] if i == 0 else th["dim"], nl=(i == 0), sep="  ")
        x += P.tw(t, f, px) + 18
    P.ln(8, 46, P.w - 8, 46, th["line"])
    rows = [("Type of file:", "PARQUET File (.parquet)"), ("Location:", "E:\\exports\\finance\\2026"),
            ("Size:", "14.2 MB (14,893,056 bytes)"), ("Size on disk:", "14.2 MB (14,897,152 bytes)"),
            ("Created:", "Friday, 12 September 2026, 08:41:07"),
            ("Modified:", "Monday, 3 November 2026, 17:20:55"), ("Accessed:", "Tuesday, 11 November 2026, 09:02:13"),
            ("Attributes:", "Read-only, Archive"),
            ("SHA-256:", "9f2c41d07a8be3156c0d2e94b7a1f683c5e02d4b9a7716e3f08c2d5a6b1e9047")]
    y = 56
    for k, v in rows:
        P.text(12, y, k, f, px, th["fg"])
        P.text(100, y, v, f, px, th["fg"], nl=False)
        y += 19
    x = P.w - 200
    for i, b in enumerate(["OK", "Cancel", "Apply"]):
        button(P, x + i * 64, P.h - 32, b, f, px, th, nl=(i == 0))
    return dict(qa=[("What is the exact size of the file in bytes (not size on disk)?",
                     ["14,893,056", "14893056", "14,893,056 bytes"]),
                    ("On what date was the file created?",
                     ["12 September 2026", "Friday, 12 September 2026", "2026-09-12"])],
                absent=["Who is the owner of the file?"])


def small_footnote(P, th):
    f, px = "Times New Roman", 10
    P.text(30, 20, "Table 3. Response times by region (median, ms)", f, 12, th["fg"], bold=True)
    rows = [["North", "182", "201", "1,204"], ["South", "167", "189", "988"], ["Central", "143", "150", "2,317"],
            ["Coastal", "211", "246", "604"], ["Overall", "171", "187", "5,113"]]
    cols = [("Region", 110), ("2025", 70), ("2026", 70), ("n", 70)]
    y = table(P, 30, 44, cols, rows, f, px, th, row_h=18, aligns="lrrr", grid=False)
    P.ln(30, 44, 350, 44, th["fg"])
    P.ln(30, 62, 350, 62, th["fg"])
    P.ln(30, y, 350, y, th["fg"])
    y += 12
    notes = ["1 Medians computed over all requests between 1 January and 30 June of each year.",
             "2 Requests from the Coastal region were routed through a backup link for 11 days in March 2026.",
             "3 n is the number of sampled sessions in 2026; sampling rate was 2 percent.",
             "Source: Operations telemetry, extract of 4 August 2026."]
    for t in notes:
        for l in wrap(t, f, px, P.w - 60):
            P.text(30, y, l, f, px, th["fg"])
            y += px * 1.45
        y += 2
    return dict(qa=[("What is the 2026 median response time for the Coastal region?", ["246"]),
                    ("For how many days were Coastal requests routed through a backup link?", ["11"])],
                absent=["What is the 2024 median response time for the North region?"])


def small_agenda(P, th):
    f, px = "Calibri", 9
    P.text(12, 10, "Tuesday 18 November 2026", f, 12, th["fg"], bold=True)
    ev = [("08:30", "09:00", "Inbox triage", ""), ("09:30", "09:45", "Standup", "Teams"),
          ("10:00", "11:00", "Vendor review: Kestrel Media", "Room Alder 2.14"),
          ("11:30", "12:00", "1:1 with Lena", "Room Birch"), ("13:00", "14:30", "Budget planning FY27", "Room Cedar 4.02"),
          ("15:00", "15:30", "Interview - backend engineer", "Teams"),
          ("16:00", "16:45", "Release 4.18 go/no-go", "Room Birch"), ("17:15", "17:30", "Pick up parcel", "Locker 6")]
    y = 36
    for a, b, t, loc in ev:
        P.rect(12, y - 2, 15, y + 22, fill=th["accent"])
        P.text(22, y, f"{a} - {b}", f, px, th["dim"])
        P.text(100, y, t, f, px, th["fg"], nl=False)
        if loc:
            P.text(100, y + 11, loc, f, px, th["dim"])
        y += 30
    P.text(12, y + 4, "8 events  |  Free after 17:30", f, px, th["dim"])
    return dict(qa=[("In which room is the Budget planning FY27 meeting?", ["Room Cedar 4.02", "Cedar 4.02"]),
                    ("What time does the Release 4.18 go/no-go start?", ["16:00"])],
                absent=["What time is the dentist appointment?"])


# ---------------------------------------------------------------- page registry
# (id, builder, width, height, theme, category, family, font_px, hidpi_variant)
PAGES = [
    ("term_cargo", terminal_cargo, 760, 240, "dark", "terminal", "Consolas", 12, False),
    ("term_npm", terminal_npm, 470, 220, "dark", "terminal", "Consolas", 10, False),
    ("term_git", terminal_git, 640, 300, "light", "terminal", "Consolas", 14, False),
    ("term_pytest", terminal_pytest, 470, 160, "dark", "terminal", "Consolas", 9, False),
    ("buildlog_ci", buildlog_ci, 700, 230, "dark", "build_log", "Consolas", 11, False),
    ("code_rust", code_rust, 820, 440, "dark", "code", "Consolas", 12, False),
    ("code_ts", code_ts, 860, 300, "light", "code", "Consolas", 14, True),
    ("code_py", code_py, 600, 330, "dark", "code", "Consolas", 10, False),
    ("code_sql", code_sql, 640, 330, "light", "code", "Consolas", 16, False),
    ("dialog_sync", dialog_error_win, 760, 400, "light", "dialog", "Segoe UI", 12, True),
    ("dialog_debug", dialog_error_dark, 700, 370, "dark", "dialog", "Segoe UI", 11, False),
    ("dialog_delete", dialog_confirm, 820, 410, "light", "dialog", "Arial", 14, False),
    ("settings_storage", settings_app, 760, 480, "light", "settings", "Segoe UI", 12, True),
    ("settings_notif", settings_dark, 640, 420, "dark", "settings", "Segoe UI", 11, False),
    ("settings_network", settings_network, 680, 520, "light", "settings", "Calibri", 14, False),
    ("sheet_sales", sheet_sales, 560, 440, "light", "table", "Calibri", 11, True),
    ("sheet_inventory", sheet_inventory, 650, 440, "light", "table", "Arial", 10, False),
    ("sheet_fleet", sheet_dark, 720, 390, "dark", "table", "Segoe UI", 12, False),
    ("table_fx", table_fx, 450, 270, "light", "small_font", "Arial", 9, False),
    ("receipt_cafe", receipt_cafe, 300, 500, "paper", "receipt", "Consolas", 12, False),
    ("receipt_hardware", receipt_hardware, 280, 400, "paper", "receipt", "Arial", 10, False),
    ("invoice_lumen", invoice, 820, 620, "light", "receipt", "Arial", 11, False),
    ("chat_team", chat_team, 760, 400, "light", "chat", "Segoe UI", 12, False),
    ("chat_phone", chat_dark, 420, 560, "dark", "chat", "Segoe UI", 14, False),
    ("banner_wide", banner_wide, 2400, 400, "light", "wide", "Arial", 40, False),
    ("wide_statusbar", wide_statusbar, 2400, 240, "dark", "wide", "Segoe UI", 12, False),
    ("tall_policy", tall_doc, 820, 1500, "light", "tall", "Times New Roman", 14, False),
    ("tall_release", tall_release_notes, 560, 1000, "dark", "tall", "Segoe UI", 12, False),
    ("doc_letter", doc_letter, 680, 620, "light", "doc", "Times New Roman", 12, False),
    ("doc_memo", doc_memo, 820, 520, "light", "doc", "Calibri", 16, False),
    ("small_props", small_props, 460, 290, "light", "small_font", "Segoe UI", 9, False),
    ("small_footnote", small_footnote, 520, 230, "light", "small_font", "Times New Roman", 10, False),
    ("small_agenda", small_agenda, 300, 300, "dark", "small_font", "Calibri", 9, False),
]


def theme_dict(name):
    if name == "dark":
        return DARK
    if name == "paper":
        return dict(LIGHT, bg=(250, 248, 240))
    return LIGHT


def add(id_, img, category, kind, prompt, meta, answers=None, text=None):
    it = {"id": id_, "image": img, "source": "gen", "category": category, "kind": kind, "prompt": prompt,
          "answers": answers or []}
    if text is not None:
        it["text"] = text
    it["meta"] = meta
    ITEMS.append(it)


def render(pid, fn, w, h, theme, scale):
    th = theme_dict(theme)
    P = Page(w, h, th["bg"], scale)
    spec = fn(P, th)
    path = os.path.join(OUT, f"{pid}{'_2x' if scale == 2 else ''}.png")
    P.im.save(path, optimize=True)
    return P, spec, path


for i, (pid, fn, w, h, theme, cat, fam, px, hidpi) in enumerate(PAGES):
    P, spec, path = render(pid, fn, w, h, theme, 1)
    text = "\n".join(P.lines)
    meta = dict(font_px=px, family=fam, theme=theme, hidpi=1, width=P.im.width, height=P.im.height, page=pid)
    add(f"gen_{pid}_tr", path, cat, "transcribe", TRANSCRIBE, meta, text=text)
    qa = spec["qa"] if i % 6 != 5 else spec["qa"][:1]  # trim some second QAs to keep the set near 130
    for j, (q, a) in enumerate(qa):
        add(f"gen_{pid}_qa{j + 1}", path, cat, "qa", q, meta, answers=a)
    if i % 3 != 2:
        add(f"gen_{pid}_absent", path, cat, "absent", spec["absent"][0] + ABSENT_SUFFIX, meta,
            answers=["not shown", "no"])
    if hidpi:
        P2, spec2, path2 = render(pid, fn, w, h, theme, 2)
        assert P2.lines == P.lines, f"2x text differs for {pid}"
        meta2 = dict(meta, hidpi=2, width=P2.im.width, height=P2.im.height, base_page=pid,
                     content_category=cat)
        add(f"gen_{pid}_2x_tr", path2, "hidpi", "transcribe", TRANSCRIBE, meta2, text=text)
        q, a = spec2["qa"][0]
        add(f"gen_{pid}_2x_qa1", path2, "hidpi", "qa", q, meta2, answers=a)


# ---------------------------------------------------------------- photo-like EXIF items
def photo(pid, w, h, wall, draw_sign, angle, orient, seed):
    rng = random.Random(seed)
    base = Image.new("RGB", (w, h), wall)
    d = ImageDraw.Draw(base)
    for y in range(0, h, 3):  # lighting gradient
        k = 1.0 - 0.25 * y / h
        d.rectangle([0, y, w, y + 3], fill=tuple(int(c * k) for c in wall))
    sign, lines, qa, absent = draw_sign()
    mask = Image.new("L", sign.size, 255).rotate(angle, resample=Image.BICUBIC, expand=True)
    sign = sign.rotate(angle, resample=Image.BICUBIC, expand=True)
    ox, oy = (w - sign.size[0]) // 2 + rng.randint(-30, 30), (h - sign.size[1]) // 2 + rng.randint(-20, 20)
    shadow = Image.new("RGB", sign.size, (0, 0, 0))
    base.paste(Image.blend(base.crop((ox + 8, oy + 10, ox + 8 + sign.size[0], oy + 10 + sign.size[1])), shadow, 0.35),
               (ox + 8, oy + 10), mask)
    base.paste(sign, (ox, oy), mask)
    noise = Image.effect_noise((w, h), 22).convert("RGB")
    base = ImageChops.add(base, noise, scale=1.0, offset=-128)
    base = base.filter(ImageFilter.GaussianBlur(0.9))
    upright = base
    # store the pixels rotated so that a viewer applying the EXIF orientation sees the upright image
    if orient == 6:
        stored = upright.transpose(Image.ROTATE_90)   # Orientation 6: viewer rotates 90 CW
    elif orient == 8:
        stored = upright.transpose(Image.ROTATE_270)  # Orientation 8: viewer rotates 90 CCW
    elif orient == 3:
        stored = upright.transpose(Image.ROTATE_180)
    else:
        stored = upright
    exif = Image.Exif()
    exif[0x0112] = orient
    exif[0x010F] = "EvalCam"
    exif[0x0110] = "Synthetic 1"
    path = os.path.join(OUT, pid + ".jpg")
    stored.save(path, "JPEG", quality=88, exif=exif.tobytes())
    meta = dict(font_px=None, family="mixed", theme="photo", hidpi=1, width=stored.width, height=stored.height,
                upright_width=w, upright_height=h, exif_orientation=orient, page=pid)
    return path, "\n".join(lines), qa, absent, meta


def sign_platform():
    im = Image.new("RGB", (760, 380), (20, 50, 120))
    d = ImageDraw.Draw(im)
    d.rectangle([12, 12, 748, 368], outline=(240, 240, 240), width=4)
    lines = ["PLATFORM 4", "Northgate Express", "Departs 14:25", "Calling at: Hollow Park, Eastmere, Northgate"]
    d.text((40, 30), lines[0], font=F("Arial", 84, True), fill=(255, 255, 255))
    d.text((40, 150), lines[1], font=F("Arial", 54), fill=(255, 210, 60))
    d.text((40, 225), lines[2], font=F("Arial", 54, True), fill=(255, 255, 255))
    d.text((40, 310), lines[3], font=F("Arial", 28), fill=(220, 220, 220))
    return im, lines, [("What time does the Northgate Express depart?", ["14:25"]),
                       ("What platform number is shown?", ["4"])], "What is the train's carriage count?"


def sign_bin():
    im = Image.new("RGB", (640, 420), (250, 215, 40))
    d = ImageDraw.Draw(im)
    lines = ["BIN C-17-04", "PART 88-2140-B", "HINGE PIN 6x40 ZINC", "QTY 250   REORDER AT 60"]
    d.text((30, 24), lines[0], font=F("Arial", 72, True), fill=(10, 10, 10))
    d.text((30, 120), lines[1], font=F("Consolas", 48, True), fill=(10, 10, 10))
    d.text((30, 190), lines[2], font=F("Arial", 36), fill=(10, 10, 10))
    d.text((30, 245), lines[3], font=F("Arial", 36), fill=(10, 10, 10))
    rng = random.Random(7)
    x = 30
    while x < 600:  # barcode-like stripes (no text)
        bw = rng.choice([2, 3, 5])
        d.rectangle([x, 310, x + bw, 395], fill=(10, 10, 10))
        x += bw + rng.choice([2, 3, 4])
    return im, lines, [("What is the part number on the label?", ["88-2140-B"]),
                       ("At what quantity should it be reordered?", ["60"])], "What is the supplier name on the label?"


def sign_parking():
    im = Image.new("RGB", (520, 640), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([8, 8, 512, 632], outline=(200, 30, 30), width=10)
    d.ellipse([190, 30, 330, 170], fill=(20, 70, 160))
    d.text((260, 100), "P", font=F("Arial", 100, True), fill=(255, 255, 255), anchor="mm")
    lines = ["PERMIT HOLDERS ONLY", "ZONE K", "Mon - Sat", "8am - 6:30pm", "Max fine 70"]
    y = 200
    for t, s, b in zip(lines, [40, 64, 44, 44, 32], [True, True, False, False, False]):
        d.text((260, y), t, font=F("Arial", s, b), fill=(10, 10, 10), anchor="ma")
        y += s + 34
    lines = ["P"] + lines  # the "P" in the roundel is visible text too
    return im, lines, [("Which zone is shown on the parking sign?", ["K", "Zone K"]),
                       ("Until what time do the restrictions apply?", ["6:30pm", "6:30 pm", "18:30"])], \
        "What is the maximum stay for visitors?"


def sign_whiteboard():
    im = Image.new("RGB", (820, 520), (246, 247, 244))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 819, 519], outline=(160, 160, 165), width=12)
    lines = ["Sprint 42 goals", "1. Ship CSV export (Mira)", "2. Fix login loop - #2310", "3. Load test: 600 rpm",
             "Demo: Fri 2:30pm"]
    cols = [(30, 90, 200), (20, 20, 20), (20, 20, 20), (20, 20, 20), (200, 40, 40)]
    y = 30
    for t, c, s in zip(lines, cols, [52, 40, 40, 40, 44]):
        d.text((40, y), t, font=F("Segoe Print", s), fill=c)
        y += int(s * 1.9)
    return im, lines, [("What issue number is the login loop fix?", ["2310", "#2310"]),
                       ("When is the demo?", ["Fri 2:30pm", "Friday 2:30pm", "Fri 2:30 pm"])], \
        "Who is assigned to the load test?"


PHOTOS = [("photo_platform", 1200, 800, (150, 140, 130), sign_platform, 3.0, 6, 11),
          ("photo_bin", 1000, 760, (120, 125, 135), sign_bin, -2.0, 8, 12),
          ("photo_parking", 900, 1100, (110, 120, 105), sign_parking, 1.5, 3, 13),
          ("photo_whiteboard", 1200, 860, (205, 200, 190), sign_whiteboard, -1.0, 6, 14)]
for k, (pid, w, h, wall, fn, ang, orient, seed) in enumerate(PHOTOS):
    path, text, qa, absent, meta = photo(pid, w, h, wall, fn, ang, orient, seed)
    add(f"gen_{pid}_tr", path, "photo_exif", "transcribe", TRANSCRIBE, meta, text=text)
    add(f"gen_{pid}_qa1", path, "photo_exif", "qa", qa[0][0], meta, answers=qa[0][1])
    if k % 2 == 0:
        add(f"gen_{pid}_qa2", path, "photo_exif", "qa", qa[1][0], meta, answers=qa[1][1])
    else:
        add(f"gen_{pid}_absent", path, "photo_exif", "absent", absent + ABSENT_SUFFIX, meta,
            answers=["not shown", "no"])

with open(os.path.join(OUT, "gen_items.jsonl"), "w") as f:
    for it in ITEMS:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(len(ITEMS), "gen items ->", OUT)
