//! Text read from the pictures in a PDF page: macOS Vision, Windows.Media.Ocr, or tesseract.

use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        LazyLock, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use super::table::{self, Word};
use crate::{skills::read_papers::output_until, tool::no_window};

/// Bumped whenever a script, `DPI` or the parsing changes, so no cache entry of an older OCR is
/// read; a source record names it with the text it read.
pub const VERSION: u32 = 1;

/// The resolution a page is rendered at for OCR: tesseract reads best at 300 dpi.
// ponytail: one guess for all three engines; the Mac sets it from Vision's time and quality at
// 200 and 300 dpi on the scanned contract.
const DPI: f64 = 300.0;

/// The longest one engine or renderer runs on one page; Vision's first page loads its model too.
pub(super) const LIMIT: Duration = Duration::from_secs(60);

/// A page with fewer letters and digits than this is a scan, or a slide that is one picture
/// under a title: its text layer is too thin to trust.
// ponytail: MIN_CHARS and MIN_SHARE are guesses; the Mac sets them from the scanned contract,
// the picture charts and the decks of test-cases.md, at the point where the text layer stops
// holding what the page shows.
const MIN_CHARS: usize = 100;

/// A page whose letters and digits are less than this share of its characters, ASCII punctuation
/// and the ellipsis aside, is garbage: a font with no Unicode map gives symbols and private-use
/// characters, while a contents page's dot leaders are punctuation. Digits count, as a table of
/// numbers is a good text layer.
const MIN_SHARE: f64 = 0.5;

/// Each page, by its PDF's SHA-256 and its index, that OCR failed on in this process, with why:
/// each try can take `LIMIT`, and every reopen would try again. A later process tries again.
static FAILED: LazyLock<Mutex<HashMap<(String, usize), String>>> = LazyLock::new(Mutex::default);

/// What OCR off macOS and Windows needs, said when it is missing.
pub(crate) const UNAVAILABLE: &str =
    "OCR is not available on this machine: it needs poppler's pdftoppm and tesseract installed";

/// Text OCR read from the picture of one page: its own representation (`Rep::Ocr`), with each
/// line's box.
#[derive(Clone, Debug, PartialEq)]
pub struct Ocr {
    /// The page index, from 0.
    pub page: usize,
    /// Its lines in reading order, each one `Word` with its box in points from the page's
    /// top-left corner.
    pub lines: Vec<Word>,
}

impl Ocr {
    /// The lines' text, one a line.
    pub fn text(&self) -> String {
        let lines: Vec<&str> = self.lines.iter().map(|l| l.text.as_str()).collect();
        lines.join("\n")
    }

    /// The text as the reader is shown it, marked as read from a picture: OCR gives characters,
    /// not what the picture shows.
    pub fn marked(&self) -> String {
        format!("{}\n{}", mark(self.page), self.text())
    }
}

/// The line that starts the text read from the pictures on page `page` (from 0).
pub fn mark(page: usize) -> String {
    format!("[text read from a picture on page {}]", page + 1)
}

/// Whether a page's text layer, `page`, is too thin to trust, so its picture should be read:
/// too few letters and digits, or too small a share of them.
pub fn needs_ocr(page: &str) -> bool {
    let (mut chars, mut letters) = (0, 0);
    let counted = |c: &char| !(c.is_whitespace() || c.is_ascii_punctuation() || *c == '\u{2026}');
    for c in page.chars().filter(counted) {
        chars += 1;
        letters += usize::from(c.is_alphanumeric());
    }
    letters < MIN_CHARS || (letters as f64) < MIN_SHARE * chars as f64
}

/// Whether this machine can read a page's picture: macOS and Windows always can, any other OS
/// when pdftoppm and tesseract are installed.
pub fn available() -> bool {
    cfg!(any(target_os = "macos", windows)) || (installed("pdftoppm") && installed("tesseract"))
}

/// The text OCR reads from the picture of page `page` (from 0) of `pdf`, whose bytes hash to
/// `sha256`: from `<data_dir>/agent/extract/` when this OCR version read the page before, else by
/// this OS's engine, then kept there; with no `pdf` (its original is not there as it was read),
/// from there alone. `stop` ends the engine. A page the engine failed on is not tried again in
/// this process, unless `stop` ended it.
pub fn read(
    pdf: Option<&Path>,
    sha256: &str,
    page: usize,
    data_dir: &Path,
    stop: &AtomicBool,
) -> Result<Ocr, String> {
    // The hash names a file: anything but hex could name one outside the cache.
    if sha256.is_empty() || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("not a SHA-256: {sha256:?}"));
    }
    let dir = data_dir.join("agent/extract");
    let cached = dir.join(format!("{sha256}.ocr{VERSION}.p{page}.tsv"));
    if let Some(lines) = (fs::read_to_string(&cached).ok()).and_then(|t| table::boxes(&t).ok()) {
        return Ok(Ocr { page, lines });
    }
    let Some(pdf) = pdf else {
        return Err(format!(
            "page {} was not read from its pictures before, and the original is not there as it \
             was read",
            page + 1
        ));
    };
    let failed = || {
        FAILED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    };
    let key = (sha256.to_owned(), page);
    if let Some(e) = failed().get(&key) {
        return Err(e.clone());
    }
    let lines = engine(pdf, page, stop).map_err(|e| {
        let e = format!("OCR of page {}: {e}", page + 1);
        if !stop.load(Ordering::Relaxed) {
            failed().insert(key, e.clone());
        }
        e
    })?;
    // A cache that cannot be written only costs the next read another OCR; written whole or not
    // at all, so a crash leaves no cut lines behind.
    let _ = fs::create_dir_all(&dir)
        .and_then(|()| crate::write_atomic(&cached, &table::tsv(&lines), true));
    Ok(Ocr { page, lines })
}

/// Whether `program` is a file in a folder on this machine's PATH.
pub(super) fn installed(program: &str) -> bool {
    let name = format!("{program}{}", env::consts::EXE_SUFFIX);
    env::var_os("PATH").is_some_and(|path| env::split_paths(&path).any(|d| d.join(&name).is_file()))
}

/// Whether the `pdftotext` on PATH is poppler's, which the readers here are written for, as its
/// `-v` says. Another program of that name (the Windows CI runner's) joined a page's lines into
/// one, so no page label was found, and has no `-bbox-layout`. Asked once in a process.
pub(crate) fn poppler() -> bool {
    static POPPLER: LazyLock<bool> = LazyLock::new(|| {
        let mut cmd = Command::new("pdftotext");
        no_window(&mut cmd).arg("-v");
        cmd.output().is_ok_and(|out| {
            let said = [out.stdout, out.stderr].concat();
            String::from_utf8_lossy(&said).contains("Poppler")
        })
    });
    *POPPLER
}

/// Page `page`'s lines by this OS's engine. Every branch is compiled on every OS, so each is
/// checked wherever the crate builds.
// ponytail: the file is an absolute path from every caller, so no "-" can start it.
fn engine(pdf: &Path, page: usize, stop: &AtomicBool) -> Result<Vec<Word>, String> {
    let out = if cfg!(target_os = "macos") {
        return vision(pdf, page, "lines", stop);
    } else if cfg!(windows) {
        // In the environment, as -Command takes the rest of its line as script text. Its
        // StorageFile.GetFileFromPathAsync refuses a `/`, which a path joined from "a/b" keeps
        // (a link's blob, under "agent/docs/blobs"): out of its verbatim form, where a `/` is
        // no separator, the components rejoin with `\`.
        let pdf: PathBuf = crate::tool::plain(pdf.to_path_buf()).components().collect();
        let mut cmd = Command::new("powershell.exe");
        no_window(&mut cmd)
            .args(["-NoProfile", "-NonInteractive", "-Command", WINDOWS])
            .env("LARCH_OCR_PDF", pdf)
            .env("LARCH_OCR_PAGE", page.to_string())
            .env("LARCH_OCR_SCALE", (DPI / 96.0).to_string());
        output_until(cmd, LIMIT, stop)?
    } else {
        return tesseract(pdf, page, stop);
    };
    table::boxes(&String::from_utf8_lossy(&out))
}

/// Page `page`'s `"lines"` or `"words"` with their boxes, read by macOS Vision (`VISION`).
pub(super) fn vision(
    pdf: &Path,
    page: usize,
    unit: &str,
    stop: &AtomicBool,
) -> Result<Vec<Word>, String> {
    let mut cmd = Command::new("osascript");
    cmd.args(["-l", "JavaScript", "-e", VISION])
        .arg(pdf)
        .arg(page.to_string())
        .arg((DPI / 72.0).to_string())
        .arg(unit);
    table::boxes(&String::from_utf8_lossy(&output_until(cmd, LIMIT, stop)?))
}

/// Page `page` rendered by poppler's pdftoppm and read by tesseract.
fn tesseract(pdf: &Path, page: usize, stop: &AtomicBool) -> Result<Vec<Word>, String> {
    if !(installed("pdftoppm") && installed("tesseract")) {
        return Err(UNAVAILABLE.into());
    }
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let png = dir.path().join("page");
    let (n, dpi) = ((page + 1).to_string(), DPI.to_string());
    let mut render = Command::new("pdftoppm");
    render
        .args([
            "-f",
            &n,
            "-l",
            &n,
            "-r",
            &dpi,
            "-gray",
            "-png",
            "-singlefile",
        ])
        .arg(pdf)
        .arg(&png);
    output_until(render, LIMIT, stop).map_err(|e| format!("pdftoppm: {e}"))?;
    let mut read = Command::new("tesseract");
    // One thread: with OpenMP's, four pages at once on 4 busy cores each ran past 85 s, against
    // 1.4 s for all four on one thread each (13 Nov 2026); the engine keeps cores busy.
    read.arg(png.with_extension("png"))
        .args(["stdout", "tsv"])
        .env("OMP_THREAD_LIMIT", "1");
    let tsv = output_until(read, LIMIT, stop).map_err(|e| format!("tesseract: {e}"))?;
    tesseract_lines(&String::from_utf8_lossy(&tsv), 72.0 / DPI)
}

/// The lines of tesseract's TSV, in its reading order, their boxes scaled by `k` from pixels to
/// points: a line is the words (level 5) of one block, paragraph and line number.
fn tesseract_lines(tsv: &str, k: f64) -> Result<Vec<Word>, String> {
    let mut lines: Vec<([&str; 3], Word)> = Vec::new();
    for row in tsv.lines().skip(1).filter(|r| r.starts_with("5\t")) {
        let f: Vec<&str> = row.split('\t').collect();
        let [_, _, block, par, line, _, left, top, width, height, _, text] = f[..] else {
            return Err(format!("a tesseract row is not 12 fields: {row:?}"));
        };
        let n: Vec<f64> = ([left, top, width, height].iter())
            .map(|v| v.parse::<f64>().ok())
            .collect::<Option<_>>()
            .ok_or_else(|| format!("a tesseract box is not four numbers: {row:?}"))?;
        let (text, key) = (text.trim(), [block, par, line]);
        if text.is_empty() {
            continue;
        }
        let (x0, y0, x1, y1) = (n[0] * k, n[1] * k, (n[0] + n[2]) * k, (n[1] + n[3]) * k);
        match lines.last_mut() {
            Some((at, w)) if *at == key => {
                w.text.push(' ');
                w.text.push_str(text);
                (w.x0, w.y0, w.x1, w.y1) = (w.x0.min(x0), w.y0.min(y0), w.x1.max(x1), w.y1.max(y1));
            }
            _ => lines.push((
                key,
                Word {
                    text: text.into(),
                    x0,
                    y0,
                    x1,
                    y1,
                },
            )),
        }
    }
    Ok(lines.into_iter().map(|(_, w)| w).collect())
}

/// The JavaScript for Automation script that reads one page's picture by macOS Vision, as
/// `table::boxes` reads it: argv is the file, the page index from 0, the scale from points to
/// pixels, and "words" for each line's words in place of the lines. PDFKit renders the page as it
/// is shown (turned when it is turned) into a bitmap of exact size, white under it; Vision's
/// accurate text request, with language correction, gives lines in its reading order, each box
/// turned from Vision's share of the image from its bottom-left into points from the page's
/// top-left. Table words come from here too: PDFKit's `characterBoundsAtIndex` falls out of step
/// with `page.string` on datasheets and 10-Ks, while the picture is what the page shows.
const VISION: &str = r#"ObjC.import("AppKit");
ObjC.import("PDFKit");
ObjC.import("Vision");
function run(argv) {
  var d = $.PDFDocument.alloc.initWithURL($.NSURL.fileURLWithPath(argv[0]));
  if (d.isNil()) { throw new Error("PDFKit could not open the file"); }
  var n = Number(argv[1]), scale = Number(argv[2]);
  if (!(n >= 0 && n < d.pageCount)) { throw new Error("the file has no page " + argv[1]); }
  var page = d.pageAtIndex(n);
  // 0 is kPDFDisplayBoxMediaBox, the box pdftotext measures in.
  var size = page.boundsForBox(0).size, w = size.width, h = size.height;
  if (page.rotation % 180 != 0) { var t = w; w = h; h = t; }
  var pw = Math.round(w * scale), ph = Math.round(h * scale);
  var rect = { origin: { x: 0, y: 0 }, size: { width: pw, height: ph } };
  // An image's own backing doubles on a Retina screen: the bitmap fixes the pixels.
  var bitmap = $.NSBitmapImageRep.alloc.initWithBitmapDataPlanesPixelsWidePixelsHighBitsPerSampleSamplesPerPixelHasAlphaIsPlanarColorSpaceNameBytesPerRowBitsPerPixel(null, pw, ph, 8, 4, true, false, "NSDeviceRGBColorSpace", 0, 0);
  $.NSGraphicsContext.saveGraphicsState;
  $.NSGraphicsContext.setCurrentContext($.NSGraphicsContext.graphicsContextWithBitmapImageRep(bitmap));
  $.NSColor.whiteColor.setFill;
  $.NSBezierPath.fillRect(rect);
  page.thumbnailOfSizeForBox(rect.size, 0).drawInRect(rect);
  $.NSGraphicsContext.restoreGraphicsState;
  var request = $.VNRecognizeTextRequest.alloc.init;
  request.recognitionLevel = 0; // VNRequestTextRecognitionLevelAccurate
  request.usesLanguageCorrection = true;
  var handler = $.VNImageRequestHandler.alloc.initWithDataOptions(bitmap.TIFFRepresentation, $.NSDictionary.dictionary);
  if (!handler.performRequestsError($.NSArray.arrayWithObject(request), Ref())) {
    throw new Error("Vision could not read the page");
  }
  var out = [], results = request.results, words = argv[3] == "words";
  function points(b) {
    var x = b.origin.x, y = b.origin.y;
    return [x * w, (1 - y - b.size.height) * h, (x + b.size.width) * w, (1 - y) * h];
  }
  function emit(box, text) {
    out.push(box.map(function (v) { return v.toFixed(2); }).join("\t") + "\t" + text);
  }
  var count = results.isNil() ? 0 : results.count;
  for (var i = 0; i < count; i++) {
    var line = results.objectAtIndex(i), best = line.topCandidates(1);
    if (best.count == 0) { continue; }
    var top = best.objectAtIndex(0), text = top.string.js, box = points(line.boundingBox);
    if (!words) { emit(box, text.replace(/\s+/g, " ")); continue; }
    var each = (box[2] - box[0]) / text.length;
    // NSString and JavaScript index the same UTF-16 units.
    for (var re = /\S+/g, m; (m = re.exec(text)); ) {
      var r = top.boundingBoxForRangeError({ location: m.index, length: m[0].length }, null);
      // With no box of its own, a word gets its share of the line's characters along the line's
      // box: close in a fixed-width font, a few points off in a proportional one.
      emit(r.isNil() ? [box[0] + each * m.index, box[1], box[0] + each * re.lastIndex, box[3]] : points(r.boundingBox), m[0]);
    }
  }
  return out.join("\n");
}"#;

/// The Windows PowerShell 5.1 script (WinRT is not projected into PowerShell 7) that reads one
/// page's picture by Windows.Media.Ocr, as `table::boxes` reads it: the file, the page index
/// from 0 and the scale from DIPs to pixels come in LARCH_OCR_PDF, LARCH_OCR_PAGE and
/// LARCH_OCR_SCALE. Windows.Data.Pdf renders the page, at most the engine's largest image a
/// side; a line's box is the union of its words', turned from pixels into points. Single quotes
/// only, so the argument needs no escaping; output UTF-8 and invariant numbers, whatever the
/// console's code page and the user's locale.
const WINDOWS: &str = r#"$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 }
$ofResult = $asTask | Where-Object { $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' }
$ofAction = $asTask | Where-Object { $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction' }
function Await($operation, $type) {
  $task = $ofResult.MakeGenericMethod($type).Invoke($null, @($operation))
  $task.Wait() | Out-Null
  $task.Result
}
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($env:LARCH_OCR_PDF)) ([Windows.Storage.StorageFile])
$pdf = Await ([Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($file)) ([Windows.Data.Pdf.PdfDocument])
$index = [uint32]$env:LARCH_OCR_PAGE
if ($index -ge $pdf.PageCount) { throw 'the file has no such page' }
$page = $pdf.GetPage($index)
$width = $page.Size.Width
$height = $page.Size.Height
$scale = [Math]::Min([double]$env:LARCH_OCR_SCALE, [Windows.Media.Ocr.OcrEngine]::MaxImageDimension / [Math]::Max($width, $height))
$options = New-Object Windows.Data.Pdf.PdfPageRenderOptions
$options.DestinationWidth = [uint32]($width * $scale)
$options.DestinationHeight = [uint32]($height * $scale)
$stream = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
$ofAction.Invoke($null, @($page.RenderToStreamAsync($stream, $options))).Wait() | Out-Null
$stream.Seek(0)
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) { throw 'no OCR language is installed' }
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$points = 0.75 / $scale
foreach ($line in $result.Lines) {
  $x0 = [double]::MaxValue; $y0 = [double]::MaxValue; $x1 = 0.0; $y1 = 0.0
  foreach ($word in $line.Words) {
    $r = $word.BoundingRect
    $x0 = [Math]::Min($x0, $r.X); $y0 = [Math]::Min($y0, $r.Y)
    $x1 = [Math]::Max($x1, $r.X + $r.Width); $y1 = [Math]::Max($y1, $r.Y + $r.Height)
  }
  $text = $line.Text -replace '\s+', ' '
  [Console]::Out.Write([string]::Format($invariant, '{0:F2}{5}{1:F2}{5}{2:F2}{5}{3:F2}{5}{4}{6}', ($x0 * $points), ($y0 * $points), ($x1 * $points), ($y1 * $points), $text, [char]9, [char]10))
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thin_or_garbled_text_layers_need_ocr_and_tables_of_numbers_do_not() {
        let prose = "The parties agree that the term of this Agreement starts on the Effective \
                     Date and continues for two years unless ended under Section 9.";
        let numbers = "Revenues $ 96,773 $ 81,462 $ 53,823 Cost of revenues 79,113 65,121 40,217 \
                       Gross profit 17,660 16,341 13,606 Operating income 8,891 9,354 6,523 \
                       Net income $ 14,997 $ 12,556";
        assert!(!needs_ocr(prose) && !needs_ocr(numbers));
        // A scan, a slide's title over its picture, and a font with no Unicode map.
        assert!(needs_ocr("") && needs_ocr(" \n\u{c}") && needs_ocr("Q3 Results  4"));
        assert!(needs_ocr(&"\u{e012}\u{e0a4} \u{e012}! ".repeat(40)));
        assert!(needs_ocr(&format!("{prose} {}", "\u{fffd}".repeat(200))));
    }

    #[test]
    fn a_contents_page_of_dot_leaders_is_a_text_layer_and_private_use_characters_are_not() {
        // LM317's contents page has a letter share of 0.36 with its leaders counted.
        let contents: String = (1..=12)
            .map(|i| {
                format!(
                    "{i} Electrical Characteristics {} {}\n",
                    ".".repeat(60),
                    i + 2
                )
            })
            .chain(["Appendix\u{2026}\u{2026}\u{2026} 40 -- __ 41\n".into()])
            .collect();
        assert!(!needs_ocr(&contents));
        // Enough letters, but most of the page is a font's private-use characters.
        let private: String = (0..300)
            .map(|i| char::from_u32(0xe000 + i).unwrap())
            .collect();
        let title = "Results of the third quarter. ".repeat(5);
        assert!(!needs_ocr(&title) && needs_ocr(&format!("{title}{private}")));
    }

    /// tesseract 5.3.4's TSV of a 300 dpi scan of two lines of a contract, its page, block,
    /// paragraph and line rows kept as printed and a few words of the second line left out.
    const TESSERACT: &str = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext
1\t1\t0\t0\t0\t0\t0\t0\t2550\t3300\t-1\t
2\t1\t1\t0\t0\t0\t301\t266\t771\t34\t-1\t
3\t1\t1\t1\t0\t0\t301\t266\t771\t34\t-1\t
4\t1\t1\t1\t1\t0\t301\t266\t771\t34\t-1\t
5\t1\t1\t1\t1\t1\t301\t266\t201\t34\t96.631264\tMASTER
5\t1\t1\t1\t1\t2\t518\t266\t230\t34\t96.237610\tSERVICES
5\t1\t1\t1\t1\t3\t764\t266\t308\t34\t95.672226\tAGREEMENT
2\t1\t2\t0\t0\t0\t301\t396\t1146\t115\t-1\t
3\t1\t2\t1\t0\t0\t301\t396\t1146\t115\t-1\t
4\t1\t2\t1\t1\t0\t304\t396\t1143\t47\t-1\t
5\t1\t2\t1\t1\t1\t304\t399\t30\t33\t96.512466\t1.
5\t1\t2\t1\t1\t2\t350\t398\t117\t34\t96.202148\tTerm.
5\t1\t2\t1\t1\t3\t483\t397\t87\t35\t96.202148\tThis
5\t1\t2\t1\t1\t4\t585\t398\t221\t45\t96.265144\tAgreement
4\t1\t2\t1\t2\t0\t301\t464\t1143\t47\t-1\t
5\t1\t2\t1\t2\t1\t301\t466\t189\t34\t96.241310\tcontinues
5\t1\t2\t1\t2\t2\t506\t464\t56\t36\t96.241310\tfor
5\t1\t2\t1\t2\t3\t575\t471\t73\t29\t96.908249\ttwo
5\t1\t2\t1\t2\t4\t664\t465\t54\t45\t96.482140\t(2)
";

    #[test]
    fn tesseract_words_become_lines_in_points() {
        let lines = tesseract_lines(TESSERACT, 72.0 / DPI).unwrap();
        let text: Vec<&str> = lines.iter().map(|l| l.text.as_str()).collect();
        assert_eq!(
            text,
            [
                "MASTER SERVICES AGREEMENT",
                "1. Term. This Agreement",
                "continues for two (2)"
            ]
        );
        let near = |a: f64, b: f64| (a - b).abs() < 1e-9;
        let l = &lines[1];
        assert!(
            near(l.x0, 304.0 * 0.24)
                && near(l.y0, 397.0 * 0.24)
                && near(l.x1, 806.0 * 0.24)
                && near(l.y1, 443.0 * 0.24),
            "{l:?}"
        );
        assert!(tesseract_lines("level\n5\t1\t1\t1\t1\t1\t301\t266\t201\tMASTER", 1.0).is_err());
    }

    /// In the shape VISION and WINDOWS print: points from the page's top-left, the text last.
    // ponytail: written in the scripts' format here, as no Mac or Windows machine is; replace it
    // with output recorded there.
    const ENGINE: &str = "72.00\t70.56\t257.28\t81.12\tMASTER SERVICES AGREEMENT\n\
                          72.96\t95.04\t346.32\t106.32\t1. Term. This Agreement starts on the Effective Date and\n";

    #[test]
    fn a_cached_page_is_read_without_its_engine_and_a_bad_hash_is_refused() {
        let data = tempfile::tempdir().unwrap();
        let sha = "ab".repeat(32);
        let dir = data.path().join("agent/extract");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(format!("{sha}.ocr{VERSION}.p2.tsv")), ENGINE).unwrap();
        let stop = AtomicBool::new(false);
        let missing = data.path().join("gone.pdf");
        let ocr = read(None, &sha, 2, data.path(), &stop).unwrap();
        assert_eq!(ocr.lines.len(), 2);
        assert_eq!(
            ocr.marked(),
            "[text read from a picture on page 3]\nMASTER SERVICES AGREEMENT\n\
             1. Term. This Agreement starts on the Effective Date and"
        );
        // Another page is no hit: its engine runs, and fails on a file that is not there; with no
        // file, nothing runs.
        assert!(read(Some(&missing), &sha, 3, data.path(), &stop).is_err());
        assert!(read(None, &sha, 3, data.path(), &stop).is_err());
        assert!(read(Some(&missing), "../../x", 2, data.path(), &stop).is_err());
    }

    #[test]
    fn a_page_ocr_failed_on_is_not_read_again_in_this_process() {
        let data = tempfile::tempdir().unwrap();
        let (file, sha) = (data.path().join("scan.pdf"), "cd".repeat(32));
        let stop = AtomicBool::new(false);
        let failed = read(Some(&file), &sha, 0, data.path(), &stop).unwrap_err();
        // The scan is there now, but its engine does not run again for the page.
        let scan = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pages/scanned.pdf");
        fs::copy(scan, &file).unwrap();
        assert_eq!(read(Some(&file), &sha, 0, data.path(), &stop), Err(failed));
        // Another page is tried.
        assert_eq!(
            read(Some(&file), &sha, 1, data.path(), &stop).is_ok(),
            available()
        );
    }
}
