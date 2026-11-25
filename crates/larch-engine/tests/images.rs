//! Images through the engine on the CPU reference, with the tiny llama model and the tiny projector:
//! admission, the tile cache, a prefix hit inside a tile, encode ahead, adjacent spans, and guessing
//! over an image prompt.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use larch_engine::{
    EngineEvent, EngineHandle, EngineOptions, ImageSpan, Request, RequestHandle, invalid, start,
};
use larch_kernels::CPU;
use larch_model::projector::{ImagePixels, ProjectorConfig};
use larch_model::vision::{IMAGE_TOKEN, Image, image_tokens};
use larch_model::{Arch, tiny};
use larch_records::{EngineError, HealthState};

fn paths() -> (PathBuf, PathBuf) {
    static PATHS: OnceLock<(PathBuf, PathBuf)> = OnceLock::new();
    PATHS
        .get_or_init(|| {
            let dir = std::env::temp_dir();
            let pid = std::process::id();
            let model = dir.join(format!("larch-images-tiny-{pid}.gguf"));
            let mmproj = dir.join(format!("larch-images-mmproj-{pid}.gguf"));
            std::fs::write(&model, tiny::gguf(Arch::Llama, 3)).unwrap();
            std::fs::write(&mmproj, tiny::Projector::new(tiny::D as u32, 3).gguf()).unwrap();
            (model, mmproj)
        })
        .clone()
}

fn options(prefix_blocks: usize, guesses: u32) -> EngineOptions {
    let (model, mmproj) = paths();
    let mut o = EngineOptions::new(model, CPU);
    o.mmproj = Some(mmproj);
    o.max_context = 512;
    o.kv_pool_fraction = 0.3;
    o.prefix_cache_blocks = prefix_blocks;
    o.guesses = guesses;
    o
}

fn engine(prefix_blocks: usize, guesses: u32) -> EngineHandle {
    start(options(prefix_blocks, guesses)).unwrap()
}

fn palette() -> Arc<Image> {
    let png = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../larch-model/tests/data/vision/palette.png"
    ))
    .unwrap();
    Arc::new(Image::new(png.into()).unwrap())
}

fn pixels(image: &Image, v: &ProjectorConfig) -> Arc<ImagePixels> {
    Arc::new(v.cut(&image.decode().unwrap()))
}

fn span(
    at: u32,
    image: &Arc<Image>,
    px: Option<Arc<ImagePixels>>,
    v: &ProjectorConfig,
) -> ImageSpan {
    ImageSpan {
        at,
        rows: v.tokens_for(image.width(), image.height()),
        tile_rows: v.unit_rows(image.width(), image.height()),
        image: image.clone(),
        pixels: px,
    }
}

/// `before`, the image's rows, `after`, with the span placed.
fn prompt_with_image(
    before: &[u32],
    after: &[u32],
    image: &Arc<Image>,
    px: Option<Arc<ImagePixels>>,
    v: &ProjectorConfig,
) -> (Vec<u32>, ImageSpan) {
    let sp = span(before.len() as u32, image, px, v);
    let mut p = before.to_vec();
    p.extend(image_tokens(image.digest(), sp.rows));
    p.extend_from_slice(after);
    (p, sp)
}

type Tokens = Vec<(u32, Vec<(u32, f32)>)>;

fn run(h: RequestHandle) -> (Tokens, EngineEvent) {
    let mut tokens = Vec::new();
    for e in h.events.iter() {
        match e {
            EngineEvent::Token { id, top, .. } => tokens.push((id, top)),
            EngineEvent::Prefilled { .. } => {}
            end => return (tokens, end),
        }
    }
    panic!("stream ended without Done or Error");
}

fn cached(e: &EngineEvent) -> u32 {
    match e {
        EngineEvent::Done { usage, .. } => usage.cached_tokens,
        e => panic!("not Done: {e:?}"),
    }
}

fn request(prompt: Vec<u32>, spans: Vec<ImageSpan>, max_tokens: u32) -> Request {
    let mut r = Request::new(prompt, max_tokens);
    r.logprobs = 4;
    r.images = spans;
    r
}

fn text(n: usize, seed: u32) -> Vec<u32> {
    (0..n).map(|i| (i as u32 * 7 + seed) % 200 + 1).collect()
}

#[test]
fn admission_enforces_the_span_rules() {
    let e = engine(0, 0);
    let v = e.vision().expect("a projector");
    let image = palette();
    let px = pixels(&image, &v);
    let bad = Err(EngineError::InvalidRequest(invalid::IMAGE));
    let submit = |p: Vec<u32>, spans: Vec<ImageSpan>| e.submit(request(p, spans, 1)).map(drop);
    // Pixels are required at admission (`is_none_or`): None is refused, not cut by the engine.
    let (p, sp) = prompt_with_image(&text(8, 1), &text(4, 2), &image, None, &v);
    assert_eq!(submit(p, vec![sp]), bad, "pixels None");
    // Out of order or overlapping spans.
    let (p, sp) = prompt_with_image(&text(8, 1), &text(4, 2), &image, Some(px.clone()), &v);
    let mut overlap = sp.clone();
    overlap.at += 1;
    assert_eq!(submit(p.clone(), vec![sp.clone(), overlap]), bad, "overlap");
    // `at + n` past the prompt, and the u32 → usize edge.
    let mut far = sp.clone();
    far.at = u32::MAX;
    assert_eq!(submit(p.clone(), vec![far]), bad, "at = u32::MAX");
    // A span whose rows disagree with its image.
    let mut rows = sp.clone();
    rows.rows -= 1;
    assert_eq!(submit(p.clone(), vec![rows]), bad, "rows mismatch");
    // Pixels of another tile count than the image.
    let (ImagePixels::InternVl(mut two), ProjectorConfig::InternVl(vc)) = ((*px).clone(), v) else {
        panic!("an InternVL projector");
    };
    two.tiles += 1;
    two.patches.extend(std::iter::repeat_n(
        0,
        (vc.patches() * vc.patch_len()) as usize,
    ));
    let mut other = sp.clone();
    other.pixels = Some(Arc::new(ImagePixels::InternVl(two)));
    assert_eq!(submit(p.clone(), vec![other]), bad, "pixels tiles mismatch");
    // An image id outside every span, before and after.
    let mut stray = p.clone();
    stray[0] |= IMAGE_TOKEN;
    assert_eq!(
        submit(stray, vec![sp.clone()]),
        bad,
        "stray image id before"
    );
    let mut stray = p.clone();
    let last = stray.len() - 1;
    stray[last] |= IMAGE_TOKEN;
    assert_eq!(submit(stray, vec![sp.clone()]), bad, "stray image id after");
    // The relaxed vocabulary check: a plain out-of-vocabulary id still fails with images present.
    let mut oov = p.clone();
    oov[1] = 300;
    assert_eq!(
        submit(oov, vec![sp.clone()]),
        Err(EngineError::InvalidRequest(invalid::TOKEN_OUT_OF_VOCAB))
    );
    // A span inside the prompt with everything right is admitted and runs to Done.
    let h = e.submit(request(p, vec![sp], 2)).unwrap();
    let (tokens, end) = run(h);
    assert_eq!(tokens.len(), 2, "{end:?}");
    assert!(matches!(end, EngineEvent::Done { .. }), "{end:?}");
    // A prompt that is only image rows is admitted (its last row yields the logits).
    let (p, sp) = prompt_with_image(&[], &[], &image, Some(px.clone()), &v);
    let (tokens, end) = run(e.submit(request(p, vec![sp], 1)).unwrap());
    assert_eq!(tokens.len(), 1, "{end:?}");
    // Image rows count as prompt tokens against max_context.
    let (p, sp) = prompt_with_image(&text(500, 1), &[], &image, Some(px.clone()), &v);
    let long = e.submit(request(p, vec![sp], 20)).map(drop);
    assert!(
        matches!(long, Err(EngineError::PromptTooLong { .. })),
        "{long:?}"
    );
    e.shutdown();
    // Without a projector, images are NO_VISION.
    let (model, _) = paths();
    let plain = start(EngineOptions::new(model, CPU)).unwrap();
    let (p, sp) = prompt_with_image(&text(8, 1), &text(4, 2), &image, Some(px), &v);
    assert_eq!(
        plain.submit(request(p, vec![sp], 1)).map(drop),
        Err(EngineError::InvalidRequest(invalid::NO_VISION))
    );
    plain.shutdown();
}

/// The tile cache: the first turn encodes the image's one tile; a second turn whose prefix-cache hit
/// lands inside that tile (block 16 inside rows 14..18) reads the rest from the cache and produces the
/// same tokens as the fresh run.
#[test]
fn a_prefix_hit_inside_a_tile_reads_the_cached_tile() {
    let e = engine(32, 0);
    let v = e.vision().unwrap();
    assert_eq!(v.max_unit_rows(), 4, "tiny projector: 4 rows a tile");
    let image = palette();
    let px = pixels(&image, &v);
    let (p1, sp1) = prompt_with_image(&text(14, 1), &text(40, 3), &image, Some(px.clone()), &v);
    assert_eq!((sp1.at, sp1.rows), (14, 4));
    let (fresh, end) = run(e.submit(request(p1.clone(), vec![sp1.clone()], 6)).unwrap());
    assert_eq!(cached(&end), 0);
    let encoded = e.tiles_encoded();
    assert!(
        encoded >= 1,
        "the warm-up tile is not counted; the image's is"
    );
    // Same prompt again: the prefix cache serves whole blocks, so reuse = 48 here (inside the text
    // after the image); make the hit land inside the tile by sharing only the first block.
    let (p2, sp2) = prompt_with_image(&text(14, 1), &text(40, 9), &image, Some(px.clone()), &v);
    let (again, end) = run(e.submit(request(p2.clone(), vec![sp2.clone()], 6)).unwrap());
    let reuse = cached(&end);
    assert_eq!(reuse, 16, "one shared block, ending inside the tile");
    assert_eq!(
        e.tiles_encoded(),
        encoded,
        "the tile came from the cache, not a second encode"
    );
    // Reference: the same prompt on an engine without a prefix cache.
    let plain = engine(0, 0);
    let (reference, _) = run(plain.submit(request(p2, vec![sp2], 6)).unwrap());
    assert_eq!(
        ids(&again),
        ids(&reference),
        "a partial-tile reuse matches a fresh prefill"
    );
    assert_ne!(ids(&fresh), Vec::<u32>::new());
    plain.shutdown();
    e.shutdown();
}

fn ids(t: &Tokens) -> Vec<u32> {
    t.iter().map(|x| x.0).collect()
}

/// `prepare_image` encodes while idle and the later request hits the cache; pixels of the wrong
/// patch length are refused, not dropped on the engine's thread.
#[test]
fn prepare_image_fills_the_cache_ahead_of_the_turn() {
    let e = engine(0, 0);
    let v = e.vision().unwrap();
    let image = palette();
    let px = pixels(&image, &v);
    e.prepare_image(&image, px.clone()).unwrap();
    let t0 = Instant::now();
    while e.tiles_encoded() < 1 && t0.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(e.tiles_encoded(), 1, "one tile encoded while idle");
    let (p, sp) = prompt_with_image(&text(8, 1), &text(4, 2), &image, Some(px.clone()), &v);
    let (_, end) = run(e.submit(request(p, vec![sp], 1)).unwrap());
    assert!(matches!(end, EngineEvent::Done { .. }), "{end:?}");
    assert_eq!(e.tiles_encoded(), 1, "the request found its tile");
    let ImagePixels::InternVl(mut short) = (*px).clone() else {
        panic!("an InternVL projector");
    };
    short.patches.pop();
    assert_eq!(
        e.prepare_image(&image, Arc::new(ImagePixels::InternVl(short))),
        Err(EngineError::InvalidRequest(invalid::IMAGE))
    );
    e.shutdown();
}

/// Adjacent spans are admitted (`at >= end`) and a request alone reads them in one chunk; each
/// image's rows are their own run, as a run's gather ids hold one image (`MAX_TILES + 1` tiles = 28
/// rows here): 8 adjacent one-tile images are 32 rows.
#[test]
fn eight_adjacent_image_spans_complete() {
    let e = engine(0, 0);
    let v = e.vision().unwrap();
    let image = palette();
    let px = pixels(&image, &v);
    let mut p = text(3, 1);
    let mut spans = Vec::new();
    for _ in 0..8 {
        let sp = span(p.len() as u32, &image, Some(px.clone()), &v);
        p.extend(image_tokens(image.digest(), sp.rows));
        spans.push(sp);
    }
    p.extend(text(3, 2));
    let (tokens, end) = run(e.submit(request(p, spans, 2)).unwrap());
    assert_eq!(
        (tokens.len(), e.health().state),
        (2, HealthState::Ready),
        "{end:?}"
    );
    e.shutdown();
}

/// With guessing on, the prompt-lookup drafter proposes the tokens that followed the sequence's
/// last 2 tokens earlier in the prompt. When those were an image's rows, the guesses stop before
/// them: an image id is never a guess.
#[test]
fn guesses_over_an_image_prompt_complete() {
    let image = palette();
    // Find a prompt `[x, c, IMG, ..., x]` whose greedy first token is `c`, so the tail `[x, c]`
    // matches the pair before the image: iterate c on the greedy answer.
    let probe = engine(0, 0);
    let v = probe.vision().unwrap();
    let px = pixels(&image, &v);
    let mut c = 5u32;
    let mut found = None;
    for _ in 0..16 {
        let mut before = text(6, 1);
        before.extend([77, c]);
        let mut after = text(6, 4);
        after.push(77);
        let (p, sp) = prompt_with_image(&before, &after, &image, Some(px.clone()), &v);
        let (tokens, _) = run(probe
            .submit(request(p.clone(), vec![sp.clone()], 1))
            .unwrap());
        let t1 = tokens[0].0;
        if t1 == c {
            found = Some((p, sp));
            break;
        }
        c = t1;
    }
    probe.shutdown();
    let (p, sp) = found.expect("a prompt whose first token repeats the pair before the image");
    let e = engine(0, 4);
    let (tokens, end) = run(e.submit(request(p, vec![sp], 6)).unwrap());
    assert_eq!(
        (tokens.len(), e.health().state),
        (6, HealthState::Ready),
        "{end:?}"
    );
    e.shutdown();
}
