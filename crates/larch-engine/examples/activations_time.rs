//! The activations request's time on a real model: a 900-token prefix plus a 20-token question,
//! once cold and once after the prefix sits in the automatic prefix cache. Prints one JSON line.
//! Usage: activations_time GGUF DEVICE LAYER[,LAYER...]

use std::time::Instant;

use larch_engine::{EngineEvent, EngineHandle, EngineOptions, Request, start};

/// Wall ms from submit to Done, and Done's cached tokens and row count.
fn time(e: &EngineHandle, r: Request) -> (f64, u32, usize) {
    let t = Instant::now();
    let h = e.submit(r).unwrap();
    for ev in h.events.iter() {
        match ev {
            EngineEvent::Token { .. } | EngineEvent::Prefilled { .. } => {}
            EngineEvent::Done {
                usage, activations, ..
            } => {
                return (
                    t.elapsed().as_secs_f64() * 1e3,
                    usage.cached_tokens,
                    activations.len(),
                );
            }
            EngineEvent::Error(err) => panic!("{err:?}"),
        }
    }
    panic!("no terminal event")
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "GGUF DEVICE LAYER[,LAYER...]");
    let layers: Vec<u16> = args[3].split(',').map(|l| l.parse().unwrap()).collect();
    let mut opts = EngineOptions::new(args[1].clone().into(), args[2].parse().unwrap());
    opts.prefix_cache_blocks = 256;
    let e = start(opts).unwrap();
    let vocab = e.deployment().vocab;
    let tokens = |from: u32, n: u32| (from..from + n).map(|t| t % vocab).collect::<Vec<_>>();
    let (prefix, question) = (tokens(1, 900), tokens(2000, 20));
    let ask = |prompt: Vec<u32>| Request {
        activations: layers.clone(),
        ..Request::new(prompt, 0)
    };
    time(&e, Request::new(vec![1], 1)); // warm-up, outside both timings
    // Cold: a prompt that shares nothing with the cached one below.
    let cold = time(&e, ask([tokens(5000, 900), question.clone()].concat()));
    time(&e, Request::new(prefix.clone(), 1)); // warms the cache with the prefix
    let cached = time(&e, ask([prefix, question].concat()));
    println!(
        "{{\"kind\":\"activations\",\"prompt_tokens\":920,\"layers\":{layers:?},\"cold_ms\":{:.1},\"cached_ms\":{:.1},\"cached_tokens\":{},\"rows\":{}}}",
        cold.0, cached.0, cached.1, cached.2
    );
    e.shutdown();
}
