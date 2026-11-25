//! Benchmark-only localhost adapter: exact scripted prompts, natural EOS and conversation-local KV.
//! HTTP text chunk gaps are not individual token gaps. This is not a production HTTP API.
use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use larch::{GenParams, Larch, LoadOptions, Prompt, Session, SessionOptions, TextEvent};
use larch_records::knob_id;
use serde_json::{Value, json};

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    model: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 11436)]
    port: u16,
    #[arg(long, default_value_t = 2560)]
    context: u32,
    #[arg(long, default_value_t = 16)]
    sequences: u32,
    /// End after this many connections (including benchmark warmup), then join their workers.
    #[arg(long, default_value_t = 201)]
    requests: usize,
}

#[derive(Debug, PartialEq)]
struct SessionRequest {
    conversation: String,
    turn: u64,
    last: bool,
}

#[derive(Debug, PartialEq)]
struct Request {
    prompt: String,
    max_tokens: u32,
    session: Option<SessionRequest>,
}

struct Conversation {
    session: Session,
    next_turn: u64,
}

type Sessions = Mutex<HashMap<String, Arc<Mutex<Conversation>>>>;

fn detach(
    sessions: &Sessions,
    meta: &SessionRequest,
    entry: &Arc<Mutex<Conversation>>,
) -> Result<()> {
    let mut map = sessions
        .lock()
        .map_err(|_| anyhow::anyhow!("session map poisoned"))?;
    if map
        .get(&meta.conversation)
        .is_some_and(|stored| Arc::ptr_eq(stored, entry))
    {
        map.remove(&meta.conversation);
    }
    Ok(())
}

fn request(s: &TcpStream, context: u32) -> Result<Request> {
    let mut r = BufReader::new(s);
    let mut line = String::new();
    ensure!(r.read_line(&mut line)? < 4096, "request line too long");
    ensure!(
        line == "POST /api/generate HTTP/1.0\r\n",
        "expected benchmark endpoint"
    );
    let mut bytes = 0;
    let mut len = None;
    loop {
        line.clear();
        bytes += r.read_line(&mut line)?;
        ensure!(bytes <= 16384 && !line.is_empty(), "invalid headers");
        if line == "\r\n" {
            break;
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            ensure!(len.is_none(), "duplicate content length");
            len = Some(v.trim().parse::<usize>()?);
        }
    }
    let len = len.context("missing content length")?;
    ensure!(len <= 1024 * 1024, "request body too large");
    let mut body = vec![0; len];
    r.read_exact(&mut body)?;
    let v: Value = serde_json::from_slice(&body)?;
    ensure!(
        v["raw"] == true && v["stream"] == true,
        "raw streaming only"
    );
    let o = &v["options"];
    ensure!(
        o["temperature"] == 0.0 && o["repeat_penalty"] == 1.0,
        "greedy only"
    );
    ensure!(
        o["num_ctx"].as_u64() == Some(u64::from(context)),
        "context mismatch"
    );
    let n = u32::try_from(o["num_predict"].as_u64().context("missing num_predict")?)?;
    ensure!(n > 0 && n <= context, "invalid output cap");
    let session = if let Some(meta) = v.get("larch_session") {
        let conversation = meta["conversation"]
            .as_str()
            .context("missing conversation identity")?;
        ensure!(
            !conversation.is_empty() && conversation.len() <= 1024,
            "invalid conversation identity"
        );
        Some(SessionRequest {
            conversation: conversation.into(),
            turn: meta["turn"].as_u64().context("missing turn index")?,
            last: meta["last"]
                .as_bool()
                .context("missing final-turn marker")?,
        })
    } else {
        None
    };
    Ok(Request {
        prompt: v["prompt"].as_str().context("missing prompt")?.into(),
        max_tokens: n,
        session,
    })
}

/// The scripted turn is taken: the next one may follow, and a last one retires the identity.
fn take_turn(
    held: &mut Option<MutexGuard<'_, Conversation>>,
    req: &Request,
    entry: Option<&Arc<Mutex<Conversation>>>,
    sessions: &Sessions,
) -> Result<()> {
    if let Some(c) = held {
        c.next_turn = c.next_turn.checked_add(1).context("turn index overflow")?;
    }
    if let (Some(meta), Some(entry)) = (&req.session, entry)
        && meta.last
    {
        detach(sessions, meta, entry)?;
    }
    Ok(())
}

/// Answers a request the adapter refuses with its status and reason, then closes.
fn refuse(s: &mut TcpStream, status: &str, reason: impl std::fmt::Display) -> Result<()> {
    write!(s, "HTTP/1.0 {status}\r\nConnection: close\r\n\r\n{reason}")?;
    Ok(())
}

fn serve(mut s: TcpStream, v: &Larch, context: u32, sessions: &Sessions) -> Result<()> {
    s.set_read_timeout(Some(Duration::from_secs(300)))?;
    s.set_write_timeout(Some(Duration::from_secs(300)))?;
    s.set_nodelay(true)?;
    let req = match request(&s, context) {
        Ok(req) => req,
        Err(e) => return refuse(&mut s, "400 Bad Request", e),
    };
    let start = Instant::now();
    let params = GenParams {
        max_tokens: req.max_tokens,
        temperature: Some(0.0),
        repeat_penalty: Some(1.0),
        ..GenParams::default()
    };
    // The map lock protects only identity lookup; independent conversations generate concurrently.
    let conversation = if let Some(meta) = &req.session {
        let mut map = sessions
            .lock()
            .map_err(|_| anyhow::anyhow!("session map poisoned"))?;
        if !map.contains_key(&meta.conversation) {
            if meta.turn != 0 {
                drop(map); // a slow client must not hold up every other conversation's lookup
                return refuse(
                    &mut s,
                    "409 Conflict",
                    "conversation must start at turn zero",
                );
            }
            map.insert(
                meta.conversation.clone(),
                Arc::new(Mutex::new(Conversation {
                    session: v.session(SessionOptions::default()),
                    next_turn: 0,
                })),
            );
        }
        Some(map[&meta.conversation].clone())
    } else {
        None
    };
    let mut held = conversation
        .as_ref()
        .map(|c| {
            c.lock()
                .map_err(|_| anyhow::anyhow!("conversation poisoned"))
        })
        .transpose()?;
    if let (Some(meta), Some(c)) = (&req.session, &held)
        && meta.turn != c.next_turn
    {
        let reason = format!(
            "out-of-order scripted turn {}, next {}",
            meta.turn, c.next_turn
        );
        drop(held);
        return refuse(&mut s, "409 Conflict", reason);
    }
    let result = (|| -> Result<()> {
        let prompt = Prompt::Text(req.prompt.clone());
        let mut stream = if let Some(c) = &mut held {
            c.session.generate(prompt, params)?
        } else {
            v.generate(prompt, params)?
        };
        // A write fails only once the client hung up (the bench's --cancel-rate does after its
        // first chunk): dropping the stream cancels the request, and the scripted turn still
        // counts as taken, so the conversation's later turns go on.
        let mut hung_up = write!(
            s,
            "HTTP/1.0 200 OK\r\nContent-Type: application/x-ndjson\r\nConnection: close\r\n\r\n"
        )
        .is_err();
        let (mut failure, mut taken) = (None, false);
        while !hung_up && let Some(event) = stream.next() {
            let body = match event {
                TextEvent::Prefilled { .. } | TextEvent::Thinking { .. } => continue,
                TextEvent::Text { delta, .. } => json!({"response": delta, "done": false}),
                TextEvent::Done { finish, usage, .. } => {
                    // Before the terminal line: a client that saw it may reuse the identity.
                    take_turn(&mut held, &req, conversation.as_ref(), sessions)?;
                    taken = true;
                    json!({
                        "response": "", "done": true, "done_reason": format!("{finish:?}"),
                        "eval_count": usage.completion_tokens, "prompt_eval_count": usage.prompt_tokens,
                        "prompt_eval_cached_count": usage.cached_tokens,
                        "total_duration": start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
                        "conversation": req.session.as_ref().map(|m| &m.conversation),
                        "turn": req.session.as_ref().map(|m| m.turn),
                    })
                }
                TextEvent::Error(e) => {
                    failure = Some(e);
                    json!({"error": format!("{e:?}")})
                }
            };
            hung_up = writeln!(s, "{body}").is_err();
        }
        drop(stream);
        ensure!(failure.is_none(), "generation failed: {failure:?}");
        if !taken {
            take_turn(&mut held, &req, conversation.as_ref(), sessions)?;
        }
        Ok(())
    })();
    if let (Some(meta), Some(entry)) = (&req.session, &conversation)
        && result.is_err()
    {
        detach(sessions, meta, entry)?;
    }
    drop(held);
    result
}

fn main() -> Result<()> {
    let a = Args::parse();
    let load_start = Instant::now();
    let mut opts = LoadOptions::new(a.model, a.device);
    opts.max_context = a.context;
    opts.knob_overrides = vec![
        (knob_id::MAX_CONTEXT, f64::from(a.context)),
        (knob_id::MAX_BATCH_SEQS, f64::from(a.sequences)),
    ];
    let v = Arc::new(Larch::load(opts).map_err(|e| anyhow::anyhow!("load: {e:?}"))?);
    ensure!(
        v.rejected_overrides().is_empty(),
        "rejected startup settings: {:?}",
        v.rejected_overrides()
    );
    let effective = v.knobs().effective();
    ensure!(
        effective.values[knob_id::MAX_CONTEXT as usize] == f64::from(a.context)
            && effective.values[knob_id::MAX_BATCH_SEQS as usize] == f64::from(a.sequences),
        "effective context or sequence limit differs from requested settings"
    );
    eprintln!(
        "model_load_ms={:.3}",
        load_start.elapsed().as_secs_f64() * 1000.0
    );
    let sessions = Arc::new(Sessions::new(HashMap::new()));
    let listener = TcpListener::bind(("127.0.0.1", a.port))?;
    eprintln!(
        "ready localhost:{} context={} sequences={}",
        a.port, a.context, a.sequences
    );
    let mut workers = Vec::new();
    for _ in 0..a.requests {
        let (s, _) = listener.accept()?;
        let v = v.clone();
        let sessions = sessions.clone();
        let context = a.context;
        workers.push(thread::spawn(move || {
            if let Err(e) = serve(s, &v, context, &sessions) {
                eprintln!("request: {e:#}");
            }
        }));
    }
    for worker in workers {
        worker
            .join()
            .map_err(|_| anyhow::anyhow!("worker panicked"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use larch_engine::{EngineOptions, start_with};
    use larch_kernels::CPU;
    use larch_model::{ChatTemplate, FakeModel, Tokenizer, gguf::Gguf, tiny};

    fn fake_runtime(context: u32) -> Arc<Larch> {
        let bytes = std::fs::read(tiny::TOKENIZER_GGUF).unwrap();
        let g = Gguf::parse(&bytes).unwrap();
        let tokenizer = Tokenizer::from_gguf(&g).unwrap();
        let template = ChatTemplate::from_gguf(&g, &tokenizer).unwrap();
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = context;
        let engine =
            start_with(opts, Box::new(|_, _, _| Ok(Box::new(FakeModel::new(331))))).unwrap();
        Arc::new(Larch::from_engine(engine, tokenizer, template))
    }

    fn exchange(
        v: Arc<Larch>,
        sessions: Arc<Sessions>,
        prompt: &str,
        context: u32,
    ) -> (String, thread::JoinHandle<Result<()>>) {
        let meta = json!({"conversation":"scripted-identity","turn":0,"last":true});
        send(v, sessions, prompt, context, (1, meta), false)
    }

    /// One request of `num_predict` tokens for the `larch_session` meta; `hang_up` closes the
    /// client once the request is sent, as the bench's --cancel-rate does after a first chunk.
    fn send(
        v: Arc<Larch>,
        sessions: Arc<Sessions>,
        prompt: &str,
        context: u32,
        (num_predict, meta): (u32, Value),
        hang_up: bool,
    ) -> (String, thread::JoinHandle<Result<()>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let (server, _) = listener.accept().unwrap();
        let worker = thread::spawn(move || serve(server, &v, context, &sessions));
        let body = json!({"prompt":prompt,"raw":true,"stream":true,
            "options":{"temperature":0.0,"repeat_penalty":1.0,"num_predict":num_predict,"num_ctx":context},
            "larch_session":meta})
        .to_string();
        write!(
            client,
            "POST /api/generate HTTP/1.0\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        if hang_up {
            return (String::new(), worker);
        }
        let mut reader = BufReader::new(client);
        let mut response = String::new();
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap() == 0 {
                break;
            }
            response.push_str(&line);
            if serde_json::from_str::<Value>(&line).is_ok_and(|v| v["done"] == true) {
                break;
            }
        }
        (response, worker)
    }

    #[test]
    fn failed_admission_retires_identity_before_same_turn_retry() {
        let context = 16;
        let bytes = std::fs::read(tiny::TOKENIZER_GGUF).unwrap();
        let tokenizer = Tokenizer::from_gguf(&Gguf::parse(&bytes).unwrap()).unwrap();
        let too_long = "a ".repeat(64);
        assert!(tokenizer.encode(&too_long, true).len() > context as usize);
        let v = fake_runtime(context);
        let sessions = Arc::new(Sessions::new(HashMap::new()));
        let (_, worker) = exchange(v.clone(), sessions.clone(), &too_long, context);
        assert!(worker.join().unwrap().is_err());
        assert!(sessions.lock().unwrap().is_empty());
        let (response, worker) = exchange(v, sessions, "a", context);
        assert!(response.contains("\"done\":true"), "{response}");
        worker.join().unwrap().unwrap();
    }

    #[test]
    fn final_done_retires_identity_before_immediate_turn_zero_reuse() {
        let context = 16;
        let v = fake_runtime(context);
        let sessions = Arc::new(Sessions::new(HashMap::new()));
        let (first, first_worker) = exchange(v.clone(), sessions.clone(), "a", context);
        assert!(first.contains("\"done\":true"), "{first}");
        // Observe retirement at the HTTP terminal boundary, before joining its producer.
        assert!(sessions.lock().unwrap().is_empty());
        let (second, second_worker) = exchange(v, sessions, "a", context);
        let done = second
            .lines()
            .find_map(|line| {
                serde_json::from_str::<Value>(line)
                    .ok()
                    .filter(|v| v["done"] == true)
            })
            .unwrap();
        assert_eq!(done["prompt_eval_cached_count"], 0);
        assert_eq!(done["turn"], 0);
        first_worker.join().unwrap().unwrap();
        second_worker.join().unwrap().unwrap();
    }

    fn meta(turn: u64, last: bool) -> Value {
        json!({"conversation":"corpus-001","turn":turn,"last":last})
    }

    #[test]
    fn a_cancelled_turn_keeps_its_conversations_later_turns() {
        let context = 64;
        let v = fake_runtime(context);
        let sessions = Arc::new(Sessions::new(HashMap::new()));
        let (_, worker) = send(
            v.clone(),
            sessions.clone(),
            "a",
            context,
            (48, meta(0, false)),
            true,
        );
        worker.join().unwrap().unwrap();
        let (response, worker) = send(
            v,
            sessions.clone(),
            "a b",
            context,
            (1, meta(1, true)),
            false,
        );
        assert!(response.starts_with("HTTP/1.0 200 OK"), "{response}");
        assert!(response.contains("\"turn\":1"), "{response}");
        worker.join().unwrap().unwrap();
        assert!(sessions.lock().unwrap().is_empty());
    }

    #[test]
    fn out_of_order_turns_answer_409_with_the_reason() {
        let context = 16;
        let v = fake_runtime(context);
        let sessions = Arc::new(Sessions::new(HashMap::new()));
        let (response, worker) = send(
            v.clone(),
            sessions.clone(),
            "a",
            context,
            (1, meta(1, false)),
            false,
        );
        assert!(response.starts_with("HTTP/1.0 409 Conflict"), "{response}");
        assert!(response.contains("must start at turn zero"), "{response}");
        worker.join().unwrap().unwrap();
        let (response, worker) = send(
            v.clone(),
            sessions.clone(),
            "a",
            context,
            (1, meta(0, false)),
            false,
        );
        assert!(response.contains("\"done\":true"), "{response}");
        worker.join().unwrap().unwrap();
        let (response, worker) = send(v, sessions, "a", context, (1, meta(2, false)), false);
        assert!(response.starts_with("HTTP/1.0 409 Conflict"), "{response}");
        assert!(
            response.contains("out-of-order scripted turn 2, next 1"),
            "{response}"
        );
        worker.join().unwrap().unwrap();
    }

    fn parse(body: Value, context: u32) -> Result<Request> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let writer = thread::spawn(move || {
            let mut s = TcpStream::connect(address).unwrap();
            let body = body.to_string();
            write!(
                s,
                "POST /api/generate HTTP/1.0\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        });
        let (s, _) = listener.accept()?;
        let parsed = request(&s, context);
        writer.join().unwrap();
        parsed
    }

    fn body() -> Value {
        json!({"prompt":"raw prompt", "raw":true, "stream":true,
               "options":{"temperature":0.0,"repeat_penalty":1.0,"num_predict":64,"num_ctx":2560}})
    }

    #[test]
    fn accepts_raw_greedy_request_with_exact_context_and_output_cap() {
        assert_eq!(
            parse(body(), 2560).unwrap(),
            Request {
                prompt: "raw prompt".into(),
                max_tokens: 64,
                session: None
            }
        );
    }

    #[test]
    fn preserves_scripted_prompt_and_requires_complete_session_identity() {
        let mut b = body();
        b["larch_session"] = json!({"conversation":"corpus-007", "turn":3, "last":false});
        let r = parse(b.clone(), 2560).unwrap();
        assert_eq!(r.prompt, "raw prompt");
        assert_eq!(
            r.session.unwrap(),
            SessionRequest {
                conversation: "corpus-007".into(),
                turn: 3,
                last: false
            }
        );
        for field in ["conversation", "turn", "last"] {
            let mut bad = b.clone();
            bad["larch_session"].as_object_mut().unwrap().remove(field);
            assert!(parse(bad, 2560).is_err());
        }
    }

    #[test]
    fn rejects_workload_configuration_mismatches() {
        for (key, value) in [
            ("temperature", json!(0.1)),
            ("repeat_penalty", json!(1.1)),
            ("num_ctx", json!(8192)),
            ("num_predict", json!(0)),
            ("num_predict", json!(u64::MAX)),
        ] {
            let mut b = body();
            b["options"][key] = value;
            assert!(parse(b, 2560).is_err(), "accepted {key}");
        }
        let mut b = body();
        b["raw"] = json!(false);
        assert!(parse(b, 2560).is_err());
    }
}
