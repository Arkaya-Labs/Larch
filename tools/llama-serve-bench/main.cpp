// llama-serve-bench (architecture 16, D13): larch's benchmark workloads on llama.cpp in-process, with
// llama-server's update_slots policy and slot cache and without its HTTP layer.
//   llama-serve-bench run      --workload <w.jsonl> [--concurrency N] [--conversations K] [--cache-prompt]
//                              [--machine M] --json <out.json> <llama.cpp flags: -m -c -b -ub -ngl -fa -ctk -ctv -t>
//   llama-serve-bench fixed    -npp 512 -ntg 128 -npl 1,4,8,16 [--machine M] --json <out.json> <llama.cpp flags>
//   llama-serve-bench manifest --workload <w.jsonl>... -m <gguf>    (per-turn token evidence, JSONL on stdout)
// -c is the whole context, split evenly over the sequences (kv_unified off), as llama-server's -c and -np.
// The workload format, the result schema and the policy are in bench/README.md.
#include "arg.h"
#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "greedy.h"
#include "hash/hash.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// Every engine renders the workloads with this date, so prompt-token hashes compare across engines and
// days: as date_string for templates that print it, as `now` (noon UTC that day) for strftime_now.
static const char *      DATE     = "26 Jul 2024";
static const std::time_t DATE_NOW = 1721995200;

static std::string slurp(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot read " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string basename_of(const std::string & path) {
    const auto i = path.find_last_of("/\\");
    return i == std::string::npos ? path : path.substr(i + 1);
}

// The prompt-token hash every engine reports: sha256 over the IDs as little-endian int32. Every
// supported host is little-endian, so the vector's bytes are that string.
static std::string token_sha256(const llama_tokens & t) {
    return hash_sha256_hex(t.data(), t.size() * sizeof(llama_token));
}

static size_t common_prefix(const llama_tokens & a, const llama_tokens & b) {
    const size_t n = std::min(a.size(), b.size());
    return size_t(std::mismatch(a.begin(), a.begin() + n, b.begin()).first - a.begin());
}

// --- workloads ---------------------------------------------------------------------------------------

// One timed answer: the prompt is messages [0, n_msgs) with the generation prompt, the answer is
// max_tokens long, and messages[n_msgs], when the conversation continues, is its scripted text.
struct Turn {
    size_t n_msgs;
    int    max_tokens;
};

struct Conv {
    std::string                  id;
    std::vector<common_chat_msg> msgs;
    std::vector<Turn>            turns;
};

struct Workload {
    std::string       name;
    std::string       sha256;
    std::vector<Conv> convs;
};

static Workload load_workload(const std::string & path) {
    const std::string  bytes = slurp(path);
    Workload           w{ basename_of(path), hash_sha256_hex(bytes.data(), bytes.size()), {} };
    std::istringstream in(bytes);
    size_t             line_no = 0;
    for (std::string line; std::getline(in, line);) {
        ++line_no;
        if (line.empty()) {
            continue;
        }
        const auto   where = path + ":" + std::to_string(line_no) + ": ";
        const auto   j     = json::parse(line);
        const auto & msgs  = j.at("messages");
        Conv         c;
        c.id = j.at("id").get<std::string>();
        for (size_t i = 0; i < msgs.size(); ++i) {
            const auto &    m = msgs[i];
            common_chat_msg msg;
            msg.role       = m.at("role").get<std::string>();
            msg.content    = m.value("content", "");
            const bool last = i + 1 == msgs.size();
            if (msg.role == "assistant") {
                const int n = m.at("max_tokens").get<int>();
                if (n < 1 || i == 0 || c.msgs.back().role != "user") {
                    throw std::runtime_error(where + "an assistant turn needs max_tokens >= 1 and a user message before it");
                }
                if (!last && !m.contains("content")) {
                    throw std::runtime_error(where + "an assistant message the conversation continues after needs its scripted content");
                }
                c.turns.push_back({ i, n });
            } else if (msg.role != "system" && msg.role != "user") {
                throw std::runtime_error(where + "role must be system, user or assistant");
            } else if (last) {
                throw std::runtime_error(where + "a conversation ends with the assistant turn that answers it");
            }
            c.msgs.push_back(std::move(msg));
        }
        if (c.turns.empty()) {
            throw std::runtime_error(where + "a conversation needs at least one assistant turn");
        }
        w.convs.push_back(std::move(c));
    }
    if (w.convs.empty()) {
        throw std::runtime_error(path + ": no conversations");
    }
    return w;
}

// The GGUF's own Jinja template through common_chat_templates_apply, then the whole prompt tokenised
// with add_special and parse_special: llama-server's chat path, done fresh for every turn.
struct Renderer {
    common_chat_templates_ptr tmpls;
    const llama_vocab *       vocab;

    llama_tokens operator()(const Conv & c, size_t n_msgs) const {
        common_chat_templates_inputs in;
        in.messages.assign(c.msgs.begin(), c.msgs.begin() + long(n_msgs));
        in.use_jinja                           = true;
        in.add_generation_prompt               = true;
        in.now                                 = std::chrono::system_clock::from_time_t(DATE_NOW);
        in.chat_template_kwargs["date_string"] = json(DATE).dump();
        return common_tokenize(vocab, common_chat_templates_apply(tmpls.get(), in).prompt, true, true);
    }
};

// --- the serving loop --------------------------------------------------------------------------------

struct Req {
    int          client = -1;
    std::string  conv;
    int          turn = 0;
    llama_tokens prompt;
    int          max_tokens = 0;
    int          slot       = -1;
    int          cached     = 0;
    int          gen        = 0;
    double       t_submit = 0, t_start = 0, t_first = 0, t_done = 0;
    std::string  error;
};

struct Slot {
    int          id;
    llama_tokens tokens;  // what this sequence holds in the KV cache, in order
    int64_t      t_last_used = -1;
    Req *        req         = nullptr;
    bool         prompt_done = false;
    int32_t      i_batch     = -1;
    llama_token  sampled     = LLAMA_TOKEN_NULL;
};

// The next request of an idle client, or nullptr once that client is finished.
using NextFn = std::function<std::unique_ptr<Req>(int client)>;

struct Server {
    llama_context *                       ctx;
    bool                                  cache_prompt;
    bool                                  phased = false;  // fixed shape: no decode step until every prompt is in
    int32_t                               n_vocab;
    std::vector<Slot>                     slots;
    std::vector<char>                     eog;
    common_batch                          batch;
    std::chrono::steady_clock::time_point t0;

    Server(llama_context * c, int n_slots, bool reuse) :
        ctx(c),
        cache_prompt(reuse),
        n_vocab(llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(c)))),
        batch(c) {
        for (int i = 0; i < n_slots; ++i) {
            slots.push_back({ i, {} });
        }
        // ignore_eos: llama-server biases every end-of-generation token to -inf.
        const auto * vocab = llama_model_get_vocab(llama_get_model(ctx));
        eog.resize(size_t(n_vocab));
        for (llama_token t = 0; t < n_vocab; ++t) {
            eog[size_t(t)] = llama_vocab_is_eog(vocab, t);
        }
    }

    double now_ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    // llama-server's get_available_slot: the free slot whose cached tokens cover the largest fraction
    // of the new prompt, above 0.10 (strictly; the first best wins); else the least recently used free
    // slot (ties go to the later slot).
    Slot * pick(const Req & r) {
        Slot * best  = nullptr;
        float  f_max = 0;
        for (auto & s : slots) {
            if (s.req || s.tokens.empty()) {
                continue;
            }
            const float f = float(common_prefix(s.tokens, r.prompt)) / float(r.prompt.size());
            if (f > f_max && f > 0.10f) {
                f_max = f;
                best  = &s;
            }
        }
        if (best) {
            return best;
        }
        int64_t t_last = -1;
        for (auto & s : slots) {
            if (!s.req && (!best || s.t_last_used <= t_last)) {
                t_last = s.t_last_used;
                best   = &s;
            }
        }
        return best;
    }

    // Keep the longest common token prefix, but evaluate at least one prompt token (TAG_PROMPT_LOGITS).
    void launch(Slot & s, Req & r) {
        size_t n_past = cache_prompt ? common_prefix(s.tokens, r.prompt) : 0;
        if (n_past == r.prompt.size()) {
            --n_past;
        }
        r.slot    = s.id;
        r.cached  = int(n_past);
        r.t_start = now_ms();
        s.tokens.resize(n_past);
        llama_memory_seq_rm(llama_get_memory(ctx), s.id, llama_pos(n_past), -1);
        s.req         = &r;
        s.prompt_done = false;
    }

    // Runs until every client is finished; returns the requests in submit order.
    std::vector<std::unique_ptr<Req>> run(int n_clients, const NextFn & next) {
        const size_t  n_ctx_seq = llama_n_ctx_seq(ctx);
        const int32_t n_batch   = int32_t(llama_n_batch(ctx));
        std::vector<std::unique_ptr<Req>> all;
        std::vector<Req *>                active(size_t(n_clients), nullptr);
        std::vector<char>                 finished(size_t(n_clients), 0);
        std::deque<Req *>                 queue;
        const auto done = [&](Slot & s, double t) {
            s.req->t_done             = t;
            active[size_t(s.req->client)] = nullptr;
            s.req                     = nullptr;
            s.t_last_used             = ggml_time_us();
        };
        t0 = std::chrono::steady_clock::now();
        for (;;) {
            // Closed loop: an idle client submits its next request at once.
            for (int c = 0; c < n_clients; ++c) {
                while (!active[size_t(c)] && !finished[size_t(c)]) {
                    auto r = next(c);
                    if (!r) {
                        finished[size_t(c)] = 1;
                        break;
                    }
                    r->client   = c;
                    r->t_submit = now_ms();
                    Req * rp    = r.get();
                    all.push_back(std::move(r));
                    if (rp->prompt.size() >= n_ctx_seq) {
                        rp->error  = "prompt of " + std::to_string(rp->prompt.size()) + " tokens does not fit n_ctx_seq " +
                                     std::to_string(n_ctx_seq);
                        rp->t_done = rp->t_submit;
                        continue;
                    }
                    active[size_t(c)] = rp;
                    queue.push_back(rp);
                }
            }
            if (std::none_of(active.begin(), active.end(), [](Req * r) { return r != nullptr; })) {
                return all;
            }
            // Tasks wait in arrival order for a free slot.
            while (!queue.empty()) {
                Slot * s = pick(*queue.front());
                if (!s) {
                    break;
                }
                launch(*s, *queue.front());
                queue.pop_front();
            }
            // One token per generating slot first, then prompt tokens in slot order up to n_batch.
            batch.clear();
            const bool prefilling =
                phased && std::any_of(slots.begin(), slots.end(), [](const Slot & s) { return s.req && !s.prompt_done; });
            for (auto & s : slots) {
                s.i_batch = -1;
                if (s.req && s.prompt_done && !prefilling) {
                    s.i_batch = batch.add(s.sampled, llama_pos(s.tokens.size()), s.id, true);
                    s.tokens.push_back(s.sampled);
                }
            }
            for (auto & s : slots) {
                if (batch.size() >= n_batch) {
                    break;
                }
                if (!s.req || s.prompt_done) {
                    continue;
                }
                const auto & p = s.req->prompt;
                while (s.tokens.size() < p.size() && batch.size() < n_batch) {
                    batch.add(p[s.tokens.size()], llama_pos(s.tokens.size()), s.id, false);
                    s.tokens.push_back(p[s.tokens.size()]);
                }
                if (s.tokens.size() == p.size()) {
                    s.i_batch = batch.size() - 1;
                    batch.set_output(s.i_batch, true);
                    s.prompt_done = true;
                }
            }
            const int ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (ret != 0) {
                throw std::runtime_error("llama_process failed: " + std::to_string(ret));
            }
            llama_synchronize(ctx);
            // One vocabulary scan per sequence on this thread, at the speed of reading the logits (greedy.h):
            // llama-batched-bench reads none, so every microsecond here is a gap in the fixed-shape mirror, and
            // waking a thread team costs more than it saves (wsl5070, 16 rows: 165 us here, 191 us on 4 threads).
            // The first llama_get_logits_ith applies any output reordering, so a row stays put once fetched.
            // ponytail: one host read of every logit, 2 % of a decode step at npl 16 on wsl5070; if a machine's
            // fixed-shape check misses at npl 16, move greedy to llama.cpp's backend sampler without the logit bias
            // (llama_sampler_seq_config: -3.8 % at npl 16 on the 5070 Ti in A1B3, the bias upload -18 %).
            for (auto & s : slots) {
                if (s.i_batch < 0) {
                    continue;
                }
                s.sampled    = argmax_no_eog(llama_get_logits_ith(ctx, s.i_batch), n_vocab, eog);
                Req & r      = *s.req;
                const double t = now_ms();
                if (r.gen++ == 0) {
                    r.t_first = t;
                }
                if (r.gen == r.max_tokens) {
                    done(s, t);
                } else if (s.tokens.size() + 1 >= n_ctx_seq) {
                    // llama-server would shift the context here; no workload is sized to reach it.
                    r.error = "context full after " + std::to_string(r.gen) + " tokens";
                    done(s, t);
                }
            }
        }
    }
};

// --- results -----------------------------------------------------------------------------------------

// Nearest-rank percentile, the definition every engine's summary uses.
static double pct(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    const auto k = size_t(std::ceil(q * double(v.size())));
    return v[std::clamp<size_t>(k, 1, v.size()) - 1];
}

static json stats(const std::vector<double> & v) {
    if (v.empty()) {
        return nullptr;
    }
    double sum = 0;
    for (double x : v) {
        sum += x;
    }
    return { { "p50", pct(v, 0.50) }, { "p99", pct(v, 0.99) }, { "mean", sum / double(v.size()) }, { "max", pct(v, 1.0) } };
}

static json request_json(const Req & r) {
    const bool ok = r.error.empty();
    return {
        { "client", r.client },
        { "conv", r.conv },
        { "turn", r.turn },
        { "slot", r.slot },
        { "prompt_tokens", r.prompt.size() },
        { "cached_tokens", r.cached },
        { "evaluated_tokens", ok ? json(int(r.prompt.size()) - r.cached) : json(nullptr) },
        { "prompt_sha256", token_sha256(r.prompt) },
        { "max_tokens", r.max_tokens },
        { "gen_tokens", r.gen },
        { "t_submit_ms", r.t_submit },
        { "t_first_ms", ok ? json(r.t_first) : json(nullptr) },
        { "t_done_ms", r.t_done },
        { "ttft_ms", ok ? json(r.t_first - r.t_submit) : json(nullptr) },
        { "tpot_ms", ok && r.gen > 1 ? json((r.t_done - r.t_first) / (r.gen - 1)) : json(nullptr) },
        { "e2e_ms", ok ? json(r.t_done - r.t_submit) : json(nullptr) },
        { "error", ok ? json(nullptr) : json(r.error) },
    };
}

static json summary_json(const std::vector<std::unique_ptr<Req>> & reqs) {
    std::vector<double> ttft, tpot, e2e;
    long                gen = 0, prompt = 0, cached = 0, failures = 0;
    double              first = INFINITY, last = 0;
    for (const auto & r : reqs) {
        first = std::min(first, r->t_submit);
        last  = std::max(last, r->t_done);
        if (!r->error.empty()) {
            ++failures;
            continue;
        }
        gen += r->gen;
        prompt += long(r->prompt.size());
        cached += r->cached;
        ttft.push_back(r->t_first - r->t_submit);
        e2e.push_back(r->t_done - r->t_submit);
        if (r->gen > 1) {
            tpot.push_back((r->t_done - r->t_first) / (r->gen - 1));
        }
    }
    const double wall_s = reqs.empty() ? 0 : (last - first) / 1000.0;
    return {
        { "requests", reqs.size() },
        { "failures", failures },
        { "prompt_tokens", prompt },
        { "cached_tokens", cached },
        { "evaluated_tokens", prompt - cached },
        { "gen_tokens", gen },
        { "wall_s", wall_s },
        { "agg_tok_s", wall_s > 0 ? double(gen) / wall_s : 0.0 },
        { "ttft_ms", stats(ttft) },
        { "tpot_ms", stats(tpot) },
        { "e2e_ms", stats(e2e) },
    };
}

static std::string utc_now() {
    char              buf[32];
    const std::time_t t = std::time(nullptr);
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return buf;
}

static json header_json(const char * mode, const common_params & p, llama_context * ctx, const std::string & machine) {
    return {
        { "schema", "larch-bench/1" },
        { "engine", "llama.cpp" },
        { "driver", "llama-serve-bench" },
        { "mode", mode },
        { "llama_cpp", { { "tag", LLAMA_TAG }, { "commit", llama_commit() } } },
        { "build", DRIVER_BUILD },
        { "system_info", llama_print_system_info() },
        { "machine", machine },
        { "started_at", utc_now() },
        // The file name only: the bench/models.toml alias, never a local path.
        { "model", basename_of(p.model.path) },
        { "date_string", DATE },
        { "params",
         {
             { "n_ctx", llama_n_ctx(ctx) },
             { "n_ctx_seq", llama_n_ctx_seq(ctx) },
             { "n_seq_max", llama_n_seq_max(ctx) },
             { "n_batch", llama_n_batch(ctx) },
             { "n_ubatch", llama_n_ubatch(ctx) },
             { "n_gpu_layers", p.n_gpu_layers },
             { "flash_attn", llama_flash_attn_type_name(p.flash_attn_type) },
             { "type_k", ggml_type_name(p.cache_type_k) },
             { "type_v", ggml_type_name(p.cache_type_v) },
             { "n_threads", llama_n_threads(ctx) },
             { "n_threads_batch", llama_n_threads_batch(ctx) },
             { "kv_unified", p.kv_unified },
         } },
    };
}

static void write_json(const std::string & path, const json & j) {
    std::ofstream f(path, std::ios::binary);
    f << j.dump(1) << "\n";
    if (!f) {
        throw std::runtime_error("cannot write " + path);
    }
}

static double field(const json & s, const char * k, const char * q) {
    return s[k].is_null() ? NAN : s[k][q].get<double>();
}

// --- commands ----------------------------------------------------------------------------------------

struct Opts {
    std::vector<std::string> workloads;
    std::string              json_out;
    std::string              machine;
    int                      concurrency   = 1;
    int                      conversations = 0;  // 0 = all
    bool                     cache_prompt  = false;
};

using ModelPtr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using CtxPtr   = std::unique_ptr<llama_context, decltype(&llama_free)>;

static ModelPtr load_model(common_params & p, bool vocab_only) {
    auto mp       = common_model_params_to_llama(p);
    mp.vocab_only = vocab_only;
    ModelPtr m(llama_model_load_from_file(p.model.path.c_str(), mp), llama_model_free);
    if (!m) {
        throw std::runtime_error("cannot load " + p.model.path);
    }
    return m;
}

// llama-batched-bench's context: its flags, n_seq_max sequences sharing -c, and its 16-token warm-up.
static CtxPtr make_context(llama_model * model, const common_params & p, int n_seq_max) {
    auto cp      = common_context_params_to_llama(p);
    cp.n_seq_max = uint32_t(n_seq_max);
    CtxPtr ctx(llama_init_from_model(model, cp), llama_free);
    if (!ctx) {
        throw std::runtime_error("cannot create the llama_context");
    }
    common_batch warm(ctx.get());
    for (int i = 0; i < 16; ++i) {
        warm.add(std::rand() % llama_vocab_n_tokens(llama_model_get_vocab(model)), i, 0, false);
    }
    if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, warm.get()) != 0) {
        throw std::runtime_error("warm-up decode failed");
    }
    llama_synchronize(ctx.get());
    llama_memory_clear(llama_get_memory(ctx.get()), false);
    return ctx;
}

static int cmd_run(common_params & p, const Opts & o) {
    if (o.workloads.size() != 1 || o.json_out.empty() || o.concurrency < 1) {
        throw std::runtime_error("run needs one --workload, --json and --concurrency >= 1");
    }
    const Workload w      = load_workload(o.workloads[0]);
    const size_t   n_conv = o.conversations > 0 ? std::min(size_t(o.conversations), w.convs.size()) : w.convs.size();
    auto           model  = load_model(p, false);
    auto           ctx    = make_context(model.get(), p, o.concurrency);

    // Scripted replay: the history continues with the scripted answers, so every prompt is known up
    // front and every engine sees the same one. They are rendered and tokenised before the clock
    // starts, which leaves tokenisation (about 1 ms per 8k tokens) out of llama.cpp's TTFT: the
    // baseline gets the benefit of the doubt.
    const Renderer                         render{ common_chat_templates_init(model.get(), ""), llama_model_get_vocab(model.get()) };
    std::vector<std::vector<llama_tokens>> prompts(n_conv);
    for (size_t i = 0; i < n_conv; ++i) {
        for (const auto & t : w.convs[i].turns) {
            prompts[i].push_back(render(w.convs[i], t.n_msgs));
        }
    }

    // Each client takes the next conversation in file order and runs its turns back to back.
    struct Cursor {
        size_t conv = SIZE_MAX;
        size_t turn = 0;
    };
    std::vector<Cursor> cur(size_t(o.concurrency));
    size_t              next_conv = 0;
    const NextFn        next      = [&](int c) -> std::unique_ptr<Req> {
        auto & k = cur[size_t(c)];
        if (k.conv == SIZE_MAX || k.turn == w.convs[k.conv].turns.size()) {
            if (next_conv == n_conv) {
                return nullptr;
            }
            k = { next_conv++, 0 };
        }
        auto r        = std::make_unique<Req>();
        r->conv       = w.convs[k.conv].id;
        r->turn       = int(k.turn);
        r->prompt     = prompts[k.conv][k.turn];
        r->max_tokens = w.convs[k.conv].turns[k.turn].max_tokens;
        ++k.turn;
        return r;
    };
    Server     srv(ctx.get(), o.concurrency, o.cache_prompt);
    const auto reqs = srv.run(o.concurrency, next);

    json out            = header_json("run", p, ctx.get(), o.machine);
    out["workload"]     = { { "file", w.name }, { "sha256", w.sha256 }, { "conversations", n_conv } };
    out["concurrency"]  = o.concurrency;
    out["cache_prompt"] = o.cache_prompt;
    out["summary"]      = summary_json(reqs);
    out["requests"]     = json::array();
    for (const auto & r : reqs) {
        out["requests"].push_back(request_json(*r));
    }
    write_json(o.json_out, out);
    const auto & s = out["summary"];
    printf("%s c=%d: %ld requests, %ld failed, %.2f tok/s, TTFT p50 %.1f p99 %.1f ms, TPOT p50 %.2f p99 %.2f ms\n",
           w.name.c_str(), o.concurrency, s["requests"].get<long>(), s["failures"].get<long>(), s["agg_tok_s"].get<double>(),
           field(s, "ttft_ms", "p50"), field(s, "ttft_ms", "p99"), field(s, "tpot_ms", "p50"), field(s, "tpot_ms", "p99"));
    return s["failures"].get<long>() == 0 ? 0 : 1;
}

// The llama-batched-bench shape through the serving loop: npl requests submitted at once, each a
// pp-token random prompt that feeds pp + tg tokens as batched-bench does (so tg + 1 are sampled), and
// its phases: every prompt in n_batch chunks, then npl-token decode steps. Continuous batching would
// overlap the two and run a ragged tail, which is a different shape. S = npl * (pp + tg) / T is the
// aggregate compared against batched-bench's S.
static int cmd_fixed(common_params & p, const Opts & o) {
    if (o.json_out.empty() || p.n_pp.empty() || p.n_tg.empty() || p.n_pl.empty()) {
        throw std::runtime_error("fixed needs -npp, -ntg, -npl and --json");
    }
    auto       model   = load_model(p, false);
    auto       ctx     = make_context(model.get(), p, *std::max_element(p.n_pl.begin(), p.n_pl.end()));
    const int  n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    json       out     = header_json("fixed", p, ctx.get(), o.machine);
    out["fixed"]       = json::array();
    for (int pp : p.n_pp) {
        for (int tg : p.n_tg) {
            for (int pl : p.n_pl) {
                if (pp + tg >= int(llama_n_ctx_seq(ctx.get()))) {
                    fprintf(stderr, "skipping pp=%d tg=%d: n_ctx_seq %u is too small\n", pp, tg, llama_n_ctx_seq(ctx.get()));
                    continue;
                }
                llama_memory_clear(llama_get_memory(ctx.get()), false);
                std::vector<char> issued(size_t(pl), 0);
                const NextFn      next = [&](int c) -> std::unique_ptr<Req> {
                    if (issued[size_t(c)]) {
                        return nullptr;
                    }
                    issued[size_t(c)] = 1;
                    auto r            = std::make_unique<Req>();
                    r->conv           = "fixed";
                    r->max_tokens     = tg + 1;
                    for (int i = 0; i < pp; ++i) {
                        r->prompt.push_back(std::rand() % n_vocab);
                    }
                    return r;
                };
                Server srv(ctx.get(), pl, false);
                srv.phased      = true;
                const auto reqs = srv.run(pl, next);
                double     t_pp = 0, t = 0;
                for (const auto & r : reqs) {
                    if (!r->error.empty()) {
                        throw std::runtime_error(r->error);
                    }
                    t_pp = std::max(t_pp, r->t_first / 1000.0);
                    t    = std::max(t, r->t_done / 1000.0);
                }
                const double t_tg = t - t_pp;
                const double s    = pl * (pp + tg) / t;
                out["fixed"].push_back({
                    { "pp", pp }, { "tg", tg }, { "pl", pl }, { "n_kv", pl * (pp + tg) },
                    { "t_pp", t_pp }, { "speed_pp", pl * pp / t_pp },
                    { "t_tg", t_tg }, { "speed_tg", pl * tg / t_tg },
                    { "t", t }, { "speed", s },
                });
                printf("fixed pp=%d tg=%d pl=%d: T %.3f s, S %.2f t/s\n", pp, tg, pl, t, s);
            }
        }
    }
    write_json(o.json_out, out);
    return 0;
}

// Per-turn token evidence for one GGUF: what each scripted-replay prompt tokenises to, its hash, how
// much of it the previous turn's prompt already holds, and the content lengths the workload targets.
static int cmd_manifest(common_params & p, const Opts & o) {
    if (o.workloads.empty()) {
        throw std::runtime_error("manifest needs --workload");
    }
    auto           model = load_model(p, true);
    const auto *   vocab = llama_model_get_vocab(model.get());
    const Renderer render{ common_chat_templates_init(model.get(), ""), vocab };
    const auto     n_content = [&](const std::string & s) { return common_tokenize(vocab, s, false, false).size(); };
    for (const auto & path : o.workloads) {
        const Workload w = load_workload(path);
        for (const auto & c : w.convs) {
            llama_tokens prev;
            for (size_t k = 0; k < c.turns.size(); ++k) {
                const Turn & t      = c.turns[k];
                const auto   tokens = render(c, t.n_msgs);
                const size_t lcp    = common_prefix(prev, tokens);
                const bool   cont   = t.n_msgs + 1 < c.msgs.size();
                const json   line   = {
                    { "workload", w.name },
                    { "workload_sha256", w.sha256 },
                    { "conv", c.id },
                    { "turn", k },
                    { "prompt_tokens", tokens.size() },
                    { "prompt_sha256", token_sha256(tokens) },
                    { "prefix_prev", lcp },
                    { "new_tokens", tokens.size() - lcp },
                    { "user_tokens", n_content(c.msgs[t.n_msgs - 1].content) },
                    { "scripted_tokens", cont ? json(n_content(c.msgs[t.n_msgs].content)) : json(nullptr) },
                    { "max_tokens", t.max_tokens },
                    { "peak_tokens", tokens.size() + size_t(t.max_tokens) },
                };
                printf("%s\n", line.dump().c_str());
                prev = tokens;
            }
        }
    }
    return 0;
}

int main(int argc, char ** argv) {
    try {
        const std::string cmd = argc > 1 ? argv[1] : "";
        if (cmd != "run" && cmd != "fixed" && cmd != "manifest") {
            fprintf(stderr, "usage: llama-serve-bench run|fixed|manifest ... (see the top of main.cpp)\n");
            return 2;
        }
        // Our flags first; the rest goes to llama.cpp's own parser, so -fa, -ctk, -b, -ub, -t and -ngl
        // mean exactly what they mean to llama-server and llama-batched-bench.
        Opts                o;
        std::vector<char *> rest = { argv[0] };
        for (int i = 2; i < argc; ++i) {
            const std::string a     = argv[i];
            const auto        value = [&]() -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error(a + " needs a value");
                }
                return argv[++i];
            };
            if (a == "--workload") {
                o.workloads.push_back(value());
            } else if (a == "--json") {
                o.json_out = value();
            } else if (a == "--machine") {
                o.machine = value();
            } else if (a == "--concurrency") {
                o.concurrency = std::stoi(value());
            } else if (a == "--conversations") {
                o.conversations = std::stoi(value());
            } else if (a == "--cache-prompt") {
                o.cache_prompt = true;
            } else {
                rest.push_back(argv[i]);
            }
        }
        common_params p;
        common_init();
        if (!common_params_parse(int(rest.size()), rest.data(), p, LLAMA_EXAMPLE_BENCH)) {
            return 2;
        }
        llama_backend_init();
        llama_numa_init(p.numa);
        const int rc = cmd == "run" ? cmd_run(p, o) : cmd == "fixed" ? cmd_fixed(p, o) : cmd_manifest(p, o);
        llama_backend_free();
        return rc;
    } catch (const std::exception & e) {
        fprintf(stderr, "llama-serve-bench: %s\n", e.what());
        return 1;
    }
}
