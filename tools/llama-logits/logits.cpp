// The model oracle's reference (architecture 15): llama.cpp's teacher-forced log-probabilities.
//   llama-logits <gguf> <prompts.jsonl> [--first N] [--ngl N] [--threads N] [--batch N] [--ubatch N]
//                [--fa auto|on|off] [--date S] [--mmproj F --embd-dir D]
//                [--image-min-tokens N] [--image-max-tokens N] > dump.jsonl
// Each prompts line is {"prompt": "..."} (tokenised as is) or {"messages": [{"role", "content"}...]}
// (a trailing assistant message without content, a bench workload's timed turn, is dropped; the rest is
// rendered with the GGUF's own template and the generation prompt, as tokdump chat renders it); other
// keys are ignored. Tokens are common_tokenize(add_special, parse_special), exactly tokdump's. The whole
// prompt runs with logits at every position, so each position's distribution is the model's prediction
// given the true prefix (teacher forcing).
// Output, JSON lines: one header object with everything needed to reproduce the run (the pinned tag
// and commit, the GGUF's and prompt file's sha256, the backend and devices, the context parameters),
// then one line per prompt:
//   {"index": i, "prompt": "<text tokenised>", "tokens": [...], "top64": [[[id, logprob] x64] per token]}
// logprob = logit - logsumexp(all logits), so values are normalised over the whole vocabulary; the
// 64 are by logit descending, ties by id. --ngl 0 (the default) keeps every op on the CPU; a GPU run
// (--ngl above the layer count) uses the first GPU device alone, named in the header.
// With --mmproj, a line may also carry "images": [paths relative to the prompts file], one per mtmd
// marker (<__media__>) in its prompt: libmtmd tokenises and encodes them, each image's embedding rows
// are written to <embd-dir>/p<index>-i<k>.f32 (f32, rows x n_embd) and decoded as llama.cpp's
// helpers decode them, and "tokens" holds -1 at those rows, with
// "images": [{"at", "n", "embd", "file", "id"}] (id: mtmd's sha256 of the file). Positions are
// sequential, but on a Qwen-VL model (M-RoPE: mtmd_decode_use_mrope) as llama.cpp's helpers lay them
// out: a text row one position, an image row (t, y, x, 0) = (p0, p0 + i / nx, p0 + i % nx, 0) over
// its merged grid from the image's first position p0, and the rows after it going on from p0 +
// max(nx, ny); such an image also carries "grid": [nx, ny] and "n_pos" (the positions it takes).
// --image-min-tokens and --image-max-tokens bound a dynamic-resolution image's rows (mtmd's).
#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "llama-tools.hpp"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

extern "C" {
#include "hash/sha256/sha256.h"
}

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

constexpr int TOP_K = 64;

static std::string hex(const unsigned char * d, size_t n) {
    std::string s;
    for (size_t i = 0; i < n; ++i) {
        char b[3];
        snprintf(b, sizeof b, "%02x", d[i]);
        s += b;
    }
    return s;
}

// Streamed, so a multi-GB GGUF is never held in memory.
static std::string sha256_file(const std::string & path, uint64_t & bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot read " + path);
    }
    sha256_t h;
    sha256_init(&h);
    std::vector<char> buf(1 << 20);
    bytes = 0;
    while (f) {
        f.read(buf.data(), buf.size());
        sha256_update(&h, reinterpret_cast<const unsigned char *>(buf.data()), f.gcount());
        bytes += f.gcount();
    }
    unsigned char d[SHA256_DIGEST_SIZE];
    sha256_final(&h, d);
    return hex(d, sizeof d);
}

static std::string sha256_str(const std::string & s) {
    unsigned char d[SHA256_DIGEST_SIZE];
    sha256_hash(d, reinterpret_cast<const unsigned char *>(s.data()), s.size());
    return hex(d, sizeof d);
}

struct Args {
    std::string gguf, prompts, fa = "auto", date = "26 Jul 2024", mmproj, embd_dir;
    int first = 0, ngl = 0, threads = 0, batch = 512, ubatch = 512;
    int image_min_tokens = -1, image_max_tokens = -1;  // -1: the projector's own
};

static Args parse_args(int argc, char ** argv) {
    if (argc < 3) {
        throw std::invalid_argument("usage");
    }
    Args a;
    a.gguf    = argv[1];
    a.prompts = argv[2];
    for (int i = 3; i < argc; i += 2) {
        const std::string k = argv[i];
        if (i + 1 >= argc) {
            throw std::invalid_argument("usage");
        }
        const std::string v = argv[i + 1];
        if (k == "--first") {
            a.first = std::stoi(v);
        } else if (k == "--ngl") {
            a.ngl = std::stoi(v);
        } else if (k == "--threads") {
            a.threads = std::stoi(v);
        } else if (k == "--batch") {
            a.batch = std::stoi(v);
        } else if (k == "--ubatch") {
            a.ubatch = std::stoi(v);
        } else if (k == "--fa" && (v == "auto" || v == "on" || v == "off")) {
            a.fa = v;
        } else if (k == "--date") {
            a.date = v;
        } else if (k == "--mmproj") {
            a.mmproj = v;
        } else if (k == "--embd-dir") {
            a.embd_dir = v;
        } else if (k == "--image-min-tokens") {
            a.image_min_tokens = std::stoi(v);
        } else if (k == "--image-max-tokens") {
            a.image_max_tokens = std::stoi(v);
        } else {
            throw std::invalid_argument("usage");
        }
    }
    if (a.first < 0 || a.ngl < 0 || a.threads < 0 || a.batch < 1 || a.ubatch < 1 ||
        a.mmproj.empty() != a.embd_dir.empty() || a.image_min_tokens == 0 || a.image_max_tokens == 0 ||
        a.image_min_tokens < -1 || a.image_max_tokens < -1 ||
        (a.mmproj.empty() && (a.image_min_tokens > 0 || a.image_max_tokens > 0))) {
        throw std::invalid_argument("usage");
    }
    return a;
}

// Logprobs need the full sum: double, since a 150k vocabulary of f32 exps loses digits in f32.
static void top64(const float * logits, int n_vocab, int index, size_t pos, std::vector<int> & ids,
                  std::string & out) {
    double max = -INFINITY;
    for (int i = 0; i < n_vocab; ++i) {
        if (!std::isfinite(logits[i])) {
            throw std::runtime_error("prompt " + std::to_string(index) + " position " + std::to_string(pos) +
                                     ": non-finite logit for token " + std::to_string(i));
        }
        max = std::max(max, (double) logits[i]);
    }
    double sum = 0;
    for (int i = 0; i < n_vocab; ++i) {
        sum += std::exp((double) logits[i] - max);
    }
    const double lse = max + std::log(sum);
    std::iota(ids.begin(), ids.end(), 0);
    const int k = std::min(TOP_K, n_vocab);
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                      [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
    out += '[';
    for (int j = 0; j < k; ++j) {
        char b[64];
        snprintf(b, sizeof b, "%s[%d,%.9g]", j ? "," : "", ids[j], (double) logits[ids[j]] - lse);
        out += b;
    }
    out += ']';
}

static int run(int argc, char ** argv) {
    Args a;
    try {
        a = parse_args(argc, argv);
    } catch (const std::exception &) {
        fprintf(stderr,
                "usage: llama-logits <gguf> <prompts.jsonl> [--first N] [--ngl N] [--threads N] [--batch N]\n"
                "                    [--ubatch N] [--fa auto|on|off] [--date S] [--mmproj F --embd-dir D]\n"
                "                    [--image-min-tokens N] [--image-max-tokens N]\n");
        return 2;
    }
    // Warnings and errors go to stderr; --fa auto's outcome is only told in the log, so keep it for the
    // header.
    static std::string fa_log;
    llama_log_set(
        [](ggml_log_level level, const char * text, void *) {
            if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
                fputs(text, stderr);
            }
            if (strstr(text, "Flash Attention")) {
                fa_log += text;
            }
        },
        nullptr);
    llama_backend_init();

    uint64_t model_bytes = 0;
    const std::string model_sha = sha256_file(a.gguf, model_bytes);
    const std::string prompts   = slurp(a.prompts);
    uint64_t          mmproj_bytes = 0;
    const std::string mmproj_sha   = a.mmproj.empty() ? "" : sha256_file(a.mmproj, mmproj_bytes);

    // One device: none for --ngl 0, so a GPU build still computes everything on the CPU; else the first
    // GPU alone, so a multi-GPU machine cannot split the run across devices the header doesn't name.
    std::string        backend    = "CPU";
    ggml_backend_dev_t devices[2] = { nullptr, nullptr };
    for (size_t i = 0; a.ngl > 0 && !devices[0] && i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d    = ggml_backend_dev_get(i);
        const auto         type = ggml_backend_dev_type(d);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            devices[0] = d;
            backend    = ggml_backend_reg_name(ggml_backend_dev_backend_reg(d));
        }
    }
    if (a.ngl > 0 && !devices[0]) {
        throw std::runtime_error("--ngl " + std::to_string(a.ngl) + " but this build has no GPU device");
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers       = a.ngl;
    mp.devices            = devices;
    llama_model * model = llama_model_load_from_file(a.gguf.c_str(), mp);
    if (!model) {
        throw std::runtime_error("cannot load " + a.gguf);
    }
    // A partial offload would compute most layers on the CPU under a GPU backend's name.
    if (a.ngl > 0 && a.ngl <= llama_model_n_layer(model)) {
        throw std::runtime_error("--ngl must be 0 (CPU) or above the model's " +
                                 std::to_string(llama_model_n_layer(model)) + " layers (all on the GPU)");
    }
    const auto fa_type = a.fa == "on"  ? LLAMA_FLASH_ATTN_TYPE_ENABLED :
                         a.fa == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED :
                                         LLAMA_FLASH_ATTN_TYPE_AUTO;
    const int  threads = a.threads ? a.threads : common_cpu_get_num_math();
    mtmd_context * mctx = nullptr;
    if (!a.mmproj.empty()) {
        mtmd_context_params mp2 = mtmd_context_params_default();
        mp2.use_gpu             = a.ngl > 0;
        mp2.print_timings       = false;
        mp2.n_threads           = threads;
        mp2.flash_attn_type     = fa_type;
        mp2.image_min_tokens    = a.image_min_tokens;
        mp2.image_max_tokens    = a.image_max_tokens;
        mctx                    = mtmd_init_from_file(a.mmproj.c_str(), model, mp2);
        if (!mctx) {
            throw std::runtime_error("cannot load " + a.mmproj);
        }
    }
    const llama_vocab *       vocab   = llama_model_get_vocab(model);
    const int                 n_vocab = llama_vocab_n_tokens(vocab);
    common_chat_templates_ptr tmpls;  // only for messages lines: a raw-text dump needs no template

    struct Prompt {
        std::string              text;
        std::vector<llama_token> tokens;
        json                     images = json::array();
        mtmd_input_chunks *      chunks = nullptr;  // a line with images
    };
    const std::string dir = a.prompts.find('/') == std::string::npos ? "." : a.prompts.substr(0, a.prompts.rfind('/'));

    std::vector<Prompt> ps;
    std::istringstream  in(prompts);
    for (std::string line; std::getline(in, line) && (a.first == 0 || (int) ps.size() < a.first);) {
        const auto j = json::parse(line);
        Prompt     p;
        if (j.contains("messages")) {
            if (!tmpls) {
                tmpls = common_chat_templates_init(model, "");
            }
            p.text = render(tmpls.get(), j.at("messages"), true, a.date);
        } else {
            p.text = j.at("prompt").get<std::string>();
        }
        if (j.contains("images")) {
            if (!mctx) {
                throw std::runtime_error("a line has images: pass --mmproj");
            }
            std::vector<mtmd_bitmap *> bitmaps;
            for (const auto & f : j.at("images")) {
                const std::string path = dir + "/" + f.get<std::string>();
                mtmd_bitmap * b = mtmd_helper_bitmap_init_from_file(mctx, path.c_str(), false,
                                                                     mtmd_helper_init_opt_default()).bitmap;
                if (!b) {
                    throw std::runtime_error("cannot read " + path);
                }
                bitmaps.push_back(b);
            }
            p.chunks = mtmd_input_chunks_init();
            const mtmd_input_text t = { p.text.c_str(), p.text.size(), true, true };
            if (mtmd_tokenize(mctx, p.chunks, &t, const_cast<const mtmd_bitmap **>(bitmaps.data()), bitmaps.size()) != 0) {
                throw std::runtime_error("mtmd_tokenize failed on prompt " + std::to_string(ps.size()));
            }
            for (size_t c = 0; c < mtmd_input_chunks_size(p.chunks); ++c) {
                const mtmd_input_chunk * ch = mtmd_input_chunks_get(p.chunks, c);
                size_t                   n  = mtmd_input_chunk_get_n_tokens(ch);
                if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    const llama_token * tk = mtmd_input_chunk_get_tokens_text(ch, &n);
                    p.tokens.insert(p.tokens.end(), tk, tk + n);
                } else if (c > 0 && mtmd_input_chunk_get_type(mtmd_input_chunks_get(p.chunks, c - 1)) ==
                                        MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                    // A tiled image is one chunk per tile, back to back: the same image.
                    p.images.back()["n"] = p.images.back()["n"].get<size_t>() + n;
                    p.tokens.insert(p.tokens.end(), n, -1);
                } else {
                    const std::string file = "p" + std::to_string(ps.size()) + "-i" + std::to_string(p.images.size()) + ".f32";
                    p.images.push_back({ { "at", p.tokens.size() }, { "n", n }, { "embd", file },
                                         { "file", j.at("images").at(p.images.size()) },
                                         { "id", mtmd_input_chunk_get_id(ch) } });
                    if (mtmd_decode_use_mrope(mctx)) {
                        // The last row's place from position 0: (nx - 1, ny - 1).
                        const mtmd_decoder_pos last =
                            mtmd_image_tokens_get_decoder_pos(mtmd_input_chunk_get_tokens_image(ch), 0, n - 1);
                        p.images.back()["grid"]  = { last.x + 1, last.y + 1 };
                        p.images.back()["n_pos"] = mtmd_input_chunk_get_n_pos(ch);
                    }
                    p.tokens.insert(p.tokens.end(), n, -1);
                }
            }
            for (mtmd_bitmap * b : bitmaps) {
                mtmd_bitmap_free(b);
            }
        } else {
            p.tokens = common_tokenize(vocab, p.text, true, true);
        }
        ps.push_back(std::move(p));
    }
    if (a.first != 0 && (int) ps.size() < a.first) {
        throw std::runtime_error("--first " + std::to_string(a.first) + " but the file has " +
                                 std::to_string(ps.size()) + " prompts");
    }
    size_t longest = 1;
    for (const auto & p : ps) {
        longest = std::max(longest, p.tokens.size());
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = longest;
    cp.n_batch              = std::min<size_t>(a.batch, longest);
    cp.n_ubatch             = std::min<uint32_t>(a.ubatch, cp.n_batch);
    cp.n_seq_max            = 1;
    cp.n_threads = cp.n_threads_batch = threads;
    cp.flash_attn_type = fa_type;
    cp.op_offload      = a.ngl > 0;
    cp.no_perf         = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        throw std::runtime_error("cannot create a context of " + std::to_string(longest) + " tokens");
    }

    json registered = json::array();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        registered.push_back({ { "name", ggml_backend_dev_name(d) },
                               { "description", ggml_backend_dev_description(d) },
                               { "backend", ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)) } });
    }
    char desc[256];
    llama_model_desc(model, desc, sizeof desc);
    const bool mrope = mctx && mtmd_decode_use_mrope(mctx);
    const json header = {
        { "larch_oracle", 1 },
        { "llama_cpp_tag", LLAMA_TAG },
        { "llama_cpp_commit", llama_commit() },
        { "compiler", llama_compiler() },
        { "build_target", llama_build_target() },
        { "model_sha256", model_sha },
        { "model_bytes", model_bytes },
        { "model_desc", desc },
        { "n_vocab", n_vocab },
        { "prompts_sha256", sha256_str(prompts) },
        { "prompts", ps.size() },
        { "first", a.first },
        { "date_string", a.date },
        { "add_special", true },
        { "parse_special", true },
        { "top_k", TOP_K },
        { "backend", backend },
        { "device", devices[0] ? ggml_backend_dev_name(devices[0]) : "CPU" },
        { "devices", registered },
        { "n_gpu_layers", a.ngl },
        { "op_offload", cp.op_offload },
        { "n_ctx", llama_n_ctx(ctx) },
        { "n_batch", llama_n_batch(ctx) },
        { "n_ubatch", llama_n_ubatch(ctx) },
        { "n_threads", cp.n_threads },
        { "flash_attn", a.fa },
        { "flash_attn_resolved", a.fa != "auto"                                   ? a.fa :
                                 fa_log.find("not supported") != std::string::npos ? "off" :
                                 fa_log.find("enabled") != std::string::npos       ? "on" :
                                                                                     "unknown" },
        { "type_k", ggml_type_name(cp.type_k) },
        { "type_v", ggml_type_name(cp.type_v) },
        { "system_info", llama_print_system_info() },
        { "mmproj_sha256", mmproj_sha },
        { "positions", mrope ? "mrope" : "sequential" },
        { "image_min_tokens", a.image_min_tokens },
        { "image_max_tokens", a.image_max_tokens },
    };
    std::cout << header.dump() << "\n";

    std::vector<int> ids(n_vocab);
    const int        n_batch = llama_n_batch(ctx);
    const int        n_embd  = llama_model_n_embd(model);
    llama_batch      batch   = llama_batch_init(n_batch, 0, 1);
    llama_batch      ebatch  = llama_batch_init(n_batch, mctx ? n_embd : 0, 1);
    llama_batch_ext * xbatch = mrope ? llama_batch_ext_init(ctx) : nullptr;
    for (size_t pi = 0; pi < ps.size(); ++pi) {
        const auto & p = ps[pi];
        llama_memory_clear(llama_get_memory(ctx), true);
        json head = { { "index", pi }, { "prompt", p.text }, { "tokens", p.tokens } };
        if (p.chunks) {
            head["images"] = p.images;
        }
        std::string out = head.dump();
        out.pop_back();  // the closing brace: top64 follows
        out += ",\"top64\":[";
        // Rows [start, end) of the prompt at positions start..: token ids, or embedding rows `embd`.
        auto decode = [&](size_t start, size_t end, const float * embd) {
            for (size_t at = start; at < end; at += n_batch) {
                const size_t stop = std::min(end, at + n_batch);
                llama_batch & b   = embd ? ebatch : batch;
                b.n_tokens        = (int32_t) (stop - at);
                for (size_t t = at; t < stop; ++t) {
                    const size_t i = t - at;
                    if (embd) {
                        std::copy_n(embd + (t - start) * n_embd, n_embd, b.embd + i * n_embd);
                    } else {
                        b.token[i] = p.tokens[t];
                    }
                    b.pos[i]       = (llama_pos) t;
                    b.n_seq_id[i]  = 1;
                    b.seq_id[i][0] = 0;
                    b.logits[i]    = true;
                }
                if (llama_decode(ctx, b) != 0) {
                    throw std::runtime_error("llama_decode failed on prompt " + std::to_string(pi));
                }
                for (size_t t = at; t < stop; ++t) {
                    if (t) {
                        out += ',';
                    }
                    const float * l = llama_get_logits_ith(ctx, (int32_t) (t - at));
                    if (!l) {
                        throw std::runtime_error("no logits for prompt " + std::to_string(pi));
                    }
                    top64(l, n_vocab, (int) pi, t, ids, out);
                }
            }
        };
        // Rows [start, end) at M-RoPE positions from p0 through the extended batch: a text row one
        // position (p0 on), an image row `img`'s (t, y, x, z) from p0.
        auto decode_mrope = [&](size_t start, size_t end, const float * embd, const mtmd_image_tokens * img,
                                llama_pos p0) {
            for (size_t at = start; at < end; at += n_batch) {
                const size_t stop = std::min(end, at + n_batch);
                llama_batch_ext_clear(xbatch);
                for (size_t t = at; t < stop; ++t) {
                    const size_t  r = t - start;
                    const int32_t i = embd ? llama_batch_ext_add_embd(xbatch, 0, { embd + r * n_embd, 1, (size_t) n_embd }) :
                                             llama_batch_ext_add_token(xbatch, 0, p.tokens[t]);
                    llama_pos pos[4] = { p0 + (llama_pos) r, 0, 0, 0 };
                    if (img) {
                        const mtmd_decoder_pos d = mtmd_image_tokens_get_decoder_pos(img, p0, r);
                        pos[0]                   = (llama_pos) d.t;
                        pos[1]                   = (llama_pos) d.y;
                        pos[2]                   = (llama_pos) d.x;
                        pos[3]                   = (llama_pos) d.z;
                    }
                    if (i != (int32_t) (t - at) || !llama_batch_ext_set_pos(xbatch, i, pos) ||
                        !llama_batch_ext_set_output_logits(xbatch, i, true)) {
                        throw std::runtime_error("cannot batch row " + std::to_string(t) + " of prompt " +
                                                 std::to_string(pi));
                    }
                }
                if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, xbatch) != 0) {
                    throw std::runtime_error("llama_process failed on prompt " + std::to_string(pi));
                }
                for (size_t t = at; t < stop; ++t) {
                    if (t) {
                        out += ',';
                    }
                    const float * l = llama_get_logits_ith(ctx, (int32_t) (t - at));
                    if (!l) {
                        throw std::runtime_error("no logits for prompt " + std::to_string(pi));
                    }
                    top64(l, n_vocab, (int) pi, t, ids, out);
                }
            }
        };
        if (!p.chunks) {
            decode(0, p.tokens.size(), nullptr);
        } else if (mrope) {
            size_t    at = 0, k = 0;
            llama_pos n_past = 0;
            for (size_t c = 0; c < mtmd_input_chunks_size(p.chunks); ++c) {
                const mtmd_input_chunk * ch = mtmd_input_chunks_get(p.chunks, c);
                const size_t             n  = mtmd_input_chunk_get_n_tokens(ch);
                if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    decode_mrope(at, at + n, nullptr, nullptr, n_past);
                } else {
                    if (mtmd_encode_chunk(mctx, ch) != 0) {
                        throw std::runtime_error("mtmd_encode_chunk failed on prompt " + std::to_string(pi));
                    }
                    const float *     e    = mtmd_get_output_embd(mctx);
                    const std::string path = a.embd_dir + "/" + p.images[k++].at("embd").get<std::string>();
                    std::ofstream     f(path, std::ios::binary | std::ios::trunc);
                    f.write(reinterpret_cast<const char *>(e), std::streamsize(n * n_embd * sizeof(float)));
                    if (!f) {
                        throw std::runtime_error("cannot write " + path);
                    }
                    decode_mrope(at, at + n, e, mtmd_input_chunk_get_tokens_image(ch), n_past);
                }
                n_past += mtmd_input_chunk_get_n_pos(ch);
                at += n;
            }
        } else {
            size_t at = 0, k = 0;
            bool   image = false;  // the last chunk was an image's
            for (size_t c = 0; c < mtmd_input_chunks_size(p.chunks); ++c) {
                const mtmd_input_chunk * ch = mtmd_input_chunks_get(p.chunks, c);
                const size_t             n  = mtmd_input_chunk_get_n_tokens(ch);
                if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    decode(at, at + n, nullptr);
                    image = false;
                } else {
                    if (mtmd_encode_chunk(mctx, ch) != 0) {
                        throw std::runtime_error("mtmd_encode_chunk failed on prompt " + std::to_string(pi));
                    }
                    const float * e = mtmd_get_output_embd(mctx);
                    // An image's tiles append to its one file.
                    const size_t      img  = image ? k - 1 : k++;
                    const std::string path = a.embd_dir + "/" + p.images[img].at("embd").get<std::string>();
                    std::ofstream f(path, std::ios::binary | (image ? std::ios::app : std::ios::trunc));
                    image = true;
                    f.write(reinterpret_cast<const char *>(e), std::streamsize(n * n_embd * sizeof(float)));
                    if (!f) {
                        throw std::runtime_error("cannot write " + path);
                    }
                    decode(at, at + n, e);
                }
                at += n;
            }
        }
        std::cout << out << "]}\n";
    }
    std::cout.flush();
    if (!std::cout) {
        throw std::runtime_error("write failed");
    }
    llama_batch_free(batch);
    llama_batch_free(ebatch);
    if (xbatch) {
        llama_batch_ext_free(xbatch);
    }
    for (auto & p : ps) {
        mtmd_input_chunks_free(p.chunks);
    }
    mtmd_free(mctx);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

int main(int argc, char ** argv) {
    // A bad prompt line or a failed decode ends the run with its message, not an abort.
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "llama-logits: %s\n", e.what());
        return 1;
    }
}
