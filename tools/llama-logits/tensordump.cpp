// Named intermediate tensors of one llama.cpp forward pass, for finding where a port first departs from the
// reference (the gate stays llama-logits' logits):
//   tensordump <gguf> <prompts.jsonl> <index> <out dir> [--fa on|off] <name regex>...
// runs prompt <index> of the file (tokenised as llama-logits does) as one batch on the CPU and writes each f32
// graph tensor whose name matches a regex to <out dir>/<name>.f32 (its data in ggml order, ne0 fastest, made
// contiguous), listing "<name>\t<ne0>,<ne1>,<ne2>,<ne3>" lines in <out dir>/index.tsv. A name the graph computes
// twice keeps its last value.
#include "common.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

struct dump {
    std::vector<std::regex> names;
    std::string             dir;
    std::vector<uint8_t>    buf;
    FILE *                  index = nullptr;
};

static bool on_tensor(ggml_tensor * t, bool ask, void * user) {
    auto * d     = static_cast<dump *>(user);
    bool   match = false;
    for (const auto & re : d->names) {
        match = match || std::regex_search(t->name, re);
    }
    if (ask) {
        return match;
    }
    if (!match || t->type != GGML_TYPE_F32) {
        return true;
    }
    const size_t n = ggml_nelements(t);
    d->buf.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, d->buf.data(), 0, d->buf.size());
    std::vector<float> out(n);
    size_t             i = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    const size_t at = i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                    std::memcpy(&out[i++], d->buf.data() + at, 4);
                }
            }
        }
    }
    std::ofstream(d->dir + "/" + t->name + ".f32", std::ios::binary)
        .write(reinterpret_cast<const char *>(out.data()), out.size() * 4);
    std::fprintf(d->index, "%s\t%lld,%lld,%lld,%lld\n", t->name, (long long) t->ne[0], (long long) t->ne[1],
                 (long long) t->ne[2], (long long) t->ne[3]);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: tensordump <gguf> <prompts.jsonl> <index> <out dir> [--fa on|off] <regex>...\n");
        return 2;
    }
    dump d;
    d.dir                     = argv[4];
    llama_flash_attn_type fa  = LLAMA_FLASH_ATTN_TYPE_AUTO;
    for (int i = 5; i < argc; ++i) {
        if (std::string(argv[i]) == "--fa" && i + 1 < argc) {
            fa = std::string(argv[++i]) == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
            continue;
        }
        d.names.emplace_back(argv[i]);
    }
    std::ifstream f(argv[2]);
    std::string   line;
    for (int i = 0; i <= std::stoi(argv[3]) && std::getline(f, line); ++i) {
    }
    const std::string prompt = nlohmann::json::parse(line).at("prompt");

    llama_backend_init();
    auto mp         = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return 1;
    }
    auto cp            = llama_context_default_params();
    cp.n_ctx           = 512;
    cp.n_batch         = 512;
    cp.n_ubatch        = 512;
    cp.n_threads       = 1;
    cp.n_threads_batch = 1;
    cp.flash_attn_type = fa;
    cp.cb_eval         = on_tensor;
    cp.cb_eval_user_data = &d;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        return 1;
    }
    d.index = std::fopen((d.dir + "/index.tsv").c_str(), "w");
    const std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true, true);
    llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data()), (int32_t) tokens.size());
    const int rc = llama_decode(ctx, batch);
    std::fclose(d.index);
    llama_free(ctx);
    llama_model_free(model);
    return rc;
}
