// llama.cpp's tokens and chat renders, the reference larch-model's tests compare against:
//   tokdump tokens <gguf> <prompts.jsonl>
//       one {"prompt":"...","tokens":[...]} line per {"prompt": "..."} line, the tokens from
//       llama_tokenize(add_special, parse_special): the --ref of `larch tokcheck`
//   tokdump chat <gguf> <template.jinja | -> <conversation.json> <date_string> [true | false]
//       for every prefix of the conversation that ends in a user or tool message, and for the whole of it,
//       one {"messages":k,"add_generation_prompt":b,"prompt":"..."} line from common_chat_templates_apply;
//       `-` renders the GGUF's own template. The conversation is a list of messages, or
//       {"tools": [...], "messages": [...]} for a render with tools (llama-tools.hpp's `render`). The last
//       argument sets `enable_thinking`, and each line records it as "enable_thinking"; without it the
//       render has llama.cpp's default, true, and the lines have no such key
#include "chat.h"
#include "common.h"
#include "llama-tools.hpp"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

using json = nlohmann::ordered_json;

static int run(int argc, char ** argv) {
    const std::string mode     = argc > 1 ? argv[1] : "";
    const std::string thinking = argc == 7 ? argv[6] : "";
    if (!((mode == "tokens" && argc == 4) ||
          (mode == "chat" && (argc == 6 || (argc == 7 && (thinking == "true" || thinking == "false")))))) {
        fprintf(stderr, "usage: tokdump tokens <gguf> <prompts.jsonl>\n"
                        "       tokdump chat <gguf> <template.jinja | -> <conversation.json> <date_string> "
                        "[true | false]\n");
        return 2;
    }
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[2], mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", argv[2]);
        return 1;
    }
    if (mode == "tokens") {
        std::ifstream in(argv[3]);
        if (!in) {
            throw std::runtime_error(std::string("cannot read ") + argv[3]);
        }
        for (std::string line; std::getline(in, line);) {
            const auto text = json::parse(line).at("prompt").get<std::string>();
            const auto ids  = common_tokenize(llama_model_get_vocab(model), text, true, true);
            std::cout << json{ { "prompt", text }, { "tokens", ids } }.dump() << "\n";
        }
    } else {
        auto tmpls = common_chat_templates_init(model, std::string(argv[3]) == "-" ? "" : slurp(argv[3]));
        const auto file  = json::parse(slurp(argv[4]));
        const auto conv  = file.is_array() ? file : file.at("messages");
        const auto tools = file.is_array() ? json::array() : file.at("tools");
        for (size_t k = 1; k <= conv.size(); ++k) {
            const bool gen = conv[k - 1].at("role") == "user" || conv[k - 1].at("role") == "tool";
            if (!gen && k != conv.size()) {
                continue;
            }
            const json prefix(conv.begin(), conv.begin() + k);
            json       line{ { "messages", k }, { "add_generation_prompt", gen } };
            if (!thinking.empty()) {
                line["enable_thinking"] = thinking == "true";
            }
            line["prompt"] = render(tmpls.get(), prefix, gen, argv[5], tools, thinking != "false");
            std::cout << line.dump() << "\n";
        }
    }
    llama_model_free(model);
    return 0;
}

int main(int argc, char ** argv) {
    // A bad prompt line or a template that raises ends the run with its message, not an abort.
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "tokdump: %s\n", e.what());
        return 1;
    }
}
