// What tokdump and llama-logits share, so their prompts and renders cannot drift apart.
#pragma once

#include "chat.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

inline std::string slurp(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot read " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// `messages` ([{"role", "content"}...]) and `tools` through the template as llama-server's --jinja renders
// them, with `date_string` for templates that print the date and `enable_thinking` where llama-server
// puts a request's flag. A message may carry larch-model's
// `tool_calls` ([{"name", "arguments", "id"}]) and `tool_call_id`; `tools` are OpenAI function specs.
// Arguments and parameters go in as llama-server reads them: keys in the caller's order, and string
// arguments as they are. A bench workload's last message is its timed assistant turn, which has no content
// (bench/README): the prompt is every message before it.
// The rendered prompt as one string: b11458 returns it as one, later tags as parts (common_chat_prompt::str).
inline std::string prompt_text(const std::string & p) {
    return p;
}

template <class P> inline auto prompt_text(const P & p) -> decltype(p.str()) {
    return p.str();
}

inline std::string render(const common_chat_templates * tmpls, const nlohmann::ordered_json & messages,
                          bool add_generation_prompt, const std::string & date,
                          const nlohmann::ordered_json & tools           = nlohmann::ordered_json::array(),
                          bool                           enable_thinking = true) {
    common_chat_templates_inputs in;
    size_t n = messages.size();
    if (n > 0 && messages[n - 1].at("role") == "assistant" && !messages[n - 1].contains("content")) {
        --n;
    }
    for (size_t i = 0; i < n; ++i) {
        const auto &    m = messages[i];
        common_chat_msg msg;
        msg.role    = m.at("role");
        msg.content = m.at("content");
        for (const auto & c : m.value("tool_calls", nlohmann::ordered_json::array())) {
            const auto & args = c.at("arguments");
            msg.tool_calls.push_back(
                { c.at("name"), args.is_string() ? args.get<std::string>() : args.dump(), c.value("id", "") });
        }
        msg.tool_call_id = m.value("tool_call_id", "");
        in.messages.push_back(msg);
    }
    for (const auto & t : tools) {
        const auto & f = t.at("function");
        in.tools.push_back(
            { f.at("name"), f.value("description", ""), f.value("parameters", nlohmann::ordered_json::object()).dump() });
    }
    in.use_jinja                            = true;
    in.add_generation_prompt                = add_generation_prompt;
    in.enable_thinking                      = enable_thinking;
    in.chat_template_kwargs["date_string"] = nlohmann::ordered_json(date).dump();
    return prompt_text(common_chat_templates_apply(tmpls, in).prompt);
}
