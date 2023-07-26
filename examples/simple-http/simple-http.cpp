// adapted from examples/simple/simple.cpp
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common.h"
#include "llama.h"
#include "build-info.h"
#include "http.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <experimental/filesystem>

#include "deps/popl/include/popl.hpp"
#include "deps/json/single_include/nlohmann/json.hpp"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <signal.h>
#endif

namespace fs = std::experimental::filesystem;

std::string run_one_prompt(gpt_params &params, nlohmann::json model_spec, struct llama_timings *timings = nullptr)
{
    llama_model *model;
    llama_context *ctx;

    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    if (model == nullptr)
    {
        LOGGER("error: unable to load model\n");
        exit(-1);
    }

    std::string pre = "";
    std::string post = "";

    if (!model_spec.is_null() && !model_spec["promptWrappers"].is_null())
    {
        pre = model_spec["promptWrappers"]["pre"];
        post = model_spec["promptWrappers"]["post"];
    }

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx, pre + params.prompt + post, true);

    const int max_context_size = llama_n_ctx(ctx);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int)tokens_list.size() > max_tokens_list_size)
    {
        LOGGER("error: prompt too long (%d tokens, max %d)\n",
               (int)tokens_list.size(), max_tokens_list_size);
        return "";
    }

    // The LLM keeps a contextual cache memory of previous token evaluation.
    // Usually, once this cache is full, it is required to recompute a compressed context based on previous
    // tokens (see "infinite text generation via context swapping" in the main example), but in this minimalist
    // example, we will just stop the loop once this cache is full or once an end of stream is detected.

    std::stringstream outstream;
    while (llama_get_kv_cache_token_count(ctx) < max_context_size)
    {
        if (llama_eval(ctx, tokens_list.data(), tokens_list.size(), llama_get_kv_cache_token_count(ctx), params.n_threads))
        {
            LOGGER("failed to eval\n");
            return "";
        }

        tokens_list.clear();
        llama_token new_token_id = 0;

        auto logits = llama_get_logits(ctx);
        auto n_vocab = llama_n_vocab(ctx); // the size of the LLM vocabulary (in tokens)

        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);

        for (llama_token token_id = 0; token_id < n_vocab; token_id++)
        {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }

        llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

        // Select it using the "Greedy sampling" method :
        new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

        // is it an end of stream ?
        if (new_token_id == llama_token_eos())
        {
            break;
        }

        outstream << llama_token_to_str(ctx, new_token_id);

        // Push this new token for next evaluation :
        tokens_list.push_back(new_token_id);
    }

    if (timings)
    {
        auto local_timings = llama_get_timings(ctx);
        memcpy(timings, &local_timings, sizeof(struct llama_timings));
    }

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return outstream.str();
}

models_map_t discover_valid_models(std::string model_path)
{
    std::vector<fs::path> bins;
    std::map<std::string, fs::path> jsons;
    for (const fs::directory_entry &entry : fs::directory_iterator{model_path})
    {
        auto path_ent = entry.path();
        if (path_ent.extension() == ".bin")
        {
            bins.push_back(entry.path());
        }
        else if (path_ent.extension() == ".json")
        {
            jsons.emplace(std::make_pair(path_ent.filename(), path_ent));
        }
    }

    models_map_t valid_models;
    for (const auto &bin : bins)
    {
        auto expect_json_name = bin.filename().string() + ".json";
        auto have_json = jsons.find(expect_json_name);
        if (have_json != jsons.end())
        {
            std::ifstream json_read{have_json->second.string()};
            auto json_parsed = nlohmann::json::parse(json_read, nullptr, false);
            if (!json_parsed.is_discarded() &&
                !json_parsed["displayName"].is_null() &&
                !json_parsed["sourceURL"].is_null())
            {
                valid_models.emplace(std::make_pair(bin.filename().string(), json_parsed));
                LOGGER("Found valid model %s\n", bin.filename().string().c_str());
            }
        }
    }

    return valid_models;
}

void sighandler(int signal)
{
    (void)signal;
    exit(0);
}

int main(int argc, char **argv)
{
    popl::OptionParser op("allowed options");
    auto help_opt = op.add<popl::Switch>("h", "help", "This help");
    auto model_opt = op.add<popl::Value<std::string>>("m", "model-path", "Path to model binaries & their sidecar JSONs");
    auto host_opt = op.add<popl::Value<std::string>>("H", "host", "Hostname on which to bind & listen", "localhost");
    auto port_opt = op.add<popl::Value<int>>("p", "port", "Port on which to bind & listen", 42000);
    auto ctx_sz_opt = op.add<popl::Value<int>>("c", "context-size", "Set the model's context size (in tokens)", 2048);
    op.parse(argc, argv);

    if (!model_opt->is_set())
    {
        LOGGER("Must set a model file.\n\n");
        help_opt->set_value(true);
    }

    if (help_opt->is_set())
    {
        std::cout << argv[0] << " " << op.help();
        exit(0);
    }

    auto models = discover_valid_models(model_opt->value());

    if (!models.size())
    {
        LOGGER("No valid models found in '%s'!\n", model_opt->value().c_str());
    }

    gpt_params params;
    params.n_ctx = ctx_sz_opt->value();
    std::string hname = host_opt->value();
    uint16_t port = port_opt->value();

    llama_backend_init(params.numa);
    signal(SIGINT, sighandler);
    LOGGER("Using context size of %d\n", params.n_ctx);
    auto prompt_servicer = http_server_run(hname, port, params.n_ctx, models);
    LOGGER("Listening on %s:%d\n", hname.c_str(), port);

    std::string model;
    while (true)
    {
        std::string response;
        struct llama_timings timings;
        bzero(&timings, sizeof(struct llama_timings));

        if (params.prompt.size())
        {
            response = run_one_prompt(params, models[model], &timings);
            LOGGER("Response to \"%s\":\n%s\n", params.prompt.c_str(), response.c_str());
        }

        std::string prompt;
        std::tie(prompt, model) = prompt_servicer(
            response.size() ? &response : nullptr,
            (timings.t_end_ms - timings.t_start_ms),
            timings.n_sample);

        params.prompt = prompt;
        params.model = fs::path{model_opt->value()} / model;
        LOGGER("Using model %s, prompting with: \"%s\"\n", model.c_str(), params.prompt.c_str());
    }

    return 0;
}
