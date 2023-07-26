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

#include "deps/popl/include/popl.hpp"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <signal.h>
#endif

std::string run_one_prompt(gpt_params &params)
{
    llama_model *model;
    llama_context *ctx;

    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    if (model == NULL)
    {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        exit(-1);
    }

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx, params.prompt, true);

    const int max_context_size = llama_n_ctx(ctx);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int)tokens_list.size() > max_tokens_list_size)
    {
        fprintf(stderr, "%s: error: prompt too long (%d tokens, max %d)\n",
                __func__, (int)tokens_list.size(), max_tokens_list_size);
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
            fprintf(stderr, "%s : failed to eval\n", __func__);
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

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return outstream.str();
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
    auto model_opt = op.add<popl::Value<std::string>>("m", "model", "Model file to use");
    auto host_opt = op.add<popl::Value<std::string>>("H", "host", "Hostname on which to bind & listen", "localhost");
    auto port_opt = op.add<popl::Value<int>>("p", "port", "Port on which to bind & listen", 42000);
    op.parse(argc, argv);

    if (!model_opt->is_set())
    {
        printf("Must set a model file.\n\n");
        help_opt->set_value(true);
    }

    if (help_opt->is_set())
    {
        std::cout << argv[0] << " " << op.help();
        exit(0);
    }

    gpt_params params;
    params.model = model_opt->value();
    std::string hname = host_opt->value();
    uint16_t port = port_opt->value();

    llama_backend_init(params.numa);
    signal(SIGINT, sighandler);
    auto prompt_servicer = http_server_run(hname, port, params.n_ctx);
    printf("Listening on %s:%d\n", hname.c_str(), port);

    while (true)
    {
        std::string response;
        if (params.prompt.size())
        {
            response = run_one_prompt(params);
            fprintf(stderr, "[%s] >> Response to \"%s\":\n%s\n", iso8601_timestamp().c_str(),
                    params.prompt.c_str(), response.c_str());
        }

        params.prompt = prompt_servicer(response.size() ? &response : NULL, /*FIX THESE ->*/ -1, 0);
        fprintf(stderr, "[%s] >> Prompting with: \"%s\"\n", iso8601_timestamp().c_str(), params.prompt.c_str());
    }

    return 0;
}
