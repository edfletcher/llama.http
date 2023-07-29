#pragma once

#include <string>
#include <functional>
#include <map>

#include "deps/json/single_include/nlohmann/json.hpp"

#define HTTP_LOGGER(fmt_str, ...) fprintf(stdout, "[%s] " fmt_str, iso8601_timestamp().c_str(), ##__VA_ARGS__)

using models_map_t = std::map<std::string, nlohmann::json>;

struct ServicerResponse
{
    std::string id;
    std::string prompt;
    std::string model;
};

struct ResponsePlusMetrics
{
    std::string response = "";
    float elapsed_ms = -1.0;
    int tokens = -1;
    size_t queue_position = -1;
    std::string model = "";
    std::string remote_addr = "";
    std::string queued_iso8601 = "";
    std::string end_iso8601 = "";
};

// blocks until the next prompt is available
// the first parameter must be the response to the *last* prompt; null if no response available (e.g. on first call)
// the second parameter is the total elapsed prediction time, in milliseconds
// the third parameter is the number of tokens processed in the prediction
using http_prompt_servicer = std::function<ServicerResponse(std::string *, float, int)>;

http_prompt_servicer http_server_run(
    std::string &hostname,
    uint16_t port,
    int32_t context_size,
    models_map_t models,
    // set to nullptr to disable the session private endpoint entirely
    std::shared_ptr<std::string> *session_ep,
    struct llama_timings *total_timings);
