#pragma once

#include <string>
#include <functional>
#include <map>

#include "deps/json/single_include/nlohmann/json.hpp"

#define HTTP_LOGGER(fmt_str, ...) fprintf(stdout, "[%s] " fmt_str, iso8601_timestamp().c_str(), ##__VA_ARGS__)

using models_map_t = std::map<std::string, nlohmann::json>;

enum QueuePriority
{
    LOW = -128,
    NORMAL = 0,
    HIGH = 128,
};

struct QueueElement
{
    uint64_t id;
    int64_t queued_ts_ms;
    std::string prompt;
    QueuePriority priority;
    uint mirostat;
};

struct ServicerResponse
{
    std::string id;
    std::string prompt;
    std::string model;
    uint mirostat;
};

struct ResponsePlusMetrics
{
    std::string response = "";
    float elapsed_ms = -1.0;
    int tokens = -1;
    std::string model = "";
    std::string remote_addr = "";
    std::string queued_iso8601 = "";
    std::string end_iso8601 = "";
};

struct KeyedRequestAuditLog
{
    uint64_t count = 0;
    struct
    {
        std::string remote_addr = "";
        std::string path = "";
    } last;
};

// Any higher value implies those below it (except "None", obviously).
// Put another way: anything higher-valued than the current setting will
// *not* require authorization.
// XXX: damn, should probably be a bitfield...
enum AuthLevel
{
    None = 0,
    HighPriority = 1,
    Runtime = 2,
    Default = Runtime,
    POSTPrompt = 252,
    GETPromptById = 253,
    All = 254
};

struct AuthOptions
{
    // anything but "None" requires `keys` be set & valid
    AuthLevel level = AuthLevel::None;
    std::map<std::string, KeyedRequestAuditLog> *keys = nullptr;
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
    struct llama_timings *total_timings,
    AuthOptions auth_options);
