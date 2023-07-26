#pragma once

#include <string>
#include <functional>
#include <map>

#include "deps/json/single_include/nlohmann/json.hpp"

using models_map_t = std::map<std::string, nlohmann::json>;

// blocks until the next prompt is available
// the first parameter must be the response to the *last* prompt; null if no response available (e.g. on first call)
// the second parameter is the total elapsed prediction time, in milliseconds
// the third parameter is the number of tokens processed in the prediction
using http_prompt_servicer = std::function<std::pair<std::string, std::string>(std::string *, float, int)>;

http_prompt_servicer http_server_run(std::string &hostname, uint16_t port, int32_t context_size, models_map_t models);
