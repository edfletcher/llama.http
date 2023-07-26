#pragma once

#include <string>
#include <functional>

// blocks until the next prompt is available
// the first parameter must be the response to the *last* prompt; null if no response available (e.g. on first call)
// the second parameter is the total elapsed prediction time, in milliseconds
// the third parameter is the number of tokens processed in the prediction
using http_prompt_servicer = std::function<std::string(std::string *, float, int)>;

http_prompt_servicer http_server_run(std::string &hostname, uint16_t port, int32_t context_size);
