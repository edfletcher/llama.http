#include "http.h"
#include "common.h"

#include "deps/cpp-httplib/httplib.h"
#include "deps/json/single_include/nlohmann/json.hpp"

#include <thread>
#include <mutex>
#include <chrono>
#include <sstream>
#include <ctime>
#include <vector>
#include <random>
#include <iomanip>

std::mt19937_64 rng(time(NULL));

struct ResponsePlusMetrics
{
    std::string response = "";
    float elapsed_ms = -1.0;
    int tokens = -1;
};

using _http_user_handler = std::function<std::string(const httplib::Request &, httplib::Response &)>;
using _http_server_starter = std::function<void(httplib::Server &)>;
using _http_put_prompt_on_queue = std::function<uint64_t(std::string)>;
using _http_get_prompt_result = std::function<std::pair<std::string, ResponsePlusMetrics>(uint64_t)>;
using _queue_t = std::deque<std::pair<uint64_t, std::string>>;
using _map_t = std::map<uint64_t, std::pair<std::string, ResponsePlusMetrics>>;

void _log_request(const httplib::Request &req, const std::string &extra_logging)
{
    auto remote_addr = req.remote_addr;
    if (req.has_header("X-Forwarded-For"))
    {
        remote_addr = req.get_header_value("X-Forwarded-For");
    }

    fprintf(stderr, "[%s] %s %s %s     %s    %s\n",
            iso8601_timestamp().c_str(),
            req.method.c_str(),
            req.path.c_str(),
            remote_addr.c_str(),
            req.body.c_str(),
            extra_logging.c_str());
}

httplib::Server::Handler _request_wrapper(_http_user_handler user_handler)
{
    return [user_handler](const httplib::Request &req, httplib::Response &res)
    {
        auto extra_logging = user_handler(req, res);
        _log_request(req, extra_logging);
    };
}

void _http_server_run(_http_server_starter go, _http_put_prompt_on_queue put_q, _http_get_prompt_result get_res)
{
    httplib::Server server;

    server.Post("/prompt",
                _request_wrapper([put_q](const httplib::Request &req, httplib::Response &res)
                                 {
        auto new_id = put_q(req.body);
        if (new_id == 0) {
            res.status = 413;
            return std::string("");
        }
        std::stringstream s;
        s << std::hex << new_id;
        res.set_content(s.str() + "\n", "text/plain");
        return s.str(); }));

    server.Get("/prompt/([\\da-f]+)",
               _request_wrapper([get_res](const httplib::Request &req, httplib::Response &res)
                                {
        std::stringstream ss;
        uint64_t prompt_id;
        ss << std::hex << req.matches[1].str();
        ss >> prompt_id;

        auto response_pair = get_res(prompt_id);

        if (response_pair.first.empty()) {
            res.status = 404;
        } else if (response_pair.second.response.empty()) {
            res.status = 202;
        }
        else {
            nlohmann::json json {
                {"prompt", response_pair.first},
                {"response", response_pair.second.response},
                {"elapsed_ms", response_pair.second.elapsed_ms},
                {"tokens", response_pair.second.tokens},
                {"ms_per_token", response_pair.second.elapsed_ms / response_pair.second.tokens}
            };
            res.set_content(json.dump(), "application/json");
        }

        return ""; }));

    go(server);
}

uint_fast64_t _unique_id(_map_t *m)
{
    auto try_id = rng();
    while (m->find(try_id) != m->end())
    {
        try_id = rng();
    }
    return try_id;
}

http_prompt_servicer http_server_run(std::string &hostname, uint16_t port, int32_t context_size)
{
    std::mutex *q_lock = new std::mutex;
    _queue_t *q = new _queue_t;
    _map_t *m = new _map_t;
    uint64_t *pending_id = new uint64_t(0);

    std::thread(
        _http_server_run,
        [hostname, port](httplib::Server &server)
        {
            server.listen(hostname, port);
        },
        [q, q_lock, m, context_size](std::string prompt)
        {
            if (prompt.length() > (std::size_t)context_size)
            {
                return (long unsigned)0;
            }
            std::lock_guard<std::mutex> lg(*q_lock);
            auto id = _unique_id(m);
            q->push_back(std::make_pair(id, prompt));
            m->emplace(std::make_pair(id, std::make_pair(prompt, ResponsePlusMetrics())));
            return id;
        },
        [m](uint64_t id)
        {
            auto ele = m->find(id);
            if (ele == m->end())
            {
                return std::make_pair(std::string(""), ResponsePlusMetrics());
            }

            return ele->second;
        })
        .detach();

    return [hostname, port, q, q_lock, pending_id, m](std::string *response, float predict_elapsed_ms = -1.0, int num_tokens_predicted = -1)
    {
        if (response && *pending_id > 0)
        {
            ResponsePlusMetrics resp_obj;
            resp_obj.response = *response;
            resp_obj.elapsed_ms = predict_elapsed_ms;
            resp_obj.tokens = num_tokens_predicted;
            (*m)[*pending_id] = std::make_pair(m->at(*pending_id).first, resp_obj);
            *pending_id = 0;
        }

        std::string r = "";

        while (!q->size())
        {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }

        {
            std::lock_guard<std::mutex> lg(*q_lock);
            auto q_pair = q->front();
            *pending_id = q_pair.first;
            r = q_pair.second;
            q->pop_front();
        }

        return r;
    };
}