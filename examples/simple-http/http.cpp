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

using _http_user_handler = std::function<std::string(const httplib::Request &, httplib::Response &)>;
using _http_server_starter = std::function<void(httplib::Server &)>;
using _http_put_prompt_on_queue = std::function<std::pair<uint64_t, size_t>(std::string, std::string, std::string)>;
using _http_get_prompt_result = std::function<std::pair<std::string, ResponsePlusMetrics>(uint64_t)>;
using _queue_t = std::deque<std::pair<uint64_t, std::string>>;
using _map_t = std::map<uint64_t, std::pair<std::string, ResponsePlusMetrics>>;

std::string _hexify_id(uint64_t id)
{
    std::stringstream ss;
    ss << std::hex << id;
    return ss.str();
}

std::string _remote_addr(const httplib::Request &req)
{
    auto remote_addr = req.remote_addr;
    if (req.has_header("X-Forwarded-For"))
    {
        remote_addr = req.get_header_value("X-Forwarded-For");
    }

    return remote_addr;
}

void _log_request(const httplib::Request &req, const std::string &extra_logging)
{
    HTTP_LOGGER("%s %s %s     %s    %s\n",
                req.method.c_str(),
                req.path.c_str(),
                _remote_addr(req).c_str(),
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

void _http_server_run(
    models_map_t models,
    std::shared_ptr<std::string> *session_ss,
    std::function<std::string()> session_private,
    _http_server_starter go,
    _http_put_prompt_on_queue put_q,
    _http_get_prompt_result get_res)
{
    httplib::Server server;

    if (session_ss)
    {
        server.Get(*(session_ss->get()), _request_wrapper([session_private](const httplib::Request &req, httplib::Response &res)
                                                          {
        auto resp = session_private();
        res.set_content(resp, "application/json");
        return ""; }));
    }

    server.Get("/models", _request_wrapper([models](const httplib::Request &req, httplib::Response &res)
                                           {
        nlohmann::json json(models);
        res.set_content(json.dump(), "application/json");
        return std::string(""); }));

    server.Post("/prompt",
                _request_wrapper([put_q, &models](const httplib::Request &req, httplib::Response &res)
                                 {
        auto parsed_body = nlohmann::json::parse(req.body);
        if (parsed_body.is_discarded() 
            || parsed_body["prompt"].is_null()
            || !parsed_body["prompt"].is_string()
            || parsed_body["model"].is_null()
            || !parsed_body["model"].is_string()) {
            res.status = 400;
            HTTP_LOGGER("Bad JSON body!\n%s", req.body.c_str());
            return std::string("400 Bad Request");
        }

        std::string pre = "";
        std::string post = "";
        auto& model_spec = models[parsed_body["model"]];

        // first, look for JSON-sidecar-configured wrappers
        if (!model_spec.is_null() && !model_spec["promptWrappers"].is_null())
        {
            pre = model_spec["promptWrappers"]["pre"];
            post = model_spec["promptWrappers"]["post"];
        }

        // then, allow user-specified wrappers to override the configured
        if (!parsed_body["promptWrappers"].is_discarded() && parsed_body["promptWrappers"].is_object()) {
            auto& pw = parsed_body["promptWrappers"];
            if (pw["pre"].is_string()) {
                pre = pw["pre"];
            }

            if (pw["post"].is_string()) {
                post = pw["post"];
            }
        }

        std::string prompt = pre + (std::string)parsed_body["prompt"] + post;
        uint64_t new_id = 0;
        size_t q_pos = -1;
        std::tie(new_id, q_pos) = put_q(prompt, parsed_body["model"], _remote_addr(req));

        if (new_id == 0) {
            res.status = 413;
            return std::string("413 Content Too Large");
        }

        nlohmann::json json {
            {"promptId", _hexify_id(new_id)},
            {"queuePosition", q_pos}
        };

        res.set_content(json.dump(), "application/json");
        return _hexify_id(new_id); }));

    server.Get("/prompt/([\\da-f]+)",
               [get_res](const httplib::Request &req, httplib::Response &res)
               {
        std::stringstream ss;
        uint64_t prompt_id;
        ss << std::hex << req.matches[1].str();
        ss >> prompt_id;

        auto response_pair = get_res(prompt_id);

        if (response_pair.first.empty()) {
            res.status = 404;
        } else if (response_pair.second.response.empty()) {
            nlohmann::json json {
                {"queuePosition", response_pair.second.queue_position},
                {"model", response_pair.second.model},
                {"prompt", response_pair.first},
            };
            res.set_content(json.dump(), "application/json");
            res.status = 202;
        }
        else {
            nlohmann::json json {
                {"prompt", response_pair.first},
                {"response", response_pair.second.response},
                {"elapsed_ms", response_pair.second.elapsed_ms},
                {"tokens", response_pair.second.tokens},
                {"model", response_pair.second.model},
                {"ms_per_token", response_pair.second.elapsed_ms / response_pair.second.tokens}
            };
            res.set_content(json.dump(), "application/json");
        }

        return ""; });

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

http_prompt_servicer http_server_run(
    std::string &hostname,
    uint16_t port,
    int32_t context_size,
    models_map_t models,
    std::shared_ptr<std::string> *session_ep,
    llama_timings *total_timings)
{
    std::mutex *q_lock = new std::mutex;
    _queue_t *q = new _queue_t;
    _map_t *m = new _map_t;
    uint64_t *pending_id = new uint64_t(0);
    uint32_t *lifetime_queued = new uint32_t(0);

    if (session_ep)
    {
        auto idA = rng(), idB = rng();
        std::string prefix = "runtime";
        if (*session_ep && (**session_ep).length())
        {
            prefix = **session_ep;
        }
        std::stringstream session_ss;
        session_ss << "/" << prefix << "/" << std::hex << idA << idB;
        *session_ep = std::make_shared<std::string>(session_ss.str());
    }

    std::thread(
        _http_server_run,
        models,
        session_ep,

        // session private handler
        [q, m, pending_id, total_timings, lifetime_queued]()
        {
            std::vector<std::string> q_json;
            for (const auto &q_pair : *q)
            {
                q_json.emplace_back(_hexify_id(q_pair.first));
            }

            nlohmann::json json{
                {"queue", q_json},
                {"totals", {
                               {"prompts", *lifetime_queued},
                               {"eval_ms", total_timings->t_eval_ms},
                               {"load_ms", total_timings->t_load_ms},
                               {"prompt_eval_ms", total_timings->t_p_eval_ms},
                               {"tokens", total_timings->n_sample},
                           }}};

            auto &processed = json["prompts"] = std::map<std::string, nlohmann::json>{};
            for (const auto &outer_pair : *m)
            {
                auto &inner_pair = outer_pair.second;
                auto &metrics = inner_pair.second;
                processed[_hexify_id(outer_pair.first)] = {
                    {"prompt", inner_pair.first},
                    {"model", metrics.model},
                    {"remote_addr", metrics.remote_addr},
                    {"metrics", nlohmann::json{
                                    {"elapsed_ms", metrics.elapsed_ms},
                                    {"tokens", metrics.tokens},
                                    {"queued_time", metrics.queued_iso8601},
                                    {"end_time", metrics.end_iso8601},
                                }},
                };
            }

            if (*pending_id)
            {
                json["pendingId"] = _hexify_id(*pending_id);
            }

            return json.dump();
        },

        // server startup (_http_server_starter)
        [hostname, port](httplib::Server &server)
        {
            server.listen(hostname, port);
        },

        // POST handler to put a prompt on the queue (_http_put_prompt_on_queue)
        [q, q_lock, m, context_size, lifetime_queued](std::string prompt, std::string model, std::string remote_addr)
        {
            if (prompt.length() > (std::size_t)context_size)
            {
                return std::make_pair((long unsigned)0, (size_t)-1);
            }

            auto id = _unique_id(m);
            ResponsePlusMetrics rpm;
            rpm.model = model;
            rpm.remote_addr = remote_addr;
            rpm.queued_iso8601 = iso8601_timestamp();

            {
                std::lock_guard<std::mutex> lg(*q_lock);
                rpm.queue_position = q->size();
                q->push_back(std::make_pair(id, prompt));
                m->emplace(std::make_pair(id, std::make_pair(prompt, rpm)));
            }

            (*lifetime_queued)++;
            return std::make_pair(id, rpm.queue_position);
        },

        // GET promptId handler (_http_get_prompt_result)
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
            ResponsePlusMetrics resp_obj = m->at(*pending_id).second;
            resp_obj.response = *response;
            resp_obj.elapsed_ms = predict_elapsed_ms;
            resp_obj.tokens = num_tokens_predicted;
            resp_obj.end_iso8601 = iso8601_timestamp();
            (*m)[*pending_id] = std::make_pair(m->at(*pending_id).first, resp_obj);
            *pending_id = 0;
        }

        std::string r = "";
        std::string model = "";

        while (!q->size())
        {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }

        {
            std::lock_guard<std::mutex> lg(*q_lock);
            auto q_pair = q->front();
            *pending_id = q_pair.first;
            r = q_pair.second;
            model = m->at(q_pair.first).second.model;
            q->pop_front();

            for (const auto &qele : *q)
            {
                m->at(qele.first).second.queue_position -= 1;
            }
        }

        return ServicerResponse{_hexify_id(*pending_id), r, model};
    };
}