#include "http.h"
#include "common.h"

#include "deps/cpp-httplib/httplib.h"
#include "deps/json/single_include/nlohmann/json.hpp"
#include "deps/cppcodec/cppcodec/base64_rfc4648.hpp"

#include <thread>
#include <mutex>
#include <chrono>
#include <sstream>
#include <ctime>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>

std::mt19937_64 rng(time(NULL));

struct _http_get_prompt_result_return
{
    std::string prompt;
    ResponsePlusMetrics rpm;
    ssize_t queue_position;
};

using _http_user_handler = std::function<std::string(const httplib::Request &, httplib::Response &)>;
using _http_server_starter = std::function<void(httplib::Server &)>;
using _http_put_prompt_on_queue = std::function<std::pair<uint64_t, ssize_t>(std::string, std::string, std::string, QueuePriority, uint)>;
using _http_get_prompt_result = std::function<_http_get_prompt_result_return(uint64_t)>;

// the actual queue is a priority queue with QueueElementCmp as the comparator function
// std::priority_queue isn't used because it doesn't allow access to the underlying container
using _queue_t = std::deque<QueueElement>;
using _map_t = std::map<uint64_t, std::pair<std::string, ResponsePlusMetrics>>;

bool QueueElementCmp(const QueueElement &us, const QueueElement &them)
{
    if (us.priority == them.priority)
    {
        return us.queued_ts_ms > them.queued_ts_ms;
    }

    return us.priority < them.priority;
}

// expects `q` already be a heap! -1 returned if `id` is not found in `q`.
ssize_t QueueElementPosById(uint64_t id, const _queue_t *q, std::mutex *q_lock)
{
    q_lock->lock();
    auto local_q = *q;
    q_lock->unlock();

    std::sort_heap(local_q.begin(), local_q.end(), QueueElementCmp);
    std::reverse(local_q.begin(), local_q.end());

    ssize_t pos = 0;
    for (const auto &ele : local_q)
    {
        if (ele.id == id)
        {
            return pos;
        }

        ++pos;
    }

    return -1;
}

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

using _check_auth_t = std::function<std::string(AuthLevel min_auth_level, const httplib::Request &req, httplib::Response &res)>;
using _check_auth_bound_t = std::function<std::string(const httplib::Request &req, httplib::Response &res)>;

httplib::Server::Handler _request_wrapper(_check_auth_bound_t check_auth, _http_user_handler user_handler, bool log = true)
{
    return [check_auth, user_handler, log](const httplib::Request &req, httplib::Response &res)
    {
        auto auth_res = check_auth(req, res);
        if (auth_res.length())
        {
            _log_request(req, "AUTH_FAILED " + auth_res);
            return;
        }

        auto extra_logging = user_handler(req, res);
        if (log)
        {
            _log_request(req, extra_logging);
        }
    };
}

void _http_server_run(
    models_map_t models,
    std::shared_ptr<std::string> *session_ss,
    std::function<std::string()> session_private,
    _http_server_starter go,
    _http_put_prompt_on_queue put_q,
    _http_get_prompt_result get_res,
    AuthOptions auth_options)
{
    httplib::Server server;

    _check_auth_t check_auth = [auth_options](AuthLevel min_auth_level, const httplib::Request &req, httplib::Response &res)
    {
        if (auth_options.level == AuthLevel::None)
        {
            return std::string{};
        }

        if (min_auth_level > auth_options.level)
        {
            return std::string{};
        }

        auto fail = [&res](std::string reason = "Unknown") -> std::string
        {
            res.set_header("WWW-Authenticate", "Basic");
            res.status = 401;
            return reason;
        };

        if (!req.has_header("Authorization"))
        {
            return fail("No Header");
        }

        auto auth_header = req.get_header_value("Authorization");
        auto scheme = std::string{"Basic "};
        auto basic_idx = auth_header.find(scheme);

        if (basic_idx != 0)
        {
            return fail("Bad Scheme: " + auth_header);
        }

        auto base64_basic = auth_header.substr(scheme.length());
        std::string basic_decoded = cppcodec::base64_rfc4648::decode<std::string>(base64_basic);

        auto colon_idx = basic_decoded.find(":");
        if (colon_idx == std::string::npos)
        {
            return fail("Bad Base64 Decode: " + basic_decoded);
        }

        std::string api_key = basic_decoded.substr(colon_idx + 1);
        auto found_key_iter = auth_options.keys->find(api_key);
        if (found_key_iter != auth_options.keys->end())
        {
            KeyedRequestAuditLog &al_ref = (*found_key_iter).second;
            al_ref.count++;
            al_ref.last.remote_addr = _remote_addr(req);
            al_ref.last.path = req.path;
            return std::string{};
        }

        return fail("Bad Key: " + api_key);
    };

    auto bind_check_auth = [check_auth](AuthLevel min_auth_level) -> _check_auth_bound_t
    {
        return [check_auth, min_auth_level](const httplib::Request &req, httplib::Response &res) -> std::string
        {
            return check_auth(min_auth_level, req, res);
        };
    };

    if (session_ss)
    {
        server.Get(*(session_ss->get()), _request_wrapper(
                                             bind_check_auth(AuthLevel::Runtime),
                                             [session_private](const httplib::Request &req, httplib::Response &res)
                                             {
        auto resp = session_private();
        res.set_content(resp, "application/json");
        return ""; }));
    }

    server.Get("/models", _request_wrapper(
                              bind_check_auth(AuthLevel::All),
                              [models](const httplib::Request &req, httplib::Response &res)
                              {
        nlohmann::json json(models);
        res.set_content(json.dump(), "application/json");
        return std::string(""); }));

    server.Post("/prompt",
                _request_wrapper(
                    bind_check_auth(AuthLevel::POSTPrompt),
                    [put_q, &models, check_auth](const httplib::Request &req, httplib::Response &res)
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
        if (!parsed_body["promptWrappers"].is_discarded() && parsed_body["promptWrappers"].is_object())
        {
            auto& pw = parsed_body["promptWrappers"];
            if (pw["pre"].is_string()) {
                pre = pw["pre"];
            }

            if (pw["post"].is_string()) {
                post = pw["post"];
            }
        }

        QueuePriority priority{QueuePriority::NORMAL};
        if (!parsed_body["priority"].is_discarded() && parsed_body["priority"].is_string())
        {
            if (parsed_body["priority"] == "LOW") {
                priority = QueuePriority::LOW;
            }

            if (parsed_body["priority"] == "HIGH") {
                const auto auth_res = check_auth(AuthLevel::HighPriority, req, res);

                if (auth_res.length()) {
                    return auth_res;
                }

                priority = QueuePriority::HIGH;
            }
        }

        uint mirostat = 0;
        if (!parsed_body["mirostat"].is_discarded() && parsed_body["mirostat"].is_number_unsigned()) {
            mirostat = parsed_body["mirostat"];
        }

        std::string prompt = pre + (std::string)parsed_body["prompt"] + post;
        uint64_t new_id = 0;
        size_t q_pos = -1;
        std::tie(new_id, q_pos) = put_q(prompt, parsed_body["model"], _remote_addr(req), priority, mirostat);

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
               _request_wrapper(
                   bind_check_auth(AuthLevel::GETPromptById),
                   [get_res](const httplib::Request &req, httplib::Response &res)
                   {
        std::stringstream ss;
        uint64_t prompt_id;
        ss << std::hex << req.matches[1].str();
        ss >> prompt_id;

        auto get_response = get_res(prompt_id);

        if (get_response.prompt.empty()) 
        {
            res.status = 404;
        }
        else if (get_response.rpm.response.empty())
        {
            nlohmann::json json {
                {"queuePosition", get_response.queue_position},
                {"model", get_response.rpm.model},
                {"prompt", get_response.prompt},
            };
            res.set_content(json.dump(), "application/json");
            res.status = 202;
        }
        else
        {
            nlohmann::json json {
                {"prompt", get_response.prompt},
                {"response", get_response.rpm.response},
                {"elapsed_ms", get_response.rpm.elapsed_ms},
                {"tokens", get_response.rpm.tokens},
                {"model", get_response.rpm.model},
                {"ms_per_token", get_response.rpm.elapsed_ms / get_response.rpm.tokens}
            };
            res.set_content(json.dump(), "application/json");
        }

        return ""; },
                   false));

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
    llama_timings *total_timings,
    AuthOptions auth_options)
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
        session_ss << "/" << prefix << "/";

        if (auth_options.level > AuthLevel::None && auth_options.keys)
        {
            session_ss << "data";
        }
        else
        {
            session_ss << std::hex << idA << idB;
        }

        *session_ep = std::make_shared<std::string>(session_ss.str());
    }

    // server startup (_http_server_starter)
    auto server_startup_handler = [hostname, port](httplib::Server &server)
    {
        server.listen(hostname, port);
    };

    auto runtime_info_ep_handler = [q, q_lock, m, pending_id, total_timings, lifetime_queued, auth_options]()
    {
        q_lock->lock();
        _queue_t local_q = *q;
        q_lock->unlock();

        std::sort_heap(local_q.begin(), local_q.end(), QueueElementCmp);
        std::reverse(local_q.begin(), local_q.end());

        std::vector<nlohmann::json> q_json;
        for (const auto &q_element : local_q)
        {
            q_json.emplace_back(nlohmann::json{
                {"id", _hexify_id(q_element.id)},
                {"priority", q_element.priority}});
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

        if (auth_options.level > AuthLevel::None)
        {
            auto &kr = json["keys"] = std::map<std::string, nlohmann::json>{};
            for (const auto &keyent : *auth_options.keys)
            {
                kr[keyent.first] = nlohmann::json{
                    {"count", keyent.second.count},
                    {"last", nlohmann::json{
                                 {"remote_add", keyent.second.last.remote_addr},
                                 {"path", keyent.second.last.path},
                             }}};
            }
        }

        return json.dump();
    };

    // POST handler to put a prompt on the queue (_http_put_prompt_on_queue)
    auto POST_handler = [q, q_lock, m, context_size, lifetime_queued](
                            std::string prompt,
                            std::string model,
                            std::string remote_addr,
                            QueuePriority priority,
                            uint mirostat) -> std::pair<uint64_t, ssize_t>
    {
        if (prompt.length() > (std::size_t)context_size)
        {
            return std::make_pair((long unsigned)0, (ssize_t)-1);
        }

        auto id = _unique_id(m);
        ResponsePlusMetrics rpm;
        rpm.model = model;
        rpm.remote_addr = remote_addr;
        rpm.queued_iso8601 = iso8601_timestamp();

        {
            using namespace std::chrono;
            auto qtsms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            QueueElement qe{id, qtsms, prompt, priority, mirostat};

            std::lock_guard<std::mutex> lg(*q_lock);
            q->push_back(qe);
            std::push_heap(q->begin(), q->end(), QueueElementCmp);
            m->emplace(std::make_pair(id, std::make_pair(prompt, rpm)));
        }

        (*lifetime_queued)++;
        return std::make_pair(id, QueueElementPosById(id, q, q_lock));
    };

    // GET promptId handler (_http_get_prompt_result)
    auto GET_promptId_handler = [q, q_lock, m](uint64_t id) -> _http_get_prompt_result_return
    {
        auto ele = m->find(id);
        if (ele == m->end())
        {
            return _http_get_prompt_result_return{};
        }

        return _http_get_prompt_result_return{
            ele->second.first,
            ele->second.second,
            QueueElementPosById(ele->first, q, q_lock)};
    };

    std::thread(
        _http_server_run,
        models,
        session_ep,
        runtime_info_ep_handler,
        server_startup_handler,
        POST_handler,
        GET_promptId_handler,
        auth_options)
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

        QueueElement q_element;
        {
            std::lock_guard<std::mutex> lg(*q_lock);
            std::pop_heap(q->begin(), q->end(), QueueElementCmp);
            q_element = q->back();
            q->pop_back();
        }

        *pending_id = q_element.id;
        r = q_element.prompt;
        model = m->at(q_element.id).second.model;

        return ServicerResponse{_hexify_id(*pending_id), r, model, q_element.mirostat};
    };
}