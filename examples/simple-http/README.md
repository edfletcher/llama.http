# simple-http

An HTTP server with a very simple REST interface.

## Interface

### List available models

`GET /models` will list all available models and metadata about then in `application/json` with the following shape:

```json
{
    "<model name>": {
        "displayName": "...",
        "sourceURL": "...",
        "promptWrappers": {
            "pre": "...",
            "post": "..."
        }
    },
}
```

### Post a prompt to the queue for processing

`POST /prompt` with a request body in `application/json` following this shape:

```json
{
    "prompt": "<user prompt>",
    "model": "<model name>"
}
```

Optionally, the `promptWrappers` data structure seen elsewhere may be included, where the constituent parameters are optional and if included will override the configure prompt wrappers.

Will return HTTP 413 if the prompt is too large, or `application/json` in the following shape on success:

```json
{
    "promptId": "...",
    "queuePosition": 12345
}
```

### Get the status or result of a queued prompt

`GET /prompt/:id` with a prompt `:id` to retrieve the prompt & response as `application/json`. If the response is still pending, will return HTTP code 202 with only the model name and queue position in the response JSON. If the `:id` is not valid, returns HTTP 404.

## Example

```shell
$ $ curl -s -d '{"prompt": "tell me about alpacas", "model": "airoboros-l2-13b-gpt4-1.4.1.ggmlv3.q4_0.bin" }' http://127.0.0.1:42000/prompt | jq .
{
  "promptId": "6af0c8af11830f91",
  "queuePosition": 1
}
$ curl -vs http://127.0.0.1:42000/prompt/6af0c8af11830f91 | jq .
...
< HTTP/1.1 202 Accepted
...
{
  "model": "airoboros-l2-13b-gpt4-1.4.1.ggmlv3.q4_0.bin",
  "prompt": "tell me about alpacas",
  "queuePosition": 1
}

# some time later...

$ curl -vs http://127.0.0.1:42000/prompt/6af0c8af11830f91 | jq .
{
  "elapsed_ms": 703788.25,
  "model": "airoboros-l2-13b-gpt4-1.4.1.ggmlv3.q4_0.bin",
  "ms_per_token": 1679.685546875,
  "prompt": "tell me about alpacas",
  "response": "Alpacas are domesticated South American camelids. They are related to llamas, guanacos, and vicuñas and belong to the same genus as the llama, but they are smaller in size. Alpacas have been bred for their fleece, which is used in textiles, and they are also raised for meat and hides.\n\nAlpacas have a lifespan of around 15-20 years and can weigh between 100-180 pounds. They have a distinctive appearance with a small head, long neck, and a short, compact body. Their fur is very soft and comes in various colors such as black, white, brown, gray, and tan.\n\nAlpacas are herbivores and graze on grass, hay, and other vegetation. They are social animals that live in herds led by a dominant male called a \"herdsire.\" Females give birth to a single baby (called a \"cria\") after a gestation period of about 11 months. Crias typically stand within an hour of birth and can walk within a few hours.\n\nAlpacas are relatively easy to care for and can be kept in small spaces, making them popular among hobby farmers and those with limited land. They are also known for their gentle nature and can be trained to walk on a leash or be halter-broken.\n\nIn summary, alpacas are domesticated South American camelids bred for their fleece, meat, and hides. They have a lifespan of around 15-20 years, weigh between 100-180 pounds, and have a distinctive appearance with soft fur that comes in various colors. Alpacas are herbivores, social animals, and easy to care for, making them popular among hobby farmers and those with limited land.",
  "tokens": 419
}
```

## Building

You needn't build this yourself as I've made a docker image available [here](https://hub.docker.com/r/edfletcher/llama.http) (see [below](#from-docker-hub) for futher information on using it).

### Locally

Initialize the repository's submodule(s):

```shell
$ git submodule update --init --recursive
Submodule 'deps/cpp-httplib' (https://github.com/yhirose/cpp-httplib.git) registered for path 'deps/cpp-httplib'
Cloning into '/home/noname/repos/alpaca.http/deps/cpp-httplib'...
Submodule path 'deps/cpp-httplib': checked out 'a66a013ed78dee11701d6075c6b713307004a126'
```

Then:

```shell
$ make simple-http
```

### With docker

This must be run _from the project's root directory_, e.g. two directories _up_ from where this README and Dockerfile live.

`<TAG>` will be your image's user tag, such as `simplehttp:1`.

```shell
$ pwd
/home/edfletcher/llama.http
$ docker build -t <TAG> -f examples/simple-http/Dockerfile .
...
=> => naming to docker.io/library/<TAG>  
```

## Running

### Locally

Run with `-h` for full, up-to-date help.

```shell
$ ./simple-http -m /path/to/models/
```

The model path `/path/to/models` must have paired model binaries and sidecar JSON as specified below.

### With docker

#### From Docker Hub

Replace the `<TAG>` in the command invocation below with `edfletcher/llama.http:latest`. That's it.

See [here](https://hub.docker.com/r/edfletcher/llama.http) for full image information.

#### From a local build

Using the same `<TAG>` placeholder as in the build section above:

```shell
$ docker run -d -it -v <PATH_TO_MODELS>:/models -p 42000:42000 <TAG> -m /models -H 0.0.0.0
```

where `<PATH_TO_MODELS>` is the path where models and their sidecar JSONs are stored _on the local machine (host)_.

### Model Sidecar JSON

Each model binary you'd like to run must be accompanied by a sidecar JSON file named `<binary-file-name>.json`, where `<binary-file-name>` is the *full name* including extension. For example, to use `airoboros-l2-13b-gpt4-1.4.1.ggmlv3.q4_0.bin`, make a file name `airoboros-l2-13b-gpt4-1.4.1.ggmlv3.q4_0.bin.json` in the same path as the binary.

These sidecar files must have the following shape:

```json
{
    "displayName": "<freeform name to show to users>",
    "sourceURL": "<URL where the model is described & can be downloaded>",
    "promptWrappers": {
        "pre": "<string to be prepended to the user prompt>",
        "post": "<string to be appended to the user prompt>"
    }
}
```

`displayName` and `sourceURL` are **required**. `promptWrappers` are optional: by default, the prompt will _not_ be wrapped with anything unless specified in the sidecar JSON.

Most model cards specify which prompt wrappers (if any) the model was trained with.
