FROM ubuntu:20.04 as simplehttp_build
WORKDIR /simplehttp_build
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get clean all && \
  apt update && \
  apt -y upgrade && \
  apt -y autoremove && \
  apt -y install make g++ gcc git
COPY . .
COPY .git .
RUN git submodule update --init --recursive
RUN make simple-http

FROM ubuntu:20.04 as simplehttp
WORKDIR /simplehttp
COPY --from=simplehttp_build /simplehttp_build/simple-http .
COPY --from=simplehttp_build /simplehttp_build/examples/simple-http/entry.sh .
RUN useradd -u 1001 -U -p simplehttp simplehttp
USER simplehttp
STOPSIGNAL SIGINT
ENTRYPOINT ["/simplehttp/entry.sh"]