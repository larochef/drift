# drift in a container: the backend and its page, on port 4321. Runtimes
# (sd-cpp, llama.cpp) are installed from the page once it runs. Not in the
# image: ffmpeg (video stills and the runner's videos need it) and the drift
# runner's GPU kernels, which are built against a ROCm tree.
FROM eclipse-temurin:25-jdk-alpine-3.23 AS build
COPY . /root/data
WORKDIR /root/data
RUN apk add --no-cache curl nodejs npm
RUN ./mill --no-server backend.universalStage

FROM eclipse-temurin:25-jre

COPY --from=build /root/data/out/backend/universalStage.dest /opt/drift

RUN chmod +x /opt/drift/bin/backend
USER 1001
WORKDIR /opt/drift

EXPOSE 4321

ENTRYPOINT ["/opt/drift/bin/backend"]
