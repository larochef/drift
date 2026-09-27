FROM eclipse-temurin:25-jdk-alpine-3.23 AS build
COPY . /root/data
WORKDIR /root/data
RUN apk add --no-cache curl nodejs npm
RUN ./mill --no-server backend.universalStage

FROM eclipse-temurin:25-jre
LABEL maintainer="NuMind <tech@numind.ai>"

COPY --from=build /root/data/out/backend/universalStage.dest /opt/extract-platform

RUN chmod +x /opt/extract-platform/bin/backend
USER 1001
WORKDIR /opt/extract-platform

ENV PORT=9000
EXPOSE 9000

ENTRYPOINT ["/opt/extract-platform/bin/backend"]
