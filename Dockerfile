FROM ubuntu:latest
LABEL authors="aandreev"

ENTRYPOINT ["top", "-b"]