# The VibePrevisBench sandbox with Codex CLI installed: the agent image of the codex preset.
#   docker build -t vibeprevisbench-agent:codex -f codex.Dockerfile .
# Pin a version with --build-arg CODEX_VERSION=<version>.
ARG BASE=vibeprevisbench-sandbox:5.2.2
FROM ${BASE}
ARG CODEX_VERSION=latest
RUN apt-get update && apt-get install -y --no-install-recommends git ripgrep nodejs npm && rm -rf /var/lib/apt/lists/*
RUN npm install -g @openai/codex@${CODEX_VERSION} && codex --version
