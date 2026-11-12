# The VibePrevisBench sandbox with Claude Code installed: the agent image of the claude-code preset.
#   docker build -t vibeprevisbench-agent:claude-code -f claude-code.Dockerfile .
# The paper used Claude Code 2.1.308; pin a version with --build-arg CLAUDE_CODE_VERSION=2.1.308.
ARG BASE=vibeprevisbench-sandbox:5.2.2
FROM ${BASE}
ARG CLAUDE_CODE_VERSION=stable
RUN apt-get update && apt-get install -y --no-install-recommends git ripgrep && rm -rf /var/lib/apt/lists/*
# The native installer puts the binary in /root/.local/bin; link it into /usr/local/bin because the login shell the
# agent runs in (bash -l) resets PATH.
RUN curl -fsSL https://claude.ai/install.sh | bash -s ${CLAUDE_CODE_VERSION} \
 && ln -sf /root/.local/bin/claude /usr/local/bin/claude \
 && claude --version
