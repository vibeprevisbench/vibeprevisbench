# Plugging an agent into the evaluation

Read this when the agent is not a plain preset run, or when a preset needs extra settings.

## What every agent gets

The `agent` stage starts one container per task. The container is created from the agent image, which is the
sandbox image plus the agent's CLI. Inside it, the stage runs:

```bash
docker exec -w /work/blender_agent [-e NAME ...] vpb-<name>-<task> \
    timeout --kill-after=60 <limit> bash -lc '<agent command>'
```

The contract is the paper's:

- **Task text.** The task is in `/opt/task/TASK.md`: the brief, followed by `agent_instructions.txt`, which explains
  the workspace, the managed `blender` command, `pvtools` tagging and the delivery rules. The presets pass the whole
  file as the prompt, as the paper did.
- **Workspace.** The working directory is `/work/blender_agent`. `input/` holds the entity table, the category table
  and the reference media, with 1 fps frames and viewing copies.
- **Delivery.** The agent must leave a standalone `solution/solution.py`. Nothing else is scored: the replay reruns
  that file in a fresh container without network, and Scene and Screen are computed from that replay.
- **Limits.** 8 CPUs, 16 GB, no GPU and 4 hours, the paper's agent sandbox. The time limit kills the agent; whatever
  `solution.py` exists at that moment is collected.
- **Network.** The container uses Docker's `bridge` network so the agent can reach its model API. For the paper
  protocol, the agent may use no other network resources and no external assets. On a host that restricts egress,
  use `--network NAME` and an HTTP proxy variable passed with `--pass-env`.
- **Ground truth.** The dataset's `rubrics/` folder is never mounted. Do not add mounts that expose it.

## Presets

Both presets reproduce the paper's command line. `{model}` and `{effort}` are filled in only when given.

| Preset | Command | Image |
| --- | --- | --- |
| `claude-code` | `claude -p "$(cat /opt/task/TASK.md)" --model M --effort E --dangerously-skip-permissions --output-format json` | `vibeprevisbench-agent:claude-code` (`docker/claude-code.Dockerfile`) |
| `codex` | `codex exec "$(cat /opt/task/TASK.md)" --model M -c model_reasoning_effort="E" --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -C /work/blender_agent --json` | `vibeprevisbench-agent:codex` (`docker/codex.Dockerfile`) |

The setup stage builds the image when it is missing. To pin the CLI version:
- `--build-arg CLAUDE_CODE_VERSION=2.1.308` (the paper's Claude Code version);
- `--build-arg CODEX_VERSION=<version>`.

When the version changes, remove the old image (`docker rmi vibeprevisbench-agent:claude-code`) so it is rebuilt.
Both images were build-tested with Claude Code 2.1.311 and codex-cli 0.167.2.

**claude-code** sets these environment variables in the container:
- `IS_SANDBOX=1`, which allows bypassing permissions as root;
- `DISABLE_AUTOUPDATER=1`;
- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`;
- `CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000`, the paper's 64k output tokens per request.

Credentials:
- an API key: `ANTHROPIC_API_KEY`;
- a subscription: `CLAUDE_CODE_OAUTH_TOKEN`, created with `claude setup-token`;
- a gateway: `ANTHROPIC_BASE_URL` together with `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY`;
- Bedrock or Vertex: `CLAUDE_CODE_USE_BEDROCK` / `CLAUDE_CODE_USE_VERTEX` and their cloud variables.

The preset passes all of these when they are set. Effort values are `low`, `medium`, `high`, `xhigh` and `max`;
which ones exist depends on the model.

**codex**: `codex exec` authenticates with `CODEX_API_KEY`. `OPENAI_BASE_URL` is passed for compatible endpoints.
Reasoning effort goes through `-c model_reasoning_effort=...`.

The paper's settings per model are in the README table "Evaluated models, agent harness and reasoning effort".

### Model-specific settings

Add non-secret settings with `--agent-env NAME=VALUE`, e.g. `--agent-env API_TIMEOUT_MS=7200000`. Pass secrets
with `--pass-env NAME`.

The paper ran every non-GPT model, including Gemini, DeepSeek, Kimi, GLM, Doubao and Muse, in Claude Code. To do
the same, put an Anthropic-compatible gateway in front of the model and give its URL as `ANTHROPIC_BASE_URL`, with
its token as `ANTHROPIC_AUTH_TOKEN`. Use the model ID the gateway expects as `--model`. The gateway must be
reachable from inside the container: `localhost` there is the container itself, not the host.

## Custom agents

Any command-line agent works if it can run non-interactively in the container.

1. Build an image `FROM vibeprevisbench-sandbox:5.2.2` that installs the agent. Keep everything the sandbox has:
   the agent needs the managed `blender` command, `ffmpeg` and `pvtools`, which `prepare` installs into the
   workspace at run time. Put the executable in `/usr/local/bin`: the agent runs in a login shell (`bash -l`),
   which resets `PATH`.
2. Pass the task text the way the agent expects it, for example `"$(cat /opt/task/TASK.md)"`, a file argument, or
   stdin with `< /opt/task/TASK.md`.
3. Make sure it runs unattended: no approval prompts and no questions to a user.

```bash
python3 SKILL_DIR/scripts/run_eval.py --name myagent-v1 --agent-image myagent:latest \
    --agent-cmd 'myagent run --task-file /opt/task/TASK.md --yes' --pass-env MYAGENT_API_KEY --tasks smoke
```

**Agents that run outside the container.** An agent that runs on the host can still be used: `--agent-cmd` can
start a small client inside the container that talks to it. The simpler alternative is to run `prepare`, drive the
container with `docker exec` from your own loop, then `collect`, as in README step 3. Keep the 4-hour limit and the
same container limits.

## What the stage records per task

`runs/<name>/<task_id>/` contains:

| File | Content |
| --- | --- |
| `agent_run.json` | Exit code, duration, whether the time limit hit, whether `solution.py` exists, and token or cost totals parsed from the CLI's JSON output when available |
| `agent_log.txt` | The agent's stdout and stderr |
| `prepare_log.txt` | Workspace preparation output |
| `agent/` | The container workspace collected at the end, including `workspace/` |
| `solution.py` | The delivery |

The replay adds `scene.blend`, `video.mp4`, `replay/` and `replay.json`.
