---
name: vibeprevisbench-eval
description: Evaluate a coding agent on VibePrevisBench end to end with one command. The command checks the environment, downloads the dataset and input media, runs the agent on each task in its own Docker sandbox under the paper's limits, replays every delivery, scores it with the Rule-based Scene Check and the VLM-based Screen Check, and writes a report next to the published leaderboard. Use this skill whenever someone in this repository wants to evaluate, benchmark, test, score or rank an agent or model (Claude Code, Codex CLI or any command-line agent) on VibePrevisBench. Also use it to run a quick smoke test, resume or monitor an evaluation, rescore a run, or compare a new agent with the paper's results, even if the skill is not named.
---

# VibePrevisBench one-command evaluation

`scripts/run_eval.py` runs the full pipeline from the repository README. Each stage is resumable:

```text
setup    Python packages, Docker, ffmpeg; sandbox image (+ agent image); Blender 5.2.2; dataset; input media
agent    per task: fresh container (8 CPUs, 16 GB) -> agent works on /opt/task/TASK.md (4 h limit) -> collect
replay   rerun each solution.py in a fresh container with no network; the paper's three delivery gates
score    Rule-based Scene Check, CPU only, no API
judge    evaluator orbit for photo tasks + VLM-based Screen Check (paid Gemini-compatible API)
report   Scene / Screen / Overall, per input kind, and the published agents on the same tasks
```

Your job:
1. Collect a valid configuration.
2. Prove it on a small smoke run.
3. Launch and monitor the full run.
4. Report the result honestly.

Run the script yourself with Bash and read what it prints. The script only calls the documented `vpbench.py`
commands, which keeps a run comparable with the paper. Do not hand-edit anything under `runs/` or `results/`.

All paths are relative to the repository root:
- `SKILL_DIR` is `.claude/skills/vibeprevisbench-eval`.
- `PY` is the Python with `requirements.txt` installed, normally `.venv/bin/python`.

## Requirements

Check these before promising the user anything:
- **A Linux host with Docker** that the current user can use (`docker info` works). Every task runs in its own
  container, and every delivery is replayed in a container with no network.
- **CPU and memory:** about 8 CPUs and 16 GB per agent running in parallel. No GPU is needed.
- **Python 3.12 with `requirements.txt`.** If `.venv` is missing, create it, since it is git-ignored:
  `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
- **ffmpeg and ffprobe** 5.1 or newer on `PATH`.

If Docker is missing, stop and tell the user. Ask for a host that has it, or for Docker to be installed. Do not fall
back to `--backend local` to evaluate an agent: it runs the agent on the host, where it can read `data/rubrics/`, the
hidden ground truth. That backend is only for debugging the pipeline.

## 1. Agree on the configuration

Find out these settings. Ask only for what you cannot infer from the request:

| Setting | Flag | Notes |
| --- | --- | --- |
| Agent | `--agent claude-code` / `--agent codex`, or `--agent-cmd '...'` | The presets are the paper's harnesses: GPT models ran in Codex CLI, all others in Claude Code. |
| Model | `--model` | The provider's model ID, e.g. `claude-opus-5-5`. A non-Anthropic model in Claude Code needs an Anthropic-compatible gateway; see `references/agents.md`. |
| Effort | `--effort` | To reproduce a model the paper evaluated, use the effort in the README table "Evaluated models, agent harness and reasoning effort". For a new model, use the highest effort it accepts, and report it. |
| Run name | `--name` | Names `runs/<name>/` and `results/<name>/`. Use one name per configuration: for example `opus55-smoke` for the smoke run and `opus55` for the full run. |
| Credentials | environment variables | See below. |
| Tasks | `--tasks smoke`, `all`, or `ID,ID` | Start with `smoke`. |
| Parallelism | `--parallel N` | Size N to the host: 8 CPUs and 16 GB per agent. |
| Judge | `VPB_JUDGE_API_KEY` or `GEMINI_API_KEY`; optional `VPB_JUDGE_BASE_URL`, `VPB_JUDGE_MODEL`, `VPB_JUDGE_AUTH=bearer` | Without a key, the report shows Scene only. The judge can be added later. |

**Credentials never appear in the chat, in a file or on a command line.** Do not ask the user to paste a key, and
do not suggest `! export ...` inside Claude Code: that would put the key into the conversation. Instead, the user
exports the variables in their terminal and then starts Claude Code from that terminal. Your Bash commands, and the
script, inherit them, and the script hands them to the container by name (`docker exec -e NAME`).
- **claude-code preset:** passes `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `CLAUDE_CODE_OAUTH_TOKEN`,
  `ANTHROPIC_BASE_URL` and the Bedrock/Vertex variables, if they are set.
- **codex preset:** passes `CODEX_API_KEY`, `OPENAI_API_KEY` and `OPENAI_BASE_URL`. `codex exec` authenticates with
  `CODEX_API_KEY`.
- **Custom agent:** add `--pass-env NAME1,NAME2`.

Verify the credentials with the dry-run below. It lists the variable names that will be passed and warns when a
preset has none.

The paper protocol is the default and should stay so unless the user explicitly wants something else:
- 4 h per task (`--time-limit`), 8 CPUs / 16 GB per agent container;
- the `release` briefs;
- Gemini 4 Argon as the judge.

Changing any of these makes the result not comparable with the leaderboard; record it in the report.

For a custom agent, or a preset with unusual needs (a gateway or proxy, a self-hosted model, a pinned CLI version),
read `SKILL_DIR/references/agents.md` first.

## 2. Check the plan, then set up

```bash
PY SKILL_DIR/scripts/run_eval.py --name opus55-smoke --agent claude-code --model claude-opus-5-5 \
    --effort high --tasks smoke --dry-run
```

`--dry-run` costs nothing and changes nothing. It prints:
- the image and the exact agent command;
- the credential names that will be passed (never their values);
- whether the judge is on;
- a `preflight` section: Docker, the images, the dataset, Blender and the input media of the selected tasks.

Show the plan to the user. Then run the setup stage with the same name and tasks:

```bash
PY SKILL_DIR/scripts/run_eval.py --name opus55-smoke --agent claude-code --tasks smoke --stages setup
```

Setup makes no paid API calls, so you can run it once the plan is shown. The first run builds the sandbox image and
copies out Blender 5.2.2, which takes 10–40 minutes. It then downloads about 130 MB of annotations and fetches the
input media of the selected tasks from their original sources. If it stops, the message names the problem;
`SKILL_DIR/references/troubleshooting.md` has the usual causes and fixes.

## 3. Smoke run (4 tasks, one per input kind)

Get the user's go-ahead first: this is the first step that calls the agent's model API and costs money.
`results/leaderboard.csv` lists, for each published agent, the average input and output tokens and the wall time
per task, which gives a rough cost estimate.

```bash
PY SKILL_DIR/scripts/run_eval.py --name opus55-smoke --agent claude-code --model claude-opus-5-5 \
    --effort high --tasks smoke --parallel 4
```

The smoke run covers TP101 (script), M50_028 (photos), B250_174 (continuation) and B250_207 (video). Their media
come from public URLs. It takes up to the time limit; the paper's agents averaged 40 minutes to 3 hours per task.

Its purpose is to prove that the agent really works in the sandbox before committing 242 tasks. Check:
- **Each task's result:** `runs/opus55-smoke/<ID>/agent_run.json` should have `"solution": true` and a plausible
  duration. `agent_log.txt` holds the agent's own output.
- **Quick failures:** a task that ends within minutes without `solution.py` almost always means a configuration
  problem: authentication, a wrong model name, or a missing CLI. After three such failures, the script stops by
  itself and prints the last log. Fix the cause and run the same command again.
- **The summary:** `results/opus55-smoke/summary.md` compares the run with the published agents on the same 4
  tasks. Four tasks are far too few to rank an agent; use them only as a sanity check.

## 4. Full run

Confirm with the user before starting, because a full run is long and paid:
- **Time:** about 242 × the median smoke-task time ÷ `--parallel`.
- **Agent cost:** 242 × the smoke cost per task. The summary shows the cost the CLI reported, where available.
- **Judge cost:** about 242 Gemini requests with video.

Run it detached so it survives the session, then monitor it:

```bash
nohup PY SKILL_DIR/scripts/run_eval.py --name opus55 --agent claude-code --model claude-opus-5-5 \
    --effort high --tasks all --parallel 8 > results/opus55.out 2>&1 &
PY SKILL_DIR/scripts/run_eval.py --name opus55 --tasks all --status    # progress, any time
```

**To resume after an interruption, run exactly the same command again.** Finished tasks are kept, and a
half-finished task is restarted from a fresh container. The script refuses to resume a run name with a different
agent configuration, because mixed configurations make the result meaningless. Start a new `--name` instead.

`--status` lists tasks that ended within 5 minutes without `solution.py`. Read their `agent_log.txt`.
- **Infrastructure failure:** an API outage, rate limiting, a network error or a crashed host. Fix the cause, then
  rerun those tasks with the same command plus `--rerun-failed`. The paper retried only infrastructure failures, and
  at most twice.
- **The agent tried and failed:** keep the result. Rerunning it would cherry-pick, and the summary lists every task
  that was run more than once.

To stop a run, kill the script, then remove its containers:

```bash
docker rm -f $(docker ps -q --filter name=vpb-opus55-)
```

## 5. Report

The report stage writes `results/NAME/summary.md`, `summary.json` and `report.json`. Present them to the user:
- **Overall, Scene and Screen.** Overall = (Scene + Screen) / 2. Tasks that fail or deliver nothing count as 0 over
  all 242 tasks.
- **Scores per input kind,** and how many tasks delivered a solution and passed the replay gates.
- **The table of published agents on the same tasks,** which is the fair comparison.
- **Any deviation from the paper protocol:** other limits, another judge model, skipped tasks, or a subset.

A subset is never a leaderboard score. Say so plainly, as the summary does.

Stages can be rerun on their own with the run's `--tasks`. For example, after exporting a judge key for a finished
run:

```bash
PY SKILL_DIR/scripts/run_eval.py --name opus55 --tasks all --stages judge,report
```

## What keeps a result valid

- **The agent must never see `data/rubrics/`.** It holds the hidden ground truth. The script mounts nothing into the
  agent's container; keep it that way for custom agents, and never run agents with `--backend local`.
- **Scoring uses only the replayed delivery.** The agent's own renders and claims do not count, so a solution that
  only works inside the agent's container scores 0.
- **Retry only infrastructure failures.** Never retry a task because its score was low.
- **Use one configuration per run name, and state the judge model.** Screen scores from another judge are not
  comparable with the Gemini 4 Argon column.
- **Small replay differences are expected.** Rescoring the same delivery can move a task by a few hundredths of a
  point, because mesh sampling differs in the last floating-point digits. See `docs/reproducibility.md`.

## Running without Claude

The script is plain Python. Anyone can run the commands above directly. `--help` lists every option.
