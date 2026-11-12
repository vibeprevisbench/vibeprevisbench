#!/usr/bin/env python3
r"""One-command VibePrevisBench evaluation of a coding agent.

    python3 .claude/skills/vibeprevisbench-eval/scripts/run_eval.py --name opus55 \
        --agent claude-code --model claude-opus-5-5 --effort high --tasks smoke

Stages (all by default, or pick with --stages):
  setup    check Python packages, Docker, ffmpeg; build the sandbox image (and the agent image of a preset);
           take Blender 5.2.2 from the sandbox image; download and verify the dataset; fetch the input media
  agent    for every task: prepare a fresh container (8 CPUs, 16 GB), run the agent in it with a time limit
           (4 h, as in the paper), collect solution/solution.py and the workspace
  replay   rerun every solution.py in a fresh container without network (the paper's three delivery gates)
  score    Rule-based Scene Check (no API)
  judge    render the evaluator orbit of photo tasks, then VLM-based Screen Check (paid API calls;
           skipped when no judge key is set)
  report   Scene, Screen and Overall, per input kind, and the published models on the same tasks

Every stage keeps finished work: run the same command again to resume. --status shows progress, --dry-run prints
the plan. Results: runs/<name>/<task_id>/ (agent output) and results/<name>/ (scores, report, summary.md).
Credentials are never written to disk or command lines: export them in the shell and pass their names with
--pass-env (presets pass the usual names automatically).
"""
import argparse, csv, datetime as dt, hashlib, json, os, re, shlex, shutil, statistics, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve()
SKILL = HERE.parents[1]
STAGES = ["setup", "agent", "replay", "score", "judge", "report"]
SANDBOX_IMAGE = os.environ.get("VPB_SANDBOX_IMAGE", "vibeprevisbench-sandbox:5.2.2")
TASK_FILE, WORKSPACE = "/opt/task/TASK.md", "/work/blender_agent"
# One task per input kind whose media can be fetched without accounts, licences or yt-dlp.
SMOKE = ["TP101", "M50_028", "B250_174", "B250_207"]
KINDS = ["text", "image", "continuation", "video"]
KIND_NAMES = {"text": "Script", "image": "Photos", "continuation": "Continuation", "video": "Reference video"}

# The two agent harnesses of the paper. The commands are the paper's (the task file as the prompt, permissions
# bypassed inside the throwaway container, JSON output); {model}/{effort} are filled in when given.
PRESETS = {
    "claude-code": {
        "dockerfile": "claude-code.Dockerfile",
        "image": "vibeprevisbench-agent:claude-code",
        "command": 'claude -p "$(cat %s)"{model}{effort} --dangerously-skip-permissions --output-format json' % TASK_FILE,
        "model_flag": " --model %s", "effort_flag": " --effort %s",
        # IS_SANDBOX=1 lets Claude Code bypass permissions as root inside the container.
        "env": {"IS_SANDBOX": "1", "DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000"},
        "pass_env": ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_BASE_URL",
                     "ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
                     "AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_BEARER_TOKEN_BEDROCK",
                     "ANTHROPIC_VERTEX_PROJECT_ID", "CLOUD_ML_REGION", "API_TIMEOUT_MS"],
        "auth_any": ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK",
                     "CLAUDE_CODE_USE_VERTEX"],
    },
    "codex": {
        "dockerfile": "codex.Dockerfile",
        "image": "vibeprevisbench-agent:codex",
        "command": ('codex exec "$(cat %s)"{model}{effort} --dangerously-bypass-approvals-and-sandbox '
                    '--skip-git-repo-check -C %s --json' % (TASK_FILE, WORKSPACE)),
        "model_flag": " --model %s", "effort_flag": ' -c model_reasoning_effort="%s"',
        "env": {},
        "pass_env": ["CODEX_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"],
        "auth_any": ["CODEX_API_KEY"],
    },
}

LOCK = threading.Lock()
LOG = None


def say(*parts):
    line = " ".join(str(p) for p in parts)
    with LOCK:
        print(line, flush=True)
        if LOG:
            with open(LOG, "a") as f:
                f.write("%s %s\n" % (dt.datetime.now().strftime("%F %T"), line))


def die(msg):
    say("ERROR:", msg)
    sys.exit(2)


def run(cmd, check=False, env=None, quiet=False, **kw):
    if not quiet:
        say("$", " ".join(shlex.quote(str(c)) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], env=env, **kw)
    if check and r.returncode:
        die("command failed (exit %d): %s" % (r.returncode, " ".join(str(c) for c in cmd)))
    return r


def parse_duration(s):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd]?)", s.strip())
    if not m:
        raise argparse.ArgumentTypeError("use e.g. 4h, 90m or 3600")
    return int(float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)])


def find_repo(arg):
    for cand in ([Path(arg)] if arg else []) + [HERE.parents[4], Path.cwd()]:
        if (cand / "vpbench.py").is_file() and (cand / "harness/vpb_harness.py").is_file():
            return cand.resolve()
    die("cannot find the VibePrevisBench repository (vpbench.py); pass --repo")


class Eval:
    def __init__(self, a):
        self.a = a
        self.repo = find_repo(a.repo)
        self.py = sys.executable
        self.vpb = [self.py, str(self.repo / "vpbench.py")]
        self.data = (self.repo / a.data).resolve() if not Path(a.data).is_absolute() else Path(a.data)
        self.run_dir = self.repo / "runs" / a.name
        self.out = self.repo / "results" / a.name
        # subprocesses find the venv's console scripts (yt-dlp) even when the venv is not activated
        self.env = dict(os.environ, PATH=str(Path(self.py).parent) + os.pathsep + os.environ.get("PATH", ""))
        self.preset = PRESETS.get(a.agent) if a.agent else None
        self.blender = None

    # ------------------------------------------------------------------ configuration
    def agent_command(self):
        a = self.a
        if a.agent_cmd:
            return a.agent_cmd
        p = self.preset
        return p["command"].format(model=(p["model_flag"] % shlex.quote(a.model)) if a.model else "",
                                   effort=(p["effort_flag"] % a.effort) if a.effort else "")

    def agent_image(self):
        return self.a.agent_image or (self.preset["image"] if self.preset else SANDBOX_IMAGE)

    def pass_env(self):
        names = list(self.preset["pass_env"]) if self.preset else []
        for n in self.a.pass_env or []:
            names += [x for x in n.split(",") if x]
        return [n for n in dict.fromkeys(names) if n in os.environ]

    def fixed_env(self):
        env = dict(self.preset["env"]) if self.preset else {}
        for kv in self.a.agent_env or []:
            k, _, v = kv.partition("=")
            env[k] = v
        return env

    def index(self):
        p = self.data / "tasks/index.json"
        if not p.is_file():
            return None
        return json.loads(p.read_text())["tasks"]

    def tasks(self):
        idx = self.index()
        if idx is None:
            return []
        ids = [t["task_id"] for t in idx]
        sel = self.a.tasks
        if sel == "all":
            return ids
        want = SMOKE if sel == "smoke" else [x.strip() for x in sel.split(",") if x.strip()]
        bad = [x for x in want if x not in ids]
        if bad:
            die("unknown task ids: " + ",".join(bad))
        return [i for i in ids if i in set(want)]

    def is_full(self, tasks):
        idx = self.index() or []
        return len(tasks) == len(idx)

    def config(self):
        a = self.a
        cfg = {"name": a.name, "agent": a.agent or "custom", "agent_command": self.agent_command(),
               "agent_image": self.agent_image(), "model": a.model, "effort": a.effort,
               "agent_env": self.fixed_env(), "credential_env_names": self.pass_env(),
               "time_limit_s": a.time_limit, "cpus": a.cpus, "memory": a.memory, "network": a.network,
               "protocol": "release", "tasks": a.tasks, "judge_model": os.environ.get("VPB_JUDGE_MODEL", "gemini-4-argon"),
               "judge_base_url": os.environ.get("VPB_JUDGE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")}
        rev = subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"], capture_output=True, text=True)
        cfg["code_revision"] = rev.stdout.strip() or None
        sums = self.data / "SHA256SUMS"
        cfg["dataset_sha256sums"] = hashlib.sha256(sums.read_bytes()).hexdigest() if sums.is_file() else None
        return cfg

    def check_config(self):
        """A run directory belongs to one agent configuration: resuming with another one would mix results."""
        cfg = self.config()
        path = self.out / "config.json"
        key = ["agent", "agent_command", "agent_image", "model", "effort", "agent_env", "time_limit_s", "cpus", "memory",
               "network", "protocol"]
        if path.is_file():
            old = json.loads(path.read_text())
            diff = {k: (old.get(k), cfg.get(k)) for k in key if old.get(k) != cfg.get(k)}
            if diff and not self.a.force_config:
                die("results/%s was started with another configuration %s; use a new --name, or --force-config "
                    "if the change does not affect the agent" % (self.a.name, json.dumps(diff)))
            cfg["started_at"] = old.get("started_at")
        cfg.setdefault("started_at", None)
        cfg["started_at"] = cfg["started_at"] or dt.datetime.now(dt.timezone.utc).strftime("%FT%TZ")
        cfg["updated_at"] = dt.datetime.now(dt.timezone.utc).strftime("%FT%TZ")
        self.out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=1) + "\n")
        return cfg

    # ------------------------------------------------------------------ setup
    def setup(self, tasks):
        say("== setup")
        problems = []
        if sys.version_info < (3, 10):
            problems.append("Python 3.10+ is needed (3.12 recommended), found %s" % sys.version.split()[0])
        miss, optional = self.missing_packages()
        if miss:
            problems.append("missing Python packages %s for %s. Create the environment: python3.12 -m venv .venv && "
                            ".venv/bin/pip install -r requirements.txt, then run this script with .venv/bin/python"
                            % (", ".join(miss), self.py))
        if optional:
            say("note: %s is not installed; it is needed only to fetch YouTube and Bilibili media" % ", ".join(optional))
        for tool in ("docker", "ffmpeg", "ffprobe"):
            if not shutil.which(tool):
                problems.append("%s is not on PATH%s" % (tool, " (every task runs in a Docker container; "
                                "--backend local is not valid for evaluating agents)" if tool == "docker" else ""))
        if shutil.which("docker") and run(["docker", "info"], capture_output=True, quiet=True).returncode:
            problems.append("the Docker daemon is not reachable (is it running, and may this user use it?)")
        if problems:
            die("setup:\n  - " + "\n  - ".join(problems))
        ver = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0]
        say("ffmpeg:", ver)

        # images: the sandbox (agent workspace and replay) and, for a preset, the same image with the agent CLI
        if run(["docker", "image", "inspect", SANDBOX_IMAGE], capture_output=True, quiet=True).returncode:
            say("building %s (downloads Blender 5.2.2; about 10-40 minutes)" % SANDBOX_IMAGE)
            run(["docker", "build", "-t", SANDBOX_IMAGE, str(self.repo / "harness/docker")], check=True)
        img = self.agent_image()
        if img != SANDBOX_IMAGE and run(["docker", "image", "inspect", img], capture_output=True, quiet=True).returncode:
            if not self.preset or self.a.agent_image:
                die("agent image %s does not exist: build it first (see references/agents.md)" % img)
            say("building agent image", img)
            run(["docker", "build", "-t", img, "--build-arg", "BASE=" + SANDBOX_IMAGE, "-f",
                 str(SKILL / "docker" / self.preset["dockerfile"]), str(SKILL / "docker")]
                + [x for kv in self.a.build_arg for x in ("--build-arg", kv)], check=True)
        self.find_blender(build=True)

        # dataset
        if not (self.data / "SHA256SUMS").is_file():
            run(self.vpb + ["download", "--repo", self.a.repo_id, "--data", str(self.data), "--data-only"],
                check=True, env=self.env)
        else:
            r = run(self.vpb + ["verify", "--data", str(self.data)], env=self.env, capture_output=True, text=True)
            if r.returncode:
                die("the dataset in %s does not match SHA256SUMS:\n%s\nre-download it: %s download --data %s --data-only"
                    % (self.data, r.stdout[-1500:], " ".join(self.vpb), self.data))
        tasks = self.tasks()
        self.fetch_media(tasks)
        say("setup ok:", len(tasks), "tasks selected")
        return tasks

    def missing_packages(self):
        """(required, optional) packages of requirements.txt that the interpreter cannot import"""
        need = {"numpy": "numpy", "scipy": "scipy", "huggingface_hub": "huggingface_hub", "cv2": "opencv-python-headless",
                "PIL": "Pillow", "requests": "requests"}
        opt = {"yt_dlp": "yt-dlp"}
        bad = lambda mods: [pkg for mod, pkg in mods.items()
                            if subprocess.run([self.py, "-c", "import " + mod], capture_output=True).returncode]
        return bad(need), bad(opt)

    def preflight(self, tasks):
        """What setup would have to do or would fail on, without changing anything."""
        pf = {}
        miss, optional = self.missing_packages()
        pf["python"] = "%s %s" % (self.py, sys.version.split()[0])
        if miss or optional:
            pf["python_packages_missing"] = miss + ["%s (optional: YouTube/Bilibili media)" % o for o in optional]
        pf["ffmpeg_ffprobe"] = "ok" if shutil.which("ffmpeg") and shutil.which("ffprobe") else "MISSING"
        if not shutil.which("docker"):
            pf["docker"] = "MISSING: a Docker host is required; --backend local is not valid for evaluating agents"
        elif run(["docker", "info"], capture_output=True, quiet=True).returncode:
            pf["docker"] = "installed, but the daemon is not reachable for this user"
        else:
            pf["docker"] = "ok"
            has = lambda img: not run(["docker", "image", "inspect", img], capture_output=True, quiet=True).returncode
            pf["sandbox_image"] = "present" if has(SANDBOX_IMAGE) else "setup will build it (10-40 min)"
            if self.agent_image() != SANDBOX_IMAGE:
                pf["agent_image"] = "present" if has(self.agent_image()) else (
                    "setup will build it" if self.preset and not self.a.agent_image else "MISSING: build it first")
        if self.a.blender == "auto":
            pf["blender"] = "present" if (self.repo / ".cache/blender-5.2.2/blender").is_file() else \
                "setup will copy it from the sandbox image"
        else:
            pf["blender"] = self.a.blender
        if (self.data / "SHA256SUMS").is_file():
            pf["dataset"] = "present (setup verifies SHA256SUMS)"
            if tasks:
                m = self.missing_media(tasks)
                pf["input_media"] = "complete" if not m else "setup will fetch the media of %d tasks: %s" % (
                    len(m), ",".join(m[:10]))
        else:
            pf["dataset"] = "setup will download it (about 130 MB)"
        return pf

    def missing_media(self, tasks):
        r = run(self.vpb + ["fetch-media", "--data", str(self.data), "--tasks", ",".join(tasks), "--check"]
                + self.a.fetch_arg, env=self.env, capture_output=True, text=True, quiet=True)
        return sorted({m.group(1) for m in re.finditer(r"^missing (\S+?)/", r.stdout, re.M)})

    def fetch_media(self, tasks):
        missing = self.missing_media(tasks)
        if missing:
            say("fetching the input media of %d tasks from their sources" % len(missing))
            run(self.vpb + ["fetch-media", "--data", str(self.data), "--tasks", ",".join(missing)] + self.a.fetch_arg,
                env=self.env)
            missing = self.missing_media(tasks)
        if missing:
            msg = ("input media still missing for %d tasks: %s. See data/fetch_report.json and "
                   "docs/external-inputs.md (scan datasets need --fetch-arg=--datasets-root=PATH; videos need yt-dlp)."
                   % (len(missing), ",".join(missing)))
            if not self.a.skip_missing_media:
                die(msg + " Fix the sources, or pass --skip-missing-media to run the other tasks (the result is then "
                          "a subset and these tasks count 0 in the full report).")
            say("WARNING:", msg, "They are skipped.")
        self.skip = set(missing)

    def find_blender(self, build=False):
        if self.blender:
            return self.blender
        if self.a.blender != "auto":
            self.blender = self.a.blender
        else:
            dest = self.repo / ".cache/blender-5.2.2"
            if not (dest / "blender").is_file():
                if not build:
                    die("Blender is not set up: run the setup stage, or pass --blender PATH")
                # the very binary the sandboxes run, copied out of the sandbox image
                say("copying Blender 5.2.2 out of", SANDBOX_IMAGE, "to", dest)
                cid = run(["docker", "create", SANDBOX_IMAGE], check=True, capture_output=True, text=True,
                          quiet=True).stdout.strip()
                tmp = dest.with_name(dest.name + ".tmp")
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.parent.mkdir(parents=True, exist_ok=True)
                try:
                    run(["docker", "cp", cid + ":/opt/blender", str(tmp)], check=True, quiet=True)
                finally:
                    run(["docker", "rm", cid], capture_output=True, quiet=True)
                tmp.rename(dest)
            self.blender = str(dest / "blender")
        r = subprocess.run([self.blender, "-b", "--factory-startup", "--version"], capture_output=True, text=True)
        if r.returncode or "Blender 5.2.2" not in r.stdout:
            die("Blender 5.2.2 does not start on this host (%s):\n%s\nInstall the libraries listed in "
                "references/troubleshooting.md, or pass --blender PATH." % (self.blender, (r.stdout + r.stderr)[-1500:]))
        return self.blender

    # ------------------------------------------------------------------ agent
    def agent_state(self, t):
        p = self.run_dir / t / "agent_run.json"
        return json.loads(p.read_text()) if p.is_file() else None

    def run_agent(self, t, stop):
        if stop.is_set():
            return t, "not started"
        d = self.run_dir / t
        st = self.agent_state(t)
        if st and st.get("status") == "done" and not self.a.rerun_agent and not (self.a.rerun_failed and not st.get("solution")):
            return t, "kept"
        previous = ((st or {}).pop("previous", []) + [st]) if st else []
        if (d / "agent_log.txt").is_file():
            (d / "agent_log.txt").rename(d / ("agent_log.attempt%d.txt" % len(previous)))
        name = "vpb-%s-%s" % (self.a.name, t.lower())
        run(["docker", "rm", "-f", name], capture_output=True, quiet=True)
        if (d / "agent").exists():                          # an interrupted attempt: keep it aside, start fresh
            (d / "agent").rename(d / ("agent.stale-" + time.strftime("%Y%m%d%H%M%S")))
        for f in ("solution.py", "agent_run.json"):
            (d / f).unlink(missing_ok=True)
        self.invalidate(t)
        d.mkdir(parents=True, exist_ok=True)
        env = dict(self.env, VPB_SANDBOX_IMAGE=self.agent_image())
        r = run(self.vpb + ["prepare", "--data", str(self.data), "--task", t, "--run", str(self.run_dir),
                            "--network", self.a.network, "--cpus", self.a.cpus, "--memory", self.a.memory],
                env=env, capture_output=True, text=True, quiet=True)
        (d / "prepare_log.txt").write_text(r.stdout + r.stderr)
        if r.returncode:
            run(["docker", "rm", "-f", name], capture_output=True, quiet=True)
            self.write_state(t, {"status": "prepare_failed", "error": (r.stdout + r.stderr)[-800:], "previous": previous})
            return t, "prepare failed (see %s)" % (d / "prepare_log.txt")
        cmd = ["docker", "exec", "-w", WORKSPACE]
        for k, v in self.fixed_env().items():
            cmd += ["-e", "%s=%s" % (k, v)]
        for n in self.pass_env():                            # value taken from this process: never on a command line
            cmd += ["-e", n]
        cmd += [name, "timeout", "--kill-after=60", str(self.a.time_limit), "bash", "-lc", self.agent_command()]
        t0 = time.time()
        with open(d / "agent_log.txt", "wb") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=self.env).returncode
        secs = round(time.time() - t0, 1)
        c = run(self.vpb + ["collect", "--task", t, "--run", str(self.run_dir)], capture_output=True, text=True,
                quiet=True, env=self.env)
        if c.returncode:
            run(["docker", "rm", "-f", name], capture_output=True, quiet=True)
        has = (d / "solution.py").is_file()
        state = {"status": "done", "exit_code": rc, "timed_out": rc in (124, 137), "seconds": secs, "solution": has,
                 "usage": usage_of(d / "agent_log.txt", self.a.agent), "attempt": len(previous) + 1, "previous": previous}
        self.write_state(t, state)
        return t, "%s in %.0f min, exit %d%s" % ("solution.py" if has else "NO solution", secs / 60, rc,
                                                 " (time limit)" if state["timed_out"] else "")

    def invalidate(self, t, replay=True):
        """Drop everything derived from a task's delivery, so that nothing stale is scored or judged."""
        d = self.run_dir / t
        if replay:
            for f in ("scene.blend", "video.mp4", "replay.json"):
                (d / f).unlink(missing_ok=True)
            shutil.rmtree(d / "replay", ignore_errors=True)
        shutil.rmtree(d / "orbit", ignore_errors=True)
        shutil.rmtree(self.out / "scene/dumps/submission" / t, ignore_errors=True)
        (self.out / "screen" / (t + ".json")).unlink(missing_ok=True)

    def write_state(self, t, state):
        state.update(task_id=t, at=dt.datetime.now(dt.timezone.utc).strftime("%FT%TZ"))
        (self.run_dir / t / "agent_run.json").write_text(json.dumps(state, indent=1) + "\n")

    def agents(self, tasks):
        todo = [t for t in tasks if t not in self.skip]
        say("== agent: %d tasks, %d at a time, limit %ss each, image %s" % (len(todo), self.a.parallel,
                                                                           self.a.time_limit, self.agent_image()))
        if self.preset and not any(n in os.environ for n in self.preset["auth_any"]):
            say("WARNING: none of %s is set; the agent CLI may not be able to reach its model"
                % ", ".join(self.preset["auth_any"]))
        stop, quick_fail, ok = threading.Event(), [], []
        with ThreadPoolExecutor(self.a.parallel) as ex:
            futs = [ex.submit(self.run_agent, t, stop) for t in todo]
            for n, f in enumerate(as_completed(futs), 1):
                t, msg = f.result()
                say("[agent %d/%d] %s: %s" % (n, len(todo), t, msg))
                st = self.agent_state(t) or {}
                if st.get("solution") or msg == "kept":
                    ok.append(t)
                elif st.get("status") == "prepare_failed" or (st.get("seconds", 1e9) < 300):
                    quick_fail.append(t)
                # several quick failures and nothing delivered: the agent is misconfigured, stop wasting time
                if len(quick_fail) >= 3 and not ok and not stop.is_set():
                    stop.set()
                    say("STOPPING: %d tasks failed within 5 minutes without a solution (%s). Last agent output:\n%s"
                        % (len(quick_fail), ",".join(quick_fail), tail(self.run_dir / quick_fail[-1] / "agent_log.txt")))
        if stop.is_set():
            for t in quick_fail:                       # not a result of the agent: run them again next time
                st = self.agent_state(t)
                if st and not st.get("solution"):
                    st["status"] = "aborted"
                    self.write_state(t, st)
            die("the agent stage was stopped; fix the agent configuration and run the same command again")

    # ------------------------------------------------------------------ replay, score, judge, report
    def sub(self, tasks):
        return [] if self.is_full(tasks) else ["--tasks", ",".join(tasks)]

    def replay(self, tasks):
        # a delivery is replayed once; again only if solution.py changed since (a failed replay is final, as a failed
        # delivery scores 0 however often it is rerun)
        todo = []
        for t in tasks:
            sol, rec = self.run_dir / t / "solution.py", self.run_dir / t / "replay.json"
            if sol.is_file() and (not rec.is_file() or rec.stat().st_mtime < sol.stat().st_mtime):
                todo.append(t)
        say("== replay: %d deliveries to replay" % len(todo))
        if todo:
            for t in todo:
                self.invalidate(t, replay=False)
            run(self.vpb + ["replay", "--data", str(self.data), "--run", str(self.run_dir), "--jobs",
                            str(self.a.replay_jobs), "--force", "--tasks", ",".join(todo)], env=self.env)
            # exit 1 = some deliveries failed the gates: they score 0, go on

    def score(self, tasks):
        say("== score (Rule-based Scene Check)")
        r = run(self.vpb + ["score", "--data", str(self.data), "--run", str(self.run_dir), "--blender",
                            self.find_blender(), "--out", str(self.out / "scene")] + self.sub(tasks), env=self.env)
        if r.returncode:
            die("scoring failed; see the output above")

    def judge(self, tasks):
        say("== judge (VLM-based Screen Check)")
        if not (os.environ.get("VPB_JUDGE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            say("no VPB_JUDGE_API_KEY / GEMINI_API_KEY: Screen Check skipped (the report shows Scene only)")
            return
        run(self.vpb + ["orbit", "--data", str(self.data), "--run", str(self.run_dir), "--blender", self.find_blender()]
            + self.sub(tasks), env=self.env)
        run(self.vpb + ["judge", "--data", str(self.data), "--run", str(self.run_dir), "--scores", str(self.out / "scene"),
                        "--out", str(self.out / "screen")] + (["--workers", str(self.a.judge_workers)]
                                                             if self.a.judge_workers else []) + self.sub(tasks),
            env=self.env)

    def report(self, tasks):
        say("== report")
        if not (self.out / "scene/scores.csv").is_file():
            die("no Scene scores yet: run the score stage first")
        screen_dir = self.out / "screen"
        cmd = self.vpb + ["report", "--data", str(self.data), "--scores", str(self.out / "scene"), "--out",
                          str(self.out / "report.json")]
        if screen_dir.is_dir():
            cmd += ["--screen", str(screen_dir)]
        run(cmd, env=self.env, capture_output=True, quiet=True)
        rep = json.loads((self.out / "report.json").read_text())
        summary = summarize(self, rep, tasks)
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        md = render_summary(summary)
        (self.out / "summary.md").write_text(md)
        print(md)

    # ------------------------------------------------------------------ status
    def status(self, tasks):
        rows = {"selected": len(tasks)}
        st = [self.agent_state(t) or {} for t in tasks]
        rows["agent finished"] = sum(1 for s in st if s.get("status") == "done")
        rows["agent delivered solution.py"] = sum(1 for s in st if s.get("solution"))
        rows["agent hit the time limit"] = sum(1 for s in st if s.get("timed_out"))
        rows["agent prepare failed"] = sum(1 for s in st if s.get("status") == "prepare_failed")
        quick = [t for t, s in zip(tasks, st) if s.get("status") == "done" and not s.get("solution")
                 and s.get("seconds", 1e9) < 300]
        rows["ended within 5 min without solution"] = "%d%s" % (len(quick), (" (%s)" % ",".join(quick[:8])) if quick else "")
        rows["agent retries"] = sum(1 for s in st if s.get("attempt", 1) > 1)
        running = run(["docker", "ps", "--filter", "name=vpb-%s-" % self.a.name, "--format", "{{.Names}}"],
                      capture_output=True, text=True, quiet=True).stdout.split() if shutil.which("docker") else []
        rows["containers running"] = len(running)
        rows["replayed ok (scene.blend)"] = sum(1 for t in tasks if (self.run_dir / t / "scene.blend").is_file())
        sc = self.out / "scene/scores.csv"
        if sc.is_file():
            with open(sc) as f:
                rows["Scene scored"] = sum(1 for r in csv.DictReader(f) if r["task_id"] in set(tasks)
                                           and r.get("status") == "scored")
        rows["Screen judged"] = sum(1 for t in tasks if (self.out / "screen" / (t + ".json")).is_file())
        width = max(len(k) for k in rows)
        print("\n".join("%-*s  %s" % (width, k, v) for k, v in rows.items()))
        secs = [s["seconds"] for s in st if s.get("seconds")]
        if secs:
            print("agent time per task: median %.0f min, max %.0f min" % (statistics.median(secs) / 60, max(secs) / 60))
        if (self.out / "summary.md").is_file():
            print("\nlast report: %s" % (self.out / "summary.md"))


def tail(path, n=25):
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def usage_of(log, agent):
    """Best-effort token/cost totals from the agent's JSON output (Claude Code result object, Codex JSONL events)."""
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return None
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "result" and "total_cost_usd" in ev:          # Claude Code --output-format json
            u = ev.get("usage") or {}
            out = {"cost_usd": ev.get("total_cost_usd"), "turns": ev.get("num_turns"),
                   "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
                   "cache_read_tokens": u.get("cache_read_input_tokens")}
        elif ev.get("type") == "turn.completed" and isinstance(ev.get("usage"), dict):   # Codex --json
            for k, v in ev["usage"].items():
                if isinstance(v, (int, float)):
                    out[k] = out.get(k, 0) + v
    return out or None


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def published(repo, tasks):
    """Each published model's Scene and Screen on exactly these tasks (results/cells.csv; failed = 0)."""
    path = repo / "results/cells.csv"
    names = {}
    lb = repo / "results/leaderboard.csv"
    if lb.is_file():
        with open(lb, encoding="utf-8-sig") as f:
            names = {r["model"]: r["name"] for r in csv.DictReader(f)}
    if not path.is_file():
        return []
    cells = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cells.setdefault(r["model"], {})[r["item"]] = r
    out = []
    for model, rows in cells.items():
        sc = [float(rows[t]["rule"] or 0) if t in rows and rows[t]["status"] == "scored" else 0.0 for t in tasks]
        vl = [float(rows[t]["vlm_gemini4"] or 0) if t in rows and rows[t]["status"] == "scored" else 0.0 for t in tasks]
        out.append({"model": names.get(model, model), "scene": mean(sc), "screen": mean(vl),
                    "overall": (mean(sc) + mean(vl)) / 2})
    return sorted(out, key=lambda r: -r["overall"])


def summarize(ev, rep, tasks):
    sel = set(tasks)
    rows = [r for r in rep["results"] if r["task_id"] in sel]
    has_screen = rep.get("screen") is not None
    pending = [t for t in rep.get("pending_screen") or [] if t in sel]
    s = {"name": ev.a.name, "tasks": len(rows), "full_benchmark": ev.is_full(tasks), "protocol": "release",
         "scene": mean([r["scene"] for r in rows]),
         "screen": mean([r["screen"] for r in rows]) if has_screen else None,
         "screen_complete": has_screen and not pending, "pending_screen": pending}
    s["overall"] = (s["scene"] + s["screen"]) / 2 if s["screen_complete"] else None
    s["by_kind"] = {}
    for k in KINDS:
        rk = [r for r in rows if r["kind"] == k]
        if rk:
            s["by_kind"][k] = {"tasks": len(rk), "scene": mean([r["scene"] for r in rk]),
                               "screen": mean([r["screen"] for r in rk]) if has_screen else None}
    states = [ev.agent_state(t) or {} for t in tasks]
    s["delivered"] = sum(1 for x in states if x.get("solution"))
    s["timed_out"] = sum(1 for x in states if x.get("timed_out"))
    s["retried"] = sorted(t for t, x in zip(tasks, states) if x.get("attempt", 1) > 1)
    s["replay_ok"] = sum(1 for t in tasks if (ev.run_dir / t / "scene.blend").is_file())
    s["replay_retried"] = []                     # the harness retries environment failures of a replay (at most 3 tries)
    for t in tasks:
        rec = ev.run_dir / t / "replay.json"
        if rec.is_file() and len(json.loads(rec.read_text()).get("attempts") or []) > 1:
            s["replay_retried"].append(t)
    s["skipped_missing_media"] = sorted(getattr(ev, "skip", set()) & sel)
    secs = [x["seconds"] for x in states if x.get("seconds")]
    s["agent_minutes_median"] = statistics.median(secs) / 60 if secs else None
    costs = [((x.get("usage") or {}).get("cost_usd")) for x in states]
    s["agent_cost_usd_total"] = sum(c for c in costs if isinstance(c, (int, float))) or None
    s["published_same_tasks"] = published(ev.repo, tasks)
    s["per_task"] = rows
    s["config"] = json.loads((ev.out / "config.json").read_text()) if (ev.out / "config.json").is_file() else None
    return s


def fmt(x):
    return "–" if x is None else "%.2f" % x


def render_summary(s):
    c = s.get("config") or {}
    scope = ("all %d tasks" % s["tasks"]) if s["full_benchmark"] else (
        "**subset of %d tasks** (not comparable with the 242-task leaderboard; the table below scores the published "
        "models on the same tasks)" % s["tasks"])
    lines = ["# VibePrevisBench result: %s" % s["name"], "",
             "Agent: %s%s%s · protocol %s · %s" % (c.get("agent", "not recorded"), (" · model %s" % c["model"]) if c.get("model") else "",
                                                   (" · effort %s" % c["effort"]) if c.get("effort") else "",
                                                   s["protocol"], scope), "",
             "| Overall | Scene | Screen |", "| ---: | ---: | ---: |",
             "| %s | %s | %s |" % (fmt(s["overall"]), fmt(s["scene"]), fmt(s["screen"])), ""]
    if not s["screen_complete"]:
        lines += ["Screen Check is %s, so there is no Overall score." % (
            "missing (no judge key was set)" if s["screen"] is None else
            "incomplete for %d tasks (%s)" % (len(s["pending_screen"]), ",".join(s["pending_screen"][:10]))), ""]
    lines += ["| Input kind | Tasks | Scene | Screen |", "| --- | ---: | ---: | ---: |"]
    for k, v in s["by_kind"].items():
        lines.append("| %s | %d | %s | %s |" % (KIND_NAMES[k], v["tasks"], fmt(v["scene"]), fmt(v["screen"])))
    lines += ["", "Deliveries: %d of %d tasks left solution.py, %d passed the replay gates, %d hit the time limit."
              % (s["delivered"], s["tasks"], s["replay_ok"], s["timed_out"])]
    if s["agent_minutes_median"] is not None:
        lines.append("Agent time per task: median %.0f min." % s["agent_minutes_median"])
    if s["agent_cost_usd_total"]:
        lines.append("Agent cost reported by the CLI: $%.2f in total." % s["agent_cost_usd_total"])
    if s["replay_retried"]:
        lines.append("Replay retried after an environment failure: %s." % ", ".join(s["replay_retried"]))
    if s["retried"]:
        lines.append("Agent rerun after an infrastructure failure: %s." % ", ".join(s["retried"]))
    if s["skipped_missing_media"]:
        lines.append("Skipped (input media missing): %s." % ", ".join(s["skipped_missing_media"]))
    pub = s["published_same_tasks"]
    if pub:
        note = ""
        if "image" in s["by_kind"]:
            note = (" On photo tasks the published agents received the paper's photo-only briefs, while this run used "
                    "the release briefs, which add an orbit (Scene scores read the photo frames only).")
        lines += ["", "## Published agents on the same tasks", "",
                  "Paper results (`results/cells.csv`), Screen judged by Gemini 4 Argon." + note, "",
                  "| Agent | Overall | Scene | Screen |", "| --- | ---: | ---: | ---: |"]
        mine = {"model": "**%s (this run)**" % s["name"], "scene": s["scene"], "screen": s["screen"], "overall": s["overall"]}
        rows = pub + [mine]
        by = "overall" if s["overall"] is not None else "scene"      # without Screen, rank by Scene alone
        rows.sort(key=lambda r: -r[by])
        for r in rows:
            lines.append("| %s | %s | %s | %s |" % (r["model"], fmt(r["overall"]), fmt(r["scene"]), fmt(r["screen"])))
    lines += ["", "Per-task scores: `results/%s/report.json`; per-check reports: `results/%s/scene/engine/`; agent logs: "
              "`runs/%s/<task_id>/agent_log.txt`." % (s["name"], s["name"], s["name"]), ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name: runs/<name>, results/<name> (letters, digits, _ . -)")
    g = ap.add_argument_group("agent")
    g.add_argument("--agent", choices=sorted(PRESETS), help="a preset harness (the paper's two)")
    g.add_argument("--agent-cmd", help="custom: shell command started in /work/blender_agent; the task is in "
                                       + TASK_FILE + "; it must leave solution/solution.py")
    g.add_argument("--agent-image", help="image with the agent installed (default: the preset's, or the sandbox image)")
    g.add_argument("--model", help="model name passed to the preset CLI")
    g.add_argument("--effort", help="reasoning effort passed to the preset CLI (paper: max for GPT models; see README)")
    g.add_argument("--pass-env", action="append", help="names of host environment variables to pass into the agent "
                                                       "container (comma-separated; values are never logged)")
    g.add_argument("--agent-env", action="append", help="NAME=VALUE for non-secret settings of the agent")
    g.add_argument("--time-limit", type=parse_duration, default=parse_duration("4h"), help="per task (paper: 4h)")
    g.add_argument("--parallel", type=int, default=max(1, min(4, (os.cpu_count() or 8) // 8)),
                   help="agent containers at once (8 CPUs and 16 GB each)")
    g.add_argument("--cpus", default="8", help="CPU limit of an agent container (paper: 8)")
    g.add_argument("--memory", default="16g", help="memory limit of an agent container (paper: 16g)")
    g.add_argument("--network", default="bridge", help="Docker network of the agent containers (needs the model API)")
    g.add_argument("--rerun-agent", action="store_true", help="run the agent again on tasks that already finished")
    g.add_argument("--rerun-failed", action="store_true",
                   help="run the agent again on finished tasks without solution.py (for infrastructure failures only: "
                        "an API outage, a crashed host; never to retry tasks the agent itself failed)")
    g = ap.add_argument_group("tasks and data")
    g.add_argument("--tasks", default="smoke", help="smoke (4 tasks, one per input kind), all, or ID,ID,...")
    g.add_argument("--data", default="data", help="dataset folder (default data/ in the repository)")
    g.add_argument("--repo-id", default=os.environ.get("VPB_HF_REPO", "AidanYuzhe/vibeprevisbench"))
    g.add_argument("--fetch-arg", action="append", default=[], help="extra argument for fetch-media, e.g. "
                                                                    "--fetch-arg=--datasets-root=/data/scannetpp")
    g.add_argument("--skip-missing-media", action="store_true", help="run the tasks whose media could be fetched")
    g = ap.add_argument_group("evaluation")
    g.add_argument("--stages", default=",".join(STAGES), help="comma-separated subset of " + ",".join(STAGES))
    g.add_argument("--blender", default="auto", help="Blender 5.2.2 for scoring (auto: copied from the sandbox image)")
    g.add_argument("--replay-jobs", type=int, default=2, help="replay containers at once (4 CPUs and 8 GB each)")
    g.add_argument("--judge-workers", type=int, help="parallel judge requests")
    g.add_argument("--build-arg", action="append", default=[], metavar="NAME=VALUE",
                   help="docker build argument for the images, e.g. CLAUDE_CODE_VERSION=2.1.308")
    g.add_argument("--repo", help="repository root (default: found from this script)")
    g.add_argument("--force-config", action="store_true", help="resume although the agent configuration changed")
    g = ap.add_argument_group("inspection")
    g.add_argument("--status", action="store_true", help="show progress and exit")
    g.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    a = ap.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", a.name):
        ap.error("--name may contain letters, digits, _ . - only")
    stages = [s for s in a.stages.split(",") if s]
    if set(stages) - set(STAGES):
        ap.error("unknown stages: " + ",".join(sorted(set(stages) - set(STAGES))))
    ev = Eval(a)
    ev.skip = set()
    needs_agent = "agent" in stages and not (a.status or a.dry_run)
    if needs_agent and bool(a.agent) == bool(a.agent_cmd):
        ap.error("give exactly one of --agent (claude-code, codex) or --agent-cmd")
    if (a.model or a.effort) and not a.agent and not (a.status or a.dry_run):
        ap.error("--model and --effort only apply to --agent presets; put them into --agent-cmd instead")

    if a.status:
        tasks = ev.tasks()
        if not tasks:
            die("no dataset in %s yet" % ev.data)
        return ev.status(tasks)
    if a.dry_run:
        tasks = ev.tasks()
        plan = {"repository": str(ev.repo), "data": str(ev.data), "runs": str(ev.run_dir), "results": str(ev.out),
                "stages": stages, "tasks": a.tasks if not tasks else "%d (%s%s)" % (
                    len(tasks), ",".join(tasks[:6]), ",..." if len(tasks) > 6 else ""),
                "agent_image": ev.agent_image(), "agent_command": ev.agent_command() if (a.agent or a.agent_cmd) else None,
                "agent_env": ev.fixed_env(), "credentials_passed": ev.pass_env(),
                "credentials_not_set": [n for x in a.pass_env or [] for n in x.split(",") if n and n not in os.environ],
                "time_limit_s": a.time_limit, "parallel": a.parallel, "network": a.network,
                "credentials_warning": ("none of %s is set: the agent cannot reach its model"
                                        % ", ".join(ev.preset["auth_any"]))
                if ev.preset and not any(n in os.environ for n in ev.preset["auth_any"]) else None,
                "judge": "on" if (os.environ.get("VPB_JUDGE_API_KEY") or os.environ.get("GEMINI_API_KEY")) else
                "off (no VPB_JUDGE_API_KEY / GEMINI_API_KEY)"}
        plan["preflight"] = ev.preflight(tasks)
        print(json.dumps({k: v for k, v in plan.items() if v is not None}, indent=1))
        return 0

    global LOG
    ev.out.mkdir(parents=True, exist_ok=True)
    LOG = ev.out / "run_eval.log"
    say("== VibePrevisBench evaluation %s (%s)" % (a.name, ", ".join(stages)))
    if "setup" in stages:
        tasks = ev.setup(ev.tasks())
    else:
        tasks = ev.tasks()
        if not tasks:
            die("no dataset in %s: run the setup stage first" % ev.data)
    if not tasks:
        die("no tasks selected")
    if "agent" in stages:
        ev.check_config()
        if "setup" not in stages:
            ev.skip = set(ev.missing_media(tasks))
            if ev.skip and not a.skip_missing_media:
                die("input media missing for %s: run the setup stage" % ",".join(sorted(ev.skip)))
        ev.agents(tasks)
    for st in ("replay", "score", "judge", "report"):
        if st in stages:
            getattr(ev, st)(tasks)
    say("== done: %s" % (ev.out / "summary.md" if "report" in stages else ev.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
