#!/usr/bin/env python3
"""The agent's workspace and the replay of its delivery, with the frozen harness of the paper (blender_agent/).

    python3 harness/vpb_harness.py prepare --task B250_190 --run runs/my_agent [--data data] [--network bridge] [--cpus 8] [--memory 16g]
    python3 harness/vpb_harness.py collect --task B250_190 --run runs/my_agent
    python3 harness/vpb_harness.py replay  --run runs/my_agent [--tasks ...] [--protocol release|paper] [--jobs 2]

prepare starts a Docker container from the sandbox image (harness/docker/) and builds the workspace exactly as for the
paper's runs: /work/blender_agent with the managed `blender` command, pvtools, execution_config.json and ledger.json;
input/entities.json, input/categories.json and the input media with their 1 fps frames and 768-pixel viewing copies;
an empty solution/; and the task file /opt/task/TASK.md (the brief, a blank line, agent_instructions.txt). The container
keeps running: start your agent inside it (docker exec -it <container> bash) with TASK.md as its task, in
/work/blender_agent. collect copies solution/solution.py to runs/<system>/<task_id>/solution.py and removes the container.

replay executes each runs/<system>/<task_id>/solution.py once, from scratch, in a fresh container (the harness's
end-to-end final stage with a fixture model whose one answer is the file, verbatim) and writes next to it scene.blend,
video.mp4 and replay/ (the full run record). --protocol release (default) replays against the brief in
tasks/<ID>/task.json; --protocol paper against rubrics/<ID>/task.json, the brief the paper's agents received (photo
tasks: without the orbit). A task without solution.py is skipped (it scores 0).

The workspace on a laptop without Docker: --backend local runs the same steps in a temporary directory on the host
(no isolation; the agent then works there instead of /work/blender_agent; needs blender 5.2.2 and ffmpeg on PATH).
"""
import argparse, json, shlex, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
WS = "/work/blender_agent"
TASK_FILE = "/opt/task/TASK.md"


def task_of(data, task_id, protocol="release"):
    """(the brief's task.json, the full task text, the input files, the media paths)"""
    tj = data / ("tasks" if protocol == "release" else "rubrics") / task_id / "task.json"
    task = json.loads(tj.read_text())
    text = task["prompt"] + "\n\n" + (data / "agent_instructions.txt").read_text()
    inputs = {"input/entities.json": data / "tasks" / task_id / "entities.json", "input/categories.json": data / "categories.json"}
    media = [data / "tasks" / task_id / m["path"] for m in task.get("media") or []]
    missing = [str(p) for p in media if not p.is_file()]
    if missing:
        sys.exit("input media missing (python3 vpbench.py fetch-data): " + ", ".join(missing))
    return task, text, inputs, media


def prepare(a):
    from blender_agent import docker_holder
    from blender_agent.common import Deadline, Recorder, write_json
    from blender_agent.execution import Workspace
    from blender_agent.tools import input_media
    task, text, inputs, media = task_of(a.data, a.task)
    spec = task["output_spec"]
    out = a.run / a.task
    agent = out / "agent"
    if agent.exists():
        sys.exit("%s exists: collect or remove it first" % agent)
    name = "vpb-%s-%s" % (a.run.name, a.task.lower())
    docker_holder.DEFAULTS.update(name=name, network=a.network, keep=True, cpus=a.cpus, memory=a.memory)
    rec = Recorder(agent)
    ws = Workspace(rec, Deadline(None), a.backend, "blender", "ffmpeg", 10 ** 6, spec["fps"],
                   tuple(spec["final_resolution"]), tuple(spec["final_resolution"]))
    ws.start()
    for rel, src in inputs.items():
        ws.put(rel, Path(src).read_bytes())
    if media:
        _, manifest = input_media(ws, media)
        write_json(agent / "media_manifest.json", manifest)
    ws.raw("mkdir -p %s" % shlex.quote(ws.root + "/solution"))
    if a.backend == "docker":
        ws.holder.run("mkdir -p /opt/task && printf 'export PATH=%s/bin:$PATH\\n' > /etc/profile.d/zz-blender-agent.sh" % WS,
                      timeout=30)
        ws.holder.write_bytes(TASK_FILE, text.encode())
        where = {"container": ws.holder.sandbox_id[:12], "name": name, "task_file": TASK_FILE, "workspace": ws.root}
    else:
        Path(ws.root, "TASK.md").write_text(text)
        where = {"task_file": str(Path(ws.root, "TASK.md")), "workspace": ws.root}
    write_json(agent / "prepared.json", dict(where, task_id=a.task, backend=a.backend, at=time.strftime("%FT%TZ", time.gmtime())))
    print(json.dumps(where, indent=1))
    if a.backend == "docker":
        print("\nStart your agent in the container, for example:\n  docker exec -it %s bash -l\n  cd %s  # task: %s\n"
              "Then: python3 harness/vpb_harness.py collect --task %s --run %s" % (name, WS, TASK_FILE, a.task, a.run))


def collect(a):
    import subprocess
    out = a.run / a.task
    p = json.loads((out / "agent/prepared.json").read_text())
    dst = out / "solution.py"
    if p["backend"] == "docker":
        r = subprocess.run(["docker", "exec", p["name"], "cat", p["workspace"] + "/solution/solution.py"], capture_output=True)
        if r.returncode:
            print("no solution/solution.py in %s: the task scores 0" % p["name"])
        else:
            dst.write_bytes(r.stdout)
        subprocess.run(["docker", "cp", p["name"] + ":" + p["workspace"], str(out / "agent/workspace")], capture_output=True)
        if not a.keep:
            subprocess.run(["docker", "rm", "-f", p["name"]], capture_output=True)
    else:
        src = Path(p["workspace"]) / "solution/solution.py"
        if src.is_file():
            shutil.copy2(src, dst)
        shutil.copytree(p["workspace"], out / "agent/workspace", dirs_exist_ok=True, symlinks=True)
    print(dst if dst.exists() else "no solution", flush=True)


def replay_one(data, run, task_id, protocol, backend):
    """solution.py -> scene.blend, video.mp4 (the paper's replay: end_to_end final stage, fixture model)"""
    import blender_agent.runner as runner
    from blender_agent.execution import Workspace
    from blender_agent.fixture import ModelFixture
    out = run / task_id
    code = (out / "solution.py").read_text()
    task, _, inputs, media = task_of(data, task_id, protocol)
    spec = task["output_spec"]
    rdir = out / "replay"
    for d in [rdir] + sorted(out.glob("replay.infra*")):
        if d.exists():
            shutil.rmtree(d)
    for f in ("scene.blend", "video.mp4"):
        (out / f).unlink(missing_ok=True)
    runner.extract_code = lambda text: code                      # the file verbatim
    attempts = []
    for k in range(3):                                           # infrastructure failures only are retried
        fixture = ModelFixture([{"role": "assistant", "content": "```python\n" + code + "```"}] * 3).__enter__()
        try:
            res = runner.run_task(task["prompt"], "end_to_end", fixture.model(), rdir, max_rounds=1, timeout_s=None,
                                  backend=backend, fps=spec["fps"], resolution=tuple(spec["preview_resolution"]),
                                  final_resolution=tuple(spec.get("final_resolution") or spec["preview_resolution"]),
                                  media=media, expected_frame_count=spec["frame_count"], input_files=inputs)
        finally:
            fixture.__exit__(None, None, None)
        attempts.append({"status": res.get("status"), "stop_reason": res.get("stop_reason"),
                         "failure_category": res.get("failure_category"), "error": str(res.get("error"))[:500]})
        if res.get("failure_category") != "environment_failure" or k == 2:
            break
        rdir.rename(out / ("replay.infra%d" % k))                   # kept for inspection
    final = rdir / "final"
    validation = json.loads((rdir / "delivery/validation.json").read_text()) if (rdir / "delivery/validation.json").is_file() else {}
    ok = validation.get("status") == "ok" and (final / "scene.blend").is_file() and (final / "video.mp4").is_file()
    if ok:
        shutil.copy2(final / "scene.blend", out / "scene.blend")
        shutil.copy2(final / "video.mp4", out / "video.mp4")
    (out / "replay.json").write_text(json.dumps({"task_id": task_id, "protocol": protocol, "ok": ok, "attempts": attempts},
                                                indent=1) + "\n")
    return task_id, ("ok" if ok else "failed: %s" % attempts[-1])


def replay(a):
    # the frozen layout of the paper's replay: the fresh workspace has the empty solution/ the agent's workspace had
    from blender_agent.execution import Workspace
    _start = Workspace.start

    def start(self):
        ws = _start(self)
        r = self.raw("mkdir -p %s && test -d %s && test -z \"$(ls -A %s)\"" % ((shlex.quote(self.root + "/solution"),) * 3))
        if r["exit_code"]:
            raise RuntimeError("cannot create the empty solution/ in %s: %s" % (self.root, r.get("stderr", "")[-300:]))
        return ws
    Workspace.start = start
    index = json.loads((a.data / "tasks/index.json").read_text())["tasks"]
    want = set(a.tasks.split(",")) if a.tasks else None
    if want and not want <= {t["task_id"] for t in index}:
        sys.exit("unknown task ids: " + ",".join(sorted(want - {t["task_id"] for t in index})))
    todo = [t["task_id"] for t in index if (want is None or t["task_id"] in want) and (a.run / t["task_id"] / "solution.py").is_file()]
    if not a.force:
        todo = [t for t in todo if not (a.run / t / "scene.blend").is_file()]
    print("replaying %d tasks" % len(todo), flush=True)
    # tasks run in worker processes, one at a time in each: the replay patches runner.extract_code with the task's file
    failed = []
    with ProcessPoolExecutor(a.jobs, mp_context=multiprocessing.get_context("fork")) as ex:
        futs = [ex.submit(replay_one, a.data, a.run, t, a.protocol, a.backend) for t in todo]
        for f in futs:
            task_id, status = f.result()
            print("replay", task_id, status, flush=True)
            if status != "ok":
                failed.append(task_id)
    return int(bool(failed))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "collect", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True, type=Path, help="runs/<system>")
        p.add_argument("--data", type=Path, default=HERE.parent / "data")
        p.add_argument("--backend", choices=["docker", "local"], default="docker")
        if name in ("prepare", "collect"):
            p.add_argument("--task", required=True)
        if name == "prepare":
            p.add_argument("--network", default="bridge", help="docker network of the agent's container (default bridge)")
            p.add_argument("--cpus", default="8", help="CPU limit of the agent's container (default 8, the paper's agent sandbox)")
            p.add_argument("--memory", default="16g", help="memory limit of the agent's container (default 16g, the paper's agent sandbox)")
        if name == "collect":
            p.add_argument("--keep", action="store_true", help="keep the container")
        if name == "replay":
            p.add_argument("--tasks")
            p.add_argument("--protocol", choices=["release", "paper"], default="release")
            p.add_argument("--jobs", type=int, default=2, help="containers at once (4 CPU / 8 GB each)")
            p.add_argument("--force", action="store_true", help="replay tasks that already have scene.blend")
    a = ap.parse_args(argv)
    a.data, a.run = a.data.resolve(), a.run.resolve()
    return {"prepare": prepare, "collect": collect, "replay": replay}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
