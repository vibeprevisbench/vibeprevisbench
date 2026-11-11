"""Managed Blender command installed inside each task workspace.

Used by the blender tool AND bash/python subprocess calls. The journal records
actual executions and stores immutable source copies before running Blender.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import struct
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", help="complete bpy script path")
    ap.add_argument("--version", action="store_true")
    ap.add_argument("--purpose", choices=["render", "query", "final"], default="render")
    ap.add_argument("--blend-file")
    ap.add_argument("--continue-from", help="harness-only interrupted final directory")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root / "execution_config.json").read_text())
    if args.version:
        subprocess.run([config['blender'],'--version'],check=True,timeout=10)
        return
    if not args.python:
        ap.error('--python is required for render/query execution')
    source = Path(args.python).resolve()
    if root not in source.parents:
        raise ValueError("script must be inside task workspace")
    code = source.read_text()
    with (root / "ledger.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads((root / "ledger.json").read_text())
        if config.get("deadline_epoch") is not None and time.time() >= config["deadline_epoch"]:
            raise TimeoutError("task budget exhausted")
        if args.purpose == "render":
            if ledger["rounds"] >= config["max_rounds"]:
                raise RuntimeError("execution round budget exhausted")
            ledger["rounds"] += 1
            directory = root / f"rounds/{ledger['rounds']:03d}"
            ledger["last_source"] = str(directory / "solution.py")
        elif args.purpose == "final":
            if not os.environ.get("BA_FINAL_RENDER"):
                raise ValueError("final rendering is reserved for the harness")
            directory = root / "final"
        else:
            if not args.blend_file:
                raise ValueError("query requires --blend-file")
            ledger["queries"] += 1
            directory = root / f"queries/{ledger['queries']:03d}"
        directory.mkdir(parents=True, exist_ok=False)
        if (root / "input").is_dir() and not (directory / "input").exists():
            # Scripts resolving input/ next to themselves (Path(__file__).parent) find the task inputs.
            (directory / "input").symlink_to(root / "input", target_is_directory=True)
        (directory / "solution.py").write_text(code)
        (root / "ledger.json").write_text(json.dumps(ledger))
    (directory / "executed.py").write_text(code)
    job = {"run_dir": str(directory), "code_file": str(directory / "executed.py"),
           "res": config.get("final_resolution", config["resolution"]) if args.purpose == "final" else config["resolution"],
           "fps": config["fps"], "engine": config.get("render_engine", "MODEL_CHOICE"), "readback": False,
           "keyframes": 0, "render_all": args.purpose != "query", "save_blend": args.purpose != "query"}
    if args.purpose == "query":
        blend = Path(args.blend_file).resolve()
        if root not in blend.parents:
            raise ValueError("blend must be inside workspace")
        job["blend_file"] = str(blend)
    if args.continue_from:
        if args.purpose != "final" or not os.environ.get("BA_FINAL_RENDER"):
            raise ValueError("frame continuation is reserved for final recovery")
        prior = Path(args.continue_from).resolve()
        if root not in prior.parents:
            raise ValueError("recovery directory must be inside workspace")
        if (prior / "executed.py").read_text() != code:
            raise ValueError("recovery source does not match executed script")
        old_job=json.loads((prior/"job.json").read_text())
        if old_job['res']!=job['res'] or old_job['fps']!=job['fps']:
            raise ValueError("recovery render specification differs")
        for frame in (prior/'frames').glob('f_*.png'):
            with frame.open('rb') as png:
                header=png.read(24)
                png.seek(-8,2)
                tail=png.read()
            if header[:8]!=b'\x89PNG\r\n\x1a\n' or tail!=b'IEND\xaeB`\x82' or struct.unpack('>II',header[16:24])!=tuple(job['res']):
                raise ValueError("incomplete or mismatched recovery frame: "+frame.name)
        shutil.copy2(prior/'scene.blend',directory/'scene.blend')
        shutil.copytree(prior/'frames',directory/'frames')
        job.update(blend_file=str(directory/'scene.blend'),continue_render=True)
        old_state=json.loads((prior/'stages.json').read_text())
        job['source_provenance']={key:old_state[key] for key in
                                 ('source_sha256','script_dependencies','runtime_representation') if key in old_state}
    (directory / "job.json").write_text(json.dumps(job))
    command = [config["blender"], "-b", "--factory-startup", "-t", "4", "--python-exit-code", "1",
               "--python", str(root / "runner.py"), "--", str(directory / "job.json")]
    began = time.time()
    rc = 1
    error = None
    stages = {}
    render_budget = None
    try:
        remaining = config["deadline_epoch"] - time.time() if config.get("deadline_epoch") is not None else None
        with (directory / "blender.log").open("w") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                render_budget = remaining if remaining is not None else (3600 if args.purpose == "query" else None)
                rc = proc.wait(timeout=max(0.01, render_budget) if render_budget is not None else None)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                stage_path = directory / "stages.json"
                stages = json.loads(stage_path.read_text()) if stage_path.exists() else {}
                rc = 124
                stages.update(status="failed", error="render time allocation exhausted")
                stage_path.write_text(json.dumps(stages))
                raise TimeoutError("Blender render time allocation exhausted; reduce samples/scene cost or select Workbench and retry within the remaining budget")
        stage_path = directory / "stages.json"
        stages = json.loads(stage_path.read_text()) if stage_path.exists() else {}
        if rc or stages.get("status") != "ok":
            error = stages.get("error") or "Blender failed; see blender.log"
        if args.purpose != "query" and not (directory / "scene.blend").exists():
            error = error or "engineering file not saved"
        frames = sorted((directory / "frames").glob("f_*.png"))
        if args.purpose != "query" and not error:
            expected = stages["frame_range"][1] - stages["frame_range"][0] + 1
            if len(frames) != expected:
                raise RuntimeError(f"incomplete render: {len(frames)}/{expected}")
            cmd = [config["ffmpeg"], "-y", "-loglevel", "error", "-framerate", str(config["fps"]),
                   "-start_number", str(stages["frame_range"][0]), "-i", str(directory / "frames/f_%04d.png"),
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", str(directory / ("video.mp4" if args.purpose == "final" else "preview.mp4"))]
            subprocess.run(cmd, check=True, capture_output=True, timeout=max(0.01, config["deadline_epoch"] - time.time()) if config.get("deadline_epoch") is not None else 3600)
            kd = directory / "keyframes"
            kd.mkdir()
            for index in sorted({round(i * (len(frames) - 1) / 3) for i in range(4)}):
                shutil.copy2(frames[index], kd / frames[index].name)
    except subprocess.CalledProcessError as exc:
        error = str(exc) + "\n" + (exc.stderr or b"").decode(errors="replace")
    except Exception as exc:
        error = str(exc)
    info = {"purpose": args.purpose, "status": "failed" if error else "ok", "error": error,
            "failure_stage": stages.get("active_stage") if error else None,
            "render_engine": stages.get("engine"), "render_timeout_s": render_budget,
            "render_samples": stages.get("render_samples"), "render_device": stages.get("render_device"),
            "exit_code": rc, "elapsed_s": time.time() - began, "directory": str(directory),
            "source": str(directory / "solution.py"), "frame_range": stages.get("frame_range"),
            "fps": config["fps"],
            "duration_s": ((stages["frame_range"][1] - stages["frame_range"][0] + 1) / config["fps"])
                          if stages.get("frame_range") else None,
            "artifacts": [str(p.relative_to(root)) for p in directory.iterdir() if p.is_file()]}
    if args.purpose == "render" and stages.get("script_dependencies"):
        info["submission_warning"] = ("This executed script reads other workspace Python files: "
            + ", ".join(stages["script_dependencies"])
            + ". Before FINAL, execute a complete standalone script containing the scene code and REPRESENTATION; "
              "solution.py must run without previous round files.")
    if not error and args.purpose != "query":
        info["artifact_sha256"]={}
        for name in ("executed.py","scene.blend","video.mp4" if args.purpose=="final" else "preview.mp4"):
            h=hashlib.sha256()
            with (directory/name).open('rb') as file:
                for block in iter(lambda:file.read(1024*1024),b''):
                    h.update(block)
            info["artifact_sha256"][name]=h.hexdigest()
    (directory / "execution.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))
    if args.purpose == "query":
        print((directory / "blender.log").read_text())
    raise SystemExit(124 if error and rc == 124 else (1 if error else 0))


if __name__ == "__main__":
    main()
