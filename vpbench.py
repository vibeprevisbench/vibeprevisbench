#!/usr/bin/env python3
"""VibePrevisBench: download, verify, prepare, collect, replay, score, orbit, judge and report."""
import argparse, csv, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def run(script, args):
    return subprocess.call([sys.executable, str(ROOT / script), *args])

def verify(data):
    checks = data / "SHA256SUMS"
    if not checks.is_file():
        raise SystemExit("SHA256SUMS missing: download the complete dataset")
    bad = []
    for line in checks.read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = data / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(name)
    print(json.dumps({"checked": len(checks.read_text().splitlines()), "failed": bad}, indent=2))
    return bool(bad)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "fetch-data", "fetch-media", "verify", "prepare", "collect", "replay", "score", "orbit", "judge", "report", "doctor"])
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    cmd = args.command
    scripts = {"score": "scorer/score.py", "judge": "judge/screen_check.py", "orbit": "orbit/batch.py"}
    if cmd in scripts:
        return run(scripts[cmd], rest)
    if cmd in ("prepare", "collect", "replay"):
        return run("harness/vpb_harness.py", [cmd, *rest])
    if cmd == "doctor":
        p = argparse.ArgumentParser(); p.add_argument("--blender", default="blender"); a = p.parse_args(rest)
        tools = {n: shutil.which(n) for n in ("ffmpeg", "ffprobe", "docker")}
        tools["blender"] = shutil.which(a.blender)
        print(json.dumps({"python": sys.version, "tools": tools}, indent=2))
        return 0 if all(tools.values()) else 1
    if cmd in ("download", "fetch-data", "fetch-media", "verify"):
        p = argparse.ArgumentParser()
        p.add_argument("--data", type=Path, default=ROOT / "data")
        p.add_argument("--repo", default=os.environ.get("VPB_HF_REPO", "AidanYuzhe/vibeprevisbench"))
        p.add_argument("--revision", default="main")
        p.add_argument("--data-only", action="store_true")
        a, extra = p.parse_known_args(rest)
        if cmd == "verify":
            if extra: p.error("unrecognized arguments: " + " ".join(extra))
            return verify(a.data)
        if cmd in ("download", "fetch-data"):
            if "{" in a.repo: p.error("set --repo ORGANIZATION/DATASET or VPB_HF_REPO")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=a.repo, repo_type="dataset", revision=a.revision, local_dir=a.data)
            if verify(a.data): return 1
        if a.data_only: return 0
        return run("tools/fetch_external.py", ["--manifest", str(a.data / "external.json"), "--out", str(a.data / "tasks"),
                                              "--cache", str(a.data / ".media_cache"), "--report", str(a.data / "fetch_report.json"), *extra])
    if cmd == "report":
        p = argparse.ArgumentParser();p.add_argument("--data", type=Path, default=ROOT / "data")
        p.add_argument("--scores", type=Path, required=True);p.add_argument("--screen", type=Path)
        p.add_argument("--out", type=Path, required=True);a=p.parse_args(rest)
        tasks=json.loads((a.data / "tasks/index.json").read_text())["tasks"]
        with (a.scores / "scores.csv").open() as f: rows={r["task_id"]:r for r in csv.DictReader(f)}
        result=[];pending=[]
        for t in tasks:
            i=t["task_id"];r=rows.get(i, {});ok=r.get("status")=="scored"
            screen=0.0
            if a.screen and ok:
                path=a.screen / (i+".json")
                if path.exists(): screen=json.loads(path.read_text()).get("screen") or 0.0
                else: pending.append(i)
            result.append({"task_id":i,"kind":t["kind"],"scene":float(r.get("rule") or 0) if ok else 0.0,
                           "screen":screen if a.screen else None})
        summary={"tasks":len(tasks),"scene":sum(r["scene"] for r in result)/len(tasks),
                 "screen":sum(r["screen"] for r in result)/len(tasks) if a.screen else None,
                 "screen_complete":not pending if a.screen else False,"pending_screen":pending,"results":result}
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(summary,indent=2)+"\n")
        print(json.dumps({k:v for k,v in summary.items() if k!="results"},indent=2));return bool(pending)
if __name__ == "__main__":
    sys.exit(main())
