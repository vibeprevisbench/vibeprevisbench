#!/usr/bin/env python3
"""Rule-based Scene Check (pvscore 5.4): score the 4D scenes of one run. CPU only, no API.

    python3 scorer/score.py --run runs/<system> --blender <blender-5.2.2>/blender --out results/<system>/scene
    python3 scorer/score.py --dumps <dir> --out results/<system>/scene        # <dir>/<task_id>/dump.{json,npz}

--run holds <task_id>/scene.blend, the scene the delivered program saved (vpbench.py replay writes it). A task without
a scene counts 0, as in the paper; --tasks restricts the run to a list and then scores only those. Steps: (1) open each
scene in a fresh Blender 5.2.2 and export the dump the scorer reads (Blender's bundled Python, no add-ons), (2) run the
scoring engine (scorer/engine, Python 3.12 + numpy + scipy) on every dump against the dataset's hidden ground truth facts,
(3) write scores.csv (one row per task: rule score 0-100 and the nine dimensions, 0-1) and summary.json (the
leaderboard columns). A finished dump is reused, so a stopped run can be started again. The engine's own output
(per-check reports, read by the Screen Check) is in --out/engine.

Dimensions: A1 entities, A2 layout, A3 spaces, P1 picture structure, B1 routes, B2 events, C1 camera, C2 cuts,
D1 physical validity. A task's rule score is 100 x the mean of the dimensions it has (a dimension the task does not
test is empty, not 0); the leaderboard averages the rule score over all 242 tasks.
"""
import argparse, csv, json, os, statistics as st, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE / "engine"
DIMS = ["A1", "A2", "A3", "P1", "B1", "B2", "C1", "C2", "D1"]
NAME = "submission"                     # the model name the engine sees


def dump(job):
    task, blend, out, blender, threads, timeout = job
    if (out / "dump.json").is_file() and (out / "dump.npz").is_file():
        return task, "reused", 0.0
    out.mkdir(parents=True, exist_ok=True)
    cmd = [blender, "-b", "--factory-startup", "-t", str(threads), "--python-exit-code", "1", str(blend), "--python",
           str(ENGINE / "vendor/v5_0_12/pvscore5/blender_dump.py"), "--", "--out", str(out), "--pvtools",
           str(ENGINE / "common/pvtools.py")]
    t0 = time.time()
    try:
        with open(out.parent / (task + ".dump.log"), "wb") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return task, "dump timed out after %d s" % timeout, time.time() - t0
    ok = rc == 0 and (out / "dump.json").is_file() and (out / "dump.npz").is_file()
    return task, "ok" if ok else "dump failed (exit %d), see %s.dump.log" % (rc, task), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", type=Path, help="<task_id>/scene.blend")
    src.add_argument("--dumps", type=Path, help="<task_id>/dump.json + dump.npz (skips Blender)")
    ap.add_argument("--data", type=Path, default=HERE.parent / "data", help="the dataset folder (default: data/)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--blender", help="Blender 5.2.2 executable (needed with --run)")
    ap.add_argument("--tasks", help="comma-separated task ids; default all 242")
    ap.add_argument("--workers", type=int, default=8, help="scoring processes")
    ap.add_argument("--dump-workers", type=int, default=2, help="Blender processes at once (each uses 4 threads)")
    ap.add_argument("--dump-timeout", type=int, default=21600)
    a = ap.parse_args()
    # The engine runs with python -I. Without scipy it would silently fall back to greedy matching and a different convex
    # hull, which changes scores; the paper's scores were computed with scipy, so refuse to score without it.
    chk = subprocess.run([sys.executable, "-I", "-c", "import numpy, scipy.optimize, scipy.spatial"], capture_output=True,
                         text=True)
    if chk.returncode:
        sys.exit("the scoring engine needs numpy and scipy for %s (pip install -r requirements.txt):\n%s"
                 % (sys.executable, chk.stderr.strip().splitlines()[-1] if chk.stderr.strip() else "import failed"))
    data = a.data.resolve()
    index = json.loads((data / "tasks/index.json").read_text())["tasks"]
    want = set(a.tasks.split(",")) if a.tasks else None
    tasks = [t for t in index if want is None or t["task_id"] in want]
    if want and len(tasks) != len(want):
        sys.exit("unknown tasks: %s" % sorted(want - {t["task_id"] for t in tasks}))
    out = a.out.resolve()
    droot = out / "dumps"
    missing, failed = [], {}
    if a.run is not None:
        if not a.blender:
            sys.exit("--run needs --blender (Blender 5.2.2)")
        jobs = []
        for t in tasks:
            blend = a.run / t["task_id"] / "scene.blend"
            if blend.is_file():
                jobs.append((t["task_id"], blend.resolve(), droot / NAME / t["task_id"], a.blender, 4, a.dump_timeout))
            else:
                missing.append(t["task_id"])
        with ThreadPoolExecutor(a.dump_workers) as ex:
            for task, status, secs in ex.map(dump, jobs):
                print("dump", task, status, "%.0fs" % secs, flush=True)
                if status not in ("ok", "reused"):
                    failed[task] = status
    else:
        (droot / NAME).mkdir(parents=True, exist_ok=True)
        for t in tasks:
            d = (a.dumps / t["task_id"]).resolve()
            if (d / "dump.json").is_file() and (d / "dump.npz").is_file():
                ln = droot / NAME / t["task_id"]
                if ln.is_symlink() or ln.exists():
                    ln.unlink()
                ln.symlink_to(d, target_is_directory=True)
            else:
                missing.append(t["task_id"])
    todo = [t["task_id"] for t in tasks if t["task_id"] not in missing and t["task_id"] not in failed]
    scored = {}
    if todo:
        env = dict(os.environ, VPB_DATA=str(data), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run([sys.executable, "-I", "-B", str(ENGINE / "score51.py"), "--dumps", str(droot), "--out",
                        str(out / "engine"), "--workers", str(a.workers), "--items", ",".join(todo)], env=env, check=False)
        if result.returncode:
            sys.exit("scoring engine failed; inspect its output (exit %d)" % result.returncode)
        sf = out / "engine/scores.json"
        if sf.is_file():
            scored = {c["item"]: c for c in json.loads(sf.read_text())["cells"]}
    rows = []
    for t in tasks:
        c = scored.get(t["task_id"])
        ok = c is not None and c.get("status") == "scored"
        dims = (c or {}).get("dims") or {}
        vals = [v for v in dims.values() if v is not None]
        why = ("" if ok else ("no scene.blend" if a.run is not None else "no scene dump") if t["task_id"] in missing else failed.get(t["task_id"])
               or (c or {}).get("why") or (c or {}).get("action") or "not scored")
        rows.append({"task_id": t["task_id"], "kind": t["kind"], "status": "scored" if ok else "failed",
                     "rule": round(100 * st.mean(vals), 3) if ok and vals else 0.0, **{d: dims.get(d) for d in DIMS},
                     "note": why})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "kind", "status", "rule"] + DIMS + ["note"])
        w.writeheader()
        w.writerows(rows)

    def mean(kinds):
        r = [x["rule"] for x in rows if x["kind"] in kinds]
        return round(st.mean(r), 2) if r else None
    summ = {"scorer": "pvscore 5.4", "tasks": len(rows), "scored": sum(r["status"] == "scored" for r in rows),
            "scene": mean(("text", "image", "continuation", "video")), "text": mean(("text",)), "image": mean(("image",)),
            "continuation": mean(("continuation",)), "video": mean(("video",)),
            "failed": {r["task_id"]: r["note"] for r in rows if r["status"] != "scored"}}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summ.items() if k != "failed"}), "failed", len(summ["failed"]))
    engineering_errors = [c["item"] for c in scored.values() if c.get("status") == "pending"]
    if engineering_errors:
        sys.exit("incomplete scoring (engineering errors): " + ",".join(engineering_errors))


if __name__ == "__main__":
    main()
