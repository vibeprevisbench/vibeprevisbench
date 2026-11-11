"""pvscore 5.1 r2: model dumps and pvscore 5.0.x reports from launcher batch directories (no hard-coded run paths).

    attempts(batch_dirs)  every acquired attempt of the batches: <batch>/selfcheck/jobs.json maps the attempt's run
                          directory (runs_*/<model>/agent/<item>/<batch>_aN) to its selfcheck directory (dump.json,
                          dump.npz, acquire.json, report.json); model = the run's model directory name (the launcher
                          model name, e.g. gpt-6-astra-official; the fixture model is dry-run-fake)
    latest(batch_dirs)    {(model, item): attempt} with a dump: the attempt of the latest batch (batch.json created_at,
                          then batch_id) wins; the attempts it replaces are listed under 'replaces'

The per-layer command lines (model_view.py, combine.py, the text layer) use latest(); the scoring driver
(score51.py) selects attempts by the delivery protocol instead and only uses attempts() to find their dumps.
Reads only.
"""
import json
from pathlib import Path


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def batch_meta(b):
    m = jload(Path(b) / "batch.json") or {}
    return {"batch": str(Path(b).resolve()), "batch_id": m.get("batch_id") or Path(b).name,
            "created_at": m.get("created_at") or "", "dry_run_model": m.get("dry_run_model"),
            "models": [x.get("model") for x in m.get("models") or []]}


def attempts(batch_dirs):
    out = []
    for b in batch_dirs:
        meta = batch_meta(b)
        for j in jload(Path(b) / "selfcheck" / "jobs.json") or []:
            run, sc = Path(j["run"]), Path(j["out"])
            rep = sc / "report.json"
            out.append({"model": run.parents[2].name, "item": run.parent.name, "run": str(run), "out": str(sc),
                        "batch_id": meta["batch_id"], "created_at": meta["created_at"], "batch": meta["batch"],
                        "baseline": bool(meta["dry_run_model"]) and run.parents[2].name == meta["dry_run_model"],
                        "dump": (sc / "dump.json").is_file(), "report": str(rep) if rep.is_file() else None})
    return out


def latest(batch_dirs):
    best = {}
    for a in sorted(attempts(batch_dirs), key=lambda a: (a["created_at"], a["batch_id"])):
        if not a["dump"]:
            continue
        k = (a["model"], a["item"])
        if k in best:
            a = dict(a, replaces=best[k].get("replaces", []) + [best[k]["out"]])
        best[k] = a
    return best


def baseline_models(batch_dirs):
    return sorted({a["model"] for a in attempts(batch_dirs) if a["baseline"]})
