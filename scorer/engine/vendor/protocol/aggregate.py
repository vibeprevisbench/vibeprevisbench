"""Per-model delivery report under the delivery protocol (rules: docs/PROTOCOL_DELIVERY.md).

    python3 -I -B tools/protocol/aggregate.py --batch BATCH_DIR [--batch ...] [--overrides FILE] \
        [--worker-lost recover|rerun] [--max-rerun-batches 2] [--out FILE] [--md FILE]

Classifies every attempt (classify.py), resolves each (dataset, model, item) across the batches in creation order --
the first delivered or model_fail result is final, infra results are superseded by a later batch, a re-run of a
final result is ignored and flagged -- and reports per model:
  delivery rate          delivered / resolved items
  quality on delivered   mean total over delivered items
  overall                mean total over resolved items, model failures as 0
  pending                infra-incomplete, unscored or unfinished items with the action they need
No total is final while anything is pending or a pin check fails; the numbers are then marked provisional.
Scores come from each batch's selfcheck/ outputs (selfcheck.json, jobs.json, report.json); nothing is rescored.
Datasets (the items' data directory) are reported separately and never mixed.  Reads only.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("protocol_classify", HERE / "classify.py")
classify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(classify)

# Policy P1-P9 of docs/PROTOCOL_DELIVERY.md section 9, approved by the user on 2026-10-18 (P1b changed to 8 h).
POLICY = {"worker_lost": "recover", "max_rerun_batches": 2, "content_filter_retries": 1, "watchdog": "infra",
          "watchdog_h_new_batches": 8, "approved": True, "approved_on": "2026-10-18"}


def hours(watchdog_s):
    return None if watchdog_s is None else round(watchdog_s / 3600, 2)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def selfcheck_index(batch_dir, models):
    """run dir -> {total, acquire, schema, gt_root, report}; from the batch's selfcheck outputs."""
    sc = Path(batch_dir) / "selfcheck"
    data = classify.jload(sc / "selfcheck.json") or {}
    jobs = classify.jload(sc / "jobs.json") or []
    gt_root = (data.get("summary") or {}).get("gt_root")
    rows = {}
    for r in data.get("rows", []):
        m = r.get("model") or (models[0] if len(models) == 1 else None)
        rows[(m, r["item"])] = r
    out = {}
    for j in jobs:
        run = Path(j["run"]).resolve()
        item = run.parent.name
        model = run.parents[2].name
        rep = classify.jload(Path(j["out"]) / "report.json") or {}
        r = rows.get((model, item)) or rows.get((None, item)) or {}
        # the item's report.json is the score of record; selfcheck.json is its batch summary and must agree
        out[str(run)] = {"total": rep.get("total") if r.get("acquire") == "ok" else None, "acquire": r.get("acquire"),
                         "row_total": r.get("total"),
                         "row_error": r.get("error"), "schema": rep.get("schema"), "gt_root": gt_root,
                         "report_total": rep.get("total"), "report": str(Path(j["out"]) / "report.json")}
    return out


def cmp_report_view(batches):
    """What tools/cmp_report.py would score: later batch wins; done+dumped -> selfcheck.json total,
    model_failed -> 0, else None.  Returns {key: (kind, score)}, kind in scored / zero / unscored."""
    cell = {}
    for c, sci in batches:
        for j in c["jobs"]:
            ok = [a for a in j["attempts"] if a["evidence"]["launcher_outcome"] == "ok"]
            s = sci.get(ok[-1]["out"]) if ok else None
            if j["launcher_status"] == "done" and s and s["acquire"] == "ok":
                v = ("scored", s["row_total"])
            else:
                v = ("zero", 0.0) if j["launcher_status"] == "model_failed" else ("unscored", None)
            cell[(j["dataset"], j["model"], j["item"])] = v
    return cell


def aggregate(batch_dirs, overrides=None, policy=None):
    loaded = []
    for b in batch_dirs:
        c = classify.classify_batch(b, overrides)
        models = sorted({j["model"] for j in c["jobs"]})
        loaded.append((c, selfcheck_index(b, models)))
    return resolve(loaded, policy)


def resolve(loaded, policy=None):
    """loaded: [(classify_batch() result, selfcheck_index() result)].  Pure function of its input."""
    policy = dict(POLICY, **(policy or {}))
    loaded = sorted(loaded, key=lambda x: (x[0]["created_at"] or "", x[0]["batch_id"]))

    cells, flags, applied = {}, [], []
    for c, sci in loaded:
        for j in c["jobs"]:
            key = (j["dataset"], j["model"], j["item"])
            cell = cells.setdefault(key, {"dataset": j["dataset"], "model": j["model"], "item": j["item"],
                                          "baseline": bool(c["dry_run_model"]), "history": [], "final": None,
                                          "task_sha256": j["task_sha256"]})
            for a in j["attempts"]:
                if a["evidence"].get("override"):
                    applied.append({"batch_id": c["batch_id"], "model": j["model"], "item": j["item"],
                                    "attempt": a["attempt"], "class": a["class"], "reason": a["reason"],
                                    "evidence": a["evidence"]["override"]["evidence"]})
                if a.get("ignored_after_final"):
                    flags.append({"flag": "attempt_after_final_ignored", "batch_id": c["batch_id"],
                                  "model": j["model"], "item": j["item"], "attempt": a["attempt"]})
            if j["task_sha256"] != cell["task_sha256"]:
                flags.append({"flag": "task_changed_between_batches", "batch_id": c["batch_id"], "model": j["model"],
                              "item": j["item"], "was": cell["task_sha256"], "now": j["task_sha256"]})
            n_infra = sum(1 for a in j["attempts"] if a["evidence"]["launcher_outcome"] == "infra_fail")
            n_cf = sum(1 for a in j["attempts"] if a["evidence"]["launcher_outcome"] == "content_filter")
            if n_infra > 3 or n_cf > 1 + policy["content_filter_retries"] or j["requeued"]:
                flags.append({"flag": "retry_cap_or_requeue", "batch_id": c["batch_id"], "model": j["model"],
                              "item": j["item"], "infra_attempts": n_infra, "requeued": j["requeued"]})
            h = {"batch_id": c["batch_id"], "status": j["status"], "launcher_status": j["launcher_status"],
                 "attempts": ["%s:%s" % (a["class"], a["reason"]) for a in j["attempts"]]}
            if cell["final"] is not None:
                h["ignored"] = "rerun_after_final"
                flags.append({"flag": "rerun_after_final_ignored", "batch_id": c["batch_id"], "model": j["model"],
                              "item": j["item"], "status": j["status"]})
                cell["history"].append(h)
                continue
            cell["history"].append(h)
            if j["status"] == "delivered":
                a = next(x for x in j["attempts"] if x["attempt"] == j["final_attempt"])
                s = sci.get(a["out"])
                cell["final"] = {"status": "delivered", "batch_id": c["batch_id"], "attempt": a["attempt"],
                                 "out": a["out"], "score": s["total"] if s else None,
                                 "scorer": s["schema"] if s else None, "gt_root": s["gt_root"] if s else None,
                                 "acquire": s["acquire"] if s else None}
                if s and s["acquire"] == "ok" and s["row_total"] != s["report_total"]:
                    flags.append({"flag": "selfcheck_report_mismatch", "batch_id": c["batch_id"], "model": j["model"],
                                  "item": j["item"], "selfcheck_json": s["row_total"], "report_json": s["report_total"],
                                  "report": s["report"]})
            elif j["status"] == "model_fail":
                a = next(x for x in j["attempts"] if x["attempt"] == j["final_attempt"])
                cell["final"] = {"status": "model_fail", "batch_id": c["batch_id"], "attempt": a["attempt"],
                                 "out": a["out"], "reason": a["reason"], "score": 0.0}
            else:
                cell["last_infra"] = {"batch_id": c["batch_id"], "status": j["status"],
                                      "reason": j["attempts"][-1]["reason"] if j["attempts"] else None,
                                      "out": j["attempts"][-1]["out"] if j["attempts"] else None,
                                      "recoverable": bool(j["attempts"] and
                                                          j["attempts"][-1]["evidence"].get("recoverable"))}

    for cell in cells.values():
        f = cell["final"]
        if f and f["status"] == "delivered" and f["score"] is None:
            # delivered but not dumped / scored: a dump failure is infra, never 0
            cell["pending"] = {"action": "rescore", "why": "acquire=%s" % f.get("acquire"), "out": f["out"]}
        elif f is None:
            li = cell.get("last_infra") or {}
            infra_batches = sum(1 for h in cell["history"] if h["status"] == "infra_incomplete")
            if li.get("status") == "unfinished":
                action = "wait"
            elif li.get("reason") == "worker_lost_after_check" and policy["worker_lost"] == "recover":
                action = "score_recovered"
            elif infra_batches > 1 + policy["max_rerun_batches"]:
                action = "infra_exhausted"
            else:
                action = "rerun"
            cell["pending"] = {"action": action, "why": li.get("reason"), "infra_batches": infra_batches,
                               "out": li.get("out")}

    cmpv = cmp_report_view(loaded)
    diffs = []
    for key, cell in sorted(cells.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        # decision-level comparison; numeric drift between selfcheck.json and report.json is its own flag
        if cell.get("pending"):
            mine = ("unscored", None)
        else:
            mine = ("zero" if cell["final"]["status"] == "model_fail" else "scored", cell["final"]["score"])
        theirs = cmpv.get(key, ("unscored", None))
        if mine[0] != theirs[0]:
            diffs.append({"model": cell["model"], "item": cell["item"], "dataset": cell["dataset"],
                          "cmp_report": theirs[1], "protocol": mine[1],
                          "why": cell["pending"]["action"] + ":" + str(cell["pending"]["why"])
                          if cell.get("pending") else cell["final"]["status"]})

    datasets = {}
    for ds in sorted({k[0] for k in cells}, key=str):
        dc = [c for k, c in cells.items() if k[0] == ds]
        models = sorted({c["model"] for c in dc})
        compared = [m for m in models if not any(c["baseline"] for c in dc if c["model"] == m)]
        exhausted = {c["item"] for c in dc if (c.get("pending") or {}).get("action") == "infra_exhausted"}
        items = sorted({c["item"] for c in dc})
        common = [it for it in items if it not in exhausted and all(
            any(c["model"] == m and c["item"] == it and resolved(c) for c in dc) for m in compared)]
        per = {}
        for m in models:
            mc = [c for c in dc if c["model"] == m and c["item"] not in exhausted]
            per[m] = summarize(mc)
            per[m]["common"] = summarize([c for c in mc if c["item"] in common])
            per[m]["baseline"] = m not in compared
        datasets[ds] = {"models": per, "items": items, "common_items": common, "excluded_infra_exhausted":
                        sorted(exhausted), "pins": pin_report(loaded, ds, cells)}
        mism = ["%(model)s/%(item)s selfcheck.json %(selfcheck_json)s vs report.json %(report_json)s" % f
                for f in flags if f["flag"] == "selfcheck_report_mismatch" and (ds, f["model"], f["item"]) in cells]
        datasets[ds]["pins"]["violations"] += ["scores in flux: " + x for x in mism]
        blocked = [m for m in per if per[m]["pending"]] or datasets[ds]["pins"]["violations"]
        datasets[ds]["final"] = not blocked
    return {"protocol": classify.PROTOCOL, "policy": policy,
            "batches": [{"batch_id": c["batch_id"], "batch": c["batch"], "created_at": c["created_at"],
                         "watchdog_h": hours(c["watchdog_s"])} for c, _ in loaded],
            "datasets": datasets, "overrides_applied": applied, "flags": flags, "cmp_report_diffs": diffs,
            "cells": list(cells.values())}


def resolved(cell):
    return cell["final"] is not None and not cell.get("pending")


def summarize(mc):
    done = [c for c in mc if resolved(c)]
    deliv = [c for c in done if c["final"]["status"] == "delivered"]
    fails = [c for c in done if c["final"]["status"] == "model_fail"]
    return {"planned": len(mc), "resolved": len(done), "delivered": len(deliv), "model_fail": len(fails),
            "delivery_rate": round(len(deliv) / len(done), 4) if done else None,
            "quality_on_delivered": mean([c["final"]["score"] for c in deliv]),
            "overall": mean([c["final"]["score"] for c in done]),
            "model_fail_items": sorted("%s(%s)" % (c["item"], c["final"]["reason"]) for c in fails),
            "pending": sorted("%s:%s(%s)" % (c["item"], c["pending"]["action"], c["pending"]["why"])
                              for c in mc if c.get("pending"))}


def pin_report(loaded, ds, cells):
    """One release, pvtools, Blender, effort per model, max_rounds, watchdog, scorer and GT root per dataset.
    The watchdog is checked per dataset: old 4 h batches keep their label; a 4 h + 8 h mix is provisional."""
    seen = {k: {} for k in ("release_id", "source_sha256", "pvtools_sha256", "blender", "max_rounds",
                            "launcher_sha256", "scorer", "gt_root")}
    effort, wd = {}, {}   # wd: watchdog_s -> batch ids; a dataset-level value (4 h before 10-18, 8 h after)
    violations = []

    def add(k, v, where):
        if v is not None:
            seen[k].setdefault(str(v), []).append(where)
    for c, _ in loaded:
        jobs = [j for j in c["jobs"] if j["dataset"] == ds]
        if not jobs:
            continue
        wd.setdefault(c["watchdog_s"], []).append(c["batch_id"])
        add("launcher_sha256", c["launcher_sha256"], c["batch_id"])
        add("release_id", c["release_id"], c["batch_id"])
        add("source_sha256", c["source_sha256"], c["batch_id"])
        for j in jobs:
            for a in j["attempts"]:
                p, w = a["pins"], "%s/%s/%s/a%s" % (c["batch_id"], j["model"], j["item"], a["attempt"])
                for k in ("release_id", "source_sha256", "pvtools_sha256", "blender", "max_rounds"):
                    add(k, p.get(k), w)
                if a["class"] == "ok" and (p.get("release_id") is None or p.get("reasoning_effort") is None):
                    violations.append("delivered attempt without harness_version.json / model.json: " + w)
                if p.get("reasoning_effort") is not None:
                    effort.setdefault(j["model"], {}).setdefault(p["reasoning_effort"], []).append(w)
                    if p["reasoning_effort"] != j["expected_effort"]:
                        violations.append("effort %s != batch %s at %s" % (p["reasoning_effort"],
                                                                          j["expected_effort"], w))
    for cell in cells.values():
        f = cell["final"]
        if cell["dataset"] == ds and f and f["status"] == "delivered":
            add("scorer", f.get("scorer"), cell["model"] + "/" + cell["item"])
            add("gt_root", f.get("gt_root"), cell["model"] + "/" + cell["item"])
    if len(wd) > 1:
        violations.append("watchdog mixed: %s" % "; ".join("%s h in %s" % (hours(w), ", ".join(b))
                                                            for w, b in sorted(wd.items(), key=lambda x: str(x[0]))))
    for k, vals in seen.items():
        if len(vals) > 1:
            violations.append("%s differs: %s" % (k, {v: len(w) for v, w in vals.items()}))
    for m, vals in effort.items():
        if len(vals) > 1:
            violations.append("effort differs for %s: %s" % (m, {v: len(w) for v, w in vals.items()}))
    if len({v for vals in effort.values() for v in vals}) > 1:
        violations.append("effort differs between models: %s" % {m: sorted(v) for m, v in effort.items()})
    return {"values": {k: sorted(v) for k, v in seen.items()}, "watchdog_h": sorted((hours(w) for w in wd), key=lambda h: (h is None, h or 0)),
            "effort": {m: sorted(v) for m, v in effort.items()}, "violations": violations}


def markdown(rep):
    out = ["# Delivery report (%s)" % rep["protocol"], "",
           "Policy: %s" % json.dumps(rep["policy"]), ""]
    for ds, d in rep["datasets"].items():
        wh = d["pins"]["watchdog_h"]
        out += ["## %s -- %s -- watchdog %s" % (ds, "FINAL" if d["final"] else
                                                "PROVISIONAL (pending items or pin violations)",
                                                "%s h" % wh[0] if len(wh) == 1 else "MIXED %s h" % " + ".join(map(str, wh))),
                "",
                "| model | resolved/planned | delivery rate | quality on delivered | overall (fail=0) | "
                "common: delivery / quality / overall | model failures | pending |",
                "|---|---|---|---|---|---|---|---|"]
        for m, s in d["models"].items():
            cm = s["common"]
            out.append("| %s%s | %d/%d | %s | %s | %s | %s / %s / %s (n=%d) | %s | %s |" % (
                m, " (baseline)" if s["baseline"] else "", s["resolved"], s["planned"], s["delivery_rate"],
                s["quality_on_delivered"], s["overall"], cm["delivery_rate"], cm["quality_on_delivered"],
                cm["overall"], cm["resolved"], ", ".join(s["model_fail_items"]) or "-",
                ", ".join(s["pending"]) or "-"))
        out += ["", "Pins: %s" % ("ok" if not d["pins"]["violations"] else "; ".join(d["pins"]["violations"])), ""]
    if rep["overrides_applied"]:
        out += ["Overrides applied:"] + ["- %(batch_id)s %(model)s %(item)s a%(attempt)s -> %(class)s/%(reason)s: "
                                         "%(evidence)s" % o for o in rep["overrides_applied"]] + [""]
    if rep["cmp_report_diffs"]:
        out += ["Differences from tools/cmp_report.py:"] + [
            "- %(dataset)s %(model)s %(item)s: cmp_report=%(cmp_report)s protocol=%(protocol)s (%(why)s)" % x
            for x in rep["cmp_report_diffs"]] + [""]
    if rep["flags"]:
        out += ["Flags:"] + ["- " + json.dumps(f) for f in rep["flags"]] + [""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", action="append", required=True, type=Path)
    ap.add_argument("--overrides", type=Path, default=classify.DEFAULT_OVERRIDES)
    ap.add_argument("--worker-lost", choices=("recover", "rerun"), default=POLICY["worker_lost"])
    ap.add_argument("--max-rerun-batches", type=int, default=POLICY["max_rerun_batches"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--md", type=Path)
    a = ap.parse_args(argv)
    rep = aggregate(a.batch, classify.load_overrides(a.overrides),
                    {"worker_lost": a.worker_lost, "max_rerun_batches": a.max_rerun_batches})
    md = markdown(rep)
    print(md)
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1) + "\n")
    if a.md:
        a.md.write_text(md + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
