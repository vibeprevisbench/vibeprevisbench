"""Delivery-protocol classification of launcher attempts (rules: docs/PROTOCOL_DELIVERY.md).

    python3 -I -B tools/protocol/classify.py BATCH_DIR [BATCH_DIR ...] [--overrides FILE] [--out FILE]

Reads a launcher batch (batch.json, status.json) and every attempt's run directory; never writes into them.
Each attempt gets one class with a reason and the evidence it rests on:
  ok     delivered (harness ok and final/video.mp4 present)
  model  the model's own failure; if it ends the job, the item scores 0
  infra  infrastructure / operator failure; never scored, the (model, item) is re-run
Each job (model, item) then gets delivered / model_fail / infra_incomplete / unfinished from its attempts.
The launcher's own label (status.json "outcome") is kept next to the protocol class; differences are flagged.
Stdlib only; safe to vendor (no imports from the rest of the scorer).
"""
import argparse
import json
import re
import sys
from pathlib import Path

PROTOCOL = "delivery-1.0"
HERE = Path(__file__).resolve().parent
DEFAULT_OVERRIDES = HERE / "overrides.json"
LAUNCHER_TO_CLASS = {"ok": "ok", "model_fail": "model", "infra_fail": "infra", "content_filter": "model"}
WORKER_LOST = "Background worker exited without completion status"

# Anchored infrastructure signatures, matched against the attempt's error text (result.json "error", last line
# first).  Kept narrow on purpose: a model traceback that merely mentions "timeout" must stay a model failure.
SANDBOX_SIGS = [
    (re.compile(r"^\s*[45]\d\d: b'"), "sandbox API error (e.g. 405: b'')"),
    (re.compile(r"Request failed: error sending request for url \(http://[^)\s]*sandbox"),
     "sandbox request failed"),
    (re.compile(r"workspace bundle failed"), "sandbox workspace bundle failed"),
    (re.compile(r"executor_process_lost|initialization_error|sandbox (?:is )?not (?:found|running)", re.I),
     "sandbox lost"),
]
GATEWAY_SIGS = [
    (re.compile(r"^model stream error:"), "model stream error"),
    (re.compile(r"^model response failed"), "provider marked the response failed"),
    (re.compile(r"upstream connect error|reset before headers|connection termination"), "gateway reset"),
    (re.compile(r"'server_error'|\"server_error\""), "provider server_error"),
]
MODEL_STOPS = {"model_protocol_error", "internal_step_limit", "no_action", "max_tokens"}


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def jlines(p):
    out = []
    try:
        for line in Path(p).read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def last_line(text):
    lines = [x for x in str(text or "").strip().splitlines() if x.strip()]
    return lines[-1].strip() if lines else ""


def match(sigs, text):
    for rx, label in sigs:
        if rx.search(text):
            return label
    return None


def worker_lost_signature(run):
    """The delivery 'editable_scene_open' query reopened final/scene.blend and printed its markers, then the
    sandbox worker died before reporting (exit 137).  Returns the evidence dict, or None."""
    for e in reversed(jlines(run / "delivery" / "events.jsonl")):
        args, res = e.get("arguments"), e.get("result")
        if e.get("kind") != "tool" or not isinstance(args, dict) or not isinstance(res, dict):
            continue
        if args.get("purpose") != "query" or "BA_DELIVERY_SCENE" not in str(args.get("code")):
            continue
        out, err = str(res.get("stdout") or ""), str(res.get("stderr") or "")
        inner = None
        try:
            inner = json.loads(out.splitlines()[0]).get("status")
        except (ValueError, IndexError, AttributeError):
            pass
        ok = (res.get("exit_code") not in (0, None) and WORKER_LOST in err and "BA_DELIVERY_SCENE {" in out
              and "BA_DONE" in out and "Read blend:" in out and inner == "ok")
        return {"event_id": e.get("event_id"), "exit_code": res.get("exit_code"), "inner_status": inner,
                "markers": ["Read blend", "BA_DELIVERY_SCENE", "BA_DONE"], "stderr": last_line(err)} if ok else None
    return None


def sandbox_tool_errors(run):
    n, first = 0, None
    for e in jlines(run / "trajectory" / "events.jsonl"):
        if e.get("kind") == "tool_error":
            label = match(SANDBOX_SIGS, str(e.get("error")))
            if label:
                n += 1
                first = first or {"event_id": e.get("event_id"), "t_s": round(e.get("event_elapsed_s") or 0, 1),
                                  "error": str(e.get("error"))[:160]}
    return n, first


def pins(run):
    hv, mj, cf = jload(run / "harness_version.json"), jload(run / "model.json"), jload(run / "config.json")
    return {"release_id": (hv or {}).get("release_id"), "source_sha256": (hv or {}).get("source_sha256"),
            "pvtools_sha256": (hv or {}).get("pvtools_sha256"), "blender": (hv or {}).get("blender"),
            "reasoning_effort": (mj or {}).get("reasoning_effort"), "max_rounds": (cf or {}).get("max_rounds")}


def usage(run):
    t = (jload(run / "usage.json") or {}).get("total") or {}
    return {k: t.get(k) for k in ("model_calls", "total_tokens", "output_tokens", "task_elapsed_s")}


def classify_attempt(att, override=None):
    """att: one status.json attempt record.  Returns the attempt row (class, reason, evidence, ...)."""
    run = Path(att["out"]).resolve()
    res = jload(run / "result.json")
    final_exec = jload(run / "final" / "execution.json") or {}
    val = jload(run / "delivery" / "validation.json") or {}
    failed_checks = [c.get("name") for c in val.get("checks", []) if c.get("passed") is False]
    rounds = []
    for p in sorted((run / "rounds").glob("*/execution.json")):
        x = jload(p) or {}
        rounds.append(x.get("status") if x.get("status") == "ok" else "failed:%s" % x.get("failure_stage"))
    err = str((res or {}).get("error") or "")
    ev = {"launcher_outcome": att.get("outcome"), "exit_code": att.get("exit_code"), "note": att.get("note"),
          "result_status": (res or {}).get("status"), "stop_reason": (res or {}).get("stop_reason"),
          "failure_category": (res or {}).get("failure_category"), "error": last_line(err)[:200],
          "failed_checks": failed_checks, "rounds": rounds, "final_source_round": (res or {}).get("final_source_round"),
          "has": {k: (run / "final" / k).is_file() for k in ("solution.py", "video.mp4", "scene.blend")}}

    def row(cls, reason, **extra):
        out = {"attempt": att.get("attempt"), "out": str(run), "class": cls, "reason": reason,
               "launcher_class": LAUNCHER_TO_CLASS.get(att.get("outcome")), "evidence": dict(ev, **extra),
               "pins": pins(run), "usage": usage(run), "started_at": att.get("started_at"),
               "ended_at": att.get("ended_at")}
        out["label_diff"] = out["launcher_class"] is not None and out["launcher_class"] != cls
        return out

    if att.get("outcome") is None:
        return row(None, "unfinished")
    if att.get("outcome") == "ok" and ev["has"]["video.mp4"]:
        return row("ok", "delivered")
    if override:
        return row(override["class"], override["reason"], override=override)
    if att.get("note") == "watchdog kill":
        return row("infra", "timeout_watchdog")
    if res is None or (att.get("exit_code") or 0) < 0:
        n, first = sandbox_tool_errors(run)
        return row("infra", "killed_no_result", sandbox_tool_errors=n, first_sandbox_error=first)
    if att.get("outcome") == "content_filter":
        return row("model", "content_filter")
    model_code = final_exec.get("status") == "failed" and final_exec.get("failure_stage") == "model_code"
    # last line first; the whole error text only when the final script did not fail in the model's own code
    for text in (last_line(err),) + (() if model_code else (err,)):
        label = match(SANDBOX_SIGS, text)
        if label:
            return row("infra", "sandbox_unavailable", signature=label)
        label = match(GATEWAY_SIGS, text)
        if label:
            return row("infra", "gateway_stream", signature=label)
    if failed_checks == ["editable_scene_open"] and ev["has"]["video.mp4"] and ev["has"]["scene.blend"] \
            and final_exec.get("status") == "ok":
        sig = worker_lost_signature(run)
        if sig:
            return row("infra", "worker_lost_after_check", worker_lost=sig, recoverable=True)
    # pvscore 5.0.12 (S23): the harness's own infrastructure verdict is tested before the model checks; a failed
    # workspace restore (M50_006 a1 of the GT-as-model batch) also fails delivery checks, which are its effect
    if ev["failure_category"] in ("environment_failure", "api_protocol_error") or \
            ev["stop_reason"] in ("interrupted", "launch_error", "initialization_error"):
        return row("infra", "infra_other", review=True)
    if model_code:
        return row("model", "final_code_error", final_error=last_line(final_exec.get("error"))[:200])
    if "standalone_script" in failed_checks:
        return row("model", "not_standalone")
    if failed_checks:
        return row("model", "delivery_incomplete")
    if ev["stop_reason"] in MODEL_STOPS:
        return row("model", "model_protocol")
    return row("model", "model_other", review=True)


def resolve_job(attempt_rows, launcher_status):
    """First ok or model attempt ends the job; attempts after it are flagged, never used."""
    final = None
    for a in attempt_rows:
        if final is not None:
            a["ignored_after_final"] = True
        elif a["class"] in ("ok", "model"):
            final = a
    if final is not None and final["class"] == "ok":
        return "delivered", final["attempt"]
    if final is not None:
        return "model_fail", final["attempt"]
    if launcher_status in ("queued", "running") or not attempt_rows or attempt_rows[-1]["class"] is None:
        return "unfinished", None
    return "infra_incomplete", None


def load_overrides(path):
    data = jload(path) if path and Path(path).is_file() else None
    out = {}
    for o in (data or {}).get("overrides", []):
        assert o.get("class") in ("model", "infra") and o.get("reason") and o.get("evidence"), o
        out[(o["batch_id"], o["model"], o["item"], int(o["attempt"]))] = o
    return out


def classify_batch(batch_dir, overrides=None):
    batch_dir = Path(batch_dir).resolve()
    meta = jload(batch_dir / "batch.json") or {}
    st = jload(batch_dir / "status.json")
    assert st is not None, "no status.json in %s" % batch_dir
    bid = meta.get("batch_id") or batch_dir.name
    cases = {c["id"]: c for c in meta.get("cases", [])}
    effort = {m["model"]: m.get("reasoning_effort") for m in meta.get("models", [])}
    jobs = []
    for j in st["jobs"]:
        rows = [classify_attempt(a, (overrides or {}).get((bid, j["model"], j["item"], int(a["attempt"]))))
                for a in j["attempts"]]
        status, used = resolve_job(rows, j.get("status"))
        case = cases.get(j["item"], {})
        jobs.append({"model": j["model"], "item": j["item"], "launcher_status": j.get("status"), "status": status,
                     "final_attempt": used, "attempts": rows, "requeued": j.get("requeued") or [],
                     "task_sha256": case.get("task_sha256"), "prompt_sha256": case.get("prompt_sha256"),
                     "dataset": Path(case["dir"]).parent.name if case.get("dir") else None,
                     "expected_effort": (meta.get("reasoning_effort_overrides") or {}).get(j["model"])
                     or effort.get(j["model"]) or meta.get("reasoning_effort")})
    rel = meta.get("release") or {}
    return {"protocol": PROTOCOL, "batch": str(batch_dir), "batch_id": bid, "created_at": meta.get("created_at"),
            "dry_run_model": meta.get("dry_run_model"), "release_id": rel.get("id"),
            "source_sha256": rel.get("source_sha256"), "reasoning_effort": meta.get("reasoning_effort"),
            "effort_overrides": meta.get("reasoning_effort_overrides") or {}, "max_rounds": meta.get("max_rounds"),
            "watchdog_s": meta.get("watchdog_s"), "items_file": meta.get("items_file"),
            "launcher_sha256": meta.get("launcher_sha256"), "jobs": jobs}


def table(c):
    lines = ["batch %s (%s)" % (c["batch_id"], c["batch"])]
    for j in c["jobs"]:
        for a in j["attempts"]:
            if a["class"] == "ok" and not a["label_diff"]:
                continue
            e = a["evidence"]
            what = e.get("signature") or e.get("final_error") or e.get("error") or e.get("note") or ""
            if e.get("override"):
                what = "override: " + e["override"]["evidence"]
            if a["reason"] == "killed_no_result":
                what = "exit %s, no result.json, %s sandbox tool_errors" % (e["exit_code"], e["sandbox_tool_errors"])
            if a["reason"] == "worker_lost_after_check":
                what = "editable_scene_open: exit %s after BA_DONE" % e["worker_lost"]["exit_code"]
            lines.append("  %-22s %s a%s launcher=%-12s -> %-5s %-24s %s%s" % (
                j["model"][:22], j["item"], a["attempt"], a["evidence"]["launcher_outcome"], a["class"], a["reason"],
                what[:90], "  [LABEL DIFF]" if a["label_diff"] else ""))
        if j["status"] != "delivered":
            lines.append("  %-22s %s job: launcher=%s protocol=%s" % (j["model"][:22], j["item"],
                                                                       j["launcher_status"], j["status"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("batches", nargs="+", type=Path)
    ap.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    ap.add_argument("--out", type=Path, help="write the full classification JSON here")
    a = ap.parse_args(argv)
    ov = load_overrides(a.overrides)
    res = [classify_batch(b, ov) for b in a.batches]
    for c in res:
        print(table(c))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
