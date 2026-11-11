from __future__ import annotations

import hashlib
import json
import time
import signal
import threading
from functools import wraps
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Deadline:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.unlimited = seconds is None
        self.ends = float("inf") if self.unlimited else self.started + seconds

    def remaining(self):
        if self.unlimited:
            return 3600.0  # Per-operation safety timeout, never a task deadline.
        remaining = self.ends - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("task total time budget exhausted")
        return remaining

    def arm(self):
        """Hard wall-clock guard, including blocking HTTP/sandbox setup calls."""
        self.saved_alarm = None
        if not self.unlimited and threading.current_thread() is threading.main_thread():
            handler = signal.getsignal(signal.SIGALRM)
            timer = signal.getitimer(signal.ITIMER_REAL)
            self.saved_alarm = handler, timer
            def expired(signum, frame):
                raise TimeoutError("task total time budget exhausted")
            signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, self.remaining())

    def disarm(self):
        if getattr(self, "saved_alarm", None):
            signal.setitimer(signal.ITIMER_REAL, 0)
            handler, timer = self.saved_alarm
            signal.signal(signal.SIGALRM, handler)
            if timer[0]:
                signal.setitimer(signal.ITIMER_REAL, max(.001, timer[0] - (time.monotonic() - self.started)), timer[1])
            self.saved_alarm = None


def synchronized(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return locked


class Recorder:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "trajectory").mkdir()
        self.started = time.monotonic()
        self.calls = []
        self._lock = threading.RLock()
        self.phase = "setup"
        self.events = 0
        self.dialogue = None
        (self.root / "trajectory/dialogue.jsonl").touch()

    @synchronized
    def begin_model_call(self, number, group):
        if self.phase in ("delivery", "cleanup"):
            raise RuntimeError("model calls are forbidden after submission")
        self.phase = "agent"
        self.flush_dialogue()
        self.dialogue = {"schema_version": "blender-agent-dialogue-1", "call": number,
                         "group": group, "request": None, "response": None,
                         "assistant": None, "attempts": [], "events": [],
                         "model_visible_feedback": []}
        self.save_pending_dialogue()

    @synchronized
    def save_pending_dialogue(self):
        if self.dialogue is not None:
            target = self.root / "trajectory/dialogue_pending.json"
            temporary = target.with_suffix(".tmp")
            write_json(temporary, self.dialogue)
            temporary.replace(target)

    @synchronized
    def record_model_reply(self, folder, stat, assistant):
        if self.dialogue is None:
            return
        def read(path):
            return json.loads(path.read_text()) if path.exists() else None
        self.dialogue.update(request=read(folder / "request.json"),
                             response=read(folder / "response.json"),
                             assistant=assistant, usage=stat["usage"],
                             elapsed_s=stat["elapsed_s"], attempt_count=stat["attempts"],
                             attempts=[{"path": str(p.relative_to(self.root)), "data": read(p)}
                                       for p in sorted(folder.glob("attempt_*.json"))])
        stream = folder / "stream_events.jsonl"
        self.dialogue["stream_events_ref"] = str(stream.relative_to(self.root)) if stream.exists() else None
        self.save_pending_dialogue()

    @synchronized
    def dialogue_feedback(self, message):
        if self.phase in ("delivery", "cleanup"):
            raise RuntimeError("delivery results must not enter model feedback")
        if self.dialogue is not None:
            self.dialogue["model_visible_feedback"].append(message)
            self.save_pending_dialogue()

    @synchronized
    def flush_dialogue(self):
        if self.dialogue is not None:
            with (self.root / "trajectory/dialogue.jsonl").open("a") as f:
                f.write(json.dumps(self.dialogue, ensure_ascii=False) + "\n")
            self.dialogue = None
            (self.root / "trajectory/dialogue_pending.json").unlink(missing_ok=True)

    @synchronized
    def finish_dialogue(self):
        self.flush_dialogue()
        # Round grouping may be corrected by the executor at task completion.
        groups = {c["call"]: c["group"] for c in self.calls}
        path = self.root / "trajectory/dialogue.jsonl"
        temporary = path.with_suffix(".tmp")
        with path.open() as source, temporary.open("w") as target:
            for line in source:
                row = json.loads(line)
                row["group"] = groups.get(row["call"], row["group"])
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)

    @synchronized
    def event(self, kind, **data):
        self.events += 1
        value = dict(event_id=self.events, event_elapsed_s=time.monotonic() - self.started,
                     kind=kind, phase=self.phase, **data)
        with (self.root / "trajectory/events.jsonl").open("a") as f:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")
        if self.phase in ("delivery", "cleanup"):
            path = self.root / "delivery/events.jsonl"
            path.parent.mkdir(exist_ok=True)
            with path.open("a") as output:
                output.write(json.dumps(value, ensure_ascii=False) + "\n")
        if self.dialogue is not None and self.phase == "agent":
            self.dialogue["events"].append(value)
            self.save_pending_dialogue()
        return value

    @synchronized
    def begin_delivery(self, reason, code=None):
        already_started = self.phase == "delivery"
        self.flush_dialogue()
        self.phase = "delivery"
        (self.root / "delivery").mkdir(exist_ok=True)
        if code is not None:
            source = self.root / "delivery/submission.py"
            if source.exists() and source.read_text() != code:
                raise RuntimeError("submitted code is immutable")
            source.write_text(code)
            write_json(self.root / "delivery/submission.json", {"reason":reason,"sha256":digest(source),
                       "source":"delivery/submission.py", "model_call":len(self.calls)})
        self.event("submission_frozen" if already_started else "delivery_started", reason=reason)

    @synchronized
    def begin_cleanup(self):
        self.flush_dialogue()
        self.phase = "cleanup"



FIELDS = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "total_tokens")


def aggregate(calls):
    out = {"model_calls": len(calls), "attempts": sum(c.get("attempts", 0) for c in calls),
           "retries": sum(max(0, c.get("attempts", 0) - 1) for c in calls),
           "elapsed_s": sum(c.get("elapsed_s", 0) for c in calls), "coverage": {}}
    for field in FIELDS:
        values = [(c.get("usage") or {}).get(field) for c in calls]
        known = [v for v in values if v is not None]
        out[field] = sum(known) if known else None
        out["coverage"][field] = {"known_calls": len(known), "all_calls": len(calls)}
    paired = [c["usage"] for c in calls if (c.get("usage") or {}).get("cached_tokens") is not None
              and c["usage"].get("input_tokens") is not None]
    denominator = sum(u["input_tokens"] for u in paired)
    out["cache_hit_rate"] = sum(u["cached_tokens"] for u in paired) / denominator if denominator else None
    out["cache_hit_rate_calls"] = len(paired)
    out["unknown_usage_attempts"] = sum(c.get("attempts", 0) - (1 if c.get("usage") else 0) for c in calls)
    return out


def finish_reports(recorder, result):
    calls = recorder.calls
    groups = sorted({c["group"] for c in calls}, key=lambda s: (s == "closing", s))
    usage = {"calls": calls, "rounds": {g: aggregate([c for c in calls if c["group"] == g]) for g in groups},
             "total": aggregate(calls),
             "notes": ["Cached tokens are a subset of input tokens; reasoning tokens may be a subset of output tokens.",
                       "Null means unreported. Totals sum known usage only; see field coverage.",
                       "Failed attempts without usage have unknown token cost."]}
    usage["executions"] = [json.loads(p.read_text()) for p in sorted((recorder.root / "rounds").glob("*/execution.json"))]
    usage["total"]["scene_execution_s"] = sum(x.get("elapsed_s", 0) for x in usage["executions"])
    usage["total"]["task_elapsed_s"] = result["elapsed_s"]
    write_json(recorder.root / "usage.json", usage)
    if (recorder.root / "resume.json").exists():
        prior = Path(json.loads((recorder.root / "resume.json").read_text())["from"])
        prior_usage = prior / "usage_cumulative.json"
        previous = json.loads((prior_usage if prior_usage.exists() else prior/"usage.json").read_text())
        combined = [dict(c, run=c.get("run",str(prior))) for c in previous["calls"]]
        combined += [dict(c,run=str(recorder.root)) for c in calls]
        write_json(recorder.root / "usage_cumulative.json", {"calls":combined,"total":aggregate(combined),
                   "rounds":{g:aggregate([c for c in combined if c["group"]==g]) for g in sorted({c["group"] for c in combined})},
                   "source_runs":sorted({c["run"] for c in combined})})
    write_json(recorder.root / "result.json", result)
    lines = ["# Blender Agent run report", "", f"Mode: {result['mode']}; status: {result['status']}; stop reason: {result['stop_reason']}",
             f"Rounds: {result['rounds']}; elapsed: {result['elapsed_s']:.2f} s", "",
             "| Phase | Calls | Input tokens | Cached tokens | Output tokens | Reasoning tokens | Total tokens | Cache hit rate | Model time (s) | Retries |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group, stats in list(usage["rounds"].items()) + [("Total (known usage)", usage["total"])]:
        values = [stats.get(k) for k in ("model_calls", *FIELDS)]
        rate = stats["cache_hit_rate"]
        lines.append("| " + group + " | " + " | ".join("unknown" if v is None else str(v) for v in values)
                     + " | " + (f"{rate:.2%}" if rate is not None else "unknown") + f" | {stats['elapsed_s']:.2f} | {stats['retries']} |")
    lines += ["", "| Round | Status | Execution and encoding (s) |", "|---|---|---:|"]
    for index, execution in enumerate(usage["executions"], 1):
        lines.append(f"| {index} | {execution['status']} | {execution.get('elapsed_s', 0):.2f} |")
    lines += ["", "The cache hit rate uses only calls that report both input and cached tokens. Field coverage: [usage.json](usage.json).",
              "Failed attempts may have no usage; totals cover known tokens only; internal reasoning the API does not return cannot be saved.", "",
              "Full requests, raw responses and tool results: [trajectory/events.jsonl](trajectory/events.jsonl).", "", "Final artifacts:", ""]
    for name, item in result.get("artifacts", {}).items():
        lines.append(f"- [{name}]({item['path']})" if item.get("path") else f"- {name}: missing; {item['reason']}")
    if (recorder.root / "representation.md").exists():
        lines += ["", "Representation and simplifications stated by the model (not semantically verified):", "", (recorder.root / "representation.md").read_text()]
    if result.get("error"):
        lines += ["", "Error:", "```text", result["error"], "```"]
    (recorder.root / "report.md").write_text("\n".join(lines) + "\n")
    files = [{"path": str(p.relative_to(recorder.root)), "sha256": digest(p), "bytes": p.stat().st_size}
             for p in sorted(recorder.root.rglob("*")) if p.is_file() and p.name != "manifest.json"]
    write_json(recorder.root / "manifest.json", {"schema_version": "blender-agent-1", "status": result["status"],
               "mode": result["mode"], "files": files,
               "implementation_sha256": {str(p.relative_to(Path(__file__).parent)): digest(p)
                                         for p in sorted(Path(__file__).parent.glob("*.py"))}})
