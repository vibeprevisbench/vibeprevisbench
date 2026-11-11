"""Bounded infrastructure recovery after the answer is frozen."""
import json
from pathlib import Path
import re
import shutil
import shlex

from .execution import Workspace
from .validation import valid_png


class DeliveryFailure(RuntimeError):
    def __init__(self, message, category, workspace):
        super().__init__(message)
        self.category, self.workspace = category, workspace


def failure_category(root, error=None):
    info_path = Path(root) / "final/execution.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    if info.get("failure_stage") in ("model_code", "setup_render"):
        return "model_code_error"
    return "environment_failure"


def deliver(workspace, code, *, single=False, continue_from=None):
    recorder = workspace.recorder
    frozen = recorder.root / "delivery/submission.py"
    if not frozen.exists() or frozen.read_text() != code:
        raise DeliveryFailure("submitted code was not frozen", "delivery_incomplete", workspace)
    for attempt in range(1, 4):
        recorder.event("delivery_attempt", attempt=attempt, recovery_limit=2)
        try:
            if single and attempt == 1:
                outcome = workspace.tool("blender", {"code": code})
                script = ("import pathlib,shutil; root=pathlib.Path("+repr(workspace.root)+"); "
                          "source=root/'rounds/001'; target=root/'final'; "
                          "shutil.copytree(source,target,symlinks=True) if source.exists() else None; "
                          "preview=target/'preview.mp4'; preview.rename(target/'video.mp4') if preview.exists() else None")
                copied=workspace.raw("python3 -I -c "+shlex.quote(script))
                if copied['exit_code']:
                    raise RuntimeError('single submission delivery copy failed: '+copied['stderr'])
                workspace.sync()
            else:
                outcome = workspace.final(code, continue_from=continue_from) if continue_from else workspace.final(code)
            if outcome["exit_code"] == 0:
                return workspace, outcome
            info_path = recorder.root / "final/execution.json"
            info = json.loads(info_path.read_text()) if info_path.exists() else {}
            raise RuntimeError(info.get("error") or outcome.get("stderr") or "delivery execution failed")
        except Exception as exc:
            category = failure_category(recorder.root, exc)
            recorder.event("delivery_attempt_failed", attempt=attempt, category=category, error=str(exc))
            if category == "model_code_error" or attempt == 3:
                raise DeliveryFailure(str(exc), category, workspace) from exc
            try:
                workspace.deadline.remaining()  # Never extend a configured total budget.
            except TimeoutError:
                raise DeliveryFailure(str(exc), category, workspace) from exc
            try:
                workspace.sync()
            except Exception as checkpoint_error:
                recorder.event("delivery_checkpoint_error", error=str(checkpoint_error))

            saved = recorder.root / f"delivery/attempts/{attempt:03d}"
            saved.mkdir(parents=True, exist_ok=True)
            partial = recorder.root / "final"
            if partial.exists():
                partial.rename(saved / "final")
            old = workspace
            try:
                old.close()
            except Exception as cleanup_error:
                recorder.event("delivery_cleanup_error", error=str(cleanup_error))
            workspace = Workspace(recorder, old.deadline, old.backend, old.blender, old.ffmpeg,
                                  old.max_rounds, old.fps, old.resolution, old.final_resolution)
            continue_from = None
            try:
                workspace.start()
                archived = recorder.root / "workspace"
                if archived.exists():
                    workspace.restore_workspace(archived, {"bin", "lib", "runner.py", "launcher.py", "execution_config.json",
                                                "ledger.json", "ledger.lock", "final", "recovery_source"})
                workspace.rounds = old.rounds
                workspace.put("ledger.json", json.dumps({"rounds": old.rounds, "queries": 0, "last_source": None}).encode())
                candidate = saved / "final"
                job_path = candidate / "job.json"
                persisted = (candidate / "scene.blend").exists() and job_path.exists() and (candidate / "frames").is_dir()
                if persisted and not re.search(r"bpy\.app\.handlers|driver_namespace", code):
                    job = json.loads(job_path.read_text())
                    persisted = (candidate / "executed.py").read_text() == code and job["res"] == list(old.final_resolution) and job["fps"] == old.fps
                else:
                    persisted = False
                if persisted:
                    for frame in (candidate / "frames").glob("f_*.png"):
                        if not valid_png(frame, old.final_resolution):
                            frame.unlink()  # Interrupted writes are regenerated, never trusted.
                    workspace.restore_workspace(candidate, set(), prefix="recovery_source")
                    continue_from = "recovery_source"
                recorder.event("delivery_recovered", attempt=attempt+1, continued_frames=bool(continue_from),
                               sandbox_id=workspace.holder.sandbox_id if workspace.holder else None)
            except Exception as restore_error:
                raise DeliveryFailure(str(restore_error), "environment_failure", workspace) from restore_error
    raise AssertionError("unreachable")
