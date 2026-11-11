from __future__ import annotations

import json
import ast
from pathlib import Path, PurePosixPath
import re
import shutil
import time

from .common import Deadline, Recorder, digest, finish_reports, write_json
from .execution import Workspace
from .model import ModelProtocolError
from .delivery import DeliveryFailure, deliver
from .validation import representation_text, validate_delivery
from .tools import INITIAL_IMAGE_CAP, definitions, observations

BASE_PROMPT = """Create an executable, editable Blender white-model video from text, images, or videos with a complete bpy script built from scratch. Preserve key subjects/identities, spatial relationships, trajectories, interactions, event order/timing, outcomes, and camera information.
Explicit task requirements for representation, actions/contacts, materials, cinematography, and submission take precedence over the defaults below, within the host's tool, budget, and execution constraints. Apply defaults only where the task is unspecified; simplification must not remove required information or impose unnecessary pose detail.
Implement explicit text requirements without rewriting the story. Preserve visible image structure/layout; add actions only as instructed, otherwise use necessary presentation. Extract observable video events and cinematography. Distinguish facts, requests, assumptions, and uncertainty; sampled frames are not full video viewing.
Match motion and cinematography to user intent, scene scale/context, mood/energy, and narrative focus, honoring instructions and references. Natural does not mean uniformly slow or smooth: vary speed, force, and rhythm purposefully, avoiding mechanical loops and universal easing. Fill presentation gaps without inventing events.
Use the fewest body parts sufficient to express the task, with simple volumes by default. When limbs are necessary, convey action intent and key contacts without overconstraining downstream generation with rigid joint poses; leave nonessential joint angles, detailed poses, and limb motion open to natural completion. Preserve required motion and interactions through expressive orientation, leaning, compression, and weight shifts. Use gray/white materials with limited identity/orientation/state colors; set material.diffuse_color for Workbench colors; text or flat media cannot replace necessary 3D geometry/animation.
Respect gravity, support/contact, balance, inertia, and collisions at scene scale. Include applicable preparation, takeoff, airborne motion, landing absorption, and recovery; avoid floating, contact sliding, interpenetration, teleportation, unexplained velocity jumps, and rigid upright/mechanical motion. Choose physically constrained procedural animation, analytic trajectories, or simulation; prefer simulation for behavior driven by collisions, deformation, or multi-object interactions. Easing cannot replace gravity-driven flight. Identify explicitly requested stylized/impossible exceptions.
Maintain identity/state and action continuity. Camera movement, holds, reframing, and cuts must serve viewing intent and keep key actions visible, rather than defaulting to constant motion, uniform orbits, or rigid tracking. Preserve purposeful rapid moves/shake while avoiding accidental jitter/collisions; keep within-shot motion coherent and hard cuts explicit with cross-shot state continuity. Paths, targets, and animation must remain editable; spline cameras and fixed shot structures are optional.
Default to BLENDER_WORKBENCH; choose CYCLES when lighting, reflections, or transparency are essential, briefly stating why in REPRESENTATION. Use economical samples; on render timeout, reduce cost or switch engines within the remaining budget.
Use meters, Z up, scene.camera, frame_start=1, and frame_end. Map t to 1+round(fps*t); render 1 through N, use the last available frame for the end pose, and use start-inclusive/end-exclusive time ranges. Follow the render configuration; do not quit Blender.
Deliver a complete runnable solution.py, an editable scene.blend, all final frames, and an H.264 MP4 (yuv420p) at the task's frame rate and aspect ratio, with no audio or overlays. The harness saves the script and scene, renders every frame, and encodes the video through its managed execution interface; do not run a separate nested full-animation render inside the scene script.
Define a top-level literal string REPRESENTATION in each complete script, saved automatically as representation.md. Briefly cover three items: key object/collection-to-subject mappings; motion/timing/interaction/camera cues to follow downstream; simplifications, details open to natural completion, and material assumptions/uncertainty. No exhaustive object inventory, repeated story, or execution/verification report.
Do not use r8lib, ready-made character/action/scene assets, or AI video-generation services. Access only the task workspace, never scoring materials or reference answers."""
# v5 English conventions (scoring note, primitives, people proxies, pvtools tagging, cuts), appended verbatim.
BASE_PROMPT += "\n" + Path(__file__).with_name("prompt_conventions.txt").read_text().rstrip("\n")
AGENT_PROMPT = """
You may begin with a brief interpretation and plan, or start working with tools directly; scale any planning to task complexity.
Use only bash, python, blender, and ffmpeg as documented, with relative artifact paths and the managed Blender launcher. Never call the underlying binary, bypass accounting, or modify the executor, budgets, ledger.json, runner.py, launcher.py, or bin directory.
Execution returns text status; request visuals via media. Videos return timestamped frames, not native video. Choose code/logs, scene queries, observation times/counts as needed; no fixed per-round viewing quota. Observe and revise against the requirements above using temporal samples or animation queries; isolated frames cannot establish natural motion.
Submit the last executed version in plain text starting with FINAL:, reporting actual checks and unresolved limitations. Never submit unexecuted code or claim unsupported verification.
"""


def input_file_records(input_files):
    """Task-provided files (e.g. input/entities.json): any relative path under input/, never managed media paths."""
    records = []
    for name, source in dict(input_files or {}).items():
        relative = PurePosixPath(name) if isinstance(name, str) else PurePosixPath("/")
        parts = relative.parts
        if (relative.is_absolute() or len(parts) < 2 or parts[0] != "input" or ".." in parts or "\\" in name
                or parts[1] in ("media", "frames") or str(relative) in ("input/task.json", "input/media_manifest.json")):
            raise ValueError(f"input file path must be a relative path under input/ outside managed media: {name!r}")
        if not Path(source).is_file():
            raise ValueError(f"input file not found: {source}")
        records.append({"path": str(relative), "source": Path(source), "source_name": Path(source).name,
                        "sha256": digest(source), "bytes": Path(source).stat().st_size})
    if len({r["path"] for r in records}) != len(records):
        raise ValueError("duplicate input file path")
    return records


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    if blocks:
        return blocks[-1].strip() + "\n"
    if text and (text.lstrip().startswith("import bpy") or text.lstrip().startswith("from bpy")):
        return text.strip() + "\n"
    raise ValueError("no complete Python code in model answer")




def run_task(prompt, mode, model, output, max_rounds=8, timeout_s=3600,
             backend="docker", blender="blender", ffmpeg="ffmpeg", fps=24, resolution=(640, 360),
             media=(), final_resolution=None, resume_from=None, expected_frame_count=None, input_files=None):
    inputs = input_file_records(input_files)
    rec = Recorder(output)
    deadline = Deadline(timeout_s)
    delivery_reserve_s = min(timeout_s * 0.2, max(5, timeout_s * 0.15)) if mode == "agent" and timeout_s is not None else 0
    work_deadline = Deadline(None if timeout_s is None else timeout_s-delivery_reserve_s)
    work_deadline.ends = deadline.ends-delivery_reserve_s
    final_resolution = tuple(final_resolution or resolution)
    workspace = Workspace(rec, work_deadline if mode == "agent" else deadline, backend, blender, ffmpeg,
                          1 if mode == "end_to_end" else max_rounds, fps,
                          final_resolution if mode == "end_to_end" else resolution, final_resolution)
    result = {"mode": mode, "status": "failed", "stop_reason": "initialization_error", "rounds": 0}
    system = BASE_PROMPT
    system += f"\nBudget: at most {1 if mode == 'end_to_end' else max_rounds} production executions; failures also count as rounds. Total time limit: {str(timeout_s) + chr(32) + 'seconds' if timeout_s is not None else 'none; individual operations still have safety timeouts'}, including initialization, observation, and final rendering. Maximum tokens per response: {model.max_tokens}."
    system += AGENT_PROMPT if mode == "agent" else "\nOutput the complete code in a single Python code block in one response. There is no execution feedback or opportunity to revise."
    if mode == "agent" and timeout_s is not None:
        system += "\nThe harness reserves part of the total time for delivery; when the agent work budget expires it may submit the last successful executed version automatically."
    write_json(rec.root / "input/task.json", {"prompt": prompt, "mode": mode, "input_files":
               [{k: v for k, v in item.items() if k != "source"} for item in inputs]})
    for item in inputs:
        (rec.root / item["path"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["source"], rec.root / item["path"])
    write_json(rec.root / "config.json", {"max_rounds": 1 if mode == "end_to_end" else max_rounds,
               "timeout_s": timeout_s, "max_tokens": model.max_tokens, "backend": backend,
               "fps": fps, "resolution": resolution, "final_resolution": final_resolution,
               "expected_frame_count": expected_frame_count, "delivery_recovery_limit": 2,
               "render_engine": "MODEL_CHOICE", "blender_render_timeout_s": None, "delivery_reserve_s": delivery_reserve_s,
               "protocol_recovery_limit": 2 if mode == "agent" else 0,
               "color_policy": "gray_with_semantic_material_colors", "tool_names": ["bash", "python", "blender", "ffmpeg"],
               "video_delivery": "fps1_timestamped_frames", "initial_image_cap": INITIAL_IMAGE_CAP,
               "input_files": [item["path"] for item in inputs], "pvtools": "lib/pvtools.py",
               "local_execution_isolation": "none" if backend == "local" else backend})
    write_json(rec.root / "model.json", {"model": model.model, "endpoint": model.url, "protocol": getattr(model, "protocol", "chat_completions"),
                                        "reasoning_effort": model.reasoning_effort, "stream": getattr(model, "stream", False)})
    (rec.root / "prompts").mkdir()
    (rec.root / "prompts/system.txt").write_text(system)
    write_json(rec.root / "prompts/initial_user.json", {"role": "user", "content": prompt})
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    submission_started = False
    def call_model(messages, tools, limit, group):
        for anomaly_number in range(1, 4):
            answer = model.call(messages, tools, limit, rec, group)
            if not answer.get("_protocol_anomaly"):
                return answer
            messages.append(answer)
            rec.event("model_protocol_anomaly", reason=answer["_protocol_anomaly"],
                      consecutive=anomaly_number, recovery_limit=2)
            if mode != "agent" or anomaly_number == 3:
                raise ModelProtocolError("API returned no actionable answer or tool call; recovery limit exhausted")
            messages.append({"role":"user", "content":
                "Your previous response contained no actionable answer or tool call. Continue using tools, "
                "or submit with FINAL: if the task is complete."})
            rec.dialogue_feedback(messages[-1])
            result["protocol_recoveries"] = result.get("protocol_recoveries", 0) + 1
            rec.event("model_protocol_recovery", consecutive=anomaly_number, group=group)

    def submit(last_successful=False):
        nonlocal submission_started, workspace
        submission_started = True
        rec.begin_delivery(result.get("stop_reason", "submission"))
        workspace.deadline = deadline
        ledger = json.loads(workspace.get("ledger.json"))
        if not ledger.get("last_source"):
            raise ValueError("no executed scene code to submit")
        relative = str(Path(ledger["last_source"]).relative_to(workspace.root))
        if last_successful:
            chosen = None
            for number in range(ledger["rounds"],0,-1):
                try:
                    execution=json.loads(workspace.get(f"rounds/{number:03d}/execution.json"))
                except (OSError,RuntimeError):
                    continue
                if execution.get("status") == "ok":
                    chosen=f"rounds/{number:03d}/solution.py"
                    break
            if chosen is None:
                raise ValueError("agent budget exhausted without any successful scene execution")
            relative=chosen
            ledger["last_source"]=str(Path(workspace.root)/relative)
            workspace.put("ledger.json",json.dumps(ledger).encode())
        code=workspace.get(str(Path(relative).with_name("executed.py"))).decode()
        result["final_source_round"]=int(Path(relative).parent.name)
        rec.begin_delivery(result.get("stop_reason", "submission"), code)
        workspace, final = deliver(workspace, code)
        result["status"] = "ok" if final["exit_code"] == 0 else "failed"
    try:
        deadline.arm()
        workspace.start()
        resume_messages = None
        if resume_from:
            import base64
            if getattr(model, "protocol", "") != "responses" or mode != "agent":
                raise ValueError("resume requires a Responses agent")
            prior = Path(resume_from)
            prior_env = json.loads((prior / "environment.json").read_text())
            shutil.copytree(prior / "input", rec.root / "input", dirs_exist_ok=True)
            old_workspace = prior / "workspace"
            old_ledger = json.loads((old_workspace / "ledger.json").read_text())
            # Restore generated files/rounds; retain fresh managed launchers and deadline.
            excluded = {"bin", "lib", "runner.py", "launcher.py", "execution_config.json", "ledger.json", "ledger.lock", "final"}
            workspace.restore_workspace(old_workspace, excluded)
            if old_ledger.get("last_source"):
                old_ledger["last_source"] = str(Path(workspace.root) / Path(old_ledger["last_source"]).relative_to(prior_env["workspace"]))
            workspace.put("ledger.json", json.dumps(old_ledger).encode())
            workspace.rounds = old_ledger.get("rounds", 0)
            if workspace.rounds >= max_rounds:
                raise ValueError("resume has no remaining production rounds")
            request = sorted((prior / "trajectory/calls").glob("*/request.json"))[-1]
            def restore_media(value):
                if isinstance(value, dict):
                    if value.get("wire_encoding") == "data_url_base64" and value.get("media_ref"):
                        blob = (prior / value["media_ref"]).read_bytes()
                        return "data:" + value["mime_type"] + ";base64," + base64.b64encode(blob).decode()
                    return {k: restore_media(v) for k,v in value.items()}
                if isinstance(value, list):
                    return [restore_media(v) for v in value]
                return value
            payload = restore_media(json.loads(request.read_text()))
            model.resume_input = payload["input"]
            for item in model.resume_input:
                if item.get("role") == "system":
                    item["content"] = [{"type": "input_text", "text": system}]
            if not any(item.get("role") == "system" for item in model.resume_input):
                model.resume_input.insert(0, {"role": "system", "content": [{"type": "input_text", "text": system}]})
            # If execution stopped after a successful model reply, retain its native
            # output and report the interrupted pending tool calls truthfully.
            last_response = request.with_name("response.json")
            if last_response.exists():
                raw = json.loads(last_response.read_text())
                outputs = raw.get("output") or []
                model.resume_input += outputs
                for item in outputs:
                    if item.get("type") == "function_call":
                        model.resume_input.append({"type": "function_call_output", "call_id": item["call_id"],
                            "output": json.dumps({"status": "executor_interrupted",
                              "reason": "Prior executor stopped while handling this reply. Current source and artifacts are preserved in the restored workspace.",
                              "rounds_already_used": workspace.rounds,
                              "last_source": old_ledger.get("last_source")})})
            resume_messages = [{"role": "user", "content":
                "The prior run stopped at an infrastructure limit or interruption. All existing production attempts, including interrupted ones, remain counted. "
                "Continue from the complete conversation and restored workspace above; do not restart the task or rewrite the approved prompt. "
                f"Current workspace: {workspace.root}. Production rounds already used: {workspace.rounds}; maximum total remains {max_rounds}. "
                "The API now keeps at most 50 images, preserving initial references and newest observations; full prior observations remain archived. "
                "Continue building/reviewing with the four tools, and submit the last executed version with FINAL when ready."}]
            resume_messages.insert(0, {"role": "system", "content":
                f"Updated host runtime configuration for this continuation: total time limit {str(timeout_s) + ' seconds' if timeout_s is not None else 'none'}, with {workspace.rounds} production rounds already used and maximum {max_rounds} overall. "
                "Blender rendering has no separate time limit; any configured total task deadline still applies. This runtime update supersedes earlier time-box notes; all scene, model, tool, and round-limit requirements remain unchanged."})
            write_json(rec.root / "resume.json", {"from": str(prior), "request": str(request), "rounds_before": workspace.rounds,
                         "previous_result": json.loads((prior / "result.json").read_text()),
                         "previous_trajectory": str(prior / "trajectory/dialogue.jsonl"),
                         "previous_usage": str(prior / "usage.json"), "backend": backend})
            if (prior / "plan.md").exists():
                shutil.copy2(prior / "plan.md", rec.root / "plan.md")
            rec.event("resume", from_run=str(prior), rounds_before=workspace.rounds, request=str(request))
        for item in inputs:  # before media and the first model call; also refreshes a restored workspace
            workspace.put(item["path"], item["source"].read_bytes())
        if inputs:
            rec.event("input_files", files=[{k: v for k, v in item.items() if k != "source"} for item in inputs])
        if media and not resume_from:
            from .tools import input_media
            blocks, manifest = input_media(workspace, media)
            group = manifest["sampling"]["group_size"]
            content = [{"type": "text", "text": prompt + "\nReference media follow: images are sent as they are; videos are given as 1 fps timestamped frames"
                        + (f" (tiled {group} per image, top to bottom, each part labelled with its timestamp)" if group > 1 else "")
                        + ". All frames and the original files are in the workspace (input/frames/, input/media/); inspect further as needed."}, *blocks]
            messages[1] = {"role": "user", "content": content}
            write_json(rec.root / "input/media_manifest.json", manifest)
            from .model import portable_request
            write_json(rec.root / "prompts/initial_user.json", portable_request(messages[1], rec))
        runtime = getattr(workspace,"execution_config",{})
        versions = json.loads((rec.root/"environment.json").read_text()).get("versions","") if (rec.root/"environment.json").exists() else ""
        note = f"\nRuntime: {versions.splitlines()[0] if versions else 'see environment.json'}; managed render engine: {runtime.get('render_engine','MODEL_CHOICE')}. Remaining agent work time: {str(round(workspace.deadline.remaining(),1)) + ' seconds' if timeout_s is not None else 'unlimited'}; delivery remains within the total task budget."
        if isinstance(messages[1]["content"], str):
            messages[1]["content"] += note
        else:
            messages[1]["content"][0]["text"] += note
        from .model import portable_request
        write_json(rec.root / "prompts/initial_user.json", portable_request(messages[1], rec))
        if mode == "end_to_end":
            answer = call_model(messages, None, deadline, "round_001")
            code = extract_code(answer.get("content"))
            submission_started = True
            rec.begin_delivery("single_submission", code)
            workspace, outcome = deliver(workspace, code, single=True)
            result.update(status="ok" if outcome["exit_code"] == 0 else "failed", stop_reason="single_submission")
        else:
            if resume_messages is not None:
                messages = resume_messages
            no_action = 0
            for step in range(256):  # Internal loop protection; total time remains the user-facing budget.
                work_deadline.remaining()
                group = "closing" if workspace.rounds >= max_rounds else f"round_{workspace.rounds + 1:03d}"
                # Planning is optional: the first call already has the tools; its text, if any, is the plan.
                answer = call_model(messages, definitions(), work_deadline, group)
                messages.append(answer)
                if step == 0 and resume_messages is None and answer.get("content"):
                    (rec.root / "plan.md").write_text(answer["content"])
                    rec.event("plan", text=answer["content"], with_tool_calls=bool(answer.get("tool_calls")))
                calls = answer.get("tool_calls") or []
                if not calls:
                    content = answer.get("content") or ""
                    if content.lstrip().startswith("FINAL:"):
                        (rec.root / "final_answer.md").write_text(content)
                        result["stop_reason"] = "model_submission"
                        break
                    no_action += 1
                    if no_action >= 3:
                        result["stop_reason"] = "no_action"
                        break
                    messages.append({"role": "user", "content": "Continue using tools. When complete, submit the current executed version with FINAL:."})
                    rec.dialogue_feedback(messages[-1])
                    continue
                no_action = 0
                media_blocks = []
                for call in calls:
                    name = call["function"]["name"]
                    try:
                        args = json.loads(call["function"].get("arguments") or "{}")
                        feedback = workspace.tool(name, args)
                        feedback.update(rounds_remaining=max(0, max_rounds - workspace.rounds),
                                        remaining_s=round(work_deadline.remaining(), 2) if timeout_s is not None else None)
                        try:
                            blocks, refs = observations(workspace, args.get("media", []))
                            media_blocks += blocks
                            feedback["media"] = refs
                            rec.event("observation", tool_call_id=call["id"], media=refs)
                        except TimeoutError:
                            raise
                        except Exception as exc:
                            feedback["observation_error"] = str(exc)
                            rec.event("observation_error", tool_call_id=call["id"], error=str(exc))
                    except TimeoutError:
                        raise
                    except Exception as exc:
                        feedback = {"error": str(exc)}
                        rec.event("tool_error", name=name, arguments=call["function"].get("arguments"), error=str(exc))
                    # Raw complete feedback is on disk. Clip only model-visible stdout/stderr.
                    shown = dict(feedback)
                    for key in ("stdout", "stderr"):
                        if len(shown.get(key, "")) > 20000:
                            shown[key] = shown[key][:20000] + "\n[truncated; read the relevant log with bash/python]"
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(shown, ensure_ascii=False)})
                    rec.dialogue_feedback(messages[-1])
                if media_blocks:
                    messages.append({"role": "user", "content": media_blocks})
                    rec.dialogue_feedback(portable_request(messages[-1], rec))
                # Adapt the delivery reserve to observed render cost without a new user knob.
                if timeout_s is not None and final_resolution != tuple(resolution):
                    executions=sorted((rec.root/'rounds').glob('*/execution.json'))
                    if executions:
                        last=json.loads(executions[-1].read_text())
                        if last.get('status')=='ok':
                            ratio=final_resolution[0]*final_resolution[1]/(resolution[0]*resolution[1])
                            estimate=last.get('elapsed_s',0)*ratio*1.5+30
                            revised=min(timeout_s*0.5,max(delivery_reserve_s,estimate))
                            if revised>delivery_reserve_s:
                                delivery_reserve_s=revised
                                work_deadline.ends=deadline.ends-delivery_reserve_s
                                rec.event('delivery_reserve_updated',reserve_s=delivery_reserve_s,
                                          reason='observed render time and final pixel count')
                if workspace.rounds >= max_rounds:
                    result["stop_reason"] = "max_rounds"
                    break
            else:
                result["stop_reason"] = "internal_step_limit"
            submit()
    except DeliveryFailure as exc:
        workspace = exc.workspace
        result.update(status="failed", stop_reason="delivery_failed", failure_category=exc.category, error=str(exc))
    except TimeoutError as exc:
        result.update(stop_reason="timeout", error=str(exc))
        if (mode == "agent" and not submission_started and time.monotonic() >= work_deadline.ends
                and time.monotonic() < deadline.ends and workspace.rounds):
            try:
                result["stop_reason"]="agent_budget_reserved_delivery"
                rec.event("automatic_submission", reason="agent work budget expired; using last successful execution")
                submit(last_successful=True)
                if result["status"] == "ok":
                    result.pop("error",None)
            except Exception as final_exc:
                result.update(status="failed", error=str(final_exc))
                if isinstance(final_exc, DeliveryFailure):
                    workspace = final_exc.workspace
                    result["failure_category"] = final_exc.category
    except ModelProtocolError as exc:
        result.update(stop_reason="model_protocol_error", error=str(exc))
    except KeyboardInterrupt:
        result.update(stop_reason="interrupted", error="user interrupted execution")
    except Exception as exc:
        result.update(stop_reason="error", error=str(exc))
    finally:
        if rec.phase != "delivery":
            rec.begin_cleanup()
        deadline.disarm()
        if backend == "docker" and workspace.holder and workspace.holder.sandbox_id:
            # Recovery only: no new model calls or rendering after the work deadline.
            recovery = Deadline(30)
            original = workspace.deadline
            workspace.deadline = recovery
            try:
                recovery.arm()
                workspace.sync()
            except Exception as exc:
                rec.event("artifact_recovery_error", error=str(exc))
            finally:
                recovery.disarm()
                workspace.deadline = original
        # Local workspace survives a deadline long enough to preserve artifacts.
        if backend == "local" and workspace.root:
            # Run directories link input -> workspace input/; archive real files only, as the sandbox sync does.
            links = lambda folder, names: [n for n in names if Path(folder, n).is_symlink()]
            shutil.copytree(workspace.root, rec.root / "workspace", dirs_exist_ok=True, ignore=links)
            for name in ("rounds", "final"):
                p = Path(workspace.root) / name
                if p.exists():
                    shutil.copytree(p, rec.root / name, dirs_exist_ok=True, ignore=links)
            ledger = Path(workspace.root) / "ledger.json"
            if ledger.exists():
                workspace.rounds = json.loads(ledger.read_text()).get("rounds", 0)
        result["rounds"] = workspace.rounds
        pending = f"round_{workspace.rounds + 1:03d}"
        if mode == "agent" and workspace.rounds:
            for stat in rec.calls:
                if stat["group"] == pending:
                    stat["group"] = "closing"
        rec.event("usage_groups_finalized", assignments=[{"call": c["call"], "group": c["group"]} for c in rec.calls])
        result["elapsed_s"] = time.monotonic() - deadline.started
        stage_path = rec.root / "final/stages.json"
        if stage_path.exists():
            write_json(rec.root / "final/render.json", json.loads(stage_path.read_text()))
        source_path = rec.root / "final/solution.py"
        if not source_path.exists() and workspace.rounds:
            source_path = rec.root / f"rounds/{workspace.rounds:03d}/solution.py"
        try:
            from .validation import executed_representation
            text = executed_representation(source_path.read_text(), source_path.parent)
            (rec.root / "representation.md").write_text(text)
            result["representation_status"] = "provided_not_semantically_verified"
        except (OSError, SyntaxError, ValueError, TypeError) as exc:
            result["representation_status"] = "missing_or_invalid"
            result["representation_error"] = str(exc)
        result["artifacts"] = {}
        for name in ("solution.py", "scene.blend", "video.mp4"):
            file = rec.root / "final" / name
            result["artifacts"][name] = {"path": str(file.relative_to(rec.root)), "sha256": digest(file)} if file.exists() else {
                "path": None, "reason": result.get("error") or "not produced; see execution logs"}
        if result["status"] == "ok" and any(v["path"] is None for v in result["artifacts"].values()):
            result["status"] = "failed"
        explanation = rec.root / "representation.md"
        result["artifacts"]["representation.md"] = {"path": "representation.md", "sha256": digest(explanation)} if explanation.exists() else {
            "path": None, "reason": result["representation_error"]}
        if submission_started:
            try:
                validation = validate_delivery(rec, workspace, json.loads((rec.root/"config.json").read_text()))
                result["delivery_status"] = validation["status"]
                if validation["status"] != "ok":
                    result["status"] = "failed"
                    result.setdefault("failure_category", validation.get("failure_category") or "delivery_incomplete")
                    result.setdefault("error", "Delivery checks failed: " + ", ".join(c["name"] for c in validation["checks"] if not c["passed"]))
                if not (rec.root/"representation.md").exists():
                    result["representation_status"]="missing_or_invalid"
                    result["artifacts"]["representation.md"]={"path":None,"reason":"required representation is absent or invalid"}
            except Exception as validation_error:
                result.update(status="failed",delivery_status="failed")
                result.setdefault("failure_category","environment_failure")
                result.setdefault("error",str(validation_error))
        if result["status"] != "ok" and "failure_category" not in result:
            detail=result.get("error", "")
            if result["stop_reason"]=="model_protocol_error" or "model response" in detail or "model HTTP" in detail:
                result["failure_category"]="api_protocol_error"
            elif not rec.calls or result["stop_reason"] in ("timeout", "interrupted"):
                result["failure_category"]="environment_failure"
            else:
                result["failure_category"]="model_code_error"
        rec.begin_cleanup()
        cleanup = Deadline(15)
        try:
            cleanup.arm()
            workspace.close()
        except Exception as exc:
            rec.event("cleanup_error", error=str(exc))
        finally:
            cleanup.disarm()
        result["elapsed_s"] = time.monotonic()-deadline.started
        rec.event("end", **result)
        rec.finish_dialogue()
        finish_reports(rec, result)
    return result
