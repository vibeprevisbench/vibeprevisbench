"""Delivery checks only; never send these results to the model."""
import ast
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path

from .common import digest, write_json
from .execution import host_ffmpeg


def representation_text(code):
    for node in ast.parse(code).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(target, ast.Name) and target.id == "REPRESENTATION" for target in targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str) and value.strip():
                return value.strip() + "\n"
            break
    raise ValueError("model did not provide a nonempty top-level literal REPRESENTATION string")


def executed_representation(code, directory):
    """Use only explanation captured by the execution of this exact source."""
    try:
        return representation_text(code)
    except (ValueError, TypeError, SyntaxError):
        stage = json.loads((Path(directory) / "stages.json").read_text())
        value = stage.get("runtime_representation")
        if (stage.get("source_sha256") == hashlib.sha256(code.encode()).hexdigest()
                and isinstance(value, str) and value.strip()):
            return value.strip() + "\n"
        raise ValueError("no representation bound to this executed source")


def valid_png(path, resolution):
    try:
        with Path(path).open("rb") as source:
            header = source.read(24)
            source.seek(-8, 2)
            tail = source.read()
        return (header[:8] == b"\x89PNG\r\n\x1a\n" and tail == b"IEND\xaeB`\x82"
                and struct.unpack(">II", header[16:24]) == tuple(resolution))
    except (OSError, ValueError, struct.error):
        return False


def validate_delivery(recorder, workspace, config):
    root = recorder.root
    final = root / "final"
    checks = []

    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))

    for name in ("solution.py", "scene.blend", "video.mp4"):
        path = final / name
        check("file:" + name, path.is_file() and path.stat().st_size > 0)
    source = final / "solution.py"
    code = source.read_text() if source.exists() else ""
    try:
        ast.parse(code)
        check("script_syntax", bool(code.strip()))
    except SyntaxError as exc:
        check("script_syntax", False, error=str(exc))
    frozen = root / "delivery/submission.py"
    check("frozen_code", source.exists() and frozen.exists() and digest(source) == digest(frozen))
    try:
        explanation = executed_representation(code, final)
        (root / "representation.md").write_text(explanation)
        check("file:representation.md", True)
    except (OSError, ValueError, SyntaxError, TypeError) as exc:
        (root / "representation.md").unlink(missing_ok=True)
        check("file:representation.md", False, error=str(exc))

    stage_path = final / "stages.json"
    stage = json.loads(stage_path.read_text()) if stage_path.exists() else {}
    check("standalone_script", not stage.get("script_dependencies"),
          dependencies=stage.get("script_dependencies", {}))
    # The preinstalled tagging module must be the release copy, both in the workspace and as executed.
    released = digest(Path(__file__).with_name("pvtools.py"))
    try:
        shipped = hashlib.sha256(workspace.get("lib/pvtools.py") if workspace and workspace.root
                                 else (root / "workspace/lib/pvtools.py").read_bytes()).hexdigest()
    except Exception as exc:
        shipped = "unreadable: " + str(exc)
    check("pvtools_intact", shipped == released and stage.get("pvtools_sha256") == released,
          expected=released, workspace=shipped, executed=stage.get("pvtools_sha256"))
    frame_range = stage.get("frame_range") or []
    expected = config.get("expected_frame_count")
    declared = frame_range[1] - frame_range[0] + 1 if len(frame_range) == 2 else None
    expected = expected if expected is not None else declared
    check("render_complete", stage.get("status") == "ok")
    check("frame_range", len(frame_range) == 2 and frame_range[0] == 1 and declared == expected,
          expected=expected, actual=frame_range)
    resolution = config["final_resolution"]
    frames = sorted((final / "frames").glob("f_*.png"))
    names = {path.name for path in frames}
    wanted = {f"f_{index:04d}.png" for index in range(1, (expected or 0) + 1)}
    check("all_frames", bool(expected) and names == wanted and all(valid_png(path, resolution) for path in frames),
          expected=expected, actual=len(frames))

    video = final / "video.mp4"
    if video.exists():
        try:
            ffmpeg = host_ffmpeg(portable=True)
            decoded = subprocess.run([ffmpeg, "-hide_banner", "-i", str(video), "-map", "0:v:0",
                                      "-fps_mode", "passthrough", "-progress", "pipe:1", "-nostats", "-f", "null", "-"],
                                     capture_output=True, text=True, timeout=3600)
            metadata = decoded.stderr
            line = next((line for line in metadata.splitlines() if "Video:" in line), "")
            counts = re.findall(r"^frame=(\d+)$", decoded.stdout, re.M)
            count = int(counts[-1]) if counts else None
            size = re.search(r"\b(\d+)x(\d+)(?:\s|,)", line)
            fps = re.search(r"([\d.]+) fps", line)
            duration = re.search(r"Duration: (\d+):(\d+):([\d.]+)", metadata)
            seconds = (int(duration[1]) * 3600 + int(duration[2]) * 60 + float(duration[3])) if duration else None
            check("video_decode", decoded.returncode == 0)
            check("video_frame_count", count == expected, expected=expected, actual=count)
            check("video_resolution", bool(size) and [int(size[1]), int(size[2])] == list(resolution))
            check("video_fps", bool(fps) and abs(float(fps[1]) - config["fps"]) < 0.001)
            check("video_duration", seconds is not None and expected is not None
                  and abs(seconds - expected / config["fps"]) <= 0.02, actual=seconds)
            check("video_codec", bool(re.search(r"Video: h264\b", line)))
            check("video_pixel_format", bool(re.search(r"\byuv420p(?:\b|\()", line)))
            check("video_no_audio", "Audio:" not in metadata)
        except Exception as exc:
            check("video_probe", False, error=str(exc))

    if workspace and (final / "scene.blend").exists():
        try:
            query = ("import bpy,json; s=bpy.context.scene; "
                     "print('BA_DELIVERY_SCENE '+json.dumps({'fps':s.render.fps/s.render.fps_base,"
                     "'resolution':[s.render.resolution_x,s.render.resolution_y],"
                     "'frames':[s.frame_start,s.frame_end]}))")
            result = workspace.tool("blender", {"purpose": "query", "blend_file": "final/scene.blend", "code": query})
            match = re.search(r"BA_DELIVERY_SCENE (\{[^\n]+\})", result["stdout"])
            scene = json.loads(match[1]) if match else {}
            check("editable_scene_open", result["exit_code"] == 0 and bool(scene))
            check("scene_spec", scene.get("frames") == [1, expected]
                  and scene.get("resolution") == list(resolution) and scene.get("fps") == config["fps"], actual=scene)
        except Exception as exc:
            check("editable_scene_open", False, error=str(exc))

    failed=[item for item in checks if not item["passed"]]
    category = ("environment_failure" if any(item["name"] in ("video_probe", "editable_scene_open") and item.get("error") for item in failed)
                else "model_code_error" if any(item["name"]=="script_syntax" for item in failed) else "delivery_incomplete")
    report = {"status": "ok" if not failed else "failed", "failure_category":category if failed else None, "checks": checks,
              "frame_count_source": "task_spec" if config.get("expected_frame_count") is not None else "declared_scene",
              "content_quality_evaluated": False}
    write_json(root / "delivery/validation.json", report)
    recorder.event("delivery_validation", status=report["status"], failed=[item["name"] for item in checks if not item["passed"]])
    return report
