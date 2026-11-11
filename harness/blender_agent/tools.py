"""Only four public tools. Media requests are fields on these tools."""
import base64
import json
import math
from pathlib import Path
import re
import shlex
import shutil

from .common import digest


INITIAL_IMAGE_CAP = 30  # images in the initial user message; limit_images protects the same set
IMAGES = (".png", ".jpg", ".jpeg", ".webp")
VIDEOS = (".mp4", ".mov", ".webm", ".mkv")
# 5x7 bitmap glyphs (hex rows, high bit = left column): burned-in labels need no font in the sandbox.
GLYPHS = {"0": "0e11131519110e", "1": "040c040404040e", "2": "0e11010204081f", "3": "1f02040201110e",
          "4": "02060a121f0202", "5": "1f101e0101110e", "6": "0608101e11110e", "7": "1f010204080808",
          "8": "0e11110e11110e", "9": "0e11110f01020c", ".": "00000000000c0c", "=": "00001f001f0000",
          "t": "08081c08080906", "s": "00000e100e011e", "-": "0000001f000000"}


def sample_times(duration, frame_period):
    """1 fps: t=0,1,2,... < duration; the last sample is at most one frame period before the end."""
    return sorted({max(0.0, min(float(k), duration - frame_period)) for k in range(max(1, math.ceil(duration)))})


def tile_group(counts, images, cap=INITIAL_IMAGE_CAP):
    """1 when every frame fits; otherwise the smallest per-video group size that fits the cap."""
    if sum(counts) + images <= cap:
        return 1
    if images >= cap:
        raise ValueError(f"{images} input images leave no room for video frames under the {cap}-image cap")
    group = math.ceil(sum(counts) / (cap - images))
    while sum(math.ceil(n / group) for n in counts) + images > cap:
        group += 1
    return group


def seconds_label(t):
    text = f"{t:.2f}".rstrip("0")
    return "t=" + text + ("0" if text.endswith(".") else "") + "s"


def label_filter(text, x=8, y=8, scale=3, pad=5):
    """drawbox chain: white text on a black box at the top-left."""
    boxes = [f"drawbox=x={x}:y={y}:w={2 * pad + (6 * len(text) - 1) * scale}:h={2 * pad + 7 * scale}:color=black:t=fill"]
    for i, char in enumerate(text):
        for row, bits in enumerate(bytes.fromhex(GLYPHS[char])):
            for run in re.finditer("1+", f"{bits:05b}"):
                boxes.append(f"drawbox=x={x + pad + (6 * i + run.start()) * scale}:y={y + pad + row * scale}"
                             f":w={len(run.group()) * scale}:h={scale}:color=white:t=fill")
    return ",".join(boxes)


def video_frames(workspace, relative, folder):
    """One decode pass: 1 fps samples (long side <= 1280) with the decoded PTS of every frame."""
    probe = workspace.command(shlex.join([workspace.ffmpeg, "-hide_banner", "-i", workspace.path(relative)]))
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", probe["stderr"])
    if not match:
        raise ValueError(f"cannot determine video duration for {relative} (exit {probe['exit_code']}): " + probe['stderr'][-2000:])
    h, m, s = map(float, match.groups())
    duration = h * 3600 + m * 60 + s
    fps_match = re.search(r"(\d+(?:\.\d+)?) fps", probe["stderr"])
    frame_period = 1 / float(fps_match.group(1)) if fps_match else .1
    times = sample_times(duration, frame_period)
    raw = folder + "/.raw"
    select = f"select=lt(selected_n\\,{len(times)})*gte(t\\,min(selected_n\\,{times[-1]:.6f})),showinfo"
    scale = "scale=w=min(1280\\,iw):h=min(1280\\,ih):force_original_aspect_ratio=decrease:force_divisible_by=2"
    result = workspace.command(shlex.join(["rm", "-rf", folder]) + " && " + shlex.join(["mkdir", "-p", raw]) + " && " + shlex.join([
        workspace.ffmpeg, "-y", "-hide_banner", "-i", workspace.path(relative), "-vf", select + "," + scale,
        "-fps_mode", "passthrough", "-frames:v", str(len(times)), "-q:v", "2", workspace.path(raw) + "/f_%04d.jpg"]))
    pts = [float(x) for x in re.findall(r"pts_time:\s*(-?[\d.]+)", result["stderr"])]
    if result["exit_code"] or not pts:
        raise ValueError(f"frame extraction failed for {relative}: " + result["stderr"][-2000:])
    frames = [{"file": f"{folder}/t_{t:07.2f}.jpg", "t_requested": t, "t_pts": p} for t, p in zip(times, pts)]
    if len({f["file"] for f in frames}) != len(frames):
        raise ValueError(f"sampled timestamps of {relative} collide at 0.01 s resolution")
    moves = [shlex.join(["mv", f"{raw}/f_{i:04d}.jpg", f["file"]]) for i, f in enumerate(frames, 1)]
    return {"source": relative, "duration_s": duration, "frame_period_s": frame_period, "index": folder + "/index.json",
            "frames": frames}, moves + [shlex.join(["rmdir", raw])]


def input_media(workspace, paths, cap=INITIAL_IMAGE_CAP):
    """Initial references: images as they are; videos as 1 fps timestamped frames, tiled only to fit the cap."""
    if len(paths) > 16:
        raise ValueError("at most 16 input files")
    files, videos, script = [], {}, ["set -e"]
    for index, source in enumerate(paths):
        source = Path(source)
        if not source.is_file() or source.suffix.lower() not in IMAGES + VIDEOS:
            raise ValueError(f"unsupported input media: {source}")
        relative = f"input/media/{index:03d}{source.suffix.lower()}"
        target = workspace.recorder.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        workspace.put(relative, target.read_bytes())
        files.append({"path": relative, "original_name": source.name, "sha256": digest(target), "bytes": target.stat().st_size})
        if source.suffix.lower() in VIDEOS:
            videos[relative], moves = video_frames(workspace, relative, f"input/frames/{index:03d}")
            script += moves
    images = len(files) - len(videos)
    group = tile_group([len(v["frames"]) for v in videos.values()], images, cap)
    shown = []  # (workspace file or None, model-visible JSON description)
    for item in files:
        video = videos.get(item["path"])
        if video is None:
            shown.append((item["path"], {"source": item["path"], "delivery": "image"}))
            continue
        workspace.put(video["index"], json.dumps(video["frames"], indent=1).encode())
        frames = video["frames"]
        for start in range(0, len(frames), group):
            parts = frames[start:start + group]
            first = parts[0]["file"].rsplit("/", 1)
            if group == 1:
                output = first[0] + "/view_" + first[1]
                command = [workspace.ffmpeg, "-y", "-loglevel", "error", "-i", parts[0]["file"], "-vf",
                           "scale=768:768:force_original_aspect_ratio=decrease:force_divisible_by=2", "-q:v", "2", output]
                description = {"source": item["path"], "delivery": "timestamped_video_frame",
                               "t_s": round(parts[0]["t_pts"], 3), "file": parts[0]["file"]}
            else:
                output = first[0] + "/tile_" + first[1]
                graph = ";".join(f"[{k}:v]scale=768:-2,format=rgb24,{label_filter(seconds_label(p['t_pts']))}[p{k}]"
                                 for k, p in enumerate(parts))
                graph += ";" + "".join(f"[p{k}]" for k in range(len(parts))) + (f"vstack=inputs={len(parts)}" if len(parts) > 1 else "null")
                command = [workspace.ffmpeg, "-y", "-loglevel", "error"] + [x for p in parts for x in ("-i", p["file"])]
                command += ["-filter_complex", graph, "-frames:v", "1", "-q:v", "2", output]
                description = {"source": item["path"], "delivery": "timestamped_video_frames_tile",
                               "parts": [{"t_s": round(p["t_pts"], 3), "file": p["file"]} for p in parts], "layout": "top_to_bottom"}
            script.append(shlex.join(command))
            shown.append((output, description))
    if videos:
        workspace.put(".input_media.sh", "\n".join(script).encode())
        built = workspace.command("bash .input_media.sh; status=$?; rm -f .input_media.sh; exit $status")
        if built["exit_code"]:
            raise ValueError("input frame preparation failed: " + built["stderr"][-2000:])
    blocks, records = [], []
    requested = {f["file"]: f["t_requested"] for v in videos.values() for f in v["frames"]}
    for file, description in shown:
        data = workspace.get(file)
        local = workspace.recorder.root / "trajectory/observations" / file
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        mime = "image/jpeg" if Path(file).suffix.lower() in (".jpg", ".jpeg") else "image/" + Path(file).suffix.lstrip(".")
        parts = description.get("parts") or ([description] if "t_s" in description else [])
        records.append({**description, "path": str(local.relative_to(workspace.recorder.root)), "workspace_file": file,
                        "timestamp_s": parts[0]["t_s"] if parts else None,
                        "requested_timestamp_s": requested.get(parts[0]["file"]) if parts else None})
        blocks += [{"type": "text", "text": json.dumps(description, ensure_ascii=False)},
                   {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + base64.b64encode(data).decode()}}]
    sampling = {"fps": 1, "initial_image_cap": cap, "group_size": group, "layout": "top_to_bottom", "part_width": 768,
                "single_frame_long_side": 768, "saved_frame_long_side": 1280, "label": "t=<decoded PTS>s, white on black, top-left"}
    manifest = {"files": files, "video_delivery": "fps1_timestamped_frames", "sampling": sampling,
                "videos": list(videos.values()), "model_observations": records, "image_count": len(records)}
    workspace.recorder.event("input_media", files=files, observations=records, sampling=sampling, image_count=len(records))
    return blocks, manifest


def definitions():
    media = {"type": "array", "description": "Optional visual feedback from workspace files, e.g. [{\"path\":\"rounds/001/preview.mp4\",\"times\":[0,1.5]}].",
             "maxItems": 16, "items": {"type": "object", "properties": {
        "path": {"type": "string", "description": "workspace-relative image/video file"},
        "times": {"type": "array", "maxItems": 16, "items": {"type": "number"}, "description": "video timestamps in seconds"},
        "count": {"type": "integer", "minimum": 1, "maximum": 16, "description": "uniform video sample count; default 4"}},
        "required": ["path"]}}
    specs = [
        ("bash", "Run bash in the task workspace. Use managed `blender --python path.py` for scene execution. For an existing engineering file, use `blender --purpose query --blend-file rounds/001/scene.blend --python query.py`. Read code/logs or use ffmpeg to trim/extract media as needed.",
         {"command": {"type": "string"}}, ["command"]),
        ("python", "Execute ordinary Python in the workspace; bpy is only available inside Blender.",
         {"code": {"type": "string"}}, ["code"]),
        ("blender", "Render complete bpy code (one execution round), producing rounds/NNN/solution.py, scene.blend, preview.mp4, frames/ and keyframes/. With purpose=query and blend_file, open an existing .blend and execute bpy queries without a new scene round.",
         {"code": {"type": "string"}, "purpose": {"type": "string", "enum": ["render", "query"]},
          "blend_file": {"type": "string", "description": "workspace-relative .blend required for query"}}, ["code"]),
        ("ffmpeg", "Run ffmpeg with an argument list to encode, trim or extract frames. Request media to see results. Use `-hide_banner -i path.mp4` to read metadata; ffprobe is not guaranteed to be installed.",
         {"args": {"type": "array", "items": {"type": "string"}}}, ["args"])]
    return [{"type": "function", "function": {"name": name, "description": desc,
             "parameters": {"type": "object", "properties": dict(props, media=media), "required": required}}}
            for name, desc, props, required in specs]


def observations(workspace, requests):
    """Chat v1 uses images. Videos are explicit timestamped samples, never silently native-video claims."""
    if not isinstance(requests, list) or len(requests) > 16:
        raise ValueError("at most 16 media requests")
    blocks, records = [], []
    for request in requests:
        name = request["path"]
        workspace.path(name)
        suffix = Path(name).suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".webp"):
            files = [(name, None, None)]
        elif suffix in (".mp4", ".mov", ".webm", ".mkv"):
            probe = workspace.command(shlex.join([workspace.ffmpeg, "-hide_banner", "-i", workspace.path(name)]))
            match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", probe["stderr"])
            if not match:
                raise ValueError(f"cannot determine video duration for {name} (exit {probe['exit_code']}): " + probe['stderr'][-2000:])
            h, m, s = map(float, match.groups())
            duration = h * 3600 + m * 60 + s
            fps_match = re.search(r"(\d+(?:\.\d+)?) fps", probe["stderr"])
            frame_period = 1 / float(fps_match.group(1)) if fps_match else .1
            times = request.get("times")
            if times is None:
                count = request.get("count", 4)
                if not isinstance(count, int) or not 1 <= count <= 16:
                    raise ValueError("sample count must be 1..16")
                times = [(i + 0.5) * duration / count for i in range(count)]
            if not isinstance(times, list) or not 1 <= len(times) <= 16:
                raise ValueError("timestamps must contain 1..16 values")
            files = []
            for t in times:
                if not isinstance(t, (int, float)) or not 0 <= t < duration:
                    raise ValueError("timestamp outside video")
                target = f"observations/{workspace.tools_called:04d}_{len(records):03d}_{len(files):02d}.jpg"
                workspace.command("mkdir -p observations")
                sample_at = min(t, max(0, duration - frame_period - .01))
                result = workspace.command(shlex.join([workspace.ffmpeg, "-y", "-hide_banner", "-i", workspace.path(name),
                    "-vf", f"select=gte(t\\,{sample_at}),showinfo", "-frames:v", "1", workspace.path(target)]))
                if result["exit_code"]:
                    raise ValueError("frame extraction failed: " + result["stderr"])
                pts = re.search(r"pts_time:([\d.]+)", result["stderr"])
                if not pts:
                    raise ValueError("no decodable frame at requested timestamp")
                files.append((target, float(pts.group(1)), t))
        else:
            raise ValueError("media must be an image or video; inspect .blend with blender query")
        for file, timestamp, requested_timestamp in files:
            if len(records) >= 16:
                raise ValueError("at most 16 returned image frames per tool call")
            data = workspace.get(file)
            local = workspace.recorder.root / "trajectory/observations" / file
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            mime = "image/jpeg" if Path(file).suffix.lower() in (".jpg", ".jpeg") else "image/" + Path(file).suffix.lstrip(".")
            records.append({"source": name, "path": str(local.relative_to(workspace.recorder.root)),
                            "timestamp_s": timestamp, "requested_timestamp_s": requested_timestamp,
                            "source_duration_s": duration if timestamp is not None else None,
                            "delivery": "timestamped_video_frame" if timestamp is not None else "image"})
            blocks += [{"type": "text", "text": json.dumps(records[-1], ensure_ascii=False)},
                       {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + base64.b64encode(data).decode()}}]
    return blocks, records
