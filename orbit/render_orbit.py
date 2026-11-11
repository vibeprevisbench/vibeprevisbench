"""Evaluation-side orbit render of a static scene (protocol orbit_v1).

Usage (Blender python, scene is opened read-only and never saved):

    LP_NUM_THREADS=2 blender -b <scene.blend> --python render_orbit.py -- \
        --out <dir> [--ffmpeg ffmpeg] [--threads 2] [--keep-frames]

Writes into <dir>:
    orbit.mp4    part 1 (interior 360 pan) + part 2 (dollhouse orbit), H.264
    sheet.jpg    2x2 contact sheet (two frames of each part)
    params.json  every derived parameter + simple view-quality metrics

The procedure is fully deterministic (no randomness); see README.md.
"""
import bpy
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from mathutils import Vector

PROTOCOL = "orbit_v1"
RES_X, RES_Y, FPS = 640, 360, 24
PART_FRAMES = 96                 # 4 s per part @ 24 fps
EYE_H = 1.6                      # m above floor (part 1)
FOV_H_DEG = 70.0                 # horizontal FOV, both parts
HUGE_DIM = 50.0                  # objects bigger than this are ignored for bounds
DOLL_ELEV_DEG = 55.0
DOLL_CUT_H = 2.2                 # hide objects whose bbox min z > floor + this (part 2)
DOLL_MARGIN = 1.0                # m added to the orbit radius
DOLL_DIST_K = 1.3                # camera distance from c = K * radius (along the 55 deg line)
CLEAR_MIN = 0.35                 # m; eye needs this clearance in all horizontal rays
ENCLOSED_MED = 0.8               # m; median horizontal ray distance below this = enclosed
NEAR_DIST = 0.30                 # m; a probe ray hitting closer counts as "blocked"
BG_GRAY = "0x3c3c3c"             # background colour composited under alpha


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = {"out": None, "ffmpeg": "ffmpeg", "threads": 2, "keep_frames": False}
    i = 0
    while i < len(argv):
        k = argv[i]
        if k == "--out":
            a["out"] = argv[i + 1]; i += 2
        elif k == "--ffmpeg":
            a["ffmpeg"] = argv[i + 1]; i += 2
        elif k == "--threads":
            a["threads"] = int(argv[i + 1]); i += 2
        elif k == "--keep-frames":
            a["keep_frames"] = True; i += 1
        else:
            raise SystemExit("unknown arg " + k)
    if not a["out"]:
        raise SystemExit("--out required")
    return a


def world_bbox(obj, dg):
    ev = obj.evaluated_get(dg)
    pts = [ev.matrix_world @ Vector(c) for c in ev.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def visible_meshes(scene, vl):
    out = []
    for o in scene.objects:
        if o.type != "MESH" or o.hide_render:
            continue
        try:
            if not o.visible_get(view_layer=vl):
                continue
        except Exception:
            continue
        out.append(o)
    return sorted(out, key=lambda o: o.name)


def ray(scene, dg, origin, direction, dist):
    hit, loc, nrm, _idx, obj, _m = scene.ray_cast(dg, origin, direction, distance=dist)
    if not hit:
        return None
    return (origin - loc).length, nrm, obj


def eye_ok(scene, dg, p, small_obj):
    """Clearance test with 36 horizontal rays. Fails if any ray hits closer than
    CLEAR_MIN (against a wall/object) or the median hit distance is below
    ENCLOSED_MED (eye enclosed in a small solid, e.g. inside a cabinet).
    Also fails if >= 90 % of the rays hit one and the same object whose
    footprint is small (small_obj(name) True), i.e. the eye sits inside a
    closed prop such as a dome or tank (not a whole-room shell).
    (A back-face test was tried first and dropped: models' walls often have
    flipped or one-sided normals, giving false 'inside' verdicts.)"""
    dists = []
    hits = {}
    n = 36
    for k in range(n):
        a = 2 * math.pi * k / n
        d = Vector((math.cos(a), math.sin(a), 0.0))
        r = ray(scene, dg, p, d, 1e4)
        dist = 1e4 if r is None else r[0]
        if r is not None:
            hits[r[2].name] = hits.get(r[2].name, 0) + 1
        if dist < CLEAR_MIN:
            return False, "near_wall"
        dists.append(dist)
    dists.sort()
    if dists[n // 2] < ENCLOSED_MED:
        return False, "enclosed"
    if hits:
        name, cnt = max(hits.items(), key=lambda kv: (kv[1], kv[0]))
        if cnt >= 0.9 * n and small_obj(name):
            return False, "inside_object"
    return True, "ok"


def photo_cameras(scene):
    cams = sorted([o for o in scene.objects if o.type == "CAMERA"], key=lambda o: o.name)
    first = scene.camera if scene.camera is not None and scene.camera.type == "CAMERA" else (cams[0] if cams else None)
    return cams, first


def heading_of(obj):
    d = obj.matrix_world.to_3x3() @ Vector((0, 0, -1))
    if abs(d.x) + abs(d.y) < 1e-6:
        return 0.0
    return math.atan2(d.y, d.x)


def look_rot(cam_pos, target):
    d = (target - cam_pos).normalized()
    return d.to_track_quat("-Z", "Y").to_euler()


def probe_frame(scene, dg, cam_obj):
    """9x5 grid of rays through the image: fraction hitting nothing (void) and
    fraction hitting closer than NEAR_DIST (blocked view)."""
    mw = cam_obj.matrix_world
    o = mw.translation.copy()
    tx = math.tan(math.radians(FOV_H_DEG) / 2)
    ty = tx * RES_Y / RES_X
    void = near = 0
    tot = 0
    for iy in range(5):
        for ix in range(9):
            u = (ix / 8.0 * 2 - 1) * tx
            v = (iy / 4.0 * 2 - 1) * ty
            d = (mw.to_3x3() @ Vector((u, v, -1.0))).normalized()
            r = ray(scene, dg, o, d, 1e4)
            tot += 1
            if r is None:
                void += 1
            elif r[0] < NEAR_DIST:
                near += 1
    return void / tot, near / tot


def main():
    args = parse_args()
    t0 = time.time()
    out = os.path.abspath(args["out"])
    os.makedirs(out, exist_ok=True)
    fdir = os.path.join(out, "_frames")
    shutil.rmtree(fdir, ignore_errors=True)
    os.makedirs(fdir)

    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    vl = bpy.context.view_layer
    dg = bpy.context.evaluated_depsgraph_get()

    meshes = visible_meshes(scene, vl)
    boxes = {o.name: world_bbox(o, dg) for o in meshes}
    huge = [n for n, (lo, hi) in boxes.items() if max(hi - lo) > HUGE_DIM]
    use = [n for n in boxes if n not in huge] or list(boxes)
    if not use:
        raise SystemExit("NO_MESHES")
    lo = Vector((min(boxes[n][0].x for n in use), min(boxes[n][0].y for n in use), min(boxes[n][0].z for n in use)))
    hi = Vector((max(boxes[n][1].x for n in use), max(boxes[n][1].y for n in use), max(boxes[n][1].z for n in use)))
    ext = hi - lo
    scene_area = max(ext.x * ext.y, 1e-6)

    cams, first_cam = photo_cameras(scene)
    cam_mean = None
    if cams:
        cam_mean = sum((c.matrix_world.translation for c in cams), Vector()) / len(cams)

    # floor: large thin horizontal slabs (huge ground planes included) with top
    # <= bbox min z + 2 m. With photo cameras: the highest such slab whose top is
    # >= 0.5 m below the lowest photo camera (so a pool basin / pit is not taken
    # as the floor); otherwise the lowest. Fallback: bbox min z.
    floor_z, floor_obj = lo.z, None
    cands = []
    for n, (blo, bhi) in boxes.items():
        d = bhi - blo
        if d.z < 0.3 and (d.x * d.y >= 0.2 * scene_area or n in huge) and bhi.z <= lo.z + 2.0:
            cands.append((round(bhi.z, 6), n))
    cands.sort()
    if cands:
        floor_z, floor_obj = cands[0]
        if cams:
            cam_min_z = min(c.matrix_world.translation.z for c in cams)
            below = [cd for cd in cands if cd[0] <= cam_min_z - 0.5]
            if below:
                floor_z, floor_obj = below[-1]

    def small_obj(name):
        if name not in boxes:
            return False
        blo, bhi = boxes[name]
        return (bhi.x - blo.x) * (bhi.y - blo.y) < 0.5 * scene_area

    # ---- part 1 eye position
    center_xy = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2))
    eye_z = floor_z + EYE_H
    tried = []
    eye = None
    targets = [center_xy]
    if cam_mean is not None:
        cm = Vector((cam_mean.x, cam_mean.y))
        targets = [center_xy.lerp(cm, f / 10.0) for f in range(0, 11)]
        targets += [Vector((c.matrix_world.translation.x, c.matrix_world.translation.y)) for c in cams]
    for t in targets:
        p = Vector((t.x, t.y, eye_z))
        ok, why = eye_ok(scene, dg, p, small_obj)
        tried.append([round(p.x, 3), round(p.y, 3), why])
        if ok:
            eye = p
            break
    eye_status = "ok" if eye is not None else "no_clear_spot"
    if eye is None:
        eye = Vector((center_xy.x, center_xy.y, eye_z))
    yaw0 = heading_of(first_cam) if first_cam is not None else math.pi / 2

    # ---- render settings (in-memory only; the .blend is never saved)
    r = scene.render
    r.engine = "BLENDER_WORKBENCH"
    r.resolution_x, r.resolution_y, r.resolution_percentage = RES_X, RES_Y, 100
    r.fps, r.fps_base = FPS, 1.0
    r.film_transparent = True
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA"
    r.use_stamp = False
    try:
        r.threads_mode = "FIXED"; r.threads = args["threads"]
    except Exception:
        pass
    try:
        scene.display_settings.display_device = "sRGB"
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = "SINGLE"
    sh.single_color = (0.8, 0.8, 0.8)
    sh.show_cavity = True
    sh.cavity_type = "BOTH"
    sh.show_object_outline = True
    sh.object_outline_color = (0.0, 0.0, 0.0)
    sh.show_backface_culling = False
    sh.show_xray = False
    sh.show_shadows = False
    sh.show_specular_highlight = True
    scene.display.render_aa = "5"

    cam_data = bpy.data.cameras.new("orbit_v1_cam")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.angle = math.radians(FOV_H_DEG)
    cam_data.clip_start = 0.05
    cam_data.clip_end = 10000.0
    cam = bpy.data.objects.new("orbit_v1_cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    # never let camera markers switch the camera mid-render
    scene.timeline_markers.clear()

    def render_to(path):
        r.filepath = path
        scene.camera = cam
        bpy.ops.render.render(write_still=True)

    metrics = {"p1": [], "p2": []}
    # part 1: interior pan
    for i in range(PART_FRAMES):
        yaw = yaw0 + 2 * math.pi * i / PART_FRAMES
        cam.location = eye
        cam.rotation_euler = (math.pi / 2, 0.0, yaw - math.pi / 2)
        bpy.context.view_layer.update()
        if i % 8 == 0:
            metrics["p1"].append(probe_frame(scene, bpy.context.evaluated_depsgraph_get(), cam))
        render_to(os.path.join(fdir, "f_%04d.png" % i))

    # part 2: dollhouse orbit; hide high objects and ceilings/roofs
    hidden = []
    for o in meshes:
        blo, bhi = boxes[o.name]
        d = bhi - blo
        high = blo.z > floor_z + DOLL_CUT_H
        ceil_like = (d.z < 0.4 and d.x * d.y >= 0.15 * scene_area and blo.z > floor_z + 1.8)
        if high or ceil_like:
            o.hide_render = True
            hidden.append(o.name)
    bpy.context.view_layer.update()
    vis_top = max([boxes[o.name][1].z for o in meshes if o.name not in hidden and o.name in use] or [floor_z + DOLL_CUT_H])
    look_z = (floor_z + min(vis_top, floor_z + DOLL_CUT_H)) / 2
    c = Vector((center_xy.x, center_xy.y, look_z))
    half_diag = 0.5 * math.hypot(ext.x, ext.y)
    radius = 1.1 * half_diag + DOLL_MARGIN          # orbit radius (see README)
    elev = math.radians(DOLL_ELEV_DEG)
    dist = DOLL_DIST_K * radius                     # 3D camera-to-center distance
    height = dist * math.sin(elev)
    radius_xy = dist * math.cos(elev)
    dg2 = bpy.context.evaluated_depsgraph_get()
    for i in range(PART_FRAMES):
        th = yaw0 + math.pi + 2 * math.pi * i / PART_FRAMES   # start behind the first photo camera
        p = c + Vector((radius_xy * math.cos(th), radius_xy * math.sin(th), height))
        cam.location = p
        cam.rotation_euler = look_rot(p, c)
        bpy.context.view_layer.update()
        if i % 8 == 0:
            metrics["p2"].append(probe_frame(scene, bpy.context.evaluated_depsgraph_get(), cam))
        render_to(os.path.join(fdir, "f_%04d.png" % (PART_FRAMES + i)))

    # pixel metrics from alpha (background = alpha < 0.5)
    def bg_frac(idx):
        img = bpy.data.images.load(os.path.join(fdir, "f_%04d.png" % idx))
        px = np.empty(len(img.pixels), dtype=np.float32)
        img.pixels.foreach_get(px)
        bpy.data.images.remove(img)
        return float((px[3::4] < 0.5).mean())
    p1_bg = [bg_frac(i) for i in range(0, PART_FRAMES, 8)]
    p2_bg = [bg_frac(PART_FRAMES + i) for i in range(0, PART_FRAMES, 8)]

    # encode
    ff = args["ffmpeg"]
    enc = None
    if shutil.which(ff) or os.path.exists(ff):
        vf = "[0][1]overlay=shortest=1,format=yuv420p"
        cmd = [ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
               "color=c=%s:s=%dx%d:r=%d" % (BG_GRAY, RES_X, RES_Y, FPS),
               "-framerate", str(FPS), "-i", os.path.join(fdir, "f_%04d.png"),
               "-filter_complex", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "28",
               "-movflags", "+faststart", "-an", "-threads", "1", os.path.join(out, "orbit.mp4")]
        enc = subprocess.run(cmd, capture_output=True, text=True)
        sheet_idx = [0, PART_FRAMES // 2, PART_FRAMES, PART_FRAMES + PART_FRAMES // 2]
        cmd2 = [ff, "-y", "-loglevel", "error"]
        for k in sheet_idx:
            cmd2 += ["-i", os.path.join(fdir, "f_%04d.png" % k)]
        cmd2 += ["-f", "lavfi", "-i", "color=c=%s:s=%dx%d" % (BG_GRAY, 2 * RES_X, 2 * RES_Y),
                 "-filter_complex", "[0][1]hstack[a];[2][3]hstack[b];[a][b]vstack[t];[4][t]overlay",
                 "-frames:v", "1", "-q:v", "4", os.path.join(out, "sheet.jpg")]
        enc2 = subprocess.run(cmd2, capture_output=True, text=True)
        if enc.returncode or enc2.returncode:
            raise SystemExit("FFMPEG_FAIL " + enc.stderr[-500:] + enc2.stderr[-500:])
    if not args["keep_frames"] and enc is not None:
        shutil.rmtree(fdir, ignore_errors=True)

    rnd = lambda v: [round(float(x), 4) for x in v]
    params = {
        "protocol": PROTOCOL,
        "blend": bpy.data.filepath,
        "blender": bpy.app.version_string,
        "frame_evaluated": scene.frame_start,
        "res": [RES_X, RES_Y], "fps": FPS, "frames_per_part": PART_FRAMES,
        "fov_h_deg": FOV_H_DEG,
        "n_meshes": len(meshes), "huge_ignored_for_bounds": huge,
        "bbox_min": rnd(lo), "bbox_max": rnd(hi),
        "floor_z": round(floor_z, 4), "floor_obj": floor_obj,
        "photo_cameras": [c_.name for c_ in cams],
        "photo_cam_mean": rnd(cam_mean) if cam_mean is not None else None,
        "part1": {"eye": rnd(eye), "eye_status": eye_status, "tried": tried,
                  "yaw0_deg": round(math.degrees(yaw0), 3),
                  "yaw_from": first_cam.name if first_cam is not None else None},
        "part2": {"center": rnd(c), "radius": round(radius, 4), "dist_3d": round(dist, 4), "radius_xy": round(radius_xy, 4), "height": round(height, 4),
                  "elev_deg": DOLL_ELEV_DEG, "hidden": hidden},
        "metrics": {
            "p1_bg_frac_mean": round(float(np.mean(p1_bg)), 4),
            "p1_ray_void_mean": round(float(np.mean([m[0] for m in metrics["p1"]])), 4),
            "p1_ray_near_mean": round(float(np.mean([m[1] for m in metrics["p1"]])), 4),
            "p2_bg_frac_mean": round(float(np.mean(p2_bg)), 4),
            "p2_ray_void_mean": round(float(np.mean([m[0] for m in metrics["p2"]])), 4),
            "p2_ray_near_mean": round(float(np.mean([m[1] for m in metrics["p2"]])), 4),
        },
        "render_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out, "params.json"), "w") as f:
        json.dump(params, f, indent=1)
    print("ORBIT_DONE", json.dumps(params["metrics"]))


try:
    main()
except SystemExit as e:
    print("ORBIT_FAIL", e)
    sys.stdout.flush()
    os._exit(3)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("ORBIT_FAIL", repr(e))
    sys.stdout.flush()
    os._exit(4)
sys.stdout.flush()
os._exit(0)
