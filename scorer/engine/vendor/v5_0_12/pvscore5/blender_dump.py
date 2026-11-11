"""pvscore5 Blender-side dump (runs inside Blender 5.2.x).

    blender -b scene.blend --factory-startup --python blender_dump.py -- --out DIR --pvtools PATH
            [--analysis-hz 8] [--grid-hz 4] [--grid-w 96] [--ref-cams cams.json] [--no-grid]

Reads the saved scene exactly as a fresh process sees it (no auto-run scripts) and writes
  DIR/dump.json  static tables: meta, parts (one row per renderable object, tags resolved with the same pvtools.py
                 the model used), cameras, markers, sample-frame lists, warnings
  DIR/dump.npz   arrays:
                 cam_frame[F] cam_index[F] cam_M[F,4,4] cam_corners[F,4,3] cam_ortho[F]   (every frame)
                 an_frame[A] obj_M[A,N,4,4] obj_vis[A,N]                                   (analysis frames)
                 pts_local[P,3] pts_part[P] pts_tag[P] pts_kind[P]       (rigid parts, object-local samples)
                 def_pts[A,Q,3] def_part[Q] def_tag[Q] def_kind[Q]        (deforming parts, world samples per frame)
                 g_frame[G] g_M[G,4,4] g_corners[G,4,3] g_ortho[G] g_part[G,H,W] g_tag[G,H,W]
                 g_depth[G,H,W] g_normal[G,H,W,3]                                          (ray-cast grids)
                 r_* the same grids for --ref-cams (reference cameras given as c2w + pinhole K)
pts_kind: 0 = bound-box corner, 1 = vertex, 2 = area-weighted surface sample.  g_part = -1 for no hit.
The scene on disk is never written.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

DUMP_VERSION = "pvdump-5.0"
DEFORM_MODS = {"ARMATURE", "LATTICE", "MESH_DEFORM", "HOOK", "CLOTH", "SOFT_BODY", "SURFACE_DEFORM", "WAVE",
               "NODES", "MESH_CACHE", "MESH_SEQUENCE_CACHE", "OCEAN", "DYNAMIC_PAINT", "EXPLODE", "FLUID",
               "PARTICLE_INSTANCE", "CAST", "SIMPLE_DEFORM", "SHRINKWRAP", "DISPLACE", "SMOOTH", "WARP"}


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--pvtools", required=True)
    p.add_argument("--analysis-hz", type=float, default=8.0)
    p.add_argument("--grid-hz", type=float, default=4.0)
    p.add_argument("--grid-w", type=int, default=96)
    p.add_argument("--samples", type=int, default=192)
    p.add_argument("--max-verts", type=int, default=256)
    p.add_argument("--ref-cams", default=None)
    p.add_argument("--no-grid", action="store_true")
    return p.parse_args(argv)


A = parse()
OUT = Path(A.out)
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.monotonic()
timings, warnings = {}, []
spec = importlib.util.spec_from_file_location("pvtools", A.pvtools)
pv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pv)
sys.modules["pvtools"] = pv
scene = bpy.context.scene
view_layer = bpy.context.view_layer
rng = np.random.default_rng(12345)


def tick(name, t):
    timings[name] = round(time.monotonic() - t, 3)


# ------------------------------------------------------------------------------------------------ preparation
def make_instances_real():
    inst = [o for o in scene.objects if o.instance_type == "COLLECTION" and o.instance_collection is not None]
    if not inst:
        return 0
    try:
        with bpy.context.temp_override(selected_objects=inst, selected_editable_objects=inst, active_object=inst[0],
                                       object=inst[0], view_layer=view_layer, scene=scene):
            bpy.ops.object.duplicates_make_real(use_base_parent=True, use_hierarchy=True)
    except Exception as exc:  # reported, never fatal
        warnings.append("duplicates_make_real failed: %r" % (exc,))
    return len(inst)


def render_enabled_collections():
    ok = {scene.collection.name}

    def walk(lc, parent_ok):
        for ch in lc.children:
            c = ch.collection
            e = parent_ok and not ch.exclude and not c.hide_render
            if e:
                ok.add(c.name)
            walk(ch, e)
    walk(view_layer.layer_collection, True)
    return ok


def unhide_for_raycast():
    """Make every render-enabled object reachable by scene.ray_cast (viewport visibility never hides geometry
    that the renderer would show).  Objects hidden for render are skipped per ray instead."""
    def walk(lc):
        for ch in lc.children:
            if not ch.exclude and not ch.collection.hide_render:
                ch.hide_viewport = False
                ch.collection.hide_viewport = False
            walk(ch)
    walk(view_layer.layer_collection)
    for o in scene.objects:
        if o.name in view_layer.objects:
            try:
                o.hide_viewport = False
                o.hide_set(False)
            except Exception:
                pass


def fcurve_keys(idb):
    """(keyframe count, has drivers) of an ID's animation (layered actions in Blender 5)."""
    ad = getattr(idb, "animation_data", None)
    if ad is None:
        return 0, False, []
    n, paths = 0, set()
    act = ad.action
    if act is not None:
        try:
            for layer in act.layers:
                for strip in layer.strips:
                    for cb in strip.channelbags:
                        for fc in cb.fcurves:
                            n += len(fc.keyframe_points)
                            paths.add(fc.data_path)
        except Exception:
            try:
                for fc in act.fcurves:
                    n += len(fc.keyframe_points)
                    paths.add(fc.data_path)
            except Exception:
                pass
    return n, bool(len(ad.drivers)), sorted(paths)


t = time.monotonic()
n_instancers = make_instances_real()
unhide_for_raycast()
fps = scene.render.fps / (scene.render.fps_base or 1.0)
f0, f1 = scene.frame_start, scene.frame_end
frames = list(range(f0, f1 + 1))
unit = scene.unit_settings.scale_length or 1.0
scene.frame_set(f0)
dg = bpy.context.evaluated_depsgraph_get()
n_gn_instances = sum(1 for i in dg.object_instances if i.is_instance)
if n_gn_instances:
    warnings.append("%d evaluated instances (geometry nodes / particles) are not resolved as parts" % n_gn_instances)
parts = [o for o in scene.objects if o.type in pv.RENDERABLE and o.name in view_layer.objects]
part_index = {o.name: i for i, o in enumerate(parts)}
rows = {r["object"]: r for r in pv.resolve_all(scene)}
cparents = pv._collection_parents(scene)
tick("prepare", t)


# ------------------------------------------------------------------------------------------- static geometry
def holder_of(source, obj):
    if source is None:
        return None
    if source == "own":
        return "obj:" + obj.name
    if source.startswith("parent:"):
        return "obj:" + source[len("parent:"):]
    if source.startswith("collection:"):
        return "col:" + source[len("collection:"):].split(" (ambiguous)")[0]
    return source


def is_deforming(o):
    if any(m.type in DEFORM_MODS for m in getattr(o, "modifiers", [])):
        return True
    data = getattr(o, "data", None)
    if data is not None and getattr(data, "shape_keys", None) is not None:
        return True
    if data is not None and getattr(data, "animation_data", None) is not None:
        return True
    return False


def geometry(o, depsgraph, world=False):
    """(points [n,3], tag [n], kind [n], stats) in object-local (or world) coordinates."""
    ev = o.evaluated_get(depsgraph)
    corners = np.array([list(c) for c in ev.bound_box], np.float64)
    pts, tags, kinds = [corners], [np.zeros(8, np.int32)], [np.zeros(8, np.int8)]
    stats = {"n_verts": 0, "n_tris": 0, "area": 0.0, "tag_area": {}}
    me = None
    try:
        me = ev.to_mesh()
    except Exception:
        me = None
    if me is not None and len(me.vertices):
        nv = len(me.vertices)
        co = np.empty(nv * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3).astype(np.float64)
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        ptag = None
        attr = me.attributes.get("pv_tag")
        if attr is not None and attr.domain == "FACE" and len(attr.data) == len(me.polygons):
            ptag = np.zeros(len(me.polygons), np.int32)
            attr.data.foreach_get("value", ptag)
        vtag = np.zeros(nv, np.int32)
        stats.update(n_verts=nv, n_tris=nt)
        if nt:
            tri = np.empty(nt * 3, np.int32)
            me.loop_triangles.foreach_get("vertices", tri)
            tri = tri.reshape(-1, 3)
            poly = np.empty(nt, np.int32)
            me.loop_triangles.foreach_get("polygon_index", poly)
            ttag = ptag[poly] if ptag is not None else np.zeros(nt, np.int32)
            for k in range(3):
                vtag[tri[:, k]] = ttag
            a, b, c = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
            area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
            tot = float(area.sum())
            stats["area"] = tot
            for k in np.unique(ttag):
                stats["tag_area"][int(k)] = float(area[ttag == k].sum())
            if tot > 0:
                ns = A.samples
                pick = rng.choice(nt, ns, p=area / tot)
                r1, r2 = rng.random(ns), rng.random(ns)
                s1 = np.sqrt(r1)
                w0, w1, w2 = 1 - s1, s1 * (1 - r2), s1 * r2
                sp = a[pick] * w0[:, None] + b[pick] * w1[:, None] + c[pick] * w2[:, None]
                pts.append(sp)
                tags.append(ttag[pick])
                kinds.append(np.full(ns, 2, np.int8))
        if nv > A.max_verts:
            ext = {int(np.argmin(co[:, j])) for j in range(3)} | {int(np.argmax(co[:, j])) for j in range(3)}
            rest = rng.choice(nv, A.max_verts - len(ext), replace=False)
            idx = np.array(sorted(ext | set(int(i) for i in rest)))
        else:
            idx = np.arange(nv)
        pts.append(co[idx])
        tags.append(vtag[idx])
        kinds.append(np.ones(len(idx), np.int8))
    if me is not None:
        ev.to_mesh_clear()
    P = np.concatenate(pts)
    if world:
        M = np.array(ev.matrix_world, np.float64)
        P = P @ M[:3, :3].T + M[:3, 3]
    return P, np.concatenate(tags), np.concatenate(kinds), stats


t = time.monotonic()
part_rows, rigid = [], []
pl, pp, pt, pk = [], [], [], []
deforming = []
for i, o in enumerate(parts):
    r = rows.get(o.name, {})
    lab_src = r.get("source")
    vals, table = (pv.face_tags(o.data) if o.type == "MESH" else (None, []))
    nkeys, drivers, paths = fcurve_keys(o)
    deform = is_deforming(o)
    row = {"index": i, "name": o.name, "type": o.type, "data": o.data.name if o.data else None,
           "class": r.get("class"), "id": r.get("id"), "source": lab_src, "holder": holder_of(lab_src, o),
           "space": r.get("space"), "space_source": r.get("space_source"),
           "face_table": table if vals is not None else [], "deforming": deform,
           "parent": o.parent.name if o.parent else None,
           "collections": [c.name for c in o.users_collection],
           "display_type": o.display_type, "keys": nkeys, "drivers": drivers, "anim_paths": paths,
           "constraints": [c.type for c in o.constraints], "modifiers": [m.type for m in getattr(o, "modifiers", [])],
           "hide_render_static": bool(o.hide_render), "materials": len(getattr(o.data, "materials", []) or [])}
    P, tg, kd, st = geometry(o, dg)
    row.update(st)
    row["tag_area"] = {str(k): v for k, v in st["tag_area"].items()}
    part_rows.append(row)
    if deform:
        deforming.append(i)
    else:
        pl.append(P.astype(np.float32))
        pp.append(np.full(len(P), i, np.int32))
        pt.append(tg.astype(np.int16))
        pk.append(kd)
tick("geometry", t)


# ------------------------------------------------------------------------------------------------- cameras
def camera_state(cam):
    M = np.array(cam.matrix_world, np.float64)
    vf = cam.data.view_frame(scene=scene)
    corners = np.array([[v.x, v.y, v.z] for v in vf], np.float64)
    return M, corners, cam.data.type == "ORTHO"


markers = sorted([(m.frame, m.name, m.camera.name if m.camera else None) for m in scene.timeline_markers])
bound = [(f, c) for f, _, c in markers if c]


def marker_camera(f):
    best = None
    for mf, c in bound:
        if mf <= f:
            best = c
    return best


t = time.monotonic()
cam_names, cam_index, cam_M, cam_corners, cam_ortho, cam_marker = [], [], [], [], [], []
lens_rows = []
for f in frames:
    scene.frame_set(f)
    cam = scene.camera
    if cam is None or cam.type != "CAMERA":
        cam_index.append(-1)
        cam_M.append(np.eye(4))
        cam_corners.append(np.zeros((4, 3)))
        cam_ortho.append(False)
        cam_marker.append(marker_camera(f))
        continue
    if cam.name not in cam_names:
        cam_names.append(cam.name)
    cam_index.append(cam_names.index(cam.name))
    M, C, ortho = camera_state(cam)
    cam_M.append(M)
    cam_corners.append(C)
    cam_ortho.append(ortho)
    cam_marker.append(marker_camera(f))
    lens_rows.append(cam.data.lens)
cam_index = np.array(cam_index, np.int32)
cam_M = np.array(cam_M, np.float64)
cam_corners = np.array(cam_corners, np.float64)
tick("cameras", t)


def boundaries():
    """Candidate shot boundaries (frame index where a new shot starts) for extra grid samples."""
    out = set()
    for k in range(1, len(frames)):
        if cam_index[k] != cam_index[k - 1]:
            out.add(k)
            continue
        d = np.linalg.norm(cam_M[k, :3, 3] - cam_M[k - 1, :3, 3])
        R = cam_M[k, :3, :3] @ cam_M[k - 1, :3, :3].T
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
        if d > 0.3 or ang > 12:
            out.add(k)
    return sorted(out)


def every(hz):
    step = max(1, int(round(fps / hz)))
    ks = list(range(0, len(frames), step))
    if ks[-1] != len(frames) - 1:
        ks.append(len(frames) - 1)
    return ks


an_k = every(A.analysis_hz)
grid_k = [] if A.no_grid else every(A.grid_hz)
if not A.no_grid:
    cuts = boundaries()
    starts = [0] + cuts
    ends = [c - 1 for c in cuts] + [len(frames) - 1]
    extra = set()
    for a, b in zip(starts, ends):
        extra.update({a, b, (a + b) // 2, min(b, a + max(1, (b - a) // 10)), max(a, b - max(1, (b - a) // 10))})
    grid_k = sorted(set(grid_k) | extra)
an_k = sorted(set(an_k) | set(grid_k))


# ------------------------------------------------------------------------------------------------ ray grids
def render_visible_names():
    enabled = render_enabled_collections()
    out = set()
    for o in parts:
        if o.hide_render:
            continue
        if any(c.name in enabled for c in o.users_collection):
            out.add(o.name)
    return out


facetag_cache = {}


def poly_tags(obj_eval, name):
    me = obj_eval.data
    n = len(getattr(me, "polygons", []))
    hit = facetag_cache.get(name)
    if hit is not None and len(hit) == n:
        return hit
    arr = None
    attr = me.attributes.get("pv_tag") if hasattr(me, "attributes") else None
    if attr is not None and attr.domain == "FACE" and len(attr.data) == n:
        arr = np.zeros(n, np.int32)
        attr.data.foreach_get("value", arr)
    facetag_cache[name] = arr if arr is not None else np.zeros(0, np.int32)
    return facetag_cache[name]


def cast_rays(origins, dirs, depsgraph, vis, fwd, cam_o, clip_end):
    n = len(dirs)
    part = np.full(n, -1, np.int32)
    tagv = np.zeros(n, np.int16)
    depth = np.zeros(n, np.float32)
    normal = np.zeros((n, 3), np.float16)
    for i in range(n):
        o = Vector(origins[i])
        d = Vector(dirs[i])
        remain = clip_end
        for _ in range(8):
            ok, loc, nor, idx, obj, _m = scene.ray_cast(depsgraph, o, d, distance=remain)
            if not ok:
                break
            name = obj.original.name if obj is not None else None
            if name in vis:
                pi = part_index.get(name, -2)
                part[i] = pi
                if pi >= 0:
                    tg = poly_tags(obj, name)
                    if 0 <= idx < len(tg):
                        tagv[i] = tg[idx]
                depth[i] = float(np.dot(np.array(loc) - cam_o, fwd))
                nn = np.array(nor)
                if float(np.dot(nn, dirs[i])) > 0:
                    nn = -nn
                normal[i] = nn
                break
            step = (loc - o).length + 1e-4
            remain -= step
            if remain <= 0:
                break
            o = loc + d * 1e-4
    return part, tagv, depth, normal


def grid_rays_blender(M, corners, ortho, W, H):
    xs, ys, z = corners[:, 0], corners[:, 1], corners[0, 2]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    u = x0 + (np.arange(W) + 0.5) / W * (x1 - x0)
    v = y1 - (np.arange(H) + 0.5) / H * (y1 - y0)
    U, V = np.meshgrid(u, v)
    R, T = M[:3, :3], M[:3, 3]
    if ortho:
        loc = np.stack([U, V, np.zeros_like(U)], -1).reshape(-1, 3)
        origins = loc @ R.T + T
        d = R @ np.array([0, 0, -1.0])
        dirs = np.repeat((d / np.linalg.norm(d))[None], len(origins), 0)
    else:
        loc = np.stack([U, V, np.full_like(U, z)], -1).reshape(-1, 3)
        dirs = loc @ R.T
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        origins = np.repeat(T[None], len(dirs), 0)
    fwd = R @ np.array([0, 0, -1.0])
    return origins, dirs, fwd / np.linalg.norm(fwd), T


def grid_rays_pinhole(c2w_blender, K, Wimg, Himg, W, H):
    M = np.array(c2w_blender, np.float64)
    i = (np.arange(W) + 0.5) / W * Wimg
    j = (np.arange(H) + 0.5) / H * Himg
    I, J = np.meshgrid(i, j)
    x = (I - K["cx"]) / K["fx"]
    y = (J - K["cy"]) / K["fy"]
    loc = np.stack([x, -y, -np.ones_like(x)], -1).reshape(-1, 3)   # OpenCV -> Blender camera axes
    R, T = M[:3, :3], M[:3, 3]
    dirs = loc @ R.T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    fwd = R @ np.array([0, 0, -1.0])
    return np.repeat(T[None], len(dirs), 0), dirs, fwd / np.linalg.norm(fwd), T


res_x = scene.render.resolution_x * scene.render.resolution_percentage / 100.0
res_y = scene.render.resolution_y * scene.render.resolution_percentage / 100.0
aspect = (res_x * scene.render.pixel_aspect_x) / max(1e-6, res_y * scene.render.pixel_aspect_y)
GW = int(A.grid_w)
GH = max(1, int(round(GW / aspect)))


# ------------------------------------------------------------------------------------- analysis + grid pass
t = time.monotonic()
N = len(parts)
obj_M = np.zeros((len(an_k), N, 4, 4), np.float32)
obj_vis = np.zeros((len(an_k), N), bool)
def_lists = {i: [] for i in deforming}
def_tk = {i: [] for i in deforming}                   # r4 R4-5 (PREREG_51r4_addendum3.json): every frame's tags / kinds (memory only)
def_meta = {}
grid_set = set(grid_k)
g = {"frame": [], "M": [], "corners": [], "ortho": [], "part": [], "tag": [], "depth": [], "normal": []}
t_grid = 0.0
for ai, k in enumerate(an_k):
    f = frames[k]
    scene.frame_set(f)
    dgf = bpy.context.evaluated_depsgraph_get()
    vis = render_visible_names()
    for i, o in enumerate(parts):
        ev = o.evaluated_get(dgf)
        obj_M[ai, i] = np.array(ev.matrix_world, np.float32)
        obj_vis[ai, i] = o.name in vis
    for i in deforming:
        P, tg, kd, _ = geometry(parts[i], dgf, world=True)
        def_lists[i].append(P.astype(np.float32))
        def_meta[i] = (tg, kd)
        def_tk[i].append((tg, kd))                    # r4 R4-5 (PREREG_51r4_addendum3.json)
    if k in grid_set and cam_index[k] >= 0:
        tg0 = time.monotonic()
        M, C, ortho = cam_M[k], cam_corners[k], bool(cam_ortho[k])
        origins, dirs, fwd, T = grid_rays_blender(M, C, ortho, GW, GH)
        clip = scene.camera.data.clip_end if scene.camera else 1e4
        part, tagv, depth, normal = cast_rays(origins, dirs, dgf, vis, fwd, T, clip)
        for key, val in (("frame", f), ("M", M), ("corners", C), ("ortho", ortho), ("part", part.reshape(GH, GW)),
                         ("tag", tagv.reshape(GH, GW)), ("depth", depth.reshape(GH, GW)),
                         ("normal", normal.reshape(GH, GW, 3))):
            g[key].append(val)
        t_grid += time.monotonic() - tg0
tick("analysis", t)
timings["grid_raycast"] = round(t_grid, 3)

# reference cameras (validation and reference rendering): pinhole c2w + K at a given time
refs = {"name": [], "frame": [], "part": [], "tag": [], "depth": [], "normal": [], "M": [], "K": [], "size": []}
ref_meta = []
if A.ref_cams:
    t = time.monotonic()
    rc = json.loads(Path(A.ref_cams).read_text())
    for c in rc["cameras"]:
        f = int(round(c.get("t", 0.0) * fps)) + f0
        f = max(f0, min(f1, f))
        scene.frame_set(f)
        dgf = bpy.context.evaluated_depsgraph_get()
        vis = render_visible_names()
        Wimg, Himg = c["width"], c["height"]
        W = int(c.get("grid_w", GW))
        H = max(1, int(round(W * Himg / Wimg)))
        origins, dirs, fwd, T = grid_rays_pinhole(c["c2w_blender"], c["K"], Wimg, Himg, W, H)
        part, tagv, depth, normal = cast_rays(origins, dirs, dgf, vis, fwd, T, 1e4)
        refs["part"].append(part.reshape(H, W))
        refs["tag"].append(tagv.reshape(H, W))
        refs["depth"].append(depth.reshape(H, W))
        refs["normal"].append(normal.reshape(H, W, 3))
        refs["M"].append(np.array(c["c2w_blender"], np.float64))
        refs["K"].append([c["K"]["fx"] * W / Wimg, c["K"]["fy"] * H / Himg, c["K"]["cx"] * W / Wimg,
                          c["K"]["cy"] * H / Himg])
        refs["frame"].append(f)
        ref_meta.append({"name": c["name"], "t": c.get("t", 0.0), "frame": f, "grid": [W, H]})
    tick("ref_cams", t)

# ---------------------------------------------------------------------------------------------- write out
arrays = {"cam_frame": np.array(frames, np.int32), "cam_index": cam_index, "cam_M": cam_M.astype(np.float32),
          "cam_corners": cam_corners.astype(np.float32), "cam_ortho": np.array(cam_ortho, bool),
          "an_frame": np.array([frames[k] for k in an_k], np.int32), "obj_M": obj_M, "obj_vis": obj_vis}
if pl:
    arrays.update(pts_local=np.concatenate(pl), pts_part=np.concatenate(pp), pts_tag=np.concatenate(pt),
                  pts_kind=np.concatenate(pk))
else:
    arrays.update(pts_local=np.zeros((0, 3), np.float32), pts_part=np.zeros(0, np.int32),
                  pts_tag=np.zeros(0, np.int16), pts_kind=np.zeros(0, np.int8))
def_var = [i for i in deforming if len({len(x) for x in def_lists[i]}) > 1]   # r4 R4-5 (PREREG_51r4_addendum3.json)
if def_var:
    # a deforming part whose per-frame sample count varies (topology-changing modifier, geometry nodes ...):
    # ragged per-frame arrays, each frame's own points / tags / kinds; frame ai = rows defv_off[ai]:defv_off[ai+1]
    rv = [(def_lists[i][ai], def_tk[i][ai][0], def_tk[i][ai][1], i) for ai in range(len(an_k)) for i in def_var]
    arrays["defv_pts"] = np.concatenate([r[0] for r in rv]).astype(np.float32)
    arrays["defv_part"] = np.concatenate([np.full(len(r[0]), r[3], np.int32) for r in rv])
    arrays["defv_tag"] = np.concatenate([r[1].astype(np.int16) for r in rv])
    arrays["defv_kind"] = np.concatenate([r[2].astype(np.int8) for r in rv])
    arrays["defv_off"] = np.concatenate([[0], np.cumsum([sum(len(def_lists[i][ai]) for i in def_var)
                                                       for ai in range(len(an_k))])]).astype(np.int64)
    warnings.append("R4-5: variable topology on deforming parts %s (per-frame sample count [min, max])" % json.dumps(
        {parts[i].name: [min(len(x) for x in def_lists[i]), max(len(x) for x in def_lists[i])] for i in def_var}))
    deforming = [i for i in deforming if i not in def_var]
if deforming:
    n_by = {i: len(def_lists[i][0]) for i in deforming}
    arrays["def_pts"] = np.concatenate([np.stack(def_lists[i]) for i in deforming], axis=1)
    arrays["def_part"] = np.concatenate([np.full(n_by[i], i, np.int32) for i in deforming])
    arrays["def_tag"] = np.concatenate([def_meta[i][0].astype(np.int16) for i in deforming])
    arrays["def_kind"] = np.concatenate([def_meta[i][1] for i in deforming])
if g["frame"]:
    arrays.update(g_frame=np.array(g["frame"], np.int32), g_M=np.array(g["M"], np.float32),
                  g_corners=np.array(g["corners"], np.float32), g_ortho=np.array(g["ortho"], bool),
                  g_part=np.stack(g["part"]), g_tag=np.stack(g["tag"]), g_depth=np.stack(g["depth"]),
                  g_normal=np.stack(g["normal"]))
for i, m in enumerate(ref_meta):
    arrays["r%d_part" % i] = refs["part"][i]
    arrays["r%d_tag" % i] = refs["tag"][i]
    arrays["r%d_depth" % i] = refs["depth"][i]
    arrays["r%d_normal" % i] = refs["normal"][i]
    arrays["r%d_M" % i] = refs["M"][i].astype(np.float32)
    arrays["r%d_K" % i] = np.array(refs["K"][i], np.float32)
np.savez_compressed(OUT / "dump.npz", **arrays)
pv_src = Path(A.pvtools).read_bytes()
meta = {"version": DUMP_VERSION, "blender": bpy.app.version_string, "blend": bpy.data.filepath,
        "pvtools": getattr(pv, "VERSION", None), "pvtools_sha256": hashlib.sha256(pv_src).hexdigest(),
        "fps": fps, "frame_start": f0, "frame_end": f1, "resolution": [res_x, res_y], "aspect": aspect,
        "engine": scene.render.engine, "unit_scale": unit, "grid": [GW, GH], "analysis_hz": A.analysis_hz,
        "grid_hz": A.grid_hz, "n_parts": N, "n_instancers_realized": n_instancers,
        "camera_objects": [o.name for o in scene.objects if o.type == "CAMERA"],
        "scene_camera_start": cam_names[cam_index[0]] if len(cam_index) and cam_index[0] >= 0 else None}
if def_var:                                           # r4 R4-5 (PREREG_51r4_addendum3.json)
    meta["def_variable_topology"] = [parts[i].name for i in def_var]
marker_mismatch = sum(1 for k in range(len(frames)) if cam_marker[k] and cam_index[k] >= 0
                      and cam_names[cam_index[k]] != cam_marker[k])
if marker_mismatch:
    warnings.append("%d frames where scene.camera differs from the marker-bound camera" % marker_mismatch)
timings["total"] = round(time.monotonic() - T0, 3)
meta["timings_s"] = timings
doc = {"meta": meta, "parts": part_rows, "cameras": cam_names, "markers": markers, "warnings": warnings,
       "refs": ref_meta, "handlers": {"frame_change_pre": len(bpy.app.handlers.frame_change_pre),
                                      "frame_change_post": len(bpy.app.handlers.frame_change_post)}}
(OUT / "dump.json").write_text(json.dumps(doc, indent=1))
print("PVDUMP_DONE", json.dumps(timings), flush=True)
