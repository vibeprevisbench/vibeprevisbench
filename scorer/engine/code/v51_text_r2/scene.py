"""pvscore 5.1 text / image layer (r1 copy, tools/v51_text_r2): the GT scene as plain boxes + tracks + cameras, a box
ray-caster, transforms. r1: imports tools/v51_r2 (scene_geom, gt_root -> paths.GT_ROOT) instead of
the r0 tools; TH = thresholds.json (byte copy of the r0 file), TH1 = thresholds_v51r1.json (the r1 changes).

Used by gt_facts.py (GT facts: the GT scene seen from the GT cameras) and by score.py tests (GT-as-model: the same
scene, transformed or perturbed, read back as a model view). numpy only; GT parsing by pvscore5.gt (read-only import).
"""
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v51_r2"))           # the view layer (paths.py, scene_geom, gt_root)
import paths as PT  # noqa: E402
R = PT.ROOT
sys.path.insert(0, str(PT.PVSCORE_DIR))
from pvscore5.gt import GT, load_categories  # noqa: E402
from pvscore5.score import CATS  # noqa: E402
import scene_geom as SG  # noqa: E402  (cell rule shared with the video layer)
from gt_root import gt_json  # noqa: E402

TH = json.loads((PT.TEXT_CONF / "thresholds.json").read_text())
TH1 = json.loads((PT.TEXT_CONF / "thresholds_v51r1.json").read_text())
TH2 = json.loads((PT.TEXT_CONF / "thresholds_v51r2.json").read_text())       # r2 view-block rules
EL = TH["elements"]
VW = TH["view"]


def load_gt(item):
    cats = load_categories(CATS)
    return GT(gt_json(item), cats), cats


# --------------------------------------------------------------------------------------------- scene dict
def scene_from_gt(gt, item):
    """Plain, transformable copy of what the facts need."""
    task = json.loads((PT.VIEW_GT / item / "task.json").read_text())
    res = task["output_spec"]["final_resolution"]
    objs = []
    for o in gt.objects:
        sp = o.space or gt.space_of_point(o.center[:2])
        # (r1) "key" stays the GT object key: classes.gt_labels reads the prompt name from it
        objs.append({"key": o.key, "id": o.id, "cls": o.cls, "group": o.group, "kind": gt.kind_of(o), "space": sp,
                     "center": [float(x) for x in o.center], "size": [float(x) for x in o.size], "yaw": float(o.yaw),
                     "conf": float(o.conf), "unc": float(o.unc or 0.0), "named": bool(o.named),
                     "size_known": bool(o.size_known)})
    tracks = {}
    for eid, tr in gt.tracks.items():
        # a sample with no space gets the GT space holding it (pvscore5 GT.space_of_point)
        sps = [sp or gt.space_of_point(tr.p[i]) for i, sp in enumerate(tr.space)]
        tracks[eid] = {"t": [float(x) for x in tr.t], "p": [[float(a), float(b)] for a, b in tr.p],
                       "z": [float(x) for x in tr.z], "yaw": list(tr.yaw), "posture": list(tr.posture),
                       "space": sps, "visible": list(tr.visible), "segs": [list(s) for s in tr.segs],
                       "unc": float(tr.unc)}
    shots = []
    for sh in gt.shots:
        cams = []
        for c in sh.camera:
            if "c2w_blender" in c:
                cams.append({"t": float(c.get("t", sh.t0)), "c2w": c["c2w_blender"], "K": c["K"],
                             "W": c["width"], "H": c["height"]})
            else:
                cams.append({"t": float(c.get("t", sh.t0)), "pos": c.get("pos"), "look": c.get("look"),
                             "hfov": c.get("hfov_deg")})
        shots.append({"id": sh.id, "t0": sh.t0, "t1": sh.t1, "space": sh.space or gt.spaces[0].id, "cams": cams})
    spaces = {}
    for s in gt.spaces:
        lo, hi = s.extent
        spaces[s.id] = {"kind": s.kind, "floor_z": s.floor_z, "L": s.L, "area": s.area(),
                        "centre": [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)]}
    ents = {eid: e.get("class") for eid, e in gt.entities.items()}
    return {"item": item, "modality": gt.modality, "duration": gt.duration, "res": res, "objects": objs,
            "tracks": tracks, "shots": shots, "spaces": spaces, "entities": ents, "scale": 1.0}


# ------------------------------------------------------------------------------------------- transforms
def _rot(theta):
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    return np.array([[c, -s], [s, c]])


def transform(sc, theta=0.0, t=(0.0, 0.0), s=1.0, mirror=False):
    """Similarity (optionally mirrored x -> -x first) applied to every position, yaw, size and camera."""
    sc = copy.deepcopy(sc)
    Rm = _rot(theta)
    T = np.asarray(t, float)

    def P(xy):
        v = np.asarray(xy[:2], float)
        if mirror:
            v = np.array([-v[0], v[1]])
        return (s * (Rm @ v) + T).tolist()

    def Y(y):
        if y is None:
            return None
        y = (180.0 - y) if mirror else y
        return (y + theta) % 360.0

    def Z(z):
        return None if z is None else z * s

    for o in sc["objects"]:
        o["center"] = P(o["center"]) + [Z(o["center"][2])]
        o["size"] = [x * s for x in o["size"]]
        o["yaw"] = Y(o["yaw"])
        o["unc"] *= s
    for tr in sc["tracks"].values():
        tr["p"] = [P(p) for p in tr["p"]]
        tr["z"] = [Z(z) for z in tr["z"]]
        tr["yaw"] = [Y(y) for y in tr["yaw"]]
        tr["unc"] *= s
    for sp in sc["spaces"].values():
        sp["floor_z"] *= s
        sp["L"] *= s
        sp["area"] *= s * s
        sp["centre"] = P(sp["centre"])
    for sh in sc["shots"]:
        for c in sh["cams"]:
            if "c2w" in c:
                M = np.asarray(c["c2w"], float)
                A = np.eye(3)
                A[:2, :2] = s * Rm @ (np.diag([-1.0, 1.0]) if mirror else np.eye(2))
                A[2, 2] = s
                Rn = A @ M[:3, :3] / s
                if mirror:   # proper camera of the mirrored world: flip its x axis (the view is the mirror image)
                    Rn[:, 0] *= -1
                    c["K"] = dict(c["K"], cx=c["W"] - c["K"]["cx"])
                pos = A @ M[:3, 3]
                pos[:2] += T
                N = np.eye(4)
                N[:3, :3] = Rn
                N[:3, 3] = pos
                c["c2w"] = N.tolist()
            else:
                for k in ("pos", "look"):
                    v = c.get(k)
                    if v is None:
                        continue
                    xy = [np.nan if x is None else x for x in v[:2]]
                    c[k] = P(xy) + [Z(v[2]) if v[2] is not None else None]
    sc["scale"] = sc.get("scale", 1.0) * s
    return sc


# ------------------------------------------------------------------------------------------------ cameras
def cam_at(sh, t, res):
    """(pos[3], R[3x3] columns right/up/back, fx, fy, cx, cy, W, H) of shot sh at time t, or None."""
    cams = sh["cams"]
    if not cams:
        return None
    if "c2w" in cams[0]:
        c = min(cams, key=lambda c: abs(c["t"] - t))
        M = np.asarray(c["c2w"], float)
        K = c["K"]
        return M[:3, 3], M[:3, :3], K["fx"], K["fy"], K["cx"], K["cy"], c["W"], c["H"]
    ts = [c["t"] for c in cams]

    def vec(c, k):
        v = c.get(k)
        return np.full(3, np.nan) if v is None else np.array([np.nan if x is None else x for x in v], float)
    if t <= ts[0] or len(cams) == 1:
        a = b = cams[0] if t <= ts[0] or len(cams) == 1 else cams[-1]
        w = 0.0
    elif t >= ts[-1]:
        a = b = cams[-1]
        w = 0.0
    else:
        i = max(k for k in range(len(ts)) if ts[k] <= t)
        a, b = cams[i], cams[i + 1]
        w = (t - ts[i]) / max(1e-9, ts[i + 1] - ts[i])
    pos = (1 - w) * vec(a, "pos") + w * vec(b, "pos")
    look = (1 - w) * vec(a, "look") + w * vec(b, "look")
    hf = (1 - w) * (a.get("hfov") or np.nan) + w * (b.get("hfov") or np.nan)
    if not (np.isfinite(pos).all() and np.isfinite(look).all() and np.isfinite(hf)):
        return None
    f = look - pos
    f /= max(1e-9, np.linalg.norm(f))
    r = np.cross(f, [0.0, 0.0, 1.0])
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    W, H = res
    fx = (W / 2) / math.tan(math.radians(hf) / 2)
    return pos, np.stack([r, u, -f], 1), fx, fx, W / 2, H / 2, W, H


def stated(sh, t):
    """True when the GT camera pose at t is stated (thresholds view.text_times)."""
    cams = sh["cams"]
    if "c2w" in cams[0]:
        return True
    near = any(abs(c["t"] - t) <= VW["stated_near_sample_s"] for c in cams)
    if near:
        c = min(cams, key=lambda c: abs(c["t"] - t))
        return full(c.get("pos")) and full(c.get("look"))
    ts = [c["t"] for c in cams]
    if t < ts[0] or t > ts[-1]:
        return False
    i = max(k for k in range(len(ts)) if ts[k] <= t)
    if i + 1 >= len(cams):
        return False
    a, b = cams[i], cams[i + 1]
    return (full(a.get("pos")) and a.get("pos") == b.get("pos") and full(a.get("look"))
            and a.get("look") == b.get("look") and a.get("hfov") == b.get("hfov"))


def full(v):
    return v is not None and all(x is not None and math.isfinite(x) for x in v)


def frame_times(sc):
    out = []
    for sh in sc["shots"]:
        if sc["modality"] == "image":
            if sh["cams"]:
                out.append((sh["id"], round(sh["cams"][0]["t"], 3)))
            continue
        t0, t1 = sh["t0"], sh["t1"]
        ts = []
        k = 0
        while t0 + 0.5 + k < t1 - 0.25 + 1e-9:
            ts.append(t0 + 0.5 + k)
            k += 1
        for c in sh["cams"]:
            tc = min(t1 - 0.25, max(t0 + 0.25, c["t"]))
            if all(abs(tc - x) > 0.3 for x in ts):
                ts.append(tc)
        for t in sorted(ts):
            if stated(sh, t):
                out.append((sh["id"], round(t, 3)))
    return out


# ----------------------------------------------------------------------------------------- elements at t
def track_known(tr, t, tol=0.5):
    return any(a - tol <= t <= b + tol for a, b in tr["segs"])


def track_at(tr, t):
    ts = tr["t"]
    p = np.array(tr["p"], float)
    return np.array([np.interp(t, ts, p[:, 0]), np.interp(t, ts, p[:, 1])])


def track_label(tr, t, field):
    i = int(np.searchsorted(tr["t"], t, side="right") - 1)
    return tr[field][max(0, i)]


def person_classes(sc):
    return {eid for eid, c in sc["entities"].items() if c == "person"}


def boxes_at(sc, t, static_only=False):
    """List of boxes {key, cls, space, c[3], size[3], yaw, unc, person, named} present at time t."""
    out = []
    by_id = {o["id"]: o for o in sc["objects"] if o["id"]}
    persons = person_classes(sc)
    for o in sc["objects"]:
        if o["id"] and o["id"] in sc["tracks"] and sc["modality"] != "image":
            continue                          # placed from its track below
        key = ("b:" + o["id"]) if o["id"] else o["key"]
        out.append({"key": key, "cls": o["cls"], "space": o["space"], "c": list(o["center"]), "size": o["size"],
                    "yaw": o["yaw"], "unc": o["unc"], "person": False, "named": bool(o["id"]), "obj": o})
    if static_only or sc["modality"] == "image":
        return out
    for eid, tr in sc["tracks"].items():
        if not tr["t"] or not track_known(tr, t) or track_label(tr, t, "visible") is False:
            continue
        xy = track_at(tr, t)
        sp = track_label(tr, t, "space")
        if eid in persons:
            post = track_label(tr, t, "posture") or "default"
            h = EL["person_height_m"].get(post, EL["person_height_m"]["default"]) * sc["scale"]
            fz = sc["spaces"].get(sp, {}).get("floor_z", 0.0) if sp else 0.0
            zb = float(np.interp(t, tr["t"], [z if z is not None else 0.0 for z in tr["z"]])) + fz
            bw, bd = EL["person_box_m"]
            y = yaw_now(tr, t, sc["scale"])
            out.append({"key": "b:" + eid, "cls": "person", "space": sp, "c": [xy[0], xy[1], zb + h / 2],
                        "size": [bd * sc["scale"], bw * sc["scale"], h], "yaw": y if y is not None else 0.0,
                        "unc": tr["unc"], "person": True, "named": True, "obj": None})
        elif eid in by_id:
            o = by_id[eid]
            z = o["center"][2] if np.linalg.norm(xy - np.array(o["center"][:2])) < 0.5 * sc["scale"] \
                else (sc["spaces"].get(sp or o["space"], {}).get("floor_z", 0.0) + o["size"][2] / 2)
            out.append({"key": "b:" + eid, "cls": o["cls"], "space": sp or o["space"], "c": [xy[0], xy[1], z],
                        "size": o["size"], "yaw": o["yaw"], "unc": max(o["unc"], tr["unc"]), "person": False,
                        "named": True, "obj": o})
    out.sort(key=lambda b: b["key"])          # fixed order: render ties resolve the same way under reorder / rename
    return out


def yaw_now(tr, t, scale=1.0):
    """GT person yaw at t by the thresholds view.facing rule, or None."""
    ts = tr["t"]
    if not ts:
        return None
    p0, p1 = track_at(tr, t - 0.5), track_at(tr, t + 0.5)
    moved = float(np.linalg.norm(p1 - p0))
    for i, y in enumerate(tr["yaw"]):
        if y is not None and abs(ts[i] - t) <= 0.5 and moved < VW["facing_static_move_m"] * scale:
            return float(y)
    if track_known(tr, t - 0.5, 0.0) and track_known(tr, t + 0.5, 0.0) and moved >= VW["facing_moving_speed"] * scale:
        d = p1 - p0
        return math.degrees(math.atan2(d[1], d[0])) % 360.0
    return None


# ----------------------------------------------------------------------------------------------- render
def render(boxes, cam, grid_w=None):
    """Ray-cast oriented boxes from cam -> dict(idx [Hg,Wg] index into boxes or -1, dep z-depth, unocc per box,
    occ {box: (occluder box, visible fraction)}, right, back, tans = half-fov tangents (h, v))."""
    pos, Rc, fx, fy, cx, cy, W, H = cam
    Wg = grid_w or VW["grid_w"]
    Hg = int(round(Wg * H / W))
    xs = (np.arange(Wg) + 0.5) / Wg * W
    ys = (np.arange(Hg) + 0.5) / Hg * H
    X, Y = np.meshgrid(xs, ys)
    a = (X - cx) / fx
    b = -(Y - cy) / fy
    right, up, back = Rc[:, 0], Rc[:, 1], Rc[:, 2]
    D = (-back)[None, None] + a[..., None] * right[None, None] + b[..., None] * up[None, None]   # t = z-depth
    D = D.reshape(-1, 3)
    n = len(boxes)
    T = np.full((len(D), max(n, 1)), np.inf)
    for j, bx in enumerate(boxes):
        c = np.asarray(bx["c"], float)
        hs = np.maximum(np.asarray(bx["size"], float) / 2, 1e-3)
        cy_, sy_ = math.cos(math.radians(bx["yaw"])), math.sin(math.radians(bx["yaw"]))
        Rb = np.array([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1.0]])
        o = Rb.T @ (pos - c)
        if np.all(np.abs(o) < hs):
            continue                          # camera inside the box: treated as hollow
        d = D @ Rb
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-hs - o) / d
            t2 = (hs - o) / d
            lo = np.nanmax(np.minimum(t1, t2), 1)
            hi = np.nanmin(np.maximum(t1, t2), 1)
        hit = (hi >= lo) & (lo > 1e-6)
        T[hit, j] = lo[hit]
    tans = (W / 2 / fx, H / 2 / fy)
    # coplanar faces hit at the same depth: break the tie by box order with a relative (scale-free) 1e-7 bias,
    # so a similarity transform of the scene cannot flip which box is seen (fix r1, 10-18, found by T1)
    T = T * (1.0 + 1e-7 * np.arange(T.shape[1]))[None]
    if n == 0:
        return {"idx": np.full((Hg, Wg), -1), "dep": np.zeros((Hg, Wg)), "unocc": np.zeros(0, int), "occ": {},
                "right": right, "back": back, "tans": tans}
    idx = np.argmin(T, 1)
    dep = T[np.arange(len(D)), idx]
    idx = np.where(np.isfinite(dep), idx, -1)
    dep = np.where(np.isfinite(dep), dep, 0.0)
    fin = np.isfinite(T)
    unocc = fin.sum(0)
    occ = {}
    for j in range(n):
        if unocc[j] >= VW["occ_min_frac"] * len(D):
            vis = float((idx == j).sum()) / unocc[j]
            if vis < VW["occ_vis_max"]:
                other = idx[fin[:, j] & (idx != j) & (idx >= 0)]
                occ[j] = (int(np.bincount(other).argmax()) if len(other) else -1, vis)
    return {"idx": idx.reshape(Hg, Wg), "dep": dep.reshape(Hg, Wg), "unocc": unocc, "occ": occ,
            "right": right, "back": back, "tans": tans}


def facing_cam(yaw, right, back):
    """0 toward the camera, +90 front to screen right, 180 away (model_view.facing_cam convention)."""
    if yaw is None:
        return None
    v = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0])
    rh = np.array([right[0], right[1], 0.0])
    bh = np.array([back[0], back[1], 0.0])
    if np.linalg.norm(rh) < 1e-6 or np.linalg.norm(bh) < 1e-6:
        return None
    return round(math.degrees(math.atan2(float(v @ rh) / np.linalg.norm(rh), float(v @ bh) / np.linalg.norm(bh))), 1)


def readout(boxes, rd, sc, t):
    """Visible elements of one rendered frame in the model_view element format (+ unc, unoccluded fraction)."""
    idx, dep = rd["idx"], rd["dep"]
    Hg, Wg = idx.shape
    E = []
    for j in np.unique(idx[idx >= 0]):
        m = idx == j
        yy, xx = np.nonzero(m)
        bx = boxes[int(j)]
        z = dep[m]
        e = {"key": bx["key"], "class": bx["cls"], "space": bx["space"],
             "bbox": [round(float(xx.min()) / Wg, 4), round(float(yy.min()) / Hg, 4),
                      round(float(xx.max() + 1) / Wg, 4), round(float(yy.max() + 1) / Hg, 4)],
             "c": [round(float(xx.mean() + 0.5) / Wg, 4), round(float(yy.mean() + 0.5) / Hg, 4)],
             "area_frac": round(float(m.mean()), 5), "depth_med": round(float(np.median(z)), 4),
             "unc": bx["unc"], "unocc_frac": round(float(rd["unocc"][int(j)]) / (Hg * Wg), 5),
             "person": bx["person"]}
        e["cell"] = SG.cell(*e["c"])
        e["height_frac"] = round(e["bbox"][3] - e["bbox"][1], 4)
        if bx["person"]:
            eid = bx["key"][2:]
            y = yaw_now(sc["tracks"][eid], t, sc["scale"]) if eid in sc["tracks"] else None
            e["facing_cam"] = facing_cam(y, rd["right"], rd["back"])
        E.append(e)
    occl = [{"key": boxes[j]["key"], "occluder": boxes[o]["key"] if o >= 0 else None, "vis_frac": round(v, 3)}
            for j, (o, v) in rd["occ"].items()]
    E.sort(key=lambda e: -e["area_frac"])
    return E, occl
