"""v5.1, model side of the per-frame scene inventory (no API, no sandbox): the model frames of a 5.0.9 dump
(ray-cast grid: part / face tag / depth per cell, 4 Hz, 96 cells wide) as the same kind of element list as the video
side (data/v51_r2/<item>/scene_frames.json), the GT time -> model frame map, and the named bodies over GT time.

    ssh ... "cd $B && env/bin/python -I -B tools/v51_r2/model_view.py --batch runs_cmp/batches/<id> ... [M50_033 ...]"

GT time -> model time and model shot exactly as pvscore 5.0.9 (align, timelines, warp; camera.model_time_for), then
the nearest grid frame of that model shot; mapped every STEP s of each GT shot and at every geometry sample time. The model grid is centre-cropped to the video aspect (as P1 does).
Element = one named body (GT id from its tags) or one furnishing instance (5.0.9 Model.layout); terrain and untagged
floor are surfaces (support only). Per element: box, centre, 3x3 cell, area, median depth, box height, and for bodies
the facing relative to the camera (0 = toward, +90 = front points to screen right, 180 = away; body_yaw). Relations with the
same rules as the video side (scene_geom.relations, r1 thresholds), with no contact pairs (those come from V2 only).
Bodies at the same GT times: present, 5.0.9 posture (Timeline: stand / crouch / sit / lie by height), airborne
(5.0.9 Events.airborne), speed and camera distance in body heights (ref_h), base / top height, facing to the camera.
10-18 (GT r1 segments): xy_scale (model x, y units -> body heights) and the camera's horizontal view direction
cam_f / screen-right axis cam_r in model x, y at that time, appended after cam_dist_h.
Only reads the dumps and pvscore5 (paths.PVSCORE_DIR, imported, not changed); writes
paths.VIEW_OUT/<item>/model_view/<model>.json. r2: dumps come from launcher batch dirs (--batch, batches.py); the
model name is the launcher model name.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as PT  # noqa: E402
R = PT.ROOT
sys.path.insert(0, str(PT.PVSCORE_DIR))
from pvscore5 import blend_cams, camera, fixes, geom, layout, motion  # noqa: E402
from pvscore5.gt import GT, load_categories  # noqa: E402
from pvscore5.model import Model  # noqa: E402
from pvscore5.score import CATS, Ctx, load_thresholds  # noqa: E402
import scene_geom as SG  # noqa: E402
from gt_root import gt_json  # noqa: E402
import sframes as SF  # noqa: E402
import items as IT  # noqa: E402
import batches as BT  # noqa: E402

GEO = PT.VIEW_GT                                        # GT side: <item>/scene_frames.json
OUT = PT.VIEW_OUT                                       # writes <item>/model_view/<model>.json
SURFACE_KINDS = {"terrain", "untagged_floor"}
SURFACE_CLASSES = {"floor", "platform"}                 # 5.0.9 GROUND_STRUCT members that are walkable floors
STEP = 0.25
FACING_WIN_S = 0.25                                      # reader r1, see body_yaw
VEH_MOVE = 0.05                                          # ~0.2 m for a 4.5 m car, the 5.0.9 Timeline motion floor
FACING_UNKNOWN = "unknown"                               # 5.2 (M14): body row facing_cam when the facing is unknown
# 5.2 (M15): the dump does not record clip_start; each camera's own value is read from the delivered scene file


def build_ctx(gt_path, dump):
    cats = load_categories(CATS)
    th = load_thresholds()
    gt = GT(gt_path, cats)
    md = Model(dump, cats, th["params"])
    ctx = Ctx(gt, md, th)
    ids = set(gt.entities)
    md.layout(ids)
    md.bodies(ids)
    layout.run_alignment(ctx)
    if not ctx.aligns:
        ctx.aligns = {s.id: layout.Align(s.id, geom.Sim2(), "none", 0, 0.0, s.floor_z, s.L) for s in gt.spaces}
    ctx.timelines.update({eid: motion.Timeline(ctx, eid) for eid in md.body})
    camera.build_warp(ctx)
    return gt, md, ctx


def body_row(md, ctx, ev, air, eid, t, tm):
    tl = ctx.timelines.get(eid)
    b = md.body[eid]
    ai = md.t_to_ai(tm)
    k = md.t_to_frame_index(tm)
    if not b.present[ai]:
        return [t, False]
    s = getattr(ctx.aligns.get(tl.space[ai]) if tl is not None and tl.space[ai] else None, "sim", None)
    sc = s.s if s is not None else 1.0
    # vehicles have no Timeline ref_h: their own 95th-percentile height in GT metres (reader r1; r0 left it in model
    # units, so speed_h mixed GT metres over model units)
    ref = getattr(tl, "ref_h", None) or float(np.nanpercentile(b.top - b.base, 95)) * sc or 1.0
    cen = np.array([b.center[ai][0], b.center[ai][1], (b.base[ai] + b.top[ai]) / 2])
    dist = float(np.linalg.norm(md.cam_pos[k] - cen)) * sc / ref if md.cam_ok[k] else None
    hb = (round(float(b.base[ai]) * sc / ref, 3), round(float(b.top[ai]) * sc / ref, 3))
    if fixes.on("M5") and getattr(tl, "ref_m", None):
        # 5.2 (M5): base / top in the body's own standing height in model units, as the timeline posture classes
        hb = (round(float(b.base[ai]) / tl.ref_m, 3), round(float(b.top[ai]) / tl.ref_m, 3))
    return [t, True, tl.posture[ai] if tl is not None else None, bool(air[ai]) if air is not None else None,
            round(float(tl.speed[ai]) / ref, 3) if tl is not None else None,
            round(float(b.center[ai][0]), 3), round(float(b.center[ai][1]), 3),
            *hb,
            facing_cam(md, b, ai, k), None if dist is None else round(dist, 3),
            round(sc / ref, 5), *cam_axes(md, k)]


def cam_axes(md, k):
    """10-18 (GT r1 segments): horizontal unit view direction and screen-right axis of the camera at frame k in model
    x, y (the frame of the body x, y), or None, None; as pvscore 5.0.10 camera._segment_dir"""
    if not md.cam_ok[k]:
        return None, None
    f, r = md.cam_fwd[k][:2], md.cam_R[k][:2, 0]
    if np.linalg.norm(f) < 1e-6:            # camera looking straight down: the screen-up axis stands in for forward
        f = md.cam_R[k][:2, 1]
    if np.linalg.norm(f) < 1e-6 or np.linalg.norm(r) < 1e-6:
        return None, None
    return ([round(float(x), 4) for x in f / np.linalg.norm(f)], [round(float(x), 4) for x in r / np.linalg.norm(r)])


def key_table(md):
    """part * 4096 + tag -> (element key, class, id, kind)"""
    tab = {}
    for n, m in enumerate(md.instances):
        cls = "surface:ground" if m.kind in SURFACE_KINDS or m.cls in SURFACE_CLASSES else (m.cls or "untagged")
        for p in m.pieces:
            tab[p["part"] * 4096 + p["tag"]] = ("i%d" % n, cls, None, m.kind)
    for (pi, tg), (cls, eid) in md.label.items():
        if eid:
            tab[pi * 4096 + tg] = ("b:" + eid, cls or "person", eid, "body")
    return tab


def crop(a, src_aspect, dst_aspect):
    H, W = a.shape[:2]
    if src_aspect > dst_aspect + 1e-6:
        w = int(round(H * dst_aspect))
        x0 = (W - w) // 2
        return a[:, x0:x0 + w]
    if src_aspect < dst_aspect - 1e-6:
        h = int(round(W / dst_aspect))
        y0 = (H - h) // 2
        return a[y0:y0 + h]
    return a


def body_yaw(md, b, ai):
    """model-frame yaw (deg) of a body at analysis frame ai, or None. Reader r1 (10-18): persons the circular medoid
    of the 5.0.9 per-frame facing within +-FACING_WIN_S (the front-marker facing flips with swinging legs); vehicles,
    which have no marker, the long axis signed by the motion over +-2 frames, as 5.0.9 Timeline (moved >= VEH_MOVE x
    length; parked: None)."""
    if b.group == "vehicle":
        i, j = max(0, ai - 2), min(md.A - 1, ai + 2)
        if not (b.present[i] and b.present[j]):
            return None
        d = np.asarray(b.center[j][:2], float) - np.asarray(b.center[i][:2], float)
        if np.linalg.norm(d) < VEH_MOVE * max(float(b.long[ai]), 1e-6):
            return None
        mv = math.degrees(math.atan2(d[1], d[0]))
        ly = float(b.long_yaw[ai])
        return ly if geom.angdiff(ly, mv, 360) <= 90 else (ly + 180) % 360
    win = [x for x in range(md.A) if abs(md.an_t[x] - md.an_t[ai]) <= FACING_WIN_S + 1e-6
           and b.present[x] and not np.isnan(b.facing[x])]
    if not win:
        if fixes.on("M14") and any(b.present[x] and b.facing_unk[x] for x in range(md.A)
                                   if abs(md.an_t[x] - md.an_t[ai]) <= FACING_WIN_S + 1e-6):
            return FACING_UNKNOWN   # 5.2 (M14): no readable facing in the window, an unknown one there
        return None
    fs = [float(b.facing[x]) for x in win]
    return min(fs, key=lambda f: sum(geom.angdiff(f, g, 360) for g in fs))


def facing_cam(md, b, ai, k):
    """body facing relative to the camera at frame k: 0 toward, +90 front to screen right, 180 away (deg)"""
    f = body_yaw(md, b, ai)
    if f is None or not md.cam_ok[k]:
        return None
    if f == FACING_UNKNOWN:
        return FACING_UNKNOWN       # 5.2 (M14)
    v = np.array([math.cos(math.radians(f)), math.sin(math.radians(f)), 0.0])
    Rc = md.cam_R[k]
    right, back = Rc[:, 0], Rc[:, 2]
    rh = np.array([right[0], right[1], 0.0])
    bh = np.array([back[0], back[1], 0.0])
    if np.linalg.norm(rh) < 1e-6 or np.linalg.norm(bh) < 1e-6:
        return None
    return round(math.degrees(math.atan2(float(v @ rh) / np.linalg.norm(rh), float(v @ bh) / np.linalg.norm(bh))), 1)


def frame_elements(md, tab, gi, aspect):
    g = md.grid(gi)
    hit = g["part"] >= 0
    clip = frame_clip(md, gi) if fixes.on("M15") else None
    if clip is not None:            # 5.2 (M15): hits nearer than the camera's clip_start are not visible content
        hit = hit & (g["depth"] >= clip)
    keyarr = np.where(hit, g["part"].astype(np.int64) * 4096 + g["tag"].astype(np.int64), -1)
    src_aspect = g["part"].shape[1] / g["part"].shape[0]
    keyarr = crop(keyarr, src_aspect, aspect)
    D = crop(np.where(hit, g["depth"], 0.0), src_aspect, aspect)
    h, w = keyarr.shape
    groups = {}
    for kv in np.unique(keyarr[keyarr >= 0]):
        e = tab.get(int(kv))
        if e is None:
            continue
        groups.setdefault(e, []).append(kv)
    k = md.t_to_frame_index(float(md.g_t[gi]))
    ai = md.t_to_ai(float(md.g_t[gi]))
    E = []
    for (ek, cls, eid, kind), kvs in groups.items():
        m = np.isin(keyarr, kvs)
        yy, xx = np.nonzero(m)
        z = D[m]
        zv = z[z > 0]
        e = {"key": ek, "label": eid or cls, "id": eid, "class": cls, "kind": kind, "_m": m,
             "bbox": [round(float(xx.min()) / w, 3), round(float(yy.min()) / h, 3),
                      round(float(xx.max() + 1) / w, 3), round(float(yy.max() + 1) / h, 3)],
             "c": [round(float(xx.mean() + 0.5) / w, 3), round(float(yy.mean() + 0.5) / h, 3)],
             "area_frac": round(float(m.mean()), 4),
             "depth_med": round(float(np.median(zv)), 3) if len(zv) else None}
        e["cell"] = SG.cell(*e["c"])
        e["height_frac"] = round(e["bbox"][3] - e["bbox"][1], 3)
        e["cut_off"] = [s for s, v in (("left", e["bbox"][0] <= 0), ("right", e["bbox"][2] >= 1),
                                        ("top", e["bbox"][1] <= 0), ("bottom", e["bbox"][3] >= 1)) if v]
        if eid and eid in md.body:
            e["facing_cam"] = facing_cam(md, md.body[eid], ai, k)
            if e["facing_cam"] == FACING_UNKNOWN:
                e["facing_cam"] = None      # 5.2 (M14): frame elements keep None, flagged unknown
                e["facing_unknown"] = True
        E.append(e)
    E.sort(key=lambda e: -e["area_frac"])
    return E, D, h, w, k


def cam_clips(dump):
    """5.2 (M15): {camera object name: clip_start} from the delivered scene file the dump was made from (acquire.json
    harness.blend, its sha256 checked against harness.blend_sha256; pvscore5/blend_cams.py reads the file's own SDNA).
    A camera whose clip_start is animated is left out; {} when the file cannot be read or does not match: then no
    frame is cut (the 5.1 reading), never a guessed clip_start."""
    try:
        h = json.loads((Path(dump) / "acquire.json").read_text())["harness"]
        p = Path(h["blend"])
        if hashlib.sha256(p.read_bytes()).hexdigest() != h["blend_sha256"]:
            return {}
        return {k: v["clip_start"] for k, v in blend_cams.camera_clips(p).items() if not v["animated"]}
    except Exception:  # noqa: BLE001 - no readable scene file: clip_start unknown
        return {}


def frame_clip(md, gi):
    """5.2 (M15): clip_start of the scene camera at model grid frame gi, None when unknown"""
    clips = getattr(md, "clips", None) or {}
    k = int(md.g_frame[gi]) - md.f0
    names = md.doc.get("cameras") or []
    ci = int(md.cam_index[k]) if 0 <= k < len(md.cam_index) else -1
    return clips.get(names[ci]) if 0 <= ci < len(names) else None


def clip_frac(md, gi, aspect):
    """5.2 (M15): fraction of the cropped model frame whose ray-cast hit is nearer than its camera's clip_start
    (None when that clip_start is unknown)"""
    clip = frame_clip(md, gi)
    if clip is None:
        return None
    g = md.grid(gi)
    near = (g["part"] >= 0) & (g["depth"] < clip)
    return float(crop(near, g["part"].shape[1] / g["part"].shape[0], aspect).mean())


def event_airborne(mask, t, lo):
    """5.2 (M17): the airborne frames the event layer reads (Events.jump_runs): runs with gaps <= 0.15 s bridged,
    a run shorter than jump_min_s that does not touch a clip end is cleared (gait jitter is no take-off)"""
    keep = np.zeros(len(mask), bool)
    for t0, t1 in motion.runs(mask, t, fill=0.15):
        if t1 - t0 < lo and not (t0 <= t[0] or t1 >= t[-1]):
            continue
        keep |= (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
    return mask & keep


def run(item, name, dump):
    gt_path = gt_json(item)
    gt, md, ctx = build_ctx(gt_path, dump)
    if fixes.on("M15"):
        md.clips = cam_clips(dump)
        if not md.clips:            # 5.2 (addendum 3): the 5.1 reading is kept, but never silently
            print("M15: clip_start unknown for %s" % dump, file=sys.stderr, flush=True)
    tab = key_table(md)
    geo = SF.load(GEO / item / "scene_frames.json")
    gshot = {s.id: s for s in gt.shots}
    ev = motion.Events(ctx)
    airs = {}
    for eid in md.body:
        try:
            airs[eid] = ev.airborne(eid)
            if fixes.on("M17"):
                airs[eid] = event_airborne(airs[eid], ev.t, ev.P.get("jump_min_s", 0.1))
        except Exception:  # noqa: BLE001 - a body the detector cannot read has no airborne mask
            airs[eid] = None
    tmap, frames, bodies = [], {}, {eid: [] for eid in md.body}
    for sh in geo["shots"]:
        W, H = sh.get("size") or (16, 9)
        aspect = W / H
        t0, t1 = sh["t"]
        ts = {round(t0 + i * STEP, 3) for i in range(int((t1 - t0) / STEP) + 1)}
        ts |= {round(sm["t"], 3) for sm in sh["samples"] if sm.get("fi") is not None}
        for t in sorted(ts):
            mshot, tm = camera.model_time_for(ctx, gshot[sh["shot"]], t)
            gi = md.grid_near(tm, mshot) if tm is not None else None
            tmap.append([t, sh["shot"], mshot, None if tm is None else round(float(tm), 3), gi,
                         None if gi is None else round(abs(float(md.g_t[gi]) - tm), 3)])
            if tm is None:
                continue
            for eid in md.body:
                bodies[eid].append(body_row(md, ctx, ev, airs[eid], eid, t, tm))
            if gi is None or gi in frames:
                continue
            E, D, h, w, k = frame_elements(md, tab, gi, aspect)
            frames[gi] = {"grid_t": round(float(md.g_t[gi]), 3), "grid_hw": [h, w], "cam_ok": bool(md.cam_ok[k]),
                          "sky_frac": round(1.0 - float(sum(e["area_frac"] for e in E)), 4),
                          "elements": [{kk: v for kk, v in e.items() if not kk.startswith("_")} for e in E],
                          "relations": SG.relations(E, D, h, w)}
            cf = clip_frac(md, gi, aspect) if fixes.on("M15") else None
            if cf is not None:      # 5.2 (M15): most of the image nearer than clip_start = unreadable frame
                frames[gi]["sky_frac"] = round(1.0 - float(sum(e["area_frac"] for e in E)) - cf, 4)
                frames[gi].update(clip_frac=round(cf, 4), unreadable=cf > 0.5)
    res = {"item": item, "model": name, "dump": PT.rel(dump), "reader": "model_view.py (pvscore 5.1 r2)",
           "thresholds": SG.TH_FILE,
           "model_shots": len(md.shots), "duration_s": round(md.duration, 3), "gt_duration_s": gt.duration,
           "bodies": sorted(md.body), "instances": len(md.instances),
           "tag_report": {k: v for k, v in md.tag_report(set(gt.entities), [s.id for s in gt.spaces]).items()
                          if k in ("tagged_area_frac", "entities_missing", "unknown_classes", "spaces_used")},
           "time_map_cols": ["t", "shot", "model_shot", "model_t", "gi", "dt"],
           "time_map": tmap, "frames": {str(k): v for k, v in sorted(frames.items())},
           "body_cols": ["t", "present", "posture", "airborne", "speed_h", "x", "y", "base_h", "top_h", "facing_cam",
                         "cam_dist_h", "xy_scale", "cam_f", "cam_r"],
           "body_series": bodies,
           "body_info": {eid: {"class": b.cls, "group": b.group,
                               "ref_h": round(float(getattr(ctx.timelines.get(eid), "ref_h", 0) or 0), 3)}
                         for eid, b in md.body.items()}}
    if fixes.on("M15"):
        res["clip_start"] = {k: round(float(v), 6) for k, v in sorted(md.clips.items())} or None  # 5.2 (M15)
    o = OUT / item / "model_view"
    o.mkdir(parents=True, exist_ok=True)
    (o / f"{name}.json").write_text(json.dumps(res, indent=1) + "\n")
    ne = [len(f["elements"]) for f in frames.values()]
    print(item, name, "times", len(tmap), "frames", len(frames), "elements/frame %.1f" % (np.mean(ne) if ne else 0),
          "bodies", res["bodies"], "missing", res["tag_report"]["entities_missing"], flush=True)
    return res


def main():
    """model_view.py --batch DIR [--batch DIR ...] [--model=NAME[,NAME]] [M50_xxx ...]: every view item (video,
    continuation) with a dump in the batches; the latest batch wins per (model, item) (batches.latest). model = the
    launcher model name (dry-run-fake for the fixture)."""
    a = sys.argv[1:]
    bdirs = [a[i + 1] for i, x in enumerate(a) if x == "--batch"]
    if not bdirs:
        sys.exit("give --batch DIR (launcher batch directories)")
    only = next((set(x.split("=", 1)[1].split(",")) for x in a if x.startswith("--model=")), None)
    view = set(IT.of_kinds(IT.VIEW_KINDS))
    items = [x for x in a if x in view] or sorted(view)
    for (name, item), att in sorted(BT.latest(bdirs).items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if item not in items or (only is not None and name not in only):
            continue
        run(item, name, att["out"])


if __name__ == "__main__":
    main()
