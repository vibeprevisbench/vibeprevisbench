"""D1 physical plausibility and Q editability (Q is reported separately and never enters the total)."""
from __future__ import annotations

import math
import re

import numpy as np

from . import fixes, geom

try:
    from scipy.spatial import ConvexHull
except Exception:  # pragma: no cover
    ConvexHull = None

MOUNTED = {"picture_frame", "mirror", "tv", "monitor", "lamp", "curtain", "shelf", "cabinet", "sign", "window",
           "door", "flag", "streetlight", "traffic_light", "basketball_hoop", "net", "umbrella", "canopy", "rack"}
DEFAULT_NAME = re.compile(r"^(Cube|Sphere|UVSphere|Cylinder|Plane|Cone|Torus|Icosphere|Mesh|Object|Empty|Text|Curve|"
                          r"BezierCurve|BezierCircle|NurbsPath|NurbsCurve|Circle|Suzanne|Grid|Capsule)(\.\d+)?$")


def _obb(P):
    """(center2, R2, half extents [3], zlo, zhi) of a point set's min-area-rect box."""
    c, yaw, lg, sh = geom.min_area_rect(P[:, :2])
    return np.asarray(c), yaw, np.array([lg / 2, sh / 2]), P[:, 2].min(), P[:, 2].max()


def _inside(Q, box, shrink=0.02):
    c, yaw, half, z0, z1 = box
    R = geom.rot2(-yaw)
    L = (Q[:, :2] - c) @ R.T
    return (np.abs(L[:, 0]) < half[0] - shrink) & (np.abs(L[:, 1]) < half[1] - shrink) & \
        (Q[:, 2] > z0 + shrink) & (Q[:, 2] < z1 - shrink)


def _convex_piece(P):
    """True if a piece's points fill its oriented box well enough to be treated as a solid box."""
    if ConvexHull is None or len(P) < 8:
        return True
    c, yaw, half, z0, z1 = _obb(P)
    vol_box = 4 * half[0] * half[1] * max(1e-6, z1 - z0)
    try:
        vol_hull = ConvexHull(P).volume
    except Exception:
        return True
    return vol_hull / max(1e-9, vol_box) >= 0.5


def _gt_overlap(oa, ob):
    """True if two GT boxes interpenetrate (then the model overlap is not penalised)."""
    def grid(o):
        u = np.linspace(-0.4, 0.4, 3)
        L = np.array([[x, y, z] for x in u for y in u for z in u]) * o.size
        R = geom.rot2(o.yaw)
        return np.c_[L[:, :2] @ R.T + o.center[:2], L[:, 2] + o.center[2]]

    def box(o):
        return (o.center[:2], o.yaw, o.size[:2] / 2, o.bottom, o.top)
    return bool(_inside(grid(oa), box(ob), 0.0).any() or _inside(grid(ob), box(oa), 0.0).any())


HELD_TYPES = ("pick_up", "put_down", "carry", "hand_over", "hold")


def _held_windows(ctx):
    """5.0.12 (S10): entity id -> GT windows [t0 - tol, t1 + tol] of events that carry / hold it."""
    gt = ctx.gt
    tol_def = ctx.TH["event_tol_s"]
    out = {}
    for e in gt.events:
        if e.get("type") not in HELD_TYPES:
            continue
        t = e.get("t", [0, 0])
        t0, t1 = (float(t[0]), float(t[1])) if isinstance(t, (list, tuple)) else (float(t), float(t))
        tol = float(e.get("tol_s", tol_def))
        for x in list(e.get("objects", [])) + list(e.get("agents", [])):
            o = gt.obj_by.get(x)
            eid = o.id if o is not None and o.id else x
            out.setdefault(eid, []).append((t0 - tol, t1 + tol))
    return out


def score_D1(ctx, items):
    gt, md, TH, P = ctx.gt, ctx.model, ctx.TH, ctx.P
    insts = [m for m in md.instances if m.kind in ("object", "untagged_object")]
    # named static bodies at their first frame (objects with ids) join the penetration / support checks
    for eid, b in md.body.items():
        if b.group not in ("agent", "animal", "vehicle") and b.P0 is not None:
            bi = ctx.entity_instance(eid)     # 5.0.12 (S19): at the GT box time
            if bi is not None:
                bi.pieces = [{"part": p, "tag": 0, "P": bi.P} for p in bi.parts][:1]
                insts.append(bi)
    if insts:
        solids = []
        for m in insts:
            boxes = []
            for pc in m.pieces:
                Q = pc.get("P")
                if Q is None or len(Q) < 4 or not _convex_piece(Q):
                    continue
                boxes.append(_obb(Q))
            solids.append(boxes)
        samples = [m.P for m in insts]
        inv = {inst.key: key for key, (inst, c, _) in getattr(ctx, "matches", {}).items() if inst is not None}
        n_pen, pairs = 0, []
        for i in range(len(insts)):
            for j in range(i + 1, len(insts)):
                a, b = insts[i], insts[j]
                if geom.aabb_gap(a.lo, a.hi, b.lo, b.hi) > 0:
                    continue
                fa = max((float(_inside(samples[i], bx).mean()) for bx in solids[j]), default=0.0)
                fb = max((float(_inside(samples[j], bx).mean()) for bx in solids[i]), default=0.0)
                if max(fa, fb) > P["penetration_frac"]:
                    oa, ob = gt.obj_by.get(inv.get(a.key)), gt.obj_by.get(inv.get(b.key))
                    if oa is not None and ob is not None and _gt_overlap(oa, ob):
                        continue
                    n_pen += 1
                    pairs.append([a.key, b.key, round(max(fa, fb), 3)])
        rate = n_pen / max(1, len(insts))
        items.add("D1", "penetration", "penetration", geom.lin(rate, *TH["d1_rate"]), 1.0, value=round(rate, 4),
                  pairs=pairs[:12])
    # floating / sinking: static objects (relative to their GT height) and bodies (time below the floor) pooled
    # into one rate, so that the cost of a violation does not depend on how many well-behaved things are around
    bad, n_static, sink_sum, n_bodies = [], 0, 0.0, 0
    if insts:
        gt_h = {}
        for key, (inst, c, _) in getattr(ctx, "matches", {}).items():
            o = gt.obj_by.get(key)
            if inst is not None and o is not None:
                gt_h[inst.key] = o.bottom - gt.space_by[o.space].floor_z
        boxes = {}

        def piece_boxes(m):
            if m.key not in boxes:
                boxes[m.key] = [geom.oriented_box(pc["P"]) for pc in m.pieces
                                if pc.get("P") is not None and len(pc["P"]) >= 3]
            return boxes[m.key]

        # supports may be anything but the untagged floor slab; contact only counts against other objects (a wall
        # holds up only MOUNTED classes, and a single shell mesh would otherwise "touch" everything)
        others = [m for m in md.instances if m.kind != "untagged_floor"] + [m for m in insts if m not in md.instances]
        objs = [o for o in others if o.kind in ("object", "untagged_object") or o in insts]
        held = _held_windows(ctx) if fixes.on("S10") else {}
        for m in insts:
            if m.cls in MOUNTED:
                continue
            sp = getattr(m, "gt_space", None) or gt.spaces[0].id
            if gt.space_by[sp].sloped:   # 5.0.2: no floor level to float above or sink below
                continue
            al = ctx.aligns[sp]
            n_static += 1
            h = float(al.h(m.bottom))
            if h < -P["sink_m"]:
                bad.append([m.key, "sink", round(h, 3)])
                continue
            # the GT's own raise (an object on a table, or an annotation box a few cm off the floor) is allowed
            if h <= max(0.0, gt_h.get(m.key, 0.0)) + P["float_m"]:
                continue
            # 5.0.12 (S10): a body the GT has carried / held at the instance's time is not floating
            if m.key.startswith("id:") and any(ctx.warp(a) <= getattr(m, "t_snap", 0.0) <= ctx.warp(b)
                                               for a, b in held.get(m.key[3:], ())):
                continue
            # resting on another object: its top lies just under our bottom and our footprint centre is over it
            sup = [o.top for o in others if o is not m and m.bottom - 0.5 < o.top <= m.bottom + 0.05 and
                   geom.point_in_poly(m.center2, geom.rect_corners(o.center2, o.long_yaw, o.long + 0.04,
                                                                   o.short + 0.04))]
            if sup and m.bottom - max(sup) <= P["float_m"]:
                continue
            # hanging from / leaning on another object (oriented boxes, so a global rotation changes nothing)
            mine = piece_boxes(m)
            touching = any(geom.aabb_gap(m.lo, m.hi, o.lo, o.hi) <= 0.02 and
                           any(geom.box_gap(a, b) <= 0.02 for a in mine for b in piece_boxes(o))
                           for o in objs if o is not m)
            if not touching:
                bad.append([m.key, "float", round(h, 3)])
    body_sink = {}
    for eid, tl in ctx.timelines.items():
        on = tl.present & np.array([not (s in gt.space_by and gt.space_by[s].sloped) for s in tl.space], bool)
        if md.body[eid].group in ("agent", "animal") and on.any():
            body_sink[eid] = float(np.mean(tl.h_base[on] < -0.1))
            sink_sum += body_sink[eid]
            n_bodies += 1
    if n_static + n_bodies:
        rate = (len(bad) + sink_sum) / (n_static + n_bodies)
        items.add("D1", "float_sink", "float_sink", geom.lin(rate, *TH["d1_rate"]), 1.0, value=round(rate, 4),
                  bad=bad[:12], body_sink={k: round(v, 3) for k, v in body_sink.items() if v > 0})
    # teleports / pops within a shot: one scene-level event count
    pops_ok = {(x, e["type"]) for e in gt.events for x in e.get("agents", []) + e.get("objects", [])}
    events, per = 0.0, {}
    # 5.0.12 (D15): in-shot jumps inside the (warped) window of a dissolve between two spaces are the dissolve
    from .camera import cross_space_dissolves
    xwin = P.get("dissolve_win_s", 1.0)
    xd_m = [ctx.warp(t) for t in cross_space_dissolves(ctx)] if gt.modality != "image" else []
    for eid, tl in ctx.timelines.items():
        b = md.body[eid]
        grp = b.group if b.group in P["teleport_speed_ms"] else "default"
        vmax = P["teleport_speed_ms"].get(grp, P["teleport_speed_ms"]["default"])
        n_tp, n_pop = 0, 0
        for ai in range(1, md.A):
            k0, k1 = int(md.an_frame[ai - 1]) - md.f0, int(md.an_frame[ai]) - md.f0
            same_shot = md.shot_of_frame(k0) == md.shot_of_frame(k1)
            if tl.present[ai] and tl.present[ai - 1]:
                d = float(np.linalg.norm(tl.xy[ai] - tl.xy[ai - 1]))
                dt = max(1e-6, md.an_t[ai] - md.an_t[ai - 1])
                if d > P["teleport_m"] and d / dt > vmax and same_shot:
                    if xd_m and any(md.an_t[ai - 1] - xwin <= x <= md.an_t[ai] + xwin for x in xd_m):
                        continue
                    n_tp += 1
            elif tl.present[ai] != tl.present[ai - 1] and same_shot:
                allowed = any((eid, t) in pops_ok for t in ("appear", "disappear", "enter_vehicle", "exit_vehicle",
                                                            "enter_space", "exit_space"))
                if not allowed:
                    n_pop += 1
        events += n_tp + 0.5 * n_pop
        if n_tp or n_pop:
            per[eid] = {"teleports": n_tp, "pops": n_pop}
    if ctx.timelines:
        items.add("D1", "teleport", "teleport", geom.lin(events, *TH["d1_events"]), 1.0, value=round(events, 2),
                  bodies=per)
    if md.body:
        dup = [eid for eid, b in md.body.items() if b.present.any() and b.dup[b.present].mean() > 0.1]
        rate = len(dup) / max(1, len(md.body))
        items.add("D1", "duplicates", "duplicates", geom.lin(rate, *TH["d1_dup_rate"]), 0.5, value=round(rate, 3),
                  ids=dup)
    # class-size sanity
    viol, n = [], 0
    a0 = ctx.aligns[gt.spaces[0].id]
    s_ = a0.sim.s if a0.ok else 1.0
    for m in insts + [md.body_instance(e) for e in md.body]:
        if m is None or m.cls not in gt.cats:
            continue
        lo_, hi_ = gt.cats[m.cls]["size_m"]
        size = max(m.long, m.height) * s_
        n += 1
        if size < lo_ / 1.25 or size > hi_ * 1.25:
            viol.append([m.key, m.cls, round(size, 3)])
    if n:
        rate = len(viol) / n
        items.add("D1", "class_size", "class_size", geom.lin(rate, *TH["d1_rate"]), 1.0, value=round(rate, 4),
                  violations=viol[:12])
    for sid, al in ctx.aligns.items():
        if al.ok:
            items.add("D1", "scale", "scale:" + sid, geom.lin(abs(math.log(al.sim.s)), *TH["d1_scale_log"]),
                      0.5 if gt.modality == "image" else 1.0, value=round(al.sim.s, 4))


def score_Q(ctx, items):
    gt, md = ctx.gt, ctx.model
    tr = ctx.tag_report
    if tr.get("tagged_area_frac") is not None:
        items.add("Q", "tags", "tagged_area", tr["tagged_area_frac"], 1.0, value=tr["tagged_area_frac"])
    if gt.entities:
        ok = [eid for eid in gt.entities if eid in md.body and md.body[eid].present.any()
              and not (md.body[eid].dup[md.body[eid].present].mean() > 0.1)]
        items.add("Q", "ids", "separable_ids", len(ok) / len(gt.entities), 1.0, value=len(ok))
    if md.parts:
        good = sum(1 for p in md.parts if not DEFAULT_NAME.match(p["name"]))
        items.add("Q", "names", "named_objects", good / len(md.parts), 1.0, value=good)
        org = sum(1 for p in md.parts if p["parent"] or any(c != "Scene Collection" for c in p["collections"]))
        items.add("Q", "hierarchy", "organised", min(1.0, (org / len(md.parts)) / 0.5), 1.0, value=org)
    if gt.tracks:
        h = (md.doc.get("handlers") or {})
        nh = int(h.get("frame_change_pre", 0)) + int(h.get("frame_change_post", 0))
        moving = sum(1 for eid in gt.tracks if eid in ctx.timelines and np.nanmax(ctx.timelines[eid].speed) > 0.05)
        items.add("Q", "animation", "keyframed_motion", (moving / len(gt.tracks)) * (0.5 if nh else 1.0), 1.0,
                  value={"moving": moving, "handlers": nh})
