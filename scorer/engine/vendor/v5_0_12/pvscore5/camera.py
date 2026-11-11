"""Camera layer: GT<->model time warp and view matching, C1 (placement / framing / move), C2 (cuts and order)."""
from __future__ import annotations

import math

import numpy as np

from . import fixes, geom
from .gt import cam_pose

MOVE_TYPES = ("static", "pan", "tilt", "roll", "dolly", "truck", "pedestal", "zoom", "orbit", "follow")
NEAR_TYPES = {frozenset(("pan", "orbit")): 0.5, frozenset(("truck", "orbit")): 0.5, frozenset(("dolly", "zoom")): 0.5,
              frozenset(("follow", "truck")): 0.5, frozenset(("follow", "dolly")): 0.5,
              frozenset(("pan", "follow")): 0.25, frozenset(("pedestal", "tilt")): 0.25}
FRAMING = ("wide", "medium", "close")
DIR_AXIS = {"forward": "in", "backward": "out", "in": "in", "out": "out"}


def norm_move(t, d):
    """5.0.8: a GT move in shot-list words -> the (type, dir) alternatives decompose() can return.
    dolly_in / dolly_out and dolly forward / backward -> dolly in / out; truck forward / backward -> dolly (the
    camera moves along its look); crane up / down / none -> pedestal; crane left / right -> truck, crane forward /
    backward -> dolly, either also as a pedestal of any direction; a follow has no direction."""
    if t in ("dolly_in", "dolly_out"):
        return [("dolly", "in" if t == "dolly_in" else "out")]
    if t in ("dolly", "truck") and d in DIR_AXIS:
        return [("dolly", DIR_AXIS[d])]
    if t == "crane":
        if d in (None, "up", "down"):
            return [("pedestal", d)]
        side = ("truck", d) if d in ("left", "right") else ("dolly", DIR_AXIS.get(d))
        return [side, ("pedestal", None)]
    if t == "follow":
        return [("follow", None)]
    return [(t, d)]


def shot_frames(md, t0m, t1m):
    """First and last model frame of model time [t0m, t1m): frames k with t0m <= k / fps < t1m (at least one).
    5.0.8: a shot's end time is the next shot's first frame; rounding it (t_to_frame_index) crossed the cut."""
    a = min(md.n_frames - 1, max(0, int(math.ceil(t0m * md.fps - 1e-6))))
    b = min(md.n_frames - 1, max(a, int(math.ceil(t1m * md.fps - 1e-6)) - 1))
    return a, b


def gt_cut_bounds(gt):
    """GT boundaries as (time, transition type) for shots after the first."""
    return [(sh.t0, sh.transition_in or "cut") for sh in gt.shots[1:]]


def cross_space_dissolves(ctx):
    """5.0.12 (D15): GT times of dissolves between two shots in different spaces."""
    gt = ctx.gt
    if not fixes.on("D15") or len(gt.spaces) < 2:
        return []
    s0 = gt.spaces[0].id
    return [b.t0 for a, b in zip(gt.shots, gt.shots[1:])
            if (b.transition_in or "cut") == "dissolve" and (a.space or s0) != (b.space or s0)]


def build_warp(ctx):
    gt, md = ctx.gt, ctx.model
    gcuts = [t for t, k in gt_cut_bounds(gt) if k == "cut"]
    mcuts = list(md.cut_times)
    zero = ctx.TH["cut_s"][1]
    pairs = []
    if gcuts and mcuts:
        D = np.abs(np.array(gcuts)[:, None] - np.array(mcuts)[None])
        for i, j in geom.hungarian(np.where(D <= zero, D, np.inf)):
            pairs.append((gcuts[i], mcuts[j]))
    pairs.sort()
    mono = []
    for g, m in pairs:
        if not mono or m > mono[-1][1]:
            mono.append((g, m))
    T_g = max(gt.duration, 1e-3)
    T_m = md.duration
    xs = [0.0] + [g for g, _ in mono] + [T_g]
    ys = [0.0] + [m for _, m in mono] + [T_m]
    ctx.cut_pairs = pairs
    ctx.warp = lambda t: float(np.interp(t, xs, ys))


def view_match(ctx):
    """Image items: GT view -> model shot by aligned camera pose (fallback: order)."""
    gt, md = ctx.gt, ctx.model
    views = [sh for sh in gt.shots if sh.camera]
    ctx.view_match = {}
    if md.shots:
        # 5.0.6: a view whose GT has no camera (the photos' recon failed) is matched by order; it is scored in P1
        # only (no C1 placement, no C2 view order)
        for i, sh in enumerate(gt.shots):
            if not sh.camera and i < len(md.shots):
                ctx.view_match[sh.id] = i
    if not views or not md.shots:
        return
    al = ctx.aligns[gt.spaces[0].id]
    mids = [(s["k0"] + s["k1"]) // 2 for s in md.shots]
    if al.ok:
        C = np.zeros((len(views), len(mids)))
        for i, sh in enumerate(views):
            pg, fg, _, _, _ = cam_pose(sh.camera[0])
            for j, k in enumerate(mids):
                if not md.cam_ok[k]:
                    C[i, j] = np.inf
                    continue
                pm = al.xy(md.cam_pos[k][None])[0]
                fm = al.dir3(md.cam_fwd[k])
                ang = math.degrees(math.acos(np.clip(np.dot(fm, fg) / max(1e-9, np.linalg.norm(fm)), -1, 1)))
                C[i, j] = np.linalg.norm(pm - pg[:2]) / al.L + ang / 45.0
        for i, j in geom.hungarian(C):
            ctx.view_match[views[i].id] = j
    else:
        for i, sh in enumerate(views):
            if i < len(mids):
                ctx.view_match[sh.id] = i


def model_time_for(ctx, sh, t):
    """(model shot index, model time) where GT shot sh at GT time t is evaluated."""
    md = ctx.model
    if ctx.gt.modality == "image":
        j = ctx.view_match.get(sh.id)
        if j is None:
            return None, None
        s = md.shots[j]
        return j, (s["k0"] + s["k1"]) / 2 / md.fps
    tm = ctx.warp(t)
    k = md.t_to_frame_index(tm)
    return md.shot_of_frame(k), tm


# ------------------------------------------------------------------------------------ camera decomposition
def _bank(R):
    """5.2 (M11): bank (deg) of a camera pose: the rotation of its right vector about its forward axis away from the
    level right vector (forward x world up); > 0 = the right side tips down. The third angle beside pan (yaw of the
    forward vector) and tilt (its pitch), with the same singularity at a camera looking straight up / down."""
    f, r = -R[:, 2], R[:, 0]
    h = np.cross(f, np.array([0.0, 0.0, 1.0]))
    n = float(np.linalg.norm(h))
    if n < 1e-9:
        return 0.0
    h = h / n
    return math.degrees(math.atan2(-float(r @ np.cross(h, f)), float(r @ h)))


def decompose(p0, R0, h0, p1, R1, h1, subj0=None, subj1=None, L=10.0, metric_s=1.0):
    """Dominant camera move between two poses: (type, direction, amount, magnitudes). metric_s: GT metres per unit
    of p0 / p1 (5.2 M12: the static translation test is in GT metres)."""
    d = p1 - p0
    right, up, back = R0[:, 0], R0[:, 1], R0[:, 2]
    truck, ped, dolly = float(d @ right), float(d @ up), float(-d @ back)
    f0, f1 = -R0[:, 2], -R1[:, 2]
    yaw0, yaw1 = math.degrees(math.atan2(f0[1], f0[0])), math.degrees(math.atan2(f1[1], f1[0]))
    pan = geom.wrap180(yaw1 - yaw0)
    tilt = math.degrees(math.asin(np.clip(f1[2], -1, 1)) - math.asin(np.clip(f0[2], -1, 1)))
    m11 = fixes.on("M11")
    roll = geom.wrap180(_bank(R1) - _bank(R0)) if m11 else 0.0     # 5.2 (M11): measured, was 0
    zoom = math.log(h0 / h1) if h0 and h1 and np.isfinite(h0) and np.isfinite(h1) else 0.0
    unit = max(0.5, 0.05 * L)
    mags = {"pan": abs(pan) / 30, "tilt": abs(tilt) / 30, "truck": abs(truck) / unit, "pedestal": abs(ped) / unit,
            "dolly": abs(dolly) / unit, "zoom": abs(zoom) / math.log(1.3), "roll": abs(roll) / 30 if m11 else roll}
    orbit = None
    if subj0 is not None and subj1 is not None:
        a0 = geom.yaw_of(p0[:2] - subj0[:2])
        a1 = geom.yaw_of(p1[:2] - subj1[:2])
        dorb = geom.wrap180(a1 - a0)
        sd = np.asarray(subj1[:2]) - np.asarray(subj0[:2])
        if abs(dorb) > 20 and np.linalg.norm(sd) < 0.5 * max(0.5, np.linalg.norm(d[:2])):
            orbit = dorb
            mags["orbit"] = abs(dorb) / 30 + 0.5
        if np.linalg.norm(sd) > 0.5 and np.linalg.norm(d[:2]) > 0.5:
            cos = float(np.dot(sd, d[:2]) / (np.linalg.norm(sd) * np.linalg.norm(d[:2])))
            ratio = np.linalg.norm(d[:2]) / np.linalg.norm(sd)
            if cos > 0.7 and 0.5 < ratio < 2.0:
                mags["follow"] = 1.0 + max(mags["truck"], mags["dolly"])
    dn = np.linalg.norm(d) * metric_s if fixes.on("M12") else np.linalg.norm(d)     # 5.2 (M12): GT metres
    if (dn < 0.1 and abs(pan) < 3 and abs(tilt) < 3 and abs(zoom) < 0.05 and abs(roll) < 3):
        return "static", None, 0.0, mags
    typ = max(mags, key=mags.get)
    dirs = {"pan": ("left" if pan > 0 else "right", abs(pan)), "tilt": ("up" if tilt > 0 else "down", abs(tilt)),
            "truck": ("right" if truck > 0 else "left", abs(truck)), "pedestal": ("up" if ped > 0 else "down", abs(ped)),
            "dolly": ("in" if dolly > 0 else "out", abs(dolly)), "zoom": ("in" if zoom > 0 else "out", math.exp(abs(zoom))),
            "orbit": ("left" if (orbit or 0) > 0 else "right", abs(orbit or 0)), "follow": (None, float(np.linalg.norm(d))),
            "roll": ("right" if roll > 0 else "left", abs(roll)) if m11 else (None, 0.0)}
    dr, am = dirs[typ]
    mags = dict(mags)
    mags["_dirs"] = dirs     # 5.0.12 (D12): signed components, read by move_score for the GT label's own component
    return typ, dr, am, mags


COMPONENT_TYPES = ("truck", "pedestal", "dolly", "pan", "tilt", "zoom")


def move_score(gm, mm, TH, comps=None, comp_min=None, metric_s=1.0):
    gt_a = gm.get("amount")
    t, d, a = mm
    best = 0.0
    for gt_t, gt_d in norm_move(gm.get("type"), gm.get("dir")):
        if gt_t == "static" or t == "static":
            best = max(best, 1.0 if gt_t == t else 0.0)
            continue
        comp_t = gt_t in COMPONENT_TYPES or (gt_t == "roll" and fixes.on("M11"))     # 5.2 (M11): roll as pan / tilt
        if comps is not None and t != gt_t and comp_t and comps.get(gt_t, 0.0) >= comp_min:
            # 5.0.12 (D12): the GT label's own component of the model move (never any move): scored as that type
            # with its own direction and amount
            cd, ca = comps["_dirs"][gt_t]
            if gt_t in ("truck", "pedestal", "dolly"):
                ca = ca * metric_s
            ds = 1.0 if (gt_d is None or cd == gt_d) else 0.0
            if gt_a and ca:
                as_ = geom.lin(abs(math.log(max(1e-6, ca) / max(1e-6, gt_a))), *TH["move_amount_log"])
            else:
                as_ = 1.0 if not gt_a else 0.0
            best = max(best, 0.5 + 0.25 * ds + 0.25 * as_)
        ts = 1.0 if t == gt_t else NEAR_TYPES.get(frozenset((t, gt_t)), 0.0)
        ds = (1.0 if (gt_d is None or d == gt_d) else 0.0) if ts > 0 else 0.0
        if gt_a and a and ts > 0:
            as_ = geom.lin(abs(math.log(max(1e-6, a) / max(1e-6, gt_a))), *TH["move_amount_log"])
        else:
            as_ = 1.0 if ts > 0 and not gt_a else 0.0
        best = max(best, 0.5 * ts + 0.25 * ds + 0.25 * as_)
    return best


# ---------------------------------------------------------------------------------------- subject helpers
def model_subject(ctx, sid, t):
    """(aligned xy, base/top heights, facing) of a model body at GT time t, or None."""
    md = ctx.model
    b = md.body.get(sid)
    if b is None:
        return None
    tm = ctx.warp(t) if ctx.gt.modality != "image" else t
    ai = md.t_to_ai(tm)
    if not b.present[ai]:
        idx = np.where(b.present)[0]
        if not len(idx):
            return None
        ai = int(idx[np.argmin(np.abs(idx - ai))])
        if abs(md.an_t[ai] - tm) > 0.5:
            return None
    sp = ctx.space_at(sid, t)
    al = ctx.aligns[sp]
    f = b.facing[ai]
    return {"xy": al.xy(b.center[ai][None])[0], "xy_m": b.center[ai], "base": b.base[ai], "top": b.top[ai],
            "facing": None if np.isnan(f) else al.yaw(f) % 360, "facing_m": None if np.isnan(f) else f % 360,
            "ai": ai, "lo": b.lo[ai], "hi": b.hi[ai],
            "facing_unk": bool(np.isnan(f) and fixes.on("M14") and b.facing_unk[ai])}     # 5.2 (M14)


def side_of(cam_xy_m, ms):
    """Camera side relative to a model subject's facing (deg, -180..180), in the model's own frame (5.0.9: it needs
    no alignment; the alignment is a rotation, so an aligned space gives the same value)."""
    if ms is None or ms["facing_m"] is None:
        return None
    return geom.wrap180(geom.yaw_of(np.asarray(cam_xy_m[:2], float) - ms["xy_m"]) - ms["facing_m"])


def framing_bucket(frac):
    return "close" if frac >= 0.6 else ("medium" if frac >= 0.3 else "wide")


def angle_class(pitch_deg):
    return "high" if pitch_deg < -15 else ("low" if pitch_deg > 8 else "eye")


def rel_az_class(d):
    return "front" if d <= 45 else ("back" if d >= 135 else "side")


# ------------------------------------------------------------------------------------------------------ C1
def score_C1(ctx, items):
    gt, md, TH = ctx.gt, ctx.model, ctx.TH
    if not gt.shots:
        return
    for sh in gt.shots:
        sp = sh.space or gt.spaces[0].id
        al = ctx.aligns[sp]
        L = al.L
        floor = gt.space_by[sp].floor_z if sp in gt.space_by else 0.0
        sloped = sp in gt.space_by and gt.space_by[sp].sloped
        if gt.modality == "image" and not sh.camera:
            continue
        mshot, _ = model_time_for(ctx, sh, (sh.t0 + sh.t1) / 2)
        if mshot is None:
            for b in ("position", "direction"):
                items.add("C1", b, "%s:%s" % (b, sh.id), 0.0, 1.0, value="no model shot")
            continue
        acc = {"position": [], "height": [], "direction": [], "fov": []}
        if gt.modality != "image":
            # 5.0.8: GT samples sit at the shot's start and end times; frames are kept inside the warped shot
            kf0, kf1 = shot_frames(md, ctx.warp(sh.t0), ctx.warp(sh.t1))
            k_at = lambda t: min(kf1, max(kf0, md.t_to_frame_index(ctx.warp(t))))
        for c in sh.camera:
            pg, fg, ug, hg, _ = cam_pose(c)
            if gt.modality == "image":
                s = md.shots[mshot]
                k = (s["k0"] + s["k1"]) // 2
            else:
                k = k_at(c.get("t", sh.t0))
            if not md.cam_ok[k]:
                continue
            pm = md.cam_pos[k]
            # 5.0.2: only the parts a sample gives are compared; no height in a sloped space
            has_xy, has_dir = bool(np.isfinite(pg[:2]).all()), bool(np.isfinite(fg).all())
            if al.ok:
                if has_xy:
                    d = float(np.linalg.norm(al.xy(pm[None])[0] - pg[:2]))
                    acc["position"].append(geom.lin(d / L, *TH["cam_pos_frac"]))
                if has_dir:
                    fm = al.dir3(md.cam_fwd[k])
                    ang = math.degrees(math.acos(np.clip(np.dot(fm, fg) / max(1e-9, np.linalg.norm(fm)), -1, 1)))
                    acc["direction"].append(geom.lin(ang, *TH["cam_dir_deg"]))
            else:
                if has_xy:
                    acc["position"].append(0.0)
                if has_dir:
                    pitch_m = math.degrees(math.asin(np.clip(md.cam_fwd[k][2], -1, 1)))
                    pitch_g = math.degrees(math.asin(np.clip(fg[2], -1, 1)))
                    acc["direction"].append(0.5 * geom.lin(abs(pitch_m - pitch_g), *TH["cam_dir_deg"]))
            if np.isfinite(pg[2]) and not sloped:
                hm = float(al.h(pm[2]))
                acc["height"].append(geom.lin(abs(hm - (pg[2] - floor)), *TH["cam_height_m"]))
            if hg and np.isfinite(md.hfov[k]):
                acc["fov"].append(geom.lin(abs(math.log(md.hfov[k] / hg)), *TH["fov_log"]))
        for b, v in acc.items():
            if v:
                items.add("C1", b, "%s:%s" % (b, sh.id), float(np.mean(v)), 1.0, n=len(v))
        if gt.modality == "image":
            continue
        s = md.shots[mshot]
        ks = [k for k in range(kf0, kf1 + 1) if md.cam_ok[k]]
        if not ks:
            continue
        kmid = ks[len(ks) // 2]
        tmid = (sh.t0 + sh.t1) / 2
        subj = sh.subjects[0] if sh.subjects else None
        # subject-relative placement
        placed = [c for c in sh.camera if np.isfinite(cam_pose(c)[0][:2]).all()]
        if subj and subj in gt.tracks and gt.modality != "image":
            # 5.0.3: only where the subject's GT position is known (not past its track, not inside a long moving gap)
            placed = [c for c in placed if gt.tracks[subj].known(c.get("t", sh.t0))]
        if subj and subj in gt.tracks and placed:
            tr = gt.tracks[subj]
            c = min(placed, key=lambda c: abs(c.get("t", sh.t0) - tmid))
            tc = c.get("t", sh.t0)
            pg, fg, _, _, _ = cam_pose(c)
            sg = tr.at(tc)
            ms = model_subject(ctx, subj, tc)
            if ms is None:
                items.add("C1", "subject", "subject:%s:%s" % (sh.id, subj), 0.0, 1.0, value="subject missing")
            else:
                k = k_at(tc)
                dg = float(np.linalg.norm(pg[:2] - sg))
                dm = float(np.linalg.norm(md.cam_pos[k][:2] - ms["xy_m"])) * al.sim.s
                sc = [geom.lin(abs(math.log(max(0.05, dm) / max(0.05, dg))), *TH["cam_dist_log"])]
                yg = tr.yaw_at(tc, ctx.P["gt_yaw_window_s"]) if fixes.on("S22") else tr.yaw_at(tc)
                sidem = side_of(md.cam_pos[k], ms)
                if yg is not None and sidem is not None:
                    sideg = geom.wrap180(geom.yaw_of(pg[:2] - sg) - yg)
                    sc.append(geom.lin(abs(geom.wrap180(sideg - sidem)), *TH["cam_azimuth_deg"]))
                items.add("C1", "subject", "subject:%s:%s" % (sh.id, subj), float(np.mean(sc)), 1.0,
                          value={"dist_gt": round(dg, 2), "dist_model": round(dm, 2)})
        # keep in frame
        for sid in sh.keep_in_frame:
            b = md.body.get(sid)
            if b is None:
                items.add("C1", "in_frame", "in_frame:%s:%s" % (sh.id, sid), 0.0, 1.0, value="missing")
                continue
            hits, n = 0, 0
            for k in ks[::max(1, len(ks) // 24)]:
                ai = md.t_to_ai(k / md.fps)
                n += 1
                if not b.present[ai]:
                    continue
                c3 = np.array([b.center[ai][0], b.center[ai][1], 0.5 * (b.base[ai] + b.top[ai])])
                u, v, dd = md.project(k, c3)
                if dd[0] > 0 and 0 <= u[0] <= 1 and 0 <= v[0] <= 1:
                    hits += 1
            frac = hits / max(1, n)
            items.add("C1", "in_frame", "in_frame:%s:%s" % (sh.id, sid), geom.lin_up(frac, *TH["in_frame_q"]), 1.0,
                      value=round(frac, 3))
        # framing bucket of the main subject
        gbucket = sh.spec.get("framing")
        if subj and not gbucket and sh.camera and subj in gt.tracks:
            gbucket = _gt_framing(gt, sh, subj)
        if subj and gbucket:
            b = md.body.get(subj)
            mb = None
            if b is not None:
                ai = md.t_to_ai(kmid / md.fps)
                if b.present[ai]:
                    top = np.array([b.center[ai][0], b.center[ai][1], b.top[ai]])
                    bot = np.array([b.center[ai][0], b.center[ai][1], b.base[ai]])
                    _, v, dd = md.project(kmid, np.stack([top, bot]))
                    if dd.min() > 0:
                        mb = framing_bucket(abs(v[1] - v[0]))
            sc = 0.0 if mb is None else (1.0 if mb == gbucket else (0.5 if abs(FRAMING.index(mb) - FRAMING.index(gbucket)) == 1 else 0.0))
            items.add("C1", "framing", "framing:%s" % sh.id, sc, 1.0, value={"gt": gbucket, "model": mb})
        # camera move
        gmove = sh.move or sh.spec.get("move") and {"type": sh.spec.get("move"), "dir": sh.spec.get("move_dir"),
                                                   "amount": sh.spec.get("move_amount")}
        k0, k1 = ks[0], ks[-1]
        msegs = getattr(sh, "move_segments", None) or []     # 5.0.10 (shot objects built elsewhere may lack it)
        if not gmove and not msegs and len(sh.camera) >= 2:
            gmove = _gt_move(ctx, sh, subj)
            # a move read from GT camera samples is compared over the same (warped) sample times
            ka = k_at(sh.camera[0].get("t", sh.t0))
            kb = k_at(sh.camera[-1].get("t", sh.t1))
            if gmove and md.cam_ok[ka] and md.cam_ok[kb] and kb > ka:
                k0, k1 = ka, kb
        d12 = fixes.on("D12")
        if gmove:
            mm = _model_move(md, subj, k0, k1, L, al)
            items.add("C1", "move", "move:%s" % sh.id,
                      move_score(gmove, mm[:3], TH, mm[3] if d12 else None, ctx.P.get("move_component_min"),
                                 al.sim.s), 1.0,
                      value={"gt": gmove, "model": [mm[0], mm[1], round(float(mm[2]), 3)]})
        elif msegs:
            # 5.0.10: one item per GT move segment, read over the model frames of its warped window; the segments
            # of a shot share the weight of one move item
            w = 1.0 / len(msegs)
            for sg in msegs:
                a, b = shot_frames(md, ctx.warp(sg["t"][0]), ctx.warp(sg["t"][1]))
                kk = [k for k in ks if a <= k <= b]
                gm = {x: sg[x] for x in ("t", "type", "dir", "amount") if x in sg}
                name = "move:%s@%.2f" % (sh.id, sg["t"][0])
                if len(kk) < 2:
                    items.add("C1", "move", name, 0.0, w, value={"gt": gm, "model": "no model frames"})
                    continue
                mm = _model_move(md, subj, kk[0], kk[-1], L, al)
                items.add("C1", "move", name,
                          move_score(gm, mm[:3], TH, mm[3] if d12 else None, ctx.P.get("move_component_min"),
                                     al.sim.s), w,
                          value={"gt": gm, "model": [mm[0], mm[1], round(float(mm[2]), 3)]})
        # qualitative spec (text items)
        if sh.spec and not sh.camera:
            _spec_items(ctx, sh, ks, kmid, subj, al, items)


def _model_move(md, subj, k0, k1, L, al):
    """decompose() of the model camera from frame k0 to k1 (subject centres for follow / orbit), metric amounts."""
    s0 = s1 = None
    if subj:
        b = md.body.get(subj)
        if b is not None:
            a0, a1 = md.t_to_ai(k0 / md.fps), md.t_to_ai(k1 / md.fps)
            if b.present[a0] and b.present[a1]:
                s0, s1 = b.center[a0], b.center[a1]
    mm = decompose(md.cam_pos[k0], md.cam_R[k0], md.hfov[k0], md.cam_pos[k1], md.cam_R[k1], md.hfov[k1],
                   s0, s1, L / max(al.sim.s, 1e-6), metric_s=al.sim.s)
    if mm[2] and mm[0] in ("truck", "pedestal", "dolly", "follow"):
        mm = (mm[0], mm[1], mm[2] * al.sim.s, mm[3])
    return mm


def _gt_framing(gt, sh, subj):
    c = sh.camera[len(sh.camera) // 2]
    pg, fg, ug, hg, vg = cam_pose(c)
    if not (np.isfinite(pg).all() and np.isfinite(fg).all()):
        return None
    tr = gt.tracks[subj]
    sg = tr.at(c.get("t", sh.t0))
    post = tr.label_at(c.get("t", sh.t0), "posture") or "stand"
    hgt = {"stand": 1.7, "sit": 1.25, "crouch": 1.0, "lie": 0.4}.get(post, 1.7)
    floor = gt.space_by.get(sh.space or gt.spaces[0].id).floor_z if gt.spaces else 0.0
    d = float(np.dot(np.array([sg[0], sg[1], floor + hgt / 2]) - pg, fg))
    if d <= 0 or not vg:
        return None
    frac = hgt / (2 * d * math.tan(math.radians(vg) / 2))
    return framing_bucket(frac)


def _gt_move(ctx, sh, subj):
    gt = ctx.gt
    c0, c1 = sh.camera[0], sh.camera[-1]
    p0, f0, u0, h0, _ = cam_pose(c0)
    p1, f1, u1, h1, _ = cam_pose(c1)
    if not all(np.isfinite(x).all() for x in (p0, f0, p1, f1)):
        return None

    def R_of(f, u):
        f = f / np.linalg.norm(f)
        r = np.cross(f, u)
        r /= max(1e-9, np.linalg.norm(r))
        u2 = np.cross(r, f)
        return np.stack([r, u2, -f], 1)
    s0 = s1 = None
    if subj and subj in gt.tracks:
        s0, s1 = gt.tracks[subj].at(c0.get("t", sh.t0)), gt.tracks[subj].at(c1.get("t", sh.t1))
    sp = sh.space or gt.spaces[0].id
    t, d, a, _ = decompose(p0, R_of(f0, u0), h0, p1, R_of(f1, u1), h1, s0, s1, ctx.aligns[sp].L)
    return {"type": t, "dir": d, "amount": a}


def _spec_items(ctx, sh, ks, kmid, subj, al, items):
    md, TH = ctx.model, ctx.TH
    spec = sh.spec
    if spec.get("angle"):
        pitch = math.degrees(math.asin(np.clip(md.cam_fwd[kmid][2], -1, 1)))
        items.add("C1", "spec", "angle:%s" % sh.id, 1.0 if angle_class(pitch) == spec["angle"] else 0.0, 1.0,
                  value={"gt": spec["angle"], "model": angle_class(pitch)})
    if spec.get("rel_azimuth") and subj:
        ms_ = model_subject(ctx, subj, (sh.t0 + sh.t1) / 2)
        side = side_of(md.cam_pos[kmid], ms_)
        cls = None if side is None else rel_az_class(abs(side))
        if not (ms_ is not None and ms_["facing_unk"]):     # 5.2 (M14): subject facing unknown, not scored
            items.add("C1", "spec", "rel_azimuth:%s" % sh.id, 1.0 if cls == spec["rel_azimuth"] else 0.0, 1.0,
                      value={"gt": spec["rel_azimuth"], "model": cls})


# ------------------------------------------------------------------------------------------------------ C2
def score_C2(ctx, items):
    gt, md, TH = ctx.gt, ctx.model, ctx.TH
    if not gt.shots:
        return
    if gt.modality == "image":
        cam_views = [sh for sh in gt.shots if sh.camera]
        views = [sh for sh in cam_views if sh.id in ctx.view_match]
        order = [ctx.view_match[sh.id] for sh in views]
        n = len(order)
        if len(cam_views) >= 2:
            if n >= 2:
                conc = disc = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if order[i] < order[j]:
                            conc += 1
                        elif order[i] > order[j]:
                            disc += 1
                tau = (conc - disc) / max(1, conc + disc)
            else:
                tau = -1.0
            cover = n / len(cam_views)
            items.add("C2", "view_order", "view_order", max(0.0, tau) * cover, 1.0, value=round(tau, 3), matched=n)
        return
    bounds = gt_cut_bounds(gt)
    n_gt_cuts = sum(1 for _, k in bounds if k == "cut")
    n_m = len(md.shots)
    diff = abs(n_m - (n_gt_cuts + 1))
    xd = cross_space_dissolves(ctx)
    xwin = ctx.P.get("dissolve_win_s", 1.0)
    if xd:
        # 5.0.12 (D15): a model cut may stand for a dissolve between two spaces
        diff = max(0, (n_gt_cuts + 1) - n_m, n_m - (n_gt_cuts + 1 + len(xd)))
    items.add("C2", "count", "shot_count", geom.lin(diff, 0, max(2, 0.5 * (n_gt_cuts + 1))), 1.0,
              value={"gt": n_gt_cuts + 1, "model": n_m})
    matched_m = set()
    for g, m in ctx.cut_pairs:
        matched_m.add(m)
    pair_of = dict(ctx.cut_pairs)
    for t, k in bounds:
        if k != "cut":
            continue
        m = pair_of.get(t)
        sc = geom.lin(abs(m - t), *TH["cut_s"]) if m is not None else 0.0
        items.add("C2", "cuts", "cut@%.2f" % t, sc, 1.0, value=None if m is None else round(m - t, 3))
    xd_m = [ctx.warp(t) for t in xd]
    for m, why in zip(md.cut_times, md.cut_reasons):
        if m not in matched_m:
            if any(abs(m - x) <= xwin for x in xd_m):
                continue        # 5.0.12 (D15): the dissolve between two spaces, not an extra cut
            items.add("C2", "cuts", "extra_cut@%.2f" % m, 0.0, 0.5, value=why)
    # transitions
    for t, k in bounds:
        tm = ctx.warp(t)
        near = [m for m in md.cut_times if abs(m - tm) <= 1.0]
        if k == "cut":
            sc = 1.0 if near else _moved(md, tm, 1.0) * 0.5
        elif t in xd and any(abs(m - tm) <= xwin for m in md.cut_times):
            sc = 1.0            # 5.0.12 (D15): a model cut at a dissolve between two spaces
        else:
            moved = _moved(md, tm, 1.5)
            sc = (1.0 if moved else 0.25) if not near else 0.25
        items.add("C2", "transitions", "transition@%.2f:%s" % (t, k), sc, 1.0,
                  value={"model_cut_near": bool(near)})
    # shot content order: (space, main subject) per GT shot window
    toks_g, toks_m = [], []
    for sh in gt.shots:
        toks_g.append((sh.space, sh.subjects[0] if sh.subjects else None))
        toks_m.append(_model_token(ctx, sh))
    if len(gt.shots) >= 2:
        sim = []
        for g, m in zip(toks_g, toks_m):
            s_ = (0.5 if g[0] == m[0] else 0.0) + (0.5 if (g[1] is None or g[1] == m[1]) else 0.0)
            sim.append(s_)
        items.add("C2", "shot_order", "shot_tokens", float(np.mean(sim)), 1.0,
                  value={"gt": toks_g, "model": toks_m})
        sg = geom.collapse([t[0] for t in toks_g])
        smod = geom.collapse([_camera_space(ctx, (s["k0"] + s["k1"]) // 2) for s in md.shots])
        if len(sg) >= 2:
            items.add("C2", "space_order", "space_order", geom.seq_similarity(sg, smod) or 0.0, 1.0,
                      value={"gt": sg, "model": smod})
    lat_text = gt.modality == "text" and ctx.P.get("screen_dir_single_text") == "A_lat"    # 5.0.11
    for sh in gt.shots:
        for sid, want in (sh.screen_dir or {}).items():
            if lat_text:
                got, info = _lateral_dir(ctx, sid, sh.t0, sh.t1)
                value = {"gt": want, "model": got, "reading": "A_lat", **info}
            else:
                got = _screen_dir(ctx, sh, sid)
                value = {"gt": want, "model": got}
            items.add("C2", "screen_dir", "screen_dir:%s:%s" % (sh.id, sid), 1.0 if got == want else 0.0, 1.0,
                      value=value)
        # 5.0.10: a subject whose direction changes inside the shot: one item per segment, read over the segment's
        # own window and sharing the weight of one label; the segment direction is the subject's heading seen from
        # the camera (_segment_dir), so away / toward segments are measured too
        for sid, segs in (getattr(sh, "screen_dir_segments", None) or {}).items():
            for sg in segs:
                got, info = _segment_dir(ctx, sid, *sg["t"])
                items.add("C2", "screen_dir", "screen_dir:%s:%s@%.2f" % (sh.id, sid, sg["t"][0]),
                          1.0 if got == sg["dir"] else 0.0, 1.0 / len(segs),
                          value={"gt": sg["dir"], "t": sg["t"], "model": got, **info})
    # durations of cut-separated GT shots vs the model segment holding their warped midpoint
    groups = []
    for sh in gt.shots:
        if groups and (sh.transition_in or "cut") != "cut":
            groups[-1][1] = sh.t1
        else:
            groups.append([sh.t0, sh.t1])
    if len(groups) >= 1 and len(gt.shots) >= 2:
        for g0, g1 in groups:
            tm = ctx.warp((g0 + g1) / 2)
            s = md.shots[md.shot_of_frame(md.t_to_frame_index(tm))]
            dm = s["t1"] - s["t0"]
            err = abs(math.log(max(1e-3, dm) / max(1e-3, g1 - g0)))
            items.add("C2", "durations", "dur@%.2f" % g0, geom.lin(err, *TH["shot_dur_log"]), 1.0,
                      value={"gt": round(g1 - g0, 3), "model": round(dm, 3)})


def _moved(md, tm, win):
    k0 = md.t_to_frame_index(tm - win)
    k1 = md.t_to_frame_index(tm + win)
    if not (md.cam_ok[k0] and md.cam_ok[k1]):
        return False
    d = np.linalg.norm(md.cam_pos[k1] - md.cam_pos[k0])
    R = md.cam_R[k1] @ md.cam_R[k0].T
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
    return bool(d > 0.5 or ang > 20)


def _camera_space(ctx, k):
    """GT space the model camera is in at frame k (aligned position), or the subject's space."""
    md, gt = ctx.model, ctx.gt
    if not md.cam_ok[k]:
        return None
    if len(gt.spaces) == 1:
        return gt.spaces[0].id
    best, bd = None, None
    for s in gt.spaces:
        p = ctx.aligns[s.id].xy(md.cam_pos[k][None])[0]
        lo, hi = s.extent
        d = float(np.max(np.maximum(lo - p, p - hi)))
        if s.outline is not None and geom.point_in_poly(p, s.outline):
            d = -1.0
        if bd is None or d < bd:
            best, bd = s.id, d
    return best


def _model_token(ctx, sh):
    md, gt = ctx.model, ctx.gt
    tm = ctx.warp((sh.t0 + sh.t1) / 2)
    k = md.t_to_frame_index(tm)
    space = _camera_space(ctx, k)
    best, barea = None, 0.0
    ai = md.t_to_ai(tm)
    s0 = sh.subjects[0] if (fixes.on("D12b") and sh.subjects) else None
    cen = []     # 5.0.12 (D12b, amended 10-19): (distance of the projected box centre to the frame centre, eid)
    for eid, b in md.body.items():
        if (eid not in gt.tracks and eid not in gt.entities) or not b.present[ai]:   # 5.0.5: every GT entity
            continue
        corners = np.array([[x, y, z] for x in (b.lo[ai][0], b.hi[ai][0]) for y in (b.lo[ai][1], b.hi[ai][1])
                            for z in (b.lo[ai][2], b.hi[ai][2])])
        u, v, d = md.project(k, corners)
        if (d <= 0).all():
            continue
        u, v = np.clip(u[d > 0], 0, 1), np.clip(v[d > 0], 0, 1)
        area = float((u.max() - u.min()) * (v.max() - v.min()))
        if area > barea:
            best, barea = eid, area
        if s0 is not None and area > 0:
            cen.append((float(np.hypot(0.5 * (u.min() + u.max()) - 0.5, 0.5 * (v.min() + v.max()) - 0.5)), eid))
    # 5.0.12 (D12b): the GT shot's own subject when it is in view. Amended 10-19 (G3 guard): only when it is the
    # most central body in view, untied (a frame showing a row of bodies does not single the subject out); else the
    # largest box (5.0.11)
    cen.sort()
    if s0 is not None and cen and cen[0][1] == s0 and (len(cen) == 1 or cen[1][0] > cen[0][0] + 1e-9):
        best = s0
    # a subject visible in frame decides the space when the camera stands outside every space
    if best is not None and len(gt.spaces) > 1:
        sp = ctx.space_at(best, (sh.t0 + sh.t1) / 2, model=True)
        space = sp or space
    return (space, best)


def _screen_dir(ctx, sh, sid, t0=None, t1=None):
    """left_to_right / right_to_left / static over [t0, t1] (default the shot) without its first and last 10 %"""
    md = ctx.model
    b = md.body.get(sid)
    if b is None:
        return None
    t0, t1 = sh.t0 if t0 is None else t0, sh.t1 if t1 is None else t1
    t0m, t1m = ctx.warp(t0 + 0.1 * (t1 - t0)), ctx.warp(t1 - 0.1 * (t1 - t0))
    a0, a1 = md.t_to_ai(t0m), md.t_to_ai(t1m)
    if not (b.present[a0] and b.present[a1]):
        return None
    k0, k1 = md.t_to_frame_index(t0m), md.t_to_frame_index(t1m)
    c0 = np.array([*b.center[a0], 0.5 * (b.base[a0] + b.top[a0])])
    c1 = np.array([*b.center[a1], 0.5 * (b.base[a1] + b.top[a1])])
    u0 = md.project(k0, c0)[0][0]
    u1 = md.project(k1, c1)[0][0]
    if abs(u1 - u0) < 0.05:
        return "static"
    return "left_to_right" if u1 > u0 else "right_to_left"


def _cam_frame_disp(ctx, sid, t0, t1):
    """Body-centre displacement d between 10 % and 90 % of [t0, t1] (warped), mean body height h, and the unit
    forward f / screen-right r of the camera's horizontal frame at the window midpoint; None if not present."""
    md = ctx.model
    b = md.body.get(sid)
    if b is None:
        return None
    t0m, t1m = ctx.warp(t0 + 0.1 * (t1 - t0)), ctx.warp(t1 - 0.1 * (t1 - t0))
    a0, a1 = md.t_to_ai(t0m), md.t_to_ai(t1m)
    if not (b.present[a0] and b.present[a1]):
        return None
    km = md.t_to_frame_index(0.5 * (t0m + t1m))
    d = np.asarray(b.center[a1], float) - np.asarray(b.center[a0], float)
    h = 0.5 * ((b.top[a0] - b.base[a0]) + (b.top[a1] - b.base[a1]))
    f, r = md.cam_fwd[km][:2], md.cam_R[km][:2, 0]
    if np.linalg.norm(f) < 1e-6:            # camera looking straight down: the screen-up axis stands in for forward
        f = md.cam_R[km][:2, 1]
    f, r = f / max(1e-9, np.linalg.norm(f)), r / max(1e-9, np.linalg.norm(r))
    return d, h, f, r


def _lateral_dir(ctx, sid, t0, t1):
    """5.0.11, text single-label screen_dir (A_lat): the heading seen from the camera as in _segment_dir but without
    the away / toward cone, for the lateral / static vocabulary of single labels: left_to_right / right_to_left by the
    sign of the displacement's screen-right component, static if that component is below seg_dir_static_h body
    heights. A text label is a world heading relative to the camera, so a tracking camera that keeps the subject
    centred still reads the subject's own direction."""
    w = _cam_frame_disp(ctx, sid, t0, t1)
    if w is None:
        return None, {}
    d, h, f, r = w
    lat = float(d @ r)
    info = {"lat_h": round(float(lat / max(1e-6, h)), 3), "disp_h": round(float(np.linalg.norm(d) / max(1e-6, h)), 3),
            "heading_deg": round(math.degrees(math.atan2(lat, float(d @ f))), 1)}
    if abs(lat) < ctx.P["seg_dir_static_h"] * h:
        return "static", info
    return ("left_to_right" if lat > 0 else "right_to_left"), info


def _segment_dir(ctx, sid, t0, t1):
    """5.0.10, screen_dir_segments only: the subject's heading seen from the camera over [t0, t1] without its first
    and last 10 %. The body centre's displacement between the window ends, in the horizontal frame of the camera at
    the window midpoint, at angle th from the view direction: away |th| < seg_dir_away_deg, toward |th| > 180 - it,
    else left_to_right / right_to_left by side; static below seg_dir_static_h body heights. Film grammar: a subject
    a following camera keeps centred still moves the way it walks."""
    P = ctx.P
    w = _cam_frame_disp(ctx, sid, t0, t1)
    if w is None:
        return None, {}
    d, h, f, r = w
    th = math.degrees(math.atan2(float(d @ r), float(d @ f)))
    info = {"disp_h": round(float(np.linalg.norm(d) / max(1e-6, h)), 3), "heading_deg": round(th, 1)}
    if np.linalg.norm(d) < P["seg_dir_static_h"] * h:
        return "static", info
    if abs(th) < P["seg_dir_away_deg"]:
        return "away", info
    if abs(th) > 180.0 - P["seg_dir_away_deg"]:
        return "toward", info
    return ("left_to_right" if th > 0 else "right_to_left"), info
