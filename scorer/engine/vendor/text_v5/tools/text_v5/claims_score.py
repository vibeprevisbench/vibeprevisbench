#!/usr/bin/env python3
"""v5 text-item claims on a pvscore 5.0.x model dump: per-claim results, per-check results and the three v5 blocks
(layout claims, entity claims, camera checks) with the registered chance baseline. Rules: text_items/drafts_v5/
_build/PREREG_v5.json (`claim_types`, `camera_checks`, `matching.model_candidates`, `model_blocks`); how 5.1 uses the
blocks: text_items/drafts_v5/SCORING_v5.md. The judge is claims_core.py (the same code checks the claims on the GT in
build.py and on blind-reader scenes in blind_compare_v5.py).

    # a model dump (needs numpy and pvscore5: in the evaluation environment)
    env/bin/python -I -B tools/text_v5/claims_score.py --claims text_items/drafts_v5/claims/MS50_T01.json \\
        --dump runs_x/.../M50_001 [--out F.json]
    # stdlib self-tests on the GT (GT-as-model, mirrored, shifted, cross-item), no dump needed
    python3 -I -B tools/text_v5/claims_score.py --selftest [--items MS50_T01,...] [--out F.json]

Input: a 5.0.x dump directory (dump.json + dump.npz) and a v5 claims file (drafts_v5/claims/MS50_Txx.json).
Output (JSON): per claim {n, src, block, text, status: true | false | missing, v, note}; per camera check the same;
blocks {P, R, coverage, score, n, n_true, n_missing}; the GT key -> model instance binding; a1_counts_drop = the
5.0.x A1 count item ids that 5.1 leaves out on this item (groups holding an omitted GT object).

Model scene (from_dump), all in the model's own frame (no alignment, no metres; the prompt fixes north = +y,
east = +x):
  static elements  pvscore5 layout instances (Model.layout), untagged left out: footprint = the instance's
                   minimum-area rectangle, height = its z extent, front = the seat facing (seats only), space = its
                   space tag; pvscore 5.1 r4 (scorer PREREG_51r4.json R4-2): an instance whose NORMALISED class
                   (classes.norm, as the class credit) is in the terrain group is kept, flagged terrain (r3 left
                   pvscore5 kind terrain out, by the raw class)
  spaces           box of the xy extents of the layout pieces tagged with the space; floor = pvscore5 model_floor;
                   a GT-sloped space also gets "ground" = claims_core.ground_plane() of the instance bottoms and person
                   bases in it (prereg_v5.0.1 note n1)
  entities         Model.bodies at GT time t read at model time warp(t) (pvscore5 build_warp, nearest analysis frame,
                   absent when none lies within 0.3 s); people / animals: front = the circular medoid of the
                   per-frame facing within +-0.25 s, height = the scene's person height; others: their rectangle and
                   z extent, no front (facing claims compare the long axis: axis_only). A body's space at a time is
                   containment in the space box (bodies carry no space tag)
  cameras          per GT shot, the model camera at frame clamp(frame(warp(t)), warped shot) (pvscore5 5.0.8 rule),
                   pos / forward / up / hfov
  ph               median over people of their 95th-percentile body height (1.75 when the model has no people)
Candidates (PREREG matching.model_candidates): instances whose class has 5.1 r1 class credit > 0 for the GT key's
labels (tools/v51_text_r2/classes.py when importable, else equal normalised class), in the GT object's space or with
no space tag; r4 (PREREG_51r4.json R4-1): a terrain element only for a GT object whose own class is terrain (the same
normalised-class test); then claims_core.Matcher (maximum cardinality, local search that keeps strict gains in claims held).
Blocks (PREREG model_blocks): P = share of true among the block's non-missing claims; coverage = non-missing share;
R = mean P over 200 permutations (random.Random(5) draws the seeds) of the bound static elements' centres within
each model space (binding, entities, spaces and cameras unchanged); score = clip((P - R) / (1 - R), 0, 1) x
coverage. Camera checks: score = P x coverage (no chance correction). R = 1 (no claim can move): score = P x
coverage, flagged `degenerate_R`.
"""
import argparse
import fnmatch
import json
import math
import os
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
sys.path.insert(0, str(HERE))
import claims_core as C  # noqa: E402

C.set_rules(os.environ.get("PV_V5_PREREG") or C.PREREG_PATH)
PRE = C.RULES  # load_cf() switches to the PREREG a claims file was built under (C.use_rules_of, sha checked)
GT_ROOT = Path(os.environ.get("PV_GT_ROOT") or (R / "data" / "rubrics"))
PVSCORE_DIR = Path(os.environ.get("PV_PVSCORE_DIR") or (R / "scorer" / "v5_0"))
CLAIMS_DIR = R / "text_items" / "drafts_v5" / "claims"
PERM_N = int(PRE["model_blocks"]["perm_n"])
PERM_SEED = int(PRE["model_blocks"]["perm_seed"])
SKIP_KINDS = {"untagged_floor", "untagged_wall", "untagged_object"}  # r4: "terrain" no longer skipped (R4-2)

try:                                    # 5.1 r1 class credit (stdlib); a vendored copy may not ship it
    sys.path.insert(0, str(R / "tools" / "v51_text_r2"))
    import classes as CL  # noqa: E402

    def class_credit(key, gcls, mcls):
        return CL.credit(CL.gt_labels(key, gcls), mcls)
    CREDIT_SRC = "tools/v51_text_r2/classes.py (5.1 r1)"
except Exception:  # noqa: BLE001 - fall back to the registered equal-class rule
    def class_credit(key, gcls, mcls):
        a, b = C.norm_word(gcls), C.norm_word(mcls)
        return 1.0 if a and a == b else 0.0
    CREDIT_SRC = "equal normalised class"


# pvscore 5.1 r4 (PREREG_51r4.json R4-1 / R4-2): one terrain test, on the normalised class, for model and GT alike
def _cats_file():
    fs = []
    if CREDIT_SRC.startswith("tools/"):
        fs.append(CL.R / CL.CMAP["vocabulary"])
    fs += [R / "common" / "categories_v5.0.json", PVSCORE_DIR.parent.parent / "common" / "categories_v5.0.json"]
    for f in fs:
        if Path(f).is_file():
            return Path(f)
    raise FileNotFoundError("categories_v5.0.json for the terrain test: none of %s" % [str(f) for f in fs])


TERRAIN = frozenset(c["class"] for c in json.loads(_cats_file().read_text())["classes"] if c.get("group") == "terrain")


def norm_cls(c):
    if CREDIT_SRC.startswith("tools/"):
        return CL.norm(c)
    w = C.norm_word(c).replace(" ", "_")
    return w or None


def is_terrain(cls):
    return norm_cls(cls) in TERRAIN


def el_terrain(el):
    return bool(el["terrain"]) if "terrain" in el else is_terrain(el.get("cls"))


def gt_path_of(cf):
    return GT_ROOT / cf["gt_item"] / "gt.json"


def load_cf(path_or_key):
    p = Path(path_or_key)
    if not p.suffix:
        p = CLAIMS_DIR / (str(path_or_key) + ".json")
    cf = json.loads(p.read_text())
    if not os.environ.get("PV_V5_PREREG"):
        C.use_rules_of(cf)
    return cf


def a1_counts_drop(cf, gt=None):
    """pvscore5 A1 count item ids ("count:<space>:<class>") whose (space, class) group of GT objects without an
    entity id holds an object the prompt omits (PREREG anchoring.omitted_scope: an omitted object is left out of every
    v5 block, A1 counts included). The whole group is dropped: a 5.0.x count item is one value per group."""
    gt = gt or json.loads(gt_path_of(cf).read_text())
    om = set()
    for k in cf.get("omitted") or {}:
        om |= {o["key"] for o in gt.get("objects") or [] if fnmatch.fnmatch(o["key"], k)}
    out = set()
    for o in gt.get("objects") or []:
        if not o.get("id") and o["key"] in om:
            out.add("count:%s:%s" % (o.get("space"), o["class"]))
    return sorted(out)


# ------------------------------------------------------------------ model dump -> plain scene (numpy, pvscore5)
def times_needed(cf):
    """GT times the claims and checks read (claim times, event probes +-0.5 s, check times), on a 0.1 s grid too"""
    T = set()
    dur = float(cf["duration_s"])
    for i in range(int(dur * 10) + 1):
        T.add(round(i * 0.1, 3))
    hw = C.J("moving_half_window_s")
    for c in cf["claims"]:
        cl = c["claim"]
        if cl["t"] == "event":
            for t in (max(0.0, cl["t0"] - cl["tol"]), min(dur, cl["t1"] + cl["tol"])):
                T.update({round(t, 3), round(t - hw, 3), round(t + hw, 3)})
        for a in C.anchors_of(cl):
            m = C.ANCH_RE.match(a)
            if m and m.group(2):
                t = float(m.group(2))
                T.update({round(t, 3), round(t - hw, 3), round(t + hw, 3)})
    for k in cf["camera_checks"]:
        for f in ("at", "t0", "t1"):
            if k.get(f) is not None:
                T.add(round(float(k[f]), 3))
    return sorted(t for t in T if -1e-9 <= t <= dur + 1e-9)


def space_geom52(md, P, gt_ids, spaces, d22):
    """pvscore 5.2 M1-M3 (scorer PREREG_52.json): one geometry of the model spaces from the model's own layout pieces
    (entity pieces left out). A piece's role comes from its normalised class and the pvscore5 sets: floor =
    model.FLOOR_CLASSES + frame.GROUND_STRUCT (and untagged_floor instances); edge = the linear pieces that are not
    floor, gt.LINEAR_CLASSES + model.WALL_CLASSES + shoreline (and untagged_wall instances): wall, partition, fence,
    railing, tunnel, curb, shoreline; terrain = the categories' terrain group; contents = the rest. Per space:
      floor_polys (M1)  footprints of its floor pieces (layout.piece_footprint under D22, else the min-area rectangle)
      edge_polys (M1)   rectangles of its edge pieces (the wall run on a side)
      floor_own, edge_own (M1)  the instance key of each of those polygons (an object is not a side of its own space)
      ext (M3)          tier floor: box of its floor pieces; tier things (no floor piece): box of its non-terrain contents
                        and of its terrain pieces that are not edge pieces (hill, cliff, water, rock, pool); tier bbox
                        (none of these): the 5.1 box (spaces[sid]["bbox"]). Edge pieces never set it.
    and support (M2): (footprint rectangles, bottom, top, space tag) of every floor / terrain piece, tagged or not"""
    import numpy as np
    sys.path.insert(0, str(PVSCORE_DIR))
    from pvscore5 import geom, layout
    from pvscore5.frame import GROUND_STRUCT
    from pvscore5.gt import LINEAR_CLASSES
    from pvscore5.model import FLOOR_CLASSES, WALL_CLASSES
    FL, LN = FLOOR_CLASSES | GROUND_STRUCT, LINEAR_CLASSES | WALL_CLASSES | {"shoreline"}
    kind = {id(p): m.kind for m in md.instances for p in m.pieces}
    owner = {id(p): m.key for m in md.instances for p in m.pieces}

    def rect52(q):                      # 4 corners of a rectangle -> (centre, unit long axis, half sides) for in_rect
        e1, e2 = q[0] - q[1], q[0] - q[3]
        n1, n2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
        return (q.mean(0), e1 / n1 if n1 > 1e-12 else np.array([1.0, 0.0]), n1 / 2, n2 / 2)
    per = {s: {"fl": [], "fo": [], "flxy": [], "edge": [], "eo": [], "things": []} for s in spaces}
    support = []
    for p in md.layout_pieces:
        if p["id"] and p["id"] in gt_ids:
            continue
        c = norm_cls(p["cls"]) if p["cls"] else None
        k = kind.get(id(p))
        is_fl = (c in FL) if c else k == "untagged_floor"
        is_ed = not is_fl and ((c in LN) if c else k == "untagged_wall")
        is_te = bool(c) and is_terrain(c)
        xy = np.asarray(p["P"][:, :2], float)
        rc = geom.min_area_rect(xy)
        if is_fl or is_te:
            fp = [np.asarray(q, float) for q in
                  (layout.piece_footprint(xy, rc, P) if (is_fl and d22) else [geom.rect_corners(*rc)])]
            support.append(([rect52(q) for q in fp], float(p["lo"][2]), float(p["hi"][2]), p["space"]))
        r = per.get(p["space"])
        if r is None:
            continue
        if is_fl:
            r["fl"] += [q.tolist() for q in fp]
            r["fo"] += [owner.get(id(p))] * len(fp)
            r["flxy"].append(xy)
        elif is_ed:
            r["edge"].append(geom.rect_corners(*rc).tolist())
            r["eo"].append(owner.get(id(p)))
        else:
            r["things"].append(xy)
    out, area = {}, {}
    for s in spaces:
        r = per[s]
        tier, pts = ("floor", r["flxy"]) if r["flxy"] else (("things", r["things"]) if r["things"] else ("bbox", []))
        Q = np.concatenate(pts) if pts else None
        ext = list(spaces[s]["bbox"]) if Q is None else [float(Q[:, 0].min()), float(Q[:, 1].min()),
                                                          float(Q[:, 0].max()), float(Q[:, 1].max())]
        out[s] = {"floor_polys": r["fl"], "floor_own": r["fo"], "edge_polys": r["edge"], "edge_own": r["eo"],
                  "ext": ext, "ext_tier": tier}
        area[s] = (ext[2] - ext[0]) * (ext[3] - ext[1])
    return {"spaces": out, "support": support, "area": area}


def support_space52(g, xy, base, tol):
    """pvscore 5.2 M2 (scorer PREREG_52.json): a body's space = the space tag of the floor / ground piece under its
    base. Supports: floor / terrain pieces whose footprint grown by tol holds xy and whose bottom is at most base + tol;
    each holds the body at min(top, base) (a body inside a piece's height span, on a slope or stairs, stands in it).
    The highest support decides; the tagged spaces among the supports within tol of it give the smallest (by its
    space_geom52 extent area), so overlapping or nested spaces give the inner one. None (the 5.1 box containment)
    when no support is under the body or none of the highest is tagged."""
    from pvscore5 import geom
    cands = []
    for rects, bot, top, sp in g["support"]:
        if bot > base + tol or not any(geom.in_rect(xy, r, tol) for r in rects):
            continue
        cands.append((min(top, base), sp))
    if not cands:
        return None
    hmax = max(h for h, _ in cands)
    sps = sorted({sp for h, sp in cands if sp and h >= hmax - tol})
    if not sps:
        return None
    return min(sps, key=lambda s: (g["area"].get(s, math.inf), s))


def from_dump(cf, dump, gt_path=None):
    """r4 R4-4 (scorer PREREG_51r4_addendum1.json): the vendored pvscore 5.0.12 fix switches are set from the
    thresholds it loads for this call (as pvscore5.score.score_item does before the GT is read) and the previous
    switches restored, so the scene does not depend on what the process scored before"""
    sys.path.insert(0, str(PVSCORE_DIR))
    from pvscore5 import fixes
    from pvscore5.score import load_thresholds
    prev = fixes.ACTIVE
    fixes.set_active(fixes.ids_from(load_thresholds()))
    try:
        return _from_dump(cf, dump, gt_path)
    finally:
        fixes.set_active(prev)


def _from_dump(cf, dump, gt_path=None):
    import numpy as np
    sys.path.insert(0, str(PVSCORE_DIR))
    from pvscore5 import camera, fixes, geom, layout, motion  # noqa: F401
    from pvscore5.gt import GT, load_categories
    from pvscore5.model import BODY_GROUPS, Model
    from pvscore5.score import CATS, Ctx, load_thresholds
    gt_path = gt_path or gt_path_of(cf)
    cats = load_categories(CATS)
    th = load_thresholds()
    gt = GT(gt_path, cats)
    md = Model(dump, cats, th["params"])
    ctx = Ctx(gt, md, th)
    ids = set(gt.entities)
    md.layout(ids)
    md.bodies(ids)
    camera.build_warp(ctx)
    # spaces: boxes of the tagged layout pieces
    spaces = {}
    for s in gt.spaces:
        pcs = [p for p in md.layout_pieces if p["space"] == s.id]
        if not pcs:
            continue
        lo = np.min([p["lo"][:2] for p in pcs], axis=0)
        hi = np.max([p["hi"][:2] for p in pcs], axis=0)
        spaces[s.id] = {"poly": None, "bbox": [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])],
                        "floor": float(layout.model_floor(md.instances, md.layout_pieces, s.id))}
    elems = {}
    for m in md.instances:
        if m.kind in SKIP_KINDS or m.untagged:
            continue
        terr = is_terrain(m.cls)                                       # r4 R4-2
        if m.kind == "terrain" and not terr:
            raise RuntimeError("%s: pvscore5 kind terrain but class %r is not terrain after normalisation" % (m.key, m.cls))
        ax = float(m.long_yaw) % 180.0
        elems[m.key] = {"c": [float(m.center2[0]), float(m.center2[1])], "long": float(m.long),
                        "short": float(m.short), "axis": ax, "front": None if m.facing is None else float(m.facing) % 360,
                        "h": float(m.height), "z0": float(m.bottom), "z1": float(m.top), "space": m.space,
                        "cls": m.cls or "", "labels": [C.norm_word(m.cls)], "person": False, "terrain": terr}
    # person height of the scene
    hs = []
    for eid, b in md.body.items():
        if b.group in BODY_GROUPS and b.present.any():
            hs.append(float(np.nanpercentile((b.top - b.base)[b.present], 95)))
    ph = float(np.median(hs)) if hs else 1.75
    # pvscore 5.2 M1-M3 (scorer PREREG_52.json): one space geometry of the model's pieces (space_geom52)
    on52 = [f for f in ("M1", "M2", "M3") if fixes.on(f)]
    g52 = space_geom52(md, ctx.P, ids, spaces, fixes.on("D22")) if on52 else None
    tol52 = (C.MECH("m4") or {}).get("z_floor_frac", 0.05) * ph    # M2 support tolerance: the m4 thin-surface floor
    # entities on the GT time axis
    T = times_needed(cf)
    ents = {}
    for eid in sorted(cf["entities"]):
        b = md.body.get(eid)
        if b is None:
            continue
        person = b.group in BODY_GROUPS
        smp = []
        for t in T:
            tm = ctx.warp(t)
            ai = md.t_to_ai(tm)
            if abs(float(md.an_t[ai]) - tm) > 0.3 or not b.present[ai]:
                continue
            c = [float(b.center[ai][0]), float(b.center[ai][1])]
            if person:
                win = [x for x in range(md.A) if abs(md.an_t[x] - md.an_t[ai]) <= 0.25 + 1e-6 and b.present[x]
                       and not np.isnan(b.facing[x])]
                fs = [float(b.facing[x]) for x in win]
                fr = min(fs, key=lambda f: sum(C.angdiff(f, g) for g in fs)) if fs else None
                el = {"c": c, "long": 2 * C.J("person_radius_frac") * ph, "short": 2 * C.J("person_radius_frac") * ph,
                      "axis": None, "front": fr, "h": None, "z0": float(b.base[ai]), "z1": float(b.top[ai]),
                      "space": None, "cls": b.cls or "", "labels": [], "person": True}
                if fr is None and fixes.on("M14") and any(
                        b.facing_unk[x] for x in range(md.A)
                        if abs(md.an_t[x] - md.an_t[ai]) <= 0.25 + 1e-6 and b.present[x]):
                    el["front_unknown"] = True      # 5.2 (M14): facing unknown in the window, its claims not scored
            else:
                fr = None if np.isnan(b.facing[ai]) else float(b.facing[ai]) % 360
                el = {"c": c, "long": float(b.long[ai]), "short": float(b.short[ai]),
                      "axis": float(b.long_yaw[ai]) % 180.0, "front": fr, "h": float(b.top[ai] - b.base[ai]),
                      "z0": float(b.base[ai]), "z1": float(b.top[ai]), "space": None, "cls": b.cls or "",
                      "labels": [C.norm_word(b.cls)], "person": False}
            el["t"] = t
            if g52 is not None and "M2" in on52:
                el["space"] = support_space52(g52, el["c"], el["z0"], tol52)
            smp.append(el)
        if smp:
            ents[eid] = {"cls": b.cls or "", "person": person, "samples": smp, "static": False}
    # cameras per GT shot, at the check times and on the grid
    cams = {}
    for si, sh in enumerate(gt.shots):
        kf0, kf1 = camera.shot_frames(md, ctx.warp(sh.t0), ctx.warp(sh.t1))
        rows = []
        for t in T:
            if not (sh.t0 - 1e-6 <= t <= sh.t1 + 1e-6):
                continue
            k = min(kf1, max(kf0, md.t_to_frame_index(ctx.warp(t))))
            if not md.cam_ok[k]:
                continue
            hf = float(md.hfov[k])
            rows.append({"t": t, "pos": [float(x) for x in md.cam_pos[k]], "fwd": [float(x) for x in md.cam_fwd[k]],
                         "up": [float(x) for x in md.cam_up[k]], "hfov": None if not math.isfinite(hf) else hf,
                         "look": None})
        cams[str(si)] = rows
    tag_spaces = sorted(spaces)
    d = {"frame": "model", "spaces": spaces, "elems": elems, "ents": ents, "cams": cams, "ph": ph,
         "duration": float(cf["duration_s"]), "known": "all", "max_gap_s": 0.6, "space_tags": True,
         "axis_fallback": True,
         "_info": {"instances": len(md.instances), "elements": len(elems),
                   "terrain_elements": sorted(k for k, el in elems.items() if el["terrain"]), "bodies": sorted(md.body),
                   "spaces_tagged": tag_spaces, "model_shots": len(md.shots), "cut_pairs": ctx.cut_pairs,
                   "model_duration_s": md.duration}}
    if on52:
        d["fix52"] = {"on": on52, "seam": float(ctx.P["touch_gap_m"])}
        for k, el in elems.items():
            el["k52"] = k                                       # M1: the instance key (its own pieces are no side)
        for sid, s in spaces.items():
            s.update({k: v for k, v in g52["spaces"][sid].items() if k != "ext_tier"})
        d["_info"]["space52"] = {sid: {"tier": g["ext_tier"], "ext": [round(x, 3) for x in g["ext"]]}
                                 for sid, g in g52["spaces"].items()}
    # prereg_v5.0.1 note n1: a GT-sloped space gets the ground plane of the model's own things on it (instance
    # bottoms and the person bases, one per second), picked by the space outline (within 0.5 ph)
    sloped = [s.id for s in gt.spaces if getattr(s, "sloped", False) and s.id in spaces]
    if sloped:
        # instance bottoms carry their footprint long side (read only by prereg_v5.0.3 n1b); person bases stay (x, y, z)
        # r4 R4-2: terrain elements are not ground points (in r3 they were not in the scene)
        pts = [(el["c"][0], el["c"][1], el["z0"], el.get("long")) for el in elems.values() if el.get("z0") is not None
               and not el.get("terrain")]
        for e in ents.values():
            if not e["person"]:
                continue
            for smp in e["samples"]:
                if abs(smp["t"] - round(smp["t"])) <= 1e-6 and smp.get("z0") is not None:
                    pts.append((smp["c"][0], smp["c"][1], smp["z0"]))
        d["_info"]["ground"] = C.attach_ground(d, sloped, pts)
    return d


# ------------------------------------------------------------------ scoring
def model_candidates(cf, sc):
    gi = {k: {"cls": v["class"], "space": v.get("space"), "terrain": is_terrain(v["class"])}
          for k, v in cf["named_objects"].items()}

    def compat(g, el):
        if el_terrain(el) and not g["terrain"]:                  # r4 R4-1: terrain only for a terrain GT object
            return 0.0
        return class_credit(g["_key"], g["cls"], el.get("cls"))
    for k, g in gi.items():
        g["_key"] = k
    return C.candidates(gi, sc, compat)


def _status(r):
    return "missing" if r.get("miss") else ("true" if r["ok"] else "false")


def _round(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, (list, tuple)):
        return [_round(x) for x in v]
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    return v


def _block(rows):
    pres = [r for r in rows if r["status"] != "missing"]
    P = (sum(1 for r in pres if r["status"] == "true") / len(pres)) if pres else 0.0
    cov = (len(pres) / len(rows)) if rows else 0.0
    return P, cov, len(pres)


def score_scene(cf, sc_d, perm_n=PERM_N, seed=PERM_SEED):
    sc = C.Scene(sc_d)
    claims = [c["claim"] for c in cf["claims"]]
    cand = model_candidates(cf, sc)
    M = C.Matcher(sc, cand, claims)
    bind = M.solve()
    cfn = M.count_fn(bind)
    rows, unk = [], set()
    for i, c in enumerate(cf["claims"]):
        r = C.judge(sc, bind, c["claim"], count_fn=cfn)
        rows.append({"n": c["n"], "src": c["src"], "block": c["block"], "text": C.claim_text(c["claim"]),
                     "status": _status(r), "v": _round(r.get("v")), "note": r.get("note")})
        if r.get("unk"):        # 5.2 (M14): a claim on a facing the model leaves unknown leaves the block
            unk.add(i)
            rows[-1]["status"] = "unscored"
    cams = []
    for k in cf["camera_checks"]:
        r = C.judge_cam(sc, bind, k)
        cams.append({"n": k["n"], "si": k["si"], "t": k["t"], "text": C.check_text(k), "status": _status(r),
                     "v": _round(r.get("v")), "note": r.get("note")})
    blocks = {}
    rng = random.Random(seed)
    seeds = [rng.randrange(2 ** 31) for _ in range(perm_n)]
    perm_P = {"layout": [], "entity": []}
    idx = {b: [i for i, c in enumerate(cf["claims"]) if c["block"] == b and i not in unk] for b in perm_P}
    for s in seeds:
        sp = C.Scene(C.permuted(sc_d, bind, s))
        cfp = C.Matcher(sp, cand, claims).count_fn(bind)
        for b, ii in idx.items():
            st = [_status(C.judge(sp, bind, cf["claims"][i]["claim"], count_fn=cfp)) for i in ii]
            pres = [x for x in st if x != "missing"]
            perm_P[b].append((sum(1 for x in pres if x == "true") / len(pres)) if pres else 0.0)
    for b in ("layout", "entity"):
        rr = [rows[i] for i in idx[b]]
        P, cov, npres = _block(rr)
        Rb = (sum(perm_P[b]) / len(perm_P[b])) if perm_P[b] else 0.0
        deg = 1.0 - Rb < 1e-9
        score = P * cov if deg else max(0.0, min(1.0, (P - Rb) / (1.0 - Rb))) * cov
        blocks[b] = {"n": len(rr), "n_true": sum(1 for r in rr if r["status"] == "true"),
                     "n_missing": len(rr) - npres, "P": round(P, 4), "coverage": round(cov, 4), "R": round(Rb, 4),
                     "score": round(score, 4), "degenerate_R": deg}
    P, cov, npres = _block(cams)
    blocks["camera"] = {"n": len(cams), "n_true": sum(1 for r in cams if r["status"] == "true"),
                        "n_missing": len(cams) - npres, "P": round(P, 4), "coverage": round(cov, 4), "R": None,
                        "score": round(P * cov, 4), "degenerate_R": False}
    return {"item": cf["item"], "gt_item": cf["gt_item"], "prereg": PRE["version"], "class_credit": CREDIT_SRC,
            "perm": {"n": perm_n, "seed": seed}, "blocks": blocks,
            "claims": [r for i, r in enumerate(rows) if i not in unk], "camera_checks": cams,
            **({"unscored_5_2": [rows[i] for i in sorted(unk)]} if unk else {}),
            "binding": {k: v for k, v in sorted(bind.items())},
            "unbound": sorted(k for k, v in bind.items() if not v)}


# ------------------------------------------------------------------ self-tests (stdlib, on the GT)
def gt_as_model(gt):
    d = C.scene_from_gt(gt, anonymise=True)
    d["frame"] = "gtm"
    d["space_tags"] = True
    d["axis_fallback"] = True
    for e in d["ents"].values():          # bodies carry no space tag in a model scene either
        for smp in e["samples"]:
            smp["space"] = None
    return d


def shift_all(d):
    """every static element shifted by claims_core.shifted_elem at once"""
    out = d
    for k in sorted(d["elems"]):
        out, _ = C.shifted_elem(out, k)
    return out


def selftest(keys, perm_n=PERM_N):
    res = []
    for key in keys:
        cf = load_cf(key)
        gt = json.loads(gt_path_of(cf).read_text())
        base = gt_as_model(gt)
        r0 = score_scene(cf, base, perm_n)
        tests = {"gt_as_model": r0}
        tests["mirrored"] = score_scene(cf, C.mirror_scene(base), perm_n)
        tests["shifted"] = score_scene(cf, shift_all(base), perm_n)
        other = next(k for k in sorted(p.name for p in GT_ROOT.iterdir() if p.is_dir())
                     if k != cf["gt_item"] and (GT_ROOT / k / "gt.json").exists())
        og = json.loads((GT_ROOT / other / "gt.json").read_text())
        tests["cross_item"] = score_scene(cf, gt_as_model(og), perm_n)
        tests["cross_item"]["scene_of"] = other
        errs = []
        b0 = r0["blocks"]
        if any(b0[b]["score"] < 1.0 - 1e-9 for b in b0) or r0["unbound"]:
            errs.append("GT-as-model is not 1.0 on every block")
        # the mirror image (x -> -x) can only break a claim or check that names east / west: a block with such a
        # claim must drop, a block without one must stay as it is
        n_ew = {b: sum(1 for c in cf["claims"] if c["block"] == b and C.ew_sensitive(c["claim"]))
                for b in ("layout", "entity")}
        n_ew["camera"] = sum(1 for k in cf["camera_checks"] if C.mirror_check(k) != k)
        bm = tests["mirrored"]["blocks"]
        for b in ("layout", "entity", "camera"):
            if n_ew[b] and not bm[b]["score"] < b0[b]["score"] - 1e-9:
                errs.append("mirrored: block %s (%d east-west claims) did not drop" % (b, n_ew[b]))
            if not n_ew[b] and abs(bm[b]["score"] - b0[b]["score"]) > 1e-9:
                errs.append("mirrored: block %s has no east-west claim but changed" % b)
        # with the GT-as-model binding held fixed, every claim and check without east / west still holds on the
        # mirror image (the matcher may rebind same-class members there, e.g. swap a west and an east column, and
        # so break a few such claims while keeping more claims overall; that is the registered matching)
        sm = C.Scene(C.mirror_scene(base))
        B0 = r0["binding"]
        cfm = C.Matcher(sm, model_candidates(cf, sm), [c["claim"] for c in cf["claims"]]).count_fn(B0)
        mir_nonew = [C.claim_text(c["claim"]) for c in cf["claims"] if not C.ew_sensitive(c["claim"])
                     and not C.judge(sm, B0, c["claim"], count_fn=cfm)["ok"]]
        mir_nonew += [C.check_text(k) for k in cf["camera_checks"] if C.mirror_check(k) == k
                      and not C.judge_cam(sm, B0, k)["ok"]]
        if mir_nonew:
            errs.append("mirrored: %d claims / checks without east / west fail under the GT binding" % len(mir_nonew))
        mir_rebound = [r["text"] for r, c in zip(tests["mirrored"]["claims"], cf["claims"])
                       if r["status"] != "true" and not C.ew_sensitive(c["claim"])]
        for name in ("shifted", "cross_item"):
            bb = tests[name]["blocks"]
            if not all(bb[b]["score"] < b0[b]["score"] - 1e-9 for b in (("layout", "entity", "camera")
                                                                        if name != "shifted" else ("layout",))):
                errs.append("%s: a block did not drop" % name)
        summ = {name: {b: {k: v for k, v in t["blocks"][b].items() if k in ("P", "R", "coverage", "score")}
                       for b in t["blocks"]} for name, t in tests.items()}
        wrong_bind = sorted(k for k, v in r0["binding"].items() if v and base["elems"][v]["_key"] != k)
        res.append({"item": key, "summary": summ, "errors": errs, "pass": not errs, "east_west_claims": n_ew,
                    "mirrored_rebound_failing": mir_rebound,
                    "gt_as_model_rebound": wrong_bind,
                    "gt_as_model_failing": [r["text"] for r in r0["claims"] if r["status"] != "true"] +
                                           [r["text"] for r in r0["camera_checks"] if r["status"] != "true"],
                    "cross_item_scene": other})
    return res


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims")
    ap.add_argument("--dump", type=Path)
    ap.add_argument("--gt", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--items")
    ap.add_argument("--perm-n", type=int, default=PERM_N)
    ap.add_argument("--claims-dir", type=Path, help="claims files (default drafts_v5/claims; round 2: drafts_v5/r2/claims)")
    a = ap.parse_args(argv)
    global CLAIMS_DIR
    if a.claims_dir:
        CLAIMS_DIR = a.claims_dir
    if a.selftest:
        keys = a.items.split(",") if a.items else sorted(p.stem for p in CLAIMS_DIR.glob("MS50_T*.json"))
        res = selftest(keys, a.perm_n)
        log = {"prereg": PRE["version"], "class_credit": CREDIT_SRC, "perm_n": a.perm_n, "items": res,
               "pass": all(r["pass"] for r in res)}
        out = a.out or HERE / "claims_score_selftest.json"
        out.write_text(json.dumps(log, indent=1) + "\n")
        for r in res:
            s = r["summary"]
            print(r["item"], " | ".join("%s L %.3f E %.3f C %.3f" % (n, s[n]["layout"]["score"], s[n]["entity"]["score"],
                                                                     s[n]["camera"]["score"]) for n in s),
                  "|", "PASS" if r["pass"] else "; ".join(r["errors"]))
        print("selftest", "PASS" if log["pass"] else "FAIL", "->", out)
        return 0 if log["pass"] else 1
    if not (a.claims and a.dump):
        ap.error("give --claims and --dump, or --selftest")
    cf = load_cf(a.claims)
    sc_d = from_dump(cf, a.dump, a.gt)
    info = sc_d.pop("_info")
    res = score_scene(cf, sc_d, a.perm_n)
    res["dump"] = str(a.dump)
    res["model_scene"] = info
    res["a1_counts_drop"] = a1_counts_drop(cf)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1) + "\n")
    b = res["blocks"]
    print("%s %s: layout P %.3f R %.3f cov %.3f -> %.3f | entity P %.3f R %.3f cov %.3f -> %.3f | camera %.3f x %.3f"
          " -> %.3f | unbound %d" % (res["item"], a.dump, b["layout"]["P"], b["layout"]["R"], b["layout"]["coverage"],
                                     b["layout"]["score"], b["entity"]["P"], b["entity"]["R"],
                                     b["entity"]["coverage"], b["entity"]["score"], b["camera"]["P"],
                                     b["camera"]["coverage"], b["camera"]["score"], len(res["unbound"])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
