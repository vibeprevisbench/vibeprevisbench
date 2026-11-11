"""pvscore 5.1 text / image layer (r2): GT fact inventory per item (no API, no sandbox, no Blender).

    python3 -I -B code/v51_text_r2/gt_facts.py [ITEM ...]

r2 changes: items = every script / spatial_arkit / spatial_keyframe item of the item table (tools/v51_r2/items.py);
paths by tools/v51_r2/paths.py (output paths.TEXT_GT/<item>/gt_facts.json); presence negatives (facts
"presence_neg", thresholds_v51r2.json presence.negatives). Other facts unchanged from r1.

r1 changes: GT root by tools/v51_r2/gt_root.py (paths.GT_ROOT); the class exclusions compare normalised classes
(tools/v51_text_r2/classes.py; no GT class of the first 5 items changes eligibility); each 3D element also stores its
class labels ("labels": normalised class + prompt name, classes.gt_labels). Fact rules unchanged.
Facts come only from the GT (the GT <item>/gt.json, read-only, via pvscore5.gt) and
tools/v51_text_r2/thresholds.json:
  3D   L1 (A nearer to B than to C), L2 (side of C seen from A to B), S1 / S2 (clearly larger / about equal, footprint
       long side and height), S3 (object height in person heights, text items), R1 (L1 with a named entity at a GT
       track sample time as apex). Same GT space only; margins widened by the GT unc_m (2 sigma).
  view the GT scene (boxes, persons and moving entities where the track states the position) ray-cast from the GT
       camera at 1 fps (text, stated poses only) / the photo camera (image): presence, 3x3 cell, height fraction,
       person facing to the camera, left/right, above/below, nearer/farther; occlusion inventoried, not scored.
Writes data/v51_text_r2/<item>/gt_facts.json. Also provides build() / scene_view() for score.py (GT-as-model tests).
"""
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene as S  # noqa: E402  (puts tools/v51_r2 on the path)
import classes as CL  # noqa: E402
import paths as PT  # noqa: E402
import items as IT  # noqa: E402

R = S.R
TH = S.TH
NEG = S.TH2["presence"]["negatives"]
EL, VW = S.EL, S.VW
OUT = PT.TEXT_GT
ITEMS = IT.of_kinds(IT.TEXT_KINDS)


# ----------------------------------------------------------------------------------------------- elements
def eligible(o, sc):
    if o is None:
        return True                                   # persons
    if o["group"] in EL["exclude_groups"] or CL.excluded(o["cls"], EL["exclude_classes"]):
        return False
    sx, sy = o["size"][0], o["size"][1]
    lo, sh = max(sx, sy), max(1e-6, min(sx, sy))
    sp = sc["spaces"].get(o["space"]) or next(iter(sc["spaces"].values()))
    if lo / sh >= EL["thin_long"]["aspect_min"] and lo >= EL["thin_long"]["long_frac_of_space_L_min"] * sp["L"]:
        return False
    if sx * sy >= EL["max_footprint_frac_of_space_area"] * sp["area"]:
        return False
    return True


def t_ref_of(sc, eid):
    tr = sc["tracks"].get(eid)
    if sc["modality"] == "image" or not tr or not tr["t"]:
        return 0.0
    return float(tr["t"][0])


def elements3d(sc):
    """static 3D elements: eligible unnamed objects + named objects at their reference time"""
    E = {}
    for o in sc["objects"]:
        if not eligible(o, sc):
            continue
        sz = o["size"]
        e = {"cls": o["cls"], "labels": list(CL.gt_labels(o["key"], o["cls"])), "space": o["space"],
             "named": bool(o["id"]), "conf": o["conf"], "unc": o["unc"],
             "long": max(sz[0], sz[1]), "short": min(sz[0], sz[1]), "height": sz[2], "size_known": o["size_known"]}
        if o["id"]:
            key = "b:" + o["id"]
            tr = sc["tracks"].get(o["id"])
            e["t_ref"] = t_ref_of(sc, o["id"])
            if tr and tr["t"] and sc["modality"] != "image":
                e["xy"] = [float(x) for x in S.track_at(tr, e["t_ref"])]
                e["space"] = S.track_label(tr, e["t_ref"], "space") or o["space"]
                e["unc"] = max(o["unc"], tr["unc"])
            else:
                e["xy"] = o["center"][:2]
        else:
            key = o["key"]
            e["xy"] = o["center"][:2]
        e["w"] = e["conf"] * (EL["named_weight"] if e["named"] else 1.0)
        E[key] = e
    return E


def entity_times(sc):
    """R1 apexes: named entities at their GT track sample times (deduplicated while they stay put)"""
    out = {}
    if sc["modality"] == "image":
        return out
    for eid, tr in sc["tracks"].items():
        last = None
        for i, t in enumerate(tr["t"]):
            if not 0.0 <= t <= sc["duration"] + 1e-6:
                continue
            xy = np.asarray(tr["p"][i], float)
            if last is not None and np.linalg.norm(xy - last) < 0.3 * sc["scale"]:
                continue
            last = xy
            out["b:%s@%g" % (eid, t)] = {"eid": eid, "t": float(t), "xy": xy.tolist(), "space": tr["space"][i],
                                         "unc": tr["unc"], "w": EL["named_weight"]}
    return out


def _w(facts, wel, k):
    """element weight spread evenly over the block's facts; fact weight = mean of its elements' shares"""
    n = Counter(x for f in facts for x in f[:k])
    return [list(f) + [round(sum(wel[x] / n[x] for x in f[:k]) / k, 6)] for f in facts]


def facts3d(sc, E, ET):
    L1, L2, R1, S1, S2, S3 = [], [], [], [], [], []
    k = TH["L1"]["gt_sigma_k"]
    by_sp = defaultdict(list)
    for key, e in E.items():
        by_sp[e["space"]].append(key)
    for sp, keys in by_sp.items():
        keys = sorted(keys)
        P = {x: np.asarray(E[x]["xy"], float) for x in keys}
        U = {x: E[x]["unc"] for x in keys}
        for a in keys:
            for b, c in itertools.permutations([x for x in keys if x != a], 2):
                dab, dac = np.linalg.norm(P[a] - P[b]), np.linalg.norm(P[a] - P[c])
                sig = math.sqrt(U[a] ** 2 + U[b] ** 2 + U[c] ** 2)
                if dac >= TH["L1"]["gt_ratio_min"] * dab and dac - dab >= k * sig:
                    L1.append((a, b, c))
        for a, b, c in itertools.combinations(keys, 3):
            pa, pb, pc = P[a], P[b], P[c]
            cr = float((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]))
            sig = math.sqrt(U[a] ** 2 + U[b] ** 2 + U[c] ** 2)
            ok = True
            for (v, e1, e2) in ((pa, pb, pc), (pb, pa, pc), (pc, pa, pb)):
                el = np.linalg.norm(e2 - e1)
                d = abs(cr) / max(el, 1e-9)
                if el < 1e-9 or d < TH["L2"]["gt_fat_min"] * el or d < TH["L2"]["gt_sigma_k"] * sig:
                    ok = False
                    break
            if ok:
                L2.append((a, b, c, 1 if cr > 0 else -1))
        for a, b in itertools.combinations(keys, 2):
            ea, eb = E[a], E[b]
            if not (ea["size_known"] and eb["size_known"]):
                continue
            for m in TH["S1"]["measures"]:
                va, vb = max(ea[m], 1e-3), max(eb[m], 1e-3)
                r = va / vb
                if r >= TH["S1"]["gt_ratio_min"]:
                    S1.append((a, b, m))
                elif 1 / r >= TH["S1"]["gt_ratio_min"]:
                    S1.append((b, a, m))
                elif max(r, 1 / r) <= TH["S2"]["gt_ratio_max"]:
                    S2.append((a, b, m))
    if sc["modality"] == "text" and any(c == "person" for c in sc["entities"].values()):
        ph = EL["text_person_height_m"] * sc["scale"]
        for key, e in E.items():
            if e["size_known"] and e["height"] >= TH["S3"]["min_height_m"] * sc["scale"]:
                S3.append((key, round(e["height"] / ph, 4)))
    for ak, a in ET.items():
        keys = sorted(x for x in by_sp.get(a["space"], []) if x != "b:" + a["eid"])
        pa = np.asarray(a["xy"], float)
        for b, c in itertools.permutations(keys, 2):
            pb, pc = np.asarray(E[b]["xy"], float), np.asarray(E[c]["xy"], float)
            dab, dac = np.linalg.norm(pa - pb), np.linalg.norm(pa - pc)
            sig = math.sqrt(a["unc"] ** 2 + E[b]["unc"] ** 2 + E[c]["unc"] ** 2)
            if dac >= TH["L1"]["gt_ratio_min"] * dab and dac - dab >= k * sig:
                R1.append((ak, b, c))
    wel = {k_: e["w"] for k_, e in E.items()}
    wel.update({k_: e["w"] for k_, e in ET.items()})
    return {"L1": _w(L1, wel, 3), "L2": _w(L2, wel, 3), "R1": _w(R1, wel, 3), "S1": _w(S1, wel, 2),
            "S2": _w(S2, wel, 2), "S3": _w(S3, wel, 1)}


# -------------------------------------------------------------------------------------------------- views
def frame_render(sc, shot, t):
    sh = next(s for s in sc["shots"] if s["id"] == shot)
    cam = S.cam_at(sh, t, sc["res"])
    if cam is None:
        return None, None, None
    boxes = S.boxes_at(sc, t)
    rd = S.render(boxes, cam)
    E, occl = S.readout(boxes, rd, sc, t)
    return E, occl, (boxes, rd, sh)


def view_spaces(sc, sh):
    sp = sh["space"]
    if sc["spaces"].get(sp, {}).get("kind") == "outdoor":
        return {k for k, v in sc["spaces"].items() if v["kind"] == "outdoor"}
    return {sp}


def _inside(bx, pos):
    c = np.asarray(bx["c"], float)
    hs = np.asarray(bx["size"], float) / 2
    cy_, sy_ = math.cos(math.radians(bx["yaw"])), math.sin(math.radians(bx["yaw"]))
    Rb = np.array([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1.0]])
    return bool(np.all(np.abs(Rb.T @ (np.asarray(pos, float) - c)) < hs))


def _projects_in(bx, cam):
    """any corner or the centre of the box in front of the camera and inside the frame (cam's intrinsics)"""
    pos, Rc, fx, fy, cx, cy, W, H = cam
    c = np.asarray(bx["c"], float)
    hs = np.asarray(bx["size"], float) / 2
    cy_, sy_ = math.cos(math.radians(bx["yaw"])), math.sin(math.radians(bx["yaw"]))
    Rb = np.array([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1.0]])
    pts = [c] + [c + Rb @ (hs * np.array(s)) for s in itertools.product((-1, 1), repeat=3)]
    right, up, back = Rc[:, 0], Rc[:, 1], Rc[:, 2]
    for q in pts:
        v = q - pos
        z = -float(v @ back)
        if z <= 1e-6:
            continue
        u, w = cx + fx * float(v @ right) / z, cy - fy * float(v @ up) / z
        if 0 <= u <= W and 0 <= w <= H:
            return True
    return False


def presence_negatives(sc, boxes, cam, E, sps):
    """r2 (thresholds_v51r2.json presence.negatives): eligible GT elements out of view in this frame. Same element rules
    as the positives; not visible at all (E holds every element with a visible pixel); the box enlarged by sigma_k x unc
    on every half-size, rendered through the frame widened by margin_frac per side with occlusion ignored, gets no ray,
    does not contain the camera, and has no corner / centre projecting into the widened frame (a box smaller than the
    96-wide ray spacing can fall between rays)."""
    seen = {e["key"] for e in E}
    m, k = NEG["margin_frac"], NEG["sigma_k"]
    pos, Rc, fx, fy, cx, cy, W, H = cam
    wide = (pos, Rc, fx / (1 + 2 * m), fy / (1 + 2 * m), cx, cy, W, H)
    elig = {("b:" + o["id"]) if o["id"] else o["key"]: eligible(o, sc) for o in sc["objects"]}
    cand = []
    for bx in boxes:
        if bx["key"] in seen or not (bx["person"] or elig.get(bx["key"], False)):
            continue
        if bx["space"] not in sps or bx["unc"] > VW["max_elem_unc_m"] * sc["scale"]:
            continue
        cand.append(dict(bx, size=[s + 2 * k * bx["unc"] for s in bx["size"]]))
    if not cand:
        return []
    rd = S.render(cand, wide)
    return sorted(bx["key"] for j, bx in enumerate(cand)
                  if rd["unocc"][j] == 0 and not _inside(bx, pos) and not _projects_in(bx, wide))


def facts_view(sc, E3):
    F = {k: [] for k in ("presence", "presence_neg", "position", "size", "facing", "screen_order", "depth_order")}
    frames, occ_all = [], []
    k = VW["sigma_k"]
    elig_key = {("b:" + o["id"]) if o["id"] else o["key"]: eligible(o, sc) for o in sc["objects"]}
    for shot, t in S.frame_times(sc):
        fid = "%s@%g" % (shot, t)
        E, occl, ctx = frame_render(sc, shot, t)
        if E is None:
            continue
        boxes, rd, sh = ctx
        th, tv = rd["tans"]
        sps = view_spaces(sc, sh)
        fe = []
        for e in E:
            if not (e["person"] or elig_key.get(e["key"], False)):
                continue
            if e["space"] not in sps or e["unc"] > VW["max_elem_unc_m"] * sc["scale"]:
                continue
            if e["area_frac"] < VW["vis_min_frac"]:
                continue
            d = max(e["depth_med"], 1e-6)
            e = dict(e, sx=e["unc"] / (d * 2 * th), sy=e["unc"] / (d * 2 * tv))
            fe.append(e)
        frames.append({"fid": fid, "shot": shot, "t": t, "n_visible_all": len(E),
                       "elements": [{x: e[x] for x in ("key", "class", "c", "cell", "height_frac", "depth_med",
                                                         "area_frac", "unc") if x in e}
                                    | ({"facing_cam": e.get("facing_cam")} if e["person"] else {}) for e in fe]})
        occ_all += [dict(o, fid=fid) for o in occl]
        F["presence_neg"] += [[fid, key] for key in presence_negatives(sc, boxes, S.cam_at(sh, t, sc["res"]), E, sps)]
        m = VW["cell_margin"]
        for e in fe:
            F["presence"].append([fid, e["key"]])
            cx, cy = e["c"]
            if min(abs(cx - 1 / 3), abs(cx - 2 / 3)) >= m + k * e["sx"] and \
                    min(abs(cy - 1 / 3), abs(cy - 2 / 3)) >= m + k * e["sy"]:
                F["position"].append([fid, e["key"], e["cell"]])
            if e["height_frac"] >= VW["height_min"] and e["unc"] / max(e["depth_med"], 1e-6) <= VW["height_max_rel_unc"]:
                F["size"].append([fid, e["key"], e["height_frac"]])
            if e["person"] and e.get("facing_cam") is not None:
                F["facing"].append([fid, e["key"], e["facing_cam"]])
        for a, b in itertools.permutations(fe, 2):
            gx = VW["pair_gap_min"] + k * math.hypot(a["sx"], b["sx"])
            gy = VW["pair_gap_min"] + k * math.hypot(a["sy"], b["sy"])
            if b["c"][0] - a["c"][0] >= gx:
                F["screen_order"].append([fid, a["key"], b["key"], "left_of"])
            if b["c"][1] - a["c"][1] >= gy:
                F["screen_order"].append([fid, a["key"], b["key"], "above"])
            if (b["depth_med"] - k * b["unc"]) / max(1e-6, a["depth_med"] + k * a["unc"]) >= VW["depth_ratio_min"]:
                F["depth_order"].append([fid, a["key"], b["key"]])
    return F, frames, occ_all


def build(sc):
    E = elements3d(sc)
    ET = entity_times(sc)
    F3 = facts3d(sc, E, ET)
    FV, frames, occ = facts_view(sc, E)
    return {"item": sc["item"], "modality": sc["modality"], "elements": E, "entity_times": ET,
            "facts": {**F3, **FV}, "frames": frames, "occlusion_inventory": occ}


# ----------------------------------------------------------------------------- GT scene read as a model
def surface(o):
    """as model_view.key_table: terrain and the walkable floor classes are surfaces (support only)"""
    return o["group"] in EL["exclude_groups"] or o["kind"] == "terrain" or o["cls"] in ("floor", "platform")


def scene_view(sc, gtf, rename=None, drop_frames=False):
    """GT-as-model adapter: the (transformed / perturbed) scene in the model-view format score.py reads.
    rename: {gt object key -> model key} for unnamed objects (T1 rename / reorder)."""
    rn = rename or {}
    els = {}
    for o in sc["objects"]:
        if o["group"] in EL["exclude_groups"] or o["kind"] == "terrain" or CL.excluded(o["cls"], EL["exclude_classes"]):
            continue                          # as model_side.py: terrain and excluded classes are no candidates
        sz = o["size"]
        if o["id"]:
            key = "b:" + o["id"]
            tr = sc["tracks"].get(o["id"])
            xy = S.track_at(tr, t_ref_of(sc, o["id"])).tolist() if tr and tr["t"] and sc["modality"] != "image" \
                else o["center"][:2]
        else:
            key = rn.get(o["key"], o["key"])
            xy = o["center"][:2]
        els[key] = {"cls": o["cls"], "space": o["space"], "xy": [float(x) for x in xy], "long": max(sz[0], sz[1]),
                    "short": min(sz[0], sz[1]), "height": sz[2]}
    ent = {}
    for k_, a in gtf["entity_times"].items():
        tr = sc["tracks"].get(a["eid"])
        ent[k_] = S.track_at(tr, a["t"]).tolist() if tr and tr["t"] else None
    ph = EL["text_person_height_m"] * sc["scale"] if any(c == "person" for c in sc["entities"].values()) else None
    frames = {}
    if not drop_frames:
        for fr in gtf["frames"]:
            E, _, _ = frame_render(sc, fr["shot"], fr["t"])
            if E is None:
                frames[fr["fid"]] = None
                continue
            ob = {(("b:" + o["id"]) if o["id"] else o["key"]): o for o in sc["objects"]}
            out = []
            for e in E:
                o = ob.get(e["key"])
                surf = o is not None and surface(o)
                x = {"key": rn.get(e["key"], e["key"]), "class": ("surface:ground" if surf else e["class"]),
                     "c": e["c"], "cell": e["cell"], "height_frac": e["height_frac"], "depth_med": e["depth_med"]}
                if e["person"]:
                    x["facing_cam"] = e.get("facing_cam")
                out.append(x)
            frames[fr["fid"]] = out
    return {"source": "gt_scene", "elements": els, "ent_xy": ent, "person_h": ph, "frames": frames}


# ------------------------------------------------------------------------------------------------ report
def not_checkable(gt, sc, gtf):
    out = []
    ids_obj = {o["id"] for o in sc["objects"] if o["id"]}
    for eid, cls in sc["entities"].items():
        if eid not in ids_obj and eid not in sc["tracks"]:
            out.append({"what": "entity %s (%s)" % (eid, cls), "why": "no GT box and no GT track: no 3D or view fact"})
        elif eid in sc["tracks"] and sc["modality"] == "text":
            tr = sc["tracks"][eid]
            dur = sc["duration"]
            known = sum(min(b + 0.5, dur) - max(a - 0.5, 0) for a, b in tr["segs"])
            if known < 0.5 * dur:
                out.append({"what": "entity %s position" % eid,
                            "why": "track states the position over %.1f of %.1f s only (5.0.3 known spans); "
                                   "no view fact and no R1 apex elsewhere" % (known, dur)})
    for o in sc["objects"]:
        if not eligible(o, sc):
            out.append({"what": "object %s (%s)" % (o["key"], o["cls"]),
                        "why": "excluded by thresholds elements (terrain / surface class / thin-long / large footprint); "
                               "rendered as an occluder only"})
        elif o["unc"] > VW["max_elem_unc_m"] * sc["scale"]:
            out.append({"what": "object %s (%s) view facts" % (o["key"], o["cls"]),
                        "why": "unc_m %.2f > view.max_elem_unc_m" % o["unc"]})
    n_un = 0
    for sh in sc["shots"]:
        if sc["modality"] == "image":
            continue
        ts = [t for s_, t in S.frame_times(sc) if s_ == sh["id"]]
        tot = int(max(0, math.floor(sh["t1"] - sh["t0"] - 0.25)))
        if len(ts) < tot:
            n_un += tot - len(ts)
            out.append({"what": "shot %s camera between samples" % sh["id"],
                        "why": "moving camera with pose stated only at its samples (or open parts): %d of ~%d 1-fps "
                               "times kept" % (len(ts), tot)})
    out.append({"what": "events, contacts, timing, camera moves, cuts", "why": "not part of this layer (5.0.9 B2 / C1 move / C2 keep)"})
    out.append({"what": "L3-L6 (on / under / inside / near / between / facing relations)",
                "why": "design keeps the 5.0 rules; not re-implemented here (%d GT relations)" % len(gt.relations)})
    out.append({"what": "occlusion", "why": "inventoried (%d GT cases) but not scored: the model dump holds only the "
                                            "visible surfaces, an unoccluded model projection is not built" % len(gtf["occlusion_inventory"])})
    out.append({"what": "S4 scene scale", "why": "not computed (needs the model space extent; skipped for time)"})
    if sc["modality"] == "image":
        out.append({"what": "image content without a GT box (walls, small props, photo-only objects)",
                    "why": "image GT = GT boxes projected through the GT camera; anything the boxes miss is not a fact"})
    return out


def main():
    items = [a for a in sys.argv[1:] if a in ITEMS] or ITEMS
    for item in items:
        gt, _ = S.load_gt(item)
        sc = S.scene_from_gt(gt, item)
        gtf = build(sc)
        gtf["not_checkable"] = not_checkable(gt, sc, gtf)
        gtf["revision"] = "r2"
        gtf["thresholds"] = [PT.rel(PT.TEXT_CONF / f) for f in ("thresholds.json", "thresholds_v51r1.json",
                                                                "thresholds_v51r2.json", "class_map_v51r1.json")]
        gtf["counts"] = {k: len(v) for k, v in gtf["facts"].items()}
        gtf["counts"]["frames"] = len(gtf["frames"])
        gtf["counts"]["elements3d"] = len(gtf["elements"])
        gtf["counts"]["entity_times"] = len(gtf["entity_times"])
        gtf["counts"]["occlusion_inventory"] = len(gtf["occlusion_inventory"])
        o = OUT / item
        o.mkdir(parents=True, exist_ok=True)
        (o / "gt_facts.json").write_text(json.dumps(gtf, indent=0) + "\n")
        print(item, json.dumps(gtf["counts"]), flush=True)


if __name__ == "__main__":
    main()
