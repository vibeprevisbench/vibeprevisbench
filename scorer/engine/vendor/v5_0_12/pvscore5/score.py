"""pvscore 5.0 host scorer: GT (pv_gt_v5.0) + model dump (pvdump-5.0) -> per-item JSON report.

    env/bin/python -m pvscore5.score item --gt GT.json --dump DUMP_DIR --out report.json
    env/bin/python -m pvscore5.score agg --reports R1.json R2.json ... --out summary.json

A missing or failed dump (no scene.blend delivered) scores 0 with delivered = false.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import camera, fixes, frame, geom, layout, motion, plaus
from .gt import GT, load_categories
from .items import Items
from .model import Model, SEAT_CLASSES

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[2]
CATS = BENCH / "common" / "categories_v5.0.json"
THRESH = HERE / "thresholds_v5_0_12.json"    # 5.0.12: thresholds_v5_0_11.json + fixes_5_0_12 and params (CHANGES_5012.md)
VERSION = "pvscore-5.0.12"  # 5.0.1 (10-17): jump / land / turn detectors in B2; 5.0.2 (10-17): sloped spaces, open camera parts
# 5.0.3 (10-17): GT positions only where a track knows them (sparse text tracks: no clamping, no long moving gaps)
# 5.0.4 (10-17): P1 references may carry a 'known' mask (real refs); depth coverage and edges count only there
# 5.0.5 (10-17): video GT: space_frames "separate" (no global alignment fallback, no cross-space size / floor items),
#                objects with size_known false (size not compared), shot tokens over every GT entity
# 5.0.6 (10-17): image items: a view with no GT camera is matched to the model shot of the same order, scored in P1
#                only (no C1 placement items, not in the C2 view order)
# 5.0.7 (10-17): jump / land: airborne runs up to 4 s when the base height changes along the run (long sports
#                airtime, slow motion); a run airborne at the clip start gives its landing, one at the clip end its take-off
# 5.0.8 (10-18): C1 frames stay inside the warped shot (a GT sample at a shot's end time read the next shot's first
#                frame); text GT move names (dolly_in, crane, truck / follow forward) mapped onto decompose() types
# 5.0.9 (10-18): C1 camera side relative to the subject's facing (rel_azimuth, subject block) read in the model's
#                own frame, so it is scored when the space has no alignment (it was 0 there)
# 5.0.10 (10-18): GT r1 shot fields screen_dir_segments / move_segments: a direction or camera move that changes
#                 inside a shot is scored per segment over its own window (segments share one label's weight)
# 5.0.11 (10-18): oriented footprints in the support / airborne tests (not the axis-aligned box of rotated corners);
#                 B2 occurrence matches gated at |dt| <= tol + event_extra_s; text single-label screen_dir by A_lat
# 5.0.12 (10-19): the GT-as-model S / D fixes (S6-S25 less S12 / S23, D2-D22), each behind its id in the thresholds'
#                 fixes_5_0_12 list (pvscore5/fixes.py); an empty list reproduces 5.0.11. S23 is in tools/protocol
DIMS =["A1", "A2", "A3", "P1", "B1", "B2", "C1", "C2", "D1"]


class Ctx:
    def __init__(self, gt, model, th_all):
        self.gt, self.model = gt, model
        self.TH_all = th_all
        self.TH = th_all["curves"]
        self.P = th_all["params"]
        self.credit = layout.Credit(self.P)
        self.aligns = {}
        self.matches = {}
        self.timelines = {}
        self.view_match = {}
        self.cut_pairs = []
        self.warp = lambda t: t
        self._seats = None
        self._tops = None

    def model_space_of(self, xy_model, eid=None, t=None):
        """GT space holding a model-frame point (each space's own alignment decides).

        5.0.12 S7: a space whose GT outline contains the point wins over any extent comparison. S24: a space with no
        alignment (mode none) maps the raw model xy, which says nothing; such spaces are left out of the test, and a
        point outside every aligned space takes the GT track's space label at t (eid's track) when that label is a
        none-mode space (otherwise the 5.0.11 answer over all spaces)."""
        gt = self.gt
        if len(gt.spaces) == 1:
            return gt.spaces[0].id
        s7 = fixes.on("S7")
        cands = gt.spaces
        if fixes.on("S24"):
            ok = [s for s in gt.spaces if self.aligns[s.id].mode != "none"]
            if ok and len(ok) < len(gt.spaces):
                cands = ok
        best, bd = None, None
        for s in cands:
            p = self.aligns[s.id].xy(np.asarray(xy_model)[None])[0]
            lo, hi = s.extent
            d = float(np.max(np.maximum(lo - p, p - hi)))
            inside = s.outline is not None and geom.point_in_poly(p, s.outline)
            if s7:
                key = (0 if inside else 1, d)
            else:
                if inside:
                    d = min(d, -0.5)
                key = d
            if bd is None or key < bd:
                best, bd = s.id, key
        if cands is not gt.spaces:
            outside = (bd[0] == 1 and bd[1] > 0) if s7 else bd > 0
            if outside:
                tr = gt.tracks.get(eid) if eid is not None and t is not None else None
                lab = tr.label_at(t, "space") if tr is not None and len(tr.t) else None
                if lab in gt.space_by and self.aligns[lab].mode == "none":
                    return lab
                return self._space_of_all(xy_model, s7)
        return best

    def _space_of_all(self, xy_model, s7):
        best, bd = None, None
        for s in self.gt.spaces:
            p = self.aligns[s.id].xy(np.asarray(xy_model)[None])[0]
            lo, hi = s.extent
            d = float(np.max(np.maximum(lo - p, p - hi)))
            inside = s.outline is not None and geom.point_in_poly(p, s.outline)
            key = (0 if inside else 1, d) if s7 else (min(d, -0.5) if inside else d)
            if bd is None or key < bd:
                best, bd = s.id, key
        return best

    def snap_time(self, o):
        """5.0.12 (S19): GT time an entity object's box describes: the middle of the GT shot it was lifted from
        (src.shot, video items), else 0 (the opening layout)."""
        src = o.raw.get("src")
        if isinstance(src, dict) and src.get("shot"):
            for sh in self.gt.shots:
                if sh.id == src["shot"]:
                    return 0.5 * (sh.t0 + sh.t1)
        return 0.0

    def entity_instance(self, eid, own_row=False):
        """Static instance of a named body for the layout layers (alignment anchors, A2 objects and relations, D1
        penetration). 5.0.11: the body at its first present frame. 5.0.12 (S19): the body at the time its GT object
        box describes (snap_time; nearest present frame within snapshot_tol_s), None when it is absent then; a body
        without a GT object keeps its first present frame. Amended 10-19: for the object's own A2 row (own_row) a body
        absent at the snap time is its nearest present frame instead of no instance (the model has the object; the
        anchors, relations and D1 still skip it)."""
        md = self.model
        if not fixes.on("S19"):
            return md.body_instance(eid)
        b = md.body.get(eid)
        if b is None or b.P0 is None:
            return None
        o = next((x for x in self.gt.objects if x.id == eid), None)
        if o is None:
            return md.body_instance(eid)
        ts = self.snap_time(o)
        idx = np.where(b.present)[0]
        k = int(idx[np.argmin(np.abs(md.an_t[idx] - ts))])
        if abs(float(md.an_t[k]) - ts) > self.P["snapshot_tol_s"] and not own_row:
            return None
        return md.body_instance(eid, k)

    def space_at(self, sid, t, model=False):
        gt = self.gt
        if model:
            tl = self.timelines.get(sid)
            if tl is not None:
                ai = tl.at(self.warp(t) if gt.modality != "image" else t)
                if tl.space[ai]:
                    return tl.space[ai]
        tr = gt.tracks.get(sid)
        if tr is not None and len(tr.t):
            sp = tr.label_at(t, "space")
            if sp:
                return sp
            return gt.space_of_point(tr.at(t))
        return gt.spaces[0].id

    def seat_boxes(self):
        if self._seats is None:
            out = []
            for m in self.model.instances:
                if m.cls in SEAT_CLASSES:
                    out.append((m.lo, m.hi, m.top))
            for eid, b in self.model.body.items():
                if b.cls in SEAT_CLASSES and b.first is not None:
                    out.append((b.lo[b.first], b.hi[b.first], b.hi[b.first][2]))
            self._seats = out
        return self._seats

    def support_tops(self):
        """(oriented footprint in the GT frame (geom.gt_rect), top height above floor) of static instances."""
        if self._tops is None:
            out = []
            for m in self.model.instances:
                if m.kind == "untagged_floor":
                    continue
                sp = getattr(m, "gt_space", None) or self.gt.spaces[0].id
                al = self.aligns[sp]
                out.append((geom.gt_rect(al.xy, m.rect[0], m.long_yaw, m.long, m.short), float(al.h(m.top))))
            self._tops = out
        return self._tops

    def model_time_for(self, sh, t):
        return camera.model_time_for(self, sh, t)


def load_thresholds(path=THRESH):
    return json.loads(Path(path).read_text())


def undelivered(gt, reason):
    return {"schema": VERSION, "item_id": gt.item_id, "modality": gt.modality, "delivered": False, "total": 0.0,
            "reason": reason, "dims": {d: None for d in DIMS}, "items": [], "pass_rate": 0.0}


def score_item(gt_path, dump_dir, cats=None, th_all=None):
    t0 = time.monotonic()
    cats = cats or load_categories(CATS)
    th_all = th_all or load_thresholds()
    fixes.set_active(fixes.ids_from(th_all))    # 5.0.12: before the GT is read (S7 acts in GT.finish)
    gt = GT(gt_path, cats)
    acq = {}
    if dump_dir is not None and (Path(dump_dir) / "acquire.json").is_file():
        acq = json.loads((Path(dump_dir) / "acquire.json").read_text())
    if dump_dir is None or not (Path(dump_dir) / "dump.json").is_file():
        rep = undelivered(gt, "no dump (%s)" % (acq.get("harness", {}).get("reason") or acq.get("status") or
                                                "scene.blend missing or dump failed"))
        rep["acquire"] = {k: acq.get(k) for k in ("status", "harness")} if acq else None
        return rep
    md = Model(dump_dir, cats, th_all["params"])
    ctx = Ctx(gt, md, th_all)
    gt_ids = set(gt.entities)
    md.layout(gt_ids)
    md.bodies(gt_ids)
    ctx.tag_report = md.tag_report(gt_ids, [s.id for s in gt.spaces])
    timings = {}
    items = Items()
    errors = []

    def stage(name, fn):
        ts = time.monotonic()
        try:
            fn()
        except Exception:
            errors.append({"stage": name, "error": traceback.format_exc(limit=6)})
        timings[name] = round(time.monotonic() - ts, 3)

    stage("align", lambda: layout.run_alignment(ctx))
    if not ctx.aligns:
        ctx.aligns = {s.id: layout.Align(s.id, geom.Sim2(), "none", 0, 0.0, s.floor_z, s.L) for s in gt.spaces}
    stage("timelines", lambda: ctx.timelines.update({eid: motion.Timeline(ctx, eid) for eid in md.body}))
    stage("A1", lambda: layout.score_A1(ctx, items))
    stage("A2", lambda: layout.score_A2(ctx, items))
    stage("A3", lambda: layout.score_A3(ctx, items))
    stage("warp", lambda: camera.build_warp(ctx) if gt.modality != "image" else camera.view_match(ctx))
    stage("C1", lambda: camera.score_C1(ctx, items))
    stage("C2", lambda: camera.score_C2(ctx, items))
    stage("P1", lambda: frame.score_P1(ctx, items))
    stage("B1", lambda: motion.score_B1(ctx, items))
    stage("B2", lambda: motion.score_B2(ctx, items))
    stage("D1", lambda: plaus.score_D1(ctx, items))
    q = Items()
    stage("Q", lambda: plaus.score_Q(ctx, q))
    dims = items.dim_scores(DIMS)
    W = th_all["weights"]
    if gt.modality == "text":
        dims["P1"] = None
    num = sum(W[d] * v["score"] for d, v in dims.items() if v is not None)
    den = sum(W[d] for d, v in dims.items() if v is not None)
    total = 100.0 * num / den if den else 0.0
    for d, v in dims.items():
        if v is not None:
            v["weight"] = W[d]
            v["weight_eff"] = round(W[d] / den, 4)
    scored = [r for r in items.rows if r["dim"] in DIMS and dims.get(r["dim"]) is not None]
    rep = {"schema": VERSION, "item_id": gt.item_id, "modality": gt.modality, "image_kind": gt.image_kind,
           "delivered": True, "total": round(total, 3), "dims": dims, "harness": acq.get("harness"),
           "pass_rate": round(sum(1 for r in scored if r["score"] >= 0.999) / max(1, len(scored)), 4),
           "n_items": len(scored), "Q": q.dim_scores(["Q"])["Q"], "Q_items": q.rows,
           "alignment": {k: a.to_dict() for k, a in ctx.aligns.items()},
           "model": {"duration_s": round(md.duration, 3), "gt_duration_s": gt.duration, "n_parts": md.N,
                     "n_instances": len(md.instances), "instances": [m.summary() for m in md.instances][:200],
                     "bodies": {eid: {"class": b.cls, "present_frac": round(float(b.present.mean()), 3),
                                      "dup_frac": round(float(b.dup.mean()), 3), "parts": len(b.parts)}
                                for eid, b in md.body.items()},
                     "shots": [{"t0": round(s["t0"], 3), "t1": round(s["t1"], 3), "why": s["why"]} for s in md.shots],
                     "cut_pairs": [[round(a, 3), round(b, 3)] for a, b in ctx.cut_pairs],
                     "view_match": ctx.view_match},
           "tag_report": ctx.tag_report, "gt_problems": gt.problems, "errors": errors, "timings_s": timings,
           "items": items.rows, "dump_meta": {k: md.meta.get(k) for k in ("version", "blender", "pvtools",
                                                                           "pvtools_sha256", "fps", "frame_start",
                                                                           "frame_end", "grid")}}
    rep["timings_s"]["total"] = round(time.monotonic() - t0, 3)
    return rep


# ------------------------------------------------------------------------------------------------ aggregate
def aggregate(reports, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    tot = np.array([r["total"] for r in reports], float)
    out = {"n": len(reports), "mean_total": round(float(tot.mean()), 3) if len(tot) else None}
    if len(tot) >= 2:
        bs = [float(tot[rng.integers(0, len(tot), len(tot))].mean()) for _ in range(n_boot)]
        out["ci95"] = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    dl = [r for r in reports if r.get("delivered")]
    out["delivery_rate"] = round(len(dl) / max(1, len(reports)), 4)
    out["delivered_only_mean"] = round(float(np.mean([r["total"] for r in dl])), 3) if dl else None
    out["pass_rate"] = round(float(np.mean([r.get("pass_rate", 0.0) for r in reports])), 4) if reports else None
    dims = {}
    for d in DIMS:
        v = [r["dims"][d]["score"] for r in dl if r["dims"].get(d)]
        dims[d] = {"mean": round(float(np.mean(v)), 4), "n": len(v)} if v else None
    out["dims"] = dims
    groups = {}
    for r in reports:
        groups.setdefault(r.get("modality"), []).append(r["total"])
    out["by_modality"] = {k: {"n": len(v), "mean": round(float(np.mean(v)), 3)} for k, v in groups.items()}
    # weight sensitivity: totals recomputed with each dimension weight x0.5 and x1.5
    W = load_thresholds()["weights"]
    sens = {}
    for d in DIMS:
        res = {}
        for f in (0.5, 1.5):
            ww = dict(W)
            ww[d] = W[d] * f
            vals = []
            for r in reports:
                if not r.get("delivered"):
                    vals.append(0.0)
                    continue
                num = sum(ww[k] * v["score"] for k, v in r["dims"].items() if v)
                den = sum(ww[k] for k, v in r["dims"].items() if v)
                vals.append(100 * num / den if den else 0.0)
            res["x%.1f" % f] = round(float(np.mean(vals)), 3) if vals else None
        sens[d] = res
    out["weight_sensitivity"] = sens
    return out


def _dump(obj, path):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=default))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("item")
    a.add_argument("--gt", required=True)
    a.add_argument("--dump")
    a.add_argument("--out", required=True)
    g = sub.add_parser("agg")
    g.add_argument("--reports", nargs="+", required=True)
    g.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "item":
        rep = score_item(args.gt, args.dump)
        _dump(rep, args.out)
        print(json.dumps({"item": rep["item_id"], "total": rep["total"],
                          "dims": {k: (v["score"] if v else None) for k, v in rep["dims"].items()},
                          "errors": len(rep.get("errors", []))}))
        return 0
    reps = [json.loads(Path(p).read_text()) for p in args.reports]
    _dump(aggregate(reps), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
