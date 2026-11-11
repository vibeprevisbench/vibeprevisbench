"""Motion layer: B1 (trajectories) and B2 (events, postures) from named bodies over the analysis frames."""
from __future__ import annotations

import math
import re

import numpy as np

from . import fixes, geom
from .layout import static_target, unmatched_static
from .model import BODY_GROUPS, SEAT_CLASSES

POSTURE_ALIAS = {"walk": "stand", "run": "stand", "standing": "stand", "sitting": "sit", "lying": "lie",
                 "kneel": "crouch", "squat": "crouch"}


def norm_posture(p):
    if p is None:
        return None
    return POSTURE_ALIAS.get(p, p)


# 5.3 (M18, scorer/v5_3 config/view/PREREG_53.json): the box-span support of the airborne test. 5.0 counts a body
# whose base lies inside any box's height span (bottom <= base + 0.1 < top, centre over its footprint) as held
# ('inside': a seat, a slope, a vehicle), so a hop beside or onto a solid box below its top (a bin, a vent, a wall, a
# railing) or a vault through it has no airborne frame. M18: the span holds a body only for the classes a person sits,
# lies or stands in below the box top (SPAN_GROUPS, SPAN_CLASSES; an other:* / unknown class by its name, SPAN_NAMES,
# or its last word, the head noun, SPAN_WORDS: other:pet_bed yes, other:bike_rack no); any other box (untagged
# geometry too) holds it at its top (as before) or at the top of one of its pieces under the body (a low part of a
# merged instance). The frames this adds are airborne only inside a run (gaps <= 0.15 s bridged, as jump_runs) of at
# most jump_max_s whose base rises more than jump_air_m above its take-off (the base at the last frame before the run,
# else at its first frame): a hop or a vault rises; a body cannot stay airborne, so a longer stay inside such a span
# (sitting on a crate or a counter with the feet hanging) is held, and a step down (the lowest point is the descending
# foot, the base only falls) is held, as in 5.2. Postures (Timeline._posture_m5) and contacts (support_contact) read
# the 5.2 support unchanged.
SPAN_GROUPS = {"terrain", "vehicle"}
SPAN_CLASSES = {"floor", "stairs", "ramp", "platform", "stage", "roof", "elevator", "escalator", "tunnel", "bridge",
                "pier", "dock", "building", "chair", "armchair", "sofa", "bench", "bench_outdoor", "stool", "bed",
                "toilet", "bathtub", "shower", "picnic_table", "rug", "ladder", "fountain", "scaffold",
                "shipping_container", "crane", "skate_bowl"}
SPAN_NAMES = SPAN_CLASSES | {
    "ground", "road", "sidewalk", "crosswalk", "curb", "parking_lot", "grass", "sand", "water", "shoreline", "hill",
    "rock", "cliff", "path", "track", "field", "court", "pool",
    "car", "van", "truck", "bus", "train", "subway_car", "tram", "motorcycle", "bicycle", "scooter", "skateboard", "boat",
    "ship", "canoe", "helicopter", "airplane", "forklift", "cart", "stroller", "wheelchair", "fire_truck", "trailer",
    "tractor", "race_car", "go_kart", "gurney"}
SPAN_WORDS = {
    "seat", "seating", "chair", "armchair", "sofa", "couch", "settee", "loveseat", "recliner", "bench", "stool",
    "ottoman", "pouf", "beanbag", "cushion", "mat", "bed", "mattress", "futon", "cot", "bunk", "hammock", "toilet",
    "bathtub", "tub", "bath", "jacuzzi", "spa", "pool", "fountain", "shower", "booth", "pew", "bleacher", "bleachers",
    "grandstand", "lounger", "sunbed", "deckchair", "chaise", "crib", "cradle", "swing", "seesaw", "slide",
    "trampoline", "springboard", "playground", "playset", "carousel", "tent", "treehouse",
    "stair", "stairs", "staircase", "stairway", "stairwell", "step", "steps", "ramp", "slope", "floor", "platform",
    "stage", "deck", "roof", "rooftop", "balcony", "terrace", "porch", "bridge", "footbridge", "pier", "dock", "jetty",
    "boardwalk", "walkway", "catwalk", "gangway", "gangplank", "building", "house", "shed", "cabin", "hut", "tunnel",
    "elevator", "lift", "escalator", "ladder", "scaffold", "scaffolding", "container", "crane", "landing", "patio",
    "veranda", "mezzanine", "loft", "attic", "gazebo", "pavilion",
    "ground", "terrain", "hill", "hillside", "mound", "dune", "rock", "boulder", "cliff", "grass", "lawn", "sand",
    "dirt", "soil", "mud", "gravel", "snow", "ice", "water", "puddle", "river", "lake", "pond", "sea", "ocean", "shore",
    "shoreline", "beach", "field", "court", "road", "street", "sidewalk", "path", "trail", "track", "ditch", "trench",
    "pit", "crater", "glacier", "swamp", "marsh", "stream", "creek", "wave", "waves", "surf",
    "car", "van", "truck", "bus", "train", "tram", "subway", "carriage", "wagon", "cart", "trolley", "boat", "ship",
    "raft", "kayak", "canoe", "yacht", "ferry", "gondola", "sled", "sleigh", "motorcycle", "motorbike", "bike",
    "bicycle", "scooter", "skateboard", "stroller", "wheelchair", "forklift", "tractor", "trailer", "jeep", "suv",
    "taxi", "pickup", "vehicle", "helicopter", "airplane", "plane", "jet", "kart", "gurney", "ambulance", "rickshaw",
    "bobsled", "snowmobile", "submarine", "pram", "buggy", "unicycle", "tricycle", "atv", "minibus", "minivan",
    "limousine", "locomotive", "caravan", "camper", "hovercraft", "jetski", "sailboat", "speedboat", "rowboat",
    "lifeboat", "dinghy", "barge", "tugboat", "aircraft", "glider"}


def span_holds(cls, group):
    """5.3 (M18): a box of this class (group) holds a body anywhere in its height span"""
    if group in SPAN_GROUPS or cls in SPAN_CLASSES:
        return True
    if group in ("other", "unknown") and isinstance(cls, str):
        w = cls.split(":", 1)[-1].strip().lower()
        words = [x for x in re.split(r"[^a-z]+", w) if x]
        return w in SPAN_NAMES or bool(words) and words[-1] in SPAN_WORDS
    return False


class Timeline:
    """Aligned (GT-frame) state of one model body at every analysis frame."""

    def __init__(self, ctx, eid):
        md, gt = ctx.model, ctx.gt
        self.id = eid
        b = md.body[eid]
        self.b = b
        A = md.A
        self.t = md.an_t
        self.present = b.present.copy()
        self.xy = np.full((A, 2), np.nan)
        self.space = [None] * A
        self.h_top = np.full(A, np.nan)
        self.h_base = np.full(A, np.nan)
        self.facing = np.full(A, np.nan)
        self.long_yaw = np.full(A, np.nan)
        for ai in range(A):
            if not b.present[ai]:
                continue
            sp = ctx.model_space_of(b.center[ai], eid, float(md.an_t[ai]))   # 5.0.12 (S24): eid / t used there
            al = ctx.aligns[sp]
            self.space[ai] = sp
            self.xy[ai] = al.xy(b.center[ai][None])[0]
            self.h_top[ai] = float(al.h(b.top[ai]))
            self.h_base[ai] = float(al.h(b.base[ai]))
            if not np.isnan(b.facing[ai]):
                self.facing[ai] = al.yaw(b.facing[ai]) % 360
            self.long_yaw[ai] = al.yaw(b.long_yaw[ai]) % 180
        # speed (GT metres / s)
        self.speed = np.zeros(A)
        for ai in range(1, A):
            if self.present[ai] and self.present[ai - 1]:
                dt = max(1e-6, self.t[ai] - self.t[ai - 1])
                self.speed[ai] = float(np.linalg.norm(self.xy[ai] - self.xy[ai - 1])) / dt
        if A > 1:
            self.speed[0] = self.speed[1]
        # vehicles: facing from motion along the long axis
        if b.group == "vehicle":
            s25, s13 = fixes.on("S25"), fixes.on("S13")
            shot = [md.shot_of_frame(int(md.an_frame[ai]) - md.f0) for ai in range(A)] if (s25 or s13) else None
            vmax = ctx.P["teleport_speed_ms"]["vehicle"]
            signed, cut = np.zeros(A, bool), np.zeros(A, bool)
            for ai in range(A):
                if not self.present[ai]:
                    continue
                j = min(A - 1, ai + 2)
                i = max(0, ai - 2)
                if self.present[i] and self.present[j] and np.linalg.norm(self.xy[j] - self.xy[i]) > 0.2:
                    # 5.0.12 (S25): a window across a model cut (or faster than a vehicle) is no motion
                    if s25 and (shot[i] != shot[j] or float(np.linalg.norm(self.xy[j] - self.xy[i])) /
                                max(1e-6, self.t[j] - self.t[i]) > vmax):
                        cut[ai] = True
                        continue
                    mv = geom.yaw_of(self.xy[j] - self.xy[i])
                    ly = self.long_yaw[ai]
                    self.facing[ai] = ly if geom.angdiff(ly, mv, 360) <= 90 else (ly + 180) % 360
                    signed[ai] = True
            if s25 or s13:
                self._carry_vehicle_facing(signed, cut if not s13 else self.present.copy(), shot,
                                           ctx.P["near_square"], backfill=s13)
        self.extent = b.top - b.base
        # height read for posture and falls: the top above the floor, or (5.0.2) in a sloped space, whose floor level
        # under the body is unknown, the body's own height (top - base)
        self.h_post = self.h_top.copy()
        for ai in range(A):
            sl = ctx.gt.space_by.get(self.space[ai])
            if sl is not None and sl.sloped:
                self.h_post[ai] = float(self.extent[ai]) * ctx.aligns[self.space[ai]].sim.s
        self.posture = [None] * A
        if b.group in BODY_GROUPS:
            ext = self.extent[self.present] * ctx.aligns[ctx.gt.spaces[0].id].sim.s
            ref = max(float(np.percentile(ext, 95)) if len(ext) else 1.7, 1.55)
            if b.group == "animal":
                ref = float(np.percentile(ext, 95)) if len(ext) else 0.5
            self.ref_h = ref
            seats = ctx.seat_boxes()
            for ai in range(A):
                if not self.present[ai]:
                    self.posture[ai] = "hidden"
                    continue
                r = self.h_post[ai] / ref
                lo, hi = b.lo[ai], b.hi[ai]
                if r >= ctx.P["stand_ratio"]:
                    p = "stand"
                elif r < ctx.P["lie_ratio"]:
                    horiz = float(max(hi[0] - lo[0], hi[1] - lo[1])) * ctx.aligns[self.space[ai]].sim.s
                    p = "lie" if horiz >= 0.6 * ref else "crouch"
                else:
                    on_seat = any(np.all(lo[:2] <= s_hi[:2] + 0.15) and np.all(hi[:2] >= s_lo[:2] - 0.15)
                                  for s_lo, s_hi, _ in seats)
                    p = "sit" if on_seat else "crouch"
                self.posture[ai] = p
            if fixes.on("M5"):
                # 5.2 (M5): the classes are read again on first use (_posture_m5), when every timeline exists
                self.ref_m = standing_height(self.extent[self.present], b.group, ctx.P)
                self._m5 = ctx
                self._posture = None

    @property
    def posture(self):
        if self._posture is None:
            self._posture = self._posture_m5()
        return self._posture

    @posture.setter
    def posture(self, v):
        self._posture = v

    def _posture_m5(self):
        """5.2 (M5): posture classes from the height of the top above the support under the body (Events.support,
        the airborne test's surface; a body inside a box's height span (a slope, a vehicle, a seat) rests on it at
        its own base; in a sloped space the body's own extent, as 5.0.2) over the body's own standing height in model
        units (ref_m); lie also from the body's shape: vertical extent below lie_ratio x ref_m with the horizontal
        extent at least 0.6 ref_m (the 5.0 lie test) on whatever surface supports it (not airborne by the airborne
        test's jump_air_m). The ratio is kept in r_m5 (falls)."""
        ctx, b, A = self._m5, self.b, len(self.t)
        ev, seats, ref = Events(ctx), ctx.seat_boxes(), self.ref_m
        out, self.r_m5 = [None] * A, np.full(A, np.nan)
        for ai in range(A):
            if not self.present[ai]:
                out[ai] = "hidden"
                continue
            ext = float(self.extent[ai])
            sl = ctx.gt.space_by.get(self.space[ai])
            if sl is not None and sl.sloped:
                h, held = ext, True
            else:
                base, sup, inside = ev.support(self.id, ai, ride=False)
                # the whole body below that surface (its top not above it: parked under the set, another storey, a
                # misaligned floor) has no height above it to read: its own extent (as a slope)
                under = float(self.h_top[ai]) <= sup
                h = ext if inside or under else (float(self.h_top[ai]) - sup) / ctx.aligns[self.space[ai]].sim.s
                held = inside or base - sup <= ctx.P.get("jump_air_m", 0.12)
            r = self.r_m5[ai] = h / ref
            lo, hi = b.lo[ai], b.hi[ai]
            flat = float(max(hi[0] - lo[0], hi[1] - lo[1])) >= 0.6 * ref
            if held and ext < ctx.P["lie_ratio"] * ref and flat:
                p = "lie"
            elif r >= ctx.P["stand_ratio"]:
                p = "stand"
            elif r < ctx.P["lie_ratio"]:
                p = "lie" if flat else "crouch"
            else:
                on_seat = any(np.all(lo[:2] <= s_hi[:2] + 0.15) and np.all(hi[:2] >= s_lo[:2] - 0.15)
                              for s_lo, s_hi, _ in seats)
                p = "sit" if on_seat else "crouch"
            out[ai] = p
        return out

    def posture_ratio(self):
        """5.2 (M5): the per-frame height ratio behind the posture classes (falls)"""
        if fixes.on("M5") and hasattr(self, "_m5"):
            self.posture
            return self.r_m5
        return self.h_post / max(1e-6, getattr(self, "ref_h", 1.7))

    def _carry_vehicle_facing(self, signed, want, shot, near_sq, backfill):
        """5.0.12 (S13 / S25): frames in want without a motion sign take the long axis sign nearest the previous
        facing of the same model shot (with backfill, a shot's leading frames take its first motion facing); not for
        near-square boxes, whose long axis says nothing."""
        A = len(self.t)
        b = self.b

        def put(ai, ref):
            if not (b.long[ai] >= near_sq * b.short[ai]):
                return False
            ly = self.long_yaw[ai]
            self.facing[ai] = ly if geom.angdiff(ly, ref, 360) <= 90 else (ly + 180) % 360
            return True

        for order in ((range(A),) + ((range(A - 1, -1, -1),) if backfill else ())):
            ref, cur = None, None
            for ai in order:
                if not self.present[ai]:
                    continue
                if shot[ai] != cur:
                    ref, cur = None, shot[ai]
                if signed[ai] or not np.isnan(self.facing[ai]):
                    ref = self.facing[ai] if not np.isnan(self.facing[ai]) else ref
                    continue
                if want[ai] and ref is not None and put(ai, ref):
                    ref = self.facing[ai]

    def at(self, t):
        ai = int(np.argmin(np.abs(self.t - t)))
        return ai

    def frame(self, t, max_dt=0.6):
        """5.0.12: analysis frame pos(t) reads (None when pos(t) is None)."""
        ai = self.at(t)
        if self.present[ai]:
            return ai
        idx = np.where(self.present)[0]
        if not len(idx):
            return None
        j = int(idx[np.argmin(np.abs(self.t[idx] - t))])
        return j if abs(self.t[j] - t) <= max_dt else None

    def pos(self, t, max_dt=0.6):
        ai = self.at(t)
        if self.present[ai]:
            return self.xy[ai]
        idx = np.where(self.present)[0]
        if not len(idx):
            return None
        j = int(idx[np.argmin(np.abs(self.t[idx] - t))])
        return self.xy[j] if abs(self.t[j] - t) <= max_dt else None


def standing_height(ext, group, P):
    """5.2 (M5): a body's own standing height in model units: the median vertical extent of its upright frames, the
    frames at least stand_ratio x its 95th percentile (raised arms or a stretch do not set it). A person whose upright
    height stays below the 5.0 floor (1.55, a standing adult, now in the model's own units) never stands up in the
    clip (seated, crouched): the floor, as 5.0."""
    if not len(ext):
        return 0.5 if group == "animal" else 1.7
    p95 = float(np.percentile(ext, 95))
    h = float(np.median(ext[ext >= P["stand_ratio"] * p95]))
    return h if group == "animal" else max(h, 1.55)


def runs(mask, t, min_len=0.0, fill=0.0):
    """List of (t_start, t_end) where mask is true (gaps <= fill seconds bridged)."""
    out = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if m else i - 1
            out.append([t[start], t[end]])
            start = None
    merged = []
    for r in out:
        if merged and r[0] - merged[-1][1] <= fill:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [tuple(r) for r in merged if r[1] - r[0] >= min_len]


# ------------------------------------------------------------------------------------------------------ B1
def score_B1(ctx, items):
    gt, md, TH = ctx.gt, ctx.model, ctx.TH
    tls = ctx.timelines
    for tid, tr in gt.tracks.items():
        if len(tr.t) < 1:
            continue
        spaces = [s for s in tr.space if s] or [gt.space_of_point(tr.p[0])]
        L = gt.union_L(set(spaces))
        tl = tls.get(tid)
        if tl is None or not tl.present.any():
            has_shape = any(y - x >= 0.5 for x, y in tr.segs)  # 5.0.3: as below, no shape item without a known span
            for b in ("shape", "endpoints") if has_shape else ("endpoints",):
                items.add("B1", b, "%s:%s" % (b, tid), 0.0, 1.0, value="missing")
            continue
        aligned = all(ctx.aligns[s].ok for s in set(spaces) if s in ctx.aligns)
        t0, t1 = tr.t[0], tr.t[-1]
        sel = (tl.t >= t0 - 1e-6) & (tl.t <= t1 + 1e-6) & tl.present
        Pm = tl.xy[sel]
        Pg = np.array([tr.at(t) for t in np.linspace(t0, t1, max(2, int((t1 - t0) * 8) + 1))])
        lg = geom.path_length(Pg)
        sig = tr.unc / L
        # 5.0.3: the shape is compared over the known spans of >= 0.5 s only (a dense track is one span, as before);
        # the worst span counts. No span (a single sample, or stated positions with only long moving gaps): no item.
        frs = []
        for a, b in [(a, b) for a, b in tr.segs if b - a >= 0.5]:
            ss = (tl.t >= a - 1e-6) & (tl.t <= b + 1e-6) & tl.present
            if ss.sum() < 2:
                frs = None
                break
            G = np.array([tr.at(t) for t in np.linspace(a, b, max(2, int((b - a) * 8) + 1))])
            if aligned:
                frs.append(geom.frechet(geom.resample_path(G, 40), geom.resample_path(tl.xy[ss], 40)))
            else:
                # no alignment: compare the shape up to a rigid transform (Procrustes on resampled paths)
                A_, B_ = geom.resample_path(tl.xy[ss], 40), geom.resample_path(G, 40)
                sim = geom.umeyama2(A_, B_, fix_scale=ctx.aligns[spaces[0]].sim.s)
                frs.append(geom.frechet(B_, sim.apply(A_)))
        if frs is None:
            items.add("B1", "shape", "shape:" + tid, 0.0, 1.0, value="too few frames")
        elif frs and aligned:
            fr = max(frs)
            items.add("B1", "shape", "shape:" + tid, geom.lin(fr / L, *geom.widen(TH["frechet_frac"], sig)), 1.0,
                      value=round(fr, 3))
        elif frs:
            fr = max(frs)
            items.add("B1", "shape", "shape:" + tid, 0.5 * geom.lin(fr / L, *TH["frechet_frac"]), 1.0,
                      value=round(fr, 3), mode="rigid-only")
        for nm, t in (("start", t0), ("end", t1)):
            if not aligned:
                items.add("B1", "endpoints", "%s:%s" % (nm, tid), 0.0, 1.0, value="no alignment")
                continue
            p = tl.pos(t)
            if p is None:
                items.add("B1", "endpoints", "%s:%s" % (nm, tid), 0.0, 1.0, value="absent")
                continue
            d = float(np.linalg.norm(p - tr.at(t)))
            items.add("B1", "endpoints", "%s:%s" % (nm, tid), geom.lin(d / L, *geom.widen(TH["endpoint_frac"], sig)), 1.0,
                      value=round(d, 3))
        wps = tr.route.get("waypoints") or []
        if wps and aligned and len(Pm):
            idx0 = 0
            for wi, w in enumerate(wps):
                w = np.asarray(w, float)[:2]
                d = np.linalg.norm(Pm[idx0:] - w, axis=1) if idx0 < len(Pm) else np.array([np.inf])
                j = int(np.argmin(d))
                items.add("B1", "waypoints", "wp%d:%s" % (wi, tid), geom.lin(float(d[j]) / L, *TH["frechet_frac"]),
                          1.0, value=round(float(d[j]), 3))
                idx0 = idx0 + j
        sg = geom.collapse([s for s in tr.space if s])
        if len(sg) >= 2:
            sm = geom.collapse([tl.space[i] for i in range(md.A) if sel[i]])
            items.add("B1", "space_order", "spaces:" + tid, geom.seq_similarity(sg, sm) or 0.0, 1.0,
                      value={"gt": sg, "model": sm})
        # heading vs facing at the GT samples that carry a yaw. 5.0.12 (S13): not for a body whose facing the scorer
        # never reads (no agent / animal / vehicle / seat): no row instead of 0
        hs = []
        no_facing = fixes.on("S13") and tl.b.group not in ("agent", "animal", "vehicle") and tl.b.cls not in SEAT_CLASSES
        for i, t in enumerate([] if no_facing else tr.t):
            if tr.yaw[i] is None:
                continue
            ai = tl.at(t)
            if not tl.present[ai]:
                continue
            f = tl.facing[ai]
            if np.isnan(f) and fixes.on("M14") and tl.b.facing_unk[ai]:
                continue            # 5.2 (M14): facing unknown at that time, not scored
            hs.append(0.0 if np.isnan(f) else geom.lin(geom.angdiff(f, tr.yaw[i], 360), *TH["heading_deg"]))
        if hs and aligned:
            items.add("B1", "heading", "heading:" + tid, float(np.mean(hs)), 1.0, n=len(hs))
        # progress along the path over time
        if lg > 0.5 and len(Pm) >= 2 and len(tr.t) >= 2 and len(tr.segs) == 1:  # 5.0.3: not across unknown gaps
            tt = np.linspace(t0, t1, 25)
            cg = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Pg, axis=0), axis=1))])
            tg = np.linspace(t0, t1, len(Pg))
            pg_ = np.interp(tt, tg, cg) / max(1e-6, cg[-1])
            tm_ = tl.t[sel]
            cm = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Pm, axis=0), axis=1))])
            if cm[-1] > 1e-6:
                pm_ = np.interp(tt, tm_, cm) / cm[-1]
                err = float(np.mean(np.abs(pm_ - pg_)))
                lm = float(cm[-1])
                ratio_pen = geom.lin(abs(math.log(max(1e-3, lm) / lg)), math.log(1.3), math.log(3))
                items.add("B1", "progress", "progress:" + tid, geom.lin(err, *TH["progress"]) * ratio_pen, 1.0,
                          value=round(err, 3), length={"gt": round(lg, 2), "model": round(lm, 2)})
            else:
                items.add("B1", "progress", "progress:" + tid, 0.0, 1.0, value="no motion")
        # stops (speed < 0.15 m/s for >= 1 s) agreement over the track span
        tt = np.arange(t0, t1 + 1e-6, 0.125)
        keep = np.array([any(a - 1e-6 <= t <= b + 1e-6 for a, b in tr.segs) for t in tt], bool)  # 5.0.3: known spans
        if keep.sum() > 8 and lg > 0.5:
            pgs = np.array([tr.at(t) for t in tt])
            vg = np.r_[0, np.linalg.norm(np.diff(pgs, axis=0), axis=1) / 0.125]
            vm = np.array([tl.speed[tl.at(t)] for t in tt])
            sg_ = _sustain((vg < 0.15) & keep, 8)
            sm_ = _sustain((vm < 0.15) & keep, 8)
            if sg_.any():
                iou = float((sg_ & sm_).sum()) / max(1, float((sg_ | sm_).sum()))
                items.add("B1", "stops_turns", "stops:" + tid, geom.lin_up(iou, 0.7, 0.2), 1.0, value=round(iou, 3))
    ids = [k for k in gt.tracks if len(gt.tracks[k].t) >= 2]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = gt.tracks[ids[i]], gt.tracks[ids[j]]
            ta, tb = tls.get(ids[i]), tls.get(ids[j])
            t0, t1 = max(a.t[0], b.t[0]), min(a.t[-1], b.t[-1])
            if t1 <= t0:
                continue
            L = gt.union_L(set([s for s in a.space + b.space if s]) or {gt.spaces[0].id})
            ts = [t for t in np.linspace(t0, t1, 20) if a.known(t, 1e-6) and b.known(t, 1e-6)]  # 5.0.3
            if not ts:
                continue
            if ta is None or tb is None:
                items.add("B1", "pair_dist", "pair:%s~%s" % (ids[i], ids[j]), 0.0, 1.0, value="missing")
                continue
            errs = []
            for t in ts:
                pa, pb = ta.pos(t), tb.pos(t)
                if pa is None or pb is None:
                    continue
                dg = float(np.linalg.norm(a.at(t) - b.at(t)))
                dm = float(np.linalg.norm(pa - pb))
                errs.append(abs(dm - dg) / L)
            sc = geom.lin(float(np.mean(errs)), *TH["pair_dist_frac"]) if errs else 0.0
            items.add("B1", "pair_dist", "pair:%s~%s" % (ids[i], ids[j]), sc, 1.0,
                      value=None if not errs else round(float(np.mean(errs)) * L, 3))


def _sustain(mask, n):
    out = np.zeros_like(mask)
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= n:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


# ------------------------------------------------------------------------------------------- event finders
class Events:
    def __init__(self, ctx):
        self.ctx = ctx
        self.md = ctx.model
        self.tl = ctx.timelines
        self.t = self.md.an_t
        self.P = ctx.P

    def tl_of(self, x):
        return self.tl.get(x)

    def static_pos(self, key):
        """GT-frame position of a target: model match of a GT object key / entity id, or a GT point / space."""
        ctx = self.ctx
        if isinstance(key, (list, tuple)):
            return np.asarray(key, float)[:2]
        if key in self.tl:
            return None
        if fixes.on("S15"):
            # 5.0.12: through static_target (A2 match, or the A3 opening match of an opening / structure)
            r = static_target(ctx, key) if isinstance(key, str) else None
            if fixes.on("M8") and r is not None and r[0] is None:
                r = unmatched_static(ctx, key), r[1]       # 5.2 M8
            if r is None or r[0] is None:
                return None
            return ctx.aligns[r[1] or ctx.gt.spaces[0].id].xy(r[0].center2[None])[0]
        m = ctx.matches.get(key)
        if m and m[0] is not None:
            o = ctx.gt.obj_by.get(key)
            sp = o.space if o else ctx.gt.spaces[0].id
            return ctx.aligns[sp].xy(m[0].center2[None])[0]
        return None

    def resolvable(self, key):
        """5.0.12 (S15): a B2 target the scorer can look up at all (a point, a GT object, entity, track or space); any
        other key makes the row unsupported, not 0."""
        gt = self.ctx.gt
        if isinstance(key, (list, tuple)):
            return True
        return key in gt.obj_by or key in gt.entities or key in gt.tracks or key in gt.space_by or key in self.tl

    def static_rect(self, key):
        """5.0.12 (S9 / S11 / S17): GT-frame oriented footprint (geom.gt_rect) of a static target's model instance."""
        ctx = self.ctx
        r = static_target(ctx, key) if isinstance(key, str) else None
        if fixes.on("M8") and r is not None and r[0] is None:
            r = unmatched_static(ctx, key), r[1]           # 5.2 M8
        if r is None or r[0] is None:
            return None
        m = r[0]
        al = ctx.aligns[r[1] or ctx.gt.spaces[0].id]
        return geom.gt_rect(al.xy, m.rect[0], m.long_yaw, m.long, m.short)

    def body_rect(self, tl, ai):
        """5.0.12 (S9): GT-frame oriented footprint of a timeline body at analysis frame ai."""
        b = tl.b
        al = self.ctx.aligns[tl.space[ai] or self.ctx.gt.spaces[0].id]
        return geom.gt_rect(al.xy, b.center[ai], b.long_yaw[ai], b.long[ai], b.short[ai])

    def holding_at(self, mask, e):
        """5.0.12 (D19): the event's own t[0] when the state mask holds at a frame within [t0 - tol, t0 + tol]."""
        t = e.get("t", [0, 0])
        t0 = float(t[0] if isinstance(t, (list, tuple)) else t)
        tol = float(e.get("tol_s", self.ctx.TH["event_tol_s"]))
        w = (self.t >= t0 - tol - 1e-6) & (self.t <= t0 + tol + 1e-6)
        return [t0] if bool(np.asarray(mask, bool)[w].any()) else []

    def support_contact(self, a, gap=0.05):
        """5.0.12 (S18): frames where the body's base is within gap of the surface under it (the airborne test's
        floor / static tops / other bodies' tops)."""
        tl = self.tl_of(a)
        A = len(self.t)
        mask = np.zeros(A, bool)
        if tl is None:
            return mask
        mg = self.P.get("footprint_margin_m", 0.05)
        for ai in range(A):
            if not tl.present[ai] or np.isnan(tl.h_base[ai]):
                continue
            if fixes.on("M4") or fixes.on("M6"):
                # 5.2 (M4 / M6): the airborne test's reading (support); a box spanning the base is still no contact
                base, sup, _ = self.support(a, ai)
                mask[ai] = base - sup <= gap
                continue
            base, xy = tl.h_base[ai], tl.xy[ai]
            sup = 0.0
            for rect, bot, top in self._boxes(ai, a):
                if geom.in_rect(xy, rect, mg) and top <= base + 0.1:
                    sup = max(sup, top)
            mask[ai] = base - sup <= gap
        return mask

    def seat_named(self, ob):
        """5.0.12 (D5): the event's objects name a seat-class GT object or entity."""
        return any(self.ctx.gt.entity_class(k) in SEAT_CLASSES for k in ob if isinstance(k, str))

    def posture_runs(self, a, want):
        tl = self.tl_of(a)
        if tl is None:
            return []
        mask = np.array([p == want for p in tl.posture])
        return runs(mask, self.t, min_len=0.25, fill=0.4)

    def transitions_into(self, a, want, prev=None):
        tl = self.tl_of(a)
        if tl is None:
            return []
        out = []
        pp = None
        for i, p in enumerate(tl.posture):
            if p == "hidden":
                continue
            if p == want and pp is not None and pp != want and (prev is None or pp in prev):
                j = i
                while j < len(tl.posture) and tl.posture[j] == want:
                    j += 1
                if self.t[min(j, len(self.t) - 1)] - self.t[i] >= 0.25 or j >= len(tl.posture):
                    out.append(self.t[i])
            pp = p
        return out

    def _seat_box(self, key):
        md = self.md
        if key in md.body:
            b = md.body[key]
            if b.first is None:
                return None
            return b.lo[b.first], b.hi[b.first]
        m = self.ctx.matches.get(key)
        if m and m[0] is not None:
            return m[0].lo, m[0].hi
        return None

    def _on(self, tl, t, seat):
        ai = tl.at(t)
        lo, hi = tl.b.lo[ai], tl.b.hi[ai]
        s_lo, s_hi = seat
        return bool(np.all(lo[:2] <= s_hi[:2] + 0.2) and np.all(hi[:2] >= s_lo[:2] - 0.2))

    def carried(self, obj, person):
        to, tp = self.tl_of(obj), self.tl_of(person)
        if to is None or tp is None:
            return np.zeros(len(self.t), bool)
        A = len(self.t)
        mask = np.zeros(A, bool)
        win = 4
        tops = self.ctx.support_tops()
        mg = self.P.get("footprint_margin_m", 0.05)
        for ai in range(A):
            if not (to.present[ai] and tp.present[ai]):
                continue
            d = float(np.linalg.norm(to.xy[ai] - tp.xy[ai]))
            if d > self.P["carry_dist_m"]:
                continue
            i, j = max(0, ai - win), min(A - 1, ai + win)
            moving = False
            if to.present[i] and to.present[j] and tp.present[i] and tp.present[j]:
                do = to.xy[j] - to.xy[i]
                dp = tp.xy[j] - tp.xy[i]
                if np.linalg.norm(do) > 0.2 and np.linalg.norm(do - dp) <= 0.35 * np.linalg.norm(dp) + 0.15:
                    moving = True
            lifted = False
            if d <= 0.8:
                base = to.h_base[ai]
                under = [tz for (rect, tz) in tops if geom.in_rect(to.xy[ai], rect, mg) and tz <= base + 0.1]
                sup = max(under) if under else 0.0
                lifted = base - sup > 0.15
            mask[ai] = moving or lifted
        return mask

    def carry_runs(self, obj, person):
        return runs(self.carried(obj, person), self.t, min_len=0.25, fill=0.6)

    def inside_vehicle(self, person, veh):
        tp, tv = self.tl_of(person), self.tl_of(veh)
        if tp is None or tv is None:
            return np.zeros(len(self.t), bool)
        b = self.md.body[veh]
        A = len(self.t)
        mask = np.zeros(A, bool)
        last = None
        for ai in range(A):
            if not tv.present[ai]:
                continue
            if tp.present[ai]:
                last = tp.xy[ai]
                lo, hi = b.lo[ai], b.hi[ai]
                pb = self.md.body[person]
                c = pb.center[ai]
                inside = np.all(c >= lo[:2] - 0.2) and np.all(c <= hi[:2] + 0.2) and pb.base[ai] >= lo[2] - 0.1
                mask[ai] = bool(inside)
            elif last is not None and np.linalg.norm(last - tv.xy[ai]) <= 3.0:
                mask[ai] = True
        return mask

    def _boxes(self, ai, skip, own=(), sp=None, src=None):
        """(oriented footprint in the GT frame (geom.gt_rect), bottom, top) of static instances and of the other
        non-agent bodies at frame ai (a body's footprint is its model-frame box, mapped with its rotation). 5.0.11:
        5.0.10 used the axis-aligned box of the mapped corners, which a rotated thin wall stretches over a whole court.
        5.3 (M18): src, a list, gets (instance or body, alignment) of every box in the same order."""
        ctx, out = self.ctx, []
        for m in ctx.model.instances:
            if m.kind == "untagged_floor":
                continue
            # 5.2 (M6): sp given: every box mapped with the alignment of the tested body's space
            al = ctx.aligns[sp or getattr(m, "gt_space", None) or ctx.gt.spaces[0].id]
            out.append((geom.gt_rect(al.xy, m.rect[0], m.long_yaw, m.long, m.short), float(al.h(m.bottom)),
                        float(al.h(m.top))))
            if src is not None:
                src.append((m, al))
        for k, b in self.md.body.items():
            if k == skip or k in own or b.group in BODY_GROUPS or not b.present[ai]:
                continue
            tl = self.tl.get(k)
            al = ctx.aligns[sp or (tl.space[ai] if tl is not None else None) or ctx.gt.spaces[0].id]
            lo, hi = b.lo[ai], b.hi[ai]
            rect = geom.gt_rect(al.xy, 0.5 * (lo[:2] + hi[:2]), 0.0, hi[0] - lo[0], hi[1] - lo[1])
            out.append((rect, float(al.h(lo[2])), float(al.h(hi[2]))))
            if src is not None:
                src.append((b, al))
        return out

    def _piece_top(self, m, al, xy, base, mg):
        """5.3 (M18): the highest top at or below base + 0.1 of instance m's pieces (part + tag) whose footprint (the
        piece's own minimum-area rectangle, mapped with al) holds xy; 0.0 when none (a named body has no pieces)"""
        pcs = getattr(m, "pieces", None)
        if not pcs:
            return 0.0
        rows = getattr(m, "_m18_rows", None)     # (model-frame footprint, top) per piece, kept on the instance
        if rows is None:
            rows = m._m18_rows = [(geom.min_area_rect(np.asarray(p["P"], float)[:, :2]), float(np.asarray(p["P"])[:, 2].max()))
                                  for p in pcs if len(p["P"])]
        best = 0.0
        for (c, yaw, ln, sh), top in rows:
            h = float(al.h(top))
            if h <= base + 0.1 and h > best and geom.in_rect(xy, geom.gt_rect(al.xy, c, yaw, ln, sh), mg):
                best = h
        return best

    def airborne(self, a):
        """Frames where the body's base is clearly above whatever surface is under it (floor, static top or another
        body's top). Feet inside some box's vertical span (a vehicle, a slope) count as supported."""
        tl = self.tl_of(a)
        A = len(self.t)
        mask = np.zeros(A, bool)
        if tl is None:
            return mask
        air = self.P.get("jump_air_m", 0.12)
        mg = self.P.get("footprint_margin_m", 0.05)
        added, bases = np.zeros(A, bool), np.full(A, np.nan)
        for ai in range(A):
            if not tl.present[ai] or np.isnan(tl.h_base[ai]):
                continue
            if fixes.on("M18"):
                # 5.3 (M18): the 5.2 reading, plus the frames held only by a span that does not hold a body
                base, sup, inside, sup18, inside18 = self.support(a, ai, m18=True)
                bases[ai] = base
                mask[ai] = not inside and base - sup > air
                added[ai] = not mask[ai] and not inside18 and base - sup18 > air
                continue
            if fixes.on("M4") or fixes.on("M6"):
                base, sup, inside = self.support(a, ai)
                mask[ai] = not inside and base - sup > air
                continue
            base, xy = tl.h_base[ai], tl.xy[ai]
            sup, inside = 0.0, False
            for rect, bot, top in self._boxes(ai, a):
                if not geom.in_rect(xy, rect, mg):
                    continue
                if top <= base + 0.1:
                    sup = max(sup, top)
                elif bot <= base + 0.1:
                    inside = True
            mask[ai] = not inside and base - sup > air
        if added.any():
            # 5.3 (M18): the added frames stay airborne only inside a run (gaps <= 0.15 s bridged, as jump_runs) of
            # at most jump_max_s whose base rises more than jump_air_m above the take-off base (the last tested
            # frame before the run, else the run's first); a longer stay or a falling base (a step down) is held
            lim = self.P.get("jump_max_s", 1.5)
            ok = np.isfinite(bases)
            for t0, t1 in runs(mask | added, self.t, fill=0.15):
                w = (self.t >= t0 - 1e-9) & (self.t <= t1 + 1e-9)
                if t1 - t0 > lim + 1e-9 or not (added & w).any():
                    continue
                i0 = int(np.argmax(w))
                before = np.nonzero(ok[:i0])[0]
                ref = bases[before[-1]] if len(before) else bases[w & ok][0]
                if bases[w & ok].max() - ref > air:
                    mask |= added & w
        return mask

    def support(self, a, ai, ride=True, m18=False):
        """5.2 (M4 / M6): (base, top of the surface under it, base inside a box's height span) of body a at analysis
        frame ai, as the 5.0 airborne test reads them (floor 0, static tops and other non-agent bodies' boxes under the
        centre). M6: every box is mapped with the alignment of the space the body sample is bound to, the body's own
        frame (was: each box with its own space's alignment, a body in one aligned space against the supports of
        another). M4 (ride): the agent's own ridden / carried bodies (co_moving) are no support; when it rides one
        (its centre over that body's footprint, that body's bottom at or below its feet: the old 'inside' / 'top'
        cases), the test is the ridden body's base and centre (the rider + vehicle composite).
        5.3 (M18, m18=True): also (sup18, inside18): a span holds the body only for a class that span_holds; the
        span of any other box gives the top of its pieces under the body (_piece_top) instead."""
        tl, ctx = self.tl_of(a), self.ctx
        mg = self.P.get("footprint_margin_m", 0.05)
        sp = (tl.space[ai] or ctx.gt.spaces[0].id) if fixes.on("M6") else None
        base, xy = tl.h_base[ai], tl.xy[ai]
        own = self.co_moving(a, ai) if ride and fixes.on("M4") else []
        under = []
        for k in own:
            kxy, kbase = self._body_at(k, ai, sp)
            vrect = self._body_box(k, ai, sp)
            if geom.in_rect(xy, vrect, mg) and kbase <= base + 0.1:
                under.append((kbase, k, kxy))
        if under:
            base, _, xy = min(under, key=lambda x: x[0])
        sup, inside = 0.0, False
        sup18, inside18, src = 0.0, False, ([] if m18 else None)
        for n, (rect, bot, top) in enumerate(self._boxes(ai, a, own, sp, src)):
            if not geom.in_rect(xy, rect, mg):
                continue
            if top <= base + 0.1:
                sup = max(sup, top)
            elif bot <= base + 0.1:
                inside = True
                if m18:
                    o, al = src[n]
                    if span_holds(o.cls, o.group):
                        inside18 = True
                    else:
                        sup18 = max(sup18, self._piece_top(o, al, xy, base, mg))
        if m18:
            return base, sup, inside, max(sup, sup18), inside18
        return base, sup, inside

    def _body_at(self, k, ai, sp):
        """5.2 (M4): (xy, base height) of body k at frame ai: its timeline's, or (M6, sp) in the frame of space sp"""
        if sp is None:
            t = self.tl[k]
            return t.xy[ai], t.h_base[ai]
        al, b = self.ctx.aligns[sp], self.md.body[k]
        return al.xy(b.center[ai][None])[0], float(al.h(b.base[ai]))

    def _body_box(self, k, ai, sp):
        """5.2 (M4): body k's support footprint at frame ai, as _boxes maps it"""
        b, t, ctx = self.md.body[k], self.tl.get(k), self.ctx
        al = ctx.aligns[sp or (t.space[ai] if t is not None else None) or ctx.gt.spaces[0].id]
        lo, hi = b.lo[ai], b.hi[ai]
        return geom.gt_rect(al.xy, 0.5 * (lo[:2] + hi[:2]), 0.0, hi[0] - lo[0], hi[1] - lo[1])

    def co_moving(self, a, ai):
        """5.2 (M4): the non-agent bodies the agent a rides or carries at frame ai: footprint (model frame: the
        agent's oriented box, the body's box as _boxes) overlapping or touching the agent's, and moving with it over
        +-4 frames by carried()'s co-motion test (displacement > 0.2 m, differing by <= 0.35 x the agent's + 0.15 m;
        model displacements in the agent's space scale)."""
        md, ctx = self.md, self.ctx
        b, tl = md.body[a], self.tl_of(a)
        A = len(self.t)
        i, j = max(0, ai - 4), min(A - 1, ai + 4)
        if not (b.present[i] and b.present[j]):
            return []
        s = ctx.aligns[tl.space[ai] or ctx.gt.spaces[0].id].sim.s
        ident = lambda q: q
        ra = geom.gt_rect(ident, b.center[ai], b.long_yaw[ai], b.long[ai], b.short[ai])
        dp = (b.center[j][:2] - b.center[i][:2]) * s
        out = []
        for k, o in md.body.items():
            if k == a or o.group in BODY_GROUPS or not (o.present[ai] and o.present[i] and o.present[j]):
                continue
            lo, hi = o.lo[ai], o.hi[ai]
            ro = geom.gt_rect(ident, 0.5 * (lo[:2] + hi[:2]), 0.0, hi[0] - lo[0], hi[1] - lo[1])
            if geom.rect_gap(ra, ro) > 0:
                continue
            do = (o.center[j][:2] - o.center[i][:2]) * s
            if np.linalg.norm(do) > 0.2 and np.linalg.norm(do - dp) <= 0.35 * np.linalg.norm(dp) + 0.15:
                out.append(k)
        return out

    def jump_runs(self, a):
        """(take-off, landing) of airborne runs of jump_min_s..jump_max_s; up to jump_long_max_s when the base height
        changes by jump_long_arc_m along the run (a long jump has an arc, a hovering proxy does not).  A run airborne
        at the first sample gives only its landing, one airborne at the last sample only its take-off (None for the
        side outside the clip); a run airborne at both ends gives nothing."""
        m = self.airborne(a)
        lo, hi = self.P.get("jump_min_s", 0.1), self.P.get("jump_max_s", 1.5)
        hi_long, arc = self.P.get("jump_long_max_s", 4.0), self.P.get("jump_long_arc_m", 0.2)
        h = self.tl_of(a).h_base if m.any() else None
        out = []
        for t0, t1 in runs(m, self.t, fill=0.15):
            at0, at1 = t0 <= self.t[0], t1 >= self.t[-1]
            if at0 and at1:
                continue
            i0, i1 = int(np.searchsorted(self.t, t0)), int(np.searchsorted(self.t, t1))
            d = t1 - t0
            if d > hi:
                hb = h[i0:i1 + 1]
                hb = hb[np.isfinite(hb)]
                # 5.0.12 (D18): a run longer than jump_long_max_s is a long drop / fall, kept when it has the arc
                if (d > hi_long and not fixes.on("D18")) or not len(hb) or hb.max() - hb.min() < arc:
                    continue
            elif d < lo and not (at0 or at1):
                continue
            out.append((None if at0 else self.t[max(0, i0 - 1)], None if at1 else self.t[min(len(self.t) - 1, i1 + 1)]))
        return out

    def turns(self, a, deg=None, direction=None):
        """Start times of facing changes of at least turn_deg (or 0.6 x a stated angle) within turn_win_s;
        left = counter-clockwise."""
        tl = self.tl_of(a)
        if tl is None:
            return []
        thr = self.P.get("turn_deg", 60.0)
        if deg:
            thr = min(thr, 0.6 * abs(float(deg)))
        win = self.P.get("turn_win_s", 1.0)
        f = tl.facing
        idx = [i for i in range(len(self.t)) if tl.present[i] and not np.isnan(f[i])]
        un, prev = {}, None  # unwrapped facing over the defined samples
        for i in idx:
            un[i] = f[i] if prev is None else un[prev] + ((f[i] - f[prev] + 180.0) % 360.0 - 180.0)
            prev = i
        m = np.zeros(len(self.t), bool)
        for k, i in enumerate(idx):
            for j in idx[k + 1:]:
                if self.t[j] - self.t[i] > win:
                    break
                d = un[j] - un[i]
                if abs(d) >= thr and (direction is None or (direction == "left") == (d > 0)):
                    m[i] = True
                    break
        out = []
        for r0, r1 in runs(m, self.t, fill=0.3):
            # report the frame of the fastest change inside the run's window, not the window start
            ks = [i for i in idx if r0 <= self.t[i] <= r1 + win]
            rate = [(abs(un[b] - un[a]) / max(1e-6, self.t[b] - self.t[a]), self.t[a]) for a, b in zip(ks, ks[1:])]
            mx = max((x[0] for x in rate), default=0.0)
            out.append(next(t for v, t in rate if v >= 0.9 * mx) if rate else r0)
        return out

    def edge_times(self, mask, rising=True):
        out = []
        for i in range(1, len(mask)):
            if rising and mask[i] and not mask[i - 1]:
                out.append(self.t[i])
            if not rising and mask[i - 1] and not mask[i]:
                out.append(self.t[i])
        return out

    def find(self, e):
        """Candidate model times (list) for one GT event, or None for unsupported types."""
        typ = e["type"]
        ag = list(e.get("agents", []))
        ob = list(e.get("objects", []))
        par = e.get("params", {})
        a = ag[0] if ag else None
        if typ == "sit":
            out = self.transitions_into(a, "sit")
            if fixes.on("D5") and not self.seat_named(ob):
                # 5.0.12: a sit on no seat object (floor, mat, boat, steps) reads as crouch in the posture classes
                out = sorted(set(out) | set(self.transitions_into(a, "crouch")))
            if ob:
                seat = self._seat_box(ob[0])
                if seat is not None:
                    tl = self.tl_of(a)
                    out = [t for t in out if self._on(tl, t, seat)]
            return out
        if typ in ("stand_up", "stand"):
            return self.transitions_into(a, "stand", prev={"sit", "crouch", "lie"})
        if typ == "crouch":
            return self.transitions_into(a, "crouch")
        if typ == "lie":
            return self.transitions_into(a, "lie")
        if typ == "fall":
            tl = self.tl_of(a)
            if tl is None:
                return []
            out = []
            # 5.2 (M5): the posture classes' ratio (support-relative, own standing height)
            r = tl.posture_ratio() if fixes.on("M5") else tl.h_post / max(1e-6, getattr(tl, "ref_h", 1.7))
            for i in range(len(self.t)):
                if not tl.present[i] or r[i] > 0.45:
                    continue
                j = [k for k in range(i) if self.t[i] - self.t[k] <= 1.5 and tl.present[k] and r[k] >= 0.8]
                if j:
                    out.append(self.t[j[-1]])
            if fixes.on("M19"):
                # 5.4 (M19): every fall, one onset per drop; only the first was kept, so a later fall of the same
                # body could never match its GT event and an early dip could take the place of the real fall
                merged = []
                for t_ in sorted(set(out)):
                    if not merged or t_ - merged[-1] > 1.5:
                        merged.append(t_)
                return merged
            return sorted(set(out))[:1] if out else []
        if typ in ("pick_up", "put_down", "carry"):
            if not ob or a is None:
                return []
            rs = self.carry_runs(ob[0], a)
            if typ == "pick_up":
                return [r[0] for r in rs]
            if typ == "put_down":
                return [r[1] for r in rs if r[1] < self.t[-1] - 0.2]
            out = [r[0] for r in rs if r[1] - r[0] >= 0.5]
            if fixes.on("D19"):
                m = np.zeros(len(self.t), bool)
                for r0, r1 in rs:
                    if r1 - r0 >= 0.5:
                        m |= (self.t >= r0 - 1e-9) & (self.t <= r1 + 1e-9)
                out = sorted(set(out) | set(self.holding_at(m, e)))
            return out
        if typ == "hand_over":
            if len(ag) < 2 or not ob:
                return []
            ra, rb = self.carry_runs(ob[0], ag[0]), self.carry_runs(ob[0], ag[1])
            return [y[0] for x in ra for y in rb if 0 <= y[0] - x[1] <= 1.5 or (x[0] < y[0] <= x[1] + 0.5)]
        if typ in ("enter_vehicle", "exit_vehicle"):
            if not ob:
                veh = [k for k in self.tl if self.md.body[k].group == "vehicle"]
                ob = veh[:1]
            if not ob or a is None:
                return []
            m = self.inside_vehicle(a, ob[0])
            return self.edge_times(m, rising=typ == "enter_vehicle")
        if typ in ("meet", "approach"):
            if len(ag) < 2:
                return []
            ta, tb = self.tl_of(ag[0]), self.tl_of(ag[1])
            if ta is None or tb is None:
                return []
            dist = np.array([np.linalg.norm(ta.xy[i] - tb.xy[i]) if ta.present[i] and tb.present[i] else np.inf
                             for i in range(len(self.t))])
            close = dist <= par.get("d", self.P["meet_dist_m"])
            return self.edge_times(close, True) or ([self.t[0]] if close[0] else [])
        if typ == "talk":
            if len(ag) < 2:
                return []
            ta, tb = self.tl_of(ag[0]), self.tl_of(ag[1])
            if ta is None or tb is None:
                return []
            cone = self.P["face_cone_deg"]
            m = np.zeros(len(self.t), bool)
            for i in range(len(self.t)):
                if not (ta.present[i] and tb.present[i]):
                    continue
                v = tb.xy[i] - ta.xy[i]
                if np.linalg.norm(v) > self.P["talk_dist_m"] or np.isnan(ta.facing[i]) or np.isnan(tb.facing[i]):
                    continue
                if (geom.angdiff(ta.facing[i], geom.yaw_of(v), 360) <= cone and
                        geom.angdiff(tb.facing[i], geom.yaw_of(-v), 360) <= cone):
                    m[i] = True
            return [r[0] for r in runs(m, self.t, min_len=1.0, fill=0.4)]
        if typ == "open_door":
            door = ob[0] if ob else None
            td = self.tl_of(door) if door else None
            if td is None and fixes.on("S11") and door is not None and door not in self.ctx.gt.entities:
                # 5.0.12: a static door: the agent within reach of its matched footprint inside the event window
                if door not in self.ctx.gt.obj_by:
                    return None
                rect, tl = self.static_rect(door), self.tl_of(a)
                if rect is None or tl is None:
                    return []
                t = e.get("t", [0, 0])
                t0, t1 = (float(t[0]), float(t[1])) if isinstance(t, (list, tuple)) else (float(t), float(t))
                tol = float(e.get("tol_s", self.ctx.TH["event_tol_s"]))
                reach = self.P["door_reach_m"]
                hit = [i for i in range(len(self.t)) if tl.present[i] and t0 - tol <= self.t[i] <= t1 + tol and
                       geom.rect_dist(tl.xy[i], rect) <= reach]
                return [self.t[hit[0]]] if hit else []
            if td is None:
                return []
            idx = np.where(td.present)[0]
            if not len(idx):
                return []
            y0, p0 = td.long_yaw[idx[0]], td.xy[idx[0]]
            out = []
            for i in idx:
                if (geom.angdiff(td.long_yaw[i], y0, 180) >= self.P["door_open_deg"] or
                        np.linalg.norm(td.xy[i] - p0) >= self.P["door_slide_m"]):
                    k = i
                    while k > idx[0] and geom.angdiff(td.long_yaw[k], y0, 180) > 8 and np.linalg.norm(td.xy[k] - p0) > 0.05:
                        k -= 1
                    out.append(self.t[k])
                    break
            return out
        if typ in ("enter_space", "exit_space"):
            tl = self.tl_of(a)
            if fixes.on("S6"):
                sp = par.get("space") or par.get("to" if typ == "enter_space" else "from")
                if sp is None and ob and ob[0] in self.ctx.gt.space_by:
                    sp = ob[0]
            else:
                sp = par.get("space") or (ob[0] if ob else None)
            if tl is None or sp is None:
                return []
            m = np.array([s == sp for s in tl.space])
            return self.edge_times(m, rising=typ == "enter_space")
        if typ == "arrive":
            tl = self.tl_of(a)
            tgt = par.get("target") or (ob[0] if ob else None)
            if tl is None or tgt is None:
                return []
            if isinstance(tgt, str) and tgt in self.ctx.gt.space_by:
                m = np.array([s == tgt for s in tl.space])
                return self.edge_times(m, True)
            if isinstance(tgt, str) and tgt in self.tl:
                to = self.tl[tgt]
                m = np.array([tl.present[i] and to.present[i] and np.linalg.norm(tl.xy[i] - to.xy[i]) <= 1.0
                              for i in range(len(self.t))])
            else:
                if fixes.on("S15") and not self.resolvable(tgt):
                    return None
                p = self.static_pos(tgt)
                if p is None:
                    return []
                rect = self.static_rect(tgt) if fixes.on("S17") else None
                if rect is not None:
                    # 5.0.12 (S17): distance to the target's footprint, not to its centre
                    m = np.array([tl.present[i] and geom.rect_dist(tl.xy[i], rect) <= par.get("d", 1.0)
                                  for i in range(len(self.t))])
                else:
                    m = np.array([tl.present[i] and np.linalg.norm(tl.xy[i] - p) <= par.get("d", 1.0)
                                  for i in range(len(self.t))])
            return self.edge_times(m, True) or ([self.t[0]] if m.any() and m[0] else [])
        if typ in ("appear", "disappear"):
            tl = self.tl_of(a or (ob[0] if ob else None))
            if tl is None:
                return []
            return self.edge_times(tl.present, rising=typ == "appear")
        if typ == "chase":
            if len(ag) < 2:
                return []
            ta, tb = self.tl_of(ag[0]), self.tl_of(ag[1])
            if ta is None or tb is None:
                return []
            m = np.zeros(len(self.t), bool)
            for i in range(1, len(self.t)):
                if not (ta.present[i] and tb.present[i] and ta.present[i - 1]):
                    continue
                va = ta.xy[i] - ta.xy[i - 1]
                if ta.speed[i] < 0.5 or tb.speed[i] < 0.5:
                    continue
                to_b = tb.xy[i] - ta.xy[i]
                if np.dot(va, to_b) / max(1e-9, np.linalg.norm(va) * np.linalg.norm(to_b)) > 0.7:
                    m[i] = True
            return [r[0] for r in runs(m, self.t, min_len=1.0, fill=0.4)]
        if typ == "contact":
            s18 = fixes.on("S18")
            if len(ag) + len(ob) < 2:
                if not (s18 and len(ag) + len(ob) == 1 and (par.get("surface") or par.get("surface_space"))):
                    return []
                # 5.0.12 (S18): contact with the support surface (floor / a static top under the body)
                m = self.support_contact((ag + ob)[0])
            else:
                x, y = (ag + ob)[:2]
                bx, by = self.md.body.get(x), self.md.body.get(y)
                if s18 and (bx is None) != (by is None):
                    # 5.0.12 (S18): one participant is a static GT object: its matched instance box
                    bb, key = (bx, y) if bx is not None else (by, x)
                    if key in self.ctx.gt.entities or key in self.tl:
                        return []
                    if not self.resolvable(key):
                        return None
                    r = static_target(self.ctx, key)
                    if r is None or r[0] is None:
                        return []
                    s_lo, s_hi = r[0].lo, r[0].hi
                    m = np.array([bb.present[i] and geom.aabb_gap(bb.lo[i], bb.hi[i], s_lo, s_hi) <= 0.05
                                  for i in range(len(self.t))])
                elif bx is None or by is None:
                    return []
                else:
                    m = np.array([bx.present[i] and by.present[i] and
                                  geom.aabb_gap(bx.lo[i], bx.hi[i], by.lo[i], by.hi[i]) <= 0.05
                                  for i in range(len(self.t))])
            out = self.edge_times(m, True) or ([self.t[0]] if m.any() and m[0] else [])
            if fixes.on("D19"):
                out = sorted(set(out) | set(self.holding_at(m, e)))
            return out
        if typ in ("stop", "start"):
            tl = self.tl_of(a)
            if tl is None:
                return []
            still = _sustain(tl.speed < 0.15, 4)
            out = self.edge_times(still, rising=typ == "stop")
            if typ == "start" and fixes.on("S8"):
                # 5.0.12: moving at the first present sample is a start there
                idx = np.where(tl.present)[0]
                if len(idx) and not still[idx[0]] and self.t[idx[0]] not in out:
                    out = [self.t[idx[0]]] + out
            return out
        if typ in ("jump", "land"):
            if a is None:
                return []
            return [x for x in (r[0] if typ == "jump" else r[1] for r in self.jump_runs(a)) if x is not None]
        if typ == "turn":
            if a is None:
                return []
            return self.turns(a, par.get("deg"), par.get("dir"))
        return None


def match_event_times(gt_events, cands, gate=None, dist=None):
    """GT event -> nearest unused model time (Hungarian on |dt|). 5.0.11: gate(e) is the largest |dt| a match may
    have (tol_s + event_extra_s, where timing already scores 0); pairs beyond it are infeasible, so a far candidate
    neither counts as the event nor takes a near candidate away from another event. 5.0.12 (S16): dist(e, times) is
    the distance of each time to the event's window; ties break on |dt| to its reference time."""
    out = {}
    for key, evs in gt_events.items():
        times = cands.get(key) or []
        if not times:
            continue
        D = np.abs(np.array([e["_t"] for e in evs])[:, None] - np.array(times)[None])
        if dist is not None:
            W = np.array([dist(e, np.asarray(times, float)) for e in evs])
            C = W + 1e-3 * D
            D = W
        else:
            C = D
        if gate is not None:
            G = np.array([gate(e) for e in evs])[:, None]
            D = np.where(D <= G, C, np.inf)         # geom.hungarian: inf entries are forbidden
        else:
            D = C
        for i, j in geom.hungarian(D):
            out[evs[i]["id"]] = times[j]
    return out


def score_B2(ctx, items):
    gt, md, TH = ctx.gt, ctx.model, ctx.TH
    ev = Events(ctx)
    s16 = fixes.on("S16")
    evs = []
    for k, e in enumerate(gt.events):
        e = dict(e)
        e.setdefault("id", "E%02d" % (k + 1))
        t = e.get("t", [0, 0])
        e["_t"] = float(t[0] if isinstance(t, (list, tuple)) else t)
        e["_t0"] = e["_t"]
        e["_t1"] = float(t[1]) if isinstance(t, (list, tuple)) and len(t) > 1 else e["_t"]
        if s16 and e["_t1"] > e["_t0"] and e["type"] in COMPLETION_TYPES:
            e["_t"] = e["_t1"]          # 5.0.12 (S16): completion types are ordered at the window end
        evs.append(e)
    for k, st in enumerate(gt.space_transitions):
        evs.append({"id": "ST%02d" % (k + 1), "type": "enter_space", "agents": [st["agent"]],
                    "params": {"space": st["to"]}, "t": [st["t"], st["t"]], "_t": float(st["t"]),
                    "_t0": float(st["t"]), "_t1": float(st["t"]), "tol_s": st.get("tol_s", 1.0)})
    if fixes.on("M14"):     # 5.2 (M14): a turn / talk read on a facing the model leaves unknown is not scored
        evs = [e for e in evs if not _facing_unknown_event(ctx, e)]
    groups, cands, unsupported = {}, {}, []
    m10 = fixes.on("M10")
    for e in evs:
        if m10:
            # 5.2 (M10): list / dict params stay in the group key (as tuples), so events that differ only there no
            # longer share one group
            key = (e["type"], tuple(_hashable(x) for x in e.get("agents", [])),
                   tuple(_hashable(x) for x in e.get("objects", [])), _hashable(e.get("params") or {}))
        else:
            key = (e["type"], tuple(e.get("agents", [])), tuple(e.get("objects", [])),
                   tuple(sorted((e.get("params") or {}).items())) if all(isinstance(v, (str, int, float))
                                                                         for v in (e.get("params") or {}).values()) else ())
        r = ev.find(e)
        if r is None:
            unsupported.append(e)
            continue
        groups.setdefault(key, []).append(e)
        if m10:
            # 5.2 (M10): the candidates of every event of a group are kept (union), never the last event's only
            cands[key] = sorted(set(cands.get(key, [])) | set(r))
        else:
            cands[key] = r
    tol_def = TH["event_tol_s"]
    gate = None
    if ctx.P.get("event_gate") == "tol_plus_extra":      # 5.0.11
        gate = lambda e: float(e.get("tol_s", tol_def)) + TH["event_extra_s"]  # noqa: E731
    dist = _window_dist if s16 else None
    matched = match_event_times(groups, cands, gate, dist)
    for e in evs:
        if e in unsupported:
            # unknown types: co-presence of all participants in the event window, capped at 0.5
            t0 = e["_t0"]
            t1 = float(e["t"][1]) if isinstance(e.get("t"), (list, tuple)) else t0
            ok = True
            for x in list(e.get("agents", [])) + list(e.get("objects", [])):
                tl = ctx.timelines.get(x)
                if tl is None:
                    ok = ok and (x in ctx.matches and ctx.matches[x][0] is not None)
                    continue
                win = (tl.t >= t0 - 1) & (tl.t <= t1 + 1)
                ok = ok and bool(tl.present[win].any())
            items.add("B2", "occurrence", "event:%s:%s" % (e["id"], e["type"]), 0.5 if ok else 0.0, 1.0,
                      value="co-presence only")
            continue
        tm = matched.get(e["id"])
        tol = float(e.get("tol_s", tol_def))
        items.add("B2", "occurrence", "event:%s:%s" % (e["id"], e["type"]), 1.0 if tm is not None else 0.0, 1.0,
                  value=None if tm is None else round(tm, 3))
        dt = float(_window_dist(e, np.array([tm]))[0]) if (s16 and tm is not None) else \
            (abs(tm - e["_t"]) if tm is not None else None)
        sc = geom.lin(dt, tol, tol + TH["event_extra_s"]) if tm is not None else 0.0
        items.add("B2", "timing", "time:%s:%s" % (e["id"], e["type"]), sc, 0.5,
                  value=None if tm is None else round(tm - e["_t"], 3))
    # pairwise order of distinct-time events
    timed = [e for e in evs if e not in unsupported]
    pairs = [(a, b) for a in timed for b in timed if a["_t"] + float(a.get("tol_s", tol_def)) < b["_t"]]
    if pairs:
        w = 1.0 / max(1, len(timed) - 1)
        for a, b in pairs:
            ma, mb = matched.get(a["id"]), matched.get(b["id"])
            ok = ma is not None and mb is not None and ma < mb
            items.add("B2", "order", "order:%s<%s" % (a["id"], b["id"]), 1.0 if ok else 0.0, w)
    # posture agreement at GT samples
    d5 = fixes.on("D5")
    seat_rects = [geom.gt_rect(lambda q: q, o.center[:2], o.long_yaw, o.long, o.short)
                  for o in gt.objects if o.cls in SEAT_CLASSES] if d5 else []
    for tid, tr in gt.tracks.items():
        tl = ctx.timelines.get(tid)
        vals = []
        for i, t in enumerate(tr.t):
            g = norm_posture(tr.posture[i])
            if g is None:
                continue
            if tl is None:
                vals.append(0.0)
                continue
            m = tl.posture[tl.at(t)]
            if m == g:
                vals.append(1.0)
            elif d5 and g == "sit" and m == "crouch" and \
                    not any(geom.in_rect(tr.at(t), r, 0.2) for r in seat_rects):
                vals.append(1.0)        # 5.0.12 (D5): a sit on no GT seat object reads as crouch
            elif {m, g} <= {"sit", "crouch"}:
                vals.append(0.5)
            elif m == "hidden" and not tr.visible[i]:
                vals.append(1.0)
            else:
                vals.append(0.0)
        if vals and any(norm_posture(p) not in (None, "stand") for p in tr.posture):
            items.add("B2", "posture", "posture:" + tid, float(np.mean(vals)), 1.0, n=len(vals))
    # timed relations
    for r in gt.relations:
        if r.get("t") is None:
            continue
        v = _timed_relation(ctx, r)
        if v is not None:
            items.add("B2", "relations_t", "rel_t:%s:%s:%s@%.1f" % (r["type"], r["a"], r.get("b"), r["t"]), v, 1.0)


COMPLETION_TYPES = ("arrive", "stop", "put_down")
FACING_EVENTS = ("turn", "talk")        # 5.2 (M14): the event types Events.find reads from the facing


def _facing_unknown_event(ctx, e):
    """5.2 (M14): a turn / talk event is not scored when the facing of one of its agents is unknown at most (> 1/2)
    of the agent's present frames inside the GT window widened by the event's tolerance"""
    if e["type"] not in FACING_EVENTS:
        return False
    tol = float(e.get("tol_s", ctx.TH["event_tol_s"]))
    for x in e.get("agents", []):
        tl = ctx.timelines.get(x)
        if tl is None:
            continue
        w = tl.present & (tl.t >= e["_t0"] - tol - 1e-6) & (tl.t <= e["_t1"] + tol + 1e-6)
        if w.any() and 2 * int((w & tl.b.facing_unk).sum()) > int(w.sum()):
            return True
    return False


def _hashable(v):
    """5.2 (M10): a GT event field as a hashable group-key part (lists -> tuples, dicts -> sorted item tuples)."""
    if isinstance(v, dict):
        return tuple(sorted((str(k), _hashable(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    return v


def _window_dist(e, times):
    """5.0.12 (S16): distance of model times to the GT event window [t0, t1] (|t - t0| for a point event)."""
    return np.maximum(0.0, np.maximum(e["_t0"] - times, times - e["_t1"]))


def _gap_measure(par, P):
    """5.0.12 (S9): params.measurement / params.measure names a footprint gap (gap_measurements words)."""
    txt = " ".join(str(par.get(k, "")) for k in ("measurement", "measure")).lower()
    return any(w in txt for w in P.get("gap_measurements", []))


def _timed_relation(ctx, r):
    t = float(r["t"])
    ta, tb = ctx.timelines.get(r["a"]), ctx.timelines.get(r.get("b"))
    if ta is None:
        return 0.0
    ev = Events(ctx)
    if fixes.on("S15") and tb is None and r.get("b") is not None and not ev.resolvable(r.get("b")):
        return None             # 5.0.12: a key the scorer cannot look up is unsupported, not 0
    pa = ta.pos(t)
    pb = tb.pos(t) if tb is not None else ev.static_pos(r.get("b"))
    if pa is None or pb is None:
        return 0.0
    typ = r["type"]
    par = r.get("params", {})
    d = float(np.linalg.norm(pa - pb))
    if typ == "distance" and fixes.on("S9") and _gap_measure(par, ctx.P):
        # 5.0.12: a clearance / gap is measured between the footprints, not the centres
        v = pb - pa
        ra = ev.body_rect(ta, ta.frame(t))
        if tb is not None:
            rb = ev.body_rect(tb, tb.frame(t))
        else:
            rb = ev.static_rect(r.get("b"))
        d = max(0.0, d - geom.rect_half_extent(ra, v) - (geom.rect_half_extent(rb, v) if rb is not None else 0.0))
    if typ == "near" and fixes.on("M13"):
        # 5.2 (M13): the gap between the footprints (body rects / the target's oriented footprint, the S9 rects), not
        # the centre distance; a GT point target: the gap from the body rect to the point
        ra = ev.body_rect(ta, ta.frame(t))
        rb = ev.body_rect(tb, tb.frame(t)) if tb is not None else ev.static_rect(r.get("b"))
        d = geom.rect_gap(ra, rb) if rb is not None else geom.rect_dist(pb, ra)
    if typ == "near":
        return 1.0 if d <= par.get("d", 1.5) else 0.0
    if typ == "distance":
        dg = par["d"]
        return geom.lin(abs(d - dg), 0.15 * max(dg, 1.0), 0.6 * max(dg, 1.0))
    if typ == "facing":
        f = ta.facing[ta.at(t)]
        if np.isnan(f) and fixes.on("M14") and ta.b.facing_unk[ta.at(t)]:
            return None             # 5.2 (M14): facing unknown at that time, not scored
        return 0.0 if np.isnan(f) else geom.lin(geom.angdiff(f, geom.yaw_of(pb - pa), 360), 30, 75)
    if typ in ("left_of", "right_of", "in_front_of", "behind"):
        ref_yaw = par.get("ref_yaw")
        if ref_yaw is None and tb is not None:
            f = tb.facing[tb.at(t)]
            ref_yaw = None if np.isnan(f) else f
        if ref_yaw is None:
            return None
        v = pa - pb
        side = geom.wrap180(geom.yaw_of(v) - ref_yaw)
        if typ == "in_front_of":
            return 1.0 if abs(side) < 60 else 0.0
        if typ == "behind":
            return 1.0 if abs(side) > 120 else 0.0
        return 1.0 if (side > 20 if typ == "left_of" else side < -20) else 0.0
    return None
