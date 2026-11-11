"""pvscore 5.1 r2: score the view and body entries of the key-content checklist on model frames (no API, JSON only).

    python3 -I -B tools/v51_r2/view_score.py [ITEM ...] [--controls] [--models=a,b] [--root=DIR --out=FILE]

Inputs: data/v51_r2/<item>/checklist.json (the fixed fact list), <item>/model_view/<model>.json (model frames, time
map, bodies), <item>/scene_frames.json (V2 row times; also scored as the pseudo-model "video_geom": the video's own
SAM3 + depth frames, an upper-bound / GT-noise check for the frame-level blocks only), the GT gt.json (gt_root.py; the
shots' camera move) and annot/video_freeze_20261017.json (freeze frames). Margins, curves and rules:
tools/v51_r2/view_thresholds_v51r2.json (r2, pre-registered; r2_changes lists what differs from r1). Each entry is scored
at its GT times on the model frame the 5.0.9 time warp maps them to; a missing model element scores 0 there. Blocks:
  screen_order  frame_relation left_of / above (geometry), screen_side (V2)        R by permutation
  depth_order   frame_relation in_front_of (geometry), depth_order (V2)             R by permutation
  position      element_in_view, object_in_view, screen_track cells (V1)            R best constant cell (mirror-pooled)
  size          screen_track height_frac (V4)                                        R best constant height (r2)
  facing        facing_run, mutual_facing (V5)                                       R best constant angle (r2)
  presence      + screen_track rows, in_frame, visible half of enter / exit / occluded;
                - cast subjects absent at V2 row times (continuation: frame 1 only), absent half of
                  enter / exit / occluded
                                                                                     balanced accuracy, R = best
                                                                                     per-subject constant (r2)
  landmarks     landmark_present (class seen in some model frame; r2: its own block) R = 0
  posture       posture_run                                                          R = best constant answer
  locomotion    locomotion_segment still / moving, jump, land (r2: onset), climb, fall
                                                                                     R = best constant {still, moving}
  direction     lateral heading (screen_dir, screen_dir segments, locomotion_segment directions: definition A;
                r2: duplicates and windows with < 2 grid times parked),
                size trend (band), toward / away camera, up / down                   R = per sub-kind best constant
Block = clip((P - R) / (1 - R), 0, 1) x coverage. Every block also carries R_gt (its R from the GT facts alone) and
informative (false when R_gt >= low_info.r_max, when n < low_info.min_n (r2), or for presence when a class has
< presence.min_class_n facts; the score is kept, a consumer should leave such a block out) with info_reason. Facts the layer cannot hold a model to are
parked (counted per item in "parked"). Not read here: on / on_ground (flag only / trivially true), events, camera,
timing, 3D entries (5.0.9 keeps them), object_relation (the geometry ones repeat a frame_relation entry), posture_run
climb / fall (temporal layer). --controls adds two negative controls per real model (left-right mirror, half-shot time
shift) and writes view_scores_controls.json instead. --models scores only the named views (video_geom included) and
writes view_scores_subset.json; --root reads another data dir (needs --out). Writes data/v51_r2/view_scores.json.
"""
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from gt_root import gt_json  # noqa: E402
import sframes as SF  # noqa: E402
import paths as PT  # noqa: E402
import items as IT  # noqa: E402
if str(PT.PVSCORE_DIR) not in sys.path:
    sys.path.insert(0, str(PT.PVSCORE_DIR))
from pvscore5 import fixes  # noqa: E402  (5.2)

R = PT.ROOT
P = PT.VIEW_GT                                   # GT side: <item>/checklist.json, scene_frames.json
TH_FILE = PT.rel(PT.VIEW_CONF / "view_thresholds_v51r2.json")
TH = json.loads((PT.VIEW_CONF / "view_thresholds_v51r2.json").read_text())
FREEZE = json.loads(PT.FREEZE.read_text())
SYN = {w: i for i, g in enumerate(TH["class_synonyms"]) for w in g}
ITEMS = IT.of_kinds(IT.VIEW_KINDS)               # r2: every video / continuation item of the item table
SKIP = {"excluded_effect", "excluded_background", "needs_limbs", "needs_scene_field"}
BLOCKS = ["screen_order", "depth_order", "position", "size", "facing", "presence", "landmarks", "posture", "locomotion",
          "direction"]
PAIRWISE = ("screen_order", "depth_order")
ID = re.compile(r"^[A-Z]\d\d$")
FACING_UNKNOWN = "unknown"      # 5.2 (M14): model_view.FACING_UNKNOWN, a body row whose facing is unknown
VIDEO_MATCH_S = 0.5     # video_geom only (a reference, not a model score): V2 row times vs SAM3 sample times, as v2_merge_s
POS, NEG = ("pres", 1), ("pres", -1)      # r2: a presence row key is POS / NEG + (subject label,)
R2 = TH["r2_changes"]
GRID_STEP = 0.25                            # model_view.py STEP: every model has a row at these GT times


def pk(sign, label):
    """r2 presence row key: ("pres", +-1, subject label)"""
    return ("pres", sign, label)


def is_pos(k):
    return isinstance(k, tuple) and k[:2] == POS


def is_neg(k):
    return isinstance(k, tuple) and k[:2] == NEG
LAT_PAIR = ("left_to_right", "right_to_left")
FIELD_DOC = ("blocks[<block>] = {n, present, coverage, P, R, score, raw, R_gt, informative, info_reason, ...}: score = "
             "clip((P - R) / (1 - R), 0, 1) x coverage; R_gt = the block's chance from the GT facts alone (all facts); "
             "informative = false when R_gt >= low_info.r_max (a constant answer already gets that share), when "
             "n < low_info.min_n (r2, every block) or, for "
             "presence, when positives or negatives number < presence.min_class_n; the score is still computed and a "
             "consumer (combine.py) should leave an uninformative block out of the total instead of reading its score. "
             "low_info = not informative (the name combine.py reads). R_gt / informative are model-independent except for the pairwise blocks (R_gt null, never flagged). "
             "presence adds n_pos, n_neg, TPR, TNR (P = (TPR + TNR) / 2); direction adds sub {lateral, size, cam_path, "
             "updown: n, R}; locomotion adds classes; diag holds alternative chance values for the record")


def canon(c):
    c = (c or "").lower()
    c = c[8:] if c.startswith("surface:") else c
    c = c.split(":")[-1].replace(" ", "_")
    return SYN.get(c, c)


def ramp_up(x, zero, full):
    return max(0.0, min(1.0, (x - zero) / (full - zero)))


def ramp_down(x, full, zero):
    return max(0.0, min(1.0, (zero - x) / (zero - full)))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


# ------------------------------------------------------------------------------------------------------- views
class View:
    def __init__(self, d, video=False, item=None):
        self.video = video
        self.item = item or d.get("item")
        self.match_s = TH["time_match_s"] if not video else VIDEO_MATCH_S
        self.tm = defaultdict(list)
        for t, shot, _, _, gi, dt in d["time_map"]:
            self.tm[shot].append((t, gi, dt))
        self.frames = {int(k): v for k, v in d["frames"].items()}
        cols = d.get("body_cols", [])
        self.bodies = {eid: [dict(zip(cols, r)) for r in rows] for eid, rows in (d.get("body_series") or {}).items()}

    def frame(self, shot, t):
        best = min(self.tm.get(shot, []), key=lambda r: abs(r[0] - t), default=None)
        if best is None or abs(best[0] - t) > self.match_s or best[1] is None or best[2] > TH["frame_max_dt_s"]:
            return None, None
        return best[1], self.frames[best[1]]

    def times(self, shot, t0, t1):
        return sorted(t for t, _, _ in self.tm.get(shot, []) if t0 - 1e-6 <= t <= t1 + 1e-6)

    def body(self, eid, t):
        rows = self.bodies.get(eid)
        if not rows:
            return None
        b = min(rows, key=lambda r: abs(r["t"] - t))
        return b if abs(b["t"] - t) <= TH["time_match_s"] else None


def video_view(item):
    """the video's own SAM3 + depth frames as a pseudo-model (frame-level blocks only)"""
    g = SF.load(P / item / "scene_frames.json")
    tmap, frames, k = [], {}, 0
    for sh in g["shots"]:
        for sm in sh["samples"]:
            if sm.get("fi") is None:
                continue
            for e in sm["elements"]:
                e.setdefault("height_frac", round(e["bbox"][3] - e["bbox"][1], 3))
            frames[k] = {"elements": sm["elements"], "relations": sm["relations"]}
            tmap.append([sm["t"], sh["shot"], None, sm["t"], k, 0.0])
            k += 1
    return View({"time_map": tmap, "frames": frames}, video=True, item=item)


def find(frame, label):
    E = frame["elements"]
    if ID.match(label or ""):
        return [i for i, e in enumerate(E) if e.get("id") == label]
    cl = canon(label)
    return [i for i, e in enumerate(E) if not e.get("id")
            and (canon(e["class"]) == cl or any(canon(a) == cl for a in e.get("alt_class", [])))]


def of_class(e, cl):
    return canon(e["class"]) == cl or any(canon(a) == cl for a in e.get("alt_class", []))


def find_m7(frame, label, bind=None, other=None):
    """5.2 M7: a class label binds every element of its class, id-tagged or not. bind = (unbound, ids) from the GT
    side (gt_binding): a tagged element is a candidate when the GT bound one of the label's instances to its id, an
    untagged one when some instance is unbound; bind None (no GT instance found) keeps every element of the class.
    An element tagged with the other operand's id is never a candidate. An id label: find."""
    if ID.match(label or ""):
        return find(frame, label)
    cl = canon(label)
    out = []
    for i, e in enumerate(frame["elements"]):
        if not of_class(e, cl) or (e.get("id") and e["id"] == other):
            continue
        if bind is None or (bind[0] if not e.get("id") else e["id"] in bind[1]):
            out.append(i)
    return out


M7_CACHE = {}


def gt_binding(geo, e, label, side):
    """5.2 M7: the GT instances a geometry fact's class label refers to, as (unbound, ids): the label's elements in
    the fact's GT samples (frame_relation: the elements of the relation instances that hold, on that side, else, for
    a fact no relation instance gives (a V2 / scene-field fact), every element of the label, as for element_in_view).
    An instance is bound to a cast id when its SAM3 tracks (tids) carry that one id, with the same class, in a tagged
    sample of the shot (scene_frames tags a track only at V2 row times, so a grid sample lists the same subject by
    class); unbound otherwise. None when no instance is found."""
    k = (e["id"], label, side)
    c = M7_CACHE.get(id(geo))
    if c is None or c[0] is not geo:
        c = M7_CACHE[id(geo)] = (geo, {})
    if k in c[1]:
        return c[1][k]
    cl, shot, (t0, t1) = canon(label), e["shot"], e["t_range"]
    S = geo.get(shot, [])
    tid = defaultdict(set)
    for sm in S:
        for x in sm["elements"]:
            if x.get("id") and of_class(x, cl):
                for t in x.get("tids") or []:
                    tid[t].add(x["id"])
    inst, own = [], []
    for sm in S:
        if not t0 - 1e-6 <= sm["t"] <= t1 + 1e-6:
            continue
        E = sm["elements"]
        own += [x for x in E if x["label"] == label]
        if side is not None:
            p, rel = e["params"], e["check"].split(":")[1]
            inst += [E[r[side]] for r in sm["relations"] if r["rel"] == rel and r["b"] is not None
                     and E[r["a"]]["label"] == p["a"] and E[r["b"]]["label"] == p["b"]]
    inst = inst or own
    unbound, ids = False, set()
    for x in inst:
        b = set().union(*[tid.get(t, set()) for t in x.get("tids") or []])
        if len(b) == 1:
            ids |= b
        else:
            unbound = True
    out = (unbound, ids) if inst else None
    c[1][k] = out
    return out


# ---------------------------------------------------------------------------------------------- item context
class ItemCtx:
    """GT side of one item, model-independent: V2 row times per shot (scene_frames samples), the screen_track rows of
    every subject (skipped entries included, so a frozen or excluded row never turns into a negative), the cast, the
    freeze intervals, the GT shots' camera move; parked facts are recorded here"""

    def __init__(self, item, root):
        ck = json.loads((root / item / "checklist.json").read_text())
        sf = SF.load(root / item / "scene_frames.json")
        gt = json.loads(gt_json(item).read_text())
        self.item = item
        self.frozen = [tuple(x["t"]) for x in (FREEZE.get(ck.get("source_key")) or {}).get("intervals", [])]
        self.v2_times = {sh["shot"]: sorted({s["t"] for s in sh["samples"]}) for sh in sf["shots"]}
        self.rows, self.spans = defaultdict(list), defaultdict(list)
        for e in ck["entries"] + ck.get("not_applicable", []):     # not_applicable rows still show who is in view
            if e["check"] == "screen_track":
                key = (e["shot"], e["subjects"][0])
                self.rows[key] += e["params"]["rows"]
                self.spans[key] += [(r[0], r[0]) for r in e["params"]["rows"]]
            elif e["check"] == "in_frame" and e.get("subjects"):
                # in view at t (r1 continuation items: one entry per V2 row) or through t_range (GT keep_in_frame)
                self.spans[(e["shot"], e["subjects"][0])].append((e["t"], e["t"]) if "t" in e else tuple(e["t_range"]))
        self.cast = sorted({s for (_, s) in self.spans if s and s[0] in "PV"})
        self.move = {sh["id"]: (sh.get("move") or {}).get("type") for sh in gt["shots"]}
        # r1 scope (scope_r1.json continuation): a continuation item's model saw frame 1 and the prompt only, so a
        # derived absence is known at frame 1 only
        self.neg_t_max = TH["presence"]["continuation_neg_t_max_s"] if gt.get("image_kind") == "continuation" else None
        self.parked = {}
        self.notes = Counter()
        self.negatives = None
        # r2: the GT grid of every shot (model_view.py: t0 + i * STEP plus the sample times) - the times at which
        # every model has a row, model-independent
        self.grid = {}
        for sh in sf["shots"]:
            t0, t1 = sh["t"]
            ts = {round(t0 + i * GRID_STEP, 3) for i in range(int((t1 - t0) / GRID_STEP) + 1)}
            ts |= {round(sm["t"], 3) for sm in sh["samples"] if sm.get("fi") is not None}
            self.grid[sh["shot"]] = sorted(ts)
        self.lateral_park = self.lateral_prepass([e for e in ck["entries"] if e["scorable_new"] not in SKIP])

    def is_frozen(self, t):
        return any(a - 1e-6 <= t <= b + 1e-6 for a, b in self.frozen)

    def park(self, e, why):
        self.parked[(e["id"], why)] = why

    def parked_summary(self):
        out = dict(sorted(Counter(self.parked.values()).items()))
        out.update({"note: " + k: v for k, v in sorted(self.notes.items())})
        return out

    def lateral_prepass(self, entries):
        """r2 (GT only): {entry id: park reason} for lateral facts with < lateral_min_grid.min_grid_times GT grid
        times in the trimmed window, and for duplicates (same shot, subject, label; overlap >= overlap_min_frac of
        the shorter window: the longest is kept, tie screen_dir first, then id)"""
        L, G, D = TH["lateral"], R2["lateral_min_grid"], R2["lateral_dedup"]
        facts, park = [], {}
        for e in entries:
            if e["check"] == "screen_dir":
                t0, t1 = e["t_range"]
                if t1 - t0 < L["min_dur_s"] - 1e-6:
                    continue                      # parked by the r1 rule in score_entry
                lab, kind = e["params"]["dir"], 0
            elif e["check"] == "locomotion_segment":
                lab, why = lateral_label(e, self)
                if not lab:
                    continue
                kind = 1
            else:
                continue
            t0, t1 = e["t_range"]
            tr = L["trim"] * (t1 - t0)
            ng = sum(1 for t in self.grid.get(e["shot"], []) if t0 + tr - 1e-6 <= t <= t1 - tr + 1e-6)
            if ng < G["min_grid_times"]:
                park[e["id"]] = "lateral: < %d grid times in the trimmed window" % G["min_grid_times"]
                continue
            facts.append((-(t1 - t0), kind, e["id"], e["shot"], (e.get("subjects") or [None])[0], lab, t0, t1))
        kept = []
        for f in sorted(facts):
            _, _, eid, shot, sid, lab, t0, t1 = f
            dup = None
            for g in kept:
                if g[3] != shot or g[4] != sid:
                    continue
                ov = min(t1, g[7]) - max(t0, g[6])
                if ov >= D["overlap_min_frac"] * min(t1 - t0, g[7] - g[6]) - 1e-9 and ov > 0:
                    if g[5] == lab:
                        dup = g[2]
                        break
                    self.notes["lateral: conflicting label overlap (both scored)"] += 1
            if dup:
                park[eid] = "lateral: duplicate of %s" % dup
            else:
                kept.append(f)
        return park

    def gt_ratio(self, shot, sid, t0, t1):
        """GT height_frac ratio (last / first V2 row of the run), None when a row or its height is missing"""
        rows = self.rows.get((shot, sid), [])
        tol = TH["presence"]["row_match_s"]

        def h_at(t):
            r = min(rows, key=lambda r: abs(r[0] - t), default=None)
            return r[3] if r is not None and abs(r[0] - t) <= tol else None
        a, b = h_at(t0), h_at(t1)
        return b / a if a and b else None


_CTX = {}


def item_ctx(item):
    key = (str(P), item)
    if key not in _CTX:
        _CTX[key] = ItemCtx(item, P)
    return _CTX[key]


# ---------------------------------------------------------------------------------------------- per-time checks
def pair_ok(rel, ea, eb):
    if rel == "left_of":
        return eb["c"][0] - ea["c"][0] >= TH["screen_gap"]
    if rel == "above":
        return eb["c"][1] - ea["c"][1] >= TH["screen_gap"]
    if rel == "in_front_of":
        da, db = ea.get("depth_med"), eb.get("depth_med")
        return bool(da and db and db / da >= TH["depth_ratio"])
    raise ValueError(rel)


def cell_rc(cell):
    if cell == "center":
        return "mid", "center"
    r, c = cell.split("-")
    return r, c


def cell_score(gt_cell, m_cell):
    C = TH["cell"]
    (gr, gc), (mr, mc) = cell_rc(gt_cell), cell_rc(m_cell)
    hx = {"left": 0, "center": 1, "right": 2}
    vy = {"top": 0, "mid": 1, "bottom": 2}
    dh, dv = abs(hx[gc] - hx[mc]), abs(vy[gr] - vy[mr])
    sh = C["h_same"] if dh == 0 else C["h_adjacent"] if dh == 1 else C["h_opposite"]
    sv = C["v_same"] if dv == 0 else C["v_adjacent"] if dv == 1 else C["v_opposite"]
    return C["w_h"] * sh + C["w_v"] * sv


CELLS = [f"{r}-{c}" if (r, c) != ("mid", "center") else "center" for r in ("top", "mid", "bottom")
         for c in ("left", "center", "right")]
MIRROR_CELL = {"left": "right", "right": "left", "center": "center"}


def mirror_cell(cell):
    r, c = cell_rc(cell)
    return "center" if (r, MIRROR_CELL[c]) == ("mid", "center") else f"{r}-{MIRROR_CELL[c]}"


def cell_chance(gt_cell):
    """r0 chance (uniform random cell); r1 reports it as a diagnostic only"""
    return sum(cell_score(gt_cell, c) for c in CELLS) / 9


def height_score(g, m):
    if not g or not m or g <= 0 or m <= 0:
        return 0.0
    a, b = TH["height_log"]
    return ramp_down(abs(math.log(m / g)), a, b)


@lru_cache(maxsize=None)
def height_chance(g):
    n = 200
    lo, hi = math.log(0.05), 0.0
    return sum(height_score(g, math.exp(lo + (hi - lo) * (i + 0.5) / n)) for i in range(n)) / n


def facing_score(gt_word, ang):
    if ang is None or gt_word not in TH["facing_dir"]:
        return 0.0
    d = abs((ang - TH["facing_dir"][gt_word] + 180) % 360 - 180)
    a, b = TH["facing_deg"]
    return ramp_down(d, a, b)


FACING_CHANCE = (2 * TH["facing_deg"][0] + (TH["facing_deg"][1] - TH["facing_deg"][0])) / 360


def model_posture(b):
    if b is None or not b.get("present"):
        return None
    return "airborne" if b.get("airborne") else b.get("posture")


def posture_score(gt_word, mp):
    ok = TH["posture"].get(gt_word)
    if ok is None:
        return None
    if mp in ok:
        return 1.0
    for a, c, v in TH["posture_partial"]:
        if {a, c} == {mp, ok[0]}:
            return v
    return 0.0


def loc_class(typ):
    """r1 locomotion class of a GT type: still / moving / jump / land / climb / fall, None = parked"""
    L = TH["locomotion"]
    if typ in L["still"]:
        return "still"
    if typ in L["moving"]:
        return "moving"
    if typ in L["events"]:
        return typ
    return None


def speed_score(cls, v):
    """r1: still / moving only (complementary ramps); anything else None"""
    S = TH["speed_h"]
    if cls == "still":
        return 0.0 if v is None else ramp_down(v, S["still_full"], S["still_zero"])
    if cls == "moving":
        return 0.0 if v is None else ramp_up(v, S["move_zero"], S["move_full"])
    return None


def size_band(lab, x):
    """x = log(model height ratio); approach / recede / same are complementary inside the band"""
    T = TH["size_trend"]
    lo, hi = math.log(T["band_lo_ratio"]), math.log(T["band_hi_ratio"])
    up, dn = ramp_up(x, lo, hi), ramp_up(-x, lo, hi)
    return up if lab == "approach" else dn if lab == "recede" else max(0.0, 1.0 - up - dn)


# ------------------------------------------------------------------------------------------------- the scorer
class Scorer:
    def __init__(self, view, perm=None, ctx=None):
        self.v, self.perm = view, perm            # perm: {gi: [donor index per element]} for the chance runs
        self.ctx = ctx if ctx is not None else (item_ctx(view.item) if view.item else None)

    def el(self, gi, frame, i):
        e = frame["elements"][i]
        if self.perm is None or gi not in self.perm:
            return e
        d = frame["elements"][self.perm[gi][i]]
        return dict(e, c=d["c"], bbox=d["bbox"], depth_med=d.get("depth_med"), cell=d["cell"],
                    height_frac=d.get("height_frac"))

    def at_times(self, e):
        shot = e.get("shot")
        if "t" in e:
            return shot, [e["t"]]
        t0, t1 = e["t_range"]
        ts = self.v.times(shot, t0, t1) if t1 - t0 >= 0.25 else [t0]
        return shot, ts or [t0]

    def park(self, e, why):
        if self.ctx is not None:
            self.ctx.park(e, why)

    @staticmethod
    def unread(fr):
        """5.2 (M15): a model frame most of whose image is nearer than clip_start (model_view.py): not read"""
        return fr is not None and bool(fr.get("unreadable")) and fixes.on("M15")

    def pairwise(self, e, a, b, rel, gt_times=None, bind=None):
        shot, ts = self.at_times(e)
        ts = gt_times or ts
        sc, pres = [], False
        for t in ts:
            gi, fr = self.v.frame(shot, t)
            if self.unread(fr):
                continue                # 5.2 (M15): leaves the denominator
            if fr is None:
                sc.append(0.0)
                continue
            if bind is None:
                A, B = find(fr, a), find(fr, b)
            else:                                   # 5.2 M7
                A, B = find_m7(fr, a, bind[0], b), find_m7(fr, b, bind[1], a)
            if not A or not B:
                sc.append(0.0)
                continue
            pres = True
            sc.append(1.0 if any(pair_ok(rel, self.el(gi, fr, i), self.el(gi, fr, j))
                                 for i in A for j in B if i != j) else 0.0)
        return (sum(sc) / len(sc) if sc else None), pres

    def position(self, e, label, cell, ts=None, bind=False):
        shot, ts0 = self.at_times(e)
        sc, pres = [], False
        for t in ts or ts0:
            gi, fr = self.v.frame(shot, t)
            if self.unread(fr):
                continue                # 5.2 (M15): leaves the denominator
            if bind is not False:                   # 5.2 M7
                A = find_m7(fr, label, bind) if fr is not None else []
            else:
                A = find(fr, label) if fr is not None else []
            if not A:
                sc.append(0.0)
                continue
            pres = True
            sc.append(max(cell_score(cell, self.el(gi, fr, i)["cell"]) for i in A))
        return (sum(sc) / len(sc) if sc else None), pres

    def visible(self, shot, t, label):
        gi, fr = self.v.frame(shot, t)
        if self.unread(fr):
            return None                 # 5.2 (M15): unreadable, neither visible nor absent
        return fr is not None and bool(find(fr, label))


def geo_times(item_geo, e):
    """GT sample times of a geometry fact: the shot's samples inside its t_range where both labels were seen"""
    t0, t1 = e["t_range"]
    p = e["params"]
    labs = [p[k] for k in ("a", "b", "label") if p.get(k) and p.get(k) != "ground"]
    out = []
    for sm in item_geo.get(e["shot"], []):
        if t0 - 1e-6 <= sm["t"] <= t1 + 1e-6 and all(any(x["label"] == lb for x in sm["elements"]) for lb in labs):
            out.append(sm["t"])
    return out


def score_entry(S, e, geo):
    """-> list of (block, score, present, chance key[, note]) for one checklist entry; [] = not read by this layer
    (other check, or parked: the reason is recorded in the item context)"""
    ck, p, subj = e["check"], e["params"], e.get("subjects") or []
    out = []
    if ck.startswith("frame_relation:"):
        rel = ck.split(":")[1]
        if rel not in ("left_of", "above", "in_front_of"):
            return out
        ts = geo_times(geo, e)
        if fixes.on("M7"):
            bind = tuple(None if ID.match(p[k]) else gt_binding(geo, e, p[k], k) for k in ("a", "b"))
            s, pr = S.pairwise(e, p["a"], p["b"], rel, ts or None, bind=bind)
        else:
            s, pr = S.pairwise(e, p["a"], p["b"], rel, ts or None)
        out.append(("depth_order" if rel == "in_front_of" else "screen_order", s, pr, None))
    elif ck == "screen_side":
        a, b = subj
        l, r = (a, b) if p["value"] == "a_left" else (b, a)
        s, pr = S.pairwise(e, l, r, "left_of")
        out.append(("screen_order", s, pr, None))
    elif ck == "depth_order":
        a, b = subj
        f, k = (a, b) if p["value"] == "a_nearer" else (b, a)
        s, pr = S.pairwise(e, f, k, "in_front_of")
        out.append(("depth_order", s, pr, None))
    elif ck == "element_in_view":
        ts = geo_times(geo, e)
        if fixes.on("M7") and not ID.match(p["label"]):
            s, pr = S.position(e, p["label"], p["cell"], ts or None, bind=gt_binding(geo, e, p["label"], None))
        else:
            s, pr = S.position(e, p["label"], p["cell"], ts or None)
        out.append(("position", s, pr, ("cell", p["cell"])))
    elif ck == "object_in_view":
        lab = p.get("ref") if ID.match(p.get("ref") or "") else (p.get("class") or p.get("ref"))
        if not lab or not p.get("pos"):
            return out
        s, pr = S.position(e, lab, p["pos"])
        out.append(("position", s, pr, ("cell", p["pos"])))
    elif ck == "screen_track":
        sid = subj[0]
        n_unread = 0
        for t, cell, _depth, h in p["rows"]:
            gi, fr = S.v.frame(e["shot"], t)
            if S.unread(fr):
                n_unread += 1
                continue                # 5.2 (M15): unreadable row, not scored
            A = find(fr, sid) if fr is not None else []
            out.append(("presence", 1.0 if A else 0.0, True, pk(1, sid)))
            if A:
                el = S.el(gi, fr, A[0])
                out.append(("position", cell_score(cell, el["cell"]), True, ("cell", cell)))
                out.append(("size", height_score(h, el.get("height_frac")), True, ("h", h)))
            else:
                out.append(("position", 0.0, False, ("cell", cell)))
                out.append(("size", 0.0, False, ("h", h)))
        if n_unread and n_unread == len(p["rows"]):
            S.park(e, NO_READ)
    elif ck == "in_frame":
        shot, ts = S.at_times(e)
        vis = [v for v in (S.visible(shot, t, subj[0]) for t in ts) if v is not None]     # 5.2 (M15)
        out.append(("presence", sum(vis) / len(vis) if vis else None, True, pk(1, subj[0])))
    elif ck == "landmark_present":
        # r2: its own block (read once, by A1), R = 0
        if fixes.on("M7"):                         # 5.2 M7: every element of the class, as the GT side counts it
            hit = any(find_m7(fr, p["class"]) for fr in S.v.frames.values())
        else:
            hit = any(find(fr, p["class"]) for fr in S.v.frames.values())
        out.append(("landmarks", 1.0 if hit else 0.0, True, ("lm", p["class"])))
    elif ck in ("enter_frame", "exit_frame", "occluded"):
        # r1: two facts - visible around t (positive) and absent somewhere on the empty side within the window
        # (negative); r0 scored their conjunction
        t, shot, w = e["t"], e["shot"], TH["enter_exit_window_s"]
        ts = S.v.times(shot, t - w - 0.25, t + w + 0.25)
        if ck == "enter_frame":
            near = [x for x in ts if t - 0.25 <= x <= t + 0.5]
            side = [x for x in ts if t - w - 0.25 <= x <= t - 0.25]
        else:                                     # exit_frame / occluded: seen up to t, gone after
            near = [x for x in ts if t - 0.5 <= x <= t + 0.25]
            side = [x for x in ts if t + 0.25 <= x <= t + w + 0.25]
        near = [v for v in (S.visible(shot, x, subj[0]) for x in near) if v is not None]     # 5.2 (M15)
        side = [v for v in (S.visible(shot, x, subj[0]) for x in side) if v is not None]
        out.append(("presence", (1.0 if any(near) else 0.0) if near else None, True, pk(1, subj[0])))
        if side:
            out.append(("presence", 1.0 if any(not v for v in side) else 0.0, True,
                        pk(-1, subj[0])))
    elif ck == "screen_dir":
        t0, t1 = e["t_range"]
        if t1 - t0 < TH["lateral"]["min_dur_s"] - 1e-6:
            S.park(e, "lateral: shorter than lateral.min_dur_s")
        elif S.ctx is not None and e["id"] in S.ctx.lateral_park:
            S.park(e, S.ctx.lateral_park[e["id"]])                       # r2
        else:
            out += lateral_check(S, e, subj[0], p["dir"])
    elif S.v.video:
        return out                                # body entries need model bodies
    elif ck == "facing_run":
        shot, ts = S.at_times(e)
        bs = [S.v.body(subj[0], t) for t in ts]
        if fixes.on("M14"):             # 5.2 (M14): times where the body's facing is unknown are not scored
            bs = [b for b in bs if not (b and b.get("present") and b.get("facing_cam") == FACING_UNKNOWN)]
        pres = any(b and b.get("present") for b in bs)
        sc = [facing_score(p["facing"], b.get("facing_cam")) if b and b.get("present") else 0.0 for b in bs]
        out.append(("facing", sum(sc) / len(sc) if sc else None, pres, ("facing", p["facing"])))
    elif ck == "mutual_facing" and p.get("value") == "facing_each_other":
        shot, ts = S.at_times(e)
        a, b = subj
        sc, pres = [], False
        for t in ts:
            gi, fr = S.v.frame(shot, t)
            ba, bb = S.v.body(a, t), S.v.body(b, t)
            if S.unread(fr) or (fixes.on("M14") and any(x and x.get("present") and x.get("facing_cam") == FACING_UNKNOWN
                                                         for x in (ba, bb))):
                continue                # 5.2 (M15 unreadable frame / M14 facing unknown): not scored
            A, B = (find(fr, a), find(fr, b)) if fr is not None else ([], [])
            if not (A and B and ba and bb and ba.get("facing_cam") is not None and bb.get("facing_cam") is not None):
                sc.append(0.0)
                continue
            pres = True
            ea, eb = S.el(gi, fr, A[0]), S.el(gi, fr, B[0])
            l, r = (ba, bb) if ea["c"][0] <= eb["c"][0] else (bb, ba)
            m = TH["mutual_facing_min_deg"]
            sc.append(1.0 if l["facing_cam"] >= m and r["facing_cam"] <= -m else 0.0)
        out.append(("facing", sum(sc) / len(sc) if sc else None, pres, ("mutual", None)))
    elif ck == "posture_run":
        if p["posture"] not in TH["posture"]:
            return out
        shot, ts = S.at_times(e)
        bs = [S.v.body(subj[0], t) for t in ts]
        pres = any(b and b.get("present") for b in bs)
        sc = [posture_score(p["posture"], model_posture(b)) or 0.0 for b in bs]
        out.append(("posture", sum(sc) / len(sc), pres, ("posture", p["posture"])))
    elif ck == "locomotion_segment":
        shot, ts = S.at_times(e)
        sid, typ = subj[0], p.get("type")
        cls = loc_class(typ)
        pb = [b for b in (S.v.body(sid, t) for t in ts) if b and b.get("present")]
        pres = bool(pb)
        t0, t1 = e["t_range"]
        L = TH["locomotion"]
        if cls is None:
            S.park(e, "locomotion: type %s not read" % typ)
        elif cls in ("jump", "land"):
            sc_on, note_on = onset_score(S, shot, sid, cls, t0, t1)
            out.append(("locomotion", sc_on, pres, ("loc", cls), note_on))
        elif cls == "climb":
            pad = L["climb"]["pad_s"]
            hb = [b["base_h"] for b in (S.v.body(sid, x) for x in S.v.times(shot, t0 - pad, t1 + pad))
                  if b and b.get("present") and b.get("base_h") is not None]
            rise = max((hb[j] - min(hb[:j]) for j in range(1, len(hb))), default=0.0)
            out.append(("locomotion", 1.0 if rise >= L["climb"]["rise_h"] else 0.0, pres, ("loc", cls)))
        elif cls == "fall":
            out.append(("locomotion", fall_score(S, shot, sid, t0, t1), pres, ("loc", cls)))
        else:
            v = mean(b["speed_h"] for b in pb if b.get("speed_h") is not None)
            out.append(("locomotion", speed_score(cls, v) if pres else 0.0, pres, ("loc", cls)))
        out += direction_checks(S, e, sid, shot, ts)
    if (fixes.on("M14") or fixes.on("M15")) and any(r[1] is None for r in out):     # 5.2: no scorable model time
        S.park(e, NO_READ)
        out = [r for r in out if r[1] is not None]
    return out


NO_READ = "5.2: no scorable model time (M14 facing unknown / M15 frame unreadable)"


def onset_score(S, shot, sid, cls, t0, t1):
    """r2 (jump_land_onset): a take-off (jump) / landing (land) between consecutive present body rows at the GT grid
    times, the later row inside [t0 - tol, t1 + tol]; rows = the grid times in that window plus one before it"""
    tol = R2["jump_land_onset"]["tol_s"]
    lo, hi = t0 - tol, t1 + tol
    allts = S.v.times(shot, -1e9, 1e9)
    win = [x for x in allts if lo - 1e-6 <= x <= hi + 1e-6]
    before = [x for x in allts if x < lo - 1e-6]
    seq = before[-1:] + win
    rows = [(x, b) for x, b in ((x, S.v.body(sid, x)) for x in seq) if b and b.get("present")]
    for (_, a), (xb, b) in zip(rows, rows[1:]):
        fa, fb = bool(a.get("airborne")), bool(b.get("airborne"))
        if ((cls == "jump" and not fa and fb) or (cls == "land" and fa and not fb)) and lo - 1e-6 <= xb <= hi + 1e-6:
            return 1.0, {"onset_t": xb}
    return 0.0, {"onset_t": None, "n_rows": len(rows)}


def fall_score(S, shot, sid, t0, t1):
    """as temporal_score.score_fall / the 5.0.9 fall detector, on the body rows at the GT grid times"""
    F = TH["locomotion"]["fall"]
    rows = [b for b in (S.v.body(sid, x) for x in S.v.times(shot, t0 - F["pad_s"], t1 + F["pad_s"]))
            if b and b.get("present") and b.get("base_h") is not None and b.get("top_h") is not None]
    for i, a in enumerate(rows):
        later = [b for b in rows[i + 1:] if b["t"] <= a["t"] + F["within_s"]]
        if a["top_h"] - a["base_h"] >= F["extent_hi"] and any(b["top_h"] - b["base_h"] <= F["extent_lo"] for b in later):
            return 1.0
        if any(a["base_h"] - b["base_h"] >= F["drop_h"] for b in later):
            return 1.0
    return 0.0


def lateral_label(e, ctx):
    """GT lateral label of a locomotion_segment (V2 run) -> (label or None, park reason or None)"""
    p = e["params"]
    if not (p.get("screen_dir") or p.get("gt_screen_dir")):
        return None, None
    L = TH["lateral"]
    t0, t1 = e["t_range"]
    if p.get("type") in L["park_types"]:
        return None, "lateral: run of type %s" % p.get("type")
    if t1 - t0 < L["min_dur_s"] - 1e-6:
        return None, "lateral: shorter than lateral.min_dur_s"
    static = ctx is not None and ctx.move.get(e["shot"]) in L["static_cam_moves"]
    if static and p.get("screen_dir"):
        return p["screen_dir"], None
    if p.get("gt_screen_dir"):
        return p["gt_screen_dir"], None
    return None, "lateral: cell direction under a moving / unknown camera"


def direction_checks(S, e, sid, shot, ts):
    """lateral heading (definition A), size trend, camera path of one subject over a V2 run"""
    p = e["params"]
    out = []
    lab, why = lateral_label(e, S.ctx)
    if why:
        S.park(e, why)
    if lab and S.ctx is not None and e["id"] in S.ctx.lateral_park:
        S.park(e, S.ctx.lateral_park[e["id"]])                           # r2
    elif lab:
        out += lateral_check(S, e, sid, lab)
    if len(ts) < 2:
        return out
    els, bs = [], []
    for t in ts:
        gi, fr = S.v.frame(shot, t)
        A = find(fr, sid) if fr is not None and not S.unread(fr) else []     # 5.2 (M15)
        els.append(S.el(gi, fr, A[0]) if A else None)
        bs.append(S.v.body(sid, t))
    vis = [x for x in els if x]
    if p.get("size") in ("approach", "recede", "same"):
        lab_s = p["size"]
        rg = S.ctx.gt_ratio(shot, sid, *e["t_range"]) if S.ctx is not None else None
        if lab_s == "same" and (rg is None or abs(math.log(rg)) > math.log(TH["size_trend"]["gt_same_max_ratio"])):
            S.park(e, "size: GT 'same' outside gt_same_max_ratio")
        elif len(vis) < 2 or not vis[0].get("height_frac") or not vis[-1].get("height_frac"):
            out.append(("direction", 0.0, bool(vis), ("dir", "size", lab_s)))
        else:
            x = math.log(vis[-1]["height_frac"] / vis[0]["height_frac"])
            out.append(("direction", size_band(lab_s, x), True, ("dir", "size", lab_s), round(math.exp(x), 3)))
    path = set(p.get("path") or [])
    pb = [b for b in bs if b and b.get("present")]
    if len(path & {"toward_camera", "away_from_camera"}) == 1:
        want = "toward_camera" if "toward_camera" in path else "away_from_camera"
        ds = [b["cam_dist_h"] for b in pb if b.get("cam_dist_h")]
        if len(ds) < 2:
            out.append(("direction", 0.0, bool(pb), ("dir", "cam_path", want)))
        else:
            r = ds[-1] / ds[0]
            q = TH["path"]["cam_ratio"]
            ok = r <= 1 / q if want == "toward_camera" else r >= q
            out.append(("direction", 1.0 if ok else 0.0, True, ("dir", "cam_path", want)))
    if len(path & {"up", "down"}) == 1:
        want = "up" if "up" in path else "down"
        hb = [b["base_h"] for b in pb if b.get("base_h") is not None]
        if len(hb) < 2:
            out.append(("direction", 0.0, bool(pb), ("dir", "updown", want)))
        else:
            d = hb[-1] - hb[0]
            u = TH["path"]["up_down_h"]
            out.append(("direction", 1.0 if (d >= u if want == "up" else d <= -u) else 0.0, True,
                        ("dir", "updown", want)))
    return out


def lateral_check(S, e, sid, want):
    """definition A (pvscore 5.0.10 camera._segment_dir) for every lateral fact: the subject's heading seen from the
    camera over the fact's range without its first and last lateral.trim: the body displacement (body heights)
    between the first and last present rows, in the horizontal frame of the camera at the row nearest the middle;
    away / toward within segment_dir.away_deg of the view axis, else left_to_right / right_to_left; static below
    static_h. A following camera that keeps the subject centred still reads the walking direction. Not read on the
    video side (no camera pose)."""
    if S.v.video:
        return []
    t0, t1 = e["t_range"]
    tr = TH["lateral"]["trim"] * (t1 - t0)
    key = ("dir", "lateral", want)
    ts = S.v.times(e["shot"], t0 + tr, t1 - tr)
    pb = [b for b in (S.v.body(sid, t) for t in ts) if b and b.get("present") and b.get("xy_scale")]
    cams = [b for b in pb if b.get("cam_f") and b.get("cam_r")]
    if len(pb) < 2 or not cams:
        return [("direction", 0.0, bool(pb), key, {"got": None, "n_rows": len(pb), "t_window": [round(t0 + tr, 3),
                 round(t1 - tr, 3)], "why": "fewer than 2 present body rows with xy_scale" if len(pb) < 2
                 else "no camera axes (cam_f / cam_r)"})]
    a, b = pb[0], pb[-1]
    c = min(cams, key=lambda x: abs(x["t"] - 0.5 * (a["t"] + b["t"])))
    sc = 0.5 * (a["xy_scale"] + b["xy_scale"])
    dx, dy = (b["x"] - a["x"]) * sc, (b["y"] - a["y"]) * sc
    T = TH["segment_dir"]
    th = math.degrees(math.atan2(dx * c["cam_r"][0] + dy * c["cam_r"][1], dx * c["cam_f"][0] + dy * c["cam_f"][1]))
    got = ("static" if math.hypot(dx, dy) < T["static_h"] else "away" if abs(th) < T["away_deg"]
           else "toward" if abs(th) > 180 - T["away_deg"] else "left_to_right" if th > 0 else "right_to_left")
    note = {"got": got, "disp_h": round(math.hypot(dx, dy), 3), "heading_deg": round(th, 1), "n_rows": len(pb),
            "t_first": a["t"], "t_last": b["t"], "t_cam": c["t"], "t_window": [round(t0 + tr, 3), round(t1 - tr, 3)]}
    return [("direction", 1.0 if got == want else 0.0, True, key, note)]


def segment_dir_check(S, e, sid):
    """r0 name kept for callers: a GT r1 screen_dir segment is now one lateral fact like any other"""
    return lateral_check(S, e, sid, e["params"]["dir"])


def negative_times(C):
    """GT side of the r1 presence negatives: at every V2 row time (scene_frames sample) of a shot, each cast subject
    with no positive presence fact of that shot (screen_track row, in_frame time or range; skipped and not_applicable
    entries included) within presence.neg_margin_s must not be in view; freeze-frame times excluded; on continuation
    items only times <= presence.continuation_neg_t_max_s (frame 1) -> [(shot, subject, t)]"""
    if C.negatives is None:
        T = TH["presence"]
        C.negatives = [(shot, sid, t) for shot, ts in C.v2_times.items() for t in ts
                       if not (T["exclude_frozen"] and C.is_frozen(t))
                       and (C.neg_t_max is None or t <= C.neg_t_max + 1e-6) for sid in C.cast
                       if not any(a - T["neg_margin_s"] < t < b + T["neg_margin_s"]
                                  for a, b in C.spans.get((shot, sid), []))]
    return C.negatives


def presence_negatives(S):
    """-> [(block, score, present, key, detail id)]: 1 when the model does not show the subject there"""
    if S.ctx is None:
        return []
    return [("presence", 0.0 if v else 1.0, True, pk(-1, sid), "absent:%s:%s@%.3f" % (shot, sid, t))
            for shot, sid, t, v in ((sh, si, x, S.visible(sh, x, si)) for sh, si, x in negative_times(S.ctx))
            if v is not None]           # 5.2 (M15): an unreadable frame is no absence


# --------------------------------------------------------------------------------------------- aggregation
def constant_chance(rows, kind):
    """best constant answer: posture (r0) / locomotion (r1: still or moving; event facts score 0 under either)"""
    if not rows:
        return 0.0
    if kind == "posture":
        best = 0.0
        for c in ("stand", "crouch", "sit", "lie", "airborne"):
            sc = [posture_score(r[3][1], c) or 0.0 for r in rows]
            best = max(best, sum(sc) / len(sc))
        return best
    n = Counter(r[3][1] for r in rows)
    return max(n["still"], n["moving"]) / len(rows)


def position_chance(rows, pooled=True):
    best = 0.0
    for c in CELLS:
        m = mirror_cell(c)
        s = [0.5 * (cell_score(r[3][1], c) + cell_score(r[3][1], m)) if pooled else cell_score(r[3][1], c)
             for r in rows]
        best = max(best, sum(s) / len(s))
    return best


def pi_random(rows):
    n = len(rows)
    h = Counter(r[3][1] for r in rows)
    return sum(k / n * sum(cell_score(r[3][1], c) for r in rows) / n for c, k in h.items())


def dir_sub_chance(sub, labels):
    c = Counter(labels)
    if sub == "lateral":
        cand = [0.5 * (c["left_to_right"] + c["right_to_left"])] + [v for k, v in c.items() if k not in LAT_PAIR]
        return max(cand) / len(labels)
    return max(c.values()) / len(labels)


def dir_chance(rows):
    by = defaultdict(list)
    for r in rows:
        by[r[3][1]].append(r[3][2])
    subs = {s: {"n": len(v), "R": round(dir_sub_chance(s, v), 3)} for s, v in sorted(by.items())}
    return sum(len(v) * dir_sub_chance(s, v) for s, v in by.items()) / len(rows), subs


SIZE_GRID = [math.exp(math.log(0.05) * (1 - i / 100)) for i in range(101)]
FACING_GRID = list(range(R2["facing_chance"]["grid_deg"][0], R2["facing_chance"]["grid_deg"][1] + 1,
                         R2["facing_chance"]["grid_deg"][2]))


def size_best_constant(rows):
    """r2: best single height_frac over the 101-point log grid (a null GT height scores 0)"""
    return max(mean(height_score(r[3][1], h) if r[3][1] else 0.0 for r in rows) for h in SIZE_GRID)


def facing_best_constant(rows):
    """r2: best single facing angle; mutual_facing rows score 0 under any constant"""
    return max(mean(facing_score(r[3][1], a) if r[3][0] == "facing" else 0.0 for r in rows) for a in FACING_GRID)


def presence_label_constant(rows):
    """r2: best balanced accuracy of a per-subject constant answer: 0.5 * sum_l max(pos_l / N_pos, neg_l / N_neg);
    1.0 when one class is empty (BA undefined: the block then scores 0)"""
    pos = Counter(r[3][2] for r in rows if is_pos(r[3]))
    neg = Counter(r[3][2] for r in rows if is_neg(r[3]))
    npos, nneg = sum(pos.values()), sum(neg.values())
    if not npos or not nneg:
        return 1.0
    return 0.5 * sum(max(pos[l] / npos, neg[l] / nneg) for l in set(pos) | set(neg))


def chance(blk, rows, perm_P=None):
    if not rows:
        return None
    if blk in PAIRWISE:
        return sum(perm_P[blk]) / len(perm_P[blk]) if perm_P and perm_P.get(blk) else 0.5
    if blk == "position":
        return position_chance(rows)
    if blk == "size":
        return size_best_constant(rows)
    if blk == "facing":
        return facing_best_constant(rows)
    if blk == "presence":
        return presence_label_constant(rows)
    if blk == "landmarks":
        return R2["landmarks_block"]["chance"]
    if blk in ("posture", "locomotion"):
        return constant_chance(rows, blk)
    if blk == "direction":
        return dir_chance(rows)[0]
    raise ValueError(blk)


def diagnostics(blk, rows):
    """alternative chance values over all facts (GT only), for the record"""
    if blk == "position":
        return {"R_uniform_r0": round(mean(cell_chance(r[3][1]) for r in rows), 3),
                "R_pi_random": round(pi_random(rows), 3),
                "R_best_const_unpooled": round(position_chance(rows, pooled=False), 3)}
    if blk == "size":
        return {"R_loguniform_r1": round(mean(height_chance(r[3][1]) if r[3][1] else 0.0 for r in rows), 3)}
    if blk == "facing":
        fr = [r for r in rows if r[3][0] == "facing"]
        return {"R_uniform_r1": round(mean(FACING_CHANCE if r[3][0] == "facing" else 1 / 9 for r in rows), 3),
                "n_facing_run": len(fr), "n_mutual": len(rows) - len(fr)}
    if blk == "presence":
        return {"R_constant_r1": 0.5, "labels": len({r[3][2] for r in rows})}
    return {}


def block_scores(rows, perm_P=None):
    """rows: (block, score, present, chance key); perm_P: {block: [P under each permutation]}"""
    out = {}
    LI, PR = TH["low_info"], TH["presence"]
    for blk in BLOCKS:
        rs = [r for r in rows if r[0] == blk]
        if not rs:
            continue
        pres = [r for r in rs if r[2]]
        cov = len(pres) / len(rs)
        extra = {}
        if blk == "presence":
            pos = [r[1] for r in pres if is_pos(r[3])]
            neg = [r[1] for r in pres if is_neg(r[3])]
            tpr, tnr = mean(pos), mean(neg)
            if pos and neg:
                Pm, Rc = 0.5 * (tpr + tnr), presence_label_constant(pres)
            else:
                Pm, Rc = (tpr if pos else tnr) or 0.0, 1.0
            extra = {"n_pos": len(pos), "n_neg": len(neg), "TPR": None if tpr is None else round(tpr, 3),
                     "TNR": None if tnr is None else round(tnr, 3)}
        else:
            Pm = sum(r[1] for r in pres) / len(pres) if pres else 0.0
            Rc = chance(blk, pres, perm_P) if pres else (chance(blk, rs, perm_P) if blk in PAIRWISE else 0.0)
        R_gt = None if blk in PAIRWISE else chance(blk, rs)
        info, why = True, None
        if blk == "presence":
            n_p = sum(1 for r in rs if is_pos(r[3]))
            n_n = sum(1 for r in rs if is_neg(r[3]))
            if not (n_p >= PR["min_class_n"] and n_n >= PR["min_class_n"]):
                info, why = False, "%d positive / %d negative facts (< %d in a class)" % (n_p, n_n, PR["min_class_n"])
        if info and R_gt is not None and R_gt >= LI["r_max"]:
            info, why = False, "R_gt %.3f >= %.2f: a constant answer gets %.0f%% of the facts" % (
                R_gt, LI["r_max"], 100 * R_gt)
        if info and len(rs) < LI["min_n"]:                            # r2
            info, why = False, "n %d < low_info.min_n %d" % (len(rs), LI["min_n"])
        if blk == "direction":
            extra["sub"] = dir_chance(rs)[1]
        if blk == "locomotion":
            extra["classes"] = dict(sorted(Counter(r[3][1] for r in rs).items()))
        corr = max(0.0, min(1.0, (Pm - Rc) / (1 - Rc))) if Rc < 1 else 0.0
        d = diagnostics(blk, rs)
        out[blk] = {"n": len(rs), "present": len(pres), "coverage": round(cov, 3), "P": round(Pm, 3),
                    "R": round(Rc, 3), "score": round(corr * cov, 3), "raw": round(Pm * cov, 3),
                    "R_gt": None if R_gt is None else round(R_gt, 3), "informative": info, "low_info": not info,
                    "info_reason": why,
                    **extra, **({"diag": d} if d else {})}
    return out


def make_perm(view, seed):
    rng = random.Random(seed)
    perm = {}
    for gi, fr in view.frames.items():
        idx = [i for i, e in enumerate(fr["elements"]) if not e["class"].startswith("surface:")]
        sh = idx[:]
        rng.shuffle(sh)
        m = list(range(len(fr["elements"])))
        for a, b in zip(idx, sh):
            m[a] = b
        perm[gi] = m
    return perm


def detail_row(eid, check, r):
    d = {"id": eid, "check": check, "block": r[0], "score": round(r[1], 3), "present": r[2]}
    k = r[3]
    if is_pos(k) or is_neg(k):
        d["kind"] = "pos" if is_pos(k) else "neg"
    elif isinstance(k, tuple) and k[0] == "dir":
        d["kind"] = "%s:%s" % (k[1], k[2])
    elif isinstance(k, tuple) and k[0] == "loc":
        d["kind"] = k[1]
    if len(r) > 4 and isinstance(r[4], dict):
        d.update(r[4])
    elif len(r) > 4 and r[4] is not None:
        d["model_ratio" if d.get("kind", "").startswith("size:") else "got"] = r[4]
    return d


def score_model(item, view, entries, geo):
    C = item_ctx(item)
    rows, detail = [], []
    S = Scorer(view, ctx=C)
    for e in entries:
        for r in score_entry(S, e, geo):
            rows.append(r[:4])
            detail.append(detail_row(e["id"], e["check"], r))
    for r in presence_negatives(S):
        rows.append(r[:4])
        detail.append(detail_row(r[4], "v2_absent", r))
    pairs = [e for e in entries if e["check"] in ("frame_relation:left_of", "frame_relation:above",
                                                  "frame_relation:in_front_of", "screen_side", "depth_order")]
    perm_P = defaultdict(list)
    for k in range(TH["chance"]["perms"]):
        Sk = Scorer(view, make_perm(view, TH["chance"]["seed"] * 100003 + k), ctx=C)
        pr_rows = [r for e in pairs for r in score_entry(Sk, e, geo)]
        for blk in PAIRWISE:
            rs = [r for r in pr_rows if r[0] == blk and r[2]]
            if rs:
                perm_P[blk].append(sum(r[1] for r in rs) / len(rs))
    return block_scores(rows, perm_P), detail


def mirrored(d):
    """negative control: the model frames flipped left-right (boxes, centres, cells, facing to the camera, the
    camera's screen-right axis, so definition A reads left / right swapped)"""
    d = json.loads(json.dumps(d))
    for fr in d["frames"].values():
        for e in fr["elements"]:
            x0, y0, x1, y1 = e["bbox"]
            e["bbox"] = [round(1 - x1, 3), y0, round(1 - x0, 3), y1]
            e["c"] = [round(1 - e["c"][0], 3), e["c"][1]]
            e["cell"] = mirror_cell(e["cell"])
            if e.get("facing_cam") is not None:
                e["facing_cam"] = -e["facing_cam"]
    if "facing_cam" not in d.get("body_cols", []):
        return d
    k = d["body_cols"].index("facing_cam")
    kr = d["body_cols"].index("cam_r") if "cam_r" in d["body_cols"] else None
    for rows in d["body_series"].values():
        for r in rows:
            if len(r) > k and r[k] is not None and r[k] != FACING_UNKNOWN:     # 5.2 (M14)
                r[k] = -r[k]
            if kr is not None and len(r) > kr and r[kr] is not None:     # screen right flips with the frame
                r[kr] = [-x for x in r[kr]]
    return d


def shifted(d):
    """negative control: inside every GT shot, GT time t reads the model at t + half the shot (wrapping)"""
    d = json.loads(json.dumps(d))
    by_shot = defaultdict(list)
    for r in d["time_map"]:
        by_shot[r[1]].append(r)
    new_tm, remap = [], {}
    for rows in by_shot.values():
        rows.sort(key=lambda r: r[0])
        n = len(rows)
        for i, r in enumerate(rows):
            src = rows[(i + n // 2) % n]
            new_tm.append([r[0], r[1]] + src[2:])
            remap[r[0]] = src[0]
    d["time_map"] = new_tm
    for eid, rows in d["body_series"].items():
        at = {r[0]: r for r in rows}
        d["body_series"][eid] = [[t] + at[t2][1:] for t, t2 in remap.items() if t2 in at]
    return d


def main():
    global P
    opt = {a.split("=", 1)[0]: (a.split("=", 1)[1] if "=" in a else True) for a in sys.argv[1:] if a.startswith("--")}
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ctl = "--controls" in opt
    only = set(opt["--models"].split(",")) if isinstance(opt.get("--models"), str) else None
    if "--root" in opt:
        if "--out" not in opt:
            sys.exit("--root needs --out (never write into another data dir)")
        P = Path(opt["--root"]) if Path(opt["--root"]).is_absolute() else R / opt["--root"]
    res = {"thresholds": TH_FILE, "version": "r2", "block_fields": FIELD_DOC, "items": {}, "parked": {}}
    for item in args or ITEMS:
        ck = json.loads((P / item / "checklist.json").read_text())
        entries = [e for e in ck["entries"] if e["scorable_new"] not in SKIP]
        g = SF.load(P / item / "scene_frames.json")
        geo = {sh["shot"]: [s for s in sh["samples"] if s.get("fi") is not None] for sh in g["shots"]}
        views = {}
        if only is None or "video_geom" in only:
            views["video_geom"] = video_view(item)
        for f in sorted((PT.VIEW_OUT / item / "model_view").glob("*.json")):
            if only is not None and f.stem not in only:
                continue
            d = json.loads(f.read_text())
            views[f.stem] = View(d, item=item)
            if ctl and not (f.stem == "fake" or f.stem.startswith("dry-run")):   # no controls of the fixture model
                views[f.stem + "~mirror"] = View(mirrored(d), item=item)
                views[f.stem + "~shift"] = View(shifted(d), item=item)
        res["items"][item] = {}
        for name, v in views.items():
            blocks, detail = score_model(item, v, entries, geo)
            res["items"][item][name] = {"blocks": blocks, "entries": detail}
            print(item, f"{name:13s}", "  ".join(
                f"{b}:{x['score']:.2f}(P{x['P']:.2f}/R{x['R']:.2f}/c{x['coverage']:.2f}/n{x['n']}{'' if x['informative'] else '/LOW'})"
                for b, x in blocks.items()), flush=True)
        res["parked"][item] = item_ctx(item).parked_summary()
    if "--out" in opt:
        out = Path(opt["--out"]) if Path(opt["--out"]).is_absolute() else R / opt["--out"]
    else:
        out = PT.VIEW_OUT / ("view_scores_controls.json" if ctl else "view_scores_subset.json" if only else "view_scores.json")
    out.write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
