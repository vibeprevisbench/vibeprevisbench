"""Model-side reading of a pvdump-5.0 dump: pieces, furnishing instances, named bodies, cameras, shots, grids."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from . import fixes, geom
from .gt import OPENING_CLASSES, group_of

SEAT_CLASSES = {"chair", "armchair", "sofa", "bench", "toilet", "bed", "bench_outdoor"}
BODY_GROUPS = {"agent", "animal"}
STUFF_GROUPS = {"structure", "terrain"}
FLOOR_CLASSES = {"floor", "ground", "road", "sidewalk", "grass", "sand", "parking_lot", "field", "court", "path",
                 "platform", "stage", "crosswalk", "track"}
WALL_CLASSES = {"wall", "partition", "fence"}
CONNECTOR_CLASSES = {"door", "doorway", "archway", "gate", "stairs", "ramp", "elevator", "escalator", "tunnel",
                     "bridge", "path", "sidewalk"}
# 5.2 (M14): front-marker part words: the protocol's (CLI_CONVENTIONS "a clearly visible front marker (for example a
# chest or nose block)") and PREREG_52 M14's head-front parts (face, visor, eyes, an animal's muzzle / snout / beak);
# front / face not inside another word (waterfront, storefront, surface) unless capitalised (ChestFront, HeadFace)
FRONT_WORDS = re.compile(r"(?i)(?<![a-z])front|(?-i:Front)|nose|chest|visor|muzzle|snout|beak|eye"
                         r"|(?<![a-z])face|(?-i:Face)")


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)
            return True
        return False


def hull_area(xy):
    if len(xy) < 3:
        return 0.0
    h = geom.convex_hull(xy)
    if len(h) < 3:
        return 0.0
    return float(abs(0.5 * np.sum(h[:, 0] * np.roll(h[:, 1], -1) - np.roll(h[:, 0], -1) * h[:, 1])))


def contained_frac(xy_small, xy_big, margin):
    """Fraction of the small piece's footprint points inside the big piece's oriented footprint (+ margin)."""
    c, yaw, long, short = geom.min_area_rect(xy_big)
    a = math.radians(yaw)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    X = xy_small - np.asarray(c, float)
    inside = (np.abs(X @ u) <= long / 2 + margin) & (np.abs(X @ v) <= short / 2 + margin)
    return float(inside.mean()) if len(X) else 0.0


def _slices(P, n=8, min_pts=3):
    """Horizontal slices (xy points) of a body, bottom to top; slices with too few points are dropped."""
    z0, z1 = P[:, 2].min(), P[:, 2].max()
    e = np.linspace(z0, z1, n + 1)
    out = []
    for i in range(n):
        S = P[(P[:, 2] >= e[i]) & (P[:, 2] <= e[i + 1]), :2]
        if len(S) >= min_pts:
            out.append(S)
    return out


def body_center(P):
    """Body axis in xy: median over height slices of the slice extent midpoints (robust to markers and limbs)."""
    S = _slices(P)
    if not S:
        return P[:, :2].mean(0)
    return np.median(np.stack([(s.min(0) + s.max(0)) / 2 for s in S]), axis=0)


def person_facing(P, min_asym=0.03):
    """Facing yaw (deg) of a person / animal proxy from its front-marker asymmetry, or None.

    The body axis along every direction is the median over height slices of the slice midpoints, so a marker
    (nose, chest, bump, legs of a seated figure) that occupies a minority of the slices does not move it; the
    front is the direction whose extent beyond the axis most exceeds the extent on the opposite side.
    """
    if len(P) < 8 or P[:, 2].max() - P[:, 2].min() < 1e-3:
        return None, 0.0
    S = _slices(P)
    if len(S) < 3:
        return None, 0.0
    th = np.radians(np.arange(0, 360, 5.0))
    D = np.stack([np.cos(th), np.sin(th)], 1)
    H = np.stack([(s @ D.T).max(0) for s in S])           # support function of every slice
    Hop = np.roll(H, -36, axis=1)                          # ... in the opposite direction
    c = np.median((H - Hop) / 2, axis=0)                   # axis offset along each direction
    radius = float(np.median((H + Hop) / 2))
    Hm = H.max(0)
    asym = (Hm - c) - (np.roll(Hm, -36) + c)
    k = int(np.argmax(asym))
    if asym[k] < max(min_asym, 0.12 * radius):
        return None, float(asym[k])
    w = np.clip(asym, 0, None) * (np.abs(((np.arange(72) - k + 36) % 72) - 36) <= 6)
    ang = math.degrees(math.atan2(float((w * np.sin(th)).sum()), float((w * np.cos(th)).sum())))
    return ang % 360.0, float(asym[k])


def person_facing_52(P, fronts=(), min_asym=0.03):
    """5.2 (M14): (facing yaw deg or None, known) of a person / animal proxy.

    An explicit front signal first: fronts are the xy offsets of the body's front-marker parts (FRONT_WORDS) from the
    part each sits on (Model._front_marks); those of at least the asymmetry threshold of person_facing point to the
    front, and when the markers within 90 deg of one another are a strict majority the facing is their summed offset
    direction. Else
    the silhouette asymmetry (person_facing), only when every slice with a significant asymmetry of its own (the same
    threshold) points within the estimator's +-30 deg window of it; otherwise (None, False) = unknown, the check is not
    scored at that time. No readable front at all: (None, True), as before."""
    f, _ = person_facing(P, min_asym)
    if len(P) < 8 or P[:, 2].max() - P[:, 2].min() < 1e-3:
        return None, True
    S = _slices(P)
    if len(S) < 3:
        return None, True
    th = np.radians(np.arange(0, 360, 5.0))
    D = np.stack([np.cos(th), np.sin(th)], 1)
    H = np.stack([(s @ D.T).max(0) for s in S])
    Hop = np.roll(H, -36, axis=1)
    c = np.median((H - Hop) / 2, axis=0)
    thr = max(min_asym, 0.12 * float(np.median((H + Hop) / 2)))
    fm = _marker_yaw(fronts, thr)
    if fm is not None:
        return fm, True
    if f is None:
        return None, True
    A = (H - c) - (Hop + c)                                # every slice's own asymmetry
    k = A.argmax(1)
    sig = np.nonzero(A[np.arange(len(S)), k] >= thr)[0]
    if all(geom.angdiff(5.0 * k[i], f, 360) <= 30 for i in sig):
        return f, True
    return None, False


def _marker_yaw(fronts, thr):
    """5.2 (M14): the facing read from front-marker offsets (person_facing_52's marker rule), or None"""
    offs = [v for v in (np.asarray(m, float) for m in fronts) if np.linalg.norm(v) >= thr]
    if not offs:
        return None
    ys = [geom.yaw_of(v) for v in offs]
    best = max(([j for j in range(len(offs)) if geom.angdiff(ys[i], ys[j], 360) < 90] for i in range(len(offs))),
               key=len)
    if 2 * len(best) > len(offs):
        return geom.yaw_of(np.sum([offs[j] for j in best], axis=0)) % 360.0
    return None


def marker_facing_52(P, fronts, min_asym=0.03):
    """5.2 (M14, addendum 2): the facing read from the front markers alone (None when they do not read)"""
    if not len(fronts) or len(P) < 8 or P[:, 2].max() - P[:, 2].min() < 1e-3:
        return None
    S = _slices(P)
    if len(S) < 3:
        return None
    th = np.radians(np.arange(0, 360, 5.0))
    D = np.stack([np.cos(th), np.sin(th)], 1)
    H = np.stack([(s @ D.T).max(0) for s in S])
    Hop = np.roll(H, -36, axis=1)
    return _marker_yaw(fronts, max(min_asym, 0.12 * float(np.median((H + Hop) / 2))))


def local_front_52(Rs, yaws, win=30.0):
    """5.2 (M14, addendum 2): one front vector in a rigid part's own frame from world facing readings (yaw deg) taken
    at its world rotations Rs: the mean of the readings rotated into the part's frame, kept only when a strict
    majority of them lie within the estimator's +-30 deg window of it; else None"""
    if not len(yaws):
        return None
    V = np.array([Rm.T @ np.array([math.cos(math.radians(y)), math.sin(math.radians(y)), 0.0])
                  for Rm, y in zip(Rs, yaws)])
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    m = V.sum(0)
    if np.linalg.norm(m) < 1e-9:
        return None
    m /= np.linalg.norm(m)
    return m if 2 * int(((V @ m) >= math.cos(math.radians(win))).sum()) > len(V) else None


def seat_facing(P, rect, drop=0.8):
    """Front yaw of a seat-like object (away from the backrest / headboard / tank), or None.

    For each of the four footprint sides the top of the outer quarter strip is compared with the top of the
    opposite strip: a backrest makes one side reach the full height while the opposite side stays clearly lower.
    Symmetric shapes (boxes, benches, sofas seen along their arms) give no front.
    """
    z0, z1 = P[:, 2].min(), P[:, 2].max()
    h = z1 - z0
    if h < 0.2:
        return None
    center, yaw_l, long, short = rect
    a = math.radians(yaw_l)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    X = P[:, :2] - np.asarray(center, float)
    best = None
    for ax, ext in ((u, long), (v, short)):
        if ext < 0.15:
            continue
        q = X @ ax
        for sgn in (1.0, -1.0):
            rear, front = sgn * q > ext / 4, sgn * q < -ext / 4
            if rear.sum() < 3 or front.sum() < 3:
                continue
            zr, zf = P[rear, 2].max() - z0, P[front, 2].max() - z0
            if zr < 0.85 * h:
                continue
            ratio = zf / max(1e-6, zr)
            if ratio < drop and (best is None or ratio < best[0]):
                best = (ratio, geom.yaw_of(-sgn * ax) % 360.0)
    return None if best is None else best[1]


class Instance:
    def __init__(self, key, cls, cats, pieces, P, space, untagged=False):
        self.key, self.cls, self.untagged = key, cls, untagged
        self.group = group_of(cls, cats)
        self.pieces = pieces
        self.parts = sorted({p["part"] for p in pieces})
        self.P = P
        self.space = space
        self.lo, self.hi = P.min(0), P.max(0)
        self.rect = geom.min_area_rect(P[:, :2])
        self.center2 = np.asarray(self.rect[0], float)
        self.long_yaw, self.long, self.short = self.rect[1], self.rect[2], self.rect[3]
        self.height = float(self.hi[2] - self.lo[2])
        self.bottom, self.top = float(self.lo[2]), float(self.hi[2])
        self.facing = None
        self.facing_inferable = cls in SEAT_CLASSES or self.group in BODY_GROUPS
        if cls in SEAT_CLASSES:
            self.facing = seat_facing(P, self.rect)
        elif self.group in BODY_GROUPS:
            self.facing, _ = person_facing(P)
        if untagged:
            if self.long > 3.0 and self.height < 0.15:
                self.kind = "untagged_floor"
            elif self.height > 1.8 and self.short < 0.35 and self.long > 1.0:
                self.kind = "untagged_wall"
            else:
                self.kind = "untagged_object"
        elif self.group == "terrain":
            self.kind = "terrain"
        elif self.group == "structure":
            self.kind = "opening" if cls in OPENING_CLASSES else "structure"
        else:
            self.kind = "object"

    @property
    def diag(self):
        return float(np.linalg.norm(self.hi - self.lo))

    def summary(self):
        return {"key": self.key, "class": self.cls, "kind": self.kind, "space": self.space, "parts": len(self.parts),
                "center": [round(float(x), 3) for x in self.center2], "yaw_long": round(self.long_yaw, 1),
                "long": round(self.long, 3), "short": round(self.short, 3), "height": round(self.height, 3),
                "bottom": round(self.bottom, 3), "facing": None if self.facing is None else round(self.facing, 1)}


class Body:
    """A named entity (id) over the analysis frames."""

    def __init__(self, eid, cls, group, A):
        self.id, self.cls, self.group = eid, cls, group
        self.present = np.zeros(A, bool)
        self.center = np.full((A, 2), np.nan)
        self.base = np.full(A, np.nan)
        self.top = np.full(A, np.nan)
        self.lo = np.full((A, 3), np.nan)
        self.hi = np.full((A, 3), np.nan)
        self.facing = np.full(A, np.nan)
        self.facing_unk = np.zeros(A, bool)     # 5.2 (M14): facing unknown (silhouette slices disagree, no marker)
        self.read_mk = [None] * A               # 5.2 (M14, addendum 2): the frame's marker reading
        self.front_src = None                   # 5.2 (M14, addendum 2): "markers" (a marker front) / None
        self.long_yaw = np.full(A, np.nan)
        self.long = np.full(A, np.nan)
        self.short = np.full(A, np.nan)
        self.n_clusters = np.zeros(A, int)
        self.dup = np.zeros(A, bool)
        self.space = [None] * A
        self.parts = set()
        self.P0 = None          # points at the first frame where present
        self.first = None

    def at(self, ai):
        return self.center[ai] if self.present[ai] else None


class Model:
    def __init__(self, dump_dir, cats, params):
        dump_dir = Path(dump_dir)
        self.doc = json.loads((dump_dir / "dump.json").read_text())
        z = np.load(dump_dir / "dump.npz")
        self.z = {k: z[k] for k in z.files}
        self.cats, self.params = cats, params
        m = self.doc["meta"]
        self.meta = m
        self.fps = float(m["fps"])
        self.f0, self.f1 = int(m["frame_start"]), int(m["frame_end"])
        self.n_frames = self.f1 - self.f0 + 1
        self.duration = self.n_frames / self.fps
        self.parts = self.doc["parts"]
        self.N = len(self.parts)
        self.an_frame = self.z["an_frame"]
        self.an_t = (self.an_frame - self.f0) / self.fps
        self.A = len(self.an_frame)
        self.obj_M = self.z["obj_M"].astype(np.float64)
        self.obj_vis = self.z["obj_vis"]
        self._labels()
        self._wcache = {}
        self._cameras()
        self._grids()

    # ------------------------------------------------------------------------------------------------ labels
    def _labels(self):
        self.label = {}
        for p in self.parts:
            i = p["index"]
            self.label[(i, 0)] = (p["class"], p["id"])
            for k, ent in enumerate(p.get("face_table") or []):
                self.label[(i, k + 1)] = (ent[0], ent[1])

    def piece_label(self, part, tag):
        return self.label.get((int(part), int(tag)), self.label.get((int(part), 0), (None, None)))

    def t_to_ai(self, t):
        return int(np.argmin(np.abs(self.an_t - t)))

    def t_to_frame_index(self, t):
        return int(min(self.n_frames - 1, max(0, round(t * self.fps))))

    def world(self, ai):
        """(points[P,3], part[P], tag[P], kind[P]) for analysis frame ai (rigid + deforming parts)."""
        if ai in self._wcache:
            return self._wcache[ai]
        z = self.z
        pl, pp = z["pts_local"].astype(np.float64), z["pts_part"]
        if len(pl):
            M = self.obj_M[ai][pp]
            W = np.einsum("pij,pj->pi", M[:, :3, :3], pl) + M[:, :3, 3]
        else:
            W = np.zeros((0, 3))
        parts, tags, kinds = pp, z["pts_tag"], z["pts_kind"]
        if "def_pts" in z:
            W = np.concatenate([W, z["def_pts"][ai].astype(np.float64)])
            parts = np.concatenate([parts, z["def_part"]])
            tags = np.concatenate([tags, z["def_tag"]])
            kinds = np.concatenate([kinds, z["def_kind"]])
        if "defv_pts" in z:                               # r4 R4-5 (PREREG_51r4_addendum3.json): variable-topology parts
            o0, o1 = int(z["defv_off"][ai]), int(z["defv_off"][ai + 1])
            W = np.concatenate([W, z["defv_pts"][o0:o1].astype(np.float64)])
            parts = np.concatenate([parts, z["defv_part"][o0:o1]])
            tags = np.concatenate([tags, z["defv_tag"][o0:o1]])
            kinds = np.concatenate([kinds, z["defv_kind"][o0:o1]])
        out = (W, parts.astype(np.int64), tags.astype(np.int64), kinds.astype(np.int64))
        if len(self._wcache) > 64:
            self._wcache.clear()
        self._wcache[ai] = out
        return out

    # ----------------------------------------------------------------------------------------------- pieces
    def pieces(self, ai):
        """Labelled pieces (part, face tag) visible at analysis frame ai."""
        W, part, tag, kind = self.world(ai)
        vis = self.obj_vis[ai]
        out = []
        if not len(W):
            return out
        sel = vis[part]
        keyarr = part * 4096 + tag
        for key in np.unique(keyarr[sel]):
            m = sel & (keyarr == key)
            pi, tg = int(key // 4096), int(key % 4096)
            mk = m & (kind > 0)
            P = W[mk] if mk.sum() >= 4 else W[m]
            if not len(P):
                continue
            row = self.parts[pi]
            cls, eid = self.piece_label(pi, tg)
            if tg > 0:
                holder = "faces:%s#%d" % (row["name"], tg)
            elif row["holder"]:
                holder = row["holder"]
            else:
                holder = "untagged:" + (("obj:" + row["parent"]) if row["parent"] else
                                        ("col:" + row["collections"][0]) if row["collections"] else
                                        ("obj:" + row["name"]))
            out.append({"part": pi, "tag": tg, "cls": cls, "id": eid, "space": row["space"], "holder": holder,
                        "P": P, "lo": P.min(0), "hi": P.max(0), "name": row["name"]})
        return out

    def layout(self, gt_ids):
        """Furnishing instances (non-entity geometry) at the first frame where each part is visible."""
        P = self.params
        first = {}
        for ai in range(self.A):
            for i in np.where(self.obj_vis[ai])[0]:
                first.setdefault(int(i), ai)
        frames = sorted(set(first.values()))
        pcs = []
        for ai in frames:
            for pc in self.pieces(ai):
                if first.get(pc["part"]) == ai:
                    pcs.append(pc)
        self.layout_pieces = pcs
        free = [p for p in pcs if not p["id"] or p["id"] not in gt_ids]
        uf = UF(len(free))
        tol = P["touch_gap_m"]
        for a in range(len(free)):
            for b in range(a + 1, len(free)):
                pa, pb = free[a], free[b]
                if pa["cls"] != pb["cls"] or pa["holder"] != pb["holder"]:
                    continue
                if pa["id"] != pb["id"]:
                    continue
                if geom.aabb_gap(pa["lo"], pa["hi"], pb["lo"], pb["hi"]) <= tol:
                    uf.union(a, b)
        # non-entity ids group everything carrying them
        byid = {}
        for a, p in enumerate(free):
            if p["id"]:
                byid.setdefault(p["id"], []).append(a)
        for lst in byid.values():
            for b in lst[1:]:
                uf.union(lst[0], b)
        changed = True
        passes = 0
        while changed and passes < 6:
            changed, passes = False, passes + 1
            comps = {}
            for a in range(len(free)):
                comps.setdefault(uf.find(a), []).append(a)
            roots = list(comps)
            info = {}
            for r in roots:
                Q = np.concatenate([free[a]["P"] for a in comps[r]])
                info[r] = (Q, Q.min(0), Q.max(0), hull_area(Q[:, :2]), free[comps[r][0]]["cls"])
            for i in range(len(roots)):
                for j in range(i + 1, len(roots)):
                    ra, rb = roots[i], roots[j]
                    A_, B_ = info[ra], info[rb]
                    if A_[4] != B_[4] or uf.find(ra) == uf.find(rb):
                        continue
                    if geom.aabb_gap(A_[1], A_[2], B_[1], B_[2]) > tol:
                        continue
                    big, small = (A_, B_) if A_[3] >= B_[3] else (B_, A_)
                    if fixes.on("D4"):
                        # 5.0.12 (D4): footprint containment merges only pieces that also share height (stacked
                        # same-class pieces stay separate) unless one of them is a thin sheet
                        ha, hb = A_[2][2] - A_[1][2], B_[2][2] - B_[1][2]
                        zo = min(A_[2][2], B_[2][2]) - max(A_[1][2], B_[1][2])
                        if min(ha, hb) >= P["merge_thin_m"] and zo < P["merge_z_overlap_frac"] * min(ha, hb):
                            continue
                    if contained_frac(small[0][:, :2], big[0][:, :2], P["contain_margin_m"]) >= P["contain_frac"]:
                        uf.union(ra, rb)
                        changed = True
        comps = {}
        for a in range(len(free)):
            comps.setdefault(uf.find(a), []).append(a)
        inst = []
        for n, (r, lst) in enumerate(sorted(comps.items())):
            pcs_ = [free[a] for a in lst]
            Q = np.concatenate([p["P"] for p in pcs_])
            cls = pcs_[0]["cls"]
            spaces = [p["space"] for p in pcs_ if p["space"]]
            if fixes.on("M16"):     # 5.2 (M16): a tied vote goes to the smallest space id, not to the string-hash order
                sp = max(sorted(set(spaces)), key=spaces.count) if spaces else None
            else:
                sp = max(set(spaces), key=spaces.count) if spaces else None
            inst.append(Instance("m%03d" % n, cls, self.cats, pcs_, Q, sp, untagged=cls is None))
        self.instances = inst
        return inst

    # ------------------------------------------------------------------------------------------------ bodies
    def bodies(self, gt_ids):
        """Named bodies for every id carried by the model (the GT entity ids first)."""
        P = self.params
        out = {}
        ids = set()
        for (pi, tg), (cls, eid) in self.label.items():
            if eid:
                ids.add(eid)
        for eid in sorted(ids):
            cls_votes = [cls for (pi, tg), (cls, e) in self.label.items() if e == eid and cls]
            if fixes.on("M16"):     # 5.2 (M16): a tied vote goes to the smallest class name, not to the string-hash order
                cls = max(sorted(set(cls_votes)), key=cls_votes.count) if cls_votes else None
            else:
                cls = max(set(cls_votes), key=cls_votes.count) if cls_votes else None
            out[eid] = Body(eid, cls, group_of(cls, self.cats), self.A)
        if not out:
            self.body = out
            return out
        _, part0, tag0, _ = self.world(0)
        keys0 = part0 * 4096 + tag0
        eids = np.full(len(part0), "", dtype=object)
        for k in np.unique(keys0):
            eids[keys0 == k] = self.piece_label(k // 4096, k % 4096)[1] or ""
        self._eids0 = eids
        self._eids_v = {}                                  # r4.1 R4-6 (PREREG_51r4_addendum4.json): per-frame labels (defv dumps)
        for ai in range(self.A):
            W, part, tag, kind = self.world(ai)
            if not len(W):
                continue
            for eid, b in out.items():
                r = self._body_pick(eid, ai)
                if r is None:
                    continue
                pick, n_cl, dup, pparts = r
                b.parts.update(pparts)
                if n_cl > 1:
                    b.dup[ai] = dup
                b.n_clusters[ai] = n_cl
                self._body_frame(b, ai, pick)
        if fixes.on("M14"):
            for b in out.values():
                if b.group in BODY_GROUPS:
                    self._body_front(b)
        self.body = out
        return out

    def _body_front(self, b):
        """5.2 (M14, addendum 2): a body's facing from its own rigid rotation. The reference is the body's largest
        part (dumped points) when it is rigid; one front vector in its frame is read from the frames whose markers
        read with local_front_52 (the silhouette source of the addendum failed its registered check (2) and is not
        used); at every frame where the reference is in the body's picked points and that front's horizontal part is
        at least half of it (within 60 deg of horizontal), the facing is the front's yaw (known). A body whose front
        cannot be read (no rigid reference, no marker reading, readings that do not agree) has no readable front: every
        frame reads as no front (facing NaN, known: the 5.1 no-front path), not unscored; on a body with a front, a
        frame it leaves near vertical keeps its own reading."""
        if not hasattr(self, "_npts"):
            n = np.bincount(self.z["pts_part"], minlength=self.N).astype(int)
            if "def_part" in self.z:
                n = n + np.bincount(self.z["def_part"], minlength=self.N).astype(int)
            self._npts = n
        ref = max(sorted(b.parts), key=lambda i: self._npts[i]) if b.parts else None
        L = None
        if ref is not None and not self.parts[ref].get("deforming"):
            use = [ai for ai in range(self.A) if b.present[ai] and b.read_mk[ai] is not None]
            if use:
                L = local_front_52([self.obj_M[ai, ref, :3, :3] for ai in use], [b.read_mk[ai] for ai in use])
        if L is None:
            b.facing[:] = np.nan                # no readable front (strict): the 5.1 no-front path on every frame
            b.facing_unk[:] = False
            return
        b.front_src = "markers"
        pl = self.z["pts_local"][self.z["pts_part"] == ref].astype(np.float64)
        c0 = pl.mean(0) if len(pl) else np.zeros(3)
        g = self.params["body_cluster_gap_m"]
        for ai in range(self.A):
            if not b.present[ai]:
                continue
            Mr = self.obj_M[ai, ref]
            c = Mr[:3, :3] @ c0 + Mr[:3, 3]
            if not (np.all(c >= b.lo[ai] - g) and np.all(c <= b.hi[ai] + g)):
                continue                        # the reference is not in the picked body at this frame
            w = Mr[:3, :3] @ L
            if math.hypot(w[0], w[1]) >= 0.5 * (np.linalg.norm(w) + 1e-12):
                b.facing[ai] = math.degrees(math.atan2(w[1], w[0])) % 360.0
                b.facing_unk[ai] = False

    def _eids_at(self, ai, part, tag):
        """r4.1 R4-6 (PREREG_51r4_addendum4.json): the entity id of every world(ai) point. A dump with R4-5
        variable-topology rows (defv_*) has its own rows per frame: labelled from frame ai's part / tag exactly as
        bodies() labels frame 0. Any other dump keeps frame 0's labels (5.0.12, the same object)."""
        if "defv_pts" not in self.z:
            return self._eids0
        if ai not in self._eids_v:
            keys = part * 4096 + tag
            eids = np.full(len(part), "", dtype=object)
            for k in np.unique(keys):
                eids[keys == k] = self.piece_label(k // 4096, k % 4096)[1] or ""
            self._eids_v[ai] = eids
        return self._eids_v[ai]

    def _body_pick(self, eid, ai):
        """(points of the picked cluster, n clusters, duplicate flag, parts) of entity eid at analysis frame ai."""
        P = self.params
        W, part, tag, kind = self.world(ai)
        if not len(W):
            return None
        m = (self._eids_at(ai, part, tag) == eid) & self.obj_vis[ai][part]   # r4.1 R4-6 (PREREG_51r4_addendum4.json)
        if not m.any():
            return None
        keys = np.unique(part[m] * 4096 + tag[m])
        groups, pparts = [], set()
        for k in keys:
            mm = m & (part * 4096 + tag == k)
            sel = mm & (kind > 0)
            Q = W[sel] if sel.sum() >= 4 else W[mm]
            groups.append(Q)
            pparts.add(int(k // 4096))
        uf = UF(len(groups))
        los = [g.min(0) for g in groups]
        his = [g.max(0) for g in groups]
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if geom.aabb_gap(los[i], his[i], los[j], his[j]) <= P["body_cluster_gap_m"]:
                    uf.union(i, j)
        cl = {}
        for i in range(len(groups)):
            cl.setdefault(uf.find(i), []).append(i)
        clusters = [np.concatenate([groups[i] for i in v]) for v in cl.values()]
        clusters.sort(key=len, reverse=True)
        pick = clusters[0]
        dup = False
        if len(clusters) > 1:
            infr = [self.in_frustum_at(ai, c.mean(0)) for c in clusters]
            if any(infr) and not infr[0]:
                pick = clusters[infr.index(True)]
            d0 = float(np.linalg.norm(clusters[0].max(0) - clusters[0].min(0)))
            big = [c for c in clusters[1:] if np.linalg.norm(c.max(0) - c.min(0)) >= 0.3 * d0
                   and len(c) >= 0.2 * len(clusters[0])]
            dup = bool(big)
        return pick, len(clusters), dup, pparts

    def _body_frame(self, b, ai, Q):
        b.present[ai] = True
        b.lo[ai], b.hi[ai] = Q.min(0), Q.max(0)
        b.base[ai], b.top[ai] = Q[:, 2].min(), Q[:, 2].max()
        rect = geom.min_area_rect(Q[:, :2])
        b.long_yaw[ai], b.long[ai], b.short[ai] = rect[1], rect[2], rect[3]
        if b.group in BODY_GROUPS:
            b.center[ai] = body_center(Q)
            if fixes.on("M14"):     # 5.2 (M14): explicit front markers first, a disagreeing silhouette is unknown
                fr = self._front_marks(b.id, ai, Q)
                f, known = person_facing_52(Q, fr)
                b.facing_unk[ai] = not known
                b.read_mk[ai] = marker_facing_52(Q, fr)                 # addendum 2: the body front's readings
            else:
                f, _ = person_facing(Q)
            b.facing[ai] = np.nan if f is None else f
        else:
            b.center[ai] = rect[0]
            if b.cls in SEAT_CLASSES:
                f = seat_facing(Q, rect)
                b.facing[ai] = np.nan if f is None else f
        if b.P0 is None:
            b.P0, b.first = Q, ai

    def _front_marks(self, eid, ai, Q):
        """5.2 (M14): xy front offsets of entity eid's front-marker parts (FRONT_WORDS in the part name) at analysis
        frame ai, those inside the picked cluster's box: the marker's centroid minus the centroid of the part it sits
        on, so the offset follows that part and not the body's vertical axis (body_center), which is not under the
        head of a lying / rolling / tumbling body. Addendum 2: the part it sits on is a larger part of the body (box
        diagonal): the smallest larger part whose box holds the marker's centroid (the head round a nose, the torso
        round a chest plate), else the larger part with the nearest point; a front-word part with no larger part (the
        body's largest, e.g. a 'zip-front jacket' torso) is a body segment, not a marker"""
        if not hasattr(self, "_front_pi"):
            self._front_pi = np.array([i for i, p in enumerate(self.parts) if FRONT_WORDS.search(p.get("name") or "")],
                                      int)
        if not len(self._front_pi):
            return []
        W, part, tag, kind = self.world(ai)
        lo, hi = Q.min(0), Q.max(0)
        own = (self._eids_at(ai, part, tag) == eid) & self.obj_vis[ai][part] & np.all(W >= lo, 1) & np.all(W <= hi, 1)
        if not own.any():
            return []
        Wo, po = W[own], part[own]
        box = {int(q): (Wo[po == q].min(0), Wo[po == q].max(0)) for q in np.unique(po)}
        diag = {q: float(np.linalg.norm(bx[1] - bx[0])) for q, bx in box.items()}
        fset = set(self._front_pi.tolist())
        out = []
        for pi in sorted(q for q in box if q in fset):
            c = Wo[po == pi].mean(0)
            bigger = [q for q in box if q != pi and diag[q] > diag[pi]]
            if not bigger:
                continue
            held = [(diag[q], q) for q in bigger if np.all(c >= box[q][0]) and np.all(c <= box[q][1])]
            if held:
                q = min(held)[1]
            else:
                sel = np.isin(po, bigger)
                q = int(po[sel][int(np.argmin(np.linalg.norm(Wo[sel] - c, axis=1)))])
            out.append((c - Wo[po == q].mean(0))[:2])
        return out

    def body_instance(self, eid, ai=None):
        """Static Instance view of a named body at its first present frame (for named GT objects); 5.0.12 (S19):
        at analysis frame ai when given (None if the body is not present there)."""
        b = self.body.get(eid)
        if b is None or b.P0 is None:
            return None
        if ai is None:
            ai, Q = b.first, b.P0
        else:
            if not b.present[ai]:
                return None
            r = self._body_pick(eid, ai)
            if r is None:
                return None
            Q = r[0]
        sp = None
        for pc in self.parts:
            if pc["id"] == eid and pc["space"]:
                sp = pc["space"]
                break
        inst = Instance("id:" + eid, b.cls, self.cats, [{"part": p} for p in sorted(b.parts)], Q, sp)
        if b.group in BODY_GROUPS:
            inst.center2 = b.center[ai]
            if fixes.on("M14"):     # 5.2 (M14): the body's own facing read at that frame (markers / unknown)
                inst.facing = None if np.isnan(b.facing[ai]) else float(b.facing[ai])
                inst.facing_unknown = bool(b.facing_unk[ai])
        inst.t_snap = float(self.an_t[ai])
        return inst

    # ----------------------------------------------------------------------------------------------- cameras
    def _cameras(self):
        z = self.z
        self.cam_index = z["cam_index"]
        M = z["cam_M"].astype(np.float64)
        C = z["cam_corners"].astype(np.float64)
        self.cam_pos = M[:, :3, 3]
        self.cam_fwd = -M[:, :3, 2]
        self.cam_up = M[:, :3, 1]
        self.cam_R = M[:, :3, :3]
        self.cam_ok = self.cam_index >= 0
        xw = C[:, :, 0].max(1) - C[:, :, 0].min(1)
        yw = C[:, :, 1].max(1) - C[:, :, 1].min(1)
        zc = np.abs(C[:, 0, 2])
        ortho = z["cam_ortho"]
        with np.errstate(divide="ignore", invalid="ignore"):
            self.hfov = np.where(ortho | (zc <= 0), np.nan, np.degrees(2 * np.arctan(xw / 2 / zc)))
            self.vfov = np.where(ortho | (zc <= 0), np.nan, np.degrees(2 * np.arctan(yw / 2 / zc)))
        self.cam_corners = C
        self.segment()

    def segment(self):
        P = self.params
        n = self.n_frames
        cuts = []
        steps = np.linalg.norm(np.diff(self.cam_pos, axis=0), axis=1) if n > 1 else np.zeros(0)
        rots = []
        for k in range(1, n):
            R = self.cam_R[k] @ self.cam_R[k - 1].T
            rots.append(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))))
        rots = np.array(rots)
        reasons = {}
        w = P["cut_local_win"]

        def spike(x, i, floor):
            # a cut is an isolated step: above the absolute floor and well above the steps of its neighbouring
            # frames (no global-median floor: one camera that only ever jumps has a median step = the jump size)
            if x[i] <= floor:
                return False
            nb = np.concatenate([x[max(0, i - w):i], x[i + 1:i + 1 + w]])
            return not len(nb) or x[i] > P["cut_local_ratio"] * float(np.median(nb))
        for k in range(1, n):
            why = None
            if self.cam_index[k] != self.cam_index[k - 1]:
                why = "camera"
            elif spike(steps, k - 1, P["cut_jump_m"]):
                why = "jump"
            elif spike(rots, k - 1, P["cut_rot_deg"]):
                why = "turn"
            elif (np.isfinite(self.hfov[k]) and np.isfinite(self.hfov[k - 1]) and
                  max(self.hfov[k], self.hfov[k - 1]) / max(1e-6, min(self.hfov[k], self.hfov[k - 1]))
                  > P["cut_fov_ratio"]):
                why = "fov"
            if why:
                cuts.append(k)
                reasons[k] = why
        starts = [0] + cuts
        ends = [c - 1 for c in cuts] + [n - 1]
        self.shots = [{"k0": a, "k1": b, "t0": a / self.fps, "t1": (b + 1) / self.fps, "why": reasons.get(a)}
                      for a, b in zip(starts, ends)]
        self.cut_times = [k / self.fps for k in cuts]
        self.cut_reasons = [reasons[k] for k in cuts]

    def cam_at(self, t):
        k = self.t_to_frame_index(t)
        return k

    def shot_of_frame(self, k):
        for i, s in enumerate(self.shots):
            if s["k0"] <= k <= s["k1"]:
                return i
        return len(self.shots) - 1

    def project(self, k, X):
        """Normalised image coords (u, v in [0,1], v down) and depth of world points X for frame index k."""
        X = np.atleast_2d(np.asarray(X, float))
        R, T = self.cam_R[k], self.cam_pos[k]
        Pc = (X - T) @ R
        C = self.cam_corners[k]
        x0, x1 = C[:, 0].min(), C[:, 0].max()
        y0, y1 = C[:, 1].min(), C[:, 1].max()
        zc = abs(C[0, 2]) if abs(C[0, 2]) > 1e-9 else 1.0
        depth = -Pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = Pc[:, 0] / np.maximum(depth, 1e-9) * zc
            ys = Pc[:, 1] / np.maximum(depth, 1e-9) * zc
        u = (xs - x0) / (x1 - x0)
        v = (y1 - ys) / (y1 - y0)
        return u, v, depth

    def in_frustum_at(self, ai, X):
        f = int(self.an_frame[ai]) - self.f0
        if f < 0 or f >= self.n_frames or not self.cam_ok[f]:
            return False
        u, v, d = self.project(f, X)
        return bool(d[0] > 0 and 0 <= u[0] <= 1 and 0 <= v[0] <= 1)

    # ------------------------------------------------------------------------------------------------- grids
    def _grids(self):
        z = self.z
        if "g_frame" in z:
            self.g_frame = z["g_frame"]
            self.g_t = (self.g_frame - self.f0) / self.fps
        else:
            self.g_frame = np.zeros(0, int)
            self.g_t = np.zeros(0)

    def grid(self, gi):
        z = self.z
        return {"part": z["g_part"][gi], "tag": z["g_tag"][gi], "depth": z["g_depth"][gi].astype(np.float64),
                "normal": z["g_normal"][gi].astype(np.float64), "corners": z["g_corners"][gi],
                "ortho": bool(z["g_ortho"][gi]), "frame": int(self.g_frame[gi])}

    def grid_near(self, t, shot=None):
        """Index of the grid frame nearest to time t, restricted to a model shot when given."""
        if not len(self.g_t):
            return None
        cand = np.arange(len(self.g_t))
        if shot is not None:
            s = self.shots[shot]
            ks = self.g_frame - self.f0
            m = (ks >= s["k0"]) & (ks <= s["k1"])
            if m.any():
                cand = cand[m]
        return int(cand[np.argmin(np.abs(self.g_t[cand] - t))])

    # -------------------------------------------------------------------------------------------- reporting
    def tag_report(self, gt_ids, gt_spaces):
        rows = self.parts
        by_src = {}
        area_tot, area_tag = 0.0, 0.0
        unknown, bad_ids, ambiguous = set(), set(), []
        for r in rows:
            src = r["source"] or "none"
            k = src.split(":")[0] if src != "own" else "own"
            if src.endswith("(ambiguous)"):
                ambiguous.append(r["name"])
            by_src[k] = by_src.get(k, 0) + 1
            a = float(r.get("area") or 0.0)
            area_tot += a
            ta = r.get("tag_area") or {}
            inherit = float(ta.get("0", a if not ta else 0.0))
            face_tagged = sum(v for kk, v in ta.items() if kk != "0")
            area_tag += face_tagged + (inherit if r["class"] else 0.0)
            for cls in [r["class"]] + [e[0] for e in (r.get("face_table") or [])]:
                if cls and group_of(cls, self.cats) == "unknown":
                    unknown.add(cls)
            for eid in [r["id"]] + [e[1] for e in (r.get("face_table") or [])]:
                if eid and eid not in gt_ids:
                    bad_ids.add(eid)
        present = set(e for (_, _), (_, e) in self.label.items() if e)
        spaces_used = sorted({r["space"] for r in rows if r["space"]})
        return {"n_parts": len(rows), "by_source": by_src,
                "tagged_area_frac": round(area_tag / area_tot, 4) if area_tot > 0 else None,
                "unknown_classes": sorted(unknown), "ids_not_in_entities": sorted(bad_ids),
                "entities_missing": sorted(set(gt_ids) - present), "ambiguous_collection_tags": ambiguous[:20],
                "spaces_used": spaces_used, "spaces_unknown": sorted(set(spaces_used) - set(gt_spaces)),
                "face_tagged_parts": sum(1 for r in rows if r.get("face_table")),
                "deforming_parts": sum(1 for r in rows if r.get("deforming")),
                "warnings": self.doc.get("warnings", []), "handlers": self.doc.get("handlers")}
