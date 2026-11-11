"""GT loading, validation and normalisation (schema pv_gt_v5.0, see DESIGN.md section 2)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import fixes, geom

SCHEMA = "pv_gt_v5.0"
MODALITIES = {"text", "image", "video"}
FRONTS = {"+x": 0.0, "+y": 90.0, "-x": 180.0, "-y": 270.0}
STRUCT_GROUPS = {"structure"}
TERRAIN_GROUPS = {"terrain"}
OPENING_CLASSES = {"door", "doorway", "window", "stairs", "ramp", "archway", "gate", "elevator", "escalator",
                   "tunnel", "bridge"}
LINEAR_CLASSES = {"road", "sidewalk", "path", "track", "railing", "fence", "bridge", "tunnel", "crosswalk", "curb"}
SEG_DIRS = {"left_to_right", "right_to_left", "static", "away", "toward"}   # 5.0.10 screen_dir_segments


class GTError(ValueError):
    pass


def load_categories(path):
    t = json.loads(Path(path).read_text())
    return {c["class"]: c for c in t["classes"]}


def group_of(cls, cats):
    if cls is None:
        return "untagged"
    if cls in cats:
        return cats[cls]["group"]
    if isinstance(cls, str) and cls.startswith("other:"):
        return "other"
    return "unknown"


class Space:
    def __init__(self, d):
        self.id = d["id"]
        self.name = d.get("name", self.id)
        self.kind = d.get("kind", "indoor")
        self.floor_z = float(d.get("floor_z", 0.0))
        # 5.0.2: a sloped floor (text items) has no single level; floor_z is its mean and no height is compared there
        self.sloped = bool(d.get("sloped", False))
        self.outline = np.asarray(d["outline"], float) if d.get("outline") else None
        self.outline_conf = float(d.get("outline_conf", 1.0))
        self.height_m = d.get("height_m")
        self.extras = d.get("extras", "observed")
        self.raw = d
        self.extent = None   # (lo[2], hi[2]) filled by GT.finish()

    @property
    def L(self):
        lo, hi = self.extent
        return float(max(1.0, np.linalg.norm(np.asarray(hi) - np.asarray(lo))))

    def area(self):
        if self.outline is not None and len(self.outline) >= 3:
            p = self.outline
            return float(abs(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])))
        lo, hi = self.extent
        return float((hi[0] - lo[0]) * (hi[1] - lo[1]))


class Obj:
    def __init__(self, d, idx, cats):
        self.key = d.get("key") or d.get("id") or "obj%03d" % idx
        self.id = d.get("id")
        self.cls = d["class"]
        self.group = group_of(self.cls, cats)
        self.space = d.get("space")
        self.space_given = self.space is not None   # 5.0.12 (S7 / S14): GT.__init__ fills a missing space
        self.center = np.asarray(d["center"], float)
        self.size = np.asarray(d["size"], float)
        self.yaw = float(d.get("yaw_deg", 0.0))
        self.front = d.get("front")
        sx, sy = self.size[0], self.size[1]
        if "symmetry" in d:
            self.symmetry = int(d["symmetry"])
        elif self.front in FRONTS:
            self.symmetry = 1
        elif max(sx, sy) / max(1e-6, min(sx, sy)) < 1.15:
            self.symmetry = 4
        else:
            self.symmetry = 2
        self.support = d.get("support", "floor")
        self.named = bool(d.get("named", self.id is not None))
        self.conf = float(d.get("conf", 1.0))
        self.unc = float(d.get("unc_m", 0.0))
        self.yaw_unc = float(d.get("yaw_unc_deg", 0.0))
        # 5.0.5: a box seen from one side only (video lift) has no reliable size; its size is not compared
        self.size_known = bool(d.get("size_known", True))
        self.raw = d

    @property
    def front_yaw(self):
        """World yaw of the object's front (None if not given)."""
        if self.front not in FRONTS:
            return None
        return (self.yaw + FRONTS[self.front]) % 360.0

    @property
    def long_yaw(self):
        return (self.yaw if self.size[0] >= self.size[1] else self.yaw + 90.0) % 180.0

    @property
    def long(self):
        return float(max(self.size[0], self.size[1]))

    @property
    def short(self):
        return float(min(self.size[0], self.size[1]))

    def corners2(self):
        return geom.rect_corners(self.center[:2], self.yaw, self.size[0], self.size[1])

    @property
    def bottom(self):
        return float(self.center[2] - self.size[2] / 2)

    @property
    def top(self):
        return float(self.center[2] + self.size[2] / 2)


class Track:
    GAP_S = 6.0  # 5.0.3: a longer gap between two samples that moves the entity leaves its path in between unknown

    def __init__(self, d):
        self.id = d["id"]
        s = sorted(d.get("samples", []), key=lambda x: x["t"])
        self.t = np.array([x["t"] for x in s], float)
        self.p = np.array([x["p"][:2] for x in s], float) if s else np.zeros((0, 2))
        self.z = np.array([x.get("z", x["p"][2] if len(x["p"]) > 2 else 0.0) for x in s], float)
        self.yaw = [x.get("yaw") for x in s]
        self.posture = [x.get("posture") for x in s]
        self.space = [x.get("space") for x in s]
        self.visible = [x.get("visible", True) for x in s]
        self.unc = float(d.get("unc_m", 0.0))
        self.route = d.get("route") or {}
        self.raw = d
        # 5.0.3: spans where the path is known. Text tracks hold only the positions a script states, so a long
        # moving gap may hide a stop or a detour; a long gap at one place (the entity stays put) is kept.
        self.segs, a = [], None
        for i in range(len(self.t)):
            if a is None:
                a = float(self.t[i])
            elif self.t[i] - self.t[i - 1] > self.GAP_S and np.linalg.norm(self.p[i] - self.p[i - 1]) > 0.5:
                self.segs.append((a, float(self.t[i - 1])))
                a = float(self.t[i])
        if a is not None:
            self.segs.append((a, float(self.t[-1])))

    def known(self, t, tol=0.5):
        """True when the GT position at t is known: inside a known span or within tol s of one (5.0.3)."""
        return any(a - tol <= t <= b + tol for a, b in self.segs)

    def at(self, t):
        if len(self.t) == 0:
            return None
        return np.array([np.interp(t, self.t, self.p[:, 0]), np.interp(t, self.t, self.p[:, 1])])

    def yaw_at(self, t, window=None):
        """GT yaw at t interpolated over the stated yaws; 5.0.12 (S22): with window, None unless a stated yaw lies
        within window s of t (a yaw interpolated from distant samples is not a GT fact)."""
        idx = [i for i, y in enumerate(self.yaw) if y is not None]
        if not idx:
            return None
        if window is not None and min(abs(float(self.t[i]) - t) for i in idx) > window:
            return None
        ts = self.t[idx]
        ys = np.unwrap(np.radians([self.yaw[i] for i in idx]))
        return math.degrees(float(np.interp(t, ts, ys))) % 360.0

    def label_at(self, t, field):
        vals = getattr(self, field)
        if not len(self.t):
            return None
        i = int(np.searchsorted(self.t, t, side="right") - 1)
        return vals[max(0, i)]


class Shot:
    def __init__(self, d, idx):
        self.id = d.get("id", "K%02d" % (idx + 1))
        self.t0, self.t1 = float(d["t"][0]), float(d["t"][1])
        self.space = d.get("space")
        self.transition_in = d.get("transition_in")
        self.subjects = list(d.get("subjects", []))
        self.camera = sorted(d.get("camera", []), key=lambda c: c.get("t", 0.0))
        self.spec = d.get("spec") or {}
        self.move = d.get("move")
        self.keep_in_frame = list(d.get("keep_in_frame", []))
        self.screen_dir = d.get("screen_dir") or {}
        # 5.0.10: a direction or camera move that changes inside the shot, as time segments (GT r1)
        self.screen_dir_segments = {k: sorted(v, key=lambda s: s["t"][0])
                                    for k, v in (d.get("screen_dir_segments") or {}).items() if v}
        self.move_segments = sorted(d.get("move_segments") or [], key=lambda s: s["t"][0])
        self.p1 = list(d.get("p1", []))
        self.view = d.get("view")
        self.raw = d

    @property
    def dur(self):
        return self.t1 - self.t0


def cam_pose(c):
    """(position[3], forward[3], up[3], hfov_deg, vfov_deg) of a GT camera sample."""
    if "c2w_blender" in c:
        M = np.asarray(c["c2w_blender"], float)
        pos, fwd, up = M[:3, 3], -M[:3, 2], M[:3, 1]
        K, W, H = c.get("K"), c.get("width"), c.get("height")
        hf = math.degrees(2 * math.atan(W / 2 / K["fx"])) if K and W else c.get("hfov_deg")
        vf = math.degrees(2 * math.atan(H / 2 / K["fy"])) if K and H else c.get("vfov_deg")
        return pos, fwd / np.linalg.norm(fwd), up, hf, vf
    # 5.0.2: a text sample may leave parts open (null, e.g. the end direction of a pan); they become NaN
    pos = np.asarray([None] * 3 if c.get("pos") is None else c["pos"], float)
    look = np.asarray([None] * 3 if c.get("look") is None else c["look"], float)
    fwd = look - pos
    fwd /= max(1e-9, np.linalg.norm(fwd))
    up = np.array([0, 0, 1.0])
    return pos, fwd, up, c.get("hfov_deg"), c.get("vfov_deg")


class GT:
    def __init__(self, path_or_dict, cats):
        if isinstance(path_or_dict, (str, Path)):
            self.path = Path(path_or_dict)
            d = json.loads(self.path.read_text())
        else:
            self.path, d = None, path_or_dict
        self.d = d
        self.cats = cats
        self.problems = []
        if d.get("schema") != SCHEMA:
            self.problems.append("schema is %r, expected %r" % (d.get("schema"), SCHEMA))
        self.item_id = d.get("item_id")
        self.modality = d.get("modality")
        if self.modality not in MODALITIES:
            raise GTError("modality must be one of %s" % sorted(MODALITIES))
        self.image_kind = d.get("image_kind")
        self.duration = float(d.get("duration_s", 0.0))
        # 5.0.5: "separate" (video items): each space has its own measured frame, not placed relative to the others;
        # no global alignment fallback and no comparison of space sizes or floor levels across spaces
        self.space_frames = d.get("space_frames", "shared")
        self.spaces = [Space(s) for s in d.get("spaces", [])] or [Space({"id": "S01"})]
        self.space_by = {s.id: s for s in self.spaces}
        self.connections = list(d.get("connections", []))
        self.entities = {e["id"]: e for e in d.get("entities", [])}
        self.objects = [Obj(o, i, cats) for i, o in enumerate(d.get("objects", []))]
        for o in self.objects:
            if o.space is None:
                o.space = self.spaces[0].id
        self.obj_by = {o.key: o for o in self.objects}
        self.tracks = {t["id"]: Track(t) for t in d.get("tracks", [])}
        self.events = list(d.get("events", []))
        self.relations = list(d.get("relations", []))
        self.zones = list(d.get("zones", []))
        self.space_transitions = list(d.get("space_transitions", []))
        self.shots = [Shot(s, i) for i, s in enumerate(d.get("shots", []))]
        self.p1_grid = d.get("p1_grid", [96, 72])
        self.finish()

    def finish(self):
        for s in self.spaces:
            pts = []
            if s.outline is not None:
                pts.append(s.outline)
            for o in self.objects:
                # 5.0.12 (S7): an object without a stated space (filled with spaces[0]) does not widen that space
                if o.space == s.id and (o.space_given or len(self.spaces) == 1 or not fixes.on("S7")):
                    pts.append(o.corners2())
            for tr in self.tracks.values():
                for i, sp in enumerate(tr.space):
                    if sp == s.id or (sp is None and len(self.spaces) == 1):
                        pts.append(tr.p[i:i + 1])
            if not pts:
                s.extent = (np.array([-2.5, -2.5]), np.array([2.5, 2.5]))
                continue
            P = np.concatenate(pts)
            s.extent = (P.min(0), P.max(0))
        ids = set(self.entities)
        for o in self.objects:
            if o.id and o.id not in ids:
                self.problems.append("object %s has id %s not in entities" % (o.key, o.id))
        for tid in self.tracks:
            if tid not in ids:
                self.problems.append("track %s not in entities" % tid)
        keys = set(self.obj_by) | ids
        for e in self.events:
            for x in list(e.get("agents", [])) + list(e.get("objects", [])):
                if x not in keys and not str(x).startswith("class:"):
                    self.problems.append("event %s references unknown %s" % (e.get("id"), x))
        for o in self.objects:
            if o.group == "unknown":
                self.problems.append("object %s has unknown class %s" % (o.key, o.cls))
        for sh in self.shots:
            for sid, segs in sh.screen_dir_segments.items():
                if sid in sh.screen_dir:
                    self.problems.append("shot %s: %s has both screen_dir and screen_dir_segments" % (sh.id, sid))
                if any(s.get("dir") not in SEG_DIRS for s in segs):
                    self.problems.append("shot %s: unknown screen_dir_segments dir for %s" % (sh.id, sid))
            if sh.move_segments and sh.move:
                self.problems.append("shot %s has both move and move_segments" % sh.id)
            for segs in list(sh.screen_dir_segments.values()) + [sh.move_segments]:
                if any(a["t"][1] > b["t"][0] + 1e-6 for a, b in zip(segs, segs[1:])):
                    self.problems.append("shot %s: overlapping segments" % sh.id)
                if any(not (sh.t0 - 1e-6 <= s["t"][0] < s["t"][1] <= sh.t1 + 1e-6) for s in segs):
                    self.problems.append("shot %s: segment outside the shot" % sh.id)

    # --------------------------------------------------------------------------------------------- helpers
    def space_of_point(self, p):
        best, bd = None, None
        for s in self.spaces:
            if s.outline is not None and geom.point_in_poly(p, s.outline):
                return s.id
            lo, hi = s.extent
            d = float(np.max(np.maximum(lo - p[:2], p[:2] - hi)))
            if bd is None or d < bd:
                best, bd = s.id, d
        return best

    def union_L(self, space_ids):
        ex = [self.space_by[s].extent for s in space_ids if s in self.space_by] or [self.spaces[0].extent]
        lo = np.min([e[0] for e in ex], 0)
        hi = np.max([e[1] for e in ex], 0)
        return float(max(1.0, np.linalg.norm(hi - lo)))

    def objects_in(self, space_id, kinds=("object",)):
        out = []
        for o in self.objects:
            if o.space != space_id:
                continue
            k = self.kind_of(o)
            if k in kinds:
                out.append(o)
        return out

    def kind_of(self, o):
        if o.group in TERRAIN_GROUPS:
            return "terrain"
        if o.group in STRUCT_GROUPS:
            return "opening" if o.cls in OPENING_CLASSES else "structure"
        return "object"

    def named_ids(self):
        return set(self.entities)

    def entity_class(self, eid):
        e = self.entities.get(eid)
        if e:
            return e.get("class")
        o = self.obj_by.get(eid)
        return o.cls if o else None
