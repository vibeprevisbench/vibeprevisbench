"""v5 text-item claims: one judge for the GT, blind-reader scenes and model scenes (pure stdlib, no numpy).

The rules are text_items/drafts_v5/_build/PREREG_v5.json (prereg_v5.0.1, registered 2026-10-18 21:48:32 +0700) and
PREREG_v5.0.2.json (prereg_v5.0.2, the round-2 rules): keys `judge` and `gt_margin` hold every constant used here;
`claim_types`, `camera_checks` and `matching` say in words what this module does. prereg_v5.0.2 adds the key
`mechanisms` (m1-m7, r_touch, r_cover, r_part); every v5.0.2 rule below is gated on it (MECH), so with the v5.0.1 file
loaded this module judges exactly as v5.0.1 did. use_rules_of(claims file) loads the PREREG a claims file was built
under (checked by sha). A vendored copy (scorer/v5_1) can point RULES at its own copy of that file with set_rules(path).

Users: text_items/drafts_v5/_build/build.py (claims on the GT, with the GT margins), text_items/drafts_v5/_build/
blind_compare_v5.py (reader scenes) and tools/text_v5/claims_score.py (model dumps, GT-as-model tests).

Plain scene (a dict, the one format every side is converted to):
  frame        "gt" | "reader" | "model"
  spaces       {sid: {"poly": [[x, y], ...] or None, "bbox": [x0, y0, x1, y1] or None, "floor": z,
                      "ground": [a, b, c] or absent: a space the GT marks sloped; its ground is z = a + b x + c y,
                      fitted by ground_plane() to the bottoms of the scene's own things on it (prereg_v5.0 note n1)}}
  elems        {eid: El}   static things: GT objects without an entity id / reader object rows / model instances
  ents         {id: {"cls": str, "person": bool, "samples": [Smp, ...] (time order), "static": bool}}
  cams         {"<shot index>": [{"t", "pos": [x, y, z] (parts may be None), "look": [x, y, z] or None,
                                  "fwd": [x, y, z] or None, "up": [x, y, z] or None, "hfov": deg or None}, ...]}
  ph           person height of the scene (GT / reader 1.75, model: median body height)
  duration     clip length in s
  known        "spans" (GT: positions only inside stated spans, pvscore5 5.0.3 rule) | "all"
  max_gap_s    None, or the longest sample gap that is interpolated (model: absent across longer gaps)
  space_tags   true: space membership by an element's own space tag when it has one (model)
  axis_fallback true: a facing claim on an element without a front compares the long axis (model)
El  = {"c": [x, y], "long": l, "short": w, "axis": deg or None, "front": deg or None, "h": height or None,
       "z0": bottom or None, "z1": top or None, "space": sid or None, "cls": str, "labels": [str], "person": bool}
Smp = El fields + "t" (+ "yaw_stated": bool on the GT: the facing is a stated value)
"""
import copy
import fnmatch
import hashlib
import json
import math
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
PREREG_PATH = R / "text_items" / "drafts_v5" / "_build" / "PREREG_v5.json"
PREREG_V502_PATH = PREREG_PATH.parent / "PREREG_v5.0.2.json"
PREREG_FILES = {"prereg_v5.0.1": "PREREG_v5.json", "prereg_v5.0.2": "PREREG_v5.0.2.json",
                "prereg_v5.0.3": "PREREG_v5.0.3.json"}
RULES = {}
RULES_SHA = [None]
HOLD_S = 0.5  # pvscore5 5.0.3 known-span rule: a GT position is known within 0.5 s of a sample


def set_rules(path=None):
    b = Path(path or PREREG_PATH).read_bytes()
    p = json.loads(b.decode())
    RULES.clear()
    RULES.update(p)
    RULES_SHA[0] = hashlib.sha256(b).hexdigest()
    return p


def use_rules_of(cf, prereg_dir=None):
    """load the PREREG that the claims file cf was built under (its 'prereg' version, checked against its
    'prereg_sha256'); a no-op when those rules are loaded already"""
    ver, sha = cf.get("prereg"), cf.get("prereg_sha256")
    if RULES.get("version") == ver and (not sha or RULES_SHA[0] == sha):
        return RULES
    fn = PREREG_FILES.get(ver)
    if fn is None:
        raise ValueError("claims file %s names an unknown prereg %r" % (cf.get("item"), ver))
    p = Path(prereg_dir or PREREG_PATH.parent) / fn
    set_rules(p)
    if sha and RULES_SHA[0] != sha:
        raise ValueError("claims file %s was built under %s sha %s, but %s has sha %s" % (
            cf.get("item"), ver, sha[:12], p, RULES_SHA[0][:12]))
    return RULES


set_rules()


def MECH(k):
    """prereg_v5.0.2 mechanism k (m1 ... m7, r_touch, r_cover, r_part; prereg_v5.0.3 adds n1b = g1 and m1c = g2):
    its parameters, or None when the loaded rules do not have it (prereg_v5.0.1; v5.0.2 for n1b / m1c)"""
    m = (RULES.get("mechanisms") or {}).get(k)
    return m if isinstance(m, dict) and m.get("on", True) else None


def J(k):
    return RULES["judge"][k]


def CJ(k):
    return RULES["judge"]["cam"][k]


def GM(k):
    return RULES["gt_margin"][k]


# ------------------------------------------------------------------ geometry
def rect(c, long_, short, axis_deg):
    a = math.radians(axis_deg or 0.0)
    ux, uy, vx, vy = math.cos(a), math.sin(a), -math.sin(a), math.cos(a)
    hx, hy = long_ / 2.0, short / 2.0
    return [(c[0] + sx * hx * ux + sy * hy * vx, c[1] + sx * hx * uy + sy * hy * vy)
            for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]


def octagon(c, r):
    return [(c[0] + r * math.cos(math.pi * k / 4), c[1] + r * math.sin(math.pi * k / 4)) for k in range(8)]


def seg_pt(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    u = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.hypot(a[0] + u * dx - p[0], a[1] + u * dy - p[1])


def inside(p, poly):
    x, y, c = p[0], p[1], False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1:
            c = not c
    return c


def seg_x(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (orient(a, b, c) * orient(a, b, d) < 0) and (orient(c, d, a) * orient(c, d, b) < 0)


def poly_gap(P, Q):
    if any(inside(p, Q) for p in P) or any(inside(q, P) for q in Q):
        return 0.0
    for i in range(len(P)):
        for j in range(len(Q)):
            if seg_x(P[i], P[(i + 1) % len(P)], Q[j], Q[(j + 1) % len(Q)]):
                return 0.0
    d = min(seg_pt(p, Q[j], Q[(j + 1) % len(Q)]) for p in P for j in range(len(Q)))
    return min(d, min(seg_pt(q, P[i], P[(i + 1) % len(P)]) for q in Q for i in range(len(P))))


def grow(poly, c, d):
    """polygon grown by d. prereg_v5.0.1: every vertex pushed out from the centre by d * sqrt(2) (a thin box grows
    less than d across its short side); prereg_v5.0.2 m5: every edge moved out by d (exact offset of a convex
    footprint)"""
    if MECH("m5"):
        return offset_convex(poly, d)
    out = []
    for p in poly:
        vx, vy = p[0] - c[0], p[1] - c[1]
        n = math.hypot(vx, vy)
        out.append(p if n < 1e-12 else (p[0] + vx / n * d * math.sqrt(2), p[1] + vy / n * d * math.sqrt(2)))
    return out


def offset_convex(poly, d):
    """convex polygon with every edge moved outward by d (prereg_v5.0.2 m5)"""
    n = len(poly)
    if n < 3 or d == 0:
        return list(poly)
    A = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))
    sg = 1.0 if A >= 0 else -1.0
    nr = []
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dx, dy = q[0] - p[0], q[1] - p[1]
        L = math.hypot(dx, dy)
        nr.append(None if L < 1e-12 else (sg * dy / L, -sg * dx / L))
    out = []
    for i in range(n):
        n1, n2 = nr[i - 1], nr[i]
        n1 = n1 or n2
        n2 = n2 or n1
        if n1 is None:
            out.append(poly[i])
            continue
        k = 1.0 + n1[0] * n2[0] + n1[1] * n2[1]
        if k < 1e-9:
            out.append((poly[i][0] + d * n1[0], poly[i][1] + d * n1[1]))
        else:
            out.append((poly[i][0] + d * (n1[0] + n2[0]) / k, poly[i][1] + d * (n1[1] + n2[1]) / k))
    return out


def bearing(frm, to):
    return math.degrees(math.atan2(to[1] - frm[1], to[0] - frm[0])) % 360.0


def angdiff(a, b, period=360.0):
    d = (a - b) % period
    return min(d, period - d)


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


COMPASS = {"east": 0.0, "north-east": 45.0, "north": 90.0, "north-west": 135.0, "west": 180.0, "south-west": 225.0,
           "south": 270.0, "south-east": 315.0}
DIRV = {"east": (1.0, 0.0), "west": (-1.0, 0.0), "north": (0.0, 1.0), "south": (0.0, -1.0)}
AXIS_DEG = {"east-west": 0.0, "north-south": 90.0}
ORDER_AX = {"west-east": (0, 1.0), "east-west": (0, -1.0), "south-north": (1, 1.0), "north-south": (1, -1.0)}
REL_DEG = {"front": 0.0, "left": 90.0, "behind": 180.0, "right": 270.0}
LENS_ALIAS = {"wide angle": "wide-angle", "wide": "wide-angle", "moderately-wide": "moderately wide",
              "slightly-long": "slightly long", "normal lens": "normal"}
SPACE_RE = re.compile(r"^S\d\d$")
ENT_RE = re.compile(r"^[PAVO]\d\d$")
ANCH_RE = re.compile(r"^([A-Za-z0-9_*]+?)(?:@(-?\d+(?:\.\d+)?))?(?::(north|south|east|west|north_line|south_line|"
                     r"east_line|west_line))?$")


def fmt_t(t):
    return ("%.2f" % t).rstrip("0").rstrip(".")


# ------------------------------------------------------------------ scene
class Scene:
    def __init__(self, d):
        self.d = d
        self.spaces = d.get("spaces") or {}
        self.elems = d.get("elems") or {}
        self.ents = d.get("ents") or {}
        self.cams = d.get("cams") or {}
        self.ph = float(d.get("ph") or 1.75)
        self.duration = d.get("duration")
        self.known = d.get("known", "all")
        self.max_gap = d.get("max_gap_s")
        self.space_tags = bool(d.get("space_tags"))
        self.axis_fallback = bool(d.get("axis_fallback"))
        self.frame = d.get("frame", "?")
        self.gaps = d.get("gaps") or {}   # prereg_v5.0.2 m6 (GT scenes): {eid: [[t_a, t_b], ...]} unknown intervals
        self.use_gaps = True
        self.fix52 = set((d.get("fix52") or {}).get("on") or ())   # pvscore 5.2 M1 / M3 (model scenes, from_dump)
        for sid, s in self.spaces.items():
            if s.get("bbox") is None and s.get("poly"):
                xs, ys = [p[0] for p in s["poly"]], [p[1] for p in s["poly"]]
                s["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
        self._spans = {}

    # ---- spaces
    def space_contains(self, sid, xy):
        s = self.spaces.get(sid)
        if s is None:
            return None
        if s.get("poly"):
            return inside(xy, s["poly"])
        b = s.get("bbox")
        if not b:
            return None
        return b[0] <= xy[0] <= b[2] and b[1] <= xy[1] <= b[3]

    def near_space(self, sid, xy, d):
        """xy inside space sid or within d of its outline"""
        s = self.spaces.get(sid)
        if s is None or xy is None or xy[0] is None or xy[1] is None:
            return False
        return bool(self.space_contains(sid, xy)) or self.near_edge(sid, xy, d)

    def level(self, sid, xy):
        """floor or ground height of space sid under the point xy (a sloped space: its ground plane)"""
        s = self.spaces[sid]
        g = s.get("ground")
        if g and xy is not None and xy[0] is not None and xy[1] is not None:
            return float(g[0] + g[1] * xy[0] + g[2] * xy[1])
        return float(s.get("floor") or 0.0)

    def floor_at(self, xy, default_sid=None):
        """floor or ground under xy: the default space when xy is inside it and not on its outline; otherwise the
        highest level among the spaces whose closed outline holds xy (a point on the edge of a platform stands on
        the platform; the rule does not depend on the scene's orientation, so a mirrored scene reads the same);
        otherwise the default space, else 0"""
        eps = 1e-6
        if default_sid in self.spaces and self.space_contains(default_sid, xy) and \
                not self.near_edge(default_sid, xy, eps):
            return self.level(default_sid, xy)
        lv = [self.level(sid, xy) for sid in sorted(self.spaces) if self.near_space(sid, xy, eps)]
        if lv:
            return max(lv)
        if default_sid in self.spaces:
            return self.level(default_sid, xy)
        return 0.0

    def cam_floor_at(self, xy, default_sid=None):
        """ground under a camera. prereg_v5.0.3 n1b (g1): in a space with a fitted ground plane and local ground
        points (attach_ground), the plane of the least-perimeter local triangle that holds xy (local_ground); no
        such triangle -> floor_at. Without n1b (v5.0.1 / v5.0.2): floor_at."""
        if MECH("n1b") and xy is not None and xy[0] is not None and xy[1] is not None:
            for sid in ([default_sid] if default_sid in self.spaces else []) + sorted(self.spaces):
                s = self.spaces[sid]
                if s.get("ground") and s.get("gpts") and self.near_space(sid, xy, 1e-6):
                    ck_ = (sid, round(float(xy[0]), 6), round(float(xy[1]), 6))
                    cache = self.__dict__.setdefault("_n1b_cache", {})
                    if ck_ not in cache:
                        cache[ck_] = local_ground(s["gpts"], xy, self.ph)
                    g = cache[ck_]
                    if g is not None:
                        return float(g[0] + g[1] * xy[0] + g[2] * xy[1])
                    break
        return self.floor_at(xy, default_sid)

    def near_edge(self, sid, xy, d):
        """xy within d of the outline of space sid"""
        s = self.spaces.get(sid)
        if s is None or xy is None or xy[0] is None or xy[1] is None:
            return False
        if s.get("poly"):
            P = s["poly"]
            return min(seg_pt(xy, P[i], P[(i + 1) % len(P)]) for i in range(len(P))) <= d
        b = s.get("bbox")
        if not b:
            return False
        inside_b = b[0] <= xy[0] <= b[2] and b[1] <= xy[1] <= b[3]
        if inside_b:
            return min(xy[0] - b[0], b[2] - xy[0], xy[1] - b[1], b[3] - xy[1]) <= d
        return math.hypot(max(b[0] - xy[0], 0.0, xy[0] - b[2]), max(b[1] - xy[1], 0.0, xy[1] - b[3])) <= d

    def space_of(self, xy):
        for sid in sorted(self.spaces):
            if self.space_contains(sid, xy):
                return sid
        return None

    # ---- entities
    def spans(self, eid):
        """pvscore5 5.0.3 known spans of a GT track: split at gaps > 6 s in which it moves > 0.5 m"""
        if eid in self._spans:
            return self._spans[eid]
        S = self.ents[eid]["samples"]
        out, cur = [], [0]
        for i in range(1, len(S)):
            a, b = S[i - 1], S[i]
            if b["t"] - a["t"] > 6.0 and math.hypot(b["c"][0] - a["c"][0], b["c"][1] - a["c"][1]) > 0.5:
                out.append(cur)
                cur = [i]
            else:
                cur.append(i)
        out.append(cur)
        self._spans[eid] = out
        return out

    def ent_at(self, eid, t, linear=False):
        """entity state (El + t) at time t (None: first sample); None when absent / not known then"""
        e = self.ents.get(eid)
        if e is None or not e.get("samples"):
            return None
        S = e["samples"]
        if t is None:
            return dict(S[0])
        if e.get("static"):
            return dict(S[0], t=t)
        if self.use_gaps and self.gaps.get(eid):
            # prereg_v5.0.2 m6: inside a GT gap in which a GT event moves the entity, its position is unknown beyond
            # HOLD_S of the gap's two samples (state claims, event probes and camera subjects alike)
            for ga, gb in self.gaps[eid]:
                if ga + HOLD_S + 1e-6 < t < gb - HOLD_S - 1e-6:
                    return None
        if self.known == "spans" and not linear:
            for sp in self.spans(eid):
                a, b = S[sp[0]], S[sp[-1]]
                if a["t"] - 0.5 - 1e-6 <= t <= b["t"] + 0.5 + 1e-6:
                    return self._interp([S[i] for i in sp], t, clamp=True)
            return None
        # outside the samples: the end sample is held for HOLD_S (the GT known-span rule), then absent
        if t < S[0]["t"] - HOLD_S - 1e-6 or t > S[-1]["t"] + HOLD_S + 1e-6:
            return None
        return self._interp(S, t, clamp=True)

    def _interp(self, S, t, clamp):
        if t <= S[0]["t"]:
            return dict(S[0], t=t) if (clamp or abs(t - S[0]["t"]) <= 1e-6) else None
        if t >= S[-1]["t"]:
            return dict(S[-1], t=t) if (clamp or abs(t - S[-1]["t"]) <= 1e-6) else None
        lo, hi = 0, len(S) - 1
        while hi - lo > 1:
            m = (lo + hi) // 2
            if S[m]["t"] <= t:
                lo = m
            else:
                hi = m
        a, b = S[lo], S[hi]
        if self.max_gap is not None and b["t"] - a["t"] > self.max_gap + 1e-6:
            return None
        u = 0.0 if b["t"] - a["t"] < 1e-9 else (t - a["t"]) / (b["t"] - a["t"])
        near = a if (t - a["t"]) <= (b["t"] - t) else b
        out = dict(near)
        out["c"] = [a["c"][0] + u * (b["c"][0] - a["c"][0]), a["c"][1] + u * (b["c"][1] - a["c"][1])]
        out["t"] = t
        if abs(near["t"] - t) > 0.5 + 1e-6:
            out["front"] = None
            out["yaw_stated"] = False
            out.pop("front_unknown", None)     # 5.2 (M14): a sample gap is no front, as before
        return out

    # ---- cameras
    def cam_at(self, si, t):
        rows = self.cams.get(str(si)) or self.cams.get(si) or []
        if not rows:
            return None
        rows = sorted(rows, key=lambda r: r["t"])
        if self.frame == "model":
            r = min(rows, key=lambda r: abs(r["t"] - t))
            return _cam_norm(r) if abs(r["t"] - t) <= 0.06 else None
        if t <= rows[0]["t"]:
            return _cam_norm(rows[0])
        if t >= rows[-1]["t"]:
            return _cam_norm(rows[-1])
        for a, b in zip(rows, rows[1:]):
            if a["t"] - 1e-9 <= t <= b["t"] + 1e-9:
                if abs(t - a["t"]) <= 1e-6:
                    return _cam_norm(a)
                if abs(t - b["t"]) <= 1e-6:
                    return _cam_norm(b)
                u = 0.0 if b["t"] - a["t"] < 1e-9 else (t - a["t"]) / (b["t"] - a["t"])
                out = {"t": t, "hfov": a.get("hfov") if a.get("hfov") is not None else b.get("hfov")}
                for k in ("pos", "look"):
                    pa, pb = a.get(k), b.get(k)
                    if pa is None or pb is None or any(x is None for x in list(pa) + list(pb)):
                        out[k] = None
                    else:
                        out[k] = [pa[i] + u * (pb[i] - pa[i]) for i in range(3)]
                if out.get("look") is None:
                    fa, fb = a.get("fwd"), b.get("fwd")
                    if fa is not None and fb is not None:
                        out["fwd"] = [fa[i] + u * (fb[i] - fa[i]) for i in range(3)]
                return _cam_norm(out)
        return None


def _cam_norm(r):
    """-> {pos, fwd (unit), up (unit or None), hfov}; pos / fwd None when not given"""
    pos = r.get("pos")
    if pos is not None and any(x is None for x in pos):
        pos_ok = None
    else:
        pos_ok = pos
    fwd = r.get("fwd")
    if fwd is None and r.get("look") is not None and pos_ok is not None and all(x is not None for x in r["look"]):
        fwd = [r["look"][i] - pos_ok[i] for i in range(3)]
    if fwd is not None:
        n = math.sqrt(sum(x * x for x in fwd))
        fwd = [x / n for x in fwd] if n > 1e-9 else None
    return {"t": r.get("t"), "pos": pos, "pos_ok": pos_ok, "fwd": fwd, "up": r.get("up"), "hfov": r.get("hfov"),
            "look": r.get("look")}


# ------------------------------------------------------------------ elements and anchors
def el_u(e, ph):
    if e.get("person"):
        return float(e.get("h") or ph)
    h = e.get("h")
    return float(max(e.get("long") or 0.0, h if h is not None else (e.get("short") or 0.0)))


def el_poly(e, ph):
    if e.get("person"):
        return octagon(e["c"], J("person_radius_frac") * el_u(e, ph))
    return rect(e["c"], max(e.get("long") or 0.0, 1e-3), max(e.get("short") or 0.0, 1e-3), e.get("axis") or 0.0)


class Missing(Exception):
    pass


def resolve(sc, bind, name, linear=False):
    """-> ("space", sid, bbox) | ("line", axis 0/1, value) | ("el", El)   (Missing when the scene lacks it)"""
    if SPACE_RE.match(name):
        s = sc.spaces.get(name)
        if s is None or not s.get("bbox"):
            raise Missing("space %s" % name)
        return ("space", name, s["bbox"])
    m = re.match(r"^(S\d\d):(north|south|east|west)$", name)
    if m:
        s = sc.spaces.get(m.group(1))
        if s is None or not s.get("bbox"):
            raise Missing("space %s" % m.group(1))
        x0, y0, x1, y1 = s["bbox"]
        return ("line",) + {"north": (1, y1), "south": (1, y0), "east": (0, x1), "west": (0, x0)}[m.group(2)]
    m = ANCH_RE.match(name)
    if not m:
        raise ValueError("bad anchor %r" % name)
    base, t, face = m.group(1), m.group(2), m.group(3)
    t = float(t) if t is not None else None
    if ENT_RE.match(base):
        e = sc.ent_at(base, t, linear=linear)
        if e is None:
            raise Missing("%s%s" % (base, "" if t is None else "@" + fmt_t(t)))
    else:
        if t is not None:
            raise ValueError("time on a static anchor %r" % name)
        eid = bind.get(base) if bind is not None else base
        if eid is None or eid not in sc.elems:
            raise Missing(base)
        e = sc.elems[eid]
    if face:
        P = el_poly(e, sc.ph)
        xs, ys = [p[0] for p in P], [p[1] for p in P]
        if face in ("north", "south", "east", "west"):
            face += "_line"
        return ("line",) + {"north_line": (1, max(ys)), "south_line": (1, min(ys)), "east_line": (0, max(xs)),
                            "west_line": (0, min(xs))}[face]
    return ("el", e)


def _runs52(polys, ax, k, seam):
    """pvscore 5.2 M1: the runs (sorted, merged when closer than seam) where the line {coordinate 1 - ax = k} crosses
    the convex polygons"""
    f, iv = 1 - ax, []
    for P in polys:
        xs = []
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            if (a[f] - k) * (b[f] - k) > 0:
                continue
            if a[f] == b[f]:
                xs += [a[ax], b[ax]] if a[f] == k else []
                continue
            xs.append(a[ax] + (k - a[f]) / (b[f] - a[f]) * (b[ax] - a[ax]))
        if xs:
            iv.append([min(xs), max(xs)])
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + seam:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def space_side52(sc, sid, d, A):
    """pvscore 5.2 M1 (scorer PREREG_52.json): the line value of side d of space sid as seen from A, on the line through
    A's centre in direction d. Floor (space_geom52 floor_polys) on that line: the far end of the floor run that holds the
    centre (off the floor: of the next run ahead, else of the nearest run behind); a wall run of the space (edge_polys)
    on that line, not behind A, that stands on that end (within seam it starts before the end and ends at or after it)
    ends the space at its near face instead (a floor slab under the wall ends at the wall's inner face; a wall run
    inside the floor, with floor beyond it, does not end the space). No floor on that line: the wall run on that side:
    the near face of the edge piece that holds the centre or of the first one ahead, else the far face of the nearest
    one behind. Neither on that line: the side of the space's extent (ext, the M3 tiers). The pieces of A
    itself (floor_own / edge_own = A's instance key) are left out: an object is not a side of its own space."""
    s = sc.spaces[sid]
    ax = 1 if d in ("north", "south") else 0
    sg = 1.0 if d in ("north", "east") else -1.0
    u = sg * A["c"][ax]
    seam = float(sc.d["fix52"].get("seam") or 0.0)

    def runs(key, okey):                             # runs on the line, in walking order (A's own pieces left out)
        own = s.get(okey) or []
        pl = [q for i, q in enumerate(s.get(key) or []) if A.get("k52") is None or i >= len(own) or own[i] != A["k52"]]
        return sorted((min(sg * a, sg * b), max(sg * a, sg * b)) for a, b in _runs52(pl, ax, A["c"][1 - ax], seam))
    F, E = runs("floor_polys", "floor_own"), runs("edge_polys", "edge_own")
    for R in (F, E):
        if not R:
            continue
        hold = [r for r in R if r[0] - 1e-9 <= u <= r[1] + 1e-9]
        ahead = [r for r in R if r[0] > u + 1e-9]
        behind = [r for r in R if r[1] < u - 1e-9]
        if R is F:
            e = (hold or ahead or [behind[-1]])[0][1]
            W = [r for r in E if r[1] >= u - 1e-9 and r[0] <= e + seam and r[1] >= e - seam]
            return sg * (W[0][0] if W else e)
        return sg * (hold[0][0] if hold else ahead[0][0] if ahead else behind[-1][1])
    b = s.get("ext") or s["bbox"]
    return {"north": b[3], "south": b[1], "east": b[2], "west": b[0]}[d]


def _need_el(r, name):
    if r[0] != "el":
        raise ValueError("%s must be a thing or an entity, not a %s" % (name, r[0]))
    return r[1]


def _centre(r):
    if r[0] == "el":
        return r[1]["c"]
    if r[0] == "space":
        x0, y0, x1, y1 = r[2]
        return [(x0 + x1) / 2, (y0 + y1) / 2]
    raise ValueError("line has no centre")


def _gap(sc, A, B):
    PA = el_poly(A, sc.ph)
    if B[0] == "line":
        ax, v = B[1], B[2]
        vals = [p[ax] for p in PA]
        if min(vals) <= v <= max(vals):
            return 0.0
        return min(abs(x - v) for x in vals)
    if B[0] == "space":
        raise ValueError("gap to a whole space")
    return poly_gap(PA, el_poly(B[1], sc.ph))


def _u_pair(sc, A, B):
    ua = el_u(A, sc.ph)
    if B[0] == "el":
        return min(ua, el_u(B[1], sc.ph))
    return ua


# ------------------------------------------------------------------ claims
STATE_TYPES = ("in_space", "side", "part", "order", "touch", "next_to", "on", "in", "facing", "axis", "rel", "moving")
WRAP_TYPES = ("event", "during")  # claims that hold a predicate ('pred') judged at chosen times
ANCHORING = ("side", "part", "order", "touch", "next_to", "on", "in", "rel")
ENT_FIELDS = ("a", "b", "of", "in", "toward")


def at_time(pred, t):
    """event predicate -> the same claim with every bare entity anchor read at time t"""
    p = copy.deepcopy(pred)
    for k in ENT_FIELDS:
        v = p.get(k)
        if isinstance(v, str):
            m = re.match(r"^([PAVO]\d\d)(:.*)?$", v)
            if m:
                p[k] = "%s@%s%s" % (m.group(1), fmt_t(t), m.group(2) or "")
    return p


class Res(dict):
    """claim result: ok (bool), miss (bool), v (measured value), note, ok_m (holds with the GT margins)"""


def judge(sc, bind, cl, margin=False, linear=False, count_fn=None):
    """judge one state / event claim on scene sc with binding bind (GT key -> scene element id; None: identity)"""
    t = cl["t"]
    try:
        if t == "event":
            return _judge_event(sc, bind, cl, margin, count_fn)
        if t == "during":
            return _judge_during(sc, bind, cl, linear, count_fn)
        if t == "count":
            n = count_fn(cl["a"]) if count_fn else sum(1 for k in sc.elems if fnmatch.fnmatch(k, cl["a"]))
            ok = n == int(cl["n"])
            return Res(ok=ok, miss=False, v=n, note="count", ok_m=ok)
        if t == "moving":
            return _judge_moving(sc, bind, cl, linear)
        A = resolve(sc, bind, cl["a"], linear) if t != "order" else None
        if t == "in_space":
            Ael = _need_el(A, "a")
            sid = cl["s"]
            if sid not in sc.spaces:
                raise Missing("space %s" % sid)
            if sc.space_tags and Ael.get("space") is not None:
                inn = Ael["space"] == sid
            else:
                inn = bool(sc.space_contains(sid, Ael["c"]))
            ok = (not inn) if cl.get("neg") else inn
            return Res(ok=ok, miss=False, v=float(inn), note="inside %s" % sid, ok_m=ok)
        if t == "side":
            Ref = resolve(sc, bind, cl["of"], linear)
            ux, uy = DIRV[cl["dir"]]
            ca = _centre(A)
            if Ref[0] == "line":
                v = (ca[Ref[1]] - Ref[2]) * (ux if Ref[1] == 0 else uy)
                if (Ref[1] == 0 and ux == 0) or (Ref[1] == 1 and uy == 0):
                    raise ValueError("side %s of a %s line" % (cl["dir"], "x" if Ref[1] == 0 else "y"))
            else:
                cr = _centre(Ref)
                v = (ca[0] - cr[0]) * ux + (ca[1] - cr[1]) * uy
            ok = v > 0
            if A[0] == "space":
                x0, y0, x1, y1 = A[2]
                ext = (x1 - x0) if ux else (y1 - y0)
                need = GM("side_space_frac") * ext
            elif Ref[0] == "space":
                x0, y0, x1, y1 = Ref[2]
                ext = (x1 - x0) if ux else (y1 - y0)
                need = GM("side_space_frac") * ext
            else:
                need = GM("side_frac") * _u_pair(sc, A[1], Ref)
            return Res(ok=ok, miss=False, v=v, note="offset (need >= %.3f on GT)" % need, ok_m=v >= need)
        if t == "part":
            In = resolve(sc, bind, cl["in"], linear)
            if In[0] == "space":
                x0, y0, x1, y1 = In[2]
                if "M3" in sc.fix52 and sc.spaces[In[1]].get("ext"):   # pvscore 5.2 M3: the space's floor extent
                    x0, y0, x1, y1 = sc.spaces[In[1]]["ext"]
            elif In[0] == "el":
                P = el_poly(In[1], sc.ph)
                x0, y0 = min(p[0] for p in P), min(p[1] for p in P)
                x1, y1 = max(p[0] for p in P), max(p[1] for p in P)
            else:
                raise ValueError("part of a line")
            ca = _centre(A)
            if "M3" in sc.fix52 and A[0] == "space" and sc.spaces[A[1]].get("ext"):   # 5.2 M3: a space by its extent
                ex = sc.spaces[A[1]]["ext"]
                ca = [(ex[0] + ex[2]) / 2, (ex[1] + ex[3]) / 2]
            fx = (ca[0] - x0) / max(x1 - x0, 1e-9)
            fy = (ca[1] - y0) / max(y1 - y0, 1e-9)
            lo, hi = J("part_cuts")
            pm = GM("part_frac")
            w = cl["w"]
            conds = []
            for part in {"nw": ("north", "west"), "ne": ("north", "east"), "sw": ("south", "west"),
                         "se": ("south", "east")}.get(w, (w,)):
                f = fx if part in ("west", "east", "middle_ew") else fy
                if part in ("west", "south"):
                    conds.append((f < lo, lo - f))
                elif part in ("east", "north"):
                    conds.append((f > hi, f - hi))
                elif part in ("middle_ew", "middle_ns"):
                    conds.append((lo <= f <= hi, min(f - lo, hi - f)))
                else:
                    raise ValueError("unknown part %r" % w)
            ok = all(c[0] for c in conds)
            mg = min(c[1] for c in conds)
            return Res(ok=ok, miss=False, v=round(fx, 3) if len(conds) == 1 and w in ("west", "east", "middle_ew")
                       else (round(fy, 3) if len(conds) == 1 else [round(fx, 3), round(fy, 3)]),
                       note="fraction across %s" % cl["in"], ok_m=ok and mg >= pm)
        if t == "order":
            if len(cl["a"]) != 2:
                raise ValueError("order claims are judged as pairs (build.py expands lists)")
            A0 = resolve(sc, bind, cl["a"][0], linear)
            A1 = resolve(sc, bind, cl["a"][1], linear)
            ax, sg = ORDER_AX[cl["axis"]]
            sep = (_centre(A1)[ax] - _centre(A0)[ax]) * sg
            if A0[0] == "el" and A1[0] == "el":
                need = GM("order_frac") * min(el_u(A0[1], sc.ph), el_u(A1[1], sc.ph))
            else:
                need = 0.0
                for X in (A0, A1):
                    if X[0] == "space":
                        x0, y0, x1, y1 = X[2]
                        need = max(need, GM("side_space_frac") * ((x1 - x0) if ax == 0 else (y1 - y0)))
                    else:
                        need = max(need, GM("order_frac") * el_u(X[1], sc.ph))
            return Res(ok=sep > 0, miss=False, v=sep, note="separation (need >= %.3f on GT)" % need, ok_m=sep >= need)
        if t in ("touch", "next_to"):
            Ael = _need_el(A, "a")
            B = resolve(sc, bind, cl["b"], linear)
            m52 = re.match(r"^(S\d\d):(north|south|east|west)$", cl["b"]) if "M1" in sc.fix52 else None
            if m52:                                       # pvscore 5.2 M1: the space side beyond A, not the box side
                B = ("line", B[1], space_side52(sc, m52.group(1), m52.group(2), Ael))
            g = _gap(sc, Ael, B)
            up = _u_pair(sc, Ael, B)
            if t == "touch" and MECH("m3") and B[0] == "el" and bool(Ael.get("person")) != bool(B[1].get("person")):
                # prereg_v5.0.2 m3: a person and a thing (held, carried, caught, pushed): the person's size sets the band
                up = el_u(Ael if Ael.get("person") else B[1], sc.ph)
            band = J("touch_frac" if t == "touch" else "next_to_frac") * up
            ok = g <= band + 1e-9
            okm = ok if t == "touch" else g <= GM("next_to_frac") * band + 1e-9
            return Res(ok=ok, miss=False, v=g, note="gap (band %.3f)" % band, ok_m=okm)
        if t in ("on", "in"):
            Ael = _need_el(A, "a")
            Bel = _need_el(resolve(sc, bind, cl["b"], linear), "b")
            PB = el_poly(Bel, sc.ph)
            g = J("onin_xy_grow_frac") * el_u(Ael, sc.ph)
            xy_ok = inside(Ael["c"], grow(PB, Bel["c"], g))
            z_ok, note = True, "footprint"
            za0, za1, zb0, zb1 = Ael.get("z0"), Ael.get("z1"), Bel.get("z0"), Bel.get("z1")
            if not Ael.get("person") and None not in (za0, za1, zb0, zb1):
                ha, hb = za1 - za0, zb1 - zb0
                if t == "on" and MECH("m4"):
                    # prereg_v5.0.2 m4: A's bottom lies between B's bottom and B's top (within tol) and A reaches up
                    # to B's top (within tol): resting on it, flush in it or wound on it; tol has a floor of
                    # z_floor_frac * ph, so thin surfaces are not held to millimetres
                    tol = max(J("on_z_frac") * max(ha, hb), MECH("m4")["z_floor_frac"] * sc.ph)
                    z_ok = zb0 - tol - 1e-9 <= za0 <= zb1 + tol + 1e-9 and za1 >= zb1 - tol - 1e-9
                elif t == "on":
                    z_ok = abs(za0 - zb1) <= J("on_z_frac") * max(ha, hb) + 1e-9
                else:
                    zc = (za0 + za1) / 2
                    z_ok = zb0 - J("in_z_below_frac") * ha - 1e-9 <= zc <= zb1 + J("in_z_above_frac") * ha + 1e-9
                note = "footprint + height"
            ok = xy_ok and z_ok
            return Res(ok=ok, miss=False, v=float(ok), note=note, ok_m=ok)
        if t == "facing":
            Ael = _need_el(A, "a")
            if "toward" in cl:
                To = resolve(sc, bind, cl["toward"], linear)
                target = bearing(Ael["c"], _centre(To))
            else:
                target = COMPASS[cl["dir"]]
            fr = Ael.get("front")
            if fr is None:
                if Ael.get("front_unknown"):    # 5.2 (M14): the model's facing is unknown there, not scored
                    return Res(ok=False, miss=False, v=None, note="front unknown (5.2 M14)", ok_m=False, unk=True)
                if sc.axis_fallback and Ael.get("axis") is not None:
                    d = angdiff(Ael["axis"], target, 180.0)
                    return Res(ok=d <= J("cone_deg"), miss=False, v=d, note="axis_only", ok_m=d <= GM("cone_deg"))
                return Res(ok=False, miss=False, v=None, note="no front", ok_m=False)
            if sc.frame == "gt" and Ael.get("yaw_stated") is False:
                return Res(ok=False, miss=False, v=None, note="GT facing not stated at this time", ok_m=False)
            d = angdiff(fr, target)
            return Res(ok=d <= J("cone_deg"), miss=False, v=d, note="front vs target (deg)", ok_m=d <= GM("cone_deg"))
        if t == "axis":
            Ael = _need_el(A, "a")
            if Ael.get("axis") is None:
                return Res(ok=False, miss=False, v=None, note="no axis", ok_m=False)
            d = angdiff(Ael["axis"], AXIS_DEG[cl["w"]], 180.0)
            sq = (Ael.get("long") or 0) >= GM("axis_square_ratio") * (Ael.get("short") or 0)
            return Res(ok=d <= J("axis_deg"), miss=False, v=d, note="long axis vs %s (deg)" % cl["w"],
                       ok_m=d <= GM("axis_deg") and sq)
        if t == "rel":
            Ael = _need_el(A, "a")
            Bel = _need_el(resolve(sc, bind, cl["b"], linear), "b")
            fr = Bel.get("front")
            if fr is None and Bel.get("front_unknown"):     # 5.2 (M14): b's facing is unknown there, not scored
                return Res(ok=False, miss=False, v=None, note="b front unknown (5.2 M14)", ok_m=False, unk=True)
            if fr is None or (sc.frame == "gt" and Bel.get("yaw_stated") is False):
                return Res(ok=False, miss=False, v=None, note="b has no front", ok_m=False)
            rb = (bearing(Bel["c"], Ael["c"]) - fr) % 360.0
            d = angdiff(rb, REL_DEG[cl["w"]])
            return Res(ok=d <= J("cone_deg"), miss=False, v=d, note="bearing from b's front (deg off)",
                       ok_m=d <= GM("cone_deg"))
        raise ValueError("unknown claim type %r" % t)
    except Missing as e:
        return Res(ok=False, miss=True, v=None, note="missing %s" % e, ok_m=False)


def _judge_moving(sc, bind, cl, linear):
    m = re.match(r"^([PAVO]\d\d)@(-?\d+(?:\.\d+)?)$", cl["a"])
    if not m:
        raise ValueError("moving needs entity@t")
    eid, t = m.group(1), float(m.group(2))
    hw = J("moving_half_window_s")
    a, b = sc.ent_at(eid, t - hw, linear=True), sc.ent_at(eid, t + hw, linear=True)
    if a is None or b is None:
        return Res(ok=False, miss=True, v=None, note="missing %s around %s s" % (eid, fmt_t(t)), ok_m=False)
    v = math.hypot(b["c"][0] - a["c"][0], b["c"][1] - a["c"][1]) / (2 * hw)
    thr = J("moving_frac_per_s") * el_u(a, sc.ph)
    mv = v >= thr
    ok = (not mv) if cl.get("neg") else mv
    return Res(ok=ok, miss=False, v=v, note="speed (threshold %.3f / s)" % thr, ok_m=ok)


def _judge_event(sc, bind, cl, margin, count_fn):
    dur = sc.duration if sc.duration is not None else 1e9
    if cl.get("probe"):
        # prereg_v5.0.2 m1: the probe times build.py fixed from the GT (event_probes)
        tb, ta = float(cl["probe"][0]), float(cl["probe"][1])
    else:
        tb = max(0.0, float(cl["t0"]) - float(cl["tol"]))
        ta = min(dur, float(cl["t1"]) + float(cl["tol"]))
    pb = judge(sc, bind, at_time(cl["pred"], tb), linear=True, count_fn=count_fn)
    pa = judge(sc, bind, at_time(cl["pred"], ta), linear=True, count_fn=count_fn)
    ok = (not pb["ok"]) and pa["ok"]
    return Res(ok=ok, miss=bool(pa.get("miss")), v=[pb["ok"], pa["ok"]],
               note="pred at %s s = %s, at %s s = %s" % (fmt_t(tb), pb["ok"], fmt_t(ta), pa["ok"]),
               ok_m=(not pb["ok"]) and pa["ok_m"])


def _during_times(cl):
    t0, t1 = float(cl["w"][0]), float(cl["w"][1])
    st = float((MECH("m2") or {}).get("step_s", 0.1))
    n = max(1, int(math.ceil((t1 - t0) / st - 1e-9)))
    return [round(t0 + (t1 - t0) * i / n, 4) for i in range(n + 1)]


def _judge_during(sc, bind, cl, linear, count_fn):
    """prereg_v5.0.2 m2: the predicate holds at some time of the window w (its ends and steps of at most step_s)"""
    ok = okm = False
    first, allmiss = None, True
    for tt in _during_times(cl):
        r = judge(sc, bind, at_time(cl["pred"], tt), linear=linear, count_fn=count_fn)
        allmiss = allmiss and bool(r.get("miss"))
        if r["ok"] and first is None:
            first = tt
        ok = ok or bool(r["ok"])
        okm = okm or bool(r["ok_m"])
    return Res(ok=ok, miss=allmiss, v=first, note="holds at some time in %s-%s s (first %s)" % (
        fmt_t(cl["w"][0]), fmt_t(cl["w"][1]), "-" if first is None else fmt_t(first)), ok_m=okm)


def anchors_of(cl):
    out = []
    if cl["t"] in WRAP_TYPES:
        return anchors_of(cl["pred"])
    for k in ("a", "b", "of", "in", "toward"):
        v = cl.get(k)
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            out += v
    return out


def base_key(a):
    return re.split(r"[@:]", a, maxsplit=1)[0]


# ------------------------------------------------------------------ cameras
def lens_class(hfov):
    if hfov is None:
        return None
    L = CJ("lens_hfov")
    return min(L, key=lambda k: abs(math.log(hfov / L[k])))


def lens_hfov(word):
    if word is None:
        return None
    w = str(word).strip().lower()
    w = LENS_ALIAS.get(w, w)
    return CJ("lens_hfov").get(w)


def height_class(r):
    cuts, names = CJ("height_cuts"), CJ("height_names")
    for c, n in zip(cuts, names):
        if r < c:
            return n
    return names[-1]


def height_band(name):
    cuts, names = CJ("height_cuts"), CJ("height_names")
    i = names.index(name)
    lo = -1e9 if i == 0 else cuts[i - 1]
    hi = 1e9 if i == len(names) - 1 else cuts[i]
    return lo, hi


def cam_axes(cam):
    f = cam["fwd"]
    up = cam.get("up")
    if up is None:
        rt = [f[1], -f[0], 0.0]
        n = math.hypot(rt[0], rt[1])
        if n < 1e-9:
            return None
        rt = [rt[0] / n, rt[1] / n, 0.0]
        up = [rt[1] * f[2] - rt[2] * f[1], rt[2] * f[0] - rt[0] * f[2], rt[0] * f[1] - rt[1] * f[0]]
    else:
        n = math.sqrt(sum(x * x for x in up))
        up = [x / n for x in up]
        rt = [f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]]
        n = math.sqrt(sum(x * x for x in rt))
        rt = [x / n for x in rt]
    return rt, up


def project(cam, X):
    """-> (u, v, depth) normalised image coordinates (u right, v down) of world point X, or None"""
    if cam.get("pos_ok") is None or cam.get("fwd") is None or cam.get("hfov") is None:
        return None
    ax = cam_axes(cam)
    if ax is None:
        return None
    rt, up = ax
    v = [X[i] - cam["pos_ok"][i] for i in range(3)]
    zc = sum(v[i] * cam["fwd"][i] for i in range(3))
    xc = sum(v[i] * rt[i] for i in range(3))
    yc = sum(v[i] * up[i] for i in range(3))
    if zc <= 1e-9:
        return (None, None, zc)
    th = math.tan(math.radians(cam["hfov"]) / 2)
    tv = th / CJ("aspect")
    return (0.5 + xc / (zc * 2 * th), 0.5 - yc / (zc * 2 * tv), zc)


def subject_point(sc, bind, name, t, space=None):
    r = resolve(sc, bind, "%s@%s" % (name, fmt_t(t)) if ENT_RE.match(name) else name)
    e = _need_el(r, "subject")
    fl = sc.floor_at(e["c"], space)
    if e.get("person"):
        z = fl + CJ("subject_h_frac") * el_u(e, sc.ph)
    elif e.get("z0") is not None and e.get("z1") is not None:
        z = (e["z0"] + e["z1"]) / 2
    elif e.get("h") is not None:
        z = fl + e["h"] / 2
    else:
        z = fl + 0.5 * sc.ph
    return e, [e["c"][0], e["c"][1], z]


def in_frame(sc, bind, cam, name, t, space, margin):
    try:
        _, X = subject_point(sc, bind, name, t, space)
    except Missing:
        return None
    p = project(cam, X)
    if p is None or p[0] is None:
        return False
    u, v, _ = p
    return -margin <= u <= 1 + margin and -margin <= v <= 1 + margin if margin >= 0 else \
        (-margin <= u <= 1 + margin and -margin <= v <= 1 + margin)


def _hbear(f):
    return math.degrees(math.atan2(f[1], f[0])) % 360.0


def _pitch(f):
    return math.degrees(math.asin(max(-1.0, min(1.0, f[2]))))


def move_features(sc, c0, c1):
    if c0 is None or c1 is None or c0.get("pos_ok") is None or c1.get("pos_ok") is None or c0.get("fwd") is None \
            or c1.get("fwd") is None:
        return None
    d = [c1["pos_ok"][i] - c0["pos_ok"][i] for i in range(3)]
    ph = sc.ph
    th = math.hypot(d[0], d[1]) / ph
    tv = d[2] / ph
    tt = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) / ph
    f = c0["fwd"]
    fwd = sum(d[i] * f[i] for i in range(3)) / ph
    fh = math.hypot(f[0], f[1])
    right = ((d[0] * f[1] - d[1] * f[0]) / fh / ph) if fh > 1e-9 else 0.0
    dyaw = wrap180(_hbear(c1["fwd"]) - _hbear(c0["fwd"]))
    dpitch = _pitch(c1["fwd"]) - _pitch(c0["fwd"])
    zr = (c1["hfov"] / c0["hfov"]) if (c0.get("hfov") and c1.get("hfov")) else 1.0
    return {"T_h": th, "T_v": tv, "T": tt, "forward": fwd, "right": right, "dyaw": dyaw, "dpitch": dpitch,
            "zoom": zr}


def move_ok(F, typ, dr, follow_inframe=None):
    tr, sh = CJ("move_trans_ph"), CJ("move_share")
    yaw, pit = CJ("move_yaw_deg"), CJ("move_pitch_deg")
    zr = CJ("zoom_ratio")
    typ = (typ or "static").lower()
    dr = (dr or "").lower() or None

    def dolly(sign):
        f = F["forward"] * sign
        return f >= tr and f >= sh * F["T"]

    def truck(sign):
        r = F["right"] * sign
        return r >= tr and r >= sh * F["T_h"]
    if typ == "static":
        return F["T"] < tr and abs(F["dyaw"]) < CJ("static_rot_deg") and abs(F["dpitch"]) < CJ("static_rot_deg") \
            and 1 / zr <= F["zoom"] <= zr
    if typ == "pan":
        return F["dyaw"] >= yaw if dr == "left" else (F["dyaw"] <= -yaw if dr == "right" else abs(F["dyaw"]) >= yaw)
    if typ == "tilt":
        return F["dpitch"] >= pit if dr == "up" else (F["dpitch"] <= -pit if dr == "down" else abs(F["dpitch"]) >= pit)
    if typ in ("dolly_in",) or (typ in ("truck", "dolly") and dr in ("forward", "in")):
        return dolly(1.0)
    if typ in ("dolly_out",) or (typ in ("truck", "dolly") and dr in ("backward", "out")):
        return dolly(-1.0)
    if typ == "truck":
        return truck(1.0) if dr == "right" else (truck(-1.0) if dr == "left" else abs(F["right"]) >= tr)
    if typ == "pedestal":
        return F["T_v"] >= tr if dr == "up" else (F["T_v"] <= -tr if dr == "down" else abs(F["T_v"]) >= tr)
    if typ == "crane":
        vert = abs(F["T_v"]) >= tr
        if dr == "up":
            return F["T_v"] >= tr
        if dr == "down":
            return F["T_v"] <= -tr
        if dr in ("left", "right"):
            return truck(1.0 if dr == "right" else -1.0) or vert
        if dr in ("forward", "backward"):
            return dolly(1.0 if dr == "forward" else -1.0) or vert
        return vert or F["T_h"] >= tr
    if typ == "follow":
        moved = F["T_h"] >= tr or abs(F["dyaw"]) >= yaw
        return moved and bool(follow_inframe)
    if typ == "zoom":
        return F["zoom"] <= 1 / zr if dr == "in" else (F["zoom"] >= zr if dr == "out" else
                                                       not (1 / zr < F["zoom"] < zr))
    if typ == "orbit":
        return abs(F["dyaw"]) >= CJ("orbit_yaw_deg") and F["T_h"] >= tr
    raise ValueError("unknown move type %r" % typ)


def judge_cam(sc, bind, ck):
    """judge one camera check (PREREG camera_checks) on scene sc"""
    t = ck["t"]
    si = ck["si"]
    try:
        if t == "cam_move":
            c0, c1 = sc.cam_at(si, ck["t0"]), sc.cam_at(si, ck["t1"])
            F = move_features(sc, c0, c1)
            if F is None:
                return Res(ok=False, miss=True, v=None, note="camera missing")
            fin = None
            if ck["type"] == "follow":
                subj = ck.get("subject")
                fin = bool(subj) and bool(in_frame(sc, bind, c0, subj, ck["t0"], ck.get("space"), CJ("frame_margin"))) \
                    and bool(in_frame(sc, bind, c1, subj, ck["t1"], ck.get("space"), CJ("frame_margin")))
            ok = move_ok(F, ck["type"], ck.get("dir"), fin)
            return Res(ok=ok, miss=False, v={k: round(v, 3) for k, v in F.items()}, note="%s %s" % (
                ck["type"], ck.get("dir") or ""))
        cam = sc.cam_at(si, ck["at"])
        if cam is None:
            return Res(ok=False, miss=True, v=None, note="camera missing")
        if t == "cam_lens":
            c = lens_class(cam.get("hfov"))
            return Res(ok=c == ck["w"], miss=c is None, v=c, note="lens class")
        if cam.get("pos") is None:
            return Res(ok=False, miss=True, v=None, note="camera position missing")
        if t == "cam_height":
            p = cam["pos"]
            if p[2] is None:
                return Res(ok=False, miss=True, v=None, note="camera height missing")
            fl = sc.cam_floor_at(p[:2], ck.get("space")) if p[0] is not None else \
                float((sc.spaces.get(ck.get("space")) or {}).get("floor") or 0.0)
            r = (p[2] - fl) / sc.ph
            lo, hi = height_band(ck["w"])
            wd = CJ("height_widen")
            return Res(ok=lo - wd <= r < hi + wd, miss=False, v=round(r, 3), note="height / ph")
        if cam.get("pos_ok") is None or cam.get("fwd") is None:
            return Res(ok=False, miss=True, v=None, note="camera direction or position missing")
        if t == "cam_dir":
            f = cam["fwd"]
            if math.hypot(f[0], f[1]) < 1e-6:
                return Res(ok=False, miss=False, v=None, note="vertical view")
            d = angdiff(_hbear(f), ck["bearing"])
            return Res(ok=d <= CJ("dir_cone_deg"), miss=False, v=round(d, 1), note="view bearing off (deg)")
        if t == "cam_side":
            e, _ = subject_point(sc, bind, ck["subject"], ck["at"], ck.get("space"))
            b = bearing(e["c"], cam["pos_ok"][:2])
            d = angdiff(b, ck["bearing"])
            return Res(ok=d <= CJ("side_cone_deg"), miss=False, v=round(d, 1), note="side bearing off (deg)")
        if t == "cam_frame":
            r = in_frame(sc, bind, cam, ck["subject"], ck["at"], ck.get("space"), CJ("frame_margin"))
            if r is None:
                return Res(ok=False, miss=True, v=None, note="subject missing")
            return Res(ok=bool(r), miss=False, v=bool(r), note="in frame")
        raise ValueError("unknown camera check %r" % t)
    except Missing as e:
        return Res(ok=False, miss=True, v=None, note="missing %s" % e)


# ------------------------------------------------------------------ GT and reader scenes
FRONT_OFF = {"+x": 0.0, "+y": 90.0, "-x": 180.0, "-y": -90.0}


def norm_word(s):
    s = re.sub(r"[^a-z0-9 ]+", " ", str(s or "").lower().replace("_", " ").replace("other:", " "))
    w = [x for x in s.split() if x not in ("the", "a", "an")]
    out = []
    for x in w:
        if len(x) > 3 and x.endswith("ies"):
            x = x[:-3] + "y"
        elif len(x) > 3 and x.endswith("es") and x[:-2].endswith(("sh", "ch", "x", "ss")):
            x = x[:-2]
        elif len(x) > 3 and x.endswith("s") and not x.endswith("ss"):
            x = x[:-1]
        out.append(x)
    return " ".join(out)


# ------------------------------------------------------------------ sloped ground (prereg_v5.0 note n1)
def _plane3(P):
    """least-squares plane z = a + b x + c y through points P [(x, y, z)]; None when degenerate"""
    n = len(P)
    if n < 3:
        return None
    sx = sum(p[0] for p in P) / n
    sy = sum(p[1] for p in P) / n
    sz = sum(p[2] for p in P) / n
    xx = sum((p[0] - sx) ** 2 for p in P)
    yy = sum((p[1] - sy) ** 2 for p in P)
    xy = sum((p[0] - sx) * (p[1] - sy) for p in P)
    xz = sum((p[0] - sx) * (p[2] - sz) for p in P)
    yz = sum((p[1] - sy) * (p[2] - sz) for p in P)
    det = xx * yy - xy * xy
    if det <= 1e-9 * max(1.0, xx * yy):
        return None
    b = (xz * yy - yz * xy) / det
    c = (yz * xx - xz * xy) / det
    return [sz - b * sx - c * sy, b, c]


def ground_plane(points, ph):
    """robust ground plane of a sloped space from the bottoms [(x, y, z)] of the things standing in it: among all
    point triples whose triangle has an area >= (0.5 ph)^2 (at most 40 points, evenly picked in sorted order, form
    triples), the plane that holds the most points within 0.15 ph (ties: the smaller sum of absolute residuals),
    refitted by least squares to those points. Things on top of other things and the bounding boxes of long surfaces
    fall outside. None with fewer than 3 points or no usable triple (the space then keeps its flat floor)."""
    P = sorted((float(p[0]), float(p[1]), float(p[2])) for p in points
               if p is not None and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p[:3]))
    if len(P) < 3:
        return None
    tol = 0.15 * ph
    amin = (0.5 * ph) ** 2
    Q = P if len(P) <= 40 else [P[int(i * (len(P) - 1) / 39)] for i in range(40)]
    best = None
    for i in range(len(Q)):
        for j in range(i + 1, len(Q)):
            for k in range(j + 1, len(Q)):
                a, b, c = Q[i], Q[j], Q[k]
                area = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2
                if area < amin:
                    continue
                pl = _plane3([a, b, c])
                if pl is None:
                    continue
                res = [abs(p[2] - (pl[0] + pl[1] * p[0] + pl[2] * p[1])) for p in P]
                inl = sum(1 for r in res if r <= tol)
                key = (inl, -sum(r for r in res if r <= tol))
                if best is None or key > best[0]:
                    best = (key, pl)
    if best is None or best[0][0] < 3:
        return None
    pl = best[1]
    inl = [p for p in P if abs(p[2] - (pl[0] + pl[1] * p[0] + pl[2] * p[1])) <= tol]
    fit = _plane3(inl) or pl
    return [round(v, 6) for v in fit]


def _in_tri(xy, a, b, c, eps=1e-9):
    """xy inside the closed triangle a b c"""
    def cr(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    d1, d2, d3 = cr(a, b, xy), cr(b, c, xy), cr(c, a, xy)
    neg = d1 < -eps or d2 < -eps or d3 < -eps
    pos = d1 > eps or d2 > eps or d3 > eps
    return not (neg and pos)


def local_ground(points, xy, ph):
    """prereg_v5.0.3 n1b (g1): the plane [a, b, c] of the least-perimeter triangle (ties: sorted corners) of local
    ground points [(x, y, z)] that contains xy and has an area >= min_triangle_area_ph2 * ph^2; only the
    nearest_points points nearest to xy form triangles. None when there is no such triangle."""
    m = MECH("n1b")
    Q = sorted(points, key=lambda p: ((p[0] - xy[0]) ** 2 + (p[1] - xy[1]) ** 2, p))[:int(m["nearest_points"])]
    amin = float(m["min_triangle_area_ph2"]) * ph * ph
    best = None
    for i in range(len(Q)):
        for j in range(i + 1, len(Q)):
            for l in range(j + 1, len(Q)):
                a, b, c = Q[i], Q[j], Q[l]
                area = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2
                if area < amin or not _in_tri(xy, a, b, c):
                    continue
                per = math.dist(a[:2], b[:2]) + math.dist(b[:2], c[:2]) + math.dist(c[:2], a[:2])
                key = (round(per, 9), sorted([a, b, c]))
                if best is None or key < best[0]:
                    best = (key, [a, b, c])
    return _plane3(best[1]) if best else None


def attach_ground(d, sloped, points):
    """d: plain scene dict; sloped: space ids the GT marks sloped; points: [(x, y, bottom z), ...] of the scene's
    things, optionally (x, y, bottom z, footprint long side) (prereg_v5.0.3 n1b). A space's plane is fitted to the
    points inside its outline or within 0.5 ph of it (things on the edge of a ramp or at its foot count; the same rule
    for GT, reader and model scenes, prereg_v5.0.1 n1). Sets d["spaces"][sid]["ground"] where ground_plane() finds a
    plane; returns {sid: plane or None}. With n1b, a space with a plane also keeps d["spaces"][sid]["gpts"]: its
    local ground points for camera height (points near it whose footprint long side is unknown or at most
    max_footprint_long_ph * ph: stated ground spots, person bases, small things)."""
    out = {}
    ph = float(d.get("ph") or 1.75)
    sc = Scene({"spaces": d.get("spaces") or {}})
    P = [p for p in points or [] if p is not None and len(p) >= 3 and all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in p[:3])]
    for sid in sorted(set(sloped or ())):
        s = (d.get("spaces") or {}).get(sid)
        if s is None:
            continue
        pl = ground_plane([p for p in P if sc.near_space(sid, p[:2], 0.5 * ph)], ph)
        out[sid] = pl
        n1b = MECH("n1b")
        if pl is not None:
            s["ground"] = pl
            if n1b:
                lim = float(n1b["max_footprint_long_ph"]) * ph
                s["gpts"] = [tuple(p[:3]) for p in P if sc.near_space(sid, p[:2], float(n1b["near_ph"]) * ph) and
                             (len(p) < 4 or p[3] is None or p[3] <= lim + 1e-9)]
        else:
            s.pop("ground", None)
            s.pop("gpts", None)
    return out


def _obj_el(o, sid_of):
    size = o["size"]
    yaw = float(o.get("yaw_deg") or 0.0)
    long_, short = max(size[0], size[1]), min(size[0], size[1])
    axis = (yaw if size[0] >= size[1] else yaw + 90.0) % 180.0
    front = (yaw + FRONT_OFF[o["front"]]) % 360.0 if o.get("front") else None
    zc, h = o["center"][2], size[2]
    return {"c": [float(o["center"][0]), float(o["center"][1])], "long": float(long_), "short": float(short),
            "axis": axis, "front": front, "h": float(h), "z0": zc - h / 2, "z1": zc + h / 2,
            "space": o.get("space") or sid_of(o["center"][:2]), "cls": o["class"], "labels": [norm_word(o["class"])],
            "person": False, "pos_open": (o.get("stated") or {}).get("position") == "open"}


def scene_from_gt(gt, anonymise=False):
    """GT (pv_gt_v5.0 dict) -> plain scene. anonymise: static objects keyed m000.. (GT-as-model tests); the GT key
    goes to '_key' so a test can check the matcher."""
    spaces = {}
    for s in gt["spaces"]:
        spaces[s["id"]] = {"poly": [list(p) for p in s["outline"]], "bbox": None, "floor": float(s.get("floor_z") or 0.0)}
    tmp = Scene({"spaces": spaces})

    def sid_of(xy):
        return tmp.space_of(xy)
    ent_cls = {e["id"]: e.get("class") for e in gt.get("entities") or []}
    elems, by_id = {}, {}
    n = 0
    for o in gt["objects"]:
        el = _obj_el(o, sid_of)
        if o.get("id"):
            by_id[o["id"]] = el
            continue
        if anonymise:
            el["_key"] = o["key"]
            elems["m%03d" % n] = el
            n += 1
        else:
            elems[o["key"]] = el
    ents = {}
    tracks = {t["id"]: t for t in gt.get("tracks") or []}
    for eid in list(ent_cls) + [k for k in tracks if k not in ent_cls]:
        person = ent_cls.get(eid) in ("person", "child", "man", "woman")
        base = by_id.get(eid)
        tr = tracks.get(eid)
        smp = []
        if tr and tr.get("samples"):
            for i, s in enumerate(tr["samples"]):
                if person or base is None:
                    el = {"c": [float(s["p"][0]), float(s["p"][1])], "long": 0.5, "short": 0.5, "axis": None,
                          "front": None, "h": 1.75 if person else None, "z0": None, "z1": None,
                          "space": s.get("space"), "cls": ent_cls.get(eid) or "", "labels": [], "person": person}
                    if person:
                        el["long"] = el["short"] = 2 * J("person_radius_frac") * 1.75
                else:
                    el = dict(base)
                    el["c"] = [float(s["p"][0]), float(s["p"][1])]
                    el["space"] = s.get("space") or sid_of(s["p"][:2])
                    if i > 0:
                        el["z0"] = el["z1"] = None
                el["pos_open"] = s.get("stated") == "open"
                yaw = s.get("yaw")
                if person or base is None:
                    # a person's track yaw is its facing bearing (drafts_v4 build.py rule)
                    el["front"] = float(yaw) % 360.0 if yaw is not None else None
                    el["yaw_stated"] = yaw is not None
                else:
                    # an object entity keeps its GT box; a stated track yaw is its heading and turns the box with it
                    if yaw is not None and base["front"] is not None:
                        d = float(yaw) - base["front"]
                        el["front"] = float(yaw) % 360.0
                        el["axis"] = (base["axis"] + d) % 180.0
                    el["yaw_stated"] = el.get("front") is not None
                el["t"] = float(s["t"])
                smp.append(el)
            ents[eid] = {"cls": ent_cls.get(eid) or "", "person": person, "samples": smp, "static": False}
        elif base is not None:
            el = dict(base, t=0.0, yaw_stated=base["front"] is not None)
            ents[eid] = {"cls": ent_cls.get(eid) or "", "person": False, "samples": [el], "static": True}
    cams = {}
    for si, sh in enumerate(gt.get("shots") or []):
        rows = []
        for c in sh.get("camera") or []:
            rows.append({"t": float(c.get("t", sh["t"][0])), "pos": c.get("pos"), "look": c.get("look"),
                         "hfov": c.get("hfov_deg")})
        cams[str(si)] = rows
    d = {"frame": "gt", "spaces": spaces, "elems": elems, "ents": ents, "cams": cams, "ph": J("person_h_gt"),
         "duration": float(gt.get("duration_s") or 0.0) or None, "known": "spans", "max_gap_s": None,
         "space_tags": False, "axis_fallback": False}
    if MECH("m6"):
        d["gaps"] = gt_gaps(gt, ents)
    attach_ground(d, gt_sloped(gt), gt_ground_points(gt))
    return d


def gt_windows(gt, ids, kinds=None, exclude=None):
    """[(t0, t1, ref)]: the GT events (of the types in kinds, or all) and GT space transitions that name any of the
    entity ids (event agents / objects, transition agent); exclude: an event id left out"""
    ids = set(ids)
    out = []
    for e in gt.get("events") or []:
        if e.get("id") == exclude or (kinds is not None and e["type"] not in kinds):
            continue
        if ids & (set(e.get("agents") or []) | set(e.get("objects") or [])):
            out.append((float(e["t"][0]), float(e["t"][1]), e["id"]))
    for tr in gt.get("space_transitions") or []:
        if tr.get("agent") in ids:
            a, b = tr.get("time_range") or [tr["t"], tr["t"]]
            out.append((float(a), float(b), "tr:%s@%s" % (tr["agent"], fmt_t(float(tr["t"])))))
    return out


def gt_gaps(gt, ents):
    """prereg_v5.0.2 m6: per entity, the GT track gaps longer than gap_s whose open interval overlaps a GT event of a
    moving type or a GT space transition naming the entity (the GT says it moved there but gives no position)"""
    m6 = MECH("m6")
    out = {}
    for eid, e in ents.items():
        if e.get("static"):
            continue
        W = gt_windows(gt, [eid], kinds=set(m6["moving_event_types"]))
        S = e["samples"]
        for a, b in zip(S, S[1:]):
            if b["t"] - a["t"] > m6["gap_s"] + 1e-9 and any(w0 < b["t"] - 1e-9 and w1 > a["t"] + 1e-9
                                                             for w0, w1, _ in W):
                out.setdefault(eid, []).append([a["t"], b["t"]])
    return out


# ------------------------------------------------------------------ prereg_v5.0.2 build-side rules (build.py)
def pred_ent_ids(pred):
    return sorted({base_key(a) for a in anchors_of(pred) if ENT_RE.match(base_key(a))})


def rider_to_vehicle(cl, gev):
    """m1 (rider): an event claim 'X starts / stops moving' whose GT event (type start or stop) has a vehicle V.. among
    its agents together with X is judged on the vehicle -> (claim, changed)"""
    m1 = MECH("m1")
    if not m1 or not m1.get("rider_to_vehicle") or cl.get("t") != "event":
        return cl, False
    p = cl["pred"]
    if p["t"] != "moving" or gev["type"] not in ("start", "stop") or not re.match(r"^[PA]\d\d$", p["a"]):
        return cl, False
    vs = [a for a in gev["agents"] if re.match(r"^V\d\d$", a)]
    if not vs or p["a"] not in gev["agents"]:
        return cl, False
    c = copy.deepcopy(cl)
    c["pred"]["a"] = vs[0]
    return c, True


def event_probes(gt, sc, cl, detail=False):
    """m1: the probe times (before, after) of an event claim on its GT event window [t0, t1] with tolerance tol, or
    None when the GT cannot verify them.
    (a) a probe never enters a neighbouring window that names an entity of the predicate: a GT event of a type that
        moves things (neighbour_event_types) or a GT space transition (windows inside [t0, t1] are the event itself,
        not neighbours):
        before = max(t0 - tol, the latest end <= t0 of such windows), after = min(t1 + tol, the earliest start >= t1);
        every probe keeps at least h outside [t0, t1] (before <= t0 - h, after >= t1 + h; h = min_offset_s, and for
        a moving predicate at least the moving half window), clamped to the clip;
    (b) on the GT every entity of the predicate must be known at a probe (inside its GT track, first to last sample,
        and outside m6 gaps; prereg_v5.0.3 m1c: strictly inside an m6 gap the hold at its edges does not count); a
        probe where one is not moves toward the
        event in steps of retime_step_s (never past t0 - h / t1 + h); when no such time exists the claim is not
        verifiable on the GT and is dropped. detail=True -> (probes or None, moved by (b))"""
    m1 = MECH("m1")
    t0, t1, tol = float(cl["t0"]), float(cl["t1"]), float(cl["tol"])
    dur = sc.duration if sc.duration is not None else 1e9
    pred = cl["pred"]
    pre, post = max(0.0, t0 - tol), min(dur, t1 + tol)
    for w0, w1, _ in gt_windows(gt, pred_ent_ids(pred), kinds=set(m1["neighbour_event_types"]), exclude=cl.get("ev")):
        if w0 >= t0 - 1e-9 and w1 <= t1 + 1e-9:
            continue  # inside the event's own window (the same happening, or a part of it): not a neighbour
        if w1 <= t0 + 1e-9:
            pre = max(pre, w1)
        if w0 >= t1 - 1e-9:
            post = min(post, w0)
    h = _m1_offset(pred)
    pre = max(0.0, min(pre, t0 - h))
    post = min(dur, max(post, t1 + h))
    lim_pre, lim_post = max(pre, t0 - h), min(post, t1 + h)
    st = float(m1["retime_step_s"])

    ext = {}
    for e in pred_ent_ids(pred):
        en = sc.ents.get(e)
        if en and not en.get("static") and en.get("samples"):
            ext[e] = (en["samples"][0]["t"], en["samples"][-1]["t"])

    def known(tt):
        # a probe needs a GT position inside the entity's track (first to last sample): the known-span hold does
        # not say where it was before it first appears or after it is last seen
        if any(not (a - 1e-9 <= tt <= b + 1e-9) for a, b in ext.values()):
            return False
        # prereg_v5.0.3 m1c (g2): strictly inside an m6 gap, the HOLD_S hold at the gap's two samples does not make a
        # probe known (an amendment of m6 gaps: off with the gaps, sc.use_gaps False; sc.use_m1c False only for the
        # build's drop log, which asks whether a dropped claim had probes without m1c)
        if (MECH("m1c") and getattr(sc, "use_gaps", True) and getattr(sc, "use_m1c", True)
                and any(ga + 1e-9 < tt < gb - 1e-9 for e in ext
                        for ga, gb in ((getattr(sc, "gaps", None) or {}).get(e) or []))):
            return False
        return not judge(sc, None, at_time(pred, tt), linear=True).get("miss")
    p = pre
    while p <= lim_pre + 1e-9 and not known(p):
        p = round(p + st, 4)
    q = post
    while q >= lim_post - 1e-9 and not known(q):
        q = round(q - st, 4)
    if p > lim_pre + 1e-9 or q < lim_post - 1e-9:
        return (None, True) if detail else None
    out = [round(p, 3), round(q, 3)]
    return (out, abs(p - pre) > 1e-9 or abs(q - post) > 1e-9) if detail else out


def _m1_offset(pred):
    """m1: the least distance of a probe from its event window"""
    h = float((MECH("m1") or {}).get("min_offset_s", 0.0))
    return max(h, J("moving_half_window_s")) if pred["t"] == "moving" else h


def gt_judge_event(sc, cl):
    """GT validation of an event claim: its probes plus the gt_margin event_time_s inner probes (each moved
    event_time_s toward the event, never into its window); with m1 a probe that misses an entity of the predicate
    on the GT makes the claim unverifiable (Res 'unverifiable')"""
    r = judge(sc, None, cl)
    dur = sc.duration if sc.duration is not None else 1e9
    em = GM("event_time_s")
    pred = cl["pred"]
    if cl.get("probe"):
        tb, ta = float(cl["probe"][0]), float(cl["probe"][1])
        h = _m1_offset(pred)
        ib = min(tb + em, max(tb, float(cl["t0"]) - h))
        ia = max(ta - em, min(ta, float(cl["t1"]) + h))
        inner = [ib, ia]
    else:
        tb = max(0.0, float(cl["t0"]) - float(cl["tol"]))
        ta = min(dur, float(cl["t1"]) + float(cl["tol"]))
        inner = [tb + em, ta - em] if ta - em > tb + em else None
    ok_m = r["ok_m"]
    notes = [r["note"]]
    miss = []
    if MECH("m1"):
        for tt in (tb, ta):
            if judge(sc, None, at_time(pred, tt), linear=True).get("miss"):
                miss.append(tt)
    if inner:
        pb2 = judge(sc, None, at_time(pred, inner[0]), linear=True)
        pa2 = judge(sc, None, at_time(pred, inner[1]), linear=True)
        ok_m = ok_m and not pb2["ok"] and pa2["ok_m"]
        notes.append("inner probes %s s = %s, %s s = %s" % (fmt_t(inner[0]), pb2["ok"], fmt_t(inner[1]), pa2["ok"]))
        if MECH("m1"):
            miss += [tt for tt, x in zip(inner, (pb2, pa2)) if x.get("miss")]
    out = Res(ok=r["ok"], miss=r["miss"], v=r["v"], note="; ".join(notes), ok_m=ok_m)
    if miss:
        out["unverifiable"] = "GT lacks a predicate entity at %s s" % ", ".join(fmt_t(x) for x in miss)
    return out


def explicit_first_time(cl, sc):
    """m7: in a state claim every bare id of an entity the GT tracks becomes id@<its first GT sample time> (a bare
    id means the first GT sample; a judged scene is read at that time, not at its own first sample)"""
    if not MECH("m7") or cl.get("t") in WRAP_TYPES + ("count", "moving"):
        return cl, False
    c = copy.deepcopy(cl)
    ch = [False]

    def fix(a):
        m = re.match(r"^([PAVO]\d\d)(:.*)?$", a)
        if m and m.group(1) in sc.ents and not sc.ents[m.group(1)].get("static") and sc.ents[m.group(1)]["samples"]:
            ch[0] = True
            return "%s@%s%s" % (m.group(1), fmt_t(sc.ents[m.group(1)]["samples"][0]["t"]), m.group(2) or "")
        return a
    for k in ENT_FIELDS:
        if isinstance(c.get(k), str):
            c[k] = fix(c[k])
    if isinstance(c.get("a"), list):
        c["a"] = [fix(x) for x in c["a"]]
    return c, ch[0]


def anchor_kind(a):
    b = base_key(a)
    if re.match(r"^[PA]\d\d$", b):
        return "person"
    if re.match(r"^[VO]\d\d$", b):
        return "entity"
    if SPACE_RE.match(b):
        return "space"
    return "thing"


def touch_rule(cl):
    """r_touch: a touch between a person or animal (P.., A..) and anything that is not a V / O entity (a fixed thing,
    a space side, another person) is judged as next_to; a touch between a person and a V / O entity (held, carried,
    caught, pushed, ridden) and every touch without a person stay touch -> (claim, changed)"""
    if not MECH("r_touch"):
        return cl, False
    p = cl["pred"] if cl.get("t") in WRAP_TYPES else cl
    if p.get("t") != "touch":
        return cl, False
    ka, kb = anchor_kind(p["a"]), anchor_kind(p["b"])
    if "person" not in (ka, kb) or "entity" in (ka, kb):
        return cl, False
    c = copy.deepcopy(cl)
    (c["pred"] if c.get("t") in WRAP_TYPES else c)["t"] = "next_to"
    return c, True


def is_opening_class(cls):
    m2 = MECH("m2")
    w = norm_word(cls)
    return bool(m2) and any(x == w or x in w.split() for x in m2["opening_classes"])


def crossing_window(gt, cl, named):
    """m2 (a): a touch / next_to claim between X@t (an entity) and a named door or opening, where t lies in the window
    of a GT space transition of X whose from or to space is the opening's own GT space, is judged over that whole
    window (X touches it at some time while crossing); named: {GT key: class or {"class", "space"}} -> (claim, changed)"""
    if not MECH("m2") or cl.get("t") not in ("touch", "next_to"):
        return cl, False
    for es, ts in (("a", "b"), ("b", "a")):
        m = re.match(r"^([PAVO]\d\d)@(-?\d+(?:\.\d+)?)$", str(cl.get(es) or ""))
        thing = cl.get(ts)
        if not m or thing not in named or not is_opening_class(
                named[thing]["class"] if isinstance(named[thing], dict) else named[thing]):
            continue
        eid, t = m.group(1), float(m.group(2))
        nd = named[thing] if isinstance(named[thing], dict) else {"class": named[thing]}
        for tr in gt.get("space_transitions") or []:
            if tr.get("agent") != eid or not tr.get("time_range"):
                continue
            if nd.get("space") and nd["space"] not in (tr.get("from"), tr.get("to")):
                continue  # the opening is not on this crossing (its GT space is neither side of it)
            w0, w1 = float(tr["time_range"][0]), float(tr["time_range"][1])
            if w1 > w0 and w0 - 1e-6 <= t <= w1 + 1e-6:
                pred = copy.deepcopy(cl)
                pred[es] = eid
                return {"t": "during", "w": [w0, w1], "pred": pred}, True
    return cl, False


def in_transition(gt, eid, t):
    """m2 (b): True when t lies in the window of a GT space transition of eid (closed; a transition without a window
    is the instant t): no cam_frame check is derived for that subject at that time"""
    if not MECH("m2"):
        return False
    for tr in gt.get("space_transitions") or []:
        if tr.get("agent") != eid:
            continue
        w0, w1 = tr.get("time_range") or [tr["t"], tr["t"]]
        if float(w0) - 1e-6 <= t <= float(w1) + 1e-6:
            return True
    return False


def _poly_area(P):
    n = len(P)
    return abs(sum(P[i][0] * P[(i + 1) % n][1] - P[(i + 1) % n][0] * P[i][1] for i in range(n))) / 2.0


def _clip_convex(subj, clip):
    """Sutherland-Hodgman: polygon subj clipped by the convex polygon clip (counter-clockwise)"""
    out = [tuple(p) for p in subj]
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        inp, out = out, []
        if not inp:
            break

        def ins(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12

        def X(p, q):
            x1, y1 = p
            x2, y2 = q
            dd = (x1 - x2) * (a[1] - b[1]) - (y1 - y2) * (a[0] - b[0])
            if abs(dd) < 1e-15:
                return q
            u = ((x1 - a[0]) * (a[1] - b[1]) - (y1 - a[1]) * (a[0] - b[0])) / dd
            return (x1 + u * (x2 - x1), y1 + u * (y2 - y1))
        s0 = inp[-1]
        for e in inp:
            if ins(e):
                if not ins(s0):
                    out.append(X(s0, e))
                out.append(e)
            elif ins(s0):
                out.append(X(s0, e))
            s0 = e
    return out


def covered_space(sc, key):
    """r_cover: the space whose outline the static element key coincides with (each covers >= cover_frac of the
    other's area): the element is that space's own floor, deck or roof, not a separate thing; else None"""
    rc = MECH("r_cover")
    if not rc or key not in sc.elems:
        return None
    P = [tuple(p) for p in el_poly(sc.elems[key], sc.ph)]
    if sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1] for i in range(len(P))) < 0:
        P = P[::-1]
    aP = _poly_area(P)
    for sid in sorted(sc.spaces):
        Q = sc.spaces[sid].get("poly")
        if not Q or aP <= 0:
            continue
        aQ = _poly_area(Q)
        I = _clip_convex(Q, P)
        aI = _poly_area(I) if len(I) >= 3 else 0.0
        if aQ > 0 and aI / aQ >= rc["cover_frac"] - 1e-9 and aI / aP >= rc["cover_frac"] - 1e-9:
            return sid
    return None


def part_phrase(phrase):
    """r_part: the names phrase names a part of a place (its head word is one of part_heads: edge, end, side,
    surface ...) or ends in an adjective in -ed (not a noun ending listed in ed_nouns: bed, shed, reed ...): it does
    not name a thing a reader can draw"""
    rp = MECH("r_part")
    if not rp:
        return False
    w = re.findall(r"[a-z]+", str(phrase).lower())
    if not w:
        return False
    adj = bool(rp.get("adjective_ed")) and w[-1].endswith("ed") and not any(
        w[-1].endswith(n) for n in rp.get("ed_nouns") or [])
    return w[-1] in rp["part_heads"] or adj


def gt_sloped(gt):
    """space ids the GT marks sloped"""
    return [s["id"] for s in gt.get("spaces") or [] if s.get("sloped")]


def gt_ground_points(gt):
    """[(x, y, bottom z, footprint long side)]: the bottoms of all GT objects (entity objects included);
    attach_ground() picks each sloped space's points by its outline (prereg_v5.0.1 n1; the long side is read only by
    prereg_v5.0.3 n1b, ground_plane() uses x, y, z)"""
    pts = []
    for o in gt.get("objects") or []:
        try:
            pts.append((float(o["center"][0]), float(o["center"][1]),
                        float(o["center"][2]) - float(o["size"][2]) / 2, max(float(o["size"][0]), float(o["size"][1]))))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return pts


def gt_as_reader(gt):
    """GT read as a reader would write it: same geometry, every track linear between its samples (no known-span
    cut), static objects keyed by GT key. Used by the comparator self-test (GT-as-reader = all pass)."""
    d = scene_from_gt(gt)
    d["known"] = "all"
    d["frame"] = "gtr"
    return d


def shot_index(label, t, windows):
    """reader camera row -> shot index (0-based): the label is the shot's window as the prompt writes it ("0-5 s",
    "0-5"), or its 1-based position in the shot list; without a usable label the shot whose window holds t (a time on
    a cut belongs to the earlier shot only when no later shot starts there)."""
    W = windows or []
    if isinstance(label, bool):
        label = None
    if isinstance(label, (int, float)):
        i = int(label) - 1
        return i if 0 <= i < len(W) or not W else None
    if isinstance(label, str):
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", label)
        nums = re.findall(r"\d+(?:\.\d+)?", label)
        if m and W:
            a, b = float(m.group(1)), float(m.group(2))
            for i, (t0, t1) in enumerate(W):
                if abs(t0 - a) <= 0.05 and abs(t1 - b) <= 0.05:
                    return i
            return None
        if len(nums) == 1 and re.fullmatch(r"\s*(?:shot\s*)?\d+\s*", label, flags=re.I):
            return shot_index(int(nums[0]), t, W)
    if t is None or not W:
        return None
    inside_ = [i for i, (t0, t1) in enumerate(W) if t0 - 1e-6 <= t <= t1 + 1e-6]
    if not inside_:
        return None
    later = [i for i in inside_ if abs(W[i][0] - t) <= 1e-6]
    return later[0] if later else inside_[0]


def scene_from_reader(rd, ent_ids=None, windows=None, duration=None, sloped=None):
    """reader JSON (READER.md schema) -> plain scene; object rows keyed r000.. with their phrase / class labels.
    windows: the item's shot windows [(t0, t1), ...] (camera rows name their shot by window); duration: clip length
    (event probes are clamped to it); sloped: the space ids the GT marks sloped (their ground plane is fitted to the
    space's stated "ground" points and the bottoms of the reader's object rows on it, prereg_v5.0.1 note n1)."""
    spaces = {}
    for sid, s in (rd.get("spaces") if isinstance(rd.get("spaces"), dict) else {}).items():
        if not isinstance(s, dict):
            continue
        try:
            poly = [[float(p[0]), float(p[1])] for p in s.get("outline") or []]
        except (TypeError, ValueError, IndexError):
            poly = []
        spaces[sid] = {"poly": poly if len(poly) >= 3 else None, "bbox": None,
                       "floor": float(s.get("floor_z") or 0.0) if isinstance(s.get("floor_z"), (int, float)) else 0.0}
    gpts = []
    for sid, s in (rd.get("spaces") if isinstance(rd.get("spaces"), dict) else {}).items():
        # a sloping space's stated ground heights (READER.md "ground": [[x, y, z], ...])
        if sid in spaces and isinstance(s, dict) and isinstance(s.get("ground"), list):
            for q in s["ground"]:
                try:
                    gpts.append((float(q[0]), float(q[1]), float(q[2])))
                except (TypeError, ValueError, IndexError):
                    continue
    tmp = Scene({"spaces": spaces})
    elems, idrow = {}, {}
    n = 0
    for row in rd.get("objects") if isinstance(rd.get("objects"), list) else []:
        if not isinstance(row, dict):
            continue
        try:
            c = [float(x) for x in row["center"]]
            sz = [abs(float(x)) for x in row["size"]]
            yaw = float(row.get("yaw_deg") or 0.0)
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if len(c) < 3 or len(sz) < 3:
            continue
        long_, short = max(sz[0], sz[1]), min(sz[0], sz[1])
        el = {"c": c[:2], "long": long_, "short": short, "axis": (yaw if sz[0] >= sz[1] else yaw + 90.0) % 180.0,
              "front": yaw % 360.0, "h": sz[2], "z0": c[2] - sz[2] / 2, "z1": c[2] + sz[2] / 2,
              "space": tmp.space_of(c[:2]), "cls": str(row.get("class") or ""),
              "labels": [norm_word(row.get("phrase")), norm_word(row.get("class"))],
              "phrase": str(row.get("phrase") or ""), "person": False}
        ph = str(row.get("phrase") or "").strip()
        if ENT_RE.match(ph):
            idrow.setdefault(ph, el)
            continue
        elems["r%03d" % n] = el
        n += 1
    ents = {}
    for eid, e in (rd.get("entities") if isinstance(rd.get("entities"), dict) else {}).items():
        if not ENT_RE.match(str(eid)) or not isinstance(e, dict):
            continue
        person = eid[0] in "PA"
        base = idrow.get(eid)
        smp = []
        rows = [s for s in (e.get("samples") if isinstance(e.get("samples"), list) else []) if isinstance(s, dict)
                and isinstance(s.get("t"), (int, float))]
        cur_size = None
        for s in sorted(rows, key=lambda s: float(s["t"])):
            try:
                t = float(s["t"])
                p = [float(s["p"][0]), float(s["p"][1])]
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            yaw = s.get("yaw_deg")
            yaw = float(yaw) % 360.0 if isinstance(yaw, (int, float)) else None
            if person or base is None:
                el = {"c": p, "long": 0.5, "short": 0.5, "axis": None if yaw is None else yaw % 180.0,
                      "front": yaw, "h": 1.75 if person else None, "z0": None, "z1": None,
                      "space": tmp.space_of(p), "cls": "person" if person else "", "labels": [], "person": person}
                if person:
                    el["long"] = el["short"] = 2 * J("person_radius_frac") * 1.75
            else:
                el = dict(base)
                el["c"] = p
                el["space"] = tmp.space_of(p)
                if yaw is not None:
                    # the entity's own row gives its box and start yaw; a sample yaw turns the box with it
                    el["front"] = yaw
                    el["axis"] = (base["axis"] + (yaw - base["front"])) % 180.0
                if MECH("m5"):
                    # prereg_v5.0.2 m5: a sample may state the thing's box anew ("size": [a, b, h] as in an object
                    # row, a along its yaw); it holds from that sample on
                    try:
                        sz = [abs(float(x)) for x in s["size"]] if s.get("size") is not None else None
                        sz = sz if sz is not None and len(sz) == 3 else None
                    except (TypeError, ValueError):
                        sz = None
                    cur_size = sz or cur_size
                    if cur_size:
                        fy = el["front"] if el.get("front") is not None else base["front"]
                        el["long"], el["short"], el["h"] = max(cur_size[0], cur_size[1]), min(cur_size[0], cur_size[1]), \
                            cur_size[2]
                        el["axis"] = (fy if cur_size[0] >= cur_size[1] else fy + 90.0) % 180.0
                        if smp:
                            el["z0"] = el["z1"] = None
                        elif sz:
                            zc = (base["z0"] + base["z1"]) / 2.0
                            el["z0"], el["z1"] = zc - sz[2] / 2.0, zc + sz[2] / 2.0
                if smp:
                    el["z0"] = el["z1"] = None
            el["t"] = t
            smp.append(el)
        if not smp and base is not None:
            smp = [dict(base, t=0.0)]
        if smp:
            # one sample (or only the object row): the reader states one place for the whole clip
            ents[eid] = {"cls": "", "person": person, "samples": smp, "static": len(smp) == 1}
    cams = {}
    for row in rd.get("cameras") if isinstance(rd.get("cameras"), list) else []:
        if not isinstance(row, dict):
            continue
        try:
            pos = [float(x) for x in row["pos"]]
            look = [float(x) for x in row["look"]]
            t = float(row["t"])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if len(pos) != 3 or len(look) != 3:
            continue
        si = shot_index(row.get("shot"), t, windows)
        if si is None:
            continue
        cams.setdefault(str(si), []).append({"t": t, "pos": pos, "look": look, "hfov": lens_hfov(row.get("lens"))})
    d = {"frame": "reader", "spaces": spaces, "elems": elems, "ents": ents, "cams": cams,
         "ph": J("person_h_reader"), "duration": duration, "known": "all", "max_gap_s": None, "space_tags": False,
         "axis_fallback": False}
    if sloped:
        # stated ground spots stay (x, y, z); object rows carry their long side (prereg_v5.0.3 n1b)
        attach_ground(d, sloped, gpts + [(el["c"][0], el["c"][1], el["z0"], el["long"]) for el in list(elems.values()) +
                                         list(idrow.values()) if el.get("z0") is not None])
    return d


# ------------------------------------------------------------------ matching
class Matcher:
    """GT key -> scene element binding: maximum cardinality among compatible candidates, then a deterministic local
    search that keeps a change only if it strictly raises the number of claims that hold (PREREG matching)."""

    def __init__(self, sc, cand, claims, max_pass=30):
        self.sc, self.cand, self.claims = sc, cand, claims
        self.keys = sorted(cand)
        self.max_pass = max_pass
        self.by_key = {}
        for i, cl in enumerate(claims):
            for a in anchors_of(cl):
                k = base_key(a)
                if k in cand:
                    self.by_key.setdefault(k, set()).add(i)
            if cl["t"] == "count":
                for k in self.keys:
                    if fnmatch.fnmatch(k, cl["a"]):
                        self.by_key.setdefault(k, set()).add(i)

    def count_fn(self, bind):
        def f(glob):
            ks = [k for k in self.keys if fnmatch.fnmatch(k, glob)]
            bound = {bind[k] for k in ks if bind.get(k)}
            used = {v for v in bind.values() if v}
            extra = set()
            for k in ks:
                for e, _ in self.cand.get(k, []):
                    if e not in used:
                        extra.add(e)
            return len(bound) + len(extra)
        return f

    def _eval(self, bind, idx):
        cf = self.count_fn(bind)
        return sum(1 for i in idx if judge(self.sc, bind, self.claims[i], count_fn=cf)["ok"])

    def initial(self):
        bind = {k: None for k in self.keys}
        owner = {}
        order = sorted(self.keys, key=lambda k: (len(self.cand[k]), k))

        def aug(k, seen):
            for e, _ in self.cand[k]:
                if e in seen:
                    continue
                seen.add(e)
                if e not in owner or aug(owner[e], seen):
                    owner[e] = k
                    bind[k] = e
                    return True
            return False
        for k in order:
            aug(k, set())
        return {k: v for k, v in bind.items()}

    def order_seed(self, bind):
        """second deterministic start: keys that share one candidate list and that order claims chain along an axis
        take those candidates in the claimed order (candidates sorted along that axis, evenly spaced when there are
        more candidates than keys); everything else as in bind"""
        bind = dict(bind)
        groups = {}
        for k in self.keys:
            groups.setdefault(tuple(e for e, _ in self.cand[k]), []).append(k)
        for cands, ks in sorted(groups.items(), key=lambda kv: kv[1][0]):
            if len(ks) < 2 or not cands:
                continue
            kset = set(ks)
            edges = {}
            for cl in self.claims:
                if cl["t"] == "order" and len(cl["a"]) == 2 and cl["a"][0] in kset and cl["a"][1] in kset:
                    edges.setdefault(cl["axis"], []).append(tuple(cl["a"]))
            if not edges:
                continue
            axis = max(sorted(edges), key=lambda x: len(edges[x]))
            ax, sg = ORDER_AX[axis]
            indeg = {k: 0 for k in ks}
            nxt = {k: [] for k in ks}
            for a0, a1 in edges[axis]:
                nxt[a0].append(a1)
                indeg[a1] += 1
            topo, ready = [], sorted(k for k in ks if indeg[k] == 0)
            while ready:
                k = ready.pop(0)
                topo.append(k)
                for k2 in sorted(nxt[k]):
                    indeg[k2] -= 1
                    if indeg[k2] == 0:
                        ready.append(k2)
                ready.sort()
            if len(topo) < len(ks):          # a cycle: leave this group to the search
                continue
            for k in ks:
                bind[k] = None
            taken = {v for v in bind.values() if v}
            rows = sorted((e for e in cands if e not in taken), key=lambda e: (sg * self.sc.elems[e]["c"][ax], e))
            n, m = len(topo), len(rows)
            if m >= n:
                pick = [rows[int(round(i * (m - 1) / (n - 1)))] for i in range(n)] if n > 1 else rows[:1]
            else:
                pick = rows + [None] * (n - m)
            for k, e in zip(topo, pick):
                bind[k] = e
        return bind

    def solve(self):
        b0 = self.initial()
        best_bind, best_n = self._search(b0)
        b1 = self.order_seed(b0)
        if b1 != b0:
            bb, nn = self._search(b1)
            if nn > best_n:
                best_bind, best_n = bb, nn
        return best_bind

    def _search(self, bind):
        total = list(range(len(self.claims)))
        best = self._eval(bind, total)
        for _ in range(self.max_pass):
            improved = False
            for k in self.keys:
                cur = bind[k]
                for e, _ in self.cand[k]:
                    if e == cur:
                        continue
                    k2 = next((kk for kk, vv in bind.items() if vv == e), None)
                    if k2 is not None and cur is not None and cur not in {x for x, _ in self.cand[k2]}:
                        continue
                    if k2 is not None and cur is None:
                        continue
                    trial = dict(bind)
                    trial[k] = e
                    if k2 is not None:
                        trial[k2] = cur
                    idx = self.by_key.get(k, set()) | (self.by_key.get(k2, set()) if k2 else set())
                    old = self._eval(bind, idx)
                    new = self._eval(trial, idx)
                    if new > old:
                        bind = trial
                        best += new - old
                        improved = True
                        cur = e
            if not improved:
                break
        return bind, best


def candidates(gt_items, sc, compat):
    """gt_items: {key: {"labels": [...], "space": sid}}; compat(gt_item, el) -> credit; -> {key: [(eid, credit)]}"""
    out = {}
    for k, g in gt_items.items():
        lst = []
        for eid in sorted(sc.elems):
            el = sc.elems[eid]
            c = compat(g, el)
            if c <= 0:
                continue
            if g.get("space") and el.get("space") and el["space"] != g["space"]:
                continue
            lst.append((eid, c))
        lst.sort(key=lambda x: (-x[1], x[0]))
        out[k] = lst
    return out


# ------------------------------------------------------------------ chance baseline (model blocks)
def permuted(sc_dict, bind, seed):
    """copy of a plain scene with the centres of the bound static elements permuted within each element space"""
    d = copy.deepcopy(sc_dict)
    rng = random.Random(seed)
    groups = {}
    for k, e in bind.items():
        if e and e in d["elems"]:
            groups.setdefault(d["elems"][e].get("space"), []).append(e)
    for sp in sorted(groups, key=lambda x: str(x)):
        es = sorted(set(groups[sp]))
        cs = [list(d["elems"][e]["c"]) for e in es]
        rng.shuffle(cs)
        for e, c in zip(es, cs):
            d["elems"][e]["c"] = c
    return d


# ------------------------------------------------------------------ text form and test transforms
def claim_text(cl):
    """one-line text of a claim (reports and review pages)"""
    t = cl["t"]
    if t == "event":
        pr = (" probes %s / %s s" % (fmt_t(cl["probe"][0]), fmt_t(cl["probe"][1]))) if cl.get("probe") else ""
        return "%s [%s-%s s, tol %s%s]: %s" % (cl["ev"], fmt_t(cl.get("t0", 0)), fmt_t(cl.get("t1", 0)),
                                               fmt_t(cl.get("tol", 0)), pr, claim_text(cl["pred"]))
    if t == "during":
        return "at some time in %s-%s s: %s" % (fmt_t(cl["w"][0]), fmt_t(cl["w"][1]), claim_text(cl["pred"]))
    neg = "not " if cl.get("neg") else ""
    if t == "in_space":
        return "%s %sin %s" % (cl["a"], neg, cl["s"])
    if t == "side":
        return "%s %s of %s" % (cl["a"], cl["dir"], cl["of"])
    if t == "part":
        return "%s in %s part of %s" % (cl["a"], cl["w"], cl["in"])
    if t == "order":
        return "%s (%s)" % (" < ".join(cl["a"]), cl["axis"])
    if t in ("touch", "next_to", "on", "in"):
        return "%s %s %s" % (cl["a"], {"touch": "touching", "next_to": "next to", "on": "on", "in": "in"}[t], cl["b"])
    if t == "facing":
        return "%s facing %s" % (cl["a"], cl["dir"] if "dir" in cl else "toward " + cl["toward"])
    if t == "axis":
        return "%s long axis %s" % (cl["a"], cl["w"])
    if t == "rel":
        return "%s %s %s" % (cl["a"], {"front": "in front of", "behind": "behind", "left": "left of",
                                       "right": "right of"}[cl["w"]], cl["b"])
    if t == "count":
        return "count %s = %s" % (cl["a"], cl["n"])
    if t == "moving":
        return "%s %smoving" % (cl["a"], neg)
    return json.dumps(cl, sort_keys=True)


def check_text(ck):
    t = ck["t"]
    w = "%s-%s s" % (fmt_t(ck["window"][0]), fmt_t(ck["window"][1])) if ck.get("window") else "shot %s" % ck["si"]
    if t == "cam_move":
        return "%s %s-%s s move %s %s" % (w, fmt_t(ck["t0"]), fmt_t(ck["t1"]), ck["type"], ck.get("dir") or "")
    head = "%s @%s s %s" % (w, fmt_t(ck["at"]), t[4:])
    if t in ("cam_lens", "cam_height"):
        return "%s %s" % (head, ck["w"])
    if t == "cam_dir":
        return "%s bearing %s" % (head, ck["bearing"])
    if t == "cam_side":
        return "%s of %s at bearing %s" % (head, ck["subject"], ck["bearing"])
    if t == "cam_frame":
        return "%s %s in frame" % (head, ck["subject"])
    return head


MIRROR_DIR = {"east": "west", "west": "east"}
MIRROR_PART = {"west": "east", "east": "west", "nw": "ne", "ne": "nw", "sw": "se", "se": "sw"}
MIRROR_AXIS = {"west-east": "east-west", "east-west": "west-east"}
MIRROR_REL = {"left": "right", "right": "left"}
MIRROR_LR = {"left": "right", "right": "left"}


def _mirror_anchor(a):
    m = re.match(r"^(.*):(east|west)(_line)?$", a)
    if m:
        return "%s:%s%s" % (m.group(1), MIRROR_DIR[m.group(2)], m.group(3) or "")
    return a


def mirror_claim(cl):
    """the claim that holds on a scene exactly when cl holds on its mirror image (x -> -x)"""
    c = copy.deepcopy(cl)
    if c["t"] in WRAP_TYPES:
        c["pred"] = mirror_claim(c["pred"])
        return c
    for k in ENT_FIELDS:
        if isinstance(c.get(k), str):
            c[k] = _mirror_anchor(c[k])
    if isinstance(c.get("a"), list):
        c["a"] = [_mirror_anchor(x) for x in c["a"]]
    if c["t"] == "side":
        c["dir"] = MIRROR_DIR.get(c["dir"], c["dir"])
    elif c["t"] == "part":
        c["w"] = MIRROR_PART.get(c["w"], c["w"])
    elif c["t"] == "order":
        c["axis"] = MIRROR_AXIS.get(c["axis"], c["axis"])
    elif c["t"] == "facing" and "dir" in c:
        b = (180.0 - COMPASS[c["dir"]]) % 360.0
        c["dir"] = min(COMPASS, key=lambda k: angdiff(COMPASS[k], b))
    elif c["t"] == "rel":
        c["w"] = MIRROR_REL.get(c["w"], c["w"])
    return c


def mirror_check(ck):
    c = copy.deepcopy(ck)
    if "bearing" in c:
        c["bearing"] = round((180.0 - c["bearing"]) % 360.0, 6)
    if c["t"] == "cam_move" and (c.get("dir") or "") in MIRROR_LR:
        c["dir"] = MIRROR_LR[c["dir"]]
    return c


def ew_sensitive(cl):
    """True when the claim names an east / west direction (its mirror claim differs)"""
    return mirror_claim(cl) != cl


def _mx(p):
    return None if p is None else [None if p[0] is None else -p[0]] + list(p[1:])


def mirror_scene(d):
    """plain scene mirrored x -> -x (bearings b -> 180 - b, axes a -> 180 - a)"""
    d = copy.deepcopy(d)
    for s in d.get("spaces", {}).values():
        if s.get("poly"):
            s["poly"] = [[-p[0], p[1]] for p in reversed(s["poly"])]
        if s.get("bbox"):
            x0, y0, x1, y1 = s["bbox"]
            s["bbox"] = [-x1, y0, -x0, y1]
        if s.get("ground"):
            s["ground"] = [s["ground"][0], -s["ground"][1], s["ground"][2]]
        if s.get("gpts"):
            s["gpts"] = [(-q[0], q[1], q[2]) for q in s["gpts"]]

    def el(e):
        e["c"] = [-e["c"][0], e["c"][1]]
        if e.get("axis") is not None:
            e["axis"] = (180.0 - e["axis"]) % 180.0
        if e.get("front") is not None:
            e["front"] = (180.0 - e["front"]) % 360.0
    for e in d.get("elems", {}).values():
        el(e)
    for e in d.get("ents", {}).values():
        for smp in e["samples"]:
            el(smp)
    for rows in d.get("cams", {}).values():
        for r in rows:
            for k in ("pos", "look", "fwd"):
                if r.get(k) is not None:
                    r[k] = _mx(r[k])
            if r.get("up") is not None:
                r["up"] = _mx(r["up"])
    return d


def _scene_box(d):
    xs, ys = [], []
    for s in d.get("spaces", {}).values():
        b = s.get("bbox")
        if b is None and s.get("poly"):
            b = [min(p[0] for p in s["poly"]), min(p[1] for p in s["poly"]),
                 max(p[0] for p in s["poly"]), max(p[1] for p in s["poly"])]
        if b:
            xs += [b[0], b[2]]
            ys += [b[1], b[3]]
    return [min(xs), min(ys), max(xs), max(ys)] if xs else [0.0, 0.0, 0.0, 0.0]


def shifted_elem(d, eid):
    """copy of a plain scene with static element eid point-reflected through the centre of its space's box (the
    scene box without a space); when that moves it less than its own size u, moved by half the box's longer
    extent instead (towards the far side). -> (scene, moved distance)"""
    d = copy.deepcopy(d)
    e = d["elems"][eid]
    sp = (d.get("spaces") or {}).get(e.get("space") or "")
    b = None
    if sp:
        b = sp.get("bbox") or ([min(p[0] for p in sp["poly"]), min(p[1] for p in sp["poly"]),
                                max(p[0] for p in sp["poly"]), max(p[1] for p in sp["poly"])] if sp.get("poly") else None)
    b = b or _scene_box(d)
    mx, my = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    c = e["c"]
    n = [2 * mx - c[0], 2 * my - c[1]]
    u = el_u(e, float(d.get("ph") or 1.75))
    if math.hypot(n[0] - c[0], n[1] - c[1]) < u:
        ax = 0 if (b[2] - b[0]) >= (b[3] - b[1]) else 1
        half = ((b[2] - b[0]) if ax == 0 else (b[3] - b[1])) / 2
        n = list(c)
        n[ax] += half if c[ax] <= (mx, my)[ax] else -half
    e["c"] = n
    return d, math.hypot(n[0] - c[0], n[1] - c[1])


def shifted_ent(d, eid):
    """copy of a plain scene with every sample of entity eid point-reflected through the centre of the scene box"""
    d = copy.deepcopy(d)
    b = _scene_box(d)
    mx, my = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    for smp in d["ents"][eid]["samples"]:
        smp["c"] = [2 * mx - smp["c"][0], 2 * my - smp["c"][1]]
        if smp.get("front") is not None:
            smp["front"] = (smp["front"] + 180.0) % 360.0
    return d


def mentions(cl, key):
    """the claim names key (a GT object key or an entity id) as an anchor, or counts a glob that covers it"""
    if cl["t"] == "count":
        return fnmatch.fnmatch(key, cl["a"])
    return any(base_key(a) == key for a in anchors_of(cl))

