"""Top-down layer: per-space alignment, A1 (what exists), A2 (where / which way), A3 (spaces and openings)."""
from __future__ import annotations

import math

import numpy as np

from . import fixes, geom
from .model import CONNECTOR_CLASSES, FLOOR_CLASSES, WALL_CLASSES

LARGE_GROUPS = {"furniture", "appliance", "industrial", "vehicle", "sports"}
OBJECT_KINDS = ("object", "untagged_object")


# --------------------------------------------------------------------------------------------- class credit
class Credit:
    def __init__(self, params):
        self.sim = {}
        for grp in params["class_similar"]:
            for a in grp:
                for b in grp:
                    if a != b:
                        self.sim.setdefault(a, set()).add(b)
        self.similar = params["similar_credit"]
        self.untagged = params["untagged_credit"]

    def __call__(self, gt_cls, inst):
        if inst.cls is None:
            return self.untagged if inst.kind == "untagged_object" else 0.0
        if inst.cls == gt_cls:
            return 1.0
        if inst.cls in self.sim.get(gt_cls, ()):
            return self.similar
        return 0.0


# ------------------------------------------------------------------------------------------------ alignment
class Align:
    def __init__(self, space, sim, mode, n_inliers, floor_m, floor_gt, L, score=0.0, source=""):
        self.space, self.sim, self.mode, self.n_inliers = space, sim, mode, n_inliers
        self.floor_m, self.floor_gt, self.L, self.score, self.source = floor_m, floor_gt, L, score, source

    def xy(self, P):
        return self.sim.apply(np.asarray(P, float))

    def z(self, zm):
        return (np.asarray(zm, float) - self.floor_m) * self.sim.s + self.floor_gt

    def h(self, zm):
        """Height above the space floor in GT metres."""
        return (np.asarray(zm, float) - self.floor_m) * self.sim.s

    def yaw(self, ym):
        return ym + self.sim.theta

    def dir3(self, v):
        v = np.asarray(v, float)
        xy = geom.rot2(self.sim.theta) @ v[:2]
        return np.array([xy[0], xy[1], v[2]])

    @property
    def ok(self):
        return self.mode in ("aligned", "global")

    def to_dict(self):
        return {"space": self.space, "mode": self.mode, "source": self.source, "inliers": self.n_inliers,
                "sim": self.sim.to_dict(), "floor_model": round(float(self.floor_m), 3),
                "floor_gt": round(float(self.floor_gt), 3), "L": round(self.L, 3), "score": round(self.score, 3)}


def _hyp_score(sim, G, Gw, Mxy, C, sigma, anchors):
    T = sim.apply(Mxy) if len(Mxy) else np.zeros((0, 2))
    s = 0.0
    if len(G) and len(T):
        D2 = ((G[:, None, :] - T[None]) ** 2).sum(2)
        s += float((Gw * (C * np.exp(-D2 / (2 * sigma ** 2))).max(1)).sum())
    for g, m, w in anchors:
        d2 = float(((np.asarray(g) - sim.apply(np.asarray(m)[None])[0]) ** 2).sum())
        s += 2.0 * w * math.exp(-d2 / (2 * sigma ** 2))
    return s


def align(gt_objs, insts, credit, anchors, L, params, rng):
    """Best similarity (model -> GT) from class-aware hypotheses + Hungarian/Umeyama refinement."""
    sigma = max(0.15, params["align_sigma_frac"] * L)
    gate = params["align_gate_frac"] * L
    sb = tuple(params["align_scale_bounds"])
    G = np.array([o.center[:2] for o in gt_objs]) if gt_objs else np.zeros((0, 2))
    Gw = np.array([(2.0 if o.named else 1.0) * o.conf for o in gt_objs]) if gt_objs else np.zeros(0)
    Mxy = np.array([m.center2 for m in insts]) if insts else np.zeros((0, 2))
    C = np.array([[credit(o.cls, m) for m in insts] for o in gt_objs]) if gt_objs and insts else np.zeros((len(G), len(Mxy)))
    hyps = []
    if len(anchors) >= 2:
        hyps.append(geom.umeyama2([a[1] for a in anchors], [a[0] for a in anchors], [a[2] for a in anchors], sb))
    for i, o in enumerate(gt_objs):
        for j, m in enumerate(insts):
            if C[i, j] < 0.75:
                continue
            ys = [o.long_yaw - m.long_yaw, o.long_yaw - m.long_yaw + 180]
            if o.symmetry == 4 or m.long < 1.15 * max(m.short, 1e-6):
                ys += [ys[0] + 90, ys[0] + 270]
            if o.front_yaw is not None and m.facing is not None:
                ys = [o.front_yaw - m.facing]
            for th in ys:
                t = o.center[:2] - geom.rot2(th) @ m.center2
                hyps.append(geom.Sim2(1.0, th, t))
    for g, mm, w in anchors:
        for th in range(0, 360, 15):
            hyps.append(geom.Sim2(1.0, th, np.asarray(g) - geom.rot2(th) @ np.asarray(mm)))
    pairs_g = [(i, j) for i in range(len(G)) for j in range(i + 1, len(G)) if np.linalg.norm(G[i] - G[j]) > 0.4]
    if len(pairs_g) > 300:
        pairs_g = [pairs_g[k] for k in rng.choice(len(pairs_g), 300, replace=False)]
    budget = 6000
    for i, j in pairs_g:
        ci = np.where(C[i] >= 0.75)[0]
        cj = np.where(C[j] >= 0.75)[0]
        dg = np.linalg.norm(G[i] - G[j])
        for a in ci:
            for b in cj:
                if a == b:
                    continue
                dm = np.linalg.norm(Mxy[a] - Mxy[b])
                if dm < 1e-6 or not 0.5 < dg / dm < 2.0:
                    continue
                h = geom.sim_from_pairs(Mxy[a], Mxy[b], G[i], G[j])
                if h is not None:
                    hyps.append(h)
                    budget -= 1
            if budget <= 0:
                break
        if budget <= 0:
            break
    if not hyps:
        return None
    scores = [_hyp_score(h, G, Gw, Mxy, C, sigma, anchors) for h in hyps]
    order = np.argsort(scores)[::-1][:8]
    best = None
    s20 = fixes.on("S20")
    for k in order:
        sim = hyps[int(k)]
        for _ in range(5):
            T = sim.apply(Mxy) if len(Mxy) else np.zeros((0, 2))
            src, dst, w = [], [], []
            matches = []
            if len(G) and len(T):
                D = np.linalg.norm(G[:, None] - T[None], axis=2)
                cost = np.where((C > 0) & (D <= gate), D / sigma - np.log(np.maximum(C, 1e-6)), np.inf)
                for i, j in geom.hungarian(cost):
                    matches.append((i, j, float(D[i, j])))
                    src.append(Mxy[j])
                    dst.append(G[i])
                    w.append(Gw[i] * C[i, j])
            for g, mm, wa in anchors:
                # 5.0.12 (S20): only anchors within the gate of the current hypothesis enter the refit
                if s20 and np.linalg.norm(np.asarray(g) - sim.apply(np.asarray(mm)[None])[0]) > gate:
                    continue
                src.append(mm)
                dst.append(g)
                w.append(2.0 * wa)
            if len(src) < 2:
                break
            if s20 and _n_distinct(dst) < 3:
                # S20: scale is refit only from >= 3 pairs (amended 3: >= 3 distinct GT points)
                new = geom.umeyama2(src, dst, w, sb, fix_scale=sim.s)
            else:
                new = geom.umeyama2(src, dst, w, sb)
                if s20 and min(abs(new.s - sb[0]), abs(new.s - sb[1])) < 1e-9:
                    break                                              # S20: a refit on a scale bound is rejected
            if abs(new.theta - sim.theta) < 0.05 and np.linalg.norm(new.t - sim.t) < 1e-3 and abs(new.s - sim.s) < 1e-4:
                sim = new
                break
            sim = new
        sc = _hyp_score(sim, G, Gw, Mxy, C, sigma, anchors)
        if best is None or sc > best[1]:
            T = sim.apply(Mxy) if len(Mxy) else np.zeros((0, 2))
            inl = [(mm, g, wa) for g, mm, wa in anchors
                   if np.linalg.norm(np.asarray(g) - sim.apply(np.asarray(mm)[None])[0]) <= gate]
            nin = _n_distinct([g for _, g, _ in inl]) if s20 else len(inl)   # S20 amended 3: distinct GT points
            ms = []
            if len(G) and len(T):
                D = np.linalg.norm(G[:, None] - T[None], axis=2)
                cost = np.where((C > 0) & (D <= gate), D / sigma - np.log(np.maximum(C, 1e-6)), np.inf)
                ms = [(i, j, float(D[i, j])) for i, j in geom.hungarian(cost)]
            # 5.0.12 (D17): the inlier pairs (model -> GT, weight) travel with the result for a 2-pair rigid fit
            pairs = [(np.asarray(mm, float), np.asarray(g, float), 2.0 * wa) for mm, g, wa in inl]
            if s20:   # amended 3: one pair per distinct GT point (the first), so two pairs are two points
                pairs = [q for k, q in enumerate(pairs) if not any(np.linalg.norm(q[1] - r[1]) <= 1e-6
                                                                   for r in pairs[:k])]
            pairs += [(Mxy[j], G[i], Gw[i] * C[i, j]) for i, j, _ in ms]
            best = (sim, sc, nin + len(ms), ms, pairs)
    return best


def _n_distinct(pts):
    """5.0.12 (S20, amended 3, G3 guard): number of distinct 2D points (within 1e-6 m). A GT point entered twice
    (a carrier and the object it carries at the track start, or a repeated anchor) is one constraint."""
    out = []
    for q in pts:
        q = np.asarray(q, float)[:2]
        if not any(np.linalg.norm(q - r) <= 1e-6 for r in out):
            out.append(q)
    return len(out)


def model_floor(insts, pieces, space=None):
    fl = [p for p in pieces if p["cls"] in FLOOR_CLASSES and (space is None or p["space"] in (space, None))
          and (p["hi"][2] - p["lo"][2]) < 0.6]
    zs = [p["hi"][2] for p in fl]
    if zs and fixes.on("S21"):
        # 5.0.12 (S21): area-weighted median of the floor piece tops, so that small raised / sunk pieces (tracks,
        # kerbs, gullies) do not move the floor of the whole space
        w = np.array([max(1e-6, geom.min_area_rect(p["P"][:, :2])[2] * geom.min_area_rect(p["P"][:, :2])[3]) for p in fl])
        o = np.argsort(zs)
        cw = np.cumsum(w[o])
        return float(np.asarray(zs)[o][int(np.searchsorted(cw, 0.5 * cw[-1]))])
    if zs:
        return float(np.median(zs))
    fl = [m.top for m in insts if m.kind == "untagged_floor"]
    if fl:
        return float(np.median(fl))
    bs = [m.bottom for m in insts if m.kind in OBJECT_KINDS]
    if bs:
        return float(np.percentile(bs, 5))
    return 0.0


def run_alignment(ctx):
    gt, md, P = ctx.gt, ctx.model, ctx.P
    rng = np.random.default_rng(7)
    credit = ctx.credit
    tagged_spaces = {m.space for m in md.instances if m.space} | {p["space"] for p in md.layout_pieces if p["space"]}
    use_tags = bool(tagged_spaces & set(gt.space_by))
    ctx.use_space_tags = use_tags

    def cands(space_id):
        out = [m for m in md.instances if m.kind in OBJECT_KINDS + ("opening",)]
        if use_tags and len(gt.spaces) > 1:
            out = [m for m in out if m.space in (space_id, None)]
        return out

    def anchors_for(space_id):
        out = []
        for o in gt.objects:
            if o.id and (o.space == space_id or space_id is None):
                bi = ctx.entity_instance(o.id)
                if bi is not None:
                    out.append((o.center[:2], bi.center2, o.conf))
        for tid, tr in gt.tracks.items():
            if not len(tr.t):
                continue
            sp = tr.space[0] or gt.spaces[0].id
            if space_id is not None and sp != space_id:
                continue
            b = md.body.get(tid)
            if b is None:
                continue
            ai = md.t_to_ai(tr.t[0])
            idx = [k for k in range(md.A) if b.present[k]]
            if not idx:
                continue
            k = ai if b.present[ai] else min(idx, key=lambda q: abs(q - ai))
            if abs(md.an_t[k] - tr.t[0]) > 1.0:
                continue
            if fixes.on("S20") and any(np.linalg.norm(np.asarray(g, float)[:2] - np.asarray(tr.p[0], float)[:2]) <= 1e-6
                                       and np.linalg.norm(np.asarray(m, float)[:2] - np.asarray(b.center[k], float)[:2])
                                       <= 1e-6 for g, m, _ in out):
                continue   # 5.0.12 (S20, amended 3): the track start repeats the entity's own object box anchor
            out.append((tr.p[0], b.center[k], 1.0))
        return out

    aligns = {}
    # global (all spaces jointly) as the fallback
    allobjs = [o for o in gt.objects if gt.kind_of(o) in ("object", "opening") and not o.id]
    Lg = gt.union_L([s.id for s in gt.spaces])
    g_best = align(allobjs, [m for m in md.instances if m.kind in OBJECT_KINDS + ("opening",)], credit,
                   anchors_for(None), Lg, P, rng)
    for s in gt.spaces:
        objs = [o for o in gt.objects if o.space == s.id and gt.kind_of(o) in ("object", "opening") and not o.id]
        L = s.L
        anc = anchors_for(s.id)
        best = align(objs, cands(s.id), credit, anc, L, P, rng) if len(gt.spaces) > 1 else g_best
        src = "space" if len(gt.spaces) > 1 else "global"
        if best is None or best[2] < P["min_anchors"]:
            if g_best is not None and g_best[2] >= P["min_anchors"] and len(gt.spaces) > 1 \
                    and gt.space_frames != "separate":
                best, src = g_best, "global"
        floor_m = model_floor(md.instances, md.layout_pieces, s.id if use_tags else None)
        if best is None:
            aligns[s.id] = Align(s.id, geom.Sim2(), "none", 0, floor_m, s.floor_z, L, 0.0, "none")
            continue
        sim, sc, nin = best[:3]
        mode = ("aligned" if src == "space" else "global") if nin >= P["min_anchors"] else "relations"
        if mode == "relations" and fixes.on("D17") and nin == 2 and len(best) > 4 and len(best[4]) == 2 \
                and _pair_sep_ok(best[4], Lg if src == "global" else L, P):
            # 5.0.12 (D17): two inlier pairs fix rotation and translation; a rigid fit (s = 1) on them is a frame.
            # Amended 10-19 (G3 guard): only when the two pairs' separations agree (_pair_sep_ok)
            pr = best[4]
            sim = geom.umeyama2([q[0] for q in pr], [q[1] for q in pr], [q[2] for q in pr], fix_scale=1.0)
            mode = "aligned" if src == "space" else "global"
        if mode == "relations":
            sim = geom.Sim2(1.0, sim.theta, sim.t)   # scale is not identifiable from < 3 anchors
        aligns[s.id] = Align(s.id, sim, mode, nin, floor_m, s.floor_z, L, sc, src)
    ctx.aligns = aligns
    claim_spaces(ctx)
    return aligns


def _pair_sep_ok(pr, L, P):
    """5.0.12 (D17, amended 10-19, G3 guard): with s = 1 the two pairs' separation is the only constraint a 2-pair
    rigid fit does not absorb; the fit is a frame only when the model and GT separations differ by at most the
    alignment sigma (max(0.15, align_sigma_frac * L)), otherwise the space stays 'relations' (5.0.11)."""
    (m1, g1, _), (m2, g2, _) = pr
    sep_m = float(np.linalg.norm(np.asarray(m1, float) - np.asarray(m2, float)))
    sep_g = float(np.linalg.norm(np.asarray(g1, float) - np.asarray(g2, float)))
    return abs(sep_m - sep_g) <= max(0.15, P["align_sigma_frac"] * L)


def claim_spaces(ctx):
    """Space of every model instance: its tag, else the GT space whose aligned extent contains it."""
    gt, md = ctx.gt, ctx.model
    for m in md.instances:
        if ctx.use_space_tags and m.space in gt.space_by:
            m.gt_space = m.space
            continue
        if len(gt.spaces) == 1:
            m.gt_space = gt.spaces[0].id
            continue
        best, bd = None, None
        for s in gt.spaces:
            a = ctx.aligns[s.id]
            p = a.xy(m.center2[None])[0]
            lo, hi = s.extent
            d = float(np.max(np.maximum(lo - p, p - hi)))
            if s.outline is not None and geom.point_in_poly(p, s.outline):
                d = min(d, -0.01)
            if bd is None or d < bd:
                best, bd = s.id, d
        m.gt_space = best


# --------------------------------------------------------------------------------------------- yaw scoring
def yaw_score(o, inst, al, curve, near_square):
    """Symmetry-aware orientation score in [0,1] (None when the GT object is round).

    A GT front is checked against the model front when the model geometry shows one (seats, bodies).  When it
    does not, only the axis is checked: at half credit for classes whose front the scorer reads from geometry
    (a seat without a backrest), at full credit for classes it never reads (cabinets, screens, appliances).
    """
    if o.symmetry == 0:
        return None, None
    full, zero = geom.widen(curve, o.yaw_unc)
    inst_sq = inst.long < near_square * max(inst.short, 1e-6)
    gt_sq = o.long < near_square * max(o.short, 1e-6)
    if o.symmetry == 1 and o.front_yaw is not None:
        if fixes.on("M14") and getattr(inst, "facing_unknown", False):
            return None, None       # 5.2 (M14): body facing unknown, orientation not scored
        if inst.facing is not None:
            d = geom.angdiff(al.yaw(inst.facing) % 360, o.front_yaw, 360)
            return geom.lin(d, full, zero), d
        period = 90 if (inst_sq or gt_sq) else 180
        d = geom.angdiff(al.yaw(inst.long_yaw) % period, o.long_yaw % period, period)
        return (0.5 if inst.facing_inferable else 1.0) * geom.lin(d, full, zero), d
    period = 90 if (o.symmetry == 4 or inst_sq) else 180
    d = geom.angdiff(al.yaw(inst.long_yaw) % period, o.long_yaw % period, period)
    return geom.lin(d, full, zero), d


# ------------------------------------------------------------------------------------------------------ A1
def score_A1(ctx, items):
    gt, md, P, TH = ctx.gt, ctx.model, ctx.P, ctx.TH
    present_ids = {eid for eid, b in md.body.items() if b.present.any()}
    for eid, e in gt.entities.items():
        ok = eid in present_ids
        items.add("A1", "entities", "entity:" + eid, 1.0 if ok else 0.0, P_named(ctx), value=ok)
    for s in gt.spaces:
        objs = [o for o in gt.objects if o.space == s.id and gt.kind_of(o) == "object" and not o.id]
        insts = [m for m in md.instances if getattr(m, "gt_space", None) == s.id and m.kind == "object"]
        if objs:
            C = np.array([[ctx.credit(o.cls, m) if m.cls is not None else 0.0 for m in insts] for o in objs]) \
                if insts else np.zeros((len(objs), 0))
            got = np.zeros(len(objs))
            if insts:
                for i, j in geom.hungarian(np.where(C > 0, 1.0 - C, np.inf)):
                    got[i] = C[i, j]
            by = {}
            for i, o in enumerate(objs):
                by.setdefault(o.cls, []).append((got[i], o.conf))
            for cls, vals in sorted(by.items()):
                n = len(vals)
                if fixes.on("D2"):
                    # 5.0.12 (D2): GT conf weights the row (sum of conf); an exact match keeps its full credit
                    wc = sum(c for _, c in vals)
                    sc = sum(g * c for g, c in vals) / wc if wc > 0 else 0.0
                    items.add("A1", "counts", "count:%s:%s" % (s.id, cls), sc, float(wc),
                              value="%.2f/%d" % (sum(g for g, _ in vals), n))
                    continue
                vals = [g * c for g, c in vals]
                items.add("A1", "counts", "count:%s:%s" % (s.id, cls), sum(vals) / n, float(n),
                          value="%.2f/%d" % (sum(vals), n))
        # structure presence (indoor spaces: floor + walls; outdoor: ground)
        pcs = [p for p in md.layout_pieces if (not ctx.use_space_tags or p["space"] in (s.id, None))]
        has_floor = any(p["cls"] in FLOOR_CLASSES for p in pcs) or any(m.kind == "untagged_floor" for m in md.instances)
        has_wall = any(p["cls"] in WALL_CLASSES for p in pcs) or any(m.kind == "untagged_wall" for m in md.instances)
        if s.kind == "indoor":
            items.add("A1", "structure", "structure:" + s.id, 0.5 * has_floor + 0.5 * has_wall, 1.0,
                      value={"floor": has_floor, "walls": has_wall})
        else:
            ground = has_floor or any(p["cls"] in ("ground", "terrain") for p in pcs)
            items.add("A1", "structure", "structure:" + s.id, 1.0 if ground else 0.0, 1.0, value={"ground": ground})
        # extras: large object-like model instances inside the observed area left over after a class-only
        # assignment to the GT objects of the space
        if s.extras != "none" and objs:
            al = ctx.aligns[s.id]
            lo, hi = s.extent
            obs = []
            for m in insts:
                if m.group not in LARGE_GROUPS or m.long < 0.3:
                    continue
                if s.extras == "observed" and al.ok:
                    p = al.xy(m.center2[None])[0]
                    if np.any(p < lo - 0.5) or np.any(p > hi + 0.5):
                        continue
                obs.append(m)
            used = set()
            if obs:
                C2 = np.array([[ctx.credit(o.cls, m) for m in obs] for o in objs])
                used = {j for i, j in geom.hungarian(np.where(C2 > 0, 1.0 - C2, np.inf))}
            surplus = len(obs) - len(used)
            n_large = sum(1 for o in objs if o.group in LARGE_GROUPS)
            ratio = surplus / max(1, n_large)
            items.add("A1", "extras", "extras:" + s.id, geom.lin(ratio, *TH["extras_ratio"]), 1.0, value=ratio,
                      surplus=surplus, n_gt_large=n_large,
                      extra_classes=sorted({obs[j].cls for j in range(len(obs)) if j not in used}))
    for z in gt.zones:
        al = ctx.aligns.get(z.get("space", gt.spaces[0].id))
        poly = np.asarray(z["polygon"], float)
        want = set(z.get("classes", []))
        hit = False
        if al is not None and al.ok:
            for m in md.instances:
                if m.cls in want and geom.point_in_poly(al.xy(m.center2[None])[0], poly):
                    hit = True
                    break
        items.add("A1", "zones", "zone:" + z.get("id", "?"), 1.0 if hit else 0.0, 1.0, value=hit)


def P_named(ctx):
    return ctx.TH_all["named_weight"]


# ------------------------------------------------------------------------------------------------------ A2
def match_objects(ctx, s, objs, insts, al):
    """GT object -> (instance, credit, dist) for one space after alignment (named objects by id)."""
    out = {}
    free = []
    for o in objs:
        if o.id:
            bi = ctx.entity_instance(o.id, own_row=True)     # 5.0.12 (S19, amended 10-19): the object's own row
            out[o.key] = (bi, ctx.credit(o.cls, bi) if bi is not None else 0.0, None)
        else:
            free.append(o)
    if not free or not insts:
        for o in free:
            out[o.key] = (None, 0.0, None)
        return out
    gate = 0.35 * al.L
    C = np.array([[ctx.credit(o.cls, m) for m in insts] for o in free])
    if al.ok:
        T = al.xy(np.array([m.center2 for m in insts]))
        G = np.array([o.center[:2] for o in free])
        D = np.linalg.norm(G[:, None] - T[None], axis=2)
        cost = np.where((C > 0) & (D <= gate), D / (0.1 * al.L) + (1 - C), np.inf)
    else:
        # relations mode: class + size similarity only
        D = np.zeros_like(C)
        sz = np.array([[abs(math.log(max(1e-3, m.long * al.sim.s) / max(1e-3, o.long))) if o.size_known else 0.0
                        for m in insts] for o in free])
        cost = np.where(C > 0, (1 - C) + sz, np.inf)
    hit = {}
    for i, j in geom.hungarian(cost):
        hit[i] = j
    for i, o in enumerate(free):
        j = hit.get(i)
        out[o.key] = (insts[j], float(C[i, j]), float(D[i, j])) if j is not None else (None, 0.0, None)
    return out


def score_A2(ctx, items):
    gt, md, TH, P = ctx.gt, ctx.model, ctx.TH, ctx.P
    ctx.matches = {}
    for s in gt.spaces:
        al = ctx.aligns[s.id]
        objs = [o for o in gt.objects if o.space == s.id and gt.kind_of(o) == "object"]
        if not objs:
            continue
        insts = [m for m in md.instances if getattr(m, "gt_space", None) == s.id and m.kind in OBJECT_KINDS]
        mt = match_objects(ctx, s, objs, insts, al)
        ctx.matches.update(mt)
        L = al.L
        if al.ok:
            for o in objs:
                inst, c, _ = mt[o.key]
                w = (ctx.TH_all["named_weight"] if o.named else 1.0) * o.conf
                if inst is None or c <= 0:
                    items.add("A2", "objects", "obj:%s:%s" % (o.key, o.cls), 0.0, w, value="missing")
                    continue
                p = al.xy(inst.center2[None])[0]
                d = float(np.linalg.norm(p - o.center[:2]))
                sp = geom.lin(d / L, *geom.widen(TH["pos_frac"], o.unc / L))
                ys, yd = yaw_score(o, inst, al, TH["yaw_deg"], P["near_square"])
                dims_m = np.array([inst.long, inst.short, inst.height]) * al.sim.s
                dims_g = np.array([o.long, o.short, o.size[2]])
                serr = float(np.mean(np.abs(np.log(np.maximum(dims_m, 0.02) / np.maximum(dims_g, 0.02)))))
                ss = geom.lin(serr, *TH["size_log"])
                hb = float(al.h(inst.bottom))
                hg = o.bottom - s.floor_z
                sup = geom.lin(abs(hb - hg), *geom.widen(TH["support_m"], o.unc))
                # position gates the item (a misplaced object is not reproduced however well it is shaped);
                # orientation, size and support qualify it (5.0.2: not support in a sloped space)
                parts = ([(0.5, ys)] if ys is not None else []) + ([(0.3, ss)] if o.size_known else []) \
                    + ([] if s.sloped else [(0.2, sup)])
                sc = c * sp * sum(a * b for a, b in parts) / sum(a for a, _ in parts)
                items.add("A2", "objects", "obj:%s:%s" % (o.key, o.cls), sc, w, value=round(d, 3),
                          credit=c, pos=round(sp, 3), yaw=None if ys is None else round(ys, 3),
                          yaw_err=None if yd is None else round(yd, 1), size=round(ss, 3) if o.size_known else None,
                          support=None if s.sloped else round(sup, 3),
                          inst=inst.key, inst_class=inst.cls)
            occupancy(ctx, s, objs, insts, al, items)
        else:
            pairwise(ctx, s, objs, mt, al, items)
    relations(ctx, items)


def pairwise(ctx, s, objs, mt, al, items):
    """Relations-only mode (< 3 anchors): pairwise distances and relative yaw of the matched objects."""
    TH = ctx.TH
    L = al.L
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            ia, ib = mt[a.key][0], mt[b.key][0]
            w = 0.5 * ((2.0 if a.named else 1.0) + (2.0 if b.named else 1.0))
            if ia is None or ib is None:
                items.add("A2", "objects", "pair:%s~%s" % (a.key, b.key), 0.0, w, value="missing")
                continue
            dg = float(np.linalg.norm(a.center[:2] - b.center[:2]))
            dm = float(np.linalg.norm(ia.center2 - ib.center2))
            sc = geom.lin(abs(dm - dg) / L, *TH["pair_dist_frac"])
            items.add("A2", "objects", "pair:%s~%s" % (a.key, b.key), sc * mt[a.key][1] * mt[b.key][1], w,
                      value=round(abs(dm - dg), 3))


def occupancy(ctx, s, objs, insts, al, items):
    """Three-level occupancy F1 (floor / table / high bands) in the aligned frame."""
    P, TH = ctx.P, ctx.TH
    lo, hi = s.extent
    L = al.L
    cell = max(P["occ_cell_m"], L / 400.0)
    lo2 = lo - 1.0
    W = int(math.ceil((hi[0] - lo[0] + 2.0) / cell))
    H = int(math.ceil((hi[1] - lo[1] + 2.0) / cell))
    l1, l2 = P["occ_levels_m"]
    if fixes.on("S14"):
        # 5.0.12 (S14): the same set on both sides. Entity boxes are not rasterised (the model side holds untagged
        # layout instances only, as in A1); in a multi-space item a GT object that states no space is left out of
        # spaces[0], together with the model instances matched to it.
        multi = len(ctx.gt.spaces) > 1
        drop = [o for o in objs if multi and not o.id and not o.space_given]
        gone = {id(ctx.matches[o.key][0]) for o in drop if o.key in ctx.matches and ctx.matches[o.key][0] is not None}
        objs = [o for o in objs if not o.id and not (multi and not o.space_given)]
        insts = [m for m in insts if id(m) not in gone]
    # 5.0.2: a sloped space has no floor level to cut bands from, so it is one band
    bands = [(-99.0, 99.0)] if s.sloped else [(-0.5, l1), (l1, l2), (l2, 99.0)]
    for bi, (b0, b1) in enumerate(bands):
        gpolys = [o.corners2() for o in objs if (o.bottom - s.floor_z) < b1 and (o.top - s.floor_z) > b0]
        mpolys = []
        for m in insts:
            hb, ht = float(al.h(m.bottom)), float(al.h(m.top))
            if hb < b1 and ht > b0:
                c = geom.rect_corners(m.rect[0], m.long_yaw, m.long, m.short)
                mpolys.append(al.xy(c))
        if not gpolys:
            continue
        G = geom.raster_polys(gpolys, lo2, cell, (H, W))
        M = geom.raster_polys(mpolys, lo2, cell, (H, W)) if mpolys else np.zeros((H, W), bool)
        if s.extras == "observed":
            region = geom.dilate(G, int(math.ceil(1.0 / cell)))
            if s.outline is not None:
                region |= geom.raster_polys([s.outline], lo2, cell, (H, W))
            M = M & region
        f1 = geom.tolerant_f1(G, M, P["occ_tol_cells"])
        if f1 is None:
            continue
        items.add("A2", "occupancy", "occ:%s:L%d" % (s.id, bi), geom.lin(1 - f1, *TH["occ_1mf1"]),
                  float(G.sum()) * cell * cell, value=round(f1, 3))


def _relation_holds(ctx, r):
    """Model-side truth (score in [0,1]) of an explicit GT relation, or None if unresolvable."""
    gt, md = ctx.gt, ctx.model

    s15 = fixes.on("S15")
    UNRES = "unresolvable"

    m9 = set()

    def inst_of(key):
        if key in ctx.matches:
            if fixes.on("M9") and ctx.matches[key][0] is None:
                co = colocated_instance(ctx, key)      # 5.2 M9
                if co is not None:
                    m9.add(key)
                return co
            return ctx.matches[key][0]
        if s15:
            # 5.0.12 (S15): a GT opening / structure resolves through the A3 opening match; a key that is no GT
            # object, entity or model body cannot be resolved at all (the row is unsupported, not 0)
            o = gt.obj_by.get(key)
            if o is not None and gt.kind_of(o) in ("opening", "structure"):
                m, sc = match_opening(ctx, o)
                return m if sc > 0 else None
            eid = o.id if o is not None and o.id else key
            if eid in md.body:
                return ctx.entity_instance(eid)
            if o is None and key not in gt.entities:
                return UNRES
            return None
        bi = ctx.entity_instance(key)
        return bi
    a, b = inst_of(r["a"]), inst_of(r.get("b")) if r.get("b") else None
    if a is UNRES or b is UNRES or (r.get("type") == "between" and inst_of(r.get("c")) is UNRES):
        return None
    typ = r["type"]
    if a is None or (r.get("b") and b is None):
        return 0.0
    if m9 and (a is b or (typ == "between" and inst_of(r.get("c")) in (a, b))):
        return 0.0      # 5.2 M9: a co-located stand-in that is the other operand's own element decides nothing
    par = r.get("params", {})
    if typ in ("near", "distance"):
        dg = par.get("d")
        dm = float(np.linalg.norm(a.center2 - b.center2))
        if typ == "near" and fixes.on("M13"):
            # 5.2 (M13): the gap between the two oriented footprints (model frame), not the centre distance
            dm = geom.rect_gap(geom.gt_rect(lambda q: q, *a.rect), geom.gt_rect(lambda q: q, *b.rect))
        if typ == "near":
            return 1.0 if dm <= (dg or 1.0) * 1.5 else geom.lin(dm, (dg or 1.0) * 1.5, (dg or 1.0) * 3)
        return geom.lin(abs(dm - dg), 0.15 * max(dg, 1.0), 0.6 * max(dg, 1.0))
    if typ == "on":
        inside = np.all(a.center2 >= b.lo[:2] - 0.1) and np.all(a.center2 <= b.hi[:2] + 0.1)
        return 1.0 if inside and abs(a.bottom - b.top) <= 0.15 else 0.0
    if typ == "under":
        inside = np.all(a.center2 >= b.lo[:2] - 0.1) and np.all(a.center2 <= b.hi[:2] + 0.1)
        return 1.0 if inside and a.top <= b.top + 0.05 and a.bottom < b.bottom + 0.3 else 0.0
    if typ == "inside":
        return 1.0 if np.all(a.lo >= b.lo - 0.1) and np.all(a.hi <= b.hi + 0.1) else 0.0
    if typ == "facing":
        to_b = geom.yaw_of(b.center2 - a.center2)
        if fixes.on("M14") and getattr(a, "facing_unknown", False):
            return None             # 5.2 (M14): body facing unknown, not scored
        if a.facing is not None:
            return geom.lin(geom.angdiff(a.facing, to_b, 360), 30, 75)
        # no readable front: the target must lie on the front axis (the short axis; either axis when square)
        sq = a.long < ctx.P["near_square"] * max(a.short, 1e-6)
        period = 90 if sq else 180
        d = geom.angdiff((a.long_yaw + 90) % period, to_b % period, period)
        return (0.5 if a.facing_inferable else 1.0) * geom.lin(d, 30, 75)
    if typ in ("left_of", "right_of", "in_front_of", "behind"):
        ref = b
        ref_yaw = par.get("ref_yaw")
        if ref_yaw is not None:
            # a GT-frame reference direction needs the alignment rotation of the referenced object's space
            o = gt.obj_by.get(r.get("b"))
            al = ctx.aligns.get(o.space if o else gt.spaces[0].id)
            if al is None or not al.ok:
                return None
            ref_yaw = ref_yaw - al.sim.theta
        else:
            if ref.facing is None:
                return None
            ref_yaw = ref.facing
        v = a.center2 - ref.center2
        rel = geom.angdiff(geom.yaw_of(v), ref_yaw, 360)
        side = geom.wrap180(geom.yaw_of(v) - ref_yaw)
        if typ == "in_front_of":
            return 1.0 if rel < 60 else 0.0
        if typ == "behind":
            return 1.0 if rel > 120 else 0.0
        return 1.0 if (side > 20 if typ == "left_of" else side < -20) else 0.0
    if typ == "between":
        c = inst_of(r["c"])
        if c is None or c is UNRES:
            return 0.0
        u = c.center2 - b.center2
        t = float(np.dot(a.center2 - b.center2, u) / max(1e-9, np.dot(u, u)))
        off = float(np.linalg.norm(b.center2 + t * u - a.center2))
        return 1.0 if 0 < t < 1 and off < 0.35 * np.linalg.norm(u) else 0.0
    return None


def poly_area(P):
    """5.2 (M9): signed shoelace area of a polygon (xy rows)"""
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def poly_iou(P, Q):
    """5.2 (M9): IoU of two convex polygons (xy corners, e.g. Obj.corners2): Sutherland-Hodgman clip + shoelace"""
    P, Q = [np.asarray(x, float)[:, :2] for x in (P, Q)]
    P, Q = (P if poly_area(P) >= 0 else P[::-1]), (Q if poly_area(Q) >= 0 else Q[::-1])
    out = list(P)
    for k in range(len(Q)):
        a, b = Q[k], Q[(k + 1) % len(Q)]
        inp, out = out, []
        for i in range(len(inp)):
            p, q = inp[i], inp[(i + 1) % len(inp)]
            sp = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            sq = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                out.append(p + (sp / (sp - sq)) * (q - p))
        if not out:
            break
    inter = abs(poly_area(np.array(out))) if len(out) >= 3 else 0.0
    union = abs(poly_area(P)) + abs(poly_area(Q)) - inter
    return inter / union if union > 1e-12 else 0.0


def colocated_instance(ctx, key):
    """5.2 M9: the model element of an unmatched GT object's co-located twin: the matched GT object of the same space
    whose 2D footprint overlaps this one's by IoU > 0.5 and whose height interval overlaps this one's by IoU > 0.5 too
    (the same box, not one stacked on the other: a cooktop under a wall cabinet or an oven in a base cabinet share a
    footprint but not a height), the best footprint IoU; a relation to the object then reads the one model element
    that stands for both. None when there is none."""
    gt = ctx.gt
    o = gt.obj_by.get(key)
    if o is None:
        return None
    best, bv = None, 0.5
    for o2 in gt.objects:
        m = ctx.matches.get(o2.key)
        if o2 is o or o2.space != o.space or m is None or m[0] is None:
            continue
        hi, hu = min(o.top, o2.top) - max(o.bottom, o2.bottom), max(o.top, o2.top) - min(o.bottom, o2.bottom)
        if hu <= 0 or hi / hu <= 0.5:
            continue
        v = poly_iou(o.corners2(), o2.corners2())
        if v > bv:
            best, bv = m[0], v
    return best


def relations(ctx, items):
    for r in ctx.gt.relations:
        if r.get("t") is not None:
            continue   # timed relations belong to B2
        v = _relation_holds(ctx, r)
        if v is None:
            continue
        items.add("A2", "relations", "rel:%s:%s:%s" % (r["type"], r["a"], r.get("b")), v, r.get("weight", 1.0))


# ------------------------------------------------------------------------------------------------------ A3
def space_footprints(ctx):
    """Model footprint polygons per GT space (in model coordinates) + the source used."""
    md = ctx.model
    out = {}
    for s in ctx.gt.spaces:
        def mine(p_space, p_center):
            if ctx.use_space_tags:
                return p_space == s.id
            if len(ctx.gt.spaces) == 1:
                return True
            al = ctx.aligns[s.id]
            p = al.xy(np.asarray(p_center)[None])[0]
            lo, hi = s.extent
            return bool(np.all(p >= lo - 0.3) and np.all(p <= hi + 0.3))
        polys, src = [], None
        for p in md.layout_pieces:
            if p["cls"] in FLOOR_CLASSES and mine(p["space"], (p["lo"][:2] + p["hi"][:2]) / 2):
                c, y, lg, sh = geom.min_area_rect(p["P"][:, :2])
                if fixes.on("D22"):
                    polys.extend(piece_footprint(p["P"][:, :2], (c, y, lg, sh), ctx.P))
                    continue
                polys.append(geom.rect_corners(c, y, lg, sh))
        for m in md.instances:
            if m.kind == "untagged_floor" and mine(m.space, m.center2):
                polys.append(geom.rect_corners(m.rect[0], m.long_yaw, m.long, m.short))
        if polys:
            src = "floor"
        else:
            pts = [p["P"][:, :2] for p in md.layout_pieces if p["cls"] in WALL_CLASSES and mine(p["space"], (p["lo"][:2] + p["hi"][:2]) / 2)]
            pts += [m.P[:, :2] for m in md.instances if m.kind == "untagged_wall" and mine(m.space, m.center2)]
            if pts:
                polys, src = [geom.convex_hull(np.concatenate(pts))], "walls"
            else:
                pts = [m.P[:, :2] for m in md.instances if m.kind in OBJECT_KINDS and getattr(m, "gt_space", None) == s.id]
                if pts:
                    h = geom.convex_hull(np.concatenate(pts))
                    c = h.mean(0)
                    polys, src = [c + (h - c) * 1.0 + 0.3 * (h - c) / np.maximum(1e-6, np.linalg.norm(h - c, axis=1))[:, None]], "objects"
        out[s.id] = (polys, src)
    return out


def piece_footprint(xy, rect, P):
    """5.0.12 (D22): footprint polygons of one floor piece from its surface samples. The piece is rasterised in its
    min-area rectangle frame (cell = footprint_cell_k * sqrt(hull area / points), occupied cells, 3x3 closing); when
    the closed raster fills less than footprint_fill_min of the rectangle (an L-shaped or otherwise non-convex
    floor) its row runs are the footprint, else the rectangle."""
    c, y, lg, sh = rect
    box = [geom.rect_corners(c, y, lg, sh)]
    q = np.unique(np.round(np.asarray(xy, float), 4), axis=0)
    if len(q) < 12 or lg * sh <= 1e-6:
        return box
    ha = hull_area2(q)
    if ha <= 1e-6:
        return box
    cell = P["footprint_cell_k"] * math.sqrt(ha / len(q))
    R = geom.rot2(-y)
    loc = (q - np.asarray(c)[None]) @ R.T                      # rectangle frame: x along the long side
    x0, y0 = -0.5 * lg, -0.5 * sh
    W = max(1, int(math.ceil(lg / cell)))
    H = max(1, int(math.ceil(sh / cell)))
    if W * H > 250000:
        return box
    cw, ch = lg / W, sh / H
    ix = np.clip(((loc[:, 0] - x0) / cw).astype(int), 0, W - 1)
    iy = np.clip(((loc[:, 1] - y0) / ch).astype(int), 0, H - 1)
    G = np.zeros((H + 2, W + 2), bool)
    G[iy + 1, ix + 1] = True
    G = ~geom.dilate(~geom.dilate(G, 1), 1)                   # closing (padding keeps the border open)
    G = G[1:-1, 1:-1]
    if G.sum() >= P["footprint_fill_min"] * G.size:
        return box
    Rb = geom.rot2(y)
    out = []
    for r in range(H):
        row = G[r]
        k = 0
        while k < W:
            if not row[k]:
                k += 1
                continue
            j = k
            while j < W and row[j]:
                j += 1
            xa, xb = x0 + k * cw, x0 + j * cw
            ya, yb = y0 + r * ch, y0 + (r + 1) * ch
            sq = np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]])
            out.append(sq @ Rb.T + np.asarray(c)[None])
            k = j
    return out or box


def hull_area2(xy):
    h = geom.convex_hull(xy)
    if len(h) < 3:
        return 0.0
    x, y_ = h[:, 0], h[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y_, -1)) - np.dot(y_, np.roll(x, -1))))


CONN_CLS = {"door", "doorway", "archway", "gate"}
LINEAR_OPENINGS = ("road", "path", "bridge", "tunnel", "stairs", "ramp", "escalator")


def _opening_score(ctx, o, m, al):
    c = 1.0 if m.cls == o.cls else (0.75 if (m.cls in CONN_CLS and o.cls in CONN_CLS) else 0.0)
    if c <= 0 or not al.ok:
        return 0.0
    TH = ctx.TH
    d = float(np.linalg.norm(al.xy(m.center2[None])[0] - o.center[:2]))
    sc = c * geom.lin(d / al.L, *geom.widen(TH["opening_frac"], o.unc / al.L))
    if o.cls in LINEAR_OPENINGS:
        ad = geom.angdiff(al.yaw(m.long_yaw) % 180, o.long_yaw, 180)
        sc = 0.7 * sc + 0.3 * c * geom.lin(ad, *TH["yaw_deg"])
    return sc


def match_opening(ctx, o):
    """(model instance, score) of a GT opening / structure object: the A3 openings search over the untagged opening /
    structure instances of its space. 5.0.12 (D3): a GT opening that is an entity is its tagged body (S19 snapshot)
    when the model carries that id and the space's frame rests on >= min_anchors inliers (amended 10-19)."""
    md = ctx.model
    al = ctx.aligns[o.space]
    if o.id and fixes.on("D3") and al.n_inliers >= ctx.P["min_anchors"]:
        # amended 10-19 (G3 guard): only in a frame that rests on >= min_anchors inliers, so that the opening's own
        # body (itself an alignment anchor) cannot have placed itself; else the untagged search (5.0.11)
        bi = ctx.entity_instance(o.id) if o.id in md.body else None
        if bi is not None:
            return bi, _opening_score(ctx, o, bi, al)
    cand = [m for m in md.instances if m.kind in ("opening", "structure") and
            (not ctx.use_space_tags or m.space in (o.space, None))]
    best, bsc = None, 0.0
    for m in cand:
        sc = _opening_score(ctx, o, m, al)
        if sc > bsc:
            best, bsc = m, sc
    return best, bsc


def static_target(ctx, key):
    """5.0.12 (S15 / S17 / S18): (model instance or None, GT space) of a static B2 target key: the A2 match of a GT
    object, the A3 opening match of a GT opening / structure; None when the key is no GT object."""
    gt = ctx.gt
    o = gt.obj_by.get(key)
    if o is None:
        return None
    if key in ctx.matches:
        return ctx.matches[key][0], o.space
    if gt.kind_of(o) in ("opening", "structure"):
        m, sc = match_opening(ctx, o)
        return (m if sc > 0 else None), o.space
    return None, o.space


_CL51 = []


def _classes51():
    """5.2 M8: the 5.1 class credit (code/v51_text_r2/classes.py of the scorer package that holds this pvscore5),
    loaded once by file path; sys.path is left as it was"""
    if not _CL51:
        import importlib.util
        import sys
        from pathlib import Path
        f = Path(__file__).resolve().parents[3] / "code/v51_text_r2/classes.py"
        spec = importlib.util.spec_from_file_location("pv52_classes51", f)
        mod = importlib.util.module_from_spec(spec)
        saved = list(sys.path)
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path[:] = saved
        _CL51.append(mod)
    return _CL51[0]


def unmatched_static(ctx, key):
    """5.2 M8: a model instance for a static B2 target (a GT object of kind object) that A2 left unmatched, among the
    space's free layout instances (not matched to another GT object), in the aligned frame: (1) by class synonym, a
    5.1 class credit > 0 (classes.credit of classes.gt_labels) within the A2 gate (0.35 L), the A2 cost decides;
    else (2) by place, an instance whose oriented footprint holds the target's centre and whose centre lies in the
    target's footprint, both within the target's own position uncertainty (unc_m), the nearest. None otherwise."""
    cache = ctx.__dict__.setdefault("_m8", {})
    if key in cache:
        return cache[key]
    gt, out = ctx.gt, None
    o = gt.obj_by.get(key)
    al = ctx.aligns.get(o.space) if o is not None else None
    if o is not None and gt.kind_of(o) == "object" and al is not None and al.ok:
        taken = {id(v[0]) for k, v in ctx.matches.items() if k != key and v[0] is not None}
        insts = [m for m in ctx.model.instances if getattr(m, "gt_space", None) == o.space and
                 m.kind in OBJECT_KINDS and id(m) not in taken]
        if insts:
            CL = _classes51()
            labels = CL.gt_labels(o.key, o.cls)
            T = np.asarray(al.xy(np.array([m.center2 for m in insts])), float)[:, :2]
            D = np.linalg.norm(T - o.center[:2], axis=1)
            syn = [(D[j] / (0.1 * al.L) + (1 - c), j) for j, c in
                   enumerate(CL.credit(labels, m.cls) for m in insts) if c > 0 and D[j] <= 0.35 * al.L]
            if syn:
                out = insts[min(syn)[1]]
            else:
                g = geom.gt_rect(lambda q: q, o.center[:2], o.yaw, o.size[0], o.size[1])
                geo = [(D[j], j) for j, m in enumerate(insts)
                       if geom.in_rect(o.center[:2], geom.gt_rect(al.xy, m.rect[0], m.long_yaw, m.long, m.short), o.unc)
                       and geom.in_rect(T[j], g, o.unc)]
                if geo:
                    out = insts[min(geo)[1]]
    cache[key] = out
    return out


def score_A3(ctx, items):
    gt, md, TH, P = ctx.gt, ctx.model, ctx.TH, ctx.P
    fps_ = space_footprints(ctx)
    ctx.footprints = fps_
    areas_m, areas_g = {}, {}
    for s in gt.spaces:
        al = ctx.aligns[s.id]
        polys, src = fps_[s.id]
        if polys:
            lo = np.min([p.min(0) for p in polys], 0)
            hi = np.max([p.max(0) for p in polys], 0)
            cell = 0.05
            sh = (int(math.ceil((hi[1] - lo[1]) / cell)) + 1, int(math.ceil((hi[0] - lo[0]) / cell)) + 1)
            if sh[0] * sh[1] < 4e6:
                areas_m[s.id] = float(geom.raster_polys(polys, lo, cell, sh).sum()) * cell * cell
            else:
                areas_m[s.id] = float(np.prod(hi - lo))
        areas_g[s.id] = s.area()
        if s.outline is not None:
            if not polys or not al.ok:
                items.add("A3", "outline", "outline:" + s.id, 0.0, s.outline_conf, value=None, source=src,
                          mode=al.mode)
            else:
                tp = [al.xy(p) for p in polys]
                lo = np.minimum(s.outline.min(0), np.min([p.min(0) for p in tp], 0)) - 0.5
                hi = np.maximum(s.outline.max(0), np.max([p.max(0) for p in tp], 0)) + 0.5
                cell = max(P["outline_cell_m"], float(np.linalg.norm(hi - lo)) / 500)
                shape = (int(math.ceil((hi[1] - lo[1]) / cell)), int(math.ceil((hi[0] - lo[0]) / cell)))
                G = geom.raster_polys([s.outline], lo, cell, shape)
                M = geom.raster_polys(tp, lo, cell, shape)
                iou = float((G & M).sum()) / max(1, float((G | M).sum()))
                sig = (1.0 - s.outline_conf) * 0.3
                sc = geom.lin(1 - iou, *geom.widen(TH["outline_1miou"], sig))
                if src == "objects":
                    sc *= 0.5
                items.add("A3", "outline", "outline:" + s.id, sc, s.outline_conf, value=round(iou, 3), source=src)
        # openings / structures with position
        ops = [o for o in gt.objects if o.space == s.id and gt.kind_of(o) in ("opening",)]
        for o in ops:
            best, bsc = match_opening(ctx, o)
            items.add("A3", "openings", "opening:%s:%s" % (o.key, o.cls), bsc, o.conf,
                      inst=best.key if best else None)
    # connectivity (model frame: footprints adjacent + connector near the shared boundary)
    for cn in gt.connections:
        a, b = cn["a"], cn["b"]
        pa, pb = fps_.get(a, ([], None))[0], fps_.get(b, ([], None))[0]
        if not pa or not pb:
            items.add("A3", "connectivity", "conn:%s-%s" % (a, b), 0.0, 1.0, value="no footprint")
            continue
        A_ = np.concatenate(pa)
        B_ = np.concatenate(pb)
        dmin = float(np.min(np.linalg.norm(A_[:, None] - B_[None], axis=2))) if len(A_) * len(B_) < 4e6 else 0.0
        gap = _poly_gap(pa, pb)
        adj = gap <= 1.0
        conn = False
        via = cn.get("via")
        # 5.0.12 (D21, amended 10-19, G3 guard): D21 needs two distinct model spaces; when one model footprint polygon
        # serves both GT spaces (one floor for everything) their gap is 0 by construction and the 5.0.11 rule applies
        d21 = fixes.on("D21") and not any(np.array_equal(x, y) for x in pa for y in pb)
        if adj and d21 and (not via or via not in gt.obj_by):
            conn = True     # 5.0.12 (D21): an open boundary (no GT via object) needs no connector
        elif adj and d21:
            # 5.0.12 (D21): a connector-class instance / piece or the via's own model match whose footprint (not its
            # centre) comes within 1 m of both space footprints
            rects = [geom.rect_corners(m.rect[0], m.long_yaw, m.long, m.short) for m in md.instances
                     if m.cls in CONNECTOR_CLASSES]
            for p_ in md.layout_pieces:
                if p_["cls"] in CONNECTOR_CLASSES:
                    c_, y_, l_, s_ = geom.min_area_rect(p_["P"][:, :2])
                    rects.append(geom.rect_corners(c_, y_, l_, s_))
            st = static_target(ctx, via)
            if st is not None and st[0] is not None:
                m = st[0]
                rects.append(geom.rect_corners(m.rect[0], m.long_yaw, m.long, m.short))
            for rc in rects:
                if _poly_gap([rc], pa) <= 1.0 and _poly_gap([rc], pb) <= 1.0:
                    conn = True
                    break
        elif adj:
            cands = [m.center2 for m in md.instances if m.cls in CONNECTOR_CLASSES]
            cands += [(p["lo"][:2] + p["hi"][:2]) / 2 for p in md.layout_pieces if p["cls"] in CONNECTOR_CLASSES]
            for c in cands:
                if _pt_polys_dist(c, pa) <= 1.0 and _pt_polys_dist(c, pb) <= 1.0:
                    conn = True
                    break
        sc = 1.0 if (adj and conn) else (0.5 if adj else 0.0)
        items.add("A3", "connectivity", "conn:%s-%s" % (a, b), sc, 1.0, value={"gap": round(gap, 3),
                  "connector": conn, "vertex_gap": round(dmin, 3)})
    if len(gt.spaces) >= 2 and len(areas_m) == len(gt.spaces) and gt.space_frames != "separate":
        tm, tg = sum(areas_m.values()), sum(areas_g.values())
        for s in gt.spaces:
            if tm <= 0 or tg <= 0:
                continue
            err = abs(math.log(max(1e-6, areas_m[s.id] / tm) / max(1e-6, areas_g[s.id] / tg)))
            items.add("A3", "space_size", "size:" + s.id, geom.lin(err, *TH["space_size_log"]), 1.0,
                      value=round(err, 3))
        f = {s.id: ctx.aligns[s.id].floor_m for s in gt.spaces}
        flat = [s for s in gt.spaces if not s.sloped]   # 5.0.2: a sloped floor has no level to compare
        s0 = flat[0] if flat else None
        for s in flat[1:]:
            dz_g = s.floor_z - s0.floor_z
            dz_m = (f[s.id] - f[s0.id]) * ctx.aligns[s.id].sim.s
            items.add("A3", "floor_dz", "floor_dz:%s-%s" % (s0.id, s.id), geom.lin(abs(dz_m - dz_g), *TH["floor_dz_m"]),
                      1.0, value=round(dz_m - dz_g, 3))


def _poly_gap(pa, pb):
    """Smallest distance between two unions of convex polygons (0 if they overlap)."""
    best = None
    for a in pa:
        for b in pb:
            d = _convex_gap(a, b)
            best = d if best is None else min(best, d)
    return float(best if best is not None else 1e9)


def _convex_gap(a, b):
    if any(geom.point_in_poly(p, b) for p in a) or any(geom.point_in_poly(p, a) for p in b):
        return 0.0
    best = 1e9
    for P_, Q_ in ((a, b), (b, a)):
        for p in P_:
            for i in range(len(Q_)):
                u, v = Q_[i], Q_[(i + 1) % len(Q_)]
                e = v - u
                t = np.clip(np.dot(p - u, e) / max(1e-12, np.dot(e, e)), 0, 1)
                best = min(best, float(np.linalg.norm(u + t * e - p)))
    return best


def _pt_polys_dist(p, polys):
    p = np.asarray(p, float)
    best = 1e9
    for poly in polys:
        if geom.point_in_poly(p, poly):
            return 0.0
        for i in range(len(poly)):
            u, v = poly[i], poly[(i + 1) % len(poly)]
            e = v - u
            t = np.clip(np.dot(p - u, e) / max(1e-12, np.dot(e, e)), 0, 1)
            best = min(best, float(np.linalg.norm(u + t * e - p)))
    return best
