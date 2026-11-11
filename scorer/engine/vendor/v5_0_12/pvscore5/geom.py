"""Small geometry / scoring helpers (numpy, scipy)."""
from __future__ import annotations

import math

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover
    _lsa = None


def lin(x, full, zero):
    """Error curve: 1 at x <= full, 0 at x >= zero, linear in between (x is an error)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    if zero <= full:
        return 1.0 if x <= full else 0.0
    return float(min(1.0, max(0.0, (zero - x) / (zero - full))))


def lin_up(x, full, zero):
    """Quality curve: 1 at x >= full, 0 at x <= zero (x is a quality, e.g. an IoU)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    if full <= zero:
        return 1.0 if x >= full else 0.0
    return float(min(1.0, max(0.0, (x - zero) / (full - zero))))


def widen(th, sigma):
    """Thresholds (full, zero) widened by an uncertainty sigma (same unit)."""
    full, zero = th
    return full + sigma, zero + 2.0 * sigma


def angdiff(a, b, period=360.0):
    """Smallest absolute difference of two angles (degrees) modulo period."""
    d = (a - b) % period
    return min(d, period - d)


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def yaw_of(v):
    return math.degrees(math.atan2(v[1], v[0]))


def rot2(theta_deg):
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


def hungarian(cost):
    """Minimum-cost assignment for a (possibly rectangular) matrix; inf entries are forbidden."""
    cost = np.asarray(cost, float)
    if cost.size == 0:
        return []
    big = 1e9
    c = np.where(np.isfinite(cost), cost, big)
    if _lsa is not None:
        r, k = _lsa(c)
    else:  # greedy fallback
        r, k, used = [], [], set()
        for i, j in sorted(((i, j) for i in range(c.shape[0]) for j in range(c.shape[1])), key=lambda x: c[x]):
            if i in r or j in used:
                continue
            r.append(i)
            k.append(j)
            used.add(j)
    return [(int(i), int(j)) for i, j in zip(r, k) if c[i, j] < big]


class Sim2:
    """p_gt = s * R(theta) @ p_model + t   (2D, proper rotation, positive scale)."""

    def __init__(self, s=1.0, theta=0.0, t=(0.0, 0.0)):
        self.s, self.theta, self.t = float(s), float(theta), np.asarray(t, float)

    @property
    def R(self):
        return rot2(self.theta)

    def apply(self, P):
        P = np.asarray(P, float)
        return (self.s * (self.R @ P[..., :2].reshape(-1, 2).T)).T.reshape(P[..., :2].shape) + self.t

    def yaw(self, yaw_model_deg):
        return yaw_model_deg + self.theta

    def to_dict(self):
        return {"s": round(self.s, 4), "theta_deg": round(self.theta, 2), "t": [round(float(x), 3) for x in self.t]}


def umeyama2(P, Q, w=None, s_bounds=(0.67, 1.5), fix_scale=None):
    """Weighted similarity (proper rotation) with P -> Q; scale clamped to s_bounds."""
    P, Q = np.asarray(P, float)[:, :2], np.asarray(Q, float)[:, :2]
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    W = w.sum()
    if len(P) == 0 or W <= 0:
        return Sim2()
    mp, mq = (w[:, None] * P).sum(0) / W, (w[:, None] * Q).sum(0) / W
    X, Y = P - mp, Q - mq
    C = (Y * w[:, None]).T @ X / W
    U, S, Vt = np.linalg.svd(C)
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    R = U @ D @ Vt
    varp = (w * (X ** 2).sum(1)).sum() / W
    if fix_scale is not None:
        s = fix_scale
    else:
        s = float(np.trace(np.diag(S) @ D) / varp) if varp > 1e-9 else 1.0
        s = min(max(s, s_bounds[0]), s_bounds[1])
    theta = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    t = mq - s * (rot2(theta) @ mp)
    return Sim2(s, theta, t)


def sim_from_pairs(p1, p2, q1, q2, s_bounds=(0.5, 2.0)):
    a, b = np.asarray(p2, float)[:2] - np.asarray(p1, float)[:2], np.asarray(q2, float)[:2] - np.asarray(q1, float)[:2]
    la, lb = np.linalg.norm(a), np.linalg.norm(b)
    if la < 1e-6 or lb < 1e-6:
        return None
    s = lb / la
    if not s_bounds[0] <= s <= s_bounds[1]:
        return None
    theta = math.degrees(math.atan2(b[1], b[0]) - math.atan2(a[1], a[0]))
    t = np.asarray(q1, float)[:2] - s * (rot2(theta) @ np.asarray(p1, float)[:2])
    return Sim2(s, theta, t)


def min_area_rect(xy):
    """(center[2], yaw_long_deg, long, short) of the minimum-area rectangle around 2D points."""
    xy = np.asarray(xy, float)
    if len(xy) == 0:
        return np.zeros(2), 0.0, 0.0, 0.0
    if len(xy) < 3:
        c = xy.mean(0)
        if len(xy) == 2:
            d = xy[1] - xy[0]
            return c, yaw_of(d), float(np.linalg.norm(d)), 0.0
        return c, 0.0, 0.0, 0.0
    hull = convex_hull(xy)
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        n = np.linalg.norm(e)
        if n < 1e-9:
            continue
        u = e / n
        v = np.array([-u[1], u[0]])
        pu, pv = hull @ u, hull @ v
        area = (pu.max() - pu.min()) * (pv.max() - pv.min())
        if best is None or area < best[0] - 1e-12:
            best = (area, u, v, pu.min(), pu.max(), pv.min(), pv.max())
    if best is None:
        c = xy.mean(0)
        return c, 0.0, 0.0, 0.0
    _, u, v, a0, a1, b0, b1 = best
    c = u * (a0 + a1) / 2 + v * (b0 + b1) / 2
    lu, lv = a1 - a0, b1 - b0
    if lu >= lv:
        return c, yaw_of(u) % 180.0, float(lu), float(lv)
    return c, yaw_of(v) % 180.0, float(lv), float(lu)


def convex_hull(xy):
    pts = sorted(set(map(tuple, np.round(np.asarray(xy, float), 6))))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def rect_corners(center, yaw_deg, long, short):
    u = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))])
    v = np.array([-u[1], u[0]])
    c = np.asarray(center, float)
    return np.array([c + u * long / 2 + v * short / 2, c - u * long / 2 + v * short / 2,
                     c - u * long / 2 - v * short / 2, c + u * long / 2 - v * short / 2])


def gt_rect(xy, center, yaw_deg, long, short):
    """5.0.11: a model-frame rectangle mapped into the GT frame by xy (an alignment's xy map), kept oriented:
    (centre, unit long axis, half long, half short) for in_rect. 5.0.10 took the axis-aligned box of the mapped corners
    instead, which a rotation stretches (a thin wall turned by 45 deg covers a square)."""
    q = np.asarray(xy(rect_corners(np.asarray(center, float)[:2], yaw_deg, long, short)), float)[:, :2]
    e1, e2 = q[0] - q[1], q[0] - q[3]          # along long, along short
    n1, n2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    if n1 > 1e-9:
        u = e1 / n1
    elif n2 > 1e-9:
        u = np.array([e2[1], -e2[0]]) / n2
    else:
        u = np.array([1.0, 0.0])
    return q.mean(0), u, 0.5 * n1, 0.5 * n2


def in_rect(p, rect, margin=0.0):
    """p (xy) within an oriented rectangle from gt_rect, grown by margin on every side."""
    c, u, hl, hs = rect
    d = np.asarray(p, float)[:2] - c
    return abs(float(d @ u)) <= hl + margin and abs(float(d[1] * u[0] - d[0] * u[1])) <= hs + margin


def rect_dist(p, rect):
    """5.0.12: distance from p (xy) to an oriented rectangle from gt_rect (0 inside)."""
    c, u, hl, hs = rect
    d = np.asarray(p, float)[:2] - c
    x = abs(float(d @ u)) - hl
    y = abs(float(d[1] * u[0] - d[0] * u[1])) - hs
    return float(math.hypot(max(x, 0.0), max(y, 0.0)))


def rect_half_extent(rect, v):
    """5.0.12 (S9): half extent of an oriented rectangle (gt_rect) along the direction v."""
    c, u, hl, hs = rect
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return 0.0
    w = np.asarray(v, float)[:2] / n
    return float(hl * abs(float(w @ u)) + hs * abs(float(w[1] * u[0] - w[0] * u[1])))


def gt_rect_corners(rect):
    """5.2 (M13): the four corners of an oriented rectangle from gt_rect (centre, unit long axis, half long, half
    short); a zero side is kept at 1e-6 so that both axes give a separating direction in convex_gap2."""
    c, u, hl, hs = rect
    u = np.asarray(u, float)
    v = np.array([-u[1], u[0]])
    hl, hs = max(float(hl), 1e-6), max(float(hs), 1e-6)
    c = np.asarray(c, float)[:2]
    return np.array([c + u * hl + v * hs, c - u * hl + v * hs, c - u * hl - v * hs, c + u * hl - v * hs])


def rect_gap(ra, rb):
    """5.2 (M13): horizontal gap between two oriented rectangles from gt_rect (0 when they touch or overlap)."""
    return convex_gap2(gt_rect_corners(ra), gt_rect_corners(rb))


def raster_polys(polys, lo, cell, shape):
    """Boolean grid (shape) with the union of convex polygons (lists of xy corners, ccw or cw)."""
    H, W = shape
    ys = lo[1] + (np.arange(H) + 0.5) * cell
    xs = lo[0] + (np.arange(W) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    grid = np.zeros(shape, bool)
    for poly in polys:
        poly = np.asarray(poly, float)
        if len(poly) < 3:
            continue
        x0, y0 = poly.min(0)
        x1, y1 = poly.max(0)
        m = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        if not m.any():
            continue
        sgn = None
        inside = m.copy()
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            cr = (b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0])
            if sgn is None:
                area = 0.5 * sum(poly[k][0] * poly[(k + 1) % len(poly)][1] - poly[(k + 1) % len(poly)][0] * poly[k][1]
                                 for k in range(len(poly)))
                sgn = 1.0 if area >= 0 else -1.0
            inside &= (sgn * cr) >= -1e-9
        grid |= inside
    return grid


def dilate(mask, r):
    """Binary dilation by a square of radius r cells."""
    if r <= 0:
        return mask.copy()
    out = mask.copy()
    H, W = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            ys0, ys1 = max(0, dy), min(H, H + dy)
            xs0, xs1 = max(0, dx), min(W, W + dx)
            out[ys0:ys1, xs0:xs1] |= mask[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def tolerant_f1(a, b, r):
    """F1 of two binary masks where a cell counts as matched if the other mask is within r cells."""
    na, nb = int(a.sum()), int(b.sum())
    if na == 0 and nb == 0:
        return None
    if na == 0 or nb == 0:
        return 0.0
    p = float((a & dilate(b, r)).sum()) / na
    q = float((b & dilate(a, r)).sum()) / nb
    return 0.0 if p + q == 0 else 2 * p * q / (p + q)


def resample_path(P, n=50):
    P = np.asarray(P, float)
    if len(P) < 2:
        return np.repeat(P[:1], n, 0) if len(P) else np.zeros((n, P.shape[1] if P.ndim == 2 else 2))
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.repeat(P[:1], n, 0)
    q = np.linspace(0, s[-1], n)
    return np.stack([np.interp(q, s, P[:, k]) for k in range(P.shape[1])], 1)


def frechet(P, Q):
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    n, m = len(P), len(Q)
    D = np.linalg.norm(P[:, None] - Q[None], axis=2)
    C = np.full((n, m), np.inf)
    C[0, 0] = D[0, 0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            best = min(C[i - 1, j] if i else np.inf, C[i, j - 1] if j else np.inf,
                       C[i - 1, j - 1] if i and j else np.inf)
            C[i, j] = max(best, D[i, j])
    return float(C[-1, -1])


def path_length(P):
    P = np.asarray(P, float)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def levenshtein(a, b, sub_cost=None):
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = (0.0 if a[i - 1] == b[j - 1] else 1.0) if sub_cost is None else sub_cost(a[i - 1], b[j - 1])
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + c)
    return float(D[n, m])


def seq_similarity(a, b, sub_cost=None):
    if not a and not b:
        return None
    return 1.0 - levenshtein(a, b, sub_cost) / max(len(a), len(b))


def collapse(seq):
    out = []
    for x in seq:
        if x is None:
            continue
        if not out or out[-1] != x:
            out.append(x)
    return out


def point_in_poly(p, poly):
    x, y = p[0], p[1]
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xi > x:
                inside = not inside
    return inside


def aabb_gap(lo1, hi1, lo2, hi2):
    """Largest per-axis separation (<= 0 means overlapping / touching)."""
    return float(np.max(np.maximum(lo1 - hi2, lo2 - hi1)))


def _seg_dist(p, a, b):
    e = b - a
    t = float(np.clip(np.dot(p - a, e) / max(1e-12, float(np.dot(e, e))), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * e)))


def convex_gap2(A, B):
    """Distance between two convex polygons (corner arrays); 0 when they overlap or touch."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            e = poly[(i + 1) % n] - poly[i]
            nrm = np.array([-e[1], e[0]])
            if not np.any(nrm):
                continue
            pa, pb = A @ nrm, B @ nrm
            if pa.max() < pb.min() or pb.max() < pa.min():
                return min(min(_seg_dist(p, Q[i], Q[(i + 1) % len(Q)]) for i in range(len(Q)))
                           for P, Q in ((A, B), (B, A)) for p in P)
    return 0.0


def box_gap(a, b):
    """Largest separation of two oriented boxes (corners2[4,2], z0, z1); <= 0 means touching or overlapping.

    The rotation-invariant counterpart of aabb_gap: horizontal distance of the footprints vs. vertical gap."""
    dz = max(a[1] - b[2], b[1] - a[2])
    return max(convex_gap2(a[0], b[0]), dz)


def oriented_box(P):
    """(corners2[4,2], z0, z1) of a point set's min-area-rect box."""
    c, yaw, lg, sh = min_area_rect(P[:, :2])
    return rect_corners(c, yaw, lg, sh), float(P[:, 2].min()), float(P[:, 2].max())
