"""Frame layer P1: per GT reference frame, depth / normals / coarse regions / instances / depth edges."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import geom
from .gt import group_of

# coarse region codes shared with the reference maps
NONE, SKY, GROUND, WALL, CEILING, OBJECT, PERSON, VEHICLE = range(8)
GROUND_STRUCT = {"floor", "platform", "stage", "stairs", "ramp", "bridge", "pier", "dock", "escalator"}
CEIL_STRUCT = {"ceiling", "roof"}


def class_region(cls, cats, kind=None):
    if cls is None:
        return {"untagged_floor": GROUND, "untagged_wall": WALL}.get(kind, OBJECT)
    g = group_of(cls, cats)
    if g == "terrain":
        return GROUND
    if g == "structure":
        if cls in GROUND_STRUCT:
            return GROUND
        if cls in CEIL_STRUCT:
            return CEILING
        return WALL
    if g in ("agent", "animal"):
        return PERSON
    if g == "vehicle":
        return VEHICLE
    return OBJECT


def model_maps(ctx):
    """(part, tag) -> (region code, instance number) lookup for the model."""
    md, cats = ctx.model, ctx.gt.cats
    reg, inst = {}, {}
    n = 0
    for m in md.instances:
        n += 1
        for p in m.pieces:
            key = (p["part"], p["tag"])
            reg[key] = class_region(m.cls, cats, m.kind)
            if m.kind in ("object", "opening", "untagged_object"):
                inst[key] = n
    for eid, b in md.body.items():
        n += 1
        for (pi, tg), (cls, e) in md.label.items():
            if e == eid:
                reg[(pi, tg)] = class_region(cls, cats)
                inst[(pi, tg)] = n
    return reg, inst


def lookup(part, tag, table, default):
    out = np.full(part.shape, default, np.int32)
    for (pi, tg), v in table.items():
        m = (part == pi) & (tag == tg)
        if m.any():
            out[m] = v
    return out


def intrinsics_from_corners(C, W, H):
    x0, x1 = C[:, 0].min(), C[:, 0].max()
    y0, y1 = C[:, 1].min(), C[:, 1].max()
    zc = abs(C[0, 2])
    fx = W * zc / (x1 - x0)
    fy = H * zc / (y1 - y0)
    cx = -x0 / (x1 - x0) * W
    cy = y1 / (y1 - y0) * H
    return np.array([fx, fy, cx, cy])


def normals_from_depth(D, K):
    H, W = D.shape
    fx, fy, cx, cy = K
    u = np.arange(W) + 0.5
    v = np.arange(H) + 0.5
    U, V = np.meshgrid(u, v)
    X = np.stack([(U - cx) / fx * D, (V - cy) / fy * D, D], -1)
    valid = D > 0
    dx = np.zeros_like(X)
    dy = np.zeros_like(X)
    dx[:, 1:-1] = X[:, 2:] - X[:, :-2]
    dy[1:-1] = X[2:] - X[:-2]
    vx = np.zeros_like(valid)
    vy = np.zeros_like(valid)
    vx[:, 1:-1] = valid[:, 2:] & valid[:, :-2]
    vy[1:-1] = valid[2:] & valid[:-2]
    # reject differences across depth discontinuities
    rel = np.zeros_like(D)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel[:, 1:-1] = np.abs(D[:, 2:] - D[:, :-2]) / np.maximum(D[:, 1:-1], 1e-6)
        vx &= rel < 0.15
        rel2 = np.zeros_like(D)
        rel2[1:-1] = np.abs(D[2:] - D[:-2]) / np.maximum(D[1:-1], 1e-6)
        vy &= rel2 < 0.15
    N = np.cross(dx, dy)
    nn = np.linalg.norm(N, axis=-1, keepdims=True)
    ok = valid & vx & vy & (nn[..., 0] > 1e-12)
    N = np.where(ok[..., None], N / np.maximum(nn, 1e-12), 0)
    return N, ok


def crop_resize(arrs, src_aspect, dst_aspect, out_hw):
    """Centre-crop arrays (H, W, ...) from src_aspect to dst_aspect, then nearest-resize to out_hw."""
    out = []
    for a in arrs:
        H, W = a.shape[:2]
        if src_aspect > dst_aspect + 1e-6:
            w = int(round(H * dst_aspect))
            x0 = (W - w) // 2
            a = a[:, x0:x0 + w]
        elif src_aspect < dst_aspect - 1e-6:
            h = int(round(W / dst_aspect))
            y0 = (H - h) // 2
            a = a[y0:y0 + h]
        H2, W2 = a.shape[:2]
        ys = np.minimum(H2 - 1, (np.arange(out_hw[0]) + 0.5) * H2 / out_hw[0]).astype(int)
        xs = np.minimum(W2 - 1, (np.arange(out_hw[1]) + 0.5) * W2 / out_hw[1]).astype(int)
        out.append(a[ys][:, xs])
    return out


def depth_edges(D):
    valid = D > 0
    E = np.zeros(D.shape, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(D[:, 1:] - D[:, :-1]) / np.maximum(np.minimum(D[:, 1:], D[:, :-1]), 1e-6)
        e = ((r > 0.1) & valid[:, 1:] & valid[:, :-1]) | (valid[:, 1:] != valid[:, :-1])
        E[:, 1:] |= e
        r = np.abs(D[1:] - D[:-1]) / np.maximum(np.minimum(D[1:], D[:-1]), 1e-6)
        e = ((r > 0.1) & valid[1:] & valid[:-1]) | (valid[1:] != valid[:-1])
        E[1:] |= e
    return E


def known_interior(kn):
    """Known cells whose four neighbours are known too (outside the grid counts as known)."""
    p = np.pad(kn, 1, constant_values=True)
    return kn & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


def compare(ref, mod, TH):
    """Five P1 metrics of a model frame against a reference frame (both already on one grid)."""
    out = {}
    dg, dm = ref["depth"], mod["depth"]
    kn = ref.get("known")
    if kn is None:
        kn = np.ones(dg.shape, bool)
    vg, vm = dg > 0, dm > 0
    both = vg & vm & kn
    if both.sum() >= 10:
        shift = np.median(np.log(dg[both]) - np.log(dm[both]))
        r = np.exp(np.log(dm[both]) + shift) / dg[both]
        d1 = float(np.mean(np.maximum(r, 1 / r) < 1.25))
        cover = both.sum() / max(1, ((vg | vm) & kn).sum())  # 5.0.4: unknown reference cells are not counted
        d1 *= min(1.0, cover / 0.7)
    else:
        d1 = 0.0
    out["depth"] = (geom.lin_up(d1, *TH["p1_depth_q"]), d1)
    ok = ref["n_ok"] & mod["n_ok"]
    if ok.sum() >= 10:
        cos = np.clip((ref["normal"][ok] * mod["normal"][ok]).sum(-1), -1, 1)
        frac = float(np.mean(np.degrees(np.arccos(cos)) < 30))
        frac *= min(1.0, ok.sum() / max(1, ref["n_ok"].sum()) / 0.7)
    else:
        frac = 0.0
    out["normal"] = (geom.lin_up(frac, *TH["p1_normal_q"]), frac)
    rg, rm = ref["group"], mod["group"]
    tot = rg.size
    fw = 0.0
    for c in np.unique(rg):
        if c == NONE:
            continue
        g, m = rg == c, rm == c
        u = (g | m).sum()
        fw += (g.sum() / tot) * ((g & m).sum() / u if u else 0.0)
    known = (rg != NONE).sum() / tot
    fw = fw / known if known > 0 else 0.0
    out["region"] = (geom.lin(1 - fw, *TH["p1_region_1mfwiou"]), fw)
    ig, im = ref["instance"], mod["instance"]
    gid = [k for k in np.unique(ig) if k > 0 and (ig == k).sum() >= 6]
    mid = [k for k in np.unique(im) if k > 0 and (im == k).sum() >= 6]
    if gid or mid:
        iou = np.zeros((len(gid), len(mid)))
        for a, k in enumerate(gid):
            g = ig == k
            for b, j in enumerate(mid):
                m = im == j
                inter = (g & m).sum()
                if inter:
                    iou[a, b] = inter / (g | m).sum()
        tp = sum(1 for a, b in geom.hungarian(np.where(iou >= 0.3, 1 - iou, np.inf)))
        f1 = 2 * tp / (len(gid) + len(mid)) if (len(gid) + len(mid)) else 0.0
        out["instance"] = (geom.lin(1 - f1, *TH["p1_inst_1mf1"]), f1)
    eg, em = depth_edges(dg), depth_edges(dm)
    if not kn.all():  # 5.0.4: edges only inside the known reference area (its border is not a depth edge)
        kin = known_interior(kn)
        eg, em = eg & kin, em & kin
    f = geom.tolerant_f1(eg, em, 2)
    if f is not None:
        out["edge"] = (geom.lin(1 - f, *TH["p1_edge_1mf"]), f)
    return out


def model_frame(ctx, gi, reg_t, inst_t):
    md = ctx.model
    g = md.grid(gi)
    H, W = g["depth"].shape
    part, tag = g["part"], g["tag"]
    hit = part >= 0
    group = np.where(hit, lookup(part, tag, reg_t, OBJECT), SKY)
    group = np.where(part == -2, OBJECT, group)
    inst = np.where(hit, lookup(part, tag, inst_t, 0), 0)
    D = np.where(hit, g["depth"], 0.0)
    K = intrinsics_from_corners(g["corners"], W, H) if not g["ortho"] else None
    if K is not None:
        N, ok = normals_from_depth(D, K)
    else:
        N, ok = g["normal"], hit
    return {"depth": D, "group": group, "instance": inst, "normal": N, "n_ok": ok, "aspect": W / H, "hw": (H, W)}


def ref_frame(path):
    z = np.load(path)
    D = z["depth"].astype(np.float64)
    H, W = D.shape
    K = z["K"].astype(np.float64)
    N, ok = normals_from_depth(D, K)
    # 5.0.4: real references (annotation pipeline) mark where their depth is measured; sky is known and empty
    kn = z["known"].astype(bool) if "known" in z.files else np.ones(D.shape, bool)
    return {"depth": D, "group": z["group"].astype(np.int32), "instance": z["instance"].astype(np.int32),
            "normal": N, "n_ok": ok & kn, "known": kn, "aspect": W / H, "hw": (H, W)}


def score_P1(ctx, items):
    gt, md = ctx.gt, ctx.model
    if gt.modality == "text":
        return
    refs = [(sh, r) for sh in gt.shots for r in sh.p1]
    if not refs:
        return
    reg_t, inst_t = model_maps(ctx)
    base = gt.path.parent if gt.path else Path(".")
    for si, (sh, r) in enumerate(refs):
        ref = ref_frame(base / r["ref"])
        name = "p1:%s@%.2f" % (sh.id, r["t"])
        mshot, tm = ctx.model_time_for(sh, r["t"])
        gi = md.grid_near(tm, mshot) if tm is not None else None
        if gi is None or not len(md.g_t):
            for b in ("depth", "normal", "region", "instance", "edge"):
                items.add("P1", b, name, 0.0, 1.0, value="no model frame")
            continue
        mod = model_frame(ctx, gi, reg_t, inst_t)
        hw = (min(ref["hw"][0], int(round(min(ref["hw"][1], mod["hw"][1]) / ref["aspect"]))),
              min(ref["hw"][1], mod["hw"][1]))
        cr = crop_resize([mod["depth"], mod["group"], mod["instance"], mod["normal"], mod["n_ok"]],
                         mod["aspect"], ref["aspect"], hw)
        rr = crop_resize([ref["depth"], ref["group"], ref["instance"], ref["normal"], ref["n_ok"], ref["known"]],
                         ref["aspect"], ref["aspect"], hw)
        m2 = dict(zip(["depth", "group", "instance", "normal", "n_ok"], cr))
        r2 = dict(zip(["depth", "group", "instance", "normal", "n_ok", "known"], rr))
        res = compare(r2, m2, ctx.TH)
        dt = abs(float(md.g_t[gi]) - tm)
        for b, (sc, val) in res.items():
            items.add("P1", b, name, sc, 1.0, value=round(float(val), 4), model_t=round(float(md.g_t[gi]), 3),
                      dt=round(dt, 3), shot=mshot)
