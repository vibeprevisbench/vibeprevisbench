"""v5.1, geometry route of the per-frame scene inventory (no API): at about 1 s steps of every shot, the SAM3
elements of the frame (geom/seg) with their single-frame depth (geom/recon npz) and the pairwise screen / depth
relations between them. Thresholds: scene_geom_thresholds_r1.json (r0 values pre-registered; r1 = three rule changes
after the visual check of 45 frames, see rules_r1 there and data/v51_r2/README.md).

    ssh ... "cd $B && env/bin/python -I -B tools/v51_r2/scene_geom.py [M50_033 ...]"

Reads annot/v1/<key>/{v0.json, final.json, geom/{seg,link,frames,recon}} and the GT <item>/gt.json (gt_root.py)
(shot times); writes only data/v51_r2/<item>/scene_frames.json.
Sample times per shot = the V2 frame times (reviewed observations, final.json) plus a 1 s grid where V2 has none
within 0.5 s; each takes the nearest 10 fps frame (<= 0.1 s). Element = one SAM3 track in that frame (score, area,
track length filters), same-class duplicates merged by IoU / overlap; label = the GT id when geom/link ties the
track to one and a V2 row of the shot within 0.5 s lists that id, else the class. Relations are screen-space (left_of, above), single-frame depth (in_front_of,
occludes) and support (on: a subject with a GT id on another element; on_ground: any non-surface element); surfaces
only take part as the support of on_ground. No depth relation for support / contact pairs (geometry on, V2 contact)
or for a structure in front of a thing inside its box.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_root import gt_json  # noqa: E402
import paths as PT  # noqa: E402
import items as IT  # noqa: E402

R = PT.ROOT
TH_FILE = "scene_geom_thresholds_r1.json"               # r2: unchanged r1 values (relations, element rules)
TH = json.loads((PT.VIEW_CONF / TH_FILE).read_text())
RU = TH["rules_r1"]
OUT = PT.VIEW_OUT
GROUND = set(TH["surfaces"]["ground_like"])


def rj(p):
    return json.loads(Path(p).read_text())


def v2_rows(key):
    """{shot: [(item time, subject ids of the row, {(subject id, contact id)})]} from the reviewed V2 frame rows"""
    d = R / "annot/v1" / key
    v0, fin = rj(d / "v0.json"), rj(d / "final.json")["observations"]
    out = {}
    for u in v0["units"]:
        for fr in (fin.get(u["id"]) or {}).get("frames", []):
            sub = fr.get("subjects", [])
            ids = {x["id"] for x in sub if x.get("id")}
            con = {(x["id"], c) for x in sub for c in (x.get("contact") or []) if isinstance(c, str)}
            out.setdefault(u["shots"][0], []).append((round(u["clip_t"][0] + float(fr["t"]), 3), ids, con))
    return out


def rows_near(rows, t):
    w = TH["sample"]["v2_merge_s"]
    near = [r for r in rows if abs(r[0] - t) <= w]
    return set().union(*[r[1] for r in near]) if near else set(), set().union(*[r[2] for r in near]) if near else set()


def inside_frac(x, s):
    """share of box x inside box s"""
    ix = max(0.0, min(x[2], s[2]) - max(x[0], s[0])) * max(0.0, min(x[3], s[3]) - max(x[1], s[1]))
    return ix / max(1e-9, (x[2] - x[0]) * (x[3] - x[1]))


def sample_times(t0, t1, v2):
    ts = [(t, "v2") for t in sorted(v2) if t0 - 1e-6 <= t < t1]
    g = t0 + TH["sample"]["grid_offset_s"]
    while g < t1:
        if all(abs(g - t) > TH["sample"]["v2_merge_s"] for t, _ in ts):
            ts.append((round(g, 3), "grid"))
        g += TH["sample"]["grid_s"]
    if not ts:
        ts = [(round((t0 + t1) / 2, 3), "grid")]
    return sorted(ts)


def cell(cx, cy):
    r = "top" if cy < 1 / 3 else ("mid" if cy < 2 / 3 else "bottom")
    c = "left" if cx < 1 / 3 else ("center" if cx < 2 / 3 else "right")
    return "center" if (r, c) == ("mid", "center") else f"{r}-{c}"


def overlap1d(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0)) / max(1e-9, min(a1 - a0, b1 - b0))


def dilate(m):
    o = m.copy()
    o[1:] |= m[:-1]
    o[:-1] |= m[1:]
    o[:, 1:] |= m[:, :-1]
    o[:, :-1] |= m[:, 1:]
    return o


def relations(E, depth, h, w, contact=frozenset()):
    rt, out = TH["relations"], []
    pe = [i for i, e in enumerate(E) if not e["class"].startswith("surface:")]
    sup = set()                                        # r1: support / contact pairs get no depth order
    for x in range(len(pe)):
        for y in range(x + 1, len(pe)):
            i, j = pe[x], pe[y]
            if E[i]["label"] == E[j]["label"]:
                continue
            for s, t in ((i, j), (j, i)):
                if E[s]["id"] and on_top(E[s], E[t]["_m"], depth, h):
                    out.append({"a": s, "b": t, "rel": "on", "conf": "medium", "v": None})
                    sup.add(frozenset((i, j)))
            if RU["no_depth_on_support"] and ((E[i]["id"], E[j]["id"]) in contact or (E[j]["id"], E[i]["id"]) in contact):
                sup.add(frozenset((i, j)))
    struct = set(RU["structure_classes"])

    def depth_ok(f, k):
        """r1: f in front of / occluding k may be stated"""
        if RU["no_depth_on_support"] and frozenset((f, k)) in sup:
            return False
        return not (E[f]["class"] in struct and inside_frac(E[k]["bbox"], E[f]["bbox"]) >= RU["contained_frac"])

    for x in range(len(pe)):
        for y in range(x + 1, len(pe)):
            i, j = pe[x], pe[y]
            a, b = E[i], E[j]
            if a["label"] == b["label"]:
                continue
            (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = a["bbox"], b["bbox"]
            ox = overlap1d(ax0, ax1, bx0, bx1)
            if abs(a["c"][0] - b["c"][0]) >= rt["lr_min_dx"] and ox <= rt["lr_max_overlap"]:
                l, r = (i, j) if a["c"][0] < b["c"][0] else (j, i)
                out.append({"a": l, "b": r, "rel": "left_of", "conf": "high" if ox == 0 else "medium",
                            "v": round(abs(a["c"][0] - b["c"][0]), 3)})
            oy = overlap1d(ay0, ay1, by0, by1)
            if abs(a["c"][1] - b["c"][1]) >= rt["ud_min_dy"] and oy <= rt["ud_max_overlap"]:
                u, d = (i, j) if a["c"][1] < b["c"][1] else (j, i)
                out.append({"a": u, "b": d, "rel": "above", "conf": "high" if oy == 0 else "medium",
                            "v": round(abs(a["c"][1] - b["c"][1]), 3)})
            if depth is not None and a["depth_med"] and b["depth_med"]:
                f, k = (i, j) if a["depth_med"] < b["depth_med"] else (j, i)
                ratio = E[k]["depth_med"] / E[f]["depth_med"]
                if ratio >= rt["front_ratio"] and depth_ok(f, k):
                    out.append({"a": f, "b": k, "rel": "in_front_of",
                                "conf": "high" if ratio >= rt["front_ratio_high"] else "medium", "v": round(ratio, 2)})
            if depth is not None:
                ma, mb = a["_m"], b["_m"]
                ea, eb = dilate(ma) & mb, dilate(mb) & ma        # B pixels next to A, A pixels next to B
                if ea.sum() >= rt["occ_min_adj_px"] and eb.sum() >= rt["occ_min_adj_px"]:
                    da, db = depth[eb], depth[ea]
                    va, vb = da[da > 0], db[db > 0]
                    if len(va) >= rt["occ_min_valid"] * len(da) and len(vb) >= rt["occ_min_valid"] * len(db):
                        za, zb = float(np.median(va)), float(np.median(vb))
                        if zb / za >= rt["occ_ratio"] and depth_ok(i, j):
                            out.append({"a": i, "b": j, "rel": "occludes", "conf": "medium", "v": round(zb / za, 2)})
                        elif za / zb >= rt["occ_ratio"] and depth_ok(j, i):
                            out.append({"a": j, "b": i, "rel": "occludes", "conf": "medium", "v": round(za / zb, 2)})
    gm = np.zeros((h, w), bool)
    for e in E:
        if e["class"] in GROUND:
            gm |= e["_m"]
    if gm.any():
        for i in pe:
            if on_top(E[i], gm, depth, h):
                out.append({"a": i, "b": None, "rel": "on_ground", "conf": "medium", "v": None})
    return out


def on_top(a, mb, depth, h):
    """A's bottom edge rests on support mask mb: mb present in a thin band at A's bottom across enough of A's columns,
    with a similar depth there (when depth exists)."""
    rt = TH["relations"]
    ma = a["_m"]
    rows = np.where(ma.any(1))[0]
    cols = np.where(ma.any(0))[0]
    if not len(rows) or not len(cols):
        return False
    yb = rows[-1]
    y0, y1 = max(0, int(yb - rt["on_band_up"] * h)), min(h, int(yb + rt["on_band_down"] * h) + 1)
    band = mb[y0:y1, cols[0]:cols[-1] + 1]
    if band.any(0).mean() < rt["on_min_cols"]:
        return False
    if depth is None:
        return False                                   # touching in the image only is not support
    zb = depth[y0:y1, cols[0]:cols[-1] + 1][band]
    za = depth[max(0, yb - 3):yb + 1][:, cols[0]:cols[-1] + 1][ma[max(0, yb - 3):yb + 1, cols[0]:cols[-1] + 1]]
    zb, za = zb[zb > 0], za[za > 0]
    if len(zb) < rt["on_min_depth_px"] or len(za) < rt["on_min_depth_px"]:
        return True
    q = float(np.median(zb)) / float(np.median(za))
    return 1 / rt["on_depth_ratio"] <= q <= rt["on_depth_ratio"]


def shot_frames(item, key, sh, v2r, stats):
    g = R / "annot/v1" / key / "geom"
    sid = sh["id"]
    if not (g / "seg" / f"{sid}.json").exists():
        return {"shot": sid, "t": sh["t"], "space": sh.get("space"), "samples": [], "note": "no seg"}
    from pycocotools import mask as mu  # GT build only (the scoring path uses relations() alone)
    seg, fm = rj(g / "seg" / f"{sid}.json"), rj(g / "frames" / sid / "frames.json")
    link = rj(g / "link" / f"{sid}.json") if (g / "link" / f"{sid}.json").exists() else {"ids": {}}
    tid_id = {t: k for k, v in link.get("ids", {}).items() for t in v.get("tids", [])}
    W, H = int(seg["width"]), int(seg["height"])
    fps = float(fm["fps"])
    ft = {f["fi"]: f for f in fm["frames"]}
    npz = g / "recon" / f"{sid}.npz"
    D = np.load(npz) if npz.exists() else None
    dfi = {int(f): k for k, f in enumerate(D["fi"])} if D is not None else {}
    h, w = (D["depth"].shape[1:] if D is not None else (180, 320))
    ys = ((np.arange(h) + 0.5) * H / h).astype(int)
    xs = ((np.arange(w) + 0.5) * W / w).astype(int)
    et = TH["element"]
    st = stats.setdefault("shots", {}).setdefault(sid, {})
    st["raw_tracks"] = len(seg["tracks"])
    keep = {}
    for tr in seg["tracks"]:
        fis = [f["fi"] for f in tr["frames"]]
        if (max(fis) - min(fis) + 1) / fps < et["min_track_s"]:
            continue
        keep[tr["tid"]] = {f["fi"]: f for f in tr["frames"]
                           if f.get("score", 1) >= et["min_score"]
                           and (tr["tid"] in tid_id or f["area"] / (W * H) >= et["min_area_frac"])}
    cls = {tr["tid"]: tr["class"] for tr in seg["tracks"]}
    st["kept_tracks"] = sum(1 for v in keep.values() if v)
    used, merges, samples = set(), 0, []
    for t, src in sample_times(sh["t"][0], sh["t"][1], [r[0] for r in v2r]):
        fi = min(ft, key=lambda k: abs(ft[k]["t"] - t))
        if abs(ft[fi]["t"] - t) > TH["sample"]["max_frame_dt_s"]:
            samples.append({"t": t, "src": src, "fi": None, "note": "no frame within 0.1 s"})
            continue
        depth = D["depth"][dfi[fi]].astype(np.float32) if fi in dfi else None
        v2ids, contact = rows_near(v2r, t)
        E = []
        for tid, frs in keep.items():
            f = frs.get(fi)
            if not f:
                continue
            gid = tid_id.get(tid)
            if gid and RU["id_needs_v2_row"] and gid not in v2ids:   # r1: the id must be in a V2 row near t
                st["ids_dropped"] = st.get("ids_dropped", 0) + 1
                gid = None
                if f["area"] / (W * H) < et["min_area_frac"]:
                    continue
            m = mu.decode({"size": [H, W], "counts": f["rle"].encode()}).astype(bool)[ys][:, xs]
            if not m.any():
                continue
            E.append({"tids": [tid], "class": cls[tid], "id": gid, "_m": m, "area": int(f["area"])})
        E.sort(key=lambda e: -e["area"])
        M = []
        for e in E:
            for k in M:
                if k["class"] != e["class"] or (k["id"] and e["id"] and k["id"] != e["id"]):
                    continue
                inter = float((k["_m"] & e["_m"]).sum())
                iou = inter / max(1.0, float((k["_m"] | e["_m"]).sum()))
                if iou >= et["merge_iou"] or inter / max(1.0, float(min(k["_m"].sum(), e["_m"].sum()))) >= et["merge_overlap"]:
                    k["_m"] |= e["_m"]
                    k["tids"] += e["tids"]
                    k["id"] = k["id"] or e["id"]
                    merges += 1
                    break
            else:
                M.append(e)
        X = []                                         # the same mask under two prompts: keep the larger, note the other
        for e in M:
            for k in X:
                if k["class"] != e["class"] and not (k["id"] and e["id"]):
                    inter = float((k["_m"] & e["_m"]).sum())
                    if inter / max(1.0, float((k["_m"] | e["_m"]).sum())) >= et["cross_class_iou"]:
                        k.setdefault("alt_class", []).append(e["class"])
                        k["id"] = k["id"] or e["id"]
                        merges += 1
                        break
            else:
                X.append(e)
        M = X
        surf = [e for e in M if e["class"].startswith("surface:")]
        objs = [e for e in M if not e["class"].startswith("surface:")]
        objs = [e for e in objs if e["id"]] + [e for e in objs if not e["id"]][:max(0, et["max_elements"] - sum(1 for e in objs if e["id"]))]
        M = objs + surf
        for e in M:
            m = e["_m"]
            yy, xx = np.nonzero(m)
            e["label"] = e["id"] or e["class"]
            e["bbox"] = [round(float(xx.min()) / w, 3), round(float(yy.min()) / h, 3),
                         round(float(xx.max() + 1) / w, 3), round(float(yy.max() + 1) / h, 3)]
            e["c"] = [round(float(xx.mean() + 0.5) / w, 3), round(float(yy.mean() + 0.5) / h, 3)]
            e["cell"] = cell(*e["c"])
            e["area_frac"] = round(float(m.mean()), 4)
            e["depth_med"], e["depth_valid_frac"] = None, None
            if depth is not None:
                z = depth[m]
                zv = z[z > 0]
                e["depth_valid_frac"] = round(len(zv) / max(1, len(z)), 3)
                if len(zv) >= et["min_depth_px"] and e["depth_valid_frac"] >= et["min_depth_valid_frac"]:
                    e["depth_med"] = round(float(np.median(zv)), 3)
            used.update(e["tids"])
        rel = relations(M, depth, h, w, contact)
        comp = {"sky": 0.0, "ground": 0.0, "wall": 0.0}
        for e in surf:
            k = "sky" if e["class"] == "surface:sky" else ("ground" if e["class"] in GROUND else
                                                            ("wall" if e["class"] == "surface:wall" else None))
            if k:
                comp[k] = round(comp[k] + e["area_frac"], 4)
        samples.append({"t": t, "src": src, "fi": fi, "frame": f"geom/frames/{sid}/{ft[fi]['file']}",
                        "dt": round(ft[fi]["t"] - t, 3), "depth": depth is not None,
                        "depth_valid_frac": round(float((depth > 0).mean()), 3) if depth is not None else None,
                        "composition": comp,
                        "elements": [{k: v for k, v in e.items() if not k.startswith("_") and k != "area"} for e in M],
                        "relations": rel})
    st.update(used_tracks=len(used), merges=merges, samples=len(samples),
              samples_with_depth=sum(1 for s in samples if s.get("depth")))
    return {"shot": sid, "t": sh["t"], "space": sh.get("space"), "size": [W, H], "depth_hw": [int(h), int(w)],
            "recon_ok": bool(rj(g / "recon" / f"{sid}.json").get("ok")) if (g / "recon" / f"{sid}.json").exists() else None,
            "samples": samples}


def main():
    """GT build (r1 items; the other view items' scene_frames come from tools/v51_gtgeom, same rules)"""
    items = sys.argv[1:] or IT.of_kinds(IT.VIEW_KINDS)
    for item in items:
        key = IT.key_of(item)
        gt = rj(gt_json(item))
        v2 = v2_rows(key)
        stats = {}
        shots = [shot_frames(item, key, sh, v2.get(sh["id"], []), stats) for sh in gt["shots"]]
        S = [s for sh in shots for s in sh["samples"] if s.get("fi") is not None]
        dv = [s["depth_valid_frac"] for s in S if s.get("depth")]
        ne = [len(s["elements"]) for s in S]
        el = [e for s in S for e in s["elements"] if e["depth_valid_frac"] is not None]
        stats["total"] = {
            "raw_tracks": sum(v["raw_tracks"] for v in stats["shots"].values()),
            "kept_tracks": sum(v["kept_tracks"] for v in stats["shots"].values()),
            "used_tracks": sum(v["used_tracks"] for v in stats["shots"].values()),
            "merges": sum(v["merges"] for v in stats["shots"].values()),
            "samples": len(S), "samples_with_depth": len(dv),
            "frame_depth_valid_mean": round(float(np.mean(dv)), 3) if dv else None,
            "elements_per_sample_mean": round(float(np.mean(ne)), 2) if ne else 0,
            "elements_with_depth_frac": round(sum(1 for e in el if e["depth_med"]) / max(1, len(el)), 3),
            "relations": sum(len(s["relations"]) for s in S),
            "ids_dropped_r1": sum(v.get("ids_dropped", 0) for v in stats["shots"].values())}
        o = OUT / item
        o.mkdir(parents=True, exist_ok=True)
        (o / "scene_frames.json").write_text(json.dumps(
            {"item": item, "source_key": key, "thresholds": TH_FILE,
             "stats": stats, "shots": shots}, indent=1) + "\n")
        print(item, json.dumps(stats["total"]), flush=True)


if __name__ == "__main__":
    main()
