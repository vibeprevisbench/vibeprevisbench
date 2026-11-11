"""pvscore 5.1 text / image layer (r2), model side: read one pvscore 5.0.x dump into the plain format score.py scores.

    ssh ... "cd $B && env/bin/python -I -B tools/v51_text_r2/model_side.py --batch DIR [--batch DIR ...] [ITEM ...]"

r2: dumps from launcher batch directories (tools/v51_r2/batches.py: the latest batch wins per (model, item)); model =
the launcher model name; items = the text / image items of the item table (script, spatial_arkit, spatial_keyframe);
paths by tools/v51_r2/paths.py (output paths.TEXT_OUT/<item>/model/<model>.json). Reader unchanged from r1.

Uses tools/v51_r2/model_view.py (build_ctx, key_table, frame_elements; imported, not changed), tools/v51_r2/
view_thresholds.json and pvscore5 5.0.9. r1 change: the class exclusion compares normalised classes
(tools/v51_text_r2/classes.py), so other:<excluded class> is excluded too; the raw model class is kept in "cls" (score.py
normalises it). Output data/v51_text_r2/<item>/model/<model>.json:
  elements   candidates for the 3D facts, in model units: tagged furnishing instances "i<n>" (5.0.9 Model.layout;
             untagged, terrain and surface classes left out) with footprint centre / long / short / height and model
             space; named bodies "b:<id>" at the 5.0.9-warped GT reference time of the GT element (image: first frame
             the body is present)
  ent_xy     body centre for each GT R1 apex "b:<id>@<t>" at warp(t) (nearest present analysis frame within 0.5 s)
  person_h   median over person bodies of the 95th-percentile body extent (top - base), model units
  frames     model frame elements (model_view.frame_elements, cropped to the GT output aspect) for each GT frame id:
             text = 5.0.9 camera.model_time_for (cut warp) + nearest grid frame of that model shot, |dt| <= 0.3 s;
             image = the prompt frame round(t * fps), its model shot, nearest grid frame
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v51_r2"))
sys.path.insert(0, str(HERE))
import paths as PT  # noqa: E402
import model_view as MV  # noqa: E402
import classes as CL  # noqa: E402
import items as IT  # noqa: E402
import batches as BT  # noqa: E402
from pvscore5 import camera  # noqa: E402

R = PT.ROOT
TH = json.loads((PT.TEXT_CONF / "thresholds.json").read_text())
VT = json.loads((PT.VIEW_CONF / "view_thresholds.json").read_text())
GTD = PT.TEXT_GT                                       # <item>/gt_facts.json
OUT = PT.TEXT_OUT                                      # writes <item>/model/<model>.json


def nearest_present(md, b, tm, win):
    ai = md.t_to_ai(tm)
    if b.present[ai]:
        return ai
    cand = [x for x in range(md.A) if b.present[x] and abs(md.an_t[x] - tm) <= win]
    return min(cand, key=lambda x: abs(md.an_t[x] - tm)) if cand else None


def run(item, name, dump, write=True):
    t0 = time.time()
    dump = Path(dump)
    gtf = json.loads((GTD / item / "gt_facts.json").read_text())
    gt, md, ctx = MV.build_ctx(MV.gt_json(item), dump)
    tb = time.time() - t0
    image = gt.modality == "image"
    excl = set(TH["elements"]["exclude_classes"])
    els = {}
    for n, m in enumerate(md.instances):
        # fix r1 (10-18, found by T1 on M50_012): "platform" instances stay candidates (GT eligibility keeps the
        # class; model_view treats it as a walkable surface for the views only)
        if m.untagged or not m.cls or m.kind in MV.SURFACE_KINDS or CL.excluded(m.cls, excl):
            continue
        els["i%d" % n] = {"cls": m.cls, "space": m.space, "xy": [float(m.rect[0][0]), float(m.rect[0][1])],
                          "long": float(m.long), "short": float(m.short), "height": float(m.height)}
    # named bodies at the GT element's reference time
    for key, e in gtf["elements"].items():
        if not key.startswith("b:"):
            continue
        b = md.body.get(key[2:])
        if b is None:
            continue
        if image:
            pres = np.nonzero(b.present)[0]
            ai = int(pres[0]) if len(pres) else None
        else:
            ai = nearest_present(md, b, ctx.warp(e["t_ref"]), 1.0)
        if ai is None:
            continue
        els[key] = {"cls": b.cls, "space": b.space[ai], "xy": [float(b.center[ai][0]), float(b.center[ai][1])],
                    "long": float(b.long[ai]), "short": float(b.short[ai]), "height": float(b.top[ai] - b.base[ai])}
    ent = {}
    for k, a in gtf["entity_times"].items():
        b = md.body.get(a["eid"])
        ai = nearest_present(md, b, ctx.warp(a["t"]), 0.5) if b is not None else None
        ent[k] = None if ai is None else [float(b.center[ai][0]), float(b.center[ai][1])]
    ph = []
    for eid, b in md.body.items():
        if b.cls == "person" and b.present.any():          # 5.0.9 bodies: class person, group agent
            ph.append(float(np.nanpercentile((b.top - b.base)[b.present], 95)))
    person_h = float(np.median(ph)) if ph else None
    # frames
    tab = MV.key_table(md)
    W, H = gtf.get("res") or json.loads((PT.VIEW_GT / item / "task.json").read_text())[
        "output_spec"]["final_resolution"]
    aspect = W / H
    gshot = {s.id: s for s in gt.shots}
    frames, finfo = {}, {}
    for fr in gtf["frames"]:
        sh, t = fr["shot"], fr["t"]
        if image:
            k = md.t_to_frame_index(t)
            ms = md.shot_of_frame(k)
            tm = t
        else:
            ms, tm = camera.model_time_for(ctx, gshot[sh], t)
        gi = md.grid_near(tm, ms) if tm is not None else None
        dt = None if gi is None else abs(float(md.g_t[gi]) - tm)
        finfo[fr["fid"]] = [ms, None if tm is None else round(float(tm), 3), gi, None if dt is None else round(dt, 3)]
        if gi is None or dt > VT["frame_max_dt_s"]:
            frames[fr["fid"]] = None
            continue
        E, D, h, w, k = MV.frame_elements(md, tab, gi, aspect)
        frames[fr["fid"]] = [{x: e[x] for x in ("key", "class", "c", "cell", "height_frac", "depth_med", "facing_cam",
                                                "facing_unknown")      # 5.2 (M14): only set when unknown
                              if x in e} for e in E]
    tr = md.tag_report(set(gt.entities), [s.id for s in gt.spaces])
    res = {"item": item, "model": name, "dump": PT.rel(dump), "reader": "model_side.py (pvscore 5.1 r2)",
           "build_s": round(tb, 1), "n_instances": len(md.instances), "bodies": sorted(md.body),
           "tag_report": {k: v for k, v in tr.items() if k in ("tagged_area_frac", "entities_missing", "spaces_used")},
           "elements": els, "ent_xy": ent, "person_h": person_h, "frames": frames,
           "frame_info_cols": ["model_shot", "model_t", "gi", "dt"], "frame_info": finfo}
    if write:
        o = OUT / item / "model"
        o.mkdir(parents=True, exist_ok=True)
        (o / f"{name}.json").write_text(json.dumps(res, indent=0) + "\n")
    nf = sum(1 for v in frames.values() if v is not None)
    print(item, name, "build %.0fs" % tb, "elements", len(els), "bodies", res["bodies"], "ent", sum(
        v is not None for v in ent.values()), "/", len(ent), "frames", nf, "/", len(frames), "person_h", person_h,
          "total %.0fs" % (time.time() - t0), flush=True)
    return res


def main():
    a = sys.argv[1:]
    bdirs = [a[i + 1] for i, x in enumerate(a) if x == "--batch"]
    if not bdirs:
        sys.exit("give --batch DIR (launcher batch directories)")
    only = next((set(x.split("=", 1)[1].split(",")) for x in a if x.startswith("--model=")), None)
    text = set(IT.of_kinds(IT.TEXT_KINDS))
    items = [x for x in a if x in text] or sorted(text)
    for (name, item), att in sorted(BT.latest(bdirs).items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if item in items and (only is None or name in only) and (GTD / item / "gt_facts.json").is_file():
            run(item, name, att["out"])


if __name__ == "__main__":
    main()
