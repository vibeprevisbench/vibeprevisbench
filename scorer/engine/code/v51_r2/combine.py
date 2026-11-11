"""pvscore 5.1 r2: combine pvscore 5.0.x report blocks with the new layers into the nine-dimension total (JSON only).

    python3 -I -B tools/v51_r2/combine.py --batch DIR [--batch DIR ...] [--view PATH] [--text PATH] [--temporal PATH]
                                             [--weights PATH] [--out DIR] [--baseline PATH] [--tag NAME] [--all-controls]

  --batch      launcher batch directories: the 5.0.x report.json of each (model, item) is the one in the selfcheck
               directory of its attempt (batches.latest: the latest batch wins); model = launcher model name
  --view       view layer (default paths.VIEW_OUT/view_scores.json, from view_score.py)
  --view-extra more view files: models of an item that the main view file lacks are taken from them
  --text       text / image layer (default paths.TEXT_OUT/results.json; text / image items are skipped with a note
               when the file does not exist)
  --temporal   temporal layer (default paths.VIEW_OUT/temporal_scores.json; skipped when the file does not exist; not
               part of the 5.1 total, combine_weights temporal_map)
  --weights    slot table (default combine_weights_v51r2.json, pre-registered r2)
  --out        output dir (default paths.VIEW_OUT/combined)
  --baseline   real-model view layer the control totals are compared with (default: --view)
  --tag        output name suffix; default from the view file name (view_scores.json -> combined.json,
               view_scores_<x>.json -> combined_<x>.json)
  --all-controls  also combine every <VIEW_OUT>/controls/view_scores_*.json and view_scores_controls.json that exists,
               and write combined_controls.{json,md} with all control totals vs their real models (a control file
               whose "version" differs from --view is skipped as stale; real-model copies inside a control file are not
               control rows)

The rules (slots, members, fallbacks, N/A, and the r1 / r2 rules: member min n / informative flag, activation of
dimensions with at most max_509_blocks kept 5.0.x blocks, D1 content gate, only_modality) are documented in the
weights file (r2_rules; an r1 file's r1_rules are read the same way). 5.0.x totals are recomputed from the report
blocks with the 5.0.9 block weights as a parsing check. Controls: a model key '<model>~mirror' (or any key of a
controls/view_scores_<control>.json file) is combined with the 5.0.x report of its base model; the 5.0.x part is not
re-scored, so a control can only lose the points of the new layers. Writes <out>/combined[_tag].{json,md}.
The scoring driver (score51.py) calls combine_one / recompute_509 / dropped_509_points directly.
Text profile claims_v5 (text_profile_claims_v5.json, script items): profile_weights() puts its slot table over the r2
file; members of src v5 read the claims_score.py blocks (layout / entity / camera) passed as v5=, a 509 member with
drop = a1_counts_drop is recomputed from the report rows without the dropped count items, and the r2 activation
clause is waived for a dimension with a used v5 member (v5_rules).
r3 (weights with r3_rules, combine_weights_v51r3.json): the 5.0.x members of a dimension are fixed by the GT
(members_509.py, E = the item's members file, passed as E=): an expected block a delivered report lacks scores 0 with
the expected n; a block the GT does not have is not a member even when a report has it; a present block counts the
expected n (activation, min n); 'unless_509_na' / 'when_509_na' read the expected dims. Every such event is logged in
the row (r3_members). Without r3_rules the r2 behaviour is unchanged (E is refused).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as PT  # noqa: E402
import batches as BT  # noqa: E402

R = PT.ROOT
P = PT.VIEW_OUT
DIMS = ["A1", "A2", "A3", "P1", "B1", "B2", "C1", "C2", "D1"]
NEW_SRC = ("view", "temporal", "facts3d", "v5")
VERSION = "pvscore-5.1-r2"

# 5.0.9 block weights (pvscore5/items.py BLOCKS, unchanged in 5.0.10 / 5.0.11), only used to recompute the 5.0.x
# total as a check
BLOCKS_509 = {
    "A1": {"entities": 0.35, "counts": 0.35, "structure": 0.10, "zones": 0.10, "extras": 0.20},
    "A2": {"objects": 0.65, "relations": 0.15, "occupancy": 0.35},
    "A3": {"outline": 0.40, "openings": 0.25, "connectivity": 0.20, "space_size": 0.10, "floor_dz": 0.05},
    "P1": {"depth": 0.2, "normal": 0.2, "region": 0.2, "instance": 0.2, "edge": 0.2},
    "B1": {"shape": 0.25, "endpoints": 0.2, "waypoints": 0.1, "space_order": 0.1, "heading": 0.1, "progress": 0.1,
           "pair_dist": 0.1, "stops_turns": 0.05},
    "B2": {"occurrence": 0.45, "timing": 0.2, "order": 0.15, "posture": 0.15, "relations_t": 0.05},
    "C1": {"position": 0.2, "height": 0.1, "direction": 0.2, "fov": 0.1, "subject": 0.15, "in_frame": 0.1,
           "framing": 0.05, "move": 0.1, "spec": 0.3},
    "C2": {"count": 0.15, "cuts": 0.25, "transitions": 0.15, "shot_order": 0.15, "space_order": 0.1,
           "screen_dir": 0.05, "durations": 0.15, "view_order": 1.0},
    "D1": {"penetration": 0.25, "float_sink": 0.2, "teleport": 0.2, "class_size": 0.15, "scale": 0.1,
           "duplicates": 0.1},
}


def rel(p):
    return PT.rel(p)


def is_baseline(name):
    """the launcher's fixture model (batch.json dry_run_model, 'dry-run-fake'); 'fake' = its r1 short name"""
    return name == "fake" or name.startswith("dry-run")


def check_report(rep, base, item):
    """-> error string or None: the report must be the given item's and its run directory must name the model"""
    run = str((rep.get("harness") or {}).get("run") or "")
    if base not in run:
        return "report run '%s' does not name %s" % (run, base)
    if rep.get("item_id") != item:
        return "report item %s != %s" % (rep.get("item_id"), item)
    return None


def batch_reports(batch_dirs):
    """report.json loader over launcher batches: (model, item) -> (report, path, error)"""
    idx = BT.latest(batch_dirs)

    def get(base, item):
        a = idx.get((base, item))
        if a is None:
            return None, None, "no attempt with a dump for model '%s' in the batches" % base
        if not a["report"]:
            return None, rel(Path(a["out"]) / "report.json"), "missing report.json in %s" % rel(a["out"])
        rep = json.loads(Path(a["report"]).read_text())
        err = check_report(rep, base, item)
        return (None if err else rep), rel(a["report"]), err
    return get


def recompute_509(rep, dw):
    num = den = 0.0
    for d in DIMS:
        dv = rep["dims"].get(d)
        if not dv:
            continue
        bs = {b: x for b, x in dv["blocks"].items() if b in BLOCKS_509[d]}
        if not bs:
            continue
        s = sum(BLOCKS_509[d][b] * x["score"] for b, x in bs.items()) / sum(BLOCKS_509[d][b] for b in bs)
        num += dw[d] * s
        den += dw[d]
    return 100 * num / den if den else 0.0


def dropped_509_points(rep, W):
    """points of the 5.0.9 total that came from blocks 5.1 drops (dropped_509 keys 'DIM.block'; fallback-only
    blocks are listed there too and count here even when a fallback brings them back)"""
    drop = {tuple(k.split(".")) for k in W["dropped_509"]}
    den = sum(W["dim_weights"][d] for d in DIMS if rep["dims"].get(d))
    out = {}
    for d in DIMS:
        dv = rep["dims"].get(d)
        if not dv:
            continue
        sw = sum(BLOCKS_509[d][b] for b in dv["blocks"] if b in BLOCKS_509[d])
        for b, x in dv["blocks"].items():
            if (d, b) in drop and sw:
                out[d + "." + b] = round(100 * W["dim_weights"][d] / den * BLOCKS_509[d][b] / sw * x["score"], 3)
    return round(sum(out.values()), 3), out


# ----------------------------------------------------------------------------------------------- members
def rules_of(W):
    return W.get("r2_rules") or W.get("r1_rules") or {}


def informative(b, W):
    f = rules_of(W).get("informative_field", {})
    return not (b.get(f.get("false_key", "informative"), True) is False or b.get(f.get("true_key", "low_info")))


def view_member(vm, m, W):
    """-> (score, n, informative) or None"""
    if vm is None:
        return None
    if m.get("checks"):
        if m["block"] != "presence":
            raise ValueError("check subsets are only defined for presence (chance R = 0)")
        ents = vm.get("entries")
        if ents is None:
            raise ValueError("view model has no entries; cannot split presence by check type")
        rows = [e for e in ents if e["block"] == m["block"] and e["check"] in m["checks"]]
        if not rows:
            return None
        pres = [e for e in rows if e["present"]]
        cov = len(pres) / len(rows)
        pm = sum(e["score"] for e in pres) / len(pres) if pres else 0.0
        return pm * cov, len(rows), True
    b = (vm.get("blocks") or {}).get(m["block"])
    if not b or not b.get("n") or b.get("score") is None:
        return None
    return float(b["score"]), int(b["n"]), informative(b, W)


def temporal_blocks(tm, rules):
    """temporal model entry -> ({target: (score, n)}, [unmatched names])"""
    out, unmatched = {}, []
    if not tm:
        return out, unmatched
    acc = {}
    for name, b in (tm.get("blocks") or {}).items():
        tgt = next((r["target"] for r in rules if name.startswith(r["prefix"])), None)
        if tgt is None:
            unmatched.append(name)
            continue
        n = int(b.get("n") or 0)
        if n <= 0:
            continue
        s, k = acc.get(tgt, (0.0, 0))
        acc[tgt] = (s + float(b["score"]) * n, k + n)
    for tgt, (s, k) in acc.items():
        out[tgt] = (s / k, k)
    return out, unmatched


def profile_weights(W, prof):
    """r2 weights with a text profile's slot table (text_profile_claims_v5.json): dims, dropped_509 and v5_rules
    replaced; dimension weights, r2_rules (member min n, activation, d1_gate) and gates kept"""
    return dict(W, dims=prof["dims"], dropped_509=prof["dropped_509"], v5_rules=prof["v5_rules"],
                version="%s+%s" % (W.get("version"), prof["profile"]), text_profile=prof["profile"])


def dropped_rows(rep, d, block, ids):
    """a 5.0.x block recomputed without the report item rows named in ids (weighted mean over the rows of weight > 0,
    as pvscore5 items.dim_scores) -> (score or None, n kept, n dropped, ids not found among the rows)"""
    rows = [r for r in (rep or {}).get("items") or [] if r.get("dim") == d and r.get("block") == block
            and r.get("weight", 0) > 0]
    names = {r.get("name") for r in rows}
    keep = [r for r in rows if r.get("name") not in set(ids)]
    if rows:                                          # parsing check: the rows give the reported block score
        full = sum(r["weight"] * r["score"] for r in rows) / sum(r["weight"] for r in rows)
        rb = rep["dims"][d]["blocks"][block]["score"]
        if abs(full - rb) > 1e-3:
            raise ValueError("%s.%s: report rows give %.4f, block says %.4f" % (d, block, full, rb))
    s = sum(r["weight"] * r["score"] for r in keep) / sum(r["weight"] for r in keep) if keep else None
    return s, len(keep), len(rows) - len(keep), sorted(set(ids) - names)


def r3_on(W, rep):
    """the r3 member rule applies: r3 weights and a delivered report (an undelivered one keeps the r2 behaviour)"""
    return bool(W.get("r3_rules")) and rep is not None and rep.get("delivered", True) is not False


def member_509_r3(m, d, rep, v5, E, log):
    """r3 rule B1 for a 5.0.x member -> (score, n, informative) or None; log: absent_expected, unexpected_present,
    n_report_differs"""
    key = "%s.%s" % (d, m["block"])
    ex = E["blocks"].get(key)
    b = ((rep["dims"].get(d) or {}).get("blocks") or {}).get(m["block"])
    if ex is None:
        if b:
            log["unexpected_present"][key] = int(b["n"])
        return None
    if m.get("drop"):
        if v5 is None or m["drop"] not in v5:
            raise ValueError("%s.%s: drop list %s needs the v5 result" % (d, m["block"], m["drop"]))
        if not b:                                   # n: the expected rows less the dropped ones (one row per id)
            n = ex["n"] - len(set(v5[m["drop"]]))
            if n <= 0:
                return None
            log["absent_expected"][key] = n
            return 0.0, n, True
        s, n, _, _ = dropped_rows(rep, d, m["block"], v5[m["drop"]])
        return (s, n, True) if s is not None else None
    if not b:
        log["absent_expected"][key] = ex["n"]
        return 0.0, ex["n"], True
    if int(b["n"]) != ex["n"]:
        log["n_report_differs"][key] = [int(b["n"]), ex["n"]]
    return float(b["score"]), ex["n"], True


def member_value(m, d, rep, vm, tb, f3, W, v5=None, E=None, log=None):
    """-> (score, n, informative) or None"""
    src = m["src"]
    if src == "509" and E is not None:
        return member_509_r3(m, d, rep, v5, E, log)
    if src == "509" and m.get("drop"):
        if v5 is None or m["drop"] not in v5:
            raise ValueError("%s.%s: drop list %s needs the v5 result" % (d, m["block"], m["drop"]))
        b = ((rep["dims"].get(d) if rep else None) or {}).get("blocks", {}).get(m["block"])
        if not b:
            return None
        s, n, _, _ = dropped_rows(rep, d, m["block"], v5[m["drop"]])
        return (s, n, True) if s is not None else None
    if src == "v5":
        b = ((v5 or {}).get("blocks") or {}).get(m["block"])
        if not b or not b.get("n") or b.get("score") is None:
            return None
        return float(b["score"]), int(b["n"]), True
    if src == "509":
        dv = rep["dims"].get(d) if rep else None
        b = (dv or {}).get("blocks", {}).get(m["block"])
        return (float(b["score"]), int(b["n"]), True) if b else None
    if src == "view":
        return view_member(vm, m, W)
    if src == "temporal":
        v = tb.get(m["block"])
        return (v[0], v[1], True) if v else None
    if src == "facts3d":
        b = (f3 or {}).get(m["block"])
        if not b or not b.get("n") or b.get("score") is None:
            return None
        return float(b["score"]), int(b["n"]), informative(b, W)
    raise ValueError(src)


# ------------------------------------------------------------------------------------------------ combine
def combine_one(W, rep, vm, tb, f3=None, v5=None, E=None):
    R1 = rules_of(W)
    min_n = R1.get("member_min_n", 0)
    V5R = W.get("v5_rules") or {}
    modality = (rep or {}).get("modality") or "video"
    dims, num, den, gated, weak = {}, 0.0, 0.0, [], []
    r3 = r3_on(W, rep)
    if E is not None and not W.get("r3_rules"):
        raise ValueError("a members file (E) is only read under r3 weights (r3_rules)")
    if r3 and E is None:
        raise ValueError("r3 weights need the item's members_509 file (E)")
    if r3 and E.get("item") != rep.get("item_id"):
        raise ValueError("members file of %s for a report of %s" % (E.get("item"), rep.get("item_id")))
    E = E if r3 else None
    log = {"absent_expected": {}, "unexpected_present": {}, "n_report_differs": {}}

    def na509(x):                                   # r2: the report's dim; r3: the expected dims
        if E is not None:
            return x not in E["dims"]
        return rep is None or rep["dims"].get(x) is None
    for d in DIMS:
        slots, kept509 = [], set()
        for sl in W["dims"][d]:
            if sl.get("unless_509_na") and na509(sl["unless_509_na"]):
                continue
            if sl.get("when_509_na") and not na509(sl["when_509_na"]):
                continue
            vals = []
            for m in sl["members"]:
                if m.get("only_modality") and modality not in m["only_modality"]:
                    continue
                v = member_value(m, d, rep, vm, tb, f3, W, v5, E, log)
                if v is None:
                    continue
                mn = V5R.get("member_min_n", min_n) if m["src"] == "v5" else min_n
                if m["src"] in NEW_SRC and (v[1] < mn or not v[2]):        # r1 rule (1); v5: present at n >= 5
                    weak.append("%s.%s %s:%s n=%d%s" % (d, sl["slot"], m["src"], m["block"], v[1],
                                                        "" if v[2] else " not informative"))
                    continue
                vals.append((m, v[:2]))
            use = [x for x in vals if not x[0].get("fallback")] or [x for x in vals if x[0].get("fallback")]
            if not use:
                continue
            kept509 |= {x[0]["block"] for x in use if x[0]["src"] == "509"}   # r2: kept = used (fallbacks when used)
            fs = sum(x[0].get("frac", 1.0) for x in use)
            mem = [{"src": m["src"], "block": m["block"] + ("[%s]" % ",".join(m["checks"]) if m.get("checks") else ""),
                    "frac": m.get("frac", 1.0) / fs, "score": round(v[0], 4), "n": v[1],
                    "fallback": bool(m.get("fallback"))} for m, v in use]
            s = sum(x["frac"] * x["score"] for x in mem)
            slots.append({"slot": sl["slot"], "w": sl["w"], "score": round(s, 4), "members": mem})
        if not slots:
            dims[d] = None
            continue
        act = R1.get("activation")
        waived = V5R.get("activation_waived") and any(m["src"] == "v5" for x in slots for m in x["members"])
        if act and not waived and len(kept509) <= act.get("max_509_blocks", 0):   # r1 rule (2), r2 extended
            n_all = sum(m["n"] for x in slots for m in x["members"])
            if len(slots) < act["new_only_min_slots"] or n_all < act["new_only_min_n"]:
                gated.append({"dim": d, "slots": [x["slot"] for x in slots], "n": n_all, "kept_509": sorted(kept509),
                              "why": "%d kept 5.0.x block(s) (<= %d); %d slot(s), %d facts < %d slots / %d facts" % (
                                  len(kept509), act.get("max_509_blocks", 0), len(slots), n_all,
                                  act["new_only_min_slots"], act["new_only_min_n"])})
                dims[d] = None
                continue
        sw = sum(x["w"] for x in slots)
        score = sum(x["w"] * x["score"] for x in slots) / sw
        new = sum(x["w"] / sw * sum(m["frac"] * m["score"] for m in x["members"] if m["src"] in NEW_SRC) for x in slots)
        new_max = sum(x["w"] / sw * sum(m["frac"] for m in x["members"] if m["src"] in NEW_SRC) for x in slots)
        dims[d] = {"score": round(score, 4), "weight": W["dim_weights"][d], "new_part": new, "new_max": new_max,
                   "slots": slots, "new_only": not kept509}
    g1 = R1.get("d1_gate")
    d1_gate = None
    if g1 and dims.get("D1"):                                                 # r1 rule (3)
        cs = {c: dims[c]["score"] for c in g1["content_dims"] if dims.get(c)}
        if not cs:
            d1_gate = {"content": {}, "E": None, "g": None, "raw": dims["D1"]["score"], "why": "no content dim active: D1 N/A"}
            dims["D1"] = None
        else:
            ec = sum(cs.values()) / len(cs)             # (named E in r2; E is the r3 members file here)
            g = min(1.0, ec / g1["full_at"])
            raw = dims["D1"]["score"]
            dims["D1"]["score"] = round(raw * g, 4)
            dims["D1"]["gate"] = d1_gate = {"content": cs, "E": round(ec, 4), "g": round(g, 4), "raw": raw}
    for d in DIMS:
        if dims[d]:
            num += W["dim_weights"][d] * dims[d]["score"]
            den += W["dim_weights"][d]
    total = 100 * num / den if den else 0.0
    new_pts = sum(100 * v["weight"] / den * v["new_part"] for v in dims.values() if v) if den else 0.0
    new_max = sum(100 * v["weight"] / den * v["new_max"] for v in dims.values() if v) if den else 0.0
    by_slot = []
    for d, v in dims.items():
        if not v:
            continue
        sw = sum(x["w"] for x in v["slots"])
        for x in v["slots"]:
            p = sum(m["frac"] * m["score"] for m in x["members"] if m["src"] in NEW_SRC)
            if any(m["src"] in NEW_SRC for m in x["members"]):
                by_slot.append(("%s.%s" % (d, x["slot"]), round(100 * v["weight"] / den * x["w"] / sw * p, 3)))
        v["new_part"], v["new_max"] = round(v["new_part"], 4), round(v["new_max"], 4)
    by_slot.sort(key=lambda z: -z[1])
    out = {"total": round(total, 3), "new_layer_points": round(new_pts, 3), "new_layer_max_points": round(new_max, 3),
           "new_layer_by_slot": by_slot, "dims": dims, "modality": modality, "gated_dims": gated, "weak_members": weak,
           "d1_gate": d1_gate}
    if E is not None:
        out["r3_members"] = dict({k: dict(sorted(v.items())) for k, v in log.items()}, item=E["item"],
                                 gt_sha256=E.get("gt_sha256"), pvscore_version=E.get("pvscore_version"))
    return out


def row_of(W, rep, rpath, vm, tb, f3, base, ctl=None, v5=None, E=None):
    """one combined row: the 5.1 total (combine_one) + the 5.0.x reference fields; shared with score51.py"""
    c = combine_one(W, rep, vm, tb, f3, v5, E)
    drops = {}
    for d in DIMS:
        for sl in W["dims"][d]:
            for m in sl["members"]:
                if m["src"] == "509" and m.get("drop") and v5 is not None and (rep["dims"].get(d) or {}).get(
                        "blocks", {}).get(m["block"]):
                    s, n, k, nf = dropped_rows(rep, d, m["block"], v5[m["drop"]])
                    drops["%s.%s" % (d, m["block"])] = {"rule": m["drop"], "kept": n, "dropped": k,
                                                        "score_full": rep["dims"][d]["blocks"][m["block"]]["score"],
                                                        "score_kept": None if s is None else round(s, 4),
                                                        "ids_not_in_report": nf}
    if drops:
        c["dropped_rows"] = drops
    dims509 = {d: (round(rep["dims"][d]["score"], 4) if rep["dims"].get(d) else None) for d in DIMS}
    dp, dpd = dropped_509_points(rep, W)
    return {"base_model": base, "control": ctl, "report": rpath, "report_schema": rep.get("schema"),
            "total_509": rep["total"], "total_509_recomputed": round(recompute_509(rep, W["dim_weights"]), 3),
            "dims_509": dims509, "dropped_509_points": dp, "dropped_509_detail": dpd, **c}


def text_models(text_path):
    """text layer results -> {item: {model: {"vm": {"blocks": view blocks}, "f3": 3D blocks}}}"""
    d = json.loads(Path(text_path).read_text())
    out = {}
    for item, ms in d["scores"].items():
        for k, v in ms.items():
            if k.startswith("model:"):
                out.setdefault(item, {})[k[len("model:"):]] = {"vm": {"blocks": v.get("blocks_view") or {}},
                                                               "f3": v.get("blocks3d") or {}}
    return out


def load_members(d):
    """r3: members_509 version dir -> {item: members record} (index.json set sha256 recomputed and checked)"""
    d = Path(d)
    idx = json.loads((d / "index.json").read_text())
    got = hashlib.sha256("".join("%s %s\n" % (hashlib.sha256((d / v["file"]).read_bytes()).hexdigest(), v["file"])
                                 for _, v in sorted(idx["items"].items())).encode()).hexdigest()
    if got != idx["set_sha256"]:
        raise ValueError("%s: member files do not give the index set_sha256" % d)
    return {item: json.loads((d / v["file"]).read_text()) for item, v in idx["items"].items()}, idx


def run(W, get_report, view_path, temporal_path, tag, extra=(), text_path=None, members=None):
    view = json.loads(Path(view_path).read_text())
    temporal = json.loads(Path(temporal_path).read_text()) if temporal_path and Path(temporal_path).exists() else None
    rules = W["temporal_map"]["rules"]
    file_ctl = tag_of(view_path) if Path(view_path).resolve().parent.name == "controls" else ""
    out = {"weights_version": W["version"], "view": rel(view_path), "view_extra": [], "text": None,
           "temporal": rel(temporal_path) if temporal else None, "tag": tag, "items": {}, "reference_only": {},
           "notes": []}
    work = {item: {k: (vm, None) for k, vm in models.items()} for item, models in view["items"].items()}
    for xp in extra:
        if not Path(xp).exists():
            continue
        out["view_extra"].append(rel(xp))
        for item, models in json.loads(Path(xp).read_text())["items"].items():
            for k, vm in models.items():
                if item in work and k not in work[item] and k != "video_geom":
                    work[item][k] = (vm, None)
                    out["notes"].append("%s %s: view blocks from %s" % (item, k, rel(xp)))
    if text_path is not None:
        if Path(text_path).exists():
            out["text"] = rel(text_path)
            for item, ms in text_models(text_path).items():
                work[item] = {k: (x["vm"], x["f3"]) for k, x in ms.items()}
        else:
            out["notes"].append("text layer %s absent: text / image items not combined" % rel(text_path))
    for item, models in work.items():
        res = {}
        for key, (vm, f3) in models.items():
            base, _, ctl = key.partition("~")
            ctl = ctl or file_ctl
            label = base + ("~" + ctl if ctl else "")
            rep, rpath, err = get_report(base, item)
            if rep is None:
                out["reference_only"].setdefault(item, {})[key] = {
                    "why": err, "view_blocks": {b: {k: x.get(k) for k in ("score", "P", "R", "coverage", "n")}
                                                for b, x in (vm.get("blocks") or {}).items()}}
                continue
            tm = (temporal or {}).get("items", {}).get(item, {})
            tb, unmatched = temporal_blocks(tm.get(key) or tm.get(base), rules)
            if unmatched:
                out["notes"].append("%s %s: temporal blocks not mapped: %s" % (item, key, ", ".join(sorted(unmatched))))
            res[label] = row_of(W, rep, rpath, vm, tb, f3, base, ctl or None,
                                E=(members or {}).get(item) if W.get("r3_rules") else None)
        out["items"][item] = res
    out["checks"] = checks(W, out)
    return out


def checks(W, out):
    G = W["gates"]
    ck = {"recompute_509": [], "gates": {}, "ranking": {}, "block_presence_by_model": [], "na_by_model": [],
          "revived_dims": []}
    for item, res in out["items"].items():
        real = {k: v for k, v in res.items() if not v["control"]}
        for k, v in res.items():
            if abs(v["total_509"] - v["total_509_recomputed"]) > 0.05:
                ck["recompute_509"].append("%s %s: report %.3f vs recomputed %.3f" % (item, k, v["total_509"],
                                                                                    v["total_509_recomputed"]))
        g = {}
        fk = sorted(k for k in real if is_baseline(k))
        if fk:
            f = real[fk[0]]
            g["baseline_model"] = fk[0]
            g["fake_total_51"] = f["total"]
            g["fake_max_total"] = {"limit": G["fake_max_total"], "pass": f["total"] <= G["fake_max_total"]}
            g["fake_not_above_509"] = {"total_509": f["total_509"], "pass": f["total"] <= f["total_509"] + 1e-9}
            # r2: the r1 gate 'astra > fake' names its model in gates.above_baseline_models (launcher names)
            for m in G.get("above_baseline_models", []):
                if m in real:
                    g.setdefault("above_baseline", {})[m] = {
                        "model": real[m]["total"], "fake": f["total"], "pass": real[m]["total"] > f["total"],
                        "margin_51": round(real[m]["total"] - f["total"], 3),
                        "margin_509": round(real[m]["total_509"] - f["total_509"], 3)}
        ck["gates"][item] = g
        if real:
            r509 = sorted(real, key=lambda k: -real[k]["total_509"])
            r51 = sorted(real, key=lambda k: -real[k]["total"])
            ck["ranking"][item] = {"509": r509, "51": r51, "same": r509 == r51}
        # dims N/A in 5.0.9 that the new layers make active (their whole weight rests on the new members)
        for k, v in list(real.items())[:1]:
            for d in DIMS:
                if v["dims_509"][d] is None and v["dims"][d]:
                    mem = ["%s:%s n=%d" % (m["src"], m["block"], m["n"]) for s_ in v["dims"][d]["slots"]
                           for m in s_["members"]]
                    ck["revived_dims"].append("%s %s (weight %d): %s" % (item, d, W["dim_weights"][d], ", ".join(mem)))
        # GT-driven structure: the same members and N/A dims for every real model of an item
        sig = {}
        for k, v in real.items():
            sig[k] = ({d: sorted((s["slot"], m["src"], m["block"]) for s in (v["dims"][d] or {}).get("slots", [])
                                 for m in s["members"]) for d in DIMS},
                      sorted(d for d in DIMS if v["dims"][d] is None))
        ks = sorted(sig)
        for a in ks[1:]:
            for d in DIMS:
                x, y = set(sig[ks[0]][0][d]), set(sig[a][0][d])
                if x != y:
                    ck["block_presence_by_model"].append("%s %s: %s has %s, %s has %s" % (
                        item, d, ks[0], sorted("%s:%s" % (t[1], t[2]) for t in x - y) or "-",
                        a, sorted("%s:%s" % (t[1], t[2]) for t in y - x) or "-"))
            if sig[ks[0]][1] != sig[a][1]:
                ck["na_by_model"].append("%s: N/A %s=%s vs %s=%s" % (item, ks[0], sig[ks[0]][1], a, sig[a][1]))
    return ck


# ---------------------------------------------------------------------------------------------- markdown
def fmt(x, nd=1):
    return "–" if x is None else ("%.*f" % (nd, x))


def write_md(out, path, baseline=None):
    L = ["# pvscore 5.1 r2: combined totals%s" % (" (%s)" % out["tag"] if out["tag"] else ""), "",
         "Generated by `combine.py` (pvscore 5.1 r2) from `%s`%s, weights `%s` (%s, sha256 %s…). 5.0.9 blocks from the "
         "report.json of each dump; new layers = view%s." % (
             out["view"], " + `%s`" % out["temporal"] if out["temporal"] else "", out["weights_file"],
             out["weights_version"], out["weights_sha256"][:12],
             " + temporal" if out["temporal"] else " (temporal_scores.json absent: hooks empty)"),
         "", "Labels '5.0.9' / '5.0.x' in this file mean the pvscore 5.0.x report the slots read (report_schema; block weights "
         "unchanged from 5.0.9).", ""]
    L += ["## Totals", "",
          "| item | model | 5.0.9 | 5.1 | Δ | new-layer pts (max) | 5.0.9 pts in dropped blocks | " + " | ".join(DIMS) + " |",
          "|---|---|---|---|---|---|---|" + "---|" * len(DIMS)]
    for item, res in out["items"].items():
        for k, v in res.items():
            cells = []
            for d in DIMS:
                a, b = v["dims_509"][d], (v["dims"][d] or {}).get("score")
                cells.append("%s→%s" % (fmt(a, 2), fmt(b, 2)))
            L.append("| %s | %s | %.1f | %.1f | %+.1f | %.1f (%.1f) | %.1f | %s |" % (
                item, k, v["total_509"], v["total"], v["total"] - v["total_509"], v["new_layer_points"],
                v["new_layer_max_points"], v["dropped_509_points"], " | ".join(cells)))
    L += ["", "Dim cells: 5.0.9 → 5.1 dimension score (– = N/A). new-layer pts: points of the 5.1 total that come "
          "from view / temporal members; max = the same if every new member scored 1. 5.0.9 pts in dropped blocks: points "
          "of the 5.0.9 total that came from blocks 5.1 drops or keeps only as fallback (combine_weights.json dropped_509).",
          ""]
    L += ["## Where the new-layer points come from", "", "| item | model | slots (points of the 5.1 total) |", "|---|---|---|"]
    for item, res in out["items"].items():
        for k, v in res.items():
            L.append("| %s | %s | %s |" % (item, k, ", ".join("%s %.2f" % x for x in v["new_layer_by_slot"] if x[1] > 0.005)
                                           or "–"))
    L.append("")
    ck = out["checks"]
    L += ["## Gates", "", "| item | fake 5.1 ≤ %.0f | fake 5.1 ≤ fake 5.0.x | model > fake (margin 5.1 / 5.0.x) |" %
          next((g["fake_max_total"]["limit"] for g in ck["gates"].values() if g.get("fake_max_total")), 25),
          "|---|---|---|---|"]
    for item, g in ck["gates"].items():
        if not g:
            L.append("| %s | – | – | – |" % item)
            continue
        ab = g.get("above_baseline") or {}
        L.append("| %s | %s (%.1f) | %s (5.0.x %.1f) | %s |" % (
            item, "pass" if g["fake_max_total"]["pass"] else "FAIL", g["fake_total_51"],
            "pass" if g["fake_not_above_509"]["pass"] else "FAIL", g["fake_not_above_509"]["total_509"],
            "; ".join("%s %s (%+.1f / %+.1f)" % (m, "pass" if a["pass"] else "FAIL", a["margin_51"], a["margin_509"])
                      for m, a in sorted(ab.items())) or "–"))
    L += ["", "## Ranking per item", "", "| item | 5.0.9 | 5.1 | same |", "|---|---|---|---|"]
    for item, r in ck["ranking"].items():
        L.append("| %s | %s | %s | %s |" % (item, " > ".join(r["509"]), " > ".join(r["51"]), "yes" if r["same"] else "no"))
    if baseline:
        L += ["", "## Controls vs real model", "", "| item | control | base 5.1 | control 5.1 | Δ | drops |",
              "|---|---|---|---|---|---|"]
        for item, res in out["items"].items():
            for k, v in res.items():
                b = baseline.get(item, {}).get(v["base_model"])
                if v["control"] and b is not None:
                    L.append("| %s | %s | %.1f | %.1f | %+.2f | %s |" % (
                        item, k, b, v["total"], v["total"] - b, ("yes" if v["total"] < b - 1e-9 else "NO") +
                        (" (fake: nothing to lose, informational)" if is_baseline(v["base_model"]) else "")))
    L += ["", "## Per-dimension slots (5.1)", ""]
    for item, res in out["items"].items():
        for k, v in res.items():
            L.append("**%s %s**" % (item, k))
            L.append("")
            for d in DIMS:
                dv = v["dims"][d]
                if not dv:
                    L.append("- %s: N/A" % d)
                    continue
                parts = []
                for s in dv["slots"]:
                    ms = ", ".join("%s:%s %.2f%s" % (m["src"], m["block"], m["score"], " fb" if m["fallback"] else "")
                                   for m in s["members"])
                    parts.append("%s %.2f [w %.2f; %s]" % (s["slot"], s["score"], s["w"], ms))
                L.append("- %s %.3f: %s" % (d, dv["score"], "; ".join(parts)))
            L.append("")
    if out["reference_only"]:
        L += ["## Reference only (no 5.0.9 report)", ""]
        for item, ms in out["reference_only"].items():
            for k, v in ms.items():
                L.append("- %s %s (%s): %s" % (item, k, v["why"], ", ".join(
                    "%s %s (P %s / R %s / cov %s / n %s)" % (b, fmt(x["score"], 2), fmt(x["P"], 2), fmt(x["R"], 2),
                                                             fmt(x["coverage"], 2), x["n"])
                    for b, x in v["view_blocks"].items())))
        L.append("")
    L += ["## r1 / r2 rules applied (weights file r2_rules)", "",
          "| item | model | D1 raw → gated (E over A2/B1/B2, g) | dims N/A by the activation rule | new-layer members "
          "dropped (n < min or not informative) |", "|---|---|---|---|---|"]
    for item, res in out["items"].items():
        for k, v in res.items():
            g = v.get("d1_gate")
            gs = "–" if not g else ("%.2f → N/A (%s)" % (g["raw"], g["why"]) if g.get("E") is None else
                                    "%.2f → %.2f (E %.2f, g %.2f)" % (g["raw"], g["raw"] * g["g"], g["E"], g["g"]))
            L.append("| %s | %s | %s | %s | %s |" % (
                item, k, gs, "; ".join("%s (%s)" % (x["dim"], x["why"]) for x in v.get("gated_dims") or []) or "–",
                "; ".join(v.get("weak_members") or []) or "–"))
    L += ["", "## Structure checks", ""]
    L.append("- 5.0.9 total recomputed from report blocks: %s" % ("all within 0.05" if not ck["recompute_509"]
                                                                   else "; ".join(ck["recompute_509"])))
    L.append("- members present for one model and not another (model-dependent 5.0.9 blocks): %s" % (
        "none" if not ck["block_presence_by_model"] else ""))
    for x in ck["block_presence_by_model"]:
        L.append("  - %s" % x)
    L.append("- N/A dims differing by model: %s" % ("none" if not ck["na_by_model"] else "; ".join(ck["na_by_model"])))
    L.append("- dims N/A in 5.0.9 that only the new layers make active: %s" % ("none" if not ck["revived_dims"] else ""))
    for x in ck["revived_dims"]:
        L.append("  - %s" % x)
    for x in out["notes"]:
        L.append("- %s" % x)
    L += ["", "## Judgement calls (combine_weights.json)", ""] + ["- %s" % x for x in out["judgement_calls"]]
    Path(path).write_text("\n".join(L) + "\n")


def tag_of(view_path):
    stem = Path(view_path).stem
    return stem[len("view_scores_"):] if stem.startswith("view_scores_") else ""


def do(W, wfile, get_report, view_path, temporal_path, out_dir, tag, baseline_totals, extra=(), text_path=None,
       members=None):
    out = run(W, get_report, view_path, temporal_path, tag, extra, text_path, members)
    out["version"] = VERSION
    out["weights_file"] = rel(wfile)
    out["judgement_calls"] = W.get("judgement_calls", [])
    out["weights_sha256"] = hashlib.sha256(Path(wfile).read_bytes()).hexdigest()
    name = "combined" + ("_" + tag if tag else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / (name + ".json")).write_text(json.dumps(out, indent=1) + "\n")
    has_ctl = any(v["control"] for res in out["items"].values() for v in res.values())
    write_md(out, out_dir / (name + ".md"), baseline_totals if has_ctl else None)
    return out


def totals(out):
    return {item: {k: v["total"] for k, v in res.items() if not v["control"]} for item, res in out["items"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="append", required=True, help="launcher batch directory (repeatable)")
    ap.add_argument("--view", default=str(P / "view_scores.json"))
    ap.add_argument("--temporal", default=str(P / "temporal_scores.json"))
    ap.add_argument("--view-extra", nargs="*", default=[])
    ap.add_argument("--text", default=str(PT.TEXT_OUT / "results.json"))
    ap.add_argument("--weights", default=str(PT.VIEW_CONF / "combine_weights_v51r2.json"))
    ap.add_argument("--out", default=str(P / "combined"))
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--all-controls", action="store_true")
    ap.add_argument("--members", default=None, help="r3 weights: members_509 version dir (required with r3_rules)")
    a = ap.parse_args()
    W = json.loads(Path(a.weights).read_text())
    mem = None
    if W.get("r3_rules"):
        if not a.members:
            sys.exit("%s has r3_rules: give --members <members_509 version dir>" % a.weights)
        mem = load_members(a.members)[0]
    gr = batch_reports(a.batch)
    out_dir = Path(a.out)
    base = run(W, gr, a.baseline or a.view, a.temporal, "", a.view_extra, a.text, mem)
    bt = totals(base)
    tag = a.tag if a.tag is not None else tag_of(a.view)
    main_out = do(W, a.weights, gr, a.view, a.temporal, out_dir, tag, bt, a.view_extra, a.text, mem)
    for item, res in main_out["items"].items():
        for k, v in res.items():
            print("%s %-28s 5.0.x %5.1f  5.1 %5.1f  new %4.1f/%4.1f  D1 gate %s  N/A by rule %s" % (
                item, k, v["total_509"], v["total"], v["new_layer_points"], v["new_layer_max_points"],
                "-" if not v["d1_gate"] else v["d1_gate"].get("g"), [x["dim"] for x in v["gated_dims"]] or "-"))
    for x in main_out["notes"]:
        print("note:", x)
    if not a.all_controls:
        return
    files = sorted((P / "controls").glob("view_scores_*.json")) if (P / "controls").exists() else []
    if (P / "view_scores_controls.json").exists():
        files.append(P / "view_scores_controls.json")
    if not files:
        print("no control files found under %s/controls/ or view_scores_controls.json" % rel(P))
        return
    rows, skipped = [], []
    ver = json.loads(Path(a.view).read_text()).get("version")
    for f in files:
        fv = json.loads(Path(f).read_text()).get("version")
        if fv != ver:
            skipped.append("%s: version %s, main view file %s (stale control, not combined)" % (rel(f), fv, ver))
            print("skip control file", skipped[-1])
            continue
        o = do(W, a.weights, gr, f, a.temporal, out_dir, tag_of(f), bt, members=mem)
        for item, res in o["items"].items():
            for k, v in res.items():
                if "~" not in k and not v.get("control"):
                    continue  # a real-model copy inside a control file (view_scores_controls.json), not a control
                b = bt.get(item, {}).get(v["base_model"])
                rows.append({"file": rel(f), "item": item, "model": k, "base_model": v["base_model"],
                             "control": v["control"], "total_51": v["total"], "base_total_51": b,
                             "delta": None if b is None else round(v["total"] - b, 3),
                             "new_layer_points": v["new_layer_points"],
                             "drops": None if b is None else v["total"] < b - 1e-9})
    (out_dir / "combined_controls.json").write_text(json.dumps({"weights_sha256": main_out["weights_sha256"],
                                                                "skipped_files": skipped, "rows": rows}, indent=1) + "\n")
    L = ["# pvscore 5.1 r2: control totals", "",
         "Each control is combined with the 5.0.x report of its base model (5.0.x part not re-scored), so Δ is the "
         "change in the new layers only. Control files whose view version differs from the main view file are skipped "
         "(listed below); real-model copies inside a control file are not rows.", ""] + \
        ["skipped: " + x for x in skipped] + ([""] if skipped else []) + [
         "| file | item | model | real 5.1 | control 5.1 | Δ | drops |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %s | %s | %s | %.1f | %s | %s |" % (
            Path(r["file"]).name, r["item"], r["model"], fmt(r["base_total_51"]), r["total_51"],
            "–" if r["delta"] is None else "%+.2f" % r["delta"],
            "–" if r["drops"] is None else ("yes" if r["drops"] else "NO") +
            (" (fake, informational)" if is_baseline(r["base_model"]) else "")))
    (out_dir / "combined_controls.md").write_text("\n".join(L) + "\n")
    for r in rows:
        print("control %-28s %s %-34s %s -> %5.1f (%s)" % (Path(r["file"]).name, r["item"], r["model"],
                                                           fmt(r["base_total_51"]), r["total_51"],
                                                           "–" if r["delta"] is None else "%+.2f" % r["delta"]))


if __name__ == "__main__":
    sys.exit(main())
