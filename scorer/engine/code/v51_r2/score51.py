"""pvscore 5.1 (r2) scoring driver: one command for every item kind (script, continuation, spatial_arkit,
spatial_keyframe, video).

    python3 -I -B score51.py --batch DIR [--batch DIR ...] --out DIR
             [--workers 48] [--items M50_004,M50_033] [--models NAME,NAME] [--policy KEY=VALUE ...]
             [--text-profile claims_v5] [--conf score51_v51r3_v5.json]

Inputs: launcher batch directories only (batch.json, status.json, selfcheck/jobs.json and the selfcheck dumps). Rules:
score51_v51r2_v5.json (selection, datasets, text profiles, totals; = score51_v51r2.json + the claims_v5 profile),
combine_weights_v51r2.json (dimensions, gates), text_profile_claims_v5.json (script items under claims_v5), the
layer thresholds the layers load themselves. --text-profile NAME scores every script cell with that profile instead
of its dataset's (recorded in the output). --conf NAME: another driver configuration in paths.VIEW_CONF (default
paths.DRIVER_CONF; r3 = score51_v51r3_v5.json: weights combine_weights_v51r3.json, the 5.0.x members fixed by the GT
from paths.MEMBERS_ROOT/<members_509.dir>, set sha256 and pvscore version checked). Steps:
  1 delivery protocol (paths.PROTOCOL_DIR aggregate.py, overrides.json, approved policy P1-P9) over the batches: one
    cell per (dataset, model, item); the cell's attempt is chosen by the protocol, never by the launcher status
  2 delivered / recovered attempt -> its selfcheck directory (batches.attempts); no dump.json -> pending needs_acquire
  3 pvscore 5.0.11 report recomputed from the dump (pvscore5.score.score_item; the 5.0.11 reference column and the
    5.0.x blocks of 5.1)
  4 new layer: view items (continuation, video) model_view.run + view_score.score_model; text / image items
    model_side.run + the text layer (score.run_jobs, 200 permutations); script items through the dataset's text profile
    (layout_r2 = the text / image layer; claims_v5 = paths.V5_CODE claims_score.py on the dump, layer v5; a profile
    that is not available -> pending no_text_profile)
  5 combine.row_of (combine_one: the 5.1 total); model_fail = 0 for both 5.1 and 5.0.11
Writes only under --out: protocol.json, reports_5011/<model>/<item>/report.json, view/ (model views, view_scores.json),
text/ (model sides, results_models.json), v5/<item>/<model>.json (claims_v5), items/<item>/<model>.json, scores.json,
summary.json, summary.md.
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path


def _early_out():
    a = sys.argv[1:]
    return Path(a[a.index("--out") + 1]).resolve() if "--out" in a and a.index("--out") + 1 < len(a) else None


_OUT = _early_out()
if _OUT is not None and __name__ == "__main__":
    # the layers' write locations (paths.VIEW_OUT / TEXT_OUT) are read at import: point them into --out first
    os.environ["PV51_VIEW_OUT"] = str(_OUT / "view")
    os.environ["PV51_TEXT_OUT"] = str(_OUT / "text")
if __name__ == "__main__" and "--conf" in sys.argv[1:-1]:
    os.environ["PV51_CONF"] = sys.argv[sys.argv.index("--conf") + 1]

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as PT  # noqa: E402
sys.path.insert(0, str(PT.PVSCORE_DIR))
sys.path.insert(0, str(PT.TEXT_CODE))
import items as IT  # noqa: E402
import batches as BT  # noqa: E402
import combine as CB  # noqa: E402
import view_score as VS  # noqa: E402
import model_view as MV  # noqa: E402
import sframes as SF  # noqa: E402
from gt_root import gt_json  # noqa: E402
import model_side as MS  # noqa: E402
import score as TS  # noqa: E402
from pvscore5.score import score_item, VERSION as PV_VERSION  # noqa: E402

def conf_path(x):
    """--conf / paths.DRIVER_CONF: a file name in paths.VIEW_CONF, or a path (absolute or bench-relative)"""
    x = Path(x)
    return x if x.is_absolute() else (PT.ROOT / x if len(x.parts) > 1 else PT.VIEW_CONF / x)


CONF_FILE = conf_path(getattr(PT, "DRIVER_CONF", "score51_v51r2_v5.json"))
CONF = json.loads(CONF_FILE.read_text())
VERSION = CONF.get("version", "pvscore-5.1-r2")
W_FILE = PT.VIEW_CONF / CONF["weights"]
W = json.loads(W_FILE.read_text())
MEMBERS, MEMBERS_META = None, None                     # r3: {item: members_509 record}
if W.get("r3_rules"):
    _mc = CONF.get("members_509")
    if not _mc:
        raise RuntimeError("%s: weights %s have r3_rules but the configuration names no members_509" % (
            CONF_FILE.name, CONF["weights"]))
    _md = PT.MEMBERS_ROOT / _mc["dir"]
    MEMBERS, _idx = CB.load_members(_md)
    if _idx["set_sha256"] != _mc["set_sha256"]:
        raise RuntimeError("%s: set_sha256 %s, configuration %s" % (_md, _idx["set_sha256"], _mc["set_sha256"]))
    if _idx["pvscore"]["version"] != PV_VERSION:
        raise RuntimeError("%s built with %s, the driver runs %s" % (_md, _idx["pvscore"]["version"], PV_VERSION))
    MEMBERS_META = {"dir": PT.rel(_md), "set_sha256": _idx["set_sha256"], "pvscore_version": PV_VERSION,
                    "builder_sha256": _idx["builder_sha256"], "files": {i: v["sha256"] for i, v in _idx["items"].items()}}
PROF_FILES = {k: PT.VIEW_CONF / v["file"] for k, v in CONF["text_profiles"].items()
              if v.get("status", "available") == "available" and v.get("file")}
PROFILES = {k: json.loads(f.read_text()) for k, f in PROF_FILES.items()}
W_PROF = {k: CB.profile_weights(W, p) for k, p in PROFILES.items()}
OUT = _OUT
TP_OVERRIDE = None                                     # --text-profile


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_protocol():
    spec = importlib.util.spec_from_file_location("pv51_protocol_aggregate", PT.PROTOCOL_DIR / "aggregate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_CS = []


def claims_mod():
    """the v5 claims scorer (paths.V5_CODE claims_score.py), imported once per process with its rules, GT root and
    pvscore5 pointed at the copies this driver uses"""
    if not _CS:
        os.environ["PV_V5_PREREG"] = str(PT.V5_PREREG)
        os.environ["PV_GT_ROOT"] = str(PT.GT_ROOT)
        os.environ["PV_PVSCORE_DIR"] = str(PT.PVSCORE_DIR)
        sys.path.insert(0, str(PT.V5_CODE))
        import claims_score as CS
        if Path(CS.C.__file__).resolve().parent != PT.V5_CODE.resolve():
            raise RuntimeError("claims_core imported from %s, not %s" % (CS.C.__file__, PT.V5_CODE))
        if CS.PRE.get("version") != json.loads(PT.V5_PREREG.read_text()).get("version"):
            raise RuntimeError("claims_score rules %s, not those of %s" % (CS.PRE.get("version"), PT.V5_PREREG))
        CS.CLAIMS_DIR = PT.V5_CLAIMS                # explicit (its default is the r1 drafts_v5/claims); the driver
        #                                             passes each claims file by path, so this only names the set
        # class credit (r3 B3): the 5.1 classes.py, the same bytes as the copy next to the v5 code when one is
        # vendored there (claims_score.py imports <V5_ROOT>/tools/v51_text_r2/classes.py when called alone)
        cl = getattr(CS, "CL", None)
        vend = PT.V5_ROOT / "tools/v51_text_r2/classes.py"
        if cl is None or not CS.CREDIT_SRC.startswith("tools/v51_text_r2/classes.py") or \
                sha(cl.__file__) != sha(PT.TEXT_CODE / "classes.py") or (vend.is_file() and sha(vend) != sha(cl.__file__)):
            raise RuntimeError("claims_score class credit %s (%s) is not the 5.1 classes.py" % (
                CS.CREDIT_SRC, getattr(cl, "__file__", None)))
        _CS.append(CS)
    return _CS[0]


def claims_of(item):
    """-> (claims path, None) or (None, (action, why)) for the claims_v5 profile"""
    p = PT.v5_claims(item)
    if p is None:
        return None, ("no_claims", "no v5 claims file for %s in %s" % (item, PT.rel(PT.V5_CLAIMS)))
    cf = json.loads(p.read_text())
    if cf.get("gt_item") != item:
        return None, ("no_claims", "%s is for %s, not %s" % (PT.rel(p), cf.get("gt_item"), item))
    if cf.get("prereg_sha256") != sha(PT.V5_PREREG):
        return None, ("claims_prereg_mismatch", "%s prereg_sha256 %s != sha256 of %s" % (
            PT.rel(p), cf.get("prereg_sha256"), PT.rel(PT.V5_PREREG)))
    return p, None


def wjson(p, obj, indent=1):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=indent) + "\n")


# ------------------------------------------------------------------------------------------------ selection
def select(proto, bdirs, only_items, only_models):
    """protocol cells -> rows {dataset, model, item, kind, status, ...}; status scored_candidate / model_fail / pending"""
    by_run = {str(Path(a["run"]).resolve()): a for a in BT.attempts(bdirs)}
    tab = IT.table()
    rows = []
    for c in sorted(proto["cells"], key=lambda c: (str(c["dataset"]), c["model"], c["item"])):
        if (only_items and c["item"] not in only_items) or (only_models and c["model"] not in only_models):
            continue
        r = {"dataset": c["dataset"], "model": c["model"], "item": c["item"], "baseline": c["baseline"],
             "kind": (tab.get(c["item"]) or {}).get("kind"),
             "protocol": {"final": c["final"], "pending": c.get("pending")}}
        f, pend = c["final"], c.get("pending")
        ds = CONF["datasets"].get(c["dataset"])
        tdir = PT.DATASET_TASKS.get(c["dataset"])           # the task files of the cell's dataset
        task = (tdir or PT.TASKS) / c["item"] / "task.json"
        run, recovered = None, False
        if ds is None or tdir is None:
            r.update(status="pending", action="unknown_dataset", why="dataset %s not in %s" % (
                c["dataset"], CONF_FILE.name if ds is None else "paths.DATASET_TASKS"))
        elif c["item"] not in tab:
            r.update(status="pending", action="unknown_item", why="item not in %s" % PT.rel(PT.ITEMS_FILE))
        elif not task.is_file() or sha(task) != c.get("task_sha256"):
            r.update(status="pending", action="task_mismatch",
                     why="task.json sha256 differs from the batch case (or task.json missing)")
        elif f and f["status"] == "model_fail" and not pend:
            r.update(status="model_fail", why=f.get("reason"), total=0.0, total_5011=0.0)
        elif f and f["status"] == "delivered":
            run = f["out"]
        elif pend and pend.get("action") == "score_recovered":
            run, recovered = pend.get("out"), True
        else:
            r.update(status="pending", action=(pend or {}).get("action") or "unresolved",
                     why=(pend or {}).get("why"))
        if run is not None:
            att = by_run.get(str(Path(run).resolve()))
            r["run"] = PT.rel(run)
            r["recovered"] = recovered
            if att is None or not att["dump"]:
                r.update(status="pending", action="needs_acquire",
                         why="no dump.json for this attempt in the batches' selfcheck directories (acquire = "
                             "pvscore5 dump of the attempt's scene.blend in Blender)")
            else:
                prof = TP_OVERRIDE or ds.get("text_profile")
                tk = r["kind"] in CONF["text_profile_kinds"]
                cl, bad = (claims_of(c["item"]) if tk and prof in PROFILES else (None, None))
                if tk and prof != "layout_r2" and prof not in PROFILES:
                    r.update(status="pending", action="no_text_profile",
                             why="text profile %s: not available in %s" % (prof, CONF_FILE.name))
                elif bad:
                    r.update(status="pending", action=bad[0], why=bad[1])
                else:
                    r.update(status="scored_candidate", selfcheck=att["out"], batch_report=att["report"],
                             text_profile=prof if tk else None)
                    vt = json.loads((Path(att["out"]) / "dump.json").read_text())["meta"].get("def_variable_topology")
                    if vt:                                # r4 R4-5 (PREREG_51r4_addendum3.json)
                        r["r4_5"] = {"variable_topology": vt, "note": "variable topology handled (R4-5)"}
                    if cl is not None:
                        r["claims"] = str(cl)
        rows.append(r)
    return rows


def select_dumps(root, only_items, only_models):
    """--dumps: <root>/<model>/<item>/dump.json -> rows as select() builds them (no launcher batches, no delivery protocol)"""
    tab = IT.table()
    rows = []
    for dj in sorted(Path(root).glob("*/*/dump.json")):
        model, item = dj.parent.parent.name, dj.parent.name
        if (only_items and item not in only_items) or (only_models and model not in only_models):
            continue
        # the dataset whose own task directory lists the item (paths.TASKS itself holds no task)
        dataset = next((d for d in CONF["datasets"] if PT.DATASET_TASKS.get(d, PT.TASKS) != PT.TASKS and
                        (PT.DATASET_TASKS[d] / item).is_dir()), None)
        r = {"dataset": dataset, "model": model, "item": item, "baseline": False, "kind": (tab.get(item) or {}).get("kind"),
             "protocol": {"final": None, "pending": None}, "run": str(dj.parent), "recovered": False}
        if dataset is None or item not in tab:
            r.update(status="pending", action="unknown_item", why="item not in %s" % PT.rel(PT.ITEMS_FILE))
            rows.append(r)
            continue
        ds = CONF["datasets"][dataset]
        prof = TP_OVERRIDE or ds.get("text_profile")
        tk = r["kind"] in CONF["text_profile_kinds"]
        cl, bad = (claims_of(item) if tk and prof in PROFILES else (None, None))
        if bad:
            r.update(status="pending", action=bad[0], why=bad[1])
            rows.append(r)
            continue
        r.update(status="scored_candidate", selfcheck=str(dj.parent), batch_report=None, text_profile=prof if tk else None)
        vt = json.loads(dj.read_text())["meta"].get("def_variable_topology")
        if vt:
            r["r4_5"] = {"variable_topology": vt, "note": "variable topology handled (R4-5)"}
        if cl is not None:
            r["claims"] = str(cl)
        rows.append(r)
    return rows


# ------------------------------------------------------------------------------------------------ workers
def job_5011(args):
    key, item, sc = args
    t0 = time.time()
    try:
        rep = score_item(str(gt_json(item)), sc)
        p = OUT / "reports_5011" / key[0] / item / "report.json"
        wjson(p, rep)
        return key, {"report": str(p), "total": rep["total"], "schema": rep.get("schema")}, None, time.time() - t0
    except Exception:  # noqa: BLE001 - reported per cell, never scored as 0
        return key, None, traceback.format_exc(), time.time() - t0


def text_facts(item):
    """number of text / image GT facts of an item (0 without gt_facts.json)"""
    f = PT.TEXT_GT / item / "gt_facts.json"
    if not f.is_file():
        return 0
    c = json.loads(f.read_text())["counts"]
    return sum(v for k, v in c.items() if k not in ("frames", "elements3d", "entity_times", "occlusion_inventory"))


VIEW_CTX = {}


def view_ctx(item):
    if item not in VIEW_CTX:
        ck = json.loads((PT.VIEW_GT / item / "checklist.json").read_text())
        entries = [e for e in ck["entries"] if e["scorable_new"] not in VS.SKIP]
        g = SF.load(PT.VIEW_GT / item / "scene_frames.json")
        geo = {sh["shot"]: [s for s in sh["samples"] if s.get("fi") is not None] for sh in g["shots"]}
        VIEW_CTX[item] = (entries, geo)
    return VIEW_CTX[item]


def v5_score(model, item, sc, claims):
    """claims_v5: claims_score.py on the dump (its main() --claims --dump --out, with the driver's GT) -> the result
    written to OUT/v5/<item>/<model>.json"""
    CS = claims_mod()
    cf = json.loads(Path(claims).read_text())
    gp = gt_json(item)
    d = CS.from_dump(cf, Path(sc), gp)
    info = d.pop("_info")
    res = CS.score_scene(cf, d)
    res["dump"] = PT.rel(sc)
    res["model_scene"] = info
    res["a1_counts_drop"] = CS.a1_counts_drop(cf, json.loads(gp.read_text()))
    res["claims_file"], res["claims_sha256"] = PT.rel(claims), sha(claims)
    res["prereg_file"], res["prereg_sha256"] = PT.rel(PT.V5_PREREG), sha(PT.V5_PREREG)
    res["class_credit_file"] = PT.rel(CS.CL.__file__)
    p = OUT / "v5" / item / (model + ".json")
    wjson(p, res)
    return {"file": str(p), "blocks": res["blocks"], "a1_counts_drop": res["a1_counts_drop"],
            "unbound": len(res.get("unbound") or []), "claims_sha256": res["claims_sha256"]}


def job_layer(args):
    layer, key, item, sc = args[:4]
    t0 = time.time()
    try:
        if layer == "v5":
            return layer, key, v5_score(key[0], item, sc, args[4]), None, time.time() - t0
        if layer == "view":
            MV.run(item, key[0], sc)
            d = json.loads((PT.VIEW_OUT / item / "model_view" / f"{key[0]}.json").read_text())
            entries, geo = view_ctx(item)
            blocks, detail = VS.score_model(item, VS.View(d, item=item), entries, geo)
            return layer, key, {"blocks": blocks, "entries": detail}, None, time.time() - t0
        MS.run(item, key[0], sc)
        return layer, key, True, None, time.time() - t0
    except Exception:  # noqa: BLE001
        return layer, key, None, traceback.format_exc(), time.time() - t0


# ------------------------------------------------------------------------------------------------ totals
def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(rows):
    sc = [r for r in rows if r["status"] == "scored"]
    mf = [r for r in rows if r["status"] == "model_fail"]
    res = sc + mf
    return {"planned": len(rows), "scored": len(sc), "model_fail": len(mf), "resolved": len(res),
            "delivery_rate": round(len(sc) / len(res), 4) if res else None,
            "quality_on_delivered": mean(r["total"] for r in sc), "overall": mean(r["total"] for r in res),
            "quality_on_delivered_5011": mean(r["total_5011"] for r in sc),
            "overall_5011": mean(r["total_5011"] for r in res),
            "model_fail_items": sorted("%s(%s)" % (r["item"], r.get("why")) for r in mf),
            "pending": sorted("%s:%s" % (r["item"], r["action"]) for r in rows if r["status"] == "pending")}


def summary(rows, proto):
    out = {}
    for ds in sorted({r["dataset"] for r in rows}, key=str):
        dr = [r for r in rows if r["dataset"] == ds]
        models = sorted({r["model"] for r in dr})
        compared = [m for m in models if not any(r["baseline"] for r in dr if r["model"] == m)]
        exhausted = sorted({r["item"] for r in dr if r["status"] == "pending" and r["action"] == "infra_exhausted"})
        dr = [r for r in dr if r["item"] not in exhausted]           # as the protocol: not counted for any model
        items = sorted({r["item"] for r in dr})
        common = [i for i in items if all(any(r["model"] == m and r["item"] == i and r["status"] in ("scored", "model_fail")
                                              for r in dr) for m in compared)]
        per = {}
        for m in models:
            mr = [r for r in dr if r["model"] == m]
            per[m] = dict(summarize(mr), baseline=m not in compared,
                          common=summarize([r for r in mr if r["item"] in common]))
        pd = (proto["datasets"].get(ds) or {})
        out[ds] = {"models": per, "items": items, "common_items": common, "excluded_infra_exhausted": exhausted,
                   "provisional": any(per[m]["pending"] for m in per) or not pd.get("final", False),
                   "protocol_final": pd.get("final"), "protocol_pins": pd.get("pins")}
    return out


def gates(rows):
    """combine.checks on the scored rows (same row layout as combine.run): per-item gates and rankings, plus the
    gates on the dataset means of the fixture model"""
    fake = {}
    out = {"items": {}}
    for r in rows:
        if r["status"] == "scored":
            out["items"].setdefault(r["item"], {})[r["model"]] = r["row"]
    ck = CB.checks(W, out)
    G = W["gates"]
    for r in rows:
        if r["status"] == "scored" and CB.is_baseline(r["model"]):
            fake.setdefault(r["dataset"], []).append(r)
    means = {}
    for ds, fr in fake.items():
        m51, m11 = mean(r["total"] for r in fr), mean(r["total_5011"] for r in fr)
        means[ds] = {"baseline_items": len(fr), "mean_51": m51, "mean_5011": m11,
                     "fake_max_total": {"limit": G["fake_max_total"], "pass": m51 <= G["fake_max_total"]},
                     "fake_not_above_509": {"pass": m51 <= m11 + 1e-9}}
    ck["dataset_means"] = means
    return ck


def write_md(path, summ, ck, rows):
    L = ["# %s scores" % VERSION, "", "Generated by score51.py (%s). Totals per dataset and model under the delivery "
         "protocol; 5.0.11 = pvscore 5.0.11 on the same dumps (reference column). Model failures count 0; pending "
         "items are left out and mark the numbers provisional." % VERSION, ""]
    for ds, d in summ.items():
        L += ["## %s%s" % (ds, " (provisional)" if d["provisional"] else ""), "",
              "| model | scored | model_fail | pending | quality on delivered 5.1 (5.0.11) | overall 5.1 (5.0.11) | "
              "common-items overall 5.1 (5.0.11) |", "|---|---|---|---|---|---|---|"]
        for m, v in d["models"].items():
            c = v["common"]
            L.append("| %s%s | %d | %d | %s | %s (%s) | %s (%s) | %s (%s) |" % (
                m, " (baseline)" if v["baseline"] else "", v["scored"], v["model_fail"],
                ", ".join(v["pending"]) or "–", v["quality_on_delivered"], v["quality_on_delivered_5011"],
                v["overall"], v["overall_5011"], c["overall"], c["overall_5011"]))
        L.append("")
        L.append("Common items (%d): %s" % (len(d["common_items"]), ", ".join(d["common_items"])))
        L.append("")
    L += ["## Gates on the baseline means", ""]
    for ds, g in ck["dataset_means"].items():
        L.append("- %s: baseline mean 5.1 %.2f (limit %.0f: %s), 5.0.11 %.2f (5.1 <= 5.0.11: %s)" % (
            ds, g["mean_51"], g["fake_max_total"]["limit"], "pass" if g["fake_max_total"]["pass"] else "FAIL",
            g["mean_5011"], "pass" if g["fake_not_above_509"]["pass"] else "FAIL"))
    L += ["", "## Per item", "", "| item | kind | model | status | 5.1 | 5.0.11 | layer |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            r["item"], r["kind"], r["model"], r["status"] + (":" + r["action"] if r["status"] == "pending" else ""),
            r.get("total", "–"), r.get("total_5011", "–"), r.get("layer") or "–"))
    Path(path).write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------ scoring
def score_cells(cand, workers, t0=None):
    """steps 3-5 for the candidate rows (dicts with model, item, kind, selfcheck, batch_report): writes the 5.0.11
    reports, the view / text layer files under OUT and fills each row (status scored, total, total_5011, row, layer) or
    marks it pending scorer_error. Shared with the package selftest. -> (errors, seconds)"""
    t0 = time.time() if t0 is None else t0
    key = {id(r): (r["model"], r["item"]) for r in cand}
    # 3 + 4 (view layer, text model sides) in one pool
    jobs = [("5011", key[id(r)], r["item"], r["selfcheck"]) for r in cand]
    lay = []
    for r in cand:
        if r.get("text_profile") in PROFILES:
            cl = r.get("claims")
            if cl is None:
                cl, bad = claims_of(r["item"])
                if bad:
                    raise ValueError("%s %s: %s" % (r["model"], r["item"], bad[1]))
                r["claims"] = str(cl)
            lay.append(("v5", key[id(r)], r["item"], r["selfcheck"], r["claims"]))
            r["layer"] = "v5"
        elif r["kind"] in IT.VIEW_KINDS:
            lay.append(("view", key[id(r)], r["item"], r["selfcheck"]))
            r["layer"] = "view"
        elif text_facts(r["item"]):
            lay.append(("text", key[id(r)], r["item"], r["selfcheck"]))
            r["layer"] = "text"
        else:
            r["layer"] = None
            r["layer_why"] = "no new-layer GT for this item (%s): 5.0.x blocks only" % (
                ("no gt_facts.json or 0 facts (no GT boxes)") if r["kind"] in IT.TEXT_KINDS else "no checklist")
    rep5, views, v5res, errors, secs = {}, {}, {}, {}, {}
    with Pool(workers, initializer=_init_fixes) as pool:     # r4 H-1 (PREREG_51r4_addendum2.json)
        it = pool.imap_unordered(_dispatch, [("5011",) + j[1:] for j in jobs] + lay, chunksize=1)
        for n, (kind, k, res, err, dt) in enumerate(it):
            secs[(kind,) + k] = round(dt, 1)
            if err:
                errors.setdefault(k, []).append({"step": kind, "error": err})
            elif kind == "5011":
                rep5[k] = res
            elif kind == "view":
                views[k] = res
            elif kind == "v5":
                v5res[k] = res
            if n % 20 == 0:
                print("%d / %d jobs, %.0fs" % (n, len(jobs) + len(lay), time.time() - t0), flush=True)
    # 4 text layer (its own pool: 200 permutations per pair)
    tpairs = [(r["item"], "model:" + r["model"]) for r in cand if r.get("layer") == "text" and key[id(r)] not in errors]
    tres = TS.run_jobs(tpairs, workers) if tpairs else {}
    wjson(OUT / "text" / "results_models.json",
          {"schema": "pvscore-5.1-text-image", **TS.provenance(), "deps": TS.deps_record(),
           "scores": {i: {w: {k: v for k, v in r.items() if k != "corr"} for w, r in d.items()} for i, d in tres.items()}})
    vs = {"thresholds": VS.TH_FILE, "version": "r2", "block_fields": VS.FIELD_DOC, "items": {}, "parked": {}}
    for (m, item), v in sorted(views.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        vs["items"].setdefault(item, {})[m] = v
        vs["parked"][item] = VS.item_ctx(item).parked_summary()
    wjson(OUT / "view" / "view_scores.json", vs)

    # 5 combine
    for r in cand:
        k = key[id(r)]
        if k in errors or k not in rep5:
            r.update(status="pending", action="scorer_error", errors=errors.get(k))
            continue
        rp = rep5[k]
        rep = json.loads(Path(rp["report"]).read_text())
        vm, f3, v5, Wr = None, None, None, W                      # temporal layer: not in the total (tb = {})
        if r["layer"] == "v5":
            v5, Wr = v5res.get(k), W_PROF[r["text_profile"]]
            if v5 is None:
                r.update(status="pending", action="scorer_error", errors=[{"step": "v5", "error": "no v5 result"}])
                continue
            r["v5"] = {b: {f: v5["blocks"][b].get(f) for f in ("n", "n_true", "n_missing", "P", "R", "coverage", "score")}
                       for b in v5["blocks"]}
            r["v5_file"], r["v5_unbound"] = PT.rel(v5["file"]), v5["unbound"]
            r["claims_sha256"] = v5["claims_sha256"]
            r["claims"] = PT.rel(r["claims"])
        elif r["layer"] == "view":
            vm = views.get(k)
        elif r["layer"] == "text":
            t = (tres.get(r["item"]) or {}).get("model:" + r["model"])
            if t is None:
                r.update(status="pending", action="scorer_error", errors=[{"step": "text", "error": "no text result"}])
                continue
            vm, f3 = {"blocks": t.get("blocks_view") or {}}, t.get("blocks3d") or {}
        E = None
        if MEMBERS is not None:                                    # r3: the item's 5.0.x members (GT)
            E = MEMBERS.get(r["item"])
            if E is None or E["gt_sha256"] != sha(gt_json(r["item"])):
                r.update(status="pending", action="scorer_error", errors=[{"step": "members", "error": (
                    "no members_509 file for %s" % r["item"]) if E is None else "members_509 gt_sha256 differs from %s"
                    % PT.rel(gt_json(r["item"]))}])
                continue
            r["members_sha256"] = MEMBERS_META["files"][r["item"]]
        row = CB.row_of(Wr, rep, PT.rel(rp["report"]), vm, {}, f3, r["model"], None, v5, E)
        br = json.loads(Path(r["batch_report"]).read_text()) if r.get("batch_report") else None
        r.update(status="scored", total=row["total"], total_5011=rep["total"], row=row,
                 batch_report_total=None if br is None else br.get("total"),
                 batch_report_schema=None if br is None else br.get("schema"))
        if br is not None and abs(br.get("total", -1) - rep["total"]) > 1e-6:
            r["flag"] = "5.0.11 recomputed %.3f differs from the batch report.json %s (%s)" % (
                rep["total"], br.get("total"), br.get("schema"))
    return errors, secs


# ------------------------------------------------------------------------------------------------ main
def main():
    a = sys.argv[1:]
    bdirs = [str(Path(a[i + 1]).resolve()) for i, x in enumerate(a) if x == "--batch"]
    dumps = Path(a[a.index("--dumps") + 1]).resolve() if "--dumps" in a else None
    if (not bdirs and dumps is None) or OUT is None:
        sys.exit(__doc__)
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 48
    only_items = set(a[a.index("--items") + 1].split(",")) if "--items" in a else None
    only_models = set(a[a.index("--models") + 1].split(",")) if "--models" in a else None
    global TP_OVERRIDE
    if "--conf" in a and conf_path(a[a.index("--conf") + 1]).resolve() != CONF_FILE.resolve():
        sys.exit("--conf %s: this paths.py reads %s (a package has one fixed configuration)" % (
            a[a.index("--conf") + 1], CONF_FILE.name))
    if "--text-profile" in a:
        TP_OVERRIDE = a[a.index("--text-profile") + 1]
        if TP_OVERRIDE != "layout_r2" and TP_OVERRIDE not in PROFILES:
            sys.exit("--text-profile %s: not an available profile of %s" % (TP_OVERRIDE, CONF_FILE.name))
    cli_pol = {}
    for i, x in enumerate(a):
        if x == "--policy":
            k, _, v = a[i + 1].partition("=")
            cli_pol[k] = int(v) if v.isdigit() else v
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    AG = load_protocol()
    policy = dict(CONF["protocol"]["policy"], **cli_pol)
    if dumps is None:
        proto = AG.aggregate(bdirs, AG.classify.load_overrides(PT.PROTOCOL_DIR / "overrides.json"), policy)
        wjson(OUT / "protocol.json", proto)
        rows = select(proto, bdirs, only_items, only_models)
    else:
        proto = {"cells": [], "datasets": {}, "policy": policy, "protocol": None, "flags": []}
        rows = select_dumps(dumps, only_items, only_models)
    cand = [r for r in rows if r["status"] == "scored_candidate"]
    print("cells %d, to score %d, model_fail %d, pending %d" % (
        len(rows), len(cand), sum(r["status"] == "model_fail" for r in rows),
        sum(r["status"] == "pending" for r in rows)), flush=True)

    errors, secs = score_cells(cand, workers, t0)
    for r in rows:
        wjson(OUT / "items" / r["item"] / (r["model"] + ".json"), dict(r, version=VERSION))
    summ = summary(rows, proto)
    ck = gates(rows)
    flat = [{k: v for k, v in r.items() if k != "row"} | ({"dims": {d: (x or {}).get("score") for d, x in
                                                                     r["row"]["dims"].items()}} if r.get("row") else {})
            for r in rows]
    meta = {"version": VERSION, "config": PT.rel(CONF_FILE), "config_sha256": sha(CONF_FILE),
            "weights": PT.rel(W_FILE), "weights_sha256": sha(W_FILE), "batches": [PT.rel(b) for b in bdirs],
            "policy": proto["policy"], "protocol": proto["protocol"], "gt_root": PT.rel(PT.GT_ROOT),
            "items_file": PT.rel(PT.ITEMS_FILE), "text_profile_override": TP_OVERRIDE,
            "pvscore": {"version": PV_VERSION, "dir": PT.rel(PT.PVSCORE_DIR)}, "protocol_dir": PT.rel(PT.PROTOCOL_DIR),
            "text_profiles": {k: {"file": PT.rel(f), "sha256": sha(f)} for k, f in PROF_FILES.items()},
            "v5_sources": v5_sources(rows), "wall_s": round(time.time() - t0, 1)}
    if MEMBERS_META is not None:
        meta["members_509"] = {k: v for k, v in MEMBERS_META.items() if k != "files"}
    wjson(OUT / "scores.json", dict(meta, cells=flat, seconds={"|".join(k): v for k, v in sorted(secs.items())}))
    wjson(OUT / "summary.json", dict(meta, datasets=summ, gates=ck, protocol_flags=proto["flags"]))
    write_md(OUT / "summary.md", summ, ck, rows)
    bad = [r for r in rows if r["status"] == "pending" and r["action"] == "scorer_error"]
    print("wrote", OUT, "%.0fs" % (time.time() - t0), "scorer errors %d" % len(bad), flush=True)
    for ds, d in summ.items():
        for m, v in d["models"].items():
            print(ds, m, "scored", v["scored"], "fail", v["model_fail"], "pending", v["pending"],
                  "q51", v["quality_on_delivered"], "q5011", v["quality_on_delivered_5011"], flush=True)
    return 1 if bad else 0


def v5_sources(rows):
    """sha256 of the v5 code, rules and claims files the run used (claims_v5 cells only)"""
    used = sorted({r["claims"] for r in rows if r.get("claims")})
    if not used:
        return None
    used = [Path(p) if Path(p).is_absolute() else PT.ROOT / p for p in used]
    return {"v5_root": PT.rel(PT.V5_ROOT), "claims_dir": PT.rel(PT.V5_CLAIMS),
            "code": {f: sha(PT.V5_CODE / f) for f in ("claims_core.py", "claims_score.py")},
            "prereg": {"file": PT.rel(PT.V5_PREREG), "sha256": sha(PT.V5_PREREG)},
            "claims": {PT.rel(p): sha(p) for p in used}}


def _init_fixes():
    """r4 H-1 (scorer PREREG_51r4_addendum2.json): every pool worker starts with the vendored pvscore 5.0.12 fix
    switches on (the thresholds file score_item loads), so no layer depends on the job order or the worker count"""
    from pvscore5 import fixes
    from pvscore5.score import load_thresholds
    fixes.set_active(fixes.ids_from(load_thresholds()))


def _dispatch(j):
    if j[0] == "5011":
        k, res, err, dt = job_5011(j[1:])
        return "5011", k, res, err, dt
    return job_layer(j)


if __name__ == "__main__":
    sys.exit(main())
