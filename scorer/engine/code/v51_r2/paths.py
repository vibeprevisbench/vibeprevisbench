"""pvscore 5.4 (Rule-based Scene Check): every location the scoring code reads or writes.

The engine scores against the dataset folder named by the environment variable VPB_DATA (default: <repo>/data, the
Hugging Face dataset). Everything task-specific is read from <data>/rubrics/<task_id>/: gt.json (hidden ground truth
facts), checklist.json and scene_frames.json (media tasks), gt_facts.json (script and photo tasks), claims.json (script
tasks) and task.json (the brief the scorer is pinned to). The rules, thresholds and the per-task 5.0.x members live in
this package. The driver points VIEW_OUT / TEXT_OUT at its --out.
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                   # the engine root (scorer/engine)


def _env(name, default):
    v = os.environ.get(name)
    return Path(v) if v else default


DATA = _env("VPB_DATA", ROOT.parents[1] / "data").resolve()
RUBRICS = DATA / "rubrics"

PVSCORE_DIR = ROOT / "vendor/v5_0_12"                    # holds pvscore5/ (pvscore 5.0.12)
VIEW_CODE = HERE
TEXT_CODE = ROOT / "code/v51_text_r2"
VIEW_CONF = ROOT / "config/view"
TEXT_CONF = ROOT / "config/text"
GT_ROOT = RUBRICS                                        # <task_id>/gt.json (+ refs/)
VIEW_GT = RUBRICS                                        # <task_id>/checklist.json, scene_frames.json
TEXT_GT = RUBRICS                                        # <task_id>/gt_facts.json
TASKS = ROOT / "gt"                                      # holds no task directories: every task is found through
DATASET_TASKS = {"vibeprevisbench": RUBRICS}             # its dataset, whose <task_id>/task.json is the pinned brief
ITEMS_FILE = ROOT / "gt/items.jsonl"                     # task id -> kind
FREEZE = ROOT / "gt/video_freeze_20261017.json"          # held still frames inside reference videos
PROTOCOL_DIR = ROOT / "vendor/protocol"
V5_ROOT = ROOT / "vendor/text_v5"
V5_CODE = V5_ROOT / "tools" / "text_v5"                  # claims_score.py, claims_core.py
V5_PREREG = V5_ROOT / "text_items/drafts_v5/_build/PREREG_v5.0.3.json"   # the frozen rules of the script-task claims
V5_CLAIMS = RUBRICS                                      # <task_id>/claims.json, each naming its task in gt_item
MEMBERS_ROOT = ROOT / "gt/members_509"                   # <version dir>/<task_id>.json + index.json (members_509.py)
DRIVER_CONF = "score51_v54.json"                         # the one driver configuration of this package
VIEW_OUT = _env("PV51_VIEW_OUT", ROOT / "_work/view")
TEXT_OUT = _env("PV51_TEXT_OUT", ROOT / "_work/text")


def rel(p):
    """path for provenance strings: relative to the engine root when inside it"""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


_V5_INDEX = []                                           # the claims index, built once by v5_claims_index


def v5_claims_index():
    """{task id: claims file} over every <task_id>/claims.json in V5_CLAIMS, keyed by the file's own gt_item. A file
    that is not a JSON object with a non-empty gt_item string, or two files naming the same task, raise ValueError:
    scoring stops rather than pick one."""
    if not _V5_INDEX:
        idx, bad = {}, []
        for p in sorted(V5_CLAIMS.glob("*/claims.json")):
            name = "%s/%s" % (p.parent.name, p.name)
            try:
                g = json.loads(p.read_text()).get("gt_item")
            except (ValueError, AttributeError) as e:
                bad.append("%s: not a claims file (%s)" % (name, e))
                continue
            if not isinstance(g, str) or not g:
                bad.append("%s: no gt_item" % name)
            elif g in idx:
                bad.append("%s/%s and %s both name %s" % (idx[g].parent.name, idx[g].name, name, g))
            else:
                idx[g] = p
        if bad:
            raise ValueError("claims index of %s: %s" % (V5_CLAIMS, "; ".join(bad)))
        _V5_INDEX.append(idx)
    return _V5_INDEX[0]


def v5_claims(item):
    """the claims file whose gt_item is exactly this task id (v5_claims_index), None when there is none (the driver
    then leaves the cell pending no_claims; it also checks the file's gt_item and prereg)"""
    return v5_claims_index().get(item)
