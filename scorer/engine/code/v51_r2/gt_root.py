"""GT root read by the v5.1 tools: paths.GT_ROOT, the dataset's rubrics/<task_id>/gt.json."""
import paths as PT

R = PT.ROOT
GT_ROOT = PT.GT_ROOT


def gt_json(item):
    return PT.GT_ROOT / item / "gt.json"
