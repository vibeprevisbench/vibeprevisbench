"""pvscore 5.1 (frozen, r3): score launcher batches.

    python3 -I -B <package>/score51.py --batch DIR [--batch DIR ...] --out DIR [--workers 48]
                                         [--items M50_004,...] [--models NAME,...] [--policy KEY=VALUE ...]
                                         [--text-profile claims_v5]
    python3 -I -B <package>/score51.py --dumps DIR --out DIR     # DIR/<model>/<item>/dump.json, no launcher batches

Runs code/v51_r2/score51.py (see its docstring) with the package's own paths: GT from the dataset's rubrics/, pvscore 5.0.12 from
vendor/v5_0_12, the delivery protocol from vendor/protocol, the v5 claims scoring from vendor/text_v5 (text profile
claims_v5), all rules from config/. Outputs (only under --out):
summary.md / summary.json (per dataset and model: delivered, model_fail, pending, quality on delivered, overall,
common-items means, for 5.1 and the 5.0.12 reference column; gates), scores.json (every cell), items/<item>/<model>.json,
protocol.json, reports_5011/, view/, text/. Needs Python 3.12 with numpy.
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "code" / "v51_r2" / "score51.py"
sys.argv[0] = str(DRIVER)
runpy.run_path(str(DRIVER), run_name="__main__")
