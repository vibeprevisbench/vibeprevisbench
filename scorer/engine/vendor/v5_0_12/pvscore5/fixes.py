"""5.0.12 fix switches (CHANGES_5012.md). The thresholds file lists the fixes that are on (`fixes_5_0_12`);
score.score_item sets them for every item before the GT is read. An empty list reproduces 5.0.11 exactly, so each
fix can be switched off on its own for an ablation."""
from __future__ import annotations

# S: scorer bugs (RESULTS_5011.md S rows + VIEW_DIAG.md S24 / S25; S23 lives in tools/protocol/classify.py)
# D: design choices that score a correct answer wrong (RESULTS_5011.md D rows)
ALL = ("S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20", "S21", "S22",
       "S24", "S25",
       "D2", "D3", "D4", "D5", "D12", "D12b", "D15", "D17", "D18", "D19", "D21", "D22")
# 5.2 (scorer/v5_2 config/view/PREREG_52.json): measurement fixes from the 10-20 blind rule review (data/rule_review_v1
# adjudication.json). Listed in the thresholds file under fixes_5_2; an empty fixes_5_2 reproduces pvscore 5.1 r4.1.
ALL_5_2 = ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14", "M15", "M16", "M17")
ALL = ALL + ALL_5_2
# 5.3 (scorer/v5_3 config/view/PREREG_53.json): M18, the airborne test's box-span support (Events.airborne). Listed
# in the thresholds file under fixes_5_3; an empty fixes_5_3 reproduces pvscore 5.2.1.
ALL_5_3 = ("M18",)
ALL = ALL + ALL_5_3
# 5.4 (config/view/PREREG_54.json): M19, every fall of a body is a candidate (Events.find). Listed in the
# thresholds file under fixes_5_4; an empty fixes_5_4 reproduces pvscore 5.3.
ALL_5_4 = ("M19",)
ALL = ALL + ALL_5_4
ACTIVE = frozenset()


def set_active(ids):
    global ACTIVE
    bad = sorted(set(ids) - set(ALL))
    if bad:
        raise ValueError("unknown 5.0.12 fix ids: %s" % bad)
    ACTIVE = frozenset(ids)


def on(fid):
    return fid in ACTIVE


def ids_from(th):
    """5.2: the switches a thresholds dict turns on (fixes_5_0_12, then fixes_5_2; 5.3: then fixes_5_3; 5.4: then fixes_5_4)"""
    return list(th.get("fixes_5_0_12", [])) + list(th.get("fixes_5_2", [])) + list(th.get("fixes_5_3", [])) + \
        list(th.get("fixes_5_4", []))
