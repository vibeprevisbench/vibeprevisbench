"""pvscore 5.1 text / image layer r1: class-label normalisation and class credit (tools/v51_text_r2/class_map_v51r1.json).

One function for GT classes, model classes, the similar groups and the exclusion lists; no per-model or per-item rule.
    norm("Other: Traffic-Cone") -> "traffic_cone";  norm("other:lockers") -> "locker" (vocabulary-backed plural)
    gt_labels("s01_light_mast", "streetlight") -> ("streetlight", "light_mast")
    credit(("streetlight", "light_mast"), "other:light_mast") -> 1.0
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v51_r2"))
import paths as PT  # noqa: E402
R = PT.ROOT
CMAP = json.loads((PT.TEXT_CONF / "class_map_v51r1.json").read_text())
VOCAB = frozenset(c["class"] for c in json.loads((R / CMAP["vocabulary"]).read_text())["classes"])
KEY_RE = re.compile(CMAP["gt_labels"]["key_regex"])
_SEP = re.compile(r"[\s\-./]+")
_US = re.compile(r"_+")
EXACT, SIMILAR = float(CMAP["credit"]["exact"]), float(CMAP["credit"]["similar"])


def norm(c):
    """N1 spelling, N2 prefix, N3 vocabulary-backed plural; None / empty -> None"""
    if c is None:
        return None
    w = str(c).strip().lower().split(":")[-1]
    w = _US.sub("_", _SEP.sub("_", w.strip())).strip("_")
    if not w:
        return None
    if w not in VOCAB:
        for suf, rep in (("ies", "y"), ("es", ""), ("s", "")):
            if w.endswith(suf) and len(w) > len(suf) and (w[: -len(suf)] + rep) in VOCAB:
                return w[: -len(suf)] + rep
    return w


def _groups():
    v5 = json.loads((PT.PVSCORE_DIR / "pvscore5/thresholds_v5.json").read_text())["params"]["class_similar"]
    vs = json.loads((PT.VIEW_CONF / "view_thresholds.json").read_text())["class_synonyms"]
    sim = {}
    for grp in list(v5) + list(vs):
        g = {norm(x) for x in grp} - {None}
        for a in g:
            sim.setdefault(a, set()).update(g - {a})
    return {k: frozenset(v) for k, v in sim.items()}


SIM = _groups()


def gt_labels(key, cls):
    """labels of a GT element: normalised class + prompt name from a text-item key (<s|v|o>NN_<name>)"""
    out = []
    c = norm(cls)
    if c:
        out.append(c)
    m = KEY_RE.match(key or "")
    if m:
        s = norm(m.group(1))
        if s and s not in out:
            out.append(s)
    return tuple(out)


def credit_word(a, b):
    if not a or not b:
        return 0.0
    if a == b:
        return EXACT
    if b in SIM.get(a, ()):
        return SIMILAR
    return 0.0


def credit(labels, mcls):
    """credit of a model class for a GT element with these labels (max over labels)"""
    b = norm(mcls)
    return max((credit_word(a, b) for a in labels), default=0.0)


def excluded(cls, exclude_classes):
    return norm(cls) in {norm(x) for x in exclude_classes}
