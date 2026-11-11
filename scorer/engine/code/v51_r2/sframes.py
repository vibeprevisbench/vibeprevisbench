"""r2: the one loader of scene_frames.json (view_thresholds_v51r2.json r2_changes.sample_dedupe): samples deduped by
(shot, round(t, 3)), the first kept. Two V2 sub-units of a long shot that overlap by 1 s both give a sample at the
boundary (fork P known_issue dup_sample_times); one video moment is one row."""
import json
from pathlib import Path


def dedupe(g):
    for sh in g["shots"]:
        seen, keep = set(), []
        for sm in sh["samples"]:
            k = round(float(sm["t"]), 3)
            if k in seen:
                continue
            seen.add(k)
            keep.append(sm)
        sh["samples"] = keep
    return g


def load(path):
    return dedupe(json.loads(Path(path).read_text()))
