"""r2: the item table of the bench (no hard-coded item lists): paths.ITEMS_FILE (gt/items.jsonl). kind: script (text), continuation, spatial_arkit,
spatial_keyframe, video."""
import json

import paths as PT

ITEMS_FILE = PT.ITEMS_FILE
VIEW_KINDS = ("video", "continuation")                      # view layer (checklist + scene_frames)
TEXT_KINDS = ("script", "spatial_arkit", "spatial_keyframe")  # text / image layer (gt_facts)


def table():
    return {d["id"]: d for d in (json.loads(x) for x in ITEMS_FILE.read_text().splitlines() if x.strip())}


def of_kinds(kinds):
    return [i for i, d in table().items() if d["kind"] in kinds]


def key_of(item):
    return table()[item]["source_key"]


def kind_of(item):
    return table()[item]["kind"]
