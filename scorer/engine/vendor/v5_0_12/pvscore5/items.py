"""Scored items and their aggregation into blocks and dimensions."""
from __future__ import annotations

import math

# block weights inside each dimension (blocks without items are dropped and the rest renormalised)
BLOCKS = {
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
    "Q": {"tags": 0.4, "ids": 0.2, "names": 0.15, "hierarchy": 0.15, "animation": 0.1},
}
PASS = 0.999


class Items:
    def __init__(self):
        self.rows = []

    def add(self, dim, block, name, score, weight=1.0, value=None, **detail):
        s = 0.0 if score is None or (isinstance(score, float) and math.isnan(score)) else float(score)
        s = min(1.0, max(0.0, s))
        row = {"dim": dim, "block": block, "name": name, "score": round(s, 4), "weight": round(float(weight), 4)}
        if value is not None:
            row["value"] = round(value, 4) if isinstance(value, float) else value
        if detail:
            row["detail"] = detail
        self.rows.append(row)
        return row

    def extend(self, other):
        self.rows.extend(other.rows)

    def dim_scores(self, dims):
        out = {}
        for d in dims:
            rows = [r for r in self.rows if r["dim"] == d and r["weight"] > 0]
            if not rows:
                out[d] = None
                continue
            bw = BLOCKS.get(d, {})
            blocks = {}
            for r in rows:
                blocks.setdefault(r["block"], []).append(r)
            tot_w, acc = 0.0, 0.0
            bdetail = {}
            for b, rs in blocks.items():
                w = sum(r["weight"] for r in rs)
                s = sum(r["weight"] * r["score"] for r in rs) / w
                bwt = bw.get(b, 0.1)
                bdetail[b] = {"score": round(s, 4), "n": len(rs), "block_weight": bwt}
                acc += bwt * s
                tot_w += bwt
            out[d] = {"score": round(acc / tot_w, 4), "n_items": len(rows),
                      "pass_rate": round(sum(1 for r in rows if r["score"] >= PASS) / len(rows), 4),
                      "blocks": bdetail}
        return out
