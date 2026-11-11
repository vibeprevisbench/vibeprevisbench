"""Usage normalisation and tool-argument repair for OpenAI-compatible responses (the two helpers the harness model
adapter uses; code unchanged from the harness the paper's runs used)."""
from __future__ import annotations

from typing import List, Optional


def norm_usage(u) -> Optional[dict]:
    """usage of a response -> input / output / total / reasoning / cached tokens"""
    if u is None:
        return None
    if isinstance(u, dict):
        g = u.get
    else:
        g = lambda n, d=None: getattr(u, n, d)  # noqa: E731

    def first(*names):
        for n in names:
            v = g(n)
            if v is not None:
                return int(v)
        return None
    out = {"input_tokens": first("prompt_tokens", "input_tokens"),
           "output_tokens": first("completion_tokens", "output_tokens"),
           "total_tokens": first("total_tokens")}
    ctd = g("completion_tokens_details")
    rt = (ctd.get("reasoning_tokens") if isinstance(ctd, dict) else getattr(ctd, "reasoning_tokens", None)) if ctd else None
    if rt is not None:
        out["reasoning_tokens"] = int(rt)
    ptd = g("prompt_tokens_details")
    ct = (ptd.get("cached_tokens") if isinstance(ptd, dict) else getattr(ptd, "cached_tokens", None)) if ptd else None
    if ct is not None:
        out["cached_tokens"] = int(ct)
    return out


def _fix_args(tcs: List[dict]) -> None:
    """empty tool arguments become "{}"; arguments are cut after the first balanced JSON object (string-aware)"""
    for tc in tcs:
        a = (tc.get("function", {}).get("arguments") or "").strip()
        if not a:
            tc["function"]["arguments"] = "{}"
            continue
        if not a.startswith("{"):
            continue
        depth, end, in_str, esc = 0, None, False, False
        for i, ch in enumerate(a):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is not None and end < len(a):
            tc["function"]["arguments"] = a[:end]


