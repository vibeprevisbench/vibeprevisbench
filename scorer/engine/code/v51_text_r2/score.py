"""pvscore 5.1 text / image layer, r2 (tools/v51_text_r2): score model dumps (and the GT-as-model test variants)
against the GT facts.

    ssh ... "cd $B && env/bin/python -I -B tools/v51_text_r2/score.py --batch DIR [--batch DIR ...] [--workers 48]
             [--models-only | --tests-only] [--no-stem] [ITEM ...]"

r2 (thresholds_v51r2.json, pre-registered): presence = balanced accuracy over GT positives and GT negatives
(gt_facts presence_neg) with the best per-element constant as chance; position / size / facing chance = the best
constant answer (view_score.position_chance / size_best_constant / facing grid); every block flagged low-information
(informative = false, low_info = true) when n < low_info.min_n, a view block when R_gt >= low_info.r_max, presence when a
class has < presence.min_class_n facts. Items = text / image items of the item table with a gt_facts.json; models =
the model files under paths.TEXT_OUT/<item>/model/ (launcher names); the 5.0.x reference reports come from the batch
dirs (tools/v51_r2/batches.py). Paths by tools/v51_r2/paths.py. TEXT_PROFILE_HOOK (below) marks where a text profile
(drafts_v5 claims scoring) plugs in; it does nothing in r2.
r1 description follows.

Inputs: data/v51_text_r2/<item>/gt_facts.json, data/v51_text_r2/<item>/model/<model>.json, tools/v51_text_r2/
thresholds.json (r0 copy) + thresholds_v51r1.json (r1 changes) + class_map_v51r1.json, expected_tests.json (r0 copy).
Rules (r0 unless marked r1):
  class credit    (r1) classes.credit: normalised labels (other:X -> X, spelling, vocabulary plural), the GT element's
                  prompt name from its text-item key, 5.0.9 similar groups + v5.1 view synonyms at 0.75
  correspondence  named GT elements -> model body of the same id; unnamed -> model instance with credit > 0 in the same
                  space (or untagged space). (r1) maximum cardinality, then maximum L1 + R1 + S objective (L2 is not
                  optimised), by deterministic local search alternated with iterated exact assignment
                  (assign.max_weight); one correspondence per (item, model), reused by the view facts
  3D blocks       L1, L2, S (S1 + S2 + S3), R1; fact score x min class credit; missing element -> fact missing (0)
  view blocks     presence, position, size, facing, screen_order, depth_order on the model frame of each GT frame id;
                  model-side rules from tools/v51_r2/view_score.py (imported)
  block           clip((P - R) / (1 - R), 0, 1) x coverage; 3D R by 200 joint geometry permutations within model
                  spaces with the matcher re-run; view R analytic (cell, height, facing) or by 200 permutations of
                  (centre, depth) among each model frame's visible non-surface elements (pairs); presence R = 0
Tests T1 / T2 on the GT-as-model adapter (gt_facts.scene_view), T5-lite Spearman against the 5.0.9 report totals.
--no-stem: ablation without the prompt-name label (models only; writes results_models_nostem.json).
Writes data/v51_text_r2/results.json and corr_results.json (models-only: results_models.json; tests-only: the GT-as-model
tests alone, results_tests.json, used to fix the solver rule before any model was scored).
"""
import copy
import hashlib
import json
import math
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene as S  # noqa: E402  (puts scorer/v5_0 and tools/v51_r2 on the path)
import gt_facts as GF  # noqa: E402
import classes as CL  # noqa: E402
import assign as AS  # noqa: E402
import view_score as VS  # noqa: E402
import paths as PT  # noqa: E402
import items as IT  # noqa: E402
import batches as BT  # noqa: E402

R = S.R
TH = S.TH
TH1 = S.TH1
TH2 = S.TH2
MT = TH1["matching"]
EXP = json.loads((PT.TEXT_CONF / "expected_tests.json").read_text())
GTD = PT.TEXT_GT                     # <item>/gt_facts.json
OUT = PT.TEXT_OUT                    # <item>/model/<model>.json (model_side.py), results*.json
ITEMS = [i for i in IT.of_kinds(IT.TEXT_KINDS) if (GTD / i / "gt_facts.json").exists()]
REVISION = "v51r2"
LI2, PR2 = TH2["low_info"], TH2["presence"]
B3 = ["L1", "L2", "S1", "S2", "S3", "R1"]
OBJ_B = list(MT["objective_blocks"])
VBLK = TH["blocks"]["view"]
NO_STEM = "--no-stem" in sys.argv

# View rules: the text layer calls view_score.cell_score / cell_chance / height_score / height_chance / FACING_CHANCE /
# pair_ok, which read the keys below. They must hold the values of the plain tools/v51_r2/view_thresholds.json (the r0
# view rules this layer is registered with), whichever thresholds file view_score.py itself loads.
VIEW_PLAIN = PT.rel(PT.VIEW_CONF / "view_thresholds.json")
VIEW_KEYS = ["class_synonyms", "screen_gap", "depth_ratio", "cell", "height_log", "facing_deg"]
_vp = json.loads((PT.VIEW_CONF / "view_thresholds.json").read_text())
_vbad = [k for k in VIEW_KEYS if VS.TH.get(k) != _vp.get(k)] + \
    ["chance." + k for k in ("perms", "seed") if VS.TH["chance"].get(k) != _vp["chance"].get(k)]
if _vbad:
    raise SystemExit("view_score.TH differs from %s on %s" % (VIEW_PLAIN, _vbad))
VIEW_TH_LOADED = getattr(VS, "TH_FILE", VIEW_PLAIN)


def is_baseline(name):
    return name == "fake" or name.startswith("dry-run")


def models_of(item):
    d = OUT / item / "model"
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


# TEXT_PROFILE_HOOK: a text item's prompt profile (text lead: tools/text_v5/claims_score.py, SCORING_v5.md) may later
# decide which facts / blocks a text prompt states. r2 has no profile rule: the hook returns the blocks unchanged. The
# packaged scorer (scorer/v5_1) selects the profile per dataset (config text_profile); do not add rules here.
def text_profile_hook(item, blocks):
    return blocks


def labels_of(key, e):
    """GT element labels (gt_facts r1 stores them; --no-stem keeps the class label only)"""
    lab = tuple(e.get("labels") or CL.gt_labels(key, e["cls"]))
    return lab[:1] if NO_STEM else lab


def ramp_up(x, zero, full):
    return np.clip((x - zero) / (full - zero), 0.0, 1.0)


def ramp_down(x, full, zero):
    return np.clip((zero - x) / (zero - full), 0.0, 1.0)


def better(a, b):
    """(cardinality, objective) a strictly better than b"""
    return a[0] > b[0] or (a[0] == b[0] and a[1] > b[1] + MT["improve_tol"])


# ------------------------------------------------------------------------------------------- 3D problem
class Facts:
    """GT side of the 3D facts as index arrays (built once per item)."""

    def __init__(self, gtf):
        E, ET, F = gtf["elements"], gtf["entity_times"], gtf["facts"]
        self.E = E
        self.gk = sorted(E)
        self.ak = sorted(ET)
        self.n = len(self.gk)
        self.N = self.n + len(self.ak)
        idx = {k: i for i, k in enumerate(self.gk)}
        idx.update({k: self.n + j for j, k in enumerate(self.ak)})
        self.un = np.array([i for i, k in enumerate(self.gk) if not k.startswith("b:")], int)

        def arr(rows, k):
            return np.array([[idx[x] for x in r[:k]] for r in rows], int).reshape(-1, k)
        f = {}
        for b in ("L1", "R1"):
            f[b] = {"I": arr(F[b], 3), "W": np.array([r[-1] for r in F[b]], float)}
        f["L2"] = {"I": arr(F["L2"], 3), "W": np.array([r[-1] for r in F["L2"]], float),
                   "sg": np.array([r[3] for r in F["L2"]], float)}
        for b in ("S1", "S2"):
            f[b] = {"I": arr(F[b], 2), "W": np.array([r[-1] for r in F[b]], float),
                    "m": np.array([0 if r[2] == "long" else 1 for r in F[b]], int)}
        f["S3"] = {"I": arr(F["S3"], 1), "W": np.array([r[-1] for r in F["S3"]], float),
                   "g": np.array([r[1] for r in F["S3"]], float)}
        for b in f:
            f[b]["all"] = np.arange(len(f[b]["W"]))
        self.f = f
        self.inv = {b: [np.nonzero((f[b]["I"] == i).any(1))[0] for i in self.un] for b in B3}


class Prob:
    def __init__(self, fx, mv):
        self.fx = fx
        E, mel = fx.E, mv["elements"]
        self.ck = sorted(k for k in mel if not k.startswith("b:"))
        nc = self.nc = len(self.ck)
        self.CX = np.full((nc + 1, 2), np.nan)
        self.CL = np.full(nc + 1, np.nan)
        self.CH = np.full(nc + 1, np.nan)
        for c, k in enumerate(self.ck):
            self.CX[c] = mel[k]["xy"]
            self.CL[c] = mel[k]["long"]
            self.CH[c] = mel[k]["height"]
        N = fx.N
        self.fX = np.full((N, 2), np.nan)
        self.fL = np.full(N, np.nan)
        self.fH = np.full(N, np.nan)
        self.fC = np.zeros(N)
        for i, k in enumerate(fx.gk):
            if k.startswith("b:") and k in mel:
                self.fX[i] = mel[k]["xy"]
                self.fL[i] = mel[k]["long"]
                self.fH[i] = mel[k]["height"]
                self.fC[i] = 1.0
        for j, k in enumerate(fx.ak):
            xy = mv["ent_xy"].get(k)
            if xy is not None:
                self.fX[fx.n + j] = xy
                self.fC[fx.n + j] = 1.0
        nu = self.nu = len(fx.un)
        self.Cr = np.zeros((nu, nc + 1))
        cache = {}
        for u, i in enumerate(fx.un):
            g = E[fx.gk[i]]
            lab = labels_of(fx.gk[i], g)
            for c, k in enumerate(self.ck):
                m = mel[k]
                if m["space"] is None or m["space"] == g["space"]:
                    key = (lab, m["cls"])
                    if key not in cache:
                        cache[key] = CL.credit(lab, m["cls"]) if m["cls"] else 0.0
                    self.Cr[u, c] = cache[key]
        self.allowed = self.Cr[:, :nc] > 0
        self.opts = [np.nonzero(self.allowed[u])[0] for u in range(nu)]
        self.ph = mv.get("person_h")
        self.diag = {}

    # state = (X, L, H, C) over the N GT slots
    def state(self, A):
        X, L, H, C = self.fX.copy(), self.fL.copy(), self.fH.copy(), self.fC.copy()
        if len(A):
            a = np.where(A < 0, self.nc, A)
            un = self.fx.un
            X[un] = self.CX[a]
            L[un] = self.CL[a]
            H[un] = self.CH[a]
            C[un] = self.Cr[np.arange(len(A)), a]
        return X, L, H, C

    def set_slot(self, st, u, c):
        X, L, H, C = (a.copy() for a in st)
        i = self.fx.un[u]
        cc = self.nc if c < 0 else c
        X[i], L[i], H[i], C[i] = self.CX[cc], self.CL[cc], self.CH[cc], self.Cr[u, cc]
        return X, L, H, C

    def score_rows(self, b, P, Lm, Hm, Cm, rows):
        """facts `rows` of block b with member geometry P (..., k, ar, 2), long / height / credit (..., k, ar) ->
        (credit-weighted score, present), each (..., k). The r0 block_eval rules, with leading batch dimensions."""
        f = self.fx.f[b]
        with np.errstate(all="ignore"):
            if b in ("L1", "R1", "L2"):
                pa, pb, pc = P[..., 0, :], P[..., 1, :], P[..., 2, :]
                ok = np.isfinite(pa).all(-1) & np.isfinite(pb).all(-1) & np.isfinite(pc).all(-1)
                if b == "L2":
                    ab, ac, bc = pb - pa, pc - pa, pc - pb
                    cr = ab[..., 0] * ac[..., 1] - ab[..., 1] * ac[..., 0]
                    e2 = np.maximum(np.maximum((ab ** 2).sum(-1), (ac ** 2).sum(-1)), (bc ** 2).sum(-1))
                    fat = np.abs(cr) / np.maximum(e2, 1e-18)
                    s = ((np.sign(cr) == f["sg"][rows]) & (fat >= TH["L2"]["model_fat_min"])).astype(float)
                else:
                    dab = np.linalg.norm(pa - pb, axis=-1)
                    dac = np.linalg.norm(pa - pc, axis=-1)
                    r = np.where(dab <= 1e-12, np.where(dac > 1e-12, np.inf, 1.0), dac / np.maximum(dab, 1e-300))
                    s = ramp_up(r, TH["L1"]["model_zero"], TH["L1"]["model_full"])
            elif b in ("S1", "S2"):
                M = np.where((f["m"][rows] == 0)[:, None], Lm, Hm)
                va, vb = M[..., 0], M[..., 1]
                ok = np.isfinite(va) & np.isfinite(vb) & (va > 0) & (vb > 0)
                r = va / vb
                if b == "S1":
                    s = ramp_up(r, TH["S1"]["model_zero"], TH["S1"]["model_full"])
                else:
                    s = ramp_down(np.maximum(r, 1 / r), TH["S2"]["model_full"], TH["S2"]["model_zero"])
            else:
                h = Hm[..., 0]
                if not self.ph:
                    ok = np.zeros(h.shape, bool)
                    s = np.zeros(h.shape)
                else:
                    ok = np.isfinite(h) & (h > 0)
                    s = ramp_down(np.abs(np.log(h / self.ph / f["g"][rows])), TH["S3"]["full_log"],
                                  TH["S3"]["zero_log"])
            crm = Cm.min(-1)
            pres = ok & (crm > 0) & np.isfinite(s)
            return np.where(pres, s * crm, 0.0), pres

    def block_eval(self, b, st, rows=None):
        """(credit-weighted score, present) per fact of block b (rows: subset)"""
        X, L, H, C = st
        f = self.fx.f[b]
        rows = f["all"] if rows is None else rows
        I = f["I"][rows]
        if not len(I):
            return np.zeros(0), np.zeros(0, bool)
        return self.score_rows(b, X[I], L[I], H[I], C[I], rows)

    def contrib(self, st, rows):
        return sum(float((self.fx.f[b]["W"][r] * self.block_eval(b, st, r)[0]).sum()) for b, r in rows.items()
                   if len(r))

    def objective(self, st, blocks=None):
        return sum(float((self.fx.f[b]["W"] * self.block_eval(b, st)[0]).sum()) for b in (blocks or OBJ_B))

    def lex(self, A, blocks=None):
        return int((A >= 0).sum()), self.objective(self.state(A), blocks)

    # ------------------------------------------------------------------ r1 matcher: iterated exact assignment
    def gain(self, A, blocks=None):
        """G[u, c]: objective of the facts holding u with u -> c, other elements at A (facts whose other member holds
        c count 0)"""
        X, L, H, C = self.state(A)
        G = np.zeros((self.nu, self.nc))
        holder = -np.ones(self.nc, int)
        for j, c in enumerate(A):
            if c >= 0:
                holder[c] = j
        un = self.fx.un
        for u in range(self.nu):
            cs = self.opts[u]
            if not len(cs):
                continue
            slot = un[u]
            hs = holder[cs]
            hslot = np.where((hs >= 0) & (hs != u), un[np.maximum(hs, 0)], -1)
            m = len(cs)
            for b in (blocks or OBJ_B):
                rows = self.fx.inv[b][u]
                if not len(rows):
                    continue
                f = self.fx.f[b]
                I = f["I"][rows]
                isu = I == slot
                P = np.repeat(X[I][None], m, 0)
                Lm = np.repeat(L[I][None], m, 0)
                Hm = np.repeat(H[I][None], m, 0)
                Cm = np.repeat(C[I][None], m, 0)
                P[:, isu] = self.CX[cs][:, None, :]
                Lm[:, isu] = self.CL[cs][:, None]
                Hm[:, isu] = self.CH[cs][:, None]
                Cm[:, isu] = self.Cr[u, cs][:, None]
                Cm[(I[None] == hslot[:, None, None]) & ~isu[None]] = 0.0
                sc, _ = self.score_rows(b, P, Lm, Hm, Cm, rows)
                G[u, cs] += (f["W"][rows][None] * sc).sum(-1)
        return G

    def solve(self, G):
        return AS.max_weight(G + MT["tie_eps"] * self.Cr[:, :self.nc], self.allowed)

    @staticmethod
    def components(A, A2):
        """alternating paths / cycles of the change A -> A2 (each one alone keeps the assignment injective)"""
        D = [u for u in range(len(A)) if A[u] != A2[u]]
        held = {int(A[u]): u for u in range(len(A)) if A[u] >= 0}
        par = {u: u for u in D}

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for u in D:
            v = held.get(int(A2[u])) if A2[u] >= 0 else None
            if v is not None and v in par:
                par[find(u)] = find(v)
        comp = {}
        for u in D:
            comp.setdefault(find(u), []).append(u)
        return [np.array(sorted(c), int) for _, c in sorted(comp.items())]

    def ipfp(self, A, blocks=None):
        cur = self.lex(A, blocks)
        it = 0
        for it in range(1, MT["max_iter"] + 1):
            A2 = self.solve(self.gain(A, blocks))
            if np.array_equal(A2, A):
                break
            v2 = self.lex(A2, blocks)
            if better(v2, cur):
                A, cur = A2, v2
                continue
            best = None
            for comp in self.components(A, A2):
                A3 = A.copy()
                A3[comp] = A2[comp]
                v3 = self.lex(A3, blocks)
                if better(v3, cur) and (best is None or better(v3, best[0])):
                    best = (v3, A3)
            if best is None:
                break
            cur, A = best
        return A, cur, it

    def polish(self, A, blocks=None):
        """deterministic local search: elements in index order, each takes its best improving single change (move to a
        free candidate, or swap with the element holding the candidate; the holder takes u's old candidate when it is
        allowed for it, else is left out), (cardinality, objective) compared lexicographically; sweeps until a sweep
        changes nothing or after polish.max_passes"""
        blocks = blocks or OBJ_B
        A = A.copy()
        st = self.state(A)
        for _ in range(MT["polish"]["max_passes"]):
            changed = False
            for u in range(self.nu):
                best = None
                for c in self.opts[u]:
                    c = int(c)
                    if c == A[u]:
                        continue
                    own = np.nonzero(A == c)[0]
                    if len(own):
                        j = int(own[0])
                        back = int(A[u]) if (A[u] >= 0 and self.allowed[j, A[u]]) else -1
                        dk = (1 if A[u] < 0 else 0) - (1 if back < 0 else 0)
                        rows = {b: np.union1d(self.fx.inv[b][u], self.fx.inv[b][j]) for b in blocks}
                        new = self.set_slot(self.set_slot(st, u, c), j, back)
                    else:
                        j, back = None, None
                        dk = 1 if A[u] < 0 else 0
                        rows = {b: self.fx.inv[b][u] for b in blocks}
                        new = self.set_slot(st, u, c)
                    if dk < 0:
                        continue
                    d = self.contrib(new, rows) - self.contrib(st, rows)
                    if better((dk, d), (0, 0.0)) and (best is None or better((dk, d), best[0])):
                        best = ((dk, d), c, j, back, new)
                if best is not None:
                    _, c, j, back, st = best
                    A[u] = c
                    if j is not None:
                        A[j] = back
                    changed = True
            if not changed:
                break
        return A, self.lex(A, blocks)

    def refine(self, A, blocks=None, order="assign_first"):
        """alternate iterated exact assignment and polish (order: which runs first in a round) until a round improves
        nothing (or max_iter rounds)"""
        cur = self.lex(A, blocks)
        n = 0
        for n in range(1, MT["max_iter"] + 1):
            if order == "assign_first":
                A2 = self.polish(self.ipfp(A, blocks)[0], blocks)[0]
            else:
                A2 = self.ipfp(self.polish(A, blocks)[0], blocks)[0]
            v2 = self.lex(A2, blocks)
            if not better(v2, cur):
                break
            A, cur = A2, v2
        return A, cur, n

    def init_class(self):
        return AS.max_weight(self.Cr[:, :self.nc], self.allowed)

    def match(self, seed=None, blocks=None, use_polish=True):
        """r1 correspondence (deterministic; seed unused, kept for the r0 call signature). use_polish=False: the first
        r1 solver (iterated exact assignment alone), kept for the U3 record and diagnostics"""
        if self.nu == 0 or self.nc == 0:
            self.diag = {"start": None, "rounds": 0, "card": 0, "J": 0.0}
            return -np.ones(self.nu, int)
        best = None
        for name, A0 in (("empty", -np.ones(self.nu, int)), ("class_only", self.init_class())):
            for order in (MT["orders"] if use_polish else ["assign_only"]):
                if order == "assign_only":
                    A, v, it = self.ipfp(A0, blocks)
                else:
                    A, v, it = self.refine(A0, blocks, order)
                if best is None or better(v, best[1]):
                    best = (A, v, name + "/" + order, it)
        self.diag = {"start": best[2], "rounds": best[3], "card": best[1][0], "J": round(best[1][1], 6)}
        return best[0]

    # ------------------------------------------------------------- r0 matcher (diagnostics and test U3 only)
    def init_random(self, rng):
        A = -np.ones(self.nu, int)
        used = set()
        for u in rng.permutation(len(A)):
            opts = [c for c in self.opts[u] if c not in used]
            if opts:
                A[u] = int(opts[rng.integers(len(opts))])
                used.add(A[u])
        return A

    def match_r0(self, seed=0, blocks=None):
        """the r0 tools/v51_text matcher (class-credit Hungarian init, first-improvement move / swap sweeps, 2 random
        greedy restarts), on the given objective blocks"""
        M = TH["matching"]
        blocks = blocks or OBJ_B
        nu = self.nu
        if nu == 0 or self.nc == 0:
            return -np.ones(nu, int)
        rng = np.random.default_rng(seed)
        best, bobj = None, -1.0
        for r in range(M["restarts"]):
            A = self.init_class() if r == 0 else self.init_random(rng)
            st = self.state(A)
            for _ in range(M["max_passes"]):
                changed = False
                for u in range(nu):
                    bd, bo = 1e-9, None
                    for c in self.opts[u]:
                        if c == A[u]:
                            continue
                        own = np.nonzero(A == c)[0]
                        if len(own):
                            j = int(own[0])
                            back = A[u] if (A[u] < 0 or self.Cr[j, A[u]] > 0) else -1
                            rows = {b: np.union1d(self.fx.inv[b][u], self.fx.inv[b][j]) for b in blocks}
                            new = self.set_slot(self.set_slot(st, u, c), j, back)
                        else:
                            j, back = None, None
                            rows = {b: self.fx.inv[b][u] for b in blocks}
                            new = self.set_slot(st, u, c)
                        d = self.contrib(new, rows) - self.contrib(st, rows)
                        if d > bd:
                            bd, bo = d, (c, j, back, new)
                    if bo is not None:
                        c, j, back, new = bo
                        A[u] = c
                        if j is not None:
                            A[j] = back
                        st = new
                        changed = True
                if not changed:
                    break
            obj = self.objective(st, blocks)
            if obj > bobj + 1e-12:
                best, bobj = A.copy(), obj
        return best

    def blocks(self, A):
        st = self.state(A)
        raw = {}
        for b in B3:
            s, pres = self.block_eval(b, st)
            W = self.fx.f[b]["W"]
            raw[b] = {"num": float((W * s).sum()), "wp": float(W[pres].sum()), "wt": float(W.sum()), "n": int(len(W)),
                      "n_present": int(pres.sum())}
        S_ = {k: sum(raw[b][k] for b in ("S1", "S2", "S3")) for k in ("num", "wp", "wt", "n", "n_present")}
        out = {"L1": raw["L1"], "L2": raw["L2"], "S": S_, "R1": raw["R1"]}
        out["_sub"] = {b: raw[b] for b in ("S1", "S2", "S3")}
        return out

    def corr(self, A):
        return {self.fx.gk[self.fx.un[u]]: self.ck[c] for u, c in enumerate(A) if c >= 0}


def permuted(mv, seed):
    """model geometry (xy, long, short, height) permuted jointly within each model space (named bodies included)"""
    rng = random.Random(seed)
    mel = mv["elements"]
    pools = {}
    for k in sorted(mel):
        pools.setdefault(mel[k]["space"], []).append(k)
    new = {k: dict(v) for k, v in mel.items()}
    for ks in pools.values():
        sh = ks[:]
        rng.shuffle(sh)
        for a, b in zip(ks, sh):
            for f in ("xy", "long", "short", "height"):
                new[a][f] = mel[b][f]
    return dict(mv, elements=new)


def P_of(b):
    return b["num"] / b["wp"] if b["wp"] > 0 else None


# ------------------------------------------------------------------------------------------------- views
def angdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def frame_dict(fr):
    d = {}
    for e in fr or []:
        d.setdefault(e["key"], e)
    return d


def view_rows(gtf, mv, corr):
    named = set(mv.get("named") or [])
    frames = {fid: (None if fr is None else frame_dict(fr)) for fid, fr in mv["frames"].items()}

    def mkey(k):
        if k.startswith("b:"):
            return k if k[2:] in named else None
        return corr.get(k)
    rows = {b: [] for b in VBLK}     # (score, chance key (r2) or None, present, pair spec)
    F = gtf["facts"]
    for sign, facts in ((1, F["presence"]), (-1, F.get("presence_neg") or [])):
        for fid, k in facts:
            mk, fr = mkey(k), frames.get(fid)
            pres = mk is not None and fr is not None
            hit = bool(pres and mk in fr)
            rows["presence"].append((float(hit if sign > 0 else (pres and not hit)), VS.pk(sign, k), pres, None))
    for fid, k, cell in F["position"]:
        mk, fr = mkey(k), frames.get(fid)
        pres = mk is not None and fr is not None
        e = fr.get(mk) if pres else None
        rows["position"].append((VS.cell_score(cell, e["cell"]) if e else 0.0, ("cell", cell), pres, None))
    for fid, k, h in F["size"]:
        mk, fr = mkey(k), frames.get(fid)
        pres = mk is not None and fr is not None
        e = fr.get(mk) if pres else None
        rows["size"].append((VS.height_score(h, e["height_frac"]) if e else 0.0, ("height", h), pres, None))
    a_, b_ = VS.TH["facing_deg"]
    for fid, k, ang in F["facing"]:
        mk, fr = mkey(k), frames.get(fid)
        pres = mk is not None and fr is not None
        e = fr.get(mk) if pres else None
        if e and e.get("facing_unknown"):
            continue            # 5.2 (M14): the model's facing is unknown in that frame, not scored
        fc = e.get("facing_cam") if e else None
        rows["facing"].append((float(ramp_down(angdiff(fc, ang), a_, b_)) if fc is not None else 0.0,
                               ("angle", ang), pres, None))
    for fid, ka, kb, rel in F["screen_order"]:
        ma, mb, fr = mkey(ka), mkey(kb), frames.get(fid)
        pres = ma is not None and mb is not None and fr is not None
        rows["screen_order"].append((None, None, pres, (fid, ma, mb, rel)))
    for fid, ka, kb in F["depth_order"]:
        ma, mb, fr = mkey(ka), mkey(kb), frames.get(fid)
        pres = ma is not None and mb is not None and fr is not None
        rows["depth_order"].append((None, None, pres, (fid, ma, mb, "in_front_of")))
    return rows, frames


def pair_scores(rows, frames):
    out = []
    for s, r, pres, spec in rows:
        if not pres:
            out.append(0.0)
            continue
        fid, ma, mb, rel = spec
        ea, eb = frames[fid].get(ma), frames[fid].get(mb)
        out.append(float(VS.pair_ok(rel, ea, eb)) if ea is not None and eb is not None else 0.0)
    return out


def view_blocks(gtf, mv, corr):
    rows, frames = view_rows(gtf, mv, corr)
    out = {}
    pair_s = {b: pair_scores(rows[b], frames) for b in ("screen_order", "depth_order")}
    # pairwise chance: permute (centre, depth) among each model frame's visible non-surface elements
    V = TH["view"]
    rng = random.Random(VS.TH["chance"]["seed"])
    nperm = VS.TH["chance"]["perms"]
    perm_P = {b: [] for b in pair_s}
    for _ in range(nperm):
        pf = {}
        for fid, fr in frames.items():
            if fr is None:
                pf[fid] = None
                continue
            ks = [k for k, e in fr.items() if not str(e["class"]).startswith("surface:")]
            sh = ks[:]
            rng.shuffle(sh)
            d = dict(fr)
            for a, b in zip(ks, sh):
                d[a] = dict(fr[a], c=fr[b]["c"], depth_med=fr[b]["depth_med"])
            pf[fid] = d
        for b in pair_s:
            ps = pair_scores(rows[b], pf)
            pr = [x for x, rr in zip(ps, rows[b]) if rr[2]]
            perm_P[b].append(sum(pr) / len(pr) if pr else None)
    for b in VBLK:
        rs = rows[b]
        n = len(rs)
        if n == 0:
            out[b] = {"score": None, "n": 0}
            continue
        pres = [r for r in rs if r[2]]
        extra = {}
        R_gt = None
        if b in pair_s:
            sc = [x for x, r in zip(pair_s[b], rs) if r[2]]
            pp = [x for x in perm_P[b] if x is not None]
            Rb = sum(pp) / len(pp) if pp else 0.0
            P = sum(sc) / len(sc) if sc else 0.0
        elif b == "presence":                                         # r2: balanced accuracy
            pos = [r[0] for r in pres if VS.is_pos(r[1])]
            neg = [r[0] for r in pres if VS.is_neg(r[1])]
            tpr = sum(pos) / len(pos) if pos else None
            tnr = sum(neg) / len(neg) if neg else None
            if pos and neg:
                P, Rb = 0.5 * (tpr + tnr), chance_r2(b, pres)
            else:
                P, Rb = (tpr if pos else tnr) or 0.0, 1.0
            R_gt = chance_r2(b, rs)
            extra = {"n_pos": sum(1 for r in rs if VS.is_pos(r[1])), "n_neg": sum(1 for r in rs if VS.is_neg(r[1])),
                     "TPR": None if tpr is None else round(tpr, 4), "TNR": None if tnr is None else round(tnr, 4)}
        else:
            sc = [r[0] for r in pres]
            P = sum(sc) / len(sc) if sc else 0.0
            Rb = chance_r2(b, pres) if pres else 0.0
            R_gt = chance_r2(b, rs)
        cov = len(pres) / n
        info, why = True, None
        if b == "presence" and not (extra["n_pos"] >= PR2["min_class_n"] and extra["n_neg"] >= PR2["min_class_n"]):
            info, why = False, "%d positive / %d negative facts (< %d in a class)" % (
                extra["n_pos"], extra["n_neg"], PR2["min_class_n"])
        if info and R_gt is not None and R_gt >= LI2["r_max"]:
            info, why = False, "R_gt %.3f >= %.2f" % (R_gt, LI2["r_max"])
        if info and n < LI2["min_n"]:
            info, why = False, "n %d < low_info.min_n %d" % (n, LI2["min_n"])
        out[b] = {"score": round(corrected(P, Rb) * cov, 4), "P": round(P, 4), "R": round(Rb, 4),
                  "coverage": round(cov, 4), "n": n, "n_present": len(pres),
                  "R_gt": None if R_gt is None else round(R_gt, 4), "informative": info, "low_info": not info,
                  "info_reason": why, **extra}
    _ = V
    return out


def chance_r2(b, rows):
    """r2 view chance (thresholds_v51r2.json) over rows (score, key, present, spec); key as view_rows sets it"""
    if not rows:
        return None
    vr = [(b, r[0], r[2], r[1]) for r in rows]                     # view_score row layout: (block, score, present, key)
    if b == "presence":
        return VS.presence_label_constant(vr)
    if b == "position":
        return VS.position_chance(vr)
    if b == "size":
        return VS.size_best_constant(vr)
    if b == "facing":
        a_, b_ = VS.TH["facing_deg"]
        return max(sum(float(ramp_down(angdiff(a, r[1][1]), a_, b_)) for r in rows) / len(rows) for a in VS.FACING_GRID)
    raise ValueError(b)


def corrected(P, Rb):
    if P is None:
        return 0.0
    return float(min(1.0, max(0.0, (P - Rb) / (1 - Rb)))) if Rb < 1 else 0.0


# ---------------------------------------------------------------------------------------------- scoring
FX_CACHE = {}


def facts_of(item):
    if item not in FX_CACHE:
        gtf = json.loads((GTD / item / "gt_facts.json").read_text())
        FX_CACHE[item] = (gtf, Facts(gtf))
    return FX_CACHE[item]


def score_main(item, mv, diag_r0=False):
    gtf, fx = facts_of(item)
    pb = Prob(fx, mv)
    A = pb.match()
    bl = pb.blocks(A)
    corr = pb.corr(A)
    vb = view_blocks(gtf, mv, corr)
    st = pb.state(A)
    md = dict(pb.diag, J_all_blocks=round(pb.objective(st, B3), 6),
              n_unnamed_gt=int(pb.nu), n_with_candidate=int(sum(len(o) > 0 for o in pb.opts)))
    if diag_r0:
        # the r0 matcher and the iterated assignment alone on the same r1 problem (classes and objective): optimiser
        # comparison only, not a score
        A0 = pb.match_r0(TH["matching"]["seed"])
        c0, j0 = pb.lex(A0)
        md["r0_matcher"] = {"card": c0, "J": round(j0, 6), "same_corr": bool(np.array_equal(A0, A))}
        keep = dict(pb.diag)
        Ai = pb.match(use_polish=False)
        ci, ji = pb.lex(Ai)
        pb.diag = keep
        md["ipfp_only"] = {"card": ci, "J": round(ji, 6), "same_corr": bool(np.array_equal(Ai, A))}
    return {"blocks3d": bl, "view": vb, "corr": corr, "objective": pb.objective(st), "match": md}


def score_perms(item, mv, seeds):
    gtf, fx = facts_of(item)
    out = []
    for s in seeds:
        pb = Prob(fx, permuted(mv, TH["chance_3d"]["seed"] * 100003 + s))
        A = pb.match()
        bl = pb.blocks(A)
        out.append({b: P_of(bl[b]) for b in ("L1", "L2", "S", "R1")})
    return out


def finish(item, main, perms):
    out = {}
    for b in ("L1", "L2", "S", "R1"):
        x = main["blocks3d"][b]
        if x["n"] == 0:
            out[b] = {"score": None, "n": 0}
            continue
        P = P_of(x)
        pp = [p[b] for p in perms if p[b] is not None]
        Rb = sum(pp) / len(pp) if pp else 0.0
        cov = x["wp"] / x["wt"] if x["wt"] > 0 else 0.0
        info = x["n"] >= LI2["min_n"]                                  # r2: low-information below min_n facts
        out[b] = {"score": round(corrected(P, Rb) * cov, 4), "P": None if P is None else round(P, 4),
                  "R": round(Rb, 4), "coverage": round(cov, 4), "n": x["n"], "n_present": x["n_present"],
                  "informative": info, "low_info": not info,
                  "info_reason": None if info else "n %d < low_info.min_n %d" % (x["n"], LI2["min_n"])}
    wl = TH["blocks"]["layer3d"]
    have = [b for b in wl if out[b]["score"] is not None]
    l3 = sum(wl[b] * out[b]["score"] for b in have) / sum(wl[b] for b in have) if have else None
    vb = text_profile_hook(item, main["view"])                       # TEXT_PROFILE_HOOK (r2: unchanged)
    vh = [b for b in VBLK if vb[b]["score"] is not None]
    vw = sum(vb[b]["score"] for b in vh) / len(vh) if vh else None
    lay = None if l3 is None and vw is None else (0.5 * (l3 or 0) + 0.5 * (vw or 0) if l3 is not None and vw is not None
                                                   else (l3 if vw is None else vw))
    return {"layer3d": None if l3 is None else round(l3, 4), "view": None if vw is None else round(vw, 4),
            "layer": None if lay is None else round(lay, 4), "blocks3d": out, "blocks_view": vb,
            "S_sub": {k: v for k, v in main["blocks3d"]["_sub"].items()}, "n_corr": len(main["corr"]),
            "match": main["match"]}


# ------------------------------------------------------------------------------------------ test variants
def base_scene(item):
    gt, _ = S.load_gt(item)
    return S.scene_from_gt(gt, item)


def adapter(sc, gtf, rename=None):
    v = GF.scene_view(sc, gtf, rename)
    v["named"] = sorted({o["id"] for o in sc["objects"] if o["id"]} | set(sc["tracks"]))
    return v


def most_fact_unnamed(gtf):
    cnt = {}
    for b in B3:
        for r in gtf["facts"][b]:
            for x in r[:3 if b in ("L1", "L2", "R1") else (2 if b in ("S1", "S2") else 1)]:
                if isinstance(x, str) and x in gtf["elements"] and not x.startswith("b:"):
                    cnt[x] = cnt.get(x, 0) + 1
    return max(sorted(cnt), key=lambda k: cnt[k]) if cnt else None


def shift_obj(sc, o, d):
    o["center"][0] += d[0]
    o["center"][1] += d[1]
    if o["id"] and o["id"] in sc["tracks"]:
        tr = sc["tracks"][o["id"]]
        tr["p"] = [[p[0] + d[0], p[1] + d[1]] for p in tr["p"]]


def obj_key(o):
    return ("b:" + o["id"]) if o["id"] else o["key"]


def ref_xy(sc, o):
    """the xy the 3D element of o is read at (named: track at t_ref)"""
    tr = sc["tracks"].get(o["id"]) if o["id"] else None
    if tr and tr["t"] and sc["modality"] != "image":
        return S.track_at(tr, GF.t_ref_of(sc, o["id"]))
    return np.asarray(o["center"][:2], float)


def cam_other_side(sc, gtf):
    sc = copy.deepcopy(sc)
    dep = {}
    for fr in gtf["frames"]:
        ds = [e["depth_med"] for e in fr["elements"]]
        dep.setdefault(fr["shot"], []).extend(ds)
    Rz = np.diag([-1.0, -1.0, 1.0])
    for sh in sc["shots"]:
        for c in sh["cams"]:
            if "c2w" in c:
                M = np.asarray(c["c2w"], float)
                D = float(np.median(dep[sh["id"]])) if dep.get(sh["id"]) else 5.0 * sc["scale"]
                pos, fwd = M[:3, 3], -M[:3, 2]
                tgt = pos + fwd * D
                N = np.eye(4)
                N[:3, :3] = Rz @ M[:3, :3]
                N[:3, 3] = [2 * tgt[0] - pos[0], 2 * tgt[1] - pos[1], pos[2]]
                c["c2w"] = N.tolist()
            elif c.get("pos") is not None and c.get("look") is not None and S.full(c["pos"][:2]) and \
                    S.full(c["look"][:2]):
                p, lk = c["pos"], c["look"]
                c["pos"] = [2 * lk[0] - p[0], 2 * lk[1] - p[1], p[2]]
    return sc


def variant(item, name):
    """(scene, rename) for a test variant of the GT-as-model adapter"""
    gtf, _ = facts_of(item)
    sc = base_scene(item)
    rn = None
    unnamed = [o["key"] for o in sc["objects"] if not o["id"]]
    if name in ("rename_reorder", "rot37_scale1.6_rename"):
        rng = random.Random(7)
        sh = unnamed[:]
        rng.shuffle(sh)
        rn = {k: "r%03d" % i for i, k in enumerate(sh)}
        rng.shuffle(sc["objects"])
    if name == "identity" or name == "rename_reorder":
        return sc, rn
    if name == "rot37_trans":
        return S.transform(sc, 37.0, (5.3, -2.1), 1.0), rn
    if name == "scale0.6":
        return S.transform(sc, 0.0, (0.0, 0.0), 0.6), rn
    if name == "scale1.6":
        return S.transform(sc, 0.0, (0.0, 0.0), 1.6), rn
    if name == "rot37_scale1.6_rename":
        return S.transform(sc, 37.0, (5.3, -2.1), 1.6), rn
    if name == "mirror_layout":
        return S.transform(sc, mirror=True), rn
    if name == "camera_other_side":
        return cam_other_side(sc, gtf), rn
    tgt = most_fact_unnamed(gtf)
    if name == "move_one_across":
        o = next(o for o in sc["objects"] if o["key"] == tgt)
        cen = np.asarray(sc["spaces"][gtf["elements"][tgt]["space"]]["centre"], float)
        shift_obj(sc, o, 2 * cen - 2 * np.asarray(o["center"][:2], float))
        return sc, rn
    if name == "drop_one":
        sc["objects"] = [o for o in sc["objects"] if o["key"] != tgt]
        return sc, rn
    if name == "drop_20pct":
        un = sorted(k for k in gtf["elements"] if not k.startswith("b:"))
        drop = set(random.Random(0).sample(un, int(round(0.2 * len(un)))))
        sc["objects"] = [o for o in sc["objects"] if o["key"] not in drop]
        return sc, rn
    if name.startswith("jitter"):
        k = float(name[6:])
        rng = np.random.default_rng(0)
        for o in sc["objects"]:
            L = sc["spaces"].get(o["space"], next(iter(sc["spaces"].values())))["L"]
            shift_obj(sc, o, rng.normal(0.0, k * L, 2))
        return sc, rn
    if name == "shuffle_all":
        rng = random.Random(0)
        els = [o for o in sc["objects"] if obj_key(o) in gtf["elements"]]
        pools = {}
        for o in els:
            pools.setdefault(gtf["elements"][obj_key(o)]["space"], []).append(o)
        for os_ in pools.values():
            xy = [ref_xy(sc, o) for o in os_]
            sh = xy[:]
            rng.shuffle(sh)
            for o, a, b in zip(os_, xy, sh):
                shift_obj(sc, o, b - a)
        return sc, rn
    if name == "collapse":
        for o in sc["objects"]:
            if obj_key(o) in gtf["elements"]:
                cen = np.asarray(sc["spaces"][gtf["elements"][obj_key(o)]["space"]]["centre"], float)
                shift_obj(sc, o, cen - ref_xy(sc, o))
        return sc, rn
    raise ValueError(name)


T1 = EXP["T1_invariance"]["variants"]
T2 = ["move_one_across", "mirror_layout", "drop_one", "drop_20pct", "camera_other_side", "jitter0.05", "jitter0.1",
      "jitter0.2", "shuffle_all", "collapse", "empty"]


def mv_for(item, who):
    if who.startswith("model:"):
        mv = MV_INLINE.get((item, who[6:])) or json.loads((OUT / item / "model" / f"{who[6:]}.json").read_text())
        mv["named"] = mv["bodies"]
        return mv
    name = who[3:]
    gtf, _ = facts_of(item)
    if name == "empty":
        return {"elements": {}, "ent_xy": {k: None for k in gtf["entity_times"]}, "person_h": None,
                "frames": {fr["fid"]: None for fr in gtf["frames"]}, "named": []}
    sc, rn = variant(item, name)
    return adapter(sc, gtf, rn)


MV_CACHE = {}
MV_INLINE = {}           # (item, model) -> model_side result set in memory by a caller (score51.py) before forking


def job(args):
    kind, item, who, seeds = args
    t0 = time.time()
    key = (item, who)
    if key not in MV_CACHE:
        MV_CACHE.clear()
        MV_CACHE[key] = mv_for(item, who)
    mv = MV_CACHE[key]
    if kind == "main":
        return kind, item, who, None, score_main(item, mv, diag_r0=True), time.time() - t0
    return kind, item, who, seeds[0], score_perms(item, mv, seeds), time.time() - t0


# ------------------------------------------------------------------------------------------------ checks
def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else None


def blocks_flat(r):
    d = {b: r["blocks3d"][b]["score"] for b in ("L1", "L2", "S", "R1")}
    d.update({b: r["blocks_view"][b]["score"] for b in VBLK})
    d.update({"layer3d": r["layer3d"], "view": r["view"], "layer": r["layer"]})
    return d


def check_tests(res):
    out = {}
    for item in sorted(i for i in res if all("gt:" + w in res[i] for w in T1 + T2)):
        T = {w: blocks_flat(res[item]["gt:" + w]) for w in T1 + T2}
        idt = T["identity"]
        c = {}
        for w in T1:
            bad = [b for b, v in T[w].items() if v is not None and v < 0.99]
            tol = 0.01 if "rename" in w else 1e-6
            dif = [b for b, v in T[w].items() if v is not None and idt[b] is not None and abs(v - idt[b]) > tol]
            c["T1:" + w] = {"pass": not bad and not dif, "below_0.99": bad, "diff_vs_identity": dif}

        def ge(a, b):
            return a is None or b is None or a >= b - 1e-9
        m = T["move_one_across"]
        c["T2:move_one_across"] = {"pass": m["layer3d"] is not None and m["layer3d"] < idt["layer3d"] and
                                   idt["layer3d"] - m["layer3d"] <= 0.35, "layer3d": m["layer3d"]}
        mi = T["mirror_layout"]
        ok = (mi["L2"] is None or mi["L2"] <= 0.10) and all(mi[b] is None or mi[b] >= 0.99 for b in ("L1", "S")) and \
            (mi["screen_order"] is None or mi["screen_order"] <= 0.6) and \
            (mi["position"] is None or mi["position"] < idt["position"])
        c["T2:mirror_layout"] = {"pass": ok, **{b: mi[b] for b in ("L1", "L2", "S", "screen_order", "position")}}
        d1 = T["drop_one"]
        c["T2:drop_one"] = {"pass": all(ge(idt[b], d1[b]) for b in d1), "layer3d": d1["layer3d"], "view": d1["view"]}
        d2 = T["drop_20pct"]
        c["T2:drop_20pct"] = {"pass": all(d2[b] is None or 0.5 <= d2[b] <= 0.95 for b in ("layer3d", "view")),
                              "layer3d": d2["layer3d"], "view": d2["view"]}
        co = T["camera_other_side"]
        c["T2:camera_other_side"] = {"pass": all(co[b] is None or co[b] >= 0.99 for b in ("L1", "L2", "S", "R1")) and
                                     (co["view"] is None or co["view"] < 0.6), "view": co["view"],
                                     "layer3d": co["layer3d"]}
        j = [T["jitter0.05"]["layer3d"], T["jitter0.1"]["layer3d"], T["jitter0.2"]["layer3d"]]
        c["T2:jitter"] = {"pass": all(j[i] >= j[i + 1] - 1e-9 for i in range(2)), "layer3d": j,
                          "view": [T["jitter0.05"]["view"], T["jitter0.1"]["view"], T["jitter0.2"]["view"]]}
        s = T["shuffle_all"]
        c["T2:shuffle_all"] = {"pass": all(s[b] is None or s[b] <= 0.25 for b in ("L1", "L2")),
                               "L1": s["L1"], "L2": s["L2"]}
        cl = T["collapse"]
        c["T2:collapse"] = {"pass": all(cl[b] is None or cl[b] == 0 for b in ("L1", "L2", "R1")),
                            **{b: cl[b] for b in ("L1", "L2", "R1")}}
        e = T["empty"]
        c["T2:empty"] = {"pass": all(v is None or v == 0 for v in e.values())}
        out[item] = c
    return out


def run_jobs(pairs, workers, nperm=None):
    """pairs: [(item, who)] with who = "model:<name>" or "gt:<variant>" -> {item: {who: finish() + cpu_s, corr}}"""
    nperm = TH["chance_3d"]["perms"] if nperm is None else nperm
    chunk = 25
    jobs = []
    for item, who in pairs:
        jobs.append(("main", item, who, None))
        for s0 in range(0, nperm, chunk):
            jobs.append(("perm", item, who, list(range(s0, min(nperm, s0 + chunk)))))
    size = {i: json.loads((GTD / i / "gt_facts.json").read_text())["counts"]["elements3d"] for i, _ in pairs}
    jobs.sort(key=lambda j: (-size[j[1]], j[1], j[2]))              # the big items first
    t0 = time.time()
    mains, perms, secs = {}, {}, {}
    with Pool(workers) as pool:
        for n, (kind, item, who, s0, r, dt) in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            if kind == "main":
                mains[(item, who)] = r
            else:
                perms.setdefault((item, who), {})[s0] = r
            secs[(item, who)] = secs.get((item, who), 0.0) + dt
            if n % 50 == 0:
                print("%d / %d jobs, %.0fs" % (n, len(jobs), time.time() - t0), flush=True)
    res = {}
    for (item, who), m in mains.items():
        pp = [x for k in sorted(perms.get((item, who), {})) for x in perms[(item, who)][k]]
        res.setdefault(item, {})[who] = dict(finish(item, m, pp), cpu_s=round(secs[(item, who)], 1),
                                             corr=m["corr"])
    return res


def provenance():
    return {"revision": REVISION + ("-nostem" if NO_STEM else ""),
            "thresholds": [PT.rel(PT.TEXT_CONF / f) for f in ("thresholds.json", "thresholds_v51r1.json",
                                                              "thresholds_v51r2.json")],
            "class_map": PT.rel(PT.TEXT_CONF / "class_map_v51r1.json"),
            "expected": PT.rel(PT.TEXT_CONF / "expected_tests.json"), "gt_facts": PT.rel(GTD),
            "text_profile_hook": "TEXT_PROFILE_HOOK: none (r2)"}


def main():
    a = sys.argv[1:]
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 48
    bdirs = [a[i + 1] for i, x in enumerate(a) if x == "--batch"]
    only_models = "--models-only" in a or NO_STEM
    only_tests = "--tests-only" in a and not only_models
    if not only_tests and not bdirs:
        raise SystemExit("--batch DIR required (the 5.0.x reference reports come from the batch dirs)")
    items = [x for x in a if x in ITEMS] or ITEMS
    pairs = []
    for item in items:
        if not only_tests:
            pairs += [(item, "model:" + m) for m in models_of(item)]
        if not only_models:
            pairs += [(item, "gt:" + w) for w in T1 + T2]
    t0 = time.time()
    res = run_jobs(pairs, workers)
    if only_tests:
        # GT-as-model tests only (run before any model score, to fix the solver rule on tests alone)
        out = {"schema": "pvscore-5.1-text-image", **provenance(), "wall_s": round(time.time() - t0, 1),
               "deps": deps_record(),
               "scores": {i: {w: {k: v for k, v in r.items() if k != "corr"} for w, r in d.items()}
                          for i, d in res.items()}, "tests": check_tests(res)}
        out["revision"] += "-tests"
        (OUT / "results_tests.json").write_text(json.dumps(out, indent=1) + "\n")
        print("wrote", OUT / "results_tests.json", "%.0fs" % (time.time() - t0), flush=True)
        return
    # T5-lite (5.0.x reference reports of the same attempts) and model order against the baseline (fixture) model
    idx = BT.latest(bdirs)
    rep = {}
    for (mname, item), att in idx.items():
        if item in res and "model:" + mname in res[item] and att.get("report"):
            r = json.loads(Path(att["report"]).read_text())
            rep[(item, mname)] = {"total": r["total"], "schema": r.get("schema"), "report": PT.rel(att["report"]),
                                  "A2": (r["dims"].get("A2") or {}).get("score"),
                                  "C1": (r["dims"].get("C1") or {}).get("score"),
                                  "P1": (r["dims"].get("P1") or {}).get("score")}
    sp = {}
    for ours, theirs in (("layer", "total"), ("layer3d", "A2"), ("view", "C1"), ("layer3d", "total"),
                         ("view", "total")):
        xs, ys = [], []
        for (i, m), rr in sorted(rep.items()):
            x, y = res[i]["model:" + m][ours], rr[theirs]
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        sp[f"{ours}_vs_{theirs}"] = {"rho": None if len(xs) < 3 else round(spearman(xs, ys), 3), "n": len(xs)}
    mc = {}
    for item, d in res.items():
        base = [w for w in d if w.startswith("model:") and is_baseline(w[6:])]
        if not base:
            continue
        f = d[base[0]]
        mc[item] = {"baseline": base[0][6:]}
        for w, o in d.items():
            if w.startswith("model:") and not is_baseline(w[6:]):
                mc[item][w[6:]] = {k: (f[k] is None or o[k] is None or f[k] < o[k]) for k in ("layer3d", "view")}
    out = {"schema": "pvscore-5.1-text-image", **provenance(), "wall_s": round(time.time() - t0, 1),
           "batches": [PT.rel(b) for b in bdirs], "deps": deps_record(),
           "scores": {i: {w: {k: v for k, v in r.items() if k != "corr"} for w, r in d.items()} for i, d in res.items()},
           "reports_5_0_x": {f"{i}|{m}": v for (i, m), v in sorted(rep.items())},
           "T5_lite": sp, "model_order": mc}
    if not only_models:
        out["tests"] = check_tests(res)
    name = ("results_models_nostem.json" if NO_STEM else "results_models.json") if only_models else "results.json"
    (OUT / name).write_text(json.dumps(out, indent=1) + "\n")
    (OUT / ("corr_" + name)).write_text(json.dumps({i: {w: r["corr"] for w, r in d.items()} for i, d in res.items()},
                                                   indent=0) + "\n")
    print("wrote", OUT / name, "%.0fs" % (time.time() - t0), flush=True)


def deps():
    """sha256 (first 16 hex) of the files this layer reads, now"""
    vd = Path(VS.__file__).resolve().parent
    fs = [vd / f for f in ("view_score.py", "scene_geom.py", "gt_root.py", "model_view.py", "paths.py", "items.py",
                           "batches.py", "sframes.py")]
    fs += [PT.VIEW_CONF / f for f in ("view_thresholds.json", "view_thresholds_v51r1.json", "view_thresholds_v51r2.json")]
    fs += [PT.PVSCORE_DIR / "pvscore5/thresholds_v5.json", R / "common/categories_v5.0.json"]
    fs += [p for p in sorted(HERE.iterdir()) if p.suffix in (".py", ".json", ".sh")]
    fs += [p for p in sorted(PT.TEXT_CONF.iterdir()) if p.suffix == ".json"]
    return {PT.rel(f): hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in dict.fromkeys(fs) if f.exists()}


DEPS_START = deps()      # at import, i.e. the versions the workers (forked from this process) run with


def deps_record():
    end = deps()
    return {"at_start": DEPS_START, "changed_during_run": sorted(f for f in set(end) | set(DEPS_START)
                                                                  if end.get(f) != DEPS_START.get(f)),
            "view_thresholds_loaded_by_view_score": VIEW_TH_LOADED,
            "view_keys_checked_equal_to_plain": VIEW_KEYS + ["chance.perms", "chance.seed"]}


if __name__ == "__main__":
    main()
