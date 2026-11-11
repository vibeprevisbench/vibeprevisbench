"""pvscore 5.1 text / image layer r1: exact linear assignment (numpy only, the same code on every host).

lsa(cost) -> col index per row (rows <= cols; transposed internally otherwise), minimum total cost, exact.
Shortest augmenting paths with dual potentials (Jonker-Volgenant form, as Crouse 2016 / scipy's
linear_sum_assignment): one Dijkstra-like search per row over reduced costs. Ties: the lowest column index wins, an
unassigned column before an assigned one at equal distance. O(n^2 m).

max_weight(W, allowed) -> per row: column or -1. Maximum cardinality first (as many rows assigned to allowed columns as
possible), then maximum total weight among those, exactly: dummy 'unassigned' columns of weight 0 and a cardinality
bonus larger than any weight difference.

greedy(cost) is the pvscore5 geom.hungarian fallback when scipy is missing (sort all pairs by cost, take a pair when
its row and column are free); kept here only for the U2 test.
"""
import numpy as np


def lsa(cost):
    C = np.asarray(cost, float)
    if C.size == 0:
        return np.zeros(C.shape[0], int) - 1
    tr = C.shape[0] > C.shape[1]
    if tr:
        C = C.T
    if not np.isfinite(C).all():
        raise ValueError("lsa: cost must be finite (use max_weight for forbidden pairs)")
    n, m = C.shape
    u = np.zeros(n)
    v = np.zeros(m)
    col4row = -np.ones(n, int)
    row4col = -np.ones(m, int)
    for cur in range(n):
        shortest = np.full(m, np.inf)
        path = -np.ones(m, int)
        SR = np.zeros(n, bool)
        SC = np.zeros(m, bool)
        min_val = 0.0
        i = cur
        sink = -1
        while sink < 0:
            SR[i] = True
            r = min_val + C[i] - u[i] - v
            upd = (~SC) & (r < shortest)
            path[upd] = i
            shortest[upd] = r[upd]
            free = np.nonzero(~SC)[0]
            sh = shortest[free]
            low = sh.min()
            ties = free[sh <= low]
            un = ties[row4col[ties] < 0]
            j = int(un[0]) if len(un) else int(ties[0])
            min_val = float(low)
            SC[j] = True
            if row4col[j] < 0:
                sink = j
            else:
                i = int(row4col[j])
        u[cur] += min_val
        others = SR.copy()
        others[cur] = False
        idx = np.nonzero(others)[0]
        u[idx] += min_val - shortest[col4row[idx]]
        v[SC] -= min_val - shortest[SC]
        j = sink
        while True:
            i = int(path[j])
            row4col[j] = i
            col4row[i], j = j, col4row[i]
            if i == cur:
                break
    if tr:
        out = -np.ones(C.shape[1], int)
        for j, i in enumerate(col4row):
            out[i] = j
        return out
    return col4row


def max_weight(W, allowed):
    """rows -> column or -1: maximum cardinality over allowed pairs, then maximum sum of W (exact)"""
    W = np.asarray(W, float)
    allowed = np.asarray(allowed, bool)
    n, m = W.shape
    if n == 0:
        return np.zeros(0, int)
    if m == 0 or not allowed.any():
        return -np.ones(n, int)
    w = np.where(allowed, W, 0.0)
    span = float(np.abs(w).sum()) + 1.0
    bonus = 4.0 * span                           # one more assigned row outweighs any weight difference
    C = np.zeros((n, m + n))
    C[:, :m] = np.where(allowed, -(bonus + w), 0.0)
    col = lsa(C)
    return np.array([c if (c < m and allowed[r, c]) else -1 for r, c in enumerate(col)], int)


def greedy(cost):
    """the pvscore5 geom.hungarian fallback (no scipy): pairs by increasing cost, row and column free"""
    C = np.asarray(cost, float)
    out = -np.ones(C.shape[0], int)
    used = set()
    for i, j in sorted(((i, j) for i in range(C.shape[0]) for j in range(C.shape[1])), key=lambda x: C[x]):
        if out[i] >= 0 or j in used or not np.isfinite(C[i, j]):
            continue
        out[i] = j
        used.add(j)
    return out
