"""Give every GPS point ONE reference arc by aligning the track to the reference route.

Why this is needed: `.route` is an ordered list of arcs, but nothing says which arc each GPS
point belongs to (1315 points vs 525 arcs in trip 1). We recover the correspondence with a small
dynamic program over route positions:

    cost(i, j) = distance(point i, route arc j)
    dp[i, j]   = cost(i, j) + min_{j' <= j} ( dp[i-1, j'] + LAMBDA * (j - j') )

i.e. the aligned route position can stay or move FORWARD as time advances (never backwards), and
skipping arcs costs a tiny LAMBDA per skipped arc (fast vehicles legitimately skip arcs between
1 Hz samples). The best path is recovered by back-pointers.

This is used ONLY to build training/evaluation labels on partitions that ship a .route file. It is
never part of the prediction pipeline.
"""
import numpy as np

from .data_io import point_segment_distance

LAMBDA = 0.05  # metres of penalty per skipped route arc


def _dist_matrix(P, Ar, Br, chunk=2000):
    n, m = len(P), len(Ar)
    D = np.empty((n, m), dtype=np.float32)
    ab = Br - Ar
    L2 = np.maximum((ab ** 2).sum(1), 1e-9)
    for s in range(0, n, chunk):
        p = P[s:s + chunk, None, :]
        t = np.clip(((p - Ar[None]) * ab[None]).sum(2) / L2[None], 0, 1)
        foot = Ar[None] + ab[None] * t[..., None]
        D[s:s + chunk] = np.linalg.norm(p - foot, axis=2)
    return D


def align_to_route(P, A, B, route, lam=LAMBDA):
    """P: (n,2) projected GPS points. A,B: (n_arcs,2) projected arc endpoints. route: arc ids.
    Returns (pos, arc, dist): aligned route position, aligned arc id, distance point->that arc (m)."""
    Ar, Br = A[route], B[route]
    D = _dist_matrix(P, Ar, Br)
    n, m = D.shape
    j = np.arange(m, dtype=np.float32)
    back = np.empty((n, m), dtype=np.int32)
    dp = D[0].copy()
    back[0] = np.arange(m)
    idx = np.arange(m, dtype=np.int32)
    for i in range(1, n):
        val = dp - lam * j
        cm = np.minimum.accumulate(val)
        arg = np.maximum.accumulate(np.where(val <= cm, idx, 0))
        back[i] = arg
        dp = D[i] + cm + lam * j
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(dp))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    dist = D[np.arange(n), pos].astype(float)
    return pos, route[pos], dist
