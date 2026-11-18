"""Feature engineering: one row per (GPS point, candidate arc).

Rule: ONLY relative geometry/kinematics. No arc ids, no absolute coordinates (every trip is in a
different city, and ids mean nothing across networks). This module never touches labels, so the
exact same code is used at training time and inside the API at inference time.

Feature groups
  position   : dist, t_frac, dist_minus_min, dist_rank, n_cands
  direction  : cos_heading, heading_ok, ds_span, has_reverse
  fit-to-trajectory : d_prev, d_next, d_mean_win, speed_ms
  road shape : arc_len, deg_out_end, deg_in_start
  time       : dt_prev, dt_next
"""
import numpy as np
import pandas as pd

from .candidates import CandidateIndex
from .data_io import point_segment_distance

K = 2          # neighbour offset (points) for heading / trajectory-fit features
W = 5          # window half-width for d_mean_win
MAX_SPAN_S = 10.0   # heading only trusted if the +-K window spans <= this many seconds
MIN_MOVE_M = 3.0    # ... and the vehicle moved at least this far (else heading is noise)

FEATURES = ["dist", "t_frac", "dist_minus_min", "dist_rank", "n_cands",
            "cos_heading", "heading_ok", "ds_span", "has_reverse",
            "d_prev", "d_next", "d_mean_win", "speed_ms",
            "arc_len", "deg_out_end", "deg_in_start", "dt_prev", "dt_next"]


def arc_static_features(arcs, n_nodes):
    """Per-arc features that depend only on the network: has_reverse, degrees."""
    u, v = arcs[:, 0].astype(np.int64), arcs[:, 1].astype(np.int64)
    fwd = u * n_nodes + v
    rev = v * n_nodes + u
    sorted_fwd = np.sort(fwd)
    pos = np.searchsorted(sorted_fwd, rev)
    pos = np.clip(pos, 0, len(sorted_fwd) - 1)
    has_rev = (sorted_fwd[pos] == rev).astype(np.int8)
    out_deg = np.bincount(u, minlength=n_nodes)
    in_deg = np.bincount(v, minlength=n_nodes)
    return has_rev, out_deg[v].astype(np.int16), in_deg[u].astype(np.int16)


def _shift(P, idx, k):
    return P[np.clip(idx + k, 0, len(P) - 1)]


def build_features(P, t, A, B, arcs, n_nodes, radius, index=None):
    """P: (n,2) projected GPS (m); t: (n,) seconds; A,B: projected arc endpoints; arcs: (n_arcs,2) node ids.
    Returns DataFrame with columns pt_idx, arc_id + FEATURES (float32)."""
    n = len(P)
    index = index or CandidateIndex(A, B)
    pi, arc, d, tfrac = index.query(P, radius)
    has_rev, deg_out, deg_in = arc_static_features(arcs, n_nodes)

    ab = B[arc] - A[arc]
    alen = np.linalg.norm(ab, axis=1)
    u = ab / np.maximum(alen, 1e-9)[:, None]

    # ---- point-level kinematics (computed once per point, then gathered) ----
    i = np.arange(n)
    ip, inx = np.clip(i - K, 0, n - 1), np.clip(i + K, 0, n - 1)
    span_t = t[inx] - t[ip]
    mv = P[inx] - P[ip]
    mv_len = np.linalg.norm(mv, axis=1)
    heading = mv / np.maximum(mv_len, 1e-9)[:, None]
    ok = (span_t > 0) & (span_t <= MAX_SPAN_S) & (mv_len >= MIN_MOVE_M)
    speed = np.where(span_t > 0, mv_len / np.maximum(span_t, 1e-9), 0.0)
    dt_prev = np.r_[0.0, np.diff(t)]
    dt_next = np.r_[np.diff(t), 0.0]

    heading_ok = ok[pi]
    cos_h = np.where(heading_ok, (heading[pi] * u).sum(1), 0.0)

    # ---- how well does THIS arc fit the neighbouring GPS points? ----
    qp, qn = _shift(P, pi, -K), _shift(P, pi, K)
    d_prev, _ = point_segment_distance(qp, A[arc], B[arc])
    d_next, _ = point_segment_distance(qn, A[arc], B[arc])
    s_prev = ((qp - A[arc]) * u).sum(1)
    s_next = ((qn - A[arc]) * u).sum(1)
    ds_span = s_next - s_prev                # >0: the vehicle moves along the arc's direction
    dw = np.zeros(len(pi))
    for k in range(-W, W + 1):
        dk, _ = point_segment_distance(_shift(P, pi, k), A[arc], B[arc])
        dw += dk
    d_mean_win = dw / (2 * W + 1)

    df = pd.DataFrame({"pt_idx": pi.astype(np.int32), "arc_id": arc.astype(np.int64),
                       "dist": d, "t_frac": tfrac})
    g = df.groupby("pt_idx")["dist"]
    df["dist_minus_min"] = d - g.transform("min").to_numpy()
    df["dist_rank"] = g.rank(method="first").to_numpy()
    df["n_cands"] = g.transform("size").to_numpy()
    df["cos_heading"], df["heading_ok"], df["ds_span"] = cos_h, heading_ok.astype(np.int8), ds_span
    df["has_reverse"] = has_rev[arc]
    df["d_prev"], df["d_next"], df["d_mean_win"] = d_prev, d_next, d_mean_win
    df["speed_ms"] = speed[pi]
    df["arc_len"], df["deg_out_end"], df["deg_in_start"] = alen, deg_out[arc], deg_in[arc]
    df["dt_prev"], df["dt_next"] = dt_prev[pi], dt_next[pi]
    cols = [c for c in df.columns if c not in ("pt_idx", "arc_id")]
    df[cols] = df[cols].astype(np.float32)
    return df
