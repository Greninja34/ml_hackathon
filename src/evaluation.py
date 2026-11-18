"""Point-level evaluation helpers shared by tuning, the HMM comparison and the API tests.

A prediction table has one row per (trip, GPS point, candidate arc) with a model score.
"""
import numpy as np


def best_per_point(trip, pt, prob):
    """Row index of the highest-scoring candidate of every GPS point, ordered by (trip, point)."""
    key = np.asarray(trip, dtype=np.int64) * 10_000_000 + np.asarray(pt, dtype=np.int64)
    order = np.lexsort((-np.asarray(prob), key))
    ks = key[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    return order[first]


def top1_accuracy(label, idx):
    """Observation-level top-1 accuracy: share of points whose top-scored candidate is the reference arc."""
    return float(np.asarray(label)[idx].mean())


def route_continuity(trip, arc, idx, arcs_by_trip):
    """Share of consecutive GPS points whose chosen arcs are identical or directly connected
    (end node of the first == start node of the second)."""
    trips, arcs_sel = np.asarray(trip)[idx], np.asarray(arc)[idx]
    ok = tot = 0
    for t in np.unique(trips):
        a = arcs_sel[trips == t]
        arcs = arcs_by_trip[int(t)]
        same = a[1:] == a[:-1]
        conn = arcs[a[:-1], 1] == arcs[a[1:], 0]
        ok += int((same | conn).sum()); tot += len(a) - 1
    return ok / max(tot, 1)
