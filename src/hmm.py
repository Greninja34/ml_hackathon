"""HMM + Viterbi route reconstruction on top of the classifier's candidate probabilities.

Hidden state   : which candidate arc the vehicle is on at GPS point k (the TOP_N best-scored candidates of that point).
Emission       : log of the classifier probability of that candidate.
Transition     : from arc a (point k-1) to arc b (point k), using the road network:
    route distance d(a,b)   same arc  -> (t_b - t_a) * len          (negative = moving backwards)
                            otherwise -> (1 - t_a)*len_a + dist(end_a -> start_b) + t_b*len_b   if start_b is reachable
                                         within `reach_limit` metres, else "unreachable"
    cost = min(|d - delta| / beta, break_pen)   where delta = straight-line distance between the two GPS points
    unreachable                       -> cost = break_pen
    exact reverse of a (U-turn)       -> cost += uturn_pen
    time gap dt > gap_s               -> the vehicle may have driven far beyond `reach_limit`, so being unreachable is
                                         expected: unreachable pairs cost gap_pen instead of break_pen and no transition
                                         costs more than gap_pen; reachable, consistent pairs are still preferred
    log T = - w_t * cost
Viterbi        : best chain maximises  sum_k [ log p_k(state_k) + log T_k(state_{k-1}, state_k) ].
Points with no candidate are skipped by the chain; `fill_missing` gives them the arc of the nearest decoded point in time.
`stitch_route` turns the per-point arcs into one arc sequence, filling non-adjacent consecutive arcs with the shortest
network path (bounded), so the returned route is drivable wherever the network allows it.
"""
import heapq
from dataclasses import dataclass
import numpy as np

REACH_LIMIT_M = 400.0
TOP_N = 8


@dataclass
class HMMParams:
    w_t: float = 1.0          # weight of transition cost relative to emission
    beta: float = 30.0        # metres of route-vs-straight-line mismatch that cost 1 nat
    break_pen: float = 8.0    # cost of an unreachable / wildly inconsistent transition (nats before weighting)
    uturn_pen: float = 4.0    # extra cost of jumping to the exact reverse arc
    gap_s: float = 20.0       # time gap (s) above which unreachability is no longer treated as a break
    gap_pen: float = 2.0      # cap on any transition cost after such a gap (and the cost of an unreachable pair)


class TripGraph:
    """Directed arc graph of one trip with cached bounded shortest paths between nodes."""

    def __init__(self, arcs, node_xy, limit=REACH_LIMIT_M):
        self.u, self.v = arcs[:, 0].astype(np.int64), arcs[:, 1].astype(np.int64)
        self.len = np.linalg.norm(node_xy[self.v] - node_xy[self.u], axis=1)
        n = len(node_xy)
        order = np.argsort(self.u, kind="stable")
        self.indptr = np.r_[0, np.cumsum(np.bincount(self.u, minlength=n))].tolist()
        self.nbr = self.v[order].tolist()
        self.w = self.len[order].tolist()
        self.aid = order.tolist()                       # arc id of every adjacency entry
        self.u_list = self.u.tolist()
        self.limit = limit
        self._cache = {}

    def reach(self, s):
        """dict node -> shortest distance from node s, only for nodes within `limit` metres."""
        r = self._cache.get(s)
        if r is not None:
            return r
        dist = {s: 0.0}; h = [(0.0, s)]
        indptr, nbr, w, limit = self.indptr, self.nbr, self.w, self.limit
        while h:
            d, x = heapq.heappop(h)
            if d > dist[x]:
                continue
            for k in range(indptr[x], indptr[x + 1]):
                y, nd = nbr[k], d + w[k]
                if nd <= limit and nd < dist.get(y, 1e18):
                    dist[y] = nd; heapq.heappush(h, (nd, y))
        self._cache[s] = dist
        return dist

    def path(self, s, t, limit=1500.0):
        """Arc ids of the shortest path node s -> node t (bounded by `limit` metres), or None if not reachable."""
        if s == t:
            return []
        dist, prev, h = {s: 0.0}, {}, [(0.0, s)]
        indptr, nbr, w, aid = self.indptr, self.nbr, self.w, self.aid
        while h:
            d, x = heapq.heappop(h)
            if x == t:
                break
            if d > dist[x]:
                continue
            for k in range(indptr[x], indptr[x + 1]):
                y, nd = nbr[k], d + w[k]
                if nd <= limit and nd < dist.get(y, 1e18):
                    dist[y] = nd; prev[y] = aid[k]; heapq.heappush(h, (nd, y))
        if t not in prev:
            return None
        out, x = [], t
        while x != s:
            a = prev[x]; out.append(a); x = self.u_list[a]
        return out[::-1]


def build_steps(pt_idx, arc_id, prob, t_frac, P, t, G, top_n=TOP_N):
    """Turn the candidate rows of ONE trip into a chain of steps with precomputed transition geometry.
    Returns list of dicts: pt, arc, lp (log prob), tf, and for k>0: delta, dt, R (route distance, nan = unreachable), rev (U-turn)."""
    order = np.argsort(pt_idx, kind="stable")
    pt_idx, arc_id, prob, t_frac = pt_idx[order], arc_id[order], prob[order], t_frac[order]
    cuts = np.flatnonzero(np.diff(pt_idx)) + 1
    steps, prev = [], None
    for sl in np.split(np.arange(len(pt_idx)), cuts):
        if len(sl) == 0:
            continue
        top = sl[np.argsort(-prob[sl])[:top_n]]
        s = dict(pt=int(pt_idx[top[0]]), arc=arc_id[top].astype(np.int64),
                 lp=np.log(np.clip(prob[top].astype(np.float64), 1e-6, 1.0)), tf=t_frac[top].astype(np.float64))
        if prev is not None:
            pa, ptf = prev["arc"], prev["tf"]
            a, tf = s["arc"], s["tf"]
            la, lb = G.len[pa], G.len[a]
            R = np.full((len(pa), len(a)), np.nan)
            for i, aa in enumerate(pa):
                D = G.reach(int(G.v[aa])); da = (1.0 - ptf[i]) * la[i]
                for j, bb in enumerate(a):
                    if aa == bb:
                        R[i, j] = (tf[j] - ptf[i]) * la[i]
                    else:
                        r = D.get(int(G.u[bb]))
                        if r is not None:
                            R[i, j] = da + r + tf[j] * lb[j]
            s["R"] = R
            s["rev"] = (G.u[a][None, :] == G.v[pa][:, None]) & (G.v[a][None, :] == G.u[pa][:, None]) & (pa[:, None] != a[None, :])
            s["delta"] = float(np.linalg.norm(P[s["pt"]] - P[prev["pt"]]))
            s["dt"] = float(t[s["pt"]] - t[prev["pt"]])
        steps.append(s); prev = s
    return steps


def decode(steps, prm: HMMParams):
    """Viterbi. Returns, per step, the index (into that step's candidate list) of the chosen arc."""
    if not steps:
        return np.array([], dtype=int)
    score = steps[0]["lp"].copy()
    back = [None]
    for s in steps[1:]:
        cap = prm.gap_pen if s["dt"] > prm.gap_s else prm.break_pen
        cost = np.where(np.isnan(s["R"]), cap, np.minimum(np.abs(s["R"] - s["delta"]) / prm.beta, cap))
        cost = cost + prm.uturn_pen * s["rev"]
        M = score[:, None] - prm.w_t * cost
        bp = M.argmax(0)
        score = s["lp"] + M[bp, np.arange(M.shape[1])]
        back.append(bp)
    idx = np.empty(len(steps), dtype=int)
    idx[-1] = int(score.argmax())
    for k in range(len(steps) - 1, 0, -1):
        idx[k - 1] = back[k][idx[k]]
    return idx


def classifier_only(steps):
    return np.array([int(s["lp"].argmax()) for s in steps], dtype=int)


def chosen_arcs(steps, idx):
    """Arc id chosen at every step."""
    return np.array([s["arc"][i] for s, i in zip(steps, idx)], dtype=np.int64)


def fill_missing(n_points, step_pts, arcs):
    """One arc per GPS point: points without candidates take the arc of the nearest decoded point (by index;
    ties go to the earlier point). Returns (arc per point, bool mask of filled points)."""
    step_pts = np.asarray(step_pts, dtype=np.int64); arcs = np.asarray(arcs, dtype=np.int64)
    if len(step_pts) == 0:
        return np.full(n_points, -1, dtype=np.int64), np.ones(n_points, bool)
    allp = np.arange(n_points)
    right = np.clip(np.searchsorted(step_pts, allp), 0, len(step_pts) - 1)
    left = np.clip(right - 1, 0, len(step_pts) - 1)
    j = np.where(np.abs(allp - step_pts[left]) <= np.abs(step_pts[right] - allp), left, right)
    out = arcs[j]
    filled = np.ones(n_points, bool); filled[step_pts] = False
    return out, filled


def stitch_route(arcs, G, limit=1500.0):
    """Collapse repeated arcs and join non-adjacent consecutive arcs by the shortest network path.
    Returns (route arc ids, number of joins that could not be bridged within `limit`)."""
    route, breaks = [], 0
    for a in arcs:
        a = int(a)
        if route and a == route[-1]:
            continue
        if route and G.v[route[-1]] != G.u[a]:
            p = G.path(int(G.v[route[-1]]), int(G.u[a]), limit)
            if p is None:
                breaks += 1
            else:
                route.extend(p)
        route.append(a)
    return np.array(route, dtype=np.int64), breaks
