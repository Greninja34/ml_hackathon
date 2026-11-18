"""Server-side scoring of a prediction against the reference route (.route).

Kept separate from src/pipeline.py on purpose: the prediction never sees the reference route. The API calls this only
after predicting, and returns metrics only, never the reference arcs themselves.

Metrics (same definitions as Phase 6, scripts/10_hmm.py)
  top1_cond     : share of GPS points (whose reference arc lies within the candidate radius) matched to the reference arc
  top1_all      : the same over ALL GPS points (candidate misses count as wrong)
  top1_clf_cond : top1_cond of the classifier alone, without the HMM (shows what the decoder adds)
  continuity    : share of consecutive decoded arcs that are identical or directly connected (from the prediction)
  route_f1      : F1 between the stitched predicted arc set and the reference .route arc set
"""
import numpy as np

from .data_io import project
from .labeling import align_to_route


def score(trip, pred, radius):
    trk, nd, arcs, route = trip.track, trip.nodes, trip.arcs, trip.route
    if route is None or len(route) == 0:
        return None
    lon0, lat0 = trk.lon.mean(), trk.lat.mean()
    P = project(trk.lon, trk.lat, lon0, lat0); N = project(nd.lon, nd.lat, lon0, lat0)
    _, ref_arc, ref_dist = align_to_route(P, N[arcs[:, 0]], N[arcs[:, 1]], route)
    has = ref_dist <= radius
    ok = pred.point_arc == ref_arc
    ok_clf = pred.point_arc_clf == ref_arc
    ref_set, pred_set = set(route.tolist()), set(pred.route.tolist())
    f1 = 2 * len(ref_set & pred_set) / max(len(ref_set) + len(pred_set), 1)
    return dict(top1_cond=float(ok[has].mean()) if has.any() else None,
                top1_all=float(ok.mean()),
                top1_clf_cond=float(ok_clf[has].mean()) if has.any() else None,
                continuity=pred.continuity,
                route_f1=float(f1),
                n_points_scored=int(has.sum()),
                reference_median_dist_m=float(np.median(ref_dist)))
