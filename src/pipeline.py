"""Phase 7: the complete prediction pipeline for ONE trip, used by the API and by the offline checks.

    raw trip (track, nodes, arcs)  ->  local metric projection  ->  candidate arcs within 75 m
    ->  18 relative features  ->  tuned XGBoost probability per candidate  ->  HMM / Viterbi decoding
    ->  fallback for points without candidates  ->  stitched route (shortest-path gap filling)

This module never reads a .route file or any label: it only sees GPS points and the road network. Scoring against the
reference route lives in src/scoring.py and is applied by the caller AFTER prediction (server-side only).
Artifacts: artifacts/xgb_tuned.json (classifier), model_meta.json (feature order, radius), hmm_config.json (decoder).
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xgboost as xgb

from .data_io import project
from .features import FEATURES, build_features
from .hmm import HMMParams, TripGraph, build_steps, chosen_arcs, classifier_only, decode, fill_missing, stitch_route

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


@dataclass
class Prediction:
    n_points: int
    point_arc: np.ndarray            # (n_points,) decoded arc id per GPS point (after fallback)
    point_filled: np.ndarray         # (n_points,) True where the point had no candidate and took a neighbour's arc
    point_arc_clf: np.ndarray        # (n_points,) classifier-only arc per point (for comparison; same fallback)
    point_t: np.ndarray              # (n_points,) position along the chosen arc (0..1), nan for filled points
    route: np.ndarray                # stitched arc sequence of the predicted route
    unbridged: int                   # joins that could not be bridged by a shortest path
    continuity: float                # share of consecutive decoded arcs that are identical or connected
    n_candidate_rows: int
    timings: dict = field(default_factory=dict)


class PathSnapModel:
    """Loads the exported artifacts once; `predict` runs the whole pipeline for one trip."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS, device: str = "cpu"):
        self.meta = json.load(open(artifacts_dir / "model_meta.json"))
        self.hmm_cfg = json.load(open(artifacts_dir / "hmm_config.json"))
        if self.meta["features"] != FEATURES:
            raise RuntimeError("feature order in model_meta.json does not match src/features.py")
        self.radius = float(self.meta["candidate_radius_m"])
        self.top_n = int(self.hmm_cfg["top_n"])
        self.stitch_limit = float(self.hmm_cfg.get("stitch_limit_m", 1500.0))
        self.params = HMMParams(**self.hmm_cfg["params"])
        self.booster = xgb.Booster()
        self.booster.load_model(str(artifacts_dir / "xgb_tuned.json"))
        self.booster.set_param({"device": device})

    def predict(self, track, nodes, arcs) -> Prediction:
        """track: DataFrame lon, lat, t; nodes: DataFrame lon, lat; arcs: (n_arcs, 2) int node ids."""
        tm = {}; t0 = time.perf_counter()
        lon0, lat0 = track.lon.mean(), track.lat.mean()
        P = project(track.lon, track.lat, lon0, lat0)
        N = project(nodes.lon, nodes.lat, lon0, lat0)
        A, B = N[arcs[:, 0]], N[arcs[:, 1]]
        t = track.t.to_numpy(dtype=float)
        n = len(P)

        df = build_features(P, t, A, B, arcs, len(nodes), self.radius)
        tm["candidates_features_s"] = time.perf_counter() - t0; t1 = time.perf_counter()

        prob = self.booster.predict(xgb.DMatrix(df[FEATURES].to_numpy(np.float32)))
        tm["classifier_s"] = time.perf_counter() - t1; t2 = time.perf_counter()

        G = TripGraph(arcs, N)
        steps = build_steps(df.pt_idx.to_numpy(), df.arc_id.to_numpy(), prob, df.t_frac.to_numpy(), P, t, G, self.top_n)
        idx = decode(steps, self.params)
        tm["hmm_viterbi_s"] = time.perf_counter() - t2; t3 = time.perf_counter()

        pts = np.array([s["pt"] for s in steps], dtype=np.int64)
        chosen = chosen_arcs(steps, idx)
        point_arc, filled = fill_missing(n, pts, chosen)
        point_arc_clf, _ = fill_missing(n, pts, chosen_arcs(steps, classifier_only(steps)))
        point_t = np.full(n, np.nan)
        point_t[pts] = [s["tf"][i] for s, i in zip(steps, idx)]
        route, unbridged = stitch_route(chosen, G, self.stitch_limit)
        same = chosen[1:] == chosen[:-1]; conn = G.v[chosen[:-1]] == G.u[chosen[1:]]
        continuity = float((same | conn).mean()) if len(chosen) > 1 else 1.0
        tm["stitch_s"] = time.perf_counter() - t3
        tm["total_s"] = time.perf_counter() - t0
        return Prediction(n_points=n, point_arc=point_arc, point_filled=filled, point_arc_clf=point_arc_clf, point_t=point_t,
                          route=route, unbridged=unbridged, continuity=continuity, n_candidate_rows=len(df), timings=tm)
