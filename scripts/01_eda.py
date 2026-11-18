"""Phase 1: exploratory data analysis over train (80) + student_test (10) trips.

Outputs (results/eda/):
  inventory.csv          one row per trip: sizes, missing values, time gaps, network stats, noise
  summary.md             short human-readable summary used for the report
  *.png                  distribution plots
  trip_<id>_matplotlib.png, trip_<id>_map.html   example trip visualisations
Run:  .venv/bin/python scripts/01_eda.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_io import PARTITIONS, list_trips, load_trip, project, point_segment_distance
from src.plotstyle import COLORS, apply, finish

apply()

OUT = Path(__file__).resolve().parents[1] / "results" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def analyse(partition, tid):
    tr = load_trip(partition, tid)
    trk, nd, arcs, route = tr.track, tr.nodes, tr.arcs, tr.route
    row = dict(partition=partition, trip_id=tid, n_points=len(trk), n_nodes=len(nd),
               n_arcs=len(arcs), n_route_arcs=len(route))

    # ---- missing / invalid values ----
    row["track_nan"] = int(trk.isna().sum().sum())
    row["nodes_nan"] = int(nd.isna().sum().sum())
    row["lonlat_out_of_range"] = int(((trk.lon.abs() > 180) | (trk.lat.abs() > 90)).sum())
    row["dup_timestamps"] = int(trk.t.duplicated().sum())
    dt = np.diff(trk.t.to_numpy())
    row["non_monotonic_t"] = int((dt <= 0).sum())
    row["dt_median"], row["dt_max"] = float(np.median(dt)), float(dt.max())
    row["frac_dt_eq_1s"] = float((dt == 1).mean())
    row["n_gaps_gt5s"] = int((dt > 5).sum())
    row["n_gaps_gt60s"] = int((dt > 60).sum())
    row["arc_bad_node_ref"] = int(((arcs < 0) | (arcs >= len(nd))).any(1).sum())
    row["route_bad_arc_ref"] = int(((route < 0) | (route >= len(arcs))).sum())

    # ---- geometry in a local metric plane ----
    lon0, lat0 = trk.lon.mean(), trk.lat.mean()
    P = project(trk.lon, trk.lat, lon0, lat0)
    N = project(nd.lon, nd.lat, lon0, lat0)
    A, B = N[arcs[:, 0]], N[arcs[:, 1]]
    alen = np.linalg.norm(B - A, axis=1)
    row["track_extent_km"] = float(np.hypot(*np.ptp(P, axis=0)) / 1000)
    row["track_len_km"] = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() / 1000)
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    row["step_median_m"] = float(np.median(step))
    row["implied_speed_p99_ms"] = float(np.percentile(step / np.maximum(dt, 1), 99))
    row["arc_len_median_m"], row["zero_len_arcs"] = float(np.median(alen)), int((alen < 0.01).sum())

    # ---- network structure ----
    n = len(nd)
    g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    used = np.zeros(n, bool); used[arcs.ravel()] = True
    row["isolated_nodes"] = int((~used).sum())
    ncomp, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab[used]) if used.any() else np.array([0])
    row["weak_components"] = int((sizes > 0).sum())
    row["largest_component_frac"] = float(sizes.max() / used.sum())
    keys = set((arcs[:, 0].astype(np.int64) * n + arcs[:, 1]).tolist())
    rev = (arcs[:, 1].astype(np.int64) * n + arcs[:, 0])
    row["frac_arcs_with_reverse"] = float(np.mean([k in keys for k in rev.tolist()]))

    # ---- reference route checks ----
    row["route_is_connected"] = float((arcs[route[:-1], 1] == arcs[route[1:], 0]).mean()) if len(route) > 1 else 1.0
    row["route_len_km"] = float(alen[route].sum() / 1000)
    row["route_arcs_unique_frac"] = float(len(set(route.tolist())) / len(route))

    # ---- GPS noise: distance from each GPS point to the nearest reference-route arc ----
    mid = (A[route] + B[route]) / 2
    tree = cKDTree(mid)
    k = min(12, len(route))
    _, idx = tree.query(P, k=k)
    idx = idx.reshape(len(P), -1)
    best = np.full(len(P), np.inf)
    for j in range(idx.shape[1]):
        ra = route[idx[:, j]]
        d, _ = point_segment_distance(P, A[ra], B[ra])
        best = np.minimum(best, d)
    row["noise_median_m"], row["noise_p95_m"], row["noise_max_m"] = (
        float(np.median(best)), float(np.percentile(best, 95)), float(best.max()))
    return row, best


def main():
    rows, noise_all = [], []
    for part in PARTITIONS:
        for tid in list_trips(part):
            r, d = analyse(part, tid)
            rows.append(r); noise_all.append(d)
            print(part, tid, r["n_points"], r["n_arcs"], f"noise_med={r['noise_median_m']:.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "inventory.csv", index=False)

    # ---- plots (shared style) ----
    def hist(col, title, fname, log=False, xlabel=None, unit=""):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        lo, hi = df[col].min(), df[col].max()
        bins = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi + 1), 28) if log and lo > 0 else 28
        for part in PARTITIONS:
            v = df.loc[df.partition == part, col]
            ax.hist(v, bins=bins, alpha=.75, color=COLORS[part], edgecolor="white",
                    label=f"{part} (n={len(v)}, median {v.median():,.1f}{unit})")
        if log and lo > 0:
            ax.set_xscale("log")
        ax.set_title(title); ax.set_xlabel(xlabel or col); ax.set_ylabel("number of trips"); ax.legend()
        finish(fig, OUT / fname, "Source: results/eda/inventory.csv (train 80 trips + student_test 10 trips)")

    hist("n_points", "GPS points per trip", "hist_points_per_trip.png", log=True, xlabel="GPS points (log scale)")
    hist("n_arcs", "Road arcs in each trip's network", "hist_arcs_per_trip.png", log=True, xlabel="arcs (log scale)")
    hist("n_route_arcs", "Reference-route length", "hist_route_arcs.png", xlabel="arcs in reference route")
    hist("dt_max", "Largest time gap within each trip", "hist_max_gap.png", log=True, xlabel="seconds (log scale)", unit=" s")
    hist("noise_median_m", "Median distance from GPS point to reference route", "hist_noise_median.png",
         xlabel="metres", unit=" m")

    allnoise = np.concatenate(noise_all)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(np.clip(allnoise, 0, 60), bins=60, color=COLORS["train"], edgecolor="white")
    ax.set_yscale("log")
    for pc, c in ((50, COLORS["reference"]), (90, COLORS["student_test"]), (99, COLORS["predicted"])):
        v = np.percentile(allnoise, pc)
        ax.axvline(min(v, 60), color=c, ls="--", lw=1.6, label=f"p{pc} = {v:.1f} m")
    ax.set_title("GPS-to-reference-route distance, all points")
    ax.set_xlabel("metres (clipped at 60; the tail reaches %.0f m)" % allnoise.max()); ax.set_ylabel("GPS points (log scale)")
    ax.legend()
    finish(fig, OUT / "hist_noise_all_points.png", "Source: train 80 + student_test 10 trips")

    # ---- summary markdown ----
    n_tr = (df.partition == "train").sum(); n_st = (df.partition == "student_test").sum()
    q = lambda p: float(np.percentile(allnoise, p))
    md = f"""# EDA summary (partitions: train={n_tr} trips, student_test={n_st} trips)

| item | value |
|---|---|
| total GPS points | {int(df.n_points.sum()):,} |
| points / trip (min, median, max) | {df.n_points.min()}, {int(df.n_points.median())}, {df.n_points.max()} |
| arcs / trip (min, median, max) | {df.n_arcs.min():,}, {int(df.n_arcs.median()):,}, {df.n_arcs.max():,} |
| route arcs / trip (min, median, max) | {df.n_route_arcs.min()}, {int(df.n_route_arcs.median())}, {df.n_route_arcs.max()} |
| NaN in tracks / nodes | {int(df.track_nan.sum())} / {int(df.nodes_nan.sum())} |
| lon/lat out of range | {int(df.lonlat_out_of_range.sum())} |
| duplicate timestamps | {int(df.dup_timestamps.sum())} |
| non-monotonic timestamps | {int(df.non_monotonic_t.sum())} |
| fraction of steps exactly 1 s (mean over trips) | {df.frac_dt_eq_1s.mean():.3f} |
| trips with a gap > 60 s | {int((df.n_gaps_gt60s > 0).sum())} |
| largest gap (s) | {df.dt_max.max():.0f} |
| arcs with invalid node ref | {int(df.arc_bad_node_ref.sum())} |
| route arcs with invalid arc ref | {int(df.route_bad_arc_ref.sum())} |
| zero-length arcs | {int(df.zero_len_arcs.sum())} |
| isolated nodes (total / trips affected) | {int(df.isolated_nodes.sum()):,} / {int((df.isolated_nodes > 0).sum())} |
| weak components (median / max per trip) | {int(df.weak_components.median())} / {df.weak_components.max()} |
| largest-component fraction (min) | {df.largest_component_frac.min():.3f} |
| arcs with reverse twin (mean over trips) | {df.frac_arcs_with_reverse.mean():.3f} |
| reference route fully connected (min over trips) | {df.route_is_connected.min():.4f} |
| GPS->route distance m: p50 / p90 / p95 / p99 / max | {q(50):.1f} / {q(90):.1f} / {q(95):.1f} / {q(99):.1f} / {allnoise.max():.0f} |
| 99th-pct implied speed (m/s), max over trips | {df.implied_speed_p99_ms.max():.1f} |
"""
    (OUT / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
