"""Phase 2: label alignment + candidate-radius analysis over train (80) + student_test (10) trips.

For every trip:
  1. project to a local metric plane
  2. align each GPS point to ONE reference arc (src/labeling.py)
  3. query all arcs within R_MAX of every point (src/candidates.py)
  4. record, for several radii R: candidate miss rate (reference arc not within R) and mean candidates / point
Outputs in results/candidates/ : per_trip_stats.csv, radius_sweep.csv, radius_sweep.png, summary.md
Aligned labels are cached in results/cache/ (git-ignored) for the feature step.
Run: python scripts/03_candidates.py
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_io import PARTITIONS, list_trips, load_trip, project
from src.labeling import align_to_route
from src.candidates import CandidateIndex
from src.plotstyle import COLORS, apply, finish

apply()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "candidates"
CACHE = ROOT / "results" / "cache"
OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)

RADII = [10, 15, 20, 25, 30, 40, 50, 75, 100]
R_MAX = max(RADII)


def main():
    trip_rows, sweep_rows = [], []
    for part in PARTITIONS:
        for tid in list_trips(part):
            t0 = time.time()
            tr = load_trip(part, tid)
            trk, nd, arcs, route = tr.track, tr.nodes, tr.arcs, tr.route
            lon0, lat0 = trk.lon.mean(), trk.lat.mean()
            P = project(trk.lon, trk.lat, lon0, lat0)
            N = project(nd.lon, nd.lat, lon0, lat0)
            A, B = N[arcs[:, 0]], N[arcs[:, 1]]

            pos, ref_arc, ref_dist = align_to_route(P, A, B, route)
            idx = CandidateIndex(A, B)
            pi, arc, d, tfrac = idx.query(P, R_MAX)
            np.savez_compressed(CACHE / f"labels_{tid}.npz", pos=pos, ref_arc=ref_arc, ref_dist=ref_dist)

            n = len(P)
            row = dict(partition=part, trip_id=tid, n_points=n, n_arcs=len(arcs),
                       align_dist_median=float(np.median(ref_dist)), align_dist_p95=float(np.percentile(ref_dist, 95)),
                       route_pos_monotone=bool(np.all(np.diff(pos) >= 0)),
                       route_arcs_visited_frac=float(len(np.unique(pos)) / len(route)))
            counts_per_pt = np.bincount(pi, minlength=n)
            for R in RADII:
                miss = float((ref_dist > R).mean())
                cands = float((d <= R).sum() / n)
                row[f"miss_{R}"] = miss; row[f"cands_{R}"] = cands
                sweep_rows.append(dict(partition=part, trip_id=tid, radius=R, miss=miss, cands_per_pt=cands,
                                       n_points=n, n_missed=int((ref_dist > R).sum())))
            # how often is the reverse twin of the reference arc also a candidate? (direction trap)
            trip_rows.append(row)
            print(f"{part} {tid} n={n:6d} arcs={len(arcs):6d} align_med={row['align_dist_median']:5.1f} "
                  f"miss@30={row['miss_30']:.3f} miss@50={row['miss_50']:.3f} cands@50={row['cands_50']:.1f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    tdf = pd.DataFrame(trip_rows); tdf.to_csv(OUT / "per_trip_stats.csv", index=False)
    sdf = pd.DataFrame(sweep_rows); sdf.to_csv(OUT / "radius_sweep.csv", index=False)

    # "reliable" trips: the reference route actually follows the GPS (median aligned distance <= 15 m)
    reliable = set(tdf.loc[tdf.align_dist_median <= 15, "trip_id"])
    sdf["reliable"] = sdf.trip_id.isin(reliable)

    def agg(df):
        g = df.groupby("radius").apply(lambda x: pd.Series({
            "miss_rate": x.n_missed.sum() / x.n_points.sum(),
            "cands_per_pt": (x.cands_per_pt * x.n_points).sum() / x.n_points.sum()}), include_groups=False)
        return g
    all_g, rel_g = agg(sdf), agg(sdf[sdf.reliable])
    comp = pd.DataFrame({"miss_all": all_g.miss_rate, "miss_reliable": rel_g.miss_rate,
                         "cands_all": all_g.cands_per_pt, "cands_reliable": rel_g.cands_per_pt})
    comp.to_csv(OUT / "radius_summary.csv")
    print(comp.round(4))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(comp.index, comp.miss_all * 100, "-o", color=COLORS["predicted"], label=f"all {len(tdf)} trips")
    ax[0].plot(comp.index, comp.miss_reliable * 100, "-o", color=COLORS["reference"],
               label=f"reliable trips only ({len(reliable)})")
    ax[0].set_yscale("log"); ax[0].set_xlabel("candidate radius R (m)"); ax[0].set_ylabel("candidate miss rate (%)")
    ax[0].set_title("Reference arc missing from candidate set"); ax[0].legend()
    ax[1].plot(comp.index, comp.cands_all, "-o", color=COLORS["train"], label="all trips")
    ax[1].plot(comp.index, comp.cands_reliable, "-o", color=COLORS["student_test"], label="reliable trips")
    ax[1].set_yscale("log"); ax[1].set_xlabel("candidate radius R (m)"); ax[1].set_ylabel("mean candidates per GPS point")
    ax[1].set_title("Cost: candidates to score"); ax[1].legend()
    finish(fig, OUT / "radius_sweep.png", "Partitions: train 80 + student_test 10. Reliable = median GPS-to-reference distance <= 15 m")

    bad = tdf[tdf.align_dist_median > 15][["partition", "trip_id", "align_dist_median", "miss_50", "miss_100"]]
    md = ["# Phase 2 summary", "", f"Trips: {len(tdf)} ({len(reliable)} reliable, {len(tdf)-len(reliable)} flagged)", "",
          comp.round(4).to_markdown(), "", "## Flagged trips (median aligned distance > 15 m)", "",
          bad.round(3).to_markdown(index=False)]
    (OUT / "summary.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
