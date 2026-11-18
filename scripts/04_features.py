"""Phase 3: build the tabular candidate-feature dataset (one row = one GPS point x one candidate arc).

Outputs
  data/processed/features_train.parquet, features_student_test.parquet   (git-ignored, big)
  data/processed/features_sample.csv                                      (first 2000 rows, for inspection)
  results/features/feature_stats.csv, feature_separation.png, dataset_summary.md
Columns: trip_id, partition, pt_idx, arc_id, label, point_has_positive, trip_reliable + FEATURES
  label              1 if arc_id == reference arc of that GPS point
  point_has_positive 1 if the reference arc is inside the candidate set (else no positive exists)
  trip_reliable      1 if the trip's median GPS->reference distance <= 15 m
Run: python scripts/04_features.py
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CANDIDATE_RADIUS_M as R
from src.data_io import PARTITIONS, list_trips, load_trip, project
from src.candidates import CandidateIndex
from src.features import FEATURES, build_features
from src.plotstyle import COLORS, apply, finish

apply()
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"
PROC = ROOT / "data" / "processed"; PROC.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "results" / "features"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    frames = {p: [] for p in PARTITIONS}
    for part in PARTITIONS:
        for tid in list_trips(part):
            t0 = time.time()
            tr = load_trip(part, tid)
            trk, nd, arcs = tr.track, tr.nodes, tr.arcs
            lon0, lat0 = trk.lon.mean(), trk.lat.mean()
            P = project(trk.lon, trk.lat, lon0, lat0)
            N = project(nd.lon, nd.lat, lon0, lat0)
            A, B = N[arcs[:, 0]], N[arcs[:, 1]]
            df = build_features(P, trk.t.to_numpy(), A, B, arcs, len(nd), R)

            L = np.load(CACHE / f"labels_{tid}.npz")          # labels only attached here, never inside features.py
            ref_arc, ref_dist = L["ref_arc"], L["ref_dist"]
            df.insert(0, "trip_id", np.int32(int(tid)))
            df.insert(1, "partition", part)
            df.insert(4, "label", (df.arc_id.to_numpy() == ref_arc[df.pt_idx.to_numpy()]).astype(np.int8))
            df.insert(5, "point_has_positive", (ref_dist[df.pt_idx.to_numpy()] <= R).astype(np.int8))
            df.insert(6, "trip_reliable", np.int8(np.median(ref_dist) <= 15))
            frames[part].append(df)
            print(f"{part} {tid}: {len(df):7d} rows, {df.pt_idx.nunique():6d}/{len(P):6d} points have candidates, "
                  f"pos rate {df.label.mean():.3f} ({time.time()-t0:.1f}s)", flush=True)

    allf = []
    for part in PARTITIONS:
        d = pd.concat(frames[part], ignore_index=True)
        d.to_parquet(PROC / f"features_{part}.parquet", index=False)
        allf.append(d)
    full = pd.concat(allf, ignore_index=True)
    full.head(2000).to_csv(PROC / "features_sample.csv", index=False)

    # ---- dataset summary + feature sanity stats ----
    rel = full[(full.trip_reliable == 1) & (full.point_has_positive == 1)]
    stats = rel.groupby("label")[FEATURES].agg(["mean", "median"]).T
    stats.to_csv(OUT / "feature_stats.csv")
    tr = full[full.partition == "train"]; st = full[full.partition == "student_test"]
    md = f"""# Phase 3 dataset summary

| | train | student_test |
|---|---|---|
| trips | {tr.trip_id.nunique()} | {st.trip_id.nunique()} |
| candidate rows | {len(tr):,} | {len(st):,} |
| GPS points with candidates | {tr.groupby('trip_id').pt_idx.nunique().sum():,} | {st.groupby('trip_id').pt_idx.nunique().sum():,} |
| positive rate (all rows) | {tr.label.mean():.4f} | {st.label.mean():.4f} |
| rows usable for training (reliable trip, point has positive) | {int(((tr.trip_reliable==1)&(tr.point_has_positive==1)).sum()):,} | n/a |
| NaN values | {int(full[FEATURES].isna().sum().sum())} | |
| features | {len(FEATURES)} | |
"""
    (OUT / "dataset_summary.md").write_text(md); print(md)

    # ---- do the features separate right from wrong arcs? (train, reliable trips) ----
    show = [("dist", "distance to arc (m)"), ("d_mean_win", "mean distance over +-5 points (m)"),
            ("cos_heading", "cos(heading vs arc direction)"), ("ds_span", "movement along arc direction (m)")]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    base = rel[rel.partition == "train"]
    for a, (col, name) in zip(ax.ravel(), show):
        lo, hi = np.percentile(base[col], [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        a.hist(base.loc[base.label == 0, col], bins=bins, alpha=.65, density=True, color=COLORS["predicted"], label="wrong arc (label 0)")
        a.hist(base.loc[base.label == 1, col], bins=bins, alpha=.65, density=True, color=COLORS["reference"], label="reference arc (label 1)")
        a.set_title(name); a.set_xlabel(col); a.set_ylabel("density"); a.legend(fontsize=8)
    finish(fig, OUT / "feature_separation.png", "Partition: train, reliable trips, points whose reference arc is a candidate")


if __name__ == "__main__":
    main()
