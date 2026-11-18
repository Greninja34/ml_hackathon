"""Phase 7 check: run the exported end-to-end pipeline (src/pipeline.py, exactly what the API runs) on every student_test
trip from the RAW files, score it server-side (src/scoring.py), and compare with the Phase 6 offline results
(results/hmm/per_trip_student_test.csv). Small differences are expected only from CPU vs GPU XGBoost inference.

Outputs: results/pipeline/check_<partition>.csv, results/pipeline/check_run_<partition>.log
Run: python scripts/11_check_pipeline.py [partition]
     evaluator trips (kept OUTSIDE the repo, next to it):  PATHSNAP_DATA=.. python scripts/11_check_pipeline.py evaluator_test
     (a single final evaluation run: nothing in the model or decoder is changed after it)
"""
import sys, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_io import list_trips, load_trip
from src.pipeline import PathSnapModel
from src.scoring import score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pipeline"; OUT.mkdir(parents=True, exist_ok=True)
PART = sys.argv[1] if len(sys.argv) > 1 else "student_test"


class Tee:
    def __init__(self, path):
        self.f, self.o = open(path, "w"), sys.stdout
    def write(self, x):
        self.o.write(x); self.f.write(x)
    def flush(self):
        self.o.flush(); self.f.flush()


def main():
    sys.stdout = Tee(OUT / f"check_run_{PART}.log")
    t0 = time.time()
    model = PathSnapModel(device="cpu")
    print(f"model loaded in {time.time()-t0:.1f}s (CPU inference); partition {PART}", flush=True)
    rows = []
    for tid in list_trips(PART):
        trip = load_trip(PART, tid)
        pred = model.predict(trip.track, trip.nodes, trip.arcs)
        m = score(trip, pred, model.radius)
        rows.append(dict(trip_id=int(tid), n_points=pred.n_points, candidate_rows=pred.n_candidate_rows,
                         filled_points=int(pred.point_filled.sum()), unbridged=pred.unbridged, **m,
                         **{k: round(v, 3) for k, v in pred.timings.items()}))
        print(f"  {tid}: {pred.n_points:5d} pts | top1 {m['top1_cond']:.4f} (classifier only {m['top1_clf_cond']:.4f}) | "
              f"continuity {m['continuity']:.4f} | route F1 {m['route_f1']:.4f} | {pred.timings['total_s']:.2f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"check_{PART}.csv", index=False)

    tot = (df.top1_cond * df.n_points_scored).sum() / df.n_points_scored.sum()
    clf = (df.top1_clf_cond * df.n_points_scored).sum() / df.n_points_scored.sum()
    print(f"\npooled top1_cond: HMM {tot:.4f} | classifier only {clf:.4f} | mean route F1 {df.route_f1.mean():.4f} | "
          f"mean time per trip {df.total_s.mean():.2f}s (max {df.total_s.max():.2f}s)", flush=True)
    ref_path = ROOT / "results" / "hmm" / f"per_trip_{PART}.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path).set_index("trip_id")
        cmp = df.set_index("trip_id")[["top1_cond", "route_f1"]].join(ref[["hmm_top1", "hmm_route_f1_full"]])
        cmp["d_top1_pts"] = 100 * (cmp.top1_cond - cmp.hmm_top1)
        cmp["d_f1_pts"] = 100 * (cmp.route_f1 - cmp.hmm_route_f1_full)
        print("\nagainst Phase 6 offline results (difference in points):")
        print(cmp.round(4).to_string(), flush=True)
        print(f"max |difference|: top-1 {cmp.d_top1_pts.abs().max():.3f} pts, route F1 {cmp.d_f1_pts.abs().max():.3f} pts", flush=True)
    print(f"total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
