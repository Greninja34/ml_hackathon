"""Phase 6: HMM/Viterbi route reconstruction. Tune decoder settings, compare classifier-only vs classifier+HMM.

Protocol
  * probabilities: out-of-fold for the 78 reliable train trips, final tuned model for the 10 student_test trips
  * decoder settings are tuned with Optuna (TPE) on a random subset of TRAIN trips only (out-of-fold probabilities)
  * final comparison on all train trips (OOF) and on student_test (never used to tune the decoder here)
Metrics (all per GPS point unless stated)
  top1_cond    : top-1 accuracy over points whose reference arc is among the candidates (comparable with Phases 4-5)
  top1_e2e     : end-to-end accuracy over ALL points (candidate misses and points without candidates count as wrong)
  continuity   : share of consecutive chosen arcs that are identical or directly connected
  route_f1     : F1 between the set of arcs of the predicted route and the set of reference arcs (sequence level)
Outputs: results/hmm/*, artifacts/hmm_config.json
Run: python scripts/10_hmm.py [n_tuning_trials]
"""
import sys, json, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RANDOM_SEED as SEED, CANDIDATE_RADIUS_M
from src.data_io import list_trips, load_trip, project
from src.hmm import HMMParams, TripGraph, build_steps, classifier_only, decode, TOP_N, REACH_LIMIT_M
from src.plotstyle import COLORS, apply, finish
from src.viz import build_trip_map

apply()
ROOT = Path(__file__).resolve().parents[1]
PROC, CACHE = ROOT / "data" / "processed", ROOT / "results" / "cache"
OUT = ROOT / "results" / "hmm"; OUT.mkdir(parents=True, exist_ok=True)
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def prepare(partition, probs):
    """Build decoder inputs for every trip of a partition."""
    data = {}
    for tid in list_trips(partition):
        tn = int(tid)
        g = probs[probs.trip_id == tn]
        if g.empty:
            continue
        tr = load_trip(partition, tid)
        lon0, lat0 = tr.track.lon.mean(), tr.track.lat.mean()
        P = project(tr.track.lon, tr.track.lat, lon0, lat0); N = project(tr.nodes.lon, tr.nodes.lat, lon0, lat0)
        G = TripGraph(tr.arcs, N)
        t0 = time.time()
        steps = build_steps(g.pt_idx.to_numpy(), g.arc_id.to_numpy(), g.prob.to_numpy(), g.t_frac.to_numpy(), P, tr.track.t.to_numpy(), G)
        L = np.load(CACHE / f"labels_{tid}.npz")
        data[tn] = dict(steps=steps, G=G, n_points=len(P), ref_arc=L["ref_arc"], ref_dist=L["ref_dist"], tr=tr,
                        route_set=set(tr.route.tolist()))
        print(f"  {partition} {tid}: {len(steps):6d} steps built in {time.time()-t0:5.1f}s", flush=True)
    return data


def score(d, idx):
    """Metrics of one trip for chosen candidate indices `idx` (one per step)."""
    steps, G = d["steps"], d["G"]
    chosen = np.array([s["arc"][i] for s, i in zip(steps, idx)], dtype=np.int64)
    pts = np.array([s["pt"] for s in steps])
    ref = d["ref_arc"][pts]; has = d["ref_dist"][pts] <= CANDIDATE_RADIUS_M
    correct = chosen == ref
    same = chosen[1:] == chosen[:-1]; conn = G.v[chosen[:-1]] == G.u[chosen[1:]]
    ref_set = set(np.unique(d["ref_arc"]).tolist()); pred_set = set(np.unique(chosen).tolist())
    inter = len(ref_set & pred_set)
    return dict(n_points=d["n_points"], n_steps=len(steps), n_cond=int(has.sum()), correct_cond=int((correct & has).sum()),
                correct_all=int(correct.sum()), cont_ok=int((same | conn).sum()), cont_tot=max(len(chosen) - 1, 0),
                f1_num=2 * inter, f1_den=len(ref_set) + len(pred_set)), chosen


def evaluate(data, decoder):
    rows = []
    for tn, d in data.items():
        m, _ = score(d, decoder(d))
        rows.append(dict(trip_id=tn, **m))
    df = pd.DataFrame(rows)
    return df, dict(top1_cond=df.correct_cond.sum() / df.n_cond.sum(), top1_e2e=df.correct_all.sum() / df.n_points.sum(),
                    continuity=df.cont_ok.sum() / df.cont_tot.sum(), route_f1=float((df.f1_num / df.f1_den).mean()))


def main():
    t0 = time.time()
    tr_probs = pd.read_parquet(PROC / "probs_train_oof.parquet"); st_probs = pd.read_parquet(PROC / "probs_student_test.parquet")
    print("building decoder inputs ...", flush=True)
    train = prepare("train", tr_probs); test = prepare("student_test", st_probs)
    print(f"built in {(time.time()-t0)/60:.1f} min: {len(train)} train trips, {len(test)} student_test trips", flush=True)

    clf = lambda d: classifier_only(d["steps"])
    hmm_default = lambda d: decode(d["steps"], HMMParams())

    # ---- tune decoder on a random subset of train trips (out-of-fold probabilities) ----
    rng = random.Random(SEED)
    tune_ids = rng.sample(sorted(train), min(30, len(train)))
    tune_set = {k: train[k] for k in tune_ids}
    base_tune = evaluate(tune_set, clf)[1]["top1_cond"]
    print(f"tuning on {len(tune_set)} train trips; classifier-only top1_cond = {base_tune:.4f}", flush=True)

    def objective(trial):
        p = HMMParams(w_t=trial.suggest_float("w_t", 0.2, 10.0, log=True), beta=trial.suggest_float("beta", 5.0, 120.0, log=True),
                      break_pen=trial.suggest_float("break_pen", 2.0, 60.0), uturn_pen=trial.suggest_float("uturn_pen", 0.0, 20.0),
                      gap_s=trial.suggest_float("gap_s", 5.0, 180.0), gap_pen=trial.suggest_float("gap_pen", 0.0, 12.0))
        return evaluate(tune_set, lambda d: decode(d["steps"], p))[1]["top1_cond"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True))
    study.enqueue_trial(dict(w_t=1.0, beta=30.0, break_pen=8.0, uturn_pen=4.0, gap_s=20.0, gap_pen=2.0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS, timeout=12 * 60)
    study.trials_dataframe().to_csv(OUT / "decoder_trials.csv", index=False)
    best = HMMParams(**study.best_params)
    print(f"decoder tuning: {len(study.trials)} trials, best {study.best_value:.4f} vs default {study.trials[0].value:.4f}\n{study.best_params}", flush=True)
    json.dump(dict(params=study.best_params, top_n=TOP_N, reach_limit_m=REACH_LIMIT_M, candidate_radius_m=CANDIDATE_RADIUS_M,
                   tuned_on="30 random train trips, out-of-fold probabilities", n_trials=len(study.trials)),
              open(ROOT / "artifacts" / "hmm_config.json", "w"), indent=2)
    shutil_params = best.__dict__

    # ---- final comparison ----
    res, per_trip = [], {}
    for name, data in (("train (out-of-fold)", train), ("student_test", test)):
        c_df, c = evaluate(data, clf); d_df, h = evaluate(data, hmm_default); t_df, t = evaluate(data, lambda d: decode(d["steps"], best))
        for label, m in (("classifier only", c), ("classifier + HMM (default settings)", h), ("classifier + HMM (tuned)", t)):
            res.append(dict(partition=name, method=label, **m))
        per_trip[name] = (c_df, t_df)
        pt = c_df[["trip_id", "n_points", "n_cond"]].copy()
        pt["clf_top1"] = c_df.correct_cond / c_df.n_cond; pt["hmm_top1"] = t_df.correct_cond / t_df.n_cond
        pt["delta_pts"] = 100 * (pt.hmm_top1 - pt.clf_top1); pt["clf_cont"] = c_df.cont_ok / c_df.cont_tot; pt["hmm_cont"] = t_df.cont_ok / t_df.cont_tot
        pt.to_csv(OUT / f"per_trip_{'train' if name.startswith('train') else 'student_test'}.csv", index=False)
    R = pd.DataFrame(res); R.to_csv(OUT / "hmm_vs_classifier.csv", index=False)
    (OUT / "hmm_vs_classifier.md").write_text(R.round(4).to_markdown(index=False))
    print(R.round(4).to_string(index=False), flush=True)

    # ---- figures ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
    mets = [("top1_cond", "top-1 (true arc among candidates)"), ("top1_e2e", "top-1 end-to-end (all points)"),
            ("continuity", "route continuity"), ("route_f1", "route F1 (arc set)")]
    for a, part in zip(ax, ["train (out-of-fold)", "student_test"]):
        sub = R[R.partition == part]; w = 0.26
        for i, (meth, col) in enumerate(zip(["classifier only", "classifier + HMM (default settings)", "classifier + HMM (tuned)"],
                                            [COLORS["neutral"], COLORS["student_test"], COLORS["reference"]])):
            vals = [sub[sub.method == meth][m].iloc[0] * 100 for m, _ in mets]
            bars = a.bar(np.arange(4) + (i - 1) * w, vals, w, color=col, label=meth)
            for b, v in zip(bars, vals):
                a.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points")
        a.set_xticks(range(4)); a.set_xticklabels([n for _, n in mets], fontsize=8, rotation=12); a.set_ylim(60, 101); a.set_ylabel("%")
        a.set_title(part)
    ax[0].legend(fontsize=7, loc="lower left")
    finish(fig, OUT / "hmm_vs_classifier.png", "y-axis starts at 60%. Decoder settings tuned on 30 train trips (out-of-fold probabilities)")

    c_df, t_df = per_trip["student_test"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(c_df)); ca = c_df.correct_cond / c_df.n_cond * 100; ta = t_df.correct_cond / t_df.n_cond * 100
    ax.bar(x - .2, ca, .4, color=COLORS["neutral"], label="classifier only"); ax.bar(x + .2, ta, .4, color=COLORS["reference"], label="+ HMM (tuned)")
    ax.set_xticks(x); ax.set_xticklabels([f"{int(t):d}" for t in c_df.trip_id], rotation=45); ax.set_ylim(60, 101)
    ax.set_xlabel("student_test trip"); ax.set_ylabel("top-1 (%)"); ax.set_title("Per-trip effect of the HMM"); ax.legend()
    finish(fig, OUT / "per_trip_student_test.png", "Partition: student_test")

    # ---- example map: the student_test trip where the HMM helped most ----
    gain = (t_df.correct_cond / t_df.n_cond - c_df.correct_cond / c_df.n_cond)
    tn = int(c_df.trip_id.iloc[int(gain.values.argmax())]); d = test[tn]
    _, chosen = score(d, decode(d["steps"], best))
    seq = [chosen[0]] + [a for p, a in zip(chosen[:-1], chosen[1:]) if a != p]
    m = build_trip_map(d["tr"].track, d["tr"].nodes, d["tr"].arcs, route=d["tr"].route, pred=np.array(seq), title=f"Trip {tn:08d} (student_test)")
    m.save(OUT / f"map_student_test_{tn:08d}.html")
    print(f"map saved for trip {tn}; total {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
