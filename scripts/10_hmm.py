"""Phase 6: HMM/Viterbi route reconstruction. Tune decoder settings, compare classifier-only vs classifier+HMM.

Protocol
  * probabilities: out-of-fold for the 78 reliable train trips, final tuned model for the 10 student_test trips
  * decoder settings are tuned with Optuna (TPE) on ALL 78 train trips (out-of-fold probabilities), objective top1_cond
  * student_test is never used for tuning here, so its numbers are the held-out result; train numbers are in-sample
    for the decoder settings (but the probabilities are still out-of-fold)
Metrics (all per GPS point unless stated)
  top1_cond      : top-1 accuracy over points whose reference arc is among the candidates (comparable with Phases 4-5)
  top1_e2e       : accuracy over ALL points; points without candidates get the nearest decoded point's arc (fill_missing)
  continuity     : share of consecutive chosen arcs that are identical or directly connected
  route_f1       : F1 between the predicted arc set and the reference arcs matched to GPS points (sequence level)
  route_f1_full  : F1 between the stitched predicted route (shortest-path gap filling) and the full .route arc set
  unbridged      : joins in the stitched route that could not be bridged within 1.5 km of network distance
Outputs: results/hmm/* (including hmm_run.log, written by this script), artifacts/hmm_config.json
Run: python scripts/10_hmm.py [n_tuning_trials]
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RANDOM_SEED as SEED, CANDIDATE_RADIUS_M
from src.data_io import list_trips, load_trip, project
from src.hmm import (HMMParams, TripGraph, build_steps, classifier_only, decode, chosen_arcs, fill_missing, stitch_route,
                     TOP_N, REACH_LIMIT_M)
from src.plotstyle import COLORS, apply, finish
from src.viz import build_trip_map

apply()
ROOT = Path(__file__).resolve().parents[1]
PROC, CACHE = ROOT / "data" / "processed", ROOT / "results" / "cache"
OUT = ROOT / "results" / "hmm"; OUT.mkdir(parents=True, exist_ok=True)
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 150


class Tee:
    """Mirror stdout into results/hmm/hmm_run.log so the log always belongs to the run that wrote the results."""
    def __init__(self, path):
        self.f, self.o = open(path, "w"), sys.stdout
    def write(self, x):
        self.o.write(x); self.f.write(x)
    def flush(self):
        self.o.flush(); self.f.flush()


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
                        route_set=set(tr.route.tolist()), pts=np.array([s["pt"] for s in steps]))
        print(f"  {partition} {tid}: {len(steps):6d} steps built in {time.time()-t0:5.1f}s", flush=True)
    return data


def f1(a, b):
    return 2 * len(a & b) / max(len(a) + len(b), 1)


def score(d, idx, full=False):
    """Metrics of one trip for chosen candidate indices `idx` (one per step). `full` adds the end-to-end and stitched
    route metrics (slower: shortest paths for the stitching), used for the final comparison, not inside tuning."""
    steps, G, pts = d["steps"], d["G"], d["pts"]
    chosen = chosen_arcs(steps, idx)
    has = d["ref_dist"][pts] <= CANDIDATE_RADIUS_M
    correct = chosen == d["ref_arc"][pts]
    same = chosen[1:] == chosen[:-1]; conn = G.v[chosen[:-1]] == G.u[chosen[1:]]
    m = dict(n_points=d["n_points"], n_steps=len(steps), n_cond=int(has.sum()), correct_cond=int((correct & has).sum()),
             cont_ok=int((same | conn).sum()), cont_tot=max(len(chosen) - 1, 0),
             route_f1=f1(set(np.unique(d["ref_arc"]).tolist()), set(np.unique(chosen).tolist())))
    if full:
        per_pt, filled = fill_missing(d["n_points"], pts, chosen)
        route, breaks = stitch_route(chosen, G)
        m.update(n_filled=int(filled.sum()), correct_all=int((per_pt == d["ref_arc"]).sum()),
                 route_f1_full=f1(d["route_set"], set(route.tolist())), unbridged=breaks, route_len=len(route))
        return m, chosen, route
    return m, chosen, None


def evaluate(data, decoder, full=False):
    rows = []
    for tn, d in data.items():
        m, _, _ = score(d, decoder(d), full)
        rows.append(dict(trip_id=tn, **m))
    df = pd.DataFrame(rows)
    out = dict(top1_cond=df.correct_cond.sum() / df.n_cond.sum(), continuity=df.cont_ok.sum() / df.cont_tot.sum(),
               route_f1=float(df.route_f1.mean()))
    if full:
        out.update(top1_e2e=df.correct_all.sum() / df.n_points.sum(), route_f1_full=float(df.route_f1_full.mean()),
                   unbridged=int(df.unbridged.sum()), points_filled=int(df.n_filled.sum()))
    return df, out


def main():
    sys.stdout = Tee(OUT / "hmm_run.log")
    t0 = time.time()
    print(f"run started {time.strftime('%Y-%m-%d %H:%M:%S')}; n_trials={N_TRIALS}; seed={SEED}", flush=True)
    tr_probs = pd.read_parquet(PROC / "probs_train_oof.parquet"); st_probs = pd.read_parquet(PROC / "probs_student_test.parquet")
    print("building decoder inputs ...", flush=True)
    train = prepare("train", tr_probs); test = prepare("student_test", st_probs)
    print(f"built in {(time.time()-t0)/60:.1f} min: {len(train)} train trips, {len(test)} student_test trips", flush=True)
    for name, data in (("train", train), ("student_test", test)):
        dts = np.array([s["dt"] for d in data.values() for s in d["steps"][1:]])
        print(f"  {name}: {len(dts):,} transitions; dt > 20 s: {(dts > 20).sum()}, > 60 s: {(dts > 60).sum()}, "
              f"> 180 s: {(dts > 180).sum()}, max {dts.max():.0f} s", flush=True)

    clf = lambda d: classifier_only(d["steps"])
    hmm_default = lambda d: decode(d["steps"], HMMParams())

    # ---- tune decoder on all train trips (out-of-fold probabilities); student_test stays held out ----
    tune_set = train
    print(f"tuning on {len(tune_set)} train trips; classifier-only top1_cond = {evaluate(tune_set, clf)[1]['top1_cond']:.4f}", flush=True)
    space = dict(w_t=(0.2, 30.0, True), beta=(5.0, 200.0, True), break_pen=(2.0, 200.0, False), uturn_pen=(0.0, 40.0, False),
                 gap_s=(5.0, 600.0, False), gap_pen=(0.0, 30.0, False))
    print("search space: " + ", ".join(f"{k} {'log ' if lg else ''}[{lo}, {hi}]" for k, (lo, hi, lg) in space.items()), flush=True)

    def objective(trial):
        p = HMMParams(**{k: trial.suggest_float(k, lo, hi, log=lg) for k, (lo, hi, lg) in space.items()})
        v = evaluate(tune_set, lambda d: decode(d["steps"], p))[1]["top1_cond"]
        if trial.number % 10 == 0:
            print(f"  trial {trial.number:3d}: {v:.4f}  ({time.time()-t0:.0f}s elapsed)", flush=True)
        return v

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True))
    study.enqueue_trial(dict(w_t=1.0, beta=30.0, break_pen=8.0, uturn_pen=4.0, gap_s=20.0, gap_pen=2.0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t_tune = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    study.trials_dataframe().to_csv(OUT / "decoder_trials.csv", index=False)
    best = HMMParams(**study.best_params)
    print(f"decoder tuning: {len(study.trials)} trials in {(time.time()-t_tune)/60:.1f} min, best {study.best_value:.4f} "
          f"vs default {study.trials[0].value:.4f}\n{study.best_params}", flush=True)
    for k, (lo, hi, _) in space.items():
        v = study.best_params[k]
        if (v - lo) / (hi - lo) < 0.03 or (hi - v) / (hi - lo) < 0.03:
            print(f"  note: best {k}={v:.3g} is at the edge of its range [{lo}, {hi}]", flush=True)
    json.dump(dict(params=study.best_params, top_n=TOP_N, reach_limit_m=REACH_LIMIT_M, candidate_radius_m=CANDIDATE_RADIUS_M,
                   stitch_limit_m=1500.0, tuned_on=f"all {len(tune_set)} reliable train trips, out-of-fold probabilities",
                   objective="top1_cond", n_trials=len(study.trials), seed=SEED,
                   search_space={k: dict(low=lo, high=hi, log=lg) for k, (lo, hi, lg) in space.items()}),
              open(ROOT / "artifacts" / "hmm_config.json", "w"), indent=2)

    # ---- final comparison ----
    res, per_trip = [], {}
    tuned = lambda d: decode(d["steps"], best)
    for name, data in (("train (out-of-fold)", train), ("student_test", test)):
        t1 = time.time(); c_df, c = evaluate(data, clf, True)
        t2 = time.time(); d_df, h = evaluate(data, hmm_default, True)
        t3 = time.time(); t_df, t = evaluate(data, tuned, True); t4 = time.time()
        for label, m, sec in (("classifier only", c, t2 - t1), ("classifier + HMM (default settings)", h, t3 - t2),
                              ("classifier + HMM (tuned)", t, t4 - t3)):
            res.append(dict(partition=name, method=label, **m, eval_seconds=sec))
        per_trip[name] = (c_df, t_df)
        pt = c_df[["trip_id", "n_points", "n_cond"]].copy()
        pt["clf_top1"] = c_df.correct_cond / c_df.n_cond; pt["hmm_top1"] = t_df.correct_cond / t_df.n_cond
        pt["delta_pts"] = 100 * (pt.hmm_top1 - pt.clf_top1); pt["clf_cont"] = c_df.cont_ok / c_df.cont_tot; pt["hmm_cont"] = t_df.cont_ok / t_df.cont_tot
        pt["clf_route_f1_full"] = c_df.route_f1_full; pt["hmm_route_f1_full"] = t_df.route_f1_full; pt["hmm_unbridged"] = t_df.unbridged
        pt.to_csv(OUT / f"per_trip_{'train' if name.startswith('train') else 'student_test'}.csv", index=False)
        hurt = pt[pt.delta_pts < 0]
        print(f"  {name}: HMM lowered top-1 on {len(hurt)} of {len(pt)} trips"
              + (f" (worst: trip {int(hurt.loc[hurt.delta_pts.idxmin(), 'trip_id'])}, {hurt.delta_pts.min():.2f} pts)" if len(hurt) else ""), flush=True)
    R = pd.DataFrame(res); R.to_csv(OUT / "hmm_vs_classifier.csv", index=False)
    R = R.drop(columns="eval_seconds")
    (OUT / "hmm_vs_classifier.md").write_text(R.round(4).to_markdown(index=False))
    print(R.round(4).to_string(index=False), flush=True)

    # ---- figures ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
    mets = [("top1_cond", "top-1 (true arc among candidates)"), ("top1_e2e", "top-1 end-to-end (all points)"),
            ("continuity", "route continuity"), ("route_f1", "route F1 (matched arcs)"), ("route_f1_full", "route F1 (full .route)")]
    for a, part in zip(ax, ["train (out-of-fold)", "student_test"]):
        sub = R[R.partition == part]; w = 0.26
        for i, (meth, col) in enumerate(zip(["classifier only", "classifier + HMM (default settings)", "classifier + HMM (tuned)"],
                                            [COLORS["neutral"], COLORS["student_test"], COLORS["reference"]])):
            vals = [sub[sub.method == meth][m].iloc[0] * 100 for m, _ in mets]
            bars = a.bar(np.arange(len(mets)) + (i - 1) * w, vals, w, color=col, label=meth)
            for b, v in zip(bars, vals):
                a.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points")
        a.set_xticks(range(len(mets))); a.set_xticklabels([n for _, n in mets], fontsize=8, rotation=12); a.set_ylim(60, 101); a.set_ylabel("%")
        a.set_title(part)
    ax[0].legend(fontsize=7, loc="lower left")
    finish(fig, OUT / "hmm_vs_classifier.png", "y-axis starts at 60%. Decoder settings tuned on all 78 train trips (out-of-fold probabilities); student_test held out")

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
    _, _, route = score(d, tuned(d), full=True)
    m = build_trip_map(d["tr"].track, d["tr"].nodes, d["tr"].arcs, route=d["tr"].route, pred=route, title=f"Trip {tn:08d} (student_test)")
    m.save(OUT / f"map_student_test_{tn:08d}.html")
    print(f"map saved for trip {tn}; total {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
