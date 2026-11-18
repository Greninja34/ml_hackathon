"""Phase 5: Bayesian hyperparameter optimisation of XGBoost (Optuna TPE) with group-aware CV.

Search protocol
  data        : 78 reliable train trips (same rows and the same GroupKFold(5) folds as scripts/05_models.py)
  objective   : maximise mean CV observation-level top-1 accuracy (the metric the decoder relies on);
                CV macro F1 and log loss are stored with every trial
  sampler     : TPE (Bayesian), seed 42, trial 0 = the untuned default configuration (so tuned vs untuned
                is measured under identical conditions)
  pruning     : MedianPruner on the running mean across folds
  hardware    : XGBoost on the GPU; each fold's quantised data is built once and reused by every trial
Afterwards: refit best params on all train rows, evaluate on student_test, compare with untuned.
Outputs: results/tuning/*, artifacts/xgb_tuned.json
Run: python scripts/06_tune_xgboost.py [n_trials] [timeout_minutes]
"""
import sys, time, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RANDOM_SEED as SEED
from src.data_io import DATA_ROOT
from src.evaluation import best_per_point, route_continuity, top1_accuracy
from src.features import FEATURES
from src.plotstyle import COLORS, apply, finish

warnings.filterwarnings("ignore")
apply()
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "tuning"; OUT.mkdir(parents=True, exist_ok=True)
(ROOT / "artifacts").mkdir(exist_ok=True)
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
TIMEOUT_S = 60 * float(sys.argv[2]) if len(sys.argv) > 2 else 35 * 60

DEFAULT = dict(n_estimators=300, learning_rate=0.1, max_depth=8, min_child_weight=1.0, subsample=1.0,
               colsample_bytree=1.0, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0)   # the untuned model of Phase 4
BASE = dict(objective="binary:logistic", tree_method="hist", device="cuda", eval_metric="logloss", seed=SEED, verbosity=0)


def to_params(p):
    return {**BASE, **{k: v for k, v in p.items() if k != "n_estimators"}}


def main():
    tr = pd.read_parquet(PROC / "features_train.parquet")
    st = pd.read_parquet(PROC / "features_student_test.parquet")
    tr = tr[(tr.trip_reliable == 1) & (tr.point_has_positive == 1)].reset_index(drop=True)
    st = st[st.point_has_positive == 1].reset_index(drop=True)
    X, y = tr[FEATURES].to_numpy(np.float32), tr.label.to_numpy()
    folds = list(GroupKFold(n_splits=5).split(tr, groups=tr.trip_id))
    print(f"train rows {len(tr):,}; building GPU data for {len(folds)} folds ...", flush=True)
    dtr, dva = [], []
    for a, b in folds:
        d = xgb.QuantileDMatrix(X[a], y[a]); dtr.append(d); dva.append(xgb.DMatrix(X[b], y[b]))
    trip_np, pt_np = tr.trip_id.to_numpy(), tr.pt_idx.to_numpy()

    def cv_metrics(i, prob):
        b = folds[i][1]
        idx = best_per_point(trip_np[b], pt_np[b], prob)
        pred = (prob >= 0.5).astype(int)
        return (top1_accuracy(y[b], idx), f1_score(y[b], pred, average="macro"),
                log_loss(y[b], np.clip(prob, 1e-6, 1 - 1e-6)))

    def objective(trial):
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 700),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
            gamma=trial.suggest_float("gamma", 0.0, 5.0))
        res, t0 = [], time.time()
        for i in range(len(folds)):
            bst = xgb.train(to_params(p), dtr[i], num_boost_round=p["n_estimators"])
            res.append(cv_metrics(i, bst.predict(dva[i])))
            trial.report(float(np.mean([r[0] for r in res])), i)
            if trial.should_prune():
                trial.set_user_attr("pruned_after_fold", i)
                raise optuna.TrialPruned()
        r = np.array(res)
        trial.set_user_attr("cv_macro_f1", float(r[:, 1].mean())); trial.set_user_attr("cv_log_loss", float(r[:, 2].mean()))
        trial.set_user_attr("cv_top1_std", float(r[:, 0].std())); trial.set_user_attr("cv_seconds", time.time() - t0)
        return float(r[:, 0].mean())

    db = OUT / "study.db"
    if db.exists():
        db.unlink()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1),
                                storage=f"sqlite:///{db}", study_name="xgb_top1")
    study.enqueue_trial(DEFAULT)                    # trial 0 = untuned baseline, evaluated identically
    optuna.logging.set_verbosity(optuna.logging.INFO)
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_S)
    search_s = time.time() - t0

    trials = study.trials_dataframe()
    trials.to_csv(OUT / "trials.csv", index=False)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = study.best_trial; base = study.trials[0]
    best_params = dict(best.params)
    json.dump({"best_params": best_params, "best_cv_top1": best.value, "trials_total": len(study.trials),
               "trials_complete": len(done), "trials_pruned": len(study.trials) - len(done),
               "search_seconds": search_s, "untuned_cv_top1": base.value, "seed": SEED,
               "search_space": {"n_estimators": "int 100-700", "learning_rate": "log 0.02-0.3", "max_depth": "int 3-12",
                                "min_child_weight": "log 1-30", "subsample": "0.5-1", "colsample_bytree": "0.5-1",
                                "reg_lambda": "log 0.01-30", "reg_alpha": "0-5", "gamma": "0-5"}},
              open(OUT / "best_params.json", "w"), indent=2)
    print(f"\n{len(study.trials)} trials ({len(done)} complete) in {search_s/60:.1f} min; best CV top-1 {best.value:.4f} "
          f"vs untuned {base.value:.4f}\n{best_params}", flush=True)

    # ---------- refit both configs on all train rows, evaluate on student_test ----------
    arcs_by_trip = {}
    for t in st.trip_id.unique():
        tid = f"{t:08d}"
        arcs_by_trip[int(t)] = pd.read_csv(DATA_ROOT / "student_test" / tid / f"{tid}.arcs", sep=r"\s+", header=None).to_numpy()
    dall = xgb.QuantileDMatrix(X, y); dst = xgb.DMatrix(st[FEATURES].to_numpy(np.float32))
    yst = st.label.to_numpy()
    rows = []
    for name, p, trial in (("XGBoost untuned", DEFAULT, base), ("XGBoost tuned (Bayesian)", best_params, best)):
        t0 = time.time(); bst = xgb.train(to_params(p), dall, num_boost_round=p["n_estimators"]); fit_s = time.time() - t0
        t0 = time.time(); prob = bst.predict(dst); inf_s = time.time() - t0
        idx = best_per_point(st.trip_id.to_numpy(), st.pt_idx.to_numpy(), prob)
        pred = (prob >= 0.5).astype(int)
        # CV numbers for this configuration (recomputed from the study record)
        cvf1 = trial.user_attrs.get("cv_macro_f1")
        rows.append(dict(model=name, cv_top1=trial.value, cv_macro_f1=cvf1, cv_log_loss=trial.user_attrs.get("cv_log_loss"),
                         test_top1=top1_accuracy(yst, idx), test_macro_f1=f1_score(yst, pred, average="macro"),
                         test_precision=precision_score(yst, pred, average="macro"), test_recall=recall_score(yst, pred, average="macro"),
                         test_log_loss=log_loss(yst, np.clip(prob, 1e-6, 1 - 1e-6)),
                         route_continuity=route_continuity(st.trip_id.to_numpy(), st.arc_id.to_numpy(), idx, arcs_by_trip),
                         train_time_s=fit_s, inference_ms_per_1k_rows=1000 * inf_s / (len(st) / 1000), params=json.dumps(p)))
        if "tuned" in name and "un" not in name:
            bst.save_model(str(ROOT / "artifacts" / "xgb_tuned.json"))
        print(f"  {name}: test top1={rows[-1]['test_top1']:.4f} F1={rows[-1]['test_macro_f1']:.4f} "
              f"logloss={rows[-1]['test_log_loss']:.4f} cont={rows[-1]['route_continuity']:.4f} fit={fit_s:.1f}s", flush=True)
    cmp = pd.DataFrame(rows); cmp.to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    (OUT / "tuned_vs_untuned.md").write_text(cmp.drop(columns="params").round(4).to_markdown(index=False))

    # ---------- figures ----------
    vals = [t.value for t in study.trials if t.value is not None]
    nums = [t.number for t in study.trials if t.value is not None]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    pr = [t.number for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    ax.scatter(nums, vals, color=COLORS["train"], s=26, label="completed trial", zorder=3)
    ax.plot(nums, np.maximum.accumulate(vals), color=COLORS["reference"], lw=2.2, label="best so far")
    ax.axhline(base.value, color=COLORS["student_test"], ls="--", lw=1.5, label=f"untuned default ({base.value:.4f})")
    ax.set_xlabel("trial"); ax.set_ylabel("CV top-1 accuracy"); ax.set_title(f"Bayesian optimisation history ({len(pr)} trials pruned early)"); ax.legend()
    finish(fig, OUT / "optimization_history.png", "Partition: train (78 trips), 5-fold CV grouped by trip")
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ks = list(imp)[::-1]; ax.barh(ks, [imp[k] for k in ks], color=COLORS["accent"])
        ax.set_title("Which hyperparameters mattered"); ax.set_xlabel("importance (fANOVA-style, Optuna)")
        finish(fig, OUT / "param_importance.png", "Computed from the completed trials")
    except Exception as e:
        print("importance skipped:", e)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, col, title in zip(ax, ["test_top1", "test_macro_f1", "test_log_loss"], ["Top-1 accuracy", "Macro F1", "Log loss (lower better)"]):
        b = a.bar(cmp.model.str.replace("XGBoost ", ""), cmp[col], color=[COLORS["neutral"], COLORS["reference"]])
        for r, v in zip(b, cmp[col]): a.annotate(f"{v:.4f}", (r.get_x() + r.get_width() / 2, v), ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")
        a.set_title(title)
        if col != "test_log_loss": a.set_ylim(min(cmp[col]) - 0.02, max(cmp[col]) + 0.01)
    finish(fig, OUT / "tuned_vs_untuned.png", "Partition: student_test (10 trips); y-axes are zoomed to show the difference")
    print(cmp.drop(columns="params").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
