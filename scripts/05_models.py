"""Phase 4: train and compare 7 models from different families on the candidate-feature table.

Protocol (Models and Training Guidelines, section 4):
  * fit + 5-fold GroupKFold CV (groups = trips) on the 80 TRAIN trips only (reliable trips, points whose
    reference arc is a candidate)
  * final refit on all of those rows, evaluation on the 10 STUDENT_TEST trips (development comparison)
  * evaluator data is never touched; scaler is fitted inside each pipeline on training rows only
Per model we record: CV macro-F1 / top-1 / log-loss (mean, std), train time, test top-1 / macro-F1 /
precision / recall / log-loss / route continuity / inference time, confusion matrix, config, seed.
Outputs: results/models/*.csv|json|png
Run: python scripts/05_models.py
"""
import sys, time, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RANDOM_SEED as SEED
from src.data_io import DATA_ROOT
from src.features import FEATURES
from src.plotstyle import COLORS, MODEL_COLORS, apply, finish

warnings.filterwarnings("ignore")
apply()
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "models"; OUT.mkdir(parents=True, exist_ok=True)
N_JOBS = 32
CV_EVAL_POINTS = 15000      # held-out points scored per fold (keeps slow models tractable)

# name -> (family, estimator, max training POINTS used per fit or None for all, config summary)
MODELS = {
    "Logistic Regression": ("Linear", make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300, random_state=SEED)), 40000, "C=1, lbfgs, 40k pts"),
    "Gaussian Naive Bayes": ("Probabilistic", make_pipeline(StandardScaler(), GaussianNB()), None, "var_smoothing=1e-9"),
    "KNN": ("Instance-based", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=N_JOBS)), 8000, "k=25, distance-weighted, 8k pts"),
    "Random Forest": ("Bagging ensemble", RandomForestClassifier(n_estimators=150, max_depth=16, min_samples_leaf=5, n_jobs=N_JOBS, random_state=SEED), 40000, "150 trees, depth 16, leaf 5, 40k pts"),
    "LightGBM": ("Gradient boosting", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63, n_jobs=N_JOBS, random_state=SEED, verbose=-1), None, "300 trees, lr 0.1, 63 leaves"),
    "XGBoost": ("Gradient boosting", xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=8, tree_method="hist", n_jobs=N_JOBS, random_state=SEED, verbosity=0), None, "300 trees, lr 0.1, depth 8"),
    "MLP": ("Neural network", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=512, max_iter=60, early_stopping=True, random_state=SEED)), 40000, "64-32 ReLU, adam, early stop, 40k pts"),
}


def load():
    tr = pd.read_parquet(PROC / "features_train.parquet")
    st = pd.read_parquet(PROC / "features_student_test.parquet")
    tr = tr[(tr.trip_reliable == 1) & (tr.point_has_positive == 1)].reset_index(drop=True)
    st = st[st.point_has_positive == 1].reset_index(drop=True)
    return tr, st


def sample_points(df, n_points, rng):
    """Subsample whole GPS points (keeps all candidates of a chosen point together)."""
    if n_points is None:
        return df
    key = df.trip_id.astype(np.int64) * 10_000_000 + df.pt_idx
    u = key.unique()
    if len(u) <= n_points:
        return df
    chosen = rng.choice(u, n_points, replace=False)
    return df[key.isin(chosen)]


def top1(df, prob):
    d = pd.DataFrame({"t": df.trip_id.to_numpy(), "p": df.pt_idx.to_numpy(), "prob": prob, "y": df.label.to_numpy()})
    best = d.loc[d.groupby(["t", "p"])["prob"].idxmax()]
    return float(best.y.mean()), best


def continuity(best, arcs_by_trip, arc_ids):
    """Share of consecutive GPS points whose predicted arcs are identical or directly connected."""
    best = best.assign(arc=arc_ids[best.index.to_numpy()])
    ok = tot = 0
    for t, g in best.groupby("t"):
        a = g.sort_values("p").arc.to_numpy()
        arcs = arcs_by_trip[t]
        same = a[1:] == a[:-1]
        conn = arcs[a[:-1], 1] == arcs[a[1:], 0]
        ok += int((same | conn).sum()); tot += len(a) - 1
    return ok / max(tot, 1)


def metrics(y, prob, df):
    pred = (prob >= 0.5).astype(int)
    acc, best = top1(df, prob)
    return dict(top1=acc, macro_f1=f1_score(y, pred, average="macro"),
                precision=precision_score(y, pred, average="macro", zero_division=0),
                recall=recall_score(y, pred, average="macro", zero_division=0),
                log_loss=log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6))), best, pred


def main():
    rng = np.random.default_rng(SEED)
    tr, st = load()
    print(f"train rows {len(tr):,} ({tr.trip_id.nunique()} trips) | student_test rows {len(st):,} ({st.trip_id.nunique()} trips)", flush=True)
    arcs_by_trip = {}
    for t in st.trip_id.unique():
        tid = f"{t:08d}"
        arcs_by_trip[t] = pd.read_csv(DATA_ROOT / "student_test" / tid / f"{tid}.arcs", sep=r"\s+", header=None).to_numpy()
    gkf = GroupKFold(n_splits=5)
    rows, cms, curves = [], {}, {}

    for name, (family, est, cap, cfg) in MODELS.items():
        print(f"\n=== {name} ({family}) ===", flush=True)
        folds, fit_times = [], []
        for k, (a, b) in enumerate(gkf.split(tr, groups=tr.trip_id)):
            fit_df = sample_points(tr.iloc[a], cap, rng)
            ev_df = sample_points(tr.iloc[b], CV_EVAL_POINTS, rng)
            m = clone(est); t0 = time.time()
            m.fit(fit_df[FEATURES].to_numpy(), fit_df.label.to_numpy()); fit_times.append(time.time() - t0)
            prob = m.predict_proba(ev_df[FEATURES].to_numpy())[:, 1]
            r, _, _ = metrics(ev_df.label.to_numpy(), prob, ev_df); folds.append(r)
            print(f"  fold {k}: top1={r['top1']:.4f} macroF1={r['macro_f1']:.4f} logloss={r['log_loss']:.4f} fit={fit_times[-1]:.1f}s", flush=True)
        cv = pd.DataFrame(folds)

        # ---- final refit on all train rows (same cap), evaluate on student_test ----
        fit_df = sample_points(tr, cap, rng)
        m = clone(est); t0 = time.time()
        m.fit(fit_df[FEATURES].to_numpy(), fit_df.label.to_numpy()); train_time = time.time() - t0
        t0 = time.time(); prob = m.predict_proba(st[FEATURES].to_numpy())[:, 1]; infer = time.time() - t0
        r, best, pred = metrics(st.label.to_numpy(), prob, st)
        cont = continuity(best, arcs_by_trip, st.arc_id.to_numpy())
        cms[name] = confusion_matrix(st.label.to_numpy(), pred)
        if name == "MLP":
            curves[name] = (m[-1].loss_curve_, m[-1].validation_scores_)
        train_acc, _, _ = (None, None, None)
        tr_eval = sample_points(tr, 15000, rng)
        tr_prob = m.predict_proba(tr_eval[FEATURES].to_numpy())[:, 1]
        tr_r, _, _ = metrics(tr_eval.label.to_numpy(), tr_prob, tr_eval)
        rows.append(dict(model=name, family=family, config=cfg,
                         cv_macro_f1_mean=cv.macro_f1.mean(), cv_macro_f1_std=cv.macro_f1.std(),
                         cv_top1_mean=cv.top1.mean(), cv_top1_std=cv.top1.std(), cv_logloss_mean=cv.log_loss.mean(),
                         train_macro_f1=tr_r["macro_f1"], train_top1=tr_r["top1"],
                         test_top1=r["top1"], test_macro_f1=r["macro_f1"], test_precision=r["precision"],
                         test_recall=r["recall"], test_log_loss=r["log_loss"], route_continuity=cont,
                         train_time_s=train_time, cv_fit_time_mean_s=float(np.mean(fit_times)),
                         inference_s_total=infer, inference_ms_per_1k_rows=1000 * infer / (len(st) / 1000),
                         seed=SEED))
        print(f"  TEST top1={r['top1']:.4f} macroF1={r['macro_f1']:.4f} logloss={r['log_loss']:.4f} "
              f"continuity={cont:.4f} train={train_time:.1f}s infer={infer:.2f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "model_comparison.csv", index=False)   # saved after every model

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "model_comparison.csv", index=False)
    (OUT / "model_comparison.md").write_text(res.round(4).to_markdown(index=False))
    json.dump({"seed": SEED, "cv": "GroupKFold(5) by trip", "cv_eval_points_per_fold": CV_EVAL_POINTS,
               "models": {k: v[3] for k, v in MODELS.items()}}, open(OUT / "experiment_config.json", "w"), indent=2)

    # ---------- figures ----------
    names = list(res.model); cols = MODEL_COLORS[:len(names)]
    def bars(ax, vals, errs=None, title="", ylabel="", fmt="{:.3f}", ylim=None):
        b = ax.bar(range(len(names)), vals, yerr=errs, color=cols, capsize=4, edgecolor="white")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_title(title); ax.set_ylabel(ylabel)
        if ylim: ax.set_ylim(*ylim)
        for rect, v in zip(b, vals):
            ax.annotate(fmt.format(v), (rect.get_x() + rect.get_width() / 2, v), ha="center", va="bottom", fontsize=8,
                        xytext=(0, 3 + (0 if errs is None else 6)), textcoords="offset points")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars(ax, res.cv_macro_f1_mean, res.cv_macro_f1_std, "Cross-validation macro F1 (5-fold, grouped by trip)", "macro F1", ylim=(0, 1.05))
    finish(fig, OUT / "cv_macro_f1.png", "Partition: train (80 trips), error bars = std across folds")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    bars(ax[0], res.test_top1, None, "Student-test top-1 accuracy", "accuracy", ylim=(0, 1.05))
    bars(ax[1], res.test_macro_f1, None, "Student-test macro F1", "macro F1", ylim=(0, 1.05))
    finish(fig, OUT / "test_top1_f1.png", "Partition: student_test (10 trips)")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars(ax, res.test_log_loss, None, "Student-test log loss (lower is better)", "log loss")
    finish(fig, OUT / "test_log_loss.png", "Partition: student_test (10 trips)")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    bars(ax[0], res.test_top1, None, "Quality: top-1 accuracy", "accuracy", ylim=(0, 1.05))
    bars(ax[1], res.inference_ms_per_1k_rows, None, "Cost: inference ms per 1000 candidate rows", "ms", fmt="{:.1f}")
    for a in ax: a.set_yscale("linear")
    finish(fig, OUT / "quality_vs_speed.png", "Partition: student_test")
    n = len(names); cw = 4
    fig, axs = plt.subplots((n + cw - 1) // cw, cw, figsize=(4 * cw, 3.6 * ((n + cw - 1) // cw)))
    for a, nm in zip(axs.ravel(), names):
        cm = cms[nm]; a.imshow(cm, cmap="Blues"); a.grid(False)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(nm, fontsize=11)
    for a in axs.ravel()[n:]: a.axis("off")
    finish(fig, OUT / "confusion_matrices.png", "Candidate-level, threshold 0.5, partition: student_test")
    if curves:
        loss, val = curves["MLP"]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(loss, color=COLORS["train"], label="training loss"); ax.set_xlabel("epoch"); ax.set_ylabel("log-loss")
        ax2 = ax.twinx(); ax2.plot(val, color=COLORS["student_test"], label="validation score (accuracy)"); ax2.set_ylabel("validation accuracy"); ax2.grid(False)
        ax.set_title("MLP training history (real epochs)"); ax.legend(loc="center right")
        finish(fig, OUT / "mlp_training_curve.png", "Partition: train (internal 10% early-stopping split)")
    print("\n", res[["model", "cv_macro_f1_mean", "test_top1", "test_macro_f1", "test_log_loss", "route_continuity", "train_time_s"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
