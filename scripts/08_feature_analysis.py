"""Phase 3 evidence: which features does the model use, and what breaks without each feature group?

1. Gain importance from the saved tuned XGBoost (artifacts/xgb_tuned.json)
2. Ablation: retrain the tuned XGBoost on all train rows WITHOUT one feature group, score on student_test.
This only explains the existing feature set; it is not used to pick features (no feature was changed afterwards).
Outputs: results/features/feature_importance.csv|png, feature_ablation.csv|png
Run: python scripts/08_feature_analysis.py
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import f1_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RANDOM_SEED as SEED
from src.evaluation import best_per_point, top1_accuracy
from src.features import FEATURES
from src.plotstyle import COLORS, apply, finish

apply()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "features"
best = json.load(open(ROOT / "results" / "tuning" / "best_params.json"))["best_params"]
tr = pd.read_parquet(ROOT / "data" / "processed" / "features_train.parquet")
st = pd.read_parquet(ROOT / "data" / "processed" / "features_student_test.parquet")
tr = tr[(tr.trip_reliable == 1) & (tr.point_has_positive == 1)]
st = st[st.point_has_positive == 1].reset_index(drop=True)

GROUPS = {
    "position (dist, t_frac, rank, gap-to-min, n_cands)": ["dist", "t_frac", "dist_minus_min", "dist_rank", "n_cands"],
    "direction (cos_heading, heading_ok, ds_span, has_reverse)": ["cos_heading", "heading_ok", "ds_span", "has_reverse"],
    "fit to trail (d_prev, d_next, d_mean_win, speed)": ["d_prev", "d_next", "d_mean_win", "speed_ms"],
    "road shape (arc_len, junction degrees)": ["arc_len", "deg_out_end", "deg_in_start"],
    "time (dt_prev, dt_next)": ["dt_prev", "dt_next"],
}
params = dict(objective="binary:logistic", tree_method="hist", device="cuda", eval_metric="logloss", seed=SEED, verbosity=0,
              **{k: v for k, v in best.items() if k != "n_estimators"})


def fit_eval(cols):
    d = xgb.QuantileDMatrix(tr[cols].to_numpy(np.float32), tr.label.to_numpy())
    bst = xgb.train(params, d, num_boost_round=best["n_estimators"])
    prob = bst.predict(xgb.DMatrix(st[cols].to_numpy(np.float32)))
    idx = best_per_point(st.trip_id.to_numpy(), st.pt_idx.to_numpy(), prob)
    y = st.label.to_numpy()
    return dict(test_top1=top1_accuracy(y, idx), test_macro_f1=f1_score(y, (prob >= .5).astype(int), average="macro"),
                test_log_loss=log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6)))


# ---- 1. importance (from the saved model) ----
m = xgb.Booster(); m.load_model(str(ROOT / "artifacts" / "xgb_tuned.json"))
m.feature_names = None
gain = m.get_score(importance_type="total_gain")
imp = pd.Series({FEATURES[int(k[1:])]: v for k, v in gain.items()}).reindex(FEATURES).fillna(0)
imp = (imp / imp.sum()).sort_values()
imp.rename("share_of_total_gain").to_csv(OUT / "feature_importance.csv")
fig, ax = plt.subplots(figsize=(7.5, 5.6))
ax.barh(imp.index, imp.values * 100, color=COLORS["accent"])
ax.set_xlabel("share of total split gain (%)"); ax.set_title("Which features the tuned XGBoost relies on")
finish(fig, OUT / "feature_importance.png", "Model: artifacts/xgb_tuned.json (trained on 78 train trips)")

# ---- 2. ablation ----
rows = [dict(removed="nothing (all 18 features)", n_features=len(FEATURES), **fit_eval(FEATURES))]
for name, cols in GROUPS.items():
    keep = [f for f in FEATURES if f not in cols]
    rows.append(dict(removed=name, n_features=len(keep), **fit_eval(keep)))
    print(rows[-1], flush=True)
ab = pd.DataFrame(rows); ab["top1_change_pts"] = 100 * (ab.test_top1 - ab.test_top1.iloc[0])
ab.to_csv(OUT / "feature_ablation.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 4.4))
short = ["all 18"] + [k.split(" (")[0] for k in GROUPS]
bars = ax.bar(short, ab.test_top1 * 100, color=[COLORS["reference"]] + [COLORS["predicted"]] * len(GROUPS))
for b, v in zip(bars, ab.test_top1 * 100):
    ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")
ax.set_ylim(60, 95); ax.set_ylabel("student-test top-1 (%)"); ax.set_title("Accuracy when one feature group is removed")
ax.tick_params(axis="x", labelrotation=15)
finish(fig, OUT / "feature_ablation.png", "Partition: student_test (y-axis starts at 60%); tuned XGBoost retrained on all train rows each time")
print(ab.round(4).to_string(index=False))
print(imp.sort_values(ascending=False).round(4).to_string())
