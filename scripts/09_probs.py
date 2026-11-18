"""Classifier probabilities for the decoder.

train        : OUT-OF-FOLD probabilities (5-fold GroupKFold by trip, tuned XGBoost params). Every train trip is scored by a
               model that never saw it, so decoder settings tuned on them are not biased by an over-confident classifier.
student_test : probabilities from the final tuned model (artifacts/xgb_tuned.json, trained on the 78 train trips).
Probabilities are produced for ALL candidate rows of a trip (including points whose reference arc is not a candidate), so
the decoder sees the same chain it would see at demo time. Unreliable trips (33, 54) are not decoded.
Outputs: data/processed/probs_train_oof.parquet, probs_student_test.parquet
Run: python scripts/09_probs.py
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RANDOM_SEED as SEED
from src.features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
best = json.load(open(ROOT / "results" / "tuning" / "best_params.json"))["best_params"]
params = dict(objective="binary:logistic", tree_method="hist", device="cuda", eval_metric="logloss", seed=SEED, verbosity=0,
              **{k: v for k, v in best.items() if k != "n_estimators"})

full = pd.read_parquet(PROC / "features_train.parquet")
full = full[full.trip_reliable == 1].reset_index(drop=True)          # all rows of the 78 reliable trips
fit = full[full.point_has_positive == 1]                              # rows the classifier may learn from
trips = np.array(sorted(fit.trip_id.unique()))
# same fold assignment logic as Phase 4/5 (GroupKFold over the fit rows' trips)
fit_r = fit.reset_index(drop=True)
oof = np.zeros(len(full), dtype=np.float32)
for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(fit_r, groups=fit_r.trip_id)):
    val_trips = set(fit_r.trip_id.iloc[b].unique())
    tr_rows = fit_r.iloc[a]
    bst = xgb.train(params, xgb.QuantileDMatrix(tr_rows[FEATURES].to_numpy(np.float32), tr_rows.label.to_numpy()),
                    num_boost_round=best["n_estimators"])
    m = full.trip_id.isin(val_trips).to_numpy()
    oof[m] = bst.predict(xgb.DMatrix(full.loc[m, FEATURES].to_numpy(np.float32)))
    print(f"fold {k}: scored {m.sum():,} rows from {len(val_trips)} unseen trips", flush=True)
out = full[["trip_id", "pt_idx", "arc_id", "t_frac", "label", "point_has_positive"]].copy(); out["prob"] = oof
out.to_parquet(PROC / "probs_train_oof.parquet", index=False)

st = pd.read_parquet(PROC / "features_student_test.parquet")
m = xgb.Booster(); m.load_model(str(ROOT / "artifacts" / "xgb_tuned.json"))
o2 = st[["trip_id", "pt_idx", "arc_id", "t_frac", "label", "point_has_positive"]].copy()
o2["prob"] = m.predict(xgb.DMatrix(st[FEATURES].to_numpy(np.float32)))
o2.to_parquet(PROC / "probs_student_test.parquet", index=False)
print("saved", len(out), "train OOF rows and", len(o2), "student_test rows")
