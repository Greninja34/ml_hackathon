"""Phase 7 (part): refit the tuned XGBoost on ALL training rows with the best Optuna parameters and export it.

Reads results/tuning/best_params.json, writes artifacts/xgb_tuned.json and artifacts/model_meta.json
(feature order, candidate radius, params, seed, student-test sanity check).
Run: python scripts/07_export_model.py
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CANDIDATE_RADIUS_M, RANDOM_SEED as SEED
from src.evaluation import best_per_point, top1_accuracy
from src.features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
best = json.load(open(ROOT / "results" / "tuning" / "best_params.json"))["best_params"]
tr = pd.read_parquet(ROOT / "data" / "processed" / "features_train.parquet")
st = pd.read_parquet(ROOT / "data" / "processed" / "features_student_test.parquet")
tr = tr[(tr.trip_reliable == 1) & (tr.point_has_positive == 1)]
st = st[st.point_has_positive == 1].reset_index(drop=True)

params = dict(objective="binary:logistic", tree_method="hist", device="cuda", eval_metric="logloss", seed=SEED, verbosity=0,
              **{k: v for k, v in best.items() if k != "n_estimators"})
t0 = time.time()
bst = xgb.train(params, xgb.QuantileDMatrix(tr[FEATURES].to_numpy(np.float32), tr.label.to_numpy()), num_boost_round=best["n_estimators"])
fit_s = time.time() - t0
out = ROOT / "artifacts"; out.mkdir(exist_ok=True)
bst.save_model(str(out / "xgb_tuned.json"))

# sanity check: reload from disk and score student_test
re = xgb.Booster(); re.load_model(str(out / "xgb_tuned.json"))
prob = re.predict(xgb.DMatrix(st[FEATURES].to_numpy(np.float32)))
acc = top1_accuracy(st.label.to_numpy(), best_per_point(st.trip_id.to_numpy(), st.pt_idx.to_numpy(), prob))
json.dump({"model": "XGBoost (Optuna-tuned)", "features": FEATURES, "candidate_radius_m": CANDIDATE_RADIUS_M, "params": best,
           "seed": SEED, "fit_seconds": fit_s, "student_test_top1_after_reload": acc}, open(out / "model_meta.json", "w"), indent=2)
print(f"saved artifacts/xgb_tuned.json ({(out / 'xgb_tuned.json').stat().st_size/1e6:.1f} MB); fit {fit_s:.1f}s; student_test top-1 after reload = {acc:.4f}")
