# PathSnap: map matching and route reconstruction

PathSnap reconstructs the road route a vehicle actually drove from noisy GPS points. For every GPS point it finds the
nearby road segments (arcs), scores each one with a tuned XGBoost classifier, and then decodes one coherent route with an
HMM / Viterbi decoder that uses the road-network connectivity. A FastAPI service (Laptop 1) runs the pipeline and scores
it against the reference route on the server; a Streamlit + Folium UI (Laptop 2) lets the evaluator pick a trip and see
the predicted route and its metrics.

```
raw trip (.track .nodes .arcs) -> local metric projection -> candidate arcs within 75 m -> 18 relative features
  -> tuned XGBoost probability per candidate -> HMM / Viterbi -> fallback for points without candidates
  -> stitched route (shortest-path gap filling) -> [server-side scoring against .route]
```

## Results

All numbers below are produced by the scripts in this repository; the files they come from are listed.

**Model comparison** (`results/models/model_comparison.csv`; 5-fold CV grouped by trip on 78 train trips, test = 10 student_test trips)

| Model | CV macro F1 | Test top-1 | Test macro F1 | Test log loss | Continuity | Train time |
|---|---|---|---|---|---|---|
| **XGBoost (GPU)** | **0.9349** | **0.8952** | **0.9393** | **0.0305** | **0.9387** | 5.9 s |
| LightGBM | 0.9330 | 0.8889 | 0.9377 | 0.0356 | 0.9240 | 23.9 s |
| Random Forest | 0.9342 | 0.8886 | 0.9364 | 0.0314 | 0.9305 | 72.6 s |
| MLP (PyTorch, GPU) | 0.9299 | 0.8871 | 0.9340 | 0.0330 | 0.9374 | 19.7 s |
| Logistic Regression | 0.9152 | 0.8734 | 0.9175 | 0.0440 | 0.9267 | 10.6 s |
| KNN (8k-point subsample) | 0.8992 | 0.8353 | 0.8997 | 0.0576 | 0.9197 | 0.03 s |
| Gaussian Naive Bayes | 0.7502 | 0.7953 | 0.7421 | 0.5412 | 0.9214 | 0.7 s |

**Bayesian optimisation of XGBoost** (`results/tuning/`; Optuna TPE, 60 trials with median pruning, objective = CV top-1):
CV top-1 0.8857 -> 0.8869; student_test top-1 0.8952 -> 0.8995, macro F1 0.9393 -> 0.9412, log loss 0.0305 -> 0.0299.
A small gain: the untuned model was already close to the best region of the search space.

**HMM / Viterbi** (`results/hmm/hmm_vs_classifier.csv`; decoder tuned with Optuna on the 78 train trips using
out-of-fold probabilities, 150 trials; student_test never used for tuning)

| Partition | Method | Top-1 | Continuity | Route F1 (full .route) |
|---|---|---|---|---|
| student_test | classifier only | 89.95% | 93.84% | 81.03% |
| student_test | classifier + HMM (tuned) | **96.51%** | **98.18%** | **97.75%** |

**End-to-end pipeline** (`results/pipeline/`; exactly the code the API runs, CPU only)

| Trip set | Pooled top-1 (classifier only -> + HMM) | Mean route F1 | Mean / max time per trip |
|---|---|---|---|
| student_test (10 trips) | 89.95% -> 96.51% | 97.75% | 0.35 s / 0.62 s |
| evaluator_test (10 trips, single final run) | 84.73% -> 94.93% | 96.57% | 0.28 s / 0.65 s |

The evaluator trips were run once, after every model and decoder setting was frozen; nothing was changed afterwards.
Weakest evaluator trips: 00000030 (top-1 83.8%), 00000015 (route F1 81.7%); on 00000067 the HMM lowered top-1
(95.1% -> 93.6%).

## Repository layout

| Folder | Contents |
|---|---|
| `src/` | data loading and projection (`data_io.py`), route alignment for labels (`labeling.py`), candidate search (`candidates.py`), features (`features.py`), HMM/Viterbi (`hmm.py`), end-to-end pipeline (`pipeline.py`), server-side scoring (`scoring.py`), evaluation helpers, PyTorch MLP, plotting and Folium maps |
| `scripts/` | numbered pipeline steps `01`-`11` (see below) |
| `artifacts/` | `xgb_tuned.json` (model), `model_meta.json` (feature order, radius), `hmm_config.json` (decoder settings) |
| `results/` | every table, figure, Folium map, trial record and run log produced by the scripts |
| `api/` | FastAPI backend (Laptop 1) |
| `ui/` | Streamlit evaluator UI (Laptop 2) + its own `requirements.txt` |

## Setup

```bash
conda create -n ml_lab python=3.13 -y && conda activate ml_lab    # or a python -m venv
pip install -r requirements.txt
```

Data is not committed. Place it like this (paths can be overridden with `PATHSNAP_DATA`):

```
final/data/raw/train/<trip_id>/<trip_id>.track|.nodes|.arcs|.route          80 trips
final/data/raw/student_test/<trip_id>/...                                    10 trips
evaluator_test/<trip_id>/...      (next to the repository folder, never inside it)
```

## Reproducing the experiments

Run from the repository root, in order (each script writes to `results/` and prints a log):

| Step | Script | Output |
|---|---|---|
| EDA | `python scripts/01_eda.py`, `python scripts/02_trip_map.py` | `results/eda/` (inventory, histograms, Folium maps) |
| Labels + candidate radius | `python scripts/03_candidates.py` | `results/candidates/` (radius sweep, miss rate) |
| Features | `python scripts/04_features.py` | `data/processed/*.parquet`, `results/features/` |
| 7 models | `python scripts/05_models.py` | `results/models/` |
| Bayesian tuning | `python scripts/06_tune_xgboost.py` | `results/tuning/` (trials.csv, study.db, comparison) |
| Export model | `python scripts/07_export_model.py` | `artifacts/xgb_tuned.json`, `model_meta.json` |
| Feature analysis | `python scripts/08_feature_analysis.py` | `results/features/` (importance, ablation) |
| Out-of-fold probabilities | `python scripts/09_probs.py` | `data/processed/probs_*.parquet` |
| HMM tuning + comparison | `python scripts/10_hmm.py` | `results/hmm/`, `artifacts/hmm_config.json` |
| End-to-end check | `python scripts/11_check_pipeline.py` | `results/pipeline/` |

Seed 42 everywhere. XGBoost and the MLP were trained on an NVIDIA GPU; GPU training is not bit-for-bit deterministic,
so a rerun can differ in the last decimals.

## Running the two-laptop demo

**Laptop 1 (API)**, from the repository root:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-access-log
```

It serves two trip sets, `student_test` and `evaluator_test` (override with
`PATHSNAP_TRIP_SETS="name=/path;name2=/path2"`). Every request is logged in this terminal: trip, sizes, metrics, timings.
Endpoints: `GET /health`, `GET /sets`, `GET /trips?trip_set=...`, `POST /predict` with
`{"trip_set": "evaluator_test", "trip_id": "00000015", "include_network": true}`.

**Laptop 2 (UI)**, needs only Python 3.11+ and `ui/requirements.txt`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

Enter `http://<laptop-1-ip>:8000` as the API address in the sidebar, choose the trip set, then **Reconstruct route** or
**Run all trips**. The UI shows the route on a Folium map (GPS points, predicted route, road network, snapped positions),
the metrics, a per-trip comparison with conclusions, and the request log. Both laptops must be on the same network
(port 8000 reachable on Laptop 1).

## Data-integrity rules followed

- Splits are by trip; cross-validation is grouped by trip. Scalers are fitted inside each pipeline on training rows only.
- Features use only relative geometry and kinematics: no arc ids and no absolute coordinates.
- The evaluator trips were never used for training, features, model choice, hyperparameters or decoder settings.
- Reference routes are read only by `src/scoring.py` on the API server, after prediction; the prediction code
  (`src/pipeline.py`) never reads them, and the API returns metrics only, never reference arcs. No outputs are hardcoded.
- Two train trips (00000033, 00000054) whose reference route does not follow the GPS (median distance 122 m and 140 m)
  are excluded from training and reported separately.
