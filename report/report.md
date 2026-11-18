---
title: "PathSnap: Distributed Map-Matching and Route Reconstruction"
subtitle: "Technical report"
course: "UE24CS352A Machine Learning"
team: "Team 6"
members: "Vignesh Nunna (PES2UG24CS582) \\newline Vidushi Singh (PES2UG24CS581) \\newline Vikas N Naik (PES2UG24CS583) \\newline Vikhyat Agrawal (PES2UG24CS585)"
repo: "https://github.com/Greninja34/ml_hackathon"
date: "18 November 2026"
---

# Problem and approach

## Problem

Given a track (longitudes, latitudes, times), node positions (`.nodes`), directed arcs between nodes (`.arcs`, index = arc id), and optionally a true route (list of arc ids, `.route`), we aim to find which arc in the graph each GPS point is on, and predict a sensible route.

## Approach

Candidate arcs are generated, then classified, then decoded:

1. **Candidate generation.** For each GPS point in a track, candidates are generated from all arcs within a 75 m radius (found with an R-tree).
2. **Feature engineering.** 18 relative features are calculated for each GPS point – candidate arc pair.
3. **Classification.** A model predicts the probability of the candidate being the true arc. 7 models were considered. XGBoost was chosen, and optimised using Bayesian optimisation.
4. **Decoding.** An HMM is built over the candidate arcs, with transition scores based on distances between arcs in the graph and emission scores based on the probabilities output by the classifier; its parameters are tuned, not trained. This gives one arc for each GPS point, and the arcs can be concatenated to give a continuous route.

![Prediction pipeline (`src/pipeline.py`; the API runs exactly this function).](img/pipeline.png){width=100%}

# Dataset understanding and exploratory data analysis

The data was split into training (80 tracks), student test (10 tracks) and evaluator data (10 tracks). Fitting, cross-validating and optimising models was done using only training data. Evaluator data was not opened until final evaluation (Section 8). The data split is indicated in the figure captions.

## Data overview, structure and quality

| Item | Value | Item | Value |
|:-------------------------------|------------:|:-------------------------------|------------:|
| Trips (train / student_test / evaluator) | 80 / 10 / 10 | GPS points (train + student_test) | 227,577 |
| Points per trip (min / median / max) | 262 / 1,496 / 19,313 | Arcs per trip (min / median / max) | 649 / 18,761 / 307,392 |
| Route arcs per trip (min / median / max) | 59 / 272 / 2,266 | Steps exactly 1 s apart (mean) | 94.3 % |
| Trips with a time gap > 60 s | 23 | Largest gap | 153,064 s |
| Arcs with a reverse twin (mean) | 92.2 % | Weak components per trip (median / max) | 19 / 421 |
| Largest-component share of nodes (min) | 0.693 | Reference route fully connected (min) | 0.9813 |

See `results/eda/summary.md` and `results/eda/inventory.csv` (`scripts/01_eda.py`), for training and student test data.

**Missingness and validity.**

| Check | Result | Decision |
|:--------------------------------------------|-----------:|:----------------------------------------------|
| NaN values in tracks / nodes | 0 / 0 | none needed |
| Longitude / latitude out of range | 0 | none needed |
| Duplicate / non-monotonic timestamps | 0 / 0 | none needed |
| Arcs with a non-existent node; route arcs with a non-existent arc | 0 / 0 | none needed |
| Zero-length arcs | 106 | never used as candidates (no geometry) |
| Isolated nodes | 0 | none needed |
| Time gaps > 60 s | 23 trips | kept (real behaviour); time-gap features; handled in the HMM |
| GPS points with no arc within 75 m | 13,572 (6.0 %) | 9 trips, 12,855 in trip 54; none in student_test; filled after decoding (Section 5) |
| Reference route far from the GPS (median > 15 m) | trips 33, 54 | flagged unreliable, excluded from training |

The data is complete and internally valid. However, the real data has other quality issues: noisy GPS positions, time gaps between points, disconnected graphs, and reference routes that do not always follow the GPS points.

```{=latex}
\begin{figure}[H]\centering
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/eda/hist_points_per_trip.png}\caption{GPS points per trip (train + student\_test).}\end{minipage}\hfill
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/eda/hist_arcs_per_trip.png}\caption{Road-network size (arcs) per trip (train + student\_test).}\end{minipage}
\end{figure}
\begin{figure}[H]\centering
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/eda/hist_max_gap.png}\caption{Largest time gap per trip (train + student\_test).}\end{minipage}\hfill
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/eda/hist_route_arcs.png}\caption{Reference-route length in arcs (train + student\_test).}\end{minipage}
\end{figure}
```

## Pre-processing

1. **Projection to a local metric coordinate system.** Longitudes and latitudes are angular units. At 60° N, one degree of longitude corresponds to approx. 56 km, compared to 111 km at the equator. As tracks occurred in different countries, a local metric projection was performed on each track using the mean GPS position as a reference point. For a track with mean latitude $\varphi_0$ and longitude $\lambda_0$, $x = (\lambda-\lambda_0)\,R\cos\varphi_0$, $y = (\varphi-\varphi_0)\,R$, where $R = 6{,}371{,}008.8$ m. For distances up to a few km this introduces negligible error (well below the GPS error of ~5 m), and no UTM zone has to be chosen for individual tracks (`src/data_io.py`).
2. **Label assignment.** `.route` only lists the arc ids of the route, not which arc each GPS point is on. For example, track 1 has 1,315 GPS points and 525 arcs. Using dynamic programming, each GPS point is matched to a position along the route. The match must move in the forward direction (never backwards), and incurs a cost of the distance to the arc + 0.05 m for each skipped arc. All 90 tracks are matched forward, and typically have point-to-arc distances of 4–7 m on normal tracks. These labels are only used for training and scoring (`src/labeling.py`).
3. **Reliability.** Tracks with a median distance between GPS and reference > 15 m are removed from training (33, with 122 m; 54, with 140 m). Labelling these points would introduce noise.
4. **Candidate arcs.** All arcs within radius R of the given point, found with a shapely STRtree. R = 75 m (radius sweep, see Figure \ref{fig:radius}).
5. **Training data.** Only the reliable tracks, with true arcs among the candidates: 2,880,132 rows across 181,659 points (6.3 % positive). If the true arc is not among the candidates, it counts as a missed candidate, and not a model error.
6. **Scaling.** Models requiring scaling (Logistic Regression, Naive Bayes, KNN, MLP) have a StandardScaler fitted on the training data within their pipeline.

## Observations

```{=latex}
\begin{figure}[H]\centering
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/eda/trip_00000003_matplotlib.png}\caption{GPS trajectory and road network of a typical trip (train, trip 3): the reference route follows the GPS.}\end{minipage}\hfill
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/eda/trip_00000033_matplotlib.png}\caption{Trip 33 (train): the reference route covers only fragments of the driven loop, so the trip is flagged unreliable.}\end{minipage}
\end{figure}
\begin{figure}[H]\centering
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/eda/hist_noise_all_points.png}\caption{Distance from each GPS point to the reference route (train + student\_test). p50 5.6 m, p90 48.8 m, p95 137.7 m, p99 200 m.}\end{minipage}\hfill
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/eda/hist_noise_median.png}\caption{Median GPS-to-reference distance per trip; the two outliers are trips 33 and 54.}\end{minipage}
\end{figure}
```

- **Long tails.** Half of the points are less than 5.6 m from the route, but 1 in 10 points is ~50 m away (likely to be confused with parallel roads).
- **Directed duplicates.** 92.2 % of arcs have an exact duplicate in the opposite direction, at the same distance from all points. Therefore choosing the closest arc gives only **58.6 %** top-1 accuracy on student test. Applying a manually designed rule of "choose the closest arc, and prioritise based on direction" gives **85.4 %**. Direction is important.
- **Sampling.** Data is mostly sampled at 1 Hz. Consecutive points are very similar, so data must be split by track, never by point.
- **Disconnected graphs** (median 19 weak components per track) and small disconnections within the reference route in some cases (minimum proportion of connected route = 0.981). Therefore the decoder must allow transitions between disconnected arcs.

![Interactive Folium map of train trip 3 (`results/eda/trip_00000003_map.html`, made by `scripts/02_trip_map.py`): raw GPS observations, road network and reference route, with switchable base and data layers. Partition: train.](img/folium_eda_trip3.png){width=96%}

![Candidate-radius sweep (`results/candidates/radius_summary.csv`; train + student_test, reliable trips in green). At 75 m the reference arc is missing for 0.8 % of points of reliable trips (2.3 % at 30 m, 1.3 % at 50 m, 0.6 % at 100 m) at about 16 candidates per point.](../results/candidates/radius_sweep.png){#fig:radius width=88%}

## Features

Only relative geometric and kinematic features were used. Arc ids and absolute positions were excluded, as the tracks come from different cities and these values would not be transferable. The same code is used for training and within the API (`src/features.py`), and it never reads in labels.

| Group | Features | Reason |
|:--------------|:-----------------------------|:--------------------------------------------------------|
| Position (5) | `dist`, `t_frac`, `dist_minus_min`, `dist_rank`, `n_cands` | How close the arc is, and how close relative to the other candidates of the same point; relative closeness transfers across cities. |
| Direction (4) | `cos_heading`, `heading_ok`, `ds_span`, `has_reverse` | Separates an arc from its reverse twin: cosine between the motion over points k−2…k+2 and the arc direction; heading trusted only if the vehicle moved ≥ 3 m within ≤ 10 s; forward progress of neighbouring points along the arc. |
| Fit to trail (4) | `d_prev`, `d_next`, `d_mean_win`, `speed_ms` | Does the arc also fit the neighbouring points (±2 and an 11-point window)? One noisy point cannot pull the decision. |
| Road shape (3) | `arc_len`, `deg_out_end`, `deg_in_start` | Segment length and junction degree. |
| Time (2) | `dt_prev`, `dt_next` | Marks points next to a time gap. |

```{=latex}
\begin{figure}[H]\centering
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/features/feature_ablation.png}\caption{Ablation: student\_test top-1 after removing each group and retraining (tuned XGBoost). Direction $-23.3$ points, position $-7.1$, road shape $-1.5$, fit to trail $-1.0$, time $-0.3$.}\end{minipage}\hfill
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/features/feature_importance.png}\caption{XGBoost gain importance (share of total): dist\_minus\_min 0.29, dist\_rank 0.20, ds\_span 0.19, dist 0.09.}\end{minipage}
\end{figure}
```

![Distributions of key features for reference arcs vs other candidates (train, reliable trips, points whose reference arc is a candidate).](../results/features/feature_separation.png){width=68%}

The features were decided before the ablation study. Therefore no feature selection was done using student test data.

# Model experimentation and comparison

## Approach

All models were trained on the same data, with the same seed (42) and the same folds. 5-fold `GroupKFold` was performed with groups defined as tracks, over the 78 reliable tracks: approx. 575,000 rows and 15–16 tracks in each fold. Each model was then retrained on all 78 tracks, and evaluated once on the student test data (19,139 points, 356,362 candidate rows). See `scripts/05_models.py`; experiment log: `results/models/experiment_run.log`.

## Measures

Top-1 accuracy on points is the main measure, as it represents what the system does. Macro F1, precision and recall are calculated on candidate rows with a probability threshold of 0.5; these give equal weighting to the rare positive class (6 %). Log loss shows how well probabilities are predicted (important for the log-probabilities used in the HMM). Continuity is the proportion of consecutive chosen arcs that are the same or directly connected. All models have the same rate of missed candidates (0.8 %) for R = 75 m.

## Summary of model performance

| Model | Family | Key configuration | CV macro F1 (std) | Top-1 acc. | Route contin. | Test macro F1 | Log loss | Inference ms/1k rows |
|:------------|:-----------|:----------------------------|---------:|--------:|--------:|--------:|--------:|---------:|
| **XGBoost** | Boosting | 300 trees, lr 0.1, depth 8, hist, GPU | 0.9349 (0.023) | **0.8952** | **0.9387** | **0.9393** | **0.0305** | 0.19 |
| Random Forest | Bagging | 100 trees, depth 16, min leaf 5, 25 % bootstrap | 0.9342 (0.022) | 0.8886 | 0.9305 | 0.9364 | 0.0314 | 0.70 |
| LightGBM | Boosting | 300 trees, lr 0.1, 63 leaves | 0.9330 (0.023) | 0.8889 | 0.9240 | 0.9377 | 0.0356 | 2.19 |
| MLP | Neural net | 64-32 ReLU, Adam 1e-3, batch 4096, early stop, GPU | 0.9299 (0.021) | 0.8871 | 0.9374 | 0.9340 | 0.0330 | 0.03 |
| Logistic Regr. | Linear | C = 1, lbfgs, scaled | 0.9152 (0.021) | 0.8734 | 0.9267 | 0.9175 | 0.0440 | 0.08 |
| KNN | Instance | k = 25, distance-weighted, 8,000 points | 0.8992 (0.020) | 0.8353 | 0.9197 | 0.8997 | 0.0576 | 108.27 |
| Gaussian NB | Probabilistic | var_smoothing 1e-9, scaled | 0.7502 (0.010) | 0.7953 | 0.9214 | 0.7421 | 0.5412 | 0.10 |

Source: `results/models/model_comparison.csv`. Top-1, continuity, macro F1 and log loss are measured on student test data. Model training times (full re-fit on all 78 tracks): XGBoost 5.9 s, Random Forest 72.6 s, LightGBM 23.9 s, MLP 19.7 s, Logistic Regression 10.6 s, KNN 0.03 s, Naive Bayes 0.7 s. Cross-validation top-1 accuracy: XGBoost 0.8857, Random Forest 0.8827, MLP 0.8797, LightGBM 0.8786, Logistic Regression 0.8749, KNN 0.8401, Naive Bayes 0.8307. All metrics are given for all models, as all of them produce probabilities for candidate arcs.

## Model selection

As the standard deviation of macro F1 scores across CV folds (~0.02) is larger than the differences between the top 4 models, they are statistically tied in terms of F1 score. XGBoost was chosen for having the highest CV macro F1 and top-1 score, the best log loss (important for the HMM emissions), the best continuity score, and the quickest training time of the strong models. This decision was made based on CV performance, and is confirmed by the performance on student test data. The variance in model scores between folds is mostly due to which tracks are in each fold: all models except Naive Bayes have their lowest scores in fold 1 and highest in fold 4. XGBoost has top-1 scores per fold of 0.825, 0.863, 0.918, 0.938 and 0.885.

## Model answers

Each model answers parts A, B and C. "Train top-1" refers to a sample of 15,000 points from the training data, scored with the final model.

### Logistic Regression

**A1–A2.** The model is a linear combination of the 18 features, passed through a sigmoid function. This is a natural baseline for a binary classification problem (is this the correct arc?), and it outputs proper probabilities. Advantages: fast (10.6 s). Disadvantages: it cannot model interaction effects; for example, direction should only be considered if the vehicle is moving.

**B5–B6.** C = 1.0 (inverse of the L2 regularisation strength), solver = lbfgs, max_iter = 300. Inputs standardised. Minimises binary log loss with L2 regularisation.

**C7–C9.** Top-1: 0.8734, macro F1: 0.9175 (precision: 0.9173, recall: 0.9176), log loss: 0.0440. Slight underfitting, as the training (0.8765) and test (0.8734) scores are similar, and quite low.

### Gaussian Naive Bayes

**A1–A2.** A Gaussian distribution is fitted for each feature and class, and the probabilities are multiplied (independence assumption). A cheap probabilistic baseline from a different family. Advantages: very fast training (< 1 s). Disadvantages: the features are highly correlated (`dist`, `d_prev` and `d_mean_win` are all related to distance), so the same evidence is weighted multiple times, giving overconfident probabilities.

**B5–B6.** var_smoothing = 1e-9. Inputs standardised. Maximises the likelihood of the Gaussian distributions per class (analytical solution).

**C7–C9.** Top-1: 0.7953, macro F1: 0.7421 (precision: 0.6834, recall: 0.9404), log loss: 0.5412 (around 10× poorer than the other models). Underfitting due to violated assumptions; the test score (0.7953) is much lower than the training score (0.8301). Its probabilities are poorly calibrated and would mislead the HMM.

### K-Nearest Neighbours

**A1–A2.** For each candidate row, the nearest 25 rows in the training data are found and give a distance-weighted vote. Advantages: no training required, flexible decision boundary. Disadvantages: time-consuming (38.6 s on student test data, as it takes 108 ms per 1,000 rows), and distance in 18 mixed features gives a poor measure of similarity. It was trained on 8,000 points; for the full data of 2.88 M rows, prediction would take hours.

**B5–B6.** k = 25, weighted by distance, standardised inputs, 8,000 training points. No loss function (lazy learner).

**C7–C9.** Top-1: 0.8353, macro F1: 0.8997 (precision: 0.9218, recall: 0.8801), log loss: 0.0576. Because of the sub-sampled training data, the test score (0.8353) is 3.2 points lower than the training score (0.8676): mild overfitting.

### Random Forest

**A1–A2.** Many decision trees are averaged. Each tree is trained on a bootstrapped sample of the data, with randomly selected features. Trees learn conditional relationships in the data without scaling. Advantages: stable, good probabilities. Disadvantages: slowest to train (72.6 s on 32 CPU threads), large.

**B5–B6.** n_estimators = 100, max_depth = 16, min_samples_leaf = 5, max_samples = 0.25. Gini impurity as the split criterion.

**C7–C9.** Top-1: 0.8886, macro F1: 0.9364 (precision: 0.9476, recall: 0.9259), log loss: 0.0314. Slight overfitting: the test score is 1.7 points lower than the training score, a healthy level.

### LightGBM

**A1–A2.** Gradient boosting with leaf-wise tree growth. Each tree fits the residual errors of the previous trees. Advantages: accurate, fast on CPU (23.9 s). Disadvantages: overfits without regularisation, many hyperparameters.

**B5–B6.** n_estimators = 300, learning_rate = 0.1, num_leaves = 63. Binary log loss.

**C7–C9.** Top-1: 0.8889, macro F1: 0.9377 (precision: 0.9412, recall: 0.9342), log loss: 0.0356. Moderate overfitting (the test score is 2.3 points lower than the training score). Slightly worse probabilities than XGBoost.

### XGBoost (chosen)

**A1–A2.** Gradient boosting with second-order updates, explicit regularisation (γ per leaf, L1/L2 regularisation on leaf weights), and level-wise tree growth. Interaction effects are important for this problem (direction only matters if moving; distance matters relative to the other candidates), and trees can easily represent interaction effects. The log-loss objective means the probabilities can be used as emissions in the HMM. Advantages: best scores (CV and test), best log loss and continuity, fast (5.9 s on GPU). Disadvantages: the greatest difference between train and test scores, so regularisation is required.

**B5–B6.** Initial parameters: n_estimators = 300, learning_rate = 0.1, max_depth = 8, tree_method = hist, device = cuda. The other parameters use their default values (min_child_weight = 1, subsample = 1, colsample_bytree = 1, λ = 1, α = 0, γ = 0). Objective function `binary:logistic`: the sum of the binary log loss and, for each tree, $\gamma T + \tfrac12\lambda\lVert w\rVert^2 + \alpha\lVert w\rVert_1$.

**C7–C9.** Untuned: top-1: 0.8952, macro F1: 0.9393 (precision: 0.9466, recall: 0.9322), log loss: 0.0305. Tuned (see Section 4): top-1: 0.8995, macro F1: 0.9412, log loss: 0.0299. Greatest overfitting of the group (5.2 point difference between train and test scores), which is consistent with tuning moving towards more regularisation and subsampling.

### Multi-layer perceptron

**A1–A2.** A feedforward neural network (18 → 64 → 32 → 1, ReLU activation) implemented in PyTorch (`src/torch_mlp.py`) and trained on the GPU. It can learn complex interaction effects. Advantages: smooth non-linear decision boundary, fast inference (0.03 ms per 1,000 rows). Disadvantages: requires scaling, sensitive to parameter choices, and rarely outperforms gradient-boosted trees on tabular data.

**B5–B6.** Hidden layers: 64 → 32. ReLU activation. Optimiser: Adam (lr = 1e-3). Batch size: 4,096. Maximum epochs: 40. Early stopping after 5 epochs of no improvement on an internal validation split (10 % of the data). Inputs standardised. Loss function: binary cross-entropy.

**C7–C9.** Top-1: 0.8871, macro F1: 0.9340 (precision: 0.9443, recall: 0.9243), log loss: 0.0330. Generalises well (only a 0.9 point difference between train and test scores). It is the only model with a per-epoch history (see Figure \ref{fig:mlp}).

## Visualisations of models

```{=latex}
\begin{figure}[H]\centering
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/models/cv_macro_f1.png}\caption{Cross-validation macro F1 per model (5 folds grouped by trip, train; error bars = std).}\end{minipage}\hfill
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/models/test_log_loss.png}\caption{Student-test log loss (lower is better).}\end{minipage}
\end{figure}
```

![Student-test top-1 accuracy and macro F1 per model.](../results/models/test_top1_f1.png){width=92%}

![Candidate-level confusion matrices at threshold 0.5 (student_test). Naive Bayes trades many false positives for recall.](../results/models/confusion_matrices.png){width=96%}

```{=latex}
\begin{figure}[H]\centering
\begin{minipage}[t]{0.42\linewidth}\centering\includegraphics[width=\linewidth]{../results/models/mlp_training_curve.png}\caption{MLP training and validation loss per epoch (real history; train partition with an internal 10\,\% hold-out).}\label{fig:mlp}\end{minipage}\hfill
\begin{minipage}[t]{0.56\linewidth}\centering\includegraphics[width=\linewidth]{../results/models/quality_vs_speed.png}\caption{Quality vs inference cost (student\_test). KNN is two to three orders of magnitude slower.}\end{minipage}
\end{figure}
```

# Bayesian hyperparameter optimisation (XGBoost)

## Hyperparameters

The hyperparameters were optimised within the ranges below. The ranges include values on either side of the library defaults. The regularisation hyperparameters have wide ranges, as the untuned model had the greatest difference between train and test scores.

| Hyperparameter | Distribution | Range | Untuned | Best found |
|:-------------------|:------------------|:------------|--------:|--------:|
| `n_estimators` | integer, uniform | 100 – 700 | 300 | 408 |
| `learning_rate` | log-uniform | 0.02 – 0.3 | 0.1 | 0.0907 |
| `max_depth` | integer, uniform | 3 – 12 | 8 | 10 |
| `min_child_weight` | log-uniform | 1 – 30 | 1 | 1.170 |
| `subsample` | uniform | 0.5 – 1 | 1 | 0.607 |
| `colsample_bytree` | uniform | 0.5 – 1 | 1 | 0.654 |
| `reg_lambda` (λ) | log-uniform | 0.01 – 30 | 1 | 0.237 |
| `reg_alpha` (α) | uniform | 0 – 5 | 0 | 3.431 |
| `gamma` (γ) | uniform | 0 – 5 | 0 | 1.663 |

## Optimisation objective

Maximise the average top-1 accuracy on points over the same 5 folds as in Section 3, grouped by track. Student test data and evaluator data are not involved.

## Optimisation details

Optuna was used with the Tree-structured Parzen Estimator (TPE), seed 42. 60 trials took 14.0 minutes on the GPU. TPE separates completed trials into good and bad performers, fits a density function to each group, $l(x)$ and $g(x)$, and suggests new trials where $l(x)/g(x)$ is greatest, so that subsequent trials focus on the promising area. A MedianPruner was used: after 6 starting trials and 1 warm-up step, a trial was pruned when its running mean score after a fold was worse than the median of previous trials at that fold. 12 trials completed and 48 were pruned. The optimisation history is saved in `results/tuning/trials.csv` and `results/tuning/study.db`.

## Best parameters

Trial #23 gives the best performance (CV top-1: 0.8869); its parameters are given in the table above (`results/tuning/best_params.json`). The model was retrained on all 78 training tracks with these parameters (`artifacts/xgb_tuned.json`, `scripts/07_export_model.py`).

## Improvement in the tuned model?

| Model | CV macro F1 | CV top-1 | Test top-1 | Test macro F1 | Log loss | Inference ms/1k rows | Train time |
|:--------------------------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| XGBoost untuned | 0.9349 | 0.8857 | 0.8952 | 0.9393 | 0.0305 | 0.117 | 5.1 s |
| **XGBoost tuned (Bayesian)** | **0.9353** | **0.8869** | **0.8995** | **0.9412** | **0.0299** | 0.116 | 7.0 s |

Source: `results/tuning/tuned_vs_untuned.csv` (both models in the same run, with the same data and splits).

Yes, it improved on all metrics. Top-1 improved by 0.12 percentage points in CV and by 0.43 percentage points on student test data, with slightly better calibrated probabilities. Inference speed was unchanged. As most trials were very similar (within 0.003), the default parameters were already close to optimal: the model is more restricted by the information in the features than by its hyperparameters. Tuning increased tree depth, but held the trees back with stronger pruning and shrinkage (γ: 0 → 1.66, α: 0 → 3.43) and with row and column sampling. This agrees with the overfitting observed in Section 3.5.6. According to the completed trials (Figure \ref{fig:paramimp}), `reg_lambda` and `colsample_bytree` have the highest importance and `learning_rate` and `max_depth` the lowest; as only 12 trials completed, this is indicative.

```{=latex}
\begin{figure}[H]\centering
\begin{minipage}[t]{0.52\linewidth}\centering\includegraphics[width=\linewidth]{../results/tuning/optimization_history.png}\caption{Optimisation history: CV top-1 per trial and best so far (train, 5-fold grouped CV).}\end{minipage}\hfill
\begin{minipage}[t]{0.46\linewidth}\centering\includegraphics[width=\linewidth]{../results/tuning/param_importance.png}\caption{Hyperparameter importance (Optuna, completed trials).}\label{fig:paramimp}\end{minipage}
\end{figure}
```

![Tuned vs untuned XGBoost on student_test (axes zoomed to show the difference).](../results/tuning/tuned_vs_untuned.png){width=92%}

# HMM and Viterbi route reconstruction

The top candidates given by the classifier may be inconsistent. For example, points may indicate a main road, then a parallel road, and then the main road again, all within 2 seconds. Including sequential and network information in the model allows for more consistent routes. Code: `src/hmm.py`, `scripts/09_probs.py`, `scripts/10_hmm.py`. Log: `results/hmm/hmm_run.log`.

## State space

For GPS point $k$, the state is the arc the vehicle is currently on. Only the 8 highest-probability candidates from the classifier are considered. The true arc is among these 8 candidates for 99.8 % of student test points whose true arc is among the candidates at all. Arc ids are stored, so the arcs chosen by the Viterbi decoding can be mapped directly to road segments.

## Emissions

The log-probability from the tuned classifier of the candidate being the true arc, clipped from below at $10^{-6}$. Distance, bearing and the fit to neighbouring GPS points are already included in the classifier, so no geometric features are calculated in the emission.

## Transitions

Between arc $a$ (at time $k-1$, with fraction along the arc $t_a$) and arc $b$ (at time $k$):
$$\begin{aligned} R(a\to b) &= (1-t_a)\,\ell_a + d_{\text{net}}\big(\text{end}(a),\text{start}(b)\big) + t_b\,\ell_b \\ R(a\to a) &= (t_b-t_a)\,\ell_a \quad \text{(same arc)} \end{aligned}$$
where $d_{\text{net}}$ is the shortest network distance from the end of $a$ to the start of $b$, capped at 400 m (calculated with Dijkstra's algorithm; for distances longer than 400 m, the transition from $a$ to $b$ is considered unreachable), and $\delta$ is the Euclidean distance between the two GPS positions. Hence,
$$\begin{aligned} \text{cost}(a\to b) &= \min\!\Big(\tfrac{|R-\delta|}{\beta},\ \text{break\_pen}\Big) + \text{uturn\_pen}\cdot\big[\,b = \text{reverse}(a)\,\big] \\ \log T(a\to b) &= -\,w_t\cdot \text{cost}(a\to b) \end{aligned}$$
where unreachable pairs have a transition cost of break_pen. If the vehicle moved 20 m in a straight line, an arc that can be reached in about 20 m of driving is plausible; an arc that requires a long detour to reach is unlikely. Note that these are log-scores (costs), not normalised log-probabilities. This makes the model similar to the map-matching HMM of Newson and Krumm. The Viterbi algorithm maximises $\sum_k \big[\log p_k(s_k) + \log T(s_{k-1}\to s_k)\big]$ using
$$\text{score}_k(b) = \log p_k(b) + \max_a\big[\text{score}_{k-1}(a) - w_t\,\text{cost}(a\to b)\big],$$
where the best previous arc $a$ is stored as a back-pointer. Once the best overall score is found, the back-pointers are followed to trace back to point 0. The total complexity is $O(K\cdot 8^2)$, rather than $8^K$. As the shortest distances for a given trip are calculated once and stored (as in Fast Map Matching), the 182,465 training transitions take ~2 seconds to decode. The implementation was checked against an independent implementation (shortest distances via networkx and Viterbi decoding in plain Python): for 3 student test trips, the transition (route) distances and the decoded arcs matched exactly.

## Missing candidates and large gaps

- **No candidate within 75 m.** The point is skipped (the transition is made between the previous and the next point). The arc of the closest decoded point in time is then assigned to this point (`fill_missing`), so a prediction is made for every observation. There are 717 such points in the training data, and none in student_test.
- **Large gaps** ($\Delta t >$ gap_s). After a long pause between points, the vehicle may be more than 400 m away. Unreachable pairs then cost gap_pen rather than break_pen, and no transition costs more than gap_pen. No effect was observed: with the tuned gap_s of 299 s, 12 training trips were affected by this rule (none in student_test), and switching the rule off did not change the accuracy on these trips (still 0.9506). After a long gap all candidate pairs are equally unreachable, so all transitions have the same constant cost, and the argmax is unchanged. Essentially, after a gap, decoding proceeds in the same way with or without this rule.

## Consecutive observations on disconnected segments

Unreachable transitions are penalised with a high cost (tuned break_pen = 147.8), but they are not forbidden. Hence, even if there is a missing candidate, a disconnected network or a break in the true route, the decoder does not get stuck. The sequence of arcs is then converted into a route (`stitch_route`): if consecutive arcs are not directly connected, they are joined by the shortest network path (capped at 1.5 km). In student test, all joins could be made.

## Tuning

Optuna TPE optimisation was applied to the 6 decoding parameters (seed 42, 150 trials, 4.8 min), maximising the top-1 score across all 78 reliable training trips. Search space: $w_t$ (log) [0.2, 30], $\beta$ (log) [5, 200] m, break_pen [2, 200], uturn_pen [0, 40], gap_s [5, 600] s, gap_pen [0, 30]. None of the optimal values lie at the search-space boundaries. The classifier probabilities for the training trips were out-of-fold: each trip was scored by a 5-fold model that had never seen it (`scripts/09_probs.py`). Using probabilities from a model trained on the same trips would give overconfident probabilities, so the decoder would learn to put less weight on the map information. Trial 0, with the default parameters ($w_t$ = 1, $\beta$ = 30, break_pen = 8, uturn_pen = 4, gap_s = 20, gap_pen = 2), obtained 92.19 %. The best parameters ($w_t$ = 7.99, $\beta$ = 13.75 m, break_pen = 147.8, uturn_pen = 25.9, gap_s = 299 s, gap_pen = 24.6) obtained 94.35 %. Student test was not used in the tuning of the decoding parameters.

## Improvement in route quality?

| Partition | Method | Top-1 (ref. is a cand.) | Top-1 (all points) | Route contin. | Route F1 (matched arcs) | Route F1 (full .route) |
|:-------------|:------------------------------|-------:|-------:|-------:|-------:|-------:|
| student_test | classifier only | 0.8995 | 0.8985 | 0.9384 | 0.9289 | 0.8103 |
| student_test | classifier + HMM (default) | 0.9344 | 0.9334 | 0.9783 | 0.9698 | 0.9597 |
| student_test | **classifier + HMM (tuned)** | **0.9651** | **0.9640** | **0.9818** | **0.9719** | **0.9775** |
| train (out-of-fold) | classifier only | 0.8861 | 0.8798 | 0.9458 | 0.9118 | 0.8046 |
| train (out-of-fold) | classifier + HMM (tuned) | 0.9435 | 0.9360 | 0.9811 | 0.9641 | 0.9609 |

Source: `results/hmm/hmm_vs_classifier.csv`. The F1 score across the entire route (route F1, full .route) compares the stitched predicted route with all arcs in the true route. In the training rows, the decoder parameter settings are in-sample; on student test the estimate is clean.

Yes. With HMM decoding, the top-1 score increases by 6.6 percentage points, the continuity score by 4.3 percentage points, and the route F1 score improves from 0.810 to 0.978 on student test. The route F1 score improves the most because the classifier alone produces a route which alternates between twin roads and parallel roads; when this is stitched together, many incorrect arcs are added to the route. In contrast, the coherent route produced by decoding only requires real gaps to be filled in.

```{=latex}
\begin{figure}[H]\centering
\begin{minipage}[t]{0.6\linewidth}\centering\includegraphics[width=\linewidth]{../results/hmm/hmm_vs_classifier.png}\caption{Classifier only vs classifier + HMM (default and tuned), train (out-of-fold) and student\_test.}\end{minipage}\hfill
\begin{minipage}[t]{0.38\linewidth}\centering\includegraphics[width=\linewidth]{../results/hmm/per_trip_student_test.png}\caption{Per-trip top-1 on student\_test, classifier only vs + HMM (tuned).}\end{minipage}
\end{figure}
```

## Where HMM decoding helped and where it did not

- **Student test:** all 10 trips improved (median +3.3 points). The largest improvement was in trip 77 (78.2 % → 95.9 %), the smallest in trip 0 (+1.8 points).
- **Training data (out-of-fold):** 74 of 78 trips improved. Performance worsened on trip 63 (−3.5 points), 66 (−1.7), 22 (−1.3) and 97 (−0.5). In trip 66 there is a section of 202 points with no true route, which affects the labelling. For trips 63, 22 and 97 it is unclear why performance worsened; it is plausible that the strong network prior smoothed over real short detours.
- **Gap rule:** no noticeable effect (see "Missing candidates and large gaps").

# Geographical visualisation using Folium

Maps are created in `src/viz.py` (`build_trip_map`), using `scripts/02_trip_map.py` and `scripts/10_hmm.py`, and are also displayed in the UI (`ui/app.py`). Multiple base maps are available (OpenStreetMap, light grey, satellite). Different layers represent the road network, the true route, the predicted route (dashed line with direction arrows) and the raw GPS points; start and end markers are also present. A legend, a layer selector and a scale bar are included, and the map is zoomed to the extent of the trip. The interactive HTML files have been committed: `results/eda/trip_00000003_map.html`, `results/eda/trip_00000033_map.html`, `results/hmm/map_student_test_00000077.html`.

![Student-test trip 77 (Syktyvkar): raw GPS (blue), road network (grey), reference route (green) and HMM-predicted route (red) on the satellite base layer. This is the trip where the HMM helped most (78.2 % → 95.9 % top-1).](img/folium_hmm_trip77.png){width=100%}

# API, UI and deployment

## Architecture

Laptop 1 hosts the FastAPI backend (`api/main.py`). The model and configuration (`artifacts/`), the road networks and the true routes are stored here; the full pipeline is run (`src/pipeline.py`) and the results are scored on the backend (`src/scoring.py`). Laptop 2 hosts the Streamlit UI (`ui/app.py`), which only requires the dependencies in `ui/requirements.txt`; no model or data is needed. Endpoints: `GET /health`, `GET /sets`, `GET /trips?trip_set=...`, and `POST /predict` (body: `{"trip_set", "trip_id", "include_network"}`).

## Requests

The UI lists the trips available in the API, the evaluator selects a trip, and the UI sends its trip id to the API. The API checks that it is valid (unknown ids and attempts to access invalid paths return 404), loads the raw files and makes the prediction using only the GPS track and the network. The prediction is then compared to the true route, and the route geometry, snapped points, metrics and times are returned. True arcs are never sent outside Laptop 1 and are never seen by the prediction code. All requests are logged in the terminal on Laptop 1, recording the client address, trip, sizes, metrics and duration. In the two-laptop test (see Figure \ref{fig:apilog}), all 10 student test trips were requested from Laptop 2 (10.1.34.101). The time taken per request ranged from 0.07 to 1.05 s, and the metrics match the offline results exactly.

## Speed

The average pipeline time per evaluator trip is 0.28 s (CPU), and the longest is 0.65 s. For trip 77 (3,911 points) the times were 0.19 s (candidate selection + feature engineering), 0.10 s (XGBoost), 0.30 s (HMM) and 0.004 s (stitching).

## Deployment

Laptop 1: `pip install -r requirements.txt`, then `uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-access-log`. Laptop 2: `pip install -r ui/requirements.txt`, then `streamlit run ui/app.py`, and enter `http://<laptop-1-ip>:8000` as the API address. Both laptops must be connected to the same network, with port 8000 open.

![UI on Laptop 2: evaluator trip set, trip bar, metrics returned by the API and the Folium map of evaluator trip 15.](img/ui_eval_map.png){width=100%}

![UI "All trips & conclusions" tab after running all 10 evaluator trips through the API.](img/ui_eval_conclusions.png){width=100%}

![API terminal on the receiving laptop (Laptop 1, `uvicorn` output) during the two-laptop run: start-up lines listing both trip sets, then the UI on Laptop 2 (10.1.34.101) running all 10 student-test trips; each request prints a PREDICT line, a RESULT line with the server-side metrics, and the request line with Laptop 2's address. Full log: `results/api/two_laptop_demo_laptop1.log`.](img/api_log.png){#fig:apilog width=100%}

# Final evaluation, issues and conclusions

## Evaluation trips

Once the classifier hyperparameters and the decoding parameters were fixed, the 10 evaluation trips were run once through the pipeline (`scripts/11_check_pipeline.py`). The results are saved in `results/pipeline/check_evaluator_test.csv`.

| Trip | Points | Top-1 classifier only | Top-1 + HMM | Continuity | Route F1 | Time (s) |
|:-----------|-------:|-------:|-------:|-------:|-------:|-------:|
| 00000015 | 6,854 | 0.6584 | 0.9205 | 0.9902 | 0.8171 | 0.65 |
| 00000017 | 1,346 | 0.9376 | 0.9539 | 0.8848 | 0.9775 | 0.38 |
| 00000023 | 458 | 0.9432 | 0.9716 | 0.9847 | 0.9516 | 0.13 |
| 00000030 | 654 | 0.8012 | 0.8379 | 0.9005 | 0.9607 | 0.22 |
| 00000049 | 2,642 | 0.9902 | 0.9962 | 0.9947 | 0.9988 | 0.24 |
| 00000059 | 1,031 | 0.9670 | 0.9816 | 0.9272 | 0.9774 | 0.23 |
| 00000067 | 916 | 0.9509 | 0.9356 | 0.9891 | 1.0000 | 0.10 |
| 00000073 | 1,144 | 0.9825 | 0.9843 | 0.9921 | 0.9958 | 0.16 |
| 00000087 | 1,399 | 0.9535 | 0.9957 | 0.9900 | 0.9989 | 0.24 |
| 00000094 | 3,230 | 0.9276 | 0.9508 | 0.9724 | 0.9789 | 0.44 |
| **Pooled / mean** | **19,674** | **0.8473** | **0.9493** |  | **0.9657** | **0.28** |

The evaluation trips were unseen, and still achieved a high accuracy (94.9 %) compared to student test (96.5 %). Accuracy with only the classifier is noticeably lower than on student test (84.7 % vs 90.0 %), largely due to trip 15 (6,854 points, 65.8 %); however, 26 points were recovered in this trip using the HMM. Performance was poorer on trip 30 (top-1 = 83.8 %, continuity = 0.90), trip 15 (route F1 = 0.82), trip 17 (continuity = 0.88) and trip 67 (top-1 reduced by the HMM from 95.1 % to 93.6 %, despite a route F1 of 1.0).

## Issues

- True routes contain inaccuracies (unreliable trips 33 and 54, missing sections in trips 2 and 66, and small breaks in 11 trips), so there is some noise in the labelling and scoring.
- 0.8 % of points in reliable trips have no correct candidate within 75 m, so they are incorrect by design.
- The transition model only considers the difference between the route length and the straight-line distance; speed limits, turn restrictions and road type are not considered.
- The scores used in decoding are not normalised probabilities, and the long-gap rule has no measurable effect on this data.
- Bayesian tuning gave only a slight improvement to the classifier, and the 4 best classifiers differ by less than the fold-to-fold variation in their scores.

## Conclusions

This dataset works well with map matching using candidate classification and decoding. Directional features contribute heavily to the high accuracy of the classifier (removing them reduces accuracy by 23 points), while the choice between strong tabular models contributes much less. After the features, the HMM gives the largest improvement: accuracy increased from 90.0 % to 96.5 % (observation level) and route F1 from 0.81 to 0.98 on student test, and this improvement generalises to the unseen evaluation trips (94.9 %). The full pipeline runs quickly (under 1 s per trip on a CPU) and has been deployed as an API and UI across two laptops, with the true data remaining on the server.
