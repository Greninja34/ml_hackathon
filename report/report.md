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

1. **Candidate generation:** for each GPS point in a track, candidates are generated from all arcs within a 75 m radius (found with an R-tree).
2. **Feature engineering:** 18 relative features are calculated for each GPS point-candidate arc pair.
3. **Classification:** a model predicts the probability of the candidate being the true arc. 7 models were considered, and XGBoost was chosen and optimized using Bayesian optimization.
4. **Decoding:** an HMM is built over the candidate arcs, with transition scores based on distances between arcs in the graph and emission scores based on the probabilities output by the classifier. Its parameters are tuned, not trained. This gives one arc for each GPS point, and the arcs can be concatenated to give a continuous route.

![Prediction pipeline (`src/pipeline.py`, the same function the API runs).](img/pipeline.png){width=100%}

# Dataset understanding and exploratory data analysis

The data was split into training (80 tracks), student test (10 tracks) and evaluator data (10 tracks). Fitting, cross-validating and optimizing models was done using only training data. Evaluator data was not opened until the final evaluation (Section 8). The data split is indicated in the figure captions.

## Data overview, structure and quality

```{=latex}
\begin{rtable}
\begin{tabularx}{\linewidth}{@{}L r@{}}
\toprule
\textbf{Item} & \textbf{Value} \\
\midrule
Trips (train / student\_test / evaluator) & 80 / 10 / 10 \\
GPS points (train + student\_test) & 227,577 \\
Points per trip (min / median / max) & 262 / 1,496 / 19,313 \\
Arcs per trip (min / median / max) & 649 / 18,761 / 307,392 \\
Route arcs per trip (min / median / max) & 59 / 272 / 2,266 \\
Steps exactly 1 s apart (mean over trips) & 94.3 \% \\
Trips with a time gap $>$ 60 s & 23 \\
Largest time gap & 153,064 s \\
Arcs with a reverse twin (mean over trips) & 92.2 \% \\
Weakly connected components per trip (median / max) & 19 / 421 \\
Largest-component share of nodes (min over trips) & 0.693 \\
Reference route fully connected (min over trips) & 0.9813 \\
\bottomrule
\end{tabularx}
\end{rtable}
```

*Source: `results/eda/summary.md` and `results/eda/inventory.csv` (`scripts/01_eda.py`), training and student test data.*

### Missingness and validity

```{=latex}
\begin{rtable}
\begin{tabularx}{\linewidth}{@{}L r L@{}}
\toprule
\textbf{Check} & \textbf{Result} & \textbf{Decision} \\
\midrule
NaN values in tracks / nodes & 0 / 0 & none needed \\
Longitude / latitude out of range & 0 & none needed \\
Duplicate / non-monotonic timestamps & 0 / 0 & none needed \\
Arcs with a non-existent node, route arcs with a non-existent arc & 0 / 0 & none needed \\
Zero-length arcs & 106 & never used as candidates (no geometry) \\
Isolated nodes & 0 & none needed \\
Time gaps $>$ 60 s & 23 trips & kept (real behavior), time-gap features, handled in the HMM \\
GPS points with no arc within 75 m & 13,572 (6.0 \%) & 9 trips (12,855 in trip 54), none in student\_test, filled after decoding (Section 5) \\
Reference route far from the GPS (median $>$ 15 m) & trips 33, 54 & flagged unreliable, excluded from training \\
\bottomrule
\end{tabularx}
\end{rtable}
```

The data is complete and internally valid. However, the real data has other quality issues:

- noisy GPS positions
- time gaps between points
- disconnected graphs
- reference routes that do not always follow the GPS points

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

- **Projection to a local metric coordinate system:** longitudes and latitudes are angular units. At 60° N, one degree of longitude corresponds to approx. 56 km, compared to 111 km at the equator. As tracks occurred in different countries, a local metric projection was performed on each track using the mean GPS position as a reference point. For a track with mean latitude $\varphi_0$ and longitude $\lambda_0$, $x = (\lambda-\lambda_0)\,R\cos\varphi_0$ and $y = (\varphi-\varphi_0)\,R$, where $R = 6{,}371{,}008.8$ m. For distances up to a few km this introduces negligible error (well below the GPS error of $\sim 5$ m), and no UTM zone has to be chosen for individual tracks (`src/data_io.py`).
- **Label assignment:** `.route` only lists the arc ids of the route, not which arc each GPS point is on. For example, track 1 has 1,315 GPS points and 525 arcs. Using dynamic programming, each GPS point is matched to a position along the route. The match must move in the forward direction (never backwards), and incurs a cost of the distance to the arc + 0.05 m for each skipped arc. All 90 tracks are matched forward, and typically have point-to-arc distances of 4–7 m on normal tracks. These labels are only used for training and scoring (`src/labeling.py`).
- **Reliability:** tracks with a median distance between GPS and reference > 15 m are removed from training (33 with 122 m, and 54 with 140 m). Labeling these points would introduce noise.
- **Candidate arcs:** all arcs within radius R of the given point, found with a shapely STRtree. R = 75 m (radius sweep, see Figure \ref{fig:radius}).
- **Training data:** only the reliable tracks, with true arcs among the candidates: 2,880,132 rows across 181,659 points (6.3 % positive). If the true arc is not among the candidates, it counts as a missed candidate, and not a model error.
- **Scaling:** models requiring scaling (Logistic Regression, Naive Bayes, KNN, MLP) have a StandardScaler fitted on the training data within their pipeline.

## Observations

```{=latex}
\begin{figure}[H]\centering
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/eda/trip_00000003_matplotlib.png}\caption{GPS trajectory and road network of a typical trip (train, trip 3). The reference route follows the GPS.}\end{minipage}\hfill
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/eda/trip_00000033_matplotlib.png}\caption{Trip 33 (train). The reference route covers only fragments of the driven loop, so the trip is flagged unreliable.}\end{minipage}
\end{figure}
\begin{figure}[H]\centering
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/eda/hist_noise_all_points.png}\caption{Distance from each GPS point to the reference route (train + student\_test): p50 5.6 m, p90 48.8 m, p95 137.7 m, p99 200 m.}\end{minipage}\hfill
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/eda/hist_noise_median.png}\caption{Median GPS-to-reference distance per trip. The two outliers are trips 33 and 54.}\end{minipage}
\end{figure}
```

- **Long tails:** half of the points are less than 5.6 m from the route, but 1 in 10 points is $\sim 50$ m away (likely to be confused with parallel roads).
- **Directed duplicates:** 92.2 % of arcs have an exact duplicate in the opposite direction, at the same distance from all points. Therefore choosing the closest arc gives only **58.6 %** top-1 accuracy on student test. Applying a manually designed rule of 'choose the closest arc, and prioritize based on direction' gives **85.4 %**. Direction is important.
- **Sampling:** data is mostly sampled at 1 Hz. Consecutive points are very similar, so data must be split by track, never by point.
- **Disconnected graphs:** median 19 weak components per track, and small disconnections within the reference route in some cases (minimum proportion of connected route = 0.981). Therefore the decoder must allow transitions between disconnected arcs.

![Interactive Folium map of train trip 3 (`results/eda/trip_00000003_map.html`, made by `scripts/02_trip_map.py`) showing raw GPS observations, road network and reference route, with switchable base and data layers. Partition: train.](img/folium_eda_trip3.png){width=96%}

![Candidate-radius sweep (`results/candidates/radius_summary.csv`, train + student_test, reliable trips in green). At 75 m the reference arc is missing for 0.8 % of points of reliable trips (2.3 % at 30 m, 1.3 % at 50 m, 0.6 % at 100 m) at about 16 candidates per point.](../results/candidates/radius_sweep.png){#fig:radius width=88%}

## Features

Only relative geometric and kinematic features were used:

- arc ids and absolute positions were excluded, as the tracks come from different cities and these values would not be transferable
- the same code is used for training and within the API (`src/features.py`), and it never reads in labels
- the features were decided before the ablation study, so no feature selection was done using student test data

```{=latex}
\begin{rtable}
\begin{tabularx}{\linewidth}{@{}l >{\raggedright\arraybackslash}p{0.27\linewidth} L@{}}
\toprule
\textbf{Group} & \textbf{Features} & \textbf{Reason} \\
\midrule
Position (5) & \texttt{dist}, \texttt{t\_frac}, \texttt{dist\_minus\_min}, \texttt{dist\_rank}, \texttt{n\_cands} & How close the arc is, and how close relative to the other candidates of the same point. Relative closeness transfers across cities. \\
Direction (4) & \texttt{cos\_heading}, \texttt{heading\_ok}, \texttt{ds\_span}, \texttt{has\_reverse} & Separates an arc from its reverse twin: cosine between the motion over points $k-2 \ldots k+2$ and the arc direction, heading trusted only if the vehicle moved ≥ 3 m within ≤ 10 s, forward progress of neighboring points along the arc. \\
Fit to trail (4) & \texttt{d\_prev}, \texttt{d\_next}, \texttt{d\_mean\_win}, \texttt{speed\_ms} & Does the arc also fit the neighboring points (±2 and an 11-point window)? One noisy point cannot pull the decision. \\
Road shape (3) & \texttt{arc\_len}, \texttt{deg\_out\_end}, \texttt{deg\_in\_start} & Segment length and junction degree. \\
Time (2) & \texttt{dt\_prev}, \texttt{dt\_next} & Marks points next to a time gap. \\
\bottomrule
\end{tabularx}
\end{rtable}
```

```{=latex}
\begin{figure}[H]\centering
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/features/feature_ablation.png}\caption{Ablation: student\_test top-1 after removing each group and retraining (tuned XGBoost). Direction $-23.3$ points, position $-7.1$, road shape $-1.5$, fit to trail $-1.0$, time $-0.3$.}\end{minipage}\hfill
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/features/feature_importance.png}\caption{XGBoost gain importance (share of total): dist\_minus\_min 0.29, dist\_rank 0.20, ds\_span 0.19, dist 0.09.}\end{minipage}
\end{figure}
```

![Distributions of key features for reference arcs vs other candidates (train, reliable trips, points whose reference arc is a candidate).](../results/features/feature_separation.png){width=68%}

# Model experimentation and comparison

## Approach

- All models were trained on the same data, with the same seed (42) and the same folds.
- 5-fold `GroupKFold` was performed with groups defined as tracks, over the 78 reliable tracks: approx. 575,000 rows and 15–16 tracks in each fold.
- Each model was then retrained on all 78 tracks, and evaluated once on the student test data (19,139 points, 356,362 candidate rows).
- Code: `scripts/05_models.py`. Experiment log: `results/models/experiment_run.log`.

## Measures

- **Top-1 accuracy on points:** the main measure, as it represents what the system does.
- **Macro F1, precision and recall:** calculated on candidate rows with a probability threshold of 0.5. These give equal weighting to the rare positive class (6 %).
- **Log loss:** shows how well probabilities are predicted (important for the log-probabilities used in the HMM).
- **Continuity:** the proportion of consecutive chosen arcs that are the same or directly connected.
- **Missed candidates:** all models have the same rate (0.8 %) for R = 75 m.

## Summary of model performance

```{=latex}
\begin{rtable}
\begin{tabularx}{\linewidth}{@{}L l r r r r r r@{}}
\toprule
\textbf{\hd{Model\\(key configuration)}} & \textbf{\hd{Algorithm\\family}} & \textbf{\hd{CV macro\\F1 (std)}} & \textbf{\hd{Top-1\\accuracy}} & \textbf{\hd{Route\\continuity}} & \textbf{\hd{Test\\macro F1}} & \textbf{\hd{Log\\loss}} & \textbf{\hd{Inference\\ms / 1k rows}} \\
\midrule
\textbf{XGBoost}\newline{\itshape\color{muted}300 trees, lr 0.1, depth 8, hist, GPU} & Boosting & 0.9349 (0.023) & \textbf{0.8952} & \textbf{0.9387} & \textbf{0.9393} & \textbf{0.0305} & 0.19 \\
\textbf{Random Forest}\newline{\itshape\color{muted}100 trees, depth 16, min leaf 5, 25 \% bootstrap} & Bagging & 0.9342 (0.022) & 0.8886 & 0.9305 & 0.9364 & 0.0314 & 0.70 \\
\textbf{LightGBM}\newline{\itshape\color{muted}300 trees, lr 0.1, 63 leaves} & Boosting & 0.9330 (0.023) & 0.8889 & 0.9240 & 0.9377 & 0.0356 & 2.19 \\
\textbf{MLP}\newline{\itshape\color{muted}64-32 ReLU, Adam 1e-3, batch 4096, early stopping, GPU} & Neural network & 0.9299 (0.021) & 0.8871 & 0.9374 & 0.9340 & 0.0330 & 0.03 \\
\textbf{Logistic Regression}\newline{\itshape\color{muted}C = 1, lbfgs, scaled inputs} & Linear & 0.9152 (0.021) & 0.8734 & 0.9267 & 0.9175 & 0.0440 & 0.08 \\
\textbf{KNN}\newline{\itshape\color{muted}k = 25, distance-weighted, 8,000 training points} & Instance-based & 0.8992 (0.020) & 0.8353 & 0.9197 & 0.8997 & 0.0576 & 108.27 \\
\textbf{Gaussian Naive Bayes}\newline{\itshape\color{muted}var\_smoothing 1e-9, scaled inputs} & Probabilistic & 0.7502 (0.010) & 0.7953 & 0.9214 & 0.7421 & 0.5412 & 0.10 \\
\bottomrule
\end{tabularx}
\end{rtable}
```

*Source: `results/models/model_comparison.csv`. Top-1, continuity, macro F1 and log loss are measured on student test data.*

- **Training times (full re-fit on all 78 tracks):** XGBoost 5.9 s, Random Forest 72.6 s, LightGBM 23.9 s, MLP 19.7 s, Logistic Regression 10.6 s, KNN 0.03 s, Naive Bayes 0.7 s.
- **Cross-validation top-1 accuracy:** XGBoost 0.8857, Random Forest 0.8827, MLP 0.8797, LightGBM 0.8786, Logistic Regression 0.8749, KNN 0.8401, Naive Bayes 0.8307.
- All metrics are given for all models, as all of them produce probabilities for candidate arcs.

## Model selection

- As the standard deviation of macro F1 scores across CV folds ($\sim 0.02$) is larger than the differences between the top 4 models, they are statistically tied in terms of F1 score.
- XGBoost was chosen for having the highest CV macro F1 and top-1 score, the best log loss (important for the HMM emissions), the best continuity score, and the quickest training time of the strong models.
- This decision was made based on CV performance, and is confirmed by the performance on student test data.
- The variance in model scores between folds is mostly due to which tracks are in each fold: all models except Naive Bayes have their lowest scores in fold 1 and highest in fold 4. XGBoost has top-1 scores per fold of 0.825, 0.863, 0.918, 0.938 and 0.885.

## Model answers

Each model below covers its suitability, strengths and limitations, its hyperparameters and objective, and its results. 'Train top-1' refers to a sample of 15,000 points from the training data, scored with the final model.

```{=latex}
\begin{adjustwidth}{1.5em}{0pt}
```

### Logistic Regression

- **Suitability:** a linear combination of the 18 features, passed through a sigmoid function. This is a natural baseline for a binary classification problem (is this the correct arc?), and it outputs proper probabilities.
- **Strengths:** fast (10.6 s).
- **Limitations:** it cannot model interaction effects. For example, direction should only be considered if the vehicle is moving.
- **Hyperparameters:** C = 1.0 (inverse of the L2 regularization strength), solver = lbfgs, max_iter = 300, standardized inputs.
- **Objective:** minimizes binary log loss with L2 regularization.
- **Results:** top-1 0.8734, macro F1 0.9175 (precision 0.9173, recall 0.9176), log loss 0.0440.
- **Fit:** slight underfitting, as the training (0.8765) and test (0.8734) scores are similar, and quite low.

### Gaussian Naive Bayes

- **Suitability:** a Gaussian distribution is fitted for each feature and class, and the probabilities are multiplied (independence assumption). A cheap probabilistic baseline from a different family.
- **Strengths:** very fast training (< 1 s).
- **Limitations:** the features are highly correlated (`dist`, `d_prev` and `d_mean_win` are all related to distance), so the same evidence is weighted multiple times, giving overconfident probabilities.
- **Hyperparameters:** var_smoothing = 1e-9, standardized inputs.
- **Objective:** maximizes the likelihood of the Gaussian distributions per class (analytical solution).
- **Results:** top-1 0.7953, macro F1 0.7421 (precision 0.6834, recall 0.9404), log loss 0.5412 (around 10× poorer than the other models).
- **Fit:** underfitting due to violated assumptions. The test score (0.7953) is much lower than the training score (0.8301), and the poorly calibrated probabilities would mislead the HMM.

### K-Nearest Neighbors

- **Suitability:** for each candidate row, the nearest 25 rows in the training data are found and give a distance-weighted vote.
- **Strengths:** no training required, flexible decision boundary.
- **Limitations:** time-consuming (38.6 s on student test data, as it takes 108 ms per 1,000 rows), and distance in 18 mixed features gives a poor measure of similarity. It was trained on 8,000 points; for the full data of 2.88 M rows, prediction would take hours.
- **Hyperparameters:** k = 25, weighted by distance, standardized inputs, 8,000 training points.
- **Objective:** no loss function (lazy learner).
- **Results:** top-1 0.8353, macro F1 0.8997 (precision 0.9218, recall 0.8801), log loss 0.0576.
- **Fit:** because of the sub-sampled training data, the test score (0.8353) is 3.2 points lower than the training score (0.8676), i.e. mild overfitting.

### Random Forest

- **Suitability:** many decision trees are averaged. Each tree is trained on a bootstrapped sample of the data, with randomly selected features, and trees learn conditional relationships in the data without scaling.
- **Strengths:** stable, good probabilities.
- **Limitations:** slowest to train (72.6 s on 32 CPU threads), large.
- **Hyperparameters:** n_estimators = 100, max_depth = 16, min_samples_leaf = 5, max_samples = 0.25.
- **Objective:** Gini impurity as the split criterion.
- **Results:** top-1 0.8886, macro F1 0.9364 (precision 0.9476, recall 0.9259), log loss 0.0314.
- **Fit:** slight overfitting. The test score is 1.7 points lower than the training score, a healthy level.

### LightGBM

- **Suitability:** gradient boosting with leaf-wise tree growth. Each tree fits the residual errors of the previous trees.
- **Strengths:** accurate, fast on CPU (23.9 s).
- **Limitations:** overfits without regularization, many hyperparameters.
- **Hyperparameters:** n_estimators = 300, learning_rate = 0.1, num_leaves = 63.
- **Objective:** binary log loss.
- **Results:** top-1 0.8889, macro F1 0.9377 (precision 0.9412, recall 0.9342), log loss 0.0356.
- **Fit:** moderate overfitting (the test score is 2.3 points lower than the training score). Slightly worse probabilities than XGBoost.

### XGBoost (chosen) {#xgboost-chosen}

- **Suitability:** gradient boosting with second-order updates, explicit regularization (γ per leaf, L1/L2 regularization on leaf weights), and level-wise tree growth. Interaction effects are important for this problem (direction only matters if moving, distance matters relative to the other candidates), and trees can easily represent interaction effects. The log-loss objective means the probabilities can be used as emissions in the HMM.
- **Strengths:** best scores (CV and test), best log loss and continuity, fast (5.9 s on GPU).
- **Limitations:** the greatest difference between train and test scores, so regularization is required.
- **Hyperparameters:** initial n_estimators = 300, learning_rate = 0.1, max_depth = 8, tree_method = hist, device = cuda. The other parameters use their default values (min_child_weight = 1, subsample = 1, colsample_bytree = 1, λ = 1, α = 0, γ = 0).
- **Objective:** `binary:logistic`, the sum of the binary log loss and, for each tree, $\gamma T + \tfrac12\lambda\lVert w\rVert^2 + \alpha\lVert w\rVert_1$.
- **Results:** untuned top-1 0.8952, macro F1 0.9393 (precision 0.9466, recall 0.9322), log loss 0.0305. Tuned (see Section 4): top-1 0.8995, macro F1 0.9412, log loss 0.0299.
- **Fit:** greatest overfitting of the group (5.2 point difference between train and test scores), which is consistent with tuning moving toward more regularization and subsampling.

### Multi-layer perceptron

- **Suitability:** a feedforward neural network (18 → 64 → 32 → 1, ReLU activation) implemented in PyTorch (`src/torch_mlp.py`) and trained on the GPU. It can learn complex interaction effects.
- **Strengths:** smooth non-linear decision boundary, fast inference (0.03 ms per 1,000 rows).
- **Limitations:** requires scaling, sensitive to parameter choices, and rarely outperforms gradient-boosted trees on tabular data.
- **Hyperparameters:** hidden layers 64 → 32, ReLU activation, Adam (lr = 1e-3), batch size 4,096, at most 40 epochs, early stopping after 5 epochs of no improvement on an internal validation split (10 % of the data), standardized inputs.
- **Objective:** binary cross-entropy.
- **Results:** top-1 0.8871, macro F1 0.9340 (precision 0.9443, recall 0.9243), log loss 0.0330.
- **Fit:** generalizes well (only a 0.9 point difference between train and test scores). It is the only model with a per-epoch history (see Figure \ref{fig:mlp}).

```{=latex}
\end{adjustwidth}
```

## Visualizations of models

```{=latex}
\begin{figure}[H]\centering
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/models/cv_macro_f1.png}\caption{Cross-validation macro F1 per model (5 folds grouped by trip, train). Error bars show the standard deviation.}\end{minipage}\hfill
\begin{minipage}[t]{0.49\linewidth}\centering\includegraphics[width=\linewidth]{../results/models/test_log_loss.png}\caption{Student-test log loss (lower is better).}\end{minipage}
\end{figure}
```

![Student-test top-1 accuracy and macro F1 per model.](../results/models/test_top1_f1.png){width=92%}

![Candidate-level confusion matrices at threshold 0.5 (student_test). Naive Bayes trades many false positives for recall.](../results/models/confusion_matrices.png){width=96%}

```{=latex}
\begin{figure}[H]\centering
\begin{minipage}[t]{0.42\linewidth}\centering\includegraphics[width=\linewidth]{../results/models/mlp_training_curve.png}\caption{MLP training and validation loss per epoch (real history, train partition with an internal 10\,\% hold-out).}\label{fig:mlp}\end{minipage}\hfill
\begin{minipage}[t]{0.56\linewidth}\centering\includegraphics[width=\linewidth]{../results/models/quality_vs_speed.png}\caption{Quality vs inference cost (student\_test). KNN is two to three orders of magnitude slower.}\end{minipage}
\end{figure}
```

# Bayesian hyperparameter optimization (XGBoost)

## Hyperparameters

The hyperparameters were optimized within the ranges below:

- the ranges include values on either side of the library defaults
- the regularization hyperparameters have wide ranges, as the untuned model had the greatest difference between train and test scores

```{=latex}
\begin{rtable}
\begin{tabularx}{\linewidth}{@{}L l l r r@{}}
\toprule
\textbf{Hyperparameter} & \textbf{Distribution} & \textbf{Range} & \textbf{Untuned} & \textbf{Best found} \\
\midrule
\texttt{n\_estimators} & integer, uniform & 100 -- 700 & 300 & 408 \\
\texttt{learning\_rate} & log-uniform & 0.02 -- 0.3 & 0.1 & 0.0907 \\
\texttt{max\_depth} & integer, uniform & 3 -- 12 & 8 & 10 \\
\texttt{min\_child\_weight} & log-uniform & 1 -- 30 & 1 & 1.170 \\
\texttt{subsample} & uniform & 0.5 -- 1 & 1 & 0.607 \\
\texttt{colsample\_bytree} & uniform & 0.5 -- 1 & 1 & 0.654 \\
\texttt{reg\_lambda} (λ) & log-uniform & 0.01 -- 30 & 1 & 0.237 \\
\texttt{reg\_alpha} (α) & uniform & 0 -- 5 & 0 & 3.431 \\
\texttt{gamma} (γ) & uniform & 0 -- 5 & 0 & 1.663 \\
\bottomrule
\end{tabularx}
\end{rtable}
```

## Optimization objective

Maximize the average top-1 accuracy on points over the same 5 folds as in Section 3, grouped by track. Student test data and evaluator data are not involved.

## Optimization details

- **Method:** Optuna with the Tree-structured Parzen Estimator (TPE), seed 42. 60 trials took 14.0 minutes on the GPU.
- **How TPE works:** it separates completed trials into good and bad performers, fits a density function to each group, $l(x)$ and $g(x)$, and suggests new trials where $l(x)/g(x)$ is greatest, so that subsequent trials focus on the promising area.
- **Pruning:** a MedianPruner was used. After 6 starting trials and 1 warm-up step, a trial was pruned when its running mean score after a fold was worse than the median of previous trials at that fold. 12 trials completed and 48 were pruned.
- **Search record:** saved in `results/tuning/trials.csv` and `results/tuning/study.db`.

## Best parameters

Trial #23 gives the best performance (CV top-1: 0.8869). Its parameters are given in the table above (`results/tuning/best_params.json`). The model was retrained on all 78 training tracks with these parameters (`artifacts/xgb_tuned.json`, `scripts/07_export_model.py`).

## Improvement in the tuned model?

```{=latex}
\begin{rtable}
\begin{tabularx}{\linewidth}{@{}L r r r r r r r@{}}
\toprule
\textbf{Model} & \textbf{\hd{CV\\macro F1}} & \textbf{\hd{CV\\top-1}} & \textbf{\hd{Test\\top-1}} & \textbf{\hd{Test\\macro F1}} & \textbf{\hd{Log\\loss}} & \textbf{\hd{Inference\\ms / 1k rows}} & \textbf{\hd{Train\\time}} \\
\midrule
XGBoost untuned & 0.9349 & 0.8857 & 0.8952 & 0.9393 & 0.0305 & 0.117 & 5.1 s \\
\textbf{XGBoost tuned (Bayesian)} & \textbf{0.9353} & \textbf{0.8869} & \textbf{0.8995} & \textbf{0.9412} & \textbf{0.0299} & 0.116 & 7.0 s \\
\bottomrule
\end{tabularx}
\end{rtable}
```

*Source: `results/tuning/tuned_vs_untuned.csv` (both models in the same run, with the same data and splits).*

Yes, it improved on all metrics:

- top-1 improved by 0.12 percentage points in CV and by 0.43 percentage points on student test data, with slightly better calibrated probabilities, and inference speed was unchanged
- as most trials were very similar (within 0.003), the default parameters were already close to optimal: the model is more restricted by the information in the features than by its hyperparameters
- tuning increased tree depth, but held the trees back with stronger pruning and shrinkage (γ: 0 → 1.66, α: 0 → 3.43) and with row and column sampling, which agrees with the overfitting observed in Section \ref{xgboost-chosen}
- according to the completed trials (Figure \ref{fig:paramimp}), `reg_lambda` and `colsample_bytree` have the highest importance and `learning_rate` and `max_depth` the lowest. As only 12 trials completed, this is indicative.

```{=latex}
\begin{figure}[H]\centering
\begin{minipage}[t]{0.52\linewidth}\centering\includegraphics[width=\linewidth]{../results/tuning/optimization_history.png}\caption{Optimization history: CV top-1 per trial and best so far (train, 5-fold grouped CV).}\end{minipage}\hfill
\begin{minipage}[t]{0.46\linewidth}\centering\includegraphics[width=\linewidth]{../results/tuning/param_importance.png}\caption{Hyperparameter importance (Optuna, completed trials).}\label{fig:paramimp}\end{minipage}
\end{figure}
```

![Tuned vs untuned XGBoost on student_test (axes zoomed to show the difference).](../results/tuning/tuned_vs_untuned.png){width=92%}

# HMM and Viterbi route reconstruction

The top candidates given by the classifier may be inconsistent. For example, points may indicate a main road, then a parallel road, and then the main road again, all within 2 seconds. Including sequential and network information in the model allows for more consistent routes. Code: `src/hmm.py`, `scripts/09_probs.py`, `scripts/10_hmm.py`. Log: `results/hmm/hmm_run.log`.

## State space

- For GPS point $k$, the state is the arc the vehicle is currently on.
- Only the 8 highest-probability candidates from the classifier are considered. The true arc is among these 8 candidates for 99.8 % of student test points whose true arc is among the candidates at all.
- Arc ids are stored, so the arcs chosen by the Viterbi decoding can be mapped directly to road segments.

## Emissions

The log-probability from the tuned classifier of the candidate being the true arc, clipped from below at $10^{-6}$. Distance, bearing and the fit to neighboring GPS points are already included in the classifier, so no geometric features are calculated in the emission.

## Transitions

Between arc $a$ (at time $k-1$, with fraction along the arc $t_a$) and arc $b$ (at time $k$):
$$\begin{aligned} R(a\to b) &= (1-t_a)\,\ell_a + d_{\text{net}}\big(\text{end}(a),\text{start}(b)\big) + t_b\,\ell_b \\ R(a\to a) &= (t_b-t_a)\,\ell_a \quad \text{(same arc)} \end{aligned}$$
where $d_{\text{net}}$ is the shortest network distance from the end of $a$ to the start of $b$, capped at 400 m (calculated with Dijkstra's algorithm; for distances longer than 400 m, the transition from $a$ to $b$ is considered unreachable), and $\delta$ is the Euclidean distance between the two GPS positions. Hence,
$$\begin{aligned} \text{cost}(a\to b) &= \min\!\Big(\tfrac{|R-\delta|}{\beta},\ \text{break\_pen}\Big) + \text{uturn\_pen}\cdot\big[\,b = \text{reverse}(a)\,\big] \\ \log T(a\to b) &= -\,w_t\cdot \text{cost}(a\to b) \end{aligned}$$
where unreachable pairs have a transition cost of break_pen.

- **Intuition:** if the vehicle moved 20 m in a straight line, an arc that can be reached in about 20 m of driving is plausible, while an arc that requires a long detour to reach is unlikely.
- **Scores, not probabilities:** these are log-scores (costs), not normalized log-probabilities. This makes the model similar to the map-matching HMM of Newson and Krumm.
- **Viterbi:** maximizes $\sum_k \big[\log p_k(s_k) + \log T(s_{k-1}\to s_k)\big]$ using
$$\text{score}_k(b) = \log p_k(b) + \max_a\big[\text{score}_{k-1}(a) - w_t\,\text{cost}(a\to b)\big],$$
where the best previous arc $a$ is stored as a back-pointer. Once the best overall score is found, the back-pointers are followed to trace back to point 0. The total complexity is $O(K\cdot 8^2)$, rather than $8^K$.
- **Speed:** as the shortest distances for a given trip are calculated once and stored (as in Fast Map Matching), the 182,465 training transitions take $\sim 2$ seconds to decode.
- **Verification:** the implementation was checked against an independent implementation (shortest distances via networkx and Viterbi decoding in plain Python). For 3 student test trips, the transition (route) distances and the decoded arcs matched exactly.

## Missing candidates and large gaps

- **No candidate within 75 m:** the point is skipped (the transition is made between the previous and the next point). The arc of the closest decoded point in time is then assigned to this point (`fill_missing`), so a prediction is made for every observation. There are 717 such points in the training data, and none in student_test.
- **Large gaps ($\Delta t >$ gap_s):** after a long pause between points, the vehicle may be more than 400 m away. Unreachable pairs then cost gap_pen rather than break_pen, and no transition costs more than gap_pen.
- **Effect of the gap rule:** none was observed. With the tuned gap_s of 299 s, 12 training trips were affected by this rule (none in student_test), and switching the rule off did not change the accuracy on these trips (still 0.9506). After a long gap all candidate pairs are equally unreachable, so all transitions have the same constant cost, and the argmax is unchanged. Essentially, after a gap, decoding proceeds in the same way with or without this rule.

## Consecutive observations on disconnected segments

- Unreachable transitions are penalized with a high cost (tuned break_pen = 147.8), but they are not forbidden. Hence, even if there is a missing candidate, a disconnected network or a break in the true route, the decoder does not get stuck.
- The sequence of arcs is then converted into a route (`stitch_route`): if consecutive arcs are not directly connected, they are joined by the shortest network path (capped at 1.5 km). In student test, all joins could be made.

## Tuning

- **Method:** Optuna TPE optimization of the 6 decoding parameters (seed 42, 150 trials, 4.8 min), maximizing the top-1 score across all 78 reliable training trips.
- **Search space:** $w_t$ (log) [0.2, 30], $\beta$ (log) [5, 200] m, break_pen [2, 200], uturn_pen [0, 40], gap_s [5, 600] s, gap_pen [0, 30]. None of the optimal values lie at the search-space boundaries.
- **Out-of-fold probabilities:** each training trip was scored by a 5-fold model that had never seen it (`scripts/09_probs.py`). Using probabilities from a model trained on the same trips would give overconfident probabilities, so the decoder would learn to put less weight on the map information.
- **Default (trial 0):** $w_t$ = 1, $\beta$ = 30, break_pen = 8, uturn_pen = 4, gap_s = 20, gap_pen = 2, giving 92.19 %.
- **Best:** $w_t$ = 7.99, $\beta$ = 13.75 m, break_pen = 147.8, uturn_pen = 25.9, gap_s = 299 s, gap_pen = 24.6, giving 94.35 %.
- Student test was not used in the tuning of the decoding parameters.

## Improvement in route quality?

```{=latex}
\begin{rtable}
\begin{tabularx}{\linewidth}{@{}l L r r r r r@{}}
\toprule
\textbf{Partition} & \textbf{Method} & \textbf{\hd{Top-1 (ref.\\is a cand.)}} & \textbf{\hd{Top-1\\(all points)}} & \textbf{\hd{Route\\continuity}} & \textbf{\hd{Route F1\\(matched arcs)}} & \textbf{\hd{Route F1\\(full .route)}} \\
\midrule
student\_test & classifier only & 0.8995 & 0.8985 & 0.9384 & 0.9289 & 0.8103 \\
student\_test & classifier + HMM (default) & 0.9344 & 0.9334 & 0.9783 & 0.9698 & 0.9597 \\
student\_test & \textbf{classifier + HMM (tuned)} & \textbf{0.9651} & \textbf{0.9640} & \textbf{0.9818} & \textbf{0.9719} & \textbf{0.9775} \\
\midrule
train (out-of-fold) & classifier only & 0.8861 & 0.8798 & 0.9458 & 0.9118 & 0.8046 \\
train (out-of-fold) & classifier + HMM (tuned) & 0.9435 & 0.9360 & 0.9811 & 0.9641 & 0.9609 \\
\bottomrule
\end{tabularx}
\end{rtable}
```

*Source: `results/hmm/hmm_vs_classifier.csv`. The route F1 (full .route) compares the stitched predicted route with all arcs in the true route. In the training rows the decoder settings are in-sample, while on student test the estimate is clean.*

Yes:

- with HMM decoding, the top-1 score increases by 6.6 percentage points, the continuity score by 4.3 percentage points, and the route F1 score improves from 0.810 to 0.978 on student test
- the route F1 score improves the most because the classifier alone produces a route which alternates between twin roads and parallel roads. When this is stitched together, many incorrect arcs are added to the route, whereas the coherent route produced by decoding only requires real gaps to be filled in.

```{=latex}
\begin{figure}[H]\centering
\begin{minipage}[t]{0.6\linewidth}\centering\includegraphics[width=\linewidth]{../results/hmm/hmm_vs_classifier.png}\caption{Classifier only vs classifier + HMM (default and tuned), train (out-of-fold) and student\_test.}\end{minipage}\hfill
\begin{minipage}[t]{0.38\linewidth}\centering\includegraphics[width=\linewidth]{../results/hmm/per_trip_student_test.png}\caption{Per-trip top-1 on student\_test, classifier only vs + HMM (tuned).}\end{minipage}
\end{figure}
```

## Where HMM decoding helped and where it did not

- **Student test:** all 10 trips improved (median +3.3 points). The largest improvement was in trip 77 (78.2 % → 95.9 %), the smallest in trip 0 (+1.8 points).
- **Training data (out-of-fold):** 74 of 78 trips improved. Performance worsened on trip 63 (−3.5 points), 66 (−1.7), 22 (−1.3) and 97 (−0.5). In trip 66 there is a section of 202 points with no true route, which affects the labeling. For trips 63, 22 and 97 it is unclear why performance worsened, but it is plausible that the strong network prior smoothed over real short detours.
- **Gap rule:** no noticeable effect (see 'Missing candidates and large gaps').

# Geographical visualization using Folium

- **Code:** maps are created in `src/viz.py` (`build_trip_map`), using `scripts/02_trip_map.py` and `scripts/10_hmm.py`, and are also displayed in the UI (`ui/app.py`).
- **Base maps:** OpenStreetMap, light gray and satellite.
- **Layers:** road network, true route, predicted route (dashed line with direction arrows) and raw GPS points, plus start and end markers.
- **Controls:** a legend, a layer selector and a scale bar, and the map is zoomed to the extent of the trip.
- **Committed interactive files:** `results/eda/trip_00000003_map.html`, `results/eda/trip_00000033_map.html`, `results/hmm/map_student_test_00000077.html`.

![Student-test trip 77 (Syktyvkar) showing raw GPS (blue), road network (gray), reference route (green) and HMM-predicted route (red) on the satellite base layer. This is the trip where the HMM helped most (78.2 % → 95.9 % top-1).](img/folium_hmm_trip77.png){width=100%}

# API, UI and deployment

## Architecture

- **Laptop 1** hosts the FastAPI backend (`api/main.py`). The model and configuration (`artifacts/`), the road networks and the true routes are stored here. The full pipeline is run (`src/pipeline.py`) and the results are scored on the backend (`src/scoring.py`).
- **Laptop 2** hosts the Streamlit UI (`ui/app.py`), which only requires the dependencies in `ui/requirements.txt`. No model or data is needed.
- **Endpoints:** `GET /health`, `GET /sets`, `GET /trips?trip_set=...`, and `POST /predict` (body: `{"trip_set", "trip_id", "include_network"}`).

## Requests

1. The UI lists the trips available in the API, and the evaluator selects a trip.
2. The UI sends its trip id to the API, which checks that it is valid (unknown ids and attempts to access invalid paths return 404).
3. The API loads the raw files and makes the prediction using only the GPS track and the network.
4. The prediction is compared to the true route, and the route geometry, snapped points, metrics and times are returned. True arcs are never sent outside Laptop 1 and are never seen by the prediction code.
5. All requests are logged in the terminal on Laptop 1, recording the client address, trip, sizes, metrics and duration.

In the two-laptop test (see Figure \ref{fig:apilog}), all 10 student test trips were requested from Laptop 2 (10.1.34.101). The time taken per request ranged from 0.07 to 1.05 s, and the metrics match the offline results exactly.

## Speed

- The average pipeline time per evaluator trip is 0.28 s (CPU), and the longest is 0.65 s.
- For trip 77 (3,911 points): 0.19 s for candidate selection and feature engineering, 0.10 s for XGBoost, 0.30 s for the HMM and 0.004 s for stitching.

## Deployment

- **Laptop 1:** `pip install -r requirements.txt`, then `uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-access-log`.
- **Laptop 2:** `pip install -r ui/requirements.txt`, then `streamlit run ui/app.py`, and enter `http://<laptop-1-ip>:8000` as the API address.
- Both laptops must be connected to the same network, with port 8000 open.

![UI on Laptop 2 showing the evaluator trip set, the trip bar, the metrics returned by the API and the Folium map of evaluator trip 15.](img/ui_eval_map.png){width=100%}

![UI 'All trips & conclusions' tab after running all 10 evaluator trips through the API.](img/ui_eval_conclusions.png){width=100%}

![API terminal on the receiving laptop (Laptop 1, `uvicorn` output) during the two-laptop run. After the start-up lines listing both trip sets, the UI on Laptop 2 (10.1.34.101) runs all 10 student-test trips, and each request prints a PREDICT line, a RESULT line with the server-side metrics, and the request line with Laptop 2's address. Full log: `results/api/two_laptop_demo_laptop1.log`.](img/api_log.png){#fig:apilog width=100%}

# Final evaluation, issues and conclusions

## Evaluation trips

Once the classifier hyperparameters and the decoding parameters were fixed, the 10 evaluation trips were run once through the pipeline (`scripts/11_check_pipeline.py`). The results are saved in `results/pipeline/check_evaluator_test.csv`.

```{=latex}
\begin{rtable}
\begin{tabularx}{\linewidth}{@{}L r r r r r r@{}}
\toprule
\textbf{Trip} & \textbf{Points} & \textbf{\hd{Top-1\\classifier only}} & \textbf{\hd{Top-1\\+ HMM}} & \textbf{Continuity} & \textbf{Route F1} & \textbf{Time (s)} \\
\midrule
00000015 & 6,854 & 0.6584 & 0.9205 & 0.9902 & 0.8171 & 0.65 \\
00000017 & 1,346 & 0.9376 & 0.9539 & 0.8848 & 0.9775 & 0.38 \\
00000023 & 458 & 0.9432 & 0.9716 & 0.9847 & 0.9516 & 0.13 \\
00000030 & 654 & 0.8012 & 0.8379 & 0.9005 & 0.9607 & 0.22 \\
00000049 & 2,642 & 0.9902 & 0.9962 & 0.9947 & 0.9988 & 0.24 \\
00000059 & 1,031 & 0.9670 & 0.9816 & 0.9272 & 0.9774 & 0.23 \\
00000067 & 916 & 0.9509 & 0.9356 & 0.9891 & 1.0000 & 0.10 \\
00000073 & 1,144 & 0.9825 & 0.9843 & 0.9921 & 0.9958 & 0.16 \\
00000087 & 1,399 & 0.9535 & 0.9957 & 0.9900 & 0.9989 & 0.24 \\
00000094 & 3,230 & 0.9276 & 0.9508 & 0.9724 & 0.9789 & 0.44 \\
\midrule
\textbf{Pooled / mean} & \textbf{19,674} & \textbf{0.8473} & \textbf{0.9493} & & \textbf{0.9657} & \textbf{0.28} \\
\bottomrule
\end{tabularx}
\end{rtable}
```

- The evaluation trips were unseen, and still achieved a high accuracy (94.9 %) compared to student test (96.5 %).
- Accuracy with only the classifier is noticeably lower than on student test (84.7 % vs 90.0 %), largely due to trip 15 (6,854 points, 65.8 %). However, 26 points were recovered in this trip using the HMM.
- Performance was poorer on:
    - trip 30 (top-1 = 83.8 %, continuity = 0.90)
    - trip 15 (route F1 = 0.82)
    - trip 17 (continuity = 0.88)
    - trip 67 (top-1 reduced by the HMM from 95.1 % to 93.6 %, despite a route F1 of 1.0)

## Issues

- True routes contain inaccuracies (unreliable trips 33 and 54, missing sections in trips 2 and 66, and small breaks in 11 trips), so there is some noise in the labeling and scoring.
- 0.8 % of points in reliable trips have no correct candidate within 75 m, so they are incorrect by design.
- The transition model only considers the difference between the route length and the straight-line distance. Speed limits, turn restrictions and road type are not considered.
- The scores used in decoding are not normalized probabilities, and the long-gap rule has no measurable effect on this data.
- Bayesian tuning gave only a slight improvement to the classifier, and the 4 best classifiers differ by less than the fold-to-fold variation in their scores.

## Conclusions

- This dataset works well with map matching using candidate classification and decoding.
- Directional features contribute heavily to the high accuracy of the classifier (removing them reduces accuracy by 23 points), while the choice between strong tabular models contributes much less.
- After the features, the HMM gives the largest improvement: accuracy increased from 90.0 % to 96.5 % (observation level) and route F1 from 0.81 to 0.98 on student test, and this improvement generalizes to the unseen evaluation trips (94.9 %).
- The full pipeline runs quickly (under 1 s per trip on a CPU) and has been deployed as an API and UI across two laptops, with the true data remaining on the server.
