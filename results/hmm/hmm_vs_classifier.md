| partition           | method                              |   top1_cond |   continuity |   route_f1 |   top1_e2e |   route_f1_full |   unbridged |   points_filled |
|:--------------------|:------------------------------------|------------:|-------------:|-----------:|-----------:|----------------:|------------:|----------------:|
| train (out-of-fold) | classifier only                     |      0.8861 |       0.9458 |     0.9118 |     0.8798 |          0.8046 |        2091 |             717 |
| train (out-of-fold) | classifier + HMM (default settings) |      0.9219 |       0.9793 |     0.9593 |     0.9153 |          0.9365 |          75 |             717 |
| train (out-of-fold) | classifier + HMM (tuned)            |      0.9435 |       0.9811 |     0.9641 |     0.936  |          0.9609 |          20 |             717 |
| student_test        | classifier only                     |      0.8995 |       0.9384 |     0.9289 |     0.8985 |          0.8103 |         185 |               0 |
| student_test        | classifier + HMM (default settings) |      0.9344 |       0.9783 |     0.9698 |     0.9334 |          0.9597 |           1 |               0 |
| student_test        | classifier + HMM (tuned)            |      0.9651 |       0.9818 |     0.9719 |     0.964  |          0.9775 |           0 |               0 |