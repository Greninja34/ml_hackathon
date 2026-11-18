| partition           | method                              |   top1_cond |   top1_e2e |   continuity |   route_f1 |
|:--------------------|:------------------------------------|------------:|-----------:|-------------:|-----------:|
| train (out-of-fold) | classifier only                     |      0.8861 |     0.8783 |       0.9458 |     0.9118 |
| train (out-of-fold) | classifier + HMM (default settings) |      0.9219 |     0.9138 |       0.9791 |     0.9589 |
| train (out-of-fold) | classifier + HMM (tuned)            |      0.9368 |     0.9286 |       0.9824 |     0.9686 |
| student_test        | classifier only                     |      0.8995 |     0.8985 |       0.9384 |     0.9289 |
| student_test        | classifier + HMM (default settings) |      0.9343 |     0.9333 |       0.9781 |     0.9697 |
| student_test        | classifier + HMM (tuned)            |      0.9554 |     0.9543 |       0.9821 |     0.9747 |