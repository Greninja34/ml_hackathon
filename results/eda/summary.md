# EDA summary (partitions: train=80 trips, student_test=10 trips)

| item | value |
|---|---|
| total GPS points | 227,577 |
| points / trip (min, median, max) | 262, 1496, 19313 |
| arcs / trip (min, median, max) | 649, 18,761, 307,392 |
| route arcs / trip (min, median, max) | 59, 272, 2266 |
| NaN in tracks / nodes | 0 / 0 |
| lon/lat out of range | 0 |
| duplicate timestamps | 0 |
| non-monotonic timestamps | 0 |
| fraction of steps exactly 1 s (mean over trips) | 0.943 |
| trips with a gap > 60 s | 23 |
| largest gap (s) | 153064 |
| arcs with invalid node ref | 0 |
| route arcs with invalid arc ref | 0 |
| zero-length arcs | 106 |
| isolated nodes (total / trips affected) | 0 / 0 |
| weak components (median / max per trip) | 19 / 421 |
| largest-component fraction (min) | 0.693 |
| arcs with reverse twin (mean over trips) | 0.922 |
| reference route fully connected (min over trips) | 0.9813 |
| GPS->route distance m: p50 / p90 / p95 / p99 / max | 5.6 / 48.8 / 137.7 / 200.0 / 569 |
| 99th-pct implied speed (m/s), max over trips | 91.1 |
