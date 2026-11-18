"""Phase 1: Folium + matplotlib visualisation of ONE trip (GPS, road network, reference route).

Run:  python scripts/02_trip_map.py [partition] [trip_id]      (default: train 00000003)
Output: results/eda/trip_<id>_map.html and trip_<id>_matplotlib.png
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_io import load_trip
from src.plotstyle import COLORS, apply, finish
from src.viz import build_trip_map

apply()
OUT = Path(__file__).resolve().parents[1] / "results" / "eda"
OUT.mkdir(parents=True, exist_ok=True)
part = sys.argv[1] if len(sys.argv) > 1 else "train"
tid = sys.argv[2] if len(sys.argv) > 2 else "00000003"

tr = load_trip(part, tid)
trk, nd, arcs, route = tr.track, tr.nodes, tr.arcs, tr.route
print(f"{tid}: {len(trk)} GPS pts, {len(arcs)} arcs, {len(route)} route arcs")

m = build_trip_map(trk, nd, arcs, route=route, title=f"Trip {tid} ({part})")
m.save(OUT / f"trip_{tid}_map.html")

# ---- static version for the PDF ----
nlon, nlat = nd.lon.to_numpy(), nd.lat.to_numpy()
pad = 0.003
x0, x1, y0, y1 = trk.lon.min() - pad, trk.lon.max() + pad, trk.lat.min() - pad, trk.lat.max() + pad
inside = (nlon[arcs[:, 0]] >= x0) & (nlon[arcs[:, 0]] <= x1) & (nlat[arcs[:, 0]] >= y0) & (nlat[arcs[:, 0]] <= y1)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.set_facecolor("white"); ax.grid(False)
u, v = arcs[inside, 0], arcs[inside, 1]
ax.plot(np.c_[nlon[u], nlon[v]].T, np.c_[nlat[u], nlat[v]].T, color=COLORS["network"], lw=.7, alpha=.8, zorder=1)
u, v = arcs[route, 0], arcs[route, 1]
ax.plot(np.c_[nlon[u], nlon[v]].T, np.c_[nlat[u], nlat[v]].T, color=COLORS["reference"], lw=3, zorder=2)
ax.scatter(trk.lon, trk.lat, s=7, c=COLORS["gps"], zorder=3, edgecolor="white", linewidth=.2)
ax.plot([], [], color=COLORS["network"], label="road network")
ax.plot([], [], color=COLORS["reference"], lw=3, label="reference route")
ax.scatter([], [], s=18, c=COLORS["gps"], label="raw GPS")
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
ax.set_aspect(1 / np.cos(np.radians(trk.lat.mean())))
ax.set_title(f"Trip {tid}: GPS vs road network vs reference route")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend(loc="upper right", frameon=True)
finish(fig, OUT / f"trip_{tid}_matplotlib.png", f"Partition: {part}")
print("saved to", OUT)
