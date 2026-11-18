"""Phase 1: Folium + matplotlib visualisation of ONE trip (GPS, road network, reference route).

Run:  .venv/bin/python scripts/02_trip_map.py [partition] [trip_id]
Default: train 00000003
Output: results/eda/trip_<id>_map.html and trip_<id>_matplotlib.png
"""
import sys
from pathlib import Path
import numpy as np
import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_io import load_trip

OUT = Path(__file__).resolve().parents[1] / "results" / "eda"
OUT.mkdir(parents=True, exist_ok=True)
part = sys.argv[1] if len(sys.argv) > 1 else "train"
tid = sys.argv[2] if len(sys.argv) > 2 else "00000003"

tr = load_trip(part, tid)
trk, nd, arcs, route = tr.track, tr.nodes, tr.arcs, tr.route
nlon, nlat = nd.lon.to_numpy(), nd.lat.to_numpy()

# network arcs inside the (padded) GPS bounding box
pad = 0.003
x0, x1, y0, y1 = trk.lon.min() - pad, trk.lon.max() + pad, trk.lat.min() - pad, trk.lat.max() + pad
inside = (nlon[arcs[:, 0]] >= x0) & (nlon[arcs[:, 0]] <= x1) & (nlat[arcs[:, 0]] >= y0) & (nlat[arcs[:, 0]] <= y1)
near = np.where(inside)[0]
print(f"{tid}: {len(trk)} GPS pts, {len(arcs)} arcs total, {len(near)} arcs in view, {len(route)} route arcs")

# ---------- Folium ----------
m = folium.Map(location=[trk.lat.mean(), trk.lon.mean()], zoom_start=15, tiles="OpenStreetMap")
net = folium.FeatureGroup(name="Road network (grey)", show=True)
for a in near:
    u, v = arcs[a]
    folium.PolyLine([(nlat[u], nlon[u]), (nlat[v], nlon[v])], color="#888888", weight=2, opacity=.7).add_to(net)
net.add_to(m)
ref = folium.FeatureGroup(name="Reference route (green)", show=True)
for a in route:
    u, v = arcs[a]
    folium.PolyLine([(nlat[u], nlon[u]), (nlat[v], nlon[v])], color="green", weight=5, opacity=.9,
                    tooltip=f"arc {a}").add_to(ref)
ref.add_to(m)
gps = folium.FeatureGroup(name="Raw GPS points (blue)", show=True)
step = max(1, len(trk) // 600)
for _, r in trk.iloc[::step].iterrows():
    folium.CircleMarker((r.lat, r.lon), radius=3, color="blue", fill=True, fill_opacity=.8,
                        tooltip=f"t={r.t:.0f}s").add_to(gps)
gps.add_to(m)
folium.Marker((trk.lat.iloc[0], trk.lon.iloc[0]), tooltip="start", icon=folium.Icon(color="green")).add_to(m)
folium.Marker((trk.lat.iloc[-1], trk.lon.iloc[-1]), tooltip="end", icon=folium.Icon(color="red")).add_to(m)
legend = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px;
border:1px solid grey;font-size:13px"><b>Legend</b><br><span style="color:blue">&#9679;</span> raw GPS<br>
<span style="color:green">&#9473;</span> reference route<br><span style="color:grey">&#9473;</span> road network</div>"""
m.get_root().html.add_child(folium.Element(legend))
folium.LayerControl().add_to(m)
m.fit_bounds([[trk.lat.min(), trk.lon.min()], [trk.lat.max(), trk.lon.max()]])
m.save(OUT / f"trip_{tid}_map.html")

# ---------- matplotlib ----------
fig, ax = plt.subplots(figsize=(7, 7))
for a in near:
    u, v = arcs[a]
    ax.plot([nlon[u], nlon[v]], [nlat[u], nlat[v]], color="#bbbbbb", lw=.8, zorder=1)
for a in route:
    u, v = arcs[a]
    ax.plot([nlon[u], nlon[v]], [nlat[u], nlat[v]], color="green", lw=2.5, zorder=2)
ax.scatter(trk.lon, trk.lat, s=6, c="blue", zorder=3)
ax.plot([], [], color="#bbbbbb", label="road network"); ax.plot([], [], color="green", label="reference route")
ax.scatter([], [], s=10, c="blue", label="raw GPS")
ax.set_title(f"Trip {tid} ({part}): GPS vs network vs reference route"); ax.set_aspect("equal")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / f"trip_{tid}_matplotlib.png", dpi=140)
print("saved to", OUT)
