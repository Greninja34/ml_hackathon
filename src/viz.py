"""Folium map builder shared by the EDA script, later experiments, and the UI.

Layers: raw GPS (blue), road network (grey), reference route (green, with direction arrows),
optional predicted route (red, with direction arrows). Layer control, legend, fullscreen,
mini-map and a distance-measure tool are included.
"""
import numpy as np
import folium
from folium import plugins

from .plotstyle import COLORS


def _chains(arc_ids, arcs):
    """Group consecutive arcs into connected chains -> list of node-id lists (so breaks are not bridged)."""
    chains, cur = [], []
    for a in arc_ids:
        u, v = arcs[a]
        if cur and cur[-1] == u:
            cur.append(v)
        else:
            if cur:
                chains.append(cur)
            cur = [u, v]
    if cur:
        chains.append(cur)
    return chains


def _route_layer(name, color, arc_ids, arcs, nlat, nlon, weight=6, dash=None):
    fg = folium.FeatureGroup(name=name, show=True)
    for ch in _chains(arc_ids, arcs):
        pts = [(nlat[n], nlon[n]) for n in ch]
        line = folium.PolyLine(pts, color=color, weight=weight, opacity=.85, dash_array=dash,
                               tooltip=name)
        line.add_to(fg)
        plugins.PolyLineTextPath(line, "      ►      ", repeat=True, offset=weight - 3,
                                 attributes={"fill": color, "font-size": "14"}).add_to(fg)
    return fg


def build_trip_map(trk, nodes, arcs, route=None, pred=None, title="", max_net_arcs=60000, pad_deg=0.003):
    """trk: DataFrame lon/lat/t; nodes: DataFrame lon/lat; arcs: (n,2) ints;
    route/pred: arrays of arc ids or None."""
    nlon, nlat = nodes.lon.to_numpy(), nodes.lat.to_numpy()
    m = folium.Map(location=[trk.lat.mean(), trk.lon.mean()], zoom_start=15, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Light Gray", name="Light grey", control=True).add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", control=True).add_to(m)

    # road network inside the padded GPS box, drawn as ONE multi-polyline (keeps the HTML small)
    x0, x1 = trk.lon.min() - pad_deg, trk.lon.max() + pad_deg
    y0, y1 = trk.lat.min() - pad_deg, trk.lat.max() + pad_deg
    inside = ((nlon[arcs[:, 0]] >= x0) & (nlon[arcs[:, 0]] <= x1) &
              (nlat[arcs[:, 0]] >= y0) & (nlat[arcs[:, 0]] <= y1))
    near = np.where(inside)[0][:max_net_arcs]
    net = folium.FeatureGroup(name=f"Road network ({len(near):,} arcs)", show=True)
    lines = [[(nlat[arcs[a, 0]], nlon[arcs[a, 0]]), (nlat[arcs[a, 1]], nlon[arcs[a, 1]])] for a in near]
    folium.PolyLine(lines, color=COLORS["network"], weight=2, opacity=.7).add_to(net)
    net.add_to(m)

    if route is not None:
        _route_layer("Reference route", COLORS["reference"], route, arcs, nlat, nlon, weight=7).add_to(m)
    if pred is not None:
        _route_layer("Predicted route", COLORS["predicted"], pred, arcs, nlat, nlon, weight=4, dash="8 6").add_to(m)

    gps = folium.FeatureGroup(name="Raw GPS points", show=True)
    step = max(1, len(trk) // 800)
    sub = trk.iloc[::step]
    for lat, lon, t in zip(sub.lat, sub.lon, sub.t):
        folium.CircleMarker((lat, lon), radius=3, color="white", weight=1, fill=True,
                            fill_color=COLORS["gps"], fill_opacity=.95,
                            tooltip=f"t = {t:.0f} s").add_to(gps)
    gps.add_to(m)

    folium.Marker((trk.lat.iloc[0], trk.lon.iloc[0]), tooltip="Start",
                  icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
    folium.Marker((trk.lat.iloc[-1], trk.lon.iloc[-1]), tooltip="End",
                  icon=folium.Icon(color="red", icon="stop", prefix="fa")).add_to(m)

    rows = [(COLORS["gps"], "&#9679;", "Raw GPS observation"),
            (COLORS["network"], "&#9473;", "Road network")]
    if route is not None:
        rows.append((COLORS["reference"], "&#9473;", "Reference route"))
    if pred is not None:
        rows.append((COLORS["predicted"], "&#9473;", "Predicted route"))
    items = "".join(f'<div><span style="color:{c};font-size:16px">{s}</span>&nbsp;{t}</div>' for c, s, t in rows)
    legend = (f'<div style="position:fixed;bottom:28px;left:12px;z-index:9999;background:rgba(255,255,255,.95);'
              f'padding:10px 14px;border-radius:8px;box-shadow:0 1px 6px rgba(0,0,0,.35);'
              f'font:13px/1.6 sans-serif"><b>{title or "Legend"}</b>{items}</div>')
    m.get_root().html.add_child(folium.Element(legend))

    plugins.Fullscreen().add_to(m)
    plugins.MiniMap(toggle_display=True, position="bottomright").add_to(m)
    plugins.MeasureControl(primary_length_unit="meters").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[trk.lat.min(), trk.lon.min()], [trk.lat.max(), trk.lon.max()]])
    return m
