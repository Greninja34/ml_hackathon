"""PathSnap evaluator UI (Laptop 2).

Lists the trips offered by the API (Laptop 1), sends the selected trip id to POST /predict, and shows the returned
predicted route on an interactive Folium map together with the server-side evaluation metrics.
The UI never reads trip files or reference routes: everything it shows comes from the API response.

Run:  PATHSNAP_API_URL=http://<laptop-1-ip>:8000  streamlit run ui/app.py
      (the API address can also be changed in the sidebar)
Needs only ui/requirements.txt (streamlit, streamlit-folium, folium, requests, pandas): no model, no data.
The trip set selector lists whatever sets the API serves (student_test, evaluator_test).
"""
import os
import time

import folium
import pandas as pd
import requests
import streamlit as st
from folium import plugins
from streamlit_folium import st_folium

COL = dict(gps="#1f4fd8", predicted="#e63946", snapped="#6a4c93", network="#9aa0a6", start="#1b9e55", end="#111111")
MAX_GPS_MARKERS = 1500

st.set_page_config(page_title="PathSnap", page_icon="🛰️", layout="wide")
ss = st.session_state
ss.setdefault("results", {})      # trip_id -> API response (this session)
ss.setdefault("history", [])      # one row per request made from this UI


# ---------------------------------------------------------------- API calls
def api_get(base, path, timeout=10):
    r = requests.get(base.rstrip("/") + path, timeout=timeout); r.raise_for_status(); return r.json()


SET_LABELS = {"student_test": "Student test (development)", "evaluator_test": "Evaluator test (final evaluation)"}


def api_predict(base, trip_set, trip_id):
    t0 = time.perf_counter()
    r = requests.post(base.rstrip("/") + "/predict", json=dict(trip_set=trip_set, trip_id=trip_id, include_network=True),
                      timeout=300)
    r.raise_for_status()
    body = r.json(); body["_roundtrip_s"] = time.perf_counter() - t0
    return body


def record(body):
    m = body.get("metrics") or {}
    ss.history.append(dict(time=time.strftime("%H:%M:%S"), trip_set=body["trip_set"], trip_id=body["trip_id"],
                           gps_points=body["n_points"],
                           top1_hmm=m.get("top1_cond"), top1_classifier=m.get("top1_clf_cond"),
                           continuity=m.get("continuity"), route_f1=m.get("route_f1"),
                           pipeline_s=body["timings"]["total_s"], roundtrip_s=round(body["_roundtrip_s"], 2)))


# ---------------------------------------------------------------- map
def build_map(d, show_network, show_snapped):
    gps = d["gps"]
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], zoom_start=15, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Light Gray", name="Light grey", show=False).add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Imagery", name="Satellite", show=False).add_to(m)

    if d.get("network_segments"):
        fg = folium.FeatureGroup(name=f"Road network ({len(d['network_segments']):,} arcs)", show=show_network)
        folium.PolyLine(d["network_segments"], color=COL["network"], weight=2, opacity=0.6).add_to(fg)
        fg.add_to(m)

    step = max(1, len(gps) // MAX_GPS_MARKERS)
    fg = folium.FeatureGroup(name=f"Raw GPS points ({len(gps):,}{', every ' + str(step) + 'th shown' if step > 1 else ''})", show=True)
    for i in range(0, len(gps), step):
        folium.CircleMarker(gps[i], radius=3, color=COL["gps"], fill=True, fill_opacity=0.55, weight=0,
                            tooltip=f"GPS point {i}").add_to(fg)
    fg.add_to(m)

    fg = folium.FeatureGroup(name="Predicted route", show=True)       # added after the GPS points: drawn on top
    for chain in d["route_polylines"]:
        line = folium.PolyLine(chain, color=COL["predicted"], weight=4, opacity=0.95, tooltip="Predicted route")
        line.add_to(fg)
        plugins.PolyLineTextPath(line, "     ►     ", repeat=True, offset=4,
                                 attributes={"fill": COL["predicted"], "font-size": "13"}).add_to(fg)
    fg.add_to(m)

    fg = folium.FeatureGroup(name="Snapped positions", show=show_snapped)
    for i in range(0, len(d["snapped"]), step):
        folium.CircleMarker(d["snapped"][i], radius=2, color=COL["snapped"], fill=True, fill_opacity=0.9, weight=0,
                            tooltip=f"point {i} snapped to arc {d['point_arc'][i]}").add_to(fg)
    fg.add_to(m)

    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="black", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    legend = f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: #ffffff; color: #1f1f1f;
                padding: 8px 12px; border: 1px solid #bbb; border-radius: 6px; font: 13px sans-serif; line-height: 1.6;
                box-shadow: 0 1px 4px rgba(0,0,0,.25);">
      <b style="color:#1f1f1f">Trip {d['trip_id']}</b> <span style="color:#555">({d['trip_set']})</span><br>
      <span style="color:{COL['gps']}">&#9679;</span> <span style="color:#1f1f1f">raw GPS point</span><br>
      <span style="color:{COL['predicted']}">&#9644;&#9644;</span> <span style="color:#1f1f1f">predicted route (HMM)</span><br>
      <span style="color:{COL['snapped']}">&#9679;</span> <span style="color:#1f1f1f">snapped position</span><br>
      <span style="color:{COL['network']}">&#9644;&#9644;</span> <span style="color:#1f1f1f">road network</span>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m


def pct(x):
    return "n/a" if x is None else f"{100 * x:.2f}%"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🛰️ PathSnap")
    st.caption("Map matching: noisy GPS → road route")
    api = st.text_input("API address (Laptop 1)", os.environ.get("PATHSNAP_API_URL", "http://localhost:8000"))
    try:
        health = api_get(api, "/health", timeout=5)
        sets = [x["name"] for x in api_get(api, "/sets")["sets"]]
        st.success(f"Connected · {health['model']}")
    except Exception as e:
        st.error(f"Cannot reach the API at {api}\n\n{type(e).__name__}: {e}")
        st.stop()
    if not sets:
        st.error("The API serves no trip sets."); st.stop()
    trip_set = st.radio("Trip set", sets, format_func=lambda k: f"{SET_LABELS.get(k, k)} · {health['trip_sets'][k]} trips")
    trips = api_get(api, f"/trips?trip_set={trip_set}")["trips"]

    labels = {t["trip_id"]: f"{t['trip_id']}  ({t['n_points']:,} GPS points)" for t in trips}
    trip_id = st.selectbox("Evaluation trip", list(labels), format_func=labels.get)
    run = st.button("▶ Reconstruct route", type="primary", width="stretch")
    run_all = st.button(f"Run all {len(trips)} trips", width="stretch")
    st.divider()
    show_network = st.checkbox("Show road network", True)
    show_snapped = st.checkbox("Show snapped positions", False)
    st.caption("Ground truth stays on the API server; this page only receives the prediction and its scores.")

# ---------------------------------------------------------------- actions
if run:
    with st.spinner(f"API is processing trip {trip_id} …"):
        try:
            ss.results[(trip_set, trip_id)] = api_predict(api, trip_set, trip_id); record(ss.results[(trip_set, trip_id)])
        except Exception as e:
            st.error(f"Prediction failed for trip {trip_id}: {e}")
if run_all:
    bar = st.progress(0.0, text="Running all trips …")
    for k, t in enumerate(trips):
        try:
            key = (trip_set, t["trip_id"])
            ss.results[key] = api_predict(api, trip_set, t["trip_id"]); record(ss.results[key])
        except Exception as e:
            st.error(f"Prediction failed for trip {t['trip_id']}: {e}")
        bar.progress((k + 1) / len(trips), text=f"Trip {t['trip_id']} done")
    bar.empty()

# ---------------------------------------------------------------- main view
st.header("Route reconstruction")
st.caption(f"Trip set: **{SET_LABELS.get(trip_set, trip_set)}** · trip **{trip_id}**")
d = ss.results.get((trip_set, trip_id))
if d is None:
    st.info("Choose a trip in the sidebar and press **Reconstruct route**. The API (Laptop 1) runs the full pipeline: "
            "candidate roads → features → tuned XGBoost → HMM/Viterbi → stitched route, then scores it server-side.")
else:
    m = d.get("metrics") or {}
    c = st.columns(5)
    gain = None if m.get("top1_cond") is None or m.get("top1_clf_cond") is None else 100 * (m["top1_cond"] - m["top1_clf_cond"])
    c[0].metric("Top-1 accuracy (classifier + HMM)", pct(m.get("top1_cond")),
                None if gain is None else f"{gain:+.2f} pts vs classifier only")
    c[1].metric("Classifier only", pct(m.get("top1_clf_cond")))
    c[2].metric("Route continuity", pct(m.get("continuity")))
    c[3].metric("Route F1 (vs reference route)", pct(m.get("route_f1")))
    c[4].metric("Processing time (API)", f"{d['timings']['total_s']:.2f} s", f"round trip {d['_roundtrip_s']:.2f} s", delta_color="off")
    if not m:
        st.warning("This trip has no reference route on the server, so only the prediction is shown.")

    st_folium(build_map(d, show_network, show_snapped), height=640, use_container_width=True, returned_objects=[],
              key=f"map_{trip_set}_{trip_id}_{show_network}_{show_snapped}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Trip summary")
        st.table(pd.DataFrame({"value": {
            "GPS points": f"{d['n_points']:,}",
            "points scored (reference within 75 m)": f"{m.get('n_points_scored', 0):,}" if m else "n/a",
            "points without candidates (filled)": f"{d['n_filled_points']:,}",
            "candidate (point, road) pairs scored": f"{d['n_candidate_rows']:,}",
            "arcs in predicted route": f"{len(d['route_arc_ids']):,}",
            "route breaks not bridgeable": f"{d['unbridged_joins']}",
            "top-1 over all points": pct(m.get("top1_all")) if m else "n/a"}}))
    with right:
        st.subheader("Where the time goes (API side)")
        t = d["timings"]
        st.table(pd.DataFrame({"seconds": {"load trip files": t["load_s"], "candidates + features": t["candidates_features_s"],
                                           "XGBoost scoring": t["classifier_s"], "HMM / Viterbi": t["hmm_viterbi_s"],
                                           "route stitching": t["stitch_s"], "pipeline total": t["total_s"]}}).round(3))
        with st.expander("Model and decoder settings"):
            st.json(d["model"])

if ss.history:
    st.subheader("Requests made from this UI")
    h = pd.DataFrame(ss.history)
    fmt = {c: "{:.2%}" for c in ["top1_hmm", "top1_classifier", "continuity", "route_f1"]}
    st.dataframe(h.style.format(fmt, na_rep="n/a"), width="stretch", hide_index=True)
    done = {(r["trip_set"], r["trip_id"]): r for r in ss.history}
    last = pd.DataFrame(done.values())
    for name, g in last.groupby("trip_set"):
        st.caption(f"**{SET_LABELS.get(name, name)}**: {len(g)} distinct trips run · mean top-1 {g.top1_hmm.mean():.2%} · "
                   f"mean classifier-only {g.top1_classifier.mean():.2%} · mean route F1 {g.route_f1.mean():.2%} · "
                   f"mean API time {g.pipeline_s.mean():.2f} s")
