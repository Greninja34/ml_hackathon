"""PathSnap API (Laptop 1): runs the full prediction pipeline for a selected trip and scores it server-side.

Endpoints
  GET  /health                 model + data status
  GET  /trips                  trip ids available on this server (and their number of GPS points)
  POST /predict                body {"trip_id": "00000077", "include_network": false}
                               -> GPS points, predicted route (geometry + arc ids), snapped points, metrics, timings

The reference route (.route) stays on this machine: it is read ONLY by src/scoring.py after the prediction is made,
and only the resulting metrics are returned. The prediction pipeline (src/pipeline.py) never reads it.
Every request is logged to this terminal (client address, trip, sizes, timings, metrics).

Run (from the repo root, conda env ml_lab):
  PATHSNAP_TRIPS_DIR=/path/to/trips  uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-access-log
  (--no-access-log: our own middleware already logs every request with timings; this avoids duplicate lines)
  Laptop 2 then reaches it at http://<laptop-1-ip>:8000 (find the IP with `ip addr` / `ipconfig`).
  PATHSNAP_TRIPS_DIR is a folder holding one sub-folder per trip (<id>/<id>.track|.nodes|.arcs[|.route]).
  Default: data/raw/student_test (the development stand-in for the evaluator trips).
"""
import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_io import load_trip  # noqa: E402
from src.pipeline import PathSnapModel  # noqa: E402
from src.scoring import score  # noqa: E402

TRIPS_DIR = Path(os.environ.get("PATHSNAP_TRIPS_DIR", ROOT / "data" / "raw" / "student_test")).resolve()
MAX_NETWORK_SEGMENTS = 20000

log = logging.getLogger("pathsnap")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    log.addHandler(h); log.setLevel(logging.INFO); log.propagate = False

app = FastAPI(title="PathSnap map-matching API", version="1.0")
model = PathSnapModel(device="cpu")
lock = threading.Lock()


def list_trip_ids():
    if not TRIPS_DIR.is_dir():
        return []
    return sorted(p.name for p in TRIPS_DIR.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


log.info(f"PathSnap API ready | trips dir: {TRIPS_DIR} | {len(list_trip_ids())} trips | model: {model.meta['model']} "
         f"| radius {model.radius:.0f} m | HMM top-{model.top_n} | inference on CPU")


class PredictRequest(BaseModel):
    trip_id: str
    include_network: bool = False


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    client = f"{request.client.host}:{request.client.port}" if request.client else "?"
    log.info(f"{client} {request.method} {request.url.path} -> {response.status_code} ({(time.perf_counter()-t0)*1000:.0f} ms)")
    return response


@app.get("/health")
def health():
    return dict(status="ok", model=model.meta["model"], trips_dir=str(TRIPS_DIR), n_trips=len(list_trip_ids()),
                candidate_radius_m=model.radius, hmm=model.hmm_cfg["params"])


@app.get("/trips")
def trips():
    out = []
    for tid in list_trip_ids():
        with open(TRIPS_DIR / tid / f"{tid}.track") as f:
            n = sum(1 for line in f if line.strip())
        out.append(dict(trip_id=tid, n_points=n))
    return dict(trips=out)


def _latlon(nodes, ids):
    return np.column_stack([nodes.lat.to_numpy()[ids], nodes.lon.to_numpy()[ids]])


def _route_chains(route, arcs, nodes):
    """Split the arc sequence into continuous polylines [[lat, lon], ...] (a new chain wherever arcs do not touch)."""
    chains, cur = [], []
    for a in route:
        u, v = int(arcs[a, 0]), int(arcs[a, 1])
        if cur and cur[-1] != u:
            chains.append(cur); cur = []
        if not cur:
            cur.append(u)
        cur.append(v)
    if cur:
        chains.append(cur)
    lat, lon = nodes.lat.to_numpy(), nodes.lon.to_numpy()
    return [[[float(lat[n]), float(lon[n])] for n in ch] for ch in chains]


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    tid = req.trip_id.strip()
    if tid not in list_trip_ids():                      # also blocks path traversal: only known folder names pass
        log.warning(f"unknown trip id requested: {tid!r}")
        raise HTTPException(404, f"unknown trip id {tid!r}")
    t0 = time.perf_counter()
    trip = load_trip(TRIPS_DIR.name, tid, root=TRIPS_DIR.parent, with_route=True)
    t_load = time.perf_counter() - t0
    track, nodes, arcs = trip.track, trip.nodes, trip.arcs
    log.info(f"[PREDICT] trip {tid} | {len(track)} GPS points | {len(nodes):,} nodes | {len(arcs):,} arcs")

    with lock:
        pred = model.predict(track, nodes, arcs)        # sees only track + network, never trip.route
    metrics = score(trip, pred, model.radius)          # server-side; reference arcs are not returned

    # geometry for the UI
    gps = np.column_stack([track.lat.to_numpy(), track.lon.to_numpy()])
    a0, a1 = _latlon(nodes, arcs[pred.point_arc, 0]), _latlon(nodes, arcs[pred.point_arc, 1])
    tt = np.where(np.isnan(pred.point_t), 0.5, pred.point_t)[:, None]
    snapped = a0 + (a1 - a0) * tt
    body = dict(trip_id=tid, n_points=pred.n_points, n_candidate_rows=pred.n_candidate_rows,
                n_filled_points=int(pred.point_filled.sum()), unbridged_joins=pred.unbridged,
                gps=gps.round(7).tolist(), snapped=snapped.round(7).tolist(),
                point_arc=pred.point_arc.tolist(), route_arc_ids=pred.route.tolist(),
                route_polylines=_route_chains(pred.route, arcs, nodes),
                metrics=metrics, timings={**{k: round(v, 3) for k, v in pred.timings.items()}, "load_s": round(t_load, 3)},
                model=dict(name=model.meta["model"], candidate_radius_m=model.radius, hmm_params=model.hmm_cfg["params"]))
    if req.include_network:
        lat, lon = nodes.lat.to_numpy(), nodes.lon.to_numpy(); pad = 0.003
        inside = ((lon[arcs[:, 0]] >= track.lon.min() - pad) & (lon[arcs[:, 0]] <= track.lon.max() + pad) &
                  (lat[arcs[:, 0]] >= track.lat.min() - pad) & (lat[arcs[:, 0]] <= track.lat.max() + pad))
        sel = np.flatnonzero(inside)[:MAX_NETWORK_SEGMENTS]
        body["network_segments"] = np.stack([_latlon(nodes, arcs[sel, 0]), _latlon(nodes, arcs[sel, 1])], 1).round(7).tolist()

    total = time.perf_counter() - t0
    m = metrics or {}
    log.info(f"[RESULT]  trip {tid} | top-1 {m.get('top1_cond', float('nan')):.4f} (classifier only "
             f"{m.get('top1_clf_cond', float('nan')):.4f}) | continuity {pred.continuity:.4f} | route F1 "
             f"{m.get('route_f1', float('nan')):.4f} | route {len(pred.route)} arcs | pipeline {pred.timings['total_s']:.2f}s "
             f"| request {total:.2f}s")
    return body
