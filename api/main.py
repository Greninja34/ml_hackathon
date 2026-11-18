"""PathSnap API (Laptop 1): runs the full prediction pipeline for a selected trip and scores it server-side.

Endpoints
  GET  /health                 model + data status
  GET  /sets                   trip sets served (e.g. student_test, evaluator_test) and their number of trips
  GET  /trips?trip_set=NAME    trip ids of one set (and their number of GPS points)
  POST /predict                body {"trip_set": "evaluator_test", "trip_id": "00000015", "include_network": false}
                               -> GPS points, predicted route (geometry + arc ids), snapped points, metrics, timings

The reference route (.route) stays on this machine: it is read ONLY by src/scoring.py after the prediction is made,
and only the resulting metrics are returned. The prediction pipeline (src/pipeline.py) never reads it.
Every request is logged to this terminal (client address, trip, sizes, timings, metrics).

Run (from the repo root, conda env ml_lab):
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-access-log
  (--no-access-log: our own middleware already logs every request with timings; this avoids duplicate lines)
  Laptop 2 then reaches it at http://<laptop-1-ip>:8000 (find the IP with `ip addr` / `ipconfig`).
Trip sets: each is a folder with one sub-folder per trip (<id>/<id>.track|.nodes|.arcs[|.route]).
  Default: student_test = data/raw/student_test, evaluator_test = ../evaluator_test (next to the repo, never inside it).
  Override: PATHSNAP_TRIP_SETS="student_test=/path/a;evaluator_test=/path/b"   (sets whose folder is missing are skipped)
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

def _trip_sets():
    spec = os.environ.get("PATHSNAP_TRIP_SETS")
    pairs = ([kv.split("=", 1) for kv in spec.split(";") if "=" in kv] if spec else
             [("student_test", ROOT / "data" / "raw" / "student_test"), ("evaluator_test", ROOT.parent / "evaluator_test")])
    return {name.strip(): Path(path).expanduser().resolve() for name, path in pairs if Path(path).expanduser().is_dir()}


TRIP_SETS = _trip_sets()
MAX_NETWORK_SEGMENTS = 20000

log = logging.getLogger("pathsnap")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    log.addHandler(h); log.setLevel(logging.INFO); log.propagate = False

app = FastAPI(title="PathSnap map-matching API", version="1.0")
model = PathSnapModel(device="cpu")
lock = threading.Lock()


def list_trip_ids(trip_set):
    d = TRIP_SETS.get(trip_set)
    if d is None:
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def _check_set(trip_set):
    if trip_set not in TRIP_SETS:
        raise HTTPException(404, f"unknown trip set {trip_set!r}; available: {list(TRIP_SETS)}")


log.info(f"PathSnap API ready | model: {model.meta['model']} | radius {model.radius:.0f} m | HMM top-{model.top_n} | inference on CPU")
for _n, _d in TRIP_SETS.items():
    log.info(f"  trip set {_n!r}: {len(list_trip_ids(_n))} trips from {_d}")
if not TRIP_SETS:
    log.warning("no trip set folder found; set PATHSNAP_TRIP_SETS")


class PredictRequest(BaseModel):
    trip_id: str
    trip_set: str = "student_test"
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
    return dict(status="ok", model=model.meta["model"], trip_sets={k: len(list_trip_ids(k)) for k in TRIP_SETS},
                candidate_radius_m=model.radius, hmm=model.hmm_cfg["params"])


@app.get("/sets")
def sets():
    return dict(sets=[dict(name=k, n_trips=len(list_trip_ids(k))) for k in TRIP_SETS])


@app.get("/trips")
def trips(trip_set: str = "student_test"):
    _check_set(trip_set)
    out = []
    for tid in list_trip_ids(trip_set):
        with open(TRIP_SETS[trip_set] / tid / f"{tid}.track") as f:
            n = sum(1 for line in f if line.strip())
        out.append(dict(trip_id=tid, n_points=n))
    return dict(trip_set=trip_set, trips=out)


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
    tid, tset = req.trip_id.strip(), req.trip_set.strip()
    _check_set(tset)
    if tid not in list_trip_ids(tset):                  # also blocks path traversal: only known folder names pass
        log.warning(f"unknown trip id requested: {tset}/{tid!r}")
        raise HTTPException(404, f"unknown trip id {tid!r} in set {tset!r}")
    t0 = time.perf_counter()
    d = TRIP_SETS[tset]
    trip = load_trip(d.name, tid, root=d.parent, with_route=True)
    t_load = time.perf_counter() - t0
    track, nodes, arcs = trip.track, trip.nodes, trip.arcs
    log.info(f"[PREDICT] {tset}/{tid} | {len(track)} GPS points | {len(nodes):,} nodes | {len(arcs):,} arcs")

    with lock:
        pred = model.predict(track, nodes, arcs)        # sees only track + network, never trip.route
    metrics = score(trip, pred, model.radius)          # server-side; reference arcs are not returned

    # geometry for the UI
    gps = np.column_stack([track.lat.to_numpy(), track.lon.to_numpy()])
    a0, a1 = _latlon(nodes, arcs[pred.point_arc, 0]), _latlon(nodes, arcs[pred.point_arc, 1])
    tt = np.where(np.isnan(pred.point_t), 0.5, pred.point_t)[:, None]
    snapped = a0 + (a1 - a0) * tt
    body = dict(trip_id=tid, trip_set=tset, n_points=pred.n_points, n_candidate_rows=pred.n_candidate_rows,
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
    log.info(f"[RESULT]  {tset}/{tid} | top-1 {m.get('top1_cond', float('nan')):.4f} (classifier only "
             f"{m.get('top1_clf_cond', float('nan')):.4f}) | continuity {pred.continuity:.4f} | route F1 "
             f"{m.get('route_f1', float('nan')):.4f} | route {len(pred.route)} arcs | pipeline {pred.timings['total_s']:.2f}s "
             f"| request {total:.2f}s")
    return body
