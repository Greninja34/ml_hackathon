"""Raw-data loading and geometry helpers for PathSnap.

Raw files per trip (all headerless, whitespace separated):
  .track  : lon lat timestamp_s
  .nodes  : lon lat              (row index = node id)
  .arcs   : from_node to_node    (row index = arc id)
  .route  : arc_id               (reference route, in travel order)

Every trip lives in a different part of the world, so each trip is projected to its OWN
local metric plane (equirectangular around the track centroid). Over the few-km extent of a
trip the error of this projection is far below the GPS noise (~5 m), and it avoids picking
a UTM zone per trip.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

EARTH_R = 6371008.8  # metres

DATA_ROOT = Path(__file__).resolve().parents[2] / "Student folder"
PARTITIONS = ("train", "student_test")


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame          # columns lon, lat, t
    nodes: pd.DataFrame          # columns lon, lat
    arcs: np.ndarray             # (n_arcs, 2) int  from_node, to_node
    route: np.ndarray | None     # (n_route,) int arc ids, or None if not available


def list_trips(partition: str, root: Path = DATA_ROOT):
    return sorted(p.name for p in (root / partition).iterdir() if p.is_dir())


def _read(path, names, dtype=None):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def load_trip(partition: str, trip_id: str, root: Path = DATA_ROOT, with_route: bool = True) -> Trip:
    d = root / partition / trip_id
    track = _read(d / f"{trip_id}.track", ["lon", "lat", "t"])
    nodes = _read(d / f"{trip_id}.nodes", ["lon", "lat"])
    arcs = _read(d / f"{trip_id}.arcs", ["u", "v"], dtype="int64").to_numpy()
    route = None
    rpath = d / f"{trip_id}.route"
    if with_route and rpath.exists():
        route = _read(rpath, ["arc"], dtype="int64")["arc"].to_numpy()
    return Trip(trip_id, partition, track, nodes, arcs, route)


def project(lon, lat, lon0, lat0):
    """lon/lat (degrees) -> local metres (x east, y north) around (lon0, lat0)."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return np.column_stack([x, y])


def point_segment_distance(p, a, b):
    """Distance from points p (n,2) to segments a->b (n,2). Also returns projection fraction t."""
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-9)
    t = np.clip(((p - a) * ab).sum(1) / L2, 0.0, 1.0)
    foot = a + ab * t[:, None]
    return np.linalg.norm(p - foot, axis=1), t
