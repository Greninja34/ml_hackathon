"""Candidate generation: for every GPS point, all road arcs within `radius` metres.

Uses a shapely STRtree (R-tree) over the arc line segments, queried with the `dwithin`
predicate, then exact point->segment distances are computed with numpy.
Zero-length arcs (106 in the whole dataset) carry no geometry and are never candidates.
"""
import numpy as np
import shapely
from shapely.strtree import STRtree

from .data_io import point_segment_distance


class CandidateIndex:
    def __init__(self, A, B):
        """A, B: (n_arcs, 2) projected endpoints of every arc (metres)."""
        self.A, self.B = A, B
        self.valid = np.where(np.linalg.norm(B - A, axis=1) > 0.01)[0]
        lines = shapely.linestrings(np.stack([A[self.valid], B[self.valid]], axis=1))
        self.tree = STRtree(lines)

    def query(self, P, radius):
        """Returns arrays (pt_idx, arc_id, dist, t) of every (point, arc) pair with dist <= radius.
        t in [0,1] is the position of the projection foot along the arc."""
        pts = shapely.points(P)
        pi, li = self.tree.query(pts, predicate="dwithin", distance=radius)
        arc = self.valid[li]
        d, t = point_segment_distance(P[pi], self.A[arc], self.B[arc])
        keep = d <= radius
        return pi[keep], arc[keep], d[keep], t[keep]
