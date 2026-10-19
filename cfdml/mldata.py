import json
from pathlib import Path

import numpy as np

from . import foam
from .errors import CfdmlError


class Traj:
    def __init__(self, fields, times, names, x=None, y=None, points=None, mask=None, params=None, source=""):
        self.fields = np.asarray(fields, np.float64)
        self.times = np.asarray(times, float)
        self.names = list(names)
        self.x, self.y, self.points = x, y, points
        self.mask = None if mask is None else np.asarray(mask, bool)
        self.params = params or {}
        self.source = source

    @property
    def grid(self):
        return self.fields.ndim == 4

    def channel(self, name):
        if name not in self.names:
            raise CfdmlError(f"no channel '{name}' in {self.source}", hint="channels: " + ", ".join(self.names))
        i = self.names.index(name)
        return self.fields[:, i] if self.grid else self.fields[:, :, i]

    def flat(self):
        if self.grid:
            X = self.fields.transpose(0, 2, 3, 1).reshape(len(self.times), -1, len(self.names))
            if self.mask is not None:
                X = X[:, self.mask.ravel()]
            return X
        return self.fields

    def spacing(self):
        if not self.grid or self.x is None or self.y is None:
            return None
        return float(np.mean(np.diff(self.x))), float(np.mean(np.diff(self.y)))


def _components(name, v, two_d):
    if v.ndim == 1:
        return [name], v[:, None]
    keep = list(range(v.shape[1]))
    if v.shape[1] == 3 and two_d is not None:
        keep = [j for j in keep if j != two_d]
    return [f"{name}{'xyz'[j] if v.shape[1] == 3 else j}" for j in keep], v[:, keep]


def load_case(case, fields, after=None, before=None, every=1, min_times=2):
    case = Path(case)
    ts = None
    for f in fields:
        t = [(tt, d) for tt, d in foam.times(case, [f]) if (after is None or tt >= after) and (before is None or tt <= before)]
        ts = t if ts is None else [p for p in ts if p[0] in {q[0] for q in t}]
    ts = (ts or [])[::every]
    if len(ts) < min_times:
        raise CfdmlError(f"{case}: fewer than {min_times} time folders with {', '.join(fields)} in the window",
                         hint="fields found: " + ", ".join(sorted(foam.field_names(case))))
    nc = foam.n_cells(case)
    try:
        pts = foam.cell_centres(case)
    except CfdmlError:
        pts = None
    two_d = None
    if pts is not None:
        span = np.ptp(pts, 0)
        if span.min() < 1e-6 * span.max() or np.unique(pts[:, int(np.argmin(span))]).size == 1:
            two_d = int(np.argmin(span))
    frames, names = [], None
    for _, d in ts:
        cols, nm = [], []
        for f in fields:
            n, v = _components(f, foam.read_field(d / f, nc), two_d)
            nm += n
            cols.append(v)
        frames.append(np.concatenate(cols, 1))
        if names is None:
            names = nm
        elif nm != names:
            raise CfdmlError(f"{case}: the components of {', '.join(fields)} change between time folders")
    return Traj(np.stack(frames), [t for t, _ in ts], names, points=pts, source=str(case))


def load(path, fields=None):
    p = Path(path)
    if p.is_dir() and (p / "system").is_dir():
        if not fields:
            raise CfdmlError(f"{p} is an OpenFOAM case: say which fields with -f", hint="e.g.  -f U -f p")
        return load_case(p, fields)
    if p.suffix == ".npz" and p.exists():
        z = np.load(p, allow_pickle=False)
        if "fields" not in z.files:
            raise CfdmlError(f"{p} has no 'fields' array", hint="arrays found: " + ", ".join(z.files)
                             + ". A trajectory file needs fields (T, C, H, W) or (T, N, C), and ideally times and names")
        F = z["fields"]
        if F.ndim not in (3, 4):
            raise CfdmlError(f"'fields' in {p} has shape {F.shape}; expected (T, C, H, W) or (T, N, C)")
        nch = F.shape[1] if F.ndim == 4 else F.shape[2]
        names = [str(s) for s in z["names"]] if "names" in z.files else [f"c{i}" for i in range(nch)]
        times = z["times"] if "times" in z.files else np.arange(F.shape[0], dtype=float)
        params = json.loads(str(z["params"])) if "params" in z.files else {}
        return Traj(F, times, names, x=z["x"] if "x" in z.files else None, y=z["y"] if "y" in z.files else None,
                    points=z["points"] if "points" in z.files else None, mask=z["mask"] if "mask" in z.files else None,
                    params=params, source=str(p))
    raise CfdmlError(f"cannot read {p}", hint="give an OpenFOAM case folder or a trajectory .npz")


def save(traj, path):
    out = dict(fields=traj.fields.astype(np.float32), times=traj.times, names=np.array(traj.names),
               params=json.dumps(traj.params))
    for k in ("x", "y", "points", "mask"):
        v = getattr(traj, k)
        if v is not None:
            out[k] = v
    np.savez_compressed(path, **out)


class GridMap:

    def __init__(self, pts, nx, ny, box=None):
        from scipy.spatial import Delaunay, cKDTree
        a, b = np.argsort(np.ptp(pts, 0))[::-1][:2]
        P = pts[:, [a, b]]
        lo, hi = P.min(0), P.max(0)
        if box:
            lo, hi = np.array([box[0], box[2]], float), np.array([box[1], box[3]], float)
        self.x = np.linspace(lo[0], hi[0], nx)
        self.y = np.linspace(lo[1], hi[1], ny)
        G = np.stack(np.meshgrid(self.x, self.y), -1).reshape(-1, 2)
        if len(P) > 300000:
            print(f"  triangulating {len(P)} cells ...")
        tri = Delaunay(P)
        simp = tri.find_simplex(G)
        T = tri.transform[simp]
        bary = np.einsum("ijk,ik->ij", T[:, :2], G - T[:, 2])
        self.w = np.c_[bary, 1 - bary.sum(1)]
        self.idx = tri.simplices[simp]
        h = np.median(cKDTree(P).query(P, k=2)[0][:, 1])
        d = cKDTree(P).query(G)[0]
        self.mask = ((simp >= 0) & (d < 1.5 * h)).reshape(ny, nx)
        self.shape = (ny, nx)
        self.axes = (a, b)

    def __call__(self, v):
        out = np.einsum("ij,ij...->i...", self.w, v[self.idx])
        out[~self.mask.ravel()] = 0
        return out.reshape(self.shape + v.shape[1:])


def to_grid(traj, nx, ny, box=None):
    if traj.points is None:
        raise CfdmlError(f"{traj.source}: no cell centres, so the fields cannot be put on a grid",
                         hint="run  postProcess -func writeCellCentres , or export with --no-grid")
    gm = GridMap(traj.points, nx, ny, box)
    F = np.stack([gm(f) for f in traj.fields]).transpose(0, 3, 1, 2)
    return Traj(F, traj.times, traj.names, x=gm.x, y=gm.y, mask=gm.mask, params=traj.params, source=traj.source)
