import re
from pathlib import Path

import numpy as np

from .errors import CfdmlError
from .foam import _NUM


def _rows(path):
    comments, rows = [], []
    for ln in Path(path).read_text(errors="replace").splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            comments.append(s.lstrip("#").strip())
            continue
        toks = s.replace("(", " ").replace(")", " ").split()
        try:
            rows.append([float(x) for x in toks])
        except ValueError:
            continue
    return comments, rows


def _names(comments, width):
    for c in reversed(comments):
        if not c.lower().startswith("time"):
            continue
        low = c.lower()
        if "forces(pressure viscous porous)" in low or "forces(pressure viscous)" in low:
            parts = ["pressure", "viscous", "porous"] if "porous" in low else ["pressure", "viscous"]
            names = ["time"] + [f"{p}_{a}" for p in parts for a in "xyz"]
            if "moment(" in low:
                names += [f"moment_{p}_{a}" for p in parts for a in "xyz"]
            if len(names) == width:
                return names
        toks = re.findall(r"[A-Za-z]+_[xyz]", c)
        if len(toks) == width - 1:
            return ["time"] + toks
    return None


def read_forces(path):
    path = Path(path)
    files = [path]
    if path.is_dir():
        files = series_files(path, ("force.dat", "forces.dat"))
    data = None
    for f in files:
        comments, rows = _rows(f)
        if not rows:
            continue
        lengths = np.array([len(r) for r in rows])
        width = int(np.bincount(lengths).argmax())
        if (lengths < width).any():
            print(f"  warning: {f.name}: {(lengths < width).sum()} incomplete row(s) skipped")
        rows = [r for r in rows if len(r) >= width]
        names = _names(comments, width)
        if names is None:
            raise CfdmlError(f"cannot recognise the force file layout of {f}",
                             hint="expected an OpenFOAM forces function object output (force.dat or forces.dat)")
        a = np.array([r[:width] for r in rows])
        part = {n: a[:, j] for j, n in enumerate(names)}
        data = part if data is None else _append(data, part)
    if data is None:
        raise CfdmlError(f"no numeric rows in {path}")
    if "total_x" not in data:
        for a in "xyz":
            data[f"total_{a}"] = sum(data[f"{p}_{a}"] for p in ("pressure", "viscous", "porous") if f"{p}_{a}" in data)
    return data


def _append(a, b):
    t0 = b["time"][0]
    keep = a["time"] < t0
    return {k: np.concatenate([a[k][keep], b[k]]) for k in a if k in b}


def series_files(folder, names):
    folder = Path(folder)
    subs = sorted((float(d.name), d) for d in folder.iterdir() if d.is_dir() and _NUM.match(d.name))
    out = []
    for _, d in subs:
        for n in names:
            if (d / n).exists():
                out.append(d / n)
                break
    if not out:
        for n in names:
            if (folder / n).exists():
                return [folder / n]
        raise CfdmlError(f"no {' or '.join(names)} in {folder} or its time subfolders")
    return out
