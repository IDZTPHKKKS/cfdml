import re
from pathlib import Path

import numpy as np

from .errors import CfdmlError


class Table:

    def __init__(self, names, data, source=""):
        self.names, self.data, self.source = list(names), np.asarray(data, float), source

    def __contains__(self, name):
        return self._index(name, strict=False) is not None

    def _index(self, name, strict=True):
        if name in self.names:
            return self.names.index(name)
        low = [n.lower() for n in self.names]
        if name.lower() in low:
            return low.index(name.lower())
        if strict:
            raise CfdmlError(f"no column '{name}' in {self.source}",
                             hint="columns are: " + ", ".join(self.names[:40]) + (" ..." if len(self.names) > 40 else ""))
        return None

    def __getitem__(self, name):
        return self.data[:, self._index(name)]

    @property
    def time(self):
        for n in ("time", "Time", "t", "TIME", "flow time", "Flow Time"):
            if n in self.names:
                return self[n]
        return self.data[:, 0]

    @property
    def value_names(self):
        tn = next((n for n in self.names if n.lower() in ("time", "t", "flow time")), self.names[0])
        return [n for n in self.names if n != tn]


def _split(line):
    line = line.replace("(", " ").replace(")", " ")
    return [x for x in re.split(r"[,\s;]+", line.strip()) if x]


def _is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def read_table(path):
    path = Path(path)
    if path.is_dir():
        subs = sorted((float(d.name), d) for d in path.iterdir() if d.is_dir() and _is_number(d.name))
        files = [sorted(f for f in d.iterdir() if f.is_file()) for _, d in subs]
        names = {f.name for fs in files for f in fs}
        if subs and len(names) == 1:
            parts = [read_table(fs[0]) for fs in files if fs]
            out = parts[0]
            for p in parts[1:]:
                keep = out.time < p.time[0]
                out = Table(out.names, np.vstack([out.data[keep], p.data[:, :out.data.shape[1]]]), str(path))
            return out
        dats = sorted(path.rglob("*.dat")) + sorted(path.rglob("*.csv"))
        raise CfdmlError(f"{path} is a folder; give a file", hint="files found: " + ", ".join(str(d) for d in dats[:8])
                         if dats else None)
    if not path.exists():
        raise CfdmlError(f"file not found: {path}")
    comments, rows, header = [], [], None
    vector_cols = None
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("%"):
            comments.append(s.lstrip("#% ").strip())
            continue
        toks = _split(s)
        if not all(_is_number(t) for t in toks):
            if not rows:
                header = [t for t in re.split(r"[,\t;]+|\s{2,}", s.strip().strip(",")) if t] if ("," in s or "\t" in s) \
                    else s.split()
            continue
        if vector_cols is None:
            vector_cols = [len(t.split()) for t in re.findall(r"\(([^()]*)\)", s)]
        rows.append([float(t) for t in toks])
    if not rows:
        raise CfdmlError(f"no numeric rows in {path}")
    width = min(len(r) for r in rows)
    data = np.array([r[:width] for r in rows])
    names = header
    if names is None and comments:
        probe_lines = [c for c in comments if re.match(r"Probe\s+\d+\s*\(", c)]
        last = comments[-1]
        cand = [t for t in re.split(r"\t|\s{2,}|,", last) if t.strip()]
        if len(cand) < 2:
            cand = last.split()
        if probe_lines and cand and cand[0].lower() == "time" and vector_cols:
            k = vector_cols[0]
            names = ["Time"] + [f"probe{i}" + (f"_{'xyz'[j] if k == 3 else j}" if k > 1 else "")
                                for i in range(len(probe_lines)) for j in range(k)]
        elif len(cand) == width:
            names = cand
        elif vector_cols and len(cand) - 1 == len(vector_cols) and len(cand) - 1 + sum(k - 1 for k in vector_cols) == width - 1:
            names = [cand[0]] + [f"{c}_{'xyz'[j] if k == 3 else j}" for c, k in zip(cand[1:], vector_cols) for j in range(k)]
    if names is None or len(names) != width:
        names = ["time"] + [f"col{i}" for i in range(1, width)]
    names = [n.strip() for n in names]
    data = data[np.argsort(data[:, 0], kind="stable")]
    _, keep = np.unique(data[:, 0][::-1], return_index=True)
    data = data[np.sort(len(data) - 1 - keep)]
    return Table(names, data, str(path))


def uniform(t, y):
    dt = np.diff(t)
    if dt.size == 0:
        raise CfdmlError("the series has a single sample")
    step = float(np.median(dt))
    if np.max(np.abs(dt - step)) <= 0.01 * step:
        return t, y, step, False
    tu = np.arange(t[0], t[-1] + 0.5 * step, step)
    return tu, np.interp(tu, t, y), step, True
