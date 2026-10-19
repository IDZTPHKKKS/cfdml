import re
from pathlib import Path

import numpy as np

from .errors import CfdmlError

_TIME = re.compile(r"^Time = ([-+\d.eE]+)")
_SOLVE = re.compile(r"Solving for (\w+), Initial residual = ([^,]+), Final residual = ([^,]+), No Iterations (\d+)")
_CO = re.compile(r"^(Interface )?Courant Number mean: ([^ ]+) max: ([^ \n]+)")
_CONT = re.compile(r"continuity errors : sum local = ([^,]+), global = ([^,]+)")
_EXEC = re.compile(r"ExecutionTime = ([\d.eE+-]+) s")
_FATAL = re.compile(r"FOAM FATAL (IO )?ERROR|Floating point exception|sigFpe|Segmentation fault|FOAM aborting")


def _num(s):
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_log(path):
    path = Path(path)
    if not path.exists():
        raise CfdmlError(f"file not found: {path}")
    steps, cur, fatal, ended, last_lines, pending = [], None, None, False, [], {}
    for i, line in enumerate(path.read_text(errors="replace").splitlines()):
        last_lines = (last_lines + [line])[-30:]
        m = _TIME.match(line)
        if m:
            cur = dict(time=_num(m.group(1)), init={}, final={}, iters={}, co=np.nan, ico=np.nan, cont=np.nan, exec=np.nan)
            cur.update(pending)
            pending = {}
            steps.append(cur)
            continue
        if fatal is None and _FATAL.search(line):
            fatal = (i + 1, steps[-1]["time"] if steps else None)
        if line.strip() == "End":
            ended = True
        m = _CO.match(line)
        if m:
            key = "ico" if m.group(1) else "co"
            if cur is None or cur["init"]:
                pending[key] = max(_num(m.group(3)), pending.get(key, -1))
            else:
                cur[key] = max(_num(m.group(3)), np.nan_to_num(cur[key], nan=-1))
            continue
        if cur is None:
            continue
        m = _SOLVE.search(line)
        if m:
            f = m.group(1)
            if f not in cur["init"]:
                cur["init"][f] = _num(m.group(2))
            cur["final"][f] = _num(m.group(3))
            cur["iters"][f] = cur["iters"].get(f, 0) + int(m.group(4))
            continue
        m = _CONT.search(line)
        if m:
            cur["cont"] = abs(_num(m.group(1)))
            continue
        m = _EXEC.search(line)
        if m:
            cur["exec"] = _num(m.group(1))
    if not steps:
        raise CfdmlError(f"no 'Time = ...' lines in {path}", hint="is this an OpenFOAM solver log (e.g. log.pimpleFoam)?")
    fields = []
    for s in steps:
        for f in s["init"]:
            if f not in fields:
                fields.append(f)
    t = np.array([s["time"] for s in steps])
    init = {f: np.array([s["init"].get(f, np.nan) for s in steps]) for f in fields}
    iters = {f: np.array([s["iters"].get(f, np.nan) for s in steps]) for f in fields}
    out = dict(time=t, init=init, iters=iters, co=np.array([s["co"] for s in steps]),
               ico=np.array([s["ico"] for s in steps]), cont=np.array([s["cont"] for s in steps]),
               exec=np.array([s["exec"] for s in steps]), fatal=fatal, ended=ended, tail=last_lines, source=str(path))
    out["nan"] = [f for f in fields if any(f in s["init"] and not np.isfinite(s["init"][f]) for s in steps)]
    return out


def parse_solver_info(path):
    path = Path(path)
    lines = [ln for ln in path.read_text(errors="replace").splitlines() if ln.strip()]
    header = None
    rows = []
    for ln in lines:
        if ln.startswith("#"):
            toks = [x for x in re.split(r"\s+", ln.lstrip("#").strip()) if x]
            if toks and toks[0].lower() == "time":
                header = toks
            continue
        rows.append(re.split(r"\s+", ln.strip()))
    if header is None or not rows:
        raise CfdmlError(f"cannot read residuals from {path}", hint="expected a '# Time ...' header line")
    width = min(len(r) for r in rows)
    header = header[:width]
    cols = {h: [r[j] for r in rows] for j, h in enumerate(header)}
    t = np.array([_num(v) for v in cols[header[0]]])
    if any(h.endswith("_initial") for h in header):
        fields = [h[:-len("_initial")] for h in header if h.endswith("_initial")]
        init = {f: np.array([_num(v) for v in cols[f + "_initial"]]) for f in fields}
        iters = {f: np.array([_num(v) for v in cols[f + "_iters"]]) for f in fields if f + "_iters" in cols}
    else:
        fields = header[1:]
        init = {f: np.array([_num(v) for v in cols[f]]) for f in fields}
        iters = {}
    nan = np.full(len(t), np.nan)
    return dict(time=t, init=init, iters=iters, co=nan, ico=nan, cont=nan, exec=nan, fatal=None, ended=None,
                tail=[], nan=[], source=str(path))


def find_residuals(path):
    p = Path(path)
    if p.is_file():
        head = p.read_text(errors="replace")[:4000]
        return parse_solver_info(p) if p.suffix == ".dat" or head.lstrip().startswith("#") else parse_log(p)
    if p.is_dir():
        logs = sorted((f for f in p.glob("log.*") if "Foam" in f.name or f.name in ("log.run", "log.solver")),
                      key=lambda f: f.stat().st_mtime)
        if logs:
            return parse_log(logs[-1])
        dats = sorted(list(p.glob("postProcessing/solverInfo*/*/solverInfo*.dat"))
                      + list(p.glob("postProcessing/residuals*/*/residuals*.dat")), key=lambda f: f.stat().st_mtime)
        if dats:
            return parse_solver_info(dats[-1])
        raise CfdmlError(f"no solver log (log.*Foam) or residuals file in {p}",
                         hint="run the solver with  > log.pimpleFoam 2>&1  or add the solverInfo function object")
    raise CfdmlError(f"not found: {p}")
