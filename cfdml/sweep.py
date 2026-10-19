import csv
import pickle
import warnings
from pathlib import Path

import numpy as np
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from . import foam
from .analysis import pod, rank_for, spectrum
from .errors import CfdmlError
from .table import read_table, uniform

class NotSettled(CfdmlError):
    pass


STATS = ("mean", "std", "rms", "min", "max", "final", "amplitude", "frequency", "period")


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise CfdmlError(f"config not found: {path}", hint="create one with:  cfdml init sweep")
    try:
        cfg = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise CfdmlError(f"{path} is not valid YAML: {e}", hint="check indentation; use spaces, not tabs")
    if not isinstance(cfg, dict) or "inputs" not in cfg:
        raise CfdmlError(f"{path} needs an 'inputs:' list", hint="see the template from  cfdml init sweep")
    cfg["_dir"] = path.parent.resolve()
    return cfg


def _stat(t, y, stat, after=None, before=None):
    if after == "auto":
        from .verify import transient_end
        tu, yu, _, _ = uniform(t, y)
        after, _ = transient_end(tu, yu)
        if after is None:
            raise NotSettled(f"still changing at the end of the record (t = {t[-1]:g})")
    elif after is not None:
        after = float(after)
    m = np.ones(len(t), bool)
    if after is not None:
        m &= t >= after
    if before is not None:
        m &= t <= before
    t, y = t[m], y[m]
    if len(y) < 2:
        raise CfdmlError(f"fewer than 2 samples between after={after} and before={before}")
    if stat == "mean":
        return float(y.mean())
    if stat == "std":
        return float(y.std())
    if stat == "rms":
        return float(np.sqrt(np.mean(y ** 2)))
    if stat == "min":
        return float(y.min())
    if stat == "max":
        return float(y.max())
    if stat == "final":
        return float(y[-1])
    if stat == "amplitude":
        return float(np.sqrt(2) * y.std())
    if stat in ("frequency", "period"):
        tu, yu, dt, _ = uniform(t, y)
        f0 = spectrum(yu, dt)["f0"]
        return f0 if stat == "frequency" else 1 / f0
    raise CfdmlError(f"unknown stat '{stat}'", hint="use one of: " + ", ".join(STATS))


def collect(cfg):
    inputs = list(cfg["inputs"])
    outs = cfg.get("outputs") or {}
    base = cfg["_dir"]
    runs = list(cfg.get("runs") or [])
    if cfg.get("table"):
        with open(base / cfg["table"]) as fh:
            for r in csv.DictReader(fh):
                runs.append({k.strip(): (float(v) if v not in ("", None) and _num(v) else v) for k, v in r.items()})
    if not runs:
        raise CfdmlError("no runs: add a 'runs:' list or a 'table:' CSV to the config")
    rows = []
    for i, r in enumerate(runs):
        name = r.get("name") or r.get("case") or f"run{i}"
        missing = [k for k in inputs if k not in r]
        if missing:
            raise CfdmlError(f"run '{name}' has no value for input(s) {missing}")
        row = dict(name=str(name), x={k: float(r[k]) for k in inputs}, y={}, case=None)
        if r.get("case"):
            row["case"] = (base / str(r["case"])).resolve()
            if not row["case"].exists():
                raise CfdmlError(f"run '{name}': case folder {row['case']} not found")
        for oname, spec in outs.items():
            if oname in r and r[oname] not in ("", None):
                row["y"][oname] = float(r[oname])
                continue
            if not isinstance(spec, dict) or "file" not in spec:
                continue
            if row["case"] is None:
                raise CfdmlError(f"run '{name}': output '{oname}' reads a file, but the run has no 'case:'")
            try:
                tb = read_table(row["case"] / spec["file"])
                if spec.get("after") == "auto":
                    try:
                        _stat(tb.time, tb[spec.get("column", tb.value_names[0])], spec.get("stat", "mean"), "auto")
                    except NotSettled as e:
                        print(f"  warning: run '{name}', output '{oname}' {e}; left out of the fit")
                        continue
                v = _stat(tb.time, tb[spec.get("column", tb.value_names[0])], spec.get("stat", "mean"),
                          spec.get("after"), spec.get("before"))
            except CfdmlError as e:
                raise CfdmlError(f"run '{name}', output '{oname}': {e}",
                                 hint=e.hint or f"the file ends at t = {tb.time[-1]:g}; did this run finish?"
                                 if "tb" in locals() else e.hint)
            row["y"][oname] = v * float(spec.get("scale", 1.0))
        rows.append(row)
    return inputs, rows


def _num(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


class Scaler:
    def __init__(self, X, log):
        self.log = np.asarray(log, bool)
        Z = self._fwd(X)
        self.mu, self.sd = Z.mean(0), Z.std(0)
        self.sd[self.sd == 0] = 1.0

    def _fwd(self, X):
        X = np.array(X, float)
        if np.any(X[:, self.log] <= 0):
            raise CfdmlError("an input marked as log has a value <= 0")
        X[:, self.log] = np.log(X[:, self.log])
        return X

    def __call__(self, X):
        return (self._fwd(X) - self.mu) / self.sd


def _gp(n_in):
    k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(n_in), (1e-2, 1e2), nu=2.5) + WhiteKernel(1e-4, (1e-8, 1e-1))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=0)


def _fit(Z, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return _gp(Z.shape[1]).fit(Z, y)


def _loo(Z, y):
    mu, sd = np.zeros(len(y)), np.zeros(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        g = _fit(Z[m], y[m])
        a, b = g.predict(Z[i:i + 1], return_std=True)
        mu[i], sd[i] = a[0], b[0]
    return mu, sd


def _relevance(g, n_in):
    ls = np.atleast_1d(g.kernel_.k1.k2.length_scale)
    inv = 1 / np.resize(ls, n_in)
    return inv / inv.sum()


class SweepModel:

    def __init__(self, inputs, log_inputs, rows):
        self.inputs, self.rows = inputs, rows
        self.X = np.array([[r["x"][k] for k in inputs] for r in rows])
        self.scaler = Scaler(self.X, [k in log_inputs for k in inputs])
        self.Z = self.scaler(self.X)
        self.lo, self.hi = self.X.min(0), self.X.max(0)
        self.scalar, self.fields = {}, {}

    def fit_scalar(self, name):
        have = np.array([name in r["y"] for r in self.rows])
        y = np.array([r["y"].get(name, np.nan) for r in self.rows])[have]
        if have.sum() < 3:
            raise CfdmlError(f"output '{name}' has {have.sum()} values; at least 3 runs are needed")
        Z = self.Z[have]
        g = _fit(Z, y)
        mu, sd = _loo(Z, y)
        ss = ((y - y.mean()) ** 2).sum()
        self.scalar[name] = dict(gp=g, y=y, have=have, loo=mu, loo_sd=sd,
                                 r2=float(1 - ((mu - y) ** 2).sum() / ss) if ss > 0 else float("nan"),
                                 rmse=float(np.sqrt(np.mean((mu - y) ** 2))),
                                 coverage=float(np.mean(np.abs(mu - y) <= 2 * sd + 1e-12)),
                                 relevance=_relevance(g, len(self.inputs)))

    def fit_field(self, name, time="latest", after=None, energy=0.999):
        snaps, cases = [], []
        for r in self.rows:
            if r["case"] is None:
                raise CfdmlError(f"field '{name}': run '{r['name']}' has no case folder")
            snaps.append(_run_field(r["case"], name, time, after))
            cases.append(r["case"])
        shapes = {s.shape for s in snaps}
        if len(shapes) > 1:
            raise CfdmlError(f"field '{name}' has different sizes across runs {sorted(shapes)}",
                             hint="field surrogates need every run on the same mesh")
        F = np.stack([s.reshape(-1) for s in snaps], 1)
        P = pod(F)
        k = min(rank_for(P["energy"], energy), F.shape[1] - 1)
        C = P["coeff"][:k].T
        gps = [_fit(self.Z, C[:, j]) for j in range(k)]
        loo_err = []
        for i in range(len(self.rows)):
            m = np.arange(len(self.rows)) != i
            Pi = pod(F[:, m])
            ki = min(k, m.sum() - 1)
            Ci = Pi["coeff"][:ki].T
            ci = np.array([_fit(self.Z[m], Ci[:, j]).predict(self.Z[i:i + 1])[0] for j in range(ki)])
            rec = Pi["mean"] + Pi["modes"][:, :ki] @ ci
            loo_err.append(float(np.linalg.norm(rec - F[:, i]) / max(np.linalg.norm(F[:, i] - F.mean(1)), 1e-30)))
        hdr = foam.field_header(_field_file(cases[0], name, time))
        self.fields[name] = dict(mean=P["mean"], modes=P["modes"][:, :k], gps=gps, shape=snaps[0].shape,
                                 energy=P["energy"], rank=k, loo_err=np.array(loo_err), header=hdr,
                                 centres=_safe_centres(cases[0]))

    def predict(self, Xnew):
        Z = self.scaler(np.atleast_2d(Xnew))
        out = {}
        for name, m in self.scalar.items():
            mu, sd = m["gp"].predict(Z, return_std=True)
            out[name] = (mu, sd)
        return out

    def predict_field(self, name, x):
        m = self.fields[name]
        Z = self.scaler(np.atleast_2d(x))
        c = np.array([g.predict(Z)[0] for g in m["gps"]])
        return (m["mean"] + m["modes"] @ c).reshape(m["shape"])

    def outside(self, x):
        x = np.asarray(x, float)
        return [k for k, v, a, b in zip(self.inputs, x, self.lo, self.hi) if v < a or v > b]

    def suggest(self, n=5, samples=4000, seed=0):
        rng = np.random.default_rng(seed)
        U = rng.random((samples, len(self.inputs)))
        log = self.scaler.log
        lo, hi = self.lo.copy(), self.hi.copy()
        lo[log], hi[log] = np.log(lo[log]), np.log(hi[log])
        Xc = lo + U * (hi - lo)
        Xc[:, log] = np.exp(Xc[:, log])
        Z = self.scaler(Xc)
        score = np.zeros(len(Xc))
        for m in self.scalar.values():
            _, sd = m["gp"].predict(Z, return_std=True)
            score += sd / (m["y"].std() or 1.0)
        picks = []
        for _ in range(n):
            i = int(np.argmax(score))
            picks.append((Xc[i], float(score[i] / max(len(self.scalar), 1))))
            score -= score[i] * np.exp(-np.sum((Z - Z[i]) ** 2, 1) / 0.1)
        return picks

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)


def load_model(path):
    path = Path(path)
    if path.is_dir():
        path = path / "model.pkl"
    if not path.exists():
        raise CfdmlError(f"model not found: {path}", hint="train one first with:  cfdml sweep <config.yaml>")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _field_file(case, name, time):
    ts = foam.times(case, [name])
    if not ts:
        raise CfdmlError(f"no time folder in {case} has the field '{name}'",
                         hint="fields found: " + ", ".join(sorted(foam.field_names(case))))
    if time == "latest":
        return ts[-1][1] / name
    t = float(time)
    return min(ts, key=lambda p: abs(p[0] - t))[1] / name


def _run_field(case, name, time, after):
    nc = foam.n_cells(case)
    if after is not None:
        ts = [(t, d) for t, d in foam.times(case, [name]) if t >= float(after)]
        if not ts:
            raise CfdmlError(f"{case}: no '{name}' at or after time {after}")
        return np.mean([foam.read_field(d / name, nc) for _, d in ts], 0)
    return foam.read_field(_field_file(case, name, time), nc)


def _safe_centres(case):
    try:
        return foam.cell_centres(case)
    except CfdmlError:
        return None


def parse_point(text, inputs):
    vals = {}
    for part in text.replace(" ", "").split(","):
        if "=" not in part:
            raise CfdmlError(f"cannot read '{part}'", hint="write values as  name=value,name=value  e.g. Re=350,angle=5")
        k, v = part.split("=", 1)
        vals[k] = float(v)
    missing = [k for k in inputs if k not in vals]
    extra = [k for k in vals if k not in inputs]
    if missing or extra:
        raise CfdmlError(f"inputs must be exactly {inputs}" + (f"; missing {missing}" if missing else "")
                         + (f"; unknown {extra}" if extra else ""))
    return np.array([vals[k] for k in inputs])
