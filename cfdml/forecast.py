import numpy as np

from .analysis import spectrum
from .errors import CfdmlError


def _period_steps(y, dt):
    f0 = spectrum(y, dt)["f0"]
    return max(2, int(round(1 / f0 / dt))) if f0 > 0 else len(y) // 2


def repeat_cycle(y, n, dt):
    p = min(_period_steps(y, dt), len(y))
    last = y[-p:]
    return np.resize(last, n)


def _lagged(y, p):
    return np.stack([y[i:len(y) - p + i] for i in range(p)], 1), y[p:]


def ar_fit(y, p, alpha):
    X, t = _lagged(y, p)
    X1 = np.c_[X, np.ones(len(X))]
    w = np.linalg.solve(X1.T @ X1 + alpha * np.eye(p + 1), X1.T @ t)
    return w


def ar_run(y, w, n):
    p = len(w) - 1
    buf = list(y[-p:])
    out = []
    for _ in range(n):
        v = float(np.dot(w[:p], buf[-p:]) + w[p])
        out.append(v); buf.append(v)
    return np.array(out)


def ar(y, n, dt, p=None):
    y = np.asarray(y, float)
    cut = int(len(y) * 0.8)
    per = _period_steps(y, dt)
    cands = sorted({min(max(2, k), cut // 3) for k in ([p] if p else [4, 8, 16, per // 2, per, 2 * per])})
    best = None
    for k in cands:
        for alpha in (1e-6, 1e-3, 1e-1):
            if cut - k < 2 * k:
                continue
            w = ar_fit(y[:cut], k, alpha)
            pred = ar_run(y[:cut], w, len(y) - cut)
            e = np.sqrt(np.mean((pred - y[cut:]) ** 2)) if np.all(np.isfinite(pred)) else np.inf
            if best is None or e < best[0]:
                best = (e, k, alpha)
    if best is None:
        raise CfdmlError("the training part is too short for the autoregressive model")
    w = ar_fit(y, best[1], best[2])
    return ar_run(y, w, n), dict(lags=best[1], ridge=best[2])


def hankel_dmd(y, n, dt, delays=None, energy=0.9999):
    y = np.asarray(y, float)
    mu = y.mean()
    z = y - mu
    d = delays or min(len(z) // 3, max(10, 2 * _period_steps(z, dt)))
    H = np.stack([z[i:len(z) - d + i + 1] for i in range(d)])
    X1, X2 = H[:, :-1], H[:, 1:]
    U, s, Vt = np.linalg.svd(X1, full_matrices=False)
    r = int(np.searchsorted(np.cumsum(s ** 2) / (s ** 2).sum(), energy) + 1)
    U, s, Vt = U[:, :r], s[:r], Vt[:r]
    A = U.T @ X2 @ Vt.T / s
    lam, W = np.linalg.eig(A)
    lam = np.where(np.abs(lam) > 1, lam / np.abs(lam), lam)
    Phi = U @ W
    b = np.linalg.lstsq(Phi, H[:, -1].astype(complex), rcond=None)[0]
    steps = np.arange(1, n + 1)
    out = (Phi[-1] * b) @ (lam[:, None] ** steps)
    return out.real + mu, dict(delays=d, rank=r)


METHODS = {"repeat-cycle": lambda y, n, dt: (repeat_cycle(y, n, dt), {}), "ar": ar, "hankel-dmd": hankel_dmd}


def evaluate(t, y, train=0.7, horizon=None, methods=None):
    y = np.asarray(y, float)
    dt = float(np.median(np.diff(t)))
    cut = int(len(y) * train)
    if cut < 30 or len(y) - cut < 5:
        raise CfdmlError(f"{len(y)} samples is too few to split {train:.0%} / {1 - train:.0%}",
                         hint="use a longer series or change --train")
    scale = np.std(y[cut:]) or 1.0
    res = {}
    for name in methods or METHODS:
        try:
            pred, info = METHODS[name](y[:cut], len(y) - cut, dt)
            nrmse = float(np.sqrt(np.mean((pred - y[cut:]) ** 2)) / scale)
            res[name] = dict(test_pred=pred, nrmse=nrmse if np.isfinite(nrmse) else np.inf, info=info)
        except (CfdmlError, np.linalg.LinAlgError, ValueError) as e:
            res[name] = dict(test_pred=None, nrmse=np.inf, info={"error": str(e)})
    best = min(res, key=lambda k: res[k]["nrmse"])
    h = horizon or (len(y) - cut)
    future, _ = METHODS[best](y, h, dt)
    good_until = None
    p = res[best]["test_pred"]
    if p is not None:
        err = np.abs(p - y[cut:]) / scale
        bad = np.nonzero(np.convolve(err > 0.5, np.ones(5) / 5, "same") > 0.6)[0]
        good_until = float((bad[0] if bad.size else len(err)) * dt)
    return dict(dt=dt, cut=cut, results=res, best=best, future=future,
                future_t=t[-1] + dt * np.arange(1, h + 1), good_for=good_until)
