import numpy as np

from .analysis import spectrum
from .errors import CfdmlError


def align(pred, truth):
    if pred.fields.shape[1:] != truth.fields.shape[1:]:
        raise CfdmlError(f"prediction {pred.fields.shape[1:]} and truth {truth.fields.shape[1:]} have different shapes",
                         hint="both need the same grid / mesh and channels; export the truth with the same settings")
    if pred.names != truth.names:
        raise CfdmlError(f"channels differ: prediction {pred.names}, truth {truth.names}")
    n = min(len(pred.times), len(truth.times))
    return n


def rel_l2(P, T, axis=1):
    num = np.sqrt(np.sum((P - T) ** 2, axis))
    den = np.sqrt(np.sum((T - T.mean(0, keepdims=True)) ** 2, axis))
    return num / np.maximum(den, 1e-30)


def velocity_channels(names, given=None):
    if given:
        u, v = given.split(",")
        return names.index(u), names.index(v)
    for a, b in (("Ux", "Uy"), ("u", "v"), ("U0", "U1"), ("vx", "vy")):
        if a in names and b in names:
            return names.index(a), names.index(b)
    return None


def ke_spectrum(u, v, dx, dy, mask=None):
    ny, nx = u.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    if mask is not None:
        u, v = np.where(mask, u, 0), np.where(mask, v, 0)
    U = np.fft.fft2((u - u.mean()) * win)
    V = np.fft.fft2((v - v.mean()) * win)
    E = 0.5 * (np.abs(U) ** 2 + np.abs(V) ** 2)
    kx = np.fft.fftfreq(nx, dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, dy) * 2 * np.pi
    K = np.sqrt(kx[None] ** 2 + ky[:, None] ** 2)
    dk = 2 * np.pi / max(nx * dx, ny * dy)
    bins = np.arange(0.5 * dk, K.max(), dk)
    idx = np.digitize(K.ravel(), bins)
    Ek = np.bincount(idx, E.ravel(), len(bins) + 1)[1:-1]
    return 0.5 * (bins[1:] + bins[:-1]), Ek


def divergence(u, v, dx, dy, mask=None):
    du = np.gradient(u, dx, axis=-1)
    dv = np.gradient(v, dy, axis=-2)
    div = du + dv
    scale = np.sqrt(du ** 2 + np.gradient(u, dy, axis=-2) ** 2 + np.gradient(v, dx, axis=-1) ** 2 + dv ** 2)
    if mask is not None:
        from scipy.ndimage import binary_erosion
        inner = binary_erosion(mask, iterations=2)
        div, scale = div[..., inner], scale[..., inner]
    return float(np.mean(np.abs(div)) / max(np.mean(scale), 1e-30))


def pick_probes(traj, k=4):
    X = traj.flat()
    var = X.var(0).sum(1)
    order = np.argsort(var)[::-1]
    if traj.grid:
        ny, nx = traj.fields.shape[2:]
        cells = np.nonzero(traj.mask.ravel())[0] if traj.mask is not None else np.arange(ny * nx)
        pos = np.c_[cells % nx / nx, cells // nx / ny]
    elif traj.points is not None:
        P = traj.points
        pos = (P - P.min(0)) / np.maximum(np.ptp(P, 0), 1e-30)
    else:
        pos = np.c_[np.arange(len(var)) / len(var), np.zeros(len(var))]
    picks = []
    for i in order:
        if all(np.linalg.norm(pos[i] - pos[j]) > 0.15 for j in picks):
            picks.append(i)
        if len(picks) == k:
            break
    return picks


def recurrence(Y, dt, f0):
    if not f0 or f0 <= 0:
        return np.nan
    lag = int(round(1 / f0 / dt))
    best = np.inf
    for L in range(max(2, lag - 2), min(len(Y) - 4, lag + 3)):
        d = np.linalg.norm(Y[L:] - Y[:-L], axis=1).mean()
        best = min(best, d / max(np.linalg.norm(Y - Y.mean(0), axis=1).mean(), 1e-30))
    return float(best) if np.isfinite(best) else np.nan


def kind(r):
    if not np.isfinite(r):
        return "too short"
    return "periodic" if r < 0.25 else ("chaotic" if r > 0.6 else "in between")


def dynamics(Y, dt):
    Y = np.asarray(Y, float)
    if not np.isfinite(Y).all():
        return dict(f=np.nan, amp=np.nan, mean=np.nan, rec=np.nan, kind="blew up")
    if np.ptp(Y) < 1e-12 * max(1.0, np.abs(Y).max()):
        return dict(f=0.0, amp=0.0, mean=float(Y.mean()), rec=0.0, kind="steady")
    main = Y[:, np.argmax(Y.std(0))]
    sp = spectrum(main, dt) if len(main) >= 16 else dict(f0=np.nan, cycles=0)
    r = recurrence(Y, dt, sp["f0"]) if sp.get("cycles", 0) >= 2 else np.nan
    amp = float(Y.std(0).mean())
    return dict(f=float(sp["f0"]), amp=amp, mean=float(Y.mean()), rec=r,
                kind="steady" if amp < 1e-3 * max(abs(Y.mean()), 1e-12) else kind(r))


def windows(Y, dt, period_steps, n_periods=4):
    w = max(int(n_periods * period_steps), 16)
    starts = np.arange(0, max(len(Y) - w, 0) + 1, max(w // 4, 1))
    out = []
    for s in starts:
        seg = Y[s:s + w]
        main = seg[:, np.argmax(seg.std(0))]
        f = spectrum(main, dt)["f0"] if np.ptp(main) > 0 else 0.0
        out.append((s + w / 2, f, float(seg.std(0).mean())))
    return np.array(out)


def field_sample(traj, n=None, cap=2000, seed=0, idx=None):
    X = traj.flat()
    X = X[:n] if n else X
    if idx is None:
        rng = np.random.default_rng(seed)
        idx = np.arange(X.shape[1]) if X.shape[1] <= cap else np.sort(rng.choice(X.shape[1], cap, replace=False))
    return X[:, idx, :].reshape(len(X), -1), idx


def jumps(X, z=10.0):
    st = np.sqrt(np.nansum(np.diff(X, axis=0) ** 2, axis=tuple(range(1, X.ndim))))
    if st.size < 5:
        return np.array([], int), np.array([], int)
    med = np.median(st)
    mad = np.median(np.abs(st - med)) + 1e-12 * max(med, 1e-30)
    big = np.nonzero((st - med) / mad > z)[0]
    rep = np.nonzero(st <= 1e-12 * max(np.sqrt(np.nansum(X[0] ** 2)), 1e-30))[0]
    return big, rep
