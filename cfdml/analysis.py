import numpy as np
from scipy import signal

from .errors import CfdmlError


def spectrum(y, dt, segments=4):
    y = np.asarray(y, float) - np.mean(y)
    if len(y) < 16:
        raise CfdmlError(f"only {len(y)} samples; a spectrum needs at least 16")
    if not np.any(y):
        raise CfdmlError("the signal is constant, so it has no frequency")
    nper = max(16, len(y) // segments)
    f, P = signal.welch(y, fs=1 / dt, nperseg=nper, noverlap=nper // 2)
    pad = 1 << int(np.ceil(np.log2(len(y) * 8)))
    F = np.fft.rfftfreq(pad, dt)
    A = np.abs(np.fft.rfft(y * np.hanning(len(y)), pad))
    k = np.argmax(A[1:]) + 1
    if 1 <= k < len(A) - 1:
        a, b, c = np.log(A[k - 1:k + 2] + 1e-300)
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    f0 = float(k * (F[1] - F[0]))
    peaks, _ = signal.find_peaks(P, height=P.max() * 0.05)
    peaks = peaks[np.argsort(P[peaks])[::-1]][:5]
    cycles = f0 * dt * len(y)
    return dict(f=f, P=P, f0=f0, peaks=[float(f[i]) for i in peaks], cycles=float(cycles), resolution=float(1 / (dt * len(y))),
                f_fine=F, P_fine=A ** 2 / max(float((A ** 2).max()), 1e-300))


def snapshot_matrix(fields):
    X = np.stack([np.asarray(f, float).reshape(len(f), -1) for f in fields], -1)
    keep = [j for j in range(X.shape[1]) if np.ptp(X[:, j]) > 0 or X.shape[1] == 1]
    if not keep:
        raise CfdmlError("the field does not change between snapshots")
    return X[:, keep].reshape(-1, X.shape[2]), keep


def pod(X, weights=None, subtract_mean=True):
    X = np.asarray(X, float)
    mean = X.mean(1, keepdims=True) if subtract_mean else np.zeros((X.shape[0], 1))
    Y = X - mean
    w = np.ones(X.shape[0]) if weights is None else np.asarray(weights, float)
    sw = np.sqrt(w / w.mean())[:, None]
    U, s, Vt = np.linalg.svd(sw * Y, full_matrices=False)
    modes = U / sw
    energy = s ** 2 / max((s ** 2).sum(), 1e-300)
    coeff = s[:, None] * Vt
    return dict(mean=mean[:, 0], modes=modes, energy=energy, sv=s, coeff=coeff)


def rank_for(energy, target):
    return int(np.searchsorted(np.cumsum(energy), target) + 1)


def dmd(X, dt, rank=None, energy=0.999):
    X = np.asarray(X, float)
    if X.shape[1] < 3:
        raise CfdmlError("DMD needs at least 3 snapshots")
    X1, X2 = X[:, :-1], X[:, 1:]
    U, s, Vt = np.linalg.svd(X1, full_matrices=False)
    r = rank or rank_for(s ** 2 / (s ** 2).sum(), energy)
    r = min(r, len(s))
    U, s, Vt = U[:, :r], s[:r], Vt[:r]
    A = U.T @ X2 @ Vt.T / s
    lam, W = np.linalg.eig(A)
    Phi = X2 @ Vt.T / s @ W
    b = np.linalg.lstsq(Phi, X1[:, 0].astype(complex), rcond=None)[0]
    omega = np.log(lam.astype(complex)) / dt
    amp = np.abs(b) * np.linalg.norm(Phi, axis=0)
    order = np.argsort(amp)[::-1]
    return dict(lam=lam[order], freq=omega.imag[order] / (2 * np.pi), growth=omega.real[order], amp=amp[order],
                modes=Phi[:, order], b=b[order], rank=r)


def dmd_reconstruct(d, n, dt):
    t = np.arange(n) * dt
    om = np.log(d["lam"].astype(complex)) / dt
    return (d["modes"] * d["b"]) @ np.exp(np.outer(om, t))
