import numpy as np

from .analysis import spectrum
from .errors import CfdmlError


def gci(h, phi, fs=1.25):
    h, phi = np.asarray(h, float), np.asarray(phi, float)
    if len(h) != 3 or len(phi) != 3:
        raise CfdmlError("a GCI study needs exactly three meshes")
    o = np.argsort(h)
    h, phi = h[o], phi[o]
    r21, r32 = h[1] / h[0], h[2] / h[1]
    if min(r21, r32) < 1.1:
        raise CfdmlError(f"refinement ratios {r21:.3f} and {r32:.3f} are too close to 1",
                         hint="each mesh should be at least 1.3 times finer than the next (Celik et al. 2008)")
    e21, e32 = phi[1] - phi[0], phi[2] - phi[1]
    if e21 == 0 or e32 == 0:
        raise CfdmlError("two meshes give the same value; the order of accuracy cannot be estimated")
    s = np.sign(e32 / e21)
    p, converged = 2.0, False
    with np.errstate(all="ignore"):
        for _ in range(500):
            q = np.log((r21 ** p - s) / (r32 ** p - s))
            p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
            if not np.isfinite(p_new):
                break
            if abs(p_new - p) < 1e-10:
                p, converged = p_new, True
                break
            p = p_new
    if not converged or p < 0.1 or r21 ** p - 1 < 1e-3:
        raise CfdmlError("the observed order of accuracy could not be determined (the values do not converge with "
                         "refinement, or change by the same amount between every pair of meshes)",
                         hint="check that the three values come from systematically refined meshes and are themselves "
                              "converged in time; a GCI is not meaningful here")
    ext = (r21 ** p * phi[0] - phi[1]) / (r21 ** p - 1)
    ea21 = abs((phi[0] - phi[1]) / phi[0])
    ea32 = abs((phi[1] - phi[2]) / phi[1])
    gci21 = fs * ea21 / (r21 ** p - 1)
    gci32 = fs * ea32 / (r32 ** p - 1)
    return dict(h=h, phi=phi, r21=r21, r32=r32, p=float(p), ext=float(ext), ea21=float(ea21),
                eext21=float(abs((ext - phi[0]) / ext)), gci21=float(gci21), gci32=float(gci32),
                asymptotic=float(gci32 / (r21 ** p * gci21)), oscillatory=bool(s < 0))


def _settled_from(b, tol):
    tail = b[-max(len(b) // 4, 2):]
    final = np.median(tail)
    band = tol + 3 * np.std(np.diff(tail)) / np.sqrt(2)
    off = np.nonzero(np.abs(b - final) > band)[0]
    return 0 if off.size == 0 else int(off[-1]) + 1


def transient_end(t, y, tol=0.02):
    dt = float(np.median(np.diff(t)))
    f0 = spectrum(y, dt)["f0"] if len(y) >= 16 and np.ptp(y) > 0 else 0
    period = int(round(1 / f0 / dt)) if f0 > 0 and 1 / f0 / dt < len(y) / 8 else 1
    w = max(2 * period, 10)
    m = len(y) // w
    if m < 8:
        return t[0], period
    blocks = y[len(y) - m * w:].reshape(m, w)
    scale = max(np.abs(y[-len(y) // 4:]).max(), 1e-30)
    d = _settled_from(blocks.mean(1), tol * scale)
    if period > 1:
        amp = blocks.std(1)
        d = max(d, _settled_from(amp, tol * max(np.median(amp[-max(m // 4, 2):]), 1e-30)))
    if d >= m - 1:
        return None, period
    return float(t[len(y) - m * w + d * w]), period


def batch_means(y, n_batches=20, min_len=1):
    n = len(y) // max(n_batches, 1)
    if n < min_len:
        n_batches = max(len(y) // min_len, 2)
        n = len(y) // n_batches
    b = y[:n * n_batches].reshape(n_batches, n).mean(1)
    return float(y.mean()), float(b.std(ddof=1) / np.sqrt(n_batches)), n_batches


REFERENCES = {
    "cylinder-st-williamson": dict(
        x="Re", y="St", lo=49, hi=178, f=lambda re: -3.3265 / re + 0.1816 + 1.6e-4 * re,
        cite="Williamson (1988), Phys. Fluids 31, 2742: parallel shedding, 49 < Re < 178"),
    "cylinder-st-fey": dict(
        x="Re", y="St", lo=47, hi=180, f=lambda re: 0.2684 - 1.0356 / np.sqrt(re),
        cite="Fey, Konig & Eckelmann (1998), Phys. Fluids 10, 1547: 47 < Re < 180"),
}


def time_weights(t):
    t = np.asarray(t, float)
    if len(t) < 2:
        return np.ones(len(t))
    d = np.diff(t)
    w = np.r_[d[0], d]
    return w / w.mean() if w.sum() > 0 else np.ones(len(t))
