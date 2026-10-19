import csv
import shutil
from pathlib import Path

import numpy as np

from . import foam
from .analysis import dmd, pod, rank_for, snapshot_matrix, spectrum
from .errors import CfdmlError
from .forecast import evaluate
from .report import COLORS, Report, fmt, plt
from .sweep import collect, load_config, load_model, parse_point, SweepModel
from .table import read_table, uniform

TEMPLATES = Path(__file__).parent / "templates"


def _out(args, default):
    return Path(args.out or default)


def _series(path, column, after=None, before=None):
    tb = read_table(path)
    col = column or tb.value_names[0]
    t, y = tb.time, tb[col]
    m = np.isfinite(t) & np.isfinite(y)
    if (~m).any():
        print(f"  note: skipped {(~m).sum()} rows with missing values in {col}")
    if after is not None:
        m &= t >= after
    if before is not None:
        m &= t <= before
    if m.sum() < 16:
        raise CfdmlError(f"only {m.sum()} samples left after the time limits", hint="check --after / --before")
    t, y, dt, resampled = uniform(t[m], y[m])
    return tb, col, t, y, dt, resampled


def info(args):
    p = Path(args.path)
    if foam.is_case(p):
        nc = foam.n_cells(p)
        ts = foam.times(p)
        print(f"OpenFOAM case: {p}")
        print(f"  cells: {nc}")
        print(f"  time folders: {len(ts)}" + (f"  ({fmt(ts[0][0])} ... {fmt(ts[-1][0])})" if ts else ""))
        for name, n in sorted(foam.field_names(p).items()):
            print(f"  field {name:<12} in {n} time folder(s)")
        pp = p / "postProcessing"
        if pp.is_dir():
            files = sorted(f for f in pp.rglob("*") if f.is_file() and f.suffix in (".dat", ".csv", "") and f.stat().st_size)
            print("  postProcessing files (use these with spectrum / forecast):")
            for f in files[:30]:
                try:
                    tb = read_table(f)
                    print(f"    {f.relative_to(p)}  columns: {', '.join(tb.names[:8])}{' ...' if len(tb.names) > 8 else ''}"
                          f"  ({len(tb.data)} rows)")
                except CfdmlError:
                    pass
        return
    if p.suffix.lower() in (".vtk", ".vtu", ".vtp", ".pvtu"):
        from .vtk import read_vtk
        pts, fields = read_vtk(p)
        print(f"VTK file: {p}\n  points/cells: {len(pts)}")
        for k, v in fields.items():
            print(f"  field {k:<12} shape {v.shape}")
        return
    tb = read_table(p)
    t = tb.time
    print(f"table: {p}\n  rows: {len(t)}, time {fmt(t[0])} ... {fmt(t[-1])}, step {fmt(float(np.median(np.diff(t))))}")
    print("  columns: " + ", ".join(tb.names))


def cmd_spectrum(args):
    tb, col, t, y, dt, resampled = _series(args.file, args.column, args.after, args.before)
    s = spectrum(y, dt)
    rep = Report(_out(args, f"cfdml_spectrum_{col}"), f"Spectrum of {col}")
    rep.say(f"file: `{args.file}`, column `{col}`, time {fmt(t[0])} to {fmt(t[-1])}, {len(y)} samples, dt {fmt(dt)}")
    if resampled:
        rep.say("the time step was not constant; the series was resampled to a uniform step first", summary=True)
    rep.say(f"dominant frequency: **{fmt(s['f0'])}** (period {fmt(1 / s['f0'])})", summary=True)
    if args.length and args.velocity:
        rep.say(f"Strouhal number St = f L / U = **{fmt(s['f0'] * args.length / args.velocity)}**", summary=True)
    rep.say(f"mean {fmt(float(y.mean()))}, standard deviation {fmt(float(y.std()))}, "
            f"amplitude (sqrt 2 x std) {fmt(float(np.sqrt(2) * y.std()))}", summary=True)
    if s["cycles"] < 5:
        rep.say(f"warning: only {s['cycles']:.1f} cycles in the window; the frequency is uncertain by about "
                f"+/-{fmt(s['resolution'] / 2)}. Use a longer signal for a reliable value.", summary=True)
    if len(s["peaks"]) > 1:
        rep.say("other peaks: " + ", ".join(fmt(f) for f in s["peaks"][1:]))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].plot(t, y, color=COLORS[0], lw=0.8)
    ax[0].set_xlabel("time"); ax[0].set_ylabel(col); ax[0].set_title("signal")
    fmax = min(s["f_fine"][-1], max(6 * s["f0"], max(s["peaks"] or [0]) * 1.5))
    k = (s["f_fine"] > 0) & (s["f_fine"] <= fmax)
    ax[1].semilogy(s["f_fine"][k], s["P_fine"][k] + 1e-12, color=COLORS[0], lw=0.9)
    ax[1].axvline(s["f0"], color=COLORS[3], ls="--", lw=0.8, label=f"f = {fmt(s['f0'])}")
    ax[1].set_ylim(1e-8, 2); ax[1].set_xlim(0, fmax)
    ax[1].set_xlabel("frequency"); ax[1].set_ylabel("power (relative to peak)"); ax[1].set_title("power spectrum")
    ax[1].legend()
    rep.figure(fig, "spectrum")
    rep.close()


def _check_case(case):
    if not foam.is_case(case) and not (case.is_dir() and (case / "system").is_dir()):
        raise CfdmlError(f"{case} does not look like an OpenFOAM case (no system/ folder)")
    if foam.is_decomposed(case):
        print(f"  decomposed case: reading {len(foam._proc_dirs(case))} processor folders")


def _case_snapshots(args):
    case = Path(args.case)
    _check_case(case)
    ts = foam.times(case, [args.field])
    ts = [(t, d) for t, d in ts if (args.after is None or t >= args.after) and (args.before is None or t <= args.before)]
    if args.every > 1:
        ts = ts[::args.every]
    if ts:
        foam._read_text(ts[0][1] / args.field)
    if len(ts) < 3:
        raise CfdmlError(f"found {len(ts)} time folders with '{args.field}' in the time window; need at least 3",
                         hint="fields found: " + ", ".join(sorted(foam.field_names(case))))
    nc = foam.n_cells(case)
    C, sel = None, None
    if getattr(args, "box", None):
        C = foam.cell_centres(case)
        b = [float(v) for v in args.box.split(",")]
        if len(b) != 6:
            raise CfdmlError("--box takes xmin,xmax,ymin,ymax,zmin,zmax")
        sel = np.nonzero(np.all((C >= b[0::2]) & (C <= b[1::2]), 1))[0]
        if sel.size == 0:
            raise CfdmlError("no cells inside --box", hint=f"the mesh spans {C.min(0).round(3)} to {C.max(0).round(3)}")
    n_used = nc if sel is None else len(sel)
    first = foam.read_field(ts[0][1] / args.field, nc)
    k = 1 if first.ndim == 1 else first.shape[1]
    gb = n_used * k * len(ts) * 8 * 4 / 1e9
    if gb > 4:
        print(f"  warning: this needs about {gb:.0f} GB of memory; use --every, --box or a shorter time window if it fails")
    print(f"  reading {args.field} from {len(ts)} time folders ({n_used} cells) ...")
    snaps = [foam.read_field(d / args.field, nc) for _, d in ts]
    if sel is not None:
        snaps = [f[sel] for f in snaps]
    tt = np.array([t for t, _ in ts])
    dts = np.diff(tt)
    if np.ptp(dts) > 0.01 * np.median(dts):
        print("  note: the time folders are not equally spaced; DMD frequencies assume they are")
    return case, tt, snaps, nc, sel


def _centres_2d(case, sel=None):
    try:
        C = foam.cell_centres(case)
    except CfdmlError:
        return None, None
    if sel is not None:
        C = C[sel]
    span = np.ptp(C, 0)
    a, b = np.argsort(span)[::-1][:2]
    return C, (a, b)


def _field_plot(ax, C, axes, values, title, cmap="RdBu_r", sym=True):
    a, b = axes
    v = np.asarray(values)
    lim = np.max(np.abs(v)) if sym else None
    tc = ax.tricontourf(C[:, a], C[:, b], v, levels=40, cmap=cmap, vmin=-lim if sym else None, vmax=lim if sym else None)
    ax.set_aspect("equal"); ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    return tc


def cmd_pod(args):
    case, tt, snaps, nc_all, sel = _case_snapshots(args)
    nc = len(snaps[0])
    X, keep = snapshot_matrix(snaps)
    V = foam.cell_volumes(case)
    if V is not None and sel is not None:
        V = V[sel]
    w = np.repeat(V, len(keep)) if V is not None else None
    P = pod(X, weights=w)
    rep = Report(_out(args, f"cfdml_pod_{args.field}"), f"POD of {args.field} in {case.name}")
    comp = "xyz" if snaps[0].ndim == 2 else ""
    rep.say(f"{len(tt)} snapshots from t = {fmt(tt[0])} to {fmt(tt[-1])}; components used: "
            + (", ".join(comp[k] for k in keep) if comp else "scalar")
            + ("; weighted by cell volume" if w is not None else "; equal cell weights (write V for volume weights)"))
    cum = np.cumsum(P["energy"])
    for target in (0.9, 0.99, 0.999):
        rep.say(f"modes for {target:.1%} of the fluctuation energy: **{rank_for(P['energy'], target)}**", summary=True)
    rep.table(["mode", "energy %", "cumulative %"],
              [[i + 1, f"{100 * P['energy'][i]:.2f}", f"{100 * cum[i]:.2f}"] for i in range(min(10, len(cum)))])
    pairs = [i for i in range(min(8, len(cum) - 1)) if abs(P["energy"][i] - P["energy"][i + 1]) < 0.1 * P["energy"][i]]
    if pairs:
        rep.say(f"modes {pairs[0] + 1} and {pairs[0] + 2} have nearly equal energy: together they usually describe "
                "one travelling structure (e.g. vortex shedding)", summary=True)
    n = min(args.modes, len(cum))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    k = min(20, len(cum))
    ax[0].bar(range(1, k + 1), 100 * P["energy"][:k], color=COLORS[0])
    ax[0].set_xlabel("mode"); ax[0].set_ylabel("energy %"); ax[0].set_title("energy per mode")
    for i in range(min(n, 4)):
        ax[1].plot(tt, P["coeff"][i], lw=0.9, color=COLORS[i], label=f"mode {i + 1}")
    ax[1].set_xlabel("time"); ax[1].set_title("time coefficients"); ax[1].legend(fontsize=7)
    rep.figure(fig, "pod_energy")
    C, axes = _centres_2d(case, sel)
    if C is not None:
        nk = len(keep)
        aspect = np.ptp(C[:, axes[1]]) / max(np.ptp(C[:, axes[0]]), 1e-30)
        fig, ax = plt.subplots(n, nk, figsize=(3.6 * nk, (3.6 * aspect + 0.45) * n), squeeze=False)
        for i in range(n):
            M = P["modes"][:, i].reshape(nc, nk)
            for j in range(nk):
                _field_plot(ax[i, j], C, axes, M[:, j], f"mode {i + 1}" + (f", {comp[keep[j]]}" if comp else ""))
        rep.figure(fig, "pod_modes", "POD modes")
    else:
        rep.say("no cell centres found, so the modes are saved but not drawn (run  postProcess -func writeCellCentres)")
    np.savez_compressed(rep.out / "pod.npz", times=tt, mean=P["mean"], modes=P["modes"][:, :max(n, 10)],
                        energy=P["energy"], coeff=P["coeff"][:max(n, 10)], components=np.array(keep))
    rep.say("modes, energies and time coefficients saved in `pod.npz`")
    if args.write_modes:
        hdr = foam.field_header(foam.times(case, [args.field])[0][1] / args.field)
        for i in range(n):
            M = P["modes"][:, i].reshape(nc, len(keep))
            idx = np.arange(nc_all) if sel is None else sel
            full = np.zeros((nc_all, snaps[0].shape[1])) if snaps[0].ndim == 2 else np.zeros(nc_all)
            if full.ndim == 2:
                full[np.ix_(idx, keep)] = M
            else:
                full[idx] = M[:, 0]
            foam.write_field(foam.out_target(case, rep.out / "foam_modes", f"{i + 1}", args.field), args.field, full, hdr,
                             note=f"POD mode {i + 1}")
        rep.say(f"modes written as OpenFOAM fields in `foam_modes/<n>/{args.field}`: copy a folder into the case as a time "
                "folder to view it in ParaView")
    rep.close()


def cmd_dmd(args):
    case, tt, snaps, nc_all, sel = _case_snapshots(args)
    nc = len(snaps[0])
    X, keep = snapshot_matrix(snaps)
    X = X - X.mean(1, keepdims=True)
    dt = float(np.median(np.diff(tt)))
    D = dmd(X, dt, rank=args.rank)
    rep = Report(_out(args, f"cfdml_dmd_{args.field}"), f"DMD of {args.field} in {case.name}")
    rep.say(f"{len(tt)} snapshots, dt = {fmt(dt)}, rank {D['rank']} (mean removed)")
    rows, shown = [], 0
    for i in range(len(D["freq"])):
        if D["freq"][i] < -1e-12:
            continue
        rows.append([shown + 1, fmt(D["freq"][i]), fmt(1 / D["freq"][i]) if D["freq"][i] > 1e-12 else "-",
                     fmt(D["growth"][i]), fmt(D["amp"][i] / D["amp"][0])])
        shown += 1
        if shown >= 10:
            break
    rep.table(["mode", "frequency", "period", "growth rate", "relative amplitude"], rows)
    osc = [r for r in rows if r[1] not in ("0", "-") and float(r[1]) > 1e-9]
    if osc:
        change = np.exp(float(osc[0][3]) * (tt[-1] - tt[0])) - 1
        trend = "steady" if abs(change) < 0.1 else f"{'growing' if change > 0 else 'decaying'} ({100 * change:+.0f}% over the window)"
        rep.say(f"strongest oscillation: frequency **{osc[0][1]}** (period {osc[0][2]}), {trend}", summary=True)
    if args.length and args.velocity and osc:
        rep.say(f"Strouhal number St = f L / U = **{fmt(float(osc[0][1]) * args.length / args.velocity)}**", summary=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    th = np.linspace(0, 2 * np.pi, 200)
    ax[0].plot(np.cos(th), np.sin(th), color="k", lw=0.6)
    sc = ax[0].scatter(D["lam"].real, D["lam"].imag, c=np.log10(D["amp"] / D["amp"].max() + 1e-12), cmap="viridis", s=18)
    ax[0].set_aspect("equal"); ax[0].set_title("eigenvalues (on the circle = steady oscillation)")
    plt.colorbar(sc, ax=ax[0], label="log10 relative amplitude")
    pos = D["freq"] >= 0
    ax[1].stem(D["freq"][pos], D["amp"][pos] / D["amp"].max(), basefmt=" ")
    ax[1].set_xlabel("frequency"); ax[1].set_ylabel("relative amplitude"); ax[1].set_title("DMD spectrum")
    rep.figure(fig, "dmd_spectrum")
    C, axes = _centres_2d(case, sel)
    if C is not None and osc:
        i = int(np.nonzero(D["freq"] > 1e-9)[0][0])
        M = D["modes"][:, i].reshape(nc, len(keep))
        fig, ax = plt.subplots(1, 2, figsize=(9, 2.6))
        _field_plot(ax[0], C, axes, M[:, 0].real, f"strongest oscillating mode, real part (f = {fmt(D['freq'][i])})")
        _field_plot(ax[1], C, axes, M[:, 0].imag, "imaginary part (a quarter period later)")
        rep.figure(fig, "dmd_mode")
    np.savez_compressed(rep.out / "dmd.npz", freq=D["freq"], growth=D["growth"], amp=D["amp"], lam=D["lam"],
                        modes=D["modes"][:, :10])
    rep.close()


def cmd_fieldstats(args):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    case = Path(args.case)
    _check_case(case)
    ts = [(t, d) for t, d in foam.times(case, [args.field])
          if (args.after is None or t >= args.after) and (args.before is None or t <= args.before)]
    if not ts:
        raise CfdmlError(f"no time folder has '{args.field}'", hint="fields found: " + ", ".join(sorted(foam.field_names(case))))
    nc = foam.n_cells(case)
    V = foam.cell_volumes(case)
    weighted = V is not None
    V = np.ones(nc) if V is None else V
    C = None
    try:
        C = foam.cell_centres(case)
    except CfdmlError:
        pass
    sel = np.ones(nc, bool)
    if args.box:
        if C is None:
            raise CfdmlError("--box needs cell centres", hint="run  postProcess -func writeCellCentres")
        b = [float(v) for v in args.box.split(",")]
        if len(b) != 6:
            raise CfdmlError("--box takes xmin,xmax,ymin,ymax,zmin,zmax")
        sel = np.all((C >= b[0::2]) & (C <= b[1::2]), 1)
    first = foam.read_field(ts[0][1] / args.field, nc)
    phase = first.ndim == 1 and (args.phase or (first.min() >= -1e-6 and first.max() <= 1 + 1e-6
                                               and args.field.lower().startswith("alpha")))
    graph = None
    if phase:
        f = foam.internal_faces(case)
        if f is not None:
            o, n = f
            m = sel[o] & sel[n]
            graph = (o[m], n[m])
    print(f"  reading {args.field} from {len(ts)} time folders ({nc} cells) ...")
    rows = []
    for t, d in ts:
        F = foam.read_field(d / args.field, nc)[sel]
        v = V[sel]
        r = {"time": t}
        if phase:
            vol = float((F * v).sum())
            r["phase_volume"] = vol
            if C is not None and vol > 0:
                for j, ax in enumerate("xyz"):
                    r[f"centroid_{ax}"] = float((F * v * C[sel][:, j]).sum() / vol)
            r["interface_cells"] = int(((F > 0.05) & (F < 0.95)).sum())
            if graph is not None:
                inside = np.zeros(nc, bool); inside[np.nonzero(sel)[0][F > args.threshold]] = True
                o, n = graph
                k = inside[o] & inside[n]
                idx = np.nonzero(inside)[0]
                pos = -np.ones(nc, int); pos[idx] = np.arange(len(idx))
                A = coo_matrix((np.ones(k.sum()), (pos[o[k]], pos[n[k]])), shape=(len(idx), len(idx)))
                r["blobs"] = int(connected_components(A, directed=False)[0]) if len(idx) else 0
        elif F.ndim == 2:
            mag = np.linalg.norm(F, axis=1)
            r.update(mean_magnitude=float((mag * v).sum() / v.sum()), max_magnitude=float(mag.max()),
                     kinetic_energy=float((0.5 * mag ** 2 * v).sum()))
            for j in range(F.shape[1]):
                r[f"mean_{'xyz'[j] if F.shape[1] == 3 else j}"] = float((F[:, j] * v).sum() / v.sum())
        else:
            r.update(mean=float((F * v).sum() / v.sum()), min=float(F.min()), max=float(F.max()),
                     integral=float((F * v).sum()))
        rows.append(r)
    names = list(rows[0])
    for r in rows:
        for k in names:
            r.setdefault(k, float("nan"))
    out = Path(args.out) if args.out else case / "postProcessing/cfdml_fieldstats"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.field}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(names)
        w.writerows([[r[k] for k in names] for r in rows])
    print(f"  {len(rows)} times -> {path}")
    print(f"  columns: {', '.join(names[1:])}")
    if phase:
        print("  (treated as a phase fraction" + (", blobs = separate regions with " + args.field + f" > {args.threshold})"
                                                   if graph is not None else "; no blob count: " + (
                                                       "droplets can cross processor boundaries in a decomposed case)"
                                                       if foam.is_decomposed(case) else "no ASCII polyMesh)")))
    if not weighted:
        print("  note: no cell volumes (V) found, so cells are weighted equally; for exact volumes and means run"
              "  postProcess -func writeCellVolumes  in the case")
    print(f"  use it with:  cfdml spectrum {path} -c <column>   or in a sweep config as a 'file:'")
    fig, axs = plt.subplots(1, min(3, len(names) - 1), figsize=(10, 2.8), squeeze=False)
    tt = np.array([r["time"] for r in rows])
    for a, k in zip(axs[0], names[1:4] if not phase else [k for k in ("phase_volume", "centroid_y", "blobs") if k in names]):
        a.plot(tt, [r[k] for r in rows], color=COLORS[0], lw=0.9); a.set_xlabel("time"); a.set_title(k)
    fig.tight_layout(); fig.savefig(out / f"{args.field}.png"); plt.close(fig)


def cmd_forecast(args):
    tb, col, t, y, dt, resampled = _series(args.file, args.column, args.after, args.before)
    horizon = int(round(args.horizon / dt)) if args.horizon else None
    R = evaluate(t, y, train=args.train, horizon=horizon)
    rep = Report(_out(args, f"cfdml_forecast_{col}"), f"Forecast of {col}")
    cut = R["cut"]
    rep.say(f"trained on t = {fmt(t[0])} to {fmt(t[cut - 1])}, tested on t = {fmt(t[cut])} to {fmt(t[-1])} "
            "(the test part was hidden from every method)")
    rows = []
    for name, r in sorted(R["results"].items(), key=lambda kv: kv[1]["nrmse"]):
        rows.append([name, fmt(r["nrmse"], 3) if np.isfinite(r["nrmse"]) else "failed",
                     ", ".join(f"{k}={v}" for k, v in r["info"].items())])
    rep.table(["method", "error on test part (RMSE / std)", "settings"], rows)
    b = R["results"][R["best"]]["nrmse"]
    base = R["results"]["repeat-cycle"]["nrmse"]
    rep.say(f"best method: **{R['best']}**, error {fmt(b, 3)} of the signal's spread "
            f"(repeat-last-cycle baseline: {fmt(base, 3)})", summary=True)
    if R["best"] == "repeat-cycle" or b > 0.9 * base:
        rep.say("no method beats simply repeating the last cycle by much: treat the forecast as 'more of the same'",
                summary=True)
    if b > 0.7:
        rep.say("the error is large: this signal is hard to predict from its own past (it may be chaotic or "
                "still changing); do not rely on the forecast", summary=True)
    if R["good_for"] is not None:
        test_len = t[-1] - t[cut]
        if R["good_for"] >= test_len - dt:
            rep.say(f"the best method stays within half a standard deviation for the whole test part "
                    f"({fmt(test_len)} time units)", summary=True)
        else:
            rep.say(f"the best method stays within half a standard deviation for about {fmt(R['good_for'])} of the "
                    f"{fmt(test_len)} time units of the test part: trust forecasts about that far ahead", summary=True)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(t, y, color="k", lw=1.0, label="data")
    ax.axvline(t[cut], color="gray", ls=":", lw=1)
    for i, (name, r) in enumerate(R["results"].items()):
        if r["test_pred"] is not None:
            ax.plot(t[cut:], r["test_pred"], lw=0.9, color=COLORS[i], label=f"{name} (test)")
    ax.plot(R["future_t"], R["future"], lw=1.2, color=COLORS[3], ls="--", label=f"forecast ({R['best']})")
    ax.set_xlabel("time"); ax.set_ylabel(col)
    ax.legend(fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    rep.figure(fig, "forecast")
    with open(rep.out / "forecast.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["time", col])
        w.writerows(zip(R["future_t"], R["future"]))
    rep.say(f"forecast values saved in `forecast.csv` ({len(R['future'])} steps)")
    rep.close()


def cmd_sweep(args):
    cfg = load_config(args.config)
    inputs, rows = collect(cfg)
    log_inputs = cfg.get("log_inputs") or []
    print(f"  {len(rows)} runs, inputs {inputs}")
    M = SweepModel(inputs, log_inputs, rows)
    outs = sorted({k for r in rows for k in r["y"]})
    rep = Report(_out(args, cfg["_dir"] / "cfdml_sweep"), f"Sweep: {Path(args.config).name}")
    rep.say(f"{len(rows)} runs; inputs: {', '.join(inputs)}" + (f" (log scale: {', '.join(log_inputs)})" if log_inputs else ""))
    with open(rep.out / "runs.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["name"] + inputs + outs)
        for r in rows:
            w.writerow([r["name"]] + [r["x"][k] for k in inputs] + [r["y"].get(o, "") for o in outs])
    rep.say("all collected values: `runs.csv`")
    for o in outs:
        print(f"  fitting {o} ...")
        M.fit_scalar(o)
    fields = cfg.get("fields") or {}
    for name, spec in fields.items():
        spec = spec or {}
        print(f"  fitting field {name} (reads every run's field) ...")
        M.fit_field(name, time=spec.get("time", "latest"), after=spec.get("mean_after"))
    rep.say("## How well each output can be predicted", summary=False)
    rep.say("Leave-one-out: each run is predicted by a model trained on all the other runs.")
    trows = []
    for o in outs:
        m = M.scalar[o]
        top = np.argsort(m["relevance"])[::-1]
        trows.append([o, fmt(m["r2"], 3), fmt(m["rmse"]), f"{100 * m['coverage']:.0f}%",
                      ", ".join(f"{inputs[i]} {100 * m['relevance'][i]:.0f}%" for i in top)])
        verdict = "good" if m["r2"] > 0.9 else "usable, check the error" if m["r2"] > 0.6 else "poor: needs more runs or another input"
        rep.say(f"**{o}**: R2 = {fmt(m['r2'], 3)} ({verdict}); typical error {fmt(m['rmse'])}; most important input: "
                f"{inputs[top[0]]}", summary=True)
    rep.table(["output", "R2 (1 = perfect)", "typical error", "inside +/-2 sigma (want ~95%)", "input importance"], trows)
    rep.say("Input importance: Sobol total-effect indices of the fitted model over the sweep range, normalised to "
            "100% (cfdml sensitivity gives the full indices with confidence intervals).")
    for o in outs:
        m = M.scalar[o]
        fig, ax = plt.subplots(1, 1 + len(inputs), figsize=(3.2 * (1 + len(inputs)), 3))
        y = m["y"]
        ax[0].errorbar(y, m["loo"], yerr=2 * m["loo_sd"], fmt="o", ms=4, color=COLORS[0], ecolor="#bbbbbb", lw=0.8)
        lim = [min(y.min(), m["loo"].min()), max(y.max(), m["loo"].max())]
        ax[0].plot(lim, lim, "k--", lw=0.8)
        ax[0].set_xlabel(f"{o} (CFD)"); ax[0].set_ylabel(f"{o} (predicted, run left out)"); ax[0].set_title("leave-one-out")
        Xh = M.X[m["have"]]
        for j, k in enumerate(inputs):
            g = np.geomspace(M.lo[j], M.hi[j], 80) if k in log_inputs else np.linspace(M.lo[j], M.hi[j], 80)
            Xg = np.tile(np.median(M.X, 0), (80, 1)); Xg[:, j] = g
            mu, sd = M.predict(Xg)[o]
            a = ax[1 + j]
            a.fill_between(g, mu - 2 * sd, mu + 2 * sd, color=COLORS[0], alpha=0.2, lw=0)
            a.plot(g, mu, color=COLORS[0])
            a.plot(Xh[:, j], y, "o", ms=3, color="k", alpha=0.5)
            a.set_xlabel(k); a.set_title(f"{o} vs {k} (others at median)")
            if k in log_inputs:
                a.set_xscale("log")
        rep.figure(fig, f"sweep_{o}")
    if len(inputs) == 2 and outs:
        o = outs[0]
        g0 = np.geomspace(M.lo[0], M.hi[0], 60) if inputs[0] in log_inputs else np.linspace(M.lo[0], M.hi[0], 60)
        g1 = np.geomspace(M.lo[1], M.hi[1], 60) if inputs[1] in log_inputs else np.linspace(M.lo[1], M.hi[1], 60)
        A, B = np.meshgrid(g0, g1)
        mu, sd = M.predict(np.c_[A.ravel(), B.ravel()])[o]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for a, v, ttl in ((ax[0], mu, f"{o} predicted"), (ax[1], sd, "uncertainty (1 sigma)")):
            c = a.contourf(A, B, v.reshape(A.shape), levels=30, cmap="viridis")
            plt.colorbar(c, ax=a)
            a.plot(M.X[:, 0], M.X[:, 1], "w.", ms=6, mec="k")
            a.set_xlabel(inputs[0]); a.set_ylabel(inputs[1]); a.set_title(ttl)
            if inputs[0] in log_inputs:
                a.set_xscale("log")
            if inputs[1] in log_inputs:
                a.set_yscale("log")
        rep.figure(fig, "sweep_map")
    for name, f in M.fields.items():
        rep.say(f"**field {name}**: {f['rank']} POD modes; leave-one-out field error median "
                f"{100 * np.median(f['loo_err']):.1f}%, worst {100 * f['loo_err'].max():.1f}% "
                "(relative to the spread between runs)", summary=True)
    if outs:
        sug = M.suggest(args.suggest)
        rep.say("## Where to run CFD next")
        rep.say("These conditions are inside your sweep range where the model is least sure. Running them improves "
                "the model the most.")
        rep.table(inputs + ["relative uncertainty"], [[fmt(v) for v in x] + [fmt(s, 3)] for x, s in sug])
        rep.say("next suggested run: " + ", ".join(f"{k}={fmt(v)}" for k, v in zip(inputs, sug[0][0])), summary=True)
    M.save(rep.out / "model.pkl")
    rep.say(f"model saved: `model.pkl` (use  cfdml predict {rep.out}/model.pkl --at ...)")
    rep.close()


def cmd_predict(args):
    M = load_model(args.model)
    points = []
    if args.at:
        for a in args.at:
            points.append(parse_point(a, M.inputs))
    if args.grid:
        axes = {}
        for g in args.grid:
            k, rng = g.split("=")
            lo, hi, n = rng.split(":")
            axes[k] = np.linspace(float(lo), float(hi), int(n))
        if set(axes) != set(M.inputs):
            missing = [k for k in M.inputs if k not in axes]
            raise CfdmlError(f"--grid needs every input; missing {missing}", hint="fix one with e.g. --grid angle=5:5:1")
        mesh = np.meshgrid(*[axes[k] for k in M.inputs], indexing="ij")
        points += list(np.stack([m.ravel() for m in mesh], 1))
    if not points:
        raise CfdmlError("give at least one --at point or a --grid", hint="e.g.  --at Re=350,angle=5")
    X = np.array(points)
    pred = M.predict(X)
    for x in X:
        out = M.outside(x)
        if out:
            print(f"  warning: {', '.join(out)} outside the sweep range for "
                  + ", ".join(f"{k}={fmt(v)}" for k, v in zip(M.inputs, x)) + ": this is extrapolation, expect larger errors")
    names = list(pred)
    print("  " + "  ".join(f"{k:>10}" for k in M.inputs) + "  " + "  ".join(f"{n:>22}" for n in names))
    for i, x in enumerate(X[:50]):
        print("  " + "  ".join(f"{fmt(v):>10}" for v in x) + "  "
              + "  ".join(f"{fmt(pred[n][0][i]):>11} +/- {fmt(2 * pred[n][1][i]):<8}" for n in names))
    if len(X) > 50:
        print(f"  ... {len(X) - 50} more rows in the CSV")
    out = Path(args.out or "cfdml_predictions")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(M.inputs + [c for n in names for c in (n, f"{n}_2sigma")])
        for i, x in enumerate(X):
            w.writerow(list(x) + [v for n in names for v in (pred[n][0][i], 2 * pred[n][1][i])])
    print(f"  (+/- is 2 sigma, about a 95% range)\n  saved {out}/predictions.csv")
    for fname in M.fields:
        for x in X[:args.max_fields]:
            tag = "_".join(f"{k}{fmt(v)}" for k, v in zip(M.inputs, x))
            F = M.predict_field(fname, x)
            f = M.fields[fname]
            foam.write_field(out / tag / fname, fname, F, f["header"],
                             note="predicted by cfdml; boundary values copied from the first run")
            if f["centres"] is not None:
                C = f["centres"]
                span = np.ptp(C, 0); a, b = np.argsort(span)[::-1][:2]
                v = np.linalg.norm(F, axis=1) if F.ndim == 2 else F
                fig, ax = plt.subplots(figsize=(7, 3))
                tc = _field_plot(ax, C, (a, b), v, f"predicted |{fname}| at " + tag.replace("_", ", "), cmap="viridis", sym=False)
                plt.colorbar(tc, ax=ax, shrink=0.75)
                fig.tight_layout(); fig.savefig(out / tag / f"{fname}.png"); plt.close(fig)
        print(f"  predicted field '{fname}' written to {out}/<point>/{fname} (OpenFOAM format) and {fname}.png")


def cmd_init(args):
    src = TEMPLATES / f"{args.kind}.yaml"
    dst = Path(args.out or f"{args.kind}.yaml")
    if dst.exists() and not args.force:
        raise CfdmlError(f"{dst} already exists", hint="choose another name with -o, or add --force to overwrite")
    shutil.copy(src, dst)
    print(f"  wrote {dst}: open it, fill in your runs, then run  cfdml sweep {dst}")


def cmd_demo(args):
    from .demo import make_demo
    make_demo(Path(args.out))


def cmd_gci(args):
    from .verify import gci
    name = args.name
    if args.cases:
        if len(args.cases) != 3 or not args.file:
            raise CfdmlError("--cases needs three case folders and --file", hint="e.g. --cases fine medium coarse --file "
                             "postProcessing/forceCoeffs/0/coefficient.dat -c Cd --after auto")
        cells, vals = [], []
        for c in args.cases:
            from .sweep import _stat
            tb = read_table(Path(c) / args.file)
            col = args.column or tb.value_names[0]
            name = name or col
            vals.append(_stat(tb.time, tb[col], args.stat, "auto" if args.after is None else args.after))
            cells.append(foam.n_cells(c))
        h = np.array(cells, float) ** (-1 / args.dim)
    else:
        if not args.values or not (args.cells or args.h):
            raise CfdmlError("give --values and --cells (or --h), or --cases",
                             hint="e.g.  cfdml gci --cells 25000,100000,400000 --values 1.42,1.39,1.385")
        vals = [float(v) for v in args.values.split(",")]
        if args.h:
            h = np.array([float(v) for v in args.h.split(",")])
        else:
            cells = [float(v) for v in args.cells.split(",")]
            h = np.array(cells) ** (-1 / args.dim)
    name = name or "value"
    R = gci(h, vals)
    rep = Report(_out(args, f"cfdml_gci_{name}"), f"Mesh convergence of {name}")
    rep.say("Celik et al. (2008), J. Fluids Eng. 130, 078001; safety factor 1.25")
    rep.table(["mesh", "h (relative)", name], [[lab, fmt(hh / R["h"][0], 4), fmt(v, 6)]
                                               for lab, hh, v in zip(("fine", "medium", "coarse"), R["h"], R["phi"])])
    rep.say(f"refinement ratios {R['r21']:.3f} (medium/fine), {R['r32']:.3f} (coarse/medium)")
    rep.say(f"observed order of accuracy p = **{R['p']:.2f}**", summary=True)
    rep.say(f"extrapolated (mesh-independent) {name} = **{fmt(R['ext'], 6)}**", summary=True)
    rep.say(f"GCI on the fine mesh = **{100 * R['gci21']:.2f}%**, i.e. {name} = {fmt(R['phi'][0], 6)} "
            f"+/- {fmt(R['gci21'] * abs(R['phi'][0]), 3)}", summary=True)
    rep.say(f"GCI on the medium mesh = {100 * R['gci32']:.2f}%")
    a = R["asymptotic"]
    sane = 0.5 <= R["p"] <= 4 and not R["oscillatory"]
    if sane:
        rep.say(f"asymptotic range check GCI32 / (r^p GCI21) = {a:.3f} "
                + ("(close to 1: in the asymptotic range)" if abs(a - 1) < 0.1 else
                   "(not close to 1: the meshes may not be in the asymptotic range; treat the GCI with care)"), summary=True)
    if R["oscillatory"]:
        rep.say("warning: the values oscillate with refinement (fine-medium and medium-coarse differences have "
                "opposite signs); the GCI is less reliable", summary=True)
    if R["p"] > 4 or R["p"] < 0.5:
        rep.say(f"warning: p = {R['p']:.2f} is far from the scheme's formal order; check the meshes and the "
                "quantity (e.g. averaging window)", summary=True)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    hh = np.r_[0, R["h"]] / R["h"][0]
    ax.plot(R["h"] / R["h"][0], R["phi"], "o-", color=COLORS[0], label="meshes")
    ax.errorbar([1], [R["phi"][0]], yerr=[R["gci21"] * abs(R["phi"][0])], color=COLORS[3], capsize=4, label="GCI (fine)")
    ax.plot([0], [R["ext"]], "s", color="k", label="extrapolated")
    g = np.linspace(0, hh[-1], 50)
    ax.plot(g, R["ext"] + (R["phi"][0] - R["ext"]) * g ** R["p"], "--", color="#8c8c8c", lw=0.8)
    ax.set_xlabel("h / h_fine"); ax.set_ylabel(name); ax.legend(fontsize=7)
    rep.figure(fig, "gci")
    rep.close()


def cmd_steady(args):
    tb = read_table(args.file)
    cols = args.column or [c for c in ("Cd", "Cl") if c in tb] or tb.value_names[:3]
    t = tb.time
    if len(t) < 100:
        raise CfdmlError(f"the file has only {len(t)} samples, up to t = {fmt(t[-1])}", hint="did the run finish?")
    ok = np.isfinite(t)
    for c in cols:
        ok &= np.isfinite(tb[c])
    if (~ok).any():
        print(f"  note: skipped {(~ok).sum()} rows with missing values")
    t = t[ok]
    data = {c: tb[c][ok] for c in cols}
    rep = _steady_core(args, t, data, cols, f"Is the run settled? {Path(args.file).name}", "cfdml_steady")
    if rep is not None:
        rep.close()


def _steady_core(args, t, data, cols, title, out_default):
    from scipy import stats
    from .verify import batch_means, transient_end
    ends, periods = {}, {}
    for c in cols:
        tu, yu, dt, _ = uniform(t, data[c])
        ends[c], periods[c] = transient_end(tu, yu)
    period = max(periods.values())
    never = [c for c, e in ends.items() if e is None]
    if never and args.after is None:
        print(f"  {', '.join(never)} still changing at the end of the record (t = {fmt(t[-1])}): the run has not "
              "settled; run longer (or give --after to force statistics)")
        return None
    t0 = args.after if args.after is not None else max(ends.values())
    rep = Report(_out(args, out_default), title)
    rep.t0, rep.period, rep.stats = t0, period, {}
    span = t[-1] - t[0]
    rep.say(f"record t = {fmt(t[0])} to {fmt(t[-1])}; " + (f"start-up transient ends at t = **{fmt(t0)}**"
            if args.after is None else f"statistics from t = {fmt(t0)} (given)") +
            f" ({100 * (t0 - t[0]) / span:.0f}% of the record)", summary=True)
    if (t0 - t[0]) > 0.5 * span:
        rep.say("the transient takes more than half of the record: run longer before trusting these statistics",
                summary=True)
    rows, verdicts = [], []
    fig, ax = plt.subplots(len(cols), 1, figsize=(9, 2.2 * len(cols)), squeeze=False)
    for a, c in zip(ax[:, 0], cols):
        tu, yu, dt, _ = uniform(t, data[c])
        k = tu >= t0
        y = yu[k]
        if len(y) < 40:
            raise CfdmlError(f"only {len(y)} samples after t = {fmt(t0)}", hint="the run is too short; run longer")
        m, se, nb = batch_means(y, 20, 2 * period)
        if nb < 8:
            rows.append([c, fmt(float(y.mean()), 5), "-", fmt(float(y.std()), 4), "-",
                         "too few cycles after the transient for an error bar: run longer"])
            rep.say(f"**{c}**: mean {fmt(float(y.mean()), 5)}, too few cycles after the transient for an error bar "
                    f"({nb} independent batches, need 8): run longer", summary=True)
            a.plot(tu, yu, color=COLORS[0], lw=0.6); a.axvspan(tu[0], t0, color="#c44e52", alpha=0.08, lw=0)
            a.set_ylabel(c)
            continue
        ci = stats.t.ppf(0.975, nb - 1) * se
        half = len(y) // 2
        m1, se1, _ = batch_means(y[:half], 10, 2 * period)
        m2, se2, _ = batch_means(y[half:], 10, 2 * period)
        drift = abs(m2 - m1) / max(np.hypot(se1, se2), 1e-300)
        sd = float(y.std())
        a1, a2 = y[:half].std(), y[half:].std()
        amp_change = (a2 - a1) / max(a1, 1e-300) if period > 1 and sd > 1e-5 + 1e-3 * abs(m) else 0.0
        zero_mean = abs(m) < 0.1 * sd
        rel = ci / abs(m) if not zero_mean else None
        need = None
        if rel is not None and rel > args.target:
            need = (tu[-1] - t0) * ((rel / args.target) ** 2 - 1)
        if abs(amp_change) > 0.05:
            v = f"oscillation amplitude still {'growing' if amp_change > 0 else 'decaying'} ({100 * amp_change:+.0f}% between halves): run longer"
        elif drift > 3:
            v = "still drifting: run longer"
        elif need:
            v = f"settled, but the mean is only known to {100 * rel:.1f}%; about {fmt(need, 3)} more time units for {100 * args.target:g}%"
        else:
            v = "settled"
        verdicts.append(v)
        rep.stats[c] = dict(mean=m, ci=ci, std=sd, verdict=v)
        rows.append([c, fmt(m, 5), f"+/- {fmt(ci, 3)}" + (f" ({100 * rel:.2f}%)" if rel is not None else ""),
                     fmt(sd, 4), f"{drift:.1f}", v])
        rep.say(f"**{c}**: mean {fmt(m, 5)} +/- {fmt(ci, 3)} (95%), {v}", summary=True)
        a.plot(tu, yu, color=COLORS[0], lw=0.6)
        a.axvspan(tu[0], t0, color="#c44e52", alpha=0.08, lw=0)
        a.axhline(m, color="k", lw=0.8); a.axhspan(m - ci, m + ci, color="k", alpha=0.12, lw=0)
        a.set_ylabel(c)
    ax[-1, 0].set_xlabel("time")
    ax[0, 0].set_title("shaded red: start-up transient; grey band: 95% interval of the mean")
    rep.table(["column", "mean", "95% interval", "std", "drift (sigma)", "verdict"], rows)
    rep.say("Error bars from batch means (batches of at least two periods), so correlated samples are not counted "
            "as independent. Drift compares the first and second half of the settled part.")
    rep.figure(fig, "steady")
    return rep


def _load_xy(path, x, y):
    with open(path, newline="") as fh:
        sample = fh.read(4096)
    if "," in sample and not sample.lstrip().startswith("#"):
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        names = list(rows[0]) if rows else []
        for c in (x, y):
            if c not in names:
                raise CfdmlError(f"no column '{c}' in {path}", hint="columns are: " + ", ".join(names))
        pts = [(float(r[x]), float(r[y])) for r in rows if r[x] not in ("", None) and r[y] not in ("", None)]
        return np.array(pts).T
    tb = read_table(path)
    return tb[x], tb[y]


def cmd_compare(args):
    from .verify import REFERENCES
    if args.list:
        for k, r in REFERENCES.items():
            print(f"  {k:<26} {r['y']} vs {r['x']}: {r['cite']}")
        return
    if not args.file or not args.x or not args.y:
        raise CfdmlError("give a results file with -x and -y", hint="e.g.  cfdml compare runs.csv -x Re -y St --ref "
                         "cylinder-st-williamson   (cfdml compare --list shows built-in references)")
    X, Y = _load_xy(args.file, args.x, args.y)
    o = np.argsort(X); X, Y = X[o], Y[o]
    if args.ref:
        if args.ref not in REFERENCES:
            raise CfdmlError(f"unknown reference '{args.ref}'", hint="built-in: " + ", ".join(REFERENCES))
        R = REFERENCES[args.ref]
        lo, hi, cite = R["lo"], R["hi"], R["cite"]
        ref = lambda x: R["f"](x)
        gx = np.linspace(max(lo, X.min() * 0.9), min(hi, X.max() * 1.1), 200)
        refx, refy = gx, ref(gx)
    elif args.ref_file:
        rx, ry = _load_xy(args.ref_file, args.ref_x or args.x, args.ref_y or args.y)
        o = np.argsort(rx); rx, ry = rx[o], ry[o]
        lo, hi, cite = rx.min(), rx.max(), str(args.ref_file)
        ref = lambda x: np.interp(x, rx, ry)
        refx, refy = rx, ry
    else:
        raise CfdmlError("give --ref NAME or --ref-file FILE", hint="cfdml compare --list shows built-in references")
    inside = (X >= lo) & (X <= hi)
    rep = Report(_out(args, f"cfdml_compare_{args.y}"), f"{args.y} vs {args.x} against {args.ref or Path(args.ref_file).name}")
    rep.say(f"reference: {cite}")
    err = 100 * (Y - ref(X)) / ref(X)
    rows = [[fmt(x), fmt(y, 5), fmt(float(ref(x)), 5) if i else "outside range", f"{e:+.2f}%" if i else "-"]
            for x, y, e, i in zip(X, Y, err, inside)]
    rep.table([args.x, f"{args.y} (yours)", f"{args.y} (reference)", "error"], rows)
    if inside.any():
        e = err[inside]
        rep.say(f"{inside.sum()} points inside the reference range: mean |error| **{np.mean(np.abs(e)):.2f}%**, "
                f"max {np.max(np.abs(e)):.2f}% at {args.x} = {fmt(X[inside][np.argmax(np.abs(e))])}; "
                f"mean signed error {np.mean(e):+.2f}%", summary=True)
        bad = X[inside][np.abs(e) > args.tol]
        if bad.size:
            rep.say(f"outside +/-{args.tol:g}%: {args.x} = " + ", ".join(fmt(b) for b in bad), summary=True)
        if abs(np.mean(e)) > 0.7 * np.mean(np.abs(e)) and inside.sum() > 3:
            rep.say("the errors share one sign: a systematic offset (e.g. blockage, domain size, mesh or time step) "
                    "rather than scatter", summary=True)
    if (~inside).any():
        rep.say(f"{(~inside).sum()} points outside the reference range were not compared", summary=True)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    ax[0].plot(refx, refy, color="k", lw=1.2, label="reference")
    ax[0].plot(X, Y, "o", ms=4, color=COLORS[0], label="yours")
    ax[0].set_xlabel(args.x); ax[0].set_ylabel(args.y); ax[0].legend(fontsize=7)
    ax[1].axhspan(-args.tol, args.tol, color="#55a868", alpha=0.12, lw=0)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].plot(X[inside], err[inside], "o", ms=4, color=COLORS[0])
    ax[1].set_xlabel(args.x); ax[1].set_ylabel("error %"); ax[1].set_title(f"band: +/-{args.tol:g}%")
    rep.figure(fig, "compare")
    rep.close()


def _end_time(case):
    cd = Path(case) / "system/controlDict"
    if cd.exists():
        import re
        m = re.search(r"^\s*endTime\s+([-+\d.eE]+)\s*;", cd.read_text(errors="replace"), re.M)
        if m:
            return float(m.group(1))
    return None


def cmd_residuals(args):
    from .logs import find_residuals
    R = find_residuals(args.path)
    t = R["time"]
    rep = Report(_out(args, "cfdml_residuals"), f"Residuals: {Path(R['source']).name}")
    rep.say(f"source: `{R['source']}`; {len(t)} time steps, t = {fmt(t[0])} to {fmt(t[-1])}")
    if R["fatal"]:
        line, tf = R["fatal"]
        rep.say(f"**the solver crashed** (line {line} of the log" + (f", at t = {fmt(tf)}" if tf is not None else "") + ")",
                summary=True)
        rep.lines += ["", "last lines of the log:", "", "```"] + R["tail"][-15:] + ["```", ""]
        for ln in R["tail"][-6:]:
            print("    | " + ln)
    elif R["ended"]:
        rep.say("the run finished (End)", summary=True)
    elif R["ended"] is not None:
        rep.say("no 'End' and no error: the run is still going, or was stopped", summary=True)
    if R["nan"]:
        rep.say(f"**NaN residuals** in {', '.join(R['nan'])}: the solution diverged", summary=True)
    rows = []
    steady_like = len(t) > 2 and np.allclose(np.diff(t[np.isfinite(t)]), 1)
    for f, r in R["init"].items():
        ok = np.isfinite(r) & (r > 0)
        if not ok.any():
            continue
        rr = r[ok]
        tail = rr[-max(len(rr) // 10, 1):]
        drop = np.log10(rr[0] / np.median(tail)) if rr[0] > 0 else np.nan
        lt = np.log10(rr[-max(len(rr) // 5, 2):])
        slope = np.polyfit(np.arange(len(lt)), lt, 1)[0] * len(lt) if len(lt) > 2 else 0.0
        trend = "still falling" if slope < -0.3 else ("rising" if slope > 0.3 else "flat")
        it = R["iters"].get(f)
        rows.append([f, fmt(rr[0], 3), fmt(float(np.median(tail)), 3), f"{drop:.1f}", trend,
                     fmt(float(np.nanmean(it[-max(len(it) // 10, 1):])), 3) if it is not None and len(it) else "-"])
    rep.table(["field", "first initial residual", "recent initial residual", "orders dropped", "last 20%",
               "iterations per step (recent)"], rows)
    worst = max(rows, key=lambda r: float(r[2])) if rows else None
    if worst:
        if steady_like:
            ok = float(worst[3]) >= 3 and worst[4] != "rising"
            rep.say(f"steady-state run: residuals dropped at least {min(float(r[3]) for r in rows):.1f} orders; "
                    + ("converged by the usual 3-orders rule" if ok else
                       f"{worst[0]} has not dropped 3 orders or is rising: not converged"), summary=True)
        else:
            rep.say(f"transient run: highest recent initial residual {worst[2]} ({worst[0]})"
                    + ("; above 1e-3, consider more outer correctors or a smaller time step"
                       if float(worst[2]) > 1e-3 else ""), summary=True)
    co = R["co"][np.isfinite(R["co"])]
    if co.size:
        over = (co > 1).sum()
        rep.say(f"Courant number: max {fmt(float(co.max()), 3)}, recent max {fmt(float(co[-1]), 3)}"
                + (f"; above 1 in {over} of {co.size} steps" if over else ""), summary=True)
    ico = R["ico"][np.isfinite(R["ico"])]
    if ico.size:
        rep.say(f"interface Courant number: max {fmt(float(ico.max()), 3)}", summary=True)
    cont = R["cont"][np.isfinite(R["cont"])]
    if cont.size:
        rep.say(f"continuity error (sum local): recent {fmt(float(cont[-1]), 3)}, max {fmt(float(cont.max()), 3)}")
    ex = R["exec"]
    k = np.isfinite(ex)
    if k.sum() > 10:
        te, xe = t[k], ex[k]
        rate = (xe[-1] - xe[max(0, len(xe) - len(xe) // 5 - 1)]) / max(te[-1] - te[max(0, len(te) - len(te) // 5 - 1)], 1e-300)
        rep.say(f"wall time: {fmt(float(xe[-1]) / 3600, 3)} h so far, {fmt(rate, 3)} s per unit of simulated time recently")
        tend = _end_time(Path(R["source"]).parent)
        if tend and tend > te[-1] and not R["ended"]:
            rep.say(f"at this speed, endTime {fmt(tend)} is reached in about {fmt((tend - te[-1]) * rate / 3600, 3)} h",
                    summary=True)
    fig, ax = plt.subplots(2 if co.size else 1, 1, figsize=(9, 5.5 if co.size else 3.2), squeeze=False, sharex=True)
    for i, (f, r) in enumerate(R["init"].items()):
        ax[0, 0].semilogy(t, r, lw=0.8, color=COLORS[i % len(COLORS)], label=f)
    ax[0, 0].set_ylabel("initial residual"); ax[0, 0].legend(fontsize=7, ncol=6)
    if co.size:
        ax[1, 0].plot(t, R["co"], color="k", lw=0.8, label="Courant max")
        if ico.size:
            ax[1, 0].plot(t, R["ico"], color=COLORS[3], lw=0.8, label="interface Courant max")
        ax[1, 0].axhline(1, color=COLORS[3], ls=":", lw=0.8); ax[1, 0].set_ylabel("Courant"); ax[1, 0].legend(fontsize=7)
    ax[-1, 0].set_xlabel("time" if not steady_like else "iteration")
    rep.figure(fig, "residuals")
    with open(rep.out / "residuals.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        names = list(R["init"])
        w.writerow(["time"] + names + ["courant_max"])
        for j in range(len(t)):
            w.writerow([t[j]] + [R["init"][n][j] for n in names] + [R["co"][j]])
    rep.close()


def _find_force_file(path):
    p = Path(path)
    if p.is_file() or (p.is_dir() and not (p / "system").is_dir()):
        return p
    cands = sorted({f.parent.parent for f in p.glob("postProcessing/*/*/force*.dat")})
    if not cands:
        raise CfdmlError(f"no forces function object output in {p}/postProcessing",
                         hint="add a forces function object to system/controlDict")
    if len(cands) > 1:
        print("  several force outputs found, using the first: " + ", ".join(c.name for c in cands))
    return cands[0]


def cmd_forces(args):
    from .forces import read_forces
    src = _find_force_file(args.path)
    F = read_forces(src)
    d, l = args.drag, args.lift
    t = F["time"]
    coeff = args.rho and args.velocity and args.area
    q = 0.5 * args.rho * args.velocity ** 2 * args.area if coeff else 1.0
    names = ("Cd", "Cl") if coeff else (f"F{d}", f"F{l}")
    data = {names[0]: F[f"total_{d}"] / q, names[1]: F[f"total_{l}"] / q}
    if "pressure_x" in F:
        data[names[0] + "_pressure"] = F[f"pressure_{d}"] / q
        data[names[0] + "_viscous"] = F[f"viscous_{d}"] / q
    out = _out(args, "cfdml_forces")
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / "forces.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["time"] + list(data))
        w.writerows(zip(t, *data.values()))
    print(f"  {len(t)} samples from {src}; time series saved as {out}/forces.csv")
    if not coeff:
        print("  give --rho, -U and --area to get coefficients (Cd, Cl) instead of forces")
    rep = _steady_core(args, t, data, list(names), f"Forces: {Path(src).name}", "cfdml_forces")
    if rep is None:
        return
    k = t >= rep.t0
    if names[0] + "_pressure" in data and rep.stats.get(names[0]):
        mp, mv = data[names[0] + "_pressure"][k].mean(), data[names[0] + "_viscous"][k].mean()
        tot = mp + mv
        if tot:
            rep.say(f"drag split: pressure {100 * mp / tot:.1f}%, viscous {100 * mv / tot:.1f}%", summary=True)
    lift = data[names[1]][k]
    if lift.std() > 1e-9 * max(1.0, abs(lift.mean())):
        tu, yu, dt, _ = uniform(t[k], lift)
        sp = spectrum(yu, dt)
        rep.say(f"{names[1]}: rms {fmt(float(np.sqrt(np.mean((yu - yu.mean()) ** 2))), 4)}, frequency {fmt(sp['f0'])}"
                + (f", St = {fmt(sp['f0'] * args.length / args.velocity)}" if args.length and args.velocity else ""),
                summary=True)
    rep.close()


def _transient_from_case(case):
    from .verify import transient_end
    files = sorted(Path(case).glob("postProcessing/*/*/coefficient.dat")) + sorted(Path(case).glob("postProcessing/*/*/force*.dat"))
    for f in files:
        try:
            if f.name == "coefficient.dat":
                tb = read_table(f.parent.parent)
                col = "Cl" if "Cl" in tb else tb.value_names[0]
                t, y = tb.time, tb[col]
            else:
                from .forces import read_forces
                F = read_forces(f.parent.parent)
                t, y = F["time"], F["total_y"]
            tu, yu, _, _ = uniform(t, y)
            te, _ = transient_end(tu, yu)
            if te is None:
                raise CfdmlError(f"the forces in {f.parent.parent} are still changing at the end of the run",
                                 hint="the run has not settled; give --after explicitly to average anyway")
            return te, f
        except CfdmlError as e:
            if "still changing" in str(e):
                raise
            continue
    raise CfdmlError("--after auto needs forces or forceCoeffs output in postProcessing to find the transient",
                     hint="give --after <time> instead")


def cmd_average(args):
    case = Path(args.case)
    _check_case(case)
    after = args.after
    if after == "auto":
        after, src = _transient_from_case(case)
        print(f"  transient ends at t = {fmt(after)} (from {src.relative_to(case)})")
    elif after is not None:
        after = float(after)
    nc = foam.n_cells(case)
    rep = Report(_out(args, "cfdml_average"), f"Time averages in {case.name}")
    for field in args.field:
        ts = [(t, d) for t, d in foam.times(case, [field])
              if (after is None or t >= after) and (args.before is None or t <= args.before)][::args.every]
        if len(ts) < 2:
            raise CfdmlError(f"found {len(ts)} time folders with '{field}' in the window; need at least 2",
                             hint="fields found: " + ", ".join(sorted(foam.field_names(case))))
        print(f"  averaging {field} over {len(ts)} time folders ...")
        n, mean, m2, half = 0, None, None, None
        for i, (_, d) in enumerate(ts):
            x = foam.read_field(d / field, nc)
            x = x.reshape(nc, -1)
            if mean is None:
                mean = np.zeros_like(x); m2 = np.zeros((nc, x.shape[1], x.shape[1]))
            n += 1
            delta = x - mean
            mean += delta / n
            m2 += delta[:, :, None] * (x - mean)[:, None, :]
            if i == len(ts) // 2 - 1:
                half = mean.copy()
        cov = m2 / max(n - 1, 1)
        k = mean.shape[1]
        mfield = mean[:, 0] if k == 1 else mean
        if k == 1:
            pfield = cov[:, 0, 0]
        else:
            iu = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)] if k == 3 else [(i, j) for i in range(k) for j in range(i, k)]
            pfield = np.stack([cov[:, i, j] for i, j in iu], 1)
        tl = ts[-1][1].name
        hdr = foam.field_header(ts[-1][1] / field)
        mean_name, p_name = f"{field}Mean", f"{field}Prime2Mean"
        hdr_p = dict(hdr, cls="volSymmTensorField" if k == 3 else hdr["cls"],
                     dims=_square_dims(hdr["dims"]))
        for into in ([False, True] if args.into_case else [False]):
            for nm, val, h in ((mean_name, mfield, hdr), (p_name, pfield, hdr_p)):
                fp = ts[-1][1] / nm if into else foam.out_target(case, rep.out, tl, nm)
                if into and fp.exists() and not args.force:
                    raise CfdmlError(f"{fp} already exists", hint="add --force to overwrite it")
                foam.write_field(fp, nm, val, _calc_boundary(h),
                                 note=f"cfdml average: {len(ts)} snapshots, t = {fmt(ts[0][0])} to {fmt(ts[-1][0])}")
        change = float(np.linalg.norm(mean - half) / max(np.linalg.norm(mean - mean.mean(0)), 1e-300)) if half is not None else np.nan
        rep.say(f"**{field}**: {len(ts)} snapshots from t = {fmt(ts[0][0])} to {fmt(ts[-1][0])}; wrote {mean_name} and "
                f"{p_name}" + (f" into the case ({tl}/)" if args.into_case else f" to {rep.out.name}/{tl}/"), summary=True)
        rep.say(f"  the mean after half the snapshots differs from the final mean by {100 * change:.1f}% "
                + ("(well converged)" if change < 0.02 else "(average over a longer time for a converged mean)"
                   if change > 0.05 else "(acceptable)"), summary=True)
        C, axes = _centres_2d(case)
        if C is not None:
            s = _slab(C, axes)
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
            mag = np.linalg.norm(mean, axis=1) if k > 1 else mean[:, 0]
            rms = np.sqrt(np.trace(cov, axis1=1, axis2=2) / k)
            tc = _field_plot(ax[0], C[s], axes, mag[s], f"{mean_name}" + (" magnitude" if k > 1 else ""),
                             cmap="viridis", sym=False)
            plt.colorbar(tc, ax=ax[0], shrink=0.8)
            tc = _field_plot(ax[1], C[s], axes, rms[s], f"rms fluctuation of {field}", cmap="magma", sym=False)
            plt.colorbar(tc, ax=ax[1], shrink=0.8)
            rep.figure(fig, f"average_{field}")
    rep.close()


def _square_dims(d):
    try:
        v = [float(x) for x in d.strip("[]").split()]
        return "[" + " ".join(f"{2 * x:g}" for x in v) + "]"
    except ValueError:
        return d


def _calc_boundary(h):
    zero = {"volVectorField": "(0 0 0)", "volSymmTensorField": "(0 0 0 0 0 0)", "volTensorField": "(0 0 0 0 0 0 0 0 0)"}
    z = zero.get(h["cls"], "0")
    return dict(h, boundary="boundaryField\n{\n    \".*\"\n    {\n        type calculated;\n        value uniform "
                + z + ";\n    }\n}\n")


def _slab(C, axes):
    third = ({0, 1, 2} - set(axes)).pop()
    span = np.ptp(C, 0)
    if span[third] <= 0.05 * max(span[axes[0]], span[axes[1]]):
        return np.ones(len(C), bool)
    mid = np.median(C[:, third])
    width = np.sort(np.abs(C[:, third] - mid))[min(len(C) - 1, max(len(C) // 50, 100))]
    return np.abs(C[:, third] - mid) <= width


def cmd_sensitivity(args):
    from .sweep import sobol
    M = load_model(args.model)
    if len(M.inputs) < 2:
        raise CfdmlError("sensitivity indices need at least two inputs")
    outs = args.output or list(M.scalar)
    rep = Report(_out(args, "cfdml_sensitivity"), "Sobol sensitivity indices")
    rep.say(f"inputs sampled uniformly over the sweep range (log-uniform for {', '.join(k for k, l in zip(M.inputs, M.scaler.log) if l) or 'none'}); "
            f"{args.n} base samples through the fitted Gaussian-process mean; 95% intervals by bootstrap")
    fig, ax = plt.subplots(1, len(outs), figsize=(3.6 * len(outs) + 0.6, 3), squeeze=False)
    for a, o in zip(ax[0], outs):
        if o not in M.scalar:
            raise CfdmlError(f"no output '{o}' in the model", hint="outputs: " + ", ".join(M.scalar))
        S = sobol(M, o, n=args.n)
        rows = [[k, f"{S['S1'][i]:.3f} [{S['S1_ci'][0][i]:.3f}, {S['S1_ci'][1][i]:.3f}]",
                 f"{S['ST'][i]:.3f} [{S['ST_ci'][0][i]:.3f}, {S['ST_ci'][1][i]:.3f}]"] for i, k in enumerate(M.inputs)]
        rep.say(f"### {o}  (leave-one-out R2 of the model {fmt(M.scalar[o]['r2'], 3)})")
        rep.table(["input", "first order S1", "total ST"], rows)
        inter = 1 - S["S1"].sum()
        rep.say(f"**{o}**: " + ", ".join(f"{k} {S['ST'][i]:.2f}" for i, k in enumerate(M.inputs))
                + f" (total effect); interactions carry {max(inter, 0):.0%} of the variance", summary=True)
        x = np.arange(len(M.inputs))
        a.bar(x - 0.18, S["S1"], 0.36, color=COLORS[0], label="first order",
              yerr=np.abs(S["S1_ci"] - S["S1"]), capsize=2, error_kw=dict(lw=0.7))
        a.bar(x + 0.18, S["ST"], 0.36, color=COLORS[1], label="total",
              yerr=np.abs(S["ST_ci"] - S["ST"]), capsize=2, error_kw=dict(lw=0.7))
        a.set_xticks(x); a.set_xticklabels(M.inputs); a.set_ylim(0, 1.05); a.set_title(o)
        if M.scalar[o]["r2"] < 0.8:
            rep.say(f"  the model of {o} is not accurate (R2 {fmt(M.scalar[o]['r2'], 3)}), so treat these indices "
                    "as rough", summary=True)
    ax[0, 0].legend(fontsize=7)
    rep.say("S1: share of the output variance from that input alone. ST: including all its interactions. "
            "ST - S1 large means the input matters mostly through interactions.")
    rep.figure(fig, "sensitivity")
    rep.close()


def _constraint(text, outputs):
    import re
    m = re.match(r"^\s*([\w.()]+)\s*(<=|>=|<|>)\s*([-+\d.eE]+)\s*$", text)
    if not m:
        raise CfdmlError(f"cannot read constraint '{text}'", hint="write it like  St<=0.21  or  Cl_amp>=0.5")
    name, op, v = m.group(1), m.group(2), float(m.group(3))
    if name not in outputs:
        raise CfdmlError(f"constraint on unknown output '{name}'", hint="outputs: " + ", ".join(outputs))
    return name, op.startswith("<"), v


def cmd_optimize(args):
    from scipy.optimize import minimize
    from scipy.stats import norm
    from .sweep import sample_inputs
    M = load_model(args.model)
    goal = args.minimize or args.maximize
    if not goal or (args.minimize and args.maximize):
        raise CfdmlError("give exactly one of --minimize or --maximize", hint="e.g.  --minimize Cd")
    if goal not in M.scalar:
        raise CfdmlError(f"no output '{goal}' in the model", hint="outputs: " + ", ".join(M.scalar))
    sign = 1 if args.minimize else -1
    cons = [_constraint(c, list(M.scalar)) for c in args.subject_to or []]
    fixed = {}
    for f in args.fix or []:
        k, v = f.split("=")
        if k not in M.inputs:
            raise CfdmlError(f"unknown input '{k}' in --fix", hint="inputs: " + ", ".join(M.inputs))
        fixed[M.inputs.index(k)] = float(v)
    free = [j for j in range(len(M.inputs)) if j not in fixed]
    log = M.scaler.log

    lo2, hi2 = M.lo.astype(float).copy(), M.hi.astype(float).copy()
    lo2[log], hi2[log] = np.log(lo2[log]), np.log(hi2[log])

    def to_x(u):
        x = np.empty(len(M.inputs))
        for j, v in fixed.items():
            x[j] = v
        for i, j in enumerate(free):
            val = lo2[j] + min(max(u[i], 0.0), 1.0) * (hi2[j] - lo2[j])
            x[j] = np.exp(val) if log[j] else val
        return x

    def pfeas(pred, i=None):
        p = np.ones_like(pred[goal][0])
        for name, le, v in cons:
            mu, sd = pred[name]
            z = (v - mu) / np.maximum(sd, 1e-12)
            p = p * (norm.cdf(z) if le else norm.cdf(-z))
        return p

    rng = np.random.default_rng(0)
    X = sample_inputs(M, 20000, rng, fixed)
    pred = M.predict(X)
    obj, pf = sign * pred[goal][0], pfeas(pred)
    feas = pf >= args.confidence
    if not feas.any():
        best = int(np.argmax(pf))
        raise CfdmlError(f"no point in the sweep range meets the constraints with {args.confidence:.0%} confidence "
                         f"(best is {pf[best]:.0%})", hint="relax the constraints, lower --confidence, or run more cases "
                                                            "where they are met")
    starts = np.argsort(np.where(feas, obj, np.inf))[:8]

    def penalised(u):
        x = to_x(u)[None]
        pr = M.predict(x)
        return float(sign * pr[goal][0][0] + 1e3 * max(0.0, args.confidence - pfeas(pr)[0]))

    def to_u(x):
        return np.array([((np.log(x[j]) if log[j] else x[j]) - lo2[j]) / ((hi2[j] - lo2[j]) or 1.0) for j in free])

    def objective(u):
        return float(sign * M.predict(to_x(u)[None])[goal][0][0])

    def margin(u):
        return float(pfeas(M.predict(to_x(u)[None]))[0] - args.confidence)

    best_u, best_f = None, np.inf
    for s in starts:
        if not free:
            break
        u0 = to_u(X[s])
        if cons:
            r = minimize(objective, u0, method="SLSQP", bounds=[(0, 1)] * len(free),
                         constraints=[dict(type="ineq", fun=margin)], options=dict(maxiter=200, ftol=1e-10))
            if margin(r.x) < -1e-6:
                continue
            f = objective(r.x)
        else:
            r = minimize(penalised, u0, method="L-BFGS-B", bounds=[(0, 1)] * len(free))
            f = r.fun
        if f < best_f:
            best_u, best_f = np.clip(r.x, 0, 1), f
    if free and best_u is None:
        best_u = to_u(X[starts[0]])
    xb = to_x(best_u) if free else X[starts[0]]
    pb = M.predict(xb[None])
    rep = Report(_out(args, f"cfdml_optimize_{goal}"), f"{'Minimise' if args.minimize else 'Maximise'} {goal}")
    rep.say("searched the sweep range on the fitted Gaussian processes" + (f", with {', '.join(args.subject_to)} required at "
            f"{args.confidence:.0%} confidence" if cons else "") + (f"; fixed {', '.join(args.fix)}" if fixed else ""))
    rep.say("optimum (predicted): " + ", ".join(f"{k}={fmt(v)}" for k, v in zip(M.inputs, xb)), summary=True)
    rep.say(f"  {goal} = {fmt(pb[goal][0][0], 5)} +/- {fmt(2 * pb[goal][1][0], 3)} (2 sigma)", summary=True)
    for name, le, v in cons:
        mu, sd = pb[name]
        p = norm.cdf((v - mu[0]) / max(sd[0], 1e-12)) if le else norm.cdf((mu[0] - v) / max(sd[0], 1e-12))
        rep.say(f"  {name} = {fmt(mu[0], 5)} +/- {fmt(2 * sd[0], 3)} ({p:.0%} chance of {name} {'<=' if le else '>='} {fmt(v)})",
                summary=True)
    edge = [k for j, k in enumerate(M.inputs) if j in free and (np.isclose(xb[j], M.lo[j], rtol=1e-3) or
                                                               np.isclose(xb[j], M.hi[j], rtol=1e-3))]
    if edge:
        rep.say(f"  the optimum sits on the edge of the sweep range in {', '.join(edge)}: the true optimum may lie "
                "outside it; extend the sweep in that direction", summary=True)
    y = M.scalar[goal]["y"]
    have = M.scalar[goal]["have"]
    Xr = M.X[have]
    pr = M.predict(Xr)
    ok_runs = pfeas(pr) >= 0.5
    for j, v in fixed.items():
        ok_runs &= np.isclose(Xr[:, j], v, rtol=0.02)
    if ok_runs.any():
        i = int(np.argmin(np.where(ok_runs, sign * y, np.inf)))
        rep.say("best run you already have: " + ", ".join(f"{k}={fmt(v)}" for k, v in zip(M.inputs, Xr[i]))
                + f", {goal} = {fmt(y[i], 5)}", summary=True)
        fbest = sign * y[i]
        mu, sd = pred[goal]
        z = (fbest - sign * mu) / np.maximum(sd, 1e-12)
        ei = (fbest - sign * mu) * norm.cdf(z) + sd * norm.pdf(z)
        score = ei * pf
        j = int(np.argmax(score))
    if ok_runs.any():
        rep.say("run to do next (largest expected improvement that meets the constraints): "
                + ", ".join(f"{k}={fmt(v)}" for k, v in zip(M.inputs, X[j])), summary=True)
    rep.say("Predictions come from the surrogate. Confirm the optimum with a CFD run before relying on it.")
    if len(free) in (1, 2):
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        if len(free) == 1:
            j = free[0]
            g = np.linspace(0, 1, 300)
            Xg = np.array([to_x(np.array([u])) for u in g])
            pg = M.predict(Xg)
            mu, sd = pg[goal]
            ok = pfeas(pg) >= args.confidence
            ax.fill_between(Xg[:, j], mu - 2 * sd, mu + 2 * sd, color=COLORS[0], alpha=0.2, lw=0)
            ax.plot(Xg[:, j], np.where(ok, mu, np.nan), color=COLORS[0], label="meets constraints")
            ax.plot(Xg[:, j], np.where(ok, np.nan, mu), color="#bbbbbb", label="violates")
            ax.plot(xb[j], pb[goal][0][0], "*", ms=14, color=COLORS[3], label="optimum")
            ax.set_xlabel(M.inputs[j]); ax.set_ylabel(goal); ax.legend(fontsize=7)
            if log[j]:
                ax.set_xscale("log")
        else:
            a, b = free
            g = np.linspace(0, 1, 70)
            U1, U2 = np.meshgrid(g, g)
            Xg = np.array([to_x(np.array([p, q])) for p, q in zip(U1.ravel(), U2.ravel())])
            pg = M.predict(Xg)
            mu = pg[goal][0].reshape(U1.shape)
            ok = (pfeas(pg) >= args.confidence).reshape(U1.shape)
            A, B = Xg[:, a].reshape(U1.shape), Xg[:, b].reshape(U1.shape)
            c = ax.contourf(A, B, mu, levels=30, cmap="viridis")
            plt.colorbar(c, ax=ax, label=goal)
            if cons:
                ax.contourf(A, B, (~ok).astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
            ax.plot(M.X[:, a], M.X[:, b], "w.", mec="k", ms=6)
            ax.plot(xb[a], xb[b], "*", ms=16, color=COLORS[3], mec="w")
            ax.set_xlabel(M.inputs[a]); ax.set_ylabel(M.inputs[b])
            ax.set_title("hatched: constraints not met" if cons else goal)
            if log[a]:
                ax.set_xscale("log")
            if log[b]:
                ax.set_yscale("log")
        rep.figure(fig, "optimize")
    rep.close()
