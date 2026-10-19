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
    m = np.ones(len(t), bool)
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


def _case_snapshots(args):
    case = Path(args.case)
    if not foam.is_case(case):
        raise CfdmlError(f"{case} does not look like an OpenFOAM case (no system/ folder)")
    ts = foam.times(case, [args.field])
    ts = [(t, d) for t, d in ts if (args.after is None or t >= args.after) and (args.before is None or t <= args.before)]
    if args.every > 1:
        ts = ts[::args.every]
    if len(ts) < 3:
        raise CfdmlError(f"found {len(ts)} time folders with '{args.field}' in the time window; need at least 3",
                         hint="fields found: " + ", ".join(sorted(foam.field_names(case))))
    nc = foam.n_cells(case)
    print(f"  reading {args.field} from {len(ts)} time folders ({nc} cells) ...")
    snaps = [foam.read_field(d / args.field, nc) for _, d in ts]
    tt = np.array([t for t, _ in ts])
    dts = np.diff(tt)
    if np.ptp(dts) > 0.01 * np.median(dts):
        print("  note: the time folders are not equally spaced; DMD frequencies assume they are")
    return case, tt, snaps, nc


def _centres_2d(case):
    try:
        C = foam.cell_centres(case)
    except CfdmlError:
        return None, None
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
    case, tt, snaps, nc = _case_snapshots(args)
    X, keep = snapshot_matrix(snaps)
    V = foam.cell_volumes(case)
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
    C, axes = _centres_2d(case)
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
            full = np.zeros((nc, snaps[0].shape[1])) if snaps[0].ndim == 2 else None
            if full is not None:
                full[:, keep] = M
            foam.write_field(rep.out / "foam_modes" / f"{i + 1}" / args.field, args.field,
                             full if full is not None else M[:, 0], hdr, note=f"POD mode {i + 1}")
        rep.say(f"modes written as OpenFOAM fields in `foam_modes/<n>/{args.field}`: copy a folder into the case as a time "
                "folder to view it in ParaView")
    rep.close()


def cmd_dmd(args):
    case, tt, snaps, nc = _case_snapshots(args)
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
    C, axes = _centres_2d(case)
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
    if not foam.is_case(case):
        raise CfdmlError(f"{case} does not look like an OpenFOAM case (no system/ folder)")
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
                                                   if graph is not None else "; no ASCII polyMesh, so no blob count)"))
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
    rep.say("Input importance comes from how quickly each output changes with each input in the fitted model.")
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
        rep.say(f"next suggested run: " + ", ".join(f"{k}={fmt(v)}" for k, v in zip(inputs, sug[0][0])), summary=True)
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
