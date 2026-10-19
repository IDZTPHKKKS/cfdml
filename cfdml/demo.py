from pathlib import Path

import numpy as np

from . import foam

HDR = {"U": dict(cls="volVectorField", dims="[0 1 -1 0 0 0 0]"), "p": dict(cls="volScalarField", dims="[0 2 -2 0 0 0 0]"),
       "C": dict(cls="volVectorField", dims="[0 1 0 0 0 0 0]")}
BOUNDARY = ("boundaryField\n{\n    inlet { type fixedValue; value uniform (1 0 0); }\n    outlet { type zeroGradient; }\n"
            "    walls { type slip; }\n    body { type noSlip; }\n    frontAndBack { type empty; }\n}\n")


def _write(case, t, name, values):
    h = dict(HDR[name], boundary=BOUNDARY if name != "C" else "boundaryField\n{\n}\n")
    foam.write_field(Path(case) / t / name, name, values, h)


def _skeleton(case):
    case = Path(case)
    for d in ("system", "constant"):
        (case / d).mkdir(parents=True, exist_ok=True)
    (case / "system/controlDict").write_text("FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
                                             "application pimpleFoam;\nwriteFormat ascii;\n")


def _grid(nx, ny, L=15.0, H=5.0):
    x = (np.arange(nx) + 0.5) * L / nx
    y = (np.arange(ny) + 0.5) * H / ny - H / 2
    X, Y = np.meshgrid(x, y)
    return X.ravel(), Y.ravel()


def _wake(X, Y, t, f=0.2, width=0.8, amp=0.4, deflect=0.0, ramp=6.0):
    k = 2 * np.pi * f / 0.85
    Yc = Y - deflect * np.clip(X - 2, 0, None)
    env = np.exp(-(Yc / width) ** 2) * np.clip((X - 2.5) / 2, 0, 1) * np.exp(-np.clip(X - 2.5, 0, None) / 20)
    a = amp * (1 - np.exp(-t / ramp))
    ph = k * X - 2 * np.pi * f * t
    u = 1 - 0.5 * np.exp(-(Yc / (1.5 * width)) ** 2) * np.exp(-np.clip(X - 2, 0, None) / 8) \
        + a * env * (2 * Yc / width) * np.cos(ph) + 0.1 * a * env * np.cos(2 * ph)
    v = a * env * np.sin(ph)
    inside = (X - 2) ** 2 + Y ** 2 < 0.25
    u[inside] = v[inside] = 0
    p = -0.5 * (u ** 2 + v ** 2 - 1)
    return np.c_[u, v, np.zeros_like(u)], p


def make_wake_case(case, nx=60, ny=20, t_end=40.0, write_every=0.5, f=0.2, seed=0):
    rng = np.random.default_rng(seed)
    _skeleton(case)
    X, Y = _grid(nx, ny)
    _write(case, "0", "C", np.c_[X, Y, np.zeros_like(X)])
    for t in np.arange(write_every, t_end + 1e-9, write_every):
        U, p = _wake(X, Y, t, f=f)
        U[:, :2] += 0.005 * rng.standard_normal((len(X), 2))
        _write(case, f"{t:g}", "U", U)
        _write(case, f"{t:g}", "p", p)
    ts = np.arange(0, 3 * t_end + 1e-9, 0.05)
    probes = [(5.0, 0.5), (8.0, 0.0), (11.0, -0.5)]
    d = Path(case) / "postProcessing/probes/0"
    d.mkdir(parents=True, exist_ok=True)
    lines = [f"# Probe {i} ({x} {y} 0)" for i, (x, y) in enumerate(probes)]
    lines.append("#" + "Probe".rjust(16) + "".join(f"{i:>32}" for i in range(len(probes))))
    lines.append("#" + "Time".rjust(16))
    for t in ts:
        U, _ = _wake(np.array([x for x, _ in probes]), np.array([y for _, y in probes]), t, f=f)
        lines.append(f"{t:>17.6g}" + "".join(f"    ({u:.6g} {v:.6g} 0)" for u, v, _ in U))
    (d / "U").write_text("\n".join(lines) + "\n")
    _force_file(Path(case), ts, f, cd=1.33, cl_amp=0.6, rng=rng)


def _force_file(case, ts, f, cd, cl_amp, rng, ramp=6.0):
    d = case / "postProcessing/forceCoeffs/0"
    d.mkdir(parents=True, exist_ok=True)
    a = 1 - np.exp(-ts / ramp)
    Cl = cl_amp * a * np.sin(2 * np.pi * f * ts) + 0.003 * rng.standard_normal(len(ts))
    Cd = cd + 0.04 * a * np.sin(4 * np.pi * f * ts) + 0.002 * rng.standard_normal(len(ts))
    rows = ["# Force coefficients", "# CofR      : (0 0 0)", "#", "# Time\tCd\tCd(f)\tCd(r)\tCl\tCl(f)\tCl(r)\tCm"]
    rows += [f"{t:g}\t{d_:.6g}\t{d_ / 2:.6g}\t{d_ / 2:.6g}\t{l:.6g}\t{l / 2:.6g}\t{l / 2:.6g}\t{0.1 * l:.6g}"
             for t, d_, l in zip(ts, Cd, Cl)]
    (d / "coefficient.dat").write_text("\n".join(rows) + "\n")


def truth(Re, angle):
    St = 0.212 * (1 - 21.2 / Re) + 0.002 * angle
    Cd = 1.1 + 6.0 / np.sqrt(Re) + 0.004 * angle ** 2
    Cl_amp = 0.2 + 0.25 * np.log(Re / 100) + 0.03 * angle
    return St, Cd, Cl_amp


def make_sweep(root, n=16, seed=1):
    rng = np.random.default_rng(seed)
    root = Path(root)
    u = (np.arange(n) + rng.random(n)) / n
    Re = np.round(100 * 10 ** u)
    angle = np.round(rng.permutation(np.linspace(0, 10, n)), 1)
    runs = []
    X, Y = _grid(60, 20)
    for i, (r, a) in enumerate(zip(Re, angle)):
        St, Cd, Cla = truth(r, a)
        case = root / "runs" / f"Re{int(r)}_a{a:g}"
        _skeleton(case)
        ts = np.arange(0, 40 + 1e-9, 0.05)
        _force_file(case, ts, St, cd=Cd, cl_amp=Cla, rng=rng)
        _write(case, "0", "C", np.c_[X, Y, np.zeros_like(X)])
        U, _ = _wake(X, Y, 1e9, f=St, width=0.6 + 0.4 * np.log10(r / 100), amp=0.0, deflect=-0.02 * a)
        _write(case, "40", "U", U)
        runs.append((case.relative_to(root), r, a))
    cfg = [
           "inputs: [Re, angle]", "log_inputs: [Re]", "", "outputs:",
           "  Cd:     {file: postProcessing/forceCoeffs/0/coefficient.dat, column: Cd, stat: mean, after: 20}",
           "  St:     {file: postProcessing/forceCoeffs/0/coefficient.dat, column: Cl, stat: frequency, after: 20}",
           "  Cl_amp: {file: postProcessing/forceCoeffs/0/coefficient.dat, column: Cl, stat: amplitude, after: 20}",
           "", "fields:", "  U: {time: latest}", "", "runs:"]
    cfg += [f"  - {{case: {c}, Re: {int(r)}, angle: {a:g}}}" for c, r, a in runs]
    (root / "sweep.yaml").write_text("\n".join(cfg) + "\n")


def make_demo(root):
    root = Path(root)
    print(f"  writing example data to {root}/ ...")
    make_wake_case(root / "wake")
    make_drops_case(root / "drops")
    make_sweep(root)
    print(f"""  done. Try:
    cfdml info {root}/wake
    cfdml spectrum {root}/wake/postProcessing/forceCoeffs/0/coefficient.dat -c Cl --after 20 -L 1 -U 1
    cfdml pod {root}/wake -f U --after 20 -o {root}/pod
    cfdml dmd {root}/wake -f U --after 20 -o {root}/dmd
    cfdml fieldstats {root}/drops -f alpha.water
    cfdml forecast {root}/wake/postProcessing/probes/0/U -c probe1_y --after 20 --horizon 10 -o {root}/forecast
    cfdml sweep {root}/sweep.yaml
    cfdml predict {root}/cfdml_sweep/model.pkl --at Re=350,angle=4
  (the wake sheds at frequency 0.2; the sweep outputs follow known formulas, so you can check the answers)""")


def _foam_list(path, cls, obj, items, note=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class " + cls + ";\n"
            + (f'    note "{note}";\n' if note else "") + "    object " + obj + ";\n}\n\n")
    path.write_text(head + f"{len(items)}\n(\n" + "\n".join(items) + "\n)\n")


def write_mesh_2d(case, nx, ny, Lx, Ly):
    mesh = Path(case) / "constant/polyMesh"
    P = lambda i, j, k: k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
    c = lambda i, j: j * nx + i
    pts = [f"({i * Lx / nx:g} {j * Ly / ny:g} {k * 0.1:g})" for k in (0, 1) for j in range(ny + 1) for i in range(nx + 1)]
    internal = []
    for j in range(ny):
        for i in range(nx):
            if i < nx - 1:
                internal.append((c(i, j), c(i + 1, j), (P(i + 1, j, 0), P(i + 1, j + 1, 0), P(i + 1, j + 1, 1), P(i + 1, j, 1))))
            if j < ny - 1:
                internal.append((c(i, j), c(i, j + 1), (P(i, j + 1, 0), P(i, j + 1, 1), P(i + 1, j + 1, 1), P(i + 1, j + 1, 0))))
    internal.sort(key=lambda f: (f[0], f[1]))
    patches = {"left": [(c(0, j), (P(0, j, 0), P(0, j, 1), P(0, j + 1, 1), P(0, j + 1, 0))) for j in range(ny)],
               "right": [(c(nx - 1, j), (P(nx, j, 0), P(nx, j + 1, 0), P(nx, j + 1, 1), P(nx, j, 1))) for j in range(ny)],
               "bottom": [(c(i, 0), (P(i, 0, 0), P(i + 1, 0, 0), P(i + 1, 0, 1), P(i, 0, 1))) for i in range(nx)],
               "top": [(c(i, ny - 1), (P(i, ny, 0), P(i, ny, 1), P(i + 1, ny, 1), P(i + 1, ny, 0))) for i in range(nx)],
               "frontAndBack": [(c(i, j), q) for j in range(ny) for i in range(nx)
                                for q in ((P(i, j, 0), P(i, j + 1, 0), P(i + 1, j + 1, 0), P(i + 1, j, 0)),
                                          (P(i, j, 1), P(i + 1, j, 1), P(i + 1, j + 1, 1), P(i, j + 1, 1)))]}
    faces = [f for _, _, f in internal] + [f for v in patches.values() for _, f in v]
    owner = [o for o, _, _ in internal] + [o for v in patches.values() for o, _ in v]
    note = f"nPoints:{len(pts)}  nCells:{nx * ny}  nFaces:{len(faces)}  nInternalFaces:{len(internal)}"
    _foam_list(mesh / "points", "vectorField", "points", pts)
    _foam_list(mesh / "faces", "faceList", "faces", [f"4({' '.join(map(str, f))})" for f in faces])
    _foam_list(mesh / "owner", "labelList", "owner", [str(o) for o in owner], note)
    _foam_list(mesh / "neighbour", "labelList", "neighbour", [str(n) for _, n, _ in internal], note)
    start, items = len(internal), []
    for name, v in patches.items():
        kind = "empty" if name == "frontAndBack" else "wall"
        items.append(f"{name}\n{{\n    type {kind};\n    nFaces {len(v)};\n    startFace {start};\n}}")
        start += len(v)
    _foam_list(mesh / "boundary", "polyBoundaryMesh", "boundary", items)


def make_drops_case(case, nx=80, ny=60, t_end=10.0, write_every=0.25):
    _skeleton(case)
    write_mesh_2d(case, nx, ny, 4.0, 3.0)
    x = (np.arange(nx) + 0.5) * 4.0 / nx
    y = (np.arange(ny) + 0.5) * 3.0 / ny
    X, Y = [a.ravel() for a in np.meshgrid(x, y)]
    h = dict(cls="volScalarField", dims="[0 0 0 0 0 0 0]",
             boundary="boundaryField\n{\n    \".*\" { type zeroGradient; }\n    frontAndBack { type empty; }\n}\n")
    for t in np.arange(0, t_end + 1e-9, write_every):
        gap = max(0.0, 0.8 - 0.16 * t)
        yc = 2.3 - 0.12 * t
        alpha = np.zeros_like(X)
        for xc in (2 - gap / 2 - 0.25, 2 + gap / 2 + 0.25):
            r = np.hypot(X - xc, Y - yc)
            alpha = np.maximum(alpha, np.clip((0.3 - r) / 0.05 + 0.5, 0, 1))
        foam.write_field(Path(case) / f"{t:g}" / "alpha.water", "alpha.water", alpha, h)
