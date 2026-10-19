import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cfdml import foam
from cfdml.demo import write_mesh_2d
from cfdml.report import plt

IMG = ROOT / "docs/images"
HDR = dict(cls="volVectorField", dims="[0 1 -1 0 0 0 0]",
           boundary="boundaryField\n{\n    \".*\" { type calculated; value uniform (0 0 0); }\n"
                    "    frontAndBack { type empty; }\n}\n")
GALLERY = [("lab/sq_re-136_angle-30_rD-0p1", "square, rounded corners, 30°, Re 136"),
           ("sim/ellipse_re100_a30", "ellipse, 30°, Re 100"),
           ("sim/triangle_re100_a30", "triangle, 30°, Re 100"),
           ("sim/hexagon_re100_a15", "hexagon, 15°, Re 100"),
           ("lab/osc_re-103_ad-1p0", "oscillating cylinder, Re 103"),
           ("sim/square_re100_a45", "square, 45°, Re 100")]


def run(*args):
    subprocess.run([sys.executable, "-m", "cfdml", *map(str, args)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


def load(path):
    z = np.load(Path(path) / "fields.npz")
    F = z["fields"].astype(float)
    u, v = F[:, 1], F[:, 2]
    body = np.isnan(u[0]) | (np.hypot(np.nan_to_num(u[0]), np.nan_to_num(v[0])) < 1e-3)
    return z["x"], z["y"], z["times"], np.nan_to_num(u), np.nan_to_num(v), body


def write_case(path, case):
    x, y, t, u, v, body = load(path)
    (Path(case) / "system").mkdir(parents=True, exist_ok=True)
    (Path(case) / "system/controlDict").write_text("FoamFile { format ascii; }\nwriteFormat ascii;\n")
    write_mesh_2d(case, len(x), len(y), np.ptp(x) + x[1] - x[0], np.ptp(y) + y[1] - y[0])
    for k in range(len(t)):
        U = np.c_[u[k].ravel(), v[k].ravel(), np.zeros(u[k].size)]
        foam.write_field(Path(case) / f"{t[k]:g}" / "U", "U", U, HDR)


def vorticity(x, y, u, v, body):
    w = np.gradient(v, x, axis=1) - np.gradient(u, y, axis=0)
    return np.ma.masked_where(body, w)


def banner(fields):
    fig, ax = plt.subplots(2, 3, figsize=(12, 3.9))
    for a, (rel, title) in zip(ax.ravel(), GALLERY):
        x, y, t, u, v, body = load(fields / rel)
        a.set_facecolor("#3b3b3b")
        a.imshow(vorticity(x, y, u[-1], v[-1], body), origin="lower", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                 extent=[x[0], x[-1], y[0], y[-1]], interpolation="bicubic")
        a.set_xlim(-1.5, 13.5); a.set_ylim(-3.2, 3.2)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.set_title(title, fontsize=8.5)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    return fig


def read_msh(path):
    L = Path(path).read_text().split("\n")
    j = L.index("$Nodes")
    nodes = {}
    for line in L[j + 2:j + 2 + int(L[j + 1])]:
        k, x, y, zz = line.split()
        nodes[int(k)] = (float(x), float(y), float(zz))
    i = L.index("$Elements")
    polys = []
    for line in L[i + 2:i + 2 + int(L[i + 1])]:
        e = line.split()
        if e[1] in ("2", "3"):
            ids = [int(v) for v in e[3 + int(e[2]):]]
            pts = [nodes[v] for v in ids]
            if all(abs(p[2]) < 1e-9 for p in pts):
                polys.append(np.array([p[:2] for p in pts]))
    return polys


def overview(d, fields, forces, mesh):
    from matplotlib.collections import PolyCollection
    from cfdml.analysis import spectrum
    from cfdml.sweep import load_model
    from cfdml.table import read_table
    lab = dict(fontsize=8.5, loc="left", fontweight="bold")
    cmd = dict(fontsize=8.5, loc="left", fontweight="bold", family="monospace")
    fig, ax = plt.subplots(2, 4, figsize=(13, 5.2))
    a = ax[0, 0]
    polys = [p for p in read_msh(mesh) if np.all(np.abs(p[:, 0] - 0.9) < 2.4) and np.all(np.abs(p[:, 1]) < 1.6)]
    a.add_collection(PolyCollection(polys, facecolors="#eef2f7", edgecolors="#4c72b0", linewidths=0.25))
    a.add_patch(plt.Circle((0, 0), 0.5, color="#3b3b3b", zorder=3))
    a.set_xlim(-1.4, 3.2); a.set_ylim(-1.5, 1.5); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.grid(False)
    a.set_title("mesh (gmsh → OpenFOAM)", **lab)
    a = ax[0, 1]
    x, y, t, u, v, body = load(fields / GALLERY[0][0])
    a.set_facecolor("#3b3b3b")
    a.imshow(vorticity(x, y, u[-1], v[-1], body), origin="lower", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
             extent=[x[0], x[-1], y[0], y[-1]], interpolation="bicubic")
    a.set_xlim(-1.5, 11); a.set_ylim(-3, 3); a.set_xticks([]); a.set_yticks([]); a.grid(False)
    a.set_title("wake vorticity", **lab)
    a = ax[0, 2]
    w = np.load(fields / "sim/kolmo_n4_re40/fields.npz")["w"][-1]
    a.imshow(w[:, :], origin="lower", cmap="RdBu_r", vmin=-np.percentile(np.abs(w), 99), vmax=np.percentile(np.abs(w), 99),
             interpolation="bicubic", extent=[0, 2 * np.pi, 0, 2 * np.pi], aspect="auto")
    a.set_xticks([]); a.set_yticks([]); a.grid(False); a.set_title("2D turbulence", **lab)
    a = ax[0, 3]
    P = np.load(d / "pod/pod.npz")
    nx, ny = len(x), len(y)
    m = P["modes"][:, 0].reshape(nx * ny, -1)[:, 1].reshape(ny, nx)
    a.imshow(np.ma.masked_where(body, m), origin="lower", cmap="RdBu_r", extent=[x[0], x[-1], y[0], y[-1]],
             interpolation="bicubic")
    a.set_xlim(-1.5, 11); a.set_ylim(-3, 3); a.set_facecolor("#3b3b3b")
    a.set_xticks([]); a.set_yticks([]); a.grid(False); a.set_title("cfdml pod", **cmd)
    tb = read_table(forces / "Re-105/postProcessing/forceCoeffs/0/coefficient.dat")
    k = tb.time >= 120
    sp = spectrum(tb["Cl"][k], 0.01)
    a = ax[1, 0]
    kk = (sp["f_fine"] > 0) & (sp["f_fine"] <= 1.0)
    a.semilogy(sp["f_fine"][kk], sp["P_fine"][kk] + 1e-12, color="#4c72b0", lw=1)
    a.axvline(sp["f0"], color="#c44e52", ls="--", lw=0.8); a.set_xlim(0, 1); a.set_ylim(1e-7, 3)
    a.text(sp["f0"] + 0.03, 0.2, f"St = {sp['f0']:.3f}", fontsize=8, color="#c44e52")
    a.set_xlabel("frequency", fontsize=7); a.set_title("cfdml spectrum", **cmd)
    a = ax[1, 1]
    M = load_model(d / "sweep/model.pkl")
    g = np.geomspace(M.lo[0], M.hi[0], 200)
    mu, sd = M.predict(g[:, None])["St"]
    a.fill_between(g, mu - 2 * sd, mu + 2 * sd, color="#4c72b0", alpha=0.25, lw=0)
    a.plot(g, mu, color="#4c72b0"); a.plot(M.X[:, 0], M.scalar["St"]["y"], "o", ms=3, color="k")
    a.set_xscale("log"); a.set_xlabel("Re", fontsize=7); a.set_ylabel("St", fontsize=7)
    from matplotlib.ticker import FixedLocator, NullLocator
    a.xaxis.set_major_locator(FixedLocator([60, 80, 100, 150])); a.xaxis.set_minor_locator(NullLocator())
    a.set_xticklabels(["60", "80", "100", "150"])
    a.set_title("cfdml sweep", **cmd)
    a = ax[1, 2]
    D = np.load(d / "dmd/dmd.npz")
    th = np.linspace(0, 2 * np.pi, 200)
    a.plot(np.cos(th), np.sin(th), color="#8c8c8c", lw=0.7)
    a.scatter(D["lam"].real, D["lam"].imag, s=14, c=np.log10(D["amp"] / D["amp"].max() + 1e-9), cmap="viridis")
    a.set_aspect("equal"); a.set_xlim(-1.25, 1.25); a.set_ylim(-1.25, 1.25)
    a.set_title("cfdml dmd", **cmd)
    a = ax[1, 3]
    fc = read_table(d / "fc/forecast.csv")
    k = (tb.time > 175) & (tb.time <= 190)
    a.plot(tb.time[k], tb["Cl"][k], color="k", lw=1)
    a.plot(fc.time, fc.data[:, 1], color="#c44e52", lw=1.1, ls="--"); a.axvline(190, color="#8c8c8c", lw=0.7, ls=":")
    a.set_xlabel("time", fontsize=7); a.set_title("cfdml forecast", **cmd)
    for a in ax.ravel():
        a.tick_params(labelsize=6.5)
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    return fig


def _t_end(case):
    with open(case / "postProcessing/forceCoeffs/0/coefficient.dat", "rb") as fh:
        fh.seek(-200, 2)
        return float(fh.read().split(b"\n")[-2].split()[0])


def main():
    forces, fields, mesh = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
    IMG.mkdir(parents=True, exist_ok=True)
    runs = sorted((int(p.name.split("-")[1]), p) for p in forces.glob("Re-*")
                  if (p / "postProcessing/forceCoeffs/0/coefficient.dat").exists() and _t_end(p) > 150)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        cf = forces / "Re-105/postProcessing/forceCoeffs/0/coefficient.dat"
        run("spectrum", cf, "-c", "Cl", "--after", 120, "-L", 1, "-U", 1, "-o", d / "spec")
        run("forecast", cf, "-c", "Cl", "--after", 150, "--before", 190, "--train", 0.6, "--horizon", 20, "-o", d / "fc")
        case = d / "square_case"
        write_case(fields / GALLERY[0][0], case)
        run("pod", case, "-f", "U", "--modes", 2, "-o", d / "pod")
        run("dmd", case, "-f", "U", "-o", d / "dmd")
        lines = ["inputs: [Re]", "log_inputs: [Re]", "outputs:",
                 "  St: {file: postProcessing/forceCoeffs/0/coefficient.dat, column: Cl, stat: frequency, after: 120}",
                 "  Cd: {file: postProcessing/forceCoeffs/0/coefficient.dat, column: Cd, stat: mean, after: 120}",
                 "runs:"] + [f"  - {{case: {p}, Re: {re}}}" for re, p in runs if re >= 50]
        (d / "sweep.yaml").write_text("\n".join(lines) + "\n")
        run("sweep", d / "sweep.yaml", "-o", d / "sweep")
        for src, dst in ((d / "spec/spectrum.png", "spectrum.png"), (d / "fc/forecast.png", "forecast.png"),
                         (d / "pod/pod_modes.png", "pod_modes.png"), (d / "dmd/dmd_mode.png", "dmd_mode.png"),
                         (d / "sweep/sweep_St.png", "sweep_St.png"), (d / "sweep/sweep_Cd.png", "sweep_Cd.png")):
            shutil.copy(src, IMG / dst)
        fig = overview(d, fields, forces, mesh)
        fig.savefig(IMG / "overview.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
    fig = banner(fields)
    fig.savefig(IMG / "wakes.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    for old in ("banner.png", "fieldstats.png", "sweep_map.png", "predicted_U.png", "sweep_Cd.png"):
        (IMG / old).unlink(missing_ok=True)
    print("images written to", IMG)


if __name__ == "__main__":
    main()
