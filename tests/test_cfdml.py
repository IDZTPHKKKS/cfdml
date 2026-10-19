import gzip

import numpy as np
import pytest

from cfdml import foam
from cfdml.analysis import dmd, pod, snapshot_matrix, spectrum
from cfdml.cli import main
from cfdml.demo import make_demo, make_drops_case, truth, write_mesh_2d
from cfdml.errors import CfdmlError
from cfdml.forecast import evaluate
from cfdml.sweep import SweepModel, collect, load_config, parse_point
from cfdml.table import read_table

HDR = dict(cls="volVectorField", dims="[0 1 -1 0 0 0 0]", boundary="boundaryField\n{\n}\n")


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    make_demo(root)
    return root


def test_field_roundtrip(tmp_path):
    v = np.random.default_rng(0).standard_normal((50, 3))
    foam.write_field(tmp_path / "U", "U", v, HDR)
    assert np.allclose(foam.read_field(tmp_path / "U"), v, rtol=1e-7)
    s = np.arange(7.0)
    foam.write_field(tmp_path / "p", "p", s, dict(HDR, cls="volScalarField"))
    assert np.allclose(foam.read_field(tmp_path / "p"), s)
    (tmp_path / "q.gz").write_bytes(gzip.compress((tmp_path / "p").read_bytes()))
    assert np.allclose(foam.read_field(tmp_path / "q"), s)


def test_uniform_compact_and_binary(tmp_path):
    (tmp_path / "a").write_text("FoamFile { format ascii; class volVectorField; }\ninternalField uniform (1 2 3);\n")
    assert foam.read_field(tmp_path / "a", 4).shape == (4, 3)
    (tmp_path / "b").write_text("FoamFile { format ascii; }\ninternalField nonuniform List<scalar> 3(1 2 3);\n")
    assert list(foam.read_field(tmp_path / "b")) == [1, 2, 3]
    (tmp_path / "c").write_text("FoamFile { format binary; }\ninternalField nonuniform List<scalar> 3(...);\n")
    with pytest.raises(CfdmlError, match="binary"):
        foam.read_field(tmp_path / "c")


def test_polymesh(tmp_path):
    (tmp_path / "system").mkdir()
    write_mesh_2d(tmp_path, 4, 3, 2.0, 1.5)
    assert foam.n_cells(tmp_path) == 12
    C = foam.cell_centres(tmp_path)
    assert np.allclose(C[0], [0.25, 0.25, 0.05]) and np.allclose(C[-1], [1.75, 1.25, 0.05])
    o, n = foam.internal_faces(tmp_path)
    assert len(o) == 3 * 3 + 4 * 2


def test_tables(demo, tmp_path):
    tb = read_table(demo / "wake/postProcessing/probes/0/U")
    assert tb.names[:3] == ["Time", "probe0_x", "probe0_y"] and len(tb.names) == 10
    fc = read_table(demo / "wake/postProcessing/forceCoeffs/0/coefficient.dat")
    assert "Cl" in fc and "Cd" in fc
    (tmp_path / "r.csv").write_text("time, Cl\n0,1\n1,2\n2,3\n1,9\n2,8\n3,7\n")
    tb = read_table(tmp_path / "r.csv")
    assert list(tb.time) == [0, 1, 2, 3] and list(tb["Cl"]) == [1, 9, 8, 7]


def test_spectrum_pod_dmd():
    dt = 0.05
    t = np.arange(0, 60, dt)
    assert abs(spectrum(np.sin(2 * np.pi * 0.37 * t), dt)["f0"] - 0.37) < 2e-3
    x = np.linspace(0, 10, 200)
    X = np.stack([np.sin(2 * x - 2 * np.pi * 0.2 * tt) for tt in np.arange(0, 30, 0.25)], 1)
    P = pod(X)
    assert P["energy"][:2].sum() > 0.999
    D = dmd(X, 0.25, rank=2)
    assert abs(abs(D["freq"][0]) - 0.2) < 1e-3 and abs(D["growth"][0]) < 1e-3
    M, keep = snapshot_matrix([np.c_[np.ones(5) * k, np.zeros(5)] for k in range(4)])
    assert keep == [0] and M.shape == (5, 4)


def test_forecast_beats_baseline():
    t = np.arange(0, 80, 0.1)
    y = np.sin(2 * np.pi * 0.21 * t) + 0.3 * np.sin(2 * np.pi * 0.42 * t + 1)
    R = evaluate(t, y, train=0.7)
    assert R["results"][R["best"]]["nrmse"] < 0.05


def test_sweep_matches_truth(demo):
    inputs, rows = collect(load_config(demo / "sweep.yaml"))
    M = SweepModel(inputs, ["Re"], rows)
    for o in ("St", "Cd", "Cl_amp"):
        M.fit_scalar(o)
        assert M.scalar[o]["r2"] > 0.95
    p = M.predict(parse_point("Re=350,angle=4", inputs))
    St, Cd, _ = truth(350, 4)
    assert abs(p["St"][0][0] - St) < 0.003 and abs(p["Cd"][0][0] - Cd) < 0.02
    with pytest.raises(CfdmlError):
        parse_point("Re=350", inputs)


def test_drop_count(tmp_path):
    make_drops_case(tmp_path / "d", nx=40, ny=30, t_end=8, write_every=1)
    assert main(["fieldstats", str(tmp_path / "d"), "-f", "alpha.water"]) == 0
    tb = read_table(tmp_path / "d/postProcessing/cfdml_fieldstats/alpha.water.csv")
    assert tb["blobs"][0] == 2 and tb["blobs"][-1] == 1


def test_cli(demo, tmp_path, capsys):
    o = str(tmp_path)
    assert main(["info", str(demo / "wake")]) == 0
    assert main(["spectrum", str(demo / "wake/postProcessing/forceCoeffs/0/coefficient.dat"), "-c", "Cl", "--after", "20",
                 "-o", o + "/s"]) == 0
    assert "0.2" in (tmp_path / "s/report.md").read_text()
    assert main(["pod", str(demo / "wake"), "--after", "20", "--write-modes", "-o", o + "/p"]) == 0
    assert (tmp_path / "p/foam_modes/1/U").exists()
    assert main(["dmd", str(demo / "wake"), "--after", "20", "-o", o + "/d"]) == 0
    assert main(["forecast", str(demo / "wake/postProcessing/probes/0/U"), "-c", "probe1_y", "--after", "20",
                 "-o", o + "/f"]) == 0
    assert main(["sweep", str(demo / "sweep.yaml"), "-o", o + "/w"]) == 0
    assert main(["predict", o + "/w", "--at", "Re=300,angle=2", "--grid", "Re=200:800:3", "--grid", "angle=0:10:2",
                 "-o", o + "/q"]) == 0
    assert (tmp_path / "q/Re300_angle2/U").exists()
    assert main(["init", "sweep", "-o", o + "/t.yaml"]) == 0
    assert main(["spectrum", o + "/missing.csv"]) == 1
    assert "not found" in capsys.readouterr().err


def test_gci_celik_example():
    from cfdml.verify import gci
    r = gci(np.array([18000, 8000, 4500.0]) ** -0.5, [6.063, 5.972, 5.863])
    assert abs(r["p"] - 1.53) < 0.01 and abs(r["ext"] - 6.1685) < 1e-3 and abs(r["gci21"] - 0.022) < 1e-3


def test_transient_end():
    from cfdml.verify import transient_end
    t = np.arange(0, 200, 0.05)
    amp = 1 / (1 + np.exp(-(t - 60) / 5))
    te, _ = transient_end(t, amp * np.sin(2 * np.pi * 0.2 * t))
    assert 70 < te < 110
    te, _ = transient_end(t, np.exp(t / 40) * np.sin(2 * np.pi * 0.2 * t))
    assert te is None


def test_compare_and_steady(demo, tmp_path, capsys):
    from cfdml.verify import REFERENCES
    re = np.array([60, 80, 100, 150])
    st = REFERENCES["cylinder-st-williamson"]["f"](re) * 1.02
    (tmp_path / "r.csv").write_text("name,Re,St\n" + "".join(f"r{i},{a},{b}\n" for i, (a, b) in enumerate(zip(re, st))))
    assert main(["compare", str(tmp_path / "r.csv"), "-x", "Re", "-y", "St", "--ref", "cylinder-st-williamson",
                 "-o", str(tmp_path / "c")]) == 0
    assert "+2.00%" in (tmp_path / "c/report.md").read_text()
    t = np.arange(0, 300, 0.05)
    cl = (1 - np.exp(-t / 10)) * np.sin(2 * np.pi * 0.2 * t)
    cd = 1.3 + 0.2 * np.exp(-t / 10) + 0.02 * (1 - np.exp(-t / 10)) * np.sin(4 * np.pi * 0.2 * t)
    (tmp_path / "f.csv").write_text("time,Cd,Cl\n" + "".join(f"{a},{b},{c}\n" for a, b, c in zip(t, cd, cl)))
    capsys.readouterr()
    assert main(["steady", str(tmp_path / "f.csv"), "-o", str(tmp_path / "s")]) == 0
    out = capsys.readouterr().out
    assert out.count("settled") == 2 and "run longer" not in out
    assert main(["steady", str(demo / "wake/postProcessing/forceCoeffs/0/coefficient.dat"), "-o", str(tmp_path / "s2")]) == 0
    assert main(["gci", "--cells", "4500,8000,18000", "--values", "5.863,5.972,6.063", "-o", str(tmp_path / "g")]) == 0


def test_box(demo, tmp_path):
    assert main(["pod", str(demo / "wake"), "--after", "20", "--box", "3,12,-2,2,-1,1", "--write-modes",
                 "-o", str(tmp_path / "p")]) == 0
    F = foam.read_field(tmp_path / "p/foam_modes/1/U")
    C = foam.cell_centres(demo / "wake")
    inside = (C[:, 0] >= 3) & (C[:, 0] <= 12) & (np.abs(C[:, 1]) <= 2)
    assert np.all(F[~inside] == 0) and np.abs(F[inside]).sum() > 0


@pytest.mark.parametrize("cmd", [[], ["info"], ["spectrum"], ["pod"], ["dmd"], ["fieldstats"], ["forecast"], ["steady"],
                                 ["gci"], ["compare"], ["sweep"], ["predict"], ["init"], ["demo"]])
def test_help(cmd, capsys):
    with pytest.raises(SystemExit) as e:
        main(cmd + ["--help"])
    assert e.value.code == 0 and "usage" in capsys.readouterr().out


def test_force_layouts(tmp_path):
    from cfdml.forces import read_forces
    t = np.arange(0, 5, 0.5)
    old = ["# Forces", "# CofR  : (0 0 0)", "# Time  forces(pressure viscous porous) moment(pressure viscous porous)"]
    old += [f"{x}\t(({1 + x} {0.1 * x} 0) (0.2 0.01 0) (0 0 0)) ((0 0 {x}) (0 0 0) (0 0 0))" for x in t]
    (tmp_path / "old.dat").write_text("\n".join(old) + "\n")
    F = read_forces(tmp_path / "old.dat")
    assert np.allclose(F["total_x"], 1.2 + t) and np.allclose(F["total_y"], 0.1 * t + 0.01)
    assert np.allclose(F["moment_pressure_z"], t)
    v8 = ["# Time                  (total_x total_y total_z) (pressure_x pressure_y pressure_z) (viscous_x viscous_y viscous_z)"]
    v8 += [f"{x}  (3 {x} 0) (2.5 {x} 0) (0.5 0 0)" for x in t]
    (tmp_path / "v8.dat").write_text("\n".join(v8) + "\n")
    F = read_forces(tmp_path / "v8.dat")
    assert np.allclose(F["total_x"], 3) and np.allclose(F["viscous_x"], 0.5) and np.allclose(F["total_y"], t)


def test_residual_formats(tmp_path):
    from cfdml.logs import parse_log, parse_solver_info
    si = ["# Solver information", "# Time          \tU_solver\tUx_initial\tUx_final\tUx_iters\tp_solver\tp_initial\t"
          "p_final\tp_iters\tp_converged"]
    si += [f"{k}\tsmoothSolver\t{10.0 ** -k}\t1e-9\t3\tGAMG\t{2 * 10.0 ** -k}\t1e-8\t12\ttrue" for k in range(1, 6)]
    (tmp_path / "solverInfo.dat").write_text("\n".join(si) + "\n")
    R = parse_solver_info(tmp_path / "solverInfo.dat")
    assert list(R["init"]) == ["Ux", "p"] and np.allclose(R["init"]["p"], 2 * 10.0 ** -np.arange(1, 6))
    (tmp_path / "residuals.dat").write_text("# Residuals\n# Time p Ux Uy\n1 0.1 N/A 0.2\n2 0.01 0.02 0.03\n")
    R = parse_solver_info(tmp_path / "residuals.dat")
    assert np.isnan(R["init"]["Ux"][0]) and R["init"]["p"][1] == 0.01
    ico = ["Time = 0.005", "", "Courant Number mean: 0.1 max: 0.7",
           "DILUPBiCG:  Solving for Ux, Initial residual = 1, Final residual = 1e-6, No Iterations 8",
           "DICPCG:  Solving for p, Initial residual = 1, Final residual = 1e-7, No Iterations 35",
           "Time = 0.01", "", "Courant Number mean: 0.1 max: 0.9",
           "DILUPBiCG:  Solving for Ux, Initial residual = nan, Final residual = nan, No Iterations 1000",
           "#0  Foam::error::printStack(Foam::Ostream&)", "Floating point exception (core dumped)"]
    (tmp_path / "log.icoFoam").write_text("\n".join(ico) + "\n")
    R = parse_log(tmp_path / "log.icoFoam")
    assert list(R["co"]) == [0.7, 0.9] and R["fatal"][1] == 0.01 and R["nan"] == ["Ux"] and not R["ended"]


def test_new_commands(demo, tmp_path, capsys):
    o = str(tmp_path)
    assert main(["residuals", str(demo / "wake"), "-o", o + "/r"]) == 0
    assert "finished" in capsys.readouterr().out
    assert main(["forces", str(demo / "wake"), "--rho", "1", "-U", "1", "--area", "0.1", "-o", o + "/f"]) == 0
    out = capsys.readouterr().out
    assert "2401 samples" in out and "pressure 78.9%" in out
    assert main(["average", str(demo / "wake"), "-f", "U", "--after", "20", "-o", o + "/a"]) == 0
    M = foam.read_field(tmp_path / "a/40/UMean")
    ts = [d for t, d in foam.times(demo / "wake", ["U"]) if t >= 20]
    assert np.allclose(M, np.mean([foam.read_field(d / "U") for d in ts], 0), atol=1e-7)


def _decompose(src, dst, nproc):
    import shutil
    shutil.copytree(src / "system", dst / "system")
    (dst / "constant").mkdir()
    nc = foam.n_cells(src)
    edges = np.linspace(0, nc, nproc + 1).astype(int)
    for _, d in foam.times(src):
        for f in d.iterdir():
            v = foam.read_field(f, nc)
            h = foam.field_header(f)
            for p in range(nproc):
                foam.write_field(dst / f"processor{p}" / d.name / f.name, f.name, v[edges[p]:edges[p + 1]], h)
    shutil.copytree(src / "postProcessing", dst / "postProcessing")


def test_decomposed(demo, tmp_path):
    rec, dec = demo / "wake", tmp_path / "dec"
    _decompose(rec, dec, 11)
    assert foam.is_decomposed(dec) and foam.n_cells(dec) == foam.n_cells(rec)
    assert np.allclose(foam.cell_centres(dec), foam.cell_centres(rec))
    assert main(["pod", str(dec), "--after", "20", "--write-modes", "-o", str(tmp_path / "pd")]) == 0
    assert main(["pod", str(rec), "--after", "20", "-o", str(tmp_path / "pr")]) == 0
    assert np.allclose(np.load(tmp_path / "pd/pod.npz")["energy"], np.load(tmp_path / "pr/pod.npz")["energy"])
    assert (tmp_path / "pd/foam_modes/processor10/1/U").exists()
    assert main(["average", str(dec), "-f", "U", "--after", "20", "-o", str(tmp_path / "ad")]) == 0
    assert main(["average", str(rec), "-f", "U", "--after", "20", "-o", str(tmp_path / "ar")]) == 0
    parts = np.concatenate([foam.read_field(tmp_path / f"ad/processor{p}/40/UMean") for p in range(11)])
    assert np.allclose(parts, foam.read_field(tmp_path / "ar/40/UMean"))
    assert main(["average", str(dec), "-f", "U", "--after", "20", "--into-case", "-o", str(tmp_path / "ai")]) == 0
    assert (dec / "processor3/40/UMean").exists()
    assert main(["average", str(dec), "-f", "U", "--after", "20", "--into-case", "-o", str(tmp_path / "ai2")]) == 1


def test_sobol_estimator():
    from cfdml.sweep import sobol

    class Exact:
        inputs = ["a", "b", "c"]
        lo, hi = np.full(3, -np.pi), np.full(3, np.pi)

        class scaler:
            log = np.zeros(3, bool)

        @staticmethod
        def predict(X):
            return {"f": (np.sin(X[:, 0]) + 7 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0]), None)}

    S = sobol(Exact, "f", n=2 ** 15, boot=0)
    assert np.allclose(S["S1"], [0.314, 0.442, 0.0], atol=0.03) and np.allclose(S["ST"], [0.558, 0.442, 0.244], atol=0.03)


def test_optimize_and_sensitivity(demo, tmp_path, capsys):
    from cfdml.demo import truth
    o = str(tmp_path)
    assert main(["sweep", str(demo / "sweep.yaml"), "-o", o + "/w"]) == 0
    capsys.readouterr()
    assert main(["optimize", o + "/w", "--maximize", "Cl_amp", "--subject-to", "Cd<=1.5", "-o", o + "/op"]) == 0
    out = capsys.readouterr().out
    import re
    re_, ang = map(float, re.search(r"Re=([\d.]+), angle=([\d.]+)", out).groups())
    assert abs(truth(re_, ang)[1] - 1.5) < 0.01 and 6.5 < ang < 7.5
    assert main(["optimize", o + "/w", "--minimize", "Cd", "--subject-to", "St>=0.5", "-o", o + "/op2"]) == 1
    assert main(["sensitivity", o + "/w", "-n", "1024", "-o", o + "/s"]) == 0
    assert "Cd" in (tmp_path / "s/report.md").read_text()


def test_truncated_and_quoted_tables(demo, tmp_path, capsys):
    src = (demo / "wake/postProcessing/forceCoeffs/0/coefficient.dat").read_text()
    (tmp_path / "cut.dat").write_text(src + "120.05\t1.3")
    tb = read_table(tmp_path / "cut.dat")
    assert "Cl" in tb and tb.data.shape[1] == 8 and tb.time[-1] == 120
    assert "cut off" in capsys.readouterr().out
    (tmp_path / "q.csv").write_text('time,"Cd, mean",Cl\n0,1.0,0.1\n1,,0.2\n2,1.2,N/A\n')
    tb = read_table(tmp_path / "q.csv")
    assert tb.names == ["time", "Cd, mean", "Cl"] and np.isnan(tb["Cd, mean"][1]) and np.isnan(tb["Cl"][2])


def test_gci_degenerate(tmp_path, capsys):
    from cfdml.verify import gci
    with pytest.raises(CfdmlError):
        gci([1, 2, 4], [1.0, 1.1, 1.2])
    assert main(["gci", "--h", "1,2,4", "--values", "6.063,5.972,5.863", "-o", str(tmp_path / "g")]) == 0
    assert "asymptotic range" not in capsys.readouterr().out


def test_model_pickle_is_restricted(tmp_path):
    import pickle
    from cfdml.sweep import load_model

    class Payload:
        def __reduce__(self):
            return (open, (str(tmp_path / "ran"), "w"))

    (tmp_path / "evil.pkl").write_bytes(pickle.dumps(Payload()))
    with pytest.raises(CfdmlError):
        load_model(tmp_path / "evil.pkl")
    assert not (tmp_path / "ran").exists()


def test_sweep_robustness(demo, tmp_path, capsys):
    from cfdml.verify import time_weights
    assert np.allclose(time_weights([0, 1, 2, 3]), 1) and np.allclose(time_weights([0, 1, 1.5, 2]), np.array([1, 1, 0.5, 0.5]) / 0.75)
    lines = (demo / "sweep.yaml").read_text().split("\n")
    runs = [i for i, ln in enumerate(lines) if ln.strip().startswith("- {case")]
    for i in runs[:2]:
        lines[i] = lines[i].replace("}", ", Cm: 0.1}")
    cfg = "\n".join(lines).replace("outputs:", "outputs:\n  Cm: {}")
    (demo / "s2.yaml").write_text(cfg)
    assert main(["sweep", str(demo / "s2.yaml"), "-o", str(tmp_path / "w")]) == 0
    assert "Cm left out" in capsys.readouterr().out
    (demo / "s3.yaml").write_text(cfg.replace("log_inputs: [Re]", "log_inputs: [re]"))
    assert main(["sweep", str(demo / "s3.yaml"), "-o", str(tmp_path / "w3")]) == 1
    assert main(["predict", str(tmp_path / "w"), "--grid", "Re=100:1000:3", "--grid", "angle=0:0:1",
                 "-o", str(tmp_path / "p")]) == 0
    assert "316.2" in (tmp_path / "p/predictions.csv").read_text()


def test_cli_fixes(demo, tmp_path, capsys):
    o = tmp_path
    (o / "r.csv").write_text("Re,St\n60,0\n100,0.1643\n150,0.1849\n")
    from cfdml.verify import REFERENCES
    REFERENCES["zero-test"] = dict(x="Re", y="St", lo=0, hi=200, f=lambda r: np.where(r < 80, 0.0, 0.1643 + 0 * r),
                                   cite="test")
    assert main(["compare", str(o / "r.csv"), "-x", "Re", "-y", "St", "--ref", "zero-test", "-o", str(o / "c")]) == 0
    assert "nan%" not in capsys.readouterr().out
    del REFERENCES["zero-test"]
    (o / "e.csv").write_text("Re,St\n1,\n")
    assert main(["compare", str(o / "e.csv"), "-x", "Re", "-y", "St", "--ref", "cylinder-st-fey"]) == 1
    assert main(["info", str(demo / "wake"), "--debug"]) == 0
    t = np.arange(0, 200, 0.05)
    (o / "g.csv").write_text("time,Cl\n" + "".join(f"{a},{np.exp(a / 40) * np.sin(a)}\n" for a in t))
    assert main(["steady", str(o / "g.csv"), "-o", str(o / "s")]) == 3
    assert (o / "s/report.md").exists()
    assert main(["spectrum", str(o / "g.csv"), "-o", str(o / "s")]) == 1
    assert main(["spectrum", str(o / "g.csv"), "-o", str(o / "s"), "--overwrite"]) == 0
    assert main(["fieldstats", str(demo / "wake"), "-f", "U", "--phase", "-o", str(o / "fs")]) == 1


def test_foam_edge_cases(tmp_path):
    foam.write_field(tmp_path / "U2", "U", np.ones((4, 2)), HDR)
    assert foam.read_field(tmp_path / "U2").shape == (4, 3)
    (tmp_path / "c/system").mkdir(parents=True)
    (tmp_path / "c/constant/polyMesh").mkdir(parents=True)
    (tmp_path / "c/constant/polyMesh/owner").write_text("FoamFile { format ascii; }\n0\n(\n)\n")
    with pytest.raises(CfdmlError):
        foam.n_cells(tmp_path / "c")


def test_export_and_evaluate(demo, tmp_path, capsys):
    import json
    from cfdml.mldata import Traj, load, save
    from cfdml.mleval import divergence
    o = tmp_path
    assert main(["export", "--sweep", str(demo / "sweep.yaml"), "-f", "U", "--grid", "64", "-o", str(o / "s")]) == 0
    split = json.loads((o / "s/split.json").read_text())
    assert len(split["train"]) + len(split["val"]) + len(split["test"]) == 16 and not set(split["train"]) & set(split["test"])
    assert all(load(f).names == ["Ux", "Uy"] for f in (o / "s/cases").glob("*.npz"))
    assert main(["export", str(demo / "wake"), "-f", "U", "--after", "10", "-o", str(o / "w")]) == 0
    w = load(o / "w/cases/wake.npz")
    stats = json.loads((o / "w/stats.json").read_text())
    a, b = json.loads((o / "w/split.json").read_text())["train"]
    assert abs(stats["Ux"]["mean"] - w.flat()[a:b, :, 0].mean()) < 1e-6
    assert main(["evaluate", str(o / "w/cases/wake.npz"), str(o / "w/cases/wake.npz"), "-o", str(o / "e0")]) == 0
    assert "first step 0.000, mean 0.000" in capsys.readouterr().out
    rng = np.random.default_rng(0)
    noisy = w.fields + 0.01 * (w.fields - w.fields.mean(0)).std(axis=(0, 2, 3), keepdims=True) * rng.standard_normal(w.fields.shape) * w.mask
    save(Traj(noisy, w.times, w.names, x=w.x, y=w.y, mask=w.mask), o / "n.npz")
    assert main(["evaluate", str(o / "n.npz"), str(o / "w/cases/wake.npz"), "-o", str(o / "e1")]) == 0
    assert "mean 0.010" in capsys.readouterr().out
    x = np.linspace(0, 2 * np.pi, 128)
    X, Y = np.meshgrid(x, x)
    u, v = np.sin(X) * np.cos(Y), -np.cos(X) * np.sin(Y)
    assert divergence(u, v, x[1] - x[0], x[1] - x[0]) < 1e-3
    assert divergence(u, v + 0.5 * np.sin(Y), x[1] - x[0], x[1] - x[0]) > 0.1


def test_rollout_and_dataset(demo, tmp_path, capsys):
    import json
    from cfdml.mldata import Traj, save
    o = tmp_path
    t = np.arange(0, 200, 0.25)
    xg = np.linspace(0, 10, 40)
    X, Y = np.meshgrid(xg, xg[:20])

    def traj(f_of_t, amp_of_t, n):
        ph = np.cumsum(2 * np.pi * f_of_t(t[:n]) * 0.25)
        F = np.stack([np.stack([a * np.sin(X - p), a * np.cos(X - p) * np.exp(-Y / 5)]) for p, a in zip(ph, amp_of_t(t[:n]))])
        return Traj(F, t[:n], ["u", "v"], x=xg, y=xg[:20], mask=np.ones(X.shape, bool), params={})

    save(traj(lambda s: 0.2 + 0 * s, lambda s: 1 + 0 * s, 400), o / "truth.npz")
    save(traj(lambda s: 0.2 - 0.04 * s / 200, lambda s: np.exp(-s / 150), 800), o / "pred.npz")
    assert main(["rollout", str(o / "pred.npz"), "--truth", str(o / "truth.npz"), "-o", str(o / "r")]) == 0
    out = capsys.readouterr().out
    assert "frequency drifts" in out and "dying out" in out
    d = o / "ds"
    (d / "cases").mkdir(parents=True)
    base = traj(lambda s: 0.2 + 0 * s, lambda s: 1 + 0 * s, 200)
    for k, re_ in enumerate([100, 200, 300, 400, 500, 200]):
        F = base.fields * (1 + re_ / 1000)
        if k == 1:
            F = F.copy(); F[0] += 5
        if k == 2:
            F = F.copy(); F[50] = F[49]
        save(Traj(F, base.times, base.names, x=base.x, y=base.y, mask=base.mask, params={"Re": re_}),
             d / "cases" / f"c{k}.npz")
    (d / "split.json").write_text(json.dumps(dict(train=["c0", "c1", "c2", "c3"], val=["c4"], test=["c5"])))
    assert main(["dataset", str(d), "-o", str(o / "a")]) == 3
    rep = (o / "a/report.md").read_text()
    assert "first snapshot does not follow" in rep and "repeated snapshots" in rep
    assert "leak" in rep and "outside the training range" in rep
