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
