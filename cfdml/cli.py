import argparse
import sys

from . import __version__
from .errors import CfdmlError

EXAMPLES = """
examples:
  cfdml demo demo                                   make example data to try every command
  cfdml info demo/wake                              what is in a case / file
  cfdml spectrum demo/wake/postProcessing/forceCoeffs/0/coefficient.dat -c Cl --after 20 -L 1 -U 1
  cfdml pod demo/wake -f U --modes 4
  cfdml dmd demo/wake -f U
  cfdml fieldstats demo/drops -f alpha.water          droplet volume, centroid and count over time
  cfdml forecast demo/wake/postProcessing/probes/0/U -c probe1_y --train 0.6 --horizon 20
  cfdml steady demo/wake/postProcessing/forceCoeffs/0/coefficient.dat
  cfdml gci --cells 25000,100000,400000 --values 1.42,1.39,1.385 --name Cd
  cfdml compare runs.csv -x Re -y St --ref cylinder-st-williamson
  cfdml residuals case/log.pimpleFoam
  cfdml forces case --rho 1 -U 1 --area 0.1
  cfdml average case -f U -f p --after auto
  cfdml sensitivity demo/cfdml_sweep
  cfdml optimize demo/cfdml_sweep --minimize Cd --subject-to St>=0.2
  cfdml export --sweep demo/sweep.yaml -f U -o data
  cfdml evaluate pred.npz data/cases/Re350_a4.npz
  cfdml rollout pred.npz --truth data/cases/Re350_a4.npz
  cfdml dataset data
  cfdml init sweep                                  write a sweep config to fill in
  cfdml sweep demo/sweep.yaml                       learn outputs from a set of runs
  cfdml predict demo/cfdml_sweep/model.pkl --at Re=350,angle=4
"""


def build():
    ap = argparse.ArgumentParser(prog="cfdml", description="cfdml: analysis and machine-learning predictions for CFD results", epilog=EXAMPLES,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"cfdml {__version__}")
    ap.add_argument("--debug", action="store_true", help="show full Python errors")
    sub = ap.add_subparsers(dest="cmd", metavar="command")

    def p(name, help_, fn):
        s = sub.add_parser(name, help=help_.replace("%", "%%"), description=help_)
        s.set_defaults(fn=fn)
        s.add_argument("-o", "--out", help="output folder (default: cfdml_<command>_... in the current folder)")
        s.add_argument("--overwrite", action="store_true", help="write into an output folder that holds other files")
        s.add_argument("--debug", action="store_true", dest="debug_sub", help="show full Python errors")
        return s

    from . import commands as C
    s = p("info", "show what is in an OpenFOAM case, a probe/CSV file or a VTK file", C.info)
    s.add_argument("path")

    def window(s):
        s.add_argument("--after", type=float, help="ignore everything before this time (e.g. the start-up transient)")
        s.add_argument("--before", type=float, help="ignore everything after this time")

    s = p("spectrum", "dominant frequency, Strouhal number and spectrum of a signal (forces, probes, any CSV)",
          C.cmd_spectrum)
    s.add_argument("file"); s.add_argument("-c", "--column", help="column to analyse (default: first after time)")
    window(s)
    s.add_argument("-L", "--length", type=float, help="reference length, for the Strouhal number")
    s.add_argument("-U", "--velocity", type=float, help="reference velocity, for the Strouhal number")

    for name, help_, fn in (("pod", "POD modes and energies of a field over time (coherent structures, ROM size)", C.cmd_pod),
                            ("dmd", "DMD: frequencies and growth rates of the structures in a field", C.cmd_dmd)):
        s = p(name, help_, fn)
        s.add_argument("case", help="OpenFOAM case folder (ASCII, reconstructed)")
        s.add_argument("-f", "--field", default="U", help="field name (default U)")
        window(s)
        s.add_argument("--every", type=int, default=1, help="use every n-th time folder")
        s.add_argument("--box", help="only cells inside xmin,xmax,ymin,ymax,zmin,zmax")
        if name == "pod":
            s.add_argument("--modes", type=int, default=4, help="modes to plot and save (default 4)")
            s.add_argument("--write-modes", action="store_true", help="also write the modes as OpenFOAM fields")
        else:
            s.add_argument("--rank", type=int, help="number of DMD modes (default: enough for 99.9%% of the energy)")
            s.add_argument("-L", "--length", type=float); s.add_argument("-U", "--velocity", type=float)

    s = p("fieldstats", "turn a field into time series: phase volume, centroid, droplet/bubble count; or mean/max/integral",
          C.cmd_fieldstats)
    s.add_argument("case"); s.add_argument("-f", "--field", required=True, help="e.g. alpha.water, T, U")
    window(s)
    s.add_argument("--box", help="only cells inside xmin,xmax,ymin,ymax,zmin,zmax")
    s.add_argument("--phase", action="store_true", help="treat the field as a phase fraction even if not named alpha*")
    s.add_argument("--threshold", type=float, default=0.5, help="phase fraction that counts as inside a blob (default 0.5)")

    s = p("forecast", "learn the start of a signal, predict the rest, and say how far it can be trusted", C.cmd_forecast)
    s.add_argument("file"); s.add_argument("-c", "--column")
    window(s)
    s.add_argument("--train", type=float, default=0.7, help="fraction used for fitting; the rest scores it (default 0.7)")
    s.add_argument("--horizon", type=float, help="time to forecast past the end (default: the length of the test part)")

    s = p("sweep", "learn how outputs depend on inputs across many runs; accuracy, input importance, next runs", C.cmd_sweep)
    s.add_argument("config", help="sweep YAML (make one with  cfdml init sweep)")
    s.add_argument("--suggest", type=int, default=5, help="how many next runs to suggest (default 5)")

    s = p("predict", "predict outputs (and fields) at new conditions with a model from  cfdml sweep", C.cmd_predict)
    s.add_argument("model", help="model.pkl, or the sweep output folder")
    s.add_argument("--at", action="append", help="a point, e.g. Re=350,angle=5 (repeat for more)")
    s.add_argument("--grid", action="append", help="a range per input, e.g. Re=100:1000:10 (start:stop:count)")
    s.add_argument("--max-fields", type=int, default=5, help="write predicted fields for at most this many points")

    s = p("steady", "has the run settled? transient end, means with 95% error bars, drift, how much longer to run",
          C.cmd_steady)
    s.add_argument("file"); s.add_argument("-c", "--column", action="append", help="column (repeat; default Cd and Cl)")
    s.add_argument("--after", type=float, help="use this start time instead of detecting the transient")
    s.add_argument("--target", type=float, default=0.01, help="wanted relative precision of the means (default 0.01)")

    s = p("gci", "mesh convergence: observed order, extrapolated value and Grid Convergence Index (3 meshes)", C.cmd_gci)
    s.add_argument("--values", help="value on each mesh, e.g. 1.385,1.39,1.42")
    s.add_argument("--cells", help="cell counts of the meshes, e.g. 400000,100000,25000")
    s.add_argument("--h", help="mesh sizes instead of cell counts")
    s.add_argument("--dim", type=int, default=2, help="2 or 3, for h = cells^(-1/dim) (default 2)")
    s.add_argument("--name", help="name of the quantity, for the report")
    s.add_argument("--cases", nargs="+", help="three case folders; values read from --file, cells from the mesh")
    s.add_argument("--file", help="file inside each case, e.g. postProcessing/forceCoeffs/0/coefficient.dat")
    s.add_argument("-c", "--column"); s.add_argument("--stat", default="mean")
    s.add_argument("--after", help="start time for the statistic, or auto (default)")

    s = p("compare", "compare results with a built-in reference or experimental data, with % errors", C.cmd_compare)
    s.add_argument("file", nargs="?", help="your results: CSV or table with the x and y columns (e.g. runs.csv)")
    s.add_argument("-x"); s.add_argument("-y")
    s.add_argument("--ref", help="built-in reference (see --list)")
    s.add_argument("--ref-file", help="reference data file (CSV / table)")
    s.add_argument("--ref-x"); s.add_argument("--ref-y")
    s.add_argument("--tol", type=float, default=5.0, help="tolerance band in %% (default 5)")
    s.add_argument("--list", action="store_true", help="list the built-in references")

    s = p("residuals", "solver log or solverInfo: residuals, Courant number, crashes, convergence, time to finish",
          C.cmd_residuals)
    s.add_argument("path", help="log file (e.g. log.pimpleFoam), solverInfo.dat / residuals.dat, or a case folder")

    s = p("forces", "forces function object output: drag/lift or Cd/Cl, pressure/viscous split, statistics", C.cmd_forces)
    s.add_argument("path", help="force(s).dat, postProcessing/<forces> folder (restarts joined), or a case folder")
    s.add_argument("--rho", type=float, help="reference density (with -U and --area gives coefficients)")
    s.add_argument("-U", "--velocity", type=float, help="reference velocity")
    s.add_argument("--area", type=float, help="reference area (2D: span x length, e.g. 0.1 x D)")
    s.add_argument("-L", "--length", type=float, help="reference length, for the Strouhal number")
    s.add_argument("--drag", default="x", choices=list("xyz"), help="drag direction (default x)")
    s.add_argument("--lift", default="y", choices=list("xyz"), help="lift direction (default y)")
    s.add_argument("--after", type=float, help="start of the statistics (default: detected)")
    s.add_argument("--target", type=float, default=0.01, help="wanted relative precision of the means (default 0.01)")

    s = p("average", "time-averaged fields and Reynolds stresses (fieldMean, fieldPrime2Mean) in OpenFOAM format",
          C.cmd_average)
    s.add_argument("case")
    s.add_argument("-f", "--field", action="append", required=True, help="field to average (repeat for more)")
    s.add_argument("--after", help="start time, or auto (from the forces in postProcessing)")
    s.add_argument("--before", type=float, help="end time")
    s.add_argument("--every", type=int, default=1, help="use every n-th time folder")
    s.add_argument("--into-case", action="store_true", help="also write the averages into the last time folder")
    s.add_argument("--force", action="store_true", help="overwrite existing average fields in the case")

    s = p("sensitivity", "Sobol sensitivity indices (first-order and total) of a sweep model, with intervals",
          C.cmd_sensitivity)
    s.add_argument("model", help="model.pkl, or the sweep output folder")
    s.add_argument("--output", action="append", help="output to analyse (repeat; default all)")
    s.add_argument("-n", type=int, default=4096, help="Monte Carlo base samples (default 4096)")

    s = p("optimize", "best inputs for an output on a sweep model, with constraints; and the next run to verify it",
          C.cmd_optimize)
    s.add_argument("model", help="model.pkl, or the sweep output folder")
    s.add_argument("--minimize"); s.add_argument("--maximize")
    s.add_argument("--subject-to", action="append", help="constraint like St<=0.21 (repeat for more)")
    s.add_argument("--fix", action="append", help="hold an input fixed, e.g. angle=5")
    s.add_argument("--confidence", type=float, default=0.95, help="required probability that constraints hold (default 0.95)")

    s = p("export", "OpenFOAM cases (or a sweep) to an ML-ready dataset: npz per case, split by case, train statistics",
          C.cmd_export)
    s.add_argument("cases", nargs="*", help="case folders")
    s.add_argument("--sweep", help="take the cases and their input values from a sweep config")
    s.add_argument("-f", "--field", action="append", help="field to export (repeat)")
    s.add_argument("--grid", default="128", help="regular grid NX or NXxNY (default 128, NY from the aspect ratio)")
    s.add_argument("--no-grid", dest="grid", action="store_const", const=None, help="keep the cells (unstructured)")
    s.add_argument("--box", help="grid region xmin,xmax,ymin,ymax")
    s.add_argument("--after", help="start time, or auto (from the forces)")
    s.add_argument("--before", type=float); s.add_argument("--every", type=int, default=1)
    s.add_argument("--split", default="0.7,0.15,0.15", help="train,val,test fractions (default 0.7,0.15,0.15)")
    s.add_argument("--seed", type=int, default=0)

    s = p("evaluate", "ML prediction vs CFD truth: error growth, statistics, spectra, divergence, error maps", C.cmd_evaluate)
    s.add_argument("pred", help="prediction: trajectory .npz or OpenFOAM case")
    s.add_argument("truth", help="truth: trajectory .npz or OpenFOAM case")
    s.add_argument("-f", "--field", action="append", help="fields to compare when the inputs are OpenFOAM cases")
    s.add_argument("--velocity", help="velocity channel names u,v (default: Ux,Uy or u,v)")
    s.add_argument("--channel", help="channel for the field pictures (default: the first)")
    s.add_argument("--steps", type=int, help="compare only the first N snapshots")

    s = p("rollout", "long-horizon behaviour of a learned model: flow type, frequency drift, amplitude, blow-up", C.cmd_rollout)
    s.add_argument("pred", help="long prediction: trajectory .npz or OpenFOAM case")
    s.add_argument("--truth", help="reference trajectory (may be shorter than the prediction)")
    s.add_argument("-f", "--field", action="append")
    s.add_argument("--probes", type=int, default=4, help="number of probe points (default 4)")
    s.add_argument("--skip", type=float, default=0.2, help="fraction of the truth treated as transient (default 0.2)")

    s = p("dataset", "audit an ML dataset: bad values, jumps, transients, duplicates, flow types, split coverage and leaks",
          C.cmd_dataset)
    s.add_argument("path", help="folder from cfdml export, a folder of .npz trajectories, or one .npz")

    s = p("init", "write a commented template config", C.cmd_init)
    s.add_argument("kind", choices=["sweep"]); s.add_argument("--force", action="store_true")

    s = p("demo", "write example data (an OpenFOAM-format wake case and a sweep) to try every command", C.cmd_demo)
    s.set_defaults(out=None)
    s.add_argument("out", nargs="?", default="cfdml_demo")
    return ap


def main(argv=None):
    ap = build()
    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 0
    from .report import COMMAND
    COMMAND.update(name=args.cmd, overwrite=getattr(args, "overwrite", False))
    debug = args.debug or getattr(args, "debug_sub", False)
    try:
        rc = args.fn(args)
    except CfdmlError as e:
        print(f"\ncfdml: {e}", file=sys.stderr)
        if e.hint:
            print(f"  hint: {e.hint}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ncfdml: stopped", file=sys.stderr)
        return 130
    except MemoryError:
        print("\ncfdml: ran out of memory", file=sys.stderr)
        print("  hint: use --every, --box or a shorter time window to read less data", file=sys.stderr)
        return 1
    except Exception as e:
        if debug:
            raise
        print(f"\ncfdml: unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        print("  rerun with --debug for the full error, and send it to whoever maintains cfdml", file=sys.stderr)
        return 2
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
