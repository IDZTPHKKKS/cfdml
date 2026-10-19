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
  cfdml init sweep                                  write a commented sweep config to fill in
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
        s = sub.add_parser(name, help=help_, description=help_)
        s.set_defaults(fn=fn)
        s.add_argument("-o", "--out", help="output folder (default: cfdml_<command>_... in the current folder)")
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
    try:
        args.fn(args)
    except CfdmlError as e:
        print(f"\ncfdml: {e}", file=sys.stderr)
        if e.hint:
            print(f"  hint: {e.hint}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ncfdml: stopped", file=sys.stderr)
        return 130
    except Exception as e:
        if args.debug:
            raise
        print(f"\ncfdml: unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        print("  rerun with --debug for the full error, and send it to whoever maintains cfdml", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
