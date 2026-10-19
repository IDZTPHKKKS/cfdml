from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
    if _style in plt.style.available:
        plt.style.use(_style)
        break
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "savefig.dpi": 160, "figure.dpi": 100})
COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]


COMMAND = {"name": None, "overwrite": False}


def claim(out):
    out = Path(out)
    mark = out / ".cfdml"
    name = COMMAND["name"] or "cfdml"
    if out.exists() and any(out.iterdir()) and not COMMAND["overwrite"]:
        owner = mark.read_text().strip() if mark.exists() else None
        if owner != name:
            from .errors import CfdmlError
            raise CfdmlError(f"{out} already holds " + (f"output of  cfdml {owner}" if owner else "other files"),
                             hint="choose another folder with -o, or add --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    mark.write_text(name + "\n")
    return out


class Report:
    def __init__(self, out, title):
        self.out = claim(out)
        self.lines = [f"# {title}", ""]
        self.summary = []

    def say(self, text, summary=False):
        self.lines.append(text)
        if summary:
            self.summary.append(text)

    def table(self, header, rows):
        self.lines += ["", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        self.lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        self.lines.append("")

    def figure(self, fig, name, caption=""):
        fig.tight_layout()
        fig.savefig(self.out / f"{name}.png")
        plt.close(fig)
        self.lines += ["", f"![{caption or name}]({name}.png)", ""]
        return self.out / f"{name}.png"

    def close(self):
        (self.out / "report.md").write_text("\n".join(self.lines) + "\n")
        for s in self.summary:
            print("  " + s)
        print(f"\n  full report and plots: {self.out}/report.md")


def fmt(x, n=4):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if x == 0 or 1e-3 <= abs(x) < 1e5:
        return f"{x:.{n}g}"
    return f"{x:.{max(n - 1, 1)}e}"
