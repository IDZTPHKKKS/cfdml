import gzip
import re
from pathlib import Path

import numpy as np

from .errors import CfdmlError

_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
_BR = re.compile(r"[()]")


def _name(f):
    return f.name[:-3] if f.name.endswith(".gz") else f.name


def _read_text(path):
    if isinstance(path, ProcField):
        path = path.paths[0]
    path = Path(path)
    if not path.exists() and Path(str(path) + ".gz").exists():
        path = Path(str(path) + ".gz")
    if not path.exists():
        raise CfdmlError(f"file not found: {path}")
    raw = gzip.open(path, "rb").read() if path.suffix == ".gz" else path.read_bytes()
    head = raw[:2000].decode("latin-1")
    if re.search(r"format\s+binary", head):
        raise CfdmlError(f"{path} is written in binary format, which cfdml cannot read",
                         hint="convert the case to ASCII once with:  foamFormatConvert  (run inside the case folder), "
                              "or set  writeFormat ascii;  in system/controlDict")
    return raw.decode("latin-1")


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def is_case(path):
    p = Path(path)
    return p.is_dir() and (p / "system").is_dir() and ((p / "constant").is_dir() or any(_time_dirs(p)))


def _time_dirs(case):
    out = []
    for d in Path(case).iterdir():
        if d.is_dir() and _NUM.match(d.name):
            out.append((float(d.name), d))
    return sorted(out)


def times(case, fields=None):
    if _proc_dirs(case) and is_decomposed(case):
        return proc_times(case, fields)
    return _times(case, fields)


def _times(case, fields=None):
    case = Path(case)
    td = _time_dirs(case)
    keep = []
    for t, d in td:
        names = {_name(f) for f in d.iterdir() if f.is_file()}
        if fields is None and names or fields is not None and all(f in names for f in fields):
            keep.append((t, d))
    return keep


def field_names(case):
    if _proc_dirs(case) and is_decomposed(case):
        case = _proc_dirs(case)[0]
    count = {}
    for _, d in _time_dirs(case):
        for f in d.iterdir():
            if f.is_file() and not f.name.startswith("."):
                n = _name(f)
                count[n] = count.get(n, 0) + 1
    return count


def _list_body(text, start):
    m = re.compile(r"\s*(\d+)\s*([({])").match(text, start)
    if not m:
        raise CfdmlError("could not find a list in the OpenFOAM file")
    n = int(m.group(1))
    if m.group(2) == "{":
        end = text.index("}", m.end())
        return ("{" + text[m.end():end] + "}"), n, end + 1
    depth = 1
    for b in _BR.finditer(text, m.end()):
        depth += 1 if b.group() == "(" else -1
        if depth == 0:
            return text[m.end():b.start()], n, b.end()
    raise CfdmlError("unbalanced brackets in the OpenFOAM file (it may be cut off)")


def _values(body, n):
    if body.startswith("{"):
        v = np.array(body.strip("{}").replace("(", " ").replace(")", " ").split(), float)
        return np.tile(v, (n, 1)) if v.size > 1 else np.full(n, v[0])
    v = np.array(body.replace("(", " ").replace(")", " ").split(), float)
    if n == 0:
        return v
    if v.size % n:
        raise CfdmlError(f"list says {n} entries but holds {v.size} numbers")
    k = v.size // n
    return v if k == 1 else v.reshape(n, k)


def read_field(path, n_cells=None):
    if isinstance(path, ProcField):
        return np.concatenate([read_field(p, c) for p, c in zip(path.paths, path.counts)])
    text = _strip_comments(_read_text(path))
    m = re.search(r"internalField\s+(uniform|nonuniform)\s*", text)
    if not m:
        raise CfdmlError(f"{path} has no internalField; is it a field file?")
    if m.group(1) == "uniform":
        v = np.array(text[m.end():text.index(";", m.end())].replace("(", " ").replace(")", " ").split(), float)
        if n_cells is None:
            raise CfdmlError(f"{path} is uniform; the number of cells is needed", hint="check the case has a mesh")
        return np.full(n_cells, v[0]) if v.size == 1 else np.tile(v, (n_cells, 1))
    m2 = re.compile(r"List<\w+>\s*").match(text, m.end())
    body, n, _ = _list_body(text, m2.end() if m2 else m.end())
    return _values(body, n)


def _read_list(path):
    text = _strip_comments(_read_text(path))
    hdr = text.find("}") + 1 if "FoamFile" in text else 0
    m = re.search(r"\n\s*\d+\s*[({]", text[hdr:])
    if not m:
        raise CfdmlError(f"could not read the list in {path}")
    body, n, _ = _list_body(text, hdr + m.start())
    return body, n, text


def n_cells(case):
    if _proc_dirs(case) and is_decomposed(case):
        return sum(proc_counts(case))
    owner = Path(case) / "constant/polyMesh/owner"
    if owner.exists() or Path(str(owner) + ".gz").exists():
        text = _read_text(owner)
        m = re.search(r"nCells:\s*(\d+)", text)
        if m:
            return int(m.group(1))
        body, n, _ = _read_list(owner)
        return int(np.array(body.split(), int).max()) + 1
    for _, d in _time_dirs(case):
        for f in sorted(d.iterdir()):
            try:
                v = read_field(f)
                return len(v)
            except CfdmlError:
                continue
    raise CfdmlError(f"cannot work out the number of cells in {case}", hint="is constant/polyMesh present?")


def cell_centres(case):
    if _proc_dirs(case) and is_decomposed(case):
        return np.concatenate([cell_centres(p) for p in _proc_dirs(case)])
    case = Path(case)
    for _, d in _time_dirs(case):
        for name in ("C", "C.gz"):
            if (d / name).exists():
                return read_field(d / name)
    mesh = case / "constant/polyMesh"
    if not mesh.is_dir():
        raise CfdmlError(f"no cell centres in {case}: no C field and no constant/polyMesh",
                         hint="run  postProcess -func writeCellCentres  in the case")
    pb, npnt, _ = _read_list(mesh / "points")
    pts = _values(pb, npnt)
    fb, nf, _ = _read_list(mesh / "faces")
    faces = [np.array(f.split(), int) for f in re.findall(r"\d+\s*\(([^)]*)\)", fb)]
    if len(faces) != nf:
        raise CfdmlError("could not read constant/polyMesh/faces", hint="run  postProcess -func writeCellCentres")
    fc = np.array([pts[f].mean(0) for f in faces])
    ob, no, _ = _read_list(mesh / "owner")
    owner = np.array(ob.split(), int)
    nb, nn, _ = _read_list(mesh / "neighbour")
    neigh = np.array(nb.split(), int)
    nc = n_cells(case)
    s, c = np.zeros((nc, 3)), np.zeros(nc)
    np.add.at(s, owner, fc[:len(owner)]); np.add.at(c, owner, 1)
    np.add.at(s, neigh, fc[:len(neigh)]); np.add.at(c, neigh, 1)
    return s / np.maximum(c, 1)[:, None]


def cell_volumes(case):
    if _proc_dirs(case) and is_decomposed(case):
        v = [cell_volumes(p) for p in _proc_dirs(case)]
        return None if any(x is None for x in v) else np.concatenate(v)
    for _, d in _time_dirs(case):
        if (Path(d) / "V").exists() or (Path(d) / "V.gz").exists():
            return read_field(Path(d) / "V")
    return None


def field_header(path):
    if isinstance(path, ProcField):
        path = path.paths[0]
    text = _read_text(path)
    cls = re.search(r"class\s+(\w+)\s*;", text)
    dims = re.search(r"dimensions\s+(\[[^\]]*\])\s*;", text)
    b = text.find("boundaryField")
    return dict(cls=cls.group(1) if cls else "volScalarField", dims=dims.group(1) if dims else "[0 0 0 0 0 0 0]",
                boundary=text[b:] if b >= 0 else "boundaryField\n{\n}\n")


def write_field(path, name, values, header, note=""):
    if isinstance(path, ProcField):
        values = np.asarray(values)
        start = 0
        for p, c in zip(path.paths, path.counts):
            write_field(p, name, values[start:start + c], header, note)
            start += c
        return
    values = np.asarray(values)
    vec = values.ndim == 2
    kind = {1: "scalar", 3: "vector", 6: "symmTensor", 9: "tensor"}[values.shape[1] if vec else 1]
    lines = [f"({' '.join(f'{x:.8g}' for x in row)})" for row in values] if vec else [f"{x:.8g}" for x in values]
    text = ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class " + header["cls"] + ";\n    object " + name
            + ";\n}\n" + (f"// {note}\n" if note else "") + f"dimensions {header['dims']};\n\n"
            + f"internalField nonuniform List<{kind}>\n{len(values)}\n(\n" + "\n".join(lines) + "\n)\n;\n\n"
            + header["boundary"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def internal_faces(case):
    if _proc_dirs(case) and is_decomposed(case):
        return None
    mesh = Path(case) / "constant/polyMesh"
    if not (mesh / "neighbour").exists() and not (mesh / "neighbour.gz").exists():
        return None
    ob, _, _ = _read_list(mesh / "owner")
    nb, _, _ = _read_list(mesh / "neighbour")
    owner, neigh = np.array(ob.split(), int), np.array(nb.split(), int)
    return owner[:len(neigh)], neigh


def _proc_dirs(case):
    ps = [(int(m.group(1)), d) for d in Path(case).glob("processor*") if d.is_dir()
          for m in [re.match(r"processor(\d+)$", d.name)] if m]
    return [d for _, d in sorted(ps)]


def is_decomposed(case):
    case = Path(case)
    if not _proc_dirs(case):
        if any(case.glob("processors*")):
            raise CfdmlError(f"{case} uses the collated format (processors*/), which cfdml cannot read",
                             hint="run  reconstructPar , or decompose with  -fileHandler uncollated")
        return False
    return not any(_name(f) not in ("C", "V") for _, d in _time_dirs(case) if d.name != "0" for f in d.iterdir()
                   if f.is_file())


class ProcField:

    def __init__(self, paths, counts):
        self.paths, self.counts = list(paths), list(counts)
        self.name = self.paths[0].name
        self.parent = self.paths[0].parent

    def exists(self):
        return all(p.exists() or Path(str(p) + ".gz").exists() for p in self.paths)

    def __str__(self):
        return str(self.paths[0]).replace("processor0", "processor*")


class TimeDir:
    def __init__(self, dirs, counts):
        self.dirs, self.counts = list(dirs), list(counts)
        self.name = self.dirs[0].name

    def __truediv__(self, name):
        return ProcField([d / name for d in self.dirs], self.counts)

    def __str__(self):
        return str(self.dirs[0]).replace("processor0", "processor*")


def proc_counts(case):
    return [n_cells(p) for p in _proc_dirs(case)]


def proc_times(case, fields=None):
    procs = _proc_dirs(case)
    counts = proc_counts(case)
    out = []
    for t, d0 in times(procs[0], fields):
        dirs = [p / d0.name for p in procs]
        if all(d.is_dir() for d in dirs) and (fields is None or all((d / f).exists() or Path(str(d / f) + ".gz").exists()
                                                                  for d in dirs for f in fields)):
            out.append((t, TimeDir(dirs, counts)))
    return out


def out_target(case, base, time_name, name):
    if is_decomposed(case):
        procs = _proc_dirs(case)
        return ProcField([Path(base) / p.name / time_name / name for p in procs], proc_counts(case))
    return Path(base) / time_name / name
