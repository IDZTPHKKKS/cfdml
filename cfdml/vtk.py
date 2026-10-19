from pathlib import Path

import numpy as np

from .errors import CfdmlError


def read_vtk(path):
    try:
        import meshio
    except ImportError:
        raise CfdmlError("reading VTK files needs the meshio package", hint="pip install meshio")
    path = Path(path)
    if not path.exists():
        raise CfdmlError(f"file not found: {path}")
    m = meshio.read(path)
    if m.cell_data:
        cells = [c.data for c in m.cells]
        centres = np.concatenate([m.points[c].mean(1) for c in cells])
        fields = {k: np.concatenate([np.asarray(b) for b in v]) for k, v in m.cell_data.items()
                  if all(np.asarray(b).ndim <= 2 for b in v)}
        return centres, fields
    return m.points, {k: np.asarray(v) for k, v in m.point_data.items()}
