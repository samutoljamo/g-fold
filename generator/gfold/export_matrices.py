"""Export CVXPY canonical problem data for matrix-diff testing."""
import json
import numpy as np
import cvxpy as cp
from .config import GFoldConfig
from .solver import GFoldSolver


def export(out_path: str) -> None:
    solver = GFoldSolver(GFoldConfig())
    prob = solver.problem
    data, _, _ = prob.get_problem_data(cp.CLARABEL)
    A = data["A"]  # scipy sparse
    A_coo = A.tocoo()
    payload = {
        "n_vars": int(A.shape[1]),
        "n_rows": int(A.shape[0]),
        "A": {
            "rows": A_coo.row.tolist(),
            "cols": A_coo.col.tolist(),
            "vals": A_coo.data.tolist(),
            "shape": list(A.shape),
        },
        "b": np.asarray(data["b"]).ravel().tolist(),
        "q": np.asarray(data["c"]).ravel().tolist(),
        "cones": _cone_list(data),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"wrote {out_path}")
    print(f"n_vars={payload['n_vars']}  n_rows={payload['n_rows']}")


def _cone_list(data: dict) -> list:
    dims = data["dims"]
    out = []
    # dims attributes: zero, nonneg, soc (list of ints), exp, psd, p3d, pnd
    if getattr(dims, "zero", 0):
        out.append({"type": "z", "dim": int(dims.zero)})
    if getattr(dims, "nonneg", 0):
        out.append({"type": "nn", "dim": int(dims.nonneg)})
    for soc_dim in getattr(dims, "soc", []):
        out.append({"type": "soc", "dim": int(soc_dim)})
    return out
