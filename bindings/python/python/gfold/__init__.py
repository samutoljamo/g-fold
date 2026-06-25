"""Python bindings for the gfold Rust solver."""
import dataclasses
import json

from ._gfold import solve_json as _solve_json
from ._types import Config, Trajectory


class GFoldError(Exception):
    """Raised when the solver reports an error."""


def solve(config: Config) -> Trajectory:
    resp = json.loads(_solve_json(json.dumps(dataclasses.asdict(config))))
    if "err" in resp:
        raise GFoldError(resp["err"])
    return Trajectory(**resp["ok"])


__all__ = ["solve", "Config", "Trajectory", "GFoldError"]
