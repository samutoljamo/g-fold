"""Regenerate oracle fixtures: cases/<name>.json -> data/<name>.json.

Run: python -m gfold_oracle.dump   (from rust/gfold-fixtures, via `uv run`)
"""
import json
from pathlib import Path

from .config import Config
from .model import solve

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "cases"
DATA = ROOT / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for case_path in sorted(CASES.glob("*.json")):
        name = case_path.stem
        cfg = Config.from_dict(json.loads(case_path.read_text()))
        expected = solve(cfg)
        fixture = {"name": name, "config": cfg.to_dict(), "expected": expected}
        out = DATA / f"{name}.json"
        out.write_text(json.dumps(fixture, indent=2) + "\n")
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
