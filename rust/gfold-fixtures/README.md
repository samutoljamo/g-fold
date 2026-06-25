# gfold-fixtures — CVXPY oracle for gfold-core

`gfold-core` (Rust, direct Clarabel) is the **source of truth** and production
solver. This package is its independent mathematical spec and differential-test
oracle: a lean CVXPY statement of the same G-FOLD SOCP, used to generate the
fixtures `gfold-core`'s tests check against.

- `gfold_oracle/model.py` — the CVXPY formulation (mirrors `../gfold-core/src/assemble.rs`, including the first-order-hold dynamics). The readable spec.
- `gfold_oracle/config.py` — config matching `config.rs`'s JSON shape.
- `cases/*.json` — test-case inputs (one place to add a case).
- `data/*.json` — generated fixtures `{name, config, expected}` consumed by `gfold-core/tests/fixtures.rs`.

## Regenerate fixtures

After changing the formulation in Rust, mirror it in `model.py`, then:

```bash
cd rust/gfold-fixtures
uv run python -m gfold_oracle.dump
cd .. && cargo test -p gfold-core
```

Commit both sides plus the regenerated `data/`. CI fails if they drift.

## Test

```bash
uv run pytest          # oracle sanity checks
```

Managed with [uv](https://docs.astral.sh/uv/); `uv.lock` pins the solver stack
so oracle output is reproducible.
