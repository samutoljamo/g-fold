# gfold-fixtures — CVXPY oracle for gfold-core

`gfold-core` (Rust, direct Clarabel) is the **source of truth** and production
solver. This package is its independent mathematical spec and differential-test
oracle: a lean CVXPY statement of the same G-FOLD SOCP, used to generate the
fixtures `gfold-core`'s tests check against.

- `gfold_oracle/model.py` — the CVXPY formulation (mirrors `../gfold-core/src/assemble.rs`, including the first-order-hold dynamics). The readable spec.
- `gfold_oracle/config.py` — config matching `config.rs`'s JSON shape.
- `cases/*.json` — test-case inputs, committed (one place to add a case).
- `data/*.json` — fixtures `{name, config, expected}` consumed by `gfold-core/tests/fixtures.rs`. **Generated, not committed** (gitignored): solver output is not bit-reproducible across machines, so committing it produces spurious diffs.

## Generate fixtures and run the differential test

```bash
cd gfold-fixtures
uv run python -m gfold_oracle.dump      # cases/ -> data/
cd .. && cargo test -p gfold-core --test fixtures
```

`fixtures.rs` checks the Rust solver against the oracle within tolerance gates,
which absorb cross-machine float noise. With no `data/` present, those tests
skip, so a plain `cargo test` runs standalone without this Python toolchain.

When you change the formulation in Rust, mirror it in `model.py` and re-run the
above. CI (`.github/workflows/oracle.yml`) does exactly this on every push.

## Test the oracle itself

```bash
uv run pytest          # oracle sanity checks
```

Managed with [uv](https://docs.astral.sh/uv/); `uv.lock` pins the solver stack.
