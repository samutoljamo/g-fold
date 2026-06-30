# G-FOLD: Fuel-Optimal Powered-Descent Guidance

**🚀 [Try the interactive playground →](https://samutoljamo.github.io/g-fold/)** — edit a landing config and solve it live in your browser (runs the solver via WebAssembly, no install).

G-FOLD ("Guidance for Fuel-Optimal Large Diverts") computes the fuel-optimal
powered-descent trajectory for a landing spacecraft. A Rust core
(`gfold-core`) poses the min-fuel soft-landing problem as a second-order cone
program and solves it directly with [Clarabel](https://clarabel.org), exposed
through a CLI, Python bindings, and a WebAssembly module from the same engine.
Correctness is guarded by a CVXPY reference oracle: CI differentially tests the
Rust solver against the reference within a tolerance, so the fast path stays
honest.

This started as a high-school curiosity project built on cvxpy (with Clarabel
as the backend) plus cvxpygen for C++ codegen. The Rust core replaces that
Python + C++ stack with a single, simpler problem definition that is both
faster for large horizons and trivial to bind into other languages. The
original cvxpy implementation lives on as the reference oracle that keeps the
new core correct.

It is based on:
- [Blackmore et al. "Minimum-Landing-Error Powered-Descent Guidance for Mars Landing Using Convex Optimization"](http://larsblackmore.com/iee_tcst13.pdf)
- [Neal et al. "Fuel-Optimal Spacecraft Guidance for Landing in Planetary Pits"](https://www.ri.cmu.edu/pub_files/2016/4/Fuel-Optimal-Spacecraft-Guidance-for-Landing-in-Planetary-Pits-Neal-Bhasin.pdf)

## Install

1. **Prebuilt CLI (recommended)** — shell installer:
   ```sh
   curl --proto '=https' --tlsv1.2 -LsSf https://github.com/samutoljamo/g-fold/releases/latest/download/gfold-cli-installer.sh | sh
   ```
2. **From crates.io** (any Rust platform):
   ```sh
   cargo install gfold-cli
   ```
3. **Python library**:
   ```sh
   pip install gfold
   ```
4. **WebAssembly / npm**:
   ```sh
   npm install @samutoljamo/gfold
   ```

## Usage

### CLI

```sh
# generate a default config to edit
gfold init -o config.json

# solve and render a trajectory plot
gfold solve config.json --plot plot.png
```

![G-FOLD trajectory](examples/gfold_plot.png)

*Produced by the `solve` command above using the default config in
[`examples/landing.json`](examples/landing.json).* By default `time_of_flight`
is `null`, so the solver searches for the fuel-optimal time of flight; set it to
a number to pin a fixed value.

### Python

```python
import gfold

cfg = gfold.Config()          # sensible defaults; time_of_flight searched
traj = gfold.solve(cfg)
print(traj.final_mass, traj.time_of_flight)
```

## Repository layout

- `gfold-core` — solver core (Clarabel SOCP)
- `gfold-cli` — `gfold` command-line tool
- `gfold-py` — Python bindings (PyPI [`gfold`](https://pypi.org/project/gfold/))
- `gfold-wasm` — WebAssembly bindings (npm `@samutoljamo/gfold`)
- `gfold-fixtures` — CVXPY reference oracle + differential tests

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for build, test, and contribution
instructions.

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
