# G-FOLD Python Generator

This is the Python implementation of the G-FOLD algorithm with code generation capabilities. The problem is described using CVXPY/Python and C/C++ code is generated using CVXPYGen.

## Installation

### Option 1: Install from source

#### Prerequisites

You need to install [Rust](https://www.rust-lang.org/tools/install) and [Eigen](https://github.com/oxfordcontrol/Clarabel.cpp#installation) for the code generation feature.

Clone the repository:

```bash
git clone https://github.com/samutoljamo/g-fold.git
cd g-fold/generator
```

Install the package in development mode:

```bash
pip install -e .
```

On WSL, make sure you've installed Tkinter (version depends on the python version you're using):
```bash
sudo apt-get install python3.12-tk
```

### Option 2: Install from PyPI

```bash
pip install gfold
```

## Usage

### As a command-line tool

After installation, you can run G-FOLD from the command line:

```bash
# Solve the example problem with 100 steps and display graphs
gfold -n 100

# Generate C++ code
gfold -g -n 100 -o output_directory

# Save the plot to a file without displaying it
gfold -n 100 --save-plot --no-plot
```

### As a Python library

```python
from gfold import GFoldSolver
from gfold.visualization import plot_results

# Create a solver with 100 steps
solver = GFoldSolver()

# Solve the problem
solution = solver.solve(verbose=True)
print(f"Final mass: {solution['final_mass']:.2f} kg")

# Plot the results
plot_results(solution, save_path="gfold_plot.png")

# Generate C++ code
solver.generate_code(code_dir="generated_code")
```

### Example scripts

Check the `examples` directory for more usage examples:

- `simple_example.py`: Basic usage of the solver and visualization

## Development

To set up the development environment:

```bash
cd g-fold/generator
pip install -e .
```
