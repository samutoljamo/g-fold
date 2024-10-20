# G-FOLD / Fuel Optimal Large Divert Guidance Algorithm

This is my python/c/c++ implementation of the G-FOLD algorithm based on http://larsblackmore.com/iee_tcst13.pdf and https://www.ri.cmu.edu/pub_files/2016/4/Fuel-Optimal-Spacecraft-Guidance-for-Landing-in-Planetary-Pits-Neal-Bhasin.pdf. \
The problem is described using CVXPY/python and c/c++ code is generated using CVXPYGen.

G-FOLD is a convex-optimization algorithm. It generates the fuel-optimal path to land the spacecraft at the desired location. You can use this program to solve/plot the fuel-optimal path. \
Here's an example:
![graph](example.png)

## Running from source

### Prerequisites

You need to install [Rust](https://www.rust-lang.org/tools/install) and [Eigen](https://github.com/oxfordcontrol/Clarabel.cpp#installation)

Clone the repository

```
git clone --recurse-submodules https://github.com/samutoljamo/g-fold.git
cd g-fold

```

install requirements

```
pip3 install cvxpy cvxpygen matplotlib
```

## Usage

run `python3 main.py -n 100` to solve the example problem and draw graphs using python/CVXPY\
n specifies the number of simulation steps so higher values create more accurate solutions but require more computing power

c/c++ code can be generated using the `-g` flag
eg. `python3 main.py -g -n 100`
