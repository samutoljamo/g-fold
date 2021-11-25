# G-FOLD / Fuel Optimal Large Divert Guidance Algorithm
This is my python/c/c++ implementation of the G-FOLD algorithm based on http://larsblackmore.com/iee_tcst13.pdf and https://www.ri.cmu.edu/pub_files/2016/4/Fuel-Optimal-Spacecraft-Guidance-for-Landing-in-Planetary-Pits-Neal-Bhasin.pdf.
The problem is described using CVXPY/python and c/c++ code is generated using [CVXPY-CODEGEN](https://github.com/moehle/cvxpy_codegen)

G-FOLD is a convex-optimization algorithm. It generates the fuel-optimal path to land the spacecraft at the desired location. You can use this program to solve/plot the fuel-optimal path.


## Prerequisites
The program was tested in WSL2/Ubuntu
Code generation should work in windows too but it might be harder to compile the generated code

## Installing
Clone the repository
```
git clone --recurse-submodules https://github.com/samutoljamo/g-fold.git
cd g-fold
```
It is recommended to create a python env and use it since CVXPY-CODEGEN requires old libraries

run `install.py` to install requirements

## Usage
run `python3 main.py -n 100` to solve the example problem and draw graphs using python/CVXPY
n specifies the number of simulation steps so higher values create more accurate solutions but require more computation power

c/c++ code can be generated using the `-g` flag
eg. `python3 main.py -g -n 100`

the generated code is in a folder called 100(name depends on n)

compile the c/c++ code
```
cd 100
make
```

run `python3 draw.py` to solve the example problem

## todo
- Better python interface
- Behaviour when there is no solution
