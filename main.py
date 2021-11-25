import argparse, os, shutil

from scipy import special, misc
misc.logsumexp = special.logsumexp
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from cvxpy_codegen import codegen
from jinja2 import Template

parser = argparse.ArgumentParser(description="gfold c++ generator")
parser.add_argument('-n', type=int, help="determines the amount of steps and thus determines dt(dt = tf/N)", default=100)
parser.add_argument('-g', '--generate', help="instead of solving the example it generates c code to solver other problems fast", action="store_true")

args = parser.parse_args()

# Problem data

n = args.n
up = 0 # 0 = x, 1=y, 2=z is the altitude

x = cp.Variable(n, 6, "x") # position(3), speed(3)
u = cp.Variable(n, 3, "u") # acc due to rocket engine

s = cp.Variable(n, 1, "s") # slack variable, equal to |u|
z = cp.Variable(n, 1, "z") # ln(mass)
wet_mass = 2000 # wet mass of the rocket(fuel + dry mass)
fuel = 1700 # weight of fuel
real_max_t = 24000 # used to get the % of the thrust right in the matplotlib chart

log_mass = cp.Parameter(1, name="log_mass") # ln(mass)
log_mass.value = np.log(wet_mass)
max_vel = cp.Parameter(1, name="max_vel") # maximum velocity
max_vel.value = 1000
sin_glide_slope = cp.Parameter(1, name="sin_glide_slope", sign="positive") # equal to sine of glideslope angle
sin_glide_slope.value = np.sin(np.radians(0))
log_dry_mass = cp.Parameter(1, name="log_dry_mass") # ln(dry mass)
log_dry_mass.value = np.log(wet_mass-fuel)
min_t = cp.Parameter(1, sign="positive", name="min_thrust") # min thrust of the engine in newtons
min_t.value = real_max_t*0.2
max_t = cp.Parameter(1, sign="positive", name="max_thrust") # max thrust of the engine in newtons
max_t.value = real_max_t*0.8
a = cp.Parameter(1, name="fuel_consumption") # fuel consumption parameter
a.value = 5e-4
dt  = cp.Parameter(1, name="dt") # time of flight = dt * N
dt.value = 44.63/n
g = cp.Parameter(1, 3, name="gravity") # assume constant gravity
g.value = [[0], [0], [-3.71]]
z0 = cp.Parameter(n, name="z0") # z0(t) = ln(m - max_thrust * a * t)
exp_z0 = cp.Parameter(n, name="exp_z0", sign="positive") # e to the power of z0

c_z0 = []
c_exp_z0 = []
for i in range(n):
    z00 = np.log(wet_mass - a.value*max_t.value*i*dt.value)
    c_z0.append(z00)
    c_exp_z0.append(np.exp(-z00))
z0.value = c_z0
exp_z0.value = c_exp_z0

max_angle = cp.Parameter(1, name="max_angle") # maximum angle in radians
max_angle.value = np.cos(np.radians(180))
dt_squared = cp.Parameter(1, name="dt_squared") # cvxpy codegen doesn't allow doing maths with parameters so these must be pre calculated
dt_squared.value = (dt.value)**2
initial_pos = cp.Parameter(1, 3, name="initial_position") # position of the spacecraft, relative to the landing site
initial_pos.value = [[2400], [450], [-330]]
initial_vel = cp.Parameter(1, 3, name="initial_vel") # velocity of the spacecraft
initial_vel.value = [[-10], [-40], [10]]
target_vel = cp.Parameter(1, 3, name="target_velocity") # velocity at the landing, [0, 0, 0] pretty much all the time
target_vel.value = [[0], [0], [0]]


# starting constraints

constraints = [
x[0, :3] == initial_pos,
x[0, 3:] == initial_vel,
z[0]==log_mass,
]

# timestep constraints

for i in range(n):
    constraints.append(cp.norm(x[i, 3:]) <= max_vel) # never exceed the maximum velocity
    constraints.append(x[i, up] >= cp.norm(x[i, :3]) * sin_glide_slope) # makes sure the spacecraft doesn't go subsurface/hit terrain by defining a cone in which the spacecraft is allowed to fly
    constraints.append(s[i] >= cp.norm(u[i, :])) # |u| = s, this constraint prevents u being greater than s
    constraints.append(u[i, up] >= max_angle * s[i]) # limits throttling angle
    constraints.append(min_t * exp_z0[i] * (1 - (z[i]-z0[i]) + (cp.square(z[i]-z0[i]))/2) <= s[i]) # lower bound for s
    constraints.append(s[i] <= max_t * exp_z0[i] * (1 - (z[i]-z0[i]))) # upper bound for s
    if i != n - 1:
        acc = g + (u[i+1, :] + u[i, :])/2
        constraints += [
        x[i+1, :3] == x[i, :3] + (x[i, 3:] + x[i+1, 3:]) * dt / 2 + acc*dt_squared * (1/2), # position at t+1
        x[i+1, 3:] == x[i, 3:] + acc*dt, # velocity at t+1
        z[i+1] == z[i] - a * dt * (s[i] + s[i+1]) / 2 # update mass
        ]

# constraints on the last step

constraints += [
    x[n-1, :3] == [[0], [0], [0]], # landing site must be at [0, 0, 0] because glide slope constraints assumes that
    x[n-1, 3:] == target_vel,
    z[n-1] >= log_dry_mass,
]
obj = cp.Maximize(z[n-1])
prob = cp.Problem(obj, constraints)

# if --generate is specified, generate c++ code/python bindings and exit
if args.generate:
    save_location = str(n)
    codegen(prob, save_location)
    for f in os.listdir("files/"):
        file_location = f"files/{f}"
        if f.endswith(".templ"):
            with open(file_location, "r") as file:
                t = Template(file.read())
                t.stream(num=str(n), up=str(up)).dump(f"{save_location}/{f[:-6]}")
        else:
            shutil.copy(file_location, f"{save_location}/{f}")
    exit(0)
# otherwise solve the problem using cvxpy and show the results using matplotlib
print(prob.solve(verbose=True, max_iters=500))
print(np.exp(z[n-1].value))

x1 = np.array(x[:, 0].value).flatten()
y1 = np.array(x[:, 1].value).flatten()
z1 = np.array(x[:, 2].value).flatten()




fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax.plot(x1, y1, z1, "red", label="path")

cosx = np.sqrt(1-np.square(sin_glide_slope.value))
theta = np.arange(0, 2*np.pi, np.pi/100)
height = np.array(initial_pos.value)[0][2]
radius = np.sqrt(1-np.square(cosx))/cosx  * height
z_values = np.arange(0, height, height/20)
for count, zval in enumerate(z_values):
    x_vals = np.cos(theta) * (count+1)/20 * radius
    y_vals = np.sin(theta) * (count+1)/20 * radius
    ax.plot(x_vals, y_vals, zval, '#0000FF55')

ax.legend(loc="center left", bbox_to_anchor=(2, 0.5))
ax.set_xlim3d(-height*0.6, height*0.6)
ax.set_ylim3d(-height*0.6, height*0.6)
ax.set_zlim3d(0, height*1.2)


t = np.arange(0, dt.value*n, dt.value)

ax = fig.add_subplot(222)
v = np.linalg.norm(np.array(x[:, 3:].value), axis=1)
ax.plot(t, v, label="velocity")
ax.legend()

ax = fig.add_subplot(223)
thrust = np.linalg.norm(np.array(u[:, :].value), axis=1)
for i in range(n):
    thrust[i] *= np.exp(z.value[i])
thrust /= real_max_t
ax.plot(t, thrust*100, label="thrust")
ax.set_ylim(0, 100)
ax.legend()
ax = fig.add_subplot(224)
ax.plot(np.array(x.value)[:, 1], np.array(x.value)[:, 2], label="surface trajectory")

ax.legend()
plt.show()
