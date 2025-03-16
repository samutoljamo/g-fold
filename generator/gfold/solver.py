import cvxpy as cp
import numpy as np
import os
from cvxpygen import cpg
from .config import GFoldConfig

class GFoldSolver:
    """
    G-FOLD solver that implements the Fuel Optimal Large Divert Guidance Algorithm.
    """
    
    def __init__(self, config=None):
        """
        Initialize the G-FOLD solver.
        
        Args:
            config (GFoldConfig): Configuration object with problem parameters
        """
        self.config = config if config is not None else GFoldConfig()
            
        self.parameters = {}
        self.variables = {}
        self.constraints = []
        self.problem = None
        self._setup_problem()
        
    def _calculate_parameter(self, expression, *args, **kwargs):
        """Helper function to create parameters from expressions with values"""
        return cp.Parameter(*args, value=expression.value, **kwargs)

    def _setup_problem(self):
        """Set up the G-FOLD optimization problem."""
        config = self.config
        n = config.solver.n
        t = config.solver.time_of_flight
        
        # Variables
        x = cp.Variable((n, 6), "x")  # position(3), speed(3)
        u = cp.Variable((n, 3), "u")  # acc due to rocket engine
        s = cp.Variable(n, "s")  # slack variable, equal to |u|
        z = cp.Variable(n, "z")  # ln(mass)
        
        self.variables = {
            "x": x,
            "u": u,
            "s": s,
            "z": z
        }
        
        # Parameters
        log_mass = cp.Parameter(name="log_mass", value=config.spacecraft.log_wet_mass)
        max_vel = cp.Parameter(name="max_vel", value=config.spacecraft.max_velocity)
        sin_glide_slope = cp.Parameter(name="sin_glide_slope", nonneg=True, value=config.environment.sin_glide_slope)
        log_dry_mass = cp.Parameter(name="log_dry_mass", value=config.spacecraft.log_dry_mass)
        min_t = cp.Parameter(nonneg=True, name="min_thrust", value=config.spacecraft.min_thrust)
        max_t = cp.Parameter(nonneg=True, name="max_thrust", value=config.spacecraft.max_thrust)
        dt = cp.Parameter(name="dt", value=t/n)
        a = cp.Parameter(name="fuel_consumption", value=config.spacecraft.fuel_consumption)
        a_dt = self._calculate_parameter(a*dt, name="fuel_consumption_dt")
        
        # Derived parameters
        z0 = cp.Parameter(n, name="z0")
        exp_z0 = cp.Parameter(n, name="exp_z0", nonneg=True)
        max_exp = cp.Parameter(n, name="max_exp_z0", nonneg=True)
        min_exp = cp.Parameter(n, name="min_exp_z0", nonneg=True)
        
        # Calculate values
        c_z0 = []
        c_exp_z0 = []
        c_max_exp = []
        c_min_exp = []

        for i in range(n):
            z00 = np.log(config.spacecraft.wet_mass - a.value*dt.value*max_t.value*i)
            c_z0.append(z00)
            c_exp_z0.append(np.exp(-z00))
            c_max_exp.append(1/(np.exp(-z00) * max_t.value))
            if min_t.value != 0:
                c_min_exp.append(1/(np.exp(-z00) * min_t.value))

        z0.value = c_z0
        exp_z0.value = c_exp_z0
        max_exp.value = c_max_exp
        min_exp.value = c_min_exp

        # More parameters
        max_angle = cp.Parameter(name="max_angle", value=config.environment.cos_max_angle)
        dt_squared = self._calculate_parameter(dt**2, name="dt_squared")
        initial_pos = cp.Parameter(3, name="initial_position", value=config.spacecraft.initial_position)
        initial_vel = cp.Parameter(3, name="initial_vel", value=config.spacecraft.initial_velocity)
        target_vel = cp.Parameter(3, name="target_velocity", value=config.spacecraft.target_velocity)
        g = cp.Parameter(3, name="gravity", value=config.environment.gravity)
        g_dt = self._calculate_parameter(g*dt, 3, name="gravity_dt")
        g_dt_sq = self._calculate_parameter(g*dt*dt, 3, name="gravity_dt_squared")
        
        # Store parameters
        self.parameters = {
            "log_mass": log_mass,
            "max_vel": max_vel,
            "sin_glide_slope": sin_glide_slope,
            "log_dry_mass": log_dry_mass,
            "min_thrust": min_t,
            "max_thrust": max_t,
            "dt": dt,
            "fuel_consumption": a,
            "fuel_consumption_dt": a_dt,
            "z0": z0,
            "exp_z0": exp_z0,
            "max_exp_z0": max_exp,
            "min_exp_z0": min_exp,
            "max_angle": max_angle,
            "dt_squared": dt_squared,
            "initial_position": initial_pos,
            "initial_vel": initial_vel,
            "target_velocity": target_vel,
            "gravity": g,
            "gravity_dt": g_dt,
            "gravity_dt_squared": g_dt_sq,
        }
        
        # Constraints
        constraints = [
            x[0, :3] == initial_pos,
            x[0, 3:] == initial_vel,
            z[0] == log_mass,
        ]

        # Timestep constraints
        for i in range(n):
            constraints.append(cp.norm(x[i, 3:]) <= max_vel)  # never exceed the maximum velocity
            constraints.append(x[i, 2] >= cp.norm(x[i, :3]) * sin_glide_slope)  # glide slope constraint
            constraints.append(s[i] >= cp.norm(u[i, :]))  # |u| = s
            constraints.append((1 - (z[i]-z0[i]) + cp.square(z[i]-z0[i])/2) <= s[i] * min_exp[i])
            constraints.append(s[i] * max_exp[i] <= (1 - (z[i]-z0[i])))  # upper bound for s 
            if i != n - 1:
                acc = (u[i+1, :] + u[i, :])/2
                constraints += [
                    x[i+1, :3] == x[i, :3] + (x[i, 3:] + x[i+1, 3:]) * dt / 2 + (acc*dt_squared+g_dt_sq) * (1/2),  # position update
                    x[i+1, 3:] == x[i, 3:] + acc*dt + g_dt,  # velocity update
                    z[i+1] == z[i] - (s[i] + s[i+1]) / 2 * a_dt  # mass update
                ]

        # Constraints on the last step
        constraints += [
            x[n-1, :3] == config.spacecraft.target_position,  # landing site
            x[n-1, 3:] == target_vel,
            z[n-1] >= log_dry_mass,
        ]
        
        self.constraints = constraints
        
        # Objective: maximize final mass
        obj = cp.Maximize(z[n-1])
        self.problem = cp.Problem(obj, constraints)
        
    def solve(self, verbose=False):
        """
        Solve the G-FOLD optimization problem.
        
        Args:
            verbose (bool): Whether to print verbose output
            
        Returns:
            dict: Solution containing positions, velocities, thrusts, and other data
        """
        if not self.problem:
            raise ValueError("Problem not initialized properly")
            
        solution_val = self.problem.solve(verbose=verbose)
        
        # Extract solution data
        x_val = self.variables["x"].value
        u_val = self.variables["u"].value
        z_val = self.variables["z"].value
        s_val = self.variables["s"].value
        
        positions = x_val[:, :3]
        velocities = x_val[:, 3:]
        thrusts = np.array([np.linalg.norm(u) for u in u_val])
        
        # Adjust thrust for mass
        for i in range(self.config.solver.n):
            thrusts[i] *= np.exp(z_val[i])
        
        return {
            "solution_value": solution_val,
            "positions": positions,
            "velocities": velocities,
            "thrusts": thrusts,
            "normalized_thrusts": thrusts / self.config.spacecraft.real_max_thrust,
            "final_mass": np.exp(z_val[-1]),
            "z_values": z_val,
            "x_values": x_val,
            "u_values": u_val,
            "s_values": s_val,
            "time_points": np.arange(0, self.parameters["dt"].value * self.config.solver.n, self.parameters["dt"].value)
        }
        
    def generate_code(self, code_dir="code"):
        """
        Generate C++/Python code for the solver using cvxpygen.
        
        Args:
            code_dir (str): Directory to save the generated code
            
        Returns:
            str: Path to the generated code
        """
        if not self.problem:
            raise ValueError("Problem not initialized properly")
            
        # Create directory if it doesn't exist
        os.makedirs(code_dir, exist_ok=True)
        
        # Generate code
        cpg.generate_code(self.problem, code_dir=code_dir, solver=cp.CLARABEL, wrapper=False)
        return code_dir

    def update_parameter(self, param_name, new_value):
        """
        Update a parameter value.
        
        Args:
            param_name (str): Name of the parameter to update
            new_value: New value for the parameter
        """
        if param_name not in self.parameters:
            raise ValueError(f"Parameter {param_name} not found")
            
        self.parameters[param_name].value = new_value
    
    def update_config(self, **kwargs):
        """
        Update configuration parameters and rebuild the problem.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self.config._process_kwargs(kwargs)
        
        # Rebuild the problem with updated configuration
        self._setup_problem()
