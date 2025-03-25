import numpy as np

class SpacecraftConfig:
    """Configuration for spacecraft parameters."""
    
    def __init__(self, **kwargs):
        """
        Initialize spacecraft configuration with default values.
        
        Args:
            **kwargs: Any parameter can be overridden during initialization
        """
        # Mass parameters
        self.wet_mass = 2000  # wet mass of the rocket (kg)
        self.fuel = 1700  # weight of fuel (kg)
        
        # Thrust parameters
        self.real_max_thrust = 24000  # maximum possible thrust (N)
        self.min_thrust_pct = 0.2  # percentage of max thrust for minimum
        self.max_thrust_pct = 0.8  # percentage of max thrust for maximum
        
        # Motion constraints
        self.max_velocity = 1000  # maximum velocity (m/s)
        
        # Initial conditions
        self.initial_position = [450, -330, 2400]  # (m)
        self.initial_velocity = [-40, 10, -10]  # (m/s)
        self.target_velocity = [0, 0, 0]  # (m/s)
        self.target_position = [0, 0, 0]  # (m)
        
        # Physics parameters
        self.fuel_consumption = 5e-4  # fuel consumption rate (kg/N/s)
        
        self._update_from_kwargs(kwargs)
    
    def _update_from_kwargs(self, kwargs):
        """Update attributes from keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown spacecraft parameter: {key}")
    
    @property
    def log_wet_mass(self):
        """Natural logarithm of wet mass."""
        return np.log(self.wet_mass)
    
    @property
    def log_dry_mass(self):
        """Natural logarithm of dry mass."""
        return np.log(self.wet_mass - self.fuel)
    
    @property
    def min_thrust(self):
        """Minimum thrust value."""
        return self.real_max_thrust * self.min_thrust_pct
    
    @property
    def max_thrust(self):
        """Maximum thrust value."""
        return self.real_max_thrust * self.max_thrust_pct
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class EnvironmentConfig:
    """Configuration for environment parameters."""
    
    def __init__(self, **kwargs):
        """
        Initialize environment configuration with default Mars values.
        
        Args:
            **kwargs: Any parameter can be overridden during initialization
        """
        # Gravitational parameters
        self.gravity = [0, 0, -3.71]  # gravity vector (m/s²), default is Mars
        
        # Landing constraints
        self.glide_slope_angle = 0  # in degrees
        self.max_angle = 90  # maximum angle for approach (degrees)
        
        self._update_from_kwargs(kwargs)
    
    def _update_from_kwargs(self, kwargs):
        """Update attributes from keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown environment parameter: {key}")
    
    @property
    def sin_glide_slope(self):
        """Sine of the glide slope angle."""
        return np.sin(np.radians(self.glide_slope_angle))
    
    @property
    def cos_max_angle(self):
        """Cosine of the maximum angle."""
        return np.cos(np.radians(self.max_angle))
    
    @classmethod
    def mars(cls):
        """Create Mars environment configuration."""
        return cls()
    
    @classmethod
    def moon(cls):
        """Create Moon environment configuration."""
        return cls(gravity=[0, 0, -1.62])
    
    @classmethod
    def earth(cls):
        """Create Earth environment configuration."""
        return cls(gravity=[0, 0, -9.81])
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class SolverConfig:
    """Configuration for solver parameters."""
    
    def __init__(self, **kwargs):
        """
        Initialize solver configuration with default values.
        
        Args:
            **kwargs: Any parameter can be overridden during initialization
        """
        # Solver parameters
        self.n = 100  # number of discrete timesteps
        self.time_of_flight = 44.63  # pre-calculated time of flight (s)
        
        self._update_from_kwargs(kwargs)
    
    def _update_from_kwargs(self, kwargs):
        """Update attributes from keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown solver parameter: {key}")
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class GFoldConfig:
    """Main configuration class for G-FOLD solver parameters."""
    
    def __init__(self, spacecraft=None, environment=None, solver=None, **kwargs):
        """
        Initialize configuration with default values that can be overridden.
        
        Args:
            spacecraft (SpacecraftConfig): Spacecraft configuration
            environment (EnvironmentConfig): Environment configuration
            solver (SolverConfig): Solver configuration
            **kwargs: Any parameter can be overridden during initialization
        """
        # Initialize sub-configurations
        self.spacecraft = spacecraft if spacecraft is not None else SpacecraftConfig()
        self.environment = environment if environment is not None else EnvironmentConfig.mars()
        self.solver = solver if solver is not None else SolverConfig()
        
        # Process any remaining kwargs by trying to assign them to the appropriate sub-config
        self._process_kwargs(kwargs)
    
    def _process_kwargs(self, kwargs):
        """Process keyword arguments and assign to appropriate sub-config."""
        for key, value in kwargs.items():
            assigned = False
            
            for config_name in ['spacecraft', 'environment', 'solver']:
                config = getattr(self, config_name)
                print(config.to_dict())
                if hasattr(config, key):
                    setattr(config, key, value)
                    assigned = True
                    break
            
            if not assigned:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    # Properties to maintain backward compatibility
    @property
    def log_wet_mass(self):
        return self.spacecraft.log_wet_mass
    
    @property
    def log_dry_mass(self):
        return self.spacecraft.log_dry_mass
    
    @property
    def sin_glide_slope(self):
        return self.environment.sin_glide_slope
    
    @property
    def cos_max_angle(self):
        return self.environment.cos_max_angle
    
    @property
    def min_thrust(self):
        return self.spacecraft.min_thrust
    
    @property
    def max_thrust(self):
        return self.spacecraft.max_thrust
    
    @property
    def time_of_flight(self):
        return self.solver.time_of_flight
    
    @property
    def wet_mass(self):
        return self.spacecraft.wet_mass
    
    @property
    def real_max_thrust(self):
        return self.spacecraft.real_max_thrust
    
    def to_dict(self):
        """Convert all configuration to a flat dictionary."""
        result = {}
        for config_name in ['spacecraft', 'environment', 'solver']:
            config = getattr(self, config_name)
            result.update(config.to_dict())
        return result
