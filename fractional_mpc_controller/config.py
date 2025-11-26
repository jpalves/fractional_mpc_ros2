"""
Configuration and weight matrix utilities for MPC.

Centralizes configuration handling and weight matrix construction
to eliminate duplication and improve maintainability.
"""

import numpy as np


# ==============================================================================
# Constants
# ==============================================================================

# Mathematical constants
ANGLE_MAX = np.pi  # Maximum joint angle (radians)
ANGLE_MIN = -np.pi  # Minimum joint angle (radians)

# Default control limits
DEFAULT_CONTROL_LIMITS_MIN = -10.0
DEFAULT_CONTROL_LIMITS_MAX = 10.0
DEFAULT_CONTROL_LIMITS = (DEFAULT_CONTROL_LIMITS_MIN, DEFAULT_CONTROL_LIMITS_MAX)

# Default velocity limits
DEFAULT_VELOCITY_MAX = 100.0

# Default integral error limits
DEFAULT_INTEGRAL_ERROR_MAX = 10.0

# Velocity estimation filter coefficient (low-pass filter)
VELOCITY_ESTIMATE_ALPHA = 0.7

# Default MPC parameters
DEFAULT_FRACTIONAL_ORDER = 0.80
DEFAULT_HORIZON = 25
DEFAULT_CONTROL_DT = 0.01
DEFAULT_SPRING_CONSTANT = 2.0
DEFAULT_DAMPING = 0.5
DEFAULT_INTEGRAL_GAIN = 1.0
DEFAULT_MASS = 1.0


# ==============================================================================
# Weight Matrix Construction
# ==============================================================================

class MPCWeights:
    """
    Encapsulates MPC cost function weight matrix construction.

    The MPC cost function is:
        J = ||x_pred - x_ref||²_Q + λ||u||²_R

    Where x = [pos₁, ..., posₙ, vel₁, ..., velₙ, e_int₁, ..., e_intₙ]
    and Q = diag([Q_pos, Q_pos, ..., Q_vel, Q_vel, ..., Q_int, Q_int, ...])
    """

    @staticmethod
    def create_diagonal(q_pos, q_vel, q_int, n_joints):
        """
        Create diagonal cost matrix Q from scalar weights.

        Args:
            q_pos: position cost weight (scalar)
            q_vel: velocity cost weight (scalar)
            q_int: integral error cost weight (scalar)
            n_joints: number of joints

        Returns:
            Q: (3n × 3n) diagonal matrix with block structure:
               [[q_pos*I  0      0    ]
                [0     q_vel*I   0    ]
                [0        0    q_int*I]]

        Example:
            >>> Q = MPCWeights.create_diagonal(650, 13, 50, n_joints=3)
            >>> Q.shape
            (9, 9)
            >>> np.diag(Q)[0:3]  # Position weights
            array([650, 650, 650])
        """
        n = n_joints

        # Create diagonal elements: [pos_weights | vel_weights | int_weights]
        diag_elements = (
            [float(q_pos)] * n +     # Position cost weights
            [float(q_vel)] * n +     # Velocity cost weights
            [float(q_int)] * n       # Integral error cost weights
        )

        return np.diag(diag_elements)

    @staticmethod
    def create_from_dict(config_dict, n_joints):
        """
        Create Q matrix from configuration dictionary.

        Args:
            config_dict: dictionary with keys:
                - 'state_cost_position': q_pos (default: 650.0)
                - 'state_cost_velocity': q_vel (default: 13.0)
                - 'state_cost_integral': q_int (default: 50.0)
            n_joints: number of joints

        Returns:
            Q: diagonal weight matrix

        Example:
            >>> config = {
            ...     'state_cost_position': 600,
            ...     'state_cost_velocity': 12,
            ...     'state_cost_integral': 50
            ... }
            >>> Q = MPCWeights.create_from_dict(config, n_joints=3)
        """
        q_pos = config_dict.get('state_cost_position', 650.0)
        q_vel = config_dict.get('state_cost_velocity', 13.0)
        q_int = config_dict.get('state_cost_integral', 50.0)

        return MPCWeights.create_diagonal(q_pos, q_vel, q_int, n_joints)

    @staticmethod
    def create_control_cost(r, n_joints):
        """
        Create control cost matrix R.

        Args:
            r: control cost weight (scalar)
            n_joints: number of joints

        Returns:
            R: (n × n) diagonal matrix = r * I

        Example:
            >>> R = MPCWeights.create_control_cost(0.15, n_joints=3)
            >>> R.shape
            (3, 3)
        """
        return float(r) * np.eye(n_joints)


# ==============================================================================
# Configuration Loader
# ==============================================================================

class MPCConfig:
    """
    Configuration container for MPC solver.

    Provides unified interface for loading and managing MPC parameters.
    """

    def __init__(self, **kwargs):
        """
        Initialize configuration with keyword arguments.

        Typical parameters:
            - n_joints: number of joints
            - fractional_order: fractional order alpha (0 < α ≤ 1)
            - control_dt: sampling time (seconds)
            - horizon: prediction horizon (steps)
            - state_cost_position: Q_pos weight
            - state_cost_velocity: Q_vel weight
            - state_cost_integral: Q_int weight
            - control_cost: R weight
            - u_min: minimum control limits
            - u_max: maximum control limits
        """
        self.n_joints = kwargs.get('n_joints', 9)
        self.fractional_order = kwargs.get('fractional_order', DEFAULT_FRACTIONAL_ORDER)
        self.control_dt = kwargs.get('control_dt', DEFAULT_CONTROL_DT)
        self.horizon = kwargs.get('horizon', DEFAULT_HORIZON)

        # Cost function weights
        self.q_pos = kwargs.get('state_cost_position', 650.0)
        self.q_vel = kwargs.get('state_cost_velocity', 13.0)
        self.q_int = kwargs.get('state_cost_integral', 50.0)
        self.r = kwargs.get('control_cost', 0.15)

        # Control limits
        u_min = kwargs.get('u_min', None)
        u_max = kwargs.get('u_max', None)

        if u_min is None:
            self.u_min = np.full(self.n_joints, DEFAULT_CONTROL_LIMITS_MIN)
        else:
            self.u_min = np.asarray(u_min).flatten()

        if u_max is None:
            self.u_max = np.full(self.n_joints, DEFAULT_CONTROL_LIMITS_MAX)
        else:
            self.u_max = np.asarray(u_max).flatten()

        # System parameters
        self.spring_constant = kwargs.get('spring_constant', DEFAULT_SPRING_CONSTANT)
        self.damping = kwargs.get('damping', DEFAULT_DAMPING)
        self.integral_gain = kwargs.get('integral_gain', DEFAULT_INTEGRAL_GAIN)
        self.mass = kwargs.get('mass', DEFAULT_MASS)

    def get_Q_matrix(self):
        """Get state cost matrix Q"""
        return MPCWeights.create_diagonal(self.q_pos, self.q_vel, self.q_int, self.n_joints)

    def get_R_matrix(self):
        """Get control cost matrix R"""
        return MPCWeights.create_control_cost(self.r, self.n_joints)

    def __repr__(self):
        return (
            f"MPCConfig(n_joints={self.n_joints}, alpha={self.fractional_order}, "
            f"horizon={self.horizon}, dt={self.control_dt})"
        )
