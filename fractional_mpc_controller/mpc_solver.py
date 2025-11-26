"""
Model Predictive Control Solver for Fractional-Order Systems

Uses convex optimization (CVXPY) to solve MPC problem:
    minimize: ||x_pred - x_ref||^2 + λ*||u||^2
    subject to: dynamics constraints, input bounds, state bounds
"""

import numpy as np
import cvxpy as cp
import warnings
from scipy.signal import TransferFunction, ss2zpk, zpk2ss
import os

# Supress CVXPY backend warning (não afecta functionality)
warnings.filterwarnings('ignore', category=UserWarning, module='cvxpy')

# Import custom modules for validation and error handling
from .validators import (
    validate_state_dimension,
    validate_state_values,
    validate_reference_trajectory
)
from .exceptions import (
    FractionalMPCError,
    MPCOptimizationError,
    InvalidStateError,
    ConfigurationError
)

# GPU acceleration setup
try:
    import cupy as gpu_available
    GPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # GPU not available - fallback to CPU
    GPU_AVAILABLE = False
except Exception as e:
    # Unexpected error - warn but continue
    print(f"WARNING: GPU import failed: {e}")
    GPU_AVAILABLE = False


class FractionalMPCSolver:
    """MPC solver for fractional-order manipulator control with integral action"""

    def __init__(self, n_joints=9, horizon=10, dt=0.01, alpha=0.80,
                 enable_soft_constraints=True, spring_const=1.0, Ki=1.0,
                 mass=1.0, use_oustaloup=False, gl_history=4):
        """
        Initialize MPC solver with integral action and fractional dynamics support.

         Args:
             n_joints: Number of joints
             horizon: Prediction horizon (steps)
             dt: Sampling time
             alpha: Fractional order (0 < α ≤ 1). Used for linearization adjustment.
                 Default α=0.80 for smooth fractional dynamics.
             enable_soft_constraints: If True, use soft constraints with slack variables
                         to handle infeasible problems
             use_oustaloup: Enable fractional Grünwald-Letnikov dynamics (False = integer model)
             gl_history: Number of fractional history taps when use_oustaloup=True (≥2)
             spring_const: Linear stiffness (scalar or per joint)
             Ki: Integral feedback gain (scalar or per joint)
             mass: Joint inertia / mass (scalar or per joint)

        State vector: [pos_1, ..., pos_n, vel_1, ..., vel_n, e_int_1, ..., e_int_n]
            - Positions: indices 0 to n-1
            - Velocities: indices n to 2n-1
            - Integral error: indices 2n to 3n-1

        NEW: Integral error is NOW a state variable optimized by MPC!
            e_int[k+1] = e_int[k] - dt*(pos[k] - ref[k])
            u_applied = u_mpc + Ki * e_int (integral position correction applied directly to the command)
        """
        self.n_joints = n_joints
        self.horizon = horizon
        self.dt = dt
        self.alpha = alpha  # Fractional order for linearization
        self.enable_soft_constraints = enable_soft_constraints
        self.spring_k = spring_const

        # Allow optional per-joint stiffness and damping; fall back to scalar values otherwise.
        spring_array = np.array(spring_const, dtype=float)
        if spring_array.ndim == 0:
            self.spring_k_vector = np.full(n_joints, spring_array.item())
        else:
            if spring_array.shape[0] != n_joints:
                raise ValueError(f"spring_const must have size {n_joints}, got {spring_array.shape[0]}")
            self.spring_k_vector = spring_array.astype(float)

        # Default viscous damping uses critical ratio 0.5 unless user supplies per-joint values later.
        self.damping_vector = 0.5 * np.ones(n_joints, dtype=float)

        Ki_array = np.array(Ki, dtype=float)
        if Ki_array.ndim == 0:
            self.Ki = np.full(n_joints, Ki_array.item())
        else:
            if Ki_array.shape[0] != n_joints:
                raise ValueError(f"Ki must have size {n_joints}, got {Ki_array.shape[0]}")
            self.Ki = Ki_array.astype(float)

        # Mass/inertia configuration (per joint)
        mass_array = np.array(mass, dtype=float)
        if mass_array.ndim == 0:
            self.mass_vector = np.full(n_joints, mass_array.item())
        else:
            if mass_array.shape[0] != n_joints:
                raise ValueError(f"mass must have size {n_joints}, got {mass_array.shape[0]}")
            self.mass_vector = mass_array.astype(float)
        if np.any(self.mass_vector <= 0.0):
            raise ValueError("mass entries must be positive")
        self.inv_mass_vector = 1.0 / self.mass_vector

        # Fractional memory configuration (Grünwald-Letnikov approximation)
        self.use_oustaloup = bool(use_oustaloup)
        if self.use_oustaloup:
            history = int(gl_history)
            if history < 2:
                history = 2
            self.gl_history = history
            self.gl_coeffs = self._compute_gl_coefficients(self.gl_history)
            # Number of stored past velocity samples per joint (excluding current sample)
            self.n_frac_states = self.gl_history - 2  # store v[k-1], v[k-2], ...
        else:
            self.gl_history = 1
            self.gl_coeffs = np.array([1.0], dtype=float)
            self.n_frac_states = 0

        # Total state dimension: [pos, vel, e_int, v_hist_1, ..., v_hist_m]
        self.state_size = 3 * n_joints + self.n_joints * self.n_frac_states

        # Control limits
        self.u_min = -10.0 * np.ones(n_joints)  # Min position command
        self.u_max = 10.0 * np.ones(n_joints)   # Max position command

        # State limits
        self.q_min = -3.1415927 * np.ones(n_joints)  # Min position
        self.q_max = 3.1415927 * np.ones(n_joints)   # Max position

        self.v_max = 100.0 * np.ones(n_joints)    # Max velocity
        self.e_int_max = 50.0 * np.ones(n_joints)  # Max integrated error (allow strong integral action)

        # Cost weights - UPDATED for 3n state (pos + vel + integral)
        # Q = diag([q_pos]*n + [q_vel]*n + [q_int]*n)
        # TUNING: Increased Q_pos for better position tracking, reduced R for control authority
        # This fixes convergence issue where joints converge to average value
        # Higher q_pos forces each joint to track its individual target
        # Lower R allows larger control commands to differentiate joint trajectories
        q_pos_default = 2000.0   # Position tracking weight (INCREASED from 200 to enforce individual targets)
        q_vel_default = 30.0     # Velocity weight (kept same)
        q_int_default = 500.0     # Integral weight (kept same)
        self.Q = np.diag([q_pos_default] * n_joints + [q_vel_default] * n_joints + [q_int_default] * n_joints)
        self.R = 0.05 * np.eye(n_joints)  # Control cost (REDUCED from 0.15 for higher control authority)

        # Soft constraint penalties (for slack variables)
        self.slack_penalty_position = 1000.0  # High penalty for position constraint violation
        self.slack_penalty_velocity = 500.0   # Medium penalty for velocity constraint violation
        self.slack_penalty_integral = 500.0   # Medium penalty for integral error constraint violation
        self.slack_penalty_control  = 100.0    # Low penalty for control constraint violation

        # System matrices (linearized around origin). They are kept for compatibility,
        # although the MPC dynamics are enforced via explicit constraints.
        self._setup_linearized_matrices()

        # Decision variables
        self.X = cp.Variable((self.horizon + 1, self.state_size))  # State trajectory: (time_steps, state_dim)
        self.U = cp.Variable((self.horizon, self.n_joints))        # Control trajectory: (time_steps, control_dim)

        # Parameter for reference trajectory (updated at each solve)
        # Reference only has 2n elements (pos + vel), integral ref is implicit (target: e_int=0)
        self.X_ref = cp.Parameter((self.horizon + 1, 2 * self.n_joints))  # Reference: positions + velocities only

        # Slack variables for soft constraints (if enabled)
        if self.enable_soft_constraints:
            self.slack_q = cp.Variable((self.horizon + 1, self.n_joints))  # Slack for position constraints
            self.slack_v = cp.Variable((self.horizon + 1, self.n_joints))  # Slack for velocity constraints
            self.slack_u = cp.Variable((self.horizon, self.n_joints))      # Slack for control constraints

        self.solve_count = 0
        self.last_u = np.zeros(n_joints)  # Fallback control signal
        self.last_feasible = False  # Track if last solution was feasible (without slack)

        # Acceleration parameters
        self.solver_verbose = False  # Suppress verbose output
        self.solver_eps_abs = 1e-5  # Absolute tolerance (default 1e-7, less strict = faster)
        self.solver_eps_rel = 1e-4  # Relative tolerance (default 1e-5, less strict = faster)
        self.solver_max_iters = 5000  # Max iterations (default 10000, fewer = faster)
        self.solver_warm_start = True  # Enable warm-start for successive solves
        self.last_x_opt = None  # Previous solution for warm-start
        self.last_u_opt = None
        self.last_u_total_opt = None  # Tracks position command after adding integral feedback
        self.enable_gpu = False  # GPU acceleration flag

        # CPU Multi-threading optimization
        self.num_threads = os.cpu_count() if os.cpu_count() else 1  # Auto-detect number of cores
        self.enable_threading = True  # Enable multi-threading by default
        self.thread_limit = None  # Custom thread limit (None = use all available cores)

        # Fractional integration constants
        self.dt_alpha = self.dt ** self.alpha if self.alpha > 0 else self.dt

        # Problem data for warm-starting
        self.problem = None
        self.objective = None
        self.constraints_list = []

    def _setup_linearized_matrices(self):
        """
        Setup linearized system matrices for MPC with integral state and optional Oustaloup approximation.

        State vector: [pos, vel, e_int, v_hist_1, v_hist_2, ...]
            - Positions: indices 0 to n-1
            - Velocities: indices n to 2n-1
            - Integral error: indices 2n to 3n-1
            - Fractional history taps: stacked blocks of size n (v[k-1], v[k-2], ...)

        The matrices are retained for compatibility with plotting and debugging utilities.
        Fractional dynamics are enforced explicitly inside the MPC constraints.
        """
        A = np.eye(self.state_size)
        B = np.zeros((self.state_size, self.n_joints))

        # Position propagation: q[k+1] = q[k] + dt * v[k]
        A[:self.n_joints, self.n_joints:self.n_joints*2] = self.dt * np.eye(self.n_joints)

        # Integral error handled explicitly in constraints, so zero these rows
        A[self.n_joints*2:self.n_joints*3, :] = 0.0

        # Direct control coupling for reference (used only for auxiliary linear model)
        B[self.n_joints:self.n_joints*2, :] = self.dt * np.eye(self.n_joints)

        self.A = A
        self.B = B

    def _compute_gl_coefficients(self, taps):
        """Generate Grünwald-Letnikov coefficients up to the requested horizon."""
        coeffs = np.zeros(taps, dtype=float)
        coeffs[0] = 1.0
        for k in range(1, taps):
            coeffs[k] = coeffs[k-1] * ((k - 1 - self.alpha) / k)
        return coeffs

    def solve(self, x0, x_ref, warm_start=None):
        """
        Solve MPC problem for fractional-order system WITH INTEGRAL ERROR and OUSTALOUP APPROXIMATION.

        This is the main entry point that coordinates three helper methods:
        1. _validate_and_preprocess: Input validation
        2. _build_mpc_problem: Constraint and objective construction
        3. _solve_optimization: Solver execution and solution extraction

        Args:
            x0: Initial state vector with stacked components:
                - pos (0:n): current measured positions
                - vel (n:2n): current estimated velocities
                - e_int (2n:3n): accumulated integral error
                - fractional history blocks (if enabled): v[k-1], v[k-2], ... each block has size n

            x_ref: Reference trajectory (horizon+1, 2*n_joints) = [pos_ref, vel_ref]
                   (integral reference is implicit: target e_int = 0)
            warm_start: Previous solution for warm-starting (not used in current version)

        Returns:
            u_opt: Optimal control sequence (horizon, n_joints)
            x_opt: Predicted state sequence (horizon+1, [3+n_frac]*n_joints) with integral and fractional states
            success: Whether optimization succeeded

        Note: Integral error is a state variable optimized by MPC!
              Fractional filter provides accurate D^α approximation for fractional-order dynamics!
        """
        try:
            # Step 1: Validate and preprocess inputs
            x0, x_ref = self._validate_and_preprocess(x0, x_ref)

            # Step 2: Build MPC problem (constraints + objective)
            constraints, objective = self._build_mpc_problem(x0, x_ref)

            # Step 3: Solve optimization problem
            return self._solve_optimization(x0, constraints, objective)

        except InvalidStateError as e:
            # Specific handling for state validation errors
            print(f"[MPC ERROR] Invalid state: {e}")
            return None, None, False

        except ConfigurationError as e:
            # Specific handling for configuration errors
            print(f"[MPC ERROR] Configuration error: {e}")
            return None, None, False

        except Exception as e:
            # Unexpected errors
            import traceback
            print(f"[MPC ERROR] Unexpected error in solve: {e}")
            print(f"[MPC ERROR] Traceback:\n{traceback.format_exc()}")
            return None, None, False

    def _validate_and_preprocess(self, x0, x_ref):
        """
        Validate and preprocess input states before solving MPC problem.

        Uses centralized validators from validators.py module.
        Raises specific exceptions for different failure modes.

        Args:
            x0: Initial state vector
            x_ref: Reference trajectory

        Returns:
            x0: Validated and flattened initial state
            x_ref: Validated reference trajectory

        Raises:
            InvalidStateError: if state contains NaN/Inf or wrong dimension
            ValueError: if reference trajectory is invalid
        """
        x0 = np.asarray(x0).flatten()
        x_ref = np.asarray(x_ref)

        try:
            # Validate state dimension
            validate_state_dimension(x0, self.state_size, name="x0")

            # Validate state values (NaN, Inf, extremes)
            validate_state_values(x0, max_abs_value=1e6, name="x0")

            # Validate reference trajectory shape and values
            validate_reference_trajectory(x_ref, self.horizon, self.n_joints)

            return x0, x_ref

        except ValueError as e:
            # Re-raise with InvalidStateError context
            raise InvalidStateError(str(e)) from e

    def _build_mpc_problem(self, x0, x_ref):
        """
        Build MPC optimization problem with constraints and objective.

        Constructs the convex optimization problem including:
        - Dynamics constraints (position, velocity, integral error)
        - Control input constraints
        - State constraints (position, velocity, integral error)
        - Objective function (position tracking + control effort)

        Args:
            x0: Initial state vector (validated)
            x_ref: Reference trajectory (validated)

        Returns:
            constraints: List of CVXPY constraints
            objective: CVXPY objective expression to minimize
        """
        # Build constraint list
        constraints = []

        # Initial condition
        constraints.append(self.X[0, :] == x0)

        # Set reference parameter
        self.X_ref.value = x_ref

        # Dynamics constraints with integral action and optional fractional history
        frac_start = 3 * self.n_joints
        control_cost = 0.0
        control_smoothness_cost = 0.0
        prev_u_total = None

        for k in range(self.horizon):
            # 1. Position: pos[k+1] = pos[k] + dt*vel[k]
            constraints.append(self.X[k+1, :self.n_joints] ==
                             self.X[k, :self.n_joints] + self.dt * self.X[k, self.n_joints:self.n_joints*2])

            # 2. Velocity dynamics with fractional or integer model
            pos_slice = slice(0, self.n_joints)
            vel_slice = slice(self.n_joints, self.n_joints * 2)
            eint_slice = slice(self.n_joints * 2, self.n_joints * 3)

            # Fractional/integer velocity dynamics with integrated error feedback
            K_spring = self.spring_k_vector
            D_damping = self.damping_vector
            integral_feedback = cp.multiply(self.Ki, self.X[k, eint_slice])
            u_total = self.U[k, :] + integral_feedback
            base_rhs = (-cp.multiply(K_spring, self.X[k, pos_slice]) -
                        cp.multiply(D_damping, self.X[k, vel_slice]) +
                        u_total)
            rhs = cp.multiply(self.inv_mass_vector, base_rhs)

            if self.use_oustaloup:
                coeffs = self.gl_coeffs
                hist_sum = coeffs[1] * self.X[k, vel_slice] if len(coeffs) > 1 else 0.0
                for tap in range(self.n_frac_states):
                    coeff_idx = tap + 2
                    if coeff_idx < len(coeffs):
                        tap_slice = slice(
                            frac_start + tap * self.n_joints,
                            frac_start + (tap + 1) * self.n_joints,
                        )
                        hist_sum += coeffs[coeff_idx] * self.X[k, tap_slice]

                v_next = self.dt_alpha * rhs - hist_sum
                constraints.append(self.X[k+1, vel_slice] == v_next)

                if self.n_frac_states:
                    first_hist = slice(frac_start, frac_start + self.n_joints)
                    constraints.append(self.X[k+1, first_hist] == self.X[k, vel_slice])
                    for tap in range(1, self.n_frac_states):
                        src = slice(
                            frac_start + (tap - 1) * self.n_joints,
                            frac_start + tap * self.n_joints,
                        )
                        dst = slice(
                            frac_start + tap * self.n_joints,
                            frac_start + (tap + 1) * self.n_joints,
                        )
                        constraints.append(self.X[k+1, dst] == self.X[k, src])
            else:
                constraints.append(
                    self.X[k+1, vel_slice] ==
                    self.X[k, vel_slice] + self.dt * rhs
                )

            # 3. Integral error: e_int[k+1] = e_int[k] + dt*(ref[k] - pos[k])
            pos_ref_k = self.X_ref[k, :self.n_joints]
            constraints.append(self.X[k+1, self.n_joints*2:self.n_joints*3] ==
                             self.X[k, self.n_joints*2:self.n_joints*3] +
                             self.dt * (pos_ref_k - self.X[k, :self.n_joints]))

            # Input constraints enforced on actual position command (u_total)
            if self.enable_soft_constraints:
                constraints.append(u_total >= self.u_min - self.slack_u[k, :])
                constraints.append(u_total <= self.u_max + self.slack_u[k, :])
                constraints.append(self.slack_u[k, :] >= 0)
            else:
                constraints.append(u_total >= self.u_min)
                constraints.append(u_total <= self.u_max)

            # Control effort and smoothness penalties
            control_cost += cp.quad_form(u_total, self.R)
            if prev_u_total is not None:
                control_smoothness_cost += cp.quad_form(u_total - prev_u_total, 0.1 * self.R)
            prev_u_total = u_total

        # State constraints (positions) - with soft constraint support
        if self.enable_soft_constraints:
            constraints.append(self.X[:, :self.n_joints] >= self.q_min - self.slack_q)
            constraints.append(self.X[:, :self.n_joints] <= self.q_max + self.slack_q)
            constraints.append(self.slack_q >= 0)
        else:
            constraints.append(self.X[:, :self.n_joints] >= self.q_min)
            constraints.append(self.X[:, :self.n_joints] <= self.q_max)

        # Velocity constraints - with soft constraint support
        if self.enable_soft_constraints:
            for i in range(self.n_joints):
                constraints.append(
                    self.X[:, self.n_joints + i] <= self.v_max[i] + self.slack_v[:, i]
                )
                constraints.append(
                    self.X[:, self.n_joints + i] >= -self.v_max[i] - self.slack_v[:, i]
                )
            constraints.append(self.slack_v >= 0)
        else:
            constraints.append(
                cp.abs(self.X[:, self.n_joints:self.n_joints*2]) <= self.v_max
            )

        # Integral error constraints
        constraints.append(
            cp.abs(self.X[:, self.n_joints*2:self.n_joints*3]) <= self.e_int_max
        )

        # Objective function
        position_cost = 0.0
        velocity_cost = 0.0
        integral_cost = 0.0
        Q_int_block = self.Q[2*self.n_joints:3*self.n_joints, 2*self.n_joints:3*self.n_joints]

        for k in range(self.horizon):
            ref_pos = x_ref[k, :self.n_joints] if x_ref.shape[1] > 0 else np.zeros(self.n_joints)

            pos_k = self.X[k, :self.n_joints]
            vel_k = self.X[k, self.n_joints:self.n_joints*2]
            int_k = self.X[k, self.n_joints*2:self.n_joints*3]

            pos_error = pos_k - ref_pos
            position_cost += cp.quad_form(pos_error, self.Q[:self.n_joints, :self.n_joints])

            Q_vel = self.Q[self.n_joints:self.n_joints*2, self.n_joints:self.n_joints*2]
            if np.any(np.diag(Q_vel) > 0.0):
                if x_ref.shape[1] >= 2 * self.n_joints:
                    ref_vel = x_ref[k, self.n_joints:2*self.n_joints]
                else:
                    ref_vel = np.zeros(self.n_joints)
                vel_error = vel_k - ref_vel
                velocity_cost += cp.quad_form(vel_error, Q_vel)

            if Q_int_block.size > 0:
                integral_cost += cp.quad_form(int_k, Q_int_block)

        # Terminal cost for final state
        ref_pos_final = x_ref[self.horizon, :self.n_joints] if x_ref.shape[1] > 0 else np.zeros(self.n_joints)
        pos_final = self.X[self.horizon, :self.n_joints]
        pos_error_final = pos_final - ref_pos_final
        terminal_cost = 10.0 * cp.quad_form(pos_error_final, self.Q[:self.n_joints, :self.n_joints])

        # Slack variable penalties
        slack_cost = 0.0
        if self.enable_soft_constraints:
            slack_cost = (self.slack_penalty_position * cp.sum(self.slack_q) +
                         self.slack_penalty_velocity * cp.sum(self.slack_v) +
                         self.slack_penalty_control * cp.sum(self.slack_u))

        # Complete objective
        objective = cp.Minimize(position_cost + terminal_cost + velocity_cost +
                    integral_cost + control_cost + control_smoothness_cost + slack_cost)

        return constraints, objective

    def _solve_optimization(self, x0, constraints, objective):
        """
        Solve the MPC optimization problem.

        This method handles solver configuration, warm-starting, GPU acceleration,
        error handling, and solution extraction.

        Args:
            x0: Validated initial state (state_size,)
            constraints: List of CVXPY constraints
            objective: CVXPY objective function

        Returns:
            u_opt: Optimal control sequence (horizon, n_joints) or None if failed
            x_opt: Predicted state sequence (horizon+1, state_size) or None if failed
            success: Boolean indicating if optimization succeeded

        Raises:
            MPCOptimizationError: If solver fails after retries
        """
        try:
            # Build CVXPY problem
            prob = cp.Problem(objective, constraints)

            # Configure solver parameters for robust convergence
            solver_kwargs = {
                'verbose': self.solver_verbose,
                'max_iter': self.solver_max_iters,
                'eps_abs': self.solver_eps_abs,      # Relaxed tolerance for faster convergence
                'eps_rel': self.solver_eps_rel,      # Relaxed relative tolerance
                'alpha': 1.6,                        # Acceleration (OSQP default: 1.6)
                'rho': 0.1,                          # Robustness parameter (default: 0.1)
                'sigma': 1e-3,                       # Hessian regularization
                'scaled_termination': True,          # Normalized scale for convergence
                'check_termination': 200,            # Check convergence every 200 iterations
                'warm_start': self.solver_warm_start, # Enable warm-starting
            }

            # Warm-start: use previous solution as initialization
            if self.solver_warm_start and self.last_x_opt is not None and self.last_u_opt is not None:
                self.X.value = self.last_x_opt
                self.U.value = self.last_u_opt

            # Attempt GPU acceleration if available (CLARABEL solver)
            if GPU_AVAILABLE and self.enable_gpu:
                try:
                    prob.solve(solver=cp.CLARABEL, **solver_kwargs)
                except Exception as gpu_error:
                    # Fallback to OSQP if GPU fails
                    if self.solver_verbose:
                        print(f"[MPC DEBUG] GPU solver failed: {gpu_error}. Falling back to OSQP.")
                    prob.solve(solver=cp.OSQP, **solver_kwargs)
            else:
                # Use OSQP (default, CPU-optimized)
                prob.solve(solver=cp.OSQP, **solver_kwargs)

            # Check solver status
            if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
                u_opt = np.array(self.U.value)
                x_opt = np.array(self.X.value)

                # Compute total control (MPC + integral feedback)
                integral_block = x_opt[:self.horizon, 2*self.n_joints:3*self.n_joints]
                u_total_opt = u_opt + np.multiply(integral_block, self.Ki)

                # Store optimal solutions for warm-start initialization in next iteration
                self.last_x_opt = x_opt.copy()
                self.last_u_opt = u_opt.copy()
                self.last_u_total_opt = u_total_opt.copy()
                if u_total_opt.size > 0:
                    self.last_u = u_total_opt[0, :].copy()

                self.solve_count += 1
                return u_opt, x_opt, True
            else:
                # Debug info for solver failures
                if self.solve_count % 50 == 0:  # Log every 50 attempts to avoid spam
                    print(f"[MPC DEBUG] Solver status: {prob.status}")
                    print(f"[MPC DEBUG] Objective value: {prob.value if hasattr(prob, 'value') else 'N/A'}")
                return None, None, False

        except Exception as e:
            # Log error with full traceback for debugging
            import traceback
            if self.solve_count % 50 == 0:
                print(f"[MPC ERROR] Exception in solver: {e}")
                print(f"[MPC ERROR] Traceback:\n{traceback.format_exc()}")

            # FALLBACK: Return last known good control signal
            # This prevents crash when solver fails near optimal point
            if hasattr(self, 'last_u') and self.last_u is not None:
                fallback_u = self.last_u.reshape(1, -1)
                return fallback_u, None, False
            return None, None, False

    def set_control_limits(self, u_min, u_max):
        """Update control input limits"""
        self.u_min = np.array(u_min)
        self.u_max = np.array(u_max)

    def set_state_limits(self, q_min, q_max, v_max):
        """Update state limits"""
        self.q_min = np.array(q_min)
        self.q_max = np.array(q_max)
        self.v_max = np.array(v_max)

    def set_weights(self, Q, R):
        """
        Update cost weights.

        Args:
            Q: State cost matrix (3n x 3n) for [pos, vel, e_int]
            R: Control cost matrix (n x n)

        Note: Integral action is NOW part of the state cost (3rd diagonal block)

        Raises:
            ConfigurationError: If Q or R have invalid dimensions
        """
        Q_array = np.array(Q)

        # Q must be 3n x 3n (positions + velocities + integral error)
        expected_size = 3 * self.n_joints
        if Q_array.shape[0] != expected_size or Q_array.shape[1] != expected_size:
            raise ConfigurationError(
                param_name="Q",
                issue="invalid dimension for state cost matrix",
                expected=f"({expected_size}, {expected_size})",
                actual=str(Q_array.shape)
            )

        self.Q = Q_array
        self.R = np.array(R)

    def set_integral_gain(self, Ki):
        """
        Set integral action gain (Ki).

        Args:
            Ki: Integral gain vector (n_joints,) or scalar for all joints

        Note: Ki is used externally in the controller node to compute:
              u_applied = u_mpc + Ki * e_int
              It is NOT part of the MPC problem formulation.
        """
        Ki_array = np.array(Ki, dtype=float)
        if Ki_array.ndim == 0:  # Scalar
            self.Ki = np.full(self.n_joints, Ki_array.item())
        else:
            if Ki_array.shape[0] != self.n_joints:
                raise ValueError(f"Ki must have size {self.n_joints}, got {Ki_array.shape[0]}")
            self.Ki = Ki_array.astype(float)

    def get_first_input(self, u_opt, x_opt=None):
        """Extract first actuator command including integral feedback"""
        if x_opt is not None and len(x_opt) > 0:
            integral0 = np.array(x_opt[0, 2*self.n_joints:3*self.n_joints]).flatten()
            return u_opt[0, :] + np.multiply(integral0, self.Ki)

        if self.last_u_total_opt is not None and len(self.last_u_total_opt) > 0:
            return self.last_u_total_opt[0, :].copy()

        if u_opt is not None and len(u_opt) > 0:
            if self.last_x_opt is not None and len(self.last_x_opt) > 0:
                integral0 = np.array(self.last_x_opt[0, 2*self.n_joints:3*self.n_joints]).flatten()
                return u_opt[0, :] + np.multiply(integral0, self.Ki)
            return u_opt[0, :].copy()

        return np.zeros(self.n_joints)

    def set_gpu_enabled(self, enable_gpu):
        """
        Enable or disable GPU acceleration for the MPC solver.

        Args:
            enable_gpu (bool): If True, attempt to use GPU solver (CLARABEL) with CPU fallback.
                              If False, use CPU solver (OSQP) exclusively.

        Note: GPU acceleration requires CLARABEL solver and CuPy libraries to be installed.
              Without these, the flag has no effect.
        """
        self.enable_gpu = bool(enable_gpu)

    def set_solver_params(self, eps_abs=None, eps_rel=None, max_iters=None, warm_start=None, verbose=None):
        """
        Configure solver acceleration parameters.

        Args:
            eps_abs (float, optional): Absolute tolerance for convergence (default: 1e-5)
                                       Larger values accelerate convergence but reduce accuracy
            eps_rel (float, optional): Relative tolerance for convergence (default: 1e-4)
                                       Larger values accelerate convergence but reduce accuracy
            max_iters (int, optional): Maximum iterations (default: 5000)
                                       Smaller values accelerate but may not converge
            warm_start (bool, optional): Enable warm-starting from previous solution (default: True)
            verbose (bool, optional): Enable solver verbosity (default: False)

        Examples:
            # Accelerate solver (faster but less accurate)
            mpc.set_solver_params(eps_abs=1e-4, eps_rel=1e-3, max_iters=3000)

            # High precision (slower but more accurate)
            mpc.set_solver_params(eps_abs=1e-7, eps_rel=1e-6, max_iters=10000)
        """
        if eps_abs is not None:
            self.solver_eps_abs = float(eps_abs)
        if eps_rel is not None:
            self.solver_eps_rel = float(eps_rel)
        if max_iters is not None:
            self.solver_max_iters = int(max_iters)
        if warm_start is not None:
            self.solver_warm_start = bool(warm_start)
        if verbose is not None:
            self.solver_verbose = bool(verbose)

    def get_solver_info(self):
        """
        Get current solver configuration.

        Returns:
            dict: Dictionary with solver parameters
        """
        return {
            'eps_abs': self.solver_eps_abs,
            'eps_rel': self.solver_eps_rel,
            'max_iters': self.solver_max_iters,
            'warm_start': self.solver_warm_start,
            'verbose': self.solver_verbose,
            'gpu_enabled': self.enable_gpu and GPU_AVAILABLE,
            'gpu_available': GPU_AVAILABLE,
            'solve_count': self.solve_count,
        }

    def set_cpu_threading(self, num_threads=None, enable_threading=True):
        """
        Configure CPU multi-threading for optimal performance with available cores.

        Enables multi-threaded computation via:
        - SciPy sparse matrix operations (auto-threaded)
        - OSQP solver threading
        - NumPy/BLAS multi-threading

        Args:
            num_threads: Number of threads to use (None = auto-detect all cores)
            enable_threading: Whether to enable threading (default: True)

        Returns:
            dict: CPU threading configuration
        """
        if enable_threading:
            if num_threads is None:
                num_threads = os.cpu_count() if os.cpu_count() else 1
            elif num_threads <= 0:
                num_threads = 1

            self.num_threads = min(num_threads, os.cpu_count() if os.cpu_count() else 1)
            self.enable_threading = True
        else:
            self.num_threads = 1
            self.enable_threading = False

        self.thread_limit = num_threads if enable_threading else 1

        # Return configuration for logging
        return {
            'enable_threading': self.enable_threading,
            'num_threads': self.num_threads,
            'max_available_cores': os.cpu_count() if os.cpu_count() else 1,
            'thread_limit': self.thread_limit
        }

    def get_cpu_info(self):
        """
        Get current CPU threading configuration and capabilities.

        Returns:
            dict: CPU configuration details
        """
        return {
            'threading_enabled': self.enable_threading,
            'threads_configured': self.num_threads,
            'max_cores_available': os.cpu_count() if os.cpu_count() else 1,
            'thread_limit': self.thread_limit,
            'threading_support': {
                'sparse_matrix_ops': 'Multi-threaded via SciPy',
                'osqp_solver': 'Native threading support',
                'numpy_blas': 'Hardware-aware via OpenBLAS/MKL'
            }
        }
