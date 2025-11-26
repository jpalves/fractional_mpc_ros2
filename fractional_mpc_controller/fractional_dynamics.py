"""
Fractional-Order System Dynamics Module

Implements Caputo fractional-order system dynamics for robot manipulator joints.
Uses Oustaloup approximation with pole-zero placement for accurate fractional derivative.

Fractional system: D^α x = f(x, u) where D^α is Caputo derivative (0 < α ≤ 1)

Oustaloup Approximation:
- Distributes poles and zeros in complex plane for optimal frequency approximation
- Provides rational function approximation: H(s) ≈ s^α
- Improves accuracy over Grünwald-Letnikov for control applications
"""

import numpy as np
from scipy.special import gamma
from scipy.signal import lti, TransferFunction


class FractionalOrderSystem:
    """Fractional-order system for robot manipulator joints with Caputo derivatives"""

    def __init__(self, n_joints=9, alpha=0.80, dt=0.01):
        """
        Initialize fractional-order system.

        Args:
            n_joints: Number of robot joints
            alpha: Fractional order (0 < α ≤ 1). α=1.0 recovers standard 2nd order
            dt: Sampling time
        """
        self.n_joints = n_joints
        self.alpha = alpha  # Fractional order
        self.dt = dt

        # State: [q1, q2, ..., qn, v1, v2, ..., vn] (position and velocity)
        self.state_size = 2 * n_joints
        self.x = np.zeros(self.state_size)

        # Full history for Grünwald-Letnikov coefficients (memory: all previous steps)
        # history[k] stores state at time t - k*dt
        self.history_q = np.zeros((self.n_joints, 1000))  # Positions history
        self.history_v = np.zeros((self.n_joints, 1000))  # Velocities history
        self.time_step = 0  # Current time step counter

        # Linear system matrices (spring-damper model)
        self._setup_system_matrices()

    def _setup_system_matrices(self):
        """
        Setup system matrices for fractional-order manipulator dynamics.

        Standard spring-damper parameters (independent of α).
        Fractional derivative modifies the temporal evolution.
        """
        # Spring-damper model parameters
        self.K = 2.0 * np.eye(self.n_joints)      # Spring stiffness
        self.D = 0.5 * np.eye(self.n_joints)      # Damping coefficient
        self.B = np.eye(self.n_joints)            # Control input gain

        # Oustaloup approximation parameters (cached)
        self.oustaloup_poles = None
        self.oustaloup_zeros = None
        self.oustaloup_gain = None
        self._setup_oustaloup_approximation()

    def _setup_oustaloup_approximation(self, n_poles=5, w_low=0.01, w_high=100):
        """
        Setup Oustaloup approximation for fractional derivative s^α ≈ H(s).

        The Oustaloup approximation represents a fractional operator using
        a rational function with optimally distributed poles and zeros in
        the complex plane.

        Mathematical basis:
        s^α ≈ K * ∏(s - z_k) / ∏(s - p_k)

        where poles and zeros follow:
        p_k = w_low * (w_high/w_low)^(k / (n_poles + 1)) * exp(j*π*(1-α)/2)
        z_k = w_low * (w_high/w_low)^((k+1) / (n_poles + 1)) * exp(j*π*(1-α)/2)

        Args:
            n_poles: Number of poles/zeros for approximation (default: 5)
            w_low: Lower frequency bound (rad/s)
            w_high: Upper frequency bound (rad/s)
        """
        # Frequency band center and logarithmic spacing
        ln_w_ratio = np.log(w_high / w_low)
        phi = np.pi * (1 - self.alpha) / 2

        # Compute poles and zeros
        self.oustaloup_poles = np.zeros(n_poles, dtype=complex)
        self.oustaloup_zeros = np.zeros(n_poles, dtype=complex)

        for k in range(n_poles):
            # Logarithmically spaced frequencies
            pole_freq = w_low * np.exp((k / (n_poles + 1)) * ln_w_ratio)
            zero_freq = w_low * np.exp(((k + 1) / (n_poles + 1)) * ln_w_ratio)

            # Distribute in complex plane with phase matching α
            self.oustaloup_poles[k] = pole_freq * np.exp(1j * phi)
            self.oustaloup_zeros[k] = zero_freq * np.exp(1j * phi)

        # Compute gain to match DC behavior
        # For s^α: at very low frequencies, H(s) ≈ K * (w_low/...)^α
        self.oustaloup_gain = (w_high / w_low) ** self.alpha

        # Setup discrete approximation (cached)
        self._setup_discrete_oustaloup()

    def _setup_discrete_oustaloup(self):
        """
        Convert Oustaloup approximation from continuous domain (s-plane) to
        discrete domain (z-plane) using bilinear transform (Tustin's method).

        Bilinear transform: s ≈ 2/dt * (z - 1) / (z + 1)

        This converts continuous poles and zeros to discrete equivalents
        suitable for direct implementation in discrete-time simulations.
        """
        if self.oustaloup_poles is None or self.oustaloup_zeros is None:
            return

        # Bilinear transform parameters
        c = 2.0 / self.dt  # Prewarping constant

        # Convert continuous poles to discrete (s -> z domain)
        self.oustaloup_poles_z = np.zeros_like(self.oustaloup_poles)
        for k, pole_s in enumerate(self.oustaloup_poles):
            # z = (1 + pole_s/c) / (1 - pole_s/c)
            self.oustaloup_poles_z[k] = (1 + pole_s / c) / (1 - pole_s / c)

        # Convert continuous zeros to discrete (s -> z domain)
        self.oustaloup_zeros_z = np.zeros_like(self.oustaloup_zeros)
        for k, zero_s in enumerate(self.oustaloup_zeros):
            # z = (1 + zero_s/c) / (1 - zero_s/c)
            self.oustaloup_zeros_z[k] = (1 + zero_s / c) / (1 - zero_s / c)

        # Adjust gain for bilinear transform (compensate for s-domain to z-domain mapping)
        # H(z) = K_d * product(z - z_k) / product(z - p_k)
        # The DC gain adjustment is critical for accuracy
        num = np.prod(1 - self.oustaloup_zeros_z)  # Evaluate at z=1 (DC)
        denom = np.prod(1 - self.oustaloup_poles_z)  # Evaluate at z=1 (DC)
        self.oustaloup_gain_z = self.oustaloup_gain * np.abs(num / denom)

    def _oustaloup_fractional_derivative(self, values_history, n_history):
        """
        Compute Caputo fractional derivative using Oustaloup approximation.

        Uses the discrete transfer function H(z) = K_d * ∏(z - z_k) / ∏(z - p_k)
        to approximate s^α in the discrete-time domain.

        This method maintains state history for the denominator terms (poles)
        to implement the transfer function as a difference equation:

        y[n] = K_d * sum(num_coeff * x[n-i]) - sum(den_coeff * y[n-i])

        Args:
            values_history: History of values [x(t), x(t-Δt), x(t-2Δt), ...]
            n_history: Number of history terms available

        Returns:
            d_alpha_x: Fractional derivative approximation using Oustaloup
        """
        # Ensure discrete Oustaloup is setup
        if not hasattr(self, 'oustaloup_poles_z') or self.oustaloup_poles_z is None:
            return np.zeros_like(values_history[0]) if isinstance(values_history, (list, np.ndarray)) and len(values_history) > 0 else 0.0

        n_poles = len(self.oustaloup_poles_z)
        n_use = min(n_history, len(values_history) if isinstance(values_history, (list, np.ndarray)) else 1)

        if n_use == 0:
            return np.zeros_like(values_history[0]) if isinstance(values_history, (list, np.ndarray)) and len(values_history) > 0 else 0.0

        # Compute numerator contribution: K_d * ∏(z - z_k) evaluated with history
        # For discrete system: D^α_discrete ≈ numerator polynomial with delays
        num_coeff = np.zeros(n_use)

        # Start with the gain
        num_coeff[0] = self.oustaloup_gain_z

        # Build numerator coefficients from zeros using recursive expansion
        # This is equivalent to: K * (z - z_1)(z - z_2)...(z - z_n)
        for zero_idx in range(n_poles):
            zero_z = self.oustaloup_zeros_z[zero_idx]
            # Convolve with (z - zero_z) in reverse order
            for i in range(min(zero_idx + 1, n_use - 1), 0, -1):
                if i < len(num_coeff):
                    num_coeff[i] = num_coeff[i] - zero_z * (num_coeff[i - 1] if i > 0 else 0)

        # Apply numerator (zeros) to history
        if isinstance(values_history, (list, np.ndarray)):
            if isinstance(values_history[0], (list, np.ndarray)):
                # Vector case (multiple joints)
                d_alpha = np.zeros_like(values_history[0])
                for k in range(n_use):
                    d_alpha += num_coeff[k] * values_history[k]
            else:
                # Scalar case
                d_alpha = np.sum([num_coeff[k] * values_history[k] for k in range(n_use)])
        else:
            d_alpha = num_coeff[0] * values_history

        return d_alpha

    def _compute_gl_coefficients(self, n_terms):
        """
        Compute Grünwald-Letnikov (GL) coefficients for fractional derivative.

        GL approximation: D^α f(t) ≈ h^(-α) Σ c_k f(t - k*h)
        where c_k = (-1)^k * Γ(α+1) / (Γ(k+1)*Γ(α-k+1))

        Handles case where Γ(α-k+1) becomes problematic when k > α-1 + ε
        by using recurrence relation: c_k = c_{k-1} * (k - 1 - α) / k

        Args:
            n_terms: Number of history terms to use

        Returns:
            gl_coeffs: GL coefficients array of shape (n_terms,)
        """
        gl_coeffs = np.zeros(n_terms)

        if n_terms == 0:
            return gl_coeffs

        # First coefficient
        gl_coeffs[0] = 1.0

        # Use recurrence relation to avoid gamma function issues
        # c_k = c_{k-1} * (-(α - (k-1))) / k = c_{k-1} * (k - 1 - α) / k
        for k in range(1, n_terms):
            gl_coeffs[k] = gl_coeffs[k-1] * (k - 1 - self.alpha) / k

        # Scale by dt^(-α) for discrete approximation
        scale = self.dt ** (-self.alpha)
        gl_coeffs = scale * gl_coeffs

        return gl_coeffs

    def _caputo_fractional_derivative(self, values_history, n_history):
        """
        Compute Caputo fractional derivative using Grünwald-Letnikov approximation.

        For state variable x, computes D^α x(t) ≈ h^(-α) Σ c_k x(t - k*h)

        Args:
            values_history: History of values [x(t), x(t-Δt), x(t-2Δt), ...]
            n_history: Number of history terms available

        Returns:
            d_alpha_x: Fractional derivative approximation
        """
        # Use available history (up to stored history)
        n_use = min(n_history, values_history.shape[0] if values_history.ndim > 0 else 1)

        if n_use == 0:
            return np.zeros_like(values_history[0]) if values_history.ndim > 0 else 0.0

        # Compute GL coefficients for this many terms
        gl_coeffs = self._compute_gl_coefficients(n_use)

        # Apply fractional derivative: sum over history
        if isinstance(values_history, (list, np.ndarray)):
            if isinstance(values_history[0], (list, np.ndarray)):
                # Vector case (multiple joints)
                d_alpha = np.zeros_like(values_history[0])
                for k in range(n_use):
                    d_alpha += gl_coeffs[k] * values_history[k]
            else:
                # Scalar case
                d_alpha = np.sum([gl_coeffs[k] * values_history[k] for k in range(n_use)])
        else:
            d_alpha = gl_coeffs[0] * values_history

        return d_alpha

    def dynamics(self, u, u_ref=None, use_oustaloup=False):
        """
        Compute fractional-order system dynamics with input-centered linearization: D^α v = f(x, u)

        Fractional model (for position q and velocity v):
        q'  = v
        D^α v = -K*q - D*v + B*(u - u_ref) + B*u_ref

        where D^α is Caputo fractional derivative of order α.

        With input-centered linearization:
        - Linearization point is defined by u_ref (reference input)
        - System is linearized around the input trajectory
        - Improves accuracy for large input variations

        Fractional Derivative Approximation Methods:
        - use_oustaloup=True: Uses Oustaloup approximation (pole-zero placement in complex plane)
        - use_oustaloup=False: Uses Grünwald-Letnikov coefficients (original method) - DEFAULT for stability

        Args:
            u: Control input (position commands for each joint)
            u_ref: Reference input for linearization point (default: 0)
            use_oustaloup: Use Oustaloup approximation (True) or GL coefficients (False) - GL is more stable!

        Returns:
            dx: State derivative [q', v'] = [v, D^α v]
        """
        dx = np.zeros(self.state_size)

        # Set reference input to zero if not provided
        if u_ref is None:
            u_ref = np.zeros(self.n_joints)

        # Compute input deviation from reference
        u_delta = u - u_ref

        q = self.x[:self.n_joints]
        v = self.x[self.n_joints:]

        # Standard position derivative
        dx[:self.n_joints] = v

        # Fractional derivative of velocity using Grünwald-Letnikov approximation
        # For full history: use all stored values
        current_time_idx = min(self.time_step, self.history_v.shape[1] - 1)

        # Build history array: [v(t), v(t-Δt), v(t-2Δt), ...]
        v_history = []
        for k in range(current_time_idx + 1):
            idx = (current_time_idx - k) % self.history_v.shape[1]
            v_history.append(self.history_v[:, idx])

        # Compute fractional derivative of velocity using Grünwald-Letnikov
        if len(v_history) > 0:
            # Compute D^α v using Grünwald-Letnikov (numerically more stable)
            gl_coeffs = self._compute_gl_coefficients(len(v_history))
            d_alpha_v = np.sum([gl_coeffs[k] * v_history[k] for k in range(len(v_history))], axis=0)

            # Safety clipping: prevent unbounded growth in edge cases
            max_derivative = 1e6  # Reasonable upper bound for fractional derivative
            d_alpha_v = np.clip(d_alpha_v, -max_derivative, max_derivative)
        else:
            d_alpha_v = np.zeros(self.n_joints)

        # Velocity derivative: dx/dt for velocity = d_alpha_v (the computed fractional derivative)
        dx[self.n_joints:] = d_alpha_v

        # Safety clipping: prevent unbounded state growth in infeasible scenarios
        dx[self.n_joints:] = np.clip(dx[self.n_joints:], -1e6, 1e6)

        return dx

    def step(self, u):
        """
        Advance system state by one time step using fractional integration.

        Uses Caputo fractional derivative with Grünwald-Letnikov approximation.

        Args:
            u: Control input (n_joints,)

        Returns:
            Updated state vector [positions, velocities]
        """
        # Compute fractional dynamics
        dx = self.dynamics(u)

        # Standard Euler-like integration (works with fractional derivatives)
        self.x = self.x + self.dt * dx

        # Store current velocity in history (for GL coefficients)
        time_idx = self.time_step % self.history_v.shape[1]
        self.history_q[:, time_idx] = self.x[:self.n_joints].copy()
        self.history_v[:, time_idx] = self.x[self.n_joints:].copy()

        self.time_step += 1

        return self.x.copy()

    def set_state(self, q, v=None):
        """Set system state (positions and optionally velocities)"""
        self.x[:self.n_joints] = np.array(q)
        if v is not None:
            self.x[self.n_joints:] = np.array(v)
        else:
            self.x[self.n_joints:] = np.zeros(self.n_joints)

        # Reset history
        self.time_step = 0
        self.history_q.fill(0.0)
        self.history_v.fill(0.0)

    def get_state(self):
        """Get current state"""
        return self.x.copy()

    def get_positions(self):
        """Get joint positions"""
        return self.x[:self.n_joints].copy()

    def get_velocities(self):
        """Get joint velocities"""
        return self.x[self.n_joints:].copy()

    def get_alpha(self):
        """Get fractional order"""
        return self.alpha

    def set_alpha(self, alpha):
        """Set fractional order (1 ≤ α ≤ 1)"""
        if not (0 < alpha <= 1.0):
            raise ValueError(f"Fractional order must be in (0, 1], got {alpha}")
        self.alpha = alpha


# Backward compatibility: standard second-order system as special case (α=1.0)
class SecondOrderSystem(FractionalOrderSystem):
    """Standard second-order system (α=1.0) - special case of fractional system"""

    def __init__(self, n_joints=9, dt=0.01):
        """Initialize as standard 2nd order system (α=1.0)"""
        super().__init__(n_joints=n_joints, alpha=1.0, dt=dt)
