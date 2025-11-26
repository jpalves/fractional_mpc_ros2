"""
Custom exceptions for fractional MPC controller.

Provides specific exception types for different failure modes,
enabling targeted error handling and recovery strategies.
"""


class FractionalMPCError(Exception):
    """
    Base exception for all fractional MPC errors.

    All other exceptions in this module inherit from this class,
    allowing for catching all MPC-related errors with a single except clause.

    Example:
        >>> try:
        ...     # MPC operation
        ... except FractionalMPCError as e:
        ...     print(f"MPC failed: {e}")
    """
    pass


class MPCOptimizationError(FractionalMPCError):
    """
    Raised when MPC optimization fails to find a solution.

    This typically indicates numerical issues in the convex solver (OSQP),
    such as:
    - Infeasible problem (constraints cannot be satisfied)
    - Unbounded problem (no bounded solution exists)
    - Numerical instability (precision issues)
    - Convergence failure (solver did not converge)

    Attributes:
        status: solver status code or name
        message: additional error information
    """

    def __init__(self, status, message=""):
        """
        Initialize optimization error.

        Args:
            status: solver status code/name (str)
            message: additional context (optional)
        """
        self.status = status
        self.message = message
        full_msg = f"MPC optimization failed ({status})"
        if message:
            full_msg += f": {message}"
        super().__init__(full_msg)


class InvalidStateError(FractionalMPCError):
    """
    Raised when state vector contains invalid values.

    Common causes:
    - NaN (Not a Number) values from numerical errors
    - Inf (Infinity) values from unbounded dynamics
    - Extreme values exceeding reasonable physical limits
    - Wrong dimensions (state vector size mismatch)

    Recovery: Check sensor calibration, previous control outputs, and system initialization.

    Example:
        >>> try:
        ...     mpc.solve(invalid_state, reference)
        ... except InvalidStateError as e:
        ...     print(f"State validation failed: {e}")
        ...     # Use last known valid state or zero state
    """

    def __init__(self, error_details):
        """
        Initialize invalid state error.

        Args:
            error_details: description of what made state invalid (str)
        """
        self.error_details = error_details
        super().__init__(f"Invalid state: {error_details}")


class ConfigurationError(FractionalMPCError):
    """
    Raised when MPC configuration is invalid or inconsistent.

    Common causes:
    - Parameter count mismatch (e.g., wrong number of joints)
    - Matrix dimension mismatch (Q, R, A, B incompatible)
    - Invalid parameter values (e.g., negative mass, alpha > 1)
    - Missing required parameters
    - Incompatible solver options

    Recovery: Check configuration file and constructor parameters.

    Example:
        >>> try:
        ...     mpc = FractionalMPCSolver(
        ...         n_joints=9,
        ...         horizon=25,
        ...         alpha=1.5  # ❌ Invalid: should be 0 < α ≤ 1
        ...     )
        ... except ConfigurationError as e:
        ...     print(f"Config error: {e}")
    """

    def __init__(self, param_name, issue, expected=None, actual=None):
        """
        Initialize configuration error.

        Args:
            param_name: name of problematic parameter (str)
            issue: description of the problem (str)
            expected: expected value/range (optional)
            actual: actual value provided (optional)
        """
        self.param_name = param_name
        self.issue = issue
        self.expected = expected
        self.actual = actual

        msg = f"Configuration error in '{param_name}': {issue}"
        if expected is not None:
            msg += f" (expected: {expected}"
            if actual is not None:
                msg += f", got: {actual}"
            msg += ")"

        super().__init__(msg)


class StateVectorError(FractionalMPCError):
    """
    Raised when state vector structure is invalid.

    The state vector must have structure:
        x = [pos₁, ..., posₙ, vel₁, ..., velₙ, e_int₁, ..., e_intₙ, ...]

    Common causes:
    - Wrong total size (expected 3n or 3n + n_frac)
    - Missing components (missing integral error or fractional history)
    - Incorrect reshape/flatten operation

    Example:
        >>> # State should be size 9 (n_joints=3): [3 pos + 3 vel + 3 int]
        >>> state = np.zeros(10)  # Wrong size!
        >>> # Raises StateVectorError
    """

    def __init__(self, expected_size, actual_size, n_joints=None, n_frac=None):
        """
        Initialize state vector error.

        Args:
            expected_size: expected number of elements
            actual_size: actual number of elements
            n_joints: number of joints (optional, for context)
            n_frac: number of fractional history states (optional)
        """
        self.expected_size = expected_size
        self.actual_size = actual_size

        msg = f"State vector size mismatch: expected {expected_size}, got {actual_size}"
        if n_joints is not None:
            msg += f" (n_joints={n_joints}"
            if n_frac is not None:
                msg += f", n_frac={n_frac}"
            msg += ")"

        super().__init__(msg)


class SolverTimeoutError(FractionalMPCError):
    """
    Raised when MPC solver exceeds time limit.

    The solver may be slow due to:
    - Large horizon (more variables/constraints)
    - Tight constraints (harder to satisfy)
    - Numerical ill-conditioning
    - High-dimensional problems

    Recovery: Reduce horizon, relax constraints, or increase timeout.
    """

    def __init__(self, timeout_sec, horizon=None, n_joints=None):
        """
        Initialize timeout error.

        Args:
            timeout_sec: timeout duration in seconds
            horizon: prediction horizon (optional)
            n_joints: number of joints (optional)
        """
        msg = f"MPC solver exceeded time limit ({timeout_sec}s)"
        if horizon is not None:
            msg += f" at horizon={horizon}"
        if n_joints is not None:
            msg += f", n_joints={n_joints}"

        super().__init__(msg)


class ConstraintViolationError(FractionalMPCError):
    """
    Raised when constraints cannot be satisfied (with soft constraints disabled).

    With soft constraints enabled, violations are penalized instead of raising an error.
    This error indicates a fundamental infeasibility in the problem.

    Common causes:
    - Conflicting constraints (upper limit < lower limit)
    - Impossible reference trajectory
    - Infeasible control limits given system dynamics

    Recovery: Check and relax constraints, or enable soft constraints.
    """

    def __init__(self, constraint_type, details=""):
        """
        Initialize constraint violation error.

        Args:
            constraint_type: type of constraint (e.g., "position", "velocity", "control")
            details: additional details about the violation
        """
        msg = f"Constraint violation: {constraint_type}"
        if details:
            msg += f" ({details})"

        super().__init__(msg)


class DynamicsError(FractionalMPCError):
    """
    Raised when system dynamics computation fails.

    Common causes:
    - Invalid fractional order (alpha < 0 or > 1)
    - Oustaloup approximation configuration error
    - Numerical instability in fractional derivative
    - Invalid system parameters (negative mass, etc)
    """

    def __init__(self, issue, alpha=None):
        """
        Initialize dynamics error.

        Args:
            issue: description of the problem
            alpha: fractional order if relevant (optional)
        """
        msg = f"Dynamics computation failed: {issue}"
        if alpha is not None:
            msg += f" (alpha={alpha})"

        super().__init__(msg)
