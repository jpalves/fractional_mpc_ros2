"""
Input validation utilities for MPC and dynamics modules.

Centralizes validation logic to eliminate duplication across controllers and tests.
All validators raise specific exceptions for clear error handling.
"""

import numpy as np


def validate_state_dimension(state, expected_size, name="state"):
    """
    Validate state vector has correct dimension.

    Args:
        state: numpy array of state values
        expected_size: expected number of elements
        name: name of state for error message (default: "state")

    Raises:
        ValueError: if dimension mismatch

    Example:
        >>> state = np.zeros(9)
        >>> validate_state_dimension(state, 9)  # OK
        >>> validate_state_dimension(state, 10)  # Raises ValueError
    """
    state_array = np.asarray(state).flatten()
    if state_array.shape[0] != expected_size:
        raise ValueError(
            f"{name} dimension mismatch: expected {expected_size}, got {state_array.shape[0]}"
        )


def validate_state_values(state, max_abs_value=1e6, name="state"):
    """
    Check state contains no NaN, Inf, or extreme values.

    Args:
        state: numpy array of state values
        max_abs_value: maximum absolute value allowed (default: 1e6)
        name: name of state for error message (default: "state")

    Raises:
        ValueError: if state contains invalid values

    Example:
        >>> state = np.array([1.0, 2.0, 3.0])
        >>> validate_state_values(state)  # OK

        >>> state = np.array([np.nan, 2.0, 3.0])
        >>> validate_state_values(state)  # Raises ValueError
    """
    state_array = np.asarray(state).flatten()

    if np.any(np.isnan(state_array)):
        nan_indices = np.where(np.isnan(state_array))[0]
        raise ValueError(
            f"{name} contains NaN values at indices {nan_indices}"
        )

    if np.any(np.isinf(state_array)):
        inf_indices = np.where(np.isinf(state_array))[0]
        raise ValueError(
            f"{name} contains Inf values at indices {inf_indices}"
        )

    max_val = np.max(np.abs(state_array))
    if max_val > max_abs_value:
        raise ValueError(
            f"{name} contains extreme values (max abs: {max_val:.2e}, "
            f"threshold: {max_abs_value:.2e})"
        )


def validate_joint_count(actual, expected, logger=None):
    """
    Validate joint count matches expected number.

    Args:
        actual: actual joint count
        expected: expected joint count
        logger: optional logger for error reporting (e.g., ROS2 logger)

    Raises:
        ValueError: if count mismatch

    Example:
        >>> validate_joint_count(9, 9)  # OK
        >>> validate_joint_count(8, 9)  # Raises ValueError
    """
    if actual != expected:
        msg = (
            f"Joint count mismatch: expected {expected}, got {actual}. "
            f"Check that joint_states publisher has correct number of joints."
        )
        if logger:
            logger.error(msg)
        raise ValueError(msg)


def validate_reference_trajectory(x_ref, horizon, n_joints, name="reference"):
    """
    Validate reference trajectory shape and values.

    Args:
        x_ref: reference trajectory array (horizon+1, 2*n_joints)
        horizon: prediction horizon
        n_joints: number of joints
        name: name for error messages (default: "reference")

    Raises:
        ValueError: if trajectory shape or values are invalid

    Expected shape: (horizon+1, 2*n_joints) where columns are [pos_1, ..., pos_n, vel_1, ..., vel_n]

    Example:
        >>> x_ref = np.zeros((11, 6))  # horizon=10, n_joints=3
        >>> validate_reference_trajectory(x_ref, 10, 3)  # OK
    """
    x_ref_array = np.asarray(x_ref)

    # Check horizon dimension
    if x_ref_array.shape[0] != horizon + 1:
        raise ValueError(
            f"{name} horizon mismatch: expected {horizon + 1}, got {x_ref_array.shape[0]}"
        )

    # Check state dimension (positions + velocities)
    expected_state_dim = 2 * n_joints
    if x_ref_array.shape[1] != expected_state_dim:
        raise ValueError(
            f"{name} state dimension mismatch: expected {expected_state_dim}, "
            f"got {x_ref_array.shape[1]}. Expected {n_joints} position + {n_joints} velocity components."
        )

    # Validate values (NaN, Inf, extremes)
    validate_state_values(x_ref_array, name=name)


def validate_control_dimension(control, expected_size, name="control"):
    """
    Validate control vector has correct dimension.

    Args:
        control: numpy array of control values
        expected_size: expected number of joints
        name: name for error message (default: "control")

    Raises:
        ValueError: if dimension mismatch
    """
    control_array = np.asarray(control).flatten()
    if control_array.shape[0] != expected_size:
        raise ValueError(
            f"{name} dimension mismatch: expected {expected_size}, got {control_array.shape[0]}"
        )


def validate_control_limits(control, u_min, u_max, name="control"):
    """
    Validate control is within specified limits.

    Args:
        control: numpy array of control values
        u_min: minimum control limits
        u_max: maximum control limits
        name: name for error message (default: "control")

    Raises:
        ValueError: if control violates limits (warning only, does not raise)

    Note: This is a soft check - logs violations but does not raise
    """
    control_array = np.asarray(control).flatten()
    u_min_array = np.asarray(u_min).flatten()
    u_max_array = np.asarray(u_max).flatten()

    violations_min = control_array < u_min_array
    violations_max = control_array > u_max_array

    if np.any(violations_min):
        min_idx = np.where(violations_min)[0]
        print(
            f"WARNING: {name} violates minimum limits at indices {min_idx}"
        )

    if np.any(violations_max):
        max_idx = np.where(violations_max)[0]
        print(
            f"WARNING: {name} violates maximum limits at indices {max_idx}"
        )
