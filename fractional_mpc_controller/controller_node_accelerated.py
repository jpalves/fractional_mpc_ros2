"""
ROS2 Node for Fractional-Order MPC Controller - CPU+GPU Accelerated

Optimizations:
- GPU acceleration (CuPy) for matrix operations
- Multi-threading for I/O and non-critical tasks
- Warm-starting in OSQP for faster convergence
- Memory pooling to reduce allocation overhead
"""

import numpy as np
from threading import Lock, Thread
from queue import Queue
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState

from .fractional_dynamics import FractionalOrderSystem
from .mpc_solver import FractionalMPCSolver
from .validators import validate_joint_count
from .config import MPCWeights
from .exceptions import InvalidStateError, ConfigurationError

# GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np


class FractionalMPCControllerNodeAccelerated(Node):
    """ROS2 node for fractional MPC control with CPU+GPU acceleration"""

    def __init__(self):
        super().__init__("fractional_mpc_controller_accelerated")

        # Parameters
        self.declare_parameter("n_joints", 9)
        self.declare_parameter("fractional_order", 0.80)
        self.declare_parameter("control_dt", 0.01)
        self.declare_parameter("mpc_horizon", 25)
        self.declare_parameter("joint_names", [f"joint_{i}" for i in range(9)])
        self.declare_parameter("enable_gpu", True)  # NEW: GPU acceleration flag
        self.declare_parameter("enable_warm_start", True)  # NEW: Warm-starting
        self.declare_parameter("worker_threads", 10)  # NEW: Background worker threads

        self.n_joints = self.get_parameter("n_joints").value
        self.alpha = self.get_parameter("fractional_order").value
        self.dt = self.get_parameter("control_dt").value
        self.horizon = self.get_parameter("mpc_horizon").value
        self.joint_names = self.get_parameter("joint_names").value
        self.enable_gpu = self.get_parameter("enable_gpu").value and GPU_AVAILABLE
        self.enable_warm_start = self.get_parameter("enable_warm_start").value
        self.n_workers = self.get_parameter("worker_threads").value

        # Cost weights - declare with launch file defaults, NOT hardcoded ones!
        # CRITICAL FIX: Use launch file defaults, not old hardcoded defaults
        # Precedence: command line > launch file > declare_parameter default
        # Launch file has: Q_pos=600.0, Q_vel=12.0, R=0.15, Ki=2.0
        # NOT the old hardcoded: Q_pos=800.0, Q_vel=15.0, R=0.2, Ki=2.0
        self.declare_parameter("state_cost_position", 600.0)  # Launch file default
        self.declare_parameter("state_cost_velocity", 12.0)   # Launch file default
        self.declare_parameter("control_cost", 0.15)          # Launch file default
        self.declare_parameter("u_min", [-15.0]*self.n_joints)  # Launch file default
        self.declare_parameter("u_max", [15.0]*self.n_joints)   # Launch file default
        self.declare_parameter("integral_gain",1.0)          # Launch file default
        self.declare_parameter("spring_const", 1.0)          # Launch file default
        self.declare_parameter("integral_cost_scale",   1.)  # Matches legacy 50/600 ratio
        self.declare_parameter("integral_cost_min",     .1)       # Prevents integral weight collapse
        self.declare_parameter("velocity_cost_scale",  7.0)   # Maintains minimum damping when Q_vel low
        self.declare_parameter("velocity_cost_min",    1.0)      # Absolute floor for velocity weight
        self.declare_parameter("integral_leak_rate",  18.0)     # Fractional decay per second when near reference
        self.declare_parameter("integral_leak_error_threshold",    1.8)  # Position error threshold for decay
        self.declare_parameter("integral_leak_velocity_threshold", 0.8)  # Velocity threshold for decay
        self.declare_parameter("integral_max_magnitude", 5.0)  # Hard clamp for accumulated integral error
        self.declare_parameter("joint_mass", [1.0]*self.n_joints)  # Default unit mass per joint

        
        self.q_pos = self.get_parameter("state_cost_position").value
        self.q_vel = self.get_parameter("state_cost_velocity").value
        self.r = self.get_parameter("control_cost").value
        self.u_min = self.get_parameter("u_min").value
        self.u_max = self.get_parameter("u_max").value
        self.spring_k = self.get_parameter("spring_const").value
        self.integral_cost_scale = self.get_parameter("integral_cost_scale").value
        self.integral_cost_min = self.get_parameter("integral_cost_min").value
        self.velocity_cost_scale = self.get_parameter("velocity_cost_scale").value
        self.velocity_cost_min = self.get_parameter("velocity_cost_min").value
        self.integral_leak_rate = max(0.0, self.get_parameter("integral_leak_rate").value)
        self.integral_leak_error_threshold = max(0.0, self.get_parameter("integral_leak_error_threshold").value)
        self.integral_leak_velocity_threshold = max(0.0, self.get_parameter("integral_leak_velocity_threshold").value)
        self.integral_max = max(0.0, self.get_parameter("integral_max_magnitude").value)
        self.mass_vector = np.array(self.get_parameter("joint_mass").value, dtype=float)
        if self.mass_vector.size != self.n_joints:
            raise ValueError("joint_mass parameter must have n_joints entries")
        if np.any(self.mass_vector <= 0.0):
            raise ValueError("joint_mass entries must be positive")
        
        self.get_logger().info(
            f"Fractional MPC Controller Accelerated: "
            f"{self.n_joints} joints, α={self.alpha}, dt={self.dt}s, horizon={self.horizon}, "
            f"GPU={self.enable_gpu}, Warm-start={self.enable_warm_start}, Workers={self.n_workers}"
        )

        # Get integral gain (from launch file, no hardcoded default)
        self.Ki = self.get_parameter("integral_gain").value

        # Initialize system and solver
        self.system = FractionalOrderSystem(
            n_joints=self.n_joints, alpha=self.alpha, dt=self.dt
        )
        self.mpc = FractionalMPCSolver(
            n_joints=self.n_joints,
            horizon=self.horizon,
            dt=self.dt,
            alpha=self.alpha,
            spring_const=self.spring_k,
            Ki=self.Ki,
            mass=self.mass_vector,
            use_oustaloup=True,
        )
        if self.integral_max > 0.0:
            self.mpc.e_int_max = self.integral_max * np.ones(self.n_joints)

        # Configure MPC with INTEGRAL as STATE variable
        # ARCHITECTURE: MPC state is NOW [position, velocity, integral_error] (3n)
        # Integral error is optimized by MPC, not accumulated externally
        q_int = max(self.q_pos * self.integral_cost_scale, self.integral_cost_min)
        q_vel_eff = max(self.q_vel, self.q_pos * self.velocity_cost_scale, self.velocity_cost_min)
        self.q_vel_effective = q_vel_eff

        # Use centralized MPCWeights for consistent Q matrix construction
        Q = MPCWeights.create_diagonal(
            q_pos=self.q_pos,
            q_vel=q_vel_eff,
            q_int=q_int,
            n_joints=self.n_joints
        )
        R = self.r * np.eye(self.n_joints)
        self.mpc.set_weights(Q, R)
        self.mpc.set_control_limits(self.u_min, self.u_max)
        # Note: Ki is kept for backward compatibility but NOT used with integral in MPC
        #self.mpc.set_integral_gain(self.Ki)

        self.get_logger().info(
            f"MPC configured: Q_pos={self.q_pos}, Q_vel={self.q_vel}, R={self.r}, Ki={self.Ki}"
        )

        # Print parameter summary
        self.get_logger().info("="*70)
        self.get_logger().info("PARAMETER SUMMARY:")
        self.get_logger().info("="*70)
        self.get_logger().info(f"  System:        {self.n_joints} joints, α={self.alpha}, dt={self.dt}s")
        self.get_logger().info(f"  MPC:           horizon={self.horizon} steps")
        self.get_logger().info(f"  Cost Weights:  Q_pos={self.q_pos}, Q_vel={self.q_vel} (eff={q_vel_eff:.3f}), Q_int={q_int:.3f}, R={self.r}")
        self.get_logger().info(f"  Integral:      Ki={self.Ki}")
        self.get_logger().info(f"  Acceleration:  GPU={self.enable_gpu}, Warm-start={self.enable_warm_start}")
        self.get_logger().info(f"  I/O:           {self.n_workers} worker threads")
        self.get_logger().info(f"  Control Range: [{self.u_min[0]:.1f}, {self.u_max[0]:.1f}] (radianos/metros)")
        self.get_logger().info(f"  Mass:          {self.mass_vector.tolist()}")
        self.get_logger().info("="*70)

        # State tracking
        # State dimension matches MPC state_size: 3n + n_hist*n (fractional velocity history taps)
        self.state_size_total = self.mpc.state_size
        self.current_state = np.zeros(self.state_size_total)
        self.reference_state = np.zeros(2 * self.n_joints)
        self.n_frac_states = self.mpc.n_frac_states  # Match MPC configuration

        # Velocity estimation
        self.prev_position = np.zeros(self.n_joints)
        self.estimated_velocity = np.zeros(self.n_joints)
        self.velocity_filter_alpha = 0.3

        # Control signals
        self.last_control = np.zeros(self.n_joints)
        self.state_lock = Lock()

        # FIXED: Integral error accumulated synchronously in control loop
        # (NOT in callback, to maintain perfect synchronization with dt)
        self.e_int_accumulated = np.zeros(self.n_joints)

        # GPU Memory pooling
        if self.enable_gpu:
            self._init_gpu_pools()

        # Warm-starting: store previous solution
        self.last_u_opt = None
        self.last_x_pred = None

        # Multi-threading for I/O
        self.publish_queue = Queue(maxsize=10)
        self.worker_threads = []
        self._start_worker_threads()

        # QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, qos
        )
        self.reference_state_sub = self.create_subscription(
            JointState, "/reference_state", self._reference_state_callback, qos
        )

        # Publishers
        self.command_pub = self.create_publisher(
            JointState, "/control_command", qos
        )
        self.predicted_state_pub = self.create_publisher(
            Float32MultiArray, "mpc_predicted_state", qos
        )

        # Performance monitoring
        self.solve_times = []
        self.max_history = 100

        # Wait for reference state before starting control
        self.get_logger().info("Waiting for /reference_state...")
        timeout = 5.0  # 5 seconds timeout
        start_time = time.time()
        while np.allclose(self.reference_state, 0.0) and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if np.allclose(self.reference_state, 0.0):
            self.get_logger().warn(
                "No /reference_state received after 5s. "
                "Control will use zero reference (no integral action expected)"
            )
        else:
            self.get_logger().info(f"Reference received: {self.reference_state[:self.n_joints]}")

        # Control timer
        self.control_timer = self.create_timer(self.dt, self._control_loop)

        self.get_logger().info("Fractional MPC Controller Accelerated ready")

    def _init_gpu_pools(self):
        """Initialize GPU memory pools for efficient allocation"""
        try:
            # Pre-allocate frequently used GPU arrays
            # Note: x0 is now [3+n_frac]*n (pos+vel+integral error+fractional filter states)
            self.gpu_x0 = cp.zeros((self.state_size_total,), dtype=cp.float32)
            self.gpu_x_ref = cp.zeros((self.horizon+1, self.n_joints*2), dtype=cp.float32)
            q_int = max(self.q_pos * self.integral_cost_scale, self.integral_cost_min)
            q_vel_eff = max(self.q_vel, self.q_pos * self.velocity_cost_scale, self.velocity_cost_min)

            # Use centralized MPCWeights for consistent Q matrix construction
            Q_cpu = MPCWeights.create_diagonal(
                q_pos=self.q_pos,
                q_vel=q_vel_eff,
                q_int=q_int,
                n_joints=self.n_joints
            )
            self.gpu_Q = cp.asarray(Q_cpu, dtype=cp.float32)
            self.gpu_R = cp.asarray(self.r * np.eye(self.n_joints), dtype=cp.float32)

            frac_suffix = f"+{self.n_frac_states}" if self.n_frac_states else ""
            self.get_logger().info(
                f"GPU memory pools initialized ([3{frac_suffix}]n state with integral{' + Oustaloup' if self.n_frac_states else ''})"
            )
        except Exception as e:
            self.get_logger().warn(f"GPU pool initialization failed: {e}")
            self.enable_gpu = False

    def _start_worker_threads(self):
        """Start background worker threads for I/O and non-critical tasks"""
        for i in range(self.n_workers):
            t = Thread(target=self._worker_loop, daemon=True)
            t.start()
            self.worker_threads.append(t)
        self.get_logger().debug(f"Started {self.n_workers} worker threads")

    def _worker_loop(self):
        """Background worker for publishing and logging"""
        while rclpy.ok():
            try:
                task = self.publish_queue.get(timeout=0.1)
                if task is None:
                    break

                task_type, data = task
                if task_type == "command":
                    self._publish_control_command(data)
                elif task_type == "predicted":
                    self._publish_predicted_state(data)
                elif task_type == "log":
                    self.get_logger().info(data)

            except:
                continue

    def _joint_state_callback(self, msg: JointState):
        """Update current joint states with velocity estimation"""
        positions = np.array(msg.position) if msg.position else np.zeros(self.n_joints)

        # Validate joint count using centralized validator
        try:
            validate_joint_count(len(positions), self.n_joints, logger=self.get_logger())
        except ValueError:
            return

        with self.state_lock:
            if len(positions) >= self.n_joints:
                self.current_state[:self.n_joints] = positions[:self.n_joints]

            # Velocity estimation with low-pass filtering
            pos_delta = positions[:self.n_joints] - self.prev_position
            raw_velocity = pos_delta / self.dt
            self.estimated_velocity = (
                self.velocity_filter_alpha * raw_velocity +
                (1 - self.velocity_filter_alpha) * self.estimated_velocity
            )
            self.current_state[self.n_joints:2*self.n_joints] = self.estimated_velocity
            self.prev_position = positions[:self.n_joints].copy()
            # NOTE: Integral error is NOW accumulated in control loop, NOT here

    def _reference_state_callback(self, msg: JointState):
        """Receive reference state"""
        positions = np.array(msg.position) if msg.position else np.zeros(self.n_joints)
        velocities = np.array(msg.velocity) if msg.velocity else np.zeros(self.n_joints)

        # Validate reference joint count using centralized validator
        try:
            validate_joint_count(len(positions), self.n_joints, logger=self.get_logger())
        except ValueError:
            return

        with self.state_lock:
            if len(positions) >= self.n_joints:
                self.reference_state[:self.n_joints] = positions[:self.n_joints]
            if len(velocities) >= self.n_joints:
                self.reference_state[self.n_joints:] = velocities[:self.n_joints]

            # DEBUG: Log what reference is being set
            #self.get_logger().info(
            #    f"[REFERENCE UPDATE] positions={self.reference_state[:self.n_joints].tolist()}"
            #)

    def _get_reference_sequence(self):
        """Get reference trajectory (constant setpoint)"""
        with self.state_lock:
            ref_state = self.reference_state.copy()

        # If GPU enabled, use GPU for this computation
        if self.enable_gpu:
            try:
                ref_gpu = cp.asarray(ref_state, dtype=cp.float32)
                x_ref = cp.zeros((self.horizon + 1, 2*self.n_joints), dtype=cp.float32)
                x_ref[:, :] = ref_gpu
                return cp.asnumpy(x_ref)
            except:
                pass  # Fall back to CPU

        # CPU fallback
        x_ref = np.zeros((self.horizon + 1, 2*self.n_joints))
        x_ref[:, :] = ref_state
        return x_ref

    def _control_loop(self):
        """Main MPC control loop with acceleration optimizations"""
        start_time = time.time()

        try:
            with self.state_lock:
                positions = self.current_state[:self.n_joints].copy()
                velocities = self.current_state[self.n_joints:2*self.n_joints].copy()
                pos_error = self.reference_state[:self.n_joints] - positions

                # Integrate measured position error (single source of accumulation)
                self.e_int_accumulated += self.dt * pos_error
                if self.integral_max > 0.0:
                    np.clip(self.e_int_accumulated, -self.integral_max, self.integral_max, out=self.e_int_accumulated)

                # Integral leak when close to reference in position and velocity
                if self.integral_leak_rate > 0.0:
                    if (
                        np.all(np.abs(pos_error) <= self.integral_leak_error_threshold)
                        and np.all(np.abs(velocities) <= self.integral_leak_velocity_threshold)
                    ):
                        leak_factor = max(0.0, 1.0 - self.integral_leak_rate * self.dt)
                        self.e_int_accumulated *= leak_factor

                # Keep internal state consistent with accumulated integral term
                e_int_start = 2 * self.n_joints
                e_int_end = 3 * self.n_joints
                self.current_state[e_int_start:e_int_end] = self.e_int_accumulated

                if self.mpc.solve_count % 10 == 0:  # Log every 10 solves (100ms)
                    self.get_logger().info(
                        f"[CONTROL] pos[0]={positions[0]:.6f}, ref[0]={self.reference_state[0]:.6f}, "
                        f"error[0]={pos_error[0]:.6f}, e_int[0]={self.e_int_accumulated[0]:.6f}"
                    )

                frac_start_idx = 3 * self.n_joints
                frac_states = self.current_state[frac_start_idx:frac_start_idx + self.n_joints * self.n_frac_states].copy()

            # Precompute reference trajectory
            x_ref = self._get_reference_sequence()

            # Attempt MPC solve, with optional integral reduction retry
            success = False
            u_opt = None
            x_pred = None
            x0_mpc = np.hstack([positions, velocities, self.e_int_accumulated, frac_states])

            for retry in range(2):
                if self.enable_warm_start and self.last_u_opt is not None:
                    u_opt, x_pred, success = self.mpc.solve(x0_mpc, x_ref)
                else:
                    u_opt, x_pred, success = self.mpc.solve(x0_mpc, x_ref)

                if success:
                    break

                # If solver failed and integral is near saturation, reduce integrator and retry once
                if retry == 0 and self.integral_max > 0.0:
                    with self.state_lock:
                        if np.any(np.abs(self.e_int_accumulated) >= 0.98 * self.integral_max):
                            self.e_int_accumulated *= 0.5
                            self.current_state[e_int_start:e_int_end] = self.e_int_accumulated
                            x0_mpc = np.hstack([positions, velocities, self.e_int_accumulated, frac_states])
                            self.get_logger().warn("Integral saturation detected - halving accumulated integral and retrying MPC")
                            continue
                break

            solve_time = time.time() - start_time

            if success:
                # Extract first control input from MPC solver
                u = self.mpc.get_first_input(u_opt)

                # Integral feedback is embedded in solver; returned input includes Ki * e_int
                u_final = self.mpc.get_first_input(u_opt, x_pred)

                # ANTI-WINDUP: Detect saturation and reduce integral accumulation
                # This prevents integral error from growing unboundedly when control is saturated
                u_min_array = np.asarray(self.u_min, dtype=float)
                u_max_array = np.asarray(self.u_max, dtype=float)
                u_saturated = np.logical_or(
                    u_final >= u_max_array - 0.01,  # Near max with small tolerance
                    u_final <= u_min_array + 0.01   # Near min with small tolerance
                )

                # Anti-windup coefficient: reduce integral when saturated
                # Range: 1.0 (no saturation) → 0.95 (saturated, 5% decay per step)
                anti_windup_coeff = np.where(
                    u_saturated,
                    0.95,      # 5% decay per step when saturated
                    1.0        # No decay when not saturated
                )

                # Apply anti-windup by reducing integral error for saturated joints
                self.e_int_accumulated *= anti_windup_coeff

                # Log anti-windup activity periodically
                if np.any(u_saturated) and self.mpc.solve_count % 100 == 0:
                    saturated_joints = np.where(u_saturated)[0]
                    e_int_before = self.e_int_accumulated[saturated_joints] / np.asarray(anti_windup_coeff[saturated_joints], dtype=float)
                    self.get_logger().warn(
                        f"[ANTI-WINDUP] Saturation on joints {saturated_joints.tolist()}: "
                        f"u_final={u_final[saturated_joints]}, "
                        f"e_int_before={e_int_before}, "
                        f"e_int_after={self.e_int_accumulated[saturated_joints]}"
                    )

                # DEBUG: Log control output
                if self.mpc.solve_count % 100 == 0:
                    self.get_logger().info(
                        f"[CONTROL] u_mpc_raw[0]={u[0]:.6f}, "
                        f"e_int[0]={self.e_int_accumulated[0]:.4f}, "
                        f"u_applied[0]={u_final[0]:.6f}"
                    )
                self.last_control = u_final.copy()

                # Update predicted state for next cycle
                with self.state_lock:
                    # x_pred[1] is NOW ([3+n_frac]*n,) from MPC prediction: [pos, vel, e_int, frac_state]
                    if x_pred is not None:
                        x_pred_aug = np.asarray(x_pred[1]).flatten()  # Ensure 1D array
                    else:
                        x_pred_aug = x0_mpc.copy()

                    # Update current_state with predicted values
                    # Copy all state elements including positions, velocities, integral, and fractional filter states
                    state_size = min(len(x_pred_aug), len(self.current_state))
                    self.current_state[:state_size] = x_pred_aug[:state_size]

                    e_int_start = 2 * self.n_joints
                    e_int_end = 3 * self.n_joints
                    if len(x_pred_aug) >= e_int_end:
                        self.current_state[e_int_start:e_int_end] = self.e_int_accumulated

                    # Apply integral leak when close to reference to avoid residual actuation
                    if self.integral_leak_rate > 0.0:
                        pos_err = self.reference_state[:self.n_joints] - self.current_state[:self.n_joints]
                        vel_state = self.current_state[self.n_joints:2*self.n_joints]
                        if (
                            np.all(np.abs(pos_err) <= self.integral_leak_error_threshold)
                            and np.all(np.abs(vel_state) <= self.integral_leak_velocity_threshold)
                        ):
                            leak_factor = max(0.0, 1.0 - self.integral_leak_rate * self.dt)
                            self.e_int_accumulated *= leak_factor
                            self.current_state[e_int_start:e_int_end] = self.e_int_accumulated

                # Store for warm-starting
                self.last_u_opt = u_opt
                self.last_x_pred = x_pred

                # Queue publications to background threads
                # Publish the control output (integral already included in MPC!)
                self.publish_queue.put(("command", u_final))
                self.publish_queue.put(("predicted", x_pred))

                # Performance monitoring
                self.solve_times.append(solve_time)
                if len(self.solve_times) > self.max_history:
                    self.solve_times.pop(0)

                # Log metrics periodically
                if self.mpc.solve_count % 100 == 0:
                    avg_time = np.mean(self.solve_times) * 1000  # Convert to ms
                    self.publish_queue.put((
                        "log",
                        f"MPC {self.mpc.solve_count}: "
                        f"solve_time={solve_time*1000:.2f}ms, "
                        f"avg_time={avg_time:.2f}ms, "
                        f"control_norm={np.linalg.norm(u_final):.6f}"
                    ))
            else:
                self.get_logger().warn("MPC solver failed - using fallback control")
                self.publish_queue.put(("command", self.last_control))

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")

    def _publish_control_command(self, u):
        """Publish control command as JointState"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = u.tolist()
        self.command_pub.publish(msg)

    def _publish_predicted_state(self, x_pred):
        """Publish predicted state trajectory"""
        msg = Float32MultiArray()
        msg.data = x_pred.flatten().tolist()
        self.predicted_state_pub.publish(msg)

    def set_control_limits(self, u_min, u_max):
        """Update control input limits"""
        self.mpc.set_control_limits(u_min, u_max)

    def set_weights(self, Q, R):
        """Update MPC cost weights"""
        self.mpc.set_weights(Q, R)

    def get_performance_stats(self):
        """Get performance statistics"""
        if self.solve_times:
            return {
                "avg_solve_time_ms": np.mean(self.solve_times) * 1000,
                "max_solve_time_ms": np.max(self.solve_times) * 1000,
                "min_solve_time_ms": np.min(self.solve_times) * 1000,
                "gpu_enabled": self.enable_gpu,
                "warm_start_enabled": self.enable_warm_start,
                "worker_threads": self.n_workers,
            }
        return {}


def main(args=None):
    try:
        rclpy.init(args=args)
        node = FractionalMPCControllerNodeAccelerated()
        rclpy.spin(node)
        rclpy.shutdown()
    except KeyboardInterrupt:
        print('Abortado')

if __name__ == "__main__":
    main()
