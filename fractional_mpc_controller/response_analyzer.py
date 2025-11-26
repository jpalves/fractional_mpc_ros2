#!/usr/bin/env python3
"""
Response Analyzer Node for ROS2 - Capture and Analyze System Response

Records:
- Reference signal
- Joint state (position, velocity)
- Control output
- Integral error
- Performance metrics (overshoot, settling time, etc.)

Subscribes to:
- /joint_states (sensor_msgs/JointState)
- /reference_command (std_msgs/Float32MultiArray)
- /control_command (std_msgs/Float32MultiArray)

Saves data to CSV and generates plots
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
import csv
from datetime import datetime
from pathlib import Path
import json


class ResponseAnalyzer(Node):
    """Capture and analyze system response to reference signals"""

    def __init__(self):
        super().__init__("response_analyzer")

        # Parameters
        self.declare_parameter("data_dir", "/tmp/mpc_responses")
        self.declare_parameter("n_joints", 9)
        self.declare_parameter("recording_duration", 30.0)  # seconds
        self.declare_parameter("enable_csv_export", True)
        self.declare_parameter("enable_json_export", True)

        # Get parameters
        self.data_dir = Path(self.get_parameter("data_dir").value)
        self.n_joints = self.get_parameter("n_joints").value
        self.recording_duration = self.get_parameter("recording_duration").value
        self.enable_csv = self.get_parameter("enable_csv_export").value
        self.enable_json = self.get_parameter("enable_json_export").value

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.timestamps = []
        self.references = []
        self.positions = []
        self.velocities = []
        self.controls = []
        self.recording = False
        self.start_time = None

        # Subscribers
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            qos
        )

        self.reference_sub = self.create_subscription(
            Float32MultiArray,
            "/reference_command",
            self.reference_callback,
            qos
        )

        self.control_sub = self.create_subscription(
            Float32MultiArray,
            "/control_command",
            self.control_callback,
            qos
        )

        # Timer for checking recording status
        self.timer = self.create_timer(0.1, self.check_recording_status)

        self.last_reference = np.zeros(self.n_joints)
        self.last_position = np.zeros(self.n_joints)
        self.last_velocity = np.zeros(self.n_joints)
        self.last_control = np.zeros(self.n_joints)

        self.get_logger().info(
            f"Response Analyzer initialized: "
            f"data_dir={self.data_dir}, n_joints={self.n_joints}"
        )

    def joint_state_callback(self, msg: JointState):
        """Capture joint state (position and velocity)"""
        if len(msg.position) >= self.n_joints:
            self.last_position = np.array(msg.position[:self.n_joints])
            self.last_velocity = np.array(msg.velocity[:self.n_joints])

            if self.recording:
                self._record_sample()

    def reference_callback(self, msg: Float32MultiArray):
        """Capture reference command"""
        if len(msg.data) >= self.n_joints:
            self.last_reference = np.array(msg.data[:self.n_joints])

    def control_callback(self, msg: Float32MultiArray):
        """Capture control output"""
        if len(msg.data) >= self.n_joints:
            self.last_control = np.array(msg.data[:self.n_joints])

    def _record_sample(self):
        """Record current state sample"""
        if self.start_time is None:
            self.start_time = rclpy.clock.Clock().now().nanoseconds / 1e9

        elapsed = rclpy.clock.Clock().now().nanoseconds / 1e9 - self.start_time

        self.timestamps.append(elapsed)
        self.references.append(self.last_reference.copy())
        self.positions.append(self.last_position.copy())
        self.velocities.append(self.last_velocity.copy())
        self.controls.append(self.last_control.copy())

    def check_recording_status(self):
        """Check if recording should stop"""
        if self.recording and self.start_time is not None:
            elapsed = rclpy.clock.Clock().now().nanoseconds / 1e9 - self.start_time
            if elapsed >= self.recording_duration:
                self.stop_recording()

    def start_recording(self):
        """Start recording data"""
        self.recording = True
        self.timestamps.clear()
        self.references.clear()
        self.positions.clear()
        self.velocities.clear()
        self.controls.clear()
        self.start_time = None
        self.get_logger().info(f"Recording started for {self.recording_duration}s")

    def stop_recording(self):
        """Stop recording and export data"""
        self.recording = False

        if len(self.timestamps) == 0:
            self.get_logger().warn("No data recorded")
            return

        self.get_logger().info(
            f"Recording stopped. Captured {len(self.timestamps)} samples "
            f"over {self.timestamps[-1]:.2f}s"
        )

        # Export data
        if self.enable_csv:
            self._export_csv()
        if self.enable_json:
            self._export_json()

        # Calculate metrics
        metrics = self._calculate_metrics()
        self._print_metrics(metrics)

    def _export_csv(self):
        """Export data to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"response_{timestamp}.csv"

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = ["time"]
            for i in range(self.n_joints):
                header.extend([
                    f"ref_{i}", f"pos_{i}", f"vel_{i}", f"ctrl_{i}"
                ])
            writer.writerow(header)

            # Data
            for idx, t in enumerate(self.timestamps):
                row = [t]
                for i in range(self.n_joints):
                    row.extend([
                        self.references[idx][i],
                        self.positions[idx][i],
                        self.velocities[idx][i],
                        self.controls[idx][i]
                    ])
                writer.writerow(row)

        self.get_logger().info(f"Data exported to {filename}")

    def _export_json(self):
        """Export data to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"response_{timestamp}.json"

        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "n_joints": self.n_joints,
                "n_samples": len(self.timestamps),
                "duration": self.timestamps[-1] if self.timestamps else 0,
            },
            "time": self.timestamps,
            "reference": [r.tolist() for r in self.references],
            "position": [p.tolist() for p in self.positions],
            "velocity": [v.tolist() for v in self.velocities],
            "control": [c.tolist() for c in self.controls],
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        self.get_logger().info(f"Data exported to {filename}")

    def _calculate_metrics(self):
        """Calculate performance metrics"""
        metrics = {}

        for i in range(self.n_joints):
            ref = np.array([r[i] for r in self.references])
            pos = np.array([p[i] for p in self.positions])
            error = ref - pos

            # Steady-state error
            ss_error = np.mean(error[-100:]) if len(error) > 100 else error[-1]

            # Overshoot
            if np.max(ref) > 0:
                overshoot = (np.max(pos) - np.max(ref)) / np.max(ref) * 100
            else:
                overshoot = 0

            # Settling time (2% criterion)
            settling_idx = len(pos)
            for idx in range(len(pos) - 1, 0, -1):
                if np.abs(error[idx]) > 0.02 * np.max(np.abs(ref)) + 1e-6:
                    settling_idx = idx
                    break
            settling_time = self.timestamps[settling_idx] if settling_idx < len(self.timestamps) else 0

            # Rise time (10% to 90%)
            ref_range = np.max(ref) - np.min(ref)
            if ref_range > 0:
                lower = np.min(ref) + 0.1 * ref_range
                upper = np.min(ref) + 0.9 * ref_range
                rising_idx = np.where(pos >= lower)[0]
                if len(rising_idx) > 0:
                    t_lower = self.timestamps[rising_idx[0]]
                    falling_idx = np.where(pos >= upper)[0]
                    if len(falling_idx) > 0:
                        t_upper = self.timestamps[falling_idx[0]]
                        rise_time = t_upper - t_lower
                    else:
                        rise_time = 0
                else:
                    rise_time = 0
            else:
                rise_time = 0

            metrics[f"joint_{i}"] = {
                "steady_state_error": float(ss_error),
                "overshoot_percent": float(overshoot),
                "settling_time": float(settling_time),
                "rise_time": float(rise_time),
                "max_error": float(np.max(np.abs(error))),
                "rms_error": float(np.sqrt(np.mean(error ** 2))),
            }

        return metrics

    def _print_metrics(self, metrics):
        """Print metrics to log"""
        self.get_logger().info("=" * 60)
        self.get_logger().info("RESPONSE ANALYSIS METRICS")
        self.get_logger().info("=" * 60)

        for joint, data in metrics.items():
            self.get_logger().info(f"\n{joint}:")
            self.get_logger().info(
                f"  Steady-state error: {data['steady_state_error']:.4f} rad"
            )
            self.get_logger().info(
                f"  Overshoot: {data['overshoot_percent']:.2f}%"
            )
            self.get_logger().info(
                f"  Settling time (2%): {data['settling_time']:.2f}s"
            )
            self.get_logger().info(
                f"  Rise time (10-90%): {data['rise_time']:.2f}s"
            )
            self.get_logger().info(
                f"  Max error: {data['max_error']:.4f} rad"
            )
            self.get_logger().info(
                f"  RMS error: {data['rms_error']:.4f} rad"
            )

        self.get_logger().info("=" * 60)


def main(args=None):
    rclpy.init(args=args)
    node = ResponseAnalyzer()

    # Start recording immediately
    node.start_recording()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_recording()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
