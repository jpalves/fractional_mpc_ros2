#!/usr/bin/env python3
"""
Reference Generator Node for ROS2 - Step, Ramp, and Impulse Responses

Provides:
- Step response: sudden change in reference
- Ramp response: linear change in reference
- Impulse response: brief spike in reference
- Custom trajectories

Publishes to: /reference_command (std_msgs/Float32MultiArray)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray
from enum import Enum
import time


class ReferenceType(Enum):
    """Types of reference signals"""
    STEP = "step"
    RAMP = "ramp"
    IMPULSE = "impulse"
    SINE = "sine"
    CUSTOM = "custom"
    IDLE = "idle"


class ReferenceGenerator(Node):
    """Generate reference trajectories for testing MPC controller"""

    def __init__(self):
        super().__init__("reference_generator")

        # Parameters
        self.declare_parameter("n_joints", 9)
        self.declare_parameter("publish_rate", 100)  # Hz
        self.declare_parameter("reference_type", "step")
        self.declare_parameter("step_amplitude", 1.0)
        self.declare_parameter("step_time", 0.5)  # seconds
        self.declare_parameter("ramp_rate", 0.5)  # rad/s
        self.declare_parameter("ramp_duration", 5.0)  # seconds
        self.declare_parameter("impulse_amplitude", 2.0)
        self.declare_parameter("impulse_duration", 0.1)  # seconds
        self.declare_parameter("sine_amplitude", 0.5)
        self.declare_parameter("sine_frequency", 1.0)  # Hz

        # Get parameters
        self.n_joints = self.get_parameter("n_joints").value
        self.publish_rate = self.get_parameter("publish_rate").value
        self.reference_type = self.get_parameter("reference_type").value
        self.step_amplitude = self.get_parameter("step_amplitude").value
        self.step_time = self.get_parameter("step_time").value
        self.ramp_rate = self.get_parameter("ramp_rate").value
        self.ramp_duration = self.get_parameter("ramp_duration").value
        self.impulse_amplitude = self.get_parameter("impulse_amplitude").value
        self.impulse_duration = self.get_parameter("impulse_duration").value
        self.sine_amplitude = self.get_parameter("sine_amplitude").value
        self.sine_frequency = self.get_parameter("sine_frequency").value

        # Publisher
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.ref_publisher = self.create_publisher(
            Float32MultiArray,
            "/reference_command",
            qos
        )

        # Timer
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)
        self.start_time = time.time()

        # Logging
        self.get_logger().info(
            f"Reference Generator initialized: "
            f"type={self.reference_type}, n_joints={self.n_joints}, "
            f"rate={self.publish_rate}Hz"
        )

    def timer_callback(self):
        """Publish reference signal at regular intervals"""
        elapsed = time.time() - self.start_time

        # Generate reference based on type
        if self.reference_type == "step":
            reference = self._generate_step(elapsed)
        elif self.reference_type == "ramp":
            reference = self._generate_ramp(elapsed)
        elif self.reference_type == "impulse":
            reference = self._generate_impulse(elapsed)
        elif self.reference_type == "sine":
            reference = self._generate_sine(elapsed)
        else:
            reference = np.zeros(self.n_joints)

        # Publish
        msg = Float32MultiArray()
        msg.data = reference.astype(np.float32).tolist()
        self.ref_publisher.publish(msg)

        # Log periodically
        if int(elapsed * self.publish_rate) % 100 == 0:
            self.get_logger().info(
                f"[{self.reference_type.upper()}] t={elapsed:.2f}s, "
                f"ref[0]={reference[0]:.4f}, ref[1]={reference[1]:.4f}"
            )

    def _generate_step(self, elapsed):
        """
        Step response: constant value after t_step

        ref(t) = 0 if t < t_step
                 amplitude if t >= t_step
        """
        reference = np.zeros(self.n_joints)

        if elapsed >= self.step_time:
            reference[:] = self.step_amplitude

        return reference

    def _generate_ramp(self, elapsed):
        """
        Ramp response: linear change at constant rate

        ref(t) = ramp_rate * t  (until ramp_duration)
                 ramp_rate * ramp_duration (after)
        """
        reference = np.zeros(self.n_joints)

        if elapsed <= self.ramp_duration:
            reference[:] = self.ramp_rate * elapsed
        else:
            reference[:] = self.ramp_rate * self.ramp_duration

        return reference

    def _generate_impulse(self, elapsed):
        """
        Impulse response: brief spike followed by zero

        ref(t) = amplitude if t_start < t < t_start + duration
                 0 otherwise
        """
        reference = np.zeros(self.n_joints)

        # Impulse starts after 0.5 seconds
        t_start = 0.5
        t_end = t_start + self.impulse_duration

        if t_start < elapsed < t_end:
            reference[:] = self.impulse_amplitude

        return reference

    def _generate_sine(self, elapsed):
        """
        Sine wave response for frequency analysis

        ref(t) = amplitude * sin(2*pi*frequency*t)
        """
        reference = np.zeros(self.n_joints)
        reference[:] = self.sine_amplitude * np.sin(
            2 * np.pi * self.sine_frequency * elapsed
        )

        return reference

    def set_reference_type(self, ref_type):
        """Change reference type at runtime"""
        if ref_type in [e.value for e in ReferenceType]:
            self.reference_type = ref_type
            self.start_time = time.time()
            self.get_logger().info(f"Reference type changed to: {ref_type}")
        else:
            self.get_logger().warn(f"Unknown reference type: {ref_type}")

    def set_step_parameters(self, amplitude, time):
        """Configure step response parameters"""
        self.step_amplitude = amplitude
        self.step_time = time
        self.get_logger().info(
            f"Step parameters updated: amplitude={amplitude}, time={time}s"
        )

    def set_ramp_parameters(self, rate, duration):
        """Configure ramp response parameters"""
        self.ramp_rate = rate
        self.ramp_duration = duration
        self.get_logger().info(
            f"Ramp parameters updated: rate={rate} rad/s, duration={duration}s"
        )

    def set_impulse_parameters(self, amplitude, duration):
        """Configure impulse response parameters"""
        self.impulse_amplitude = amplitude
        self.impulse_duration = duration
        self.get_logger().info(
            f"Impulse parameters updated: amplitude={amplitude}, duration={duration}s"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ReferenceGenerator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
