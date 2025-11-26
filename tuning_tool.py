#!/usr/bin/env python3
"""
Interactive Tuning Tool for Fractional MPC Controller

Helps systematically tune MPC parameters through testing and analysis.

Usage:
    python3 tuning_tool.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tabulate import tabulate
import subprocess
import time


class TuningTool:
    """Interactive tool for MPC parameter tuning"""

    def __init__(self):
        self.data_dir = Path("/tmp/mpc_responses")
        self.results = []
        self.baseline = None

        # Default parameters
        self.current_config = {
            "q_pos": 600.0,
            "q_vel": 12.0,
            "r": 0.15,
            "q_int": 1.0,
            "integral_leak_rate": 18.0,
            "integral_leak_error_threshold": 1.8,
        }

    def print_menu(self):
        """Print main menu"""
        print("\n" + "=" * 70)
        print("FRACTIONAL MPC TUNING TOOL".center(70))
        print("=" * 70)
        print("\n1. Run Diagnostic Tests (30 min)")
        print("2. Run Single Test with Custom Parameters")
        print("3. Run Parameter Sweep")
        print("4. Compare Results")
        print("5. View Tuning Recommendations")
        print("6. Export Results")
        print("7. Exit")
        print("\n" + "-" * 70)

    def run_diagnostic(self):
        """Run diagnostic tests"""
        print("\n" + "=" * 70)
        print("PHASE 1: DIAGNOSTIC TESTS".center(70))
        print("=" * 70)

        tests = [
            {
                "name": "Step Response (1.0 rad)",
                "ref_type": "step",
                "params": {
                    "step_amplitude": 1.0,
                    "step_time": 0.5,
                },
                "duration": 10.0,
            },
            {
                "name": "Ramp Response (0.5 rad/s)",
                "ref_type": "ramp",
                "params": {
                    "ramp_rate": 0.5,
                    "ramp_duration": 10.0,
                },
                "duration": 15.0,
            },
            {
                "name": "Impulse Response (2.0 rad, 0.1s)",
                "ref_type": "impulse",
                "params": {
                    "impulse_amplitude": 2.0,
                    "impulse_duration": 0.1,
                },
                "duration": 5.0,
            },
            {
                "name": "Sine Response (1.0 Hz)",
                "ref_type": "sine",
                "params": {
                    "sine_amplitude": 0.5,
                    "sine_frequency": 1.0,
                },
                "duration": 10.0,
            },
        ]

        results = []

        for test in tests:
            print(f"\n▶ Running: {test['name']}")
            print(f"  Duration: {test['duration']}s")

            result = self._run_test(
                test["ref_type"],
                test["params"],
                test["duration"],
                self.current_config,
            )

            if result:
                results.append((test["name"], result))
                self._print_metrics(test["name"], result)
            else:
                print("  ❌ Test failed")

            print("  Waiting 3s before next test...")
            time.sleep(3)

        self.baseline = results
        print("\n✅ Diagnostic complete!")
        return results

    def run_single_test(self, config=None):
        """Run single test with custom parameters"""
        if config is None:
            config = self._prompt_parameters()

        print(f"\n▶ Running test with: {config}")

        result = self._run_test("step", {"step_amplitude": 1.0, "step_time": 0.5}, 10.0, config)

        if result:
            self._print_metrics("Custom Test", result)
            self.results.append((datetime.now().isoformat(), config, result))
            return result
        return None

    def run_parameter_sweep(self):
        """Run sweep of parameters"""
        print("\n" + "=" * 70)
        print("PARAMETER SWEEP".center(70))
        print("=" * 70)

        # Predefined sweeps
        sweeps = {
            "1": {
                "name": "Position Gain Sweep (q_pos)",
                "param": "q_pos",
                "values": [400, 600, 800, 1000, 1200],
            },
            "2": {
                "name": "Damping Sweep (q_vel)",
                "param": "q_vel",
                "values": [5, 10, 15, 25, 50],
            },
            "3": {
                "name": "Control Cost Sweep (r)",
                "param": "r",
                "values": [0.08, 0.10, 0.15, 0.20, 0.25],
            },
            "4": {
                "name": "Combined Sweep (Recommended)",
                "configs": [
                    {"q_pos": 600, "q_vel": 12, "r": 0.15},
                    {"q_pos": 800, "q_vel": 12, "r": 0.15},
                    {"q_pos": 800, "q_vel": 25, "r": 0.15},
                    {"q_pos": 1000, "q_vel": 20, "r": 0.12},
                    {"q_pos": 1000, "q_vel": 30, "r": 0.10},
                ],
            },
        }

        print("\nAvailable sweeps:")
        for key, sweep in sweeps.items():
            print(f"{key}. {sweep['name']}")

        choice = input("\nSelect sweep (1-4): ").strip()

        if choice not in sweeps:
            print("Invalid choice")
            return

        sweep = sweeps[choice]

        results = []

        if "values" in sweep:
            # Single parameter sweep
            for value in sweep["values"]:
                config = self.current_config.copy()
                config[sweep["param"]] = value

                print(f"\n▶ Testing {sweep['param']}={value}")
                result = self._run_test(
                    "step",
                    {"step_amplitude": 1.0, "step_time": 0.5},
                    10.0,
                    config,
                )

                if result:
                    metrics = self._extract_metrics(result)
                    results.append({
                        "param": value,
                        "overshoot": metrics.get("overshoot", 0),
                        "settling": metrics.get("settling_time", 0),
                        "error": metrics.get("steady_state_error", 0),
                    })
                    print(
                        f"  Overshoot: {metrics.get('overshoot', 0):.2f}%, "
                        f"Settling: {metrics.get('settling_time', 0):.2f}s, "
                        f"Error: {metrics.get('steady_state_error', 0):.4f}"
                    )

                time.sleep(2)

            # Print table
            table_data = [[r["param"], r["overshoot"], r["settling"], r["error"]] for r in results]
            print("\n" + "=" * 60)
            print(tabulate(
                table_data,
                headers=["Value", "Overshoot %", "Settling s", "Error rad"],
                tablefmt="grid"
            ))

        else:
            # Multiple config sweep
            for config in sweep["configs"]:
                print(f"\n▶ Testing config: {config}")
                result = self._run_test(
                    "step",
                    {"step_amplitude": 1.0, "step_time": 0.5},
                    10.0,
                    config,
                )

                if result:
                    metrics = self._extract_metrics(result)
                    results.append((config, metrics))
                    self._print_metrics(str(config), result)

                time.sleep(2)

        return results

    def _run_test(self, ref_type, ref_params, duration, config):
        """Run a single test"""
        try:
            # Build launch command
            cmd = [
                "ros2", "run", "fractional_mpc_controller", "reference_generator",
                "--ros-args",
                f"-p reference_type:={ref_type}",
            ]

            # Add reference parameters
            for key, value in ref_params.items():
                cmd.append(f"-p {key}:={value}")

            # Launch in background
            print("  Launching reference generator...")
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            time.sleep(1)

            # Launch analyzer
            print("  Launching response analyzer...")
            analyzer_cmd = [
                "ros2", "run", "fractional_mpc_controller", "response_analyzer",
                "--ros-args",
                f"-p recording_duration:={duration}",
            ]

            subprocess.Popen(analyzer_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Wait for completion
            time.sleep(duration + 2)

            # Load results
            response_file = max(self.data_dir.glob("response_*.json"), key=lambda p: p.stat().st_mtime)
            with open(response_file, 'r') as f:
                return json.load(f)

        except Exception as e:
            print(f"  Error: {e}")
            return None

    def _extract_metrics(self, data):
        """Extract metrics from response data"""
        timestamps = np.array(data['time'])
        references = np.array(data['reference'])
        positions = np.array(data['position'])
        errors = references - positions

        metrics = {}

        # Focus on first joint
        if len(errors) > 0:
            ref = references[:, 0]
            pos = positions[:, 0]
            error = errors[:, 0]

            # Metrics
            metrics['steady_state_error'] = float(np.mean(np.abs(error[-100:])))
            metrics['max_error'] = float(np.max(np.abs(error)))
            metrics['rms_error'] = float(np.sqrt(np.mean(error ** 2)))

            # Overshoot
            if np.max(ref) > 0:
                metrics['overshoot'] = float((np.max(pos) - np.max(ref)) / np.max(ref) * 100)
            else:
                metrics['overshoot'] = 0

            # Settling time (2% criterion)
            for idx in range(len(error) - 1, 0, -1):
                if np.abs(error[idx]) > 0.02 * (np.max(ref) - np.min(ref)) + 1e-6:
                    metrics['settling_time'] = float(timestamps[idx])
                    break
            else:
                metrics['settling_time'] = 0

        return metrics

    def _print_metrics(self, name, data):
        """Print metrics"""
        metrics = self._extract_metrics(data)
        print(f"\n  {name} - Results:")
        print(f"    Overshoot: {metrics.get('overshoot', 0):.2f}%")
        print(f"    Settling time: {metrics.get('settling_time', 0):.2f}s")
        print(f"    Steady-state error: {metrics.get('steady_state_error', 0):.4f} rad")
        print(f"    Max error: {metrics.get('max_error', 0):.4f} rad")
        print(f"    RMS error: {metrics.get('rms_error', 0):.4f} rad")

    def _prompt_parameters(self):
        """Prompt user for custom parameters"""
        config = self.current_config.copy()

        print("\nCurrent parameters:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        print("\nEnter new values (press Enter to keep current):")
        try:
            q_pos = input(f"q_pos [{config['q_pos']}]: ").strip()
            if q_pos:
                config['q_pos'] = float(q_pos)

            q_vel = input(f"q_vel [{config['q_vel']}]: ").strip()
            if q_vel:
                config['q_vel'] = float(q_vel)

            r = input(f"r [{config['r']}]: ").strip()
            if r:
                config['r'] = float(r)

        except ValueError:
            print("Invalid input, keeping current values")

        return config

    def compare_results(self):
        """Compare multiple test results"""
        if not self.results:
            print("No results to compare")
            return

        print("\n" + "=" * 70)
        print("RESULT COMPARISON".center(70))
        print("=" * 70)

        table_data = []
        for timestamp, config, data in self.results:
            metrics = self._extract_metrics(data)
            table_data.append([
                f"q_pos={config['q_pos']:.0f}",
                f"{metrics.get('overshoot', 0):.2f}",
                f"{metrics.get('settling_time', 0):.2f}",
                f"{metrics.get('steady_state_error', 0):.4f}",
            ])

        print(tabulate(
            table_data,
            headers=["Config", "Overshoot %", "Settling s", "Error rad"],
            tablefmt="grid"
        ))

    def print_recommendations(self):
        """Print tuning recommendations"""
        print("\n" + "=" * 70)
        print("TUNING RECOMMENDATIONS".center(70))
        print("=" * 70)

        print("""
Based on control theory for MPC systems:

IF overshoot is HIGH (>10%):
  • Increase q_vel (damping) by 50%
  • Increase r (control cost) slightly
  • Example: q_vel: 12 → 20

IF error is HIGH (>0.05 rad):
  • Increase q_pos (position gain) by 30-50%
  • Decrease r (control cost) by 20%
  • Example: q_pos: 600 → 800, r: 0.15 → 0.12

IF rise time is SLOW (>1 second):
  • Increase q_pos (position gain)
  • Decrease r (control cost)
  • Increase q_vel slightly to prevent overshoot
  • Example: q_pos: 600 → 900, r: 0.15 → 0.10

IF oscillation exists:
  • Increase q_vel significantly
  • Increase integral_leak_rate
  • Example: q_vel: 12 → 30, integral_leak_rate: 18 → 25

RECOMMENDED STARTING POINTS:

1. For balanced performance (RECOMMENDED):
   q_pos: 900, q_vel: 25, r: 0.12

2. For maximum speed (aggressive):
   q_pos: 1200, q_vel: 20, r: 0.08

3. For smooth response (conservative):
   q_pos: 600, q_vel: 40, r: 0.20

4. For high precision:
   q_pos: 1500, q_vel: 30, r: 0.10
        """)

    def export_results(self):
        """Export results to file"""
        if not self.results:
            print("No results to export")
            return

        filename = Path("/tmp/mpc_responses") / f"tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "config": config,
                    "metrics": self._extract_metrics(data)
                }
                for _, config, data in self.results
            ]
        }

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✅ Results exported to {filename}")

    def run(self):
        """Main loop"""
        while True:
            self.print_menu()
            choice = input("\nSelect option (1-7): ").strip()

            if choice == "1":
                self.run_diagnostic()
            elif choice == "2":
                self.run_single_test()
            elif choice == "3":
                self.run_parameter_sweep()
            elif choice == "4":
                self.compare_results()
            elif choice == "5":
                self.print_recommendations()
            elif choice == "6":
                self.export_results()
            elif choice == "7":
                print("\n👋 Goodbye!")
                break
            else:
                print("Invalid option")


def main():
    tool = TuningTool()
    tool.run()


if __name__ == "__main__":
    main()
