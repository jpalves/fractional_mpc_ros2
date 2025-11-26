#!/usr/bin/env python3
"""
Plot Response Data - Visualize system response to step, ramp, and impulse

Usage:
    python3 plot_responses.py <path_to_data.json>

Or analyze multiple files:
    python3 plot_responses.py /tmp/mpc_responses/response_*.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime


def load_response_data(filepath):
    """Load response data from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def plot_single_response(data, output_dir=None):
    """Plot single response file"""
    timestamps = np.array(data['time'])
    references = np.array(data['reference'])
    positions = np.array(data['position'])
    velocities = np.array(data['velocity'])
    controls = np.array(data['control'])

    n_joints = data['metadata']['n_joints']
    timestamp_str = data['metadata']['timestamp'].split('T')[0]

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(
        f"System Response - {timestamp_str}\n"
        f"n_joints={n_joints}, duration={timestamps[-1]:.2f}s",
        fontsize=14, fontweight='bold'
    )

    # Plot 1: Reference vs Position
    ax = axes[0]
    for i in range(min(3, n_joints)):
        ax.plot(timestamps, references[:, i], 'r--', label=f'ref_{i}', linewidth=2)
        ax.plot(timestamps, positions[:, i], 'b-', label=f'pos_{i}', linewidth=1.5)
    ax.set_ylabel('Position (rad)', fontsize=11)
    ax.set_title('Reference Tracking', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Velocity
    ax = axes[1]
    for i in range(min(3, n_joints)):
        ax.plot(timestamps, velocities[:, i], 'g-', label=f'vel_{i}', linewidth=1.5)
    ax.set_ylabel('Velocity (rad/s)', fontsize=11)
    ax.set_title('Joint Velocity', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Control Output
    ax = axes[2]
    for i in range(min(3, n_joints)):
        ax.plot(timestamps, controls[:, i], 'purple', label=f'u_{i}', linewidth=1.5)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Control (rad or N)', fontsize=11)
    ax.set_title('Control Output', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    if output_dir is None:
        output_dir = Path('/tmp/mpc_responses/plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"response_{timestamp_str}_{datetime.now().strftime('%H%M%S')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved: {filename}")

    return fig


def plot_error_analysis(data, output_dir=None):
    """Plot error analysis"""
    timestamps = np.array(data['time'])
    references = np.array(data['reference'])
    positions = np.array(data['position'])

    n_joints = data['metadata']['n_joints']
    timestamp_str = data['metadata']['timestamp'].split('T')[0]

    # Calculate errors
    errors = references - positions

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f"Error Analysis - {timestamp_str}",
        fontsize=14, fontweight='bold'
    )

    # Plot 1: Error
    ax = axes[0]
    for i in range(min(3, n_joints)):
        ax.plot(timestamps, errors[:, i], 'r-', label=f'error_{i}', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Position Error (rad)', fontsize=11)
    ax.set_title('Tracking Error', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Absolute Error
    ax = axes[1]
    for i in range(min(3, n_joints)):
        ax.semilogy(timestamps, np.abs(errors[:, i]) + 1e-6, 'b-',
                    label=f'|error_{i}|', linewidth=1.5)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Absolute Error (rad, log scale)', fontsize=11)
    ax.set_title('Absolute Error (Log Scale)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    # Save figure
    if output_dir is None:
        output_dir = Path('/tmp/mpc_responses/plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"error_{timestamp_str}_{datetime.now().strftime('%H%M%S')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Error plot saved: {filename}")

    return fig


def plot_phase_portrait(data, output_dir=None):
    """Plot phase portrait (position vs velocity)"""
    positions = np.array(data['position'])
    velocities = np.array(data['velocity'])

    n_joints = data['metadata']['n_joints']
    timestamp_str = data['metadata']['timestamp'].split('T')[0]

    fig, axes = plt.subplots(1, min(3, n_joints), figsize=(15, 4))
    if min(3, n_joints) == 1:
        axes = [axes]

    fig.suptitle(
        f"Phase Portrait - {timestamp_str}",
        fontsize=14, fontweight='bold'
    )

    for i in range(min(3, n_joints)):
        ax = axes[i]
        ax.plot(positions[:, i], velocities[:, i], 'b-', linewidth=1)
        ax.plot(positions[0, i], velocities[0, i], 'go', markersize=8, label='start')
        ax.plot(positions[-1, i], velocities[-1, i], 'ro', markersize=8, label='end')
        ax.set_xlabel('Position (rad)', fontsize=10)
        ax.set_ylabel('Velocity (rad/s)', fontsize=10)
        ax.set_title(f'Joint {i}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    if output_dir is None:
        output_dir = Path('/tmp/mpc_responses/plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"phase_{timestamp_str}_{datetime.now().strftime('%H%M%S')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Phase portrait saved: {filename}")

    return fig


def print_metrics(data):
    """Print metrics from data"""
    timestamps = np.array(data['time'])
    references = np.array(data['reference'])
    positions = np.array(data['position'])
    velocities = np.array(data['velocity'])

    n_joints = data['metadata']['n_joints']

    print("\n" + "=" * 70)
    print("RESPONSE METRICS")
    print("=" * 70)

    for i in range(min(3, n_joints)):
        ref = references[:, i]
        pos = positions[:, i]
        vel = velocities[:, i]
        error = ref - pos

        # Metrics
        ss_error = np.mean(error[-100:]) if len(error) > 100 else error[-1]
        max_error = np.max(np.abs(error))
        rms_error = np.sqrt(np.mean(error ** 2))

        # Overshoot
        if np.max(ref) > 0:
            overshoot = (np.max(pos) - np.max(ref)) / np.max(ref) * 100
        else:
            overshoot = 0

        # Rise time (10% to 90%)
        ref_range = np.max(ref) - np.min(ref)
        rise_time = 0
        if ref_range > 0:
            lower = np.min(ref) + 0.1 * ref_range
            upper = np.min(ref) + 0.9 * ref_range
            rising_idx = np.where(pos >= lower)[0]
            if len(rising_idx) > 0:
                t_lower = timestamps[rising_idx[0]]
                falling_idx = np.where(pos >= upper)[0]
                if len(falling_idx) > 0:
                    t_upper = timestamps[falling_idx[0]]
                    rise_time = t_upper - t_lower

        print(f"\nJoint {i}:")
        print(f"  Steady-state error: {ss_error:>10.4f} rad")
        print(f"  Max error:          {max_error:>10.4f} rad")
        print(f"  RMS error:          {rms_error:>10.4f} rad")
        print(f"  Overshoot:          {overshoot:>10.2f}%")
        print(f"  Rise time (10-90%): {rise_time:>10.2f}s")
        print(f"  Peak velocity:      {np.max(np.abs(vel)):>10.4f} rad/s")

    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_responses.py <path_to_data.json>")
        print("Example: python3 plot_responses.py /tmp/mpc_responses/response_*.json")
        sys.exit(1)

    filepath = Path(sys.argv[1])

    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        sys.exit(1)

    print(f"📂 Loading: {filepath}")
    data = load_response_data(filepath)

    # Print metrics
    print_metrics(data)

    # Generate plots
    print("\n📊 Generating plots...")
    plot_single_response(data)
    plot_error_analysis(data)
    plot_phase_portrait(data)

    print("\n✅ Done! Check /tmp/mpc_responses/plots/ for plots")

    # Show plots
    plt.show()


if __name__ == "__main__":
    main()
