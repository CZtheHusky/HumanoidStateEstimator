#!/usr/bin/env python3
"""
IMU Preintegrator Validation Script

Validates the ImuPreintegrator class in isolation using
NPZ sensor logs or hardware logs.

Usage:
    python validate_imu_preintegrator.py --log data.npz --fps 100
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm

# Import from state_estimator
from state_estimator.state_estimator import (
    ImuPreintegrator,
    EstimatorInput,
    ImuMeasurement,
    JointMeasurement, # Needed for EstimatorInput type hint but not used
)

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Install with: pip install plotly")
from utils import load_streaming_pickle, load_npz_log_with_slice, load_hw_log_to_npz_like_for_estimator

def plot_results_matplotlib(t, raw_accel, integrated_accel, output_path):
    """
    Plot results using Matplotlib.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    labels = ["Accel X", "Accel Y", "Accel Z"]
    
    for i in range(3):
        ax = axes[i]
        ax.plot(t, raw_accel[:, i], label="Raw Input", alpha=0.5, linewidth=1)
        ax.plot(t, integrated_accel[:, i], label="Pre-integrated", linewidth=2)
        ax.set_ylabel(f"{labels[i]} (m/s²)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    axes[2].set_xlabel("Time (s)")
    plt.suptitle("IMU Preintegration Validation")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    print(f"Matplotlib figure saved to: {output_path}")

def plot_results_plotly(t, raw_accel, integrated_accel, output_path):
    """
    Plot results using Plotly.
    """
    if not PLOTLY_AVAILABLE:
        return

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Accel X", "Accel Y", "Accel Z"))

    labels = ["x", "y", "z"]
    colors_raw = ["#aec7e8", "#ffbb78", "#98df8a"]
    colors_int = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i in range(3):
        # Raw data
        fig.add_trace(
            go.Scatter(x=t, y=raw_accel[:, i], mode='lines', name=f'Raw {labels[i]}',
                       line=dict(color=colors_raw[i], width=1), opacity=0.7),
            row=i+1, col=1
        )
        # Integrated data
        fig.add_trace(
            go.Scatter(x=t, y=integrated_accel[:, i], mode='lines', name=f'Integrated {labels[i]}',
                       line=dict(color=colors_int[i], width=2)),
            row=i+1, col=1
        )
        fig.update_yaxes(title_text="m/s²", row=i+1, col=1)

    fig.update_layout(title_text="IMU Preintegration Validation", height=900)
    fig.write_html(output_path)
    print(f"Plotly figure saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Validate ImuPreintegrator")
    parser.add_argument("--log", type=str, required=True, help="Path to log file (.npz or .pkl)")
    parser.add_argument("--fps", type=float, default=100.0, help="Target frequency for ImuPreintegrator (Hz)")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=-1, help="End index")
    parser.add_argument("--legacy_accl", action="store_true", help="Use legacy IMU acceleration reconstruction")
    parser.add_argument("--output_dir", type=str, default="validate_results_imu", help="Output directory")
    
    args = parser.parse_args()
    
    log_path = Path(args.log)
    
    # Load data
    if log_path.suffix == ".npz":
        log_data = load_npz_log_with_slice(log_path, start=args.start, end=args.end)
    else:
        log_data = load_hw_log_to_npz_like_for_estimator(log_path, start=args.start, end=args.end)
        
    # Prepare data arrays
    t = log_data["t"]
    ticks = log_data["tick"]
    
    if args.legacy_accl:
        raw_accel = log_data["imu_accel_legacy"]
        print("Using Legacy Accel (Reconstructed from Velocity)")
    else:
        raw_accel = log_data["imu_accel"]
        print("Using Raw IMU Accel")
        
    imu_quat = log_data["imu_quat_wxyz"]
    imu_gyro = log_data.get("imu_gyro", np.zeros_like(raw_accel)) # Gyro not strictly needed for accel integration test but required for struct
    
    # Initialize Integrator
    preintegrator = ImuPreintegrator()
    
    # Simulation parameters
    target_dt_ms = 1000.0 / args.fps
    tick_accumulator = target_dt_ms # Trigger immediately or wait? Let's wait for first interval.
    
    # Output buffers
    integrated_accel_hist = np.zeros_like(raw_accel)
    
    # State variables
    last_integrated_val = np.zeros(3)
    last_tick = None
    
    print(f"Running ImuPreintegrator at {args.fps} Hz (dt={target_dt_ms:.2f} ms)...")
    
    for k in tqdm(range(len(t))):
        current_tick = int(ticks[k])
        
        # Calculate dt_tick
        if last_tick is None:
            dt_tick = 0
        else:
            dt_tick = current_tick - last_tick
            
        last_tick = current_tick
        
        # Construct input
        # Note: JointMeasurement is required by EstimatorInput but not used by ImuPreintegrator
        # We create dummy joint data
        dummy_joint = JointMeasurement(
            position=np.array([]), velocity=np.array([]), torque=np.array([])
        )
        
        imu_meas = ImuMeasurement(
            quat_wxyz=imu_quat[k],
            accel_body=raw_accel[k],
            gyro_body=imu_gyro[k]
        )
        
        est_input = EstimatorInput(
            joint=dummy_joint,
            imu=imu_meas,
            tick=current_tick
        )
        
        # Push to integrator
        preintegrator.push(est_input)
        # print("Push:", imu_meas.accel_body)
        
        # Update accumulator
        tick_accumulator -= dt_tick
        
        # Check if we should trigger integration output
        if tick_accumulator <= 0:
            # Get integral and reset
            integral = preintegrator.get_integral(reset=True)
            # print("Integral:", integral.accel_body)
            
            if integral is not None:
                last_integrated_val = integral.accel_body
            
            # Reset accumulator (add target_dt_ms to maintain phase, or reset to target_dt_ms?)
            # StateEstimator uses: self._imu_tick_increment = int(self._imu_dt_ms)
            # This resets the counter completely, ignoring overshoot. We'll do the same.
            tick_accumulator = target_dt_ms
            
        # Store result (fill with last known value)
        integrated_accel_hist[k] = last_integrated_val

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # merge the last two parts of the log path as the ds name
    filename_base = f"{log_path.parts[-2]}_{log_path.stem}_{args.fps}Hz_{timestamp}"
    
    # Plotting
    plot_results_matplotlib(t, raw_accel, integrated_accel_hist, 
                          output_dir / f"{filename_base}_matplotlib.png")
    
    plot_results_plotly(t, raw_accel, integrated_accel_hist, 
                        output_dir / f"{filename_base}_plotly.html")
    
    print("Done.")

if __name__ == "__main__":
    main()
