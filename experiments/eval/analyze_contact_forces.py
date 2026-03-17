"""Analyze and plot contact forces (GRF and JRF) over the gait cycle.

This script:
1) Reads a CSV from record_contact_forces.py.
2) Detects gait cycles based on vertical GRF (Fz).
3) Resamples data to 0-100% gait phase.
4) Plots mean +/- std bands for GRFs and JRFs.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d

def detect_cycles_from_grf(df, side='r', threshold=50.0):
    """Detect gait cycles based on heel-strike (start of stance phase)."""
    # Use absolute value for detection to be robust to sign conventions
    grf_z = np.abs(df[f'grf_{side}_z'].values)
    time = df['time'].values
    
    # Simple thresholding to find stance phases
    is_stance = grf_z > threshold
    
    # Find rising edges (heel strike)
    diff = np.diff(is_stance.astype(int))
    heel_strikes = np.where(diff == 1)[0]
    
    cycles = []
    for i in range(len(heel_strikes) - 1):
        start_idx = heel_strikes[i]
        end_idx = heel_strikes[i+1]
        cycles.append((start_idx, end_idx))
    
    return cycles

def resample_cycle(data, n_points=101):
    x = np.linspace(0, 1, len(data))
    f = interp1d(x, data, kind='linear')
    return f(np.linspace(0, 1, n_points))

def plot_force_bands(ax, cycles_data, label, color):
    # cycles_data: list of arrays, each array is one cycle resampled to 101 points
    mean = np.mean(cycles_data, axis=0)
    std = np.std(cycles_data, axis=0)
    phase = np.linspace(0, 100, len(mean))
    
    ax.plot(phase, mean, label=label, color=color, linewidth=2)
    ax.fill_between(phase, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=str, help="Path to contact_analysis.csv")
    p.add_argument("--out", type=str, default="contact_plots.png")
    p.add_argument("--threshold", type=float, default=50.0, help="GRF threshold for cycle detection")
    p.add_argument("--side", type=str, default='r', choices=['r', 'l'], help="Side to use for gait cycle detection")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    
    # Pre-calculate magnitudes if they don't exist
    if 'grf_l_mag' not in df.columns:
        df['grf_l_mag'] = np.sqrt(df['grf_l_x']**2 + df['grf_l_y']**2 + df['grf_l_z']**2)
    
    # Detect cycles based on selected side heel strike
    cycles = detect_cycles_from_grf(df, side=args.side, threshold=args.threshold)
    
    if len(cycles) < 2:
        other_side = 'l' if args.side == 'r' else 'r'
        print(f"No cycles detected with side '{args.side}'. Trying other side '{other_side}'...")
        cycles = detect_cycles_from_grf(df, side=other_side, threshold=args.threshold)

    print(f"Detected {len(cycles)} gait cycles.")

    if len(cycles) < 2:
        print("Error: Not enough cycles detected. Try lowering --threshold.")
        return

    # Create a single plot for force magnitudes
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # 1. Ground Reaction Force (Magnitude)
    grf_l_cycles = [resample_cycle(df['grf_l_mag'].iloc[s:e].values) for s, e in cycles]
    plot_force_bands(ax, grf_l_cycles, "GRF Left (Mag)", "tab:red")

    # 2. Joint Reaction Forces (Magnitude)
    joints = [('hip_l', 'tab:green'), ('knee_l', 'tab:blue'), ('ankle_l', 'tab:purple')]
    for joint, color in joints:
        col_name = f'jrf_{joint}_mag'
        jrf_cycles = [resample_cycle(df[col_name].iloc[s:e].values) for s, e in cycles]
        plot_force_bands(ax, jrf_cycles, f"JRF {joint.replace('_', ' ').capitalize()} (Mag)", color)
    
    ax.set_ylabel("Force (N)")
    ax.set_xlabel("Gait Phase (%)")
    ax.set_title("Force Magnitudes (Left Leg Only)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Plots saved to {args.out}")

if __name__ == "__main__":
    main()
