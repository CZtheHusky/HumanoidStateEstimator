#!/usr/bin/env python3
"""
Compare multiple runs of state estimator validation results.

This script reads summary.yaml files from multiple run directories,
extracts metrics, and generates comparison plots.

Usage:
    python compare_runs.py --runs validate_results/run1 validate_results/run2 --output comparison_results
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def load_run_metrics(run_dir: Path) -> Tuple[str, Dict[str, Dict]]:
    """
    Load all metrics from a run directory.
    
    Args:
        run_dir: Path to run directory (e.g., validate_results/data_200_20251205_054813)
    
    Returns:
        run_name: Name of the run
        metrics_dict: Dictionary mapping subproject_name -> metrics
    """
    run_name = run_dir.name
    metrics_dict = {}
    
    # Find all subdirectories containing summary.yaml
    for subdir in sorted(run_dir.iterdir()):
        if subdir.is_dir():
            yaml_path = subdir / "summary.yaml"
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'metrics' in data:
                        # Extract subproject name (e.g., boxing_0.npz -> boxing_0)
                        subproject_name = subdir.name.replace('.npz', '').replace('.pkl', '')
                        metrics_dict[subproject_name] = data['metrics']
    
    print(f"Loaded {len(metrics_dict)} subprojects from run: {run_name}")
    return run_name, metrics_dict


def extract_all_metric_keys(all_runs_data: Dict[str, Dict[str, Dict]]) -> List[str]:
    """
    Extract all unique metric keys across all runs and subprojects.
    
    Args:
        all_runs_data: Dict[run_name -> Dict[subproject_name -> metrics]]
    
    Returns:
        List of metric keys
    """
    metric_keys = set()
    for run_name, subprojects in all_runs_data.items():
        for subproject_name, metrics in subprojects.items():
            metric_keys.update(metrics.keys())
    
    # Remove non-numeric keys (like counts that we'll use for weighting)
    # We'll handle 'num_samples' separately as the weighting factor
    metric_keys = sorted(list(metric_keys))
    return metric_keys


def plot_metric_comparison(
    all_runs_data: Dict[str, Dict[str, Dict]],
    metric_key: str,
    output_path: Path
):
    """
    Plot comparison for a single metric across all runs and subprojects.
    
    Args:
        all_runs_data: Dict[run_name -> Dict[subproject_name -> metrics]]
        metric_key: The metric to plot (e.g., 'rmse_velocity')
        output_path: Where to save the plot
    """
    # Collect all subproject names across all runs
    all_subprojects = set()
    for run_name, subprojects in all_runs_data.items():
        all_subprojects.update(subprojects.keys())
    all_subprojects = sorted(list(all_subprojects))
    
    if not all_subprojects:
        print(f"Warning: No subprojects found for metric {metric_key}")
        return
    
    # Plot bars for each run
    run_names = sorted(all_runs_data.keys())
    
    # Create figure
    n_subprojects = len(all_subprojects)
    n_runs = len(run_names)
    
    fig, ax = plt.subplots(figsize=(max(12, n_subprojects * 0.8), 6))
    
    # Set up bar positions
    x = np.arange(n_subprojects)
    width = 0.8 / n_runs  # Width of each bar
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, run_name in enumerate(run_names):
        subprojects = all_runs_data[run_name]
        values = []
        
        for subproject_name in all_subprojects:
            if subproject_name in subprojects and metric_key in subprojects[subproject_name]:
                value = subprojects[subproject_name][metric_key]
                # Handle potential None values
                values.append(value if value is not None else 0)
            else:
                values.append(0)
        
        # Calculate bar positions
        bar_positions = x + (idx - n_runs/2 + 0.5) * width
        
        bars = ax.bar(bar_positions, values, width, 
                     label=run_name, 
                     color=colors[idx % len(colors)],
                     alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}',
                       ha='center', va='bottom', fontsize=7, rotation=90)
    
    # Customize plot
    ax.set_xlabel('Subproject', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_key, fontsize=12, fontweight='bold')
    ax.set_title(f'Metric Comparison: {metric_key}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(all_subprojects, rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metric plot: {output_path}")


def plot_weighted_average_comparison(
    all_runs_data: Dict[str, Dict[str, Dict]],
    output_path: Path,
    weight_key: str = 'num_samples'
):
    """
    Plot weighted average of all metrics across runs.
    
    Args:
        all_runs_data: Dict[run_name -> Dict[subproject_name -> metrics]]
        output_path: Where to save the plot
        weight_key: Key to use for weighting (default: 'num_samples')
    """
    run_names = sorted(all_runs_data.keys())
    
    # Get all metric keys (excluding the weight key and T)
    all_metric_keys = set()
    for run_name, subprojects in all_runs_data.items():
        for subproject_name, metrics in subprojects.items():
            all_metric_keys.update(metrics.keys())
    
    # Filter out non-metric keys
    exclude_keys = {weight_key, 'T', 'count', 'innovation_mahalanobis2_count'}
    metric_keys = sorted([k for k in all_metric_keys if k not in exclude_keys])
    
    # Calculate weighted average for each metric and run
    weighted_averages = defaultdict(dict)  # run_name -> metric_key -> weighted_avg
    
    for run_name in run_names:
        subprojects = all_runs_data[run_name]
        
        for metric_key in metric_keys:
            total_weight = 0
            weighted_sum = 0
            
            for subproject_name, metrics in subprojects.items():
                if metric_key in metrics and weight_key in metrics:
                    value = metrics[metric_key]
                    weight = metrics[weight_key]
                    
                    if value is not None and weight is not None:
                        weighted_sum += value * weight
                        total_weight += weight
            
            if total_weight > 0:
                weighted_averages[run_name][metric_key] = weighted_sum / total_weight
            else:
                weighted_averages[run_name][metric_key] = None
    
    # Create figure
    n_metrics = len(metric_keys)
    n_runs = len(run_names)
    
    fig, ax = plt.subplots(figsize=(max(14, n_metrics * 1.2), 7))
    
    # Set up bar positions
    x = np.arange(n_metrics)
    width = 0.8 / n_runs  # Width of each bar
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, run_name in enumerate(run_names):
        values = [weighted_averages[run_name].get(mk, 0) for mk in metric_keys]
        
        # Calculate bar positions
        bar_positions = x + (idx - n_runs/2 + 0.5) * width
        
        bars = ax.bar(bar_positions, values, width, 
                     label=run_name, 
                     color=colors[idx % len(colors)],
                     alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}',
                       ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Customize plot
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Weighted Average Metrics Comparison (weighted by {weight_key})', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved weighted average plot: {output_path}")
    
    # Also print the table
    print("\n" + "="*80)
    print("WEIGHTED AVERAGE METRICS")
    print("="*80)
    print(f"{'Metric':<40} " + " ".join([f"{rn:>12}" for rn in run_names]))
    print("-"*80)
    for metric_key in metric_keys:
        values_str = " ".join([
            f"{weighted_averages[rn].get(metric_key, 0):>12.6f}" 
            if weighted_averages[rn].get(metric_key) is not None 
            else f"{'N/A':>12}"
            for rn in run_names
        ])
        print(f"{metric_key:<40} {values_str}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple state estimator validation runs"
    )
    parser.add_argument(
        '--runs',
        nargs='+',
        required=True,
        help='Paths to run directories (e.g., validate_results/run1 validate_results/run2)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='validate_results_comparison',
        help='Output directory for comparison plots'
    )
    parser.add_argument(
        '--weight-key',
        type=str,
        default='num_samples',
        help='Metric key to use for weighted averaging (default: num_samples)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all runs
    all_runs_data = {}  # run_name -> subproject_name -> metrics
    for run_path_str in args.runs:
        run_path = Path(run_path_str)
        if not run_path.exists():
            print(f"Warning: Run directory does not exist: {run_path}")
            continue
        
        run_name, metrics_dict = load_run_metrics(run_path)
        if metrics_dict:
            all_runs_data[run_name] = metrics_dict
    
    if not all_runs_data:
        print("Error: No valid run data found!")
        return
    
    print(f"\nLoaded {len(all_runs_data)} runs")
    
    # Extract all metric keys
    metric_keys = extract_all_metric_keys(all_runs_data)
    print(f"Found {len(metric_keys)} metric keys: {metric_keys}")
    
    # Plot comparison for each metric
    print("\nGenerating per-metric comparison plots...")
    for metric_key in metric_keys:
        if metric_key in ['num_samples', 'T']:
            # Skip count metrics in individual plots
            continue
        
        output_path = output_dir / f"metric_{metric_key}.png"
        plot_metric_comparison(all_runs_data, metric_key, output_path)
    
    # Plot weighted average comparison
    print("\nGenerating weighted average comparison plot...")
    weighted_avg_path = output_dir / "weighted_average_comparison.png"
    plot_weighted_average_comparison(
        all_runs_data, 
        weighted_avg_path,
        weight_key=args.weight_key
    )
    
    print(f"\n✓ All comparison plots saved to: {output_dir}")
    print(f"  - Individual metric plots: metric_*.png")
    print(f"  - Weighted average: weighted_average_comparison.png")


if __name__ == "__main__":
    main()

