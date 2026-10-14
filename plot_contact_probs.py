#!/usr/bin/env python3
"""
Quick visualization of GeneralizedMomentumObserver::_update_contact_probs.

Probability model:
    f_rel = (|Fz| - thresh) / thresh
    x     = scale * f_rel
    p     = 1 / (1 + exp(-x))

Use:
    python plot_contact_probs.py --thresh 150 --scale 15
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def contact_prob_curve(fz: np.ndarray, thresh: float, scale: float) -> np.ndarray:
    """Replicate _update_contact_probs formula for a single contact."""
    # Avoid division by zero
    denom = max(thresh, 1e-6)
    f_rel = (np.abs(fz) - thresh) / denom
    x = scale * f_rel
    p = 1.0 / (1.0 + np.exp(-x))
    return np.clip(p, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Plot contact probability vs Fz curve.")
    parser.add_argument(
        "--thresh",
        type=float,
        default=150.0,
        help="Contact force threshold (same意义 as contactForceThreshold).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=20.0,
        help="Slope scale (same意义 as contactForceScale).",
    )
    parser.add_argument(
        "--max-mul",
        type=float,
        default=3.0,
        help="Fz max = max_mul * thresh (default: 3).",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=500,
        help="Number of samples between 0 and max Fz.",
    )
    args = parser.parse_args()

    thresh = float(args.thresh)
    scale = float(args.scale)
    fz_max = args.max_mul * thresh

    fz = np.linspace(0.0, fz_max, args.num)
    p = contact_prob_curve(fz, thresh, scale)

    plt.figure(figsize=(8, 4))
    plt.plot(fz, p, label=f"scale={scale:.3g}, thresh={thresh:.3g}")
    plt.axvline(thresh, color="red", linestyle="--", label="threshold")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Fz (N)")
    plt.ylabel("contact probability")
    plt.title("Contact probability vs Fz")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("contact_prob_curve.png", dpi=150)


if __name__ == "__main__":
    main()

