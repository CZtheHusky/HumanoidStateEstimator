from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np
import pickle

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Install with: pip install plotly")


def load_streaming_pickle(file_path: str) -> list[Any]:
    """
    Load data from a streaming pickle file (multiple pickles appended).

    Args:
        file_path: Path to the pickle file.

    Returns:
        List of all samples in the file.
    """
    data: list[Any] = []
    with open(file_path, "rb") as f:
        while True:
            try:
                sample = pickle.load(f)
                data.append(sample)
            except EOFError:
                break
    return data


def load_npz_log_with_slice(
    log_path: Path,
    start: int = 0,
    end: int = -1,
    *,
    derive_tick: bool = True,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Load a single NPZ log file and apply start/end slicing by index.

    This is a generic helper used by validate_estimator / validate_imu_preintegrator.
    It assumes the log contains a time vector ``t`` of length ``T`` and slices all
    arrays whose first dimension matches ``T``.

    Args:
        log_path: Path to ``.npz`` log.
        start:   Start index (inclusive).
        end:     End index (exclusive, -1 means all).
        derive_tick: Whether to derive integer tick (ms) from ``t`` if ``tick`` missing.
        verbose: Whether to print basic log / slicing information.
    """
    log_path = Path(log_path)
    if verbose:
        print(f"Loading log file: {log_path}")
    log_data: Dict[str, np.ndarray] = dict(np.load(log_path, allow_pickle=True))

    if "t" not in log_data:
        raise KeyError("NPZ log is expected to contain key 't' (timestamps in seconds).")

    n_samples = len(log_data["t"])
    start_idx = max(0, int(start))
    end_idx = n_samples if end == -1 else min(int(end), n_samples)

    if start_idx >= end_idx:
        raise ValueError(f"Invalid range [{start_idx}, {end_idx}) for {n_samples} samples")

    if verbose:
        print(f"  Total samples: {n_samples}")
        print(f"  Using range: [{start_idx}, {end_idx}) ({end_idx - start_idx} samples)")

    # Slice all time-series arrays whose first dimension is n_samples
    for key, val in list(log_data.items()):
        shape = getattr(val, "shape", ())
        if len(shape) > 0 and shape[0] == n_samples:
            log_data[key] = val[start_idx:end_idx]
            if verbose:
                print(f"  Sliced '{key}': {shape} -> {log_data[key].shape}")

    # Optional: joint name information (not sliced)
    if verbose and "joint_names" in log_data:
        joint_names = [str(name) for name in log_data["joint_names"]]
        print(f"  ✅ Contains joint names information ({len(joint_names)} joints)")

    # Derive integer tick from continuous time t (seconds) if needed
    if derive_tick and "tick" not in log_data:
        sim_dt_tick = round(log_data['sim_timestep'] * 1000)
        ticks = np.arange(0, len(log_data["t"]) * sim_dt_tick, sim_dt_tick)
        log_data["tick"] = ticks
        if verbose:
            print(
                f"  Derived 'tick' from 't': "
                f"tick_step≈{sim_dt_tick} ms"
            )

    return log_data


def load_hw_log_to_npz_like_for_estimator(
    log_path: Path,
    start: int = 0,
    end: int = -1,
) -> Dict[str, np.ndarray]:
    """
    Load a hardware StreamingPickleLogger log and convert it into an NPZ-like
    dict compatible with EstimatorValidator.run_offline_validation.

    Expected per-sample dict keys:
      - "tick", "joint_q", "joint_dq", "joint_tau_est",
        "imu_rpy", "imu_gyroscope", "imu_quaternion", "imu_accelerometer",
        "base_pos", "base_lin_vel".
    """
    log_path = Path(log_path)
    if not log_path.is_file():
        raise FileNotFoundError(f"Hardware log file not found: {log_path}")

    print(f"Loading hardware log: {log_path}")
    samples = load_streaming_pickle(str(log_path))
    if not samples:
        raise RuntimeError(f"No samples in hardware log: {log_path}")

    # Convert list[dict] -> stacked arrays
    ticks = np.array([int(s["tick"]) for s in samples], dtype=np.int64)
    n = len(ticks)

    def stack(key: str, dim: Optional[int] = None) -> np.ndarray:
        arr = np.stack([np.array(s[key], dtype=np.float32) for s in samples], axis=0)
        if dim is not None and arr.shape[1] != dim:
            raise ValueError(f"{key} expected dim {dim}, got {arr.shape[1]}")
        return arr

    joint_q = stack("joint_q")
    joint_dq = stack("joint_dq")
    joint_tau_est = stack("joint_tau_est")
    imu_quat = stack("imu_quaternion", dim=4)
    imu_accel = stack("imu_accelerometer", dim=3)
    imu_gyro = stack("imu_gyroscope", dim=3)
    base_vel_gt = stack("base_lin_vel", dim=3)
    base_pos_gt = stack("base_pos", dim=3)

    # Build NPZ-like dict
    t_s = (ticks - ticks[0]).astype(np.float64) / 1000.0  # assume 1 tick = 1 ms

    # Calculate imu_accel_legacy based on base_vel_gt
    imu_accel_legacy = np.zeros_like(base_vel_gt)
    gravity_w = np.array([0.0, 0.0, -9.81])

    def _q2r(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
            ]
        )

    for i in range(n):
        if i == 0:
            accel_w = np.zeros(3, dtype=float)
        else:
            dt = float(t_s[i] - t_s[i - 1])
            if dt < 1e-6:
                dt = 0.001
            accel_w = (base_vel_gt[i] - base_vel_gt[i - 1]) / dt

        R_wb = _q2r(imu_quat[i])
        # a_body = R_wb^T * (a_world - g_world)
        imu_accel_legacy[i] = R_wb.T @ (accel_w - gravity_w)

    log_data: Dict[str, np.ndarray] = {
        "tick": ticks,
        "t": t_s,
        "q_joints": joint_q,
        "v_joints": joint_dq,
        "tau": joint_tau_est,
        "imu_accel": imu_accel,
        "imu_accel_legacy": imu_accel_legacy,
        "imu_quat_wxyz": imu_quat,
        "imu_gyro": imu_gyro,
        "base_vel_gt": base_vel_gt,
        "base_pos_gt": base_pos_gt,
    }

    # Apply slicing by index
    n_samples = n
    start_idx = max(0, int(start))
    end_idx = n_samples if end == -1 else min(int(end), n_samples)
    if start_idx >= end_idx:
        raise ValueError(f"Invalid range [{start_idx}, {end_idx}) for {n_samples} samples")

    print(f"  Total samples: {n_samples}")
    print(f"  Using range: [{start_idx}, {end_idx}) ({end_idx - start_idx} samples)")

    for key, val in list(log_data.items()):
        if hasattr(val, "ndim") and val.ndim > 0 and val.shape[0] == n_samples:
            log_data[key] = val[start_idx:end_idx]
            print(f"  Sliced '{key}': {val.shape} -> {log_data[key].shape}")

    return log_data


def set_nested_attr(obj: Any, attr_path: str, value: str) -> None:
    """
    Set a nested attribute of an object, with automatic type conversion.

    Example:
        set_nested_attr(cfg, "kf.imu_accel_noise_density", "0.5")
    """
    parts = attr_path.split(".")
    current = obj
    for part in parts[:-1]:
        if not hasattr(current, part):
            raise AttributeError(f"Object {current} has no attribute {part}")
        current = getattr(current, part)

    attr_name = parts[-1]
    if not hasattr(current, attr_name):
        raise AttributeError(f"Object {current} has no attribute {attr_name}")

    existing_value = getattr(current, attr_name)
    target_type = type(existing_value)

    new_value: Any = value
    if target_type is bool:
        new_value = str(value).lower() in ("true", "1", "yes", "on")
    elif target_type is int:
        new_value = int(value)
    elif target_type is float:
        new_value = float(value)
    elif target_type is list:
        # For lists, we assume the user knows what they are doing or we don't support
        # it yet via this simple interface unless it's a list of strings.
        if isinstance(existing_value, list) and (
            not existing_value or isinstance(existing_value[0], str)
        ):
            new_value = value.split(",")
        else:
            print(
                f"Warning: List parsing for {attr_path} not fully supported, "
                "keeping as string or trying eval"
            )
            try:
                new_value = eval(value)
            except Exception:
                pass

    setattr(current, attr_name, new_value)
    print(f"  Set {attr_path} = {new_value} (type: {type(new_value).__name__})")


def apply_config_overrides(config: Any, args_list: Sequence[str] | None) -> None:
    """
    Apply overrides from command line arguments in the format ``key=value`` to a
    (possibly nested) config dataclass.
    """
    if not args_list:
        return

    print("Applying config overrides:")
    for arg in args_list:
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Support 'cfg.' prefix or just direct access
            if key.startswith("cfg."):
                key = key[4:]

            # Remove leading -- if present (e.g. --kf.val=1)
            if key.startswith("--"):
                key = key[2:]

            try:
                set_nested_attr(config, key, value)
            except Exception as e:
                print(f"  Error setting {key}: {e}")
        else:
            # Handle flags that might be passed as --flag value (not supported by this simple parser, but we can warn)
            if arg.startswith("--"):
                print(
                    f"  Warning: Unknown argument format or flag: {arg}. "
                    "Use key=value for config overrides."
                )


# ======================================================================
# Plotting helpers for estimator validation
# ======================================================================

def _plot_series_matplotlib(ax, x, ys, labels, ylabel, styles=None):
    """
    Helper for Matplotlib line plots.

    Args:
        ax:      Matplotlib axes.
        x:       1D array-like for x-axis.
        ys:      List of 1D arrays for each series.
        labels:  List of legend labels (same length as ys).
        ylabel:  Y-axis label.
        styles:  Optional list of dicts with extra kwargs per series
                 (e.g. {'linestyle': '--', 'alpha': 0.7}).
    """
    for i, y in enumerate(ys):
        kwargs = {"linewidth": 2}
        if styles is not None and i < len(styles) and styles[i] is not None:
            kwargs.update(styles[i])
        ax.plot(x, y, label=labels[i], **kwargs)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _add_series_plotly(fig, row, x, ys, labels, ylabel, styles=None):
    """
    Helper for Plotly line plots (one subplot, multiple series).
    """
    if not PLOTLY_AVAILABLE:
        return
    for i, y in enumerate(ys):
        style: Dict[str, Any] = {"line": {"width": 2}}
        if styles is not None and i < len(styles) and styles[i] is not None:
            # allow overriding line / color / dash etc.
            for k, v in styles[i].items():
                if k == "line":
                    style.setdefault("line", {}).update(v)
                else:
                    style[k] = v
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=labels[i],
                **style,
            ),
            row=row,
            col=1,
        )
    fig.update_yaxes(title_text=ylabel, row=row, col=1)


def plot_estimator_results_matplotlib(results: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Matplotlib version of estimator validation plots (was EstimatorValidator.plot_results).
    """
    t = results["timestamps"]
    v_est = results["v_est_hist"]
    v_gt = results["base_vel_gt"]
    error = results["error_hist"]
    p_est = results["p_est_hist"]
    p_gt = results.get("base_pos_gt", None)
    pos_error = results.get("pos_error_hist", None)
    z_c_est = results.get("z_c_est", None)
    z_c_gt = results.get("z_c_gt", None)
    contact_fz = results.get("contact_fz", None)
    contact_prob = results.get("contact_prob", None)
    contact_weights_log = results.get("contact_weights_log", None)
    contact_active = results.get("contact_active", None)
    imu_bias = results.get("imu_accel_bias_hist", None)
    zmp_offsets = results.get("zmp_offsets", None)
    zmp_valid = results.get("zmp_valid", None)
    step_time_ms = results.get("step_time_ms", None)
    imu_accel = results.get("imu_accel", None)
    imu_accel_legacy = results.get("imu_accel_legacy", None)
    innovation_pos = results.get("innovation_pos", None)
    innovation_height = results.get("innovation_height", None)
    innovation_mahalanobis2 = results.get("innovation_mahalanobis2_hist", None)
    P_diag_hist = results.get("P_diag_hist", None)
    R_diag_hist = results.get("R_diag_hist", None)

    # 可视化开关（默认都开启）
    plot_R = results.get("plot_R", True)
    plot_P = results.get("plot_P", True)
    plot_innovation = results.get("plot_innovation", True)

    steps = np.arange(len(t))

    has_pos = p_gt is not None and pos_error is not None
    has_bias = imu_bias is not None

    n_subplots = 4
    if has_pos:
        n_subplots += 4
    if has_bias:
        n_subplots += 1
    if z_c_est is not None:
        n_subplots += 1  # foot heights
    if contact_fz is not None:
        n_subplots += 1  # contact forces
    if contact_prob is not None:
        n_subplots += 1  # contact probabilities
    if contact_active is not None:
        n_subplots += 1  # contact active status
    # 残差可视化：每个脚的 [-r_bc] 残差 + 足端高度残差
    if (
        plot_innovation
        and innovation_pos is not None
        and np.isfinite(innovation_pos).any()
    ):
        # 每个接触点各一个 subplot，用于 3 轴位置残差
        n_subplots += innovation_pos.shape[1]
    if (
        plot_innovation
        and innovation_height is not None
        and np.isfinite(innovation_height).any()
    ):
        n_subplots += 1  # 所有脚的高度残差放在一张图
    if plot_R and R_diag_hist is not None and np.isfinite(R_diag_hist).any():
        n_subplots += 2  # R 中位置噪声 / 高度噪声各一张
    if plot_P and P_diag_hist is not None and np.isfinite(P_diag_hist).any():
        n_subplots += 2  # P 中 base / bias 方差
    if step_time_ms is not None:
        n_subplots += 1
    if imu_accel is not None:
        n_subplots += 3
    if innovation_mahalanobis2 is not None and np.isfinite(innovation_mahalanobis2).any():
        n_subplots += 1

    fig, axes = plt.subplots(n_subplots, 1, figsize=(24, 3.5 * n_subplots))

    # Velocity components
    vel_labels = ["v_x", "v_y", "v_z"]
    for i in range(3):
        ax = axes[i]
        ys = [v_est[:, i], v_gt[:, i]]
        labels = [
            f"{vel_labels[i]} (estimated)",
            f"{vel_labels[i]} (ground truth)",
        ]
        styles = [
            None,
            {"linestyle": "--", "alpha": 0.7},
        ]
        _plot_series_matplotlib(
            ax, steps, ys, labels, ylabel=f"{vel_labels[i]} (m/s)", styles=styles
        )

    # Velocity error
    axes[3].plot(steps, error, label="Velocity error", linewidth=2, color="red")
    axes[3].axhline(
        y=results["rmse"],
        color="blue",
        linestyle="--",
        label=f'RMSE = {results["rmse"]:.4f}',
    )
    axes[3].set_ylabel("Error (m/s)")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plot_idx = 4

    # Position components
    if has_pos:
        p_labels = ["p_x", "p_y", "p_z"]
        for i in range(3):
            ax = axes[plot_idx]
            ys = [p_est[:, i], p_gt[:, i]]
            labels = [
                f"{p_labels[i]} (estimated)",
                f"{p_labels[i]} (ground truth)",
            ]
            styles = [
                None,
                {"linestyle": "--", "alpha": 0.7},
            ]
            _plot_series_matplotlib(
                ax, steps, ys, labels, ylabel=f"{p_labels[i]} (m)", styles=styles
            )
            plot_idx += 1

        ax = axes[plot_idx]
        ax.plot(
            steps,
            pos_error,
            label="Position error",
            linewidth=2,
            color="red",
        )
        if results["rmse_pos"] is not None:
            ax.axhline(
                y=results["rmse_pos"],
                color="blue",
                linestyle="--",
                label=f'RMSE = {results["rmse_pos"]:.4f}',
            )
        ax.set_ylabel("Error (m)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # IMU accel bias
    if has_bias:
        ax = axes[plot_idx]
        bias_labels = ["b_ax", "b_ay", "b_az"]
        ys = [imu_bias[:, i] for i in range(3)]
        _plot_series_matplotlib(
            ax, steps, ys, bias_labels, ylabel="IMU bias (m/s²)"
        )
        plot_idx += 1

    # IMU acceleration (IMU vs legacy reconstruction, if both available)
    if imu_accel is not None:
        accel_labels = ["a_x", "a_y", "a_z"]
        for i in range(3):
            ax = axes[plot_idx]
            ys = []
            labels = []
            styles = []
            if imu_accel is not None:
                ys.append(imu_accel[:, i])
                labels.append(f"{accel_labels[i]} (imu)")
                styles.append(None)
            _plot_series_matplotlib(
                ax, steps, ys, labels, ylabel="IMU accel (m/s²)", styles=styles
            )
            plot_idx += 1

    # Foot heights
    if z_c_est is not None:
        ax = axes[plot_idx]
        n_contacts = z_c_est.shape[1]
        foot_labels = [f"Foot {i+1}" for i in range(n_contacts)]
        ys = []
        labels = []
        if z_c_gt is not None:
            ys_gt = []
            labels_gt = []
        for i in range(n_contacts):
            ys.append(z_c_est[:, i])
            labels.append(f"{foot_labels[i]} (estimated)")
            if z_c_gt is not None:
                ys.append(z_c_gt[:, i])
                labels.append(f"{foot_labels[i]} (ground truth)")
        # 简单处理：直接画多条线
        for i, y in enumerate(ys):
            ax.plot(steps, y, linewidth=2, label=labels[i])
        ax.set_ylabel("Foot height (m)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Contact forces fz
    if contact_fz is not None:
        ax = axes[plot_idx]
        n_contacts = contact_fz.shape[1]
        foot_labels = [f"Foot {i+1}" for i in range(n_contacts)]
        for i in range(n_contacts):
            ax.plot(
                steps,
                contact_fz[:, i],
                label=f"{foot_labels[i]} fz",
                linewidth=2,
            )
        ax.axhline(y=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_ylabel("Contact force fz (N)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Contact probabilities (+ optional logged contact weights)
    if contact_prob is not None:
        ax = axes[plot_idx]
        n_contacts = contact_prob.shape[1]
        foot_labels = [f"Foot {i+1}" for i in range(n_contacts)]
        n_plot = n_contacts
        if contact_weights_log is not None:
            n_plot = min(n_contacts, contact_weights_log.shape[1])

        contact_initialized = results.get("contact_initialized", None)

        for i in range(n_plot):
            # GMO estimated contact probability
            line_prob, = ax.plot(
                steps,
                contact_prob[:, i],
                label=f"{foot_labels[i]} prob",
                linewidth=2,
            )
            # Logged contact weight from NPZ (same color, different linestyle)
            if contact_weights_log is not None:
                ax.plot(
                    steps,
                    contact_weights_log[:, i],
                    label=f"{foot_labels[i]} weight(log)",
                    linewidth=1.5,
                    linestyle="--",
                    color=line_prob.get_color(),
                )
            
            # Add markers for contact initialization events
            if contact_initialized is not None and i < contact_initialized.shape[1]:
                init_mask = contact_initialized[:, i]
                init_steps = steps[init_mask]
                init_probs = contact_prob[init_mask, i]
                if len(init_steps) > 0:
                    ax.scatter(
                        init_steps,
                        init_probs,
                        marker='*',
                        s=150,
                        color=line_prob.get_color(),
                        edgecolors='red',
                        linewidths=2,
                        label=f"{foot_labels[i]} init",
                        zorder=10
                    )

        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Contact prob.")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Contact active status (which feet are in contact)
    if contact_active is not None:
        ax = axes[plot_idx]
        n_contacts = contact_active.shape[1]
        foot_labels = [f"Foot {i+1}" for i in range(n_contacts)]
        
        for i in range(n_contacts):
            # Convert boolean to float for plotting (0 = not active, 1 = active)
            active_float = contact_active[:, i].astype(float)
            ax.plot(
                steps,
                active_float,
                label=f"{foot_labels[i]} active",
                linewidth=2,
                drawstyle='steps-post',  # Use step plot for binary signals
            )
        
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Inactive', 'Active'])
        ax.set_ylabel("Contact active")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Innovation: 每个脚的 [-r_bc]_x,y,z 残差
    if innovation_pos is not None and np.isfinite(innovation_pos).any():
        n_contacts = innovation_pos.shape[1]
        for i in range(n_contacts):
            ax = axes[plot_idx]
            ax.plot(
                steps,
                innovation_pos[:, i, 0],
                label=f"Foot {i+1} Δ[-r_bc]_x (m)",
                linewidth=1.5,
            )
            ax.plot(
                steps,
                innovation_pos[:, i, 1],
                label=f"Foot {i+1} Δ[-r_bc]_y (m)",
                linewidth=1.5,
            )
            ax.plot(
                steps,
                innovation_pos[:, i, 2],
                label=f"Foot {i+1} Δ[-r_bc]_z (m)",
                linewidth=1.5,
            )
            ax.set_ylabel(f"Foot {i+1} rel-pos resid (m)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1

    # Innovation: 足端高度残差 z_c
    if innovation_height is not None and np.isfinite(innovation_height).any():
        ax = axes[plot_idx]
        n_contacts = innovation_height.shape[1]
        foot_labels = [f"Foot {i+1}" for i in range(n_contacts)]
        for i in range(n_contacts):
            ax.plot(
                steps,
                innovation_height[:, i],
                label=f"{foot_labels[i]} Δz_c (m)",
                linewidth=2,
            )
        ax.set_ylabel("Foot height resid (m)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Estimator step time
    if step_time_ms is not None:
        ax = axes[plot_idx]
        ax.plot(steps, step_time_ms, label="Estimator step time", linewidth=2)
        ax.set_ylabel("Step time (ms)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Innovation Mahalanobis^2
    if (
        plot_innovation
        and innovation_mahalanobis2 is not None
        and np.isfinite(innovation_mahalanobis2).any()
    ):
        ax = axes[plot_idx]
        ax.plot(
            steps,
            innovation_mahalanobis2,
            label="Innovation Mahalanobis²",
            linewidth=2,
        )
        if z_c_est is not None:
            n_contacts = z_c_est.shape[1]
            dof = 4 * n_contacts  # dim(y) = 3*n_c + n_c
            ax.axhline(
                y=dof,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label=f"χ² dof ≈ {dof}",
            )
        ax.set_ylabel("Mahalanobis²")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # 状态协方差 P 的对角线可视化（只画 base pos/vel 和 bias）
    if plot_P and P_diag_hist is not None and np.isfinite(P_diag_hist).any():
        nx = P_diag_hist.shape[1]
        # 状态布局：x = [p_b(3), v_b(3), p_c1(3), ..., p_cn(3), b_a(3)]
        # 因此 n_contacts = (nx - 9) / 3
        n_contacts = max(int((nx - 9) // 3), 0)
        bias_start = 6 + 3 * n_contacts
        # Base pos/vel variances
        ax = axes[plot_idx]
        labels_pv = ["P_p_x", "P_p_y", "P_p_z", "P_v_x", "P_v_y", "P_v_z"]
        ys = [P_diag_hist[:, i] for i in range(6)]
        for i, y in enumerate(ys):
            ax.plot(steps, y, label=labels_pv[i], linewidth=2)
        ax.set_ylabel("P base pos/vel var")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        # IMU accel bias variances
        ax = axes[plot_idx]
        bias_labels = ["P_bias_ax", "P_bias_ay", "P_bias_az"]
        for i in range(3):
            ax.plot(
                steps,
                P_diag_hist[:, bias_start + i],
                label=bias_labels[i],
                linewidth=2,
            )
        ax.set_ylabel("P bias var")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # R 对角线可视化：
    #   前 3*n_c 维：每个脚 [-r_bc]_x,y,z 的噪声方差；
    #   后 n_c 维：每个脚 z_c 的噪声方差。
    if plot_R and R_diag_hist is not None and np.isfinite(R_diag_hist).any():
        n_contacts = R_diag_hist.shape[1] // 4
        # 位置噪声方差
        ax = axes[plot_idx]
        for i in range(n_contacts):
            idx_x = 3 * i + 0
            idx_y = 3 * i + 1
            idx_z = 3 * i + 2
            ax.plot(
                steps,
                R_diag_hist[:, idx_x],
                label=f"Foot {i+1} R_x ([-r_bc]_x var)",
                linewidth=1.5,
            )
            ax.plot(
                steps,
                R_diag_hist[:, idx_y],
                label=f"Foot {i+1} R_y ([-r_bc]_y var)",
                linewidth=1.5,
                linestyle="--",
            )
            ax.plot(
                steps,
                R_diag_hist[:, idx_z],
                label=f"Foot {i+1} R_z ([-r_bc]_z var)",
                linewidth=1.5,
                linestyle=":",
            )
        ax.set_ylabel("R position var (m²)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        # 高度噪声方差
        ax = axes[plot_idx]
        for i in range(n_contacts):
            idx_h = 3 * n_contacts + i
            ax.plot(
                steps,
                R_diag_hist[:, idx_h],
                label=f"Foot {i+1} R_zc (z_c var)",
                linewidth=2,
            )
        ax.set_ylabel("R height var (m²)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    axes[plot_idx - 1].set_xlabel("Step")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Results plot saved to: {save_path}")


def plot_estimator_results_interactive(results: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plot validation results using Plotly (interactive HTML), inspired by
    old_implement/validate_estimator_legacy.py.
    """
    if not PLOTLY_AVAILABLE:
        print("Error: Plotly is not installed. Install with: pip install plotly")
        return

    t = results["timestamps"]
    v_est = results["v_est_hist"]
    v_gt = results["base_vel_gt"]
    error = results["error_hist"]
    p_est = results["p_est_hist"]
    p_gt = results.get("base_pos_gt", None)
    pos_error = results.get("pos_error_hist", None)
    z_c_est = results.get("z_c_est", None)
    z_c_gt = results.get("z_c_gt", None)
    contact_fz = results.get("contact_fz", None)
    contact_prob = results.get("contact_prob", None)
    contact_weights_log = results.get("contact_weights_log", None)
    contact_active = results.get("contact_active", None)
    imu_bias = results.get("imu_accel_bias_hist", None)
    zmp_offsets = results.get("zmp_offsets", None)
    zmp_valid = results.get("zmp_valid", None)
    step_time_ms = results.get("step_time_ms", None)
    imu_accel = results.get("imu_accel", None)
    imu_accel_legacy = results.get("imu_accel_legacy", None)
    innovation_pos = results.get("innovation_pos", None)
    innovation_height = results.get("innovation_height", None)
    innovation_mahalanobis2 = results.get("innovation_mahalanobis2_hist", None)
    P_diag_hist = results.get("P_diag_hist", None)
    R_diag_hist = results.get("R_diag_hist", None)

    plot_R = results.get("plot_R", True)
    plot_P = results.get("plot_P", True)
    plot_innovation = results.get("plot_innovation", True)

    steps = np.arange(len(t))

    has_pos = p_gt is not None and pos_error is not None
    has_bias = imu_bias is not None

    # Determine number of subplots (group some signals together)
    n_subplots = 4  # velocity xyz + error
    if has_pos:
        n_subplots += 4  # position xyz + error
    if has_bias:
        n_subplots += 1  # IMU accel bias
    if imu_accel is not None:
        n_subplots += 1  # IMU acceleration
    if z_c_est is not None:
        n_subplots += 1  # foot heights
    if contact_fz is not None:
        n_subplots += 1  # contact forces
    # 残差：每个脚的位置残差 / 高度残差
    if (
        plot_innovation
        and innovation_pos is not None
        and np.isfinite(innovation_pos).any()
    ):
        n_subplots += innovation_pos.shape[1]  # 每个脚一张位置残差图
    if (
        plot_innovation
        and innovation_height is not None
        and np.isfinite(innovation_height).any()
    ):
        n_subplots += 1  # 所有脚高度残差
    if step_time_ms is not None:
        n_subplots += 1  # step time
    if contact_prob is not None:
        n_subplots += 1  # contact probabilities
    if contact_active is not None:
        n_subplots += 1  # contact active status
    if (
        plot_innovation
        and innovation_mahalanobis2 is not None
        and np.isfinite(innovation_mahalanobis2).any()
    ):
        n_subplots += 1  # innovation Mahalanobis^2
    if plot_R and R_diag_hist is not None and np.isfinite(R_diag_hist).any():
        n_subplots += 2  # R 中位置噪声 / 高度噪声
    if plot_P and P_diag_hist is not None and np.isfinite(P_diag_hist).any():
        n_subplots += 2  # P 中 base / bias 方差

    # Subplot titles
    subplot_titles = [
        "v_x (m/s)",
        "v_y (m/s)",
        "v_z (m/s)",
        "Velocity error (m/s)",
    ]
    if has_pos:
        subplot_titles.extend(
            ["p_x (m)", "p_y (m)", "p_z (m)", "Position error (m)"]
        )
    if has_bias:
        subplot_titles.append("IMU acceleration bias (m/s²)")
    if imu_accel is not None:
        subplot_titles.append("IMU acceleration (m/s²)")
    if z_c_est is not None:
        subplot_titles.append("Foot heights (m)")
    if contact_fz is not None:
        subplot_titles.append("Contact force fz (N)")
    if contact_prob is not None:
        subplot_titles.append("Contact probability")
    if contact_active is not None:
        subplot_titles.append("Contact active status")

    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.005,  # 减小垂直间距使子图更紧凑
        shared_xaxes=True,
    )

    plot_idx = 1

    # Velocity components
    vel_labels = ["v_x", "v_y", "v_z"]
    colors_est = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    colors_gt = ["#aec7e8", "#ffbb78", "#98df8a"]

    for i in range(3):
        ys = [v_est[:, i], v_gt[:, i]]
        labels = [
            f"{vel_labels[i]} (est)",
            f"{vel_labels[i]} (gt)",
        ]
        styles = [
            {"line": {"color": colors_est[i], "width": 2}, "legendgroup": f"v{i}"},
            {
                "line": {"color": colors_gt[i], "width": 2, "dash": "dash"},
                "legendgroup": f"v{i}",
            },
        ]
        _add_series_plotly(
            fig,
            row=plot_idx,
            x=steps,
            ys=ys,
            labels=labels,
            ylabel=f"{vel_labels[i]} (m/s)",
            styles=styles,
        )
        plot_idx += 1

    # Velocity error
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=error,
            mode="lines",
            name="Velocity error",
            line=dict(color="red", width=2),
        ),
        row=plot_idx,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=np.full_like(steps, results["rmse"], dtype=float),
            mode="lines",
            name=f'RMSE = {results["rmse"]:.4f}',
            line=dict(color="blue", width=2, dash="dash"),
        ),
        row=plot_idx,
        col=1,
    )
    fig.update_yaxes(title_text="Error (m/s)", row=plot_idx, col=1)
    plot_idx += 1

    # Position components
    if has_pos:
        pos_labels = ["p_x", "p_y", "p_z"]
        colors_p_est = ["#d62728", "#9467bd", "#8c564b"]
        colors_p_gt = ["#ff9896", "#c5b0d5", "#c49c94"]

        for i in range(3):
            ys = [p_est[:, i], p_gt[:, i]]
            labels = [
                f"{pos_labels[i]} (est)",
                f"{pos_labels[i]} (gt)",
            ]
            styles = [
                {
                    "line": {"color": colors_p_est[i], "width": 2},
                    "legendgroup": f"p{i}",
                },
                {
                    "line": {"color": colors_p_gt[i], "width": 2, "dash": "dash"},
                    "legendgroup": f"p{i}",
                },
            ]
            _add_series_plotly(
                fig,
                row=plot_idx,
                x=steps,
                ys=ys,
                labels=labels,
                ylabel=f"{pos_labels[i]} (m)",
                styles=styles,
            )
            plot_idx += 1

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=pos_error,
                mode="lines",
                name="Position error",
                line=dict(color="purple", width=2),
            ),
            row=plot_idx,
            col=1,
        )
        if results.get("rmse_pos") is not None:
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=np.full_like(steps, results["rmse_pos"], dtype=float),
                    mode="lines",
                    name=f'RMSE = {results["rmse_pos"]:.4f}',
                    line=dict(color="blue", width=2, dash="dash"),
                ),
                row=plot_idx,
                col=1,
            )
        fig.update_yaxes(title_text="Error (m)", row=plot_idx, col=1)
        plot_idx += 1

    # IMU accel bias
    if has_bias:
        bias_labels = ["b_ax", "b_ay", "b_az"]
        bias_colors = ["#e377c2", "#7f7f7f", "#bcbd22"]
        ys = [imu_bias[:, i] for i in range(3)]
        styles = [
            {"line": {"color": bias_colors[i], "width": 2}} for i in range(3)
        ]
        _add_series_plotly(
            fig,
            row=plot_idx,
            x=steps,
            ys=ys,
            labels=bias_labels,
            ylabel="IMU bias (m/s²)",
            styles=styles,
        )
        plot_idx += 1

    # IMU acceleration
    if imu_accel is not None:
        accel_labels = ["a_x", "a_y", "a_z"]
        accel_colors = ["#17becf", "#8c564b", "#bcbd22"]
        for i in range(3):
            ys = []
            labels = []
            styles = []
            if imu_accel is not None:
                ys.append(imu_accel[:, i])
                labels.append(f"{accel_labels[i]} (imu)")
                styles.append({"line": {"color": accel_colors[i], "width": 2}})
            _add_series_plotly(
                fig,
                row=plot_idx,
                x=steps,
                ys=ys,
                labels=labels,
                ylabel="IMU accel (m/s²)",
                styles=styles,
            )
        plot_idx += 1

    # Foot heights
    if z_c_est is not None:
        n_contacts = z_c_est.shape[1]
        foot_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i in range(n_contacts):
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=z_c_est[:, i],
                    mode="lines",
                    name=f"Foot {i+1} (est)",
                    line=dict(color=foot_colors[i % len(foot_colors)], width=2),
                    legendgroup=f"foot{i}",
                ),
                row=plot_idx,
                col=1,
            )
            if z_c_gt is not None:
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=z_c_gt[:, i],
                        mode="lines",
                        name=f"Foot {i+1} (gt)",
                        line=dict(
                            color=foot_colors[i % len(foot_colors)],
                            width=2,
                            dash="dash",
                        ),
                        legendgroup=f"foot{i}",
                    ),
                    row=plot_idx,
                    col=1,
                )
        fig.update_yaxes(title_text="Foot height (m)", row=plot_idx, col=1)
        plot_idx += 1

    # Contact forces fz
    if contact_fz is not None:
        n_contacts = contact_fz.shape[1]
        fz_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i in range(n_contacts):
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=contact_fz[:, i],
                    mode="lines",
                    name=f"Foot {i+1} fz",
                    line=dict(color=fz_colors[i % len(fz_colors)], width=2),
                ),
                row=plot_idx,
                col=1,
            )
        fig.update_yaxes(title_text="Contact force fz (N)", row=plot_idx, col=1)
        plot_idx += 1

    # Contact probabilities
    if contact_prob is not None:
        n_contacts = contact_prob.shape[1]
        prob_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        n_plot = n_contacts
        if contact_weights_log is not None:
            n_plot = min(n_contacts, contact_weights_log.shape[1])

        contact_initialized = results.get("contact_initialized", None)

        for i in range(n_plot):
            # GMO estimated contact probability
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=contact_prob[:, i],
                    mode="lines",
                    name=f"Foot {i+1} contact prob",
                    line=dict(color=prob_colors[i % len(prob_colors)], width=2),
                ),
                row=plot_idx,
                col=1,
            )
            # Logged contact weights (same foot color, dashed)
            if contact_weights_log is not None:
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=contact_weights_log[:, i],
                        mode="lines",
                        name=f"Foot {i+1} contact weight (log)",
                        line=dict(
                            color=prob_colors[i % len(prob_colors)],
                            width=2,
                            dash="dot",
                        ),
                    ),
                    row=plot_idx,
                    col=1,
                )
            
            # Add markers for contact initialization events
            if contact_initialized is not None and i < contact_initialized.shape[1]:
                init_mask = contact_initialized[:, i]
                init_steps = steps[init_mask]
                init_probs = contact_prob[init_mask, i]
                if len(init_steps) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=init_steps,
                            y=init_probs,
                            mode="markers",
                            name=f"Foot {i+1} initialized",
                            marker=dict(
                                color=prob_colors[i % len(prob_colors)],
                                size=10,
                                symbol="star",
                                line=dict(color="red", width=2)
                            ),
                            showlegend=True,
                        ),
                        row=plot_idx,
                        col=1,
                    )

        fig.update_yaxes(
            title_text="Contact probability", range=[-0.05, 1.05], row=plot_idx, col=1
        )
        plot_idx += 1

    # Contact active status (which feet are in contact)
    if contact_active is not None:
        n_contacts = contact_active.shape[1]
        active_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        
        for i in range(n_contacts):
            # Convert boolean to float for plotting (0 = not active, 1 = active)
            active_float = contact_active[:, i].astype(float)
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=active_float,
                    mode="lines",
                    name=f"Foot {i+1} active",
                    line=dict(
                        color=active_colors[i % len(active_colors)],
                        width=2,
                        shape='hv'  # horizontal-vertical step plot
                    ),
                ),
                row=plot_idx,
                col=1,
            )
        
        fig.update_yaxes(
            title_text="Contact active",
            range=[-0.1, 1.1],
            tickvals=[0, 1],
            ticktext=['Inactive', 'Active'],
            row=plot_idx,
            col=1
        )
        plot_idx += 1

    # Innovation: 每个脚的 [-r_bc]_x,y,z 残差
    if innovation_pos is not None and np.isfinite(innovation_pos).any():
        n_contacts = innovation_pos.shape[1]
        for i in range(n_contacts):
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=innovation_pos[:, i, 0],
                    mode="lines",
                    name=f"Foot {i+1} Δ[-r_bc]_x (m)",
                    line=dict(width=1.5),
                ),
                row=plot_idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=innovation_pos[:, i, 1],
                    mode="lines",
                    name=f"Foot {i+1} Δ[-r_bc]_y (m)",
                    line=dict(width=1.5, dash="dot"),
                ),
                row=plot_idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=innovation_pos[:, i, 2],
                    mode="lines",
                    name=f"Foot {i+1} Δ[-r_bc]_z (m)",
                    line=dict(width=1.5, dash="dash"),
                ),
                row=plot_idx,
                col=1,
            )
        fig.update_yaxes(
            title_text="Foot rel-pos residual [-r_bc] (m)",
            row=plot_idx,
            col=1,
        )
        plot_idx += 1

    # Innovation: 足端高度残差 z_c
    if innovation_height is not None and np.isfinite(innovation_height).any():
        n_contacts = innovation_height.shape[1]
        for i in range(n_contacts):
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=innovation_height[:, i],
                    mode="lines",
                    name=f"Foot {i+1} Δz_c (m)",
                    line=dict(width=2),
                ),
                row=plot_idx,
                col=1,
            )
        fig.update_yaxes(title_text="Foot height residual z_c (m)", row=plot_idx, col=1)
        plot_idx += 1

    # Estimator step time
    if step_time_ms is not None:
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=step_time_ms,
                mode="lines",
                name="Estimator step time",
                line=dict(color="#17becf", width=2),
            ),
            row=plot_idx,
            col=1,
        )
        fig.update_yaxes(title_text="Step time (ms)", row=plot_idx, col=1)
        plot_idx += 1

    # Innovation Mahalanobis^2
    if (
        plot_innovation
        and innovation_mahalanobis2 is not None
        and np.isfinite(innovation_mahalanobis2).any()
    ):
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=innovation_mahalanobis2,
                mode="lines",
                name="Innovation Mahalanobis²",
                line=dict(color="#8c564b", width=2),
            ),
            row=plot_idx,
            col=1,
        )
        if z_c_est is not None:
            n_contacts = z_c_est.shape[1]
            dof = 4 * n_contacts
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=np.full_like(steps, dof, dtype=float),
                    mode="lines",
                    name=f"χ² dof ≈ {dof}",
                    line=dict(color="black", width=1, dash="dash"),
                ),
                row=plot_idx,
                col=1,
            )
        fig.update_yaxes(title_text="Mahalanobis²", row=plot_idx, col=1)
        plot_idx += 1

    # P 对角线：base pos/vel + bias
    if plot_P and P_diag_hist is not None and np.isfinite(P_diag_hist).any():
        nx = P_diag_hist.shape[1]
        n_contacts = max(int((nx - 9) // 3), 0)
        bias_start = 6 + 3 * n_contacts

        # Base pos/vel variances
        labels_pv = ["P_p_x", "P_p_y", "P_p_z", "P_v_x", "P_v_y", "P_v_z"]
        for i in range(6):
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=P_diag_hist[:, i],
                    mode="lines",
                    name=labels_pv[i],
                    line=dict(width=2),
                ),
                row=plot_idx,
                col=1,
            )
        fig.update_yaxes(title_text="P base pos/vel var", row=plot_idx, col=1)
        plot_idx += 1

        # Bias variances
        bias_labels = ["P_bias_ax", "P_bias_ay", "P_bias_az"]
        for i in range(3):
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=P_diag_hist[:, bias_start + i],
                    mode="lines",
                    name=bias_labels[i],
                    line=dict(width=2),
                ),
                row=plot_idx,
                col=1,
            )
        fig.update_yaxes(title_text="P bias var", row=plot_idx, col=1)
        plot_idx += 1

    # R 对角线可视化
    if plot_R and R_diag_hist is not None and np.isfinite(R_diag_hist).any():
        n_contacts = R_diag_hist.shape[1] // 4
        # 位置噪声方差
        for i in range(n_contacts):
            idx_x = 3 * i + 0
            idx_y = 3 * i + 1
            idx_z = 3 * i + 2
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=R_diag_hist[:, idx_x],
                    mode="lines",
                    name=f"Foot {i+1} R_x ([-r_bc]_x var)",
                    line=dict(width=1.5),
                ),
                row=plot_idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=R_diag_hist[:, idx_y],
                    mode="lines",
                    name=f"Foot {i+1} R_y ([-r_bc]_y var)",
                    line=dict(width=1.5, dash="dot"),
                ),
                row=plot_idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=R_diag_hist[:, idx_z],
                    mode="lines",
                    name=f"Foot {i+1} R_z ([-r_bc]_z var)",
                    line=dict(width=1.5, dash="dash"),
                ),
                row=plot_idx,
                col=1,
            )
        fig.update_yaxes(
            title_text="R position var (m²)", row=plot_idx, col=1
        )
        plot_idx += 1

        # 高度噪声方差
        for i in range(n_contacts):
            idx_h = 3 * n_contacts + i
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=R_diag_hist[:, idx_h],
                    mode="lines",
                    name=f"Foot {i+1} R_zc (z_c var)",
                    line=dict(width=2),
                ),
                row=plot_idx,
                col=1,
            )
        fig.update_yaxes(
            title_text="R height var (m²)", row=plot_idx, col=1
        )
        plot_idx += 1

    # Layout and save/show
    fig.update_layout(
        width=1800,
        height=400 * n_subplots,
        margin=dict(l=60, r=40, t=120, b=50, pad=5),  # 添加紧凑边距
        title_text=(
            "State Estimator Validation Results<br>"
            f"<sub>RMSE(vel)={results['rmse']:.4f} m/s, "
            f"MAE(vel)={results['mae']:.4f} m/s"
            + (
                f", RMSE(pos)={results.get('rmse_pos', 0.0):.4f} m"
                if has_pos and results.get("rmse_pos") is not None
                else ""
            )
            + "</sub>"
        ),
        showlegend=True,
        hovermode="x unified",
    )

    fig.update_xaxes(title_text="Step", row=n_subplots, col=1)

    if save_path:
        fig.write_html(save_path, include_plotlyjs="cdn", full_html=True)
        print(f"Interactive plot saved to: {save_path}")
