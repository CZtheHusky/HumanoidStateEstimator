#!/usr/bin/env python3
"""
StateEstimator Validation Script (New Framework)

Validates the new state_estimator.StateEstimator implementation using
NPZ logs containing joint, IMU, and reference state data.

Usage:
    python validate_estimator.py --urdf /path/to/robot.urdf --log data.npz
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from state_estimator.state_estimator import (
    StateEstimator,
    RobotDynamicsModelConfig,
    GmoConfig,
    LinearKalmanFilterConfig,
    StateEstimatorConfig,
    JointMeasurement,
    ImuMeasurement,
    ContactMeasurement,
    EstimatorInput,
)
import sys
from tqdm import tqdm
from utils import (
    load_npz_log_with_slice,
    load_hw_log_to_npz_like_for_estimator,
    apply_config_overrides,
    plot_estimator_results_matplotlib,
    plot_estimator_results_interactive,
)

# Try to import PyYAML for saving config/metrics
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not available. Install with: pip install pyyaml")

class EstimatorValidator:
    """State Estimator Validation Tool (new framework)"""

    def __init__(
        self,
        estimator_config: StateEstimatorConfig,
        use_legacy_accel: bool = False,
    ):
        """
        Initialize validator

        Args:
            urdf_path: URDF file path
            foot_frames: List of contact point frame names
            gmo_config: GMO configuration (GmoConfig)
            kf_config: KF configuration (LinearKalmanFilterConfig)
            base_link: Base link name (RobotDynamicsModelConfig.base_link)
            imu_frame: IMU frame name (RobotDynamicsModelConfig.imu_frame)
            gravity: Gravity magnitude (RobotDynamicsModelConfig.gravity)
            soft_contact_duration: Soft contact ramp duration (StateEstimatorConfig)
            use_external_pose: Whether to use external pose (StateEstimatorConfig)
        """
        model_cfg = estimator_config.model
        self.model_config = model_cfg
        self.gmo_config = estimator_config.gmo
        self.kf_config = estimator_config.kf
        self.estimator_config = estimator_config
        self.estimator = StateEstimator(estimator_config)
        self.results_history = []
        self.needs_reorder = False  # Whether joint data needs to be reordered
        self.use_legacy_accel = use_legacy_accel

    def run_offline_validation(
        self,
        log_data: Dict[str, np.ndarray],
        print_every: int = 100,
    ) -> Dict:
        """
        Offline validation from log files (same NPZ format as legacy)

        Args:
            log_data: Log data dictionary containing:
                - 't': Timestamps (T,)
                - 'q': Generalized coordinates (T, n_q) or 'q_joints': Joint positions (T, n_j)
                - 'v': Generalized velocities (T, n_v) or 'v_joints': Joint velocities (T, n_j)
                - 'tau': Joint torques (T, n_tau)
                - 'imu_accel': IMU acceleration (T, 3)
                - 'imu_quat_wxyz': IMU quaternion (T, 4)
                - 'base_vel_gt': Ground truth base velocity (T, 3)
                - 'base_pos_gt' or infer from 'q'[:, :3]
                - 'z_c': Contact point heights (T, n_c) [optional]
                - 'contact_weights': Contact weights (T, n_c) [optional]
                - 'joint_names': Joint names (n_j,) [optional but recommended]
            print_every: Print interval

        Returns:
            results: Validation results dictionary
        """
        ticks = log_data['tick']
        T = len(ticks)

        # Extract joint data
        if "q_joints" in log_data:
            q_joints = log_data["q_joints"]
            v_joints = log_data["v_joints"]
        else:
            q = log_data["q"]
            v = log_data["v"]
            # Assume first 7 are base (pos + quat), rest are joints
            q_joints = q[:, 7:]
            # Assume first 6 are base velocity, rest are joints
            v_joints = v[:, 6:]

        tau = log_data["tau"]

        imu_accel = log_data["imu_accel"]
        imu_accel_legacy = log_data["imu_accel_legacy"]
        imu_quat_wxyz = log_data["imu_quat_wxyz"]
        base_vel_gt = log_data["base_vel_gt"]

        # Ground-truth base position (from generalized coordinates if available)
        if "base_pos_gt" in log_data:
            base_pos_gt = log_data["base_pos_gt"]
        elif "q" in log_data:
            base_pos_gt = log_data["q"][:, :3]
        else:
            base_pos_gt = None

        # IMU angular velocity
        imu_gyro = log_data.get("imu_gyro", None)

        # Ground truth foot heights / contact weights from log
        # 说明：为了与 legacy validate_estimator_legacy 完全对齐，
        # 新管线在“估计器输入”阶段不会使用这两个量，只在结果里保存和画图使用。
        z_c_gt = log_data.get("z_c", None)  # Ground truth foot heights (for logging/plot only)
        contact_weights_log = log_data.get("contact_weights", None)  # Ground truth contact weights (for plotting only)

        # Contacts count
        n_contacts = self.estimator.model.n_contacts

        # Buffers（使用列表，只在 EstimatorOutput 改变时追加，避免重复采样）
        v_est_list = []
        p_est_list = []
        error_list = []
        pos_error_list = [] if base_pos_gt is not None else None
        z_c_est_list = []
        contact_fz_list = []
        contact_prob_list = []
        imu_accel_bias_list = []
        zmp_offsets_list = []
        zmp_valid_list = []
        step_time_list = []

        # 对应用于绘图和 metric 的 GT，只在 EstimatorOutput 更新时采样
        t_kept = []
        base_vel_gt_kept = []
        base_pos_gt_kept = [] if base_pos_gt is not None else None
        z_c_gt_kept = [] if z_c_gt is not None else None
        contact_weights_log_kept = [] if contact_weights_log is not None else None

        # 残差（innovation）相关统计（仅在 EstimatorOutput 更新时记录）：
        #   - innovation_pos_hist: 每个脚的 [-r_bc]_x,y,z 残差
        #   - innovation_height_hist: 每个脚的足端高度残差 z_c
        #   - innovation_mahalanobis2_hist: 整体 Mahalanobis 距离平方
        innovation_pos_list = []
        innovation_height_list = []
        innovation_mahalanobis2_list = []

        # 测量噪声协方差 R 的对角线：长度 4*n_contacts，
        #   [(-r_bc_1)x, (-r_bc_1)y, (-r_bc_1)z, ..., (-r_bc_n)z, z_c1, ..., z_cn]
        R_diag_list = []

        # 状态协方差 P 的对角线（每次更新时记录一帧，对角长度在第一次看到 P 时确定）
        P_diag_list = []

        # 接触点初始化标志（记录哪些接触点在每个时刻被初始化）
        contact_initialized_list = []
        
        # 接触点激活状态（记录哪些接触点处于有效触地状态）
        contact_active_list = []

        # Initialize estimator
        print("Initializing estimator...")
        # 与 legacy 一致：仅用第一帧的真值位置初始化滤波器
        init_pos = base_pos_gt[0] if base_pos_gt is not None else None
        self.estimator.reset(initial_base_position=init_pos)

        print(f"Starting validation on {T} samples...")

        last_output_id = None

        for k in tqdm(range(T)):
            # Assemble EstimatorInput
            joint_meas = JointMeasurement(
                position=q_joints[k],
                velocity=v_joints[k],
                torque=tau[k],
            )

            imu_meas = ImuMeasurement(
                quat_wxyz=imu_quat_wxyz[k],
                accel_body=imu_accel_legacy[k] if self.use_legacy_accel else imu_accel[k],
                gyro_body=imu_gyro[k] if imu_gyro is not None else None,
            )

            # ContactMeasurement:
            # legacy pipeline 不会把 gt z_c / gt contact_weights 喂给估计器，只让 GMO+KF 自己推断接触，
            # 因此这里保持 contact=None，与 legacy 保持一致。
            contact_meas = None
            tick_k = int(ticks[k])

            est_input = EstimatorInput(
                joint=joint_meas,
                imu=imu_meas,
                contact=contact_meas,
                tick=tick_k,
            )

            # Run estimator and measure per-step time
            t0 = time.perf_counter()
            out = self.estimator.step(est_input)
            step_time_k = (time.perf_counter() - t0) * 1000.0

            # 仅在 EstimatorOutput 发生变化（即触发了一次 contact 更新）时记录结果
            is_new_output = (last_output_id is None) or (id(out) != last_output_id)
            last_output_id = id(out)

            if not is_new_output:
                # 输出未改变，只更新进度条和打印，不记录数据点
                if (print_every > 0 and (k % print_every) == 0) or k == T - 1:
                    print(
                        f"Step {k:6d}/{T} | t={(ticks[k] - ticks[0]) / 1000:.3f}s | "
                        f"(EstimatorOutput unchanged)"
                    )
                continue

            # === 下面开始真正记录一次“有效”的 EstimatorOutput 与对应 GT ===
            t_kept.append((ticks[k] - ticks[0]) / 1000.0)  # 转为相对秒数
            base_vel_gt_kept.append(base_vel_gt[k])
            if base_pos_gt is not None:
                base_pos_gt_kept.append(base_pos_gt[k])

            v_est_list.append(out.base_velocity_world.copy())
            p_est_list.append(out.base_position.copy())

            err_k = np.linalg.norm(out.base_velocity_world - base_vel_gt[k])
            error_list.append(err_k)
            if base_pos_gt is not None:
                pos_err_k = np.linalg.norm(out.base_position - base_pos_gt[k])
                pos_error_list.append(pos_err_k)

            z_c_est_list.append(out.contact_positions[:, 2].copy())
            contact_fz_list.append(out.contact_forces[:, 2].copy())
            contact_prob_list.append(out.contact_probabilities.copy())
            imu_accel_bias_list.append(out.imu_accel_bias.copy())
            step_time_list.append(step_time_k)

            # ZMP debug info (if available from estimator)
            if getattr(out, "zmp_offsets", None) is not None:
                zmp_offsets_list.append(out.zmp_offsets.copy())
            else:
                zmp_offsets_list.append(np.full((n_contacts, 2), 0))

            if getattr(out, "zmp_valid", None) is not None:
                zmp_valid_list.append(out.zmp_valid.copy())
            else:
                zmp_valid_list.append(np.full(n_contacts, False))

            # 记录 GT 足端高度 / contact_weights（如果日志中有）
            if z_c_gt is not None:
                z_c_gt_kept.append(z_c_gt[k])
            if contact_weights_log is not None:
                contact_weights_log_kept.append(contact_weights_log[k])

            # 状态协方差 P 的对角线
            p_valid = False
            if (
                getattr(out, "state_covariance", None) is not None
                and out.state_covariance is not None
            ):
                P = np.asarray(out.state_covariance, dtype=float)
                diag_P = np.diag(P)
                P_diag_list.append(diag_P)
                p_valid = True
            
            if not p_valid:
                dim_P = self.estimator.kf.state_dimension
                P_diag_list.append(np.full(dim_P, 0))

            # 残差统计（仅在本 step 发生 KF 接触测量更新时有效）
            innov_valid = False
            if getattr(out, "innovation", None) is not None:
                innov = out.innovation
                if innov is not None:
                    innov = np.asarray(innov, dtype=float).reshape(-1)
                    expected_dim = 4 * n_contacts  # 3*n_c + n_c
                    if innov.shape[0] == expected_dim:
                        # 前 3*n_c 维：每个脚的 [-r_bc]_x,y,z 残差
                        per_contact = np.zeros((n_contacts, 3), dtype=float)
                        for i in range(n_contacts):
                            per_contact[i, :] = innov[3 * i : 3 * (i + 1)]
                        innovation_pos_list.append(per_contact)
                        # 后 n_c 维：每个脚的高度残差 z_c
                        innovation_height_list.append(
                            innov[3 * n_contacts :].copy()
                        )
                        innov_valid = True
            
            if not innov_valid:
                innovation_pos_list.append(np.full((n_contacts, 3), 0))
                innovation_height_list.append(np.full(n_contacts, 0))

            if getattr(out, "innovation_mahalanobis2", None) is not None:
                innovation_mahalanobis2_list.append(out.innovation_mahalanobis2)
            else:
                innovation_mahalanobis2_list.append(0)

            # 测量噪声协方差 R 的对角线（仅在本 step 有接触更新时有效）
            r_valid = False
            if getattr(out, "R_diag", None) is not None and out.R_diag is not None:
                r = np.asarray(out.R_diag, dtype=float).reshape(-1)
                if r.shape[0] == 4 * n_contacts:
                    R_diag_list.append(r)
                    r_valid = True
            
            if not r_valid:
                R_diag_list.append(np.full(4 * n_contacts, 0))

            # 接触点初始化标志
            if getattr(out, "contact_initialized", None) is not None:
                contact_initialized_list.append(out.contact_initialized.copy())
            else:
                contact_initialized_list.append(np.zeros(n_contacts, dtype=bool))
            
            # 接触点激活状态
            if getattr(out, "contact_active", None) is not None:
                contact_active_list.append(out.contact_active.copy())
            else:
                contact_active_list.append(np.zeros(n_contacts, dtype=bool))

            self.results_history.append(out)

            # Progress print（仍按原始步数打印）
            if (print_every > 0 and (k % print_every) == 0) or k == T - 1:
                pos_err_str = ""
                if base_pos_gt is not None and pos_error_list is not None and pos_error_list:
                    pos_err_str = f"p_err={pos_error_list[-1]:.4f} m | "
                print(
                    f"Step {k:6d}/{T} | t={(ticks[k] - ticks[0]) / 1000:.3f}s | "
                    f"err={error_list[-1]:.4f} m/s | {pos_err_str}"
                    f"v_est=[{out.base_velocity_world[0]:.3f}, "
                    f"{out.base_velocity_world[1]:.3f}, "
                    f"{out.base_velocity_world[2]:.3f}] | "
                    f"v_gt=[{base_vel_gt[k][0]:.3f}, "
                    f"{base_vel_gt[k][1]:.3f}, "
                    f"{base_vel_gt[k][2]:.3f}]"
                )

        # Statistics
        # 将列表转换为 numpy 数组，长度只等于“有效 EstimatorOutput”次数
        v_est_hist = np.asarray(v_est_list, dtype=float)
        p_est_hist = np.asarray(p_est_list, dtype=float)
        error_hist = np.asarray(error_list, dtype=float)
        if pos_error_list is not None:
            pos_error_hist = np.asarray(pos_error_list, dtype=float)
        else:
            pos_error_hist = None

        timestamps = np.asarray(t_kept, dtype=float)
        base_vel_gt = np.asarray(base_vel_gt_kept, dtype=float)
        if base_pos_gt_kept is not None:
            base_pos_gt = np.asarray(base_pos_gt_kept, dtype=float)
        else:
            base_pos_gt = None

        z_c_est_hist = np.asarray(z_c_est_list, dtype=float) if z_c_est_list else None
        contact_fz_hist = (
            np.asarray(contact_fz_list, dtype=float) if contact_fz_list else None
        )
        contact_prob_hist = (
            np.asarray(contact_prob_list, dtype=float) if contact_prob_list else None
        )
        imu_accel_bias_hist = (
            np.asarray(imu_accel_bias_list, dtype=float) if imu_accel_bias_list else None
        )
        zmp_offsets_hist = (
            np.asarray(zmp_offsets_list, dtype=float) if zmp_offsets_list else None
        )
        zmp_valid_hist = (
            np.asarray(zmp_valid_list, dtype=bool) if zmp_valid_list else None
        )
        step_time_ms = (
            np.asarray(step_time_list, dtype=float) if step_time_list else None
        )

        if z_c_gt_kept is not None:
            z_c_gt = np.asarray(z_c_gt_kept, dtype=float)
        if contact_weights_log_kept is not None:
            contact_weights_log = np.asarray(contact_weights_log_kept, dtype=float)

        if innovation_pos_list:
            innovation_pos_hist = np.asarray(innovation_pos_list, dtype=float)
        else:
            innovation_pos_hist = None
        if innovation_height_list:
            innovation_height_hist = np.asarray(innovation_height_list, dtype=float)
        else:
            innovation_height_hist = None
        if innovation_mahalanobis2_list:
            innovation_mahalanobis2_hist = np.asarray(
                innovation_mahalanobis2_list, dtype=float
            )
        else:
            innovation_mahalanobis2_hist = None

        if P_diag_list:
            P_diag_hist = np.asarray(P_diag_list, dtype=float)
        else:
            P_diag_hist = None
        if R_diag_list:
            R_diag_hist = np.asarray(R_diag_list, dtype=float)
        else:
            R_diag_hist = None

        if contact_initialized_list:
            contact_initialized_hist = np.asarray(contact_initialized_list, dtype=bool)
        else:
            contact_initialized_hist = None
        
        if contact_active_list:
            contact_active_hist = np.asarray(contact_active_list, dtype=bool)
        else:
            contact_active_hist = None

        # Statistics（基于"有效输出"个数）
        rmse = float(np.sqrt(np.mean(error_hist**2))) if error_hist.size > 0 else 0.0
        mae = float(np.mean(np.abs(error_hist))) if error_hist.size > 0 else 0.0
        max_error = float(np.max(error_hist)) if error_hist.size > 0 else 0.0

        if base_pos_gt is not None and pos_error_hist is not None and pos_error_hist.size > 0:
            rmse_pos = float(np.sqrt(np.mean(pos_error_hist**2)))
            mae_pos = float(np.mean(np.abs(pos_error_hist)))
            max_pos_error = float(np.max(pos_error_hist))
        else:
            rmse_pos = mae_pos = max_pos_error = None

        # 残差 Mahalanobis 距离统计（仅统计有效条目）
        innovation_stats = None
        if innovation_mahalanobis2_hist is not None:
            valid_mask = np.isfinite(innovation_mahalanobis2_hist)
            n_valid_residuals = int(np.sum(valid_mask))
            if n_valid_residuals > 0:
                d2_valid = innovation_mahalanobis2_hist[valid_mask]
                innovation_stats = {
                    "count": n_valid_residuals,
                    "mean": float(np.mean(d2_valid)),
                    "median": float(np.median(d2_valid)),
                    "p95": float(np.percentile(d2_valid, 95.0)),
                }

        # 统计接触点初始化次数
        n_initializations = 0
        if contact_initialized_hist is not None:
            n_initializations = int(np.sum(contact_initialized_hist))

        print("\n" + "=" * 70)
        print("Validation Results (new estimator):")
        print("=" * 70)
        print(f"  Raw log samples: {T}")
        print(f"  Effective estimator outputs: {len(error_hist)}")
        print(f"  Contact initializations: {n_initializations}")
        print(f"  RMSE (velocity): {rmse:.4f} m/s")
        print(f"  MAE (velocity):  {mae:.4f} m/s")
        print(f"  Max error:       {max_error:.4f} m/s")
        if rmse_pos is not None:
            print(f"  RMSE (position): {rmse_pos:.4f} m")
            print(f"  MAE (position):  {mae_pos:.4f} m")
            print(f"  Max error (pos): {max_pos_error:.4f} m")
        if innovation_stats is not None:
            print("-" * 70)
            print("  Innovation Mahalanobis^2 statistics (contact measurements):")
            print(f"    Count:   {innovation_stats['count']}")
            print(f"    Mean:    {innovation_stats['mean']:.3f}")
            print(f"    Median:  {innovation_stats['median']:.3f}")
            print(f"    95%-tile:{innovation_stats['p95']:.3f}")
        print("=" * 70)

        return {
            "T": len(error_hist),
            "v_est_hist": v_est_hist,
            "p_est_hist": p_est_hist,
            "error_hist": error_hist,
            "rmse": rmse,
            "mae": mae,
            "max_error": max_error,
            "timestamps": timestamps,
            "base_vel_gt": base_vel_gt,
            "base_pos_gt": base_pos_gt,
            "pos_error_hist": pos_error_hist,
            "rmse_pos": rmse_pos,
            "mae_pos": mae_pos,
            "max_pos_error": max_pos_error,
            "imu_accel_bias_hist": imu_accel_bias_hist,
            "z_c_est": z_c_est_hist,
            "z_c_gt": z_c_gt,
            "contact_fz": contact_fz_hist,
            "contact_prob": contact_prob_hist,
            "contact_weights_log": contact_weights_log,
            "zmp_offsets": zmp_offsets_hist,
            "zmp_valid": zmp_valid_hist,
            "step_time_ms": step_time_ms,
            "imu_accel": imu_accel,
            "imu_accel_legacy": imu_accel_legacy,
            "innovation_pos": innovation_pos_hist,
            "innovation_height": innovation_height_hist,
            "innovation_mahalanobis2_hist": innovation_mahalanobis2_hist,
            "P_diag_hist": P_diag_hist,
            "R_diag_hist": R_diag_hist,
            "innovation_stats": innovation_stats,
            "contact_initialized": contact_initialized_hist,
            "contact_active": contact_active_hist,
        }

    @staticmethod
    def save_results(results: Dict, output_path: str):
        """
        Save validation results to file (npz), similar to legacy save_results.
        """
        np.savez(
            output_path,
            timestamps=results["timestamps"],
            v_est_hist=results["v_est_hist"],
            p_est_hist=results["p_est_hist"],
            base_vel_gt=results["base_vel_gt"],
            base_pos_gt=results.get("base_pos_gt"),
            error_hist=results["error_hist"],
            pos_error_hist=results.get("pos_error_hist"),
            rmse=results["rmse"],
            mae=results["mae"],
            max_error=results["max_error"],
            rmse_pos=results.get("rmse_pos"),
            mae_pos=results.get("mae_pos"),
            max_pos_error=results.get("max_pos_error"),
            imu_accel_bias_hist=results.get("imu_accel_bias_hist"),
            zmp_offsets=results.get("zmp_offsets"),
            zmp_valid=results.get("zmp_valid"),
            step_time_ms=results.get("step_time_ms"),
            imu_accel=results.get("imu_accel"),
            imu_accel_legacy=results.get("imu_accel_legacy"),
            contact_weights_log=results.get("contact_weights_log"),
            innovation_mahalanobis2_hist=results.get("innovation_mahalanobis2_hist"),
        )
        print(f"New estimator results saved to: {output_path}")

    def plot_results(self, results: Dict, save_path: str = None):
        """
        Plot validation results.

        If save_path ends with ".html", delegate to the Plotly-based helper in utils;
        otherwise use the Matplotlib-based helper.
        """
        if save_path is not None and str(save_path).lower().endswith(".html"):
            plot_estimator_results_interactive(results, save_path=save_path)
        else:
            plot_estimator_results_matplotlib(results, save_path=save_path)

def validate_file(log_path: Path, args, batch_root: Optional[Path] = None, unknown_args: list = None) -> None:
    # get the last component as well as the second last component if exists
    log_file = log_path.name
    if len(log_path.parts) >= 2:
        log_file = log_path.parts[-2] + "_" + log_file

    print(f"Loading log data from: {log_path}")
    if log_path.suffix == ".npz":
        log_data = load_npz_log_with_slice(log_path, start=args.start, end=args.end)
    else:
        log_data = load_hw_log_to_npz_like_for_estimator(log_path, start=args.start, end=args.end)

    # Build configuration objects with defaults
    # Note: We removed explicit CLI args for config fields to simplify.
    # Users should use key=value overrides.
    
    gmo_config = GmoConfig()
    kf_config = LinearKalmanFilterConfig()
    
    # RobotDynamicsModelConfig needs some args from CLI
    joint_names = log_data.get("joint_names", None)
    joint_names = [str(name) for name in joint_names] if joint_names is not None else None
    
    # We use defaults for base_link, imu_frame, gravity unless overridden later
    model_cfg = RobotDynamicsModelConfig(
        urdf_path=args.urdf,
        external_joint_names=joint_names,
    )
    
    # Create top-level config
    estimator_config = StateEstimatorConfig(
        model=model_cfg,
        gmo=gmo_config,
        kf=kf_config,
        # soft_contact_duration and use_external_pose defaults are in StateEstimatorConfig
    )

    # Apply overrides from unknown_args
    if unknown_args:
        apply_config_overrides(estimator_config, unknown_args)
    print(estimator_config.to_dict())

    # New estimator (this file)
    # We need to reconstruct the validator with the updated config
    # But EstimatorValidator takes individual configs in __init__.
    # So we pass the updated objects.
    
    new_validator = EstimatorValidator(
        estimator_config=estimator_config,
        use_legacy_accel=args.legacy_accl,
    )
    # Hack: The validator init creates a NEW StateEstimatorConfig. 
    # We should ensure it uses exactly what we prepared.
    # The current EstimatorValidator.__init__ reconstructs it.
    # Let's modify EstimatorValidator to accept a pre-built config or just trust that passing the sub-configs is enough.
    # Since we updated the sub-configs (gmo, kf, model) in place (they are mutable dataclasses), 
    # passing them to __init__ should work, provided __init__ uses them.
    # However, StateEstimatorConfig also has scalar fields (soft_contact_duration, etc).
    # We passed those explicitly above.
    
    results_new = new_validator.run_offline_validation(log_data, print_every=-1)
    # Attach plotting control flags so that utils can decide which diagnostics to render
    results_new["plot_R"] = not getattr(args, "no_plot_R", False)
    results_new["plot_P"] = not getattr(args, "no_plot_P", False)
    results_new["plot_innovation"] = not getattr(args, "no_plot_innovation", False)

    # Determine output directory
    if batch_root is not None:
        output_dir = batch_root / log_path.name
    elif args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.legacy_accl:
            output_dir = Path("validate_results") / f"{log_file.replace('/', '_')}_{args.start}_{args.end}_legacy_{timestamp}"
        else:
            output_dir = Path("validate_results") / f"{log_file.replace('/', '_')}_{args.start}_{args.end}_{timestamp}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if YAML_AVAILABLE:
        # Use the new to_dict() method to serialize the full configuration
        config_dict = new_validator.estimator_config.to_dict()

        metrics = {
            "T": float(results_new["T"]),
            "num_samples": int(len(results_new["timestamps"])),
            "rmse_velocity": float(results_new["rmse"]),
            "mae_velocity": float(results_new["mae"]),
            "max_velocity_error": float(results_new["max_error"]),
        }
        if results_new.get("rmse_pos") is not None:
            metrics["rmse_position"] = float(results_new["rmse_pos"])
        if results_new.get("mae_pos") is not None:
            metrics["mae_position"] = float(results_new["mae_pos"])
        if results_new.get("max_pos_error") is not None:
            metrics["max_position_error"] = float(results_new["max_pos_error"])
        innovation_stats = results_new.get("innovation_stats")
        if innovation_stats is not None:
            metrics["innovation_mahalanobis2_count"] = int(innovation_stats.get("count", 0))
            metrics["innovation_mahalanobis2_mean"] = float(innovation_stats.get("mean", 0.0))
            metrics["innovation_mahalanobis2_median"] = float(innovation_stats.get("median", 0.0))
            metrics["innovation_mahalanobis2_p95"] = float(innovation_stats.get("p95", 0.0))

        summary = {
            "log_path": str(log_path),
            "config": config_dict,
            "metrics": metrics,
        }

        yaml_path = output_dir / "summary.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(summary, f, sort_keys=False)
        print(f"Config and metrics saved to: {yaml_path}")
    else:
        print("Warning: PyYAML not installed, skip saving YAML summary.")

    # Determine figure paths (always save both matplotlib PNG + interactive HTML)
    static_fig_path = output_dir / "validation_matplotlib.png"
    html_fig_path = output_dir / "validation_interactive.html"

    static_fig_path.parent.mkdir(parents=True, exist_ok=True)
    html_fig_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot NEW estimator results
    new_validator.plot_results(results_new, save_path=str(static_fig_path))
    new_validator.plot_results(results_new, save_path=str(html_fig_path))


def main():
    parser = argparse.ArgumentParser(description="Validate new StateEstimator")
    parser.add_argument("--urdf", required=True, help="Robot URDF file path")
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="NPZ log file with estimator inputs",
    )
    parser.add_argument('--start', type=int, default=0, help='Start frame index (default: 0)')
    parser.add_argument('--end', type=int, default=-1, help='End frame index (default: -1 for all)')
    # Note: We removed explicit config arguments (gmo_..., kf_..., etc.)
    # to support generic key=value overrides.
    # e.g. python validate_estimator.py ... kf.imu_accel_noise_density=0.5

    # Output control
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory to save validation outputs (npz/png/html/yaml). "
            "Default: ./validate_results/{YYYYmmdd_HHMMSS}. "
            "Both matplotlib PNG and Plotly HTML figures will be saved here."
        ),
    )
    parser.add_argument(
        "--save_results_prefix",
        type=str,
        default=None,
        help=(
            "If set, save new estimator results to '<prefix>_new.npz'. "
            "If prefix is relative, it is interpreted inside output-dir. "
            "Default: <output-dir>/results_new.npz"
        ),
    )
    parser.add_argument(
        "--legacy_accl",
        action="store_true",
        help="Use legacy IMU acceleration reconstruction for state estimator.",
    )
    # Visualization control
    parser.add_argument(
        "--no_plot_R",
        action="store_true",
        help="Disable visualization of measurement noise R (R_diag).",
    )
    parser.add_argument(
        "--no_plot_P",
        action="store_true",
        help="Disable visualization of state covariance P.",
    )
    parser.add_argument(
        "--no_plot_innovation",
        action="store_true",
        help="Disable visualization of innovation and Mahalanobis².",
    )

    # Parse known args first, leaving unknown args for config overrides
    args, unknown = parser.parse_known_args()

    log_path = Path(args.log)
    if not log_path.is_file():
        # Collect all valid files
        files_to_process = [
            p for p in log_path.iterdir() 
            if p.suffix in (".npz", ".pkl")
        ]
        
        if not files_to_process:
            print(f"No .npz or .pkl files found in {log_path}")
            return

        print(f"Found {len(files_to_process)} files to process in parallel...")
        
        # Calculate batch output root
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_root = Path("validate_results") / f"{log_path.name}_{timestamp}"

        # Use ProcessPoolExecutor for parallel processing (bypasses GIL for CPU-bound tasks)
        import concurrent.futures
        # Set max_workers to 64 as requested. 
        # Note: For optimal performance, ensure OMP_NUM_THREADS=1 in your environment 
        # to avoid thread oversubscription if numpy/pinocchio are also multi-threaded.
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                # Submit all tasks
                # Pass unknown args to worker
                futures = [executor.submit(validate_file, p, args, batch_root, unknown) for p in files_to_process]
                
                # Wait for all to complete and handle exceptions
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Task generated an exception: {e}")
        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt, shutting down executor...")
            executor.shutdown(wait=False, cancel_futures=True)
            print("Executor shutdown complete.")
            sys.exit(1)
    else:
        validate_file(log_path, args, unknown_args=unknown)

if __name__ == "__main__":
    main()
