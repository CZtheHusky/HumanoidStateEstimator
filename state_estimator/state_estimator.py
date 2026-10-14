# pyright: reportInvalidTypeForm=false

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Any, Dict
import numpy as np
from numpy.typing import NDArray
import pinocchio as pin
from scipy.special import expit

# ========== 通用类型别名 ==========
Array3 = NDArray[np.float64]  # shape (3,)
Vec = NDArray[np.float64]
Mat = NDArray[np.float64]

@dataclass
class RobotDynamicsModelConfig:
    """
    对应 legged::LeggedModel 里的几何/动力学配置（去掉 ROS 部分）。
    """
    urdf_path: str = ""
    # 外部关节名顺序（例如日志 / 控制器使用的顺序）
    # 如果提供，RobotDynamicsModel 会在初始化时预先建立从 external → Pinocchio 顺序的重排索引。
    contact_frames: List[str] = field(default_factory=lambda: ["LL_FOOT", "LR_FOOT"])
    external_joint_names: List[str] = field(default_factory=lambda: [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", 
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint", 
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
    ])
    base_link: str = "base"
    imu_frame: str = "imu_link"
    gravity: float = 9.81

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



@dataclass
class GmoConfig:
    """
    对应 legged_estimation::GeneralizedMomentumObserver::Configurations.
    字段命名尽量贴近 C++ 版本。
    """
    # 低通滤波 λ
    lpf_lambda: float = 50.0
    # 伪逆阻尼
    pinv_damping: float = 1e-6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self):
        assert self.lpf_lambda > 0
        assert self.pinv_damping > 0


@dataclass
class LinearKalmanFilterConfig:
    """
    对应 legged_estimation::LinearKalmanFilter::Configurations.

    C++ 成员:
      imuAccelerationNoiseDensity      -> IMU 加速度噪声密度
      imuAccelerationBiasNoiseDensity  -> IMU 加速度 bias 随机游走噪声
      contactProcessNoisePosition      -> 接触点位置过程噪声
      contactSensorNoisePosition       -> r_bc 约束测量噪声（FK）
      contactHeightSensorNoise         -> 足端高度测量噪声
      contactRadius                    -> 接触 patch 半径（ZMP 用）
      contactForceThreshold            -> 接触判定阈值
      contactForceScale                -> 接触概率缩放
      contactZmpLengthX / Y            -> ZMP 支撑域长宽
      positionNoiseDensity             -> 额外位置传感器噪声（外部里程计）
    """
    imu_accel_noise_density: float = 1e-2
    imu_accel_bias_noise_density: float = 5e-3
    contact_process_noise_position: float = 0.002
    contact_sensor_noise_position: float = 0.002
    contact_height_sensor_noise: float = 0.002
    contact_radius: float = 0.0
    contact_force_threshold: float = 150.0
    contact_force_scale: float = 15.0
    contact_zmp_length_x: float = 0.08
    contact_zmp_length_y: float = 0.025
    # 外部位置观测噪声（离散测量标准差）
    # 单位：m
    #
    # 对 mocap/视觉里程计这类“离散 pose 测量”，更合理的模型通常是：
    #   R_pos = σ_pos^2 * I3
    # 而不是连续白噪声谱密度随 dt 离散化。
    position_noise_density: float = 1e-2
    # 是否启用基于 ZMP 的接触点初始化 / 可视化（关闭后返回零偏移）
    dynamic_height_noise: bool = True
    use_zmp: bool = False
    zmp_force_threshold: Optional[float] = None
    
    epsilon_no_contact: float = 500.0  # 非接触时噪声放大因子
    init_pos_cov: float = 0.05
    init_vel_cov: float = 0.05
    init_pc_cov: float = 0.1
    init_bias_cov: float = 0.2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self):
        assert self.imu_accel_noise_density > 0
        assert self.imu_accel_bias_noise_density > 0
        assert self.contact_process_noise_position >= 0
        assert self.contact_sensor_noise_position > 0
        assert self.contact_height_sensor_noise > 0
        assert self.epsilon_no_contact >= 1.0
        assert self.position_noise_density > 0

@dataclass
class StateEstimatorConfig:
    """
    顶层状态估计器配置：聚合 LeggedModel / GMO / KF 的配置。
    对应 legged_controllers::StateEstimator 里持有的几个对象。
    """
    model: RobotDynamicsModelConfig = field(default_factory=RobotDynamicsModelConfig)
    gmo: GmoConfig = field(default_factory=GmoConfig)
    kf: LinearKalmanFilterConfig = field(default_factory=LinearKalmanFilterConfig)
    # IMU / 接触更新频率（Hz）。EstimatorInput 需要以不低于这两个频率的 tick 进来，
    # StateEstimator 内部使用 tick 差分 + 简单调度来决定何时运行 IMU 预测和接触更新。
    imu_frenquency: float = 500.0  # Hz
    contact_frequency: float = 100.0  # Hz
    # 新接触软启动持续时间（秒）
    soft_contact_duration: float = 0.05
    # 是否启用接触 soft-start（权重从 soft_weight → 1.0 的平滑）
    use_soft_start: bool = False    # will jeopardize performance if enabled, when there is jittering in gmo estimation
    # 是否启用新接触点位置的初始化逻辑（若关闭，则沿用上一时刻的滤波器状态）
    contact_pos_init: bool = True    # TODO: we need better method to determine whether there is a solid contact

    # 可选：用来控制是否使用外部里程计 / 姿态输入（对应 poseSubscriber_）
    use_external_pose: bool = False
    # 外部“世界系”(例如 map/mocap world) -> 估计器内部 world 的刚体变换 SE(3)。
    # 用于在使用 inp.external_position 之前，将其统一预处理到内部 world。
    #
    # 约定：
    #   p_world_internal = R @ p_world_external + t
    # 其中 (q_wxyz, t_xyz) 由 external_world_to_ego 给出。
    #
    # 当前实现仅使用：yaw + xy 平移 对齐（忽略 roll/pitch 与 z 平移）：
    #   - 从 q_wxyz 提取 yaw，构造 Rz(yaw)
    #   - t 仅使用 [tx, ty]，tz 强制为 0
    #
    # 默认：单位旋转 + 零平移（表示外部 world 与内部 world 已对齐）。
    external_world_to_ego: Tuple[Vec, Vec] = field(
        default_factory=lambda: (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            np.zeros(3, dtype=float),
        )
    )

    # 可选：是否在启动时使用一段“静止 IMU 数据”做 Fast-LIO 风格的 IMU 初始化
    # （统计加速度 / 角速度的均值与方差，用于设置初始加速度 bias 等）
    use_imu_static_init: bool = False
    # 静态初始化所需的最小 IMU 样本数
    imu_init_min_samples: int = 200

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JointMeasurement:
    """
    一帧的关节传感器数据，对应 JointSensors 语义组件里读出来的内容。
    """
    position: Vec             # shape (n_j,)
    velocity: Vec             # shape (n_j,)
    torque: Vec               # shape (n_j,)



@dataclass
class ImuMeasurement:
    """
    单个 IMU 数据，假定已经选定使用哪一个 IMU（比如 pelvis 的那个）。
    """
    quat_wxyz: Vec        # shape (4,)
    accel_body: Array3       # shape (3,), 线加速度（不含重力 or 已经按你定义预处理）
    gyro_body: Optional[Array3] = None  # shape (3,), 角速度
    frame: str = "imu_link"

@dataclass
class ContactMeasurement:
    """
    每一帧的接触相关信息。

    legged_control2 里：GmObserver 使用脚底 wrench 推接触概率，
    LinearKalmanFilter 使用 contactProbabilities_、feetHeights_ 等。
    """
    # 每个接触点的 6D wrench（一般来自 GMO 估计 / 力传感器）
    wrenches: Optional[Vec] = None             # shape (n_contacts, 6) or flattened
    # 外部判定的接触概率（0~1），如果 None，就由 GMO 计算
    contact_probabilities: Optional[Vec] = None  # shape (n_contacts,)
    # 足端高度观测（世界系 z）：
    #   - 为 None 时：StateEstimator 会将内部 feet_heights 设为 0.0（假定地面在 z=0），
    #     并启用“根据接触权重自适应调整足端高度噪声”的机制；
    #   - 不为 None 时：使用外部提供的高度观测，并关闭高度噪声的动态调整，
    #     对所有脚使用统一的 contact_height_sensor_noise。
    feet_heights: Optional[Vec] = None           # shape (n_contacts,)

@dataclass
class ExternalPoseMeasurement:
    """
    外部里程计 / 姿态测量（对应 poseSubscriber_）。
    """
    position: Array3        # shape (3,), 世界系位置
    quat_wxyz: Vec         # shape (4,), 世界系姿态
    tick: Optional[int] = None  # 时间戳（与 EstimatorInput.tick 一致的单位）,但不一定完全相同

@dataclass
class EstimatorInput:
    """
    顶层 StateEstimator 每次 step 所需的所有输入。

    注意：
      - 多频率/预积分模式下，tick 为必需字段（单位：与硬件 tick 一致，这里约定为毫秒）。
      - dt 仅作为辅助调试字段，StateEstimator.step 不再使用 dt 作为 fallback。
    """
    joint: JointMeasurement
    imu: ImuMeasurement
    contact: Optional[ContactMeasurement] = None
    # 时间戳：来自硬件 / 日志的 tick（整数），单位毫秒。
    tick: Optional[int] = None
    # 旧接口中的 dt 不再作为 fallback 使用，这里保留字段仅为兼容；StateEstimator.step 不依赖该字段。
    dt: Optional[float] = None

    # 可选：外部里程计/姿态输入（对应 poseSubscriber_）
    # 约定：position/quat_wxyz 均在 mocap/world 系下。
    external_position: Optional[ExternalPoseMeasurement] = None

@dataclass
class EstimatorOutput:
    """
    StateEstimator 的主输出结构，类似于 legged_control2 中
    LinearKalmanFilter + GMO 提供给控制器的状态。
    """
    base_position: Array3
    base_velocity_world: Array3
    base_velocity_body: Array3

    contact_positions: Vec           # shape (n_contacts, 3)
    contact_forces: Vec              # shape (n_contacts, 6)
    contact_probabilities: Vec       # shape (n_contacts,)
    tau_external: Vec                # shape (n_j,)

    imu_accel_bias: Array3

    # 可选：ZMP 偏移 / 有效标志（按接触点）
    # zmp_offsets: (n_contacts, 2) -> [dx, dy] in world x-y plane
    # zmp_valid  : (n_contacts,)   -> bool mask for valid ZMP
    zmp_offsets: Optional[Vec] = None
    zmp_valid: Optional[Vec] = None

    # 可选：协方差、外力等调试信息
    state_covariance: Optional[Mat] = None
    base_wrench: Optional[Array3] = None
    # 可选：最近一次接触测量的残差及其 Mahalanobis 距离
    # 仅在该 step 内实际触发了 KF 接触测量更新时有效，其余 step 通常为 None。
    innovation: Optional[Vec] = None
    innovation_mahalanobis2: Optional[float] = None
    # 可选：最近一次构造的测量噪声协方差 R 的对角线，
    # 按照 [(-r_bc_1)x, (-r_bc_1)y, (-r_bc_1)z, ..., (-r_bc_n)z, z_c1, ..., z_cn] 排列。
    R_diag: Optional[Vec] = None
    
    # 可选：记录哪些接触点在当前时刻被初始化了（新接触点初始化）
    # contact_initialized: (n_contacts,) -> bool mask, True表示该接触点在本时刻被初始化
    contact_initialized: Optional[Vec] = None
    contact_active: Optional[Vec] = None

@dataclass
class IMUIntegral:
    """
    从上一段“积分起点”到当前时刻，IMU 预积分的结果。

    dt:      tick 差（单位：ms）
    accel_body: 这一整段时间上的“平均机体系加速度”
                （时间加权平均： ∫a(t)dt / ∫dt）
    """
    dt: int
    accel_body: Array3


@dataclass
class ImuInitResult:
    """
    IMU 静止阶段统计结果，借鉴 Fast-LIO 中 IMU_init 的思路：
      - mean_accel_body / mean_gyro_body: 均值（可用于初始化 bias）
      - cov_accel_body / cov_gyro_body: 方差（对角线），可用于估计噪声尺度
    """
    mean_accel_body: Array3
    mean_gyro_body: Array3
    bias_accel_body: Array3
    cov_accel_body: Array3
    cov_gyro_body: Array3
    sample_count: int


class ImuPreintegrator:
    """
    极简版 IMU 预积分：
    - 连续吃 EstimatorInput（主要用到 imu + tick/dt）
    - 内部累计 "accel * dt"
    - 需要的时候吐出一个 IMUIntegral（平均 accel + 时间长度），并重置
    """

    def __init__(self):
        self.reset()

    def reset(self, start_tick: Optional[int] = None,
              last_accel: Optional[Array3] = None) -> None:
        # 积分相关量
        self._sum_accel_dt = np.zeros(3, dtype=float)  # ∫ a dt
        self._sum_dt = 0.0                             # 总时间（秒）

        # 时间戳（以 tick 计数，单位与 EstimatorInput.tick 相同，通常为毫秒）
        self._start_tick: Optional[int] = start_tick
        self._end_tick: Optional[int] = start_tick

        # 保存上一帧，用梯形积分。None 表示还未开始积分。
        self._last_tick: Optional[int] = start_tick
        self._last_accel: Optional[Array3] = (
            last_accel.copy() if last_accel is not None else None
        )

    def push(self, est_in: EstimatorInput) -> None:
        """
        推入一帧输入（通常是每个估计循环来一次）。
        只使用 est_in.imu 和 est_in.tick（单位：毫秒）。
        """
        imu = est_in.imu
        accel = np.asarray(imu.accel_body, dtype=float)
        if est_in.tick is None:
            # 预积分模式下必须提供 tick
            raise ValueError("EstimatorInput.tick is required for IMU preintegration")
        if self._last_tick is None:
            # 这是第一帧，只记录时间和加速度，不积分
            self._last_tick = est_in.tick
            self._start_tick = est_in.tick
            self._end_tick = est_in.tick
            self._last_accel = accel
            return
        dt = (est_in.tick - self._last_tick) * 1e-3  # tick(ms) → s

        if dt <= 0:
            return

        # 如果是第一次真正开始积分，初始化起点
        if self._start_tick is None and est_in.tick is not None:
            self._start_tick = est_in.tick

        # 梯形积分：∫ a dt ≈ 0.5 * (a_{k-1} + a_k) * dt
        if self._last_accel is None:
            # 没有上一帧，只用当前加速度做矩形近似
            self._sum_accel_dt += accel * dt
        else:
            self._sum_accel_dt += 0.5 * (self._last_accel + accel) * dt

        self._sum_dt += dt

        # 更新时间戳 & 最后一帧
        if est_in.tick is not None:
            self._end_tick = est_in.tick
            self._last_tick = est_in.tick
        self._last_accel = accel

    def get_integral(self, reset: bool = True) -> Optional[IMUIntegral]:
        """
        取出当前这段预积分的结果，并（可选）重置内部状态。

        返回:
            - None: 还没有效积分
            - IMUIntegral: dt(ms), 平均 accel_body
        """
        if self._start_tick is None or self._end_tick is None:
            return None
        if self._sum_dt <= 0.0:
            return IMUIntegral(
                dt=20,
                accel_body=np.zeros(3, dtype=float),
            )

        dt_ms = int(self._end_tick - self._start_tick)
        assert dt_ms > 0, f"Non-positive integral dt: {dt_ms} ms"   

        avg_accel = self._sum_accel_dt / self._sum_dt

        out = IMUIntegral(
            dt=dt_ms,
            accel_body=avg_accel,
        )

        if reset:
            # 以当前 end_tick 作为下一段的起点，沿用最后一帧加速度
            self.reset(start_tick=self._end_tick, last_accel=self._last_accel)

        return out


class ImuStaticInitializer:
    """
    简单的 IMU 静态初始化器（借鉴 Fast-LIO 中 IMU_init 的思路）：

    - 假设一开始有一段“静止”阶段，机器人姿态不变、线加速度近似恒定；
    - 连续喂入 ImuMeasurement，在线统计加速度 / 角速度的均值与方差；
    - 一旦样本数达到阈值 min_samples，即可导出 ImuInitResult，
      用于设置 KF 初始加速度 bias、噪声尺度等。

    注意：本类本身不做“是否静止”的判定，只做纯统计；
          是否在静止阶段启用由上层逻辑（StateEstimatorConfig.use_imu_static_init）控制。
    """

    def __init__(self, min_samples: int = 200, gravity: float = 9.81):
        self.min_samples = int(min_samples)
        self.gravity = float(gravity)
        self.reset()

    def reset(self) -> None:
        self._count: int = 0
        # Welford 在线均值 / 方差累积
        self._mean_acc = np.zeros(3, dtype=float)
        self._mean_gyr = np.zeros(3, dtype=float)
        self._M2_acc = np.zeros(3, dtype=float)
        self._M2_gyr = np.zeros(3, dtype=float)
        # 加速度 bias 估计（body 系），使用 a_body + R_wb^T g_world 的样本均值
        self._mean_bias_acc = np.zeros(3, dtype=float)

    def push(self, imu: ImuMeasurement) -> None:
        """
        推入一帧 IMU 测量，只使用 accel_body / gyro_body。
        """
        acc = np.asarray(imu.accel_body, dtype=float).reshape(3)
        if imu.gyro_body is not None:
            gyr = np.asarray(imu.gyro_body, dtype=float).reshape(3)
        else:
            gyr = np.zeros(3, dtype=float)

        self._count += 1

        # 加速度：Welford 算法
        delta_acc = acc - self._mean_acc
        self._mean_acc += delta_acc / self._count
        delta2_acc = acc - self._mean_acc
        self._M2_acc += delta_acc * delta2_acc

        # 角速度：Welford 算法
        delta_gyr = gyr - self._mean_gyr
        self._mean_gyr += delta_gyr / self._count
        delta2_gyr = gyr - self._mean_gyr
        self._M2_gyr += delta_gyr * delta2_gyr

        # 加速度 bias 样本：b ≈ a_body + R_wb^T g_world（假设线加速度≈0）
        # ImuMeasurement.quat_wxyz 表示 world→body 旋转，quat_to_rotation 返回 R_wb（body→world）
        # 因此 R_bw = R_wb^T.
        from_frame_quat = imu.quat_wxyz
        R_wb = quat_to_rotation(from_frame_quat)
        g_world = np.array([0.0, 0.0, -self.gravity], dtype=float)
        b_sample = acc + R_wb.T @ g_world

        # 对 bias 样本做简单均值累积（与 _count 同步）
        delta_b = b_sample - self._mean_bias_acc
        self._mean_bias_acc += delta_b / self._count

    @property
    def sample_count(self) -> int:
        return self._count

    @property
    def ready(self) -> bool:
        """是否已经积累了足够的样本，可以导出初始化结果。"""
        return self._count >= self.min_samples

    def get_result(self) -> Optional[ImuInitResult]:
        """
        若样本数达到阈值，返回一个 ImuInitResult；否则返回 None。
        """
        if not self.ready:
            return None

        n = max(self._count - 1, 1)
        cov_acc = self._M2_acc / float(n)
        cov_gyr = self._M2_gyr / float(n)

        return ImuInitResult(
            mean_accel_body=self._mean_acc.copy(),
            mean_gyro_body=self._mean_gyr.copy(),
            bias_accel_body=self._mean_bias_acc.copy(),
            cov_accel_body=cov_acc.copy(),
            cov_gyro_body=cov_gyr.copy(),
            sample_count=self._count,
        )


class RobotDynamicsModel:
    """
    Robot Dynamics Model (based on Pinocchio)
    
    ⚠️ **Important Conventions**:
    ==================
    1. **Must use Free-Flyer base**
       - This class assumes model has a 6-DOF floating base
       - nv = 6 (base) + n_joints
       - Floating base is enforced during initialization
    
    2. **Quaternion Format**
       - All quaternion inputs must be in **wxyz** format: [w, x, y, z]
       - Pinocchio internally uses xyzw, this class handles conversion automatically
    
    3. **Velocity Coordinate Frames**
       - base_velocity: base linear velocity (world frame)
       - base_angular_velocity: base angular velocity (base frame)
       - Corresponds to Pinocchio free-flyer standard definition
    
    4. **Base DOF Ordering**
       - First 6 elements of generalized velocity v correspond to base: [v_lin(3), ω(3)]
       - First 7 elements of generalized coordinates q correspond to base: [pos(3), quat_xyzw(4)]
    
    Features:
    - Compute mass matrix M(q) [symmetrized]
    - Compute Coriolis matrix C(q, v)
    - Compute gravity term g(q)
    - Compute contact Jacobian J(q)
    - Forward kinematics
    """
    
    def __init__(self, cfg: RobotDynamicsModelConfig):
        """
        Initialize dynamics model
        
        Args:
            urdf_path: URDF file path
            contact_frames: List of contact point frame names
        
        Raises:
            RuntimeError: If unable to load model with free-flyer base
        """
        # 保存配置，方便其他模块访问（例如重力加速度）
        self.cfg = cfg
        self.gravity = cfg.gravity

        # ⚠️ [Priority 1] Load URDF and enforce free-flyer base
        # G1 URDF has fixed base by default, must add free-flyer
        model_loaded = False
        last_error = None
        
        try:
            # Method 1: Standard API (pinocchio >= 2.6)
            self.model = pin.buildModelFromUrdf(cfg.urdf_path, pin.JointModelFreeFlyer())
            model_loaded = True
        except (AttributeError, TypeError) as e:
            last_error = e
            try:
                # Method 2: Use urdf submodule
                from pinocchio import urdf
                self.model = urdf.buildModel(cfg.urdf_path, pin.JointModelFreeFlyer())
                model_loaded = True
            except Exception as e2:
                last_error = e2
        
        # ⚠️ If all methods fail, raise error directly - DO NOT fallback to fixed base!
        if not model_loaded:
            raise RuntimeError(
                f"❌ Failed to load URDF model with free-flyer base!\n"
                f"This class requires a free-flyer model to work properly.\n"
                f"Last error: {last_error}\n"
                f"Please check Pinocchio version and URDF file."
            )
        
        self.data = self.model.createData()
        
        # ⚠️ Verify that free-flyer base is correctly loaded
        if self.model.njoints < 2:
            raise RuntimeError(
                f"❌ Abnormal number of joints: {self.model.njoints}\n"
                f"Expected at least 2 joints (universe + free-flyer)"
            )
        
        # Verify that the first actual joint is free-flyer (should have 6 velocity DOFs)
        if self.model.njoints >= 2:
            root_joint = self.model.joints[1]  # joint[0] is universe
            if root_joint.nv != 6:
                raise RuntimeError(
                    f"❌ Root joint velocity DOF is not 6!\n"
                    f"   Actual: {root_joint.nv}\n"
                    f"   This indicates the free-flyer base may not be correctly added.\n"
                    f"   Root joint name: {self.model.names[1]}"
                )
        
        # ✅ Extract joint names (from Pinocchio model)
        self.joint_names = self._extract_joint_names_from_model()
        self.joint_name_to_idx = {name: i for i, name in enumerate(self.joint_names)}
        
        # ✅ Cache reordering indices（source_joint_names → Pinocchio 顺序）
        self._reorder_indices: Optional[np.ndarray] = None
        self._cached_source_names: Optional[List[str]] = None
        # 如果配置里提供了 external_joint_names，则在初始化时预先建立索引
        if cfg.external_joint_names:  # 非空列表才认为“真的有外部顺序”
            self._reorder_indices = self._build_reorder_indices(cfg.external_joint_names)
            self._cached_source_names = list(cfg.external_joint_names)

        
        # ✅ Optimization: cache Pinocchio kinematics state to avoid redundant computation
        self._cached_q = None
        self._cached_v = None
        
        # ✅ Optimization: pre-allocate commonly used matrices
        self._eye_6 = np.eye(6)  # For J_base construction
        
        # Get contact point frame IDs
        self.foot_frame_ids = []
        for name in cfg.contact_frames:
            try:
                frame_id = self.model.getFrameId(name)
                self.foot_frame_ids.append(frame_id)
            except Exception as e:
                raise ValueError(f"Frame '{name}' not found in URDF: {e}")
        
        self.contact_frames = cfg.contact_frames
        self._n_contacts = len(cfg.contact_frames)
        
        # ✅ Pre-build base Jacobian block (6 × nv) used in contact Jacobian
        nv = self.model.nv
        self._J_base_template = np.zeros((6, nv))
        self._J_base_template[:, :6] = self._eye_6
    
    def _extract_joint_names_from_model(self) -> List[str]:
        """
        Extract joint names from Pinocchio model
        
        Returns:
            joint_names: List of joint names (excluding free-flyer base)
        """
        joint_names = []
        # In Pinocchio model, joint 0 is usually "universe" (root)
        # joint 1 is free-flyer
        # Actual robot joints start from joint 2
        for i in range(2, self.model.njoints):  # Skip universe and freeflyer
            joint_name = self.model.names[i]
            joint_names.append(joint_name)
        return joint_names

    def _build_reorder_indices(self, source_joint_names: List[str]) -> np.ndarray:
        """
        构造从 source_joint_names → Pinocchio self.joint_names 的重排索引。
        """
        if len(source_joint_names) != len(self.joint_names):
            raise ValueError(
                f"Joint count mismatch: source={len(source_joint_names)}, "
                f"Pinocchio={len(self.joint_names)}"
            )

        indices = np.empty(len(self.joint_names), dtype=np.int32)
        source_name_to_idx = {name: i for i, name in enumerate(source_joint_names)}

        for tgt_idx, tgt_name in enumerate(self.joint_names):
            if tgt_name not in source_name_to_idx:
                raise ValueError(
                    f"Joint '{tgt_name}' not found in source data.\n"
                    f"Source data joints: {source_joint_names}"
                )
            indices[tgt_idx] = source_name_to_idx[tgt_name]

        return indices

    def reorder_joint_measurement(self, meas: JointMeasurement) -> JointMeasurement:
        """
        将整个 JointMeasurement 从数据源的关节顺序重排到 Pinocchio 模型内部顺序。

        返回一个新的 JointMeasurement，names 对齐为 self.joint_names。
        """
        # 简化逻辑：只使用在初始化时或外部显式建立的重排索引。
        # 如果没有缓存索引，则认为输入数据已经与 Pinocchio 顺序一致，直接返回原数据。
        if self._reorder_indices is None:
            return meas

        idx = self._reorder_indices
        pos = meas.position[idx]
        vel = meas.velocity[idx]
        tau = meas.torque[idx]

        return JointMeasurement(
            position=pos,
            velocity=vel,
            torque=tau,
        )
    
    @property
    def n_q(self) -> int:
        """Generalized coordinates dimension"""
        return self.model.nq
    
    @property
    def n_v(self) -> int:
        """Generalized velocity dimension"""
        return self.model.nv
    
    @property
    def n_joints(self) -> int:
        """
        Number of joints (excluding free-flyer base)
        
        ⚠️ Hardcoded assumption:
        - Free-flyer base occupies first 6 DOFs of v
        - All other joints are 1-DOF (revolute/prismatic)
        - If model contains multi-DOF joints (e.g. ball joints), this property will be inaccurate
        """
        return self.model.nv - 6  # Base velocity DOF is fixed at 6
    
    @property
    def n_contacts(self) -> int:
        """Number of contact points"""
        return self._n_contacts
    
    def update_kinematics(self, q: Vec, v: Optional[Vec] = None):
        """
        Update Pinocchio kinematics state (with caching to avoid redundant computation)
        
        Args:
            q: Generalized coordinates (n_q,)
            v: Generalized velocity (n_v,), optional
        """
        # Check if update is needed (whether q has changed)
        need_update_q = (self._cached_q is None or not np.array_equal(q, self._cached_q))
        need_update_v = (v is not None and 
                        (self._cached_v is None or not np.array_equal(v, self._cached_v)))
        
        if need_update_q or need_update_v:
            if v is not None:
                pin.forwardKinematics(self.model, self.data, q, v)
                pin.computeJointJacobians(self.model, self.data, q)
                self._cached_v = v.copy()
            else:
                pin.forwardKinematics(self.model, self.data, q)
            
            pin.updateFramePlacements(self.model, self.data)
            self._cached_q = q.copy()
    
    def compute_dynamics(self, 
                        q: Vec, 
                        v: Vec) -> Tuple[Mat, Mat, Vec]:
        """
        Compute dynamics quantities
        
        Args:
            q: Generalized coordinates (n_q,)
            v: Generalized velocity (n_v,)
        
        Returns:
            M: Mass matrix (n_v, n_v)
            C: Coriolis matrix (n_v, n_v)
            g: Gravity term (n_v,)
        """
        # Verify dimensions
        assert q.shape[0] == self.n_q, f"q dimension error: {q.shape}, expected ({self.n_q},)"
        assert v.shape[0] == self.n_v, f"v dimension error: {v.shape}, expected ({self.n_v},)"
        
        # ✅ Optimization: use cached kinematics update
        self.update_kinematics(q, v)
        
        # ⚠️ [Priority 2] Compute and symmetrize mass matrix
        # pin.crba only guarantees correct upper triangle, lower triangle may have unfilled values
        # Must symmetrize to ensure M is truly symmetric positive definite
        M = pin.crba(self.model, self.data, q)
        M = np.array(M)  # Convert to NumPy array
        
        # Symmetrize: M_sym = (M + M^T) / 2, or fill lower triangle from upper triangle
        # Use upper triangle to fill lower triangle method (more efficient)
        M = np.triu(M) + np.triu(M, 1).T  # Upper triangle + strict upper triangle transpose
        
        # Compute Coriolis matrix
        C = pin.computeCoriolisMatrix(self.model, self.data, q, v)
        C = np.array(C)
        
        # Compute gravity term
        g = pin.computeGeneralizedGravity(self.model, self.data, q)
        g = np.array(g).flatten()
        
        return M, C, g
    
    def compute_contact_jacobian(self, q: Vec) -> Mat:
        """
        Compute stacked contact Jacobian matrix
        
        Corresponds to J(q) in the algorithm, used to compute external torques:
        τ_ext = J(q)^T · f_ext
        
        Args:
            q: Generalized coordinates (n_q,)
        
        Returns:
            J: Stacked Jacobian ((n_c+1)*6, n_v)
               - First block (6, nv): Base wrench to generalized torque mapping
               - Subsequent blocks (6, nv): Spatial Jacobians of contact points (world frame)
        
        ⚠️ Hardcoded assumption:
        - Base has 6 velocity DOFs corresponding to first 6 elements of v
        - Base wrench maps directly to first 6 generalized torques (identity matrix)
        """
        # ✅ Optimization: use cached kinematics update (with Jacobian)
        # Note: computeJointJacobians requires q, so pass None as v
        if self._cached_q is None or not np.array_equal(q, self._cached_q):
            pin.forwardKinematics(self.model, self.data, q)
            pin.computeJointJacobians(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            self._cached_q = q.copy()
        
        nv = self.model.nv
        blocks = []
        
        # ⚠️ Hardcoded assumption: base occupies first 6 DOFs of v
        # Base wrench → generalized torque mapping is 6×6 identity matrix
        # ✅ Optimization: use pre-built base Jacobian block
        blocks.append(self._J_base_template)
        
        # Contact point spatial Jacobians
        for fid in self.foot_frame_ids:
            J6 = pin.computeFrameJacobian(
                self.model, self.data, q, fid, 
                pin.ReferenceFrame.WORLD
            )
            # Pin returns 6×nv spatial Jacobian
            blocks.append(J6)
        
        J = np.vstack(blocks)
        return J
    
    def compute_foot_positions(self, q: Optional[Vec] = None) -> Mat:
        """
        Forward kinematics: compute contact points' world coordinates
        
        Args:
            q: Generalized coordinates (n_q,)
        
        Returns:
            positions: Contact point positions (n_c, 3)
        """
        # ✅ Optimization: use cached kinematics update
        if q is not None:
            self.update_kinematics(q)
        
        positions = []
        for fid in self.foot_frame_ids:
            oMf = self.data.oMf[fid]
            positions.append(oMf.translation)
        
        return np.array(positions)
    
    def compute_relative_positions(self, 
                                   q: Vec,
                                   base_position: Array3) -> Mat:
        """
        Compute relative positions from base to contact points
        
        Args:
            q: Generalized coordinates (n_q,)
            base_position: Base position (3,)
        
        Returns:
            r_bc: Relative position vectors (n_c, 3)
                  r_bc[i] = p_contact[i] - base_position
        """
        foot_positions = self.compute_foot_positions(q)
        r_bc = foot_positions - base_position[None, :]
        return r_bc
    
    def build_generalized_q(self,
                           base_position: Array3,
                           base_quat_wxyz: Vec,
                           q_joints: Vec) -> Vec:
        """
        Build generalized coordinates from base state and joint state
        
        Args:
            base_position: Base position (3,) [x, y, z] **world frame**
            base_quat_wxyz: Base orientation quaternion (4,) **wxyz format: [w, x, y, z]**
                          Represents rotation from world frame to base frame
            q_joints: Joint positions (n_joints,) unit: rad
        
        Returns:
            q: Generalized coordinates (n_q,) Pinocchio format
               = [pos(3), quat_xyzw(4), joints(n_joints)]
        
        ⚠️ Important convention:
        - Quaternion input must be in **wxyz format**
        - Pinocchio internally uses xyzw, this method handles conversion automatically
        - If input is in other format (e.g. xyzw), it will cause incorrect orientation!
        """
        # Pinocchio free-flyer uses SE3: [x, y, z, qx, qy, qz, qw]
        # Input is [w, x, y, z], need to convert to [x, y, z, qw]
        q = np.concatenate([
            base_position,              # (3,)
            base_quat_wxyz[1:],        # [x, y, z] (3,)
            base_quat_wxyz[0:1],       # [w] (1,)
            q_joints                   # (n_joints,)
        ])
        return q

    # def build_generalized_q(self,
    #                         base_position: Array3,
    #                         base_quat_wxyz: Vec,
    #                         q_joints: Vec) -> Vec:
    #     """
    #     base_quat_wxyz: represents R_BW (body->world)
    #     Pinocchio expects world->body, so take conjugate
    #     """
    #     w, x, y, z = base_quat_wxyz
    #     # Conjugate: reverse rotation
    #     base_quat_conj = np.array([w, -x, -y, -z], dtype=float)

    #     q = np.concatenate([
    #         base_position,             # (3,)
    #         base_quat_conj[1:],        # x, y, z
    #         base_quat_conj[0:1],       # w
    #         q_joints
    #     ])
    #     return q
    
    def build_generalized_v(self,
                           base_linear_velocity_body: Array3,
                           base_angular_velocity_body: Array3,
                           v_joints: Vec) -> Vec:
        """
        Build generalized velocity from base velocity and joint velocity
        
        Args:
            base_linear_velocity_body: Base linear velocity (3,) [vx, vy, vz] **BODY frame** unit: m/s
            base_angular_velocity_body: Base angular velocity (3,) [ωx, ωy, ωz] **BODY frame** unit: rad/s
            v_joints: Joint velocities (n_joints,) unit: rad/s
        
        Returns:
            v: Generalized velocity (n_v,) Pinocchio format
               = [v_lin_body(3), ω_body(3), v_joints(n_joints)]
        
        ⚠️ Important conventions (coordinate frames):
        - Pinocchio free-flyer expects **BODY frame** velocities for both linear and angular parts.
        - If you have world frame velocity, you must rotate it to body frame first:
          v_body = R_WB @ v_world  (where R_WB = R_BW.T)
        """
        v = np.concatenate([
            base_linear_velocity_body,   # (3,)
            base_angular_velocity_body,  # (3,)
            v_joints                     # (n_joints,)
        ])
        return v



class GeneralizedMomentumObserver:
    """
    Python 版 GMO，接口贴近 legged_control2:

    典型使用：
      gmo = GeneralizedMomentumObserver(model, gmo_cfg)
      gmo.reset()
      gmo.set_torques(joint_torque)
      gmo.set_state(q, v)
      gmo.update(dt)
      tau_ext = gmo.get_external_torques()
      wrenches, contact_probs = gmo.get_contact_wrenches()
    """

    def __init__(self, model: RobotDynamicsModel, cfg: GmoConfig):
        self.model = model
        self.cfg = cfg
        self.cfg.validate()
        self._n_v = model.n_v

        # 内部状态
        self._y_lpf = np.zeros(self._n_v)
        self._tau_ext = np.zeros(self._n_v)
        self._contact_wrenches = np.zeros((model.n_contacts + 1, 6))
        # 当前步的输入
        self._tau = None
        self._q = None
        self._v = None

    # ----- 配置与复位 -----
    def set_config(self, cfg: GmoConfig):
        self.cfg = cfg
        self.cfg.validate()

    def reset(self):
        self._y_lpf[:] = 0.0
        self._tau_ext[:] = 0.0
        self._contact_wrenches[:] = 0.0

    # ----- 输入设置（类似 Ros2 wrapper 的 setXXX） -----
    def set_state(self, q: Vec, v: Vec):
        self._q = q
        self._v = v

    def set_torques(self, tau: Vec):
        """传入关节力矩（已经按 S 选好了）。"""
        self._tau = tau

    # ----- 主更新 -----
    def update(self, dt: float):
        """
        执行一次 GMO 更新：
          - 计算 p, u, y_lpf, tau_ext
          - 用 J 伪逆求 f_ext
          - 由接触 wrench 计算接触概率
        这里可以直接搬你现在 GeneralizedMomentumObserver.step(...) 的实现。
        """
        assert self._q is not None and self._v is not None and self._tau is not None

        # 1. 动力学量
        M, C, g = self.model.compute_dynamics(self._q, self._v)
        J = self.model.compute_contact_jacobian(self._q)

        # 2. generalized momentum
        p = M @ self._v
        lam = self.cfg.lpf_lambda
        u = self._build_u(M, C, g, p)  # 内部会用 S^T tau + C^T v - g + λp

        # 3. 一阶滤波
        alpha = float(np.clip(np.exp(-lam * dt), 1e-6, 1.0 - 1e-6))
        self._y_lpf = alpha * self._y_lpf + (1.0 - alpha) * u

        # 4. 外力矩估计
        self._tau_ext = lam * p - self._y_lpf

        # 5. 伪逆求 wrench
        f_ext = self._solve_wrench(J, self._tau_ext)  # shape ((1+n_c)*6,)
        self._contact_wrenches = f_ext.reshape(self.model.n_contacts + 1, 6)

    def _build_u(self, M: Mat, C: Mat, g: Vec, p: Vec) -> Vec:
        """
        构造广义动量观测器的一阶滤波输入：

            u = S^T τ + C^T v - g + λ p

        这里我们复用原实现里的假设：
          - 模型使用自由基 (free-flyer)，v 的前 6 个 DoF 为基座，
          - 之后的 DoF 与关节一一对应并全部为驱动关节。
        因此 S^T τ 等价于 [0_6, τ_joints]^T。
        """
        assert self._tau is not None, "Joint torques must be set before update()"
        assert self._v is not None, "Generalized velocity must be set before update()"

        n_v = self._n_v
        n_tau = self._tau.shape[0]
        # 对应 nv = 6 (base) + n_joints
        if n_v != n_tau + 6:
            raise ValueError(
                f"Dimension mismatch when building S^T τ: "
                f"n_v={n_v}, expected 6 + n_tau={n_tau + 6}"
            )

        # S^T τ: 前 6 个自由基坐标未驱动，其余与关节力矩一一对应
        sT_tau = np.zeros(n_v)
        sT_tau[6:] = self._tau

        lam = self.cfg.lpf_lambda
        u = sT_tau + (C.T @ self._v) - g + lam * p
        return u

    def _solve_wrench(self, J: Mat, tau_ext: Vec) -> Vec:
        """
        根据外广义力矩 tau_ext 求解接触/基座 wrench:

            f_ext = (J^T)^† · τ_ext

        采用带阻尼的左伪逆，与原实现保持一致：

            (J^T)^† = (J J^T + λ I)^{-1} J

        适用于 nv ≥ (n_c+1)*6 的情形（典型人形/双足机器人）。
        """
        J_t = J.T
        m, n = J_t.shape  # m = nv, n = (n_c+1)*6

        # 阻尼系数
        damping = self.cfg.pinv_damping

        # AAt = J J^T + λ I_n  (维度 n×n)
        AAt = J @ J_t + damping * np.eye(n)

        # 解 (J J^T + λI) X = J 以得到 (J^T)^† = X
        J_t_pinv = np.linalg.solve(AAt, J)  # 形状 (n, m)

        f_ext = J_t_pinv @ tau_ext  # (n,)
        return f_ext

    # ----- 输出 -----
    def get_external_torques(self) -> Vec:
        return self._tau_ext.copy()

    def get_contact_wrenches(self) -> Vec:
        """返回 (1+n_c, 6) 的 wrench 数组，第 0 个是 base。"""
        return self._contact_wrenches.copy()


class LinearKalmanFilter:
    """
    Python 版 LinearKalmanFilter（带 IMU 加速度 bias）。

      kf.setConfigurations(configs);
      kf.reset();
      kf.setAccelerationLocal(a_imu);
      kf.setContactWrenches(wrenches, weights);
      kf.updateImuProcess(dt);
      kf.updateContactsMeasurement(dt);
      pos = kf.getPosition();
      vel = kf.getVelocityGlobal();

    状态:
        x = [p_b(3), v_b(3), p_c1(3), ..., p_cn(3), b_a(3)]
        p_b: 基座位置（世界系）
        v_b: 基座速度（世界系）
        p_ci: 第 i 个接触点位置（世界系）
        b_a: IMU 加速度计偏置（body 系）
    输入:
        u = R_BW @ a_body + g_world
    观测:
        y = [-r_bc_1, ..., -r_bc_n, z_c1, ..., z_cn]
        r_bc_i: 第 i 个接触点相对于基座的位置向量（世界系）
        z_ci: 第 i 个接触点的高度测量（世界系）
    """

    def __init__(self, model: RobotDynamicsModel, cfg: LinearKalmanFilterConfig):
        # 外部配置
        self.model = model
        self.cfg = cfg
        if self.cfg.zmp_force_threshold is None:
            self.cfg.zmp_force_threshold = cfg.contact_force_threshold
        self.cfg.validate()

        # 接触点数量
        self._n_c = model.n_contacts

        # ================== 内部 Kalman 滤波器状态 ================== #
        # 状态布局：
        # x = [p_b(3), v_b(3), p_c1(3), ..., p_cn(3), b_a(3)]
        self._bias_start = 6 + 3 * self._n_c   # bias 起始索引
        self._n_bias = 3
        self._nx = self._bias_start + self._n_bias  # = 6 + 3*n_c + 3

        # 初始状态
        self._x = np.zeros(self._nx)

        # 初始协方差
        P = np.zeros((self._nx, self._nx))
        P[0:3, 0:3] = (cfg.init_pos_cov**2) * np.eye(3)
        P[3:6, 3:6] = (cfg.init_vel_cov**2) * np.eye(3)
        P[6:self._bias_start, 6:self._bias_start] = (cfg.init_pc_cov**2) * np.eye(3 * self._n_c)
        P[self._bias_start:self._bias_start+3, self._bias_start:self._bias_start+3] = \
            (cfg.init_bias_cov**2) * np.eye(3)
        self._P = P
        # 备份一份用于 reset
        self._P_init = P.copy()

        # 预分配矩阵
        self._I = np.eye(self._nx)
        self._C = self._build_C()
        self._A = np.eye(self._nx)
        self._B = np.zeros((self._nx, 3))
        self._Q = np.zeros((self._nx, self._nx))

        dim_pc = 3 * self._n_c
        dim_y = dim_pc + self._n_c
        self._R_pc = np.zeros((dim_pc, dim_pc))
        self._R_z = np.zeros((self._n_c, self._n_c))
        self._R = np.zeros((dim_y, dim_y))

        # ================== 外部接口相关缓存 ================== #
        # 重力（世界系）
        self._g_world = np.array([0.0, 0.0, -model.gravity])

        # 当前步输入缓存
        self._accel_local = np.zeros(3)
        self._R_BW = np.eye(3)
        self._wrenches = np.zeros((self._n_c, 6))
        # 语义：这里存的是 contact weights w ∈ [1/epsilon, 1]
        self._contact_weights = np.zeros(self._n_c)
        self._feet_heights = np.zeros(self._n_c)
        # 若 True，则足端高度噪声随 contact_weights 自适应调整；若 False，则高度噪声固定
        self._use_dynamic_height_noise = cfg.dynamic_height_noise
        # 最近一次测量残差及其 Mahalanobis 距离，用于残差统计/诊断
        self._last_innovation: Optional[NDArray[np.float64]] = None
        self._last_innovation_mahalanobis2: Optional[float] = None
        # 最近一次构造的测量噪声协方差对角线（R 的 diag），用于可视化 / 诊断
        self._last_R_diag: Optional[NDArray[np.float64]] = None
        # r_bc 在世界系下的测量（flatten 后长度 3*n_c）
        self._r_bc_world_flat = np.zeros(3 * self._n_c)

    # ------------------------------------------------------------------
    # 配置与重置
    # ------------------------------------------------------------------
    def set_configurations(self, cfg: LinearKalmanFilterConfig):
        self.cfg = cfg
        self.cfg.validate()
        # 仅更新配置，不重建滤波器；如需完全重置可显式调用 reset()

    def reset(self,
              initial_base_position: Optional[Array3] = None,
              initial_base_velocity: Optional[Array3] = None,
              initial_imu_accel_bias: Optional[Array3] = None,
              reset_covariance: bool = True):
        """
        重置状态 / 协方差，并允许指定初始基座状态与 IMU 加速度 bias.
        """
        # 基座位置
        if initial_base_position is not None:
            self._x[0:3] = initial_base_position
        else:
            self._x[0:3] = 0.0

        # 基座速度
        if initial_base_velocity is not None:
            self._x[3:6] = initial_base_velocity
        else:
            self._x[3:6] = 0.0

        # 接触点位置（若未显式给出，则置零）
        self._x[6:self._bias_start] = 0.0

        # IMU 加速度 bias 初值
        if initial_imu_accel_bias is not None:
            b = np.asarray(initial_imu_accel_bias, dtype=float).reshape(3)
            self._x[self._bias_start:self._bias_start+3] = b
        else:
            self._x[self._bias_start:self._bias_start+3] = 0.0

        if reset_covariance:
            np.copyto(self._P, self._P_init)

    # ------------------------------------------------------------------
    # 内部 Kalman 滤波器辅助（对应原 BaseStateKalmanFilter）
    # ------------------------------------------------------------------
    @property
    def state_dimension(self) -> int:
        return self._nx

    @property
    def _base_position(self) -> NDArray[np.float64]:
        return self._x[0:3].copy()

    @property
    def _base_velocity(self) -> NDArray[np.float64]:
        return self._x[3:6].copy()

    @property
    def _contact_positions(self) -> NDArray[np.float64]:
        return self._x[6:self._bias_start].reshape(self._n_c, 3).copy()

    @property
    def _state_covariance(self) -> NDArray[np.float64]:
        return self._P.copy()

    @property
    def _imu_accel_bias(self) -> NDArray[np.float64]:
        return self._x[self._bias_start:self._bias_start+3].copy()

    def _build_AB(self, dt: float, R_BW: NDArray[np.float64]) -> Tuple[NDArray, NDArray]:
        """
        离散化状态转移矩阵 A 和输入矩阵 B:

            p_{k+1} = p_k + dt*v_k + 0.5*dt^2*(u - R_BW @ b_a)
            v_{k+1} = v_k + dt*(u - R_BW @ b_a)
            p_c     = p_c
            b_a     = b_a
        """
        A = self._A
        B = self._B

        A.fill(0.0)
        np.fill_diagonal(A, 1.0)

        # 位置依赖速度
        A[0:3, 3:6] = np.eye(3) * dt

        # bias 影响项（通过 -R_BW @ b_a）
        A[0:3, self._bias_start:self._bias_start+3] = -0.5 * (dt**2) * R_BW
        A[3:6, self._bias_start:self._bias_start+3] = -dt * R_BW

        # 输入矩阵
        B.fill(0.0)
        B[0:3, :] = 0.5 * (dt**2) * np.eye(3)
        B[3:6, :] = dt * np.eye(3)

        return A, B

    def _build_C(self) -> NDArray:
        """
        观测矩阵 C，对应：
          - 前 3*n_c 行：y = p_b - p_ci = -r_bc_i
          - 后 n_c 行： y = z_ci
        """
        n_c = self._n_c
        nx = self._nx
        rows = 3 * n_c + n_c

        C = np.zeros((rows, nx))

        for i in range(n_c):
            row = 3 * i
            C[row:row+3, 0:3] = np.eye(3)                      # + p_b
            C[row:row+3, 6 + 3*i:6 + 3*(i+1)] = -np.eye(3)     # - p_ci

        # 足端高度：z 分量
        for i in range(n_c):
            row = 3 * n_c + i
            C[row, 6 + 3*i + 2] = 1.0

        return C

    def _build_Q(self, dt: float) -> NDArray[np.float64]:
        """
        Build process noise covariance Q from continuous-time noise densities.

        State layout:
            x = [p_b(3), v_b(3), p_c1(3), ..., p_cn(3), b_a(3)]

        Noise models (per axis, continuous-time):

            1) IMU linear acceleration white noise w_a(t)
            - density:  imu_accel_noise_density  [m/s^2 / sqrt(Hz)]
            - enters dynamics as:  v̇ = ... + w_a

            2) IMU acceleration bias random walk b_a(t)
            - density:  imu_accel_bias_noise_density  [m/s^2 / sqrt(Hz)]
            - dynamics: ḃ_a = w_b

            3) Contact position random walk
            - density:  contact_process_noise_position  [m / sqrt(Hz)]
            - dynamics: ṗ_ci = w_pc

        Discrete-time approximation (per axis, scalar 1D formulas):

            For acceleration white noise w_a (only on v):
                x = [p, v]^T,  ẋ = [v, a + w_a]^T
                Q_a (per axis) = σ_a^2 * [[dt^3/3, dt^2/2],
                                        [dt^2/2, dt     ]]

            For bias random walk w_b with coupling v̇ = ... - b_a, ḃ = w_b:
                augment state to [p, v, b]^T.
                Continuous-time model:
                    ṗ = v
                    v̇ = ... - b
                    ḃ = w_b
                Discrete-time Q_b (per axis) is:

                    Q_b = σ_b^2 * [[ dt^5/20,  dt^4/8,  -dt^3/6],
                                [ dt^4/8,   dt^3/3,  -dt^2/2],
                                [-dt^3/6,  -dt^2/2,  dt     ]]

            For contact positions random walk:
                ṗ_ci = w_pc,  w_pc density σ_pc  [m / sqrt(Hz)]
                ⇒ Q_pc (per axis) = σ_pc^2 * dt
        """
        cfg = self.cfg
        n_c = self._n_c

        # Use pre-allocated Q if available
        if hasattr(self, "_Q"):
            Q = self._Q
            Q.fill(0.0)
        else:
            Q = np.zeros((self._nx, self._nx), dtype=float)

        I3 = np.eye(3, dtype=float)

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        dt5 = dt3 * dt2

        # -------- 1) IMU accel white noise → affects [p_b, v_b] --------
        # σ_a: accel noise density [m/s^2 / sqrt(Hz)]
        sigma_a2 = float(cfg.imu_accel_noise_density) ** 2

        # Blocks for [p_b(3), v_b(3)]
        Q[0:3, 0:3] += sigma_a2 * (dt3 / 3.0) * I3
        Q[0:3, 3:6] += sigma_a2 * (dt2 / 2.0) * I3
        Q[3:6, 0:3] += sigma_a2 * (dt2 / 2.0) * I3
        Q[3:6, 3:6] += sigma_a2 * (dt * 1.0) * I3

        # -------- 2) IMU accel bias random walk → coupled [p_b, v_b, b_a] --------
        # σ_b: bias noise density [m/s^2 / sqrt(Hz)]
        sigma_b2 = float(cfg.imu_accel_bias_noise_density) ** 2

        b0 = self._bias_start
        b1 = b0 + 3

        # Q_pp, Q_pv, Q_vp from bias noise
        Q[0:3, 0:3] += sigma_b2 * (dt5 / 20.0) * I3
        Q[0:3, 3:6] += sigma_b2 * (dt4 / 8.0) * I3
        Q[3:6, 0:3] += sigma_b2 * (dt4 / 8.0) * I3

        # Coupling with bias state
        Q[0:3, b0:b1] += sigma_b2 * (-dt3 / 6.0) * I3
        Q[b0:b1, 0:3] += sigma_b2 * (-dt3 / 6.0) * I3

        Q[3:6, 3:6] += sigma_b2 * (dt3 / 3.0) * I3
        Q[3:6, b0:b1] += sigma_b2 * (-dt2 / 2.0) * I3
        Q[b0:b1, 3:6] += sigma_b2 * (-dt2 / 2.0) * I3

        # Bias self noise
        Q[b0:b1, b0:b1] += sigma_b2 * (dt * 1.0) * I3

        # -------- 3) Contact positions random walk --------
        # Each contact position p_ci: density σ_pc [m / sqrt(Hz)]
        # Q_pc (per axis) = σ_pc^2 * dt
        if n_c > 0:
            sigma_pc2 = float(cfg.contact_process_noise_position) ** 2
            pc_start = 6
            pc_end = b0  # 6 + 3*n_c
            dim_pc = 3 * n_c

            Q[pc_start:pc_end, pc_start:pc_end] += \
                sigma_pc2 * dt * np.eye(dim_pc, dtype=float)

        return Q

    def _build_R(self, contact_weights: NDArray[np.float64]) -> NDArray:
        """
        测量噪声协方差 R，随接触权重变化。
        contact_weights 越小，对应 r_bc 约束越不可信 -> 噪声越大。
        """
        cfg = self.cfg
        n_c = self._n_c

        sigmas_pc = []
        for w in contact_weights:
            # w 有下界 1/epsilon，上界 1
            w_eff = float(np.clip(w, 1.0 / cfg.epsilon_no_contact, 1.0))
            sigmas_pc.append(cfg.contact_sensor_noise_position / w_eff)

        R_pc = self._R_pc
        R_pc.fill(0.0)
        for i, s in enumerate(sigmas_pc):
            R_pc[3*i:3*(i+1), 3*i:3*(i+1)] = (s**2) * np.eye(3)

        R_z = self._R_z
        R_z.fill(0.0)
        if self._use_dynamic_height_noise:
            # 动态高度噪声：根据 contact_weights 调整
            for i, w in enumerate(contact_weights):
                w_eff = float(np.clip(w, 1.0 / cfg.epsilon_no_contact, 1.0))
                s = cfg.contact_height_sensor_noise / w_eff
                R_z[i, i] = (s**2)
        else:
            # 固定高度噪声：外部 feet_heights 已经给定，噪声不随接触权重变
            np.fill_diagonal(R_z, (cfg.contact_height_sensor_noise**2))

        R = self._R
        R.fill(0.0)
        R[:3*n_c, :3*n_c] = R_pc
        R[3*n_c:, 3*n_c:] = R_z

        # 记录对角线，便于后续可视化：
        #   - 前 3*n_c 维：每个脚的 [-r_bc]_x,y,z 观测噪声方差（单位 m^2）
        #   - 后 n_c 维：每个脚的足端高度 z_c 观测噪声方差（单位 m^2）
        # self._last_R_diag = np.diag(R).astype(float)

        return R

    def _update_contact_probs(self, contact_wrenches: Vec):
        """
        根据接触 wrench 更新 contactProbabilities_（0~1）。
        legged_control2 里是 contactForceThreshold 和 contactForceScale 做一个 saturate。
        """
        n_c = self.model.n_contacts
        probs = np.zeros(n_c)
        thresh = self.cfg.contact_force_threshold
        scale = self.cfg.contact_force_scale

        for i in range(n_c):
            fz = contact_wrenches[i + 1, 2]

            # 相对超出量
            f_rel = (fz - thresh) / max(thresh, 1e-6)

            # scale 作为斜率，越大过阈值后的陡峭越强
            x = scale * f_rel
            p = expit(x)
            probs[i] = float(np.clip(p, 0.0, 1.0))
        self._contact_probs = probs
        return probs

    # ------------------------------------------------------------------
    # ZMP / CoP 计算（可选）
    # ------------------------------------------------------------------
    def compute_zmp_offsets(self, foot_wrenches: Vec) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据每个脚的 6D wrench 计算 ZMP (CoP) 在“脚底平面”上的偏移量 (dx, dy)，
        并结合脚底几何 (圆形半径或矩形长宽) 做一次 clip。

        输入:
          - foot_wrenches: shape (n_contacts, 6)，在世界坐标系下，
                           每行 [Fx, Fy, Fz, τx, τy, τz]，
                           参考点在“脚底接触框架”的原点。

        返回:
          - offsets: shape (n_contacts, 2)，每行 [dx, dy]，单位 m，
                     表示相对“脚底接触框架原点”的 CoP 偏移（在 world x-y 平面）
          - valid:   shape (n_contacts,) 的 bool，表示该脚 ZMP 是否可用
                     （例如 Fz 太小则认为无效）

        当 cfg.use_zmp = False 时，始终返回零偏移和全 False 的有效标志。
        """
        n_c = self._n_c
        offsets = np.zeros((n_c, 2), dtype=float)
        valid = np.zeros(n_c, dtype=bool)

        # 开关关闭时直接返回零
        if not self.cfg.use_zmp or n_c == 0:
            return offsets, valid

        foot_wrenches = np.asarray(foot_wrenches, dtype=float).reshape(n_c, 6)

        cfg = self.cfg

        # 使用与接触判定相同量级的竖直力阈值
        min_fz = float(cfg.zmp_force_threshold)

        for i in range(n_c):
            Fx, Fy, Fz, Tx, Ty, Tz = foot_wrenches[i]

            if abs(Fz) < min_fz:
                # 竖直支撑太小，ZMP 不可信
                continue

            # 标准平面接触 CoP 公式（参考点在接触平面内）：
            #   τx = y_cop * Fz
            #   τy = -x_cop * Fz
            x_cop = -Ty / Fz
            y_cop =  Tx / Fz

            dx, dy = float(x_cop), float(y_cop)

            # ---------- 根据脚底几何 clip ----------
            # 优先使用圆盘半径（与脚方向无关，最安全）
            if cfg.contact_radius > 0.0:
                R = float(cfg.contact_radius)
                r2 = dx * dx + dy * dy
                if r2 > R * R:
                    scale = R / np.sqrt(r2)
                    dx *= scale
                    dy *= scale
            else:
                # 如果没设半径，就用矩形长宽 (world x-y 对齐的近似)
                Lx = float(cfg.contact_zmp_length_x)
                Ly = float(cfg.contact_zmp_length_y)
                if Lx > 0.0:
                    dx = float(np.clip(dx, -Lx, Lx))
                if Ly > 0.0:
                    dy = float(np.clip(dy, -Ly, Ly))

            offsets[i, 0] = dx
            offsets[i, 1] = dy
            valid[i] = True

        return offsets, valid

    def _predict(self, dt: float, u: NDArray[np.float64], R_BW: NDArray[np.float64]):
        """
        KF 预测步
        """
        assert dt > 0, f"dt must be greater than 0, current: {dt}"
        assert u.shape == (3,), f"u dimension error: {u.shape}, expected (3,)"
        assert R_BW.shape == (3, 3), f"R_BW shape mismatch: {R_BW.shape}, expected (3,3)"

        A, B = self._build_AB(dt, R_BW)
        Q = self._build_Q(dt)

        self._x = A @ self._x + B @ u
        self._P = A @ self._P @ A.T + Q

    def _update(self, y: NDArray[np.float64], contact_weights: NDArray[np.float64]):
        """
        KF 测量更新步
        """
        expected_y_dim = 3 * self._n_c + self._n_c
        assert y.shape == (expected_y_dim,), \
            f"y dimension error: {y.shape}, expected ({expected_y_dim},)"
        assert contact_weights.shape == (self._n_c,), \
            f"contact_weights dimension error: {contact_weights.shape}, expected ({self._n_c},)"

        C = self._C
        R = self._build_R(contact_weights)

        CP = C @ self._P
        S = CP @ C.T + R

        # K = P C^T (C P C^T + R)^{-1}
        K = np.linalg.solve(S, CP).T

        # 残差（innovation）：y - C x_pred
        innov = y - (C @ self._x)

        # # 记录最近一次残差及 Mahalanobis 距离，用于后续残差统计 / 诊断
        # self._last_innovation = innov.copy()
        # try:
        #     # d^2 = innov^T S^{-1} innov
        #     self._last_innovation_mahalanobis2 = float(
        #         innov @ np.linalg.solve(S, innov)
        #     )
        # except np.linalg.LinAlgError:
        #     self._last_innovation_mahalanobis2 = None

        self._x = self._x + K @ innov

        IKC = self._I - K @ C
        self._P = IKC @ self._P @ IKC.T + K @ R @ K.T

    # ------------------------------------------------------------------
    # 输入设置
    # ------------------------------------------------------------------
    def setAccelerationLocal(self, acceleration_local: Array3, R_BW: Mat):
        """对应 C++ setAccelerationLocal."""
        self._accel_local = acceleration_local
        self._R_BW = R_BW

    def setContactWrenches(self, wrenches: Vec, contact_weights: Optional[Vec] = None):
        """对应 C++ setContactWrenches."""
        self._wrenches = np.asarray(wrenches).reshape(self._n_c, 6)
        if contact_weights is not None:
            self._contact_weights = np.asarray(contact_weights, dtype=float)

    def setFeetHeights(self, feet_heights: Optional[Vec]):
        """
        对应 feetHeights_（地形高度观测）。

        约定：
          - feet_heights is None:
              * 内部 feet_heights 置为 0.0（假定地面在 z=0）
              * 启用“根据接触权重自适应调整足端高度噪声”的机制
          - feet_heights 不是 None:
              * 使用外部提供的高度观测
              * 关闭高度噪声的动态调整（对所有脚使用统一噪声）
        """
        if feet_heights is None:
            self._feet_heights[:] = 0.0
            self._use_dynamic_height_noise = True
        else:
            self._feet_heights = np.asarray(feet_heights, dtype=float)
            self._use_dynamic_height_noise = False

    def setRelativePositionsWorld(self, r_bc_world: Vec):
        """
        设置由几何 + 关节角 + IMU 姿态得到的 r_bc（世界系）。
        r_bc_world 形状应为 (n_c, 3) 或展平后的 (3*n_c,)。
        """
        r = np.asarray(r_bc_world)
        if r.ndim == 2:
            r = r.reshape(-1)
        assert r.shape[0] == 3 * self._n_c, \
            f"r_bc_world length mismatch: {r.shape[0]} vs 3*n_c={3*self._n_c}"
        self._r_bc_world_flat = r

    def initializeContactPosition(self, contact_index: int, position: Array3, pc_cov: float):
        """
        初始化某个接触点在内部 KF 状态中的位置，并设置其协方差。
        用于新接触从摆动转为支撑时的初始化
        """
        assert 0 <= contact_index < self._n_c, \
            f"contact_index {contact_index} out of range [0, {self._n_c})"
        position = np.asarray(position, dtype=float)
        start = 6 + 3 * contact_index
        end = start + 3
        self._x[start:end] = position
        if pc_cov is not None:
            self._P[start:end, start:end] = (pc_cov**2) * np.eye(3)

    # ------------------------------------------------------------------
    # 更新步骤（对外接口）
    # ------------------------------------------------------------------
    def updateImuProcess(self, dt: float):
        """
        纯预测步：用 IMU 加速度 + bias 模型做 x, P 预测。
        """
        # u = R_BW @ a_body + g_world
        u = self._R_BW @ self._accel_local + self._g_world
        self._predict(dt, u, self._R_BW)

    def updateContactsMeasurement(self, dt: float):
        """
        接触约束测量更新（r_bc + 足端高度），
        用 self._contact_weights 生成 measurement noise R。
        """
        # 观测向量 y = [-r_bc_world_meas, z_c]
        # 这里假设 feet_heights 已经是世界系 z_c（与原实现一致）
        z_c = self._feet_heights
        r_bc_flat = self._r_bc_world_flat
        y = np.concatenate([-r_bc_flat, z_c])

        self._update(y, self._contact_weights)

    def updatePositionMeasurement(self, dt: float, position_world: Array3):
        """
        对应外部里程计/视觉给的 base position，作为额外测量。
        这部分你可以以后再扩展。
        """
        # 仅对基座位置做一次标准 KF 测量更新：y = p_b_meas
        z = np.asarray(position_world, dtype=float).reshape(3)

        # 惯用形式：H = [I3, 0...]
        # 为减小开销，按需惰性分配并复用矩阵
        nx = self._nx
        if not hasattr(self, "_C_pos") or self._C_pos.shape[1] != nx:
            C_pos = np.zeros((3, nx), dtype=float)
            C_pos[:, 0:3] = np.eye(3, dtype=float)
            self._C_pos = C_pos
        else:
            C_pos = self._C_pos

        # 外部位置观测：使用“离散测量标准差”建模（与 mocap/视觉里程计更贴近）
        sigma = float(self.cfg.position_noise_density)
        R_pos = (sigma * sigma) * np.eye(3, dtype=float)

        C = C_pos
        P = self._P

        CP = C @ P
        S = CP @ C.T + R_pos

        # K = P C^T (C P C^T + R)^{-1}
        K = np.linalg.solve(S, CP).T

        innov = z - (C @ self._x)
        self._x = self._x + K @ innov

        IKC = self._I - K @ C
        self._P = IKC @ P @ IKC.T + K @ R_pos @ K.T

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------
    def getPosition(self) -> Array3:
        return self._base_position

    def getVelocityGlobal(self) -> Array3:
        return self._base_velocity

    def getVelocityLocal(self) -> Array3:
        # v_body = R_WB * v_world
        v_world = self.getVelocityGlobal()
        R_WB = self._R_BW.T
        return R_WB @ v_world

    def getContactPositions(self) -> Vec:
        return self._contact_positions

    def getImuAccelerationBias(self) -> Array3:
        return self._imu_accel_bias

    def getCovariance(self) -> Mat:
        return self._state_covariance

    # ------------------------------------------------------------------
    # 残差 / 噪声协方差相关输出（用于统计 / 诊断）
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 残差相关输出（用于统计 / 诊断）
    # ------------------------------------------------------------------
    def get_last_innovation(self) -> Optional[NDArray[np.float64]]:
        """
        返回最近一次测量更新使用的残差向量 y - C x_pred。
        若当前周期未做测量更新，则返回 None。
        """
        if self._last_innovation is None:
            return None
        return self._last_innovation.copy()

    def get_last_innovation_mahalanobis2(self) -> Optional[float]:
        """
        返回最近一次测量残差的 Mahalanobis 距离平方
        d^2 = e^T S^{-1} e。
        若 S 不可逆或当前周期未做测量更新，则返回 None。
        """
        return self._last_innovation_mahalanobis2

    def get_last_R_diag(self) -> Optional[NDArray[np.float64]]:
        """
        返回最近一次构造的测量噪声协方差 R 的对角线。

        维度含义（长度 = 4 * n_contacts）：
          - 对于第 i 个接触点 (i 从 0 开始)：
              * 索引 3*i+0: [-r_bc]_x 观测噪声方差 (m^2)
              * 索引 3*i+1: [-r_bc]_y 观测噪声方差 (m^2)
              * 索引 3*i+2: [-r_bc]_z 观测噪声方差 (m^2)
          - 索引 3*n_c + i: 足端高度 z_c 观测噪声方差 (m^2)
        """
        if self._last_R_diag is None:
            return None
        return self._last_R_diag.copy()


    
def rotate_to_ego(q, vecs):
    return np.array([quat_apply_inverse(q, v) for v in vecs], dtype=np.float32)

def quat_apply_inverse(quat, vec):
    """
    Apply inverse quaternion rotation to vector.
    quat: [w, x, y, z]
    """
    quat = np.asarray(quat)
    if quat.ndim == 1:
        w, x, y, z = quat
        return quat_apply(np.array([w, -x, -y, -z]), vec)
    else:
        raise NotImplementedError("Batch quaternion rotation not implemented yet")


def quat_apply(quat, vec):
    """
    Apply quaternion rotation to vector.
    quat: [w, x, y, z]
    vec: [x, y, z] or [N, 3]
    """
    # Ensure numpy arrays
    quat = np.asarray(quat)
    vec = np.asarray(vec)
    
    # Handle single quaternion, multiple vectors case
    if quat.ndim == 1 and vec.ndim == 2:
        w, x, y, z = quat
        xyz = np.array([x, y, z])
        t = 2 * np.cross(xyz, vec)
        return vec + w * t + np.cross(xyz, t)
    
    # Handle single quaternion, single vector
    elif quat.ndim == 1 and vec.ndim == 1:
        w, x, y, z = quat
        xyz = np.array([x, y, z])
        t = 2 * np.cross(xyz, vec)
        return vec + w * t + np.cross(xyz, t)
        
    # Handle batch case (N, 4) and (N, 3)
    # Not strictly needed for this deployment but good for completeness
    else:
        # Simplified for now
        raise NotImplementedError("Batch quaternion rotation not implemented yet")
    
def obs_body_vel_rel_root(body_vel_w, body_pos_w, root_quat_w, root_pos_w, root_vel_w, root_ang_w):
    # body_vel_w: (N, 3) linear velocities of each link in world frame.
    # root_vel_w: (3,) root linear velocity in world frame.
    # root_ang_w: (3,) root angular velocity in world frame.
    # Compute link velocity relative to the root frame origin in world coordinates:
    # v_rel_w = v_link_w - v_root_w - omega_root_w x (p_link_w - p_root_w).
    # Then rotate into the root (ego) frame via the inverse of root_quat_w.
    rel_pos = body_pos_w - root_pos_w
    rel_vel = body_vel_w - root_vel_w - np.cross(root_ang_w, rel_pos)
    return rotate_to_ego(root_quat_w, rel_vel)

def obs_body_pos_rel_root(body_pos_w, root_quat_w, root_pos_w):
    # body_pos_w: (N, 3) positions of each link in world frame.
    # root_pos_w: (3,) root position in world frame.
    # Rotate positions into the root (ego) frame via the inverse of root_quat_w.
    rel_pos = body_pos_w - root_pos_w
    return rotate_to_ego(root_quat_w, rel_pos)

class StateEstimator:
    """
    纯算法版 StateEstimator，结构上对齐 legged_control2:

      - RobotDynamicsModel       (几何/动力学)
      - GeneralizedMomentumObserver  (GMO)
      - LinearKalmanFilter          (LKF)

    使用方式：
      est = StateEstimator(cfg)
      est.reset()
      out = est.step(est_input)
    """

    def __init__(self, cfg: StateEstimatorConfig):
        self.cfg = cfg
        self._use_external_pose: bool = bool(cfg.use_external_pose)
        if not self._use_external_pose:
            cfg.kf.position_noise_density = 100.  # disable position measurement if not used
        # 1) 模型
        self.model = RobotDynamicsModel(cfg.model)
        # 2) 观测器 / 滤波器
        self.gmo = GeneralizedMomentumObserver(self.model, cfg.gmo)
        self.kf = LinearKalmanFilter(self.model, cfg.kf)

        # 3) selection matrix S（可以重用你之前的逻辑）
        self._build_selection_matrix()

        # 4) 重力
        self.g_world = np.array([0.0, 0.0, -cfg.model.gravity])

        # ================== Contact Soft Start Related ================== #
        # 上一帧接触状态 (True=in contact, False=not in contact)
        self._prev_contact_active = np.zeros(self.model.n_contacts, dtype=bool)
        # 新接触刚落足时的初始权重 (<1 表示“半信任”)
        self._soft_contact_weight = 0.0
        # Soft-start 持续时间（秒），直接使用 cfg.soft_contact_duration
        self._soft_contact_duration = float(cfg.soft_contact_duration)
        # 每个接触点 soft-start 剩余时间（秒）
        self._soft_contact_time_left = np.zeros(self.model.n_contacts, dtype=float)
        # ================================================================ #
        # IMU / Contact 调度相关参数（全部在 tick/ms 上运算）
        self._imu_freq = float(cfg.imu_frenquency)
        self._contact_freq = float(cfg.contact_frequency)
        assert self._imu_freq > 0.0 and self._contact_freq > 0.0
        # 目标更新周期（毫秒）
        self._imu_dt_ms = 1000.0 / self._imu_freq
        self._contact_dt_ms = 1000.0 / self._contact_freq
        # 剩余 tick 计数（毫秒），用于决定何时触发一次 IMU / Contact 更新
        self._imu_tick_increment = int(self._imu_dt_ms)
        self._contact_tick_increment = int(self._contact_dt_ms)

        # Tick-based timing（全局上一帧 tick）
        self._last_tick: Optional[int] = None
        self.imu_integral = ImuPreintegrator()
        # 可选：IMU 静态初始化器（Fast-LIO 风格），仅在 cfg.use_imu_static_init=True 时使用
        self._use_imu_static_init: bool = bool(cfg.use_imu_static_init)
        self._imu_init_done: bool = not self._use_imu_static_init
        self._imu_init_result: Optional[ImuInitResult] = None
        # 外部里程计开关

        # 首次收到 external_position 时，用它锚定 KF 的 world（消除常量平移差）
        self._external_pose_aligned: bool = False
        # 上一次外部位置测量更新的到达 tick（ms），用于按实际到达时间计算 dt
        self._last_external_pose_tick: Optional[int] = None
        # 静态阶段估计得到的陀螺 bias（body 系），供 GMO 使用
        self._gyro_bias_body = np.zeros(3, dtype=float)
        self.imu_initializer: Optional[ImuStaticInitializer] = (
            ImuStaticInitializer(
                min_samples=cfg.imu_init_min_samples,
                gravity=cfg.model.gravity,
            )
            if self._use_imu_static_init
            else None
        )
        # 最近一次完整输出（用于在某些 step 不触发 contact 更新时复用）
        self._last_output: Optional[EstimatorOutput] = None

        # 一些常用常数
        self._zero3 = np.zeros(3, dtype=float)
        self._quat_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        # ========== Scheme-B: IMU quat -> mocap(world) quat alignment ==========
        # Maintain a yaw-only world alignment quaternion q_MWi such that:
        #   q_MB(t) = q_MWi * q_WiB(t)
        # Update q_MWi whenever a new external pose (mocap) quaternion arrives.
        self._q_world_from_imu = self._quat_identity.copy()  # q_MWi (yaw-only)
        self._last_external_quat_tick: Optional[int] = None

        # 外部 world -> 内部 world 的对齐变换（用于 external_position 预处理）
        q_ext, t_ext = cfg.external_world_to_ego
        q_ext = np.asarray(q_ext, dtype=float).reshape(4)
        q_norm = float(np.linalg.norm(q_ext))
        if not np.isfinite(q_norm) or q_norm <= 0.0:
            q_ext = self._quat_identity.copy()
        else:
            q_ext = q_ext / q_norm

        # 仅使用 yaw + xy 平移对齐：R = Rz(yaw), t = [tx, ty, 0]
        w, x, y, z = q_ext
        yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        cy = float(np.cos(yaw))
        sy = float(np.sin(yaw))
        self._R_world_from_external = np.array(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float
        )

        t_ext = np.asarray(t_ext, dtype=float).reshape(3)
        self._t_world_from_external = np.array([t_ext[0], t_ext[1], 0.0], dtype=float)

        # Cache last kinematics state for external queries
        self._last_q: Optional[np.ndarray] = None
        self._last_v: Optional[np.ndarray] = None
        self._frame_id_cache: Dict[Tuple[str, ...], List[int]] = {}
        self._body_frame_cache: Dict[str, int] = {}
        # Scratch buffers to avoid per-call allocations in high-frequency query helpers.
        self._tmp_body_pos_w: Optional[np.ndarray] = None
        self._tmp_body_vel_w: Optional[np.ndarray] = None
        # Cache per body frame -> (frame_id, parent_joint_id, com_lever_in_joint, com_placement_in_joint)
        # Notes:
        # - Pinocchio associates each body's inertia with its parent joint.
        # - `Inertia.lever` is the body COM position expressed in the joint frame.
        self._body_com_cache: Dict[str, Tuple[int, int, np.ndarray, pin.SE3]] = {}
        self.imu_quat_wxyz_raw = self._quat_identity.copy()
        self.imu_quat_wxyz = self._quat_identity.copy()
        self.root_ang_w = self._zero3.copy()

    def _get_body_state_buffers(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return scratch buffers (pos_w, vel_w) with shape (n, 3).

        These buffers are reused across calls, so callers must not store references
        to them without copying.
        """
        n_int = int(n)
        if n_int <= 0:
            # Return empty views with consistent dtype.
            empty = np.empty((0, 3), dtype=np.float64)
            return empty, empty

        if (
            self._tmp_body_pos_w is None
            or self._tmp_body_vel_w is None
            or self._tmp_body_pos_w.shape[0] < n_int
            or self._tmp_body_vel_w.shape[0] < n_int
        ):
            self._tmp_body_pos_w = np.empty((n_int, 3), dtype=np.float64)
            self._tmp_body_vel_w = np.empty((n_int, 3), dtype=np.float64)

        assert self._tmp_body_pos_w is not None
        assert self._tmp_body_vel_w is not None

        return self._tmp_body_pos_w[:n_int], self._tmp_body_vel_w[:n_int]

    def _get_body_com_pos_world(self, body_name: str) -> np.ndarray:
        """Compute body COM position in world (3,)."""
        _, jid, lever, _placement = self._get_body_com_info(body_name)
        oMi = self.model.data.oMi[int(jid)]
        # Assignments into scratch buffers will copy, so no .copy() needed here.
        return np.asarray(oMi.translation + oMi.rotation @ lever, dtype=np.float64).reshape(3)

    def _get_body_com_lin_vel_world(
        self,
        body_name: str,
        *,
        reference_frame: pin.ReferenceFrame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    ) -> np.ndarray:
        """Compute body COM linear velocity expressed in world axes (3,)."""
        _fid, jid, _lever, placement = self._get_body_com_info(body_name)
        vel_com = pin.getFrameVelocity(
            self.model.model, self.model.data, int(jid), placement, reference_frame
        )
        return np.asarray(vel_com.linear, dtype=np.float64).reshape(3)

    def _resolve_frame_id(self, frame_name: str) -> int:
        model = self.model.model
        if hasattr(model, "existFrame") and not model.existFrame(frame_name):
            raise ValueError(f"Frame '{frame_name}' not found in Pinocchio model.")
        frame_id = int(model.getFrameId(frame_name))
        if frame_id < 0 or frame_id >= model.nframes:
            raise ValueError(f"Frame '{frame_name}' not found in Pinocchio model.")
        return frame_id

    def _get_frame_ids(self, body_names: List[str]) -> List[int]:
        key = tuple(body_names)
        cached = self._frame_id_cache.get(key)
        if cached is not None:
            return cached
        frame_ids = []
        for name in body_names:
            frame_ids.append(self._resolve_frame_id(name))
        self._frame_id_cache[key] = frame_ids
        return frame_ids

    def _get_body_frame_id(self, body_name: str) -> int:
        cached = self._body_frame_cache.get(body_name)
        if cached is not None:
            return cached
        frame_id = self._resolve_frame_id(body_name)
        self._body_frame_cache[body_name] = frame_id
        return frame_id

    def _get_body_world_state(self, body_name: str) -> Tuple[Array3, Mat, Array3, Array3]:
        # if self._last_q is None or self._last_v is None:
        #     return None
        # self.model.update_kinematics(self._last_q, self._last_v)
        body_id = self._get_body_frame_id(body_name)
        oMf = self.model.data.oMf[body_id]
        body_pos = oMf.translation.copy()
        body_rot = oMf.rotation.copy()
        body_vel = pin.getFrameVelocity(
                    self.model.model, self.model.data, body_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
        body_lin = body_vel.linear.copy()
        body_ang = body_vel.angular.copy()
        return body_pos, body_rot, body_lin, body_ang

    def _get_body_com_info(self, body_name: str) -> Tuple[int, int, np.ndarray, pin.SE3]:
        cached = self._body_com_cache.get(body_name)
        if cached is not None:
            return cached

        fid = self._get_body_frame_id(body_name)
        frame = self.model.model.frames[int(fid)]
        jid = int(frame.parentJoint)
        if jid <= 0:
            raise ValueError(f"Frame '{body_name}' has invalid parentJoint={jid}.")

        lever = np.asarray(self.model.model.inertias[jid].lever, dtype=np.float64).reshape(3)
        placement = pin.SE3(np.eye(3), lever)
        out = (int(fid), jid, lever, placement)
        self._body_com_cache[body_name] = out
        return out

    def _get_body_com_world_state(
        self, body_name: str, *, reference_frame: pin.ReferenceFrame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    ) -> Tuple[Array3, Mat, Array3, Array3]:
        """
        Return (p_com_w, R_wb, v_com_w, ω_w) for the body, all expressed in world axes.

        - COM position comes from Pinocchio inertia lever transformed by oMi[parentJoint].
        - COM velocity is obtained with Pinocchio's getFrameVelocity(joint_id, placement, reference_frame),
          where placement's translation is the COM lever in the joint frame (rotation = identity).
        """
        fid, jid, lever, placement = self._get_body_com_info(body_name)

        # Orientation: body frame orientation in world.
        oMf = self.model.data.oMf[int(fid)]
        R_wb = oMf.rotation.copy()

        oMi = self.model.data.oMi[int(jid)]
        p_com_w = (oMi.translation + oMi.rotation @ lever).copy()

        vel_com = pin.getFrameVelocity(
            self.model.model, self.model.data, int(jid), placement, reference_frame
        )
        v_com_w = np.asarray(vel_com.linear, dtype=np.float64).reshape(3).copy()
        w_w = np.asarray(vel_com.angular, dtype=np.float64).reshape(3).copy()

        return p_com_w, R_wb, v_com_w, w_w
    
    def get_body_world_state(
        self, body_name: str = "pelvis"
    ) -> Tuple[Array3, Mat, Array3, Array3]:
        return self._get_body_world_state(body_name)

    def get_body_pos_vel_rel_root(
        self,
        body_names: List[str],
        root_name: str = "pelvis",
        use_com: Optional[bool] = None,
        pos_mode: str = "origin",
        vel_mode: str = "com",
        debug: bool = False,
    ) -> np.ndarray:
        """
        计算每个 body(frame) 相对 root_name(frame, 默认 pelvis) 的“相对速度”，并用 root_name 坐标系表示。

        这里的“相对速度”对应：
            d/dt ( R_wr^T * (p_b - p_r) )
        也就是“相对位置向量在 rel 坐标系下的时间导数”，因此包含 - ω_r × (p_b - p_r) 项。

        训练/IsaacLab 口径对齐说明：
        - IsaacLab 的 body 线速度默认是 “rigid-body COM linear velocity”，位置也是 COM。
        - 但你当前训练数据约定可能是“pos 用 link frame origin，而 vel 用 body COM vel”。
          因此本函数允许分别控制 pos/vel 的取值点：
            - pos_mode: "origin" 或 "com"
            - vel_mode: "origin" 或 "com"
          默认使用 pos_mode="origin", vel_mode="com"（即：COM vel + 非 COM pos）。

        注意：
        - 必须用 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED 取得线速度的 world-axes 表达；
          pin.ReferenceFrame.WORLD 的 linear 分量不是“点速度”，会导致公式不自洽。
        """
        if use_com is not None:
            # Backward-compatible switch: use_com=True means both pos/vel use COM; use_com=False means both use origin.
            pos_mode = "com" if bool(use_com) else "origin"
            vel_mode = "com" if bool(use_com) else "origin"

        pos_mode = str(pos_mode).lower().strip()
        vel_mode = str(vel_mode).lower().strip()
        if pos_mode not in ("origin", "com"):
            raise ValueError(f"pos_mode must be 'origin' or 'com', got {pos_mode!r}")
        if vel_mode not in ("origin", "com"):
            raise ValueError(f"vel_mode must be 'origin' or 'com', got {vel_mode!r}")

        n_bodies = len(body_names)
        body_pos_w, body_vel_w = self._get_body_state_buffers(n_bodies)
        for i, name in enumerate(body_names):
            if pos_mode == "com" and vel_mode == "com":
                # Single Pinocchio query per body when both are COM.
                p_com_w, _R_wb, v_com_w, _w_w = self._get_body_com_world_state(name)
                body_pos_w[i, :] = p_com_w
                body_vel_w[i, :] = v_com_w
                continue

            if pos_mode == "com":
                body_pos_w[i, :] = self._get_body_com_pos_world(name)
            else:
                fid = self._get_body_frame_id(name)
                body_pos_w[i, :] = self.model.data.oMf[fid].translation

            if vel_mode == "com":
                body_vel_w[i, :] = self._get_body_com_lin_vel_world(name)
            else:
                fid = self._get_body_frame_id(name)
                body_vel = pin.getFrameVelocity(
                    self.model.model,
                    self.model.data,
                    fid,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
                body_vel_w[i, :] = np.asarray(body_vel.linear, dtype=np.float64).reshape(3)
        ego_body_vel_rel_root = obs_body_vel_rel_root(
            body_vel_w,
            body_pos_w,
            self.imu_quat_wxyz,
            self._last_output.base_position,
            self._last_output.base_velocity_world,
            self.root_ang_w,
        )
        if pos_mode == "com":
            root_pos_w, root_rot_w, _, _ = self._get_body_com_world_state(root_name)
        else:
            root_pos_w, root_rot_w, _, _ = self._get_body_world_state(root_name)
        # ego_body_pos_rel_root = obs_body_pos_rel_root(
        #     body_pos_w,
        #     self.imu_quat_wxyz,
        #     self._last_output.base_position,
        # )
        # Vectorized form of: rel = root_rot_w.T @ (p_w - root_pos_w)
        # For row-wise positions (N,3), this becomes: (p_w - root_pos_w) @ root_rot_w
        ego_body_pos_rel_root = (body_pos_w - root_pos_w.reshape(1, 3)) @ root_rot_w
        if debug:
            self.debug_dict = dict(
                body_pos_w=body_pos_w.copy(),
                body_vel_w=body_vel_w.copy(),
                imu_quat_wxyz=self.imu_quat_wxyz,
                last_base_position=self._last_output.base_position,
                last_base_velocity_world=self._last_output.base_velocity_world,
                root_ang_w=self.root_ang_w,
                root_pos_w=root_pos_w,
                root_rot_w=root_rot_w,
                ego_body_pos_rel_root=ego_body_pos_rel_root,
                ego_body_vel_rel_root=ego_body_vel_rel_root,
            )
        return ego_body_vel_rel_root.reshape(-1), ego_body_pos_rel_root.reshape(-1)

    def _build_selection_matrix(self):
        """
        和你现在一样：假定自由基坐标前 6 个是未驱动的，后面是关节。
        这里先给一个简单占位，具体可以重用你之前的实现。
        """
        nv = self.model.n_v
        n_tau = nv - 6
        S = np.zeros((n_tau, nv))
        S[np.arange(n_tau), 6 + np.arange(n_tau)] = 1.0
        self.S = S

    def _compute_tick_delta(self, tick: Optional[int]) -> int:
        """
        仅根据 tick 计算与上一帧的 tick 差（单位：tick，本实现中约定 1 tick = 1 ms）。

        约定：
          - 必须提供 tick（去掉了所有 dt fallback 分支）。
          - 如果是第一帧，或 tick 非递增 / 跳变过大，可以视作“重新开始”：返回 0，
            并重置内部调度/预积分状态。
        """
        if tick is None:
            raise ValueError("EstimatorInput.tick is required for tick-based scheduling.")

        tick_int = int(tick)
        last_tick = self._last_tick
        self._last_tick = tick_int

        # 第一帧：尚无历史 tick，只记录不做积分
        if last_tick is None:
            return 0

        dt_ticks = tick_int - int(last_tick)

        # 非递增或异常情况：视作重启
        if dt_ticks <= 0:
            self._imu_tick_increment = int(self._imu_dt_ms)
            self._contact_tick_increment = int(self._contact_dt_ms)
            self.imu_integral.reset(start_tick=tick_int)
            return 0

        # 可选：若需要，可以在此处对 dt_ticks 做上界检查，
        # 确保 dt_ticks 不超过 IMU / Contact 目标周期。
        # 目前仅在文档中约定使用者：EstimatorInput 的 tick 间隔
        # 不应超过 min(imu_dt_ms, contact_dt_ms)。

        return dt_ticks

    def check_new_contacts(self, contact_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        检查哪些接触点在本帧成为新接触
        """
        current_active = contact_weights > 0.5
        new_contacts = np.logical_and(current_active,
                                      np.logical_not(self._prev_contact_active))
        return new_contacts, current_active

    def _apply_soft_start(self, contact_weights: np.ndarray, current_active: np.ndarray, new_contacts: np.ndarray, dt: float):
        """
        多帧 soft-start：对新接触脚的权重从 soft_weight 平滑过渡到 1.0。

        输入:
            contact_weights: 当前帧的“硬”接触权重（通常为 {1, 1/epsilon} 或外部给定）。
            current_active: 当前帧接触标志（布尔）
            new_contacts: 本帧新产生接触的脚（0->1）的布尔掩码
        返回:
            contact_weights_soft: 经过 soft-start 平滑后的权重
        """

        # 为刚从摆动转为接触的脚启动 soft-start 计数器（按时间而非固定步数）
        self._soft_contact_time_left[new_contacts] = self._soft_contact_duration

        # 在 soft-start 窗口内，对权重进行线性插值：soft_weight → 1.0，
        # 使用累计实际时间 dt 来推进进度，而不依赖 cfg.update_frequency。
        if np.any(self._soft_contact_time_left > 0.0) and self._soft_contact_duration > 0.0:
            contact_weights = contact_weights.copy()
            total_time = self._soft_contact_duration

            for i in range(self.model.n_contacts):
                remaining = self._soft_contact_time_left[i]
                if remaining > 0.0:
                    if current_active[i]:
                        # 进度：已用时间 / 总 soft-start 时间
                        elapsed = total_time - remaining
                        progress = np.clip(elapsed / total_time, 0.0, 1.0)
                        w_soft = (self._soft_contact_weight
                                  + progress * (1.0 - self._soft_contact_weight))
                        # 保守做法：新权重不比原始权重更“硬”
                        contact_weights[i] = min(contact_weights[i], w_soft)
                    # 剩余 soft-start 时间减少 dt
                    self._soft_contact_time_left[i] = max(0.0, remaining - float(dt))

        return contact_weights
    
    def _initialize_new_contact_positions(
        self,
        base_pos: Array3,
        R_BW: Mat,
        r_bc_body: Vec,
        new_contacts: np.ndarray,
        foot_wrenches: np.ndarray,
        zmp_offsets: np.ndarray,
        zmp_valid: np.ndarray,
    ) -> np.ndarray:
        """
        对新接触脚的接触点位置进行初始化，并写入内部 KF 状态，
        同时利用 ZMP + 脚底几何对“接触点在脚底区域内”的位置进行修正。

        Args:
            base_pos:      预测后的基座位置估计（world）
            R_BW:          机体到世界的旋转矩阵
            r_bc_body:     每个接触点在机体坐标系下的相对位置 (n_c, 3)
            new_contacts:  bool 向量，标记本帧哪些脚从 0->1 成为新接触
            foot_wrenches: shape (n_c,6)，每个接触点的 6D wrench（不含 base 那一项）
        """
        p_b = base_pos
        pc_est = self.kf.getContactPositions()  # (n_c, 3)
        prev_support_indices = np.where(self._prev_contact_active)[0]
        used_support_mask = np.zeros(self.model.n_contacts, dtype=bool)

        for i in range(self.model.n_contacts):
            if not new_contacts[i]:
                continue

            # 1) 由当前基座估计 + 几何 得到“脚底参考点”位置（一般是脚心 Frame）
            foot_pos_world = p_b + R_BW @ r_bc_body[i]

            # 3) 若上一帧存在支撑脚，则利用脚间几何关系改进估计（保持你原来的逻辑）
            if len(prev_support_indices) > 0:
                j = prev_support_indices[0]  # 任选一个支撑脚 j
                d_ij_body = r_bc_body[i] - r_bc_body[j]
                d_ij_world = R_BW @ d_ij_body
                p_ci_from_foot = pc_est[j] + d_ij_world

                alpha = 0.5
                p_ci_init = alpha * foot_pos_world + (1.0 - alpha) * p_ci_from_foot
                used_support_mask[i] = True
            else:
                # 没有支撑脚时，仅使用 base + FK (+ ZMP 修正)
                p_ci_init = foot_pos_world

            # 2) 如果该脚的 ZMP 可用，则在参考点附近加上 (dx, dy) 修正
            if zmp_valid[i]:
                dx, dy = zmp_offsets[i]
                p_ci_init = p_ci_init + np.array([dx, dy, 0.0], dtype=float)
            else:
                # ZMP 不可用，退化为几何中心
                p_ci_init = p_ci_init

            # 4) 写回 KF 内部状态并重置该脚位置协方差
            pc_cov = self.cfg.kf.init_pc_cov
            self.kf.initializeContactPosition(i, p_ci_init, pc_cov)

        return used_support_mask

    def _preprocess_external_position(self, position_external_world: Array3) -> Array3:
        """Convert external world position to estimator internal world."""
        p_ext = np.asarray(position_external_world, dtype=float).reshape(3)
        return (self._R_world_from_external @ p_ext + self._t_world_from_external).reshape(3)

    # ---------------- Quaternion helpers (wxyz) ----------------
    def _quat_normalize(self, q: Vec) -> Vec:
        q = np.asarray(q, dtype=float).reshape(4)
        n = float(np.linalg.norm(q))
        if not np.isfinite(n) or n <= 0.0:
            return self._quat_identity.copy()
        return (q / n).astype(float, copy=False)

    def _quat_conj(self, q: Vec) -> Vec:
        q = np.asarray(q, dtype=float).reshape(4)
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

    def _quat_mul(self, q1: Vec, q2: Vec) -> Vec:
        w1, x1, y1, z1 = np.asarray(q1, dtype=float).reshape(4)
        w2, x2, y2, z2 = np.asarray(q2, dtype=float).reshape(4)
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )

    def _yaw_from_quat(self, q: Vec) -> float:
        w, x, y, z = np.asarray(q, dtype=float).reshape(4)
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _quat_from_yaw(self, yaw: float) -> Vec:
        half = 0.5 * float(yaw)
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)

    # ---------------- Scheme-B hooks ----------------
    def _get_imu_quat_at_external_time(self, external_pose: ExternalPoseMeasurement, inp: EstimatorInput) -> Vec:
        """Hook for time alignment (stub).

        Future: interpolate IMU quaternion to external_pose.tick.
        Current behavior: use the current input IMU quaternion.
        """
        return np.asarray(inp.imu.quat_wxyz, dtype=float).reshape(4)

    def _maybe_update_world_alignment(self, inp: EstimatorInput) -> None:
        """Update yaw-only q_MWi using the latest external pose quaternion."""
        if (not self._use_external_pose) or (inp.external_position is None):
            return

        ext = inp.external_position
        tick = int(ext.tick) if ext.tick is not None else (int(inp.tick) if inp.tick is not None else None)
        if tick is not None and self._last_external_quat_tick is not None and tick <= self._last_external_quat_tick:
            return

        q_mb = self._quat_normalize(ext.quat_wxyz)  # body->mocap_world
        q_wib = self._quat_normalize(self._get_imu_quat_at_external_time(ext, inp))  # body->imu_world
        q_mwi = self._quat_mul(q_mb, self._quat_conj(q_wib))
        q_mwi = self._quat_normalize(q_mwi)

        # Yaw-only alignment
        yaw = self._yaw_from_quat(q_mwi)
        self._q_world_from_imu = self._quat_from_yaw(yaw)

        if tick is not None:
            self._last_external_quat_tick = tick

    def _imu_quat_to_world(self, q_imu_wxyz: Vec) -> Vec:
        """Convert IMU quaternion (body->imu_world) to estimator world (mocap world)."""
        q_imu = self._quat_normalize(q_imu_wxyz)
        q_out = self._quat_mul(self._q_world_from_imu, q_imu)
        return self._quat_normalize(q_out)
    
    # ----- 对外 API -----
    def reset(self,
              initial_base_position: Array3 | None = None,
              initial_base_velocity: Array3 | None = None,
              reset_covariance: bool = True):
        """
        重置估计器状态。

        为了与 legacy 行为对齐，这里允许传入初始基座位置 / 速度，
        并将其下发到内部 Kalman 滤波器。
        """
        self.gmo.reset()
        # 透传到内部 KF（此时尚未注入 IMU bias，静态初始化阶段会视情况重新 reset 一次）
        self.kf.reset(
            initial_base_position=initial_base_position,
            initial_base_velocity=initial_base_velocity,
            initial_imu_accel_bias=None,
            reset_covariance=reset_covariance,
        )

        # 若 reset 已显式指定初始 base 位置，则视为已对齐；否则等待首帧 external_position 到来时对齐。
        self._external_pose_aligned = initial_base_position is not None
        self._last_external_pose_tick = None

        # 重置 tick 调度和预积分
        self._last_tick = None
        self._imu_tick_increment = int(self._imu_dt_ms)
        self._contact_tick_increment = int(self._contact_dt_ms)
        self.imu_integral.reset()

        # 重置 IMU 静态初始化器
        if self._use_imu_static_init and self.imu_initializer is not None:
            self.imu_initializer.reset()
            self._imu_init_done = False
            self._imu_init_result = None
            self._gyro_bias_body[:] = 0.0
        else:
            # 未启用静态初始化时，视为已完成，gyro bias 保持零
            self._imu_init_done = True
            self._imu_init_result = None
            self._gyro_bias_body[:] = 0.0

        # 构造一个“当前 KF 状态”的输出作为初始 _last_output
        base_pos = self.kf.getPosition()
        v_world = self.kf.getVelocityGlobal()
        v_body = self.kf.getVelocityLocal()
        n_c = self.model.n_contacts
        contact_positions = np.zeros((n_c, 3), dtype=float)
        contact_forces = np.zeros((n_c, 6), dtype=float)
        contact_probabilities = np.zeros(n_c, dtype=float)
        tau_external = np.zeros(self.model.n_v - 6, dtype=float)
        imu_bias = self.kf.getImuAccelerationBias()

        self._last_output = EstimatorOutput(
            base_position=base_pos,
            base_velocity_world=v_world,
            base_velocity_body=v_body,
            contact_positions=contact_positions,
            contact_forces=contact_forces,
            contact_probabilities=contact_probabilities,
            tau_external=tau_external,
            imu_accel_bias=imu_bias,
        )

    # ----- 分步接口 -----
    def step_imu(self, dt_eff: float, imu: ImuMeasurement) -> None:
        """
        仅运行 IMU 预测步（高频），不做 GMO / 接触更新。
        """
        R_BW = quat_to_rotation(imu.quat_wxyz)
        self.kf.setAccelerationLocal(imu.accel_body, R_BW)
        self.kf.updateImuProcess(dt_eff)

    def step_contacts(self,
                      dt_eff: float,
                      joint: JointMeasurement,
                      imu: ImuMeasurement,
                      contact: ContactMeasurement | None = None,
                      ) -> EstimatorOutput:
        """
        使用当前 KF 预测状态 + Joint/IMU/GMO 接触信息，
        运行 GMO 与 KF 的接触测量更新，返回一帧完整估计。

        注意：
          - 假定在调用本函数前，KF 已经根据 IMU 预测到了当前时刻
            （例如通过多次 step_imu 调用）。
        """
        # 1. joint 重排序（用 LeggedModel 内部的 Pinocchio 顺序）
        joint_reordered = self.model.reorder_joint_measurement(joint)
        q_j = joint_reordered.position
        v_j = joint_reordered.velocity
        tau_j = joint_reordered.torque



        # 使用“预测后的” base 状态构造 q, v
        base_pos = self.kf.getPosition()
        base_vel_world = self.kf.getVelocityGlobal()

        # Pinocchio expects Body-frame velocity for free-flyer
        base_vel_body = self._R_BW.T @ base_vel_world

        q = self.model.build_generalized_q(base_pos, self.imu_quat_wxyz, q_j)
        v = self.model.build_generalized_v(base_vel_body, self.root_ang_b, v_j)
        self._last_q = q.copy()
        self._last_v = v.copy()

        # ========== GMO 更新 ==========
        self.gmo.set_state(q, v)
        self.gmo.set_torques(tau_j)
        self.gmo.update(dt_eff)
        tau_ext = self.gmo.get_external_torques()
        contact_wrenches = self.gmo.get_contact_wrenches()

        # ====== Contact weights: 对齐 C++ contactProbabilities 语义 ====== #
        n_c = self.model.n_contacts

        # 1) 先得到“原始接触概率” p ∈ [0, 1]
        if contact is not None and contact.contact_probabilities is not None:
            probs = np.asarray(contact.contact_probabilities, dtype=float)
            assert probs.shape == (n_c,)
            probs = np.clip(probs, 0.0, 1.0)
        else:
            probs = self.kf._update_contact_probs(contact_wrenches)
            assert probs.shape == (n_c,)

        # 3) soft-start 在“权重”上做（多帧从 soft_weight → 1.0），
        #    使用实际 dt 控制软启动时长。
        new_contacts, current_active = self.check_new_contacts(probs)
        if self.cfg.use_soft_start:
            contact_weights = self._apply_soft_start(probs, current_active, new_contacts, dt_eff)
        else:
            contact_weights = probs

        # # ========== 3.1 由几何 + 关节角构造 r_bc 测量==========
        foot_positions_world = self.model.compute_foot_positions()
        r_bc_body = (self._R_BW.T @ (foot_positions_world - base_pos).T).T  # (n_c, 3)

        # ====== 新接触的接触点位置初始化（加入 ZMP + 脚底几何修正） ====== #
        # contact_wrenches[0] 是 base 的 wrench，后面每一行对应一个接触点
        foot_wrenches = contact_wrenches[1:]   # shape (n_c, 6)
        zmp_offsets, zmp_valid = self.kf.compute_zmp_offsets(foot_wrenches)
        
        # 记录哪些接触点在本时刻被初始化（仅当启用初始化且有新接触时）
        contact_initialized = np.zeros(self.model.n_contacts, dtype=bool)
        if self.cfg.contact_pos_init and np.any(new_contacts):
            contact_initialized = self._initialize_new_contact_positions(
                base_pos,
                self._R_BW,
                r_bc_body,
                new_contacts,
                foot_wrenches,
                zmp_offsets,
                zmp_valid,
            )

        r_bc_world = (self._R_BW @ r_bc_body.T).T  # (n_c, 3)

        # ========== KF 接触测量更新 ==========
        self.kf.setContactWrenches(foot_wrenches, contact_weights)
        # feet_heights: None -> 假定地面在 z=0 并启用动态噪声调整；
        # 非 None -> 使用外部高度观测并关闭动态噪声调整。
        if contact is not None:
            self.kf.setFeetHeights(contact.feet_heights)
        else:
            self.kf.setFeetHeights(None)
        self.kf.setRelativePositionsWorld(r_bc_world)
        self.kf.updateContactsMeasurement(dt_eff)

        # ========== 组装输出 ==========
        # 残差 / 噪声统计：从 KF 中取出最近一次测量残差、R 对角线及其 Mahalanobis 距离
        innovation = self.kf.get_last_innovation()
        innovation_m2 = self.kf.get_last_innovation_mahalanobis2()
        R_diag = self.kf.get_last_R_diag()

        base_pos = self.kf.getPosition()
        v_world = self.kf.getVelocityGlobal()
        v_body = self.kf.getVelocityLocal()
        contact_positions = self.kf.getContactPositions()
        imu_bias = self.kf.getImuAccelerationBias()

        out = EstimatorOutput(
            base_position=base_pos,
            base_velocity_world=v_world,
            base_velocity_body=v_body,
            contact_positions=contact_positions,
            contact_forces=foot_wrenches,
            contact_probabilities=contact_weights,
            tau_external=tau_ext,
            imu_accel_bias=imu_bias,
            zmp_offsets=zmp_offsets,
            zmp_valid=zmp_valid,
            state_covariance=self.kf.getCovariance(),
            base_wrench=contact_wrenches[0],
            innovation=innovation,
            innovation_mahalanobis2=innovation_m2,
            R_diag=R_diag,
            contact_initialized=contact_initialized,
            contact_active=current_active,
        )

        # 记录当前接触状态，用于下一帧检测 0->1 过渡
        self._prev_contact_active = current_active

        return out
    

    def process_input(self, inp: EstimatorInput) -> Optional[EstimatorOutput]:
        # 2. IMU 姿态 / 转动矩阵 / 角速度
        imu = inp.imu   
        self.imu_quat_wxyz_raw = imu.quat_wxyz.copy()
        self.imu_quat_wxyz = self._imu_quat_to_world(imu.quat_wxyz)
        self._R_BW = quat_to_rotation(self.imu_quat_wxyz)
        if imu.gyro_body is not None:
            omega_raw = np.asarray(imu.gyro_body, dtype=float).reshape(3)
        else:
            omega_raw = self._zero3

        # 若进行了静态 IMU 初始化，则在 GMO 中使用去 bias 的角速度
        if self._use_imu_static_init and self._imu_init_done:
            omega = omega_raw - self._gyro_bias_body
        else:
            omega = omega_raw
        self.root_ang_b_raw = omega_raw
        self.root_ang_b = omega
        self.root_ang_w = self._R_BW @ self.root_ang_b  # ω_r

    def step(self, inp: EstimatorInput) -> Optional[EstimatorOutput]:
        """
        一步估计（保持与旧接口兼容）：
          1) 用 IMU 做 KF 预测（step_imu）
          2) 用 Joint/IMU/GMO 做接触更新（step_contacts）

        对于多频率使用场景，可直接调用 step_imu / step_contacts；
        但在本实现中，推荐统一通过 tick 驱动的 step() 接口使用。
        """
        # 1) 若启用 IMU 静态初始化，则在真正进入滤波之前，先积累一段 IMU 统计量
        self._last_input = inp
        # Scheme-B: update alignment (yaw-only) when mocap pose arrives
        self._maybe_update_world_alignment(inp)

        # Always use world-aligned IMU quat in downstream logic
        imu_world = ImuMeasurement(
            quat_wxyz=self._imu_quat_to_world(inp.imu.quat_wxyz),
            accel_body=inp.imu.accel_body,
            gyro_body=inp.imu.gyro_body,
            frame=inp.imu.frame,
        )
        inp = EstimatorInput(
            joint=inp.joint,
            imu=imu_world,
            contact=inp.contact,
            tick=inp.tick,
            dt=inp.dt,
            external_position=inp.external_position,
        )
        self.process_input(inp)
        if self._use_imu_static_init and not self._imu_init_done:
            # 若开启 external pose，则在静态初始化阶段也允许用首帧 external_position 锚定 world。
            if (
                self._use_external_pose
                and (not self._external_pose_aligned)
                and inp.external_position is not None
            ):
                external_pos_world = self._preprocess_external_position(inp.external_position.position)
                self.kf.reset(
                    initial_base_position=external_pos_world,
                    initial_base_velocity=self.kf.getVelocityGlobal(),
                    initial_imu_accel_bias=self.kf.getImuAccelerationBias(),
                    reset_covariance=True,
                )
                self._external_pose_aligned = True
                if inp.tick is not None:
                    self._last_external_pose_tick = int(inp.tick)
                if self._last_output is not None:
                    self._last_output.base_position = self.kf.getPosition().copy()
                    self._last_output.base_velocity_world = self.kf.getVelocityGlobal().copy()
                    self._last_output.base_velocity_body = self.kf.getVelocityLocal().copy()
                    self._last_output.imu_accel_bias = self.kf.getImuAccelerationBias().copy()

            if self.imu_initializer is not None:
                self.imu_initializer.push(inp.imu)
                init_res = self.imu_initializer.get_result()
                if init_res is not None:
                    # 使用静止阶段的加速度均值作为初始 IMU 加速度 bias
                    # 注意：这里不改变基座位置 / 速度，只重置 bias 与协方差
                    self.kf.reset(
                        initial_base_position=self.kf.getPosition(),
                        initial_base_velocity=self.kf.getVelocityGlobal(),
                        initial_imu_accel_bias=init_res.bias_accel_body,
                        reset_covariance=True,
                    )
                    # 记录 gyro bias 供 GMO 使用（对应 Fast-LIO 中 init_state.bg = mean_gyr）
                    self._gyro_bias_body = init_res.mean_gyro_body.copy()
                    self._imu_init_done = True
                    self._imu_init_result = init_res
                    # 初始化完成后，重置 tick / 预积分起点，避免把初始化阶段计入积分
                    if inp.tick is not None:
                        self._last_tick = int(inp.tick)
                        self._imu_tick_increment = int(self._imu_dt_ms)
                        self._contact_tick_increment = int(self._contact_dt_ms)
                        self.imu_integral.reset(start_tick=int(inp.tick))

            # 在静态初始化阶段，直接返回当前缓存的输出（通常来自 reset）
            return self._last_output

        # 2) 计算与上一帧的 tick 差（整数 tick，单位：ms）
        dt_ticks = self._compute_tick_delta(inp.tick)

        # 3) 累积 IMU 预积分（每帧都推入一次）
        self.imu_integral.push(inp)

        # 4) 按 tick 差分更新调度计数器
        self._imu_tick_increment -= dt_ticks
        self._contact_tick_increment -= dt_ticks

        # 5) 到达或超过 IMU 更新周期：取出预积分结果，做一次 IMU 预测
        if self._imu_tick_increment <= 0:
            imu_integral = self.imu_integral.get_integral()
            if imu_integral is not None:
                dt_imu = float(imu_integral.dt) * 1e-3  # ms → s
                # 使用预积分得到的平均加速度替换当前帧加速度
                imu_meas = ImuMeasurement(
                    quat_wxyz=inp.imu.quat_wxyz,
                    accel_body=imu_integral.accel_body,
                    gyro_body=inp.imu.gyro_body,
                    frame=inp.imu.frame,
                )
                self.step_imu(dt_imu, imu_meas)
            # 重置 IMU 计数器（不考虑多次触发，约定输入频率不低于 imu_frenquency）
            self._imu_tick_increment = int(self._imu_dt_ms)

        # 5.5) 外部里程计位置测量更新（若启用）
        if self._use_external_pose and inp.external_position is not None:
            external_pos_world = self._preprocess_external_position(inp.external_position.position)
            # 不预设里程计频率：用里程计数据“实际到达时的 tick”计算 dt
            ext_tick = inp.external_position.tick if inp.external_position.tick is not None else inp.tick
            if ext_tick is not None and self._last_external_pose_tick is not None:
                dt_odom = float(int(ext_tick) - int(self._last_external_pose_tick)) * 1e-3
                if dt_odom <= 0.0:
                    dt_odom = float(max(dt_ticks, 1)) * 1e-3
            else:
                if dt_ticks > 0:
                    dt_odom = float(dt_ticks) * 1e-3
                else:
                    dt_odom = float(self._imu_dt_ms) * 1e-3
            dt_odom = max(float(dt_odom), 1e-6)

            if not self._external_pose_aligned:
                # 首帧外部位置：直接锚定 KF 的 world（消除常量平移差）
                self.kf.reset(
                    initial_base_position=external_pos_world,
                    initial_base_velocity=self.kf.getVelocityGlobal(),
                    initial_imu_accel_bias=self.kf.getImuAccelerationBias(),
                    reset_covariance=True,
                )
                self._external_pose_aligned = True

                # 若本 step 不触发接触更新，避免返回的 _last_output 仍然是旧的 world。
                if self._last_output is not None:
                    self._last_output.base_position = self.kf.getPosition().copy()
                    self._last_output.base_velocity_world = self.kf.getVelocityGlobal().copy()
                    self._last_output.base_velocity_body = self.kf.getVelocityLocal().copy()
                    self._last_output.imu_accel_bias = self.kf.getImuAccelerationBias().copy()
            else:
                self.kf.updatePositionMeasurement(dt_odom, external_pos_world)

            if ext_tick is not None:
                self._last_external_pose_tick = int(ext_tick)

        # 6) 到达或超过接触更新周期：做一次 GMO + KF 接触更新
        if self._contact_tick_increment <= 0:
            dt_eff_contact = (self._contact_dt_ms - self._contact_tick_increment) * 1e-3  # ms → s
            self._contact_tick_increment = int(self._contact_dt_ms)
            out = self.step_contacts(dt_eff_contact, inp.joint, inp.imu, inp.contact)
            self._last_output = out

        # 7) 若本次未进行接触更新，则复用上一帧输出
        return self._last_output
            
def quat_to_rotation(q_wxyz: Vec) -> Mat:
    w, x, y, z = q_wxyz
    R = np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
    ])
    return R
