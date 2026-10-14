# HumanoidStateEstimator

Python research code for estimating a humanoid robot's base position, velocity, and contact forces from joint measurements and IMU data. The estimator combines a generalized momentum observer with a Kalman filter.

## Installation

The core package needs Python, NumPy, SciPy, and Pinocchio:

```bash
pip install -e .
```

The validation and plotting scripts also need `matplotlib`, `tqdm`, `PyYAML`, and optionally `plotly`. Robot visualization needs `meshcat` and Pinocchio's visualization support.

This repository does not include a robot URDF, meshes, or recorded data. Supply files for your robot when using the examples below.

## Estimator

The implementation is in `state_estimator/state_estimator.py`. Configure it with your URDF and contact frames:

```python
from state_estimator.state_estimator import (
    RobotDynamicsModelConfig,
    StateEstimator,
    StateEstimatorConfig,
)

config = StateEstimatorConfig(
    model=RobotDynamicsModelConfig(
        urdf_path="/path/to/robot.urdf",
        contact_frames=["LL_FOOT", "LR_FOOT"],
    )
)
estimator = StateEstimator(config)
```

Pass an `EstimatorInput` to `estimator.step(...)` for each sensor sample. Its `tick` field is an integer timestamp in milliseconds. See the `JointMeasurement`, `ImuMeasurement`, and `ContactMeasurement` definitions in the same module for the expected values.

## Offline validation

With an NPZ log containing the estimator's expected signals:

```bash
pip install matplotlib tqdm pyyaml plotly
python validate_estimator.py --urdf /path/to/robot.urdf --log /path/to/log.npz
```

Validation results are written under `validate_results/`. Use `--log` with a directory to process several logs. Configuration fields can be overridden with arguments such as `kf.imu_accel_noise_density=0.01`.

To visualize a log with a URDF and meshes:

```bash
pip install meshcat
python visualize_robot.py --urdf /path/to/robot.urdf --mesh-search-path /path/to/package/root --log /path/to/log.npz
```

## License

MIT. See [LICENSE](LICENSE).
