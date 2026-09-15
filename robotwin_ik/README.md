# RoboTwin 多本体 TCP 逆运动学

将 `tcp_episode.json` 中的双臂 TCP 预测转换为目标本体的关节轨迹，并用 Rerun 对比目标 TCP 与实际 FK TCP。TCP 指两根手指末端内侧接触面的虚拟闭合中点；求解保留原轨迹的位置、朝向、时间戳和夹爪命令。

## 环境与入口

从仓库根目录运行。求解使用 `RoboTwin` Conda 环境，需要兼容 cuRobo、CUDA PyTorch、NumPy 和 PyYAML；可视化使用 `rerun` 环境，需要 Rerun SDK 0.35.0、NumPy、PyYAML 和 Pillow。`--help` 不加载 CUDA 或 Rerun。

| 本体 | 求解脚本（位于 `robotwin_ik/`） | 动作形状 | 默认结果目录（相对 episode） |
| --- | --- | --- | --- |
| Franka Panda | `solve_franka_panda.py` | `[T,16]` | `TCP_prediction_ik/franka_panda` |
| ARX-X5 | `solve_arx_x5.py` | `[T,14]` | `TCP_prediction_ik/arx_x5` |
| Piper | `solve_piper.py` | `[T,14]` | `TCP_prediction_ik/piper` |
| UR5-WSG | `solve_ur5_wsg.py` | `[T,14]` | `TCP_prediction_ik/ur5_wsg` |
| Aloha-Agilex | `solve_aloha_agilex.py` | `[T,14]` | `TCP_prediction_ik` |

Aloha 的双臂 ARX5 与单臂 ARX-X5 使用不同配置。根目录 `solve_robotwin_tcp_curobo_ik.py` 保留 Aloha 兼容入口。

## 求解并直接可视化

以 ARX-X5 为例，其他本体替换脚本名和 `--ik-dir` 最后一段即可：

```bash
conda run --no-capture-output -n RoboTwin python robotwin_ik/solve_arx_x5.py \
  4d_datasets/beat_block_hammer/episode_0000000

conda run --no-capture-output -n rerun python robotwin_ik/visualize_ik_rerun.py \
  4d_datasets/beat_block_hammer/episode_0000000 \
  --ik-dir 4d_datasets/beat_block_hammer/episode_0000000/TCP_prediction_ik/arx_x5
```

Linux 无 X11/Wayland 显示环境时自动启动 Web 查看器，也可加 `--web` 手动启用。远程运行时，在自己的电脑执行终端打印的 SSH 端口转发命令，再打开打印的浏览器地址。查看期间保持进程运行，按 `Ctrl+C` 退出。

白色表示目标 TCP，蓝/橙色表示实际 TCP，失败帧显示红色；界面同时展示机器人、RGB/点云、历史轨迹和误差曲线。点云包含源场景中的原机器人，目标本体模型叠加显示。

常用可视化参数：

- `--no-rgb`、`--no-point-cloud`：关闭 RGB 或点云。
- `--point-cloud-stride 4`：点云采样步长，默认 4。
- `--history 30`：显示最近 30 帧轨迹。
- `--show-candidate`：仅在求解失败时，明确选择查看失败候选与违规位置。

## 输入、配置与求解约束

默认读取 `<episode>/tcp_episode.json` 和 `extrinsics/<view>.npy`。预测位置单位为米，固定轴 XYZ 欧拉角单位为弧度；相机使用 OpenCV 坐标，外参为 `world_to_camera`，支持固定或逐帧矩阵。

模型资源默认从本仓库 `embodiments/RobotTwin_embodiments/` 加载，可通过 `--embodiments-root` 修改。基座间距优先级为 `--embodiment-distance` > `--task-config` 中第三项 > 0.6 米。双单臂配置示例：`embodiment: [franka-panda, franka-panda, 0.6]`。基座高度和朝向来自各本体配置；修改配置后需要重新求解，可视化使用结果中记录的变换。

求解流程为整段候选搜索、动态规划选择关节分支、联合优化所有帧。最终 float32 轨迹须满足：

- TCP 位置误差 ≤3 mm，朝向误差 ≤2°。
- 关节限位和单臂自碰撞检查通过。
- 相邻帧单关节变化 ≤0.5 rad，可用正数 `--max-joint-step-rad` 调整。

速度和加速度参与平滑优化，但不设硬上限；检查范围不包含桌面、跨臂碰撞或采样帧之间的碰撞。有限搜索失败不证明轨迹无解。

常用求解参数：`--prediction-json`、`--device cuda:0`、`--ik-seeds 64`、`--output-dir` 和 `--overwrite`。默认初态来自 `_initial_state.py` 的本体样例首帧，仅用于搜索和首帧软偏好，不固定首帧输出，也不读取源 episode 的关节状态。可用 `--initial-state-hdf5` 覆盖初态，此时额外需要 h5py。

闭合 TCP 标定见 [`tcp_calibrations.yml`](tcp_calibrations.yml)，包含指尖接触面偏移和资源校验值。模型或标定资源变化后需重新标定。

## 求解结果

| 文件 | 内容 |
| --- | --- |
| `robot_state.npy` | 整段成功的 float32 关节轨迹 |
| `robot_state_candidate.npy` | 整段失败时的诊断候选，不作为成功轨迹 |
| `metadata.json` | 本体、关节列、基座、TCP 标定、初态和资源校验值 |
| `diagnostics.json` | 每帧误差、连续性指标和失败原因 |

轨迹列顺序为 `[左臂关节, 左夹爪, 右臂关节, 右夹爪]`，关节单位为弧度，夹爪 `0=闭合，1=打开`。Franka 夹爪列为 7、15，其他本体为 6、13。

全部通过时退出码为 0；失败时只生成候选并退出 1；环境或求解异常记录为 `solver_error.json`。覆盖已有结果需显式传入 `--overwrite`。当前结果为 v3 格式；统一查看器也支持历史 v2，历史 Aloha v1 使用原可视化脚本。

## 单独查看 Franka homestate

无需 episode，直接显示配置中的初始姿态：

```bash
conda run --no-capture-output -n rerun python robotwin_ik/visualize_franka_homestate_rerun.py
```

此入口使用桌面查看器。可用 `--arm left|right|both` 和 `--gripper-open 0..1` 调整显示。

## 验证

```bash
conda run -n RoboTwin python -m unittest discover -s robotwin_ik/tests -v
conda run -n RoboTwin env ROBOTWIN_IK_GPU_TESTS=1 \
  python -m unittest discover -s robotwin_ik/tests -v
python -m robotwin_ik.tests.validate_saved_outputs /path/to/trajectory_results
```

历史整段验证结果见 [`trajectory_validation_results.json`](trajectory_validation_results.json)，其中失败候选不计为成功轨迹。
