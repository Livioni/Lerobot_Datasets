# RoboTwin 多本体 TCP 逆运动学

将 `tcp_episode.json` 中的双臂 TCP 预测转换为目标本体的关节轨迹，并用 Rerun 对比目标 TCP 与实际 FK TCP。TCP 指两根手指末端内侧接触面的虚拟闭合中点；求解保留原轨迹的位置、朝向、时间戳和夹爪命令。

## 环境与入口

从仓库根目录运行。以下求解示例使用 `curobo` Conda 环境和 `--solver differential`，需要新版 cuRobo、CUDA PyTorch、NumPy 和 PyYAML；可视化使用 `rerun` 环境，需要 Rerun SDK 0.35.0、NumPy、PyYAML 和 Pillow。`--help` 不加载 CUDA 或 Rerun。

| 本体 | 求解脚本（位于 `robotwin_ik/`） | 动作形状 | 下方配套示例结果目录（通过 `--output-dir` 指定，相对 episode） |
| --- | --- | --- | --- |
| Franka Panda | `solve_franka_panda.py` | `[T,16]` | `TCP_prediction_ik/franka_panda` |
| ARX-X5 | `solve_arx_x5.py` | `[T,14]` | `TCP_prediction_ik/arx_x5` |
| Piper | `solve_piper.py` | `[T,14]` | `TCP_prediction_ik/piper` |
| UR5-WSG | `solve_ur5_wsg.py` | `[T,14]` | `TCP_prediction_ik/ur5_wsg` |
| Aloha-Agilex | `solve_aloha_agilex.py` | `[T,14]` | `TCP_prediction_ik` |

Aloha 的双臂 ARX5 与单臂 ARX-X5 使用不同配置。根目录 `solve_robotwin_tcp_curobo_ik.py` 保留 Aloha 兼容入口。

## 使用 cuRobo Differential IK

五个求解入口均支持 `--solver differential`。该模式使用 cuRobo 2 的 LM seed solver 做离线逐帧收敛，以上一帧关节位置作为初值。每帧先执行最多 128 次 LM 数值迭代；如果 TCP、模型约束或关节步长未通过，再尝试多个 LM 初值，并优先选择满足约束且接近上一帧的解。不执行 LBFGS；失败恢复会搜索其他关节分支，因此不保证始终保持同一分支。

当前机器已创建独立的 `curobo` Conda 环境，安装 Python 3.10、PyTorch 2.9.1（CUDA 12.8）、cuda-core 1.2.0、cuda-bindings 12.9.7、Warp 1.17.0 和数据读取依赖。仓库内新版 cuRobo 已通过 editable 模式安装，无需设置 `PYTHONPATH`。从仓库根目录运行：

```bash
conda run --no-capture-output -n curobo python robotwin_ik/solve_piper.py \
  4d_datasets/beat_block_hammer/episode_0000000 --solver differential
```

替换脚本名即可求解 Franka、ARX-X5、UR5-WSG 或 Aloha。默认输出到 `<episode>/TCP_prediction_differential_ik/<本体>`；Aloha 直接输出到 `TCP_prediction_differential_ik`。state 列顺序、夹爪命令和可视化格式与原模式相同。查看时将 `--ik-dir` 指向新目录；失败候选需加 `--show-candidate`。

初态来自现有内置值或 `--initial-state-hdf5`，首帧直接求解首个 TCP，不要求机器人在一个采样间隔内从初态运动到目标。常规跟踪使用单初值，失败恢复使用 `--ik-seeds` 个初值（默认 64）。位置/朝向/速度/加速度的 LM 权重分别为 1、1、0、0，并记录在 metadata 中。原时间戳保留用于输出和速度、加速度诊断，不作为 LM 单步速度裁剪的控制周期；这适用于离线 state 转换，不代表原采样时间下已满足动力学限制。

局部极值、奇异位形以及不同本体的可达空间和关节限位仍可能导致求解失败；多初值恢复失败不证明目标无解。每帧仍检查 3 mm / 2° TCP 容差、关节限位、自碰撞和相邻输出关节变化；`--max-joint-step-rad` 是最终验收阈值，不对求解结果做裁剪。LM 路径不执行碰撞避障优化，碰撞由最终模型约束检查判定。内置初态本身不保证无碰撞。

未通过的有限候选仍作为下一帧的诊断初态，整段只保存 `robot_state_candidate.npy` 并退出 1；不会将未收敛状态标记为成功。此模式的恢复仍使用 LM，不会切换到旧版全轨迹求解器。每帧 `tcp_tracking_success` 单独表示 TCP 是否在容差内；`violated_model_constraints` 区分关节空间约束（`cspace`）与自碰撞（`self_collision`），最终 `success` 还要求步长等检查全部通过。脚本默认仍为 `--solver trajectory`，该旧版后端依赖原 `RoboTwin` 环境；在新版 `curobo` 环境中必须显式指定 `--solver differential`。

新版 GPU 回归测试：

```bash
ROBOTWIN_DIFFERENTIAL_GPU_TESTS=1 conda run --no-capture-output -n curobo \
  python -m unittest robotwin_ik.tests.test_differential -v
```

## 求解并直接可视化

以下求解命令显式指定 `--output-dir`，将结果写入原 `TCP_prediction_ik` 目录，因此现有可视化命令可继续使用。以 ARX-X5 为例，其他本体替换脚本名和 `--ik-dir` 最后一段即可：

```bash
conda run --no-capture-output -n curobo python robotwin_ik/solve_arx_x5.py \
  4d_datasets/beat_block_hammer/episode_0000000 --solver differential \
  --output-dir 4d_datasets/beat_block_hammer/episode_0000000/TCP_prediction_ik/arx_x5 --overwrite

conda run --no-capture-output -n curobo python robotwin_ik/solve_franka_panda.py \
  4d_datasets/beat_block_hammer/episode_0000000 --solver differential \
  --output-dir 4d_datasets/beat_block_hammer/episode_0000000/TCP_prediction_ik/franka_panda

conda run --no-capture-output -n curobo python robotwin_ik/solve_piper.py \
  4d_datasets/place_dual_shoes/episode_0000092 --solver differential \
  --output-dir 4d_datasets/place_dual_shoes/episode_0000092/TCP_prediction_ik/piper --overwrite

conda run --no-capture-output -n curobo python robotwin_ik/solve_ur5_wsg.py \
  4d_datasets/beat_block_hammer/episode_0000000 --solver differential \
  --output-dir 4d_datasets/beat_block_hammer/episode_0000000/TCP_prediction_ik/ur5_wsg

conda run --no-capture-output -n rerun python robotwin_ik/visualize_ik_rerun.py \
  4d_datasets/place_dual_shoes/episode_0000092 \
  --ik-dir 4d_datasets/place_dual_shoes/episode_0000092/TCP_prediction_ik/arx_x5 --show-candidate

conda run --no-capture-output -n rerun python robotwin_ik/visualize_ik_rerun.py \
  4d_datasets/place_dual_shoes/episode_0000092 \
  --ik-dir 4d_datasets/place_dual_shoes/episode_0000092/TCP_prediction_ik/franka_panda \
  --show-candidate

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

本文命令使用 Differential IK 顺序跟踪 TCP。旧版 `trajectory` 后端使用整段候选搜索、动态规划选择关节分支、联合优化所有帧。最终 float32 轨迹须满足：

- TCP 位置误差 ≤3 mm，朝向误差 ≤2°。
- 关节限位和单臂自碰撞检查通过。
- 相邻帧单关节变化 ≤0.5 rad，可用正数 `--max-joint-step-rad` 调整。

旧版 `trajectory` 的速度和加速度参与平滑优化；当前离线 Differential IK 只报告速度和加速度诊断，不设硬上限；检查范围不包含桌面、跨臂碰撞或采样帧之间的碰撞。有限搜索失败不证明轨迹无解。

常用求解参数：`--solver differential`、`--prediction-json`、`--device cuda:0`、`--output-dir` 和 `--overwrite`；`--ik-seeds 64` 在 Differential IK 中控制失败恢复初值数，在旧版 `trajectory` 中控制初始搜索种子数。默认初态来自 `_initial_state.py` 的本体样例首帧，在 Differential IK 中作为顺序跟踪初态，在旧版后端中用于搜索和首帧软偏好；不固定首帧输出，也不读取源 episode 的关节状态。可用 `--initial-state-hdf5` 覆盖初态，此时额外需要 h5py。

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

## 查看求解器内置初态（五款本体）

直接读取 `_initial_state.py` 的 `BUILTIN_INITIAL_JOINTS`，不需要 episode 或 CUDA；这与 `config.yml` 的 `homestate` 不同。默认在独立标签页展示五款本体的双臂初态、关节角（rad/deg）、基座和闭合 TCP 坐标。

```bash
conda run --no-capture-output -n rerun python robotwin_ik/visualize_initial_states_rerun.py

# 只看 Piper；也可选择 franka-panda、ARX-X5、ur5-wsg、aloha-agilex
conda run --no-capture-output -n rerun python robotwin_ik/visualize_initial_states_rerun.py --robot piper

# 导出全部初态
conda run --no-capture-output -n rerun python robotwin_ik/visualize_initial_states_rerun.py --output /tmp/robotwin_initial_states.rrd
```

支持 `--arm left|right|both`、`--gripper-open 0..1`、`--embodiment-distance 0.6`（Aloha 不使用该间距）和 `--embodiments-root`。基座变换来自当前本体配置；夹爪默认使用内置初态的开度。初态是求解种子和首帧软偏好，不是强制的输出第一帧，也不表示已经通过碰撞检查。

无图形显示环境自动启动 Web viewer，也可显式加 `--web`。远程访问须按终端提示同时转发网页和数据两个端口。修改 `_initial_state.py` 后重新运行即可查看最新初态。
