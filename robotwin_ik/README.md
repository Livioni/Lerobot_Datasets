# RoboTwin 多本体闭合 TCP 逆解

将已有 `tcp_episode.json` 中的双臂 TCP 预测转换成指定本体的关节动作。**TCP 是两根手指末端内侧接触面中心的中点**；夹爪张开时仍使用同一个虚拟闭合点。保持这个点在原场景中的位置、朝向和时序，按不同夹爪的长度及安装方向补偿末端目标，不平移或缩放原轨迹。

## 入口与环境

每个本体一个脚本，共用 `_solver.py`、`_kinematics.py` 和本体配置加载器：

| 本体 | 脚本 | 关节动作维度 | 默认结果子目录 |
|---|---|---|---|
| franka-panda | `solve_franka_panda.py` | `[T,16]` | `TCP_prediction_ik/franka_panda/` |
| ARX-X5 | `solve_arx_x5.py` | `[T,14]` | `TCP_prediction_ik/arx_x5/` |
| piper | `solve_piper.py` | `[T,14]` | `TCP_prediction_ik/piper/` |
| ur5-wsg | `solve_ur5_wsg.py` | `[T,14]` | `TCP_prediction_ik/ur5_wsg/` |
| 原 aloha-agilex | `solve_aloha_agilex.py` | `[T,14]` | `TCP_prediction_ik/`（兼容旧入口） |

`aloha-agilex` 中的双臂 ARX5 与单臂 `ARX-X5` 模型不同，不能交换配置。根目录 `solve_robotwin_tcp_curobo_ik.py` 继续支持原有命令；原可视化脚本继续用于 Aloha 的旧格式输出。

求解依赖 NumPy、PyYAML、兼容 cuRobo 和 CUDA PyTorch；此机器可使用 `RoboTwin` Conda 环境。仅显式指定 HDF5 初值时需要 h5py。Rerun 回放使用 `rerun` 环境，依赖 NumPy、PyYAML、Pillow 和支持 `rr.urdf.UrdfTree` / `frame_prefix` 的 Rerun SDK；当前验证版本为 0.35.0。`--help` 不加载 CUDA 或 Rerun。

从仓库根目录运行：

```bash
conda run --no-capture-output -n RoboTwin python robotwin_ik/solve_franka_panda.py \
  4d_datasets/beat_block_hammer/episode_0000000

conda run --no-capture-output -n RoboTwin python robotwin_ik/solve_arx_x5.py \
  4d_datasets/beat_block_hammer/episode_0000000

conda run --no-capture-output -n RoboTwin python robotwin_ik/solve_piper.py \
  4d_datasets/beat_block_hammer/episode_0000000

conda run --no-capture-output -n RoboTwin python robotwin_ik/solve_ur5_wsg.py \
  4d_datasets/beat_block_hammer/episode_0000000
```

输入默认是 `<episode>/tcp_episode.json` 和 `<episode>/extrinsics/<view>.npy`。新本体不读取源 episode 的 `robot_state.npy`。预测包含米制 `xyz_m`、弧度制固定轴 XYZ `rpy_rad`，以及二值夹爪或带阈值的夹爪概率；相机为 OpenCV 坐标，外参为 `world_to_camera`，支持固定或逐帧外参。

支持参数：`--prediction-json`、`--device cuda:0`、`--ik-seeds 64`、`--max-joint-step-rad 0.5`、`--output-dir`、`--overwrite`、`--embodiments-root`、`--embodiment-distance`、`--task-config`。已有结果需显式传入 `--overwrite`；输出目录不能是源 episode 本身。

## RoboTwin 采集配置与基座

求解和回放的资源统一默认从本仓库的 `embodiments/RobotTwin_embodiments/` 只读加载（相对仓库位置解析，不依赖运行目录），可用 `--embodiments-root /path/to/RobotTwin_embodiments` 修改。原 Aloha 入口也使用此目录。依据各本体的 `config.yml`、`curobo_tmp.yml`（缺失时读取 `curobo.yml`）、URDF，以及 `RoboTwin/envs/robot/robot.py`、`scripts/collect_data.py` 的加载规则。模板 YAML 中的 `${ASSETS_PATH}` 由加载器替换为所选本地资源路径，无需设置环境变量或生成 `curobo.yml`。求解脚本仍兼容显式 `--robotwin-root` 参数，但默认不访问外部 RoboTwin 仓库。回放旧结果时，也从本地资源目录查找同一本体，并核验 URDF 校验值。

这四种本体均通过两个独立单臂模型组成双臂。采集 YAML 应使用三个元素，例如：

```yaml
embodiment: [franka-panda, franka-panda, 0.6]
```

第三项是米制基座间距。只写 `[franka-panda]` 会进入单个双臂 URDF 的加载分支，不适用于这些单臂模型。

间距优先级为：显式 `--embodiment-distance` > `--task-config` 中第三项 > **0.6 米**。0.6 米是本工具的默认值，不是 RoboTwin 仓库中已有的四本体采集值。

```bash
conda run --no-capture-output -n RoboTwin python robotwin_ik/solve_piper.py \
  4d_datasets/place_dual_shoes/episode_0000092 \
  --embodiment-distance 0.7 --output-dir /tmp/piper_07m

# 也可以读取现有采集配置；本体名必须与脚本匹配。
# python robotwin_ik/solve_franka_panda.py <episode> --task-config /path/to/task.yml
```

0.6 米间距时，左右模型相对原 `robot_pose` 的世界 X 分量分别增加 -0.3 / +0.3：

| 本体 | 左基座 XYZ (m) | 右基座 XYZ (m) | 基座朝向 |
|---|---|---|---|
| franka-panda | `[-0.3,-0.65,0.75]` | `[0.3,-0.65,0.75]` | 绕 Z 轴 90° |
| ARX-X5 | `[-0.3,-0.35,0.784]` | `[0.3,-0.35,0.784]` | 绕 Z 轴 90° |
| piper | `[-0.3,-0.45,0.75]` | `[0.3,-0.45,0.75]` | 绕 Z 轴 90° |
| ur5-wsg | `[-0.3,-0.65,0.65]` | `[0.3,-0.65,0.75]` | 单位旋转 |

UR5 左右高度差来自原配置，未自动改为相同高度。四元数按 `wxyz` 解释并归一化。

## 闭合 TCP 与关节求解

`tcp_calibrations.yml` 保存末端坐标中的完整平移、接触面选择依据和资源 SHA256。平移由本地指尖碰撞网格的接触面面积加权中心计算，并考虑夹爪闭合时的 URDF 限位与所有固定安装变换。它不是网格包围盒中心，也不是 RoboTwin 的 `gripper_bias`。

| 本体 | 求解末端 | 闭合 TCP 平移 XYZ (mm，近似) |
|---|---|---|
| franka-panda | `panda_hand` | `[0,0,105.779542]` |
| ARX-X5 | `link6` | `[149.335045,-0.002,-0.853987]` |
| piper | `link6` | `[0.000004,0,134.833004]` |
| ur5-wsg | `ee_link` | `[143.500001,-0.013823,0.013823]` |

末端到 TCP 的旋转使用各本体 `delta_matrix`。这里已核对 SAPIEN 的关节轴坐标旋转与 `global_trans_matrix` 抵消；不能把后者再直接叠加到 URDF link 位姿上。模型或标定网格发生变化时，工具会要求重新标定，避免继续套用旧偏移。

令 `T_A_B` 表示从 B 到 A 的变换，求解目标为：

```text
T_world_tcp = inverse(T_camera_world) @ T_camera_tcp
T_base_ee   = inverse(T_root_base) @ inverse(T_world_root)
              @ T_world_tcp @ inverse(T_ee_tcp)
```

五种本体默认对整个 TCP 序列求解，保持世界坐标中的位置、朝向和时间戳。先使用批量 cuRobo IK 为每帧搜索多个有效候选，再通过候选图的动态规划回溯整段关节解分支，最后对所有帧的关节变量联合优化。后续目标可以影响前面的姿态选择。

`--ik-seeds` 默认 64（范围 2–256），搜索断开时扩展至最多 256；候选同时从前后相邻帧补充。最多选取 4 条整段路径，使用 CUDA 可微 FK、增广拉格朗日和 L-BFGS 优化速度、加速度及首帧姿态偏好。若候选图未连通，仍尝试优化违反约束较小的整段初值；有限搜索失败不证明数学上无解。

候选检查包括 cuRobo 成功、模型约束、锁定关节和独立 URDF FK。最终输出在转换成 float32 后重新验收：闭合 TCP **位置误差 ≤3 mm、旋转误差 ≤2°**，关节限位及单臂自碰撞检查通过，任意相邻帧单个关节的实际变化量不超过 **0.5 rad**。可用正数 `--max-joint-step-rad` 调整，不能用 0 关闭；不对差值取模，也不通过裁剪输出掩盖错误。

初态仍来自 `_initial_state.py` 的采集样例首帧，或显式 `--initial-state-hdf5`。它用于搜索初始化和首帧软偏好，不是额外时间采样点，也不固定首帧输出。五种本体均不读取源 episode 的关节状态。ARX-X5/Piper 的配置向量修正与 TCP 标定保持原规则。

速度和加速度按原始时间戳参与平滑及诊断，没有硬速度/加速度上限；仍只检查原采样时刻的单臂自碰撞，不新增桌面或跨臂避碰。夹爪二值命令保持原预测，不参与机械臂连续性优化。

## 输出与回放

每个结果目录包含：

- `robot_state.npy`：仅整段成功时生成，float32，列顺序为 `[左臂关节, 左夹爪, 右臂关节, 右夹爪]`。关节按 RoboTwin `arm_joints_name` 排列，单位弧度；夹爪 `0=闭合，1=打开`。
- `metadata.json`：本体、列名/索引、初始姿态、基座位姿、TCP 标定、源配置快照、文件校验值、运行环境和耗时。
- `robot_state_candidate.npy`：整段失败时另存的有限候选，供诊断回放；不同时生成 `robot_state.npy`。
- `diagnostics.json`：每侧候选搜索和联合优化记录、每帧 FK 误差、关节步长、速度、加速度及失败原因。

新结果统一使用 `schema_version: 3` / `robotwin_closed_tcp_trajectory_v3`；`output.state_file` 与 `output.role` 明确区分成功解和失败候选。

Franka 的夹爪列是 7、15；另外三种是 6、13。两臂全程通过才保存成功解并退出 **0**；否则状态为 `failed`，保存候选并退出 **1**，不再用失败保持填充轨迹。环境或求解异常记录为 `solver_error.json`，不伪造候选。已有结果需显式 `--overwrite`，成功/失败切换时清理过期状态文件。

```bash
conda run --no-capture-output -n rerun python robotwin_ik/visualize_ik_rerun.py \
  4d_datasets/place_dual_shoes/episode_0000092 \
  --ik-dir 4d_datasets/place_dual_shoes/episode_0000092/TCP_prediction_ik/ur5_wsg \
  --output rrd_output/ik/place_dual_shoes_piper.rrd \
  --point-cloud-stride 2

# 导出回放，不打开窗口。
conda run --no-capture-output -n rerun python robotwin_ik/visualize_ik_rerun.py \
  4d_datasets/beat_block_hammer/episode_0000000 \
  --ik-dir 4d_datasets/beat_block_hammer/episode_0000000/TCP_prediction_ik/piper \
  --output rrd_output/ik/beat_block_hammer_piper.rrd
```

其他本体只需替换 `--ik-dir` 最后一段。支持 `--no-rgb`、`--no-point-cloud`、`--point-cloud-stride 4`、`--history 30`。失败候选必须加 `--show-candidate`，回放明确显示 FAILED CANDIDATE 和违规位置；历史 v2 结果仍可读取，历史 Aloha v1 结果使用旧可视化脚本。新 Aloha v3 结果也使用本页的统一回放脚本。

白色是目标闭合 TCP，蓝/橙色是实际 FK TCP，失败帧显示红色；同时显示坐标轴、历史轨迹、误差连线、原 RGB/点云、关节角与夹爪曲线。源点云仍包含原机器人，新本体模型是叠加显示。Rerun 使用元数据中的基座与 TCP 变换，不重新猜测本体配置。

夹爪显示按 `gripper_scale[0] + open*(gripper_scale[1]-gripper_scale[0])` 及 mimic 关系换算，并按 URDF 限位裁剪；它不是把二值直接当手指位移。棱柱关节显式使用 `origin @ motion`。Piper 右指 DAE 中存在异常三角形，临时显示 URDF 对该指使用同源 STL，UR5 的重复 `limit` 标签也仅在临时显示副本中合并，并保留首个标签的关节限位；原始 RoboTwin 文件不变。

## 验证

CPU 测试覆盖全局分支回溯、与穷举一致的多路径搜索、角度分支、非均匀时间、float32 步长验收、成功/候选文件和回放兼容性。GPU 测试覆盖五种本体的整段 FK→IK→FK、TCP 梯度及不可达目标报告。

```bash
conda run -n RoboTwin python -m unittest discover -s robotwin_ik/tests -v
conda run -n RoboTwin env ROBOTWIN_IK_GPU_TESTS=1 \
  python -m unittest discover -s robotwin_ik/tests -v

# 可传入单个结果目录或包含多个结果的目录，独立复核 v2/v3 输出。
python -m robotwin_ik.tests.validate_saved_outputs /path/to/trajectory_results
```

历史 `validation_results.json` 属于旧逐帧策略，不能作为整段规划成功率。新验收要求所有原始帧同时满足 TCP 与连续性要求，失败候选仅用于检查。

2026-09-13，RTX 5090 D v2 / PyTorch 2.9.1+cu128：24 项测试通过，9 组完整 episode 均经过独立 FK 和连续性复核，结果如下。表中的失败均输出单独候选，未输出成功 `robot_state.npy`。UR5 使用当前左右等高 0.65 m 的基座配置。

| 本体 | hammer（121 帧） | shoes（256 帧） |
|---|---|---|
| ur5-wsg | 整段成功 | 整段成功 |
| ARX-X5 | 整段成功 | 整段成功 |
| aloha-agilex | 整段成功 | 本轮未测 |
| franka-panda | 未通过：237/242 臂帧合格 | 未通过：496/512 臂帧合格 |
| piper | 未通过：188/242 臂帧合格 | 未通过：436/512 臂帧合格 |

成功轨迹的最大单关节帧间变化不超过 0.159 rad，所有帧满足 3 mm / 2° 阈值。成功 UR5、失败 Franka 候选及统一 Aloha 回放均已导出验证。具体指标、源文件校验值和临时输出路径见 [`trajectory_validation_results.json`](trajectory_validation_results.json)。


Franka 的 D435 相机 DAE 使用毫米单位，Rerun 导入时未应用该单位；临时显示 URDF 按 DAE 的 `meter="0.001"` 设置网格缩放，恢复约 90 × 25 × 25 mm 的尺寸。相机挂载变换、源模型与 IK/TCP 标定均保持原值。

### 单独查看 Franka Panda homestate

新脚本直接读取本体 `config.yml` 中的 `homestate`，无需 episode。默认显示左右两臂（基座间距 0.6 m）、打开的夹爪、基座/TCP 坐标轴，以及每个关节的弧度和角度数值。

```bash
conda run --no-capture-output -n rerun python robotwin_ik/visualize_franka_homestate_rerun.py

# 仅左臂、夹爪闭合；保存文件供桌面 Rerun 打开。
conda run --no-capture-output -n rerun python robotwin_ik/visualize_franka_homestate_rerun.py \
  --arm left --gripper-open 0 --output rrd_output/ik/franka_left_homestate.rrd
```

支持 `--arm both|left|right`、`--gripper-open 0..1`、`--embodiment-distance` 和 `--embodiments-root`。不带 `--output` 时打开 Rerun 窗口，带该参数时只导出 `.rrd`。

### RoboTwin 初始化姿态与回放首帧

五种本体（含根目录旧 Aloha 入口）默认使用 `_initial_state.py::BUILTIN_INITIAL_JOINTS` 中的数值。这些值来自之前提供的五个 HDF5 样例第 0 帧，保留原始浮点精度，并按关节名匹配当前模型。默认不读取 HDF5，也不依赖 `example` 目录或其文件；没有重新从 config homestate 推算初值。`metadata.json.initialization.source` 与 `solver.initial_seed_source` 为 `builtin_target_example_frame_0`，同时记录具体关节值；`path` / `sha256` 为 null，表示运行时没有读取初值文件。

可通过 `--initial-state-hdf5 /path/to/recording.hdf5` 显式覆盖默认值。仅此时需要 h5py；加载器读取 `joint_action/left_arm[0]`、`right_arm[0]`，并与 `joint_action/vector[0]` 核对 `[左臂关节, 左夹爪, 右臂关节, 右夹爪]` 布局。维度、有限性、关节限位或文件不符时会报错，不会忽略用户指定的路径。该模式的来源为 `target_embodiment_example_frame_0`，并记录路径与 SHA256。样例帧数无需与待求解轨迹相同。

整段候选搜索使用内置初态和前后帧候选；优化只对首帧施加初态软偏好。样例后续状态不参与求解，夹爪列不送入臂关节 IK。RoboTwin 样例中的 joint_action 记录的是关节 drive target，并非仿真 qpos 测量值。

当前五个样例的首帧与 config homestate 基本相同：Franka 为 `[0, 0.1963495463, 0, -2.6179938316, 0, 2.9415926933, 0.7853981853]` rad，Piper、ARX-X5、Aloha 均为零，UR5 为约 `[-1.54470003, -1.54470003, -1.54470003, -1.57939994, 1.57939994, 0]` rad。更换来源不会自动解决解分支突变。

- `metadata.json` 的 `arms.<side>.initial_joint_state` 保存初始种子；`robot_state.npy[0]` 保存首帧目标的求解结果。初始化种子不等于对首帧输出的固定约束。夹爪输出跟随输入轨迹，也不强制改成初始化时的打开状态。

输入 TCP 来自另一种本体，通常与目标本体 homestate 的 TCP 不同。例如 `place_dual_shoes/episode_0000092` 默认基座配置下，Franka home TCP 与目标首帧约差 10.4 cm / 82.7°，Piper 约差 9.8 cm / 5.1°。因此回放首帧不会显示 homestate。若执行时必须从 homestate 开始，应另行规划 `homestate → 首帧 IK` 的准备动作，再执行原轨迹；准备动作需要单独的时间段和碰撞/运动约束检查，不能直接覆盖首帧并仍声称匹配原 TCP。

### UR5 双臂基座高度

RoboTwin 原始 `ur5-wsg/config.yml` 设置 `robot_pose` 为左臂 `[0, -0.65, 0.65, 1, 0, 0, 0]`、右臂 `[0, -0.65, 0.75, 1, 0, 0, 0]`。RoboTwin 对双单臂组合只追加 x 方向的 `±distance/2`，没有统一 z 高度。因此默认距离 0.6 m 时，左右基座分别是 `[-0.3, -0.65, 0.65]` 与 `[0.3, -0.65, 0.75]` m，右臂高 10 cm。求解读取本地配置，回放使用求解时保存的基座变换；本地配置可调整（当前用户已将右臂降到 0.65 m），旧输出不会自动更新。源代码未说明原始高度差的设计原因。若改为等高，需要修改求解使用的基座配置并重新求解，不能仅平移回放模型，否则世界 TCP 将不再匹配。

### 整段连续性与失败候选

默认已启用整段求解和 0.5 rad 步长硬验收，无需额外策略开关。旧 `--fallback-seeds`、逐帧首选/回退、失败保持及对应诊断字段已删除。调大 `--ik-seeds` 增加候选搜索预算，但不保证任意输入存在连续解。

```bash
conda run --no-capture-output -n RoboTwin python robotwin_ik/solve_franka_panda.py \
  4d_datasets/place_dual_shoes/episode_0000092 \
  --output-dir /tmp/franka_trajectory

# 仅当结果为 failed 时，用此选项查看候选及失败位置。
conda run --no-capture-output -n rerun python robotwin_ik/visualize_ik_rerun.py \
  4d_datasets/place_dual_shoes/episode_0000092 \
  --ik-dir /tmp/franka_trajectory --show-candidate \
  --output /tmp/franka_candidate.rrd
```
