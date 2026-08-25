# 示例数据集字段含义说明

本仓库在 `assets/example/` 下提供了三个紧凑的 LeRobot v3.0 示例数据集，分别覆盖单臂、双臂（含深度与标定）以及双臂（含速度/力）三类形态。本文档以中文逐字段解释每个数据集的 `state`、`action`、视觉与标定等字段含义，便于快速理解数据结构。

三个示例的共同字段（LeRobot v3.0 标准元数据）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | float32 | 当前帧在 episode 内的时间戳（秒），从 0 开始 |
| `frame_index` | int64 | 当前帧在 episode 内的序号（从 0 开始） |
| `episode_index` | int64 | 当前 episode 的序号 |
| `index` | int64 | 全局帧序号（跨 episode 的唯一递增索引） |
| `task_index` | int64 | 任务编号，对应 `meta/tasks.parquet` 中的 `task_index` |

每个数据集额外在 `meta/` 下包含：

- `meta/info.json`：数据集整体信息（fps、特征定义、机器人类型、splits 等）。
- `meta/tasks.parquet`：任务文本表，`task_index → task` 描述。
- `meta/episodes/chunk-000/file-000.parquet`：每个 episode 的起止索引、视频文件映射、以及 episode 级别的统计信息（min/max/mean/std/q01/q10/q50/q90/q99 等）。
- `meta/stats.json`：全数据集各特征的统计量（min/max/mean/std/q01–q99 等）。
- `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`：逐帧数值数据（state/action/标定等）。
- `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`：每个相机视角对应的视频文件。

---

## 1. LIBERO（单臂 Franka，仿真）

- **路径**：`assets/example/LIBERO`
- **机器人**：`franka`（单臂）
- **FPS**：20
- **示例任务**：`put the white mug on the left plate and put the yellow and white mug on the right plate`
- **示例长度**：1 个 episode，共 214 帧

### 1.1 视觉

| 字段 | 类型 | 分辨率 | 说明 |
|------|------|--------|------|
| `observation.images.image` | video (av1) | 256×256×3 | 第三视角（外部）RGB 图像 |
| `observation.images.wrist_image` | video (av1) | 256×256×3 | 腕部相机 RGB 图像 |

### 1.2 `observation.state`（8 维，float32）

Franka 末端执行器（end-effector）位姿 + 夹爪状态，采用**末端位姿（Euler/Axis-angle）表示**而非关节角：

| 索引 | 名称 | 含义 |
|------|------|------|
| 0 | `x` | 末端位置 x（米） |
| 1 | `y` | 末端位置 y（米） |
| 2 | `z` | 末端位置 z（米） |
| 3 | `axis_angle1` | 末端姿态 axis-angle 第 1 分量 |
| 4 | `axis_angle2` | 末端姿态 axis-angle 第 2 分量 |
| 5 | `axis_angle3` | 末端姿态 axis-angle 第 3 分量 |
| 6 | `gripper` | 夹爪状态（左指/第一分量） |
| 7 | `gripper` | 夹爪状态（右指/第二分量，与索引 6 对称） |

> 注：LIBERO 的 state 使用末端位姿而非关节角，因此本仓库的可视化脚本对它不做 URDF 关节回放，仅作为视频 + 信号展示。

### 1.3 `action`（7 维，float32）

与 state 同坐标系的目标指令：

| 索引 | 名称 | 含义 |
|------|------|------|
| 0 | `x` | 目标末端位置 x |
| 1 | `y` | 目标末端位置 y |
| 2 | `z` | 目标末端位置 z |
| 3 | `axis_angle1` | 目标姿态 axis-angle 第 1 分量 |
| 4 | `axis_angle2` | 目标姿态 axis-angle 第 2 分量 |
| 5 | `axis_angle3` | 目标姿态 axis-angle 第 3 分量 |
| 6 | `gripper` | 目标夹爪开合度（0=闭合，1=张开） |

---

## 2. RoboTwin2（双臂 + 深度 + 标定，仿真）

- **路径**：`assets/example/RoboTwin2`
- **机器人**：`unified_robot`（双臂统一表示，关节角形式）
- **FPS**：15
- **示例任务**：`Locate the smooth green plastic bottle and lift it upright with the left arm.`
- **示例长度**：1 个 episode，共 180 帧

### 2.1 视觉（RGB + 深度）

| 字段 | 类型 | 分辨率 | 说明 |
|------|------|--------|------|
| `observation.images.cam_high` | video (h264) | 240×320×3 | 高位第三视角 RGB |
| `observation.images.cam_high_depth` | video (hevc, gray12le) | 240×320×1 | 高位视角深度图，单位 mm，无效值 0 |
| `observation.images.cam_left_wrist` | video (h264) | 240×320×3 | 左腕相机 RGB |
| `observation.images.cam_left_wrist_depth` | video (hevc, gray12le) | 240×320×1 | 左腕深度图，单位 mm |
| `observation.images.cam_right_wrist` | video (h264) | 240×320×3 | 右腕相机 RGB |
| `observation.images.cam_right_wrist_depth` | video (hevc, gray12le) | 240×320×1 | 右腕深度图，单位 mm |

深度图编码参数（来自 `info.json`）：

- `depth_unit`: mm
- `invalid_value`: 0
- `depth_min`: 0.0 m，`depth_max`: 5.0 m
- `shift`: 3.5，`use_log`: true（对数编码后存入 12-bit 灰度，解码需逆变换）

### 2.2 `observation.state` / `action`（14 维，float32）

采用**左右臂关节角 + 夹爪**的统一表示，state 与 action 维度和命名一致：

| 索引 | 名称 | 含义 |
|------|------|------|
| 0 | `left_joint_0` | 左臂关节 0（rad） |
| 1 | `left_joint_1` | 左臂关节 1（rad） |
| 2 | `left_joint_2` | 左臂关节 2（rad） |
| 3 | `left_joint_3` | 左臂关节 3（rad） |
| 4 | `left_joint_4` | 左臂关节 4（rad） |
| 5 | `left_joint_5` | 左臂关节 5（rad） |
| 6 | `left_joint_6` | 左臂关节 6（rad，若有第 7 自由度则冗余/固定） |
| 7 | `right_joint_0` | 右臂关节 0（rad） |
| 8 | `right_joint_1` | 右臂关节 1（rad） |
| 9 | `right_joint_2` | 右臂关节 2（rad） |
| 10 | `right_joint_3` | 右臂关节 3（rad） |
| 11 | `right_joint_4` | 右臂关节 4（rad） |
| 12 | `right_joint_5` | 右臂关节 5（rad） |
| 13 | `right_joint_6` | 右臂关节 6（rad） |

> 注：本示例的 `info.json` 中 state/action 名为 `left_joint_0..6, right_joint_0..6`，共 14 维，**没有单独的 gripper 维**——夹爪开合通常并入 `joint_6` 或由 `joint_5` 的末端表示。与 W2 的 14 维（6 关节 + 1 夹爪 ×2）含义不同，使用时需注意区分。

### 2.3 相机标定（每个相机 3 个矩阵）

对三个相机 `cam_high`、`cam_left_wrist`、`cam_right_wrist`，各提供：

| 字段 | 形状 | 含义 |
|------|------|------|
| `calibration.<cam>.intrinsic_matrix` | 3×3 | OpenCV 针孔内参矩阵（像素坐标） |
| `calibration.<cam>.extrinsic_matrix` | 4×4 | world→camera 变换矩阵（OpenCV 相机坐标：+x 右、+y 下、+z 前） |
| `calibration.<cam>.camera_pose_matrix` | 4×4 | camera→world 变换矩阵（即外参的逆，相机在世系中的位姿） |

这些矩阵可用于将深度图反投影到世界坐标系，或在 Rerun 中重建对齐的相机视窗。

---

## 3. W2（双臂 Piper，含速度/力，真机）

- **路径**：`assets/example/W2`
- **机器人**：`agilex_piper_bimanual`（双臂 Piper，真机双臂）
- **FPS**：30
- **示例任务**：`remove the paper balls and blocks from the desktop, put them in the basket, and then use a cloth to wipe off the cola stains.`
- **示例长度**：1 个 episode，共 1034 帧

### 3.1 视觉（仅 RGB，无深度/标定）

| 字段 | 类型 | 分辨率 | 说明 |
|------|------|--------|------|
| `observation.images.cam_high` | video (av1) | 480×640×3 | 高位第三视角 RGB |
| `observation.images.cam_left_wrist` | video (av1) | 480×640×3 | 左腕相机 RGB |
| `observation.images.cam_right_wrist` | video (av1) | 480×640×3 | 右腕相机 RGB |

### 3.2 `observation.state` / `action`（14 维，float32）

采用**左右臂 6 关节 + 夹爪**的表示，state 与 action 维度和命名一致：

| 索引 | 名称 | 含义 |
|------|------|------|
| 0 | `left_joint_1` | 左臂关节 1（rad） |
| 1 | `left_joint_2` | 左臂关节 2（rad） |
| 2 | `left_joint_3` | 左臂关节 3（rad） |
| 3 | `left_joint_4` | 左臂关节 4（rad） |
| 4 | `left_joint_5` | 左臂关节 5（rad） |
| 5 | `left_joint_6` | 左臂关节 6（rad） |
| 6 | `left_gripper` | 左夹爪开合度（归一化，0=闭合，1=张开） |
| 7 | `right_joint_1` | 右臂关节 1（rad） |
| 8 | `right_joint_2` | 右臂关节 2（rad） |
| 9 | `right_joint_3` | 右臂关节 3（rad） |
| 10 | `right_joint_4` | 右臂关节 4（rad） |
| 11 | `right_joint_5` | 右臂关节 5（rad） |
| 12 | `right_joint_6` | 右臂关节 6（rad） |
| 13 | `right_gripper` | 右夹爪开合度（归一化，0=闭合，1=张开） |

### 3.3 `observation.velocity`（14 维，float32）

与 state 同维度同命名，表示每个关节的**角速度**（夹爪位为夹爪速度），单位一般为 rad/s。

### 3.4 `observation.effort`（14 维，float32）

与 state 同维度同命名，表示每个关节的**力矩/电流**反馈（夹爪位为夹爪受力），单位视驱动器而定（通常为 Nm 或归一化电流）。

> W2 是三个示例中唯一同时提供 **位置 + 速度 + 力** 三种模态的数据集，适合用于阻抗/力控相关的研究与可视化。

---

## 4. 三个数据集对比速查

| 维度 | LIBERO | RoboTwin2 | W2 |
|------|--------|-----------|-----|
| 机器人 | franka（单臂） | unified_robot（双臂） | agilex_piper_bimanual（双臂真机） |
| FPS | 20 | 15 | 30 |
| state 维度 | 8（末端位姿 + 双夹爪） | 14（左右 7 关节） | 14（左右 6 关节 + 夹爪） |
| action 维度 | 7（末端位姿 + 夹爪） | 14（同 state） | 14（同 state） |
| 速度/力 | 无 | 无 | 有（velocity + effort） |
| RGB 相机 | 第三视角 + 腕部 | 第三视角 + 左右腕 | 第三视角 + 左右腕 |
| 深度图 | 无 | 有（3 个相机） | 无 |
| 相机标定 | 无 | 有（内参 + 外参 + 位姿） | 无 |
| 示例帧数 | 214 | 180 | 1034 |
| 视频编码 | av1 | h264（RGB）/ hevc gray12le（深度） | av1 |

---

## 5. 使用提示

- **字段命名差异**：RoboTwin2 的 14 维是 `joint_0..6 × 2`（无独立夹爪维），W2 的 14 维是 `joint_1..6 + gripper × 2`。同样叫 `observation.state`，含义随机器人形态不同而不同，使用前务必对照 `info.json` 的 `names`。
- **深度图解码**：RoboTwin2 的深度以 `gray12le` + 对数变换存储，读取后需按 `info.json` 中的 `shift`、`use_log`、`depth_min/max` 反推真实米值；0 值为无效像素。
- **标定矩阵的坐标系**：所有外参均遵循 OpenCV 约定（+x 右、+y 下、+z 前），`extrinsic_matrix` 为 world→camera，`camera_pose_matrix` 为 camera→world，两者互为逆。
- **可视化**：本仓库的 `visualize_lerobot_rerun.py` 会自动读取上述字段并在 Rerun 中按 `episode_time` 时间轴同步回放；LIBERO 因 state 为末端位姿，仅作信号 + 视频展示，不做 URDF 关节回放。
