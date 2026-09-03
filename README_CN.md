# 使用 Rerun 可视化 LeRobot 数据集

[English](README.md) | 中文

这是一个面向本地 [LeRobot](https://github.com/huggingface/lerobot) 数据集的可视化与媒体处理工具集。项目使用 [Rerun](https://rerun.io/) 在同一时间轴上同步展示多相机视频、数值信号、机器人运动学、末端轨迹和标定相机视图。

仓库在 [`assets/example`](assets/example) 中提供了三个精简的 LeRobot v3.0 示例，无需下载完整基准数据集即可运行可视化。

![Rerun 可视化示例](assets/images/cover.gif)

## 功能

- 读取本地 LeRobot v3.0 和 v2.1 数据集。
- 在 `episode_time` 时间轴上同步 RGB/深度视频与数值特征。
- 将成对 RGB-D 数据重建为 world/base 坐标系下的同步彩色点云。
- 根据数据自动使用 URDF 回放 Piper/ALOHA 和 RoboTwin/Arx5 关节状态。
- 对暂不支持机器人模型的 embodiment（例如示例中的 LIBERO/Franka）显示视频和数值信号。
- 根据 OpenCV 外参重建固定标定相机视角。
- 将结果保存为可移植的 Rerun `.rrd` 记录，而不启动查看器。
- 将 episode 导出为相机拼图或独立 MP4 文件。
- 将合并存储的 LeRobot v3.0 数据集转换为逐 episode 的 v2.1 布局。

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/Livioni/Lerobot_Datasets.git
cd Lerobot_Datasets
```

### 2. 安装环境

以下命令会创建本项目测试使用的 Conda 环境。RGB-D 点云解码、MP4 导出和 v3.0→v2.1 转换会使用 FFmpeg。

```bash
conda create -n rerun -c conda-forge python=3.10 ffmpeg -y
conda activate rerun

python -m pip install \
  "rerun-sdk==0.35.0" \
  "numpy>=2.0,<3" \
  "pyarrow>=21,<26" \
  "Pillow>=11,<13" \
  "huggingface_hub>=1.0,<2"
```

这些脚本不依赖 `lerobot` Python 包，而是通过 PyArrow 直接读取 LeRobot 文件。

### 3. 下载机器人模型

从 Hugging Face 下载 [`HarrisonPENG/Embodiments`](https://huggingface.co/datasets/HarrisonPENG/Embodiments)，并直接保存到 `embodiments/`：

```bash
hf download HarrisonPENG/Embodiments \
  --local-dir embodiments
```

下载后的目录应包含：

```text
embodiments/
├── aloha-agilex/
├── aloha_new_description/
├── realsense2_description/
└── tracer2_description/
```

如果仓库为公开状态，则无需登录即可下载；如果访问需要令牌，请先执行 `hf auth login`。使用 `--no-robot` 时可以不下载机器人模型。

### 4. 运行内置示例

#### LIBERO：Franka 视频与数值信号

[`assets/example/LIBERO`](assets/example/LIBERO) 包含 episode 0，共 214 帧、20 FPS 和两路 RGB 相机。

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/LIBERO \
  --episode 0 \
  --no-robot
```

#### RoboTwin：完整 Arx5 机器人与 RGB-D 相机

[`assets/example/RoboTwin2`](assets/example/RoboTwin2) 包含 episode 0，共 180 帧、15 FPS、三路 RGB、三路深度视频和 14 维双臂状态。脚本会自动选择完整 Arx5 模型。

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/RoboTwin2 \
  --episode 0
```

加入 `--point-cloud` 后，会在统一的机器人 base 坐标系中重建主相机和两只腕部相机的 RGB-D 点云：

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/RoboTwin2 \
  --episode 0 \
  --point-cloud
```

#### Agilex-Aloha：Piper 双臂机器人

[`assets/example/W2`](assets/example/W2) 包含 episode 0，共 1,034 帧、30 FPS、三路 RGB 相机，以及 14 维位置、速度、力矩和动作信号。脚本会自动回放前方两只 Piper 机械臂和夹爪。

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/W2 \
  --episode 0
```

| 示例 | Robot type | 帧数 | 相机 | 机器人回放 |
| --- | --- | ---: | --- | --- |
| LIBERO | `franka` | 214 | 2 路 RGB | 仅视频/信号 |
| RoboTwin2 | `unified_robot` | 180 | 3 路 RGB + 3 路深度 | 完整 Arx5 |
| W2 | `agilex_piper_bimanual` | 1,034 | 3 路 RGB | Piper 前方双臂 |

## 可视化自己的 LeRobot v3.0 数据

`--root` 可以指向一个数据集，也可以指向直接包含多个数据集子目录的文件夹：

```bash
python visualize_lerobot_rerun.py \
  --root /path/to/datasets \
  --dataset dataset_directory_name \
  --episode 0
```

省略 `--dataset` 时，脚本会自动选择唯一的数据集；如果发现多个数据集，则显示交互式选择菜单。只检查数据集发现结果而不启动 Rerun：

```bash
python visualize_lerobot_rerun.py \
  --root assets/example \
  --list-datasets
```

## Base 坐标系 RGB-D 点云

传入 `--point-cloud` 后，脚本会发现兼容的 LeRobot v3 RGB-D 数据。名为 `observation.images.<camera>_depth` 的深度流会与 `observation.images.<camera>` RGB 流，以及逐帧的 `calibration.<camera>.intrinsic_matrix`、`camera_pose_matrix`（或 `extrinsic_matrix` 的逆矩阵）配对。缺少这些列时，`--camera-calibration` 会为同名相机提供静态 base→camera YAML 标定。各路点云在现有机器人三维视图中共同显示，同时保留在 `robot/scene_point_cloud` 下分别开关的能力。

深度值直接从原始高位深视频解码，并使用 `meta/info.json` 中的量化参数恢复为米。RGB 与深度使用完全相同的采样行列。默认 `--point-cloud-stride 2`，因此 320×240 相机每帧最多产生 19,200 个点：三路相机合计最多 57,600 个点，`RoboTwin2_third_view` 四路相机合计最多 76,800 个点。使用 `--point-cloud-stride 1` 可恢复全分辨率，增大步长则可降低记录体积和查看器负担。

点云最终以机器人 base/URDF `footprint` 为参考系。对于 RoboTwin/Arx5 的 `unified_robot` 数据，数据集 `world` 的 `+Y` 对应 URDF `footprint` 的 `+X`，并且两个原点相差 0.65 m；脚本会自动施加 `world → footprint` 变换 `(x, y, z) → (y + 0.65, -x, z)`，即绕 `+Z` 顺时针旋转 90° 后沿 base `+X` 平移 0.65 m。该变换通过逐帧匹配左右腕部相机标定位置与 URDF 正向运动学位置得到。其他机器人类型暂按 `world` 与 `footprint` 对齐处理。`--no-video --point-cloud` 会隐藏二维相机面板但继续生成点云；`--no-robot --point-cloud` 可以不加载 URDF，仅显示重建场景。

RoboTwin 相机坐标、SAPIEN `world`、机器人 `reference_frame` 与 URDF `footprint` 的完整关系及矩阵推导，参见 [`docs/README_RoboTwin_Camera_Coordinates.md`](docs/README_RoboTwin_Camera_Coordinates.md)。

## 机器人回放与 Embodiment 模型

v3.0 可视化脚本目前识别两类状态配置：

- `agilex_piper_bimanual`：每侧需要六个关节值和一个夹爪值。默认使用 `embodiments/aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf`，并依赖 `tracer2_description` 中的底盘网格。
- `unified_robot`：使用 RoboTwin/Arx5 的 14 维状态布局。默认使用 `embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf`，显示完整底盘、轮组、相机和前后四臂。

其他机器人类型仍可通过 `--no-robot` 完整查看视频和数值信号。

可视化进程会将 `embodiments/` 软件包目录加入 `ROS_PACKAGE_PATH`，使 Rerun 能够解析 `package://` 网格路径。脚本不会修改原始 URDF，而是创建临时可视化副本，并仅从副本中移除碰撞模型。

数据中的 `left_gripper` 和 `right_gripper` 表示两根指爪之间的总开度，单位为米。回放时会把总开度均分到两侧，使指爪保持原始尺寸并对称开合。

`realsense2_description` 只供可选的 D435/D415 等 URDF 版本使用；默认 Piper Dabai 模型和默认 Arx5 模型都不依赖它。

## 标定相机视角回放

使用 [`calibrations/w2_demo.yaml`](calibrations/w2_demo.yaml) 中的标定，从固定主相机视角查看 W2 示例：

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/W2 \
  --episode 0 \
  --camera-calibration calibrations/w2_demo.yaml \
  --camera-resolution 480 640
```

查看 RoboTwin 示例：

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/RoboTwin2 \
  --episode 0 \
  --camera-calibration calibrations/robotwin.yaml \
  --camera-resolution 240 320 \
  --point-cloud
```

含第四路 `cam_third_view` 的 RoboTwin 数据可以使用第三视角标定：

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/RoboTwin2_third_view \
  --episode 0 \
  --camera-calibration calibrations/robotwin_third_view.yaml \
  --camera-resolution 240 320 \
  --point-cloud
```

此时 RGB 与 Depth 面板会自动包含 `cam_third_view` 和 `cam_third_view_depth`，第三视角彩色点云位于 `robot/scene_point_cloud/cam_third_view`。`--camera-calibration` 同时控制唯一的标定回放视角，并在该 RGB-D 流缺少逐帧相机矩阵时提供静态点云标定；数据集自带的逐帧标定仍然优先。

![标定视角回放](assets/images/caliball.gif)

YAML 中的 `extrinsic` 按以下方式解释：

```text
p_camera = T_camera_base @ p_base
```

它会把机器人 `footprint` 坐标系中的点转换到 OpenCV 相机坐标系（`+X` 向右、`+Y` 向下、`+Z` 向前）。脚本会求逆得到相机在 base 坐标系中的位姿。Rerun 使用相机字段名命名标定回放标签页，同时保留自由视角的 `Robot replay` 标签页用于对照。

`--camera-resolution` 的参数顺序为 `HEIGHT WIDTH`。当标定文件中只有一个相机时，脚本会自动推断 `--camera-feature`。
