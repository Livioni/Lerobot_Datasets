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

例如，下面的命令可视化 G106 新采集数据的主视角 RGB、深度预览和双臂本体
回放，整体布局与 W2 示例保持一致。主相机默认使用
[`calibrations/w2_demo.yaml`](calibrations/w2_demo.yaml) 中
`observation.images.cam_high` 的内外参；数据集的
`robot_type: agilex_piper_bimanual` 会自动选择 Agilex-Aloha：Piper 本体：

```bash
conda run --no-capture-output -n rerun python visualize_lerobot_rerun.py \
  --root lerobot_datasets_v3.0/G106/stack_blocks_depth_lerobot \
  --episode 0 \
  --depth-feature observation.images.cam_high_depth \
  --camera-calibration calibrations/G106.yaml \
  --camera-feature observation.images.cam_high \
  --camera-resolution 480 640

python visualize_lerobot_rerun.py \
  --root lerobot_datasets_v3.0/G106/throw_battery_into_trash_bin \
  --episode 0 \
  --camera-calibration calibrations/G106.yaml \
  --camera-feature observation.images.cam_high \
  --camera-resolution 480 640
```

`--depth-feature` 让相机网格只增加主视角深度；省略该参数会显示数据中的全部
原生深度图。二维显示采用固定范围的压缩灰度预览，避免将全部 uint16 原始帧
通过 Rerun gRPC 发送而阻塞查看器。检测到匹配的 RGB-D 和标定后会默认生成
主相机彩色点云，`--point-cloud-stride` 默认值为 `2`；点云计算仍使用原始毫米
深度，并且只会反投影具有完整内外参的相机。可用 `--no-point-cloud` 显式关闭。

省略 `--dataset` 时，脚本会自动选择唯一的数据集；如果发现多个数据集，则显示交互式选择菜单。只检查数据集发现结果而不启动 Rerun：

```bash
python visualize_lerobot_rerun.py \
  --root assets/example \
  --list-datasets
```

## DROID 双第三人称 RGB-D 与 Franka 回放

[`visualize_droid_rerun.py`](visualize_droid_rerun.py) 用于已经解包为
`images/`、`depths/`、`intrinsic/`、`extrinsic/`、`observations/` 和
`action/` 的 DROID episode。它在同一时间轴中显示两路第三人称 RGB/深度、
彩色点云、Franka Panda + Robotiq 2F-85 实测关节回放，并在每路原始 RGB 旁边
放置同一标定机位的 robot replay 视图用于对照；同时显示实测/指令 TCP 轨迹，
以及笛卡尔、关节和夹爪的 state/action 曲线：

```bash
conda run --no-capture-output -n rerun python visualize_droid_rerun.py \
  '4d_datasets/droid_episodes/AUTOLab__Fri_Aug_18_11:40:54_2023'
```

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

## 逆运动学求解与可视化

将预测 TCP 转为指定本体的整段关节轨迹，再直接查看机器人、目标与实际 TCP、RGB/点云和误差曲线。支持 Franka Panda、ARX-X5、Piper、UR5-WSG 和 Aloha-Agilex，完整参数与验收条件见 [`robotwin_ik/README.md`](robotwin_ik/README.md)。

```bash
conda run --no-capture-output -n RoboTwin python robotwin_ik/solve_arx_x5.py \
  4d_datasets/beat_block_hammer/episode_0000000

conda run --no-capture-output -n rerun python robotwin_ik/visualize_ik_rerun.py \
  4d_datasets/beat_block_hammer/episode_0000000 \
  --ik-dir 4d_datasets/beat_block_hammer/episode_0000000/TCP_prediction_ik/arx_x5
```

ur5-wsg
```bash
conda run --no-capture-output -n RoboTwin python robotwin_ik/solve_ur5_wsg.py \
  4d_datasets/place_dual_shoes/episode_0000092

conda run --no-capture-output -n rerun python robotwin_ik/visualize_ik_rerun.py \
  4d_datasets/place_dual_shoes/episode_0000092 \
  --ik-dir 4d_datasets/place_dual_shoes/episode_0000092/TCP_prediction_ik/ur5_wsg
```

无显示环境时自动启动 Web 查看器。失败结果需加 `--show-candidate` 查看候选；覆盖已有求解结果需加 `--overwrite`。
