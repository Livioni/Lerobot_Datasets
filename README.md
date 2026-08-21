# 使用 Rerun 可视化 LeRobot 3.0


## 可视化一个回合

可视化脚本会在 `episode_time` 时间轴上同步显示相机视频、数值信号和
ALOHA URDF 模型。对于兼容的 `agilex_piper_bimanual` 数据集，脚本会自动使用
`observation.state` 回放前方两只从臂的运动。

```bash
conda run -n rerun python visualize_lerobot_rerun.py \
  --root lerobot_datasets_v3.0/w2_datasets \
  --dataset plug_in_socket_lerobot \
  --episode 0
```

```bash
conda run -n rerun python visualize_lerobot_rerun_v21.py \
  --root lerobot_datasets_v2.1 \
  --dataset table_clean \
  --episode 0
```

默认模型为：
`embodiments/aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf`。
可视化进程会将 `embodiments/` 下同级的 ROS 软件包加入 `ROS_PACKAGE_PATH`，
使 Rerun 能够解析 `package://` 格式的网格模型路径。脚本不会修改原始 URDF，
而是创建一份临时的可视化副本，并从中移除碰撞模型和后方主臂的网格模型。
脚本不会生成或使用 GLB；Rerun 会直接加载原始 DAE 网格及其 Collada 节点变换。
对于 DAE visual，临时副本会移除 URDF 的整网格单色覆盖，让 Rerun 使用 DAE 内嵌的
多材质颜色，从而让装配和外观都与直接打开原始 URDF 的查看器保持一致。
数据中的 `left_gripper` 和 `right_gripper` 表示两根指爪之间的总开度，单位为米；
回放时会将总开度均分到两侧，使指爪保持原尺寸并沿横向对称开合。

使用 `caliberations/w2_demo.yaml` 中的主相机标定，从固定主相机视角回放机器人：

```bash
conda run -n rerun python visualize_lerobot_rerun.py \
  --root lerobot_datasets_v3.0/w2_datasets \
  --dataset table_clean_lerobot \
  --episode 0 \
  --camera-calibration caliberations/w2_demo.yaml \
  --camera-resolution 480 640
```

这里 YAML 的 `extrinsic` 按 `p_camera = T_camera_base @ p_base` 解释，即从机器人
`footprint`（base）坐标系到 OpenCV 相机坐标系（X 向右、Y 向下、Z 向前）的刚体
变换。脚本会自动求逆得到相机在 base 中的位姿。Rerun 首先打开
`Main camera replay (640x480)` 标签页，并保留 `Robot replay` 自由视角供对照。
`--camera-resolution` 的参数顺序是 `HEIGHT WIDTH`；YAML 只有一个相机时无需指定
`--camera-feature`。

常用选项：

```bash
# 使用指定的 URDF 覆盖默认模型。
conda run -n rerun python visualize_lerobot_rerun.py \
  --root lerobot_datasets_v3.0/w2_datasets \
  --dataset table_clean_lerobot \
  --episode 0 \
  --urdf embodiments/aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf

# 保留原有的纯视频与信号视图，不加载机器人模型。
conda run -n rerun python visualize_lerobot_rerun.py \
  --root lerobot_datasets_v3.0/w2_datasets \
  --dataset plug_in_socket_lerobot \
  --no-robot

# 不解码视频，直接导出 Rerun 记录文件。
conda run -n rerun python visualize_lerobot_rerun.py \
  --root lerobot_datasets_v3.0/w2_datasets \
  --dataset plug_in_socket_lerobot \
  --no-video --output /tmp/plug_in_socket_episode_0.rrd
```

URDF 网格模型会嵌入 `.rrd` 记录文件，因此即使不包含视频，启用机器人模型后导出的
文件也可能超过 100 MB。如果希望减小记录文件的体积，可以传入 `--no-robot`。

## 为每个回合导出 MP4

将每个任务数据集中的所有回合导出到
`visualization/<dataset>/episode_XXXXXX.mp4`：

```bash
conda run -n rerun python export_lerobot_episode_mp4.py
```

建议先用少量回合测试：

```bash
conda run -n rerun python export_lerobot_episode_mp4.py \
  --dataset task3_bimanual_plug_Insertion \
  --episodes 0-2
```

脚本会跳过已经存在的 MP4，因此可以中断后继续运行。使用 `--overwrite` 可覆盖已有
文件；使用 `--jobs 2` 可并行处理两个回合；使用 `--mode both` 还会为每个相机
分别导出独立的 MP4。运行时传入 `--help` 可以查看全部选项。
