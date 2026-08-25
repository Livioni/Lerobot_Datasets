# RoboTwin 相机、world 与 URDF `footprint` 坐标系说明

本文说明 RoboTwin RGB-D 相机采集使用的坐标系、LeRobot 示例中的标定矩阵含义，以及为什么文档中的 `reference_frame` 虽然与 URDF `footprint` 一致，点云仍不能直接作为 `footprint` 坐标使用。

## 结论

RoboTwin 文档中的机器人 `reference_frame` 确实就是 URDF 根坐标系 `footprint`。未对齐的原因不是 URDF 定义错误，而是：

1. RoboTwin 将整个 URDF 根节点通过 `robot_pose` 放置到 SAPIEN `world` 中；
2. 相机标定记录的是相机相对于 SAPIEN `world` 的位姿；
3. 因此必须额外应用 `robot_pose` 的逆矩阵，才能把点云从 SAPIEN `world` 转回机器人 `reference_frame/footprint`。

对于本仓库的 RoboTwin2/Arx5 示例，正确变换为：

```text
(x, y, z)_world → (y + 0.65, -x, z)_footprint
```

等价于绕 `+Z` 顺时针旋转 90°，再沿 `footprint +X` 平移 0.65 m。

## 涉及的三个坐标系

为避免混淆，本文使用以下记号：

| 记号 | 坐标系 | 说明 |
|------|--------|------|
| `C_cv` | OpenCV 相机坐标系 | `+X` 向右、`+Y` 向下、`+Z` 向前 |
| `W` | SAPIEN world | 场景、桌面、相机实体所在的仿真世界坐标系 |
| `F` | robot reference/URDF `footprint` | 机器人自身局部基座坐标系，URDF 根 link |

`reference_frame == footprint` 描述的是 `F`。它不表示 `F` 必须和 SAPIEN `W` 使用相同的原点与朝向。

## RoboTwin 如何采集相机数据

RoboTwin 在 `envs/camera/camera.py` 的 `Camera.get_config()` 中读取：

```python
camera_intrinsic_cv = camera.get_intrinsic_matrix()
camera_extrinsic_cv = camera.get_extrinsic_matrix()
camera_model_matrix = camera.get_model_matrix()

return {
    "intrinsic_cv": camera_intrinsic_cv,
    "extrinsic_cv": camera_extrinsic_cv,
    "cam2world_gl": camera_model_matrix,
}
```

这些字段的含义是：

| RoboTwin 字段 | 变换方向 | 相机约定 |
|---------------|----------|----------|
| `intrinsic_cv` | 像素投影内参 | OpenCV |
| `extrinsic_cv` | `W → C_cv` | OpenCV |
| `cam2world_gl` | OpenGL camera → `W` | OpenGL/SAPIEN |

腕部渲染相机直接复制机器人相机 link 的世界位姿：

```python
self.left_camera.entity.set_pose(left_pose)
self.right_camera.entity.set_pose(right_pose)
```

因此腕部相机位姿已经包含机器人根节点 `robot_pose` 对 URDF `footprint` 的变换。

### 深度含义

RoboTwin 从 SAPIEN Position buffer 取得深度：

```python
position = camera.get_picture("Position")
depth = -position[..., 2]
depth_image = (depth * 1000.0).astype(np.float64)
```

保存值是沿光轴的 Z-depth，单位转换为毫米，不是相机中心到空间点的欧氏距离。转换到 LeRobot 后，需要先按数据集量化参数恢复为米，再使用 OpenCV 针孔模型反投影：

```text
X = (u - cx) Z / fx
Y = (v - cy) Z / fy
Z = depth
```

## LeRobot 示例中的标定矩阵

`assets/example/RoboTwin2` 为每个相机提供：

| LeRobot 字段 | 变换方向 |
|--------------|----------|
| `calibration.<camera>.intrinsic_matrix` | OpenCV 针孔内参 |
| `calibration.<camera>.extrinsic_matrix` | `W → C_cv` |
| `calibration.<camera>.camera_pose_matrix` | `C_cv → W` |

后两个矩阵互为逆矩阵：

```text
camera_pose_matrix = inverse(extrinsic_matrix)
```

当前点云实现优先使用 `camera_pose_matrix`；缺失时才对 `extrinsic_matrix` 求逆。两条路径数学上等价。

> 注意：RoboTwin 的中间转换脚本 `envs/utils/pkl2hdf5.py` 会优先把 `cam2world_gl` 写入名为 `extrinsics_matrix` 的字段。这个名称并不保证它是 OpenCV 的 `world → camera` 外参。处理原始 RoboTwin/XPolicyLab 文件时，必须根据来源区分 `cam2world_gl` 与 `extrinsic_cv`，不能只根据字段名判断方向。

## 为什么 `reference_frame` 和 `footprint` 一致仍然需要转换

Arx5 URDF 的根 link 是：

```xml
<link name="footprint" />
```

但 RoboTwin 加载 URDF 后还会执行：

```python
self.left_entity.set_root_pose(self.left_entity_origion_pose)
self.right_entity.set_root_pose(self.right_entity_origion_pose)
```

本仓库使用的 `embodiments/aloha-agilex/config.yml` 指定：

```yaml
robot_pose: [[0, -0.65, 0.0, 0.707, 0, 0, 0.707]]
```

四元数顺序为 `(qw, qx, qy, qz)`，因此该 pose 表示 `footprint → world`：

- 平移 `[0, -0.65, 0]` m；
- 绕 `+Z` 逆时针旋转 90°。

对应齐次矩阵为：

```text
T_W_F =
[[ 0, -1,  0,  0.00],
 [ 1,  0,  0, -0.65],
 [ 0,  0,  1,  0.00],
 [ 0,  0,  0,  1.00]]
```

相机点云在 `W` 中，而 Rerun 的机器人根节点放在 `F` 中，所以需要：

```text
T_F_W = inverse(T_W_F)

T_F_W =
[[ 0,  1,  0, 0.65],
 [-1,  0,  0, 0.00],
 [ 0,  0,  1, 0.00],
 [ 0,  0,  0, 1.00]]
```

完整的像素到机器人 base 变换链为：

```text
p_Ccv = depth · inverse(K) · [u, v, 1]ᵀ
p_W   = T_W_Ccv · p_Ccv
p_F   = T_F_W · p_W
```

合并后：

```text
T_F_Ccv = T_F_W · T_W_Ccv
```

## 与源码和数据的交叉验证

### 主相机

RoboTwin 配置中的 `head_camera` 世界位置为：

```text
[-0.032, -0.45, 1.35]
```

RoboTwin2 第一帧 `cam_high.camera_pose_matrix` 的平移完全相同，证明该矩阵表达的是 `camera → SAPIEN world`，而不是 `camera → footprint`。

应用 `T_F_W` 后：

```text
相机位置（footprint） = [0.20, 0.032, 1.35]
相机前向（footprint） = [0.60, 0.00, -0.80]
```

即主相机位于机器人前部上方，并沿机器人 `+X` 方向向下观察。

### 左右腕部相机

将 `cam_left_wrist`、`cam_right_wrist` 的逐帧标定位置应用 `T_F_W`，再与相同关节状态下 URDF 正向运动学得到的 `left_camera`、`right_camera` link 位置比较：

| 相机 | 位置残差 |
|------|----------|
| 右腕 | 约 `10⁻⁷ m` |
| 左腕 | 约 `1–2 mm` |

这同时验证了旋转方向和平移量。左腕的毫米级差异来自模型/标定细节，不是 world/base 轴向错误。

## 当前 Rerun 实现

`visualize_lerobot_rerun.py` 对 `robot_type=unified_robot` 使用：

```python
R_world_to_base = np.array([
    [ 0,  1, 0],
    [-1,  0, 0],
    [ 0,  0, 1],
])
t_world_to_base = np.array([0.65, 0.0, 0.0])
```

三路点云先数值变换到 `footprint`，再分别以静态单位变换连接到 Rerun 命名 frame `footprint`。因此 Rerun 中点坐标本身已经是 robot base 坐标。

运行示例：

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/RoboTwin2 \
  --episode 0 \
  --point-cloud \
  --point-cloud-stride 2
```

旧 `.rrd` 不会自动获得新的坐标变换，修改坐标逻辑后必须重新生成记录。

## 常见未对齐现象与排查

| 现象 | 优先检查 |
|------|----------|
| 三路点云彼此对齐，但整体和机器人不对齐 | 是否遗漏 `inverse(robot_pose)`；是否打开了旧 RRD |
| 方向正确但整体平移 | 是否只应用旋转而漏掉 base 原点的 0.65 m 平移 |
| 点云翻转、镜像或上下颠倒 | 是否混用了 `cam2world_gl` 与 OpenCV 外参 |
| 三路点云彼此不对齐 | RGB/depth 像素采样、相机内外参帧号、深度逆量化是否同步 |
| 腕部点云随机器人运动但有固定时间滞后 | 图像帧与 `observation.state` 是否相差一帧 |
| Rerun 报 `No transform path ... footprint` | 每个承载 `Points3D` 的实体是否显式连接到 `footprint` frame |

## 相关文件

- [`visualize_lerobot_rerun.py`](../visualize_lerobot_rerun.py)：RGB-D 反投影及 `world → footprint` 变换。
- [`embodiments/aloha-agilex/config.yml`](../embodiments/aloha-agilex/config.yml)：RoboTwin/Arx5 `robot_pose` 和主相机配置。
- [`embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf`](../embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf)：根 link `footprint` 及机器人运动链。
- RoboTwin `envs/camera/camera.py`：相机矩阵、深度和腕部相机位姿采集。
- RoboTwin `envs/robot/robot.py`：URDF 加载和 `robot_pose` 应用。
- RoboTwin `envs/utils/pkl2hdf5.py`：原始相机字段到中间 HDF5 的映射。
