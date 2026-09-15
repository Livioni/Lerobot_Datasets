# RoboTwin 第三视角本体替换

将原 Aloha/AgileX 的 URDF 投影区域涂黑，再根据已求解的 IK 关节状态渲染新本体。只处理 `images/third_views`，保留原图尺寸和文件名，输出两版 RGB PNG。

## 环境与运行

从仓库根目录运行。当前机器已有可用的 `RoboTwin` conda 环境（Python 3.10、SAPIEN 3.0.0b1），无需安装其他渲染器，也不需要启动查看器。离屏渲染需要可工作的 Vulkan/GPU 环境；数据加载、合成和普通单元测试不需要 SAPIEN、Rerun 或 cuRobo。

```bash
conda run --no-capture-output -n RoboTwin python -m rerender \
  4d_datasets/place_dual_shoes/episode_0000092 \
  --ik-dir 4d_datasets/place_dual_shoes/episode_0000092/TCP_prediction_ik/ur5_wsg \
  --mode both
```

新环境可参考 `requirements.txt`，在已配置 GPU/Vulkan 的 Python 3.10 环境执行 `python -m pip install -r rerender/requirements.txt`。现有环境优先直接使用。

### 参数

| 参数 | 含义 |
|---|---|
| `episode` | 单个 `robotwin_4d_v1` episode 目录 |
| `--ik-dir` | 必填；包含 `metadata.json`、成功 `robot_state.npy` 的目标本体目录 |
| `--mode` | `no_depth`、`with_depth`、`both`；默认 `both` |
| `--frames 0,120,255` | 仅处理指定的零起始帧号；默认全部，去重后按帧号排序 |
| `--depth-tolerance-mm 5` | 遮挡深度比较容差，默认 5 毫米，可设为 0 |
| `--embodiments-root` | 目标 URDF 根目录；默认仓库的 `embodiments/RobotTwin_embodiments` |
| `--overwrite` | 覆盖选中帧的已有输出；默认发现同名文件即报错 |

所有相对路径按当前工作目录解析。示例输出已经生成，再执行相同命令需要加 `--overwrite`。中断后也可使用它重新生成选中帧；每张 PNG 写完后才替换同名文件。

ARX-X5 抽帧示例：

```bash
conda run --no-capture-output -n RoboTwin python -m rerender \
  4d_datasets/place_dual_shoes/episode_0000092 \
  --ik-dir 4d_datasets/place_dual_shoes/episode_0000092/TCP_prediction_ik/arx_x5 \
  --frames 0,120,255 --mode both --overwrite
```

## 输入、输出与兼容性

输入包括 episode 的 `metadata.json`、`robot_state.npy`、第三视角 RGB 和相机内外参；`with_depth`、`both` 还读取同名 depth PNG。`no_depth` 不读取或要求 depth 文件。

```text
<episode>/rerender_images/third_views/
├── ur5_wsg/
│   ├── no_depth/000000.png ... 000255.png
│   └── with_depth/000000.png ... 000255.png
└── arx_x5/
    ├── no_depth/...
    └── with_depth/...
```

每个输出均为原尺寸的 RGB uint8 PNG，未参与擦除或绘制的像素与原图完全一致。源图像、深度、状态、URDF 和其他视角保持原样。新增工具代码、测试和本说明均位于 `rerender/`；运行中只在临时目录创建适配后的 URDF，不修改模型资源。

- 源本体固定为 `embodiments/aloha-agilex/config.yml` 指向的完整 URDF，包括前后机械臂、夹爪、支架、相机和底盘。14 维状态按左臂 6 关节＋夹爪、右臂 6 关节＋夹爪解释；未记录的活动关节为零。
- 支持仓库的 `robotwin_closed_tcp_ik_v2`、`robotwin_closed_tcp_trajectory_v3` 元数据：UR5-WSG、ARX-X5、Franka-Panda、Piper 和 Aloha 的关节数、夹爪及基座放置按元数据解析。
- 新本体直接采用每臂的 `world_from_root`，不再次叠加双臂间距。单臂 URDF 创建左右两个实例；共享根坐标的双臂 URDF 只创建一份，并让实测/预测活动关节覆盖另一侧求解器的锁定值。
- 夹爪按 `gripper_scale` 与 mimic 映射计算，并按 URDF 限位夹紧。关节变换使用 URDF 原始坐标定义。
- 校验帧数、帧名、状态列、有限值、关节限位、目标 URDF 哈希和相机外参哈希。可用时还核对本 episode 的预测文件哈希，防止静态相机相同但轨迹来自另一段数据。
- v2 元数据未提供整体 `status` 时，还需要同目录 `diagnostics.json` 证明每帧双臂均成功。
- 只读取成功轨迹，不读取 `robot_state_candidate.npy`。当前示例的 Franka、Piper 标记为失败，工具会拒绝这些候选；需要先得到成功 IK 输出。

## 渲染与遮挡规则

分别在只含机器人 visual 的场景中渲染旧、新本体。使用 SAPIEN 默认光栅化渲染器、固定环境光和方向光、关闭 MSAA 和投射阴影；机器人内部及左右臂之间仍有正常的深度排序。通过 `set_qpos` 直接设置状态，不推进物理仿真。URDF 重复 limit 标签在临时副本中合并；Piper 沿用已有回放的 `link8.STL` 几何修正。

投影关系：

```text
p_camera = world_to_camera @ world_from_root @ root_from_link @ p_link
```

内参支持 `[3,3]` 和 `[T,3,3]`；外参支持固定/逐帧的 `3×4` 或 `4×4` 矩阵。相机采用 OpenCV 的右、下、前坐标，并显式转换到 SAPIEN 相机实体的前、左、上坐标。原始 depth 是 uint16 毫米光轴 Z-depth，乘以 `0.001` 转为米；机器人深度从 Position buffer 的 `-Z` 取得，同样是光轴深度。

记 `M_old/M_new` 为新旧本体投影，`Z_old/Z_new` 为机器人渲染深度，`D` 为场景深度，`ε` 为容差。

### 无深度版 `no_depth`

1. 将全部 `M_old` 像素设为 `(0,0,0)`。
2. 在全部 `M_new` 像素上绘制新本体颜色。

此版本不考虑与桌面、鞋盒、鞋子等场景物体的遮挡。旧本体投影即便在桌面后面，也会参与擦除。

### 带深度版 `with_depth`

```text
valid = D 有限且 D > 0
erase = M_old & valid & (Z_old <= D + ε)
draw  = M_new & valid & (erase | (Z_new <= D + ε))
```

先将 `erase` 区域设黑，再在 `draw` 区域绘制新本体。

- 场景物体在旧机器人前方：不擦除它。
- 场景物体在新机器人前方：不绘制被遮挡的新本体像素。
- 旧本体被遮挡但新本体更靠前：允许新本体正常显示，两个判断相互独立。
- 已擦除旧本体的像素：旧深度不再阻挡新本体，即使新本体更远也允许绘制。
- 原始深度为 0 或无效值：保留原 RGB，不擦除也不覆盖。

## 误差与结果边界

- 擦除后未被新本体覆盖的位置保持纯黑；不做背景补全。
- 旧机器人后方的真实背景深度不可由当前 RGB-D 恢复，在擦除区域按“不提供遮挡”处理。
- 5 mm 容差用于应对深度毫米量化、网格和状态重放的小偏差。过大可能擦除贴近机器人的前景物体；过小可能留下机器人边缘。可用相同帧号、不同容差重新检查。
- 记录的夹爪值是归一化控制量，抓住物体时的实际指爪接触位移不一定完整记录，因此手指附近可能留下局部残影。轮廓不做额外膨胀，也不扩大擦除到没有几何依据的像素。
- 只替换本体区域，原图阴影、反射和背景光照不重建；新本体使用固定照明，外观不保证与随机化的原始场景照明完全一致。

## 测试

普通测试（深度合成、输入校验、输出保护，无 GPU 依赖）：

```bash
conda run -n RoboTwin python -m unittest discover -s rerender/tests -v
```

可选 GPU 集成测试（需要本地示例 episode 与模型资源）：

```bash
RERENDER_GPU_TESTS=1 conda run --no-capture-output -n RoboTwin \
  python -m unittest discover -s rerender/tests -v
```

集成测试对照独立 NumPy FK 检查所有 link 的世界位姿，验证相机内外参更新、原本体深度对齐、两版输出和未编辑像素；Franka/Piper 用构造的 home pose 验证几何，不读取失败候选。可通过 `RERENDER_TEST_EPISODE` 指定具有对应成功轨迹的其他测试 episode。
