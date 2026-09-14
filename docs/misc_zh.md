## 常用选项

使用指定 URDF 覆盖自动选择的模型：

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/W2 \
  --episode 0 \
  --urdf embodiments/aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf
```

只保留视频和数值视图：

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/W2 \
  --episode 0 \
  --no-robot
```

不解码视频并保存 Rerun 记录：

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/W2 \
  --episode 0 \
  --no-video \
  --output /tmp/w2_episode_0.rrd

rerun /tmp/w2_episode_0.rrd
```

URDF 网格会被嵌入 `.rrd` 记录。即使关闭视频，包含机器人几何的记录仍可能超过 100 MB；需要减小文件时可加上 `--no-robot`。

运行 `python visualize_lerobot_rerun.py --help` 可查看完整参数。

## LeRobot v2.1 可视化

v2.1 入口复用同一套 Rerun 渲染器，同时读取 JSONL episode 元数据和逐 episode 的 parquet/视频文件：

```bash
python visualize_lerobot_rerun_v21.py \
  --root lerobot_datasets_v2.1 \
  --dataset table_clean \
  --episode 0
```

机器人、相机标定、`.rrd` 和数据集发现参数与 v3.0 入口一致。运行 `python visualize_lerobot_rerun_v21.py --help` 查看详情。

## 导出 Episode MP4

将 W2 示例导出为相机拼图：

```bash
python export_lerobot_episode_mp4.py \
  --root assets/example/W2 \
  --episodes 0
```

输出位于 `visualization/<dataset>/`。脚本会跳过已有文件，因此中断后可以安全地继续运行。

常用选项：

- `--overwrite`：覆盖已有 MP4。
- `--jobs 2`：同时处理两个 episode。
- `--mode cameras`：为每个相机导出独立文件。
- `--mode both`：同时导出拼图和独立相机视频。
- `--layout grid`：使用等尺寸相机网格。
- `--dry-run`：只查看计划生成的文件，不执行编码。

运行 `python export_lerobot_episode_mp4.py --help` 可查看所有编码和相机选择参数。

## 将 LeRobot v3.0 转换为 v2.1

转换脚本会把 v3.0 中合并的 parquet/视频拆分成独立的 v2.1 episode 文件，重写元数据，并验证输出：

```bash
python convert_lerobot_v30_to_v21.py \
  --source-root assets/example/W2 \
  --output-root converted_v2.1 \
  --jobs 4
```

可以先用 `--dry-run` 检查来源和目标路径；重复传入 `--source-root` 可以转换多个目录；`--overwrite` 会重新创建已有目标。转换中断后可以再次执行相同命令。

## 上传数据集到 Hugging Face

在不登录、也不产生网络写入的情况下预览上传内容：

```bash
python upload_lerobot_datasets.py \
  --dataset-root /path/to/local/datasets \
  --dry-run
```

正式上传前请安装 `huggingface_hub`，执行 `hf auth login`，然后移除 `--dry-run`。可以通过 `--repo-id`、`--private` 或 `--public` 控制目标仓库。发布前请确认数据集和第三方资源的许可证允许相应使用。

## 仓库结构

```text
.
├── assets/example/                 # 三个独立可读的 LeRobot v3.0 示例
├── calibrations/                  # 相机标定 YAML
├── embodiments/                    # 下载的 URDF 和网格包（Git 已忽略）
├── visualize_lerobot_rerun.py      # LeRobot v3.0 Rerun 可视化
├── visualize_lerobot_rerun_v21.py  # LeRobot v2.1 Rerun 可视化
├── export_lerobot_episode_mp4.py   # 逐 episode MP4 导出
├── convert_lerobot_v30_to_v21.py   # v3.0→v2.1 转换
└── upload_lerobot_datasets.py      # Hugging Face 数据集上传
```
