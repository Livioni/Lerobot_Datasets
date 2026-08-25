# LeRobot Dataset Visualizer with Rerun

English | [中文](README_CN.md)

A local visualization and media toolkit for [LeRobot](https://github.com/huggingface/lerobot) datasets. It uses [Rerun](https://rerun.io/) to synchronize multi-camera video, numerical signals, robot kinematics, end-effector traces, and calibrated camera views on one timeline.

The repository includes three compact LeRobot v3.0 examples under [`assets/example`](assets/example), so you can try the visualizer without downloading a full benchmark dataset.

![Rerun visualization examples](assets/images/cover.gif)

## Features

- Read local LeRobot v3.0 and v2.1 datasets.
- Synchronize RGB/depth video and numerical features on the `episode_time` timeline.
- Automatically replay Piper/ALOHA and RoboTwin/Arx5 joint states with URDF models.
- Inspect unsupported embodiments, such as the included LIBERO/Franka example, as video and signals only.
- Reconstruct a fixed calibrated camera view from an OpenCV extrinsic matrix.
- Save a portable Rerun `.rrd` recording instead of opening the viewer.
- Export episodes as camera mosaics or independent MP4 files.
- Convert consolidated LeRobot v3.0 datasets to the per-episode v2.1 layout.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/Livioni/Lerobot_Datasets.git
cd Lerobot_Datasets
```

### 2. Install the environment

The commands below create the tested Conda environment. FFmpeg is used by the MP4 exporter and the v3.0-to-v2.1 converter.

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

The scripts do not require the `lerobot` Python package. They read the LeRobot files directly with PyArrow.

### 3. Download the robot models

Download [`HarrisonPENG/Embodiments`](https://huggingface.co/datasets/HarrisonPENG/Embodiments) from Hugging Face directly into `embodiments/`:

```bash
hf download HarrisonPENG/Embodiments \
  --repo-type dataset \
  --local-dir embodiments
```

The resulting layout should contain:

```text
embodiments/
├── aloha-agilex/
├── aloha_new_description/
├── realsense2_description/
└── tracer2_description/
```

If the repository is public, no authentication is required. If access requires a token, run `hf auth login` first. Robot assets are optional when using `--no-robot`.

### 4. Run an included example

#### LIBERO: Franka video and signals

[`assets/example/LIBERO`](assets/example/LIBERO) contains episode 0 with 214 frames at 20 FPS and two RGB cameras.

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/LIBERO \
  --episode 0 \
  --no-robot
```

#### RoboTwin: complete Arx5 robot and RGB-D cameras

[`assets/example/RoboTwin2`](assets/example/RoboTwin2) contains episode 0 with 180 frames at 15 FPS, three RGB streams, three depth streams, and a 14-dimensional dual-arm state. The visualizer automatically selects the complete Arx5 model.

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/RoboTwin2 \
  --episode 0
```

#### Agilex-Aloha: Piper bimanual robot

[`assets/example/W2`](assets/example/W2) contains episode 0 with 1,034 frames at 30 FPS, three RGB cameras, and 14-dimensional position, velocity, effort, and action signals. The visualizer automatically replays the two front Piper arms and grippers.

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/W2 \
  --episode 0
```

| Example | Robot type | Frames | Cameras | Robot replay |
| --- | --- | ---: | --- | --- |
| LIBERO | `franka` | 214 | 2 RGB | Video/signals only |
| RoboTwin2 | `unified_robot` | 180 | 3 RGB + 3 depth | Complete Arx5 |
| W2 | `agilex_piper_bimanual` | 1,034 | 3 RGB | Piper front arms |

## Visualizing your own LeRobot v3.0 data

Pass either a dataset directory or a directory whose immediate children are datasets:

```bash
python visualize_lerobot_rerun.py \
  --root /path/to/datasets \
  --dataset dataset_directory_name \
  --episode 0
```

If `--dataset` is omitted, the script selects the only detected dataset or opens an interactive selection menu. To inspect discovery without launching Rerun:

```bash
python visualize_lerobot_rerun.py \
  --root assets/example \
  --list-datasets
```

## Robot replay and embodiment models

The v3.0 visualizer currently recognizes two state profiles:

- `agilex_piper_bimanual`: expects six joints and one gripper value for each side. It uses `embodiments/aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf` and depends on the meshes in `tracer2_description`.
- `unified_robot`: expects the RoboTwin/Arx5 14-value state layout. It uses `embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf` and displays the complete chassis, wheels, cameras, and four arms.

Other robot types remain fully usable for videos and numerical signals with `--no-robot`.

The visualizer prepends the `embodiments/` package directory to `ROS_PACKAGE_PATH`, allowing Rerun to resolve `package://` mesh paths. It never edits a source URDF. Instead, it creates a temporary visualization copy and removes collision geometry from that copy.

`left_gripper` and `right_gripper` represent the total distance between the two fingers in meters. During replay, the value is divided equally between the fingers for symmetric opening without changing their geometry.

`realsense2_description` is used only by optional D435/D415-style URDF variants; the default Piper Dabai model and the default Arx5 model do not require it.

## Calibrated camera replay

Use [`caliberations/w2_demo.yaml`](caliberations/w2_demo.yaml) to inspect the W2 example from a fixed main-camera viewpoint:

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/W2 \
  --episode 0 \
  --camera-calibration caliberations/w2_demo.yaml \
  --camera-resolution 480 640
```

The YAML `extrinsic` is interpreted as:

```text
p_camera = T_camera_base @ p_base
```

It maps points from the robot `footprint` frame into the OpenCV camera frame (`+X` right, `+Y` down, `+Z` forward). The script inverts this transform to recover the camera pose in the base frame. Rerun opens the `Main camera replay (640x480)` tab first and keeps the free-view `Robot replay` tab for comparison.

`--camera-resolution` uses `HEIGHT WIDTH` order. If a calibration file contains exactly one camera, `--camera-feature` is inferred automatically.

## Common options

Use a custom URDF instead of the automatically selected model:

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/W2 \
  --episode 0 \
  --urdf embodiments/aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf
```

Keep only video and numerical views:

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/W2 \
  --episode 0 \
  --no-robot
```

Skip video decoding and save a Rerun recording:

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/W2 \
  --episode 0 \
  --no-video \
  --output /tmp/w2_episode_0.rrd

rerun /tmp/w2_episode_0.rrd
```

URDF meshes are embedded in `.rrd` recordings. A recording with robot geometry can therefore exceed 100 MB even when videos are disabled. Add `--no-robot` when file size matters.

Run `python visualize_lerobot_rerun.py --help` for the complete option list.

## LeRobot v2.1 visualization

The v2.1 entry point reuses the same Rerun renderer while reading JSONL episode metadata and per-episode parquet/video files:

```bash
python visualize_lerobot_rerun_v21.py \
  --root lerobot_datasets_v2.1 \
  --dataset table_clean \
  --episode 0
```

Its robot, camera-calibration, `.rrd`, and dataset-discovery options match the v3.0 entry point. Run `python visualize_lerobot_rerun_v21.py --help` for details.

## Exporting episodes to MP4

Export the W2 example as a camera mosaic:

```bash
python export_lerobot_episode_mp4.py \
  --root assets/example/W2 \
  --episodes 0
```

Outputs are written under `visualization/<dataset>/`. The exporter skips existing files, so interrupted jobs can be resumed safely.

Useful options include:

- `--overwrite` to replace existing MP4 files.
- `--jobs 2` to process two episodes concurrently.
- `--mode cameras` to export one file per camera.
- `--mode both` to export the mosaic and independent camera files.
- `--layout grid` to use an equal-size camera grid.
- `--dry-run` to inspect planned outputs without encoding.

Run `python export_lerobot_episode_mp4.py --help` for all encoding and camera-selection options.

## Converting LeRobot v3.0 to v2.1

The converter splits consolidated v3.0 parquet/video files into independent v2.1 episode files, rewrites metadata, and validates the output:

```bash
python convert_lerobot_v30_to_v21.py \
  --source-root assets/example/W2 \
  --output-root converted_v2.1 \
  --jobs 4
```

Use `--dry-run` to inspect source and destination paths first, repeat `--source-root` to convert several roots, and use `--overwrite` to recreate an existing destination. The same command can be rerun after interruption.

## Uploading datasets to Hugging Face

Preview an upload without authentication or network writes:

```bash
python upload_lerobot_datasets.py \
  --dataset-root /path/to/local/datasets \
  --dry-run
```

To upload, install `huggingface_hub`, authenticate with `hf auth login`, and rerun without `--dry-run`. Use `--repo-id`, `--private`, or `--public` to control the destination. Always review the dataset and third-party asset licenses before publishing.

## Repository layout

```text
.
├── assets/example/                 # Three self-contained LeRobot v3.0 examples
├── caliberations/                  # Camera calibration YAML files
├── embodiments/                    # Downloaded URDF and mesh packages (gitignored)
├── visualize_lerobot_rerun.py      # LeRobot v3.0 Rerun visualizer
├── visualize_lerobot_rerun_v21.py  # LeRobot v2.1 Rerun visualizer
├── export_lerobot_episode_mp4.py   # Per-episode MP4 exporter
├── convert_lerobot_v30_to_v21.py   # v3.0-to-v2.1 converter
└── upload_lerobot_datasets.py      # Hugging Face dataset uploader
```
