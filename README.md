# LeRobot Dataset Visualizer with Rerun

English | [中文](README_CN.md)

A local visualization and media toolkit for [LeRobot](https://github.com/huggingface/lerobot) datasets. It uses [Rerun](https://rerun.io/) to synchronize multi-camera video, numerical signals, robot kinematics, end-effector traces, and calibrated camera views on one timeline.

The repository includes three compact LeRobot v3.0 examples under [`assets/example`](assets/example), so you can try the visualizer without downloading a full benchmark dataset.

![Rerun visualization examples](assets/images/cover.gif)

## Features

- Read local LeRobot v3.0 and v2.1 datasets.
- Synchronize RGB/depth video and numerical features on the `episode_time` timeline.
- Reconstruct synchronized RGB-colored point clouds from paired depth streams in the dataset world/base frame.
- Automatically replay Piper/ALOHA and RoboTwin/Arx5 joint states with URDF models.
- Inspect unsupported embodiments, such as the included LIBERO/Franka example, as video and signals only.
- Reconstruct a fixed calibrated camera view from an OpenCV extrinsic matrix.
- Export episodes as camera mosaics or independent MP4 files.
- Convert consolidated LeRobot v3.0 datasets to the per-episode v2.1 layout.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/Livioni/Lerobot_Datasets.git
cd Lerobot_Datasets
```

### 2. Install the environment

The commands below create the tested Conda environment. FFmpeg is used by RGB-D point-cloud decoding, the MP4 exporter, and the v3.0-to-v2.1 converter.

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

Add `--point-cloud` to reconstruct the main and two wrist RGB-D streams in the shared robot base frame:

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/RoboTwin2 \
  --episode 0 \
  --point-cloud
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

For example, this command shows the main-camera RGB, depth preview, and
bimanual replay for the newly captured G106 dataset while retaining the W2
example layout. It uses [`calibrations/w2_demo.yaml`](calibrations/w2_demo.yaml) for
the `observation.images.cam_high` intrinsics/extrinsics, while the dataset's
`robot_type: agilex_piper_bimanual` automatically selects the Agilex-Aloha
Piper embodiment:

```bash
conda run --no-capture-output -n rerun python visualize_lerobot_rerun.py \
  --root lerobot_datasets_v3.0/G106/stack_blocks_depth_lerobot \
  --episode 0 \
  --depth-feature observation.images.cam_high_depth \
  --camera-calibration calibrations/w2_demo.yaml \
  --camera-feature observation.images.cam_high \
  --camera-resolution 480 640
```

`--depth-feature` adds only the main depth stream to the camera grid; omit it to
show all native depth images in the dataset. The 2D view is a fixed-range,
compressed grayscale preview, avoiding the Rerun gRPC congestion caused by
sending every uint16 frame uncompressed. A colored point cloud is enabled
automatically when compatible RGB-D and calibration are detected, using the
default `--point-cloud-stride 2`. Reconstruction still uses the original metric
depth, and only cameras with complete calibration are back-projected. Pass
`--no-point-cloud` to disable it explicitly.

If `--dataset` is omitted, the script selects the only detected dataset or opens an interactive selection menu. To inspect discovery without launching Rerun:

```bash
python visualize_lerobot_rerun.py \
  --root assets/example \
  --list-datasets
```

## DROID two-camera RGB-D and Franka replay

[`visualize_droid_rerun.py`](visualize_droid_rerun.py) reads a DROID episode
already extracted into `images/`, `depths/`, `intrinsic/`, `extrinsic/`,
`observations/`, and `action/`. It synchronizes both third-person RGB/depth
views, colored point clouds, a measured Franka Panda + Robotiq 2F-85 joint
replay, two camera-matched robot views paired beside the source RGB images,
measured/commanded TCP traces, and Cartesian, joint, and gripper state/action
plots:

```bash
conda run --no-capture-output -n rerun python visualize_droid_rerun.py \
  '4d_datasets/droid_episodes/AUTOLab__Fri_Aug_18_11:40:54_2023'
```


## Web viewer on headless machines

The LeRobot v3/v2.1, RoboTwin TCP/prediction, DROID, and [`robotwin_ik/visualize_ik_rerun.py`](robotwin_ik/visualize_ik_rerun.py) viewers automatically use Web mode on Linux without an X11/Wayland display. Existing commands work as-is; add `--web` to select Web mode manually.

Open the URL printed in the terminal. For remote use, first run the printed SSH forwarding command on your computer, replacing `<user>@<server>`; it forwards both the Web and data ports. Keep the visualization process running and press `Ctrl+C` to stop. See the [IK guide](robotwin_ik/README.md) for solving and visualizing joint trajectories.

## RGB-D point clouds in the base frame

Compatible LeRobot v3 RGB-D pairs are reconstructed automatically; use `--no-point-cloud` to disable them or `--point-cloud` to require at least one compatible pair. A depth feature named `observation.images.<camera>_depth` is paired with `observation.images.<camera>` and the per-frame `calibration.<camera>.intrinsic_matrix` plus `camera_pose_matrix` (or the inverse of `extrinsic_matrix`). When those columns are absent, `--camera-calibration` supplies static base-to-camera calibration for its matching stream. The resulting camera entities remain individually toggleable under `robot/scene_point_cloud`, while Rerun overlays them in the existing 3D robot view.

Depth may be stored either as an original high-bit-depth video or as a native LeRobot `image` column in parquet. Video depth is decoded with the quantization settings in `meta/info.json`; native image depth retains its physical values (such as uint16) and is converted according to `depth_unit` (`mm` or `m`). Each RGB pixel uses the same sampled row and column as its depth value. The default stride is `2`, so a 320×240 camera contributes at most 19,200 points per frame: 57,600 points for three cameras or 76,800 for the four-camera `RoboTwin2_third_view` example. Use `--point-cloud-stride 1` for full resolution or a larger value for lighter recordings.

Point clouds are ultimately expressed in the robot base/URDF `footprint` frame. For RoboTwin/Arx5 `unified_robot` data, dataset-world `+Y` corresponds to URDF-footprint `+X`, and the two origins are 0.65 m apart. The script therefore maps `world → footprint` as `(x, y, z) → (y + 0.65, -x, z)`: a 90-degree clockwise yaw about `+Z`, followed by a 0.65 m translation along base `+X`. This transform was recovered by matching both calibrated wrist-camera positions against their URDF forward-kinematics positions over the episode. Other robot types currently treat `world` and `footprint` as aligned. `--no-video --point-cloud` omits the 2D camera panels while retaining point-cloud decoding, and `--no-robot --point-cloud` shows the reconstructed scene without loading a URDF.

## Robot replay and embodiment models

The v3.0 visualizer currently recognizes two state profiles:

- `agilex_piper_bimanual`: expects six joints and one gripper value for each side. It uses `embodiments/aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf` and depends on the meshes in `tracer2_description`.
- `unified_robot`: expects the RoboTwin/Arx5 14-value state layout. It uses `embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf` and displays the complete chassis, wheels, cameras, and four arms.

Other robot types remain fully usable for videos and numerical signals with `--no-robot`.

The visualizer prepends the `embodiments/` package directory to `ROS_PACKAGE_PATH`, allowing Rerun to resolve `package://` mesh paths. It never edits a source URDF. Instead, it creates a temporary visualization copy and removes collision geometry from that copy.

`left_gripper` and `right_gripper` represent the total distance between the two fingers in meters. During replay, the value is divided equally between the fingers for symmetric opening without changing their geometry.

`realsense2_description` is used only by optional D435/D415-style URDF variants; the default Piper Dabai model and the default Arx5 model do not require it.

## Calibrated camera replay

Use [`calibrations/w2_demo.yaml`](calibrations/w2_demo.yaml) to inspect the W2 example from a fixed main-camera viewpoint:

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/W2 \
  --episode 0 \
  --camera-calibration calibrations/w2_demo.yaml \
  --camera-resolution 480 640
```

For a RoboTwin export with the additional `cam_third_view` RGB-D stream:

```bash
python visualize_lerobot_rerun.py \
  --root assets/example/RoboTwin2_third_view \
  --episode 0 \
  --camera-calibration calibrations/robotwin_third_view.yaml \
  --camera-resolution 240 320 \
  --point-cloud
```


![CaliberatedCameraReplay](assets/images/caliball.gif)

The YAML `extrinsic` is interpreted as:

```text
p_camera = T_camera_base @ p_base
```

It maps points from the robot `footprint` frame into the OpenCV camera frame (`+X` right, `+Y` down, `+Z` forward). The script inverts this transform to recover the camera pose in the base frame. Rerun names the calibrated replay tab after the camera feature and keeps the free-view `Robot replay` tab for comparison.

`--camera-resolution` uses `HEIGHT WIDTH` order. If a calibration file contains exactly one camera, `--camera-feature` is inferred automatically.
