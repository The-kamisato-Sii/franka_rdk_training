# Real-World Joint Training Data Paths

This note documents the two real-world joint SFT training paths currently used in RLinf:

1. OpenPI pi0.5 SFT on `franka_dual`
2. DreamZero real_world_joint SFT, including motion / SAM scene-flow variants

The two paths are intentionally separate. OpenPI uses its own LeRobot v3 reader under
`rlinf/data/datasets/openpi/`; DreamZero continues to use the DreamZero real_world_joint
reader under `rlinf/data/datasets/dreamzero/`.

## OpenPI pi0.5: franka_dual

Entry script:

```bash
bash examples/sft/run_real_world_franka_dual_openpi_pi05_sft.sh
```

Main config:

```text
examples/sft/config/real_world_franka_dual_openpi_pi05_sft.yaml
```

Important config fields:

```yaml
actor:
  model:
    model_type: openpi
    openpi:
      config_name: pi05_real_world_joint

data:
  dataset_type: openpi_real_world_joint
  train_data_paths: /inspire/qb-ilm2/project/robot-body/public/hairuoliu/franka_dual
  openpi_real_world_joint_require_lerobot_v3: true
  openpi_real_world_joint_num_frames: 1
  openpi_real_world_joint_action_horizon: 48
  openpi_real_world_joint_use_state: true
```

Call chain:

```text
examples/sft/run_real_world_franka_dual_openpi_pi05_sft.sh
  -> examples/sft/train_vla_sft.py
  -> rlinf/runners/sft_runner.py
  -> rlinf/workers/sft/fsdp_vla_sft_worker.py
  -> build_openpi_real_world_joint_sft_dataloader(...)
  -> rlinf/data/datasets/openpi/real_world_joint.py
```

OpenPI data reader:

```text
rlinf/data/datasets/openpi/real_world_joint.py
```

This reader is independent from DreamZero. It does not inherit from
`DreamZeroLeRobotDataset` and does not use `rlinf/data/datasets/dreamzero/real_world_joint.py`.

Expected dataset root:

```text
/inspire/qb-ilm2/project/robot-body/public/hairuoliu/franka_dual
```

Expected LeRobot v3 session layout:

```text
franka_dual/<task>/<session>/
  meta/info.json
  meta/stats.json
  meta/episodes/chunk-000/file-000.parquet
  data/chunk-000/file-000.parquet
  videos/observation.images.middle_zed/chunk-000/file-000.mp4
  videos/observation.images.left_camera/chunk-000/file-000.mp4
  videos/observation.images.right_camera/chunk-000/file-000.mp4
```

Per-sample logic:

```text
global sample index
  -> episode_index + frame_in_episode
  -> read observation.state from data parquet
  -> read action window of length 48 from data parquet
  -> decode one frame from each video using:
       videos/<camera>/from_timestamp + frame_in_episode / fps
  -> prompt comes from meta/episodes tasks
```

Camera mapping into OpenPI:

```text
observation.images.middle_zed   -> base_0_rgb
observation.images.left_camera  -> left_wrist_0_rgb
observation.images.right_camera -> right_wrist_0_rgb
```

Raw dimensions from the current `franka_dual` data:

```text
state:  16
action: 16
```

The OpenPI transform chain pads these to the pi0.5 model action dimension:

```text
raw sample
  -> repack_transforms
  -> RealWorldJointInputs
  -> Normalize
  -> model_transforms
  -> PadStatesAndActions(32)
  -> Observation + actions
```

The training worker then calls:

```python
self.model(
    forward_type=ForwardType.SFT,
    data={"observation": observation, "actions": actions},
)
```

## DreamZero baseline: franka_dual

Entry script:

```bash
bash examples/sft/dreamzero/dreamzero_real_world_sft/run_real_world_franka_dual_dreamzero_sft_multi_node.sh
```

Main config:

```text
examples/sft/config/real_world_franka_dual_dreamzero_sft.yaml
```

Important config fields:

```yaml
actor:
  model:
    model_type: dreamzero
    embodiment_tag: real_world_franka_dual
    action_horizon: 48
    action_head_cfg:
      config:
        num_frames: 33
        num_action_per_block: 48
        use_motion_modality: false

data:
  dataset_type: real_world_joint
  train_data_paths: /inspire/qb-ilm2/project/robot-body/public/hairuoliu/franka_dual
  real_world_joint_tags: [real_world_franka_dual]
  sampling_mode: multi_anchor
  macro_stride: 48
  video_in_chunk_offsets: [0, 6, 12, 18, 24, 30, 36, 42]
  scene_flow_training: false
  use_sam_scene_flow: false
```

This is the standard DreamZero path without motion supervision. It uses the
same `real_world_joint` dataloader family as the motion run, but
`use_motion_modality: false` keeps motion / SAM scene-flow files out of the
sample path.

For LeRobot v3 `franka_dual`, video decoding follows the v3 episode metadata:

```text
videos/<camera>/from_timestamp + frame_in_episode / fps
```

The motion training path still uses `frame_index` when motion is enabled so it
can align action rows to full-video motion files.

Optional metadata preparation:

```bash
python toolkits/lerobot/prepare_franka_dual_dreamzero_meta.py \
  --root /inspire/qb-ilm2/project/robot-body/public/hairuoliu/franka_dual
```

The preparation script writes missing `meta/modality.json` and
`meta/embodiment.json` sidecars for each `franka_dual/<task>/<session>` and does
not overwrite existing files unless `--overwrite-modality` or
`--overwrite-embodiment` is passed.

## DreamZero real_world_joint

Typical entry script:

```bash
bash examples/sft/run_real_world_joint_dreamzero_motion_sam_scene_flow_sft_multi_node.sh
```

Main config:

```text
examples/sft/config/real_world_joint_sft_dreamzero_motion_sam_scene_flow_5b.yaml
```

Important config fields:

```yaml
actor:
  model:
    model_type: dreamzero

data:
  dataset_type: real_world_joint
  train_data_paths: /inspire/qb-ilm2/project/robot-body/public/real_world_data_dreamzero_motion_v2
  use_sam_scene_flow: true
  scene_flow_training: true
```

Call chain:

```text
examples/sft/run_real_world_joint_dreamzero_motion_sam_scene_flow_sft_multi_node.sh
  -> examples/sft/train_vla_sft.py
  -> rlinf/runners/sft_runner.py
  -> rlinf/workers/sft/fsdp_vla_sft_worker.py
  -> build_real_world_joint_sft_dataloader(...)
  -> rlinf/data/datasets/dreamzero/real_world_joint.py
  -> RealWorldJointLeRobotDataset
  -> _RealWorldJointMotionMixin
  -> DreamZeroLeRobotDataset
```

DreamZero reader files:

```text
rlinf/data/datasets/dreamzero/real_world_joint.py
rlinf/data/datasets/dreamzero/dreamzero.py
```

This path supports both v2-style and v3-style LeRobot storage.

### v2 / v2.1 branch

If a dataset contains:

```text
meta/episodes.jsonl
```

DreamZero uses the v2-compatible branch. This branch resolves episodes from
`episodes.jsonl` and local data/video paths.

### v3 branch

If a dataset does not contain `meta/episodes.jsonl` but does contain:

```text
meta/episodes/chunk-*/file-*.parquet
```

DreamZero uses the v3 metadata branch in `DreamZeroLeRobotDataset`.

The v3 branch reads:

```text
episode_index
length
data/chunk_index
data/file_index
dataset_from_index
dataset_to_index
videos/<camera>/chunk_index
videos/<camera>/file_index
videos/<camera>/from_timestamp
videos/<camera>/to_timestamp
```

This allows multiple episodes to share one parquet shard and one mp4 shard. Video decoding uses the episode timestamp offset:

```text
videos/<camera>/from_timestamp + frame_in_episode / fps
```

This is necessary for official LeRobot v3 shared video files.

DreamZero also supports the real_world_joint temporal controls used by the motion pipeline, including:

```text
macro_stride
video_in_chunk_offsets
multi_anchor temporal sampling
motion / SAM scene-flow loading through _RealWorldJointMotionMixin
```

## Separation Rule

Do not make OpenPI depend on the DreamZero real_world_joint reader.

Current intended separation:

```text
OpenPI franka_dual:
  rlinf/data/datasets/openpi/real_world_joint.py
  independent LeRobot v3 reader
  no DreamZeroLeRobotDataset inheritance

DreamZero real_world_joint:
  rlinf/data/datasets/dreamzero/dreamzero.py
  rlinf/data/datasets/dreamzero/real_world_joint.py
  v2 and v3 real_world_joint reader with optional motion
```

If OpenPI needs new data behavior, implement it in `rlinf/data/datasets/openpi/`.
If DreamZero needs new real_world_joint behavior, implement it in `rlinf/data/datasets/dreamzero/`.
