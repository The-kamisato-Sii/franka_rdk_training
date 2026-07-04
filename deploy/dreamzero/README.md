# DreamZero Franka Dual Server

This server is the deployment-side counterpart of:

`examples/sft/dreamzero/dreamzero_real_world_sft/run_real_world_franka_dual_dreamzero_sft_multi_node.sh`

The training config uses `real_world_joint` with:

- three camera views tiled by the dataset transform: `middle_zed` on top, `left_camera` bottom-left, `right_camera` bottom-right;
- tiled image size `256x256`;
- `macro_stride=48`, `video_in_chunk_offsets=[0,6,12,18,24,30,36,42]`;
- `max_state_dim=64`, `max_action_dim=32`;
- raw Franka dual state/action statistics on the first 16 dimensions.

## Boundary

The LeRobot client does all robot-side preprocessing:

- read `norm_stats.json`;
- normalize raw 16D state with q01/q99;
- pad normalized state to 64D and send `state_mask`;
- tile multi-view RGB images into the training layout;
- send history/rollout frames;
- unnormalize the returned normalized actions with the same global action q01/q99;
- send the resulting 16D GELLO-format joint action to the robot.

The server only consumes `payload["dreamzero"]` and runs the model in normalized model space. It does not normalize state, unnormalize action, tile camera views, or fall back to raw payloads.

Each request saves artifacts under `OUTPUT_ROOT/dreamzero_<timestamp>_<index>/`:

- `gt_video.mp4`: the tiled client video frames passed to the model;
- `pred_video.mp4`: decoded predicted video when the checkpoint returns video latents;
- `normalized_actions.npy`: normalized `[48, 16]` actions returned to the client;
- `metadata.json`: request and shape metadata.

## Start

Fill `MODEL_PATH` with the converted DreamZero checkpoint path.

```bash
cd /inspire/hdd/project/robot-body/linbokai-CZXS24250037/RLinf

MODEL_PATH= \
NPROC_PER_NODE=1 \
PORT=8000 \
bash deploy/dreamzero/run_franka_dual_dreamzero_server.sh
```

Use more GPUs by setting `CUDA_VISIBLE_DEVICES` and `NPROC_PER_NODE`.
