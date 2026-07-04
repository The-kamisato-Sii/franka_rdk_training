# WMAM Franka Dual Server

This server is the deployment-side counterpart of:

`examples/sft/dreamzero/dreamzero_real_world_sft/run_real_world_franka_dual_wmam_motion_sft_multi_node.sh`

The training config uses the same `real_world_joint` camera/state/action preprocessing as DreamZero, plus motion modality:

- three camera views tiled by the dataset transform: `middle_zed` on top, `left_camera` bottom-left, `right_camera` bottom-right;
- tiled image size `256x256`;
- `macro_stride=48`, `video_in_chunk_offsets=[0,6,12,18,24,30,36,42]`;
- `motion_horizon=8`, `motion_downsample_ratio=6`;
- `max_state_dim=64`, `max_action_dim=32`;
- raw Franka dual state/action statistics on the first 16 dimensions.

## Boundary

The LeRobot `wmam_client` does all robot-side preprocessing:

- read `norm_stats.json`;
- normalize raw 16D state with q01/q99;
- pad normalized state to 64D and send `state_mask`;
- tile multi-view RGB images into the training layout;
- collect the executed 48 raw frames and send the 8 kept tiled frames for WMAM rollout feedback;
- unnormalize the returned normalized actions with the same global action q01/q99;
- send the resulting 16D GELLO-format joint action to the robot.

The server only consumes `payload["wmam"]` and runs the model in normalized model space. It does not normalize state, unnormalize action, tile camera views, or fall back to raw payloads.

Each request saves artifacts under `OUTPUT_ROOT/wmam_<timestamp>_<index>/`:

- `gt_video.mp4`: the tiled client video frames passed to the model;
- `pred_video.mp4`: decoded predicted video when the checkpoint returns video latents;
- `motion_pred.npz`: decoded motion as `point_map`/`scene_flow` when available, otherwise `motion_latent`;
- `normalized_actions.npy`: normalized `[48, 16]` actions returned to the client;
- `metadata.json`: request and shape metadata.

## Start

Fill `MODEL_PATH` with the converted WMAM checkpoint path.

```bash
cd /inspire/hdd/project/robot-body/linbokai-CZXS24250037/RLinf

MODEL_PATH= \
NPROC_PER_NODE=1 \
PORT=8000 \
bash deploy/wmam/run_franka_dual_wmam_server.sh
```

`DECODE_MOTION=true` is the default so `motion_pred.npz` can contain decoded motion when the checkpoint supports it. Use more GPUs by setting `CUDA_VISIBLE_DEVICES` and `NPROC_PER_NODE`.
