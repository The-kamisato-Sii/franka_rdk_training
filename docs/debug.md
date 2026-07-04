# Franka Dual VLA Debug Report

Date: 2026-07-04

This note summarizes the current training/deployment mismatch investigation for
the Franka dual-arm real-world VLA stack in RLinf. It is written for future
debugging agents, so it records concrete code entry points, tensor shapes,
temporal index rules, and the observed failure pattern.

Official references:

- OpenPI: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- DreamZero: [dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero)

## Current Symptom

OpenPI, DreamZero, and WMAM can all reach low training loss on the real-world
Franka dual data. However, real deployment / held-out evaluation still shows
poor manipulation behavior: the robot often misses objects, does not move toward
the correct target, or makes unhelpful motions. In OpenPI training we also saw
`train/grad_norm` keep increasing while `train/loss` kept decreasing, which is
suspicious and should be treated as a real signal, not ignored.

At this point the root cause should not be assumed. The most important question
is whether the training objective, data preprocessing, and inference payload are
exactly aligned with the official model logic and with our client/server
deployment logic.

## Shared Training Entry

All three model families enter training through:

- `examples/sft/train_vla_sft.py`
- `rlinf/runners/sft_runner.py`
- `rlinf/workers/sft/fsdp_vla_sft_worker.py`
- `rlinf/workers/sft/fsdp_sft_worker.py`

`train_vla_sft.py` builds an `FSDPVlaSftWorker`; `SFTRunner.run()` repeatedly
calls `actor.run_training()`. The base worker accumulates micro-batches, calls
`get_train_model_output()`, backprops the mean loss, then calls
`optimizer_step()`, logs `loss`, `learning_rate`, and `grad_norm`.

The VLA worker dispatches by `actor.model.model_type`:

- `openpi` -> `build_openpi_real_world_joint_sft_dataloader()`
- `dreamzero` / `wmam` -> `build_real_world_joint_sft_dataloader()`

## OpenPI Training Logic

Main scripts/configs:

- `examples/sft/pi_05/run_real_world_franka_dual_openpi_pi05_sft.sh`
- `examples/sft/pi_05/run_real_world_franka_dual_openpi_pi05_sft_debug.sh`
- `examples/sft/config/real_world_franka_dual_openpi_pi05_sft.yaml`
- `rlinf/data/datasets/openpi/real_world_joint.py`
- `rlinf/models/embodiment/openpi/dataconfig/__init__.py`
- `rlinf/models/embodiment/openpi/dataconfig/real_world_joint_dataconfig.py`
- `rlinf/models/embodiment/openpi/policies/real_world_joint_policy.py`
- `rlinf/models/embodiment/openpi/openpi_action_model.py`

Current OpenPI config facts:

- `config_name: pi05_real_world_joint`
- `Pi0Config(pi05=True, action_horizon=48, discrete_state_input=True)`
- Model input spec:
  - images: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`, each `[B, 224, 224, 3]`, `float32`
  - image masks: each `[B]`, bool
  - state: `[B, 32]`, `float32`
  - prompt tokens: `[B, 200]`, int32, plus `[B, 200]` mask
  - actions: `[B, 48, 32]`, `float32`
- Raw robot state/action are 16-D. They are padded to 32-D for the model.
- `openpi_real_world_joint_loss_action_dim: 16`; current worker masks OpenPI
  action loss to the first 16 dimensions.

OpenPI dataset index rule:

For a sampled dataset frame `t = frame_in_ep`:

- image frame index: `t`
- state row: `[t]`
- action rows: `clip(t + [0, 1, ..., 47], 0, episode_length - 1)`
- prompt: first episode task string, lowercased

This comes from `_LeRobotV3FrankaDualOpenPIDataset.__getitem__()` in
`rlinf/data/datasets/openpi/real_world_joint.py`.

OpenPI preprocessing:

1. Read three LeRobot video streams:
   - `observation.images.middle_zed`
   - `observation.images.left_camera`
   - `observation.images.right_camera`
2. Convert decoded frames to HWC `uint8`.
3. Repack into OpenPI keys:
   - `observation/image`
   - `observation/left_wrist_image`
   - `observation/right_wrist_image`
   - `observation/state`
   - `actions`
   - `prompt`
4. `RealWorldJointInputs` maps them to:
   - image dict names `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`
   - `state`
   - optional `actions`
5. `Normalize(norm_stats, use_quantiles=True)` applies q01/q99 normalization.
6. OpenPI model transforms resize/tokenize/pad to the model input spec above.

OpenPI output/loss:

- Model predicts normalized action flow for `[B, 48, 32]`.
- Noise for padded action dimensions is masked to zero when `action_mask` is
  present.
- Worker loss is masked to first 16 dims for `openpi_real_world_joint`.
- Deployment should return/slice first 16 dims and unnormalize on the client with
  the same q01/q99 statistics.

Important OpenPI checks:

- Confirm the OpenPI official training path also uses the same resize,
  normalization, prompt tokenization, and `discrete_state_input=True` behavior.
- Confirm padded dims are never contributing to training loss or noisy action
  target.
- Confirm deployment server does not normalize state or unnormalize action a
  second time.
- Investigate why `grad_norm` increases while loss decreases.

## DreamZero Training Logic

Main scripts/configs:

- `examples/sft/dreamzero/dreamzero_real_world_sft/run_real_world_franka_dual_dreamzero_sft_multi_node.sh`
- `examples/sft/dreamzero/dreamzero_real_world_sft/run_real_world_franka_dual_dreamzero_sft_debug.sh`
- `examples/sft/config/real_world_franka_dual_dreamzero_sft.yaml`
- `examples/sft/config/real_world_franka_dual_dreamzero_sft_debug.yaml`
- `rlinf/data/datasets/dreamzero/real_world_joint.py`
- `rlinf/data/datasets/dreamzero/dreamzero.py`
- `rlinf/data/datasets/dreamzero/sampling_strategy.py`
- `rlinf/models/embodiment/dreamzero/__init__.py`
- `rlinf/models/embodiment/dreamzero/dreamzero_policy.py`
- `rlinf/models/embodiment/dreamzero/patch/wan_causal_model_forward_train.py`
- `../dreamzero/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
- `../dreamzero/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`

Current DreamZero config facts:

- `dataset_type: real_world_joint`
- `sampling_mode: multi_anchor`
- `macro_stride: 48`
- `video_in_chunk_offsets: [0, 6, 12, 18, 24, 30, 36, 42]`
- `action_horizon: 48`
- `max_chunk_size: 4`
- `action_dim/max_action_dim: 32`
- `max_state_dim: 64`
- `state_horizon: 1`
- `target_video_height/width: 256`
- `action_head_cfg.config.num_frames: 33`
- `num_frame_per_block: 2`
- `num_action_per_block: 48`
- `max_num_embodiments: 14`
- `real_world_franka_dual` raw embodiment id: `49`
- internal action/state category id: `49 - 36 = 13`

DreamZero temporal index rule:

For a sampled anchor frame `a` inside one language segment:

- Each macro chunk has stride `48`.
- Per chunk, video micro offsets are `[0, 6, 12, 18, 24, 30, 36, 42]`.
- Per chunk, action offsets are `[0, 1, ..., 47]`.
- Per chunk, state uses the anchor index.
- The sampler expands backward and forward by multiples of `macro_stride` inside
  the same language label and requires a full `max_chunk_size=4` window.

Therefore, for four macro chunks:

- video raw frames: `4 * 8 + 1 = 33`
  - the extra `+1` is the boundary frame added by the sampler
- state rows: `4`
- action rows: `4 * 48 = 192`

In the Franka real-world wrapper, `real_world_joint.py` subtracts the current
anchor from the base DreamZero offsets, then re-adds `frame_in_ep`, so the final
rows remain episode-local absolute rows.

DreamZero preprocessing:

1. Decode three views.
2. Resize/tile into a single video frame:
   - top: middle ZED / agent view
   - bottom-left: left wrist
   - bottom-right: right wrist
3. Normalize state/action with q01/q99.
4. Concatenate state components in order:
   - left arm 7
   - left gripper 1
   - right arm 7
   - right gripper 1
   - raw total: 16
5. Concatenate action components in the same order; raw total: 16.
6. Pad state to `[T_state, 64]`.
7. Pad action to `[T_action, 32]`.
8. Produce masks:
   - `state_mask`: true for real dims, false for padded dims
   - `action_mask`: true for real dims, false for padded dims
9. Set `embodiment_id=49`, which must map to internal action/state category 13.

DreamZero model/loss:

- Input image tensor reaches the action head as `[B, T, H, W, C]`, then becomes
  `[B, C, T, H, W]`.
- `uint8` video is converted to `[0,1]`, normalized to `[-1,1]`, resized to the
  configured target resolution, and VAE-encoded.
- With 33 raw video frames, Wan VAE produces 9 latent frames because the temporal
  relation is `raw_T = 1 + 4 * (latent_T - 1)`.
- The action branch uses noisy normalized actions and predicts action flow.
- `action_mask` is used to zero padded action noise and padded action training
  target, and the action loss is averaged only over valid action dimensions.
- Total loss is:
  - `dynamics_loss`
  - plus `action_loss`
  - plus `motion_loss` only for WMAM.

Important DreamZero checks:

- Confirm train and inference both use raw embodiment id `49`, and both map to
  internal category `13` before `action_encoder`, `state_encoder`, and
  `action_decoder`.
- Confirm `action_loss_embodiment_ids` includes `49` and that category mapping is
  not confused with the raw id.
- Confirm inference cache semantics match training: first chunk uses only frame
  0, later chunks cache the observed/predicted anchors according to the same
  `0,6,...,48` style timing used by training.
- Confirm state/action q01/q99 are identical between training and client.

## WMAM Training Logic

Main scripts/configs:

- `examples/sft/dreamzero/dreamzero_real_world_sft/run_real_world_franka_dual_wmam_motion_sft_multi_node.sh`
- `examples/sft/dreamzero/dreamzero_real_world_sft/run_real_world_franka_dual_wmam_motion_continue_sft_multi_node.sh`
- `examples/sft/config/real_world_franka_dual_wmam_motion_sft.yaml`

WMAM uses the same DreamZero real-world joint dataset path, temporal sampler,
state/action normalization, padding, masking, and embodiment mapping, but enables
the motion modality:

- `model_type: wmam`
- `use_motion_modality: true`
- `scene_flow_training: true`
- `use_sam_scene_flow: true`
- `motion_dir_name: motions_sam`
- `motion_downsample_ratio: 6`
- `motion_horizon: 8`
- `num_motion_per_block: 8`
- `motion_latent_channels: 8`
- `per_motion_seqlen: 16`

WMAM motion index rule:

- First compute the same action rows as DreamZero: 4 chunks x 48 rows = 192 rows.
- Convert action rows to real video/motion frame indices using `frame_index`
  when motion is enabled.
- Downsample each 48-step action chunk by 6:
  - per chunk: `[0, 6, 12, 18, 24, 30, 36, 42]`
  - total for 4 chunks: `4 * 8 = 32` motion frames
- Load `motion.point_map` and `motion.scene_flow` from the configured motion
  files; SAM and visibility confidence can mask scene flow.

WMAM loss:

- Same `dynamics_loss` and `action_loss` as DreamZero.
- Adds `motion_loss` from motion latent prediction.
- The current training worker logs `motion_loss`, and also has spike logging for
  motion loss / grad norm jumps.

## Known High-Risk Areas

These are not final conclusions. They are the most important areas to inspect.

1. Low scalar training loss may not imply good unnormalized action quality.
   Always evaluate predicted action in raw 16-D robot/action units against GT.
2. OpenPI `grad_norm` increasing while loss decreases may indicate training
   instability, loss masking/scale mismatch, or optimizer/schedule mismatch
   relative to official OpenPI.
3. Video prediction quality and action quality can diverge. DreamZero can decode
   plausible future video while action tokens are wrong because action/state
   branch conditioning, masks, or embodiment categories differ.
4. Padding must be handled consistently:
   - OpenPI: action/state padded to 32; loss only first 16 dims.
   - DreamZero/WMAM: action padded to 32, state padded to 64; masks must remove
     padded dims from noise and loss.
5. Normalization must be single-sourced. The client should normalize raw state
   and unnormalize returned action; servers should not repeat this unless the
   payload contract explicitly changes.
6. Prompts and task names must be identical or semantically intended. Even small
   prompt differences can hide whether the action branch learned the right
   conditional behavior.
7. Real-world frame rate and chunk execution must match training. Training data
   assumes 30 FPS; action horizon 48 means about 1.6 seconds per chunk.

## Prompt For Future OpenPI Debug Agent

You are debugging RLinf OpenPI pi0.5 training and deployment for Franka dual-arm
real-world data. Do not rely on memory. Read the code on `qizhi-local`.

Required files:

- `examples/sft/pi_05/run_real_world_franka_dual_openpi_pi05_sft.sh`
- `examples/sft/config/real_world_franka_dual_openpi_pi05_sft.yaml`
- `rlinf/data/datasets/openpi/real_world_joint.py`
- `rlinf/models/embodiment/openpi/dataconfig/__init__.py`
- `rlinf/models/embodiment/openpi/dataconfig/real_world_joint_dataconfig.py`
- `rlinf/models/embodiment/openpi/policies/real_world_joint_policy.py`
- `rlinf/models/embodiment/openpi/openpi_action_model.py`
- `deploy/open_pi/serve_franka_dual_openpi.py`
- `external franka machine: /home/franka/franka_rdk/lerobot/src/lerobot/policies/openpi_client/processor_openpi_client.py`

Tasks:

1. Compare this RLinf training path against
   `https://github.com/Physical-Intelligence/openpi`.
2. Write exact shapes before and after every transform: images, masks, state,
   prompt tokens, actions, action mask, noise.
3. Confirm image order and names: middle ZED/base, left wrist, right wrist.
4. Confirm action row rule is exactly `t:t+48` and state row is `t`.
5. Confirm q01/q99 normalization source is exactly the same in training and
   deployment.
6. Confirm `discrete_state_input=True` is intended for this pi0.5 config and
   matches official OpenPI behavior.
7. Confirm padded action dimensions have zero noise and no loss contribution.
8. Investigate why `grad_norm` grows while loss decreases. Check LR schedule,
   gradient clipping, loss reduction, trainable parameter set, and official
   optimizer settings.
9. Run dataset-host evaluation: feed training-set observation/state/prompt to the
   server, unnormalize predicted actions, compare against GT future 48x16 action,
   and report MSE, max error, first-step error, and displacement scale.
10. Give a concrete verdict: training bug, deployment bug, dataset bug, or still
    inconclusive. Every claim must cite file paths and line ranges.

## Prompt For Future DreamZero / WMAM Debug Agent

You are debugging RLinf DreamZero and WMAM training/deployment for Franka dual-arm
real-world data. Do not rely on memory. Read the code on `qizhi-local` and compare
against `https://github.com/dreamzero0/dreamzero`.

Required files:

- `examples/sft/dreamzero/dreamzero_real_world_sft/run_real_world_franka_dual_dreamzero_sft_multi_node.sh`
- `examples/sft/dreamzero/dreamzero_real_world_sft/run_real_world_franka_dual_wmam_motion_sft_multi_node.sh`
- `examples/sft/config/real_world_franka_dual_dreamzero_sft.yaml`
- `examples/sft/config/real_world_franka_dual_wmam_motion_sft.yaml`
- `rlinf/data/datasets/dreamzero/real_world_joint.py`
- `rlinf/data/datasets/dreamzero/dreamzero.py`
- `rlinf/data/datasets/dreamzero/sampling_strategy.py`
- `rlinf/models/embodiment/dreamzero/__init__.py`
- `rlinf/models/embodiment/dreamzero/dreamzero_policy.py`
- `rlinf/models/embodiment/dreamzero/patch/wan_causal_model_forward_train.py`
- `../dreamzero/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
- `../dreamzero/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`
- `deploy/dreamzero/serve_franka_dual_dreamzero.py`
- `deploy/wmam/serve_franka_dual_wmam.py`
- `external franka machine: /home/franka/franka_rdk/lerobot/src/lerobot/policies/dreamzero_client/processor_dreamzero_client.py`
- `external franka machine: /home/franka/franka_rdk/lerobot/src/lerobot/policies/wmam_client/processor_wmam_client.py`

Tasks:

1. Compare RLinf DreamZero/WMAM training against the official DreamZero repo.
2. Write exact temporal indices for a concrete anchor frame `a`: video rows,
   state rows, action rows, and WMAM motion rows.
3. Confirm the 33 raw video frames -> 9 latent frames rule and how
   `num_frame_per_block=2` aligns with four 48-action chunks.
4. Confirm the first chunk training condition: action 0..47 should only see the
   initial observation and allowed causal context, not future GT video in a way
   unavailable at inference.
5. Confirm inference cache semantics match training, including when KV cache is
   reset.
6. Confirm raw `embodiment_id=49` maps to internal action/state category `13`
   during both train and inference.
7. Confirm state/action normalization uses the intended global q01/q99 stats and
   is not repeated on the server.
8. Confirm action/state padding and masks:
   - state raw 16 -> padded 64
   - action raw 16 -> padded 32
   - padded dims get zero noise and no loss
9. For WMAM, confirm motion frames are exactly `[0,6,...,42]` per action chunk,
   total 32 frames for 4 chunks, and check point-map/scene-flow/SAM mask shapes.
10. Run dataset-host evaluation for DreamZero and WMAM: feed training-set
    observation/state/prompt to the server, compare normalized and unnormalized
    48x16 actions against GT, and save decoded predicted video/motion artifacts.
11. Give a concrete verdict: action-branch training bug, cache/inference bug,
    normalization bug, motion bug, or still inconclusive. Every claim must cite
    file paths and line ranges.
