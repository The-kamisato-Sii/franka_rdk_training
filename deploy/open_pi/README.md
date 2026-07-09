# OpenPI Franka Dual Server

This directory serves the RLinf OpenPI pi0.5 SFT model for the separate
`franka_rdk` robot machine.

## Protocol

The server matches `franka_rdk`'s OpenPI client:

```text
POST /infer
Content-Type: application/msgpack
```

Request:

```python
{
    "openpi": {
        "state": np.ndarray,      # [32] q01/q99 normalized and padded
        "state_mask": np.ndarray, # [32] bool
        "images": {
            "base_0_rgb": np.ndarray,        # [3,224,224] float32 in [-1,1]
            "left_wrist_0_rgb": np.ndarray,  # [3,224,224] float32 in [-1,1]
            "right_wrist_0_rgb": np.ndarray, # [3,224,224] float32 in [-1,1]
        },
        "image_mask": {...},
        "action_q01": np.ndarray, # [32]
        "action_q99": np.ndarray, # [32]
    },
    "prompt": str,
}
```

The strict path uses the client-preprocessed `payload["openpi"]` block. The
server does not redo q01/q99 state normalization or image resize/pad on this
path; it only tokenizes the prompt, builds an OpenPI `Observation`, and calls
`sample_actions(...)`. It returns the normalized model-space action directly;
`openpi_pi05_client` owns the q01/q99 action inverse before publishing robot
commands.

By default, the server now fails if `payload["openpi"]` is missing, because
training-matched deployment keeps normalization and image preprocessing inside
`franka_rdk/lerobot/src/lerobot/policies/openpi_client/processor_openpi_client.py`.
There is no raw top-level `state/images` fallback in `/infer`; if a client does
not send the `payload["openpi"]` block, the request is rejected immediately.

Response:

```python
{"actions": np.ndarray}  # shape [48, 16]
```

The first 16 action dimensions are:

```text
left_joint_positions_0..7 + right_joint_positions_0..7
```

The OpenPI model internally uses the 32-d padded action/state format used during
SFT. The server always returns the full 48-step model horizon, sliced to 16 dims;
`openpi_pi05_client` applies the action q01/q99 inverse before commanding the
robot, and decides how many leading actions to execute before replanning.

Checkpoint note: this SFT run used FSDP with `sharding_strategy: no_shard`.
Although the directory is named `local_shard_checkpoint`, each
`checkpoint_rank_*.pt` file contains a full model copy. The server loads
`checkpoint_rank_0.pt` by default.

## Start Server

Use the `rlinf_openpi` environment:

```bash
cd /inspire/hdd/project/robot-body/linbokai-CZXS24250037/RLinf
conda activate rlinf_openpi

bash deploy/open_pi/run_franka_dual_openpi_server.sh
```

Defaults:

```text
OpenPI cache:
  /inspire/hdd/project/robot-body/linbokai-CZXS24250037/RLinf/checkpoints/openpi_cache

base model:
  /inspire/hdd/project/robot-body/linbokai-CZXS24250037/RLinf/checkpoints/pi05_base_pytorch_real_world_joint

SFT checkpoint:
  /inspire/hdd/project/robot-body/linbokai-CZXS24250037/results/real_world_franka_dual_openpi_pi05_sft_v2/checkpoints/global_step_10000

server:
  0.0.0.0:8000

sampling:
  NUM_STEPS=10
```

The launch script sets `OPENPI_DATA_HOME` to the same cache path used by the
SFT training script, so OpenPI can find
`big_vision/paligemma_tokenizer.model` without downloading from GCS.

Common overrides:

```bash
DEVICE=cuda:1 \
PORT=8001 \
NUM_STEPS=10 \
DEFAULT_PROMPT="fold the box" \
bash deploy/open_pi/run_franka_dual_openpi_server.sh
```

If you want to use another checkpoint:

```bash
CHECKPOINT_PATH=/path/to/global_step_xxxxx \
bash deploy/open_pi/run_franka_dual_openpi_server.sh
```

`CHECKPOINT_PATH` can be either the `global_step_xxxxx` directory or a direct
`checkpoint_rank_0.pt` file.

## Official OpenPI Checkpoints

Official OpenPI training saves JAX/Orbax checkpoints, for example:

```text
/inspire/hdd/project/robot-body/linbokai-CZXS24250037/results/openpi_official/pi05_franka_dual_test/pi05_franka_dual_test_20k/19999
```

This cannot be passed to `CHECKPOINT_PATH` directly. Convert it to a complete
PyTorch `model.safetensors` directory first:

```bash
bash -c '
source /inspire/hdd/project/robot-body/linbokai-CZXS24250037/miniconda/etc/profile.d/conda.sh
conda activate /inspire/hdd/project/robot-body/linbokai-CZXS24250037/miniconda/envs/openpi
cd /inspire/hdd/project/robot-body/linbokai-CZXS24250037/openpi
export PYTHONPATH=/inspire/hdd/project/robot-body/linbokai-CZXS24250037/openpi/src:${PYTHONPATH:-}
export OPENPI_DATA_HOME=/inspire/hdd/project/robot-body/linbokai-CZXS24250037/RLinf/checkpoints/openpi_cache
python examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /inspire/hdd/project/robot-body/linbokai-CZXS24250037/results/openpi_official/pi05_franka_dual_test/pi05_franka_dual_test_20k/19999 \
  --config-name pi05_franka_dual_test \
  --output-path /inspire/hdd/project/robot-body/linbokai-CZXS24250037/results/openpi_official/pi05_franka_dual_test/pi05_franka_dual_test_20k/19999_pytorch \
  --precision float32
mkdir -p \
  /inspire/hdd/project/robot-body/linbokai-CZXS24250037/results/openpi_official/pi05_franka_dual_test/pi05_franka_dual_test_20k/19999_pytorch/real_world_franka_dual \
  /inspire/hdd/project/robot-body/linbokai-CZXS24250037/results/openpi_official/pi05_franka_dual_test/pi05_franka_dual_test_20k/19999_pytorch/real_world_joint
cp /inspire/qb-ilm2/project/robot-body/public/bokai/franka_dual_test/norm_stats.json \
  /inspire/hdd/project/robot-body/linbokai-CZXS24250037/results/openpi_official/pi05_franka_dual_test/pi05_franka_dual_test_20k/19999_pytorch/real_world_franka_dual/norm_stats.json
cp /inspire/qb-ilm2/project/robot-body/public/bokai/franka_dual_test/norm_stats.json \
  /inspire/hdd/project/robot-body/linbokai-CZXS24250037/results/openpi_official/pi05_franka_dual_test/pi05_franka_dual_test_20k/19999_pytorch/real_world_joint/norm_stats.json
'
```

Serve the converted directory as the base model and disable local-shard loading:

```bash
BASE_MODEL_PATH=/inspire/hdd/project/robot-body/linbokai-CZXS24250037/results/openpi_official/pi05_franka_dual_test/pi05_franka_dual_test_20k/19999_pytorch \
CHECKPOINT_PATH=none \
RETURN_ACTION_DIM=16 \
NUM_STEPS=10 \
bash deploy/open_pi/run_franka_dual_openpi_server.sh
```

For official-converted checkpoints, `BASE_MODEL_PATH` is the converted full
model directory. `CHECKPOINT_PATH=none` is intentional; it tells the server not
to look for an RLinf `actor/local_shard_checkpoint/checkpoint_rank_0.pt`.

Before `global_step_10000` exists, the current v2 checkpoint can be served with:

```bash
CHECKPOINT_PATH=/inspire/hdd/project/robot-body/linbokai-CZXS24250037/results/real_world_franka_dual_openpi_pi05_sft_v2/checkpoints/global_step_8000 \
bash deploy/open_pi/run_franka_dual_openpi_server.sh
```

The server does not normalize raw state and does not inverse-normalize actions.
The `franka_rdk` OpenPI client must use the same v2 dataset stats used in SFT
for client-side state normalization and action inverse normalization:

```text
/inspire/qb-ilm2/project/robot-body/public/bokai/franka_dual_test/norm_stats.json
```

## Smoke Test

After the server is up:

```bash
python deploy/open_pi/test_openpi_server_client.py \
  --host http://127.0.0.1:8000 \
  --prompt "fold the box"
```

Expected output:

```text
actions.shape = (48, 16)
```

## Robot Machine

On the `franka_rdk` machine, point its OpenPI client at this server:

```bash
--policy.type=openpi_pi05_client
--policy.host=http://<GPU_SERVER_IP>:8000
--policy.default_prompt="fold the box"
--policy.stats_task_id=fold_box
```

The server always returns the full 48-step normalized action chunk. The
`openpi_pi05_client` controls receding-horizon execution with
`n_action_steps` (24 by default, override with `POLICY_N_ACTION_STEPS=12` in
the robot script), and already sets `include_preprocessed_payload=true`,
`send_raw_compat_payload=false`, and `server_returns_normalized_actions=true`.
It sends the `payload["openpi"]` block consumed by this server.
