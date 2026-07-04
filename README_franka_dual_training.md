# Franka Dual SFT Notes

This patch adds two RLinf SFT configs for:

- `real_world_franka_dual_openpi_pi05_sft`
- `real_world_franka_dual_dreamzero_sft`

Expected dataset root:

```bash
/inspire/qb-ilm2/project/robot-body/public/hairuoliu/franka_dual
```

The configs reuse RLinf's `real_world_joint` discovery/normalization path. OpenPI uses the no-motion adapter added previously and keeps 32-d padded actions/states; the Franka deployment client consumes the first 16 dimensions as:

```text
left_joint_positions_0..7 + right_joint_positions_0..7
```

## OpenPI pi0.5

Convert the JAX pi0.5 checkpoint once before SFT:

```bash
cd /inspire/hdd/project/robot-body/linbokai-CZXS24250037/RLinf
python rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py \
  --checkpoint-dir checkpoints/pi05_base/params \
  --config-name pi05_real_world_joint \
  --output-path checkpoints/pi05_base_pytorch_real_world_joint \
  --precision bfloat16
```

Train:

```bash
cd /inspire/hdd/project/robot-body/linbokai-CZXS24250037/RLinf
bash examples/sft/run_real_world_franka_dual_openpi_pi05_sft.sh
```

Optional overrides:

```bash
REAL_WORLD_FRANKA_DUAL_ROOT=/path/to/franka_dual \
OPENPI_PI05_PYTORCH_CKPT=/path/to/pi05_base_pytorch_real_world_joint \
bash examples/sft/run_real_world_franka_dual_openpi_pi05_sft.sh
```

## DreamZero 5B

Train from the DreamZero 5B base components:

```bash
cd /inspire/hdd/project/robot-body/linbokai-CZXS24250037/RLinf
bash examples/sft/run_real_world_franka_dual_dreamzero_sft_multi_node.sh
```

Train from a previous DreamZero SFT checkpoint:

```bash
bash examples/sft/run_real_world_franka_dual_dreamzero_sft_multi_node.sh \
  actor.model.model_path=/path/to/previous_dreamzero_sft_ckpt
```

For multi-node launches, set `NNODES`, `NUM_GPUS`, `NODE_RANK`, and `MASTER_ADDR` in the same way as the existing DreamZero real-world scripts.
