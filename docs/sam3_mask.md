# SAM3 Mask 数据说明

本文档说明我们如何用 SAM3 为 real-world DreamZero motion 数据生成 `sam3_mask`，mask 的 true/false 语义，以及最终 `motions_sam/*.npy` 的格式。

相关代码：

- `/inspire/hdd/project/robot-body/linbokai-CZXS24250037/sam3/scripts/run_sam3_masks_robomind2.sh`
- `/inspire/hdd/project/robot-body/linbokai-CZXS24250037/sam3/scripts/run_sam3_masks_robocoin.sh`
- `/inspire/hdd/project/robot-body/linbokai-CZXS24250037/sam3/scripts/run_sam3_masks_allenai.sh`
- `/inspire/hdd/project/robot-body/linbokai-CZXS24250037/sam3/scripts/generate_real_world_sam3_masks.py`
- `/inspire/hdd/project/robot-body/linbokai-CZXS24250037/sam3/scripts/generate_missing_real_world_sam3_masks.py`

## 1. 运行入口

三个入口脚本主要区别是数据集名和 batch size：

| 脚本 | `DATASET_NAME` | Python 主逻辑 |
| --- | --- | --- |
| `run_sam3_masks_robomind2.sh` | `robomind2_lerobot_v21` | `generate_real_world_sam3_masks.py` |
| `run_sam3_masks_robocoin.sh` | `robocoin_filtered_v1` | `generate_missing_real_world_sam3_masks.py` |
| `run_sam3_masks_allenai.sh` | `allenai_lerobot_v3_filtered` | `generate_real_world_sam3_masks.py` |

默认数据根目录：

```text
/inspire/qb-ilm2/project/robot-body/public/real_world_data_dreamzero_motion
```

默认 SAM3 checkpoint：

```text
/inspire/hdd/project/robot-body/linbokai-CZXS24250037/sam3/checkpoints/facebook_sam3
```

常用环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATA_ROOT` | real_world motion 公共目录 | 数据根目录 |
| `MAX_EPISODES` | `-1` | 最多处理多少 episode，`-1` 表示全部 |
| `MAX_FRAMES` | `-1` | 每个 episode 最多处理多少帧 |
| `CONFIDENCE_THRESHOLD` | `0.0` | SAM3 score 过滤阈值 |
| `MAX_MASKS_PER_PROMPT` | `1` | 每个 prompt 默认保留 top-k mask |
| `TARGET_MAX_MASKS` | `gripper=2,object=1` | 不同 target 的 top-k 数量 |
| `OVERWRITE` | `0` | 是否覆盖已有 `motions_sam` |
| `DRY_RUN` | `0` | 只扫描不推理 |
| `SAM3_VERBOSE` | `0` | 打印更详细日志 |

示例：

```bash
cd /inspire/hdd/project/robot-body/linbokai-CZXS24250037/sam3
SAM3_VERBOSE=1 bash scripts/run_sam3_masks_robomind2.sh
```

`generate_missing_real_world_sam3_masks.py` 只改任务发现逻辑：如果目标 `motions_sam/chunk-*/episode_*.npy` 已存在且没有 `OVERWRITE=1`，就跳过该 episode；推理和保存逻辑仍然来自 `generate_real_world_sam3_masks.py`。

## 2. 输入数据和 prompt

SAM3 脚本会扫描：

```text
{DATA_ROOT}/{DATASET_NAME}/**/meta/info.json
```

对每个 task，要求存在 motion npy：

```text
<task>/motions/chunk-xxx/episode_xxxxxx.npy
```

视频路径优先通过同名 `.npz` 中的 `agent_view_key` / `source_video` 解析；否则从 `meta/info.json` 里的 video key 选择，优先级包含 `top`、`agent`、`exterior`、`left`、`right` 等关键词。

prompt 来自：

```text
<task>/meta/sam2_prompt.json
```

默认 target：

```text
gripper,object
```

示例 prompt 结构：

```json
{
  "targets": {
    "gripper": {
      "prompt": "robot gripper. robotic gripper. gripper."
    },
    "object": {
      "prompt": "sauce bottle. bottle. cup."
    }
  }
}
```

如果缺少 prompt，代码会使用 fallback：

- `gripper`: `"robot gripper. robotic gripper. gripper."`
- `object`: 当前 task 目录名替换 `-` 后的文本

## 3. mask 的 true / false 要求

最终保存的 `sam3_mask` 是一个二值前景 mask，表示当前 motion 网格上哪些像素属于需要保留 scene flow 的主体区域。

`True / 1` 表示：

- robot gripper / end effector 区域；
- task-relevant manipulated object 区域；
- 多个目标的并集：`combined = gripper_mask OR object_mask`；
- 如果同一 target 保留多个 SAM3 mask，例如 `gripper=2`，则这些 mask 也会先做 OR。

`False / 0` 表示：

- 背景、桌面、墙面、地面等静态环境；
- 非任务相关物体；
- 未被 `gripper` 或 `object` prompt 命中的区域；
- 低于 `CONFIDENCE_THRESHOLD` 或没有进入 top-k 的 SAM3 候选区域；
- SAM3 在该帧没有检测到有效 mask 的区域。

注意：

- `sam3_mask` 不是 soft probability，保存时是 0/1。
- `sam3_mask` 不是 Track4World 的 `visconf`。`visconf` 表示 scene flow 几何置信度；`sam3_mask` 表示语义前景区域。
- RLinf 中 `use_sam_scene_flow=True` 时，scene flow 会同时乘上 `visconf >= 0.5` 和 `sam3_mask >= 0.5`，也就是只在高置信且语义前景的区域保留 scene flow。

## 4. 生成流程

每个 episode 的流程：

1. 读取 `motions/chunk-xxx/episode_xxxxxx.npy`，得到 motion 的 `(T,H,W,C)`。
2. 读取同 episode 的 agent-view RGB 视频，帧数最多处理到 `T`。
3. 对每个 frame，用 SAM3 对 `gripper` 和 `object` prompt 分别推理。
4. 对每个 target，按 score 和 `TARGET_MAX_MASKS` 选 top-k mask。
5. 将所有 target mask 做 OR，得到原视频分辨率下的 combined mask。
6. 用 nearest neighbor resize 到 motion 分辨率 `(H,W)`。
7. 写入 `sam_small[t, ..., 0]`。
8. 把 `sam_small` append 到原 motion npy 的最后一维，保存到 `motions_sam`。

核心逻辑：

```python
combined = np.zeros(frame_rgb.shape[:2], dtype=bool)
for mask in masks.values():
    combined |= mask

small = cv2.resize(
    combined.astype(np.uint8),
    (motion_w, motion_h),
    interpolation=cv2.INTER_NEAREST,
)
sam_small[t, ..., 0] = small
```

保存逻辑：

```python
motion = np.load(motion_npy, mmap_mode="r", allow_pickle=False)
out = np.lib.format.open_memmap(
    tmp,
    mode="w+",
    dtype=motion.dtype,
    shape=(*motion.shape[:3], motion.shape[3] + 1),
)
out[..., : motion.shape[3]] = motion
out[..., motion.shape[3] :] = sam_mask.astype(motion.dtype, copy=False)
os.replace(tmp, output_npy)
```

## 5. 最终 `motions_sam` npy 格式

输入 motion npy：

```text
<task>/motions/chunk-xxx/episode_xxxxxx.npy
shape = (T,H,W,11)
dtype = float16
```

输出 SAM motion npy：

```text
<task>/motions_sam/chunk-xxx/episode_xxxxxx.npy
shape = (T,H,W,12)
dtype = float16
```

在用户指定的路径：

```text
/inspire/qb-ilm2/project/robot-body/public/real_world_data_dreamzero_motion/robomind2_lerobot_v21/franka/motions_sam
```

当前真实样本观测到的格式是：

```text
shape = (T,64,64,12)
dtype = float16
sam3_mask value = 0.0 or 1.0
```

最后一维 channel 含义：

| channel | 含义 |
| --- | --- |
| `0:3` | `point_map[..., xyz]` |
| `3:6` | `scene_flow[..., xyz]`，已经 pad 到 `T` |
| `6:7` | `visconf`，已经 pad 到 `T` |
| `7:10` | `mu[..., xyz]`，归一化中心，broadcast 到所有位置 |
| `10:11` | `S`，归一化尺度，broadcast 到所有位置 |
| `11:12` | `sam3_mask`，SAM3 语义前景二值 mask |

维度说明：

```text
axis 0: T，episode 内时间帧
axis 1: H，motion 网格高度，当前 public 数据通常为 64
axis 2: W，motion 网格宽度，当前 public 数据通常为 64
axis 3: C，通道；`motions_sam` 为 12
```

RLinf 读取 `motions_sam` 时的使用方式：

```python
point_map = np.moveaxis(selected[..., 0:3], -1, 1)

scene_flow = selected[..., 3:6]
visconf = selected[..., 6:7]
scene_flow = scene_flow * (visconf >= 0.5)

sam_mask = selected[..., -1:]
scene_flow = scene_flow * (sam_mask >= 0.5)
scene_flow = np.moveaxis(scene_flow, -1, 1)
```

所以最终进入模型的数据仍然是：

```text
motion.point_map:  (N,3,H,W), float32
motion.scene_flow: (N,3,H,W), float32
```

其中 `motion.scene_flow` 已经被 `visconf` 和 `sam3_mask` 过滤。

