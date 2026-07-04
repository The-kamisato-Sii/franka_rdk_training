# DreamZero Motion 数据说明

本文档说明 real-world DreamZero motion 数据的定义、归一化、`.npz` 到 `.npy` 的保存格式，以及 Track4World / RLinf 中 `motion.point_map` 和 `motion.scene_flow` 的接口对应关系。

相关代码：

- `/inspire/hdd/project/robot-body/linbokai-CZXS24250037/Track4World/scripts/run_generated_gt.sh`
- `/inspire/hdd/project/robot-body/linbokai-CZXS24250037/Track4World/scripts/generate_motioncrafter_gt.py`
- `/inspire/qb-ilm2/project/robot-body/public/real_world_data_dreamzero_motion/transform_npz2npy.py`
- `/inspire/hdd/project/robot-body/linbokai-CZXS24250037/RLinf/rlinf/data/datasets/dreamzero/real_world_joint.py`

## 1. Motion 的定义

这里的 motion 是从 agent / primary camera 视频上生成的 dense 3D motion 表示。每个 episode 对应一条 motion 文件，和同 episode 的 RGB 视频帧按时间维对齐。

核心字段有两个：

| 字段 | 语义 | 形状 |
| --- | --- | --- |
| `point_map` | 每一帧、每个像素对应的 3D 点坐标。它是绝对位置，不是位移。 | `.npz` 公共格式为 `(T,H,W,3)`；RLinf 读出后变成 `(N,3,H,W)` |
| `scene_flow` | 每一帧、每个像素从当前帧到后续帧的 3D 位移。 | `.npz` 公共格式为 `(T-flow_stride,H,W,3)`；转 `.npy` 后 pad 到 `T` |

坐标含义：

- `agent-view` 指 motion 来自 agent / primary camera 对应的视频视角，不是 wrist camera。
- Track4World 原始输出先在相机坐标系中得到 `points` 和 `flow_3d`。
- `generate_motioncrafter_gt.py` 会把这些量变换到第 0 帧归一化后的世界坐标系，再做全局中心化和尺度归一化。
- 像素坐标统一记为 `(u,v)`，其中 `u` 是横向坐标，对应 width / array axis 2；`v` 是纵向坐标，对应 height / array axis 1。因此数组访问写作 `[t,v,u]`。
- `point_map[t,v,u] = (X,Y,Z)` 表示第 `t` 帧像素 `(u,v)` 在该归一化坐标系下的 3D 位置。
- `scene_flow[t,v,u] = (dX,dY,dZ)` 表示同一个源像素 `(u,v)` 从 `t` 到 `t+flow_stride` 的 3D 位移。`generate_motioncrafter_gt.py` 的 `infer_pair` 路径是相邻帧，等价于 `flow_stride=1`；公共 real-world motion 数据中实际 `.npz` 会显式保存 `flow_stride`，例如当前 `robomind2_lerobot_v21/franka` 样本为 `flow_stride=6`。

RLinf 训练时对外暴露的 sample key 是：

```text
motion.point_map
motion.scene_flow
```

在 batch collate 之后会合并为：

```python
batch["motion"] = {
    "point_map": batch.pop("motion.point_map"),
    "scene_flow": batch.pop("motion.scene_flow"),
}
```

## 2. Track4World 接口对应关系

`generate_motioncrafter_gt.py` 使用 `Track4World.infer_pair(...)`，因为这个接口天然按相邻帧 pair 输出 `t -> t+1` 的 motion：

```python
output = model.infer_pair(
    rgbs,
    iters=args.inference_iters,
    tracking3d=True,
    force_projection=True,
    aligned_scene_flow=True,
)
```

返回值是两个 dict：

```text
output[0]: per-frame geometry
output[1]: pairwise motion
```

和最终 motion 字段的关系：

| 最终字段 | Track4World 原始接口 | 后处理 |
| --- | --- | --- |
| `motion.point_map` | `output[0]["points"]`，形状 `(B,T,H,W,3)`，相机坐标系 point map | 取 batch 0，结合 `output[0]["camera_poses"]` 转到第 0 帧世界系，再全局归一化 |
| `motion.scene_flow` | `output[1]["flow_3d"]`，形状 `(B,T-1,H,W,3)` | 注意它在当前代码里是 `t` 帧像素到 `t+1` 时刻的绝对 3D 位置，不是位移；需要先减去 `points[t]` 得到相机系位移，再转世界系 |
| `visconf` | `output[1]["visconf_maps_e"]` | 当前脚本用两个 confidence channel 相乘：`visconf = ch0 * ch1` |
| `c2w` | `output[0]["camera_poses"]` | 先归一化到第 0 帧坐标系，作为 point/flow 世界系变换的中间量 |

### 2.1 flow_stride 和 Track4World stride 的关系

需要区分两个概念：

- `flow_stride` 是保存到 motion 数据里的语义字段，表示 `scene_flow[t]` 跨越原始视频中的多少帧，即从 `t` 到 `t + flow_stride`。
- Track4World 代码里的 `stride` 是长序列 sliding window 的窗口步长，用于决定窗口起点和窗口重叠，不是 `scene_flow` 的时间跨度。

在当前 Track4World 代码中，`infer_pair(...)` 虽然签名里有 `stride=None`，但这个参数没有传入真正做 pairwise 推理的 `forward_sliding1(...)`：

```python
output = model.infer_pair(...)

# track4world/nets/model.py
return_dict = self.forward_sliding1(
    images,
    iters=iters,
    sw=sw,
    is_training=is_training,
    tracking3d=tracking3d,
)
```

`forward_sliding1(...)` 内部固定通过 `pairwise_concat(...)` 把输入序列拼成相邻帧 pair：

```python
first = tensor[:, :-1]
second = tensor[:, 1:]
pairs = torch.stack([first, second], dim=2)
```

然后每个 pair 都用第 0 个元素作为源帧、第 1 个元素作为目标帧：

```python
fmap1_single = fmaps_chunk[:, 0]
fmap2 = fmaps_chunk[:, 1:2]
pm1_single = pm_anchor_chunk
pm2 = pms_chunk[:, 1:2]
```

所以 `infer_pair` 的输出永远是“输入序列相邻帧”的 flow，长度为 `T_input - 1`。如果送入的是连续视频帧 `[0, 1, 2, ...]`，输出就是 `0->1, 1->2, ...`，等价于 `flow_stride=1`。

如果需要保存 `flow_stride=6` 的 dense scene flow，目标不是抽帧成 `[0, 6, 12, ...]`。抽帧只会得到稀疏的 `0->6, 6->12, ...`。我们真正需要的是滑动 pair：

```text
0 -> 6
1 -> 7
2 -> 8
...
T-7 -> T-1
```

因此输入原始视频长度为 `T` 时，`flow_stride=6` 的 `scene_flow` 形状应是：

```text
(T - 6, H, W, 3)
```

实现上，`flow_stride` 不是由 Track4World 的某个模型接口直接控制的，而是由 Track4World 外部的数据生成逻辑控制：对每个起点 `i` 构造二帧 clip `[frame_i, frame_{i+flow_stride}]`，调用 `infer_pair` 得到这一对的 flow，再把所有 `i=0..T-flow_stride-1` 的结果按时间维拼起来。

当前 Track4World 的 `forward_sliding1` / `infer_pair` 主要按 `B=1, T=sequence_length` 的方式使用。若要把所有二帧 pair 组织成大 batch `(T-flow_stride, 2, C, H, W)`，需要先确认或修正内部 points / chunk 对齐逻辑；最稳妥的实现是外层按 pair 循环或小批量包装，并保证每个输出都对应原始帧 `i -> i+flow_stride`。

Track4World 里真正使用 `stride` 的地方是 `get_T_padded_images(...)`：

```python
step = S // 2 if stride is None else stride
```

这个 `step` 只决定长序列 `forward(...)` / `forward_sliding(...)` 的 sliding window 起点间隔。它不会把 `scene_flow[t]` 从 `t -> t+1` 改成 `t -> t+step`。

### 2.2 dreamzero/scripts/data 的 dense flow_stride 实现

公共 real-world motion 数据不是用 `Track4World/scripts/generate_motioncrafter_gt.py` 生成的，而是用 `dreamzero/scripts/data` 下的脚本生成。例如 AllenAI 入口：

```text
/inspire/hdd/project/robot-body/linbokai-CZXS24250037/dreamzero/scripts/data/generate_allenai_real_world_motion_agent_dist.sh
```

这个 shell 只是 `torchrun` 包装，实际调用：

```text
scripts/data/generate_allenai_real_world_motion_agent.py
  -> generate_real_world_motion_agent_common.py
  -> run_for_source("allenai_lerobot_v3_filtered")
```

同一套 common 逻辑也被 RoboMind2 / RoboCoin 入口复用：

```text
generate_robomind2_real_world_motion_agent.py
generate_robocoin_real_world_motion_agent.py
```

真实 `flow_stride` 参数在 `generate_real_world_motion_agent_common.py` 中定义，默认是 6：

```python
parser.add_argument("--flow-stride", type=int, default=6)
```

该脚本先对整条视频跑一次 `infer_pair`，只取 geometry，用于得到每一帧的 `point_map` 和统一的 `c2w_full`：

```python
rgbs_full = torch.stack([... for f in frames]).unsqueeze(0).to(device)
full_output = model.infer_pair(rgbs_full, ...)
points_cam_full = full_output[0]["points"][0].detach().cpu().numpy()
c2w_full = normalize_camera_poses(full_output[0]["camera_poses"].detach().cpu().numpy())
point_map = cam_points_to_world(points_cam_full, c2w_full)
```

然后按 `flow_stride` 创建 dense scene flow 容器：

```python
flow_len = t_total - args.flow_stride
scene_flow = np.zeros((flow_len, h, w, 3), dtype=np.float32)
visconf = np.zeros((flow_len, h, w), dtype=np.float32)
```

关键实现是 offset loop。对 `flow_stride=6`，它不是只跑 `[0,6,12,...]`，而是跑 6 组 offset 序列：

```python
for offset in range(args.flow_stride):
    idxs = list(range(offset, len(frames), args.flow_stride))
    output = model.infer_pair(frames[idxs], ...)
```

对应关系是：

```text
offset=0: [0, 6, 12, ...]  -> 0->6, 6->12, ...
offset=1: [1, 7, 13, ...]  -> 1->7, 7->13, ...
offset=2: [2, 8, 14, ...]  -> 2->8, 8->14, ...
...
offset=5: [5, 11, 17, ...] -> 5->11, 11->17, ...
```

把 6 组结果按原始起点 `global_i` 写回，就得到完整 dense `0->6, 1->7, ..., T-7->T-1`：

```python
pairs = [
    (local_i, global_i)
    for local_i, global_i in enumerate(idxs[:-1])
    if global_i < flow_len and global_i + args.flow_stride < t_total
]

for out_i, (local_i, global_i) in enumerate(pairs):
    scene_flow[global_i] = flow_world[out_i]
    visconf[global_i] = conf[local_i]
```

因此，若原始视频有 `T=129` 帧且 `flow_stride=6`，该脚本保存的 `scene_flow` 是：

```text
shape = (123, H, W, 3)
semantic = [0->6, 1->7, 2->8, ..., 122->128]
```

最后它会保存公共 real-world motion `.npz` 所需字段：

```python
{
    "point_map": pm_norm,
    "scene_flow": sf_norm,
    "valid_mask": valid_mask,
    "visconf": visconf,
    "mu": mu,
    "S": scale,
    "flow_stride": np.array(args.flow_stride, dtype=np.int32),
    "agent_view_key": job.agent_key,
    "source_video": str(job.video_path),
}
```

这套逻辑和 `Track4World/scripts/generate_motioncrafter_gt.py` 不同：后者只对连续帧调用一次 `infer_pair`，输出 `T-1` 个相邻帧 flow，等价于 `flow_stride=1`；`dreamzero/scripts/data/generate_real_world_motion_agent_common.py` 才是当前公共 real-world 数据中 dense `flow_stride=6` 的实现。

关键代码对应：

```python
points_cam = output[0]["points"][0].cpu().numpy()          # (T,H,W,3)
c2w = output[0]["camera_poses"].cpu().numpy()              # (T,4,4)
flow_3d_abs = output[1]["flow_3d"][0].cpu().numpy()        # (T-1,H,W,3), absolute target position
visconf_raw = output[1]["visconf_maps_e"][0].cpu().numpy() # (T-1,2,H,W)

scene_flow_cam = flow_3d_abs - points_cam[:-1]
```

## 3. 归一化方法

归一化分四步，对应 `generate_motioncrafter_gt.py` 中的 `normalize_camera_poses`、`transform_point_maps_to_world`、`transform_scene_flow_to_world`、`normalize_world_coords`。

### 3.1 相机位姿归一化到第 0 帧

输入 `c2w[i] = (R_i, t_i)`。以第 0 帧为世界坐标原点：

```text
R'_i = R_0^T R_i
t'_i = R_0^T (t_i - t_0)
```

归一化后第 0 帧相机位姿约为单位变换。

### 3.2 point_map 从相机系转世界系

对第 `i` 帧每个像素的相机系点 `X_i^C`：

```text
X_i = R'_i X_i^C + t'_i
```

`X_i` 就是世界系 `point_map`。

### 3.3 scene_flow 从相机系转世界系

Track4World `infer_pair` 输出的 `flow_3d` 是下一时刻的绝对 3D 位置，所以先得到相机系位移：

```text
V_i^C = flow_3d_abs[i] - X_i^C
X_{i -> i+1}^C = X_i^C + V_i^C
```

然后用 `i+1` 帧的相机位姿把变形后的点转到世界系：

```text
X_{i -> i+1} = R'_{i+1} X_{i -> i+1}^C + t'_{i+1}
X_i = R'_i X_i^C + t'_i
V_i = X_{i -> i+1} - X_i
```

`V_i` 就是世界系 `scene_flow`。

### 3.4 全局中心和尺度归一化

有效点由 `visconf` 产生：

```text
valid_mask = visconf > 0.5
```

`valid_mask` 原本只有 `T-flow_stride` 帧，代码会扩展到 `T` 帧，最后一帧使用最后一个有效 flow mask。

对所有有效点计算：

```text
mu = mean(valid_points)
S  = mean(||valid_points - mu||_2)
```

如果有效点少于 10 个，或者 `S < 1e-6`，退化为：

```text
mu = [0,0,0]
S = 1
```

归一化公式：

```text
point_map_norm = (point_map - mu) / S
scene_flow_norm = scene_flow / S
c2w_translation_norm = (c2w_translation - mu) / S
```

反归一化时：

```text
point_map = point_map_norm * S + mu
scene_flow = scene_flow_norm * S
```

## 4. 保存格式

### 4.1 过渡 `.npz`

`generate_motioncrafter_gt.py` 的 `save_episode_lerobot(...)` 会写：

```text
{output_root}/{dataset_name}/motion/chunk-xxx/episode_xxxxxx.npz
```

当前脚本内保存的 key：

| key | 形状 | dtype | 说明 |
| --- | --- | --- | --- |
| `point_map` | `(T,3,H,W)` | `float16` 或 `float32` | 归一化后的世界系 point map |
| `scene_flow` | `(T-1,3,H,W)` | `float16` 或 `float32` | 归一化后的世界系相邻帧 scene flow |
| `c2w` | `(T,4,4)` | `float32` | 归一化后的 camera-to-world，中间/调试用 |
| `visconf` | `(T-1,1,H,W)` | `float32` | flow confidence |
| `meta` | scalar JSON string | string | episode 元信息和 normalize params |

需要注意两点：

- 公共 real-world motion 数据目录使用的是 `motions/`，SAM3 和 RLinf 默认也找 `motions/` / `motions_sam/`，不是 singular `motion/`。
- `/inspire/qb-ilm2/project/robot-body/public/real_world_data_dreamzero_motion/transform_npz2npy.py` 读取的 `.npz` 约定是 channel-last `THWC`，并要求顶层 `mu` 和 `S`。如果直接使用 `generate_motioncrafter_gt.py` 当前输出，需要先把 `point_map/scene_flow/visconf` 转成公共 `.npz` 约定，或者同步修改转换脚本。

公共 real-world `.npz` 实际约定如下：

| key | 形状 | dtype | 说明 |
| --- | --- | --- | --- |
| `point_map` | `(T,H,W,3)` | `float16` | 归一化后的 3D position |
| `scene_flow` | `(T-flow_stride,H,W,3)` | `float16` | 归一化后的 3D displacement |
| `valid_mask` | `(T,H,W)` | `bool` | point map 有效区域 |
| `visconf` | `(T-flow_stride,H,W)` | `float16` | scene flow confidence |
| `mu` | `(3,)` | `float32` | 归一化中心 |
| `S` | scalar | `float32` | 归一化尺度 |
| `flow_stride` | scalar | `int32` | scene flow 的时间跨度 |
| `agent_view_key` | scalar string | string | 生成 motion 使用的视频 key |
| `source_video` | scalar string | string | 原始视频路径 |

### 4.2 最终训练 `.npy`

最终训练优先使用 `.npy`，由 `transform_npz2npy.py` 从同名 `.npz` 生成：

```bash
python /inspire/qb-ilm2/project/robot-body/public/real_world_data_dreamzero_motion/transform_npz2npy.py \
  /inspire/qb-ilm2/project/robot-body/public/real_world_data_dreamzero_motion/<dataset_or_task_root> \
  --workers 16
```

输出路径和 `.npz` 同目录同名：

```text
.../motions/chunk-xxx/episode_xxxxxx.npy
```

`.npy` 是 mmap-friendly 单数组：

```text
shape = (T,H,W,11)
dtype = float16
layout = THWC
```

最后一维 channel 含义：

| channel | 含义 |
| --- | --- |
| `0:3` | `point_map[..., xyz]` |
| `3:6` | `scene_flow[..., xyz]`，已经 pad 到 `T`；没有 flow 的末尾帧为 0 |
| `6:7` | `visconf`，已经 pad 到 `T`；没有 flow 的末尾帧为 0 |
| `7:10` | `mu[..., xyz]`，broadcast 到所有时空位置 |
| `10:11` | `S`，broadcast 到所有时空位置 |

当前公共 `robomind2_lerobot_v21/franka` 样本的实际 shape 是：

```text
motions/chunk-*/episode_*.npy: (T,64,64,11), float16
```

## 5. RLinf 如何读取 motion

代码位置：

```text
/inspire/hdd/project/robot-body/linbokai-CZXS24250037/RLinf/rlinf/data/datasets/dreamzero/real_world_joint.py
```

入口逻辑：

- `_motion_dir_name = "motions_sam"` if `use_sam_scene_flow=True` else `"motions"`。
- `_motion_path(episode_index)` 拼出 `.npz` 路径。
- `_motion_npy_path(episode_index)` 拼出同名 `.npy` 路径。
- `_load_motion_frames(...)` 优先读 `.npy`，没有 `.npy` 时回退到 `.npz`。

读 `.npy` 时的字段映射：

```python
selected = np.asarray(arr[frame_idx])

point_map = np.moveaxis(selected[..., 0:3], -1, 1)  # (N,3,H,W)
scene_flow = selected[..., 3:6]                     # (N,H,W,3)
visconf = selected[..., 6:7]
scene_flow = scene_flow * (visconf >= 0.5)

if use_sam_scene_flow:
    sam_mask = selected[..., -1:]
    scene_flow = scene_flow * (sam_mask >= 0.5)

scene_flow = np.moveaxis(scene_flow, -1, 1)          # (N,3,H,W)
```

因此，模型/transform 看到的 `motion.point_map` 和 `motion.scene_flow` 都是：

```text
(num_selected_motion_frames, 3, H, W), float32
```
