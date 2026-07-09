# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.utils.checkpoint
from groot.vla.model.dreamzero.modules.wan2_1_submodule import sinusoidal_embedding_1d

# This patch is DreamZero CausalWanModel._forward_train with RLinf checkpointing
# behavior preserved, plus DreamZero's real-world motion token path.


def _forward_train(
    self,
    x,
    timestep,
    timestep_action,
    context,
    seq_len,
    clean_x=None,
    aug_t=None,
    y=None,
    clip_feature=None,
    action=None,
    state=None,
    embodiment_id=None,
    motion=None,
    timestep_motion=None,
    motion_condition=None,
):
    if self.model_type == "i2v":
        assert clip_feature is not None and y is not None

    if y is not None and self.concat_first_frame_latent:
        x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

    x = self.patch_embedding(x)
    grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
    freqs = self._create_freqs(
        grid_size=grid_size,
        start_frame=0,
    )

    x = x.flatten(start_dim=2).transpose(1, 2)
    assert x.shape[1] == seq_len

    B = x.shape[0]
    F = timestep.shape[1]

    motion_condition_length = 0
    motion_condition_freqs = None
    if motion_condition is not None and self.motion_patch_embedding is not None:
        motion_condition_input = motion_condition.to(dtype=x.dtype)
        motion_condition_emb = self.motion_patch_embedding(motion_condition_input)
        motion_condition_grid_size = torch.tensor(
            motion_condition_emb.shape[2:],
            dtype=torch.long,
            device=motion_condition_input.device,
        )
        motion_condition_features = motion_condition_emb.flatten(start_dim=2).transpose(
            1, 2
        )
        motion_condition_length = motion_condition_features.shape[1]
        motion_condition_freqs = self._create_motion_freqs(
            grid_size=motion_condition_grid_size,
            start_motion_block=0,
        )
        x = torch.cat([x, motion_condition_features], dim=1)

    if action is not None:
        if embodiment_id is None:
            embodiment_id = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        else:
            embodiment_id = embodiment_id.to(device=x.device, dtype=torch.long).view(-1)
        if embodiment_id.numel() != x.shape[0]:
            raise ValueError(
                f"embodiment_id must have batch size {x.shape[0]}, got {tuple(embodiment_id.shape)}"
            )
        num_embodiments = int(getattr(self.action_encoder, "num_embodiments", 1))
        action_embodiment_id = embodiment_id
        if embodiment_id.numel() and int(embodiment_id.min().item()) >= 36:
            if num_embodiments in (13, 14):
                action_embodiment_id = embodiment_id - 36
            elif num_embodiments == 15 and int(embodiment_id.min().item()) >= 35:
                action_embodiment_id = embodiment_id - 35
        if action_embodiment_id.numel() and (
            int(action_embodiment_id.min().item()) < 0
            or int(action_embodiment_id.max().item()) >= num_embodiments
        ):
            raise ValueError(
                f"embodiment_id values {embodiment_id.detach().cpu().tolist()} map to "
                f"category ids {action_embodiment_id.detach().cpu().tolist()}, outside "
                f"[0, {num_embodiments})."
            )
        action_features = self.action_encoder(action, timestep_action, action_embodiment_id)
        action_length = action_features.shape[1]
        if state is not None and state.shape[1] > 0 and self.num_state_per_block > 0:
            state_features = self.state_encoder(state, action_embodiment_id)
        else:
            state_features = action_features.new_empty(
                action_features.shape[0], 0, action_features.shape[-1]
            )

        if motion is not None and self.motion_patch_embedding is not None:
            m = motion.to(dtype=x.dtype)
            motion_emb = self.motion_patch_embedding(m)
            motion_grid_size = torch.tensor(
                motion_emb.shape[2:],
                dtype=torch.long,
                device=m.device,
            )
            self._motion_grid_size = motion_grid_size
            motion_features = motion_emb.flatten(start_dim=2).transpose(1, 2)
            motion_length = motion_features.shape[1]
            motion_freqs = self._create_motion_freqs(
                grid_size=motion_grid_size,
                start_motion_block=0,
            )
            register_parts = [motion_features, action_features]
        else:
            motion_features = None
            motion_length = 0
            self._motion_grid_size = None
            motion_freqs = None
            register_parts = [action_features]
        if state_features.shape[1] > 0:
            register_parts.append(state_features)
        extra_register = torch.cat(register_parts, dim=1)

        action_register_length = extra_register.shape[1]
        x = torch.cat([x, extra_register], dim=1)
    else:
        action_features = None
        action_length = None
        motion_features = None
        motion_freqs = None
        motion_length = 0
        state_features = None
        action_register_length = None
        self._motion_grid_size = None

    timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)
    timestep_original = timestep.clone()

    if motion_condition_length > 0:
        timestep_motion_condition = torch.zeros(
            B,
            motion_condition_length,
            device=timestep.device,
            dtype=timestep.dtype,
        )
        timestep = torch.cat([timestep, timestep_motion_condition], dim=1)

    if action is not None:
        assert timestep_action is not None
        assert state_features is not None
        if state_features.shape[1] > 0 and self.num_state_per_block > 0:
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
        else:
            timestep_state = timestep_action.new_empty(
                timestep_action.shape[0], 0
            )
        if motion is not None and timestep_motion is not None and motion_length > 0:
            assert self._motion_grid_size is not None
            timestep_motion_exp = self._expand_motion_timestep_to_tokens(
                timestep_motion=timestep_motion,
                motion_grid_size=self._motion_grid_size,
            )
            timestep_parts = [timestep, timestep_motion_exp, timestep_action]
        else:
            timestep_parts = [timestep, timestep_action]
        if timestep_state.shape[1] > 0:
            timestep_parts.append(timestep_state)
        timestep = torch.cat(timestep_parts, dim=1)

    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x)
    )
    e = e.unflatten(dim=0, sizes=(B, -1))
    e0 = self.time_projection(e)
    e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

    assert context.shape[1] == self.text_len
    context = self.text_embedding(context)
    if clip_feature is not None:
        clip_embedding = self.img_emb(clip_feature)
        context = torch.cat([clip_embedding, context], dim=1)

    if clean_x is not None:
        if y is not None and self.concat_first_frame_latent:
            clean_x = torch.cat([clean_x, y.to(dtype=clean_x.dtype)], dim=1)
        clean_x = self.patch_embedding(clean_x)
        clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
        assert clean_x.shape[1] == seq_len

        x = torch.cat([clean_x, x], dim=1)

        if aug_t is None:
            aug_t = torch.zeros_like(timestep_original)
        assert aug_t is not None

        e_clean = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x)
        )
        e_clean = e_clean.unflatten(dim=0, sizes=timestep_original.shape)
        e0_clean = self.time_projection(e_clean)
        e0_clean = e0_clean.unflatten(dim=2, sizes=(6, self.dim))
        e0 = torch.cat([e0_clean, e0], dim=1)

    motion_tokens_per_block = (
        self.motion_token_length_per_block if motion is not None else 0
    )
    kwargs = {
        "e": e0,
        "freqs": freqs,
        "freqs_action": self.freqs_action,
        "freqs_state": self.freqs_state,
        "action_register_length": action_register_length,
        "context": context,
        "is_tf": clean_x is not None,
        "motion_length": motion_length,
        "motion_tokens_per_block": motion_tokens_per_block,
        "freqs_motion": motion_freqs,
        "motion_condition_length": motion_condition_length,
        "freqs_motion_condition": motion_condition_freqs,
    }

    def create_custom_forward(module):
        def custom_forward(*inputs, **kwargs):
            outputs, updated_kv_cache = module(*inputs, **kwargs)
            assert updated_kv_cache is None
            return outputs

        return custom_forward

    for block in self.blocks:
        use_ckpt = torch.is_grad_enabled() and self.gradient_checkpointing

        if use_ckpt:
            ckpt_use_reentrant = getattr(
                self, "gradient_checkpointing_use_reentrant", True
            )

            if ckpt_use_reentrant:
                x, _ = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    e0,
                    freqs,
                    self.freqs_action,
                    self.freqs_state,
                    action_register_length,
                    context,
                    None,  # kv_cache
                    None,  # crossattn_cache
                    0,  # current_start_frame
                    clean_x is not None,  # is_tf
                    None,  # layer_index
                    motion_length,
                    motion_tokens_per_block,
                    motion_freqs,
                    motion_condition_length,
                    motion_condition_freqs,
                    use_reentrant=True,
                )
            else:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
        else:
            x, _ = block(x, **kwargs)

    if clean_x is not None:
        x = x[:, clean_x.shape[1] :]

    if (
        action is not None
        and motion is not None
        and self.motion_head is not None
        and motion_length > 0
    ):
        motion_start = seq_len + motion_condition_length
        motion_end = motion_start + motion_length
        motion_tokens = x[:, motion_start:motion_end]
        e_motion = e[:, motion_start:motion_end]
        motion_noise_pred = self.motion_head(motion_tokens, e_motion.unsqueeze(2))
        motion_noise_pred = self.unpatchify_motion(
            motion_noise_pred,
            self._motion_grid_size,
        )
    else:
        motion_noise_pred = None

    if action is not None:
        action_start = seq_len + motion_condition_length
        if motion is not None:
            action_start += motion_length
        action_noise_pred = x[:, action_start : action_start + action_length]
        action_noise_pred = self.action_decoder(action_noise_pred, action_embodiment_id)
    else:
        action_noise_pred = None

    x_video = x[:, :seq_len]
    e_video = e[:, :seq_len]
    x_video = self.head(x_video, e_video.unsqueeze(2))
    video_noise_pred = self.unpatchify(x_video, grid_size)
    return video_noise_pred, action_noise_pred, motion_noise_pred
