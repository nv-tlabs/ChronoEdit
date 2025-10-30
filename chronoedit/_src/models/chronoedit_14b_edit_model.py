# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Tuple, Dict

import attrs
import torch
from torch import Tensor

from chronoedit._src.models.wan_t2v_model import DataType, T2VCondition, T2VModelConfig, WANDiffusionModel
from chronoedit._src.utils.misc import sync_timer
from chronoedit._src.utils.context_parallel import broadcast_split_tensor, cat_outputs_cp
from  chronoedit._ext.imaginaire.utils import misc

WAN2PT1_I2V_COND_LATENT_KEY = "i2v_WAN2PT1_cond_latents"


@attrs.define(slots=False)
class EditModelConfig(T2VModelConfig):

    is_video_prior: bool = False


class I2V_Edit_Wan2pt1Model(WANDiffusionModel):
    def __init__(self, config: T2VModelConfig):
        # Note that I2V config.shift has better value {"480p": 3.0, "720p": 5.0}
        super().__init__(config)

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]) -> Tuple[Tensor, T2VCondition]:
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        # Latent state
        raw_state = data_batch[self.input_image_key if is_image_batch else self.input_data_key]

        last_frame = raw_state[:, :, -1:, :, :]  # Shape: [1, 3, 1, 480, 832]
        # Repeat the last frame 4 times
        last_frame_repeated = last_frame.repeat(1, 1, 4, 1, 1)  # Shape: [1, 3, 4, 480, 832]
        # Concatenate with the original first tensor
        raw_state_edit = torch.cat([raw_state[:, :, :1, :, :], last_frame_repeated], dim=2)  # Shape: [1, 3, 5, 480, 832]
        latent_state = self.encode(raw_state_edit).contiguous().float()

        # try to get is_video_prior from data_batch
        is_video_prior = data_batch.get("is_video_prior", False)
        if is_video_prior:
            raw_state_video = raw_state[:, :, :-1, :, :]  # Shape: [1, 3, 25, 480, 832]
            latent_state_video = self.encode(raw_state_video).contiguous().float()
            latent_state = torch.cat([latent_state_video, latent_state[:, :, 1:, :, :]], dim=2)
            raw_state = torch.cat([raw_state_video, last_frame_repeated], dim=2)
        else:
            raw_state = raw_state_edit

        if WAN2PT1_I2V_COND_LATENT_KEY not in data_batch:
            conditional_content = torch.zeros_like(raw_state).to(**self.tensor_kwargs)
            if not is_image_batch:
                conditional_content[:, :, 0] = raw_state[:, :, 0]

            data_batch[WAN2PT1_I2V_COND_LATENT_KEY] = self.encode(conditional_content).contiguous()

        # Condition
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        return raw_state, latent_state, condition

    @sync_timer("WANDiffusionModel: generate_samples_from_batch")
    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            num_steps (int): number of steps for the diffusion process
            solver_option (str): differential equation solver option, default to "2ab"~(mulitstep solver)
        """

        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        noise = (
                misc.arch_invariant_rand(
                    (n_sample,) + tuple(state_shape),
                    torch.float32,
                    self.tensor_kwargs["device"],
                    seed,
                )
        )

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        self.sample_scheduler.set_timesteps(
            num_steps, device=self.tensor_kwargs["device"], shift=shift)

        timesteps = self.sample_scheduler.timesteps


        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)
        latents = noise

        if self.net.is_context_parallel_enabled:
            latents = broadcast_split_tensor(latents, seq_dim=2, process_group=self.get_context_parallel_group())

        with sync_timer(f"WANDiffusionModel: generate_samples_from_batch: {num_steps} diffusion_steps"):
            for _, t in enumerate(timesteps):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                velocity_field_pred = x0_fn(latent_model_input, timestep.unsqueeze(0)) # velocity field
                temp_x0 = self.sample_scheduler.step(
                    velocity_field_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = temp_x0.squeeze(0)

                if (
                    "drop_video_step" in kwargs
                    and kwargs["drop_video_step"] is not None
                    and t == timesteps[kwargs["drop_video_step"]]
                ):

                    if self.net.is_context_parallel_enabled:
                        latents = cat_outputs_cp(latents, seq_dim=2, cp_group=self.get_context_parallel_group())
                    
                    latents = latents[:,:,[0, -1]]

                    if self.net.is_context_parallel_enabled:
                        latents = broadcast_split_tensor(latents, seq_dim=2, process_group=self.get_context_parallel_group())
                    
                    if self.sample_scheduler.model_outputs != []:
                        for i in range(len(self.sample_scheduler.model_outputs)):
                            if latents.shape[-3] != self.sample_scheduler.model_outputs[i].shape[-3]:
                                if self.net.is_context_parallel_enabled:
                                    self.sample_scheduler.model_outputs[i] = cat_outputs_cp(self.sample_scheduler.model_outputs[i], seq_dim=3, cp_group=self.get_context_parallel_group())
                                self.sample_scheduler.model_outputs[i] = self.sample_scheduler.model_outputs[i][:,:,:,[0, -1]]
                                if self.net.is_context_parallel_enabled:
                                    self.sample_scheduler.model_outputs[i] = broadcast_split_tensor(self.sample_scheduler.model_outputs[i], seq_dim=3, process_group=self.get_context_parallel_group())

                        if self.sample_scheduler.last_sample is not None:
                            self.sample_scheduler.last_sample = latents
                    
                    data_batch["i2v_WAN2PT1_cond_latents"] = data_batch["i2v_WAN2PT1_cond_latents"][:,:,[0, -1]]
                    data_batch["video"] = data_batch["video"][:,:,[0, -1]]
                    data_batch["is_video_prior"] = False

                    print(f"rank {torch.distributed.get_rank()}:", data_batch["video"].shape, data_batch["i2v_WAN2PT1_cond_latents"].shape, latents.shape, data_batch["is_video_prior"])

                    x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)

        if self.net.is_context_parallel_enabled:
            latents = cat_outputs_cp(latents, seq_dim=2, cp_group=self.get_context_parallel_group())

        return latents