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


import json
from hydra.core.config_store import ConfigStore
from typing import Optional

from  chronoedit._ext.imaginaire.lazy_config import LazyCall as L
from chronoedit._ext.callbacks.every_n_draw_sample import EveryNDrawSample
import torch
from einops import rearrange
from  chronoedit._ext.imaginaire.utils import misc
from  chronoedit._ext.imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0
import torchvision
from  chronoedit._ext.imaginaire.visualize.video import save_img_or_video
from  chronoedit._ext.imaginaire.model import ImaginaireModel


class EditEveryNDrawSample(EveryNDrawSample):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        super().on_train_start(model, iteration)

    @misc.timer("EveryNDrawSample: sample")
    def sample(self, trainer, model, data_batch, output_batch, loss, iteration):
        """
        Args:
            skip_save: to make sure FSDP can work, we run forward pass on all ranks even though we only save on rank 0 and 1
        """
        if self.fix_batch is not None:
            data_batch = misc.to(self.fix_batch, **model.tensor_kwargs)

        tag = "ema" if self.is_ema else "reg"
        raw_data, x0, condition = model.get_data_and_condition(data_batch)

        is_video_prior = data_batch.get("is_video_prior", False)
        to_show = []
        for guidance in self.guidance:
            sample = model.generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                # make sure no mismatch and also works for cp
                state_shape=x0.shape[1:],
                n_sample=x0.shape[0],
                num_steps=self.num_sampling_step,
                is_negative_prompt=False,
            )

            if is_video_prior:
                first_frame_sample = sample[:, :, :1, :, :]
                last_frame_sample = sample[:, :, -1:, :, :]
                sample_edit = torch.cat([first_frame_sample, last_frame_sample], dim=2)
                sample_edit = model.decode(sample_edit).contiguous()

                sample_video = sample[:, :, :-1, :, :]
                sample_video = model.decode(sample_video).contiguous()

                sample = torch.cat([sample_video, sample_edit[:, :, 1:, :, :]], dim=2)
            else:
                if hasattr(model, "decode"):
                    sample = model.decode(sample)
            to_show.append(sample.float().cpu())


        to_show.append(raw_data.float().cpu())

        # visualize input video
        if "hint_key" in data_batch:
            hint = data_batch[data_batch["hint_key"]]
            for idx in range(0, hint.size(1), 3):
                x_rgb = hint[:, idx : idx + 3]
                to_show.append(x_rgb.float().cpu())

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}"

        batch_size = x0.shape[0]
        if is_tp_cp_pp_rank0():
            local_path = self.run_save(to_show, batch_size, base_fp_wo_ext, is_video_prior)

            # save caption
            if "caption" in data_batch or "ai_caption" in data_batch:
                caption = data_batch["caption"] if "caption" in data_batch else data_batch["ai_caption"]
                caption_path = f"{self.local_dir}/{base_fp_wo_ext}_caption.json"
                caption_dict = {}
                caption_dict["caption"] = caption
                caption_dict["path"] = data_batch["__key__"]
                with open(caption_path, "w") as f:
                    json.dump(caption_dict, f)

            return local_path
        return None

    def run_save(self, to_show, batch_size, base_fp_wo_ext, is_video_prior=False) -> Optional[str]:
        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [n, b, c, t, h, w]
        is_single_frame = to_show.shape[3] == 1
        n_viz_sample = min(self.n_viz_sample, batch_size)

        # ! we only save first n_sample_to_save video!
        # if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
        if is_video_prior:
            local_path = f"{self.local_dir}/{base_fp_wo_ext}"
            save_img_or_video(
                rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
                local_path,
                fps=self.fps,
            )
        else:
            file_base_fp = f"{base_fp_wo_ext}.jpg"
            local_path = f"{self.local_dir}/{file_base_fp}"

            to_show = to_show[:, :n_viz_sample]  # [n, b, c, 3, h, w]
            if not self.show_all_frames:
                # resize 2 frames frames so that we can display them on wandb
                _T = to_show.shape[3]
                three_frames_list = [0, _T - 1]
                to_show = to_show[:, :, :, three_frames_list]
                log_image_size = 1024
            else:
                log_image_size = 512 * to_show.shape[3]
            to_show = rearrange(
                to_show,
                "n b c t h w -> 1 c (n h) (b t w)",
            )

            # ! do not resize the image grid
            image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
            torchvision.utils.save_image(
                image_grid, local_path, nrow=1, scale_each=True
            )

        return local_path

VIZ_ONLINE_SAMPLING_EDIT_CALLBACKS = dict(
    every_n_sample_reg=L(EditEveryNDrawSample)(
        every_n=5000,
        save_s3="${upload_reproducible_setup}",
    ),
)

def edit_register_callbacks():
    cs = ConfigStore.instance()

    cs.store(
        group="callbacks", package="trainer.callbacks", name="viz_online_sampling_edit", node=VIZ_ONLINE_SAMPLING_EDIT_CALLBACKS
    )
