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


import math
import os
from contextlib import nullcontext
from functools import partial
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as torchvision_F
import wandb
from einops import rearrange, repeat
from megatron.core import parallel_state
from omegaconf import OmegaConf, DictConfig

from  chronoedit._ext.imaginaire.callbacks.every_n import EveryN
from  chronoedit._ext.imaginaire.model import ImaginaireModel
from  chronoedit._ext.imaginaire.utils import distributed, log, misc
from  chronoedit._ext.imaginaire.utils.easy_io import easy_io
from  chronoedit._ext.imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0
from  chronoedit._ext.imaginaire.visualize.video import save_img_or_video


# use first two rank to generate some images for visualization
def resize_image(image: torch.Tensor, size: int = 1024) -> torch.Tensor:
    _, h, w = image.shape
    ratio = size / max(h, w)
    new_h, new_w = int(ratio * h), int(ratio * w)
    return torchvision_F.resize(image, (new_h, new_w))


def is_primitive(value):
    return isinstance(value, (int, float, str, bool, type(None)))


def convert_to_primitive(value):
    if isinstance(value, (list, tuple)):
        return [convert_to_primitive(v) for v in value if is_primitive(v) or isinstance(v, (list, dict))]
    elif isinstance(value, dict):
        return {k: convert_to_primitive(v) for k, v in value.items() if is_primitive(v) or isinstance(v, (list, dict))}
    elif is_primitive(value):
        return value
    else:
        return "non-primitive"  # Skip non-primitive types


class EveryNDrawSample(EveryN):
    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        fix_batch_fp: Optional[str] = None,
        n_x0_level: int = 4,
        n_viz_sample: int = 3,
        n_sample_to_save: int = 128,
        num_sampling_step: int = 35,
        guidance: List[float] = [3.0, 7.0, 9.0, 13.0],
        is_x0: bool = True,
        is_sample: bool = True,
        save_s3: bool = False,
        is_ema: bool = False,
        show_all_frames: bool = False,
        fps: int = 16,
        run_at_start: bool = False,
        prompt_type: str = "umt5_xxl",
    ):
        super().__init__(every_n, step_size, run_at_start=run_at_start)
        self.fix_batch = fix_batch_fp
        self.n_x0_level = n_x0_level
        self.n_viz_sample = n_viz_sample
        self.n_sample_to_save = n_sample_to_save
        self.save_s3 = save_s3
        self.is_x0 = is_x0
        self.is_sample = is_sample
        self.name = self.__class__.__name__
        self.is_ema = is_ema
        self.show_all_frames = show_all_frames
        self.guidance = guidance
        self.num_sampling_step = num_sampling_step
        self.rank = distributed.get_rank()
        self.fps = fps
        self.prompt_type = prompt_type
        self.use_negative_prompt = False # always false


    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"Callback: local_dir: {self.local_dir}")

        if parallel_state.is_initialized():
            self.data_parallel_id = parallel_state.get_data_parallel_rank()
        else:
            self.data_parallel_id = self.rank

        if self.fix_batch is not None:
            if isinstance(self.fix_batch, DictConfig):
                self.fix_batch = OmegaConf.to_container(self.fix_batch, resolve=True)
            if isinstance(self.fix_batch, str):
                this_load_path = self.fix_batch.format(self.data_parallel_id)
                backend_args = {}
            elif isinstance(self.fix_batch, dict):
                this_load_path = self.fix_batch["path"].format(self.data_parallel_id)
                backend_args = self.fix_batch["backend_args"]
            else:
                raise ValueError(f"Invalid fix_batch type: {type(self.fix_batch)}")
            with misc.timer(f"loading fix_batch {this_load_path}"):
                fix_batch = misc.to(easy_io.load(this_load_path, backend_args=backend_args), "cpu")
                log.info(f"loading fix_batch {this_load_path} to rank {self.data_parallel_id}", rank0_only=False)
                self.fix_batch = fix_batch



    @misc.timer("EveryNDrawSample: x0")
    @torch.no_grad()
    def x0_pred(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.fix_batch is not None:
            data_batch = misc.to(self.fix_batch, **model.tensor_kwargs)
        tag = "ema" if self.is_ema else "reg"

        log.debug("starting data and condition model", rank0_only=False)
        raw_data, x0, condition = model.get_data_and_condition(data_batch)
        _, condition, x0, _ = model.broadcast_split_for_model_parallelsim(None, condition, x0, None)

        log.debug("done data and condition model", rank0_only=False)
        batch_size = x0.shape[0]
        sigmas = np.exp(
            np.linspace(math.log(model.sde.sigma_min), math.log(model.sde.sigma_max), self.n_x0_level + 1)[1:]
        )

        to_show = []
        generator = torch.Generator(device="cuda")
        generator.manual_seed(0)
        random_noise = torch.randn(*x0.shape, generator=generator, **model.tensor_kwargs)
        _ones = torch.ones(batch_size, **model.tensor_kwargs)
        mse_loss_list = []
        for _, sigma in enumerate(sigmas):
            x_sigma = sigma * random_noise + x0
            log.debug(f"starting denoising {sigma}", rank0_only=False)
            sample = model.denoise(x_sigma, _ones * sigma, condition).x0
            log.debug(f"done denoising {sigma}", rank0_only=False)
            mse_loss = distributed.dist_reduce_tensor(F.mse_loss(sample, x0))
            mse_loss_list.append(mse_loss)
            if hasattr(model, "decode"):
                sample = model.decode(sample)
            to_show.append(sample.float().cpu())
        to_show.append(
            raw_data.float().cpu(),
        )

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_x0_Iter{iteration:09d}"

        local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
        return local_path, torch.tensor(mse_loss_list).cuda(), sigmas

    @torch.no_grad()
    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.is_ema:
            if not model.config.ema.enabled:
                return
            context = partial(model.ema_scope, "every_n_sampling")
        else:
            context = nullcontext
        tag = "ema" if self.is_ema else "reg"
        sample_counter = getattr(trainer, "sample_counter", iteration)
        batch_info = {
            "data": {
                k: convert_to_primitive(v)
                for k, v in data_batch.items()
                if is_primitive(v) or isinstance(v, (list, dict))
            },
            "sample_counter": sample_counter,
            "iteration": iteration,
        }
        if is_tp_cp_pp_rank0():
            if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
                easy_io.dump(
                    batch_info,
                    f"s3://rundir/{self.name}/BatchInfo_ReplicateID{self.data_parallel_id:04d}_Iter{iteration:09d}.json",
                )

        log.debug("entering, every_n_impl", rank0_only=False)
        with context():
            log.debug("entering, ema", rank0_only=False)
            # we only use rank0 and rank to generate images and save
            # other rank run forward pass to make sure it works for FSDP
            log.debug("entering, fsdp", rank0_only=False)
            if self.is_x0:
                log.debug("entering, x0_pred", rank0_only=False)
                x0_img_fp, mse_loss, sigmas = self.x0_pred(
                    trainer,
                    model,
                    data_batch,
                    output_batch,
                    loss,
                    iteration,
                )
                log.debug("done, x0_pred", rank0_only=False)
                if self.save_s3 and self.rank == 0:
                    easy_io.dump(
                        {
                            "mse_loss": mse_loss.tolist(),
                            "sigmas": sigmas.tolist(),
                            "iteration": iteration,
                        },
                        f"s3://rundir/{self.name}/{tag}_MSE_Iter{iteration:09d}.json",
                    )
            if self.is_sample:
                log.debug("entering, sample", rank0_only=False)
                sample_img_fp = self.sample(
                    trainer,
                    model,
                    data_batch,
                    output_batch,
                    loss,
                    iteration,
                )
                log.debug("done, sample", rank0_only=False)
            if self.fix_batch is not None:
                misc.to(self.fix_batch, "cpu")

            log.debug("waiting for all ranks to finish", rank0_only=False)
            dist.barrier()
        torch.cuda.empty_cache()

    @misc.timer("EveryNDrawSample: sample")
    def sample(self, trainer, model, data_batch, output_batch, loss, iteration):
        """
        Args:
            skip_save: to make sure FSDP can work, we run forward pass on all ranks even though we only save on rank 0 and 1
        """
        if self.fix_batch is not None:
            data_batch = misc.to(self.fix_batch, **model.tensor_kwargs)
        
        # # save data_batch locally for loading
        # if distributed.get_rank() == 0:
        #     # iteration starts from 1
        #     path = f"outputs/image_cosmos_alpamayo_20250526_video_cosmos_alpamayo_20252905_finetune_2.3M_s3/EveryNDrawSample_data_batch_wan_{iteration - 1}.pkl"
        #     easy_io.dump(misc.to(data_batch, "cpu"), path)
        #     print(f"saved data_batch to {path}")
        # dist.barrier()
        # return None

        tag = "ema" if self.is_ema else "reg"
        raw_data, x0, condition = model.get_data_and_condition(data_batch)

        to_show = []

        for guidance in self.guidance:
            sample = model.generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                # make sure no mismatch and also works for cp
                state_shape=x0.shape[1:],
                n_sample=x0.shape[0],
                num_steps=self.num_sampling_step,
                is_negative_prompt=False
            )
            if hasattr(model, "decode"):
                sample = model.decode(sample)
            to_show.append(sample.float().cpu())

        to_show.append(raw_data.float().cpu())

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}"

        batch_size = x0.shape[0]
        if is_tp_cp_pp_rank0():
            local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)

        for ii, item in  enumerate(to_show[:-1]):
            base_fp_wo_ext = (
                f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}_single_{ii}"
            )
            if is_tp_cp_pp_rank0():
                _ = self.run_save([item], batch_size, base_fp_wo_ext)

        base_fp_wo_ext = (
            f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}_single_gt"
        )
        if is_tp_cp_pp_rank0():
            _ = self.run_save([to_show[-1]], batch_size, base_fp_wo_ext)

        return None

    def run_save(self, to_show, batch_size, base_fp_wo_ext) -> Optional[str]:
        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [n, b, c, t, h, w]
        is_single_frame = to_show.shape[3] == 1
        n_viz_sample = min(self.n_viz_sample, batch_size)

        # ! we only save first n_sample_to_save video!
        if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
            save_img_or_video(
                rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
                f"s3://rundir/{self.name}/{base_fp_wo_ext}",
                fps=self.fps,
            )

        file_base_fp = f"{base_fp_wo_ext}_resize.jpg"
        local_path = f"{self.local_dir}/{file_base_fp}"

        if self.rank == 0 and wandb.run:
            if is_single_frame:  # image case
                to_show = rearrange(
                    to_show[:, :n_viz_sample],
                    "n b c t h w -> t c (n h) (b w)",
                )
                image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
                # resize so that wandb can handle it
                torchvision.utils.save_image(resize_image(image_grid, 1024), local_path, nrow=1, scale_each=True)
            else:
                to_show = to_show[:, :n_viz_sample]  # [n, b, c, 3, h, w]
                if not self.show_all_frames:
                    # resize 3 frames frames so that we can display them on wandb
                    _T = to_show.shape[3]
                    three_frames_list = [0, _T // 2, _T - 1]
                    to_show = to_show[:, :, :, three_frames_list]
                    log_image_size = 1024
                else:
                    log_image_size = 512 * to_show.shape[3]
                to_show = rearrange(
                    to_show,
                    "n b c t h w -> 1 c (n h) (b t w)",
                )

                # resize so that wandb can handle it
                image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
                torchvision.utils.save_image(
                    resize_image(image_grid, log_image_size), local_path, nrow=1, scale_each=True
                )

            return local_path
        return None
