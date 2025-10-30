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

import gc

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from chronoedit._ext.imaginaire.utils import log, distributed
from chronoedit._src.datasets.chronoedit_dataset.config import DATASET_CONFIG
from chronoedit._src.datasets.chronoedit_dataset.dataloader_utils import dict_collation_fn
from chronoedit._src.datasets.chronoedit_dataset.instantiate_utils import instantiate_from_config
from chronoedit._src.datasets.chronoedit_dataset.joint_dataloader import IterativeEditDataLoader

try:
    from megatron.core import parallel_state
    USE_MEGATRON = True
except ImportError:
    parallel_state = None  # type: ignore
    USE_MEGATRON = False

import omegaconf
from omegaconf import OmegaConf

from chronoedit._ext.imaginaire.lazy_config import LazyCall as L
    

def get_chronoedit_multiple_dataloader(
    dataset_list: list[str],
    dataset_weight_list: list[float],
    shuffle=True,
    num_workers=4,
    prefetch_factor=4,
    batch_size=1,
) -> omegaconf.dictconfig.DictConfig:
    dataloader_dict = {}

    for dataset_idx in range(len(dataset_list)):
        dataset_name = dataset_list[dataset_idx]
        dataset_weight = dataset_weight_list[dataset_idx]

        dataloader_dict[dataset_name] = {
            "dataloader": L(MyDataLoader)(
                dataset=L(get_chronoedit_dataset)(
                    dataset_name=dataset_name,
                ),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                prefetch_factor=prefetch_factor,
            ),
            "ratio": dataset_weight,
        }
    return L(IterativeEditDataLoader)(dataloaders=dataloader_dict)

def get_chronoedit_dataloader(
    dataset_name: str,
    shuffle=True,
    num_workers=4,
    prefetch_factor=4,
    batch_size=1,
) -> omegaconf.dictconfig.DictConfig:
    return L(MyDataLoader)(
        dataset=L(get_chronoedit_dataset)(
            dataset_name=dataset_name,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        prefetch_factor=prefetch_factor,
    )

class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size: int = 1, *args, **kw):
        dataset_obj = dataset.build_dataset()
        if "dataloaders" in kw:
            kw.pop("dataloaders")
        super().__init__(dataset_obj, batch_size, collate_fn=dict_collation_fn, *args, **kw)

def get_chronoedit_dataset(dataset_name="re10k", **kwargs):
    return EditDataset(dataset_name, **kwargs)

class EditDataset:
    def __init__(self, dataset_name, **kwargs):
        self.dataset_name = dataset_name
        self.dataset_config = OmegaConf.create(DATASET_CONFIG[dataset_name])
        extra_config = OmegaConf.create(kwargs)
        self.dataset_config = OmegaConf.merge(self.dataset_config, extra_config)

    def build_dataset(self):
        config_dict = OmegaConf.to_container(self.dataset_config, resolve=True)
        return InfiniteDataVerse(**config_dict)

class InfiniteDataVerse:
    def __init__(
        self,
        dataset_cfg,
        batch_size=1,
        fps=24,
        is_video_prior=False,
    ):
        self.dataset = instantiate_from_config(dataset_cfg)
        self.n_data = len(self.dataset)

        self.fps = fps
        self.is_video_prior = is_video_prior

        # Split the data by node, make sure each node has different data sample
        # Ranks of the same pp/tp/cp group will have the same dp rank and thus share the same group id.
        if USE_MEGATRON and (parallel_state is not None) and parallel_state.is_initialized():
            dp_group_id = parallel_state.get_data_parallel_rank()
            dp_world_size = parallel_state.get_data_parallel_world_size()
            log.critical(
                f"Using parallelism size CP :{parallel_state.get_context_parallel_world_size()}, "
                + f"TP :{parallel_state.get_tensor_model_parallel_world_size()} for video dataset, "
                + f"DP: {dp_group_id}, DP World size: {dp_world_size}"
            )
        else:
            dp_world_size = 1
            dp_group_id = 0
        self.n_data_per_node = self.n_data // dp_world_size
        self.data_start_idx = dp_group_id * self.n_data_per_node
        self.dp_group_id = dp_group_id
        
        # Make an infinite loop
        maximum_iter = 5000000 * batch_size  # TODO:a hack to create infinite loop
        self.multiplier = maximum_iter // self.n_data_per_node

        self.norm_image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

    def __len__(self):
        return self.multiplier * self.n_data_per_node

    def transform_data(self, sampled_images):
        sampled_images = self.norm_image(sampled_images)  # (N, C, H, W)
        sample = {
            "video": sampled_images.permute(1, 0, 2, 3).contiguous(),  # (C, N, H, W) format for cosmos
            "is_preprocessed": True,
            "is_video_prior": self.is_video_prior,
        }
        return sample

    def __getitem__(self, idx):
        rank = distributed.get_rank()
        data_idx = (idx % self.n_data_per_node) + self.data_start_idx
        assert data_idx < self.n_data
        
        try:
            data = self.dataset.__getitem__(data_idx)
        except Exception as e:
            print(
                f"RANK {rank}: Data reading ERROR for video_idx {data_idx}. Skip and continue load the next Video."
            )
            return self.__getitem__((idx + 1) % len(self))

        frames = [torch.from_numpy(np.array(image)) for image in data["video"]]
        data["video"] = torch.stack(frames)  # (T, H, W, C)
        data["video"] = data["video"].permute(0, 3, 1, 2) / 255.0

        sample = self.transform_data(data["video"].clone())
        sample["__key__"] = "%s" % (data["key"])

        umt5_embed = data["umt5"]
        dummy_text_embedding = torch.zeros(512, 4096)
        dummy_text_mask = torch.zeros(512)
        n_text = umt5_embed.sum(axis=-1).nonzero().shape[0]
        dummy_text_embedding[:n_text] = umt5_embed[:n_text]
        dummy_text_mask[:n_text] = 1
        sample["t5_text_embeddings"] = dummy_text_embedding
        sample["t5_text_mask"] = dummy_text_mask

        sample["num_frames"] = data["video"].shape[0]
        actual_crop_size = sample["video"].shape[-2:]
        sample["image_size"] = torch.from_numpy(np.asarray(actual_crop_size))
        sample["fps"] = self.fps
        sample["padding_mask"] = torch.zeros(1, actual_crop_size[0], actual_crop_size[1])
        sample["caption"] = data["prompt"]

        gc.collect(1)
        return sample

# python -m chronoedit._src.datasets.chronoedit_dataset.chronoedit_dataloader
if __name__ == "__main__":
    dataset_name = "chronoedit_data_image"
    CONFIG = DATASET_CONFIG[dataset_name]
    print(CONFIG)
    dataset = InfiniteDataVerse(**CONFIG)
    for i in range(1):
        data = dataset[i]
        for key, value in data.items():
            if torch.is_tensor(value):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {value}")