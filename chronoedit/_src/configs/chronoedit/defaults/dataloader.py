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

from hydra.core.config_store import ConfigStore

from chronoedit._src.datasets.chronoedit_dataset.chronoedit_dataloader import get_chronoedit_dataloader, get_chronoedit_multiple_dataloader

def edit_register_dataloader():
    cs = ConfigStore.instance()

    cs.store(
        group="data_train",
        package="dataloader_train",        
        name="chronoedit_data_image",
        node=get_chronoedit_dataloader(
            dataset_name="chronoedit_data_image",
            num_workers=6,
            prefetch_factor=4
        )
    )

    cs.store(
        group="data_train",
        package="dataloader_train",        
        name="chronoedit_data_video",
        node=get_chronoedit_dataloader(
            dataset_name="chronoedit_data_video",
            num_workers=6,
            prefetch_factor=4
        )
    )
