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

from chronoedit._src.configs.common.mock_data import (
    MOCK_DATA_IMAGE_ONLY_CONFIG,
    MOCK_DATA_INTERLEAVE_CONFIG,
    MOCK_DATA_VIDEO_ONLY_CONFIG,
)




def register_training_and_val_data():
    cs = ConfigStore()
    cs.store(group="data_train", package="dataloader_train", name="mock", node=MOCK_DATA_INTERLEAVE_CONFIG)
    cs.store(group="data_train", package="dataloader_train", name="mock_image", node=MOCK_DATA_IMAGE_ONLY_CONFIG)
    cs.store(group="data_train", package="dataloader_train", name="mock_video", node=MOCK_DATA_VIDEO_ONLY_CONFIG)
    cs.store(group="data_val", package="dataloader_val", name="mock", node=MOCK_DATA_INTERLEAVE_CONFIG)


def register_training_and_val_data_no_cosmos():
    cs = ConfigStore()
    cs.store(group="data_train", package="dataloader_train", name="mock", node=MOCK_DATA_INTERLEAVE_CONFIG)
    cs.store(group="data_train", package="dataloader_train", name="mock_image", node=MOCK_DATA_IMAGE_ONLY_CONFIG)
    cs.store(group="data_train", package="dataloader_train", name="mock_video", node=MOCK_DATA_VIDEO_ONLY_CONFIG)
    cs.store(group="data_val", package="dataloader_val", name="mock", node=MOCK_DATA_INTERLEAVE_CONFIG)