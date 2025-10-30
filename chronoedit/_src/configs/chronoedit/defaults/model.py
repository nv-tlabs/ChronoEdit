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

from  chronoedit._ext.imaginaire.lazy_config import LazyCall as L
from chronoedit._src.models.chronoedit_14b_edit_model import I2V_Edit_Wan2pt1Model, EditModelConfig


ddp_wan2pt1_edit_config = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(I2V_Edit_Wan2pt1Model)(
        config=EditModelConfig(
            state_t=20,
        ),
        _recursive_=False,
    ),
)

fsdp_wan2pt1_edit_config = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(I2V_Edit_Wan2pt1Model)(
        config=EditModelConfig(
            fsdp_shard_size=8,
            state_t=20,
        ),
        _recursive_=False,
    ),
)




def edit_register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="ddp_wan2pt1_edit", node=ddp_wan2pt1_edit_config)
    cs.store(group="model", package="_global_", name="fsdp_wan2pt1_edit", node=fsdp_wan2pt1_edit_config)
