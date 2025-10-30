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
from chronoedit._src.models.wan_t2v_model import WANDiffusionModel as T2VModel
from chronoedit._src.models import WANMOEDiffusionModel as T2VMOEModel

from chronoedit._src.models.wan_t2v_model import WANI2VDiffusionModel as I2VModel
from chronoedit._src.models.wan_t2v_model import T2VModelConfig

DDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(T2VModel)(
        config=T2VModelConfig(),
        _recursive_=False,
    ),
)


FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(T2VModel)(
        config=T2VModelConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

FSDP_MOE_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(T2VMOEModel)(
        config=T2VModelConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)


I2V_DDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(I2VModel)(
        config=T2VModelConfig(),
        _recursive_=False,
    ),
)

I2V_FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(I2VModel)(
        config=T2VModelConfig(
            fsdp_shard_size=8,
        ),
    )
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="ddp", node=DDP_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp", node=FSDP_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp_moe", node=FSDP_MOE_CONFIG)

    cs.store(group="model", package="_global_", name="i2v_ddp", node=I2V_DDP_CONFIG)
    cs.store(group="model", package="_global_", name="i2v_fsdp", node=I2V_FSDP_CONFIG)  