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
from  chronoedit._ext.imaginaire.lazy_config import LazyDict
from chronoedit._src.modules.selective_activation_checkpoint import SACConfig as SACConfig
from chronoedit._src.networks.wan2pt1 import WanModel

WAN2PT1_1PT3B: LazyDict = L(WanModel)(
    dim=1536,
    eps=1e-06,
    ffn_dim=8960,
    freq_dim=256,
    in_dim=36,
    model_type="i2v",
    num_heads=12,
    num_layers=30,
    out_dim=16,
    text_len=512,
    cp_comm_type="p2p",
    sac_config=L(SACConfig)(
        mode="block_wise"
    ),
)

WAN2PT1_14B: LazyDict = L(WanModel)(
    dim=5120,
    eps=1e-06,
    ffn_dim=13824,
    freq_dim=256,
    in_dim=36,
    model_type="i2v",
    num_heads=40,
    num_layers=40,
    out_dim=16,
    text_len=512,
    cp_comm_type="p2p",
    sac_config=L(SACConfig)(
        mode="block_wise"
    ),
)


def register_net():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="wan2pt1_1pt3B", node=WAN2PT1_1PT3B)
    cs.store(group="net", package="model.config.net", name="wan2pt1_14B", node=WAN2PT1_14B)
