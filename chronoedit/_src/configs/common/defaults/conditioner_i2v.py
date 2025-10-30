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


from dataclasses import dataclass
from typing import Dict, Optional

import torch
from hydra.core.config_store import ConfigStore

from  chronoedit._ext.imaginaire.lazy_config import LazyCall as L
from  chronoedit._ext.imaginaire.lazy_config import LazyDict
from chronoedit._src.modules.conditioner import (
    BaseCondition,
    GeneralConditioner,
    ReMapkey,
    T2VCondition,
    TextAttrEmptyStringDrop,
    TextAttr,
)
from chronoedit._src.models.wan_i2v_model import WAN2PT1_I2V_COND_LATENT_KEY
from chronoedit._src.modules.clip import Wan2pt1CLIPEmb
from chronoedit._src.utils.context_parallel import broadcast_split_tensor



@dataclass(frozen=True)
class Img2VidWan2pt1Condition(T2VCondition):
    frame_cond_crossattn_emb_B_L_D: Optional[torch.Tensor] = None
    y_B_C_T_H_W: Optional[torch.Tensor] = None # image condition
    # latent_condition: Optional[torch.Tensor] = None # latent condition

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> BaseCondition:
        """Broadcasts and splits the condition across the checkpoint parallelism group.
        For most condition, such asT2VCondition, we do not need split.

        Args:
            process_group: The process group for broadcast and split

        Returns:
            A new BaseCondition instance with the broadcasted and split condition.
        """
        if self.is_broadcasted:
            return self

        y_B_C_T_H_W = self.y_B_C_T_H_W
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["y_B_C_T_H_W"] = None
        new_condition = T2VCondition.broadcast(
            type(self)(**kwargs),
            process_group,
        )
        kwargs = new_condition.to_dict(skip_underscore=False)
        if process_group is not None:
            y_B_C_T_H_W = broadcast_split_tensor(y_B_C_T_H_W, seq_dim=2, process_group=process_group)
        kwargs["y_B_C_T_H_W"] = y_B_C_T_H_W
        return type(self)(**kwargs)


class Img2VidWan2pt1Conditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Img2VidWan2pt1Condition:
        output = super()._forward(batch, override_dropout_rate)
        return Img2VidWan2pt1Condition(**output)


VideoConditionerFpsPaddingConfig: LazyDict = L(Img2VidWan2pt1Conditioner)(
    text=L(TextAttr)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
    ),
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    wanclip=L(Wan2pt1CLIPEmb)(
        input_key=["images", "video", WAN2PT1_I2V_COND_LATENT_KEY],
        dropout_rate=0.0,
        dtype="bfloat16",
    ),
)


VideoConditionerFpsPaddingEmptyStringDropConfig: LazyDict = L(Img2VidWan2pt1Conditioner)(
    text=L(TextAttrEmptyStringDrop)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
    ),
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    wanclip=L(Wan2pt1CLIPEmb)(
        input_key=["images", "video", WAN2PT1_I2V_COND_LATENT_KEY],
        dropout_rate=0.0,
        dtype="bfloat16",
    ),
)


def register_conditioner():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="i2v_conditioner",
        node=VideoConditionerFpsPaddingConfig,
    )
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="i2v_conditioner_empty_string_drop",
        node=VideoConditionerFpsPaddingEmptyStringDropConfig,
    )