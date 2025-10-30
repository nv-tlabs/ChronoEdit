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

from  chronoedit._ext.imaginaire.lazy_config import LazyDict

cs = ConfigStore.instance()


"""
torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train --config=chronoedit/_src/configs/chronoedit/config.py -- experiment="edit_14B_skip_pe8_mock"
"""

CHRONO_EDIT_14B_mock: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/edit_14B_skip_pe8",
            {"override /data_train": "mock_video"},
        ],
        job=dict(
            group="chronoedit_edit",
            name="edit_14B_skip_pe8_mock",
        ),
        checkpoint=dict(
            save_iter=250,
            save_to_object_store=dict(
                enabled=False,
            ),
            load_from_object_store=dict(
                enabled=False,
            ),
            load_path="/lustre/fs12/portfolios/nvr/users/huling/ChronoEdit_Release/checkpoints/ChronoEdit-14B/nvidia/chronoedit_14B",  # chronoedit_14B checkpoint here
            load_training_state=False,
            strict_resume=False,
        ),
    )
)


CHRONO_EDIT_14B_SFT_1: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{CHRONO_EDIT_14B_mock['job']['name']}",
            {"override /data_train": "chronoedit_data_image"}, #TODO: ADD dataset
        ],
        job=dict(
            group="chronoedit_edit",
            name="edit_14B_skip_pe8_sft1",
        ),
        checkpoint=dict(
            save_iter=250,
            save_to_object_store=dict(
                enabled=False,
            ),
            load_from_object_store=dict(
                enabled=False,
            ),
            load_path="/lustre/fs12/portfolios/nvr/users/huling/ChronoEdit_Release/checkpoints/ChronoEdit-14B/nvidia/chronoedit_14B",  # chronoedit_14B checkpoint here
            load_training_state=False,
            strict_resume=False,
        ),
    )
)

cs.store(
    group="experiment",
    package="_global_",
    name="edit_14B_skip_pe8_mock",
    node=CHRONO_EDIT_14B_mock,
)

cs.store(
    group="experiment",
    package="_global_",
    name="edit_14B_skip_pe8_sft1",
    node=CHRONO_EDIT_14B_SFT_1,
)