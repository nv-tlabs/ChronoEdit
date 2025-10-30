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

CHRONO_EDIT_14B: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/wan2pt1_i2v_14B_res480p_16fps",
            {"override /data_train": "mock_video"},
            {"override /data_train": "mock_video"}, 
            {"override /model": "fsdp_wan2pt1_edit"},
            {"override /net": "wan2pt1_14B_edit"},
            {"override /conditioner": "i2v_conditioner_empty_string_drop"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling_edit",
                    "wandb",
                    "cluster_speed",
                ]
            },
            "_self_",
        ],
        upload_reproducible_setup=False, # TODO: consolidate logdir
        model=dict(
            config=dict(
                shift=5,
                train_time_weight="uniform",
                net=dict(
                    temporal_skip_p=True,  # Use skip PE
                    temporal_skip_len=8,
                ),
            ),
        ),
        optimizer=dict(
            lr=2e-5,
            weight_decay=1e-3,
        ),
        checkpoint=dict(
            save_iter=1000,
            save_to_object_store=dict(
                enabled=False,
            ),
            load_from_object_store=dict(
                enabled=False,
            ),
            load_path="checkpoints/Wan2.1-I2V-14B-720P.dcp", # WAN2.1 14B checkpoint here TODO: need a conversio script
            load_training_state=False,
            strict_resume=False,
        ),
        job=dict(
            group="chronoedit_edit",
            name="edit_14B_skip_pe8",
        ),
        model_parallel=dict(
            context_parallel_size=2,
        ),
        trainer=dict(
            timestamp_seed=True,
            max_iter=1000000,
            logging_iter=20,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=1000,
                    guidance=[5],
                ),
            ),
        ),
        dataloader_train=dict(),
    )
)



cs.store(
    group="experiment",
    package="_global_",
    name="edit_14B_skip_pe8",
    node=CHRONO_EDIT_14B,
)

