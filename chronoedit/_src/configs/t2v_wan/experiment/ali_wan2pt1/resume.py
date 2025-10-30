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


import functools

from hydra.core.config_store import ConfigStore

from  chronoedit._ext.imaginaire.lazy_config import LazyDict
from chronoedit._src.datasets.cached_replay_dataloader import duplicate_batches_random

#### Benchmark speed
# 1.3B 480P wan2pt1_1pt3B_res480p_16fps_cp_new_mock_wo_resume
# [07-17 15:13:36|INFO|projects/cosmos/cosmos_predict2/callbacks/iter_speed.py:85:every_n_impl] 10 : iter_speed 1.97 seconds per iteration | Loss: 1.7784
#                                     Avg            Max            Min
# cpu_mem_gb                   112.067894     112.792061     111.944218
# peak_gpu_mem_gb               10.516008      10.524383      10.504822
# peak_gpu_mem_reserved_gb      17.958740      18.587891      17.839844
#
# 14B 480P wan2pt1_14B_res480p_16fps_mock_wo_resume
# [07-17 15:21:58|INFO|projects/cosmos/cosmos_predict2/callbacks/iter_speed.py:85:every_n_impl] 12 : iter_speed 3.93 seconds per iteration | Loss: 1.7774
#                                     Avg            Max            Min
# cpu_mem_gb                   112.164636     112.869026     112.045441
# peak_gpu_mem_gb               31.847306      31.852549      31.840982
# peak_gpu_mem_reserved_gb      58.280029      58.699219      57.923828
#
# 14B 720P wan2pt1_14B_res720p_16fps_cp_new_mock_wo_resume
# [07-17 15:53:48|INFO|projects/cosmos/cosmos_predict2/callbacks/iter_speed.py:85:every_n_impl] 12 : iter_speed 18.64 seconds per iteration | Loss: 1.7680
#                                     Avg            Max            Min
# cpu_mem_gb                   116.872467     117.555328     116.759327
# peak_gpu_mem_gb               35.343176      35.345328      35.338885
# peak_gpu_mem_reserved_gb      75.013916      75.041016      74.970703


_TRAINER_DEBUG_CONFIG = dict(
    max_iter=25,
    logging_iter=2,
    callbacks=dict(
        every_n_sample_reg=dict(
            every_n=12,
            num_sampling_step=3,
        ),
        every_n_sample_ema=dict(
            every_n=999999,
            num_sampling_step=3,
        ),
        reg_model_t2v_sora_upsampled_val_sampling=dict(
            every_n=999999,
            is_debug=True,
        ),
        ema_model_t2v_sora_upsampled_val_sampling=dict(
            every_n=999999,
            is_debug=True,
        ),
    ),
)
_CKPT_DEBUG_CONFIG = dict(
    save_iter=100,
    load_path="",
    load_training_state=False,
    strict_resume=False,
)


def build_debug_runs(job):
    wo_resume = dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}_WO_RESUME" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=_TRAINER_DEBUG_CONFIG,
        checkpoint=_CKPT_DEBUG_CONFIG,
    )

    mock_wo_resume = dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            {"override /data_train": "mock"},
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}_MOCK_WO_RESUME" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=_TRAINER_DEBUG_CONFIG,
        checkpoint=_CKPT_DEBUG_CONFIG,
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    dataloader=dict(
                        dataset=dict(t5_dim=4096),
                    ),
                    ratio=0,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        dataset=dict(t5_dim=4096),
                    ),
                    ratio=1,
                ),
            ),
        ),
    )

    return [wo_resume, mock_wo_resume]


"""
torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train --config=projects/cosmos/wan/configs/t2v_wan/config.py -- experiment="wan2pt1_1pt3B_res480p_16fps_new_mock_wo_resume"
"""
WAN2PT1_1PT3B_RES480P_FPS16: LazyDict = LazyDict(
    dict(
        defaults=[
            {
                "override /data_train": "mock"
            },
            {"override /model": "fsdp"},
            {"override /net": "wan2pt1_1pt3B"},
            {"override /conditioner": "add_fps_padding_mask_empty_string_drop"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="resume_wan2pt1",
            name="wan2pt1_1pt3B_res480p_16fps_new",
        ),
        optimizer=dict(
            lr=3e-5,
            weight_decay=1e-3,
        ),
        scheduler=dict(
            f_max=[0.99],
            f_min=[0.4],
            warm_up_steps=[100],
            cycle_lengths=[400_000],
        ),
        model=dict(
            config=dict(
                ema=dict(
                    enabled=False,
                ),
                state_t=24,  # 24
                fsdp_shard_size=4,
                shift=5,
                use_dynamic_shift=False,
                train_time_weight="reweighting"
            )
        ),
        checkpoint=dict(
            save_iter=100,
            save_to_object_store=dict(
                enabled=False,
            ),
            load_from_object_store=dict(
                enabled=False,
            ),
            load_path="Wan2.1-T2V-1.3B.dcp",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=150_000,
            logging_iter=200,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=5_000,
                    is_x0=False,
                    guidance=[0, 3, 7],
                    fps=16,
                ),
                grad_clip=dict(
                    clip_norm=0.1
                )
            ),
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    ratio=0,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.1),
                        dataset=dict(
                            resolution="480p",
                            num_video_frames=93,
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            caption_type="t2w_qwen2p5_7b",
                            embedding_type="umt5_xxl",
                            min_fps_thres=10,
                            max_fps_thres=60,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
        upload_reproducible_setup=True,
    ),
    flags={"allow_objects": True},
)

"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=projects/cosmos/wan/configs/t2v_wan/config.py  -- experiment="wan2pt1_1pt3B_res480p_16fps_cp_new_mock_wo_resume"
"""
WAN2PT1_1PT3B_RES480_FPS16_CP: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{WAN2PT1_1PT3B_RES480P_FPS16['job']['name']}",
            "_self_",
        ],
        job=dict(
            group="resume_wan2pt1",
            name="wan2pt1_1pt3B_res480p_16fps_cp_new",
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
    ),
    flags={"allow_objects": True},
)



"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=projects/cosmos/wan/configs/t2v_wan/config.py -- experiment="wan2pt1_14B_res480p_16fps"
# run with mock data
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=projects/cosmos/wan/configs/t2v_wan/config.py -- experiment="wan2pt1_14B_res480p_16fps_mock_wo_resume"
"""
WAN2PT1_14B_RES480_FPS16: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{WAN2PT1_1PT3B_RES480P_FPS16['job']['name']}",
            {
                "override /data_train": "mock"
            },
            {"override /net": "wan2pt1_14B"},
            "_self_",
        ],
        job=dict(
            group="resume_wan2pt1",
            name="wan2pt1_14B_res480p_16fps",
        ),
        checkpoint=dict(
            save_iter=100,
            load_path="Wan2.1-T2V-14B.dcp",
        ),
        trainer=dict(
            max_iter=5_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=500,
                ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                ema=dict(
                    enabled=False,
                ),
                state_t=24,  # 24
                fsdp_shard_size=32,
            )
        ),
        dataloader_train=dict(
            dataloaders=dict(
                video_data=dict(
                    dataloader=dict(
                        use_cache=False,
                        batch_size=1,
                        cache_size=32,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.5),
                        dataset=dict(
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            caption_type="t2w_qwen2p5_7b",
                            embedding_type="umt5_xxl",
                            min_fps_thres=10,
                            max_fps_thres=60,
                        ),
                    ),
                ),
                image_data=dict(
                    ratio=0,
                ),
            ),
        ),
        upload_reproducible_setup=True,
    ),
    flags={"allow_objects": True},
)

"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=projects/cosmos/wan/configs/t2v_wan/config.py  -- experiment="wan2pt1_14B_res720p_16fps_cp_new_mock_wo_resume"
"""
WAN2PT1_14B_RES720_FPS16: LazyDict = LazyDict(
    dict(
        defaults=[
            f"/experiment/{WAN2PT1_14B_RES480_FPS16['job']['name']}",
            "_self_",
        ],
        job=dict(
            group="resume_wan2pt1",
            name="wan2pt1_14B_res720p_16fps_cp_new",
        ),
        dataloader_train=dict(
            dataloaders=dict(
                image_data=dict(
                    ratio=0,
                ),
                video_data=dict(
                    dataloader=dict(
                        batch_size=1,
                        use_cache=False,
                        cache_size=16,
                        concat_size=1,
                        cache_augment_fn=functools.partial(duplicate_batches_random, n=1.1),
                        dataset=dict(
                            resolution="720p",
                            num_video_frames=93,
                            video_decoder_name="video_naive_bytes",
                            augmentor_name="video_basic_augmentor_v2",
                            caption_type="t2w_qwen2p5_7b",
                            embedding_type="umt5_xxl",
                            min_fps_thres=10,
                            max_fps_thres=60,
                        ),
                    ),
                    ratio=1,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()

for _item, _item_wo_resume, _item_mock_wo_resume in [
    [
        WAN2PT1_1PT3B_RES480P_FPS16,
        *build_debug_runs(WAN2PT1_1PT3B_RES480P_FPS16),
    ],
    [
        WAN2PT1_1PT3B_RES480_FPS16_CP,
        *build_debug_runs(WAN2PT1_1PT3B_RES480_FPS16_CP),
    ],
    [
        WAN2PT1_14B_RES480_FPS16,
        *build_debug_runs(WAN2PT1_14B_RES480_FPS16),
    ],
    [
        WAN2PT1_14B_RES720_FPS16,
        *build_debug_runs(WAN2PT1_14B_RES720_FPS16),
    ],
]:
    cs.store(
        group="experiment",
        package="_global_",
        name=_item["job"]["name"],
        node=_item,
    )
    if _item_wo_resume is not None:
        cs.store(
            group="experiment",
            package="_global_",
            name=f"{_item['job']['name']}_wo_resume",
            node=_item_wo_resume,
        )
    if _item_mock_wo_resume is not None:
        cs.store(
            group="experiment",
            package="_global_",
            name=f"{_item['job']['name']}_mock_wo_resume",
            node=_item_mock_wo_resume,
        )
