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

import torch
from PIL import Image
from diffsynth import save_video, VideoData
from chronoedit_diffsynth.wan_video_new_chronoedit import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    use_usp=True, # torchrun --standalone --nproc_per_node=8
    model_configs=[
        ModelConfig(
            path=[
                "checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00001-of-00014.safetensors",
                "checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00002-of-00014.safetensors",
                "checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00003-of-00014.safetensors",
                "checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00004-of-00014.safetensors",
                "checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00005-of-00014.safetensors",
                "checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00006-of-00014.safetensors",
                "checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00007-of-00014.safetensors",
                "checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00008-of-00014.safetensors",
                "checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00009-of-00014.safetensors",
                "checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00010-of-00014.safetensors",
                "checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00011-of-00014.safetensors",
                "checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00012-of-00014.safetensors",
                "checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00013-of-00014.safetensors",
                "checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00014-of-00014.safetensors",
            ], 
            # offload_device="cpu"
        ),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
    ],
)
# pipe.load_lora(pipe.dit, "models/train/ChronoEdit-14B_lora/epoch-4.safetensors", alpha=1)
pipe.enable_vram_management()

image = Image.open("assets/images/input.jpg")
prompt = "Add a person wearing red jacket."

video = pipe(
    prompt=prompt,
    negative_prompt="",
    input_image=image,
    seed=0, tiled=False,
    num_frames=5, # 5 or 29
    height=720, width=1280,
    cfg_scale=5.0,
    num_inference_steps=50,
    enable_temporal_reasoning=False,
    num_temporal_reasoning_steps=0,
)
save_video(video, "output.mp4", fps=4, quality=5)
