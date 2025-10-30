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

import torch, os, json
from diffsynth import load_state_dict
from chronoedit_diffsynth.wan_video_new_chronoedit import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset, LoadVideo, ImageCropAndResize, ToAbsolutePath
os.environ["TOKENIZERS_PARALLELISM"] = "false"


"""
# Full
PYTHONPATH=$(pwd) accelerate launch --config_file chronoedit_diffsynth/accelerate_config_14B.yaml scripts/train_diffsynth.py \
  --dataset_base_path data/difix_dataset \
  --dataset_metadata_path data/difix_dataset/metadata.csv \
  --height 720 \
  --width 1280 \
  --num_frames 2 \
  --dataset_repeat 5 \
  --model_paths '[["checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00001-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00002-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00003-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00004-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00005-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00006-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00007-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00008-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00009-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00010-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00011-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00012-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00013-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00014-of-00014.safetensors"]]' \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-720P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-720P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-720P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/ChronoEdit-14B_full_difix" \
  --trainable_models "dit" \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing_offload

# Lora
PYTHONPATH=$(pwd) accelerate launch scripts/train_diffsynth.py \
  --dataset_base_path data/difix_dataset \
  --dataset_metadata_path data/difix_dataset/metadata.csv \
  --height 720 \
  --width 1280 \
  --num_frames 2 \
  --dataset_repeat 5 \
  --model_paths '[["checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00001-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00002-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00003-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00004-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00005-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00006-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00007-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00008-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00009-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00010-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00011-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00012-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00013-of-00014.safetensors","checkpoints/ChronoEdit-14B-Diffusers/transformer/diffusion_pytorch_model-00014-of-00014.safetensors"]]' \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-720P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-720P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-720P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/ChronoEdit-14B_lora_difix" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing_offload
"""

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        assert len(data["video"]) == 2, "The number of frames should be 2"
        data["video"] = [data["video"][0]] + [data["video"][1]] * 4
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16))
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(dataset, model, model_logger, args=args)
