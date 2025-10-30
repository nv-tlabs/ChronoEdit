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

"""
ChronoEdit Diffusers Inference Script

This script provides a command-line interface for running video editing inference
using the ChronoEdit model with the Diffusers backend.

Example usage:

# Basic usage
PYTHONPATH=$(pwd) python scripts/run_inference_diffusers.py \
    --input assets/images/input_2.png --offload_model --use-prompt-enhancer \
    --prompt "Add a sunglasses to the cat's face" \
    --output output2.mp4 --seed 42  \
    --model-path ./checkpoints/ChronoEdit-14B-Diffusers

# Basic usage with temporal reasoning
PYTHONPATH=$(pwd) python scripts/run_inference_diffusers.py \
    --enable-temporal-reasoning assets/images/input_2.png --offload_model \
    --prompt "Add a sunglasses to the cat's face"  \
    --output output_reasoning_offload.mp4 --seed 42 \
    --model-path ./checkpoints/ChronoEdit-14B-Diffusers

# Advanced usage with lora settings
PYTHONPATH=$(pwd) python scripts/run_inference_diffusers.py \
    --input assets/images/input_2.png \
    --prompt "Add a sunglasses to the cat's face"  \
    --output output_lora.mp4 \
    --num-inference-steps 8 \
    --guidance-scale 1.0 \
    --flow-shift 2.0 \
    --lora-scale 1.0 \
    --seed 42 \
    --lora-path ./checkpoints/ChronoEdit-14B/nvidia/lora/chronoedit_distill_lora_clean.safetensors \
    --model-path ./checkpoints/ChronoEdit-14B-Diffusers
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from PIL import Image
from transformers import CLIPVisionModel

from chronoedit_diffusers.pipeline_chronoedit import ChronoEditPipeline
from chronoedit_diffusers.transformer_chronoedit import ChronoEditTransformer3DModel
from scripts.prompt_enhancer import load_model as load_prompt_enhancer
from scripts.prompt_enhancer import enhance_prompt

# Resolution presets
RESOLUTION_PRESETS = {
    "480p": 480 * 832,
    "720p": 720 * 1280,
    "1080p": 1080 * 1920,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ChronoEdit Video Editing Inference with Diffusers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")


    model_group.add_argument(
        "--use-prompt-enhancer",
        action="store_true",
        help="Whether Use Prompt Enhancer",
    )
    model_group.add_argument(
        "--prompt-enhancer-model",
        type=str,
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        choices=[
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
        ],
        help="Model to use for prompt enhancement"
    )
    model_group.add_argument(
        "--model-path",
        "--model-id",
        type=str,
        default="nvidia/ChronoEdit-14B-Diffusers",
        help="HuggingFace model ID or local path to the model"
    )
    model_group.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA weights file (.safetensors)"
    )
    model_group.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="LoRA scale factor (0.0-1.0)"
    )
    model_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on"
    )
    
    # Input/Output arguments
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input image"
    )
    io_group.add_argument(
        "-o", "--output",
        type=str,
        default="output.mp4",
        help="Path to output video"
    )
    io_group.add_argument(
        "--output-image",
        type=str,
        default=None,
        help="Path to save last frame (default: output path with .png extension)"
    )
    
    # Generation arguments
    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "-p", "--prompt",
        type=str,
        required=True,
        help="Text prompt for video editing"
    )
    gen_group.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt (not recommended, leave empty)"
    )
    gen_group.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    gen_group.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale"
    )
    gen_group.add_argument(
        "--flow-shift",
        type=float,
        default=5.0,
        help="Flow shift parameter for scheduler"
    )
    gen_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    gen_group.add_argument(
        "--offload_model",
        action="store_true",
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )

    # Temporal reasoning arguments
    temporal_group = parser.add_argument_group("Temporal Reasoning (Experimental)")
    temporal_group.add_argument(
        "--enable-temporal-reasoning",
        action="store_true",
        help="Enable temporal reasoning mode"
    )
    temporal_group.add_argument(
        "--num-temporal-reasoning-steps",
        type=int,
        default=50,
        help="Number of temporal reasoning steps"
    )

    # Misc arguments
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()

    
    if not os.path.exists(args.input):
        parser.error(f"Input image not found: {args.input}")
    
    if args.lora_path and not os.path.exists(args.lora_path):
        parser.error(f"LoRA weights file not found: {args.lora_path}")
    
    return args




def calculate_dimensions(image,  mod_value):
    """
    Calculate output dimensions based on resolution settings.
    
    Args:
        image: PIL Image
        mod_value: Modulo value for dimension alignment
        
    Returns:
        Tuple of (width, height)
    """
    
    # Get max area from preset or override 
    target_area = 720 * 1280
    
    # Calculate dimensions maintaining aspect ratio
    aspect_ratio = image.height / image.width
    calculated_height = round(np.sqrt(target_area * aspect_ratio)) // mod_value * mod_value
    calculated_width = round(np.sqrt(target_area / aspect_ratio)) // mod_value * mod_value
    
    return calculated_width, calculated_height


def main():
    """Main execution function."""
    args = parse_args()
    
    # Setup
    device = args.device
    
    if args.verbose:
        print("=" * 80)
        print("ChronoEdit Diffusers Inference")
        print("=" * 80)
        print(f"Model: {args.model_path}")
        print(f"Device: {device}")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Prompt: {args.prompt}")
        print(f"Resolution: {args.resolution}")
        print(f"Steps: {args.num_inference_steps}")
        print(f"Guidance Scale: {args.guidance_scale}")
        if args.seed is not None:
            print(f"Seed: {args.seed}")
        if args.lora_path:
            print(f"LoRA: {args.lora_path} (scale={args.lora_scale})")
        print("=" * 80)


    if args.use_prompt_enhancer:
        prompt_model, processor = load_prompt_enhancer(args.prompt_enhancer_model)

        # Enhance prompt with CoT reasoning
        cot_prompt = enhance_prompt(
            args.input,
            args.prompt,
            prompt_model,
            processor,
        )

        # Print enhanced CoT prompt
        print("\n" + "=" * 80)
        print("Enhanced CoT Prompt:")
        print("=" * 80)
        print(cot_prompt)
        print("=" * 80 + "\n")
        args.prompt = cot_prompt
        # offload prompt model to cpu
        _ = prompt_model.to("cpu")

    # Load models
    print("Loading diffusion models...")

    try:
        image_encoder = CLIPVisionModel.from_pretrained(
            args.model_path,
            subfolder="image_encoder",
            torch_dtype=torch.float32
        )
        if args.verbose:
            print("✓ Loaded image encoder")
        
        vae = AutoencoderKLWan.from_pretrained(
            args.model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16
        )
        if args.verbose:
            print("✓ Loaded VAE")
        
        transformer = ChronoEditTransformer3DModel.from_pretrained(
            args.model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16
        )
        if args.verbose:
            print("✓ Loaded transformer")
        
        pipe = ChronoEditPipeline.from_pretrained(
            args.model_path,
            image_encoder=image_encoder,
            transformer=transformer,
            vae=vae,
            torch_dtype=torch.bfloat16
        )
        if args.verbose:
            print("✓ Created pipeline")
        
        # Load LoRA if specified
        if args.lora_path:
            print(f"Loading LoRA weights from {args.lora_path}...")
            pipe.load_lora_weights(args.lora_path)

            pipe.fuse_lora(lora_scale=args.lora_scale)
            if args.verbose:
                print(f"✓ Fused LoRA with scale {args.lora_scale}")

        
        # Setup scheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            flow_shift=args.flow_shift
        )
        if args.verbose:
            print(f"✓ Configured scheduler (flow_shift={args.flow_shift})")
        
        # Move to device
        pipe.to(device)
        print(f"✓ Models loaded and moved to {device}")
        
    except Exception as e:
        print(f"Error loading models: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load and prepare input image
    print(f"Loading input image: {args.input}")
    try:
        image = load_image(args.input)
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Calculate output dimensions
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    width, height = calculate_dimensions(
        image,
        mod_value
    )
    
    print(f"Output dimensions: {width}x{height}")
    image = image.resize((width, height))
    
    # Setup generator for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        if args.verbose:
            print(f"Using seed: {args.seed}")
    
    # Run inference
    print("Running inference...")
    num_frames = 29 if args.enable_temporal_reasoning else 5

    print(f"Generating {num_frames} frames with {args.num_inference_steps} steps...")
    try:
        output = pipe(
            image=image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if args.negative_prompt else None,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            enable_temporal_reasoning=args.enable_temporal_reasoning,
            num_temporal_reasoning_steps=args.num_temporal_reasoning_steps,
            generator=generator,
            offload_model=args.offload_model,
        ).frames[0]
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Save outputs
    try:
        # Create output directory if needed
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Export video only if temporal reasoning is enabled
        if args.enable_temporal_reasoning:
            print(f"Saving video to: {args.output}")
            export_to_video(output, args.output, fps=8)
            print(f"✓ Video saved: {args.output}")
        
        # Save last frame if requested
        if args.output_image:
            image_path = args.output_image
        else:
            image_path = Path(args.output).with_suffix(".png")
        
        last_frame = (output[-1] * 255).clip(0, 255).astype("uint8")
        Image.fromarray(last_frame).save(image_path)
        print(f"✓ Last frame saved: {image_path}")
        
    except Exception as e:
        print(f"Error saving outputs: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 80)
    print("✓ Inference completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
