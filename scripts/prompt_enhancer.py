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

import argparse
import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor, 
    Qwen3VLMoeForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

# Import device utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from chronoedit.utils.device_utils import get_device_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhance a prompt with CoT reasoning given an input image and prompt"
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default="./assets/images/input.jpg",
        help="Path to the input image (default: ./assets/images/input.jpg)"
    )
    parser.add_argument(
        "--input-prompt",
        type=str,
        required=True,
        help="Input prompt to enhance with CoT reasoning"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        choices=[
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
        ],
        help="Model to use for prompt enhancement"
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=1080,
        help="Maximum resolution for the shortest edge (default: 1080)"
    )
    return parser.parse_args()


def pick_attn_implementation(prefer_flash: bool = True, device: str = "cuda") -> str:
    """
    Decide the best attn_implementation based on environment.

    Args:
        prefer_flash: Whether to prefer flash attention if available
        device: Target device ("cuda" or "cpu")

    Returns one of: "flash_attention_2", "sdpa", "eager".
    """
    # CPU only supports eager
    if device == "cpu" or not torch.cuda.is_available():
        return "eager"
    
    # Try FlashAttention v2 first (needs SM80+ and the wheel to import)
    if prefer_flash:
        try:
            import flash_attn  # noqa: F401
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                # FlashAttn requires Ampere (SM80) or newer
                if (major, minor) >= (8, 0):
                    return "flash_attention_2"
        except Exception:
            pass
    try:
        if torch.backends.cuda.sdp_kernel.is_available():
            return "sdpa"
    except Exception:
        pass

    # Fallback: eager (always works, slower)
    return "eager"
def load_model(model_name, device=None):
    """
    Load the vision-language model and processor.
    
    Args:
        model_name: Name/path of the model to load
        device: Target device (None for auto-detect, "cuda", "cpu", etc.)
    
    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading model: {model_name}")
    
    # Get device map
    device_map = get_device_map(device)
    device_type = "cpu" if device_map == "cpu" else "cuda"
    print(f"Using device: {device_map}")

    attn_impl = pick_attn_implementation(prefer_flash=True, device=device_type)
    print(f"Using attention implementation: {attn_impl}")

    if model_name == "Qwen/Qwen2.5-VL-7B-Instruct":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map=device_map
        )
        processor = AutoProcessor.from_pretrained(model_name)
    
    elif model_name == "Qwen/Qwen3-VL-30B-A3B-Instruct":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name, 
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map=device_map
        )
        processor = AutoProcessor.from_pretrained(model_name)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, processor


def resize_if_needed(image, max_resolution=1080):
    """Resize image so that the shortest edge is at most max_resolution pixels."""
    width, height = image.size
    if min(width, height) > max_resolution:
        scaling_factor = max_resolution / float(min(width, height))
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        print(f"Resizing image from {image.size} to {new_size}")
        return image.resize(new_size, Image.LANCZOS)
    return image


def _run_model_inference(messages, model, processor):
    """
    Helper function to run model inference.
    
    Args:
        messages: Chat messages for the model
        model: The loaded VL model
        processor: The model's processor
        
    Returns:
        str: Generated text
    """
    if isinstance(model, Qwen2_5_VLForConditionalGeneration):
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device).to(model.dtype)
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    elif isinstance(model, Qwen3VLMoeForConditionalGeneration):
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device).to(model.dtype)
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    else:
        raise ValueError("Unsupported model type")

    # Decode the generated text
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


def enhance_prompt(input_image_path, input_prompt, model, processor, max_resolution=1080):
    """
    Enhance a prompt with Chain-of-Thought reasoning given an input image and prompt.
    
    Args:
        input_image_path: Path to the input image
        input_prompt: The input editing instruction prompt
        model: The loaded VL model
        processor: The model's processor
        max_resolution: Maximum resolution for image resizing
        
    Returns:
        str: Enhanced CoT prompt
    """
    # Load and resize image
    print(f"Loading image: {input_image_path}")
    input_image = Image.open(input_image_path).convert("RGB")
    input_image = resize_if_needed(input_image, max_resolution)

    cot_prompt = f"""You are a professional edit instruction rewriter and prompt engineer. Your task is to generate a precise, concise, and visually achievable chain-of-thought reasoning based on the user-provided instruction and the image to be edited.

You have the following information:
1. The user provides an image (the original image to be edited)
2. question text: {input_prompt}

Your task is NOT to output the final answer or the edited image. Instead, you must:
- Generate a "thinking" or chain-of-thought process that explains how you reason about the editing task.
- First identify the task type, then provide reasoning/analysis that leads to how the image should be edited.
- Always describe pose and appearance in detail.
- Match the original visual style or genre (anime, CG art, cinematic, poster). If not explicit, choose a stylistically appropriate one based on the image.
- Incorporate motion and camera direction when relevant (e.g., walking, turning, dolly in/out, pan), implying natural human/character motion and interactions.
- Maintain quoted phrases or titles exactly (e.g., character names, series names). Do not translate or alter the original language of text.

## Task Type Handling Rules:

**1. Standard Editing Tasks (e.g., Add, Delete, Replace, Action Change):**
- For replacement tasks, specify what to replace and key visual features of the new element.
- For text editing tasks, specify text position, color, and layout concisely.
- If the user wants to "extract" something, this means they want to remove the background and only keep the specified object isolated. We should add "while removing the background" to the reasoning.
- Explicitly note what must stay unchanged: appearances (hairstyle, clothing, expression, skin tone/race, age), posture, pose, visual style/genre, spatial layout, and shot composition (e.g., medium shot, close-up, side view).

**2. Character Consistency Editing Tasks (e.g., Scenario Change):**
- For tasks that place an object/character (e.g., human, robot, animal) in a completely new scenario, preserve the object's core identity (appearance, materials, key features) but adapt its pose, interaction, and context to fit naturally in the new environment.
- Reason about how the object should interact with the new scenario (e.g., pose changes, hand positions, orientation, facial direction).
- The background and context should transform completely to match the new scenario while maintaining visual coherence.
- Describe both what stays the same (core appearance) and what must change (pose, interaction, setting) to make the scene look realistic and natural.

The length of outputs should be **around 80 - 100 words** to fully describe the transformation. Always start with "The user wants to ..."

Example Output 1 (Standard Editing Task):
The user wants to make the knight kneel on his right knee while keeping the rest of the pose intact. 
The knight should lower his stance so his right leg bends to the ground in a kneeling position, with the left leg bent upright to support balance. 
The shield with the NVIDIA logo should still be held up firmly in his left hand, angled forward in a defensive posture, while the right hand continues gripping the weapon. 
The armor reflections, proportions, and medieval style should remain consistent, emphasizing a powerful and respectful kneeling stance.

Example Output 2 (Character Consistency Editing Task):
The user wants to change the image by modifying the scene so that the woman is drinking coffee in a cozy coffee shop. 
The elegant anime-style woman keeps her same graceful expression, long flowing dark hair adorned with golden ornaments, and detailed traditional outfit with red and gold floral patterns. 
She is now seated at a wooden café table, holding a steaming cup of coffee near her lips with one hand, while soft sunlight filters through the window, highlighting her refined features. 
The background transforms into a warmly lit café interior with subtle reflections, bookshelves, and gentle ambience, maintaining the delicate, painterly aesthetic.
"""

    # Create messages for CoT generation
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": cot_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": input_image},
            ],
        }
    ]

    # Generate CoT reasoning
    print("Generating Chain-of-Thought enhanced prompt...")
    cot_prompt_output = _run_model_inference(messages, model, processor)
    
    return cot_prompt_output


def main():
    args = parse_args()
    
    # Load model
    model, processor = load_model(args.model)
    
    # Enhance prompt with CoT reasoning
    cot_prompt = enhance_prompt(
        args.input_image,
        args.input_prompt,
        model,
        processor,
        args.max_resolution
    )
    
    # Print enhanced CoT prompt
    print("\n" + "="*80)
    print("Enhanced CoT Prompt:")
    print("="*80)
    print(cot_prompt)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
