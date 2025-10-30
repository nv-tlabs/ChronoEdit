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
from functools import partial
import torch.distributed as dist
import torch as th
import os
import json
import time
import chronoedit._ext.imaginaire.utils.distributed
from megatron.core import parallel_state
from chronoedit._ext.imaginaire.visualize.video import save_img_or_video
from chronoedit._src.datasets.utils import VIDEO_RES_SIZE_INFO
from chronoedit._src.utils.get_t5_emb import get_text_embedding
from chronoedit._src.utils.model_loader import load_model_from_checkpoint
from chronoedit._src.utils.misc import read_and_process_image


th.enable_grad(False)

DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
DEFAULT_POSITIVE_PROMPT = "A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."
UMT5_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChronoEdit inference script")
    parser.add_argument("--experiment", type=str, default="???", help="inference only config")
    parser.add_argument("--guidance", type=float, default=5.0,  help="Guidance value")
    parser.add_argument("--lora_paths", type=str, default=None, nargs="+", help="Path to the lora weights")
    parser.add_argument("--lora_weights", type=float, default=None, nargs="+", help="Path to the lora weights")
    parser.add_argument("--prompt", type=str, default="", help="Editing Prompt")
    parser.add_argument("--input_image_fp", type=str, default="", help="Input image path")
    parser.add_argument("--shift", type=float, default=5.0, help="Shift value")
    parser.add_argument("--neg_prompt", type=str, default="", help="Input image path")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to the checkpoint. If not provided, will use the one specify in the config",
    )

    parser.add_argument("--num_steps", type=int, default=50, help="sampler steps")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--num_frames", type=int, default=2, help="Number of frames in the video")
    parser.add_argument("--save_root", type=str, default="./", help="Path to save the results")
    parser.add_argument("--cache_dir", type=str, default="/project/cosmos/tianshic/imaginaire4-cache", help="Path to cache the text embeddings")
    parser.add_argument("--resolution", type=str, default="720p", help="Resolution of the video")
    parser.add_argument("--fps", type=int, default=10, help="FPS of the video and model")
    parser.add_argument("--drop_video_step", type=int, default=None, help="The step to drop the video")
    return parser.parse_args()


def sample_batch_video(resolution, batch_size, num_frames, fps):
    h, w = VIDEO_RES_SIZE_INFO[resolution]["9,16"]
    data_batch = {
        "dataset_name": "video_data",
        "video": th.randint(0, 256, (batch_size, 3, num_frames, h, w), dtype=th.uint8).cuda(),
        "fps": th.ones((batch_size,)).cuda() * fps,
        "padding_mask": th.zeros(batch_size, 1, h, w).cuda(),
    }
    return data_batch


def get_sample_batch(
    num_frames: int = 93,
    resolution: str = "720p",
    batch_size: int = 1,
    fps: int = 30,
) -> th.Tensor:
    data_batch = sample_batch_video(resolution, batch_size, num_frames, fps)
    for k, v in data_batch.items():
        if isinstance(v, th.Tensor) and th.is_floating_point(data_batch[k]):
            data_batch[k] = v.cuda().to(dtype=th.bfloat16)
    return data_batch


if __name__ == "__main__":
    args = parse_arguments()
    if "{" in args.save_root:
        try:
            _ = args.save_root.format(g=args.guidance, s=args.shift, n=args.num_steps)
        except KeyError:
            raise ValueError(f"Invalid save_root: {args.save_root}, please use {{g,s,n}} in the save_root")
    th.backends.cuda.preferred_linalg_library(backend="magma")

    chronoedit._ext.imaginaire.utils.distributed.init()
    world_size = dist.get_world_size()
    print(f"world_size: {world_size}")
    if world_size > 1:
        parallel_state.initialize_model_parallel(context_parallel_size=world_size)
        process_group = parallel_state.get_context_parallel_group()


    # instantiate model, config and load checkpoint
    if args.lora_paths:
        experiment_opts = ["model.config.net.postpone_checkpoint=True"]
    else:
        experiment_opts = []
    model, config = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=args.checkpoint_path,
        enable_fsdp=False,
        config_file="chronoedit/_src/configs/chronoedit/config.py",
        experiment_opts=experiment_opts,
    )
    if world_size > 1:
        model.net.enable_context_parallel(process_group)
    if args.lora_paths:
        lora_names = []
        for lora_path in args.lora_paths:
            lora_name = model.load_lora_weights(lora_path)
            lora_names.append(lora_name)
        model.set_weights_and_activate_adapters(lora_names, args.lora_weights)
        model.net.enable_selective_checkpoint(model.net.sac_config, model.net.blocks)
    embedding_func = partial(get_text_embedding, text_encoder_class='umT5')
    if dist.get_rank() == 0:
        avg_list = {}
    save_path = f"{args.save_root}/{args.input_image_fp.split('/')[-1].split('.')[0]}_edited"

    sample_info = {}
    sample_info['prompt'] = args.prompt

    data_batch = get_sample_batch(
        num_frames=model.tokenizer.get_pixel_num_frames(model.get_num_video_latent_frames()),
        resolution=args.resolution,
        batch_size=args.num_samples,
        fps=args.fps,
    )

    data_batch["video"] = read_and_process_image(args.input_image_fp, [data_batch['video'].shape[3], data_batch['video'].shape[4]], args.num_frames).cuda()
    if "is_preprocessed" in data_batch:
        del data_batch["is_preprocessed"]

    assert args.num_frames == 2 or args.num_frames == 26, f"num_frames must be 2 or 26, but got {args.num_frames}"
    is_video_prior = True if args.num_frames == 26 else False
    data_batch["is_video_prior"] = is_video_prior
    data_batch['caption'] = [args.prompt]
    #### prepare text embeddings ####
    text_emb = embedding_func(prompts = args.prompt, cache_dir=args.cache_dir)
    th.distributed.barrier()
    data_batch["t5_text_embeddings"] = text_emb.to(dtype=th.bfloat16).cuda()

    if args.neg_prompt:
        args.neg_prompt = UMT5_NEGATIVE_PROMPT
        text_emb = embedding_func(prompts = args.neg_prompt)
        data_batch["neg_t5_text_embeddings"] = text_emb.to(dtype=th.bfloat16).cuda()
        print("Using negative prompt: ", args.neg_prompt)
        is_negative_prompt = True
        sample_info['neg_prompt'] = args.neg_prompt

    else:
        print("Using empty negative prompt")
        is_negative_prompt = False
        sample_info['neg_prompt'] = ""

    if dist.get_rank() == 0:
        key = f"{args.guidance:.2f}_{args.shift:.2f}_{args.num_steps:d}"
        if key not in avg_list:
            avg_list[key] = {
                "diffusion": [],
                "decode": [],
                "total": [],
            }
        this_list = avg_list[key]

    if "{" in args.save_root:
        save_root = args.save_root.format(g=args.guidance, s=args.shift, n=args.num_steps)
    else:
        save_root = args.save_root



    if dist.get_rank() == 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(f"{save_path}.json", "w") as f:
            json.dump(sample_info, f, indent=2)

    # generate samples
    dist.barrier()
    th.cuda.synchronize()
    start_time = time.time()

    sample_info['steps'] = args.guidance
    sample_info['guidance'] = args.num_steps
    sample_info['shift'] = args.shift
    sample_info['drop_video_step'] = args.drop_video_step

    sample = model.generate_samples_from_batch(
        data_batch,
        guidance=args.guidance,
        state_shape=[16, 2 + (args.num_frames - 2) // 4, data_batch['video'].shape[3] // 8,  data_batch['video'].shape[4] // 8],
        seed=1,
        num_steps=args.num_steps,
        is_negative_prompt=is_negative_prompt,
        shift=args.shift,
        drop_video_step=args.drop_video_step,
    )

    dist.barrier()
    th.cuda.synchronize()


    if is_video_prior and args.drop_video_step is None:
        first_frame_sample = sample[:, :, :1, :, :]
        last_frame_sample = sample[:, :, -1:, :, :]
        sample_edit = th.cat([first_frame_sample, last_frame_sample], dim=2)
        sample_edit = model.decode(sample_edit).contiguous()

        sample_video = sample[:, :, :-1, :, :]
        sample_video = model.decode(sample_video).contiguous()

        video = th.cat([sample_video, sample_edit[:, :, 1:, :, :]], dim=2)
    else:
        video = model.decode(sample)

    dist.barrier()
    th.cuda.synchronize()

    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    for i in range(args.num_samples):
        save_img_or_video((1.0 + video[i]) / 2, save_path, quality=8, fps=args.fps)
        save_img_or_video((1.0 + video[i, :, -1:]) / 2, save_path)

    if world_size > 1:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
