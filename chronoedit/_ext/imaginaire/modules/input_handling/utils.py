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

import math

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def detect_aspect_ratio(img_size):
    r"""
    Function for detecting the closest aspect ratio.
    """

    _aspect_ratios = np.array([(16 / 9), (4 / 3), 1, (3 / 4), (9 / 16)])
    _aspect_ratio_keys = ["16,9", "4,3", "1,1", "3,4", "9,16"]
    w, h = img_size
    current_ratio = w / h
    closest_aspect_ratio = np.argmin((_aspect_ratios - current_ratio) ** 2)
    return _aspect_ratio_keys[closest_aspect_ratio]


def detect_resolution(img_size):
    r"""
    Function to detect resolution.
    """
    w, h = img_size
    if max(w, h) >= 1024 and min(w, h) >= 576:
        resolution = 1024
    elif max(w, h) >= 256 and min(w, h) >= 144:
        resolution = 256
    else:
        raise ValueError("Images should be of at least 256x144 or 144x256 resolution")
    return resolution


def resize_image_to_aspect_ratio(image, resolution, aspect_ratio, center_crop=True):
    r"""
    Function for resizing to a specific aspect ratio and a resolution.
    """

    # Finding the target shape based on resolution and aspect ratio
    asp_ratio = aspect_ratio.split(",")
    asp_ratio[0] = int(asp_ratio[0])
    asp_ratio[1] = int(asp_ratio[1])
    dim_ratio = asp_ratio[0] / asp_ratio[1]
    if dim_ratio >= 1:
        target_w, target_h = resolution, int(math.ceil(resolution / dim_ratio))
    else:
        target_w, target_h = int(math.ceil(resolution * dim_ratio)), resolution

    if type(image) == torch.Tensor:
        # Perform resizing
        if center_crop:  # Do aspect ratio preserving resize, then center crop
            orig_h, orig_w = image.shape[-2:]
            scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
            resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), int(math.ceil(scaling_ratio * orig_w)))
            img_resized = torch.nn.functional.interpolate(image, resizing_shape, mode="bicubic")

            # Perform center crop
            resize_box = [int((resizing_shape[0] - target_h) / 2), int((resizing_shape[1] - target_w) / 2), 0, 0]
            resize_box[2] = resize_box[0] + target_h
            resize_box[3] = resize_box[1] + target_w
            img_resized = img_resized[:, :, resize_box[0] : resize_box[2], resize_box[1] : resize_box[3]]
        else:  # Directly resize to target aspect ratio.
            img_resized = torch.nn.functional.interpolate(image, (target_h, target_w), mode="bicubic")

    else:
        # Perform resizing
        if center_crop:  # Do aspect ratio preserving resize, then center crop
            orig_w, orig_h = image.size
            scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
            resizing_shape = (int(math.ceil(scaling_ratio * orig_w)), int(math.ceil(scaling_ratio * orig_h)))
            img_resized = image.resize(resizing_shape, Image.Resampling.BICUBIC, reducing_gap=True)

            # Perform center crop
            resize_box = [int((resizing_shape[0] - target_w) / 2), int((resizing_shape[1] - target_h) / 2), 0, 0]
            resize_box[2] = resize_box[0] + target_w
            resize_box[3] = resize_box[1] + target_h
            img_resized = img_resized.crop(resize_box)
        else:  # Directly resize to target aspect ratio.
            img_resized = image.resize((target_w, target_h), Image.Resampling.BICUBIC, reducing_gap=True)

    return img_resized


def resize_batch(images, resolution):
    r"""
    Function for resizing a batch of images.
    """
    assert isinstance(images, list), "Invalid input type. Expects a list of images as inputs"
    aspect_ratio = detect_aspect_ratio(images[0].size)
    images_resized = [
        resize_image_to_aspect_ratio(img, resolution=resolution, aspect_ratio=aspect_ratio) for img in images
    ]
    return images_resized


def process_input_image(input_image, resolution):
    if isinstance(input_image, str):
        return process_input_image(Image.open(input_image).convert("RGB"), resolution)

    if isinstance(input_image, Image.Image):
        return process_input_image([input_image], resolution)

    if isinstance(input_image, list) and all(isinstance(i, Image.Image) for i in input_image):
        # Perform resizing
        input_image = resize_batch(input_image, resolution=resolution)
        input_image = [
            (2.0 * transforms.functional.pil_to_tensor(img) / 255.0 - 1.0) for img in input_image
        ]  # [-1, 1] images,
        return process_input_image(torch.stack(input_image), resolution)

    if isinstance(input_image, torch.Tensor):
        return input_image.cuda(), input_image.shape[0]

    raise TypeError("Invalid input type. Expected one of [str, PIL.Image.Image, List[PIL.Image.Image], torch.Tensor]")
