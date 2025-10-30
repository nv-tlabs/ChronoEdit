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


# from Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

from chronoedit._src.networks.wan2pt1 import WanModel, VideoPositionEmb
import torch
from einops import repeat, rearrange
from typing import Optional

class TemproalSkipVideoRopePosition3DEmb(VideoPositionEmb):
    def __init__(
        self,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        temporal_skip_len: int = 10,
    ):
        super().__init__()
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self._dim_h = dim_h
        self._dim_t = dim_t
        self.temporal_skip_len = temporal_skip_len

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

        self._is_initialized = False

    def cache_parameters(self) -> None:
        if self._is_initialized:
            return

        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().cuda()
        self.dim_spatial_range = torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().cuda() / dim_h
        self.dim_temporal_range = torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().cuda() / dim_t
        self._is_initialized = True

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ):
        """
        Generate embeddings for the given input size.

        Args:
            B_T_H_W_C (torch.Size): Input tensor size (Batch, Time, Height, Width, Channels).
            fps (Optional[torch.Tensor], optional): Frames per second. Defaults to None.
            h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses self.h_ntk_factor.
            w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses self.w_ntk_factor.
            t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses self.t_ntk_factor.

        Returns:
            Not specified in the original code snippet.
        """
        self.cache_parameters()

        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        assert h_ntk_factor is not None
        assert w_ntk_factor is not None
        assert t_ntk_factor is not None
        
        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        B, T, H, W, _ = B_T_H_W_C
        is_video_prior = True if T == self.temporal_skip_len else False # hacky way to handle video prior
        T = self.temporal_skip_len
        
        assert (
            H <= self.max_h and W <= self.max_w
        ), f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        freqs_h = torch.outer(self.seq[:H], h_spatial_freqs)
        freqs_w = torch.outer(self.seq[:W], w_spatial_freqs)
        freqs_t = torch.outer(self.seq[:T], temporal_freqs)
        freqs_T_H_W_D = torch.cat(
            [
                repeat(freqs_t, "t d -> t h w d", h=H, w=W),
                repeat(freqs_h, "h d -> t h w d", t=T, w=W),
                repeat(freqs_w, "w d -> t h w d", t=T, h=H),
            ],
            dim=-1,
        )

        if is_video_prior:
            freqs_T_H_W_D = freqs_T_H_W_D
        else:
            first = freqs_T_H_W_D[:1, :, :, :]
            last = freqs_T_H_W_D[-1:, :, :, :]

            freqs_T_H_W_D = torch.cat([first, last], dim=0)
        return rearrange(freqs_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()

    @property
    def seq_dim(self):
        return 0

class EditWanModel(WanModel):
    def __init__(
            self,
            *args,
            temporal_skip_p: bool = False,
            temporal_skip_len: int = 10,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.temporal_skip_p = temporal_skip_p
        
        # Override the rope_position_embedding with your custom version
        # Calculate head_dim the same way as parent class
        d = self.dim // self.num_heads
        
        # Replace with your custom VideoRopePosition3DEmb
        # You can modify these parameters as needed
        if self.temporal_skip_p:
            self.rope_position_embedding = TemproalSkipVideoRopePosition3DEmb(
                head_dim=d,
                len_h=128,
                len_w=128,
                len_t=32,
                temporal_skip_len=temporal_skip_len,
            )
