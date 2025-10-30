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


def volume_render_rays(
    nerf: torch.nn.Module,
    center: torch.Tensor,
    ray_unit: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    num_samples: int,
    stratified: bool = False,
    solid_background: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given a NeRF, volume render the color and density of the rays within the near/far distance bounds.

    Args:
        nerf (torch.nn.Module): A neural field predicting the color and density from input 3D points and rays.
        center (torch.Tensor [...,3]): Center of rays.
        ray_unit (torch.Tensor [...,3]): Direction of rays (should be of unit norm).
        near (torch.Tensor [...,1]): The near bound of the range to volume render the rays.
        far (torch.Tensor [...,1]): The far bound of the range to volume render the rays.
        num_samples (int): Number of sampled points.
        stratified (bool): Whether to enable stratified sampling.
        solid_background (bool): Whether the background is assumed to be solid. Enabling this would make the sum of
            alphas along the ray to be 1. This should not be enabled if the background would be modeled separately.

    Returns:
        rgb (torch.Tensor [...,3]): The volme rendered rgb values.
        opacity (torch.Tensor [...,1]): The volme rendered opacity values.
        weights (torch.Tensor [...,N,1]): The weights for compositing the samples.
        points (torch.Tensor [...,N,3]): The sampled point locations.
        dists (torch.Tensor [...,N]): The distance of the sampled points to the camera center.
    """
    # Sample 3D points within the near/far range for all rays.
    with torch.no_grad():
        dists = sample_dists(near, far, num_samples, stratified=stratified)  # [...,N]
    points = center[..., None, :] + ray_unit[..., None, :] * dists[..., None]  # [...,N,3]
    rays_unit = ray_unit[..., None, :].expand_as(points).contiguous()  # [...,N,3]
    # Feed-forward pass on the neural field.
    rgbs, densities = nerf(points, rays_unit)  # [...,N,3],[...,N,1]
    # Volume rendering.
    dist_far = None if solid_background else far[..., None]
    alphas = volume_rendering_alphas(densities, dists[..., None], dist_far=dist_far)  # [...,N,1]
    weights = alpha_compositing_weights(alphas)  # [...,N,1]
    opacity = composite(1.0, weights)  # [...,1] # type: ignore
    rgb = composite(rgbs, weights)  # [...,3]
    return rgb, opacity, weights, points, dists


def volume_rendering_alphas(
    densities: torch.Tensor, dists: torch.Tensor, dist_far: torch.Tensor | None = None
) -> torch.Tensor:
    """Computes the alpha weights for volume rendering (density-based).

    Args:
        densities (torch.Tensor [...,samples,1]): Density values.
        dists (torch.Tensor [...,samples,1]): Distance from sampled point to camera center.
        dist_far (torch.Tensor [...,1,1] | None): Farthest distance of the volume rendering range. Defaults to a
            large value if not provided, which equivalently assumes full opacity at the farthest end, making
            sum(alphas) = 1. (default: None)

    Returns:
        alphas (torch.Tensor [...,samples,1]): The opacity of each sampled point (in [0,1]).
    """
    if dist_far is None:
        dist_far = torch.empty_like(dists[..., :1, :]).fill_(1e10)  # [...,1,1]
    dists = torch.cat([dists, dist_far], dim=-2)  # [...,N+1,1]
    # Volume rendering: compute rendering weights (using quadrature).
    dist_intvs = dists[..., 1:, :] - dists[..., :-1, :]  # [...,N,1]
    sigma_delta = densities * dist_intvs  # [...,N,1]
    alphas = 1 - (-sigma_delta).exp_()  # [...,N,1]
    return alphas


def alpha_compositing_weights(alphas: torch.Tensor) -> torch.Tensor:
    """Alpha compositing to compute the blending weights.

    Args:
        alphas (torch.Tensor [...,samples,1]): The opacity of each sampled point (in [0,1]).

    Returns:
        weights (torch.Tensor [...,samples,1]): The compositing weights (in [0,1]).
    """
    alphas_front = torch.cat([torch.zeros_like(alphas[..., :1, :]), alphas[..., :-1, :]], dim=2)  # [...,N,1]
    with torch.amp.autocast("cuda", enabled=False):  # Half precision may cause numerical instability.
        visibility = (1 - alphas_front).cumprod(dim=-2)  # [...,N,1]
    weights = alphas * visibility  # [...,N,1]
    return weights


def composite(quantities: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Composite the samples to render the corresponding pixels.

    Args:
        quantities (torch.Tensor [...,samples,channels]): The quantity to compute the weighted sum.
        weights (torch.Tensor [...,samples,1]): The compositing weights (in [0,1]).

    Returns:
        quantity (torch.Tensor [...,channels]): The expected (rendered) quantity.
    """
    # Integrate RGB and depth weighted by probability.
    quantity = (quantities * weights).sum(dim=-2)  # [...,K]
    return quantity


@torch.no_grad()
def sample_dists(near: torch.Tensor, far: torch.Tensor, num_samples: int, stratified: bool = False) -> torch.Tensor:
    """Sample points along view rays given the near/far bounds.

    Args:
        near (torch.Tensor [...,1]): The near bound of the range to sample the points from.
        far (torch.Tensor [...,1]): The far bound of the range to sample the points from.
        num_samples (int): Number of sampled points.
        stratified (bool): Whether to use stratified sampling (uniform within each interval bin); otherwise,
            sample at the midpoint of each interval (default: False).

    Returns:
        dists (torch.Tensor [...,N]): The distance of the sampled points to the camera center.
    """
    if stratified:
        rands = torch.rand(*near.shape[:-1], num_samples, dtype=near.dtype, device=near.device)  # [...,N]
    else:
        rands = torch.empty(*near.shape[:-1], num_samples, dtype=near.dtype, device=near.device).fill_(0.5)  # [...,N]
    base = torch.arange(num_samples, dtype=near.dtype, device=near.device).repeat(*near.shape[:-1], 1)  # [...,N]
    rands = (rands + base) / num_samples  # [...,N]
    dists = rands * (far - near) + near  # [...,N]
    return dists
