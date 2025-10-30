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

import numpy as np
import pytest
import torch

from chronoedit._ext.imaginaire.modules.camera import Camera, Quaternion


def _make_pose(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.cat([R, t.unsqueeze(-1)], dim=-1)


def _random_unit_quaternion(batch_shape=()) -> torch.Tensor:
    q = torch.randn(*batch_shape, 4)
    return Quaternion.normalize(q)


def _rand_quaternion(dtype: torch.dtype = torch.float32, device: str = "cpu") -> torch.Tensor:
    q = torch.randn(4, dtype=dtype, device=device)
    return Quaternion.normalize(q)


@pytest.mark.L0
@pytest.mark.GPU
def test_camera_invert_pose_identity():
    R = torch.eye(3)
    t = torch.zeros(3)
    pose = _make_pose(R, t)
    pose_inv = Camera.invert_pose(pose)
    assert torch.allclose(pose, pose_inv)


@pytest.mark.L0
@pytest.mark.GPU
def test_camera_invert_pose_roundtrip_points():
    # Create a valid random rotation via quaternion and random translation
    q = _random_unit_quaternion()
    R = Quaternion.to_rotation_matrix(q)
    t = torch.tensor([0.3, -1.2, 2.5])
    pose = _make_pose(R, t)

    points = torch.randn(7, 3)
    pc = Camera.world2camera(points, pose)
    pw = Camera.camera2world(pc, pose)
    assert torch.allclose(points, pw, atol=1e-5)


@pytest.mark.L0
@pytest.mark.GPU
def test_camera_compose_poses_matches_matrix_product():
    # Build two random valid poses
    q1 = _random_unit_quaternion()
    q2 = _random_unit_quaternion()
    R1 = Quaternion.to_rotation_matrix(q1)
    R2 = Quaternion.to_rotation_matrix(q2)
    t1 = torch.tensor([0.1, 0.2, -0.3])
    t2 = torch.tensor([-1.0, 0.5, 0.7])
    pose1 = _make_pose(R1, t1)
    pose2 = _make_pose(R2, t2)

    # Compose using implementation
    pose_comp = Camera.compose_poses([pose1, pose2])

    # Compose in homogeneous 4x4 explicitly: H = H2 @ H1
    def to_h(T):
        return torch.cat([torch.cat([T[..., :3], T[..., 3:]], dim=-1), torch.tensor([[0.0, 0.0, 0.0, 1.0]])])

    H1 = to_h(pose1)
    H2 = to_h(pose2)
    Hc = H2 @ H1
    # Back to 3x4
    pose_expected = Hc[:3, :]

    assert torch.allclose(pose_comp, pose_expected, atol=1e-5)


@pytest.mark.L0
@pytest.mark.GPU
def test_camera_image_camera_inversion():
    K = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    # Random camera-space points in homogeneous depth-1 coordinates
    cam_pts = torch.randn(11, 3)
    img_pts = Camera.camera2image(cam_pts, K)
    cam_pts_rec = Camera.image2camera(img_pts, K)
    assert torch.allclose(cam_pts, cam_pts_rec, atol=1e-6)


@pytest.mark.L0
@pytest.mark.GPU
def test_camera_image_camera_inversion_batched():
    # Batch of intrinsics
    K = torch.stack(
        [
            torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
            torch.tensor([[800.0, 0.0, 100.0], [0.0, 600.0, 50.0], [0.0, 0.0, 1.0]]),
        ],
        dim=0,
    )  # [B,3,3]
    B, N = 2, 13
    cam_pts = torch.randn(B, N, 3)
    img_pts = Camera.camera2image(cam_pts, K)
    cam_pts_rec = Camera.image2camera(img_pts, K)
    assert torch.allclose(cam_pts, cam_pts_rec, atol=1e-6)


@pytest.mark.L0
@pytest.mark.GPU
def test_intrinsics_param_matrix_roundtrip():
    params = torch.tensor([500.0, 600.0, 320.0, 240.0])
    K = Camera.intrinsic_params_to_matrices(params)
    params_rec = Camera.intrinsic_matrices_to_params(K)
    assert torch.allclose(params, params_rec, atol=1e-6)


@pytest.mark.L0
@pytest.mark.GPU
def test_intrinsics_param_matrix_roundtrip_batched():
    params = torch.tensor(
        [
            [500.0, 600.0, 320.0, 240.0],
            [800.0, 400.0, 100.0, 50.0],
        ]
    )
    K = Camera.intrinsic_params_to_matrices(params)
    params_rec = Camera.intrinsic_matrices_to_params(K)
    assert torch.allclose(params, params_rec, atol=1e-6)


@pytest.mark.L0
@pytest.mark.GPU
def test_camera_rays_identity_intrinsics():
    # Identity extrinsics -> camera and world frames coincide
    R = torch.eye(3)
    t = torch.zeros(3)
    pose = _make_pose(R, t)

    # Simple intrinsics
    fx, fy, cx, cy = 100.0, 150.0, 2.0, 1.0
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    H, W = 2, 3
    rays = Camera.get_camera_rays(pose, K, (H, W))

    # Check a couple of pixels for expected ray directions at depth=1
    def expected_ray(x, y):
        xn = (x + 0.5 - cx) / fx
        yn = (y + 0.5 - cy) / fy
        return torch.tensor([xn, yn, 1.0])

    # (y=0,x=0)
    v = expected_ray(0, 0)
    v = v / v.norm()
    assert torch.allclose(rays[0], v, atol=1e-6)
    # (y=1,x=2)
    idx = 1 * W + 2
    v = expected_ray(2, 1)
    v = v / v.norm()
    assert torch.allclose(rays[idx], v, atol=1e-6)

    # All rays should be unit length
    assert torch.allclose(rays.norm(dim=-1), torch.ones(H * W), atol=1e-6)


@pytest.mark.L0
@pytest.mark.GPU
def test_plucker_rays_properties():
    # Identity pose and simple intrinsics
    R = torch.eye(3)
    t = torch.zeros(3)
    pose = _make_pose(R, t)
    fx, fy, cx, cy = 100.0, 150.0, 2.0, 1.0
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    H, W = 2, 3
    plucker = Camera.get_plucker_rays(pose, K, (H, W))  # [HW,6]
    moment, direction = plucker[..., :3], plucker[..., 3:]
    # Directions unit
    assert torch.allclose(direction.norm(dim=-1), torch.ones(H * W), atol=1e-6)
    # For camera at origin, m = o × d = 0
    assert torch.allclose(moment, torch.zeros(H * W, 3), atol=1e-6)


@pytest.mark.L0
@pytest.mark.GPU
def test_camera_get_camera_center_matches_formula():
    q = _random_unit_quaternion()
    R = Quaternion.to_rotation_matrix(q)
    t = torch.tensor([0.3, -1.2, 2.5])
    pose = _make_pose(R, t)
    center = Camera.get_camera_center(pose)
    # Center should satisfy R @ C + t = 0 -> C = -R^T t
    expected = -R.T @ t
    assert torch.allclose(center, expected, atol=1e-6)


@pytest.mark.L0
@pytest.mark.GPU
def test_quaternion_invert_roundtrip():
    q = torch.tensor([0.2, -0.5, 0.7, 0.1])
    qn = Quaternion.normalize(q)
    inv = Quaternion.invert(qn)

    # q ⊗ q^{-1} = identity and q^{-1} ⊗ q = identity
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    prod1 = Quaternion.multiply(qn, inv)
    prod2 = Quaternion.multiply(inv, qn)
    assert torch.allclose(prod1, identity, atol=1e-6)
    assert torch.allclose(prod2, identity, atol=1e-6)

    # invert(invert(q)) = q for unit quaternions
    q_back = Quaternion.invert(inv)
    assert torch.allclose(q_back, qn, atol=1e-6)


@pytest.mark.L0
@pytest.mark.GPU
def test_quaternion_to_from_rotation_matrix_roundtrip():
    q = _random_unit_quaternion()
    R = Quaternion.to_rotation_matrix(q)
    q2 = Quaternion.from_rotation_matrix(R)
    # Account for double-cover ambiguity: q == -q
    err1 = (q - q2).abs().max()
    err2 = (q + q2).abs().max()
    assert min(err1.item(), err2.item()) < 1e-5


@pytest.mark.L0
@pytest.mark.GPU
def test_quaternion_rotation_matrix_properties():
    q = _random_unit_quaternion()
    R = Quaternion.to_rotation_matrix(q)
    I = torch.eye(3)
    assert torch.allclose(R.T @ R, I, atol=1e-6)
    det = torch.det(R)
    assert torch.allclose(det, torch.tensor(1.0), atol=1e-5)


@pytest.mark.L0
@pytest.mark.GPU
def test_quaternion_multiply_matches_rotation_composition():
    q1 = _random_unit_quaternion()
    q2 = _random_unit_quaternion()
    q12 = Quaternion.multiply(q1, q2)

    R1 = Quaternion.to_rotation_matrix(q1)
    R2 = Quaternion.to_rotation_matrix(q2)
    R12 = Quaternion.to_rotation_matrix(q12)

    assert torch.allclose(R12, R1 @ R2, atol=1e-5)


# -----------------------------------------------------------------------------
# bf16-oriented tests
# -----------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.GPU
def test_check_valid_pose_bf16_tolerance():
    device = "cpu"
    dtype = torch.bfloat16
    q = _rand_quaternion(dtype=dtype, device=device)
    R = Quaternion.to_rotation_matrix(q)
    t = torch.zeros(3, 1, dtype=dtype, device=device)
    cam_pose = torch.cat([R, t], dim=-1)
    Camera._check_valid_pose(cam_pose)


@pytest.mark.L0
@pytest.mark.GPU
def test_world2camera_camera2world_roundtrip_bf16():
    device = "cpu"
    dtype = torch.bfloat16
    q = _rand_quaternion(dtype=dtype, device=device)
    R = Quaternion.to_rotation_matrix(q)
    t = torch.tensor([[0.1], [-0.2], [0.3]], dtype=dtype, device=device)
    cam_pose = torch.cat([R, t], dim=-1)

    points = torch.tensor([[0.5, -0.1, 2.0], [1.2, 0.3, 4.0], [-0.7, 0.9, 1.5]], dtype=dtype, device=device)
    cam = Camera.world2camera(points, cam_pose)
    back = Camera.camera2world(cam, cam_pose)
    assert torch.allclose(back.to(torch.float32), points.to(torch.float32), rtol=1e-3, atol=2e-2)


@pytest.mark.L0
@pytest.mark.GPU
def test_image2camera_camera2image_roundtrip_bf16():
    device = "cpu"
    dtype = torch.bfloat16
    fx, fy, cx, cy = 500.0, 480.0, 320.0, 240.0
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=dtype, device=device)
    pts_cam = torch.tensor([[-0.2, 0.1, 1.0], [0.3, -0.4, 2.0], [0.0, 0.0, 3.0]], dtype=dtype, device=device)
    pix = Camera.camera2image(pts_cam, K)
    rec = Camera.image2camera(pix, K)
    assert torch.allclose(rec.to(torch.float32), pts_cam.to(torch.float32), rtol=1e-3, atol=2e-2)


@pytest.mark.L0
@pytest.mark.GPU
def test_get_camera_rays_bf16_unit_norm():
    device = "cpu"
    dtype = torch.bfloat16
    q = _rand_quaternion(dtype=dtype, device=device)
    R = Quaternion.to_rotation_matrix(q)
    t = torch.zeros(3, 1, dtype=dtype, device=device)
    cam_pose = torch.cat([R, t], dim=-1)
    fx, fy, cx, cy = 400.0, 400.0, 1.0, 1.0
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=dtype, device=device)
    rays = Camera.get_camera_rays(cam_pose, K, image_size=(3, 3))
    norms = rays.to(torch.float32).norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-3, atol=2e-2)


@pytest.mark.L0
@pytest.mark.GPU
def test_quaternion_roundtrip_bf16():
    device = "cpu"
    dtype = torch.bfloat16
    q = _rand_quaternion(dtype=dtype, device=device)
    R = Quaternion.to_rotation_matrix(q)
    q2 = Quaternion.from_rotation_matrix(R)
    d = torch.sum(q.to(torch.float32) * q2.to(torch.float32))
    assert torch.isfinite(d)
    assert abs(float(d)) >= 0.98


@pytest.mark.L0
@pytest.mark.GPU
def test_xyzw_t_pose_roundtrip():
    # Random unit quaternion and translation
    q = _random_unit_quaternion()
    t = torch.randn(3)
    vec = torch.cat([q, t], dim=-1)
    pose = Camera.extrinsic_params_to_matrices(vec)
    vec_rec = Camera.extrinsic_matrices_to_params(pose)
    # Compare quaternion up to sign
    q_rec, t_rec = vec_rec[:4], vec_rec[4:]
    err1 = (q - q_rec).abs().max()
    err2 = (q + q_rec).abs().max()
    assert min(err1.item(), err2.item()) < 1e-5
    assert torch.allclose(t, t_rec, atol=1e-6)


@pytest.mark.L0
@pytest.mark.GPU
def test_xyzw_t_to_pose_matches_rotation_matrix():
    q = _random_unit_quaternion()
    t = torch.tensor([0.4, -0.2, 1.1])
    pose = Camera.extrinsic_params_to_matrices(torch.cat([q, t], dim=-1))
    R = Quaternion.to_rotation_matrix(q)
    assert torch.allclose(pose[:3, :3], R, atol=1e-6)
    assert torch.allclose(pose[:3, 3], t, atol=1e-6)


@pytest.mark.L0
@pytest.mark.GPU
def test_numpy_world_camera_roundtrip():
    q = _random_unit_quaternion()
    R = Quaternion.to_rotation_matrix(q)
    t = torch.randn(3)
    pose = _make_pose(R, t)
    points = torch.randn(9, 3)

    pose_np = pose.detach().cpu().numpy()
    points_np = points.detach().cpu().numpy()

    pc_np = Camera.world2camera(points_np, pose_np)
    pw_np = Camera.camera2world(pc_np, pose_np)
    assert np.allclose(points_np, pw_np, atol=1e-5)


@pytest.mark.L0
@pytest.mark.GPU
def test_numpy_image_camera_roundtrip():
    K = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    cam_pts = torch.randn(7, 3)
    K_np = K.detach().cpu().numpy()
    cam_pts_np = cam_pts.detach().cpu().numpy()
    img_pts_np = Camera.camera2image(cam_pts_np, K_np)
    cam_pts_rec_np = Camera.image2camera(img_pts_np, K_np)
    assert np.allclose(cam_pts_np, cam_pts_rec_np, atol=1e-6)


@pytest.mark.L0
@pytest.mark.GPU
def test_numpy_intrinsic_param_matrix_roundtrip():
    params_np = np.array([500.0, 600.0, 320.0, 240.0], dtype=np.float32)
    K_np = Camera.intrinsic_params_to_matrices(params_np)
    params_rec_np = Camera.intrinsic_matrices_to_params(K_np)
    assert np.allclose(params_np, params_rec_np, atol=1e-6)


@pytest.mark.L0
@pytest.mark.GPU
def test_numpy_extrinsic_param_matrix_roundtrip():
    q = _random_unit_quaternion()
    t = torch.randn(3)
    vec_np = torch.cat([q, t], dim=-1).detach().cpu().numpy()
    pose_np = Camera.extrinsic_params_to_matrices(vec_np)
    vec_rec_np = Camera.extrinsic_matrices_to_params(pose_np)
    q_np, t_np = vec_np[:4], vec_np[4:]
    q_rec_np, t_rec_np = vec_rec_np[:4], vec_rec_np[4:]
    err1 = np.max(np.abs(q_np - q_rec_np))
    err2 = np.max(np.abs(q_np + q_rec_np))
    assert min(err1, err2) < 1e-5
    assert np.allclose(t_np, t_rec_np, atol=1e-6)


@pytest.mark.L0
@pytest.mark.GPU
def test_numpy_rays_and_plucker():
    R = torch.eye(3)
    t = torch.zeros(3)
    pose_np = _make_pose(R, t).detach().cpu().numpy()
    fx, fy, cx, cy = 100.0, 150.0, 2.0, 1.0
    K_np = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    H, W = 2, 3

    rays_np = Camera.get_camera_rays(pose_np, K_np, (H, W))
    norms = np.linalg.norm(rays_np, axis=-1)
    assert np.allclose(norms, np.ones(H * W), atol=1e-6)

    def expected_ray(x, y):
        xn = (x + 0.5 - cx) / fx
        yn = (y + 0.5 - cy) / fy
        v = np.array([xn, yn, 1.0], dtype=np.float32)
        return v / np.linalg.norm(v)

    assert np.allclose(rays_np[0], expected_ray(0, 0), atol=1e-6)

    plucker_np = Camera.get_plucker_rays(pose_np, K_np, (H, W))
    m_np, d_np = plucker_np[..., :3], plucker_np[..., 3:]
    assert np.allclose(np.linalg.norm(d_np, axis=-1), np.ones(H * W), atol=1e-6)
    assert np.allclose(m_np, np.zeros((H * W, 3)), atol=1e-6)
