"""Planar group actions and coordinate-frame transport."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor


def rotation_frames(angles: Tensor) -> Tensor:
    """Return local-to-world SO(2) frame matrices for ``angles``."""

    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    row_0 = torch.stack((cosine, -sine), dim=-1)
    row_1 = torch.stack((sine, cosine), dim=-1)
    return torch.stack((row_0, row_1), dim=-2)


def d4_matrices(
    *, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None
) -> Tensor:
    """Return the eight orthogonal matrices of the square's D4 symmetry group."""

    angles = torch.arange(4, dtype=dtype, device=device) * (math.pi / 2.0)
    rotations = rotation_frames(angles)
    reflection = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype, device=device)
    reflected = torch.matmul(rotations, reflection)
    return torch.cat((rotations, reflected), dim=0)


def apply_isometry(points: Tensor, matrix: Tensor, translation: Optional[Tensor] = None) -> Tensor:
    """Apply ``matrix @ point + translation`` to arbitrary point tensor prefixes."""

    transformed = torch.einsum("ij,...j->...i", matrix, points)
    if translation is not None:
        transformed = transformed + translation
    return transformed


def transform_poses(
    positions: Tensor,
    frames: Tensor,
    matrix: Tensor,
    translation: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Apply a global isometry to positions and local-to-world frames."""

    transformed_positions = apply_isometry(positions, matrix, translation)
    transformed_frames = torch.einsum("ij,...jk->...ik", matrix, frames)
    return transformed_positions, transformed_frames


def world_to_local(points: Tensor, position: Tensor, frame: Tensor) -> Tensor:
    """Express world points in one or more local orthogonal frames."""

    delta = points - position.unsqueeze(-2)
    return torch.einsum("...ij,...mi->...mj", frame, delta)


def local_to_world(points: Tensor, position: Tensor, frame: Tensor) -> Tensor:
    """Express local points in the world frame."""

    rotated = torch.einsum("...ij,...mj->...mi", frame, points)
    return rotated + position.unsqueeze(-2)


def all_receiver_transports(points: Tensor, positions: Tensor, frames: Tensor) -> Tensor:
    """Transport every sender map into every receiver frame.

    Args:
        points: Sender-frame points with shape ``[B, N_sender, M, 2]``.
        positions: Agent world positions with shape ``[B, N, 2]``.
        frames: Agent local-to-world frames with shape ``[B, N, 2, 2]``.

    Returns:
        Points with shape ``[B, N_receiver, N_sender, M, 2]``.
    """

    if points.ndim != 4 or points.shape[-1] != 2:
        raise ValueError("points must have shape [batch, agents, targets, 2]")
    world = positions.unsqueeze(-2) + torch.einsum("bnij,bnmj->bnmi", frames, points)
    delta = world.unsqueeze(1) - positions[:, :, None, None, :]
    return torch.einsum("brij,brsmi->brsmj", frames, delta)


def orthogonality_error(frames: Tensor) -> Tensor:
    """Return the maximum absolute error in ``F^T F = I``."""

    gram = torch.matmul(frames.transpose(-1, -2), frames)
    identity = torch.eye(2, dtype=frames.dtype, device=frames.device)
    return (gram - identity).abs().max()
