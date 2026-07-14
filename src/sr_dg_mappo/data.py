"""Synthetic partially observed geometric scenes used by the prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from sr_dg_mappo.groups import apply_isometry, rotation_frames, transform_poses, world_to_local


@dataclass
class SceneBatch:
    """Batched agent poses, targets, graph, and local target maps."""

    positions: Tensor
    frames: Tensor
    targets: Tensor
    adjacency: Tensor
    local_points: Tensor
    visibility: Tensor

    def transformed(self, matrix: Tensor, translation: Optional[Tensor] = None) -> "SceneBatch":
        """Return the same physical scene after a global coordinate transformation."""

        positions, frames = transform_poses(
            self.positions, self.frames, matrix, translation=translation
        )
        targets = apply_isometry(self.targets, matrix, translation)
        local_points = targets_in_agent_frames(positions, frames, targets)
        local_points = local_points * self.visibility.unsqueeze(-1)
        return SceneBatch(
            positions=positions,
            frames=frames,
            targets=targets,
            adjacency=self.adjacency.clone(),
            local_points=local_points,
            visibility=self.visibility.clone(),
        )


def targets_in_agent_frames(positions: Tensor, frames: Tensor, targets: Tensor) -> Tensor:
    """Express every target in every agent frame."""

    expanded_targets = targets.unsqueeze(1).expand(-1, positions.shape[1], -1, -1)
    return world_to_local(expanded_targets, positions, frames)


def _ensure_connected(adjacency: Tensor, distances: Tensor) -> Tensor:
    """Add a distance-minimal spanning tree to every undirected graph."""

    adjacency = adjacency.clone()
    batch_size, num_agents, _ = adjacency.shape
    for batch_index in range(batch_size):
        selected = torch.zeros(num_agents, dtype=torch.bool, device=adjacency.device)
        selected[0] = True
        for _ in range(num_agents - 1):
            crossing = selected[:, None] & ~selected[None, :]
            candidate = distances[batch_index].masked_fill(~crossing, float("inf"))
            flat_index = int(candidate.argmin().item())
            source = flat_index // num_agents
            target = flat_index % num_agents
            adjacency[batch_index, source, target] = 1.0
            adjacency[batch_index, target, source] = 1.0
            selected[target] = True
    identity = torch.eye(num_agents, dtype=torch.bool, device=adjacency.device)
    return adjacency.masked_fill(identity.unsqueeze(0), 0.0)


def sample_scenes(
    batch_size: int,
    num_agents: int = 6,
    num_targets: int = 2,
    world_scale: float = 1.0,
    observation_radius: float = 0.85,
    communication_radius: float = 0.9,
    ensure_connected: bool = True,
    ensure_target_observed: bool = True,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> SceneBatch:
    """Sample predator-prey-style scenes with partial target visibility."""

    if batch_size < 1 or num_agents < 1 or num_targets < 1:
        raise ValueError("batch_size, num_agents, and num_targets must be positive")

    positions = (torch.rand(batch_size, num_agents, 2, device=device, generator=generator) - 0.5)
    positions = positions * (2.0 * world_scale)
    targets = (torch.rand(batch_size, num_targets, 2, device=device, generator=generator) - 0.5)
    targets = targets * (2.0 * world_scale)
    angles = torch.rand(batch_size, num_agents, device=device, generator=generator)
    frames = rotation_frames(angles * (2.0 * torch.pi))

    agent_distances = torch.cdist(positions, positions)
    adjacency = (agent_distances <= communication_radius).to(positions.dtype)
    identity = torch.eye(num_agents, dtype=torch.bool, device=positions.device)
    adjacency = adjacency.masked_fill(identity.unsqueeze(0), 0.0)
    if ensure_connected and num_agents > 1:
        adjacency = _ensure_connected(adjacency, agent_distances)

    target_distances = torch.cdist(positions, targets)
    visibility = target_distances <= observation_radius
    if ensure_target_observed:
        unseen = ~visibility.any(dim=1)
        if unseen.any():
            nearest = target_distances.argmin(dim=1)
            forced = torch.zeros_like(visibility)
            forced.scatter_(1, nearest.unsqueeze(1), unseen.unsqueeze(1))
            visibility = visibility | forced

    local_points = targets_in_agent_frames(positions, frames, targets)
    local_points = local_points * visibility.unsqueeze(-1)
    return SceneBatch(
        positions=positions,
        frames=frames,
        targets=targets,
        adjacency=adjacency,
        local_points=local_points,
        visibility=visibility.to(positions.dtype),
    )
