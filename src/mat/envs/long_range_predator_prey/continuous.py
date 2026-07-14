"""Torch-vectorized continuous long-range predator-prey environment.

The Gymnasium wrapper exposes the multi-agent API used by this repository:
reset() -> obs, share_obs, available_actions
step(actions) -> obs, share_obs, rewards, dones, infos, available_actions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


def _wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


@dataclass
class LongRangePredatorPreyConfig:
    num_envs: int = 1
    num_predators: int = 6
    num_prey: int = 2
    world_size: float = 6.0
    dt: float = 0.1
    episode_length: int = 200
    random_start_positions: bool = False
    init_min_predator_dist: float = 0.45
    init_min_prey_dist: float = 0.45
    init_min_prey_predator_dist: float = 1.0
    obs_radius: float = 1.8
    comm_radius: float = 2.2
    ensure_connected_comm_graph: bool = True
    ensure_prey_visible: bool = True
    capture_radius: float = 0.35
    capture_k: int = 2
    predator_max_speed: float = 0.22
    predator_max_omega: float = 2.84
    prey_speed_ratio: float = 0.85
    prey_max_omega: float = 2.3
    prey_avoid_radius: float = 2.4
    collision_radius: float = 0.18
    collision_resolution_iters: int = 4
    capture_reward: float = 10.0
    all_captured_bonus: float = 15.0
    progress_scale: float = 0.6
    surround_scale: float = 0.05
    collision_penalty: float = 0.08
    control_penalty: float = 0.01
    time_penalty: float = 0.005
    wall_margin: float = 0.45
    wall_penalty: float = 0.02
    prey_noise_scale: float = 0.12
    device: str = "cpu"
    seed: Optional[int] = None


class LongRangePredatorPreyTorchCore:
    """Batched tensor dynamics for continuous predator-prey."""

    def __init__(self, cfg: LongRangePredatorPreyConfig):
        self.cfg = cfg
        if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(cfg.device)
        self.generator = torch.Generator(device=self.device)
        if cfg.seed is not None:
            self.generator.manual_seed(int(cfg.seed))

        self.n_envs = int(cfg.num_envs)
        self.n_predators = int(cfg.num_predators)
        self.n_prey = int(cfg.num_prey)
        self.half_world = float(cfg.world_size) / 2.0
        self.max_steps = int(cfg.episode_length)

        self.predator_pose = torch.zeros(self.n_envs, self.n_predators, 3, device=self.device)
        self.prey_pose = torch.zeros(self.n_envs, self.n_prey, 3, device=self.device)
        self.prey_alive = torch.ones(self.n_envs, self.n_prey, dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        self.prev_min_dist = torch.zeros(self.n_envs, self.n_prey, device=self.device)
        self.last_collision_count = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        self.reset()

    @property
    def obs_dim(self) -> int:
        return 5 + 5 * self.n_predators + 5 * self.n_prey

    @property
    def share_obs_dim(self) -> int:
        return 1 + 4 * self.n_predators + 5 * self.n_prey

    def seed(self, seed: int) -> None:
        self.generator.manual_seed(int(seed))

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        n = env_ids.numel()

        if self.cfg.random_start_positions:
            pred_xy, pred_theta, prey_xy, prey_theta = self._random_start_poses(n)
        else:
            pred_angles = torch.linspace(0.0, 2.0 * torch.pi, self.n_predators + 1, device=self.device)[:-1]
            pred_angles = pred_angles.unsqueeze(0).repeat(n, 1)
            pred_angles = pred_angles + self._rand(n, self.n_predators) * 0.35
            pred_radius = self.half_world * 0.65
            pred_xy = torch.stack([torch.cos(pred_angles), torch.sin(pred_angles)], dim=-1) * pred_radius
            pred_xy = pred_xy + (self._rand(n, self.n_predators, 2) - 0.5) * 0.55
            pred_theta = _wrap_angle(pred_angles + torch.pi + (self._rand(n, self.n_predators) - 0.5) * 0.5)

            prey_xy = (self._rand(n, self.n_prey, 2) - 0.5) * (self.cfg.world_size * 0.45)
            prey_theta = (self._rand(n, self.n_prey) - 0.5) * 2.0 * torch.pi

        self.predator_pose[env_ids, :, :2] = self._clamp_xy(pred_xy)
        self.predator_pose[env_ids, :, 2] = pred_theta
        self.prey_pose[env_ids, :, :2] = self._clamp_xy(prey_xy)
        self.prey_pose[env_ids, :, 2] = prey_theta
        self.prey_alive[env_ids] = True
        self._resolve_prey_collisions(env_ids)
        self.steps[env_ids] = 0
        self.prev_min_dist[env_ids] = self._predator_prey_dist(env_ids).amin(dim=1)
        return self.get_obs()

    def step(self, actions: torch.Tensor):
        actions = actions.to(device=self.device, dtype=torch.float32)
        actions = actions.reshape(self.n_envs, self.n_predators, 2).clamp(-1.0, 1.0)

        old_min_dist = self.prev_min_dist.clone()
        self._step_predators(actions)
        self.last_collision_count = self._collision_count()
        self._resolve_predator_collisions()
        self._step_prey()
        self._resolve_prey_collisions()
        new_captures, close_counts = self._update_captures()
        self.steps += 1

        dist = self._predator_prey_dist()
        masked_dist = torch.where(self.prey_alive.unsqueeze(1), dist, torch.zeros_like(dist))
        min_dist = torch.where(
            self.prey_alive,
            masked_dist.amin(dim=1),
            old_min_dist,
        )
        self.prev_min_dist = min_dist.detach()

        progress = ((old_min_dist - min_dist) * self.prey_alive.float()).sum(dim=1)
        capture_reward = new_captures.float().sum(dim=1) * self.cfg.capture_reward
        all_captured = ~self.prey_alive.any(dim=1)
        all_bonus = all_captured.float() * self.cfg.all_captured_bonus

        max_close = close_counts.float().max(dim=1).values / max(float(self.cfg.capture_k), 1.0)
        surround = torch.clamp(max_close, 0.0, 1.0) * self.cfg.surround_scale

        collision_penalty = self.last_collision_count.float() * self.cfg.collision_penalty
        wall_penalty = self._wall_contact_count().float() * self.cfg.wall_penalty
        control_penalty = actions.square().mean(dim=(1, 2)) * self.cfg.control_penalty
        team_reward = (
            self.cfg.progress_scale * progress
            + capture_reward
            + all_bonus
            + surround
            - collision_penalty
            - wall_penalty
            - control_penalty
            - self.cfg.time_penalty
        )

        timeout = self.steps >= self.max_steps
        done_env = all_captured | timeout
        rewards = team_reward[:, None, None].repeat(1, self.n_predators, 1)
        dones = done_env[:, None].repeat(1, self.n_predators)

        infos = self._infos(new_captures, close_counts, all_captured, timeout)
        obs, share_obs, available_actions = self.get_obs()
        return obs, share_obs, rewards, dones, infos, available_actions

    def get_obs(self):
        obs = self._local_obs()
        share_obs = self._share_obs()
        available_actions = torch.ones(self.n_envs, self.n_predators, 1, device=self.device)
        return obs, share_obs, available_actions

    def get_visibility_matrix(self) -> torch.Tensor:
        pred_xy = self.predator_pose[:, :, :2]
        dist = torch.cdist(pred_xy, pred_xy)
        adj = (dist <= self.cfg.comm_radius).float()
        if self.cfg.ensure_connected_comm_graph:
            adj = self._ensure_connected_adjacency(adj, dist)
        return adj

    def _ensure_connected_adjacency(self, adj: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        if self.n_predators <= 1:
            return adj

        adj = adj.clone()
        for env_i in range(self.n_envs):
            env_adj = adj[env_i]
            env_dist = dist[env_i]
            seen = [False] * self.n_predators
            components = []

            for start in range(self.n_predators):
                if seen[start]:
                    continue
                stack = [start]
                seen[start] = True
                comp = []
                while stack:
                    node = stack.pop()
                    comp.append(node)
                    neighbors = torch.nonzero(env_adj[node] > 0, as_tuple=False).flatten().tolist()
                    for nbr in neighbors:
                        if nbr == node or seen[nbr]:
                            continue
                        seen[nbr] = True
                        stack.append(nbr)
                components.append(comp)

            while len(components) > 1:
                best = None
                for comp_i, comp_a in enumerate(components[:-1]):
                    for comp_j in range(comp_i + 1, len(components)):
                        comp_b = components[comp_j]
                        for a in comp_a:
                            for b in comp_b:
                                d = float(env_dist[a, b].item())
                                if best is None or d < best[0]:
                                    best = (d, comp_i, comp_j, a, b)

                if best is None:
                    break

                _, comp_i, comp_j, a, b = best
                env_adj[a, b] = 1.0
                env_adj[b, a] = 1.0
                components[comp_i].extend(components[comp_j])
                components.pop(comp_j)

            env_adj.fill_diagonal_(1.0)

        return adj

    def get_edge_index_matrix(self, faulty_node: Optional[int] = None) -> torch.Tensor:
        adj = self.get_visibility_matrix()
        if faulty_node is not None and int(faulty_node) >= 0:
            idx = int(faulty_node)
            if idx < self.n_predators:
                adj[:, idx, :] = 0.0
                adj[:, :, idx] = 0.0

        edge_index = torch.full(
            (self.n_envs, 2, self.n_predators * self.n_predators),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        rows, cols = torch.meshgrid(
            torch.arange(self.n_predators, device=self.device),
            torch.arange(self.n_predators, device=self.device),
            indexing="ij",
        )
        flat_rows = rows.reshape(-1)
        flat_cols = cols.reshape(-1)
        for env_i in range(self.n_envs):
            valid = adj[env_i].reshape(-1) > 0
            count = int(valid.sum().item())
            edge_index[env_i, 0, :count] = flat_rows[valid]
            edge_index[env_i, 1, :count] = flat_cols[valid]
        return edge_index

    def render_rgb(self, env_id: int = 0, size: int = 512) -> np.ndarray:
        env_id = int(env_id)
        canvas = np.full((size, size, 3), 245, dtype=np.uint8)
        margin = 24

        def to_px(xy: np.ndarray):
            scaled = (xy + self.half_world) / (2.0 * self.half_world)
            px = margin + scaled * (size - 2 * margin)
            px[:, 1] = size - px[:, 1]
            return px.astype(np.int32)

        pred = self.predator_pose[env_id, :, :2].detach().cpu().numpy()
        prey = self.prey_pose[env_id, :, :2].detach().cpu().numpy()
        alive = self.prey_alive[env_id].detach().cpu().numpy()

        canvas[margin:size - margin, margin] = 30
        canvas[margin:size - margin, size - margin] = 30
        canvas[margin, margin:size - margin] = 30
        canvas[size - margin, margin:size - margin] = 30

        for p in to_px(pred):
            self._draw_disk(canvas, p, 8, np.array([30, 100, 220], dtype=np.uint8))
        for p, is_alive in zip(to_px(prey), alive):
            color = np.array([220, 70, 45], dtype=np.uint8) if is_alive else np.array([120, 120, 120], dtype=np.uint8)
            self._draw_disk(canvas, p, 7, color)
        return canvas

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.generator, device=self.device)

    def _sample_uniform_xy(self, n_envs: int, count: int) -> torch.Tensor:
        span = max(2.0 * (self.half_world - 0.05), 1e-3)
        return (self._rand(n_envs, count, 2) - 0.5) * span

    def _random_start_poses(self, n_envs: int):
        pred_xy = self._sample_uniform_xy(n_envs, self.n_predators)
        pred_min = max(float(self.cfg.init_min_predator_dist), float(self.cfg.collision_radius), 0.0)
        for agent_i in range(1, self.n_predators):
            for _ in range(200):
                dist = torch.norm(
                    pred_xy[:, agent_i: agent_i + 1, :] - pred_xy[:, :agent_i, :],
                    dim=-1,
                )
                invalid = dist.lt(pred_min).any(dim=1)
                if not bool(invalid.any().item()):
                    break
                pred_xy[invalid, agent_i, :] = self._sample_uniform_xy(
                    int(invalid.sum().item()),
                    1,
                )[:, 0, :]

        prey_xy = self._sample_uniform_xy(n_envs, self.n_prey)
        prey_min = max(float(self.cfg.init_min_prey_dist), float(self.cfg.collision_radius), 0.0)
        prey_pred_min = max(
            float(self.cfg.init_min_prey_predator_dist),
            float(self.cfg.capture_radius) * 2.0,
            float(self.cfg.collision_radius),
            0.0,
        )
        for prey_i in range(self.n_prey):
            for _ in range(300):
                pred_dist = torch.norm(
                    prey_xy[:, prey_i: prey_i + 1, :] - pred_xy,
                    dim=-1,
                )
                invalid = pred_dist.lt(prey_pred_min).any(dim=1)
                if prey_i > 0:
                    prey_dist = torch.norm(
                        prey_xy[:, prey_i: prey_i + 1, :] - prey_xy[:, :prey_i, :],
                        dim=-1,
                    )
                    invalid = invalid | prey_dist.lt(prey_min).any(dim=1)
                if not bool(invalid.any().item()):
                    break
                prey_xy[invalid, prey_i, :] = self._sample_uniform_xy(
                    int(invalid.sum().item()),
                    1,
                )[:, 0, :]

        pred_theta = (self._rand(n_envs, self.n_predators) - 0.5) * 2.0 * torch.pi
        prey_theta = (self._rand(n_envs, self.n_prey) - 0.5) * 2.0 * torch.pi
        return pred_xy, pred_theta, prey_xy, prey_theta

    def _clamp_xy(self, xy: torch.Tensor) -> torch.Tensor:
        return xy.clamp(-self.half_world + 0.05, self.half_world - 0.05)

    def _step_predators(self, actions: torch.Tensor) -> None:
        v = (actions[..., 0] + 1.0) * 0.5 * self.cfg.predator_max_speed
        omega = actions[..., 1] * self.cfg.predator_max_omega
        theta = _wrap_angle(self.predator_pose[..., 2] + omega * self.cfg.dt)
        dx = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1) * v.unsqueeze(-1) * self.cfg.dt
        self.predator_pose[..., :2] = self._clamp_xy(self.predator_pose[..., :2] + dx)
        self.predator_pose[..., 2] = theta

    def _resolve_predator_collisions(self) -> None:
        """Project overlapping predators apart using a small position-based solve."""
        if self.n_predators < 2 or self.cfg.collision_radius <= 0.0:
            return

        eye = torch.eye(self.n_predators, dtype=torch.bool, device=self.device).unsqueeze(0)
        base_angle = torch.linspace(
            0.0,
            2.0 * torch.pi,
            self.n_predators + 1,
            device=self.device,
        )[:-1]
        base_dir = torch.stack([torch.cos(base_angle), torch.sin(base_angle)], dim=-1)
        fallback = base_dir.unsqueeze(1) - base_dir.unsqueeze(0)
        fallback = fallback / torch.norm(fallback, dim=-1, keepdim=True).clamp_min(1e-6)
        fallback = fallback.unsqueeze(0)

        for _ in range(max(int(self.cfg.collision_resolution_iters), 1)):
            xy = self.predator_pose[:, :, :2]
            diff = xy.unsqueeze(2) - xy.unsqueeze(1)
            dist = torch.norm(diff, dim=-1, keepdim=True)
            direction = torch.where(dist > 1e-6, diff / dist.clamp_min(1e-6), fallback)

            overlap = (self.cfg.collision_radius - dist.squeeze(-1)).clamp_min(0.0)
            overlap = overlap.masked_fill(eye, 0.0)
            correction = (direction * (0.5 * overlap).unsqueeze(-1)).sum(dim=2)
            self.predator_pose[:, :, :2] = self._clamp_xy(xy + correction)

    def _step_prey(self) -> None:
        pred_xy = self.predator_pose[:, :, :2]
        prey_xy = self.prey_pose[:, :, :2]
        diff = prey_xy.unsqueeze(2) - pred_xy.unsqueeze(1)
        dist = torch.norm(diff, dim=-1).clamp_min(1e-4)
        close = dist <= self.cfg.prey_avoid_radius
        avoid = (diff / dist.unsqueeze(-1).square()).masked_fill(~close.unsqueeze(-1), 0.0).sum(dim=2)

        wall = torch.zeros_like(prey_xy)
        margin = self.cfg.wall_margin
        wall[..., 0] += torch.relu((-self.half_world + margin) - prey_xy[..., 0])
        wall[..., 0] -= torch.relu(prey_xy[..., 0] - (self.half_world - margin))
        wall[..., 1] += torch.relu((-self.half_world + margin) - prey_xy[..., 1])
        wall[..., 1] -= torch.relu(prey_xy[..., 1] - (self.half_world - margin))

        noise_angle = self._rand(self.n_envs, self.n_prey) * 2.0 * torch.pi
        noise = torch.stack([torch.cos(noise_angle), torch.sin(noise_angle)], dim=-1) * self.cfg.prey_noise_scale
        desired = avoid + 2.0 * wall + noise
        desired_norm = torch.norm(desired, dim=-1, keepdim=True).clamp_min(1e-6)
        desired_dir = desired / desired_norm
        desired_theta = torch.atan2(desired_dir[..., 1], desired_dir[..., 0])

        theta = self.prey_pose[..., 2]
        angle_error = _wrap_angle(desired_theta - theta)
        omega = angle_error.clamp(-self.cfg.prey_max_omega * self.cfg.dt, self.cfg.prey_max_omega * self.cfg.dt)
        theta = _wrap_angle(theta + omega)
        speed = self.cfg.predator_max_speed * self.cfg.prey_speed_ratio
        dx = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1) * speed * self.cfg.dt
        alive = self.prey_alive.unsqueeze(-1)
        self.prey_pose[..., :2] = torch.where(alive, self._clamp_xy(self.prey_pose[..., :2] + dx), self.prey_pose[..., :2])
        self.prey_pose[..., 2] = torch.where(self.prey_alive, theta, self.prey_pose[..., 2])

    def _resolve_prey_collisions(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Project overlapping live prey apart."""
        if self.n_prey < 2 or self.cfg.collision_radius <= 0.0:
            return

        if env_ids is None:
            prey_xy = self.prey_pose[:, :, :2]
            prey_alive = self.prey_alive
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
            prey_xy = self.prey_pose[env_ids, :, :2]
            prey_alive = self.prey_alive[env_ids]

        eye = torch.eye(self.n_prey, dtype=torch.bool, device=self.device).unsqueeze(0)
        alive_pair = prey_alive.unsqueeze(2) & prey_alive.unsqueeze(1)
        active_pair = alive_pair & ~eye

        base_angle = torch.linspace(
            0.0,
            2.0 * torch.pi,
            self.n_prey + 1,
            device=self.device,
        )[:-1]
        base_dir = torch.stack([torch.cos(base_angle), torch.sin(base_angle)], dim=-1)
        fallback = base_dir.unsqueeze(1) - base_dir.unsqueeze(0)
        fallback = fallback / torch.norm(fallback, dim=-1, keepdim=True).clamp_min(1e-6)
        fallback = fallback.unsqueeze(0)
        alive = prey_alive.unsqueeze(-1)

        for _ in range(max(int(self.cfg.collision_resolution_iters), 1)):
            xy = prey_xy
            diff = xy.unsqueeze(2) - xy.unsqueeze(1)
            dist = torch.norm(diff, dim=-1, keepdim=True)
            direction = torch.where(dist > 1e-6, diff / dist.clamp_min(1e-6), fallback)

            overlap = (self.cfg.collision_radius - dist.squeeze(-1)).clamp_min(0.0)
            overlap = overlap.masked_fill(~active_pair, 0.0)
            correction = (direction * (0.5 * overlap).unsqueeze(-1)).sum(dim=2)
            resolved_xy = self._clamp_xy(xy + correction)
            prey_xy = torch.where(alive, resolved_xy, xy)

        if env_ids is None:
            self.prey_pose[:, :, :2] = prey_xy
        else:
            self.prey_pose[env_ids, :, :2] = prey_xy

    def _predator_prey_dist(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if env_ids is None:
            pred_xy = self.predator_pose[:, :, :2]
            prey_xy = self.prey_pose[:, :, :2]
        else:
            pred_xy = self.predator_pose[env_ids, :, :2]
            prey_xy = self.prey_pose[env_ids, :, :2]
        return torch.cdist(pred_xy, prey_xy)

    def _update_captures(self):
        dist = self._predator_prey_dist()
        close_counts = (dist <= self.cfg.capture_radius).sum(dim=1)
        new_captures = self.prey_alive & (close_counts >= int(self.cfg.capture_k))
        self.prey_alive = self.prey_alive & ~new_captures
        return new_captures, close_counts

    def _local_obs(self) -> torch.Tensor:
        pred_xy = self.predator_pose[:, :, :2]
        pred_theta = self.predator_pose[:, :, 2]
        prey_xy = self.prey_pose[:, :, :2]
        time_frac = (self.steps.float() / max(float(self.max_steps), 1.0)).view(self.n_envs, 1, 1)
        own = torch.cat(
            [
                pred_xy / self.half_world,
                torch.cos(pred_theta).unsqueeze(-1),
                torch.sin(pred_theta).unsqueeze(-1),
                time_frac.repeat(1, self.n_predators, 1),
            ],
            dim=-1,
        )

        rel_pred = pred_xy.unsqueeze(1) - pred_xy.unsqueeze(2)
        pred_dist = torch.norm(rel_pred, dim=-1, keepdim=True)
        pred_observable = (pred_dist <= self.cfg.comm_radius).float()
        masked_rel_pred = rel_pred * pred_observable
        masked_pred_dist = pred_dist * pred_observable
        pred_feats = torch.cat(
            [
                masked_rel_pred / self.half_world,
                masked_pred_dist / self.cfg.world_size,
                pred_observable,
                pred_observable,
            ],
            dim=-1,
        ).reshape(self.n_envs, self.n_predators, -1)

        rel_prey = prey_xy.unsqueeze(1) - pred_xy.unsqueeze(2)
        prey_dist = torch.norm(rel_prey, dim=-1, keepdim=True)
        prey_visible = self._prey_visible_by_predator(prey_dist.squeeze(-1)).unsqueeze(-1).float()
        prey_alive = self.prey_alive.float().unsqueeze(1).unsqueeze(-1).repeat(1, self.n_predators, 1, 1)
        masked_rel_prey = rel_prey * prey_visible
        masked_prey_dist = prey_dist * prey_visible
        prey_feats = torch.cat(
            [
                masked_rel_prey / self.half_world,
                masked_prey_dist / self.cfg.world_size,
                prey_visible,
                prey_alive,
            ],
            dim=-1,
        ).reshape(self.n_envs, self.n_predators, -1)
        return torch.cat([own, pred_feats, prey_feats], dim=-1)

    def _share_obs(self) -> torch.Tensor:
        time_frac = (self.steps.float() / max(float(self.max_steps), 1.0)).view(self.n_envs, 1)
        pred = torch.cat(
            [
                self.predator_pose[..., :2] / self.half_world,
                torch.cos(self.predator_pose[..., 2]).unsqueeze(-1),
                torch.sin(self.predator_pose[..., 2]).unsqueeze(-1),
            ],
            dim=-1,
        ).reshape(self.n_envs, -1)
        prey = torch.cat(
            [
                self.prey_pose[..., :2] / self.half_world,
                torch.cos(self.prey_pose[..., 2]).unsqueeze(-1),
                torch.sin(self.prey_pose[..., 2]).unsqueeze(-1),
                self.prey_alive.float().unsqueeze(-1),
            ],
            dim=-1,
        ).reshape(self.n_envs, -1)
        share = torch.cat([time_frac, pred, prey], dim=-1)
        return share.unsqueeze(1).repeat(1, self.n_predators, 1)

    def obs_radius(self) -> float:
        return float(self.cfg.obs_radius)

    def _collision_count(self) -> torch.Tensor:
        dist = torch.cdist(self.predator_pose[:, :, :2], self.predator_pose[:, :, :2])
        eye = torch.eye(self.n_predators, dtype=torch.bool, device=self.device).unsqueeze(0)
        collisions = (dist < self.cfg.collision_radius) & ~eye
        return torch.div(collisions.sum(dim=(1, 2)), 2, rounding_mode="trunc")

    def _wall_contact_count(self) -> torch.Tensor:
        xy = self.predator_pose[:, :, :2]
        near_wall = (xy.abs() >= (self.half_world - 0.08)).any(dim=-1)
        return near_wall.sum(dim=1)

    def _infos(self, new_captures, close_counts, all_captured, timeout):
        infos = []
        visible = self._prey_visible_by_predator()
        comm = self.get_visibility_matrix()
        for env_i in range(self.n_envs):
            env_infos = []
            prey_visible_by_any = visible[env_i].any(dim=0)
            live_prey_visible_by_any = prey_visible_by_any | ~self.prey_alive[env_i]
            for agent_i in range(self.n_predators):
                env_infos.append(
                    {
                        "capture_success": bool(all_captured[env_i].item()),
                        "new_captures": int(new_captures[env_i].sum().item()),
                        "prey_remaining": int(self.prey_alive[env_i].sum().item()),
                        "capture_fraction": float(
                            1.0 - self.prey_alive[env_i].float().mean().item()
                        ),
                        "collision_count": int(self.last_collision_count[env_i].item()),
                        "all_live_prey_visible": bool(live_prey_visible_by_any.all().item()),
                        "live_prey_visible_count": int(
                            (prey_visible_by_any & self.prey_alive[env_i]).sum().item()
                        ),
                        "prey_seen_by_agent": bool(visible[env_i, agent_i].any().item()),
                        "comm_degree": int(comm[env_i, agent_i].sum().item()),
                        "max_capture_group": int(close_counts[env_i].max().item()),
                        "long_range_dependency_active": bool(
                            visible[env_i].any().item()
                            and not visible[env_i].all(dim=0).any().item()
                        ),
                        "timeout": bool(timeout[env_i].item()),
                    }
                )
            infos.append(env_infos)
        return infos

    def _prey_visible_by_predator(self, dist: Optional[torch.Tensor] = None) -> torch.Tensor:
        if dist is None:
            dist = self._predator_prey_dist()

        visible = (dist <= self.cfg.obs_radius) & self.prey_alive.unsqueeze(1)
        if not self.cfg.ensure_prey_visible or self.n_predators < 1 or self.n_prey < 1:
            return visible

        needs_reveal = self.prey_alive & ~visible.any(dim=1)
        if not bool(needs_reveal.any().item()):
            return visible

        nearest_pred = dist.argmin(dim=1)
        forced_visible = torch.zeros_like(visible)
        forced_visible.scatter_(1, nearest_pred.unsqueeze(1), needs_reveal.unsqueeze(1))
        return visible | forced_visible

    @staticmethod
    def _draw_disk(canvas: np.ndarray, center: np.ndarray, radius: int, color: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        cx, cy = int(center[0]), int(center[1])
        y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
        mask = x * x + y * y <= radius * radius
        x0, x1 = max(cx - radius, 0), min(cx + radius + 1, w)
        y0, y1 = max(cy - radius, 0), min(cy + radius + 1, h)
        mx0, mx1 = x0 - (cx - radius), x1 - (cx - radius)
        my0, my1 = y0 - (cy - radius), y1 - (cy - radius)
        canvas[y0:y1, x0:x1][mask[my0:my1, mx0:mx1]] = color


class LongRangePredatorPreyContinuousEnv(gym.Env):
    """Single Gym-style wrapper around the torch-vectorized core."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, **kwargs):
        cfg = LongRangePredatorPreyConfig(**kwargs)
        cfg.num_envs = 1
        self.cfg = cfg
        self.core = LongRangePredatorPreyTorchCore(cfg)
        self.n_agents = cfg.num_predators

        self.observation_space = spaces.Tuple([
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.core.obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ])
        self.share_observation_space = spaces.Tuple([
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.core.share_obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ])
        self.action_space = spaces.Tuple([
            spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for _ in range(self.n_agents)
        ])

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self.core.seed(int(seed))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        del options
        if seed is not None:
            self.seed(seed)
        obs, share_obs, available_actions = self.core.reset()
        return self._single(obs), self._single(share_obs), self._single(available_actions)

    def step(self, actions):
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.core.device).reshape(1, self.n_agents, 2)
        obs, share_obs, rewards, dones, infos, available_actions = self.core.step(actions_t)
        return (
            self._single(obs),
            self._single(share_obs),
            self._single(rewards),
            self._single(dones).astype(bool),
            infos[0],
            self._single(available_actions),
        )

    def get_visibility_matrix(self):
        return self._single(self.core.get_visibility_matrix())

    def get_edge_index_matrix(self, faulty_node: Optional[int] = None):
        return self._single(self.core.get_edge_index_matrix(faulty_node))

    def render(self, mode="rgb_array"):
        frame = self.core.render_rgb(0)
        if mode == "rgb_array":
            return frame
        if mode == "human":
            return frame
        raise NotImplementedError(f"Unsupported render mode: {mode}")

    def close(self):
        pass

    @staticmethod
    def _single(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()[0].astype(np.float32, copy=False)
