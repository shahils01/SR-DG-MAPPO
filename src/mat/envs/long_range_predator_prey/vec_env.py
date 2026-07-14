"""Batched vector environment for long-range predator-prey.

This wrapper keeps all predator-prey environments inside one torch core.  It is
especially useful for CUDA because it avoids creating CUDA contexts in forked
rollout worker processes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from gymnasium import spaces

from mat.envs.env_wrappers import ShareVecEnv
from mat.envs.long_range_predator_prey.continuous import (
    LongRangePredatorPreyConfig,
    LongRangePredatorPreyTorchCore,
)


class LongRangePredatorPreyTorchVecEnv(ShareVecEnv):
    """Vector env backed by one batched LongRangePredatorPreyTorchCore."""

    def __init__(self, **kwargs):
        cfg = LongRangePredatorPreyConfig(**kwargs)
        self.cfg = cfg
        self.core = LongRangePredatorPreyTorchCore(cfg)
        self.n_agents = cfg.num_predators
        self.num_envs = cfg.num_envs
        self.actions = None
        self.closed = False

        observation_space = spaces.Tuple(
            [
                spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.core.obs_dim,),
                    dtype=np.float32,
                )
                for _ in range(self.n_agents)
            ]
        )
        share_observation_space = spaces.Tuple(
            [
                spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.core.share_obs_dim,),
                    dtype=np.float32,
                )
                for _ in range(self.n_agents)
            ]
        )
        action_space = spaces.Tuple(
            [
                spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
                for _ in range(self.n_agents)
            ]
        )
        super().__init__(cfg.num_envs, observation_space, share_observation_space, action_space)

    def reset(self):
        obs, share_obs, available_actions = self.core.reset()
        return self._to_numpy(obs), self._to_numpy(share_obs), self._to_numpy(available_actions)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        if self.actions is None:
            raise RuntimeError("step_wait() called before step_async().")
        actions = self.actions
        self.actions = None
        return self.step(actions)

    def step(self, actions):
        actions_t = torch.as_tensor(
            actions,
            dtype=torch.float32,
            device=self.core.device,
        ).reshape(self.num_envs, self.n_agents, 2)

        obs, share_obs, rewards, dones, infos, available_actions = self.core.step(actions_t)

        done_envs = dones.all(dim=1)
        if bool(done_envs.any().item()):
            env_ids = torch.nonzero(done_envs, as_tuple=False).flatten()
            reset_obs, reset_share_obs, reset_available_actions = self.core.reset(env_ids)
            obs = obs.clone()
            share_obs = share_obs.clone()
            available_actions = available_actions.clone()
            obs[env_ids] = reset_obs[env_ids]
            share_obs[env_ids] = reset_share_obs[env_ids]
            available_actions[env_ids] = reset_available_actions[env_ids]

        return (
            self._to_numpy(obs),
            self._to_numpy(share_obs),
            self._to_numpy(rewards),
            self._to_numpy(dones).astype(bool, copy=False),
            tuple(infos),
            self._to_numpy(available_actions),
        )

    def get_edge_index_matrix(self, faulty_node: Optional[int] = None):
        return self._to_numpy(self.core.get_edge_index_matrix(faulty_node))

    def get_visibility_matrix(self):
        return self._to_numpy(self.core.get_visibility_matrix())

    def render(self, mode="human"):
        if mode == "rgb_array":
            frames = [self.core.render_rgb(env_id) for env_id in range(self.num_envs)]
            return np.asarray(frames, dtype=np.uint8)
        if mode == "human":
            return self.core.render_rgb(0)
        raise NotImplementedError(f"Unsupported render mode: {mode}")

    def close(self):
        self.closed = True

    @staticmethod
    def _to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()
