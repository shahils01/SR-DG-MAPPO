"""DG-MAPPO actor-critic with differentiable symmetry-reduced communication."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from mat.algorithms.utils.transformer_act import (
    continuous_decentralized_act,
    continuous_parallel_act,
    discrete_decentralized_act,
    discrete_parallel_act,
)
from mat.algorithms.utils.util import check, init
from sr_dg_mappo.codec import MapCodec, local_reconstruction_loss
from sr_dg_mappo.communication import (
    InvariantMapAttention,
    SymmetryReducedCommunicator,
    SymmetryReducedDGATCommunicator,
)


def _init_layer(layer: nn.Module, gain: float = 0.01, activate: bool = False) -> nn.Module:
    if activate:
        gain = nn.init.calculate_gain("relu")
    return init(layer, nn.init.orthogonal_, lambda bias: nn.init.constant_(bias, 0), gain)


class DistributedCritic(nn.Module):
    """Per-agent critic heads retained from the DG-MAPPO architecture."""

    def __init__(self, args, state_dim, obs_dim, hidden_dim, num_agents, num_quants) -> None:
        super().__init__()
        self.use_centralized_critic = bool(args.use_centralized_critic)
        input_dim = state_dim if self.use_centralized_critic else obs_dim + hidden_dim
        self.head_ = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(input_dim),
                    _init_layer(nn.Linear(input_dim, hidden_dim), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    _init_layer(nn.Linear(hidden_dim, hidden_dim), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    _init_layer(nn.Linear(hidden_dim, num_quants)),
                )
                for _ in range(num_agents)
            ]
        )

    def forward(self, state: Tensor, obs: Tensor):
        values = []
        representations = []
        for agent_id, head in enumerate(self.head_):
            features = state[:, agent_id] if self.use_centralized_critic else obs[:, agent_id]
            values.append(head(features))
            representations.append(features.unsqueeze(1))
        value_tensor = torch.stack(values, dim=1)
        value_tensor, _ = torch.sort(value_tensor, dim=-1)
        return value_tensor, torch.stack(representations, dim=1)


class DistributedActor(nn.Module):
    """Per-agent actor heads retained from the DG-MAPPO architecture."""

    def __init__(self, obs_dim, action_dim, hidden_dim, num_agents, action_type) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_type = action_type
        input_dim = obs_dim + hidden_dim
        self.mlp_ = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(input_dim),
                    _init_layer(nn.Linear(input_dim, hidden_dim), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    _init_layer(nn.Linear(hidden_dim, hidden_dim), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    _init_layer(nn.Linear(hidden_dim, action_dim)),
                )
                for _ in range(num_agents)
            ]
        )
        if action_type != "Discrete":
            self.log_std = nn.Parameter(torch.ones(action_dim))

    def forward(self, action, obs_rep, obs):
        del action, obs_rep
        return torch.stack(
            [head(obs[:, agent_id]) for agent_id, head in enumerate(self.mlp_)],
            dim=1,
        )

    def zero_std(self, device) -> None:
        if self.action_type != "Discrete":
            self.log_std.data = torch.zeros(self.action_dim, device=device)


class SymmetryReducedObservationEncoder(nn.Module):
    """Parse predator-prey observations, communicate maps, and emit policy features."""

    def __init__(self, args, obs_dim: int, output_dim: int) -> None:
        super().__init__()
        self.num_agents = int(args.num_predators)
        self.num_targets = int(args.num_prey)
        self.half_world = float(args.world_size) / 2.0
        self.expected_obs_dim = 5 + 5 * self.num_agents + 5 * self.num_targets
        if obs_dim != self.expected_obs_dim:
            raise ValueError(
                f"sr_mappo expects predator-prey obs_dim={self.expected_obs_dim}, got {obs_dim}"
            )

        self.reconstruction_coef = float(args.sr_reconstruction_coef)
        self.vq_coef = float(args.sr_vq_coef)
        self.codec = MapCodec(
            num_targets=self.num_targets,
            hidden_dim=int(args.sr_hidden_dim),
            latent_dim=int(args.sr_latent_dim),
            num_codebooks=int(args.sr_num_codebooks),
            codebook_size=int(args.sr_codebook_size),
            coordinate_scale=max(float(args.world_size), 1.0),
        )
        self.communicator = SymmetryReducedCommunicator(
            self.codec,
            rounds=int(args.sr_comm_rounds),
        )
        map_dim = self.num_targets * 3
        self.readout = nn.Sequential(
            nn.LayerNorm(map_dim),
            nn.Linear(map_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self._auxiliary_loss: Optional[Tensor] = None
        self._metrics: Dict[str, Tensor] = {}

    @property
    def bits_per_message(self) -> int:
        return int(self.codec.bits_per_message)

    def _parse(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_agents, _ = obs.shape
        if num_agents != self.num_agents:
            raise ValueError(f"expected {self.num_agents} agents, got {num_agents}")

        positions = obs[..., :2] * self.half_world
        cosine = obs[..., 2]
        sine = obs[..., 3]
        frames = torch.stack(
            (
                torch.stack((cosine, -sine), dim=-1),
                torch.stack((sine, cosine), dim=-1),
            ),
            dim=-2,
        )

        predator_start = 5
        predator_end = predator_start + 5 * self.num_agents
        predator_features = obs[..., predator_start:predator_end].reshape(
            batch_size, self.num_agents, self.num_agents, 5
        )
        adjacency = predator_features[..., 3].clamp(0.0, 1.0)

        prey_features = obs[..., predator_end:].reshape(
            batch_size, self.num_agents, self.num_targets, 5
        )
        relative_world = prey_features[..., :2] * self.half_world
        local_points = torch.einsum("bnij,bnmi->bnmj", frames, relative_world)
        confidence = (prey_features[..., 3] * prey_features[..., 4]).clamp(0.0, 1.0)
        local_points = local_points * confidence.unsqueeze(-1)
        return positions, frames, local_points, confidence, adjacency

    def forward(self, obs: Tensor, graph_context: Optional[Tensor] = None) -> Tensor:
        positions, frames, local_points, confidence, observed_adjacency = self._parse(obs)
        adjacency = observed_adjacency if graph_context is None else graph_context
        adjacency = adjacency.to(device=obs.device, dtype=obs.dtype)
        if adjacency.shape != observed_adjacency.shape:
            adjacency = adjacency.reshape_as(observed_adjacency)

        communicated = self.communicator(
            positions,
            frames,
            local_points,
            confidence,
            adjacency,
        )
        local_codec = self.codec(local_points, confidence)
        reconstruction = local_reconstruction_loss(local_codec, local_points, confidence)
        vq_loss = 0.5 * (local_codec.vq_loss + communicated.codec_loss)
        self._auxiliary_loss = (
            self.reconstruction_coef * reconstruction + self.vq_coef * vq_loss
        )
        self._metrics = {
            "sr_reconstruction_loss": reconstruction.detach(),
            "sr_vq_loss": vq_loss.detach(),
            "sr_bits_per_message": obs.new_tensor(float(self.bits_per_message)),
            "sr_bits_per_agent": communicated.bits_per_agent.detach(),
            "sr_codebook_perplexity": communicated.perplexity.detach(),
            "sr_map_coverage": communicated.map_confidence.detach().mean(),
        }

        normalized_points = communicated.map_points / max(self.half_world, 1e-6)
        map_features = torch.cat(
            (normalized_points, communicated.map_confidence.unsqueeze(-1)), dim=-1
        )
        return self.readout(map_features.flatten(start_dim=-2))

    def auxiliary_loss(self) -> Tensor:
        if self._auxiliary_loss is None:
            return next(self.parameters()).new_zeros(())
        return self._auxiliary_loss

    def metrics(self) -> Dict[str, Tensor]:
        return self._metrics


class DistributedSymmetryReducedObservationEncoder(SymmetryReducedObservationEncoder):
    """Per-agent SR codecs with invariant D-GAT aggregation and consensus hooks."""

    def __init__(self, args, obs_dim: int, output_dim: int) -> None:
        super().__init__(args, obs_dim, output_dim)
        del self.codec
        del self.communicator
        del self.readout

        codec_kwargs = dict(
            num_targets=self.num_targets,
            hidden_dim=int(args.sr_hidden_dim),
            latent_dim=int(args.sr_latent_dim),
            num_codebooks=int(args.sr_num_codebooks),
            codebook_size=int(args.sr_codebook_size),
            coordinate_scale=max(float(args.world_size), 1.0),
        )
        self.agent_codecs = nn.ModuleList(
            [MapCodec(**codec_kwargs) for _ in range(self.num_agents)]
        )
        attention_hidden_dim = max(int(args.sr_hidden_dim) // 2, 8)
        self.agent_fusers = nn.ModuleList(
            [
                InvariantMapAttention(
                    hidden_dim=attention_hidden_dim,
                    coordinate_scale=max(float(args.world_size), 1.0),
                )
                for _ in range(self.num_agents)
            ]
        )
        map_dim = self.num_targets * 3
        self.agent_readouts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(map_dim),
                    nn.Linear(map_dim, output_dim),
                    nn.GELU(),
                    nn.Linear(output_dim, output_dim),
                )
                for _ in range(self.num_agents)
            ]
        )
        self.communicator = SymmetryReducedDGATCommunicator(
            self.agent_codecs,
            self.agent_fusers,
            rounds=int(args.sr_comm_rounds),
        )

    @property
    def bits_per_message(self) -> int:
        return int(self.agent_codecs[0].bits_per_message)

    def forward(self, obs: Tensor, graph_context: Optional[Tensor] = None) -> Tensor:
        positions, frames, local_points, confidence, observed_adjacency = self._parse(obs)
        adjacency = observed_adjacency if graph_context is None else graph_context
        adjacency = adjacency.to(device=obs.device, dtype=obs.dtype)
        if adjacency.shape != observed_adjacency.shape:
            adjacency = adjacency.reshape_as(observed_adjacency)

        communicated = self.communicator(
            positions,
            frames,
            local_points,
            confidence,
            adjacency,
        )
        local_outputs = [
            codec(local_points[:, agent_id], confidence[:, agent_id])
            for agent_id, codec in enumerate(self.agent_codecs)
        ]
        reconstruction = torch.stack(
            [
                local_reconstruction_loss(
                    output,
                    local_points[:, agent_id],
                    confidence[:, agent_id],
                )
                for agent_id, output in enumerate(local_outputs)
            ]
        )
        local_vq = torch.stack([output.vq_loss for output in local_outputs])
        vq_loss = 0.5 * (local_vq + communicated.agent_codec_loss)
        self._auxiliary_loss = (
            self.reconstruction_coef * reconstruction + self.vq_coef * vq_loss
        ).unsqueeze(-1)
        self._metrics = {
            "sr_reconstruction_loss": reconstruction.detach().mean(),
            "sr_vq_loss": vq_loss.detach().mean(),
            "sr_bits_per_message": obs.new_tensor(float(self.bits_per_message)),
            "sr_bits_per_agent": communicated.bits_per_agent.detach(),
            "sr_codebook_perplexity": communicated.agent_perplexity.detach().mean(),
            "sr_map_coverage": communicated.map_confidence.detach().mean(),
            "sr_attention_entropy": communicated.attention_entropy.detach().mean(),
        }

        normalized_points = communicated.map_points / max(self.half_world, 1e-6)
        map_features = torch.cat(
            (normalized_points, communicated.map_confidence.unsqueeze(-1)), dim=-1
        ).flatten(start_dim=-2)
        return torch.stack(
            [
                readout(map_features[:, agent_id])
                for agent_id, readout in enumerate(self.agent_readouts)
            ],
            dim=1,
        )

    def consensus_module_lists(self):
        """Return identically structured per-agent modules mixed by D-SGD."""

        return (self.agent_codecs, self.agent_fusers, self.agent_readouts)


class SymmetryReducedMAPPO(nn.Module):
    """Original DG-MAPPO per-agent actor/critic heads with an SR message front end."""

    def __init__(
        self,
        args,
        state_dim,
        obs_dim,
        action_dim,
        n_agent,
        n_block,
        n_embd,
        n_head,
        encode_state=False,
        device=torch.device("cpu"),
        action_type="Discrete",
        dec_actor=False,
        share_actor=False,
        num_quants=1,
    ) -> None:
        super().__init__()
        self.n_agent = n_agent
        self.action_dim = action_dim
        self.action_type = action_type
        self.device = torch.device(device)
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        if args.algorithm_name == "sr_mappo":
            self.obs_encoder = DistributedSymmetryReducedObservationEncoder(
                args, obs_dim, n_embd
            )
        else:
            self.obs_encoder = SymmetryReducedObservationEncoder(args, obs_dim, n_embd)
        del n_block, n_head, encode_state, dec_actor, share_actor
        self.encoder = DistributedCritic(
            args, state_dim, obs_dim, n_embd, n_agent, num_quants
        )
        self.decoder = DistributedActor(
            obs_dim, action_dim, n_embd, n_agent, action_type
        )
        self.to(self.device)

    def _augment(self, obs: Tensor, graph_context: Optional[Tensor]) -> Tensor:
        obs = check(obs).to(**self.tpdv)
        message_features = self.obs_encoder(obs, graph_context)
        return torch.cat((obs, message_features), dim=-1)

    def forward(
        self,
        state,
        obs,
        action,
        available_actions=None,
        obs_rep=None,
        graph_context=None,
    ):
        state = check(state).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        augmented = self._augment(obs, graph_context)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        batch_size = augmented.shape[0]
        if self.action_type == "Discrete":
            action_log, entropy = discrete_parallel_act(
                self.decoder,
                obs_rep,
                augmented,
                action.long(),
                batch_size,
                self.n_agent,
                self.action_dim,
                self.tpdv,
                available_actions,
            )
        else:
            action_log, entropy = continuous_parallel_act(
                self.decoder,
                obs_rep,
                augmented,
                action,
                batch_size,
                self.n_agent,
                self.action_dim,
                self.tpdv,
            )
        values, _ = self.encoder(state, augmented)
        return action_log, values, entropy

    def get_actions(
        self,
        state,
        obs,
        available_actions=None,
        deterministic=False,
        graph_context=None,
        obs_rep=None,
    ):
        state = check(state).to(**self.tpdv)
        augmented = self._augment(obs, graph_context)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        batch_size = augmented.shape[0]
        if self.action_type == "Discrete":
            actions, action_log = discrete_decentralized_act(
                self.decoder,
                obs_rep,
                augmented,
                batch_size,
                self.n_agent,
                self.action_dim,
                self.tpdv,
                available_actions,
                deterministic,
            )
        else:
            actions, action_log = continuous_decentralized_act(
                self.decoder,
                obs_rep,
                augmented,
                batch_size,
                self.n_agent,
                self.action_dim,
                self.tpdv,
                deterministic,
            )
        values, _ = self.encoder(state, augmented)
        return actions, action_log, values

    def get_values(self, state, obs, available_actions=None, graph_context=None):
        del available_actions
        state = check(state).to(**self.tpdv)
        augmented = self._augment(obs, graph_context)
        values, _ = self.encoder(state, augmented)
        return values

    def auxiliary_loss(self) -> Tensor:
        return self.obs_encoder.auxiliary_loss()

    def auxiliary_metrics(self) -> Dict[str, Tensor]:
        return self.obs_encoder.metrics()

    def zero_std(self) -> None:
        if self.action_type != "Discrete":
            self.decoder.zero_std(self.device)
