"""Multi-hop symmetry-reduced map communication and fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from sr_dg_mappo.codec import CodecOutput
from sr_dg_mappo.groups import all_receiver_transports


@dataclass
class CommunicationOutput:
    map_points: Tensor
    map_confidence: Tensor
    codec_loss: Tensor
    perplexity: Tensor
    total_bits: Tensor
    bits_per_agent: Tensor
    last_indices: Optional[Tensor]


class SymmetryReducedCommunicator(nn.Module):
    """Exchange fixed-size local-frame maps for a configurable number of rounds."""

    def __init__(self, codec: nn.Module, rounds: int = 1) -> None:
        super().__init__()
        if rounds < 1:
            raise ValueError("rounds must be positive")
        if not hasattr(codec, "bits_per_message"):
            raise TypeError("codec must expose a bits_per_message property")
        self.codec = codec
        self.rounds = rounds

    def forward(
        self,
        positions: Tensor,
        frames: Tensor,
        initial_points: Tensor,
        initial_confidence: Tensor,
        adjacency: Tensor,
    ) -> CommunicationOutput:
        if adjacency.shape != positions.shape[:2] + (positions.shape[1],):
            raise ValueError("adjacency must have shape [batch, agents, agents]")
        num_agents = positions.shape[1]
        identity = torch.eye(num_agents, dtype=torch.bool, device=positions.device)
        communication_adjacency = adjacency.masked_fill(identity.unsqueeze(0), 0.0)

        points = initial_points
        confidence = initial_confidence
        losses = []
        perplexities = []
        last_output: Optional[CodecOutput] = None

        for _ in range(self.rounds):
            last_output = self.codec(points, confidence)
            transported = all_receiver_transports(
                last_output.decoded_points, positions, frames
            )
            weights = (
                communication_adjacency.unsqueeze(-1)
                * last_output.decoded_confidence.unsqueeze(1)
            )

            neighbor_numerator = (transported * weights.unsqueeze(-1)).sum(dim=2)
            neighbor_denominator = weights.sum(dim=2)
            numerator = points * confidence.unsqueeze(-1) + neighbor_numerator
            denominator = confidence + neighbor_denominator
            points = numerator / denominator.clamp_min(1e-8).unsqueeze(-1)
            points = torch.where(
                denominator.unsqueeze(-1) > 0.0, points, torch.zeros_like(points)
            )

            no_neighbor_report = (1.0 - weights.clamp(0.0, 1.0)).prod(dim=2)
            confidence = 1.0 - (1.0 - confidence.clamp(0.0, 1.0)) * no_neighbor_report
            losses.append(last_output.vq_loss)
            perplexities.append(last_output.perplexity)

        assert last_output is not None
        directed_edges = communication_adjacency.sum()
        total_bits = directed_edges * float(self.codec.bits_per_message * self.rounds)
        batch_agents = positions.shape[0] * positions.shape[1]
        return CommunicationOutput(
            map_points=points,
            map_confidence=confidence,
            codec_loss=torch.stack(losses).mean(),
            perplexity=torch.stack(perplexities).mean(),
            total_bits=total_bits,
            bits_per_agent=total_bits / float(batch_agents),
            last_indices=last_output.indices,
        )


@dataclass
class DistributedCommunicationOutput:
    """Outputs from per-agent codecs and invariant graph attention."""

    map_points: Tensor
    map_confidence: Tensor
    agent_codec_loss: Tensor
    agent_perplexity: Tensor
    attention_entropy: Tensor
    total_bits: Tensor
    bits_per_agent: Tensor
    last_indices: Optional[Tensor]


class InvariantMapAttention(nn.Module):
    """Score transported map entries using only orthogonal-group invariants.

    Coordinate-bearing values are aligned in the receiver frame before this
    module is called.  Attention logits depend on confidence, squared map
    disagreement, and squared sender distance, so the scalar weights are
    unchanged by a global rotation or reflection.
    """

    def __init__(self, hidden_dim: int, coordinate_scale: float) -> None:
        super().__init__()
        self.coordinate_scale_sq = max(float(coordinate_scale) ** 2, 1e-8)
        self.score = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        candidate_points: Tensor,
        candidate_confidence: Tensor,
        receiver_points: Tensor,
        receiver_confidence: Tensor,
        sender_distance_sq: Tensor,
        connectivity: Tensor,
    ):
        disagreement_sq = (
            candidate_points - receiver_points.unsqueeze(1)
        ).square().sum(dim=-1)
        features = torch.stack(
            (
                candidate_confidence,
                receiver_confidence.unsqueeze(1).expand_as(candidate_confidence),
                disagreement_sq / self.coordinate_scale_sq,
                sender_distance_sq.unsqueeze(-1).expand_as(candidate_confidence)
                / self.coordinate_scale_sq,
            ),
            dim=-1,
        )
        logits = self.score(features).squeeze(-1)
        logits = logits + torch.log(candidate_confidence.clamp_min(1e-8))

        valid = (connectivity.unsqueeze(-1) > 0.0) & (candidate_confidence > 0.0)
        has_candidate = valid.any(dim=1, keepdim=True)
        masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(masked_logits, dim=1)
        weights = torch.where(valid & has_candidate, weights, torch.zeros_like(weights))

        fused_points = (weights.unsqueeze(-1) * candidate_points).sum(dim=1)
        report_probability = (
            candidate_confidence * connectivity.unsqueeze(-1)
        ).clamp(0.0, 1.0)
        fused_confidence = 1.0 - (1.0 - report_probability).prod(dim=1)
        fused_points = torch.where(
            fused_confidence.unsqueeze(-1) > 0.0,
            fused_points,
            torch.zeros_like(fused_points),
        )
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=1)
        entropy = entropy.mean()
        return fused_points, fused_confidence, entropy


class SymmetryReducedDGATCommunicator(nn.Module):
    """Per-agent VQ messages followed by equivariant receiver-side attention."""

    def __init__(
        self,
        codecs: Sequence[nn.Module],
        fusers: Sequence[InvariantMapAttention],
        rounds: int = 1,
    ) -> None:
        super().__init__()
        if rounds < 1:
            raise ValueError("rounds must be positive")
        if len(codecs) == 0 or len(codecs) != len(fusers):
            raise ValueError("one codec and attention fuser are required per agent")
        bit_rates = {int(codec.bits_per_message) for codec in codecs}
        if len(bit_rates) != 1:
            raise ValueError("all per-agent codecs must use the same message rate")
        # The ModuleLists are owned by the observation encoder.  Keeping plain
        # references here avoids registering every module twice in state_dict().
        self.__dict__["_codecs"] = codecs
        self.__dict__["_fusers"] = fusers
        self.rounds = int(rounds)
        self.bits_per_message = bit_rates.pop()

    def _encode_agents(self, points: Tensor, confidence: Tensor):
        outputs = [
            codec(points[:, agent_id], confidence[:, agent_id])
            for agent_id, codec in enumerate(self._codecs)
        ]
        decoded_points = torch.stack([output.decoded_points for output in outputs], dim=1)
        decoded_confidence = torch.stack(
            [output.decoded_confidence for output in outputs], dim=1
        )
        losses = torch.stack([output.vq_loss for output in outputs])
        perplexities = torch.stack([output.perplexity for output in outputs])
        indices = None
        if all(output.indices is not None for output in outputs):
            indices = torch.stack([output.indices for output in outputs], dim=1)
        return decoded_points, decoded_confidence, losses, perplexities, indices

    def forward(
        self,
        positions: Tensor,
        frames: Tensor,
        initial_points: Tensor,
        initial_confidence: Tensor,
        adjacency: Tensor,
    ) -> DistributedCommunicationOutput:
        num_agents = positions.shape[1]
        if num_agents != len(self._codecs):
            raise ValueError("agent count does not match the distributed communicator")
        if adjacency.shape != positions.shape[:2] + (num_agents,):
            raise ValueError("adjacency must have shape [batch, agents, agents]")

        identity = torch.eye(num_agents, dtype=torch.bool, device=positions.device)
        communication_adjacency = adjacency.masked_fill(identity.unsqueeze(0), 0.0)
        connectivity = (adjacency + identity.to(adjacency.dtype).unsqueeze(0)).clamp(0.0, 1.0)
        sender_distance_sq = (
            positions.unsqueeze(2) - positions.unsqueeze(1)
        ).square().sum(dim=-1)

        points = initial_points
        confidence = initial_confidence
        losses = []
        perplexities = []
        entropies = []
        last_indices = None

        for _ in range(self.rounds):
            decoded_points, decoded_confidence, codec_loss, perplexity, last_indices = (
                self._encode_agents(points, confidence)
            )
            transported = all_receiver_transports(decoded_points, positions, frames)

            self_mask = identity.view(1, num_agents, num_agents, 1, 1)
            candidate_points = torch.where(
                self_mask,
                points.unsqueeze(2),
                transported,
            )
            candidate_confidence = torch.where(
                identity.view(1, num_agents, num_agents, 1),
                confidence.unsqueeze(2),
                decoded_confidence.unsqueeze(1),
            )

            next_points = []
            next_confidence = []
            round_entropy = []
            for receiver_id, fuser in enumerate(self._fusers):
                fused_points, fused_confidence, entropy = fuser(
                    candidate_points[:, receiver_id],
                    candidate_confidence[:, receiver_id],
                    points[:, receiver_id],
                    confidence[:, receiver_id],
                    sender_distance_sq[:, receiver_id],
                    connectivity[:, receiver_id],
                )
                next_points.append(fused_points)
                next_confidence.append(fused_confidence)
                round_entropy.append(entropy)

            points = torch.stack(next_points, dim=1)
            confidence = torch.stack(next_confidence, dim=1)
            losses.append(codec_loss)
            perplexities.append(perplexity)
            entropies.append(torch.stack(round_entropy))

        directed_edges = communication_adjacency.sum()
        total_bits = directed_edges * float(self.bits_per_message * self.rounds)
        batch_agents = positions.shape[0] * num_agents
        return DistributedCommunicationOutput(
            map_points=points,
            map_confidence=confidence,
            agent_codec_loss=torch.stack(losses).mean(dim=0),
            agent_perplexity=torch.stack(perplexities).mean(dim=0),
            attention_entropy=torch.stack(entropies).mean(dim=0),
            total_bits=total_bits,
            bits_per_agent=total_bits / float(batch_agents),
            last_indices=last_indices,
        )
