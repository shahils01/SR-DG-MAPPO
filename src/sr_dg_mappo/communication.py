"""Multi-hop symmetry-reduced map communication and fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
