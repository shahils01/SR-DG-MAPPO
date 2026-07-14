"""Fixed-rate and uncompressed codecs for local scene maps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class CodecOutput:
    decoded_points: Tensor
    decoded_confidence: Tensor
    confidence_logits: Tensor
    indices: Optional[Tensor]
    vq_loss: Tensor
    perplexity: Tensor


class ProductVectorQuantizer(nn.Module):
    """A product VQ bottleneck with a fixed number of codebook tokens."""

    def __init__(
        self,
        latent_dim: int,
        num_codebooks: int = 4,
        codebook_size: int = 16,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        if latent_dim % num_codebooks != 0:
            raise ValueError("latent_dim must be divisible by num_codebooks")
        if codebook_size < 2:
            raise ValueError("codebook_size must be at least 2")
        self.latent_dim = latent_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.chunk_dim = latent_dim // num_codebooks
        self.commitment_cost = commitment_cost
        scale = 1.0 / max(codebook_size, 1)
        self.codebook = nn.Parameter(
            torch.empty(num_codebooks, codebook_size, self.chunk_dim).uniform_(-scale, scale)
        )

    @property
    def bits_per_message(self) -> int:
        return self.num_codebooks * math.ceil(math.log2(self.codebook_size))

    def forward(self, latent: Tensor):
        prefix = latent.shape[:-1]
        chunks = latent.reshape(-1, self.num_codebooks, self.chunk_dim)
        distances = (
            chunks.square().sum(dim=-1, keepdim=True)
            - 2.0 * torch.einsum("bqd,qkd->bqk", chunks, self.codebook)
            + self.codebook.square().sum(dim=-1).unsqueeze(0)
        )
        indices = distances.argmin(dim=-1)
        codebook_index = torch.arange(self.num_codebooks, device=latent.device).unsqueeze(0)
        quantized = self.codebook[codebook_index, indices]

        codebook_loss = F.mse_loss(quantized, chunks.detach())
        commitment_loss = F.mse_loss(chunks, quantized.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        straight_through = chunks + (quantized - chunks).detach()

        one_hot = F.one_hot(indices, num_classes=self.codebook_size).to(latent.dtype)
        probabilities = one_hot.mean(dim=0)
        per_codebook = torch.exp(
            -(probabilities * torch.log(probabilities.clamp_min(1e-10))).sum(dim=-1)
        )
        perplexity = per_codebook.mean()
        return (
            straight_through.reshape(*prefix, self.latent_dim),
            indices.reshape(*prefix, self.num_codebooks),
            vq_loss,
            perplexity,
        )


class MapCodec(nn.Module):
    """Compress a fixed-size sender-frame target map into discrete tokens."""

    def __init__(
        self,
        num_targets: int,
        hidden_dim: int = 96,
        latent_dim: int = 32,
        num_codebooks: int = 4,
        codebook_size: int = 16,
        coordinate_scale: float = 3.0,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_targets = num_targets
        self.coordinate_scale = float(coordinate_scale)
        input_dim = num_targets * 3
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.quantizer = ProductVectorQuantizer(
            latent_dim=latent_dim,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            commitment_cost=commitment_cost,
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    @property
    def bits_per_message(self) -> int:
        return self.quantizer.bits_per_message

    def forward(self, points: Tensor, confidence: Tensor) -> CodecOutput:
        if points.shape[-2:] != (self.num_targets, 2):
            raise ValueError("points shape does not match configured target count")
        if confidence.shape != points.shape[:-1]:
            raise ValueError("confidence must match points without the coordinate axis")

        inputs = torch.cat((points / self.coordinate_scale, confidence.unsqueeze(-1)), dim=-1)
        latent = self.encoder(inputs.flatten(start_dim=-2))
        quantized, indices, vq_loss, perplexity = self.quantizer(latent)
        decoded = self.decoder(quantized).reshape(*points.shape[:-2], self.num_targets, 3)
        decoded_points = torch.tanh(decoded[..., :2]) * self.coordinate_scale
        confidence_logits = decoded[..., 2]
        return CodecOutput(
            decoded_points=decoded_points,
            decoded_confidence=torch.sigmoid(confidence_logits),
            confidence_logits=confidence_logits,
            indices=indices,
            vq_loss=vq_loss,
            perplexity=perplexity,
        )


class UncompressedMapCodec(nn.Module):
    """Float32 baseline with the same map interface and no learned distortion."""

    def __init__(self, num_targets: int, float_bits: int = 32) -> None:
        super().__init__()
        self.num_targets = num_targets
        self.float_bits = float_bits

    @property
    def bits_per_message(self) -> int:
        return self.num_targets * 3 * self.float_bits

    def forward(self, points: Tensor, confidence: Tensor) -> CodecOutput:
        zero = points.new_zeros(())
        logits = torch.logit(confidence.clamp(1e-6, 1.0 - 1e-6))
        return CodecOutput(
            decoded_points=points,
            decoded_confidence=confidence,
            confidence_logits=logits,
            indices=None,
            vq_loss=zero,
            perplexity=zero,
        )


def local_reconstruction_loss(
    output: CodecOutput,
    target_points: Tensor,
    target_confidence: Tensor,
    confidence_weight: float = 1.0,
) -> Tensor:
    """Masked coordinate reconstruction plus visibility classification."""

    squared_error = (output.decoded_points - target_points).square().sum(dim=-1)
    coordinate_loss = (squared_error * target_confidence).sum()
    coordinate_loss = coordinate_loss / target_confidence.sum().clamp_min(1.0)
    visibility_loss = F.binary_cross_entropy_with_logits(
        output.confidence_logits, target_confidence
    )
    return coordinate_loss + confidence_weight * visibility_loss
