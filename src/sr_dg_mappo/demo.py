"""Train and evaluate the initial symmetry-reduced communication prototype."""

from __future__ import annotations

import argparse
import json
from typing import Dict

import torch

from sr_dg_mappo.codec import MapCodec, UncompressedMapCodec, local_reconstruction_loss
from sr_dg_mappo.communication import SymmetryReducedCommunicator
from sr_dg_mappo.data import sample_scenes, targets_in_agent_frames
from sr_dg_mappo.groups import d4_matrices, local_to_world
from sr_dg_mappo.metrics import d4_orbit_mse, equivariance_error, map_metrics


def train_codec(
    codec: MapCodec,
    steps: int,
    batch_size: int,
    num_agents: int,
    num_targets: int,
    device: torch.device,
    learning_rate: float = 3e-3,
    seed: int = 7,
) -> float:
    """Train only from sender-local observations; no global state is a target."""

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    optimizer = torch.optim.Adam(codec.parameters(), lr=learning_rate)
    codec.train()
    final_loss = 0.0
    for _ in range(steps):
        scene = sample_scenes(
            batch_size=batch_size,
            num_agents=num_agents,
            num_targets=num_targets,
            device=device,
            generator=generator,
        )
        output = codec(scene.local_points, scene.visibility)
        reconstruction = local_reconstruction_loss(
            output, scene.local_points, scene.visibility
        )
        loss = reconstruction + output.vq_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    return final_loss


@torch.no_grad()
def evaluate_codec(
    codec,
    rounds: int,
    batch_size: int,
    num_agents: int,
    num_targets: int,
    device: torch.device,
    seed: int = 101,
) -> Dict[str, float]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    scene = sample_scenes(
        batch_size=batch_size,
        num_agents=num_agents,
        num_targets=num_targets,
        device=device,
        generator=generator,
    )
    communicator = SymmetryReducedCommunicator(codec, rounds=rounds).to(device)
    communicator.eval()
    result = communicator(
        scene.positions,
        scene.frames,
        scene.local_points,
        scene.visibility,
        scene.adjacency,
    )
    target_local = targets_in_agent_frames(scene.positions, scene.frames, scene.targets)
    quality = map_metrics(result.map_points, result.map_confidence, target_local)

    matrix = d4_matrices(dtype=scene.positions.dtype, device=device)[5]
    translation = torch.tensor([0.37, -0.22], dtype=scene.positions.dtype, device=device)
    transformed_scene = scene.transformed(matrix, translation)
    transformed_result = communicator(
        transformed_scene.positions,
        transformed_scene.frames,
        transformed_scene.local_points,
        transformed_scene.visibility,
        transformed_scene.adjacency,
    )

    receiver_zero_world = local_to_world(
        result.map_points[:, 0], scene.positions[:, 0], scene.frames[:, 0]
    )
    quotient_mse = d4_orbit_mse(
        receiver_zero_world,
        scene.targets,
        weight=result.map_confidence[:, 0],
    )
    local_codec_output = codec(scene.local_points, scene.visibility)
    reconstruction = local_reconstruction_loss(
        local_codec_output, scene.local_points, scene.visibility
    )
    return {
        "bits_per_message": float(codec.bits_per_message),
        "bits_per_agent": float(result.bits_per_agent.cpu()),
        "total_bits": float(result.total_bits.cpu()),
        "coordinate_rmse": float(quality["coordinate_rmse"].cpu()),
        "coverage": float(quality["coverage"].cpu()),
        "soft_distortion": float(quality["soft_distortion"].cpu()),
        "d4_quotient_mse": float(quotient_mse.cpu()),
        "equivariance_max_error": float(
            equivariance_error(result.map_points, transformed_result.map_points).cpu()
        ),
        "confidence_equivariance_max_error": float(
            equivariance_error(
                result.map_confidence, transformed_result.map_confidence
            ).cpu()
        ),
        "local_reconstruction_loss": float(reconstruction.cpu()),
        "codebook_perplexity": float(result.perplexity.cpu()),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    codec = MapCodec(
        num_targets=args.num_targets,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
    ).to(device)
    training_loss = train_codec(
        codec=codec,
        steps=args.steps,
        batch_size=args.batch_size,
        num_agents=args.num_agents,
        num_targets=args.num_targets,
        device=device,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    quantized = evaluate_codec(
        codec=codec,
        rounds=args.rounds,
        batch_size=args.eval_batch_size,
        num_agents=args.num_agents,
        num_targets=args.num_targets,
        device=device,
        seed=args.seed + 1,
    )
    quantized["final_training_loss"] = training_loss
    uncompressed = evaluate_codec(
        codec=UncompressedMapCodec(args.num_targets).to(device),
        rounds=args.rounds,
        batch_size=args.eval_batch_size,
        num_agents=args.num_agents,
        num_targets=args.num_targets,
        device=device,
        seed=args.seed + 1,
    )
    reduction = 1.0 - quantized["bits_per_message"] / uncompressed["bits_per_message"]
    return {
        "configuration": {
            "num_agents": args.num_agents,
            "num_targets": args.num_targets,
            "rounds": args.rounds,
            "steps": args.steps,
            "codebook_size": args.codebook_size,
            "num_codebooks": args.num_codebooks,
        },
        "quantized": quantized,
        "uncompressed": uncompressed,
        "message_bit_reduction_fraction": reduction,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--num-targets", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-codebooks", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run_experiment(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
