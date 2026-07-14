import unittest

import torch
from torch import nn

from mat.algorithms.mat.algorithm.sr_mappo_transformer import (
    DistributedSymmetryReducedObservationEncoder,
)
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.algorithms.mat.mat_trainer import MATTrainer
from mat.config import get_config
from mat.envs.long_range_predator_prey import LongRangePredatorPreyTorchVecEnv
from sr_dg_mappo.codec import MapCodec
from sr_dg_mappo.communication import (
    InvariantMapAttention,
    SymmetryReducedDGATCommunicator,
)
from sr_dg_mappo.groups import d4_matrices, rotation_frames, transform_poses
from sr_dg_mappo.train import configure_algorithm, parse_args


def distributed_args(num_agents=3):
    parser = get_config()
    args = parse_args(
        [
            "--algorithm_name",
            "sr_mappo",
            "--num_predators",
            str(num_agents),
            "--num_prey",
            "2",
            "--n_embd",
            "16",
            "--sr_hidden_dim",
            "16",
            "--sr_latent_dim",
            "8",
            "--sr_num_codebooks",
            "2",
            "--sr_codebook_size",
            "8",
            "--sr_comm_rounds",
            "1",
        ],
        parser,
    )
    configure_algorithm(args)
    return args


class DistributedSRDGATTests(unittest.TestCase):
    def test_global_d4_transform_preserves_receiver_frame_estimates(self):
        torch.manual_seed(4)
        batch_size, num_agents, num_targets = 2, 3, 2
        codecs = nn.ModuleList(
            [
                MapCodec(
                    num_targets=num_targets,
                    hidden_dim=12,
                    latent_dim=8,
                    num_codebooks=2,
                    codebook_size=8,
                    coordinate_scale=6.0,
                )
                for _ in range(num_agents)
            ]
        )
        fusers = nn.ModuleList(
            [InvariantMapAttention(hidden_dim=8, coordinate_scale=6.0) for _ in range(num_agents)]
        )
        communicator = SymmetryReducedDGATCommunicator(codecs, fusers, rounds=2)

        positions = torch.randn(batch_size, num_agents, 2)
        frames = rotation_frames(torch.randn(batch_size, num_agents))
        local_points = torch.randn(batch_size, num_agents, num_targets, 2)
        confidence = torch.rand(batch_size, num_agents, num_targets)
        adjacency = torch.tensor(
            [[[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]]
        ).repeat(batch_size, 1, 1)

        reference = communicator(
            positions, frames, local_points, confidence, adjacency
        )
        matrix = d4_matrices(dtype=positions.dtype)[5]
        transformed_positions, transformed_frames = transform_poses(
            positions,
            frames,
            matrix,
            translation=torch.tensor([1.7, -0.4]),
        )
        transformed = communicator(
            transformed_positions,
            transformed_frames,
            local_points,
            confidence,
            adjacency,
        )

        self.assertTrue(
            torch.allclose(reference.map_points, transformed.map_points, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(
                reference.map_confidence, transformed.map_confidence, atol=1e-6
            )
        )

    def test_distributed_encoder_has_agent_local_losses_and_codebooks(self):
        args = distributed_args(num_agents=4)
        env = LongRangePredatorPreyTorchVecEnv(
            num_envs=3,
            num_predators=4,
            num_prey=2,
            episode_length=5,
            device="cpu",
            seed=7,
        )
        obs, _, _ = env.reset()
        encoder = DistributedSymmetryReducedObservationEncoder(
            args, obs_dim=35, output_dim=16
        )
        features = encoder(
            torch.as_tensor(obs), torch.as_tensor(env.get_visibility_matrix())
        )

        self.assertEqual(features.shape, (3, 4, 16))
        self.assertEqual(encoder.auxiliary_loss().shape, (4, 1))
        self.assertEqual(len(encoder.agent_codecs), 4)
        codebook_ids = {
            encoder.agent_codecs[i].quantizer.codebook.data_ptr() for i in range(4)
        }
        self.assertEqual(len(codebook_ids), 4)
        self.assertIn("sr_attention_entropy", encoder.metrics())

        (features.square().mean() + encoder.auxiliary_loss().mean()).backward()
        for codec in encoder.agent_codecs:
            self.assertGreater(float(codec.quantizer.codebook.grad.abs().sum()), 0.0)

    def test_policy_uses_local_optimizers_and_graph_neighbor_consensus(self):
        args = distributed_args(num_agents=3)
        self.assertTrue(args.truelyDistributed)
        env = LongRangePredatorPreyTorchVecEnv(
            num_envs=1,
            num_predators=3,
            num_prey=2,
            episode_length=4,
            device="cpu",
            seed=3,
        )
        policy = TransformerPolicy(
            args,
            env.observation_space[0],
            env.share_observation_space[0],
            env.action_space[0],
            num_agents=3,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(policy.optimizers), 3)
        optimizer_parameter_ids = [
            {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
            for optimizer in policy.optimizers
        ]
        self.assertTrue(optimizer_parameter_ids[0].isdisjoint(optimizer_parameter_ids[1]))

        trainer = MATTrainer(args, policy, num_agents=3, device=torch.device("cpu"))
        codebooks = policy.transformer.obs_encoder.agent_codecs
        critic_parameters = [next(module.parameters()) for module in policy.transformer.encoder.head_]
        actor_parameters = [next(module.parameters()) for module in policy.transformer.decoder.mlp_]
        with torch.no_grad():
            for parameters in (
                [codec.quantizer.codebook for codec in codebooks],
                critic_parameters,
                actor_parameters,
            ):
                parameters[0].fill_(0.0)
                parameters[1].fill_(3.0)
                parameters[2].fill_(9.0)
        adjacency = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        consensus_error = trainer.apply_sr_parameter_consensus(adjacency)

        self.assertTrue(torch.isfinite(consensus_error))
        self.assertTrue(
            torch.allclose(
                codebooks[0].quantizer.codebook,
                torch.ones_like(codebooks[0].quantizer.codebook),
            )
        )
        self.assertTrue(
            torch.allclose(
                codebooks[1].quantizer.codebook,
                torch.full_like(codebooks[1].quantizer.codebook, 2.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                codebooks[2].quantizer.codebook,
                torch.full_like(codebooks[2].quantizer.codebook, 9.0),
            )
        )
        for parameters in (critic_parameters, actor_parameters):
            self.assertTrue(
                torch.allclose(parameters[0], torch.ones_like(parameters[0]))
            )
            self.assertTrue(
                torch.allclose(parameters[1], torch.full_like(parameters[1], 2.0))
            )
            self.assertTrue(
                torch.allclose(parameters[2], torch.full_like(parameters[2], 9.0))
            )

        shared_args = parse_args(
            ["--algorithm_name", "sr_mappo_shared"], get_config()
        )
        configure_algorithm(shared_args)
        self.assertFalse(shared_args.truelyDistributed)


if __name__ == "__main__":
    unittest.main()
