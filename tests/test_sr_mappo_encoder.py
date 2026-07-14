import argparse
import unittest

import torch

from mat.algorithms.mat.algorithm.sr_mappo_transformer import (
    SymmetryReducedObservationEncoder,
)
from mat.envs.long_range_predator_prey import LongRangePredatorPreyTorchVecEnv


class SymmetryReducedMAPPOEncoderTests(unittest.TestCase):
    def test_encoder_is_differentiable_and_reports_rate(self):
        args = argparse.Namespace(
            num_predators=4,
            num_prey=2,
            world_size=6.0,
            sr_reconstruction_coef=0.1,
            sr_vq_coef=0.1,
            sr_hidden_dim=32,
            sr_latent_dim=16,
            sr_num_codebooks=2,
            sr_codebook_size=8,
            sr_comm_rounds=2,
        )
        env = LongRangePredatorPreyTorchVecEnv(
            num_envs=3,
            num_predators=4,
            num_prey=2,
            episode_length=5,
            device="cpu",
            seed=5,
        )
        obs, _, _ = env.reset()
        adjacency = torch.as_tensor(env.get_visibility_matrix())
        encoder = SymmetryReducedObservationEncoder(args, obs_dim=35, output_dim=24)
        features = encoder(torch.as_tensor(obs), adjacency)
        loss = features.square().mean() + encoder.auxiliary_loss()
        loss.backward()

        self.assertEqual(features.shape, (3, 4, 24))
        self.assertEqual(encoder.bits_per_message, 6)
        self.assertGreater(
            float(encoder.codec.quantizer.codebook.grad.abs().sum()), 0.0
        )
        metrics = encoder.metrics()
        self.assertIn("sr_bits_per_agent", metrics)
        self.assertIn("sr_reconstruction_loss", metrics)
        self.assertTrue(torch.isfinite(metrics["sr_reconstruction_loss"]))


if __name__ == "__main__":
    unittest.main()
