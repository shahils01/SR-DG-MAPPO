import argparse
import unittest

from sr_dg_mappo.demo import run_experiment


class DemoTests(unittest.TestCase):
    def test_small_experiment_runs(self):
        args = argparse.Namespace(
            steps=2,
            batch_size=8,
            eval_batch_size=8,
            num_agents=4,
            num_targets=2,
            rounds=2,
            hidden_dim=32,
            latent_dim=16,
            num_codebooks=2,
            codebook_size=8,
            learning_rate=1e-3,
            device="cpu",
            seed=5,
        )
        result = run_experiment(args)
        self.assertIn("quantized", result)
        self.assertIn("uncompressed", result)
        self.assertGreater(result["message_bit_reduction_fraction"], 0.0)
        self.assertLess(result["quantized"]["equivariance_max_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
