import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sr_dg_mappo.train import main


class MARLTrainingSmokeTests(unittest.TestCase):
    def _run(self, algorithm):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = [
                "--no_cuda",
                "--env_device",
                "cpu",
                "--algorithm_name",
                algorithm,
                "--num_env_steps",
                "4",
                "--episode_length",
                "4",
                "--env_episode_length",
                "4",
                "--n_rollout_threads",
                "1",
                "--n_training_threads",
                "1",
                "--ppo_epoch",
                "1",
                "--num_mini_batch",
                "1",
                "--mini_batch_size",
                "4",
                "--n_embd",
                "16",
                "--hid_dim",
                "16",
                "--iterations",
                "1",
                "--num-layers",
                "1",
                "--dropout",
                "0",
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
                "--save_interval",
                "1",
                "--log_interval",
                "1",
                "--experiment_name",
                f"test_{algorithm}",
            ]
            with patch.dict(os.environ, {"SR_DG_MAPPO_RESULTS": temp_dir}):
                with redirect_stdout(io.StringIO()):
                    main(args)
            checkpoints = list(Path(temp_dir).rglob("transformer_0.pt"))
            self.assertEqual(len(checkpoints), 1)
            self.assertGreater(checkpoints[0].stat().st_size, 0)

    def test_original_dg_mappo_baseline_updates(self):
        self._run("mappo_dgnn")

    def test_distributed_dg_mappo_baseline_updates(self):
        self._run("mappo_dgnn_dsgd")

    def test_symmetry_reduced_mappo_updates(self):
        self._run("sr_mappo")

    def test_shared_codec_ablation_updates(self):
        self._run("sr_mappo_shared")


if __name__ == "__main__":
    unittest.main()
