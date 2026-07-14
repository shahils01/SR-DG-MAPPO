import unittest

import numpy as np

from mat.envs.long_range_predator_prey import LongRangePredatorPreyTorchVecEnv


class PredatorPreyEnvironmentTests(unittest.TestCase):
    def test_vector_environment_contract_and_graph(self):
        env = LongRangePredatorPreyTorchVecEnv(
            num_envs=3,
            num_predators=4,
            num_prey=2,
            episode_length=5,
            device="cpu",
            seed=11,
        )
        obs, share_obs, available_actions = env.reset()
        self.assertEqual(obs.shape, (3, 4, 35))
        self.assertEqual(share_obs.shape, (3, 4, 27))
        self.assertEqual(available_actions.shape, (3, 4, 1))

        adjacency = env.get_visibility_matrix()
        self.assertEqual(adjacency.shape, (3, 4, 4))
        self.assertTrue(np.allclose(adjacency, adjacency.transpose(0, 2, 1)))
        self.assertTrue(np.all(np.diagonal(adjacency, axis1=1, axis2=2) == 1.0))

        actions = np.zeros((3, 4, 2), dtype=np.float32)
        next_obs, next_share_obs, rewards, dones, infos, _ = env.step(actions)
        self.assertEqual(next_obs.shape, obs.shape)
        self.assertEqual(next_share_obs.shape, share_obs.shape)
        self.assertEqual(rewards.shape, (3, 4, 1))
        self.assertEqual(dones.shape, (3, 4))
        self.assertEqual(len(infos), 3)


if __name__ == "__main__":
    unittest.main()
