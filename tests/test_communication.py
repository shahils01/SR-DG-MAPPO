import unittest

import torch

from sr_dg_mappo.codec import UncompressedMapCodec
from sr_dg_mappo.communication import SymmetryReducedCommunicator
from sr_dg_mappo.data import SceneBatch, sample_scenes
from sr_dg_mappo.groups import d4_matrices, rotation_frames


class CommunicationTests(unittest.TestCase):
    def test_two_rounds_propagate_information_over_chain(self):
        positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
        frames = rotation_frames(torch.zeros(1, 3))
        targets = torch.tensor([[[0.25, 0.0]]])
        adjacency = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
        local_points = torch.tensor([[[[0.25, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]]])
        visibility = torch.tensor([[[1.0], [0.0], [0.0]]])
        scene = SceneBatch(positions, frames, targets, adjacency, local_points, visibility)
        communicator = SymmetryReducedCommunicator(
            UncompressedMapCodec(num_targets=1), rounds=2
        )
        output = communicator(
            scene.positions,
            scene.frames,
            scene.local_points,
            scene.visibility,
            scene.adjacency,
        )
        self.assertGreater(float(output.map_confidence[0, 2, 0]), 0.99)
        self.assertTrue(
            torch.allclose(output.map_points[0, 2, 0], torch.tensor([-1.75, 0.0]))
        )
        self.assertEqual(float(output.total_bits), 4 * 2 * 96)

    def test_communication_is_invariant_in_agent_frames(self):
        generator = torch.Generator().manual_seed(3)
        scene = sample_scenes(
            batch_size=4,
            num_agents=5,
            num_targets=2,
            generator=generator,
        )
        communicator = SymmetryReducedCommunicator(
            UncompressedMapCodec(num_targets=2), rounds=2
        )
        reference = communicator(
            scene.positions,
            scene.frames,
            scene.local_points,
            scene.visibility,
            scene.adjacency,
        )
        transformed = scene.transformed(d4_matrices()[5], torch.tensor([0.4, -0.7]))
        transformed_output = communicator(
            transformed.positions,
            transformed.frames,
            transformed.local_points,
            transformed.visibility,
            transformed.adjacency,
        )
        self.assertTrue(
            torch.allclose(reference.map_points, transformed_output.map_points, atol=2e-6)
        )
        self.assertTrue(
            torch.allclose(
                reference.map_confidence, transformed_output.map_confidence, atol=1e-6
            )
        )


if __name__ == "__main__":
    unittest.main()
