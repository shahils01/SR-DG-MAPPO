import math
import unittest

import torch

from sr_dg_mappo.groups import (
    all_receiver_transports,
    apply_isometry,
    d4_matrices,
    local_to_world,
    orthogonality_error,
    rotation_frames,
    transform_poses,
    world_to_local,
)


class GroupTests(unittest.TestCase):
    def test_d4_contains_eight_orthogonal_matrices(self):
        matrices = d4_matrices()
        self.assertEqual(tuple(matrices.shape), (8, 2, 2))
        self.assertLess(float(orthogonality_error(matrices)), 1e-6)
        determinants = torch.det(matrices)
        self.assertEqual(int((determinants > 0).sum()), 4)
        self.assertEqual(int((determinants < 0).sum()), 4)

    def test_world_local_round_trip(self):
        angles = torch.tensor([[0.2, -1.1]])
        frames = rotation_frames(angles)
        positions = torch.tensor([[[0.4, -0.2], [-0.3, 0.5]]])
        points = torch.tensor(
            [[[[0.7, 0.1], [-0.2, 0.9]], [[0.7, 0.1], [-0.2, 0.9]]]]
        )
        local = world_to_local(points, positions, frames)
        reconstructed = local_to_world(local, positions, frames)
        self.assertTrue(torch.allclose(points, reconstructed, atol=1e-6))

    def test_pairwise_transport_matches_world_conversion(self):
        positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
        frames = rotation_frames(torch.tensor([[0.0, math.pi / 2.0]]))
        sender_points = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
        transported = all_receiver_transports(sender_points, positions, frames)
        sender_one_world = local_to_world(
            sender_points[:, 1], positions[:, 1], frames[:, 1]
        )
        expected_at_receiver_zero = world_to_local(
            sender_one_world, positions[:, 0], frames[:, 0]
        )
        self.assertTrue(
            torch.allclose(transported[:, 0, 1], expected_at_receiver_zero, atol=1e-6)
        )

    def test_local_coordinates_are_invariant_to_global_isometry(self):
        positions = torch.tensor([[[0.2, -0.3]]])
        frames = rotation_frames(torch.tensor([[0.7]]))
        points = torch.tensor([[[[0.9, 0.1], [-0.4, 0.2]]]])
        local_before = world_to_local(points, positions, frames)
        matrix = d4_matrices()[6]
        translation = torch.tensor([0.3, -0.8])
        positions_after, frames_after = transform_poses(
            positions, frames, matrix, translation
        )
        points_after = apply_isometry(points, matrix, translation)
        local_after = world_to_local(points_after, positions_after, frames_after)
        self.assertTrue(torch.allclose(local_before, local_after, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
