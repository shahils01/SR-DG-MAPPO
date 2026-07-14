import unittest

import torch

from sr_dg_mappo.codec import MapCodec, UncompressedMapCodec, local_reconstruction_loss


class CodecTests(unittest.TestCase):
    def test_quantized_codec_shapes_and_rate(self):
        codec = MapCodec(
            num_targets=3,
            latent_dim=24,
            num_codebooks=3,
            codebook_size=10,
        )
        points = torch.randn(4, 5, 3, 2)
        confidence = torch.randint(0, 2, (4, 5, 3)).float()
        output = codec(points, confidence)
        self.assertEqual(output.decoded_points.shape, points.shape)
        self.assertEqual(output.decoded_confidence.shape, confidence.shape)
        self.assertEqual(tuple(output.indices.shape), (4, 5, 3))
        self.assertEqual(codec.bits_per_message, 12)
        self.assertTrue(torch.isfinite(output.vq_loss))
        self.assertGreaterEqual(float(output.perplexity), 1.0)

    def test_uncompressed_codec_is_exact(self):
        codec = UncompressedMapCodec(num_targets=2)
        points = torch.randn(3, 4, 2, 2)
        confidence = torch.randint(0, 2, (3, 4, 2)).float()
        output = codec(points, confidence)
        self.assertTrue(torch.equal(output.decoded_points, points))
        self.assertTrue(torch.equal(output.decoded_confidence, confidence))
        self.assertEqual(codec.bits_per_message, 192)

    def test_reconstruction_loss_backpropagates(self):
        codec = MapCodec(num_targets=2, latent_dim=16, num_codebooks=2)
        points = torch.randn(8, 3, 2, 2)
        confidence = torch.randint(0, 2, (8, 3, 2)).float()
        output = codec(points, confidence)
        loss = local_reconstruction_loss(output, points, confidence) + output.vq_loss
        loss.backward()
        gradient_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in codec.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
