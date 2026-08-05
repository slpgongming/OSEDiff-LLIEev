import os
import unittest

import torch

from models.rgb_event_encoder import RGBEventEncoder


class RGBEventEncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pretrained_root = os.environ.get(
            "RGB_EVENT_PRETRAINED_ROOT", "checkpoints/HDRev"
        )

    def test_strict_pretrained_load_and_shapes(self):
        model = RGBEventEncoder(event_channels=5)
        reports, manifest = model.load_pretrained(
            pretrained_root=self.pretrained_root, strict=True
        )
        self.assertEqual(
            {key: report.loaded for key, report in reports.items()},
            {"encoder_i": 52, "encoder_e": 52, "fusion": 48},
        )
        self.assertNotIn("Decoder.pth", str(manifest))

        model.eval()
        rgb = torch.rand(1, 3, 16, 16) * 2 - 1
        events = torch.randn(1, 5, 16, 16)
        with torch.no_grad():
            output = model(rgb, events)
        self.assertEqual(
            [tuple(feature.shape) for feature in output.fused_features],
            [
                (1, 64, 16, 16),
                (1, 128, 8, 8),
                (1, 256, 4, 4),
                (1, 512, 2, 2),
            ],
        )

    def test_rgb_range_and_channel_flip(self):
        rgb = torch.tensor([[[[-1.0]], [[0.0]], [[1.0]]]])
        adapted = RGBEventEncoder.adapt_rgb_input(rgb)
        expected = torch.tensor([[[[1.0]], [[0.5]], [[0.0]]]])
        torch.testing.assert_close(adapted, expected)

    def test_recurrent_state_does_not_leak(self):
        torch.manual_seed(11)
        model = RGBEventEncoder(event_channels=5).eval()
        rgb_a = torch.randn(1, 3, 16, 16)
        rgb_b = torch.randn(1, 3, 16, 16)
        event_a = torch.randn(1, 5, 16, 16)
        event_b = torch.randn(1, 5, 16, 16)
        with torch.no_grad():
            first = model(rgb_a, event_a).fused_features
            model(rgb_b, event_b)
            repeated = model(rgb_a, event_a).fused_features
        for expected, actual in zip(first, repeated):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

