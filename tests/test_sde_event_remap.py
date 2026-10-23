"""错配事件的坐标缩放必须保留绝对时间戳，且兼容整数事件。"""
import unittest

import torch

from dataloaders.sde_event_film_dataset import SDEEventFilmDataset


class EventRemapTest(unittest.TestCase):
    def setUp(self):
        self.dataset = SDEEventFilmDataset.__new__(SDEEventFilmDataset)

    def test_int64_scaling_preserves_timestamp_polarity_and_input(self):
        # 相邻时间戳只差 1；整体转 float32 会丢失此差异。
        events = torch.tensor([[1689664848265408, 3, 5, 1],
                               [1689664848265409, 7, 9, -1]], dtype=torch.int64)
        original = events.clone()
        mapped = self.dataset._remap_event_extent(events, 8, 10, 4, 20)
        self.assertEqual(mapped.dtype, torch.int64)
        torch.testing.assert_close(mapped[:, 0], original[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(mapped[:, 3], original[:, 3], rtol=0, atol=0)
        torch.testing.assert_close(mapped[:, 1:3], torch.tensor([[1, 10], [3, 18]]))
        torch.testing.assert_close(events, original, rtol=0, atol=0)

    def test_equal_extent_preserves_integer_events(self):
        events = torch.tensor([[1689664848265409, 100, 200, 1]], dtype=torch.int64)
        mapped = self.dataset._remap_event_extent(events, 346, 260, 346, 260)
        torch.testing.assert_close(mapped, events, rtol=0, atol=0)

    def test_float_events_keep_fractional_coordinates(self):
        events = torch.tensor([[10.0, 3.0, 5.0, -1.0]], dtype=torch.float64)
        mapped = self.dataset._remap_event_extent(events, 8, 10, 4, 5)
        torch.testing.assert_close(mapped, torch.tensor([[10.0, 1.5, 2.5, -1.0]], dtype=torch.float64))

    def test_empty_integer_events(self):
        events = torch.empty((0, 4), dtype=torch.int64)
        mapped = self.dataset._remap_event_extent(events, 8, 10, 4, 20)
        self.assertEqual(mapped.shape, events.shape)
        self.assertEqual(mapped.dtype, events.dtype)

    def test_voxel_nearest_resize_preserves_channels(self):
        events = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
        mapped = self.dataset._remap_event_extent(events, 2, 2, 4, 4)
        torch.testing.assert_close(mapped, events.repeat_interleave(2, -2).repeat_interleave(2, -1))


if __name__ == "__main__":
    unittest.main()
