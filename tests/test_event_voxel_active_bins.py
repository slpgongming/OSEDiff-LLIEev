import unittest

import torch

from dataloaders.event_voxel_layout import concentrate_event_voxel


class EventVoxelActiveBinsTest(unittest.TestCase):
    def setUp(self):
        # 每个输入 bin 都放置不同值，便于确认重分 bin 后的位置与总和。
        self.voxel = torch.arange(1, 6, dtype=torch.float32).view(5, 1, 1)

    def test_one_active_bin_reproduces_all_events_in_bin_zero(self):
        concentrated = concentrate_event_voxel(self.voxel, 1)
        self.assertEqual(tuple(concentrated.shape), (5, 1, 1))
        torch.testing.assert_close(concentrated[:, 0, 0], torch.tensor([15., 0., 0., 0., 0.]))

    def test_partial_active_bins_keep_channels_and_preserve_event_sum(self):
        concentrated = concentrate_event_voxel(self.voxel, 2)
        # source bins 0/1/2 -> target 0，3/4 -> target 1；后三通道为零。
        torch.testing.assert_close(concentrated[:, 0, 0], torch.tensor([6., 9., 0., 0., 0.]))
        self.assertEqual(float(concentrated.sum()), float(self.voxel.sum()))

    def test_invalid_active_bin_count_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "active_bins"):
            concentrate_event_voxel(self.voxel, 0)
        with self.assertRaisesRegex(ValueError, "active_bins"):
            concentrate_event_voxel(self.voxel, 6)


if __name__ == "__main__":
    unittest.main()
