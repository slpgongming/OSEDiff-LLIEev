import unittest

import torch

from dataloaders.sde_event_film_dataset import SDEEventFilmDataset


class EventViewTest(unittest.TestCase):
    def test_two_views_are_derived_from_same_raw_voxel(self):
        raw = torch.tensor(
            [
                [[0.0, 1.0], [-1.0, 0.0]],
                [[2.0, 0.0], [0.0, -2.0]],
            ]
        )
        dataset = object.__new__(SDEEventFilmDataset)
        clora = dataset._normalize_event_voxel(raw)
        refinement = dataset._normalize_event_voxel_refinement(raw)
        self.assertEqual(tuple(clora.shape), tuple(raw.shape))
        self.assertEqual(tuple(refinement.shape), tuple(raw.shape))
        self.assertTrue(torch.equal(refinement == 0, raw == 0))
        nonzero = refinement[raw != 0]
        self.assertAlmostEqual(float(nonzero.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(nonzero.std(unbiased=False)), 1.0, places=5)

    def test_empty_and_constant_events_are_stable(self):
        zero = torch.zeros(5, 4, 4)
        constant = torch.zeros(5, 4, 4)
        constant[0, 0, 0] = 1.0
        for raw in (zero, constant):
            view = SDEEventFilmDataset._normalize_event_voxel_refinement(raw)
            self.assertTrue(torch.isfinite(view).all())
            self.assertTrue(torch.equal(view, torch.zeros_like(view)))

    def test_dataset_level_shuffle_never_returns_self(self):
        dataset = object.__new__(SDEEventFilmDataset)
        dataset.samples = list(range(7))
        for index in range(len(dataset.samples)):
            donor = dataset._shuffled_event_index(index)
            self.assertNotEqual(index, donor)
            self.assertGreaterEqual(donor, 0)
            self.assertLess(donor, len(dataset.samples))


if __name__ == "__main__":
    unittest.main()

