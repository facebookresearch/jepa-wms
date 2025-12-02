# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.multiprocessing as mp

from src.datasets.new.iterator import DistributedIndexIterator


class DummyDataset(DistributedIndexIterator, torch.utils.data.IterableDataset):
    """Mock dataset for basic unit testing."""

    def __init__(self, bounds=(0, 10), transform=None, **kwargs):
        # transform is not used
        super().__init__(**kwargs)
        self.min, self.max = bounds

    def __len__(self):
        return self.max - self.min

    def get_sample(self, i):
        return self.min + i


class TestIterator(unittest.TestCase):
    def test_dataset_skips_samples(self):
        num_samples1 = 1000
        num_samples2 = 500
        skip_samples2 = num_samples1 - num_samples2

        ds1 = DummyDataset(bounds=(100, 200), seed=0, rank_and_world_size=(0, 1))
        ds2 = DummyDataset(bounds=(100, 200), seed=0, rank_and_world_size=(0, 1))
        ds2.set_skip_samples(skip_samples2)
        ds1 = iter(ds1)
        ds2 = iter(ds2)
        samples1 = []
        samples2 = []

        for i in range(num_samples1):
            samples1.append(next(ds1))

        for i in range(num_samples2):
            samples2.append(next(ds2))

        # Samples 0-499 from ds2 are the same as samples 500-999 from ds1
        self.assertEqual(samples1[skip_samples2:], samples2)

    def test_dataloader_skips_samples(self):
        # Batches are only guaranteed to be identical if the number of skipped samples is a multiple of
        # batch_size * num_workers (here, 1000-500 is a multiple of 5*4)
        batch_size = 5
        num_workers = 4

        num_samples1 = 100
        num_samples2 = 40
        skip_samples2 = num_samples1 - num_samples2
        skip_samples_per_worker2 = skip_samples2 // num_workers
        num_batches1 = num_samples1 // batch_size
        num_batches2 = num_samples2 // batch_size
        skip_batches2 = skip_samples2 // batch_size

        ds1 = DummyDataset(bounds=(100, 200), seed=0, rank_and_world_size=(0, 1), shuffle=False)
        ds2 = DummyDataset(bounds=(100, 200), seed=0, rank_and_world_size=(0, 1), shuffle=False)
        ds2.set_skip_samples(skip_samples_per_worker2)

        mp.set_start_method("spawn")
        dl1 = torch.utils.data.DataLoader(ds1, batch_size=batch_size, num_workers=num_workers)
        dl2 = torch.utils.data.DataLoader(ds2, batch_size=batch_size, num_workers=num_workers)

        batches1 = []
        batches2 = []

        dl1 = iter(dl1)
        for i in range(num_batches1):
            batches1.append(next(dl1))

        dl2 = iter(dl2)
        for i in range(num_batches2):
            batches2.append(next(dl2))

        # First few batches from dl1 are skipped by dl2; remaining ones are equivalent
        torch.testing.assert_close(batches1[skip_batches2:], batches2, atol=0, rtol=0)
