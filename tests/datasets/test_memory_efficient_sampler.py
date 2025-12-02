import unittest

from src.datasets.utils.weighted_sampler import (
    MemoryEfficientDistributedWeightedSampler,
    MemoryEfficientDistributedWeightedSamplerLessRepeat,
)


class MockDataset:
    def __init__(self, datasets, dataset_weights):
        self.datasets = datasets
        self.dataset_weights = dataset_weights

    def __len__(self):
        return sum(len(d) for d in self.datasets)


class TestMemoryEfficientSampler(unittest.TestCase):

    def test_shuffled_sampling_single(self):
        "The specific values returned are a function of the random sampler with the given seed."
        datasets = []
        for i in range(3):
            datasets.append([f"DS{i}"] * 100 * (i + 1))

        mock_dataset = MockDataset(datasets, [1, 1, 1])
        sampler = MemoryEfficientDistributedWeightedSampler(mock_dataset, num_replicas=1, rank=0, shuffle=True)

        smplr_it = iter(sampler)

        ex = next(smplr_it)
        self.assertIsNotNone(ex)
        self.assertEqual(ex, 202)  # Based on previous run

        ex = next(smplr_it)
        self.assertIsNotNone(ex)
        self.assertEqual(ex, 26)  # Based on previous run

    def test_shuffled_sampling(self):
        datasets = []
        for i in range(3):
            datasets.append([f"DS{i}"] * 100 * (i + 1))

        mock_dataset = MockDataset(datasets, [1, 2000, 1])
        sampler = MemoryEfficientDistributedWeightedSampler(mock_dataset, num_replicas=8, rank=3, shuffle=True)

        smplr_it = iter(sampler)

        # Notice how the following samples are drawn from the 2nd dataset, which has a weight of 2000.
        ex = next(smplr_it)
        self.assertIsNotNone(ex)
        self.assertEqual(ex, 135)  # Based on previous run

        ex = next(smplr_it)
        self.assertIsNotNone(ex)
        self.assertEqual(ex, 143)  # Based on previous run

    def test_non_shuffled_sampling(self):
        datasets = []
        for i in range(3):
            datasets.append([f"DS{i}"] * 100 * (i + 1))

        mock_dataset = MockDataset(datasets, [1, 10, 1])
        sampler = MemoryEfficientDistributedWeightedSampler(mock_dataset, num_replicas=4, rank=2, shuffle=False)

        smplr_it = iter(sampler)

        ex = next(smplr_it)
        self.assertIsNotNone(ex)
        self.assertEqual(ex, 102)  # Calculated based on the `__next__` function's non shuffled logic.

        ex = next(smplr_it)
        self.assertIsNotNone(ex)
        self.assertEqual(ex, 106)  # Calculated based on the `__next__` function's non shuffled logic.


class TestMemoryEfficientSamplerLessRepeat(unittest.TestCase):

    def test_shuffled_sampling_single(self):
        """
        Testing all weights are equal to 1.
        The specific values returned are a function of the random sampler with the given seed.
        """
        datasets = []
        for i in range(3):
            datasets.append([f"DS{i}"] * 100 * (i + 1))

        mock_dataset = MockDataset(datasets, [1, 1, 1])
        sampler = MemoryEfficientDistributedWeightedSamplerLessRepeat(
            mock_dataset, num_replicas=1, rank=0, shuffle=True
        )

        smplr_it = iter(sampler)

        ex = next(smplr_it)
        self.assertIsNotNone(ex)
        self.assertEqual(ex, 144)  # Based on previous run

        ex = next(smplr_it)
        self.assertIsNotNone(ex)
        self.assertEqual(ex, 84)  # Based on previous run

    def test_shuffled_sampling(self):
        """
        Testing one dominant dataset.
        The specific values returned are a function of the random sampler with the given seed.
        """
        datasets = []
        for i in range(3):
            datasets.append([f"DS{i}"] * 100 * (i + 1))

        mock_dataset = MockDataset(datasets, [1, 2000, 1])
        sampler = MemoryEfficientDistributedWeightedSamplerLessRepeat(
            mock_dataset, num_replicas=8, rank=3, shuffle=True
        )

        smplr_it = iter(sampler)

        # Notice how the following samples are drawn from the 2nd dataset, which has a weight of 2000.
        ex = next(smplr_it)
        self.assertIsNotNone(ex)
        self.assertEqual(ex, 255)  # Based on previous run

        ex = next(smplr_it)
        self.assertIsNotNone(ex)
        self.assertEqual(ex, 231)  # Based on previous run

    def test_non_shuffled_sampling(self):
        datasets = []
        for i in range(3):
            datasets.append([f"DS{i}"] * 100 * (i + 1))

        mock_dataset = MockDataset(datasets, [1, 10, 1])
        sampler = MemoryEfficientDistributedWeightedSamplerLessRepeat(
            mock_dataset, num_replicas=4, rank=2, shuffle=False
        )

        smplr_it = iter(sampler)

        ex = next(smplr_it)
        self.assertIsNotNone(ex)
        self.assertEqual(ex, 102)  # Calculated based on the `__next__` function's non shuffled logic.

        ex = next(smplr_it)
        self.assertIsNotNone(ex)
        self.assertEqual(ex, 106)  # Calculated based on the `__next__` function's non shuffled logic.
