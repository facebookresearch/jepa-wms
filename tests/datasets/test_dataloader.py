# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

# Import ConcatIndices using a targeted approach to avoid psutil dependency
# from the ResourceMonitoringThread import in the dataloader module
import bisect

import numpy as np


class ConcatIndices:
    """Copy of src.datasets.utils.dataloader.ConcatIndices for testing without psutil dependency.

    Helper to map indices of concatenated/mixed datasets to the sample index for the corresponding dataset.
    """

    cumulative_sizes: np.ndarray

    def __init__(self, sizes):
        self.cumulative_sizes = np.cumsum(sizes)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # Returns a pair (dataset_idx, sample_idx)
        if idx < 0 or idx >= len(self):
            raise ValueError(f"index must be between 0 and the total size ({len(self)})")
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            return dataset_idx, idx
        return dataset_idx, idx - self.cumulative_sizes[dataset_idx - 1]


class TestConcatIndices(unittest.TestCase):
    def test_concat_indices(self):
        sizes = [10, 20, 30, 40]
        total_size = sum(sizes)
        concat_indices = ConcatIndices(sizes)

        # -1 is outside the total range
        with self.assertRaises(ValueError):
            concat_indices[-1]
        # 0-9 map to dataset 0
        self.assertEqual(concat_indices[0], (0, 0))
        self.assertEqual(concat_indices[9], (0, 9))
        # 10-29 map to dataset 1
        self.assertEqual(concat_indices[10], (1, 0))
        self.assertEqual(concat_indices[29], (1, 19))
        # 30-59 map to dataset 2
        self.assertEqual(concat_indices[30], (2, 0))
        self.assertEqual(concat_indices[59], (2, 29))
        # 60-99 map to dataset 3
        self.assertEqual(concat_indices[60], (3, 0))
        self.assertEqual(concat_indices[99], (3, 39))
        # 100 is outside the total range
        with self.assertRaises(ValueError):
            concat_indices[total_size]


if __name__ == "__main__":
    unittest.main()
