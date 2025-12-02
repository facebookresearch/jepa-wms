# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from src.datasets.utils.dataloader import ConcatIndices


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
