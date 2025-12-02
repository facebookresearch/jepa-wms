# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import Counter

from omegaconf import OmegaConf

from src.datasets.new.utils import SUPPORTED_METHODS, make_dataset

SUPPORTED_METHODS["DummyDataset"] = ("tests.datasets.new.test_iterator", "DummyDataset")


class TestMixedDataset(unittest.TestCase):
    def test_samples_match_specified_weights(self):
        # Weighted mixing three datasets that always return 0, 1 or 2
        config = """
        type: MixedDataset
        components:
            - weight: .2
              config:
                type: DummyDataset
                bounds:
                - 0
                - 1
            - weight: .3
              config:
                type: DummyDataset
                bounds:
                - 1
                - 2
            - weight: .5
              config:
                type: DummyDataset
                bounds:
                - 2
                - 3
        """
        config = OmegaConf.create(config)
        mixed_dataset = make_dataset(config)

        num_samples = 10**5
        samples = []
        mixed_dataset = iter(mixed_dataset)
        for i in range(num_samples):
            samples.append(next(mixed_dataset))

        c = Counter(samples)
        # Each value's sampling rate should be very close to the config weight
        self.assertAlmostEqual(c[0] / c.total(), 0.2, places=2)
        self.assertAlmostEqual(c[1] / c.total(), 0.3, places=2)
        self.assertAlmostEqual(c[2] / c.total(), 0.5, places=2)
