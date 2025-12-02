import unittest

import torch

from src.models.vision_transformer import VIT_EMBED_DIMS, vit_tiny


class TestImageViT(unittest.TestCase):
    def setUp(self) -> None:
        self._vit_tiny = vit_tiny()
        self.height, self.width = 224, 224
        self.num_patches = (self.height // self._vit_tiny.patch_size) * (self.width // self._vit_tiny.patch_size)

    def test_model_image_nomask_batchsize_4(self):
        BS = 4
        x = torch.rand((BS, 3, self.height, self.width))
        y = self._vit_tiny(x)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, self.num_patches, VIT_EMBED_DIMS["vit_tiny"]))

    def test_model_image_nomask_batchsize_1(self):
        BS = 1
        x = torch.rand((BS, 3, self.height, self.width))
        y = self._vit_tiny(x)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, self.num_patches, VIT_EMBED_DIMS["vit_tiny"]))

    def test_model_image_masked_batchsize_4(self):
        BS = 4
        mask_indices = [6, 7, 8]
        masks = [torch.tensor(mask_indices, dtype=torch.int64) for _ in range(BS)]
        x = torch.rand((BS, 3, self.height, self.width))
        y = self._vit_tiny(x, masks=masks)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, len(mask_indices), VIT_EMBED_DIMS["vit_tiny"]))

    def test_model_image_masked_batchsize_1(self):
        BS = 1
        mask_indices = [6, 7, 8]
        masks = [torch.tensor(mask_indices, dtype=torch.int64) for _ in range(BS)]
        x = torch.rand((BS, 3, self.height, self.width))
        y = self._vit_tiny(x, masks=masks)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, len(mask_indices), VIT_EMBED_DIMS["vit_tiny"]))


class TestVideoViT(unittest.TestCase):
    def setUp(self) -> None:
        self.num_frames = 8
        self._vit_tiny = vit_tiny(num_frames=8)
        self.height, self.width = 224, 224
        self.num_patches = (
            (self.height // self._vit_tiny.patch_size)
            * (self.width // self._vit_tiny.patch_size)
            * (self.num_frames // self._vit_tiny.tubelet_size)
        )

    def test_model_video_nomask_batchsize_4(self):
        BS = 4
        x = torch.rand((BS, 3, self.num_frames, self.height, self.width))
        y = self._vit_tiny(x)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, self.num_patches, VIT_EMBED_DIMS["vit_tiny"]))

    def test_model_video_nomask_batchsize_1(self):
        BS = 1
        x = torch.rand((BS, 3, self.num_frames, self.height, self.width))
        y = self._vit_tiny(x)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, self.num_patches, VIT_EMBED_DIMS["vit_tiny"]))

    def test_model_video_masked_batchsize_4(self):
        BS = 4
        mask_indices = [6, 7, 8]
        masks = [torch.tensor(mask_indices, dtype=torch.int64) for _ in range(BS)]
        x = torch.rand((BS, 3, self.num_frames, self.height, self.width))
        y = self._vit_tiny(x, masks=masks)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, len(mask_indices), VIT_EMBED_DIMS["vit_tiny"]))

    def test_model_video_masked_batchsize_1(self):
        BS = 1
        mask_indices = [6, 7, 8]
        masks = [torch.tensor(mask_indices, dtype=torch.int64) for _ in range(BS)]
        x = torch.rand((BS, 3, self.num_frames, self.height, self.width))
        y = self._vit_tiny(x, masks=masks)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, len(mask_indices), VIT_EMBED_DIMS["vit_tiny"]))
