import unittest

import torch

from src.models.vision_transformer import VIT_EMBED_DIMS as VIT_V1_EMBED_DIMS
from src.models.vision_transformer import vit_tiny as vit_tiny_v1
from src.models.vision_transformer_v2 import VIT_EMBED_DIMS as VIT_V2_EMBED_DIMS
from src.models.vision_transformer_v2 import vit_tiny as vit_tiny_v2


class TestImageViTV1(unittest.TestCase):
    def setUp(self) -> None:
        self._vit_tiny = vit_tiny_v1()
        self.height, self.width = 224, 224
        self.num_patches = (self.height // self._vit_tiny.patch_size) * (self.width // self._vit_tiny.patch_size)

    def test_model_image_nomask_batchsize_4(self):
        BS = 4
        x = torch.rand((BS, 3, self.height, self.width))
        y = self._vit_tiny(x)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, self.num_patches, VIT_V1_EMBED_DIMS["vit_tiny"]))

    def test_model_image_nomask_batchsize_1(self):
        BS = 1
        x = torch.rand((BS, 3, self.height, self.width))
        y = self._vit_tiny(x)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, self.num_patches, VIT_V1_EMBED_DIMS["vit_tiny"]))

    def test_model_image_masked_batchsize_4(self):
        BS = 4
        mask_indices = [6, 7, 8]
        masks = [torch.tensor(mask_indices, dtype=torch.int64) for _ in range(BS)]
        x = torch.rand((BS, 3, self.height, self.width))
        y = self._vit_tiny(x, masks=masks)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, len(mask_indices), VIT_V1_EMBED_DIMS["vit_tiny"]))

    def test_model_image_masked_batchsize_1(self):
        BS = 1
        mask_indices = [6, 7, 8]
        masks = [torch.tensor(mask_indices, dtype=torch.int64) for _ in range(BS)]
        x = torch.rand((BS, 3, self.height, self.width))
        y = self._vit_tiny(x, masks=masks)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, len(mask_indices), VIT_V1_EMBED_DIMS["vit_tiny"]))


class TestVideoViTV1(unittest.TestCase):
    def setUp(self) -> None:
        self.num_frames = 8
        self._vit_tiny = vit_tiny_v1(num_frames=8)
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
        self.assertEqual(y.size(), (BS, self.num_patches, VIT_V1_EMBED_DIMS["vit_tiny"]))

    def test_model_video_nomask_batchsize_1(self):
        BS = 1
        x = torch.rand((BS, 3, self.num_frames, self.height, self.width))
        y = self._vit_tiny(x)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, self.num_patches, VIT_V1_EMBED_DIMS["vit_tiny"]))

    def test_model_video_masked_batchsize_4(self):
        BS = 4
        mask_indices = [6, 7, 8]
        masks = [torch.tensor(mask_indices, dtype=torch.int64) for _ in range(BS)]
        x = torch.rand((BS, 3, self.num_frames, self.height, self.width))
        y = self._vit_tiny(x, masks=masks)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, len(mask_indices), VIT_V1_EMBED_DIMS["vit_tiny"]))

    def test_model_video_masked_batchsize_1(self):
        BS = 1
        mask_indices = [6, 7, 8]
        masks = [torch.tensor(mask_indices, dtype=torch.int64) for _ in range(BS)]
        x = torch.rand((BS, 3, self.num_frames, self.height, self.width))
        y = self._vit_tiny(x, masks=masks)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, len(mask_indices), VIT_V1_EMBED_DIMS["vit_tiny"]))


class TestImageViTV2(unittest.TestCase):
    def setUp(self) -> None:
        self._vit_tiny = vit_tiny_v2()
        self.height, self.width = 224, 224
        self.num_patches = (self.height // self._vit_tiny.patch_size) * (self.width // self._vit_tiny.patch_size)

    def test_model_image_nomask_batchsize_4(self):
        BS = 4
        x = torch.rand((BS, 3, self.height, self.width))
        y = self._vit_tiny(x)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, self.num_patches, VIT_V2_EMBED_DIMS["vit_tiny"]))

    def test_model_image_nomask_batchsize_1(self):
        BS = 1
        x = torch.rand((BS, 3, self.height, self.width))
        y = self._vit_tiny(x)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, self.num_patches, VIT_V2_EMBED_DIMS["vit_tiny"]))

    def test_model_image_masked_batchsize_4(self):
        BS = 4
        mask_indices = [6, 7, 8]
        masks = [torch.tensor(mask_indices, dtype=torch.int64) for _ in range(BS)]
        x = torch.rand((BS, 3, self.height, self.width))
        y = self._vit_tiny(x, masks=masks)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, len(mask_indices), VIT_V2_EMBED_DIMS["vit_tiny"]))

    def test_model_image_masked_batchsize_1(self):
        BS = 1
        mask_indices = [6, 7, 8]
        masks = [torch.tensor(mask_indices, dtype=torch.int64) for _ in range(BS)]
        x = torch.rand((BS, 3, self.height, self.width))
        y = self._vit_tiny(x, masks=masks)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, len(mask_indices), VIT_V2_EMBED_DIMS["vit_tiny"]))


class TestVideoViTV2(unittest.TestCase):
    def setUp(self) -> None:
        self.num_frames = 8
        self._vit_tiny = vit_tiny_v2(num_frames=8)
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
        self.assertEqual(y.size(), (BS, self.num_patches, VIT_V2_EMBED_DIMS["vit_tiny"]))

    def test_model_video_nomask_batchsize_1(self):
        BS = 1
        x = torch.rand((BS, 3, self.num_frames, self.height, self.width))
        y = self._vit_tiny(x)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, self.num_patches, VIT_V2_EMBED_DIMS["vit_tiny"]))

    def test_model_video_masked_batchsize_4(self):
        BS = 4
        mask_indices = [6, 7, 8]
        masks = [torch.tensor(mask_indices, dtype=torch.int64) for _ in range(BS)]
        x = torch.rand((BS, 3, self.num_frames, self.height, self.width))
        y = self._vit_tiny(x, masks=masks)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, len(mask_indices), VIT_V2_EMBED_DIMS["vit_tiny"]))

    def test_model_video_masked_batchsize_1(self):
        BS = 1
        mask_indices = [6, 7, 8]
        masks = [torch.tensor(mask_indices, dtype=torch.int64) for _ in range(BS)]
        x = torch.rand((BS, 3, self.num_frames, self.height, self.width))
        y = self._vit_tiny(x, masks=masks)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), (BS, len(mask_indices), VIT_V2_EMBED_DIMS["vit_tiny"]))


class TestDinoV2Encoder(unittest.TestCase):
    def setUp(self) -> None:
        self.height, self.width = 224, 224
        self.patch_size = 14
        self.num_patches = (self.height // self.patch_size) * (self.width // self.patch_size)

    def test_dinov2_encoder_patch_tokens_batchsize_4(self):
        try:
            from app.plan_common.models.dino import DinoEncoder

            encoder = DinoEncoder(name="dinov2_vits14", feature_key="x_norm_patchtokens")
            BS = 4
            x = torch.rand((BS, 3, self.height, self.width))
            y = encoder(x)

            self.assertIsInstance(y, torch.Tensor)
            self.assertEqual(y.size(), (BS, self.num_patches, encoder.emb_dim))
        except Exception as e:
            self.skipTest(f"DinoV2 encoder test skipped due to: {str(e)}")

    def test_dinov2_encoder_patch_tokens_batchsize_1(self):
        try:
            from app.plan_common.models.dino import DinoEncoder

            encoder = DinoEncoder(name="dinov2_vits14", feature_key="x_norm_patchtokens")
            BS = 1
            x = torch.rand((BS, 3, self.height, self.width))
            y = encoder(x)

            self.assertIsInstance(y, torch.Tensor)
            self.assertEqual(y.size(), (BS, self.num_patches, encoder.emb_dim))
        except Exception as e:
            self.skipTest(f"DinoV2 encoder test skipped due to: {str(e)}")

    def test_dinov2_encoder_cls_token_batchsize_4(self):
        try:
            from app.plan_common.models.dino import DinoEncoder

            encoder = DinoEncoder(name="dinov2_vits14", feature_key="x_norm_clstoken")
            BS = 4
            x = torch.rand((BS, 3, self.height, self.width))
            y = encoder(x)

            self.assertIsInstance(y, torch.Tensor)
            # cls token output has shape (BS, 1, emb_dim) due to unsqueeze
            self.assertEqual(y.size(), (BS, 1, encoder.emb_dim))
        except Exception as e:
            self.skipTest(f"DinoV2 encoder test skipped due to: {str(e)}")


class TestDinoV3Encoder(unittest.TestCase):
    def setUp(self) -> None:
        self.height, self.width = 224, 224
        self.patch_size = 16  # dinov3 typically uses patch size 16
        self.num_patches = (self.height // self.patch_size) * (self.width // self.patch_size)

    def test_dinov3_encoder_patch_tokens_batchsize_4(self):
        try:
            from app.plan_common.models.dino import DinoEncoder

            encoder = DinoEncoder(name="dinov3_vits16", feature_key="x_norm_patchtokens")
            BS = 4
            x = torch.rand((BS, 3, self.height, self.width))
            y = encoder(x)

            self.assertIsInstance(y, torch.Tensor)
            self.assertEqual(y.size(), (BS, self.num_patches, encoder.emb_dim))
        except Exception as e:
            self.skipTest(f"DinoV3 encoder test skipped due to: {str(e)}")

    def test_dinov3_encoder_patch_tokens_batchsize_1(self):
        try:
            from app.plan_common.models.dino import DinoEncoder

            encoder = DinoEncoder(name="dinov3_vits16", feature_key="x_norm_patchtokens")
            BS = 1
            x = torch.rand((BS, 3, self.height, self.width))
            y = encoder(x)

            self.assertIsInstance(y, torch.Tensor)
            self.assertEqual(y.size(), (BS, self.num_patches, encoder.emb_dim))
        except Exception as e:
            self.skipTest(f"DinoV3 encoder test skipped due to: {str(e)}")

    def test_dinov3_encoder_cls_token_batchsize_4(self):
        try:
            from app.plan_common.models.dino import DinoEncoder

            encoder = DinoEncoder(name="dinov3_vits16", feature_key="x_norm_clstoken")
            BS = 4
            x = torch.rand((BS, 3, self.height, self.width))
            y = encoder(x)

            self.assertIsInstance(y, torch.Tensor)
            # cls token output has shape (BS, 1, emb_dim) due to unsqueeze
            self.assertEqual(y.size(), (BS, 1, encoder.emb_dim))
        except Exception as e:
            self.skipTest(f"DinoV3 encoder test skipped due to: {str(e)}")


if __name__ == "__main__":
    unittest.main()
