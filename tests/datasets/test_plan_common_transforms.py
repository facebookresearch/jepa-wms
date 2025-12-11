import unittest

import torch

from app.plan_common.datasets.transforms import (
    InverseVideoTransform,
    VideoTransform,
    _tensor_normalize_inplace,
    make_inverse_transforms,
    make_transforms,
    tensor_normalize,
)


class TestMakeTransforms(unittest.TestCase):
    """Test the make_transforms factory function."""

    def setUp(self):
        self.g = torch.Generator()
        self.g.manual_seed(42)
        self.T, self.C, self.H, self.W = 8, 3, 280, 320
        self.crop_size = 224

    def test_make_transforms_returns_video_transform(self):
        transform = make_transforms()
        self.assertIsInstance(transform, VideoTransform)

    def test_make_transforms_default_params(self):
        transform = make_transforms()
        self.assertEqual(transform.crop_size, 224)
        self.assertTrue(transform.random_horizontal_flip)
        self.assertEqual(transform.random_resize_aspect_ratio, (3 / 4, 4 / 3))
        self.assertEqual(transform.random_resize_scale, (0.3, 1.0))
        self.assertFalse(transform.auto_augment)
        self.assertFalse(transform.motion_shift)
        self.assertFalse(transform.hwc)
        self.assertFalse(transform.do_255_to_1)

    def test_make_transforms_custom_img_size(self):
        transform = make_transforms(img_size=112)
        self.assertEqual(transform.crop_size, 112)

    def test_make_transforms_custom_normalize(self):
        custom_mean = (0.5, 0.5, 0.5)
        custom_std = (0.5, 0.5, 0.5)
        transform = make_transforms(normalize=(custom_mean, custom_std))
        torch.testing.assert_close(transform.mean, torch.tensor(custom_mean, dtype=torch.float32))
        torch.testing.assert_close(transform.std, torch.tensor(custom_std, dtype=torch.float32))


class TestVideoTransformSingleVideo(unittest.TestCase):
    """Test VideoTransform with single video input [T, C, H, W]."""

    def setUp(self):
        self.g = torch.Generator()
        self.g.manual_seed(42)
        self.T, self.C, self.H, self.W = 8, 3, 280, 320
        self.crop_size = 224

    def test_single_video_tchw_output_shape(self):
        transform = make_transforms(img_size=self.crop_size, random_horizontal_flip=False)
        video = torch.randint(0, 255, (self.T, self.C, self.H, self.W), generator=self.g, dtype=torch.float32)
        output = transform(video)
        self.assertEqual(output.shape, (self.T, self.C, self.crop_size, self.crop_size))

    def test_single_video_thwc_output_shape(self):
        transform = make_transforms(img_size=self.crop_size, random_horizontal_flip=False, hwc=True)
        video = torch.randint(0, 255, (self.T, self.H, self.W, self.C), generator=self.g, dtype=torch.float32)
        output = transform(video)
        self.assertEqual(output.shape, (self.T, self.C, self.crop_size, self.crop_size))

    def test_single_video_output_dtype(self):
        transform = make_transforms(img_size=self.crop_size)
        video = torch.randint(0, 255, (self.T, self.C, self.H, self.W), generator=self.g, dtype=torch.float32)
        output = transform(video)
        self.assertEqual(output.dtype, torch.float32)

    def test_single_video_uint8_input(self):
        transform = make_transforms(img_size=self.crop_size)
        video = torch.randint(0, 255, (self.T, self.C, self.H, self.W), generator=self.g, dtype=torch.uint8)
        output = transform(video)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.shape, (self.T, self.C, self.crop_size, self.crop_size))


class TestVideoTransformBatchedVideo(unittest.TestCase):
    """Test VideoTransform with batched video input [B, T, C, H, W]."""

    def setUp(self):
        self.g = torch.Generator()
        self.g.manual_seed(42)
        self.B, self.T, self.C, self.H, self.W = 4, 8, 3, 280, 320
        self.crop_size = 224

    def test_batched_video_btchw_output_shape(self):
        transform = make_transforms(img_size=self.crop_size, random_horizontal_flip=False)
        video = torch.randint(0, 255, (self.B, self.T, self.C, self.H, self.W), generator=self.g, dtype=torch.float32)
        output = transform(video)
        self.assertEqual(output.shape, (self.B, self.T, self.C, self.crop_size, self.crop_size))

    def test_batched_video_bthwc_output_shape(self):
        transform = make_transforms(img_size=self.crop_size, random_horizontal_flip=False, hwc=True)
        video = torch.randint(0, 255, (self.B, self.T, self.H, self.W, self.C), generator=self.g, dtype=torch.float32)
        output = transform(video)
        self.assertEqual(output.shape, (self.B, self.T, self.C, self.crop_size, self.crop_size))

    def test_batched_video_batchsize_1(self):
        transform = make_transforms(img_size=self.crop_size)
        video = torch.randint(0, 255, (1, self.T, self.C, self.H, self.W), generator=self.g, dtype=torch.float32)
        output = transform(video)
        self.assertEqual(output.shape, (1, self.T, self.C, self.crop_size, self.crop_size))


class TestVideoTransformAugmentations(unittest.TestCase):
    """Test VideoTransform augmentation options."""

    def setUp(self):
        self.g = torch.Generator()
        self.g.manual_seed(42)
        self.T, self.C, self.H, self.W = 8, 3, 280, 320
        self.crop_size = 224

    def test_with_motion_shift(self):
        transform = make_transforms(img_size=self.crop_size, motion_shift=True)
        video = torch.randint(0, 255, (self.T, self.C, self.H, self.W), generator=self.g, dtype=torch.float32)
        output = transform(video)
        self.assertEqual(output.shape, (self.T, self.C, self.crop_size, self.crop_size))

    def test_with_random_erasing(self):
        transform = make_transforms(img_size=self.crop_size, reprob=0.5)
        video = torch.randint(0, 255, (self.T, self.C, self.H, self.W), generator=self.g, dtype=torch.float32)
        output = transform(video)
        self.assertEqual(output.shape, (self.T, self.C, self.crop_size, self.crop_size))

    def test_with_255_to_1_normalization(self):
        transform = make_transforms(img_size=self.crop_size, do_255_to_1=True)
        video = torch.randint(0, 255, (self.T, self.C, self.H, self.W), generator=self.g, dtype=torch.float32)
        output = transform(video)
        self.assertEqual(output.shape, (self.T, self.C, self.crop_size, self.crop_size))

    def test_different_crop_sizes(self):
        for crop_size in [112, 224, 256]:
            transform = make_transforms(img_size=crop_size)
            video = torch.randint(0, 255, (self.T, self.C, self.H, self.W), generator=self.g, dtype=torch.float32)
            output = transform(video)
            self.assertEqual(output.shape, (self.T, self.C, crop_size, crop_size))


class TestMakeInverseTransforms(unittest.TestCase):
    """Test the make_inverse_transforms factory function."""

    def setUp(self):
        self.g = torch.Generator()
        self.g.manual_seed(42)
        self.T, self.C, self.H, self.W = 8, 3, 224, 224

    def test_make_inverse_transforms_returns_inverse_video_transform(self):
        transform = make_inverse_transforms()
        self.assertIsInstance(transform, InverseVideoTransform)

    def test_make_inverse_transforms_default_params(self):
        transform = make_inverse_transforms()
        self.assertEqual(transform.img_size, 224)
        torch.testing.assert_close(transform.mean, torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32))
        torch.testing.assert_close(transform.std, torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32))

    def test_make_inverse_transforms_custom_params(self):
        custom_mean = (0.5, 0.5, 0.5)
        custom_std = (0.5, 0.5, 0.5)
        transform = make_inverse_transforms(img_size=112, normalize=(custom_mean, custom_std))
        self.assertEqual(transform.img_size, 112)
        torch.testing.assert_close(transform.mean, torch.tensor(custom_mean, dtype=torch.float32))
        torch.testing.assert_close(transform.std, torch.tensor(custom_std, dtype=torch.float32))


class TestInverseVideoTransform(unittest.TestCase):
    """Test InverseVideoTransform class."""

    def setUp(self):
        self.g = torch.Generator()
        self.g.manual_seed(42)
        self.T, self.C, self.H, self.W = 8, 3, 224, 224

    def test_inverse_transform_output_shape_tchw(self):
        transform = make_inverse_transforms()
        normalized_video = torch.randn(self.T, self.C, self.H, self.W, generator=self.g)
        output = transform(normalized_video)
        self.assertEqual(output.shape, (self.T, self.C, self.H, self.W))

    def test_inverse_transform_output_dtype(self):
        transform = make_inverse_transforms()
        normalized_video = torch.randn(self.T, self.C, self.H, self.W, generator=self.g)
        output = transform(normalized_video)
        self.assertEqual(output.dtype, torch.float32)

    def test_denormalize_operation(self):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = InverseVideoTransform(mean=mean, std=std)
        normalized_frame = torch.zeros(3, 224, 224)
        denormalized = transform.denormalize(normalized_frame)
        expected = torch.full((3, 224, 224), 0.5)
        torch.testing.assert_close(denormalized, expected)


class TestTransformInverseConsistency(unittest.TestCase):
    """Test that inverse transform approximately reverses the forward transform."""

    def setUp(self):
        self.g = torch.Generator()
        self.g.manual_seed(42)
        self.T, self.C, self.H, self.W = 8, 3, 224, 224
        self.normalize = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def test_forward_inverse_approximate_identity(self):
        forward_transform = make_transforms(
            img_size=224,
            random_horizontal_flip=False,
            random_resize_scale=(1.0, 1.0),
            random_resize_aspect_ratio=(1.0, 1.0),
            normalize=self.normalize,
        )
        inverse_transform = make_inverse_transforms(img_size=224, normalize=self.normalize)

        original = torch.rand(self.T, self.C, self.H, self.W, generator=self.g)
        transformed = forward_transform(original)
        recovered = inverse_transform(transformed)

        self.assertEqual(original.shape, recovered.shape)


class TestTensorNormalize(unittest.TestCase):
    """Test tensor_normalize utility function."""

    def setUp(self):
        self.g = torch.Generator()
        self.g.manual_seed(42)

    def test_tensor_normalize_with_tensor_mean_std(self):
        tensor = torch.rand(8, 224, 224, 3, generator=self.g)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalized = tensor_normalize(tensor, mean, std)
        self.assertEqual(normalized.shape, tensor.shape)

    def test_tensor_normalize_with_list_mean_std(self):
        tensor = torch.rand(8, 224, 224, 3, generator=self.g)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalized = tensor_normalize(tensor, mean, std)
        self.assertEqual(normalized.shape, tensor.shape)

    def test_tensor_normalize_uint8_input(self):
        tensor = torch.randint(0, 255, (8, 224, 224, 3), generator=self.g, dtype=torch.uint8)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalized = tensor_normalize(tensor, mean, std)
        self.assertEqual(normalized.dtype, torch.float32)

    def test_tensor_normalize_with_255_to_1(self):
        tensor = torch.randint(0, 255, (8, 224, 224, 3), generator=self.g, dtype=torch.float32)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalized = tensor_normalize(tensor, mean, std, do_255_to_1=True)
        self.assertEqual(normalized.shape, tensor.shape)


class TestTensorNormalizeInplace(unittest.TestCase):
    """Test _tensor_normalize_inplace utility function."""

    def setUp(self):
        self.g = torch.Generator()
        self.g.manual_seed(42)

    def test_inplace_normalize_shape(self):
        C, T, H, W = 3, 8, 224, 224
        tensor = torch.rand(C, T, H, W, generator=self.g)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalized = _tensor_normalize_inplace(tensor, mean, std)
        self.assertEqual(normalized.shape, (C, T, H, W))

    def test_inplace_normalize_uint8(self):
        C, T, H, W = 3, 8, 224, 224
        tensor = torch.randint(0, 255, (C, T, H, W), generator=self.g, dtype=torch.uint8)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalized = _tensor_normalize_inplace(tensor, mean, std)
        self.assertEqual(normalized.dtype, torch.float32)

    def test_inplace_normalize_with_255_to_1(self):
        C, T, H, W = 3, 8, 224, 224
        tensor = torch.randint(0, 255, (C, T, H, W), generator=self.g, dtype=torch.float32)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalized = _tensor_normalize_inplace(tensor, mean, std, do_255_to_1=True)
        self.assertEqual(normalized.shape, (C, T, H, W))


if __name__ == "__main__":
    unittest.main()
