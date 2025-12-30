import torch
from torchvision import transforms

import src.datasets.utils.video.transforms as video_transforms
from src.datasets.utils.video.randerase import RandomErasing


class InverseVideoTransform:
    """
    Input:
        [T C H W] or [B T C H W] Tensor because of broadcasting between frame and self.std / self.mean
        Expects a float tensor with entries normalized by mean and std.
    This transform's Denormalize does
        output[channel] = (input[channel] * std[channel]) + mean[channel].
    Output:
        [T C H W] or [B T C H W] Tensor
    """

    def __init__(
        self,
        img_size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):
        self.img_size = img_size
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def denormalize(self, frame):
        return (frame * self.std[:, None, None]) + self.mean[:, None, None]

    def resize_back(self, frame):
        # Assuming original size is not known, we can't exactly reverse the Resize and CenterCrop operations.
        # However, if you know the original size, you can use transforms.Resize(original_size) here.
        return frame

    def __call__(self, frames):
        transformed_frames = []
        for frame in frames:
            denormalized_frame = self.denormalize(frame)
            resized_frame = self.resize_back(denormalized_frame)
            transformed_frames.append(resized_frame)
        return torch.stack(transformed_frames)


def make_inverse_transforms(
    img_size=224,
    normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    **kwargs,
):
    transform = InverseVideoTransform(img_size=img_size, mean=normalize[0], std=normalize[1])
    return transform


def make_transforms(
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(3 / 4, 4 / 3),
    random_resize_scale=(0.3, 1.0),
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    img_size=224,
    normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    hwc=False,
    do_255_to_1=False,
):
    transform = VideoTransform(
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=img_size,
        normalize=normalize,
        hwc=hwc,
        do_255_to_1=do_255_to_1,
    )
    return transform


class VideoTransform(object):
    """Video transformation class for augmentation and normalization of video data.

    This class applies a series of transformations to video frames including:
    - Optional auto-augmentation (RandAugment)
    - Random resized cropping (with optional motion shift)
    - Optional horizontal flipping
    - Normalization
    - Optional random erasing
    Input:
        [B T C H W] or [T C H W] if not self.hwc
        [B T H W C] or [T H W C] if self.hwc
    Returns: [B T C H W] or [T C H W] Tensor after applying the transformations.
    """

    def __init__(
        self,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(3 / 4, 4 / 3),
        random_resize_scale=(0.3, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=224,
        normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        hwc=False,
        do_255_to_1=False,
    ):
        self.hwc = hwc
        self.random_horizontal_flip = random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.auto_augment = auto_augment
        self.motion_shift = motion_shift
        self.crop_size = crop_size
        self.do_255_to_1 = do_255_to_1
        self.mean = torch.tensor(normalize[0], dtype=torch.float32)
        self.std = torch.tensor(normalize[1], dtype=torch.float32)

        self.autoaug_transform = video_transforms.create_random_augment(
            input_size=(crop_size, crop_size),
            auto_augment="rand-m7-n4-mstd0.5-inc1",
            interpolation="bicubic",
        )

        self.spatial_transform = (
            video_transforms.random_resized_crop_with_shift if motion_shift else video_transforms.random_resized_crop
        )

        self.reprob = reprob
        self.erase_transform = RandomErasing(
            reprob,
            mode="pixel",
            max_count=1,
            num_splits=1,
            device="cpu",
        )

    def __call__(self, buffer):
        """
        buffer: List of frames (numpy arrays or tensors) or a tensor of shape
            [B T C H W] or [T C H W] if not self.hwc
            [B T H W C] or [T H W C] if self.hwc
        Returns: [B T C H W] or [T C H W] Tensor after applying the transformations.
        """
        if self.auto_augment:
            # Auto-augment expects list of frames (no batch dimension)
            buffer = [transforms.ToPILImage()(frame) for frame in buffer]
            buffer = self.autoaug_transform(buffer)
            buffer = [transforms.ToTensor()(img) for img in buffer]
            buffer = torch.stack(buffer)  # T C H W
            buffer = buffer.permute(0, 2, 3, 1)  # T H W C
        else:
            # Convert to tensor if not already
            if not torch.is_tensor(buffer):
                buffer = torch.tensor(buffer, dtype=torch.float32)

            # Handle batched input [B, T, C, H, W] or single video [T, C, H, W]
            if buffer.dim() == 5:
                processed = []
                for b in range(buffer.size(0)):
                    batch_slice = buffer[b]  # [T, C, H, W] or [T, H, W, C] if self.hwc
                    processed_slice = self._process_single_video(batch_slice)
                    processed.append(processed_slice.unsqueeze(0))
                buffer = torch.cat(processed, dim=0)
            else:
                buffer = self._process_single_video(buffer)
        return buffer

    def _process_single_video(self, buffer):
        """
        Expects a single video torch.tensor of shape [T, C, H, W] / [T, H, W, C] if self.hwc
        Returns [T, C, H, W] always
        """
        if buffer.dtype != torch.float32:
            buffer = buffer.to(torch.float32)

        if self.hwc:
            buffer = buffer.permute(3, 0, 1, 2)  # T H W C -> C T H W
        else:
            buffer = buffer.permute(1, 0, 2, 3)  # T C H W -> C T H W

        buffer = self.spatial_transform(
            images=buffer,
            target_height=self.crop_size,
            target_width=self.crop_size,
            scale=self.random_resize_scale,
            ratio=self.random_resize_aspect_ratio,
        )
        if self.random_horizontal_flip:
            buffer, _ = video_transforms.horizontal_flip(0.5, buffer)

        buffer = _tensor_normalize_inplace(buffer, self.mean, self.std, do_255_to_1=self.do_255_to_1)
        if self.reprob > 0:
            self.erase_transform.device = buffer.device
            buffer = buffer.permute(1, 0, 2, 3)  # C T H W -> T C H W
            buffer = self.erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)  # T C H W -> C T H W

        # Added by Basile to ensure output shape is [T, C, H, W]
        buffer = buffer.permute(1, 0, 2, 3)  # C T H W -> T C H W
        return buffer


def tensor_normalize(tensor, mean, std, do_255_to_1=False):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
    if do_255_to_1:
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def _tensor_normalize_inplace(tensor, mean, std, do_255_to_1=False):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize (with dimensions C, T, H, W).
        mean (tensor): mean value to subtract (in 0 to 255 floats).
        std (tensor): std to divide (in 0 to 255 floats).
    """
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)

    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
    if do_255_to_1:
        tensor.div_(255.0)
    C, T, H, W = tensor.shape
    tensor = tensor.view(C, -1).permute(1, 0)  # Make C the last dimension
    tensor.sub_(mean).div_(std)
    tensor = tensor.permute(1, 0).view(C, T, H, W)  # Put C back in front
    return tensor
