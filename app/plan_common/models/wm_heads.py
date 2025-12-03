# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import OrderedDict

import torch
from einops import rearrange

from app.plan_common.models.decoder import VisionTransformerDecoder
from app.plan_common.models.lpips_loss import LPIPSWithDiscriminator
from app.plan_common.models.state_decoder import StateReadoutViT
from app.plan_common.models.trainable_model import TrainableModel


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


class WorldModelViTImageHead(TrainableModel):
    """
    Our model is trained to predict the processed rgb format, a Tensor[float] in [0,1],
    composed with a transform noramlizing with meand and std 0.5.
    The decode() function returns Tensor[float] in [0, 255]
    """

    def __init__(self, head_config, train_config=None, inverse_transform=None, device="cpu"):
        model = VisionTransformerDecoder(**head_config)
        super().__init__(model, train_config=train_config, device=device)
        self.device = device
        self.unwrapped = model
        self.inverse_transform = inverse_transform

        # LPIPS configuration
        self.use_lpips = head_config.get("use_lpips", False)
        self.pixelloss_weight = head_config.get("pixelloss_weight", 1.0)
        self.perceptual_weight = head_config.get("perceptual_weight", 1.0)

        # Initialize LPIPS if needed
        if self.use_lpips:
            pretrained_ckpt_root = os.environ.get("PRETRAINED_CKPT_ROOT")
            lpips_ckpt = f"{pretrained_ckpt_root}/vgg_lpips.pth"

            self.combined_loss = LPIPSWithDiscriminator(
                disc_start=1000000000000,
                logvar_init=0.0,
                kl_weight=0.0,
                pixelloss_weight=self.pixelloss_weight,
                perceptual_weight=self.perceptual_weight,
                lpips_ckpt=lpips_ckpt,
                # --- Discriminator Loss ---
                disc_num_layers=4,
                disc_in_channels=3,
                disc_factor=0.0,
                disc_weight=0.5,
                disc_loss="hinge",
                add_discriminator=False,
                using_3d_discriminator=False,
            ).to(device)

    def compute_loss(self, features, rgb_target, global_step=None):
        """
        rgb_target: b t v h w (p q c),
            with the patch size being e.g. p=q=8
            with normalize transform applied, so mean 0 and std 1
        """
        predicted_pixels = self.model(features)  #  b t v h w (p q c)
        if self.use_lpips:
            # Process predicted pixels for LPIPS
            pred_processed = rearrange(
                predicted_pixels,
                "b t v h w (p q c) -> (b t v) c (h p) (w q)",
                p=self.unwrapped.patch_size,
                q=self.unwrapped.patch_size,
                c=self.unwrapped.in_chans,
            ).cpu()

            # Process target pixels for LPIPS
            target_processed = rearrange(
                rgb_target,
                "b t v h w (p q c) -> (b t v) c (h p) (w q)",
                p=self.unwrapped.patch_size,
                q=self.unwrapped.patch_size,
                c=self.unwrapped.in_chans,
            ).cpu()

            # Apply inverse transform to get to [0,1] range but keep as float
            if self.inverse_transform:
                pred_processed = self.inverse_transform(pred_processed).to(self.device)
                target_processed = self.inverse_transform(target_processed).to(self.device)

            # Compute loss with LPIPS
            loss, loss_logs = self.combined_loss(target_processed, pred_processed, 0, global_step)
        else:
            loss = (torch.abs(rgb_target - predicted_pixels)).mean(-1)
        return {"loss": loss.mean()}

    def decode(self, features):
        predicted_pixels = self.model(features)
        return self.postprocess_rgb(predicted_pixels)

    def preprocess_rgb(self, rgb_target):
        rgb_target = rearrange(
            rgb_target,
            "b t c (v n p) (m q) -> b t v n m (p q c)",
            n=self.unwrapped.grid_size,
            m=self.unwrapped.grid_size,
            p=self.unwrapped.patch_size,
            q=self.unwrapped.patch_size,
        )
        # We do not use it since our images are already preprocessed.
        # rgb_target = rgb_target / 255.
        return rgb_target

    def postprocess_rgb(self, rgb_output):
        rgb_output = rgb_output.float()
        B, T, V, _, _, _ = rgb_output.shape
        rgb_ = rearrange(
            rgb_output,
            "b t v h w (p q c) -> (b t v) c (h p) (w q)",
            p=self.unwrapped.patch_size,
            q=self.unwrapped.patch_size,
            c=self.unwrapped.in_chans,
        ).cpu()
        if self.inverse_transform is not None:
            rgb_ = self.inverse_transform(rgb_)
        rgb_ = (255.0 * rgb_).clip(0.0, 255.0).to(torch.uint8)
        rgb_ = rearrange(
            rgb_,
            "(b t v) c (h p) (w q) -> b t v (h p) (w q) c",
            p=self.unwrapped.patch_size,
            q=self.unwrapped.patch_size,
            c=self.unwrapped.in_chans,
            b=B,
            t=T,
        )
        return rgb_


class WorldModelPoseReadoutHead(TrainableModel):
    def __init__(self, head_config, train_config=None, device="cpu"):
        model = StateReadoutViT(**head_config)
        super().__init__(model, train_config=train_config, device=device)
        self.unwrapped = model
        self.proprio_dim = head_config.get("proprio_dim", 4)

    def compute_loss(self, features, camera_extrinsics, states, reduce_mean=True):
        predicted_states = self.model(features, camera_extrinsics)
        state_loss = torch.abs(predicted_states - states)
        if reduce_mean:
            state_loss_comps = {f"loss_{i}": state_loss[..., i].mean() for i in range(state_loss.shape[-1])}
            state_loss_comps.update({"loss": state_loss.mean()})
            state_loss_comps.update({"loss_proprio": state_loss[..., : self.proprio_dim].mean()})
        else:
            state_loss_comps = {f"loss_{i}": state_loss[..., i] for i in range(state_loss.shape[-1])}
            state_loss_comps.update({"loss": state_loss})
            state_loss_comps.update({"loss_proprio": state_loss[..., : self.proprio_dim]})
        return state_loss_comps

    def decode(self, features, camera_extrinsics):
        outputs = self.model(features, camera_extrinsics)
        # out = {f'dim_{i}': outputs[..., i] for i in range(outputs.shape[-1])}
        return outputs
        # {
        #     'x': outputs[..., 2], # positive - forward, negative - backward
        #     'y': outputs[..., 1], # positive - right, negarive - left
        #     'z': outputs[..., 3], # positive - up, negative - down
        #     'th1': outputs[..., 4],
        #     'th2': outputs[..., 5],
        #     'th3': outputs[..., 6],
        #     'gripper': outputs[..., 0],
        # }


class WorldModelRewardReadoutHead(TrainableModel):
    def __init__(self, head_config, train_config=None, device="cpu"):
        model = StateReadoutViT(**head_config)
        super().__init__(model, train_config=train_config, device=device)
        self.unwrapped = model

    def compute_loss(self, features, rewards):
        predicted_rewards = self.model(features, None)
        reward_loss = torch.abs(predicted_rewards - rewards)
        reward_loss_comps = {"loss": reward_loss.mean()}
        return reward_loss_comps

    def decode(self, features):
        outputs = self.model(features, None)
        return {
            "reward": outputs[..., 0],
        }
