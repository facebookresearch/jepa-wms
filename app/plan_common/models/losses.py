import torch


def compute_loss(
    pred_video_features,
    pred_proprio_features,
    video_features,
    proprio_features,
    proprio_loss=True,
    visual_loss=True,
    shift=1,
    num_views=1,
    reduce_mean=True,
):
    """
    Input:
        proprio_features: [B, T, 1, D]
        video_features: [B, T, V, H, W, D]
    """
    # if not self.use_proprio: proprio_loss = False
    B, T, V, H, W, C = pred_video_features.shape
    V = num_views
    # H, W = self.grid_size, self.grid_size
    pred_video_features = pred_video_features.reshape(B, T, V * H * W, C)
    video_features = video_features.reshape(B, T, V * H * W, C)
    # targets = targets.reshape(B, T, V * H * W, -1)
    if shift != 0:
        visual_targets_ = video_features[:, shift:]
        visual_features_ = pred_video_features[:, :-shift]
        if proprio_loss:
            proprio_targets_ = proprio_features[:, shift:]
            proprio_features_ = pred_proprio_features[:, :-shift]
    else:
        visual_targets_ = video_features
        visual_features_ = pred_video_features
        if proprio_loss:
            proprio_targets_ = proprio_features
            proprio_features_ = pred_proprio_features
    loss = 0.0
    visual_cos_loss = -(
        visual_features_
        * visual_targets_
        / (visual_features_.norm(dim=-1, keepdim=True) * visual_targets_.norm(dim=-1, keepdim=True))
    ).sum(-1)
    visual_l1_loss = l1_(visual_features_, visual_targets_).mean(dim=-1)
    visual_l2_loss = l2_(visual_features_, visual_targets_).mean(dim=-1)
    visual_smooth_l1_loss = smooth_l1_(visual_features_, visual_targets_).mean(dim=-1)
    if proprio_loss:
        proprio_cos_loss = -(
            proprio_features_
            * proprio_targets_
            / (proprio_features_.norm(dim=-1, keepdim=True) * proprio_targets_.norm(dim=-1, keepdim=True))
        ).sum(-1)
        proprio_l1_loss = l1_(proprio_features_, proprio_targets_).mean(dim=-1)
        proprio_l2_loss = l2_(proprio_features_, proprio_targets_).mean(dim=-1)
        proprio_smooth_l1_loss = smooth_l1_(proprio_features_, proprio_targets_).mean(dim=-1)
    # Combine losses
    if visual_loss:
        loss += visual_cos_loss
        loss += visual_l1_loss
        loss += visual_l2_loss
        loss += visual_smooth_l1_loss
    if proprio_loss:
        loss += proprio_cos_loss
        loss += proprio_l1_loss
        loss += proprio_l2_loss
        loss += proprio_smooth_l1_loss
    out = {
        "loss": loss.mean() if reduce_mean else loss,
        "visual_cos_loss": visual_cos_loss.mean() if reduce_mean else visual_cos_loss,
        "visual_l1_loss": visual_l1_loss.mean() if reduce_mean else visual_l1_loss,
        "visual_l2_loss": visual_l2_loss.mean() if reduce_mean else visual_l2_loss,
        "visual_smooth_l1_loss": visual_smooth_l1_loss.mean() if reduce_mean else visual_smooth_l1_loss,
    }
    if proprio_loss:
        out.update(
            {
                "proprio_cos_loss": proprio_cos_loss.mean() if reduce_mean else proprio_cos_loss,
                "proprio_l1_loss": proprio_l1_loss.mean() if reduce_mean else proprio_l1_loss,
                "proprio_l2_loss": proprio_l2_loss.mean() if reduce_mean else proprio_l2_loss,
                "proprio_smooth_l1_loss": proprio_smooth_l1_loss.mean() if reduce_mean else proprio_smooth_l1_loss,
            }
        )
    return out


l2_ = torch.nn.MSELoss(reduction="none")
l1_ = torch.nn.L1Loss(reduction="none")
smooth_l1_ = torch.nn.SmoothL1Loss(reduction="none")
