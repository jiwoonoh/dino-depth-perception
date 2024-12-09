from __future__ import division
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from inverse_warp import inverse_warp
from utils import projective_inverse_warp

def photometric_reconstruction_loss(tgt_img, ref_img, intrinsics, depth, pose, 
                                    rotation_mode='euler', padding_mode='zeros'):
    """
    Computes the photometric reconstruction loss between the target and reference images.

    Args:
        tgt_img (Tensor): Target image (shape: [B, 3, H, W]).
        ref_img (Tensor): Reference image (shape: [B, 3, H, W]).
        intrinsics (Tensor): Camera intrinsics (shape: [B, 3, 3]).
        depth (Tensor): Depth map for the target view (shape: [B, H, W]).
        pose (Tensor): Relative pose between target and reference (shape: [B, 6]).
        rotation_mode (str): Rotation parameterization mode ('euler' or 'matrix').
        padding_mode (str): Padding mode for `grid_sample` ('zeros' or 'border').

    Returns:
        loss (Tensor): Photometric reconstruction loss (scalar).
    """
    batch_size, _, H, W = tgt_img.shape

    # Ensure depth has correct shape [B, 1, H, W]
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)  # [B, 1, H, W]

    # Warp the reference image to the target frame
    ref_img_warped, valid_points = projective_inverse_warp(
        src_image=ref_img, 
        depth=depth.squeeze(1),  # Depth expected as [B, H, W]
        pose=pose, 
        intrinsics=intrinsics
    )  # ref_img_warped: [B, 3, H, W], valid_points: [B, H, W]

    # Compute L1 photometric loss
    diff = (tgt_img - ref_img_warped) * valid_points.unsqueeze(1).float()  # Masked difference
    l1_loss = diff.abs().mean()

    return l1_loss

def smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # don't ask me why it works better
    return loss