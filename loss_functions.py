from __future__ import division
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from inverse_warp import inverse_warp

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

    # Reshape depth to [B, 1, H, W] for compatibility
    depth = depth.unsqueeze(1)  # [B, 1, H, W]

    # Warp the reference image to the target frame
    ref_img_warped, valid_mask = inverse_warp(
        ref_img, depth.squeeze(1), pose, intrinsics, rotation_mode, padding_mode
    )  # ref_img_warped: [B, 3, H, W], valid_mask: [B, H, W]

    # Compute L1 photometric loss
    diff = (tgt_img - ref_img_warped) * valid_mask.unsqueeze(1).float()  # Masked difference
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


@torch.no_grad()
def compute_depth_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1
    skipped = 0
    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask
        if valid.sum() == 0:
            continue

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid]

        median_pred = torch.median(valid_pred.clamp(1e-3, 80))
        valid_pred = valid_pred * torch.median(valid_gt) / median_pred
        valid_pred = valid_pred.clamp(1e-3, 80)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)
    if skipped == batch_size:
        return None

    return [metric.item() / (batch_size - skipped) for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]


@torch.no_grad()
def compute_pose_errors(gt, pred):
    RE = 0
    for (current_gt, current_pred) in zip(gt, pred):
        snippet_length = current_gt.shape[0]
        scale_factor = torch.sum(current_gt[..., -1] * current_pred[..., -1]) / torch.sum(current_pred[..., -1] ** 2)
        ATE = torch.norm((current_gt[..., -1] - scale_factor * current_pred[..., -1]).reshape(-1)).cpu().numpy()
        R = current_gt[..., :3] @ current_pred[..., :3].transpose(-2, -1)
        for gt_pose, pred_pose in zip(current_gt, current_pred):
            # Residual matrix to which we compute angle's sin and cos
            R = (gt_pose[:, :3] @ torch.inverse(pred_pose[:, :3])).cpu().numpy()
            s = np.linalg.norm([R[0, 1]-R[1, 0],
                                R[1, 2]-R[2, 1],
                                R[0, 2]-R[2, 0]])
            c = np.trace(R) - 1
            # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
            RE += np.arctan2(s, c)

    return [ATE/snippet_length, RE/snippet_length]