from __future__ import division
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import StructuralSimilarityIndexMeasure
import numpy as np
from PIL import Image

def euler_to_matrix(vec_rot):
    """Converts Euler angles to rotation matrix
    Args:
        vec_rot: Euler angles in the order of rx, ry, rz -- [B, 3] torch.tensor
    Returns:
        A rotation matrix -- [B, 3, 3] torch.tensor
    """
    batch_size = vec_rot.shape[0]
    rx, ry, rz = vec_rot[:, 0], vec_rot[:, 1], vec_rot[:, 2]
    
    cos_rx, sin_rx = torch.cos(rx), torch.sin(rx)
    cos_ry, sin_ry = torch.cos(ry), torch.sin(ry)
    cos_rz, sin_rz = torch.cos(rz), torch.sin(rz)
    
    R_x = torch.stack([torch.ones(batch_size), torch.zeros(batch_size), torch.zeros(batch_size),
                       torch.zeros(batch_size), cos_rx, -sin_rx,
                       torch.zeros(batch_size), sin_rx, cos_rx], dim=1).view(batch_size, 3, 3)
    
    R_y = torch.stack([cos_ry, torch.zeros(batch_size), sin_ry,
                       torch.zeros(batch_size), torch.ones(batch_size), torch.zeros(batch_size),
                       -sin_ry, torch.zeros(batch_size), cos_ry], dim=1).view(batch_size, 3, 3)
    
    R_z = torch.stack([cos_rz, -sin_rz, torch.zeros(batch_size),
                       sin_rz, cos_rz, torch.zeros(batch_size),
                       torch.zeros(batch_size), torch.zeros(batch_size), torch.ones(batch_size)], dim=1).view(batch_size, 3, 3)
    
    rotation_matrix = torch.bmm(R_z, torch.bmm(R_y, R_x))
    
    return rotation_matrix

def dof_vec_to_matrix(dof_vec):
    """Converts 6DoF parameters to transformation matrix
    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6] torch.tensor
    Returns:
        A transformation matrix -- [B, 4, 4] torch.tensor
        R11 R12 R13 tx
        R21 R22 R23 ty
        R31 R32 R33 tz
        0   0   0   1
    """
    batch_size = dof_vec.shape[0]
    translation = dof_vec[:,:3]
    # Add a one at the end of translation
    ones = torch.ones(batch_size, 1)
    translation = torch.cat((translation, ones), dim=1)
    rot_vec = dof_vec[:, 3:]
    # print("rot_vec", rot_vec)
    rot_matrix = euler_to_matrix(rot_vec)
    # add zero at 4 row
    zeros = torch.zeros(batch_size, 1, 3)
    rot_matrix = torch.cat((rot_matrix, zeros), dim=1)
    transformation_matrix = torch.cat((rot_matrix, translation.unsqueeze(2)), dim=2)
    return transformation_matrix

def inverse_dof(dof_vec):
    """
    Computes the inverse of 6DoF parameters.
    
    Args:
        dof_vec: Tensor of shape [B, 6], representing 6DoF parameters (tx, ty, tz, rx, ry, rz).
    
    Returns:
        Inverted 6DoF parameters: Tensor of shape [B, 6].
    """
    # Negate both the translation and rotation parts
    translation_inv = -dof_vec[:, :3]
    rotation_inv = -dof_vec[:, 3:]
    return torch.cat((translation_inv, rotation_inv), dim=1)

def step_cloud(I_t, dof_vec):
    """
    Applies a 6DoF transformation to a point cloud.
    
    Args:
        I_t: Tensor of shape [B, N, 3], representing a batch of point clouds.
        dof_vec: Tensor of shape [B, 6], representing 6DoF parameters (tx, ty, tz, rx, ry, rz).
    
    Returns:
        I_t_1: Transformed point cloud, Tensor of shape [B, N, 3].
    """
    batch_size, num_points = I_t.shape[0], I_t.shape[1]
    
    # Step 1: Convert to homogeneous coordinates
    ones = torch.ones(batch_size, num_points, 1, device=I_t.device)  # [B, N, 1]
    I_t_augmented = torch.cat((I_t, ones), dim=2)  # [B, N, 4]
    
    # Step 2: Get the transformation matrix
    transf_mat = dof_vec_to_matrix(dof_vec)  # [B, 4, 4]
    
    # Step 3: Apply the transformation
    # Transpose the transformation matrix for compatibility
    transf_mat = transf_mat.transpose(1, 2)  # [B, 4, 4]
    I_t_1_homo = torch.bmm(I_t_augmented, transf_mat)  # [B, N, 4]
    
    # Step 4: Convert back to Cartesian coordinates
    I_t_1 = I_t_1_homo[:, :, :3]  # Drop the homogeneous coordinate
    
    return I_t_1

def pixel_to_3d(points, intrins):
    """
    Converts pixel coordinates and depth to 3D coordinates.
    
    Args:
        points: Tensor of shape [B, N, 3], where B is the batch size, N is the number of points,
                and each point is represented as (u, v, w) where u and v are pixel coordinates and w is depth.
        intrins: List of intrinsic parameters of the camera.
    
    Returns:
        Tensor of shape [B, N, 3] representing the 3D coordinates.
    """
    fx = intrins[0][0]
    fy = intrins[1][1]
    cx = intrins[0][2]
    cy = intrins[1][2]
    
    u = points[:, :, 0]
    v = points[:, :, 1]
    w = points[:, :, 2]
    
    x = ((u - cx) * w) / fx
    y = ((v - cy) * w) / fy
    z = w
    
    return torch.stack((x, y, z), dim=2)

def _3d_to_pixel(points_3d, intrins):
    """
    Converts 3D coordinates to pixel coordinates and depth.
    
    Args:
        points_3d: Tensor of shape [B, N, 3], where B is the batch size, N is the number of points,
                   and each point is represented as (x, y, z) where x, y, and z are 3D coordinates.
        intrins: List of intrinsic parameters of the camera.
    
    Returns:
        Tensor of shape [B, N, 3] representing the pixel coordinates and depth.
    """
    fx = intrins[0][0]
    fy = intrins[1][1]
    cx = intrins[0][2]
    cy = intrins[1][2]
    
    x = points_3d[:, :, 0]
    y = points_3d[:, :, 1]
    z = points_3d[:, :, 2]
    
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    w = z
    
    return torch.stack((u, v, w), dim=2)
    
def projective_inverse_warp(src_image, depth, pose, intrinsics):
    """
    Warps the source image to the target frame using depth and pose.

    Args:
        src_image (Tensor): Source image tensor (shape: [B, C, H, W]).
        depth (Tensor): Depth map for the target view (shape: [B, H, W]).
        pose (Tensor): 6-DoF pose parameters (shape: [B, 6]).
        intrinsics (Tensor): Camera intrinsics matrix (shape: [B, 3, 3]).

    Returns:
        warped_image (Tensor): Source image warped to the target frame (shape: [B, C, H, W]).
    """
    batch_size, C, H, W = src_image.shape

    # Step 1: Create pixel grid
    device = src_image.device
    u, v = torch.meshgrid(torch.arange(0, W, device=device),
                          torch.arange(0, H, device=device))
    u = u.float().unsqueeze(0).expand(batch_size, -1, -1)  # [B, H, W]
    v = v.float().unsqueeze(0).expand(batch_size, -1, -1)  # [B, H, W]
    ones = torch.ones_like(u)  # [B, H, W]
    pixel_coords = torch.stack([u, v, ones], dim=1)  # [B, 3, H, W]

    # Step 2: Backproject to 3D space
    depth = depth.unsqueeze(1)  # [B, 1, H, W]
    cam_coords = torch.bmm(intrinsics, pixel_coords.view(batch_size, 3, -1))  # [B, 3, H*W]
    cam_coords = cam_coords * depth.view(batch_size, 1, -1)  # [B, 3, H*W]

    # Step 3: Apply pose transformation
    transf_mat = dof_vec_to_matrix(pose)  # [B, 4, 4]
    cam_coords_homo = torch.cat([cam_coords, torch.ones(batch_size, 1, H*W, device=device)], dim=1)  # [B, 4, H*W]
    warped_cam_coords_homo = torch.bmm(transf_mat, cam_coords_homo)  # [B, 4, H*W]
    warped_cam_coords = warped_cam_coords_homo[:, :3, :] / warped_cam_coords_homo[:, 3:4, :]  # [B, 3, H*W]

    # Step 4: Project back to pixel space
    warped_pixel_coords = torch.bmm(intrinsics, warped_cam_coords)  # [B, 3, H*W]
    warped_pixel_coords = warped_pixel_coords[:, :2, :] / warped_pixel_coords[:, 2:3, :]  # [B, 2, H*W]
    warped_pixel_coords = warped_pixel_coords.view(batch_size, 2, H, W)  # [B, 2, H, W]

    # Normalize grid to [-1, 1]
    warped_pixel_coords_x = warped_pixel_coords[:, 0, :, :] / (W - 1) * 2 - 1
    warped_pixel_coords_y = warped_pixel_coords[:, 1, :, :] / (H - 1) * 2 - 1
    grid = torch.stack([warped_pixel_coords_x, warped_pixel_coords_y], dim=3)  # [B, H, W, 2]

    # Step 5: Sample from source image
    warped_image = F.grid_sample(
        src_image, grid, align_corners=False, padding_mode='border'
    )  # [B, C, H, W]

    return warped_image

# def projective_inverse_warp(src_image, depth, pose, intrinsics):
#     """
#     Warps the source image to the target frame using depth and pose.

#     Args:
#         src_image: Source image tensor (shape: [B, H, W, 3]).
#         depth: Depth map for the target view (shape: [B, H, W]).
#         pose: 6-DoF pose parameters (shape: [B, 6]).
#         intrinsics: Camera intrinsics matrix (shape: [B, 3, 3]).

#     Returns:
#         warped_image: Source image warped to the target frame (shape: [B, H, W, 3]).
#     """
#     batch_size, img_height, img_width, _ = src_image.shape

#     # Step 1: Create pixel grid
#     u, v = torch.meshgrid(torch.arange(0, img_width, device=src_image.device),
#                           torch.arange(0, img_height, device=src_image.device))
#     u = u.flatten().float()
#     v = v.flatten().float()
#     pixel_coords = torch.stack([u, v, torch.ones_like(u)], dim=1)  # [HW, 3]
#     pixel_coords = pixel_coords.unsqueeze(0).expand(batch_size, -1, -1)  # [B, HW, 3]

#     # Step 2: Backproject pixels to 3D space
#     cam_coords = pixel_to_3d(pixel_coords, intrinsics)  # [B, HW, 3]
#     cam_coords = cam_coords * depth.view(batch_size, -1, 1)  # Scale by depth

#     # Step 3: Apply 6-DoF transformation
#     cam_coords_transformed = step_cloud(cam_coords, pose)  # [B, HW, 3]

#     # Step 4: Reproject to 2D space
#     pixel_coords_proj = _3d_to_pixel(cam_coords_transformed, intrinsics)  # [B, HW, 3]
#     u_proj = pixel_coords_proj[:, :, 0].view(batch_size, img_height, img_width)
#     v_proj = pixel_coords_proj[:, :, 1].view(batch_size, img_height, img_width)

#     # Step 5: Sample from source image
#     grid = torch.stack([u_proj / img_width * 2 - 1, v_proj / img_height * 2 - 1], dim=-1)  # [B, H, W, 2]
#     warped_image = torch.nn.functional.grid_sample(src_image, grid, align_corners=False)

#     return warped_image

def compute_smoothness_loss(pred_depth, image):
    """
    Computes edge-aware smoothness loss for the predicted depth map.

    Args:
        pred_depth: Predicted depth map (Tensor [B, H, W]).
        image: Corresponding RGB image for edge awareness (Tensor [B, C, H, W]).

    Returns:
        smoothness_loss: Edge-aware smoothness loss (Tensor).
    """
    # Ensure pred_depth has shape [B, 1, H, W] for consistency
    if pred_depth.dim() == 3:
        pred_depth = pred_depth.unsqueeze(1)  # [B, 1, H, W]
    
    # Convert RGB image to grayscale by taking the mean across channels
    grayscale = torch.mean(image, dim=1, keepdim=True)  # [B, 1, H, W]

    # Compute gradients of depth map
    depth_gradient_x = torch.abs(pred_depth[:, :, :, 1:] - pred_depth[:, :, :, :-1])  # [B, 1, H, W-1]
    depth_gradient_y = torch.abs(pred_depth[:, :, 1:, :] - pred_depth[:, :, :-1, :])  # [B, 1, H-1, W]

    # Compute gradients of image
    image_gradient_x = torch.abs(grayscale[:, :, :, 1:] - grayscale[:, :, :, :-1])  # [B, 1, H, W-1]
    image_gradient_y = torch.abs(grayscale[:, :, 1:, :] - grayscale[:, :, :-1, :])  # [B, 1, H-1, W]

    # Weight depth gradients with image gradients
    # Exponential weighting: edges in image lead to less smoothing
    weighted_smoothness_x = depth_gradient_x * torch.exp(-image_gradient_x)
    weighted_smoothness_y = depth_gradient_y * torch.exp(-image_gradient_y)

    # Compute mean loss
    smoothness_loss = torch.mean(weighted_smoothness_x) + torch.mean(weighted_smoothness_y)
    
    return smoothness_loss

def compute_loss(pred_depth, pred_poses, tgt_image, src_image_stack, intrinsics, smooth_weight=0.01):
    """
    Computes photometric loss, smoothness loss, and total loss.

    Args:
        pred_depth (List[Tensor]): List of predicted depth maps for different scales (List of tensors [B, H, W]).
        pred_poses (Tensor): Predicted 6-DoF poses for source frames (Tensor [B, num_source, 6]).
        tgt_image (Tensor): Target image tensor (shape: [B, 3, H, W]).
        src_image_stack (Tensor): Source image stack tensor (shape: [B, 3*num_source, H, W]).
        intrinsics (Tensor): Camera intrinsics matrix (shape: [B, 3, 3]).
        num_source (int, optional): Number of source images. If None, inferred from pred_poses. Defaults to None.
        smooth_weight (float, optional): Weight for the smoothness loss. Defaults to 0.1.

    Returns:
        total_loss (Tensor): Combined loss.
        photometric_loss (Tensor): Photometric loss.
        smoothness_loss (Tensor): Smoothness loss.
    """
    # Initialize SSIM metric
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    photometric_loss = 0.0
    smoothness_loss = 0.0

    # Infer num_source if not provided
    if num_source is None:
        num_source = pred_poses.shape[1]

    for s in range(len(pred_depth)):
        curr_depth = pred_depth[s]  # [B, H, W]

        # Resize images for the current scale
        scale_factor = 1 / (2 ** s)
        curr_tgt_image = F.interpolate(
            tgt_image, scale_factor=scale_factor, mode='bilinear', align_corners=False
        )  # [B, 3, H', W']
        curr_src_image_stack = F.interpolate(
            src_image_stack, scale_factor=scale_factor, mode='bilinear', align_corners=False
        )  # [B, 3*num_source, H', W']

        # Extract the current source image
        src_image = curr_src_image_stack[:, i*3:(i+1)*3, :, :]  # [B, 3, H', W']

        # Warp the source image to the target frame
        warped_image = projective_inverse_warp(src_image, curr_depth, pred_poses[:, i, :], intrinsics)  # [B, 3, H', W']

        # Compute photometric loss (L1 + SSIM)
        l1_loss = F.l1_loss(warped_image, curr_tgt_image, reduction='mean')  # Scalar
        ssim_loss = 1 - ssim_metric(warped_image, curr_tgt_image)  # [B]
        ssim_loss = ssim_loss.mean()  # Scalar
        photometric_loss += (0.85 * l1_loss + 0.15 * ssim_loss)  # Scalar

        # Compute smoothness loss for the current scale
        smoothness_loss += compute_smoothness_loss(curr_depth, curr_tgt_image)  # Scalar

    # Combine photometric and smoothness loss
    total_loss = photometric_loss + smooth_weight * smoothness_loss  # Scalar

    return total_loss, photometric_loss, smoothness_loss

# def compute_ssim(img1, img2):
#     """
#     Computes the Structural Similarity (SSIM) index between two images.

#     Args:
#         img1: First image tensor (shape: [B, 3, H, W]).
#         img2: Second image tensor (shape: [B, 3, H, W]).

#     Returns:
#         ssim_map: SSIM map (Tensor [B, 1, H, W]).
#     """
#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2

#     mu1 = torch.nn.functional.avg_pool2d(img1, 3, 1, padding=1)
#     mu2 = torch.nn.functional.avg_pool2d(img2, 3, 1, padding=1)

#     sigma1 = torch.nn.functional.avg_pool2d(img1 * img1, 3, 1, padding=1) - mu1 ** 2
#     sigma2 = torch.nn.functional.avg_pool2d(img2 * img2, 3, 1, padding=1) - mu2 ** 2
#     sigma12 = torch.nn.functional.avg_pool2d(img1 * img2, 3, 1, padding=1) - mu1 * mu2

#     ssim_num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
#     ssim_den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)

#     ssim_map = ssim_num / ssim_den
#     return ssim_map.clamp(0, 1)