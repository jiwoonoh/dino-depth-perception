from __future__ import division
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
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

def pixel_to_3d(points, intrinsics):
    """
    Converts pixel coordinates and depth to 3D coordinates.

    Args:
        points: Tensor of shape [B, N, 3], representing (u, v, w).
        intrinsics: Camera intrinsics tensor of shape [B, 3, 3].

    Returns:
        Tensor of shape [B, N, 3], representing 3D coordinates.
    """
    fx = intrinsics[:, 0, 0].unsqueeze(1)  # [B, 1]
    fy = intrinsics[:, 1, 1].unsqueeze(1)  # [B, 1]
    cx = intrinsics[:, 0, 2].unsqueeze(1)  # [B, 1]
    cy = intrinsics[:, 1, 2].unsqueeze(1)  # [B, 1]

    u = points[:, :, 0]  # [B, N]
    v = points[:, :, 1]  # [B, N]
    w = points[:, :, 2]  # [B, N]

    x = ((u - cx) * w) / fx  # [B, N]
    y = ((v - cy) * w) / fy  # [B, N]
    z = w  # Depth remains unchanged

    return torch.stack((x, y, z), dim=2)  # [B, N, 3]

    
    return torch.stack((x, y, z), dim=2)

def _3d_to_pixel(points_3d, intrinsics):
    """
    Converts 3D coordinates to pixel coordinates and depth.

    Args:
        points_3d: Tensor of shape [B, N, 3], where B is the batch size, N is the number of points,
                   and each point is represented as (x, y, z).
        intrinsics: Camera intrinsics tensor of shape [B, 3, 3].

    Returns:
        Tensor of shape [B, N, 3] representing the pixel coordinates (u, v) and depth (w).
    """
    # Extract intrinsics components
    fx = intrinsics[:, 0, 0].unsqueeze(1)  # [B, 1]
    fy = intrinsics[:, 1, 1].unsqueeze(1)  # [B, 1]
    cx = intrinsics[:, 0, 2].unsqueeze(1)  # [B, 1]
    cy = intrinsics[:, 1, 2].unsqueeze(1)  # [B, 1]

    # Extract 3D points
    x = points_3d[:, :, 0]  # [B, N]
    y = points_3d[:, :, 1]  # [B, N]
    z = points_3d[:, :, 2]  # [B, N]

    # Prevent division by zero
    z = torch.clamp(z, min=1e-6)

    # Compute pixel coordinates
    u = (x * fx) / z + cx  # [B, N]
    v = (y * fy) / z + cy  # [B, N]
    w = z  # Depth remains unchanged

    return torch.stack((u, v, w), dim=2)  # [B, N, 3]

    
def projective_inverse_warp(src_image, depth, pose, intrinsics):
    """
    Warps the source image to the target frame using depth and pose.

    Args:
        src_image: Source image tensor (shape: [B, H, W, 3]).
        depth: Depth map for the target view (shape: [B, H, W]).
        pose: 6-DoF pose parameters (shape: [B, 6]).
        intrinsics: Camera intrinsics matrix (shape: [B, 3, 3]).

    Returns:
        warped_image: Source image warped to the target frame (shape: [B, H, W, 3]).
    """
    batch_size, img_height, img_width, _ = src_image.shape

    # Step 1: Create pixel grid
    u, v = torch.meshgrid(torch.arange(0, img_width, device=src_image.device),
                          torch.arange(0, img_height, device=src_image.device))
    u = u.flatten().float()
    v = v.flatten().float()
    pixel_coords = torch.stack([u, v, torch.ones_like(u)], dim=1)  # [HW, 3]
    pixel_coords = pixel_coords.unsqueeze(0).expand(batch_size, -1, -1)  # [B, HW, 3]

    # Step 2: Backproject pixels to 3D space
    cam_coords = pixel_to_3d(pixel_coords, intrinsics)  # [B, HW, 3]
    cam_coords = cam_coords * depth.view(batch_size, -1, 1)  # Scale by depth

    # Step 3: Apply 6-DoF transformation
    cam_coords_transformed = step_cloud(cam_coords, pose)  # [B, HW, 3]

    # Step 4: Reproject to 2D space
    pixel_coords_proj = _3d_to_pixel(cam_coords_transformed, intrinsics)  # [B, HW, 3]
    u_proj = pixel_coords_proj[:, :, 0].view(batch_size, img_height, img_width)
    v_proj = pixel_coords_proj[:, :, 1].view(batch_size, img_height, img_width)

    # Step 5: Sample from source image
    grid = torch.stack([u_proj / img_width * 2 - 1, v_proj / img_height * 2 - 1], dim=-1)  # [B, H, W, 2]
    warped_image = torch.nn.functional.grid_sample(src_image, grid, align_corners=False)

    return warped_image


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

def compute_loss(pred_depth, pred_poses, tgt_image, src_image_stack, intrinsics, 
                smooth_weight=0.001, num_source=None, use_ssim=True, ssim_metric=None):
    """
    Computes photometric loss, smoothness loss, and total loss.

    Args:
        pred_depth (List[Tensor]): List of predicted depth maps for different scales 
                                   (List of tensors [B, 1, H, W]).
        pred_poses (Tensor): Predicted 6-DoF poses for source frames 
                             (Tensor [B, num_source, 6] or [B, 6]).
        tgt_image (Tensor): Target image tensor (shape: [B, 3, H, W]).
        src_image_stack (Tensor): Source image stack tensor (shape: [B, 3*num_source, H, W]).
        intrinsics (Tensor): Camera intrinsics matrix (shape: [B, 3, 3]).
        smooth_weight (float, optional): Weight for the smoothness loss. Defaults to 0.01.
        num_source (int, optional): Number of source images. If None, inferred from pred_poses. Defaults to None.
        use_ssim (bool, optional): Whether to include SSIM in photometric loss. Defaults to True.
        ssim_metric (Metric, optional): Initialized SSIM metric from torchmetrics.image. 
                                        If None and use_ssim is True, it will be initialized inside.

    Returns:
        total_loss (Tensor): Combined loss.
        photometric_loss (Tensor): Photometric loss.
        smoothness_loss (Tensor): Smoothness loss.
    """
    photometric_loss = 0.0
    smoothness_loss = 0.0

    # Determine num_source based on pred_poses shape if not provided
    if num_source is None:
        if pred_poses.dim() == 3:
            num_source = pred_poses.shape[1]
        elif pred_poses.dim() == 2:
            num_source = 1
        else:
            raise ValueError(f"pred_poses must be a 2D or 3D tensor, but got {pred_poses.dim()}D tensor.")

    # Initialize SSIM metric if needed
    if use_ssim:
        if ssim_metric is None:
            device = tgt_image.device
            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Iterate over each scale
    for s in range(len(pred_depth)):
        curr_depth = pred_depth[s]  # [B, 1, H, W]

        # Print current depth shape
        # print(f"Scale {s}:")
        # print(f"  curr_depth shape: {curr_depth.shape}")  # Should be [B, 1, H, W]

        # Check if depth_map[s] is correctly batched
        if curr_depth.dim() != 4 or curr_depth.shape[1] != 1:
            raise ValueError(f"pred_depth[{s}] has incorrect shape: {curr_depth.shape}. Expected [B, 1, H, W].")

        # Resize images for the current scale
        scale_factor = 1 / (2 ** s)
        curr_tgt_image = F.interpolate(
            tgt_image, scale_factor=scale_factor, mode='bilinear', align_corners=False
        )  # [B, 3, H', W']
        curr_src_image_stack = F.interpolate(
            src_image_stack, scale_factor=scale_factor, mode='bilinear', align_corners=False
        )  # [B, 3*num_source, H', W']

        # Print shapes to verify alignment
        # print(f"  curr_tgt_image shape: {curr_tgt_image.shape}")
        # print(f"  curr_src_image_stack shape: {curr_src_image_stack.shape}")

        # Check if depth matches the image dimensions
        if curr_depth.shape[2:] != curr_tgt_image.shape[2:]:
            # print(f"  Resizing curr_depth from {curr_depth.shape[2:]} to {curr_tgt_image.shape[2:]}")
            curr_depth = F.interpolate(
                curr_depth,
                size=curr_tgt_image.shape[2:],
                mode='bilinear',
                align_corners=False
            )  # [B, 1, H', W']
            # print(f"  Resized curr_depth shape: {curr_depth.shape}")

        # Iterate over each source
        for i in range(num_source):
            if num_source == 1:
                # For single source, src_image_stack has shape [B, 3, H', W']
                src_image = curr_src_image_stack[:, :3, :, :]  # [B, 3, H', W']
                # For single source, pred_poses has shape [B, 6]
                pose = pred_poses if pred_poses.dim() == 2 else pred_poses[:, i, :]  # [B, 6]
            else:
                # For multiple sources, src_image_stack has shape [B, 3*num_source, H', W']
                src_image = curr_src_image_stack[:, i*3:(i+1)*3, :, :]  # [B, 3, H', W']
                # pred_poses has shape [B, num_source, 6]
                pose = pred_poses[:, i, :]  # [B, 6]

            # Debug: Print pose shape
            # print(f"  Source {i}: pose shape: {pose.shape}")

            # Warp the source image to the target frame
            warped_image = projective_inverse_warp(src_image, curr_depth.squeeze(1), pose, intrinsics)  # [B, 3, H', W']

            # Compute photometric loss (L1 + SSIM if enabled)
            l1_loss = F.l1_loss(warped_image, curr_tgt_image, reduction='mean')  # Scalar

            if use_ssim:
                ssim_loss = 1 - ssim_metric(warped_image, curr_tgt_image)  # [B]
                ssim_loss = ssim_loss.mean()  # Scalar
                photometric_loss += (0.85 * l1_loss + 0.15 * ssim_loss)  # Scalar
            else:
                photometric_loss += l1_loss  # Scalar

        # Compute smoothness loss for the current scale
        smoothness_loss += compute_smoothness_loss(curr_depth, curr_tgt_image)  # Scalar

    # Combine photometric and smoothness loss
    total_loss = photometric_loss + smooth_weight * smoothness_loss  # Scalar

    return photometric_loss #, photometric_loss, smoothness_loss
