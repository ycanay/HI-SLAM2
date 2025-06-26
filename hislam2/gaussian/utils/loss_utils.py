#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from math import exp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from util.utils import ele_multip_in_chunks


def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l1_loss_weight(network_output, gt):
    image = gt.detach().cpu().numpy().transpose((1, 2, 0))
    rgb_raw_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    sobelx = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_merge = np.sqrt(sobelx * sobelx + sobely * sobely) + 1e-10
    sobel_merge = np.exp(sobel_merge)
    sobel_merge /= np.max(sobel_merge)
    sobel_merge = torch.from_numpy(sobel_merge)[None, ...].to(gt.device)

    return torch.abs((network_output - gt) * sobel_merge).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def separation_loss(feat_mean_stack):
    """ inter-mask contrastive loss Eq.(2) in the paper
    Constrain the instance features within different masks to be as far apart as possible.
    """
    N, _ = feat_mean_stack.shape

    # expand feat_mean_stack[N, 6] to [N, N, 6]
    feat_expanded = feat_mean_stack.unsqueeze(1).expand(-1, N, -1)
    feat_transposed = feat_mean_stack.unsqueeze(0).expand(N, -1, -1)
    
    # distance
    diff_squared = (feat_expanded - feat_transposed).pow(2).sum(2)
    
    # Calculate the inverse of the distance to enhance discrimination
    epsilon = 1     # 1e-6
    inverse_distance = 1.0 / (diff_squared + epsilon)
    # Exclude diagonal elements (distance from itself) and calculate the mean inverse distance
    mask = torch.eye(N, device=feat_mean_stack.device).bool()
    inverse_distance.masked_fill_(mask, 0)  

    # note: weight
    # sorted by distance
    sorted_indices = inverse_distance.argsort().argsort()
    loss_weight = (sorted_indices.float() / (N - 1)) * (1.0 - 0.1) + 0.1    # scale to 0.1 - 1.0, [N, N]
    # small weight
    inverse_distance *= loss_weight     # [N, N]

    # final loss
    loss = inverse_distance.sum() / (N * (N - 1))

    return loss

def cohesion_loss(gt_masks, feat_map, feature_mean):
    """
    Computes cohesion loss using externally provided per-mask mean features.
    
    Args:
        gt_masks (Tensor): [num_masks, H1, W1] boolean instance masks
        feat_map (Tensor): [6, H, W] feature map with 6 channels
        feature_mean (Tensor): [num_masks, 6] mean features for each mask

    Returns:
        Tensor: cohesion loss
    """
    num_masks, mask_h, mask_w = gt_masks.shape
    C, H, W = feat_map.shape
    assert C == 6, "Feature map must have 6 channels."

    # Resize masks if needed
    if (mask_h, mask_w) != (H, W):
        gt_masks = torch.nn.functional.interpolate(gt_masks.unsqueeze(1).float(), size=(H, W), mode='nearest')
        gt_masks = gt_masks.squeeze(1)  # [num_masks, H, W]

    # Expand tensors
    feat_exp = feat_map.unsqueeze(0).expand(num_masks, -1, -1, -1)       # [num_masks, 6, H, W]
    mask_exp = gt_masks.unsqueeze(1).expand(-1, C, -1, -1)               # [num_masks, 6, H, W]

    # Apply masks to features
    masked_feats = ele_multip_in_chunks(feat_exp, mask_exp, 2)

    # Expand mean feature per mask to spatial layout
    mean_feats = feature_mean.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
    mean_feats_masked = ele_multip_in_chunks(mean_feats, mask_exp, 2)

    # Mean L1 loss between masked features and masked means
    l1 = (torch.abs(masked_feats - mean_feats_masked)).mean()
    return l1