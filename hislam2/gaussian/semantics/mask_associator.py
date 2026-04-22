import torch
import torch.nn.functional as F
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hislam2.gaussian.utils.camera_utils import Camera


def warp_masks(
    masks_src: Tensor,
    depth_src: Tensor,
    view_src: "Camera",
    depth_tgt: Tensor,
    view_tgt: "Camera",
    *,
    occlusion_rel_tol: float = 0.1,
    min_depth: float = 0.2,
) -> Tensor:
    """Warp source masks into the target view using depth unprojection.

    Unprojects every source pixel to a world-frame 3D point using *depth_src*
    and the source camera pose, then projects those points into the target
    view.  Pixels whose reprojected depth disagrees with *depth_tgt* by more
    than *occlusion_rel_tol* (relative) are treated as occluded and excluded.
    The resulting binary masks indicate which target pixels are covered by
    each source mask after warping.

    Args:
        masks_src: ``[N, H_src, W_src]`` bool or uint8 source masks.
        depth_src: ``[H_src, W_src]`` or ``[1, H_src, W_src]`` depth in metres.
        view_src: Camera object for the source view (provides R, T, fx, fy, cx, cy).
        depth_tgt: ``[H_tgt, W_tgt]`` or ``[1, H_tgt, W_tgt]`` depth in metres.
        view_tgt: Camera object for the target view.
        occlusion_rel_tol: Relative depth tolerance |dz|/z for occlusion filtering.
        min_depth: Points closer than this (in either view) are discarded.

    Returns:
        ``[N, H_tgt, W_tgt]`` bool tensor on CUDA.
    """
    H_src = view_src.image_height
    W_src = view_src.image_width
    H_tgt = view_tgt.image_height
    W_tgt = view_tgt.image_width
    device = "cuda"

    depth_src = depth_src.squeeze().to(device=device, dtype=torch.float32)  # [H_src, W_src]
    depth_tgt = depth_tgt.squeeze().to(device=device, dtype=torch.float32)  # [H_tgt, W_tgt]

    # Build per-pixel direction vectors in source camera frame
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H_src, device=device, dtype=torch.float32),
        torch.arange(W_src, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # [H_src*W_src, 3]
    pixels = torch.stack(
        [grid_x.reshape(-1), grid_y.reshape(-1), torch.ones(H_src * W_src, device=device, dtype=torch.float32)],
        dim=1,
    )

    intrins_src = torch.tensor(
        [[view_src.fx, 0.0, view_src.cx],
         [0.0, view_src.fy, view_src.cy],
         [0.0, 0.0, 1.0]],
        device=device, dtype=torch.float32,
    )

    # Camera-to-world transform for source view
    c2w_src = view_src.world_view_transform.T.inverse()  # [4, 4]

    # Unproject source pixels to world-frame 3D points
    rays_d = pixels @ intrins_src.inverse().T @ c2w_src[:3, :3].T  # [H*W, 3]
    rays_o = c2w_src[:3, 3]                                          # [3]
    pts_world = depth_src.reshape(-1, 1) * rays_d + rays_o           # [H*W, 3]

    # Transform world points into target camera frame (world-to-camera: R, T)
    R_tgt = view_tgt.R.to(device=device, dtype=torch.float32)  # [3, 3]
    T_tgt = view_tgt.T.to(device=device, dtype=torch.float32)  # [3]
    pts_cam_tgt = (R_tgt @ pts_world.T + T_tgt[:, None]).T      # [H*W, 3]

    Z = pts_cam_tgt[:, 2]  # [H*W] — depth in target camera space

    # Project to target pixel coords
    u = view_tgt.fx * pts_cam_tgt[:, 0] / Z.clamp(min=1e-8) + view_tgt.cx  # [H*W]
    v = view_tgt.fy * pts_cam_tgt[:, 1] / Z.clamp(min=1e-8) + view_tgt.cy  # [H*W]

    # Validity: positive depth and within target image bounds
    valid = (Z > min_depth) & (u >= 0) & (u < W_tgt) & (v >= 0) & (v < H_tgt)

    # Occlusion check via grid_sample on depth_tgt
    u_norm = (2.0 * u / max(W_tgt - 1, 1) - 1.0)  # [H*W]
    v_norm = (2.0 * v / max(H_tgt - 1, 1) - 1.0)  # [H*W]
    grid = torch.stack([u_norm, v_norm], dim=1).unsqueeze(0).unsqueeze(2)  # [1, H*W, 1, 2]
    sampled_depth = F.grid_sample(
        depth_tgt.unsqueeze(0).unsqueeze(0),  # [1, 1, H_tgt, W_tgt]
        grid,
        mode="nearest",
        align_corners=False,
        padding_mode="zeros",
    ).squeeze()  # [H*W]

    depth_diff_rel = (Z - sampled_depth).abs() / Z.clamp(min=1e-8)
    valid = valid & (sampled_depth > min_depth) & (depth_diff_rel < occlusion_rel_tol)

    if not valid.any():
        return torch.zeros(masks_src.shape[0], H_tgt, W_tgt, device=device, dtype=torch.bool)

    # Scatter valid src-pixel mask labels into target image coords
    u_int = u[valid].long().clamp(0, W_tgt - 1)  # [K]
    v_int = v[valid].long().clamp(0, H_tgt - 1)  # [K]
    src_idx = torch.where(valid)[0]               # [K] indices into H_src*W_src

    N = masks_src.shape[0]
    masks_src_gpu = masks_src.bool().to(device=device)  # [N, H_mask, W_mask]
    if masks_src_gpu.shape[1:] != (H_src, W_src):
        masks_src_gpu = torch.nn.functional.interpolate(
            masks_src_gpu.unsqueeze(1).float(), size=(H_src, W_src), mode="nearest"
        ).squeeze(1).bool()
    masks_src_flat = masks_src_gpu.reshape(N, H_src * W_src)  # [N, H_src*W_src]
    masks_at_valid = masks_src_flat[:, src_idx]  # [N, K] — which masks cover each valid point

    tgt_flat = (v_int * W_tgt + u_int).unsqueeze(0).expand(N, -1)  # [N, K]
    warped_flat = torch.zeros(N, H_tgt * W_tgt, device=device, dtype=torch.float32)
    warped_flat.scatter_add_(1, tgt_flat, masks_at_valid.float())

    return (warped_flat > 0).reshape(N, H_tgt, W_tgt)


def masks_iou(warped: Tensor, target: Tensor) -> Tensor:
    """Compute pairwise IoU between two mask sets on the same grid.

    Args:
        warped: ``[N_src, H, W]`` bool tensor — warped source masks.
        target: ``[N_tgt, H, W]`` bool tensor — target masks.

    Returns:
        ``[N_src, N_tgt]`` float tensor of IoU values.
    """
    N_src = warped.shape[0]
    N_tgt = target.shape[0]

    warped_flat = warped.reshape(N_src, -1).float()   # [N_src, H*W]
    target_flat = target.reshape(N_tgt, -1).float()   # [N_tgt, H*W]

    intersection = warped_flat @ target_flat.T                         # [N_src, N_tgt]
    union = warped_flat.sum(1, keepdim=True) + target_flat.sum(1, keepdim=True).T - intersection

    return intersection / union.clamp(min=1.0)


def associate_masks(
    masks_src: Tensor,
    depth_src: Tensor,
    view_src: "Camera",
    masks_tgt: Tensor,
    depth_tgt: Tensor,
    view_tgt: "Camera",
    *,
    iou_threshold: float = 0.33,
    occlusion_rel_tol: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Associate source masks with target masks via depth-based warping.

    Warps *masks_src* into the target view, computes the pairwise IoU matrix,
    and returns one-to-one greedy pairs with ``IoU >= iou_threshold``.  The
    greedy pass processes source masks in descending order of their best IoU
    score so that the highest-confidence associations are resolved first.

    Args:
        masks_src: ``[N_src, H, W]`` source masks (bool or uint8).
        depth_src: Source view depth map.
        view_src: Source camera.
        masks_tgt: ``[N_tgt, H, W]`` target masks (bool or uint8).
        depth_tgt: Target view depth map.
        view_tgt: Target camera.
        iou_threshold: Minimum IoU to accept a pair.
        occlusion_rel_tol: Forwarded to :func:`warp_masks`.

    Returns:
        Tuple of:
        - ``[K, 2]`` int64 tensor of ``(src_idx, tgt_idx)`` pairs.
        - ``[K]`` float tensor of the IoU score for each pair.
        Both are empty when no pairs qualify.
    """
    empty_pairs = torch.empty(0, 2, dtype=torch.int64, device="cuda")
    empty_ious = torch.empty(0, dtype=torch.float32, device="cuda")

    if masks_src.shape[0] == 0 or masks_tgt.shape[0] == 0:
        return empty_pairs, empty_ious

    warped = warp_masks(
        masks_src, depth_src, view_src, depth_tgt, view_tgt,
        occlusion_rel_tol=occlusion_rel_tol,
    )

    # Resize masks_tgt to the target camera resolution so it matches warped
    masks_tgt_gpu = masks_tgt.bool().to("cuda")
    H_tgt, W_tgt = warped.shape[1], warped.shape[2]
    if masks_tgt_gpu.shape[1:] != (H_tgt, W_tgt):
        masks_tgt_gpu = torch.nn.functional.interpolate(
            masks_tgt_gpu.unsqueeze(1).float(), size=(H_tgt, W_tgt), mode="nearest"
        ).squeeze(1).bool()
    iou_matrix = masks_iou(warped, masks_tgt_gpu)  # [N_src, N_tgt]

    max_ious, _ = iou_matrix.max(dim=1)
    src_order = max_ious.argsort(descending=True).tolist()

    pairs: list[list[int]] = []
    ious: list[float] = []
    used_tgt: set[int] = set()

    for src_i in src_order:
        row = iou_matrix[src_i]
        for tgt_j in row.argsort(descending=True).tolist():
            if tgt_j in used_tgt:
                continue
            if row[tgt_j].item() >= iou_threshold:
                pairs.append([src_i, tgt_j])
                ious.append(row[tgt_j].item())
                used_tgt.add(tgt_j)
            break  # sorted descending — no later candidate can beat this

    if not pairs:
        return empty_pairs, empty_ious

    return (
        torch.tensor(pairs, dtype=torch.int64, device="cuda"),
        torch.tensor(ious, dtype=torch.float32, device="cuda"),
    )
