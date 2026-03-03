import json
import os
import cv2
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from hislam2.util.utils import Log, distinct_colors
from hislam2.gaussian.renderer import render
from hislam2.gaussian.utils.loss_utils import ssim, psnr
from hislam2.gaussian.utils.camera_utils import Camera
from tqdm import tqdm
import math


def _save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _tensor_to_json(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy().tolist()
    elif hasattr(value, 'item'):
        return value.item()
    return value


def _convert_cluster_features(cluster_features):
    return {
        str(key): {
            inner_key: _tensor_to_json(inner_value)
            for inner_key, inner_value in value.items()
        }
        for key, value in cluster_features.items()
    }


def _compute_instance_ids(ins_feat_original, cluster_features):
    C, H, W = ins_feat_original.shape
    ins_feat_flat = ins_feat_original.permute(1, 2, 0).reshape(-1, C)

    cluster_labels = list(cluster_features.keys())
    mean_feats = torch.stack([
        cluster_features[label]["mean_feature"] for label in cluster_labels
    ])

    dist = (ins_feat_flat[:, None, :] -
            mean_feats[None, :, :]).pow(2).sum(dim=-1)
    closest_idx = dist.argmin(dim=1)
    pixel_clusters = torch.tensor([cluster_labels[i] for i in closest_idx])

    return pixel_clusters.reshape(H, W)


def _save_rendering_outputs(idx, image, ins_feat, depth, instance_ids, save_dirs):
    pred = (image.detach().cpu().numpy().transpose(
        1, 2, 0) * 255).astype(np.uint8)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

    ins_feat_cpu = ins_feat.detach().cpu().numpy()
    pred_feat_1 = cv2.cvtColor((ins_feat_cpu[0:3].transpose(
        1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    pred_feat_2 = cv2.cvtColor((ins_feat_cpu[3:6].transpose(
        1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    unique_ids = torch.unique(instance_ids)
    K = len(unique_ids)
    colors = distinct_colors(K)

    # Create a color mapping for each instance ID
    instance_ids_img = cv2.cvtColor(
        instance_ids.cpu().numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)

    cv2.imwrite(f'{save_dirs["image"]}/{idx:06d}.jpg', pred)
    cv2.imwrite(f'{save_dirs["depth"]}/{idx:06d}.png',
                np.clip(depth * 6553.5, 0, 65535).astype(np.uint16))
    cv2.imwrite(f'{save_dirs["ins_feat"]}/{idx:06d}_1.png', pred_feat_1)
    cv2.imwrite(f'{save_dirs["ins_feat"]}/{idx:06d}_2.png', pred_feat_2)
    cv2.imwrite(f'{save_dirs["cluster"]}/{idx:06d}.png', instance_ids_img)


def eval_rendering(
    gtimages,
    gtdepthdir,
    traj,
    gaussians,
    save_dir,
    background,
    empty_ins_feats,
    projection_matrix,
    K,
    kf_idx,
    iteration="final",
):
    gtdepths = sorted(os.listdir(gtdepthdir)
                      ) if gtdepthdir is not None else None
    psnr_array, ssim_array, lpips_array, l1_array = [], [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True).to("cuda")

    save_dirs = {
        "image": f'{save_dir}/renders/image_{iteration}',
        "ins_feat": f'{save_dir}/renders/ins_feat_{iteration}',
        "depth": f'{save_dir}/renders/depth_{iteration}',
        "cluster": f'{save_dir}/renders/cluster_{iteration}',
    }

    for dir_path in save_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    json_compatible_features = _convert_cluster_features(
        gaussians.cluster_features)
    _save_json(json_compatible_features, os.path.join(
        save_dir, "cluster_features.json"))

    for i, (idx, image) in tqdm(enumerate(gtimages.items()), total=len(gtimages), desc=f"Eval {iteration}"):
        if idx % 5 != 0 and idx not in kf_idx and i != len(gtimages) - 1:
            continue

        frame = Camera.init_from_tracking(
            image.squeeze() /
            255.0, None, None, traj[idx], idx, projection_matrix, K
        )
        gtimage = frame.original_image.cuda()

        rendering = render(frame, gaussians, background, empty_ins_feats)
        rendered_image = torch.clamp(rendering["render"], 0.0, 1.0)
        ins_feat = torch.clamp(rendering["rendered_features"], 0.0, 1.0)
        depth = rendering["depth"].detach().squeeze().cpu().numpy()

        instance_ids = _compute_instance_ids(
            rendering["rendered_features"], gaussians.cluster_features)

        if gtdepthdir is not None:
            gtdepth = cv2.imread(
                os.path.join(gtdepthdir, gtdepths[idx]), cv2.IMREAD_ANYDEPTH
            ) / 6553.5
            gtdepth = cv2.resize(
                gtdepth, (depth.shape[-1], depth.shape[-2]
                          ), interpolation=cv2.INTER_NEAREST
            )
            invalid = gtdepth <= 0
            depth[invalid] = 0

            if idx in kf_idx:
                valid_mask = depth > 0
                l1_array.append(
                    np.abs(gtdepth[valid_mask] - depth[valid_mask]).mean())

        _save_rendering_outputs(idx, rendered_image,
                                ins_feat, depth, instance_ids, save_dirs)

        mask = gtimage > 0
        psnr_array.append(psnr(rendered_image[mask].unsqueeze(
            0), gtimage[mask].unsqueeze(0)).item())
        ssim_array.append(ssim(rendered_image.unsqueeze(0),
                          gtimage.unsqueeze(0)).item())
        lpips_array.append(
            cal_lpips(rendered_image.unsqueeze(0), gtimage.unsqueeze(0)).item())

    output = {
        "mean_psnr": float(np.mean(psnr_array)),
        "mean_ssim": float(np.mean(ssim_array)),
        "mean_lpips": float(np.mean(lpips_array)),
        "mean_l1": float(np.mean(l1_array)) if l1_array else 0
    }

    Log(f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, '
        f'lpips: {output["mean_lpips"]}, depth l1: {output["mean_l1"]}', tag="Eval")

    _save_json(output, os.path.join(save_dir, "psnr",
               str(iteration), "final_result.json"))

    return output


def eval_rendering_kf(
    viewpoints,
    gaussians,
    save_dir,
    background,
    empty_ins_feats,
    iteration="final",
):
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True).to("cuda")

    for frame in viewpoints.values():
        gtimage = frame.original_image.cuda()

        rendering = render(frame, gaussians, background, empty_ins_feats)
        image = torch.clamp(
            torch.exp(frame.exposure_a) *
            rendering["render"] + frame.exposure_b,
            0.0, 1.0
        )

        mask = gtimage > 0
        psnr_array.append(psnr(image[mask].unsqueeze(
            0), gtimage[mask].unsqueeze(0)).item())
        ssim_array.append(
            ssim(image.unsqueeze(0), gtimage.unsqueeze(0)).item())
        lpips_array.append(cal_lpips(image.unsqueeze(0),
                           gtimage.unsqueeze(0)).item())

    output = {
        "mean_psnr": float(np.mean(psnr_array)),
        "mean_ssim": float(np.mean(ssim_array)),
        "mean_lpips": float(np.mean(lpips_array))
    }

    Log(f'kf mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, '
        f'lpips: {output["mean_lpips"]}', tag="Eval")

    _save_json(output, os.path.join(save_dir, "psnr",
               str(iteration), "final_result_kf.json"))

    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return

    point_cloud_path = os.path.join(
        name, "point_cloud", "final" if final else f"iteration_{iteration}"
    )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    print('saved to', point_cloud_path)


def create_mappings():
    with open("/storage/user/ayu/repos/HI-SLAM2/data/Replica_semantics/mappings.json", "r") as f:
        data = json.load(f)

    return {class_id['id']: class_id['mapping']['idx'] for class_id in data["classes"]}
