import json
import os
import colorsys
import cv2
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.detection.panoptic_qualities import PanopticQuality
from hislam2.util.utils import Log
from hislam2.gaussian.renderer import render
from hislam2.gaussian.utils.loss_utils import ssim, psnr
from hislam2.gaussian.utils.camera_utils import Camera
from tqdm import tqdm
import json


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

    image_save_dir = f'{save_dir}/renders/image_{iteration}'
    ins_feat_save_dir = f'{save_dir}/renders/ins_feat_{iteration}'
    depth_save_dir = f'{save_dir}/renders/depth_{iteration}'
    cluster_save_dir = f'{save_dir}/renders/cluster_{iteration}'
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(ins_feat_save_dir, exist_ok=True)
    os.makedirs(depth_save_dir, exist_ok=True)
    # os.makedirs(vis_save_dir, exist_ok=True)
    os.makedirs(cluster_save_dir, exist_ok=True)
    json_compatible_features = {}
    for key, value in gaussians.cluster_features.items():
        json_key = str(key)  # Convert key to string
        json_value = {}

        for inner_key, inner_value in value.items():
            if isinstance(inner_value, torch.Tensor):
                # Convert tensor to list
                json_value[inner_key] = inner_value.cpu().numpy().tolist()
            elif hasattr(inner_value, 'item'):
                # Convert numpy scalars to Python types
                json_value[inner_key] = inner_value.item()
            else:
                json_value[inner_key] = inner_value

        json_compatible_features[json_key] = json_value

    # Save the JSON-compatible version
    json.dump(json_compatible_features, open(os.path.join(
        save_dir, "cluster_features.json"), "w", encoding="utf-8"), indent=4)

    for i, (idx, image) in tqdm(enumerate(gtimages.items()), total=len(gtimages), desc=f"Eval {iteration}"):
        if idx % 5 != 0 and idx not in kf_idx and i != len(gtimages) - 1:
            continue
        frame = Camera.init_from_tracking(
            image.squeeze()/255.0, None, None, traj[idx], idx, projection_matrix, K)
        gtimage = frame.original_image.cuda()

        rendering = render(frame, gaussians, background, empty_ins_feats)
        image = torch.clamp(rendering["render"], 0.0, 1.0)
        ins_feat = torch.clamp(rendering["rendered_features"], 0.0, 1.0)
        ins_feat_1 = ins_feat[0:3]
        ins_feat_2 = ins_feat[3:6]
        depth = rendering["depth"].detach().squeeze().cpu().numpy()
        ins_feat_original = rendering["rendered_features"]

        C, H, W = ins_feat_original.shape
        ins_feat_flat = ins_feat_original.permute(
            1, 2, 0).reshape(-1, C)  # [H*W, C]

        cluster_labels = list(gaussians.cluster_features.keys())
        mean_feats = torch.stack(
            [gaussians.cluster_features[l]["mean_feature"]
                for l in cluster_labels]
        )  # [K, C]
        diff = ins_feat_flat[:, None, :] - mean_feats[None, :, :]
        dist = diff.pow(2).sum(dim=-1)  # [H*W, K]
        closest_idx = dist.argmin(dim=1)
        pixel_clusters = torch.tensor(
            [cluster_labels[i] for i in closest_idx])  # [H*W]
        instance_ids = pixel_clusters.reshape(H, W)  # [H, W]

        if gtdepthdir is not None:
            # 1000.
            gtdepth = cv2.imread(os.path.join(
                gtdepthdir, gtdepths[idx]), cv2.IMREAD_ANYDEPTH) / 6553.5
            gtdepth = cv2.resize(
                gtdepth, (depth.shape[-1], depth.shape[-2]), interpolation=cv2.INTER_NEAREST)
            invalid = gtdepth <= 0
            depth[invalid] = 0

        pred = (image.detach().cpu().numpy().transpose(
            (1, 2, 0)) * 255).astype(np.uint8)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        pred_feat_1 = (ins_feat_1.detach().cpu().numpy(
        ).transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred_feat_2 = (ins_feat_2.detach().cpu().numpy(
        ).transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred_feat_1 = cv2.cvtColor(pred_feat_1, cv2.COLOR_BGR2RGB)
        pred_feat_2 = cv2.cvtColor(pred_feat_2, cv2.COLOR_BGR2RGB)
        instance_ids_img = instance_ids.cpu().numpy().astype(np.uint8)
        instance_ids_img = cv2.cvtColor(instance_ids_img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f'{image_save_dir}/{idx:06d}.jpg', pred)
        cv2.imwrite(f'{depth_save_dir}/{idx:06d}.png',
                    np.clip(depth*6553.5, 0, 65535).astype(np.uint16))
        cv2.imwrite(f'{ins_feat_save_dir}/{idx:06d}_1.png', pred_feat_1)
        cv2.imwrite(f'{ins_feat_save_dir}/{idx:06d}_2.png', pred_feat_2)
        cv2.imwrite(f'{cluster_save_dir}/{idx:06d}.png', instance_ids_img)
        # vis = np.concatenate((pred, cv2.imread(f'{save_dir}/renders/depth_{iteration}/{idx:06d}.png')), axis=0)
        # cv2.imwrite(f'{vis_save_dir}/{idx:06d}.jpg', vis)

        if gtdepthdir is not None and idx in kf_idx:
            l1_array.append(
                np.abs(gtdepth[depth > 0] - depth[depth > 0]).mean().item())

        # if idx in kf_idx:
        #     continue
        mask = gtimage > 0
        psnr_score = psnr((image[mask]).unsqueeze(0),
                          (gtimage[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gtimage).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gtimage).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["mean_l1"] = float(np.mean(l1_array)) if l1_array else 0

    Log(f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}, depth l1: {output["mean_l1"]}', tag="Eval")

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    os.makedirs(psnr_save_dir, exist_ok=True)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"),
             "w", encoding="utf-8"),
        indent=4,
    )
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
        image = (torch.exp(frame.exposure_a)) * \
            rendering["render"] + frame.exposure_b
        image = torch.clamp(image, 0.0, 1.0)

        mask = gtimage > 0
        psnr_score = psnr((image[mask]).unsqueeze(0),
                          (gtimage[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gtimage).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gtimage).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(f'kf mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}', tag="Eval")

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    os.makedirs(psnr_save_dir, exist_ok=True)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result_kf.json"),
             "w", encoding="utf-8"),
        indent=4,
    )
    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    print('saved to ', point_cloud_path)


def distinct_colors(K):
    color_map = []
    for i in range(K):
        h = i / K
        # alternate between high/low saturation
        s = 0.6 + 0.4 * ((i % 2))
        v = 0.7 + 0.3 * ((i // 2) % 2)  # alternate brightness
        rgb = [int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v)]
        color_map.append(torch.tensor(rgb, dtype=torch.uint8))
    return color_map


def create_mappings():
    with open("/storage/user/ayu/repos/HI-SLAM2/data/Replica_semantics/mappings.json", "r") as f:
        data = f.read()
    classes = json.loads(data)["classes"]
    mappings = {}
    for class_id in classes:
        mappings[class_id['id']] = class_id['mapping']['idx']
    return mappings
