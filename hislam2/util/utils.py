import yaml
import numpy as np
import rich
import copy
import torch
from matplotlib import cm
from pathlib import Path
from PIL import Image
import math

_log_styles = {
    "GSBackend": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
    "PGBA": "bold blue",
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="GSBackend"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)


def load_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from per scene/dataset cfg.
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    inherit_from = cfg_special.get("inherit_from")

    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # merge per dataset cfg. and main cfg.
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively. dict1 get masked by dict2, and we retuen dict1.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def colorize_np(x, cmap_name='jet', range=None):
    if range is not None:
        vmin, vmax = range
    else:
        vmin, vmax = np.percentile(x, (1, 99))

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]
    return x_new


def clone_obj(obj):
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        # check if its a property
        if hasattr(clone_obj.__class__, attr) and isinstance(
            getattr(clone_obj.__class__, attr), property
        ):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj


def mask_feature_mean(feat_map, gt_masks):
    """Compute the average instance features within each mask.
    feat_map: [C=D, H, W]         the instance features of the entire image
    gt_masks: [num_mask, mask_h, mask_w]  num_mask boolean masks
    """
    num_mask, mask_h, mask_w = gt_masks.shape
    C, H, W = feat_map.shape

    # Resize masks to match feature map size using nearest neighbor interpolation
    if (mask_h, mask_w) != (H, W):
        # [num_mask, 1, mask_h, mask_w]
        gt_masks = gt_masks.unsqueeze(1).float()
        gt_masks = torch.nn.functional.interpolate(
            gt_masks, size=(H, W), mode='nearest')
        gt_masks = gt_masks.squeeze(1)  # [num_mask, H, W]

    # expand feat and masks for batch processing
    feat_expanded = feat_map.unsqueeze(0).expand(
        num_mask, *feat_map.shape)  # [num_mask, D, H, W]
    masks_expanded = gt_masks.unsqueeze(
        1).expand(-1, feat_map.shape[0], -1, -1)  # [num_mask, D, H, W]
    masked_feats = ele_multip_in_chunks(
        feat_expanded, masks_expanded, chunk_size=2)   # in chuck to avoid OOM
    mask_counts = masks_expanded.sum(dim=(2, 3))  # [num_mask, D]

    # the number of pixels within each mask
    mask_counts = mask_counts.clamp(min=1)

    # the mean features of each mask
    sum_per_channel = masked_feats.sum(dim=[2, 3])
    mean_per_channel = sum_per_channel / mask_counts    # [num_mask, D]

    return mean_per_channel   # [num_mask, D]


def ele_multip_in_chunks(feat_expanded, masks_expanded, chunk_size=5):
    result = torch.zeros_like(feat_expanded)
    for i in range(0, feat_expanded.size(0), chunk_size):
        end_i = min(i + chunk_size, feat_expanded.size(0))
        for j in range(0, feat_expanded.size(1), chunk_size):
            end_j = min(j + chunk_size, feat_expanded.size(1))
            chunk_feat = feat_expanded[i:end_i, j:end_j]
            chunk_mask = masks_expanded[i:end_i, j:end_j].float()

            result[i:end_i, j:end_j] = chunk_feat * chunk_mask
    return result


def distinct_colors(K: int) -> list[torch.Tensor]:
    color_div = math.ceil(K ** (1/3))
    steps = 256 // color_div
    colors = []
    last_color = torch.tensor([-steps, 0, 0])
    for _ in range(K):
        r = (last_color[0] + steps)
        g = last_color[1]
        b = last_color[2]
        if r >= 256:
            r = 0
            g = (last_color[1] + steps)
            if g >= 256:
                g = 0
                b = (last_color[2] + steps)
        colors.append(torch.tensor([r, g, b]))
        last_color = torch.tensor([r, g, b])
    return colors
