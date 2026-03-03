import torch
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import json


def read_gt_masks(image_idx, mask_path):
    """
    Read the SAM masks from a .png file.
    Args:
        image_idx (int): the index of the image.
        mask_path (str): path to the all masks directory.
    Returns:
        masks (torch.Tensor): the masks in shape of [num_masks, H, W].
    """
    mask_directory = Path(f"{mask_path}/frame{int(image_idx):06d}")
    mask_files = sorted(list(mask_directory.glob("*.png")))
    masks = []
    for mask_file in mask_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.array(Image.open(mask_file).convert("L"))
        masks.append(mask)

    if masks:
        masks = np.stack(masks, axis=0, dtype=np.uint8)  # [num_masks, H, W]
        masks = torch.from_numpy(masks)
    else:
        masks = torch.empty((0, 0, 0), dtype=torch.uint8)

    return masks


def read_sam3_masks(image_idx: int, mask_path: str) -> dict[int, list[torch.Tensor]]:
    """
    Read the SAM3 masks from a .png file.
    Args:
        image_idx (int): the index of the image.
        mask_path (str): path to the all masks directory.
    Returns:
        (masks (dict[int -> list[torch.Tensor]]): the masks in shape of [num_masks, H, W] for each label.
    """
    mask_directory = Path(f"{mask_path}/frame{int(image_idx):06d}")
    mask_files = sorted(list(mask_directory.glob("*.png")))
    masks = {}
    for mask_file in mask_files:
        # Assuming filename format is '{class_id}_{instance_no}.png'
        label = int(mask_file.stem.split('_')[0])
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.array(Image.open(mask_file).convert("L"))
        mask_tensor = torch.from_numpy(mask.astype(np.uint8))
        if label not in masks:
            masks[label] = [mask_tensor]
        else:
            masks[label].append(mask_tensor)
    return masks


def sam_masks_semantic_image(masks: dict[int, list[torch.Tensor]], image_idx: int, mask_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert SAM masks to a semantic segmentation image.
    Args:
        masks (dict: int -> list[torch.Tensor]): the masks in shape of [num_masks, H, W] for each label.
        image_idx (int): the index of the image.
        mask_path (str): path to the all masks directory.
    Returns:
        semantic_masks (torch.Tensor): the semantic segmentation image in shape of [num_semantic_masks, H, W].
        mask_ids (torch.Tensor): the list of mask ids corresponding to the semantic segmentation image shape of [num_semantic_masks].
    """
    if masks is None or len(masks) == 0:
        return None, None
    semantic_image = -1 * \
        torch.ones_like(masks[list(masks.keys())[0]][0], dtype=torch.int64)
    semantic_masks = []
    mask_directory = Path(f"{mask_path}/frame{int(image_idx):06d}")
    json_file = mask_directory / "masks.json"
    with open(json_file, 'r') as f:
        metadata = json.load(f)
    for item in metadata:
        label = item['item_id']
        instance = item['instance_no']
        mask = masks[label][instance]
        semantic_image[mask > 0] = int(label)
    mask_ids = list(set(semantic_image.flatten().tolist()))
    mask_ids.remove(-1)
    mask_ids = torch.tensor(mask_ids, dtype=torch.int64)
    for mask_id in mask_ids:
        semantic_masks.append((semantic_image == mask_id).to(torch.uint8))
    semantic_masks = torch.stack(semantic_masks, dim=0)
    return semantic_masks, mask_ids


def sam3_dict_to_tensor(masks: dict[int, list[torch.Tensor]]) -> torch.Tensor:
    """
    Convert SAM3 masks dictionary to a tensor.
    Args:
        masks (dict: int -> list[torch.Tensor]): the masks in shape of [num_masks, H, W] for each label.
    Returns:
        all_masks (torch.Tensor): the masks in shape of [total_num_masks, H, W].
    """
    all_masks = []
    for label_masks in masks.values():
        all_masks.extend(label_masks)
    if all_masks:
        all_masks = torch.stack(all_masks, dim=0)
    else:
        all_masks = torch.empty((0, 0, 0), dtype=torch.uint8)
    return all_masks
