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


def sam_masks_semantic_image(
    masks: dict[int, list[torch.Tensor]],
    image_idx: int,
    mask_path: str,
    hierarchy: dict[int, int] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Convert SAM masks to a semantic segmentation image.

    When *hierarchy* is provided, label conflicts at overlapping pixels are
    resolved pairwise: a new label overwrites an existing one only when the
    new label is the *direct* child of the existing label in the hierarchy
    (i.e. ``hierarchy[new_label] == existing_label``).  Multi-hop chains are
    intentionally ignored — only immediate parent-child pairs are considered.
    Unrelated conflicts keep the first-assigned label.  Without a hierarchy
    the last mask in metadata order wins (original behaviour).

    Args:
        masks: Instance masks keyed by label id, each value a list of per-instance tensors.
        image_idx: Frame index used to locate the masks.json metadata file.
        mask_path: Root directory containing per-frame mask subdirectories.
        hierarchy: Optional mapping ``{child_id: parent_id}`` loaded from
            ``hierarchy.json``.  Keys and values are integer label ids.

    Returns:
        semantic_masks: Binary masks stacked as ``[num_semantic_masks, H, W]``,
            or *None* when no valid masks exist.
        mask_ids: 1-D tensor of label ids corresponding to each slice in
            *semantic_masks*, or *None*.
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
        if 'item_id' in item:
            label = item['item_id']
            instance = item['instance_no']
        else:
            label = item['semantic_id']
            instance = item['instance_id']
        if label not in masks:
            continue
        if instance < 0 or instance >= len(masks[label]):
            continue
        mask = masks[label][instance]
        active = mask > 0
        if hierarchy is not None:
            assign = active & (semantic_image == -1)
            direct_parent = hierarchy.get(label)
            if direct_parent is not None:
                assign = assign | (active & (semantic_image == direct_parent))
            semantic_image[assign] = int(label)
        else:
            semantic_image[active] = int(label)

    semantic_ids = torch.unique(semantic_image)
    semantic_ids = semantic_ids[semantic_ids >= 0].to(torch.int64)
    if semantic_ids.numel() == 0:
        return None, None

    for mask_id in semantic_ids.tolist():
        semantic_masks.append((semantic_image == mask_id).to(torch.uint8))
    semantic_masks = torch.stack(semantic_masks, dim=0)
    return semantic_masks, semantic_ids


def sam3_dict_to_tensor(masks: dict[int, list[torch.Tensor]]) -> torch.Tensor:
    """
    Convert SAM3 masks dictionary to a tensor.
    Args:
        masks (dict: int -> list[torch.Tensor]): the masks in shape of [num_masks, H, W] for each label.
    Returns:
        all_masks (torch.Tensor): the masks in shape of [total_num_masks, H, W].
    """
    all_mask_tensors = []
    for label_masks in masks.values():
        all_mask_tensors.extend(label_masks)
    if all_mask_tensors:
        return torch.stack(all_mask_tensors, dim=0)
    return torch.empty((0, 0, 0), dtype=torch.uint8)
