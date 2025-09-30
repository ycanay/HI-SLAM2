import torch
import numpy as np
from pathlib import Path
import cv2
from PIL import Image


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
