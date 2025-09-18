import torch
import numpy as np
from pathlib import Path
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
    mask_directory = Path(f"{mask_path}/{int(image_idx):05d}")
    if not mask_directory.exists():
        mask_directory = Path(f"{mask_path}/frame{int(image_idx):06d}")
    mask_files = mask_directory.glob("*.png")
    masks = []
    for mask_file in mask_files:
        mask = np.array(Image.open(mask_file).convert("L"))
        masks.append(mask)
    masks = np.stack(masks, axis=0)  # [num_masks, H, W]
    masks = torch.from_numpy(masks)  # convert to torch tensor

    return masks
