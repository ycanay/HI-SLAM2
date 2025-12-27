from transformers import pipeline
from transformers.utils import logging as hf_logging
from PIL import Image
import numpy as np
import torch
from PIL import Image
import cv2
from scipy import ndimage
from hislam2.gaussian.utils.camera_utils import Camera
from pathlib import Path
import json
# Option 2: only filter that specific warning
hf_logging.set_verbosity_error()


class MaskGenerator:
    def __init__(self, config, save_dir):
        save_dir = Path(save_dir)
        save_path = save_dir/'masks'
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        self.semantic_masks_path = save_path/'semantic'
        if not self.semantic_masks_path.exists():
            self.semantic_masks_path.mkdir(parents=True, exist_ok=True)
        self.segmenter = pipeline(
            task="image-segmentation",
            model=config["masks"]["network"],
            dtype=torch.float16,
            device=0,
        )
        self.model = self.segmenter.model
        # Check the model's configuration
        self.config = self.model.config
        self.id2label = self.config.id2label if hasattr(
            self.config, 'id2label') else None
        self.label2id = self.config.label2id if hasattr(
            self.config, 'label2id') else None

    def generate_and_save_masks(self, viewpoint: Camera):
        image = viewpoint.original_image
        semantic_dict = {}

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()  # ensure on CPU
            if image.ndim == 3:  # C,H,W
                image = image.permute(1, 2, 0)  # H,W,C
            image = image.numpy()
            # scale to 0-255 if float
            if image.dtype != 'uint8':
                image = (image * 255).clip(0, 255).astype('uint8')
            image = Image.fromarray(image)
        segmentation_results = self.segmenter(image)
        semantic_masks_path = self.semantic_masks_path / \
            f"frame{int(viewpoint.tstamp):06d}"
        if semantic_masks_path.exists():
            return
        semantic_masks_path.mkdir(parents=True, exist_ok=True)
        semantic_count = 0
        for result in segmentation_results:
            if (semantic_masks_path / f"{semantic_count:03d}.png").exists():
                break
            mask = np.array(result['mask'])
            label = result['label']
            semantic_dict[f"{semantic_count:03d}.png"] = {"label": label,
                                                          "id": self.label2id[label] if self.label2id is not None else None}
            semantic_mask = mask.copy()
            if semantic_mask.dtype == bool:
                semantic_segmentation = semantic_mask.astype("uint8") * 255
            elif semantic_mask.dtype != "uint8":
                semantic_segmentation = semantic_mask.astype("uint8")
            else:
                semantic_segmentation = semantic_mask
            cv2.imwrite(str(semantic_masks_path /
                        f"{semantic_count:03d}.png"), semantic_segmentation)
            semantic_count += 1

        with open(semantic_masks_path / "metadata.json", 'w') as f:
            json.dump(semantic_dict, f, indent=4)

    def read_masks(self, viewpoint, type):
        if type == 'instance':
            mask_path = self.semantic_masks_path  # Not in use
        elif type == 'semantic':
            mask_path = self.semantic_masks_path
        else:
            raise ValueError("type must be 'instance' or 'semantic'")

        frame_id = f"frame{int(viewpoint.tstamp):06d}"
        mask_directory = mask_path / frame_id
        if not mask_directory.exists():
            mask_directory = mask_path / f"{int(viewpoint.tstamp):06d}"

        metadata_file = mask_directory / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        mask_files = sorted(mask_directory.glob("*.png"))

        if not mask_files:
            return None, None

        first_mask = np.array(Image.open(mask_files[0]).convert("L"))
        num_masks = len(mask_files)
        masks = np.empty((num_masks, *first_mask.shape), dtype=np.uint8)
        mask_ids = np.empty(num_masks, dtype=np.int64)

        masks[0] = first_mask
        mask_ids[0] = metadata[mask_files[0].name]['id']

        for i, mask_file in enumerate(mask_files[1:], start=1):
            masks[i] = np.array(Image.open(mask_file).convert("L"))
            mask_ids[i] = metadata[mask_file.name]['id']

        # Convert to torch tensors
        masks = torch.from_numpy(masks)
        mask_ids = torch.from_numpy(mask_ids)

        return masks, mask_ids
