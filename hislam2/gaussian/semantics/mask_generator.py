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
        self.instance_masks_path = save_path/'instance'
        self.semantic_masks_path = save_path/'semantic'
        if not self.instance_masks_path.exists():
            self.instance_masks_path.mkdir(parents=True, exist_ok=True)
        if not self.semantic_masks_path.exists():
            self.semantic_masks_path.mkdir(parents=True, exist_ok=True)
        self.segmenter = pipeline(
            task="image-segmentation",
            model=config["masks"]["network"],
            dtype=torch.float16,
            device=0,
        )
        model = self.segmenter.model
        # Check the model's configuration
        config = model.config
        self.id2label = config.id2label if hasattr(
            config, 'id2label') else None
        self.label2id = config.label2id if hasattr(
            config, 'label2id') else None

    def split_connected_components(self, mask):
        """
        Splits connected components from a binary mask into separate masks.

        Args:
            mask (np.ndarray): Binary mask (0s and 1s).

        Returns:
            list of np.ndarray: Each element is a binary mask of one connected component.
        """
        # Label connected components
        labeled_mask, num_components = ndimage.label(mask)

        # Extract individual masks
        component_masks = [(labeled_mask == i).astype(np.uint8)
                           for i in range(1, num_components + 1)]

        return component_masks

    def generate_and_save_masks(self, viewpoint: Camera):
        image = viewpoint.original_image
        semantic_dict = {}
        instance_dict = {}

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
        instance_masks_path = self.instance_masks_path / \
            f"frame{int(viewpoint.tstamp):06d}"
        semantic_masks_path = self.semantic_masks_path / \
            f"frame{int(viewpoint.tstamp):06d}"
        if semantic_masks_path.exists() and instance_masks_path.exists():
            return
        semantic_masks_path.mkdir(parents=True, exist_ok=True)
        instance_masks_path.mkdir(parents=True, exist_ok=True)
        semantic_count = 0
        instance_count = 0
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
            components = self.split_connected_components(mask)
            for idx, comp in enumerate(components):
                # check if number of true pixels is less than 100, skip
                if comp.sum() < 100:
                    continue
                instance_dict[f"{instance_count:03d}.png"] = {"label": label,
                                                              "id": self.label2id[label] if self.label2id is not None else None,
                                                              "component": idx}

                mask_copy = comp.copy()
                if mask_copy.dtype == bool:
                    instance_segmentation = mask_copy.astype("uint8") * 255
                elif mask_copy.dtype != "uint8":
                    instance_segmentation = mask_copy.astype("uint8")
                else:
                    instance_segmentation = mask_copy
                cv2.imwrite(str(instance_masks_path /
                            f"{instance_count:03d}.png"), instance_segmentation)
                instance_count += 1

        with open(semantic_masks_path / "metadata.json", 'w') as f:
            json.dump(semantic_dict, f, indent=4)
        with open(instance_masks_path / "metadata.json", 'w') as f:
            json.dump(instance_dict, f, indent=4)

    def read_masks(self, viewpoint, type):
        if type == 'instance':
            mask_path = self.instance_masks_path
        elif type == 'semantic':
            mask_path = self.semantic_masks_path
        else:
            raise ValueError("type must be 'instance' or 'semantic'")
        mask_directory = Path(f"{mask_path}/{int(viewpoint.tstamp):06d}")
        if not mask_directory.exists():
            mask_directory = Path(
                f"{mask_path}/frame{int(viewpoint.tstamp):06d}")
        metadata_file = mask_directory / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        mask_files = mask_directory.glob("*.png")
        mask_files = sorted(list(mask_files))
        masks = []
        mask_ids = []
        for mask_file in mask_files:
            mask = np.array(Image.open(mask_file).convert("L"))
            masks.append(mask)
            mask_ids.append(metadata[mask_file.stem + ".png"]['id'])
        masks = np.stack(masks, axis=0)  # [num_masks, H, W]
        masks = torch.from_numpy(masks)  # convert to torch tensor
        mask_ids = np.stack(mask_ids, axis=0)
        mask_ids = torch.from_numpy(mask_ids)

        return masks, mask_ids
