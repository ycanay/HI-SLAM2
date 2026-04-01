from transformers import pipeline
from transformers.utils import logging as hf_logging
from PIL import Image
import numpy as np
import torch
import cv2
from hislam2.gaussian.utils.camera_utils import Camera
from pathlib import Path
import json
# Option 2: only filter that specific warning
hf_logging.set_verbosity_error()


class MaskGenerator:
    def __init__(self, config, save_dir):
        save_dir = Path(save_dir)
        masks_cfg = config.get("masks", {})
        default_output = save_dir / "masks" / "mask2former"
        self.output_masks_path = Path(
            masks_cfg.get("mask2former_masks_dir", default_output)
        )
        self.output_masks_path.mkdir(parents=True, exist_ok=True)

        model_name = masks_cfg.get(
            "mask2former_network",
            masks_cfg.get(
                "network", "facebook/mask2former-swin-large-coco-panoptic"),
        )
        use_cuda = torch.cuda.is_available()
        self.segmenter = pipeline(
            task="image-segmentation",
            model=model_name,
            dtype=torch.float16 if use_cuda else torch.float32,
            device=0 if use_cuda else -1,
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
        if segmentation_results is None:
            segmentation_results = []
        elif isinstance(segmentation_results, dict):
            segmentation_results = [segmentation_results]
        frame_dir = self.output_masks_path / \
            f"frame{int(viewpoint.tstamp):06d}"
        if (frame_dir / "masks.json").exists():
            sam_masks, _instance_ids = self.read_masks(viewpoint, "instance")
            sem_masks, sem_ids = self.read_masks(viewpoint, "semantic")
            return sam_masks, sem_masks, sem_ids

        frame_dir.mkdir(parents=True, exist_ok=True)
        class_instance_counter = {}
        masks_json = []
        all_instance_masks = []

        for result in segmentation_results:
            if not isinstance(result, dict):
                continue

            label_name = result.get("label")
            mask_value = result.get("mask")
            if label_name is None or mask_value is None:
                continue

            if self.label2id is not None and label_name in self.label2id:
                label_id = int(self.label2id[label_name])
            else:
                try:
                    label_id = int(label_name)
                except (TypeError, ValueError):
                    continue

            mask = np.array(mask_value)
            binary_mask = (mask > 0).astype(np.uint8) * 255
            if binary_mask.max() == 0:
                continue

            instance_id = class_instance_counter.get(label_id, 0)
            class_instance_counter[label_id] = instance_id + 1
            mask_name = f"{label_id}_{instance_id}.png"
            cv2.imwrite(str(frame_dir / mask_name), binary_mask)
            all_instance_masks.append(binary_mask > 0)

            masks_json.append(
                {
                    "semantic_id": int(label_id),
                    "instance_id": int(instance_id),
                }
            )

        with open(frame_dir / "masks.json", "w") as f:
            json.dump(masks_json, f, indent=2)

        if not all_instance_masks:
            empty_masks = torch.empty((0, 0, 0), dtype=torch.uint8)
            return empty_masks, None, None

        sam_masks = torch.from_numpy(
            np.stack(all_instance_masks, axis=0).astype(np.uint8))
        sem_masks, sem_ids = self.read_masks(viewpoint, "semantic")
        return sam_masks, sem_masks, sem_ids

    def read_masks(self, viewpoint, type):
        if type in ["instance", "semantic"]:
            mask_path = self.output_masks_path
        else:
            raise ValueError("type must be 'instance' or 'semantic'")

        frame_id = f"frame{int(viewpoint.tstamp):06d}"
        mask_directory = mask_path / frame_id
        if not mask_directory.exists():
            mask_directory = mask_path / f"{int(viewpoint.tstamp):06d}"

        mask_files = sorted(mask_directory.glob("*.png"))

        if not mask_files:
            return None, None

        first_mask = np.array(Image.open(mask_files[0]).convert("L"))
        num_masks = len(mask_files)
        masks = np.empty((num_masks, *first_mask.shape), dtype=np.uint8)
        instance_label_ids = np.empty(num_masks, dtype=np.int64)

        masks[0] = first_mask
        instance_label_ids[0] = int(mask_files[0].stem.split("_")[0])

        for i, mask_file in enumerate(mask_files[1:], start=1):
            masks[i] = np.array(Image.open(mask_file).convert("L"))
            instance_label_ids[i] = int(mask_file.stem.split("_")[0])

        masks = torch.from_numpy(masks)
        instance_label_ids = torch.from_numpy(instance_label_ids)

        if type == "instance":
            return masks, instance_label_ids

        semantic_masks = []
        semantic_ids = torch.unique(instance_label_ids)
        for semantic_id in semantic_ids:
            semantic_masks.append(
                (masks[instance_label_ids == semantic_id] > 0).any(
                    dim=0).to(torch.uint8)
            )
        semantic_masks = torch.stack(semantic_masks, dim=0)

        return semantic_masks, semantic_ids
