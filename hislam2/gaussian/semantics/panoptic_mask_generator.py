from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
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
        save_path.mkdir(parents=True, exist_ok=True)
        self.instance_masks_path = save_path/'instance'
        self.semantic_masks_path = save_path/'semantic'
        self.instance_masks_path.mkdir(parents=True, exist_ok=True)
        self.semantic_masks_path.mkdir(parents=True, exist_ok=True)

        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/mask2former-swin-large-ade-panoptic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-large-ade-panoptic")
        config = self.model.config
        self.id2label = config.id2label
        self.label2id = config.label2id

        # Move model to GPU if available for faster inference
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

    def generate_and_save_masks(self, viewpoint: Camera):
        image = viewpoint.original_image

        # Early exit check - moved before processing
        frame_id = f"frame{int(viewpoint.tstamp):06d}"
        instance_masks_path = self.instance_masks_path / frame_id
        semantic_masks_path = self.semantic_masks_path / frame_id

        if semantic_masks_path.exists() and instance_masks_path.exists():
            return

        # Optimized image conversion
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
            if image.ndim == 3:  # C,H,W
                image = image.permute(1, 2, 0)  # H,W,C
            image = image.numpy()
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Move inputs to device
        inputs = self.processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        prediction = self.processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0]

        # Create directories
        semantic_masks_path.mkdir(parents=True, exist_ok=True)
        instance_masks_path.mkdir(parents=True, exist_ok=True)

        # Get segmentation once
        segmentation = prediction['segmentation'].cpu().numpy()
        segments_info = prediction['segments_info']

        # Pre-allocate dictionaries
        instance_dict = {}
        semantic_dict = {}
        semantic_masks = {}

        # Single loop for both instance and semantic processing
        for instance_count, segment in enumerate(segments_info):
            segment_id = segment['id']
            segment_mask = (segmentation == segment_id)
            segment_label_id = segment['label_id']
            segment_label = self.id2label[segment_label_id]

            # Process instance mask
            instance_segmentation = segment_mask.astype(np.uint8) * 255
            instance_filename = f"{instance_count:03d}.png"
            cv2.imwrite(str(instance_masks_path / instance_filename),
                        instance_segmentation)
            instance_dict[instance_filename] = {
                "label": segment_label,
                "id": self.label2id[segment_label],
                "component": 0
            }

            # Accumulate semantic masks
            if segment_label in semantic_masks:
                semantic_masks[segment_label] |= segment_mask
            else:
                semantic_masks[segment_label] = segment_mask.copy()

        # Write semantic masks
        for semantic_count, (segment_label, segment_mask) in enumerate(semantic_masks.items()):
            semantic_segmentation = segment_mask.astype(np.uint8) * 255
            semantic_filename = f"{semantic_count:03d}.png"
            cv2.imwrite(str(semantic_masks_path / semantic_filename),
                        semantic_segmentation)
            semantic_dict[semantic_filename] = {
                "label": segment_label,
                "id": self.label2id[segment_label]
            }

        # Write metadata files
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

        frame_id = f"frame{int(viewpoint.tstamp):06d}"
        mask_directory = mask_path / frame_id
        if not mask_directory.exists():
            mask_directory = mask_path / f"{int(viewpoint.tstamp):06d}"

        metadata_file = mask_directory / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Get sorted list of mask files
        mask_files = sorted(mask_directory.glob("*.png"))

        if not mask_files:
            return None, None

        # Pre-allocate numpy arrays for better performance
        first_mask = np.array(Image.open(mask_files[0]).convert("L"))
        num_masks = len(mask_files)
        masks = np.empty((num_masks, *first_mask.shape), dtype=np.uint8)
        mask_ids = np.empty(num_masks, dtype=np.int64)

        # First mask already loaded
        masks[0] = first_mask
        mask_ids[0] = metadata[mask_files[0].name]['id']

        # Load remaining masks
        for i, mask_file in enumerate(mask_files[1:], start=1):
            masks[i] = np.array(Image.open(mask_file).convert("L"))
            mask_ids[i] = metadata[mask_file.name]['id']

        # Convert to torch tensors
        masks = torch.from_numpy(masks)
        mask_ids = torch.from_numpy(mask_ids)

        return masks, mask_ids
