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


def filter_overlapping_masks(
    masks: dict[int, list[torch.Tensor]],
    threshold: float = 0.8,
) -> dict[int, list[torch.Tensor]]:
    """Reject masks that are nearly fully contained inside a larger mask.

    For every pair of instance masks (across all classes), compute
    ``intersection / min(area_a, area_b)``. When the ratio exceeds
    *threshold*, drop the smaller mask of the pair; the larger mask is
    always kept. Ties on area drop the later index, making the result
    deterministic.

    The returned dict preserves the input keys and per-list lengths so
    that downstream code which uses the metadata's ``instance`` field as
    a list index keeps working unchanged. Rejected slots are replaced
    with zero-filled tensors of the same shape; ``resolve_sam_masks_conflicts``
    skips them naturally (``active.any()`` is False) and
    ``sam3_dict_to_tensor`` drops them at the stacking step.

    Args:
        masks: Instance masks keyed by label id, each value a list of
            per-instance binary tensors of identical ``[H, W]`` shape.
        threshold: Containment ratio above which the smaller mask is
            rejected. Must be in ``(0.0, 1.0]``.

    Returns:
        New masks dict with the same structure as the input, where
        rejected instance slots are zero tensors.
    """
    if masks is None or len(masks) == 0:
        return masks

    flat: list[tuple[int, int]] = []
    bool_masks: list[torch.Tensor] = []
    areas: list[int] = []
    for label, instances in masks.items():
        for inst_idx, mask in enumerate(instances):
            bm = mask > 0
            flat.append((label, inst_idx))
            bool_masks.append(bm)
            areas.append(int(bm.sum().item()))

    n = len(flat)
    keep = [True] * n
    for i in range(n):
        if not keep[i] or areas[i] == 0:
            continue
        for j in range(i + 1, n):
            if not keep[j] or areas[j] == 0:
                continue
            intersection = int((bool_masks[i] & bool_masks[j]).sum().item())
            if intersection == 0:
                continue
            min_area = min(areas[i], areas[j])
            if intersection / min_area > threshold:
                if areas[i] <= areas[j]:
                    keep[i] = False
                    break
                else:
                    keep[j] = False

    keep_lookup = {flat[k]: keep[k] for k in range(n)}
    result: dict[int, list[torch.Tensor]] = {}
    for label, instances in masks.items():
        new_list = []
        for inst_idx, mask in enumerate(instances):
            if keep_lookup.get((label, inst_idx), True):
                new_list.append(mask)
            else:
                new_list.append(torch.zeros_like(mask))
        result[label] = new_list
    return result


def _read_metadata(image_idx: int, mask_path: str) -> list[dict]:
    """Read and normalise masks.json into ``[{label, instance, confidence}]``.

    Supports both supported JSON formats (``item_id``/``instance_no`` and
    ``semantic_id``/``instance_id``).  Confidence defaults to ``0.0`` when
    missing from the JSON entry.

    Args:
        image_idx: Frame index used to locate the masks.json metadata file.
        mask_path: Root directory containing per-frame mask subdirectories.

    Returns:
        List of dicts with keys ``label`` (int), ``instance`` (int),
        ``confidence`` (float) in the order given by masks.json.
    """
    mask_directory = Path(f"{mask_path}/frame{int(image_idx):06d}")
    json_file = mask_directory / "masks.json"
    with open(json_file, 'r') as f:
        raw = json.load(f)
    normalised = []
    for item in raw:
        if 'item_id' in item:
            label = int(item['item_id'])
            instance = int(item['instance_no'])
        else:
            label = int(item['semantic_id'])
            instance = int(item['instance_id'])
        confidence = float(item.get('confidence', 0.0))
        normalised.append({
            'label': label,
            'instance': instance,
            'confidence': confidence,
        })
    return normalised


def _build_semantic_image(
    masks: dict[int, list[torch.Tensor]],
    metadata: list[dict],
    hierarchy: dict[int, int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a pixel-wise single-class semantic image via metadata-order fill.

    Iterates metadata in order.  At every pixel covered by the current
    instance mask, pick a winner using these rules:

      - Empty pixel (``-1``): assign the new label.
      - Same class already present, ``new_confidence > stored``: overwrite.
      - Different class already present and ``hierarchy[existing] == new_label``
        (existing is a direct child of the new label): overwrite.
      - Otherwise: keep the existing assignment (first-come-first-served for
        unrelated classes; parents keep against newer children).

    Args:
        masks: Instance masks keyed by label id.
        metadata: Entries from :func:`_read_metadata`, in file order.
        hierarchy: Optional ``{child_id: parent_id}`` mapping.

    Returns:
        semantic_image: ``[H, W]`` int64 tensor of the winning label per
            pixel (``-1`` where no instance covers the pixel).
        confidence_image: ``[H, W]`` float32 tensor holding the confidence
            of the winning instance at each pixel.
    """
    any_mask = next(iter(masks.values()))[0]
    semantic_image = -1 * torch.ones_like(any_mask, dtype=torch.int64)
    confidence_image = torch.zeros_like(any_mask, dtype=torch.float32)

    children_of: dict[int, list[int]] = {}
    if hierarchy:
        for child, parent in hierarchy.items():
            children_of.setdefault(parent, []).append(child)

    for item in metadata:
        label = item['label']
        instance = item['instance']
        confidence = item['confidence']
        if label not in masks:
            continue
        if instance < 0 or instance >= len(masks[label]):
            continue
        mask = masks[label][instance]
        active = mask > 0
        if not active.any():
            continue

        empty_assign = active & (semantic_image == -1)
        same_class_better = (
            active & (semantic_image == label) & (confidence_image < confidence)
        )

        parent_overwrite = torch.zeros_like(active)
        child_labels = children_of.get(label)
        if child_labels:
            children_tensor = torch.tensor(
                child_labels,
                dtype=semantic_image.dtype,
                device=semantic_image.device,
            )
            parent_overwrite = active & torch.isin(semantic_image, children_tensor)

        assign = empty_assign | same_class_better | parent_overwrite
        if not assign.any():
            continue
        semantic_image[assign] = label
        confidence_image[assign] = float(confidence)

    return semantic_image, confidence_image


def sam_masks_semantic_image(
    masks: dict[int, list[torch.Tensor]],
    image_idx: int,
    mask_path: str,
    hierarchy: dict[int, int] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Build per-class binary masks with strictly pairwise hierarchy.

    For each class the binary mask is the union of its instance masks.  The
    hierarchy is then applied pairwise: for every registered
    ``hierarchy[child] == parent`` pair, the child loses pixels that are
    also claimed by the parent class (parent overwrites child at overlap).

    Intentional properties of this design:
      - **Non-conflicting pixels are passed through unchanged.**  A pixel
        covered by a single class's masks is always in that class's output.
      - **Unrelated different-class overlaps are preserved** — the output
        is a per-class binary mask, so two unrelated classes can both claim
        the same pixel.  Only pairs explicitly registered in ``hierarchy``
        cause a class to lose pixels.
      - **No transitive chains.**  A grandparent does not implicitly beat a
        grandchild; only direct pairs matter.

    Args:
        masks: Instance masks keyed by label id, each value a list of per-instance tensors.
        image_idx: Frame index (unused; kept for API compatibility).
        mask_path: Root directory (unused; kept for API compatibility).
        hierarchy: Optional mapping ``{child_id: parent_id}`` loaded from
            ``hierarchy.json``.  Applied strictly pairwise.

    Returns:
        semantic_masks: Binary masks stacked as ``[num_classes, H, W]``,
            one per class present in *masks*.  May overlap across slices.
            *None* when no mask covers any pixel.
        mask_ids: 1-D tensor of label ids corresponding to each slice in
            *semantic_masks*, or *None*.
    """
    del image_idx, mask_path
    if masks is None or len(masks) == 0:
        return None, None

    class_masks: dict[int, torch.Tensor] = {}
    for label, instances in masks.items():
        if not instances:
            continue
        merged = torch.zeros_like(instances[0], dtype=torch.bool)
        for inst in instances:
            merged |= inst > 0
        class_masks[label] = merged

    if hierarchy is not None:
        for child_label, parent_label in hierarchy.items():
            if child_label in class_masks and parent_label in class_masks:
                class_masks[child_label] = class_masks[child_label] & ~class_masks[parent_label]

    valid = sorted(
        ((lbl, m) for lbl, m in class_masks.items() if m.any()),
        key=lambda x: x[0],
    )
    if not valid:
        return None, None

    semantic_masks = torch.stack([m.to(torch.uint8) for _, m in valid], dim=0)
    semantic_ids = torch.tensor([lbl for lbl, _ in valid], dtype=torch.int64)
    return semantic_masks, semantic_ids


def resolve_sam_masks_conflicts(
    masks: dict[int, list[torch.Tensor]],
    image_idx: int,
    mask_path: str,
    hierarchy: dict[int, int] | None = None,
) -> dict[int, list[torch.Tensor]]:
    """Resolve pixel-level conflicts at the instance level.

    Produces a new masks dict whose binary instance masks are mutually
    non-overlapping.  Conflict rules mirror :func:`sam_masks_semantic_image`:

    1. Cross-class conflicts are resolved by the pairwise hierarchy from
       ``sam_masks_semantic_image`` — a direct parent overwrites its direct
       child only.  Pixels where a different class wins the semantic vote
       are removed from this instance's mask.
    2. Among the instances whose class won the semantic vote at a pixel,
       the instance with strictly higher ``confidence`` keeps the pixel.
    3. First-come-first-served for same-class equal-confidence conflicts.

    Instances referenced in ``masks.json`` but missing from ``masks`` (or
    with an out-of-range instance index) are skipped.  Instances present in
    ``masks`` but absent from ``masks.json`` are returned untouched at the
    end of their label's list, which preserves the input shape for
    ``sam3_dict_to_tensor``.

    Args:
        masks: Instance masks keyed by label id, each value a list of
            per-instance binary tensors.
        image_idx: Frame index used to locate the masks.json metadata file.
        mask_path: Root directory containing per-frame mask subdirectories.
        hierarchy: Optional mapping ``{child_id: parent_id}``.

    Returns:
        A new masks dict with the same keys and list lengths as the input,
        whose tensors are binary (``uint8``) and non-overlapping across
        instances and classes.
    """
    if masks is None or len(masks) == 0:
        return masks

    metadata = _read_metadata(image_idx, mask_path)
    semantic_image, _ = _build_semantic_image(masks, metadata, hierarchy)

    any_mask = next(iter(masks.values()))[0]
    instance_image = -1 * torch.ones_like(any_mask, dtype=torch.int64)
    inst_confidence_image = torch.zeros_like(any_mask, dtype=torch.float32)

    flat_to_key: dict[int, tuple[int, int]] = {}
    for flat_idx, item in enumerate(metadata):
        label = item['label']
        instance = item['instance']
        confidence = item['confidence']
        if label not in masks:
            continue
        if instance < 0 or instance >= len(masks[label]):
            continue
        mask = masks[label][instance]
        active = (mask > 0) & (semantic_image == label)

        empty_assign = active & (instance_image == -1)
        conf_overwrite = active & (inst_confidence_image < confidence)
        assign = empty_assign | conf_overwrite

        instance_image[assign] = flat_idx
        inst_confidence_image[assign] = float(confidence)
        flat_to_key[flat_idx] = (label, instance)

    resolved: dict[int, list[torch.Tensor]] = {
        label: [m.clone() for m in insts] for label, insts in masks.items()
    }
    for flat_idx, (label, instance) in flat_to_key.items():
        original = masks[label][instance]
        won = instance_image == flat_idx
        lost = (original > 0) & ~won
        if lost.any():
            cleaned = original.clone()
            cleaned[lost] = 0
            resolved[label][instance] = cleaned
    return resolved


def sam3_dict_to_tensor(masks: dict[int, list[torch.Tensor]]) -> torch.Tensor:
    """Convert a SAM3 masks dict into a single stacked tensor.

    Empty (all-zero) instance masks are dropped at this step so that
    downstream losses never receive zero-area masks. Empty masks can
    arise either from upstream rejection in
    :func:`filter_overlapping_masks` or from full pixel removal in
    :func:`resolve_sam_masks_conflicts`.

    Args:
        masks: Instance masks keyed by label id, each value a list of
            ``[H, W]`` binary tensors.

    Returns:
        Stacked binary masks of shape ``[total_num_kept_masks, H, W]``,
        or an empty tensor when no non-empty mask is present.
    """
    all_mask_tensors = []
    for label_masks in masks.values():
        for mask in label_masks:
            if (mask > 0).any():
                all_mask_tensors.append(mask)
    if all_mask_tensors:
        return torch.stack(all_mask_tensors, dim=0)
    return torch.empty((0, 0, 0), dtype=torch.uint8)
