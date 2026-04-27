import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hislam2.gaussian.semantics.mask_reader import (
    read_sam3_masks,
    resolve_sam_masks_conflicts,
)
from hislam2.util.utils import distinct_colors


def discover_frame_indices(masks_dir: Path, rgb_dir: Path) -> list[int]:
    """Return sorted frame indices present in both masks and RGB directories.

    A mask frame directory is expected at ``masks_dir/frame{idx:06d}`` and the
    matching RGB image at ``rgb_dir/frame{idx:06d}.jpg``. Frames missing from
    either location are skipped.

    Args:
        masks_dir: Root directory containing per-frame mask subdirectories.
        rgb_dir: Directory containing the source RGB ``.jpg`` frames.

    Returns:
        Sorted list of integer frame indices that exist on both sides.
    """
    mask_frames = {int(p.name.replace("frame", "")) for p in masks_dir.glob("frame??????") if p.is_dir()}
    rgb_frames = {int(p.stem.replace("frame", "")) for p in rgb_dir.glob("frame??????.jpg")}
    return sorted(mask_frames & rgb_frames)


def load_hierarchy(json_path: Path | None) -> dict[int, int] | None:
    """Load a ``{child_id: parent_id}`` label hierarchy from disk.

    Mirrors the loader in
    :meth:`hislam2.gaussian.semantics.mask_cache.MaskCache._load_hierarchy`
    so this script applies the exact same conflict resolution semantics as
    the main pipeline.

    Args:
        json_path: Path to the hierarchy JSON (string keys, int values), or
            ``None`` to disable hierarchy-based resolution.

    Returns:
        Mapping from child label id to parent label id, or ``None`` when
        ``json_path`` is ``None``.
    """
    if json_path is None:
        return None
    with open(json_path) as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def load_class_names(json_path: Path) -> dict[int, str]:
    """Load a class-id to class-name mapping from a dataset classes JSON file.

    Args:
        json_path: Path to a JSON file containing a list of
            ``{"id": int, "name": str}`` entries.

    Returns:
        Mapping from integer class id to class name string.
    """
    with open(json_path) as f:
        entries = json.load(f)
    return {int(e["id"]): str(e["name"]) for e in entries}


def build_class_palette(class_ids: list[int]) -> dict[int, np.ndarray]:
    """Build a deterministic BGR color per class id using ``distinct_colors``.

    Args:
        class_ids: Sorted unique class ids the palette should cover.

    Returns:
        Mapping from class id to a ``(3,)`` uint8 BGR numpy array.
    """
    colors = distinct_colors(max(len(class_ids), 1))
    palette: dict[int, np.ndarray] = {}
    for cid, rgb in zip(class_ids, colors):
        rgb_np = rgb.cpu().numpy().astype(np.uint8)
        palette[cid] = np.array([rgb_np[2], rgb_np[1], rgb_np[0]], dtype=np.uint8)
    return palette


def _instance_color(class_id: int, instance_no: int) -> np.ndarray:
    """Deterministic, visually distinct BGR color per instance.

    The (class_id, instance_no) pair seeds a RNG, so the same instance gets
    the same color across frames.

    Args:
        class_id: Semantic class id.
        instance_no: Instance index within the class.

    Returns:
        A ``(3,)`` uint8 BGR color array.
    """
    rng = np.random.default_rng(seed=(class_id * 1000003) ^ (instance_no + 1))
    return rng.integers(30, 226, size=3).astype(np.uint8)


def _blend(rgb_bgr: np.ndarray, overlay: np.ndarray, any_mask: np.ndarray, alpha: float) -> np.ndarray:
    """Alpha-blend ``overlay`` onto ``rgb_bgr`` only where ``any_mask`` is set.

    Args:
        rgb_bgr: Source frame as ``[H, W, 3]`` uint8 BGR.
        overlay: Color overlay image of the same shape.
        any_mask: Boolean ``[H, W]`` union-of-masks selector.
        alpha: Mask blend weight in ``[0, 1]``.

    Returns:
        The blended frame, same shape/dtype as ``rgb_bgr``.
    """
    blended = cv2.addWeighted(rgb_bgr, 1.0 - alpha, overlay, alpha, 0.0)
    out = rgb_bgr.copy()
    out[any_mask] = blended[any_mask]
    return out


def overlay_semantic(
    rgb_bgr: np.ndarray,
    masks: dict[int, list[torch.Tensor]],
    palette: dict[int, np.ndarray],
    alpha: float,
) -> tuple[np.ndarray, list[int]]:
    """Paint mask pixels by semantic class color and blend onto the frame.

    Args:
        rgb_bgr: Source frame as ``[H, W, 3]`` uint8 BGR.
        masks: Instance masks keyed by class id.
        palette: Class-to-BGR color mapping.
        alpha: Mask blend weight in ``[0, 1]``.

    Returns:
        Tuple of (overlaid image, sorted list of class ids actually present
        in the frame — i.e. those that contributed any pixel).
    """
    h, w = rgb_bgr.shape[:2]
    overlay = np.zeros_like(rgb_bgr)
    any_mask = np.zeros((h, w), dtype=bool)
    classes_present: set[int] = set()

    for class_id, instances in masks.items():
        color = palette.get(int(class_id))
        if color is None:
            continue
        for mask_tensor in instances:
            m = mask_tensor.cpu().numpy() > 0
            if not m.any():
                continue
            overlay[m] = color
            any_mask |= m
            classes_present.add(int(class_id))

    return _blend(rgb_bgr, overlay, any_mask, alpha), sorted(classes_present)


def overlay_instance(
    rgb_bgr: np.ndarray,
    masks: dict[int, list[torch.Tensor]],
    alpha: float,
) -> np.ndarray:
    """Paint each mask instance with its own distinct color and blend.

    Args:
        rgb_bgr: Source frame as ``[H, W, 3]`` uint8 BGR.
        masks: Instance masks keyed by class id.
        alpha: Mask blend weight in ``[0, 1]``.

    Returns:
        The overlaid frame as ``[H, W, 3]`` uint8 BGR.
    """
    h, w = rgb_bgr.shape[:2]
    overlay = np.zeros_like(rgb_bgr)
    any_mask = np.zeros((h, w), dtype=bool)

    for class_id, instances in masks.items():
        for instance_no, mask_tensor in enumerate(instances):
            m = mask_tensor.cpu().numpy() > 0
            if not m.any():
                continue
            overlay[m] = _instance_color(int(class_id), instance_no)
            any_mask |= m

    return _blend(rgb_bgr, overlay, any_mask, alpha)


def render_legend(
    classes_present: list[int],
    palette: dict[int, np.ndarray],
    class_names: dict[int, str],
    image_height: int,
    legend_width: int,
) -> np.ndarray:
    """Render a vertical legend panel mapping class colors to class names.

    Each row shows a colored swatch next to ``{class_id}: {class_name}``.
    Rows are sized to fit all present classes within ``image_height`` with
    reasonable font scaling.

    Args:
        classes_present: Sorted list of class ids that appear in the frame.
        palette: Class-to-BGR color mapping.
        class_names: Class id to name mapping.
        image_height: Height of the legend panel in pixels.
        legend_width: Width of the legend panel in pixels.

    Returns:
        A ``[image_height, legend_width, 3]`` uint8 BGR legend image.
    """
    legend = np.full((image_height, legend_width, 3), 255, dtype=np.uint8)
    if not classes_present:
        return legend

    padding = 10
    usable_height = max(image_height - 2 * padding, 1)
    n = len(classes_present)
    row_height = max(18, min(34, usable_height // n))
    swatch_size = max(10, row_height - 6)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(0.6, row_height / 44.0))
    font_thickness = 1

    y = padding
    for cid in classes_present:
        if y + row_height > image_height - padding:
            break
        color = palette.get(cid, np.array([128, 128, 128], dtype=np.uint8))
        sw_top = y + (row_height - swatch_size) // 2
        top_left = (padding, sw_top)
        bottom_right = (padding + swatch_size, sw_top + swatch_size)
        cv2.rectangle(legend, top_left, bottom_right, color.tolist(), thickness=-1)
        cv2.rectangle(legend, top_left, bottom_right, (0, 0, 0), thickness=1)

        label = f"{cid}: {class_names.get(cid, 'unknown')}"
        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_x = bottom_right[0] + 8
        text_y = y + (row_height + text_size[1]) // 2
        cv2.putText(legend, label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        y += row_height
    return legend


def run(
    masks_dir: Path,
    rgb_dir: Path,
    output_dir: Path,
    classes_json: Path,
    hierarchy_json: Path | None,
    alpha: float,
    legend_width: int,
    limit: int | None,
) -> None:
    """Generate semantic (with legend) and instance overlays for all frames.

    Mask conflicts are resolved via
    :func:`hislam2.gaussian.semantics.mask_reader.resolve_sam_masks_conflicts`,
    which uses the same pairwise hierarchy + confidence rules as the main
    pipeline. Each pixel ends up assigned to at most one instance.

    Writes two images per frame:
      - ``output_dir/semantic/frame{idx:06d}.png`` — class-colored overlay
        with a legend panel appended on the right.
      - ``output_dir/instance/frame{idx:06d}.png`` — per-instance colored
        overlay.

    Args:
        masks_dir: Root directory of SAM3 per-frame mask subdirectories.
        rgb_dir: Directory containing source RGB ``.jpg`` frames.
        output_dir: Root output directory; ``semantic/`` and ``instance/``
            subdirectories are created inside.
        classes_json: Path to a JSON file with ``{"id", "name"}`` entries
            for the legend labels.
        hierarchy_json: Optional path to a ``{child_id: parent_id}`` JSON
            hierarchy used for conflict resolution. ``None`` disables it.
        alpha: Mask blend weight in ``[0, 1]``.
        legend_width: Width in pixels of the appended legend panel.
        limit: Optional cap on number of frames processed (for smoke tests).
    """
    semantic_dir = output_dir / "semantic"
    instance_dir = output_dir / "instance"
    semantic_dir.mkdir(parents=True, exist_ok=True)
    instance_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = discover_frame_indices(masks_dir, rgb_dir)
    if limit is not None:
        frame_indices = frame_indices[:limit]
    if not frame_indices:
        raise RuntimeError(f"No overlapping frames found between {masks_dir} and {rgb_dir}")

    class_ids: set[int] = set()
    for idx in frame_indices:
        class_ids.update(int(p.stem.split("_")[0]) for p in (masks_dir / f"frame{idx:06d}").glob("*.png"))
    palette = build_class_palette(sorted(class_ids))
    class_names = load_class_names(classes_json)
    hierarchy = load_hierarchy(hierarchy_json)

    for idx in tqdm(frame_indices, desc="Overlaying masks", unit="frame"):
        rgb_path = rgb_dir / f"frame{idx:06d}.jpg"
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"Failed to read RGB frame: {rgb_path}")

        masks = read_sam3_masks(idx, str(masks_dir))
        masks = resolve_sam_masks_conflicts(masks, idx, str(masks_dir), hierarchy=hierarchy)

        semantic_img, classes_present = overlay_semantic(rgb_bgr, masks, palette, alpha)
        legend = render_legend(classes_present, palette, class_names, semantic_img.shape[0], legend_width)
        semantic_with_legend = np.concatenate([semantic_img, legend], axis=1)
        cv2.imwrite(str(semantic_dir / f"frame{idx:06d}.png"), semantic_with_legend)

        instance_img = overlay_instance(rgb_bgr, masks, alpha)
        cv2.imwrite(str(instance_dir / f"frame{idx:06d}.png"), instance_img)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the mask overlay script.

    Returns:
        The parsed ``argparse.Namespace`` with mask/rgb/output paths, the
        classes JSON path, blend weight, legend width, and optional frame
        limit.
    """
    parser = argparse.ArgumentParser(description="Overlay SAM3 masks onto RGB frames (semantic + instance views).")
    parser.add_argument("--masks-dir", type=Path, default=REPO_ROOT / "data/masks/room0/masks")
    parser.add_argument("--rgb-dir", type=Path, default=REPO_ROOT / "data/Replica_semantics/room0/frames")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/overlaid/room0")
    parser.add_argument("--classes-json", type=Path, default=REPO_ROOT / "data/Replica_semantics/dataset_classes.json")
    parser.add_argument("--hierarchy-json", type=Path, default=REPO_ROOT / "data/hierarchy.json")
    parser.add_argument("--no-hierarchy", action="store_true", help="Disable hierarchy-based conflict resolution.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--legend-width", type=int, default=320)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hierarchy_json = None if args.no_hierarchy else args.hierarchy_json
    if hierarchy_json is not None and not hierarchy_json.exists():
        hierarchy_json = None
    run(
        args.masks_dir,
        args.rgb_dir,
        args.output_dir,
        args.classes_json,
        hierarchy_json,
        args.alpha,
        args.legend_width,
        args.limit,
    )
