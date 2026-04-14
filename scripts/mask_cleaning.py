"""
Mask cleaning pipeline for SAM3 pseudo-GT masks.

Each pixel should have exactly one label. This script:
  1. Derives a label hierarchy from real GT (semantic_ids) by voting on
     which label "owns" contested pixels.
  2. Cleans masks using the hierarchy: parent/root labels beat children;
     ties broken by SAM3 confidence scores.
  3. Evaluates the cleaned output against GT and a confidence-only baseline.

Usage:
    python scripts/mask_cleaning.py build-hierarchy \\
        --scenes office0 room0 \\
        --masks_dir data/masks \\
        --gt_dir data/Replica_semantics \\
        --output hierarchy.json

    python scripts/mask_cleaning.py clean \\
        --scene office0 \\
        --masks_dir data/masks \\
        --hierarchy hierarchy.json \\
        --output_dir outputs/cleaned_masks

    python scripts/mask_cleaning.py evaluate \\
        --scene office0 \\
        --cleaned_dir outputs/cleaned_masks \\
        --gt_dir data/Replica_semantics
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_gt_image(gt_root: Path, scene: str, frame_idx: int) -> np.ndarray:
    """
    Load GT semantic_ids image for a frame.

    Args:
        gt_root: Root directory of Replica_semantics data.
        scene: Scene name (e.g. "office0").
        frame_idx: Zero-based frame index.

    Returns:
        Array of shape (H, W) with dtype uint8; each pixel is a Replica label_id.
    """
    path = gt_root / scene / "semantic_ids" / f"semantic_id{frame_idx:06d}.png"
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.array(Image.open(path).convert("L"))
    return img


def load_masks_with_confidence(
    frame_dir: Path,
) -> dict[int, list[tuple[np.ndarray, float]]]:
    """
    Load all SAM3 masks for a frame together with per-instance confidence scores.

    Mask PNGs are named ``{label_id}_{instance_no}.png``.  Confidence values
    are read from the accompanying ``masks.json``.  If a mask file has no
    matching entry in masks.json the confidence defaults to 0.0.

    Args:
        frame_dir: Directory containing the mask PNGs and masks.json.

    Returns:
        Dict mapping label_id → [(binary_mask_uint8, confidence), ...] where
        masks are sorted by instance_no.
    """
    json_path = frame_dir / "masks.json"
    confidence_map: dict[tuple[int, int], float] = {}
    if json_path.exists():
        with open(json_path) as f:
            metadata = json.load(f)
        for item in metadata:
            label = item.get("item_id", item.get("semantic_id", -1))
            inst = item.get("instance_no", item.get("instance_id", 0))
            conf = float(item.get("confidence", 0.0))
            confidence_map[(label, inst)] = conf

    result: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
    for png in sorted(frame_dir.glob("*.png")):
        parts = png.stem.split("_")
        if len(parts) != 2:
            continue
        label_id, inst_no = int(parts[0]), int(parts[1])
        mask = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.array(Image.open(png).convert("L"))
        conf = confidence_map.get((label_id, inst_no), 0.0)
        result[label_id].append((mask.astype(np.uint8), conf))

    return dict(result)


def _frame_indices(masks_root: Path, scene: str) -> list[int]:
    """Return sorted list of frame indices available in a scene's mask directory."""
    scene_mask_dir = masks_root / scene / "masks"
    dirs = sorted(scene_mask_dir.glob("frame??????"))
    indices = []
    for d in dirs:
        try:
            indices.append(int(d.name[5:]))
        except ValueError:
            pass
    return indices


# ---------------------------------------------------------------------------
# Phase 1: Build hierarchy
# ---------------------------------------------------------------------------

def compute_pair_dominance(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    gt_image: np.ndarray,
    label_a: int,
    label_b: int,
) -> str | None:
    """
    Determine which of two overlapping masks is the GT-dominant label.

    Looks at pixels claimed by both masks and counts how many the GT assigns
    to label_a vs label_b.

    Args:
        mask_a: Binary mask for label_a (uint8).
        mask_b: Binary mask for label_b (uint8).
        gt_image: GT semantic label image (uint8, Replica label IDs).
        label_a: Replica label ID for mask_a.
        label_b: Replica label ID for mask_b.

    Returns:
        ``"A"`` if label_a dominates in the overlap, ``"B"`` if label_b
        dominates, ``None`` if the overlap is empty or tied.
    """
    overlap = (mask_a > 0) & (mask_b > 0)
    if not overlap.any():
        return None

    gt_overlap = gt_image[overlap]
    count_a = int((gt_overlap == label_a).sum())
    count_b = int((gt_overlap == label_b).sum())

    if count_a > count_b:
        return "A"
    elif count_b > count_a:
        return "B"
    return None


def build_hierarchy(
    scenes: list[str],
    masks_root: Path,
    gt_root: Path,
    min_vote_ratio: float = 0.6,
) -> dict[int, int]:
    """
    Empirically derive a label hierarchy by voting over all frames and scenes.

    For every pair of overlapping SAM3 masks (A, B) in a frame, the GT label
    at the overlapping pixels casts a vote for which label is the "parent".
    After aggregating votes, a parent relationship ``child → parent`` is
    established when one label dominates the other in at least ``min_vote_ratio``
    of contested frames.

    Args:
        scenes: List of scene names to process.
        masks_root: Root of the SAM3 masks directory.
        gt_root: Root of the Replica_semantics GT directory.
        min_vote_ratio: Fraction of frames where A must beat B for A to be
            declared parent of B.

    Returns:
        Dict mapping child_label_id → parent_label_id.
    """
    # votes[(a, b)] = [count_a_wins, count_b_wins]  where a < b
    votes: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])

    for scene in scenes:
        frame_indices = _frame_indices(masks_root, scene)
        for idx in tqdm(frame_indices, desc=f"Hierarchy: {scene}", unit="frame"):
            frame_dir = masks_root / scene / "masks" / f"frame{idx:06d}"
            masks = load_masks_with_confidence(frame_dir)
            label_ids = list(masks.keys())

            try:
                gt = _load_gt_image(gt_root, scene, idx)
            except (FileNotFoundError, OSError):
                continue

            for i in range(len(label_ids)):
                for j in range(i + 1, len(label_ids)):
                    lid_a, lid_b = label_ids[i], label_ids[j]
                    # Canonical key: smaller id first
                    if lid_a > lid_b:
                        lid_a, lid_b = lid_b, lid_a
                    key = (lid_a, lid_b)

                    # Merge all instances per label into a single mask
                    merged_a = np.zeros(gt.shape, dtype=np.uint8)
                    for m, _ in masks[label_ids[i]]:
                        merged_a |= m
                    merged_b = np.zeros(gt.shape, dtype=np.uint8)
                    for m, _ in masks[label_ids[j]]:
                        merged_b |= m

                    winner = compute_pair_dominance(
                        merged_a, merged_b, gt, label_ids[i], label_ids[j]
                    )
                    if winner == "A":
                        # label_ids[i] wins
                        if label_ids[i] == key[0]:
                            votes[key][0] += 1
                        else:
                            votes[key][1] += 1
                    elif winner == "B":
                        # label_ids[j] wins
                        if label_ids[j] == key[0]:
                            votes[key][0] += 1
                        else:
                            votes[key][1] += 1

    hierarchy: dict[int, int] = {}
    for (la, lb), (wins_a, wins_b) in votes.items():
        total = wins_a + wins_b
        if total == 0:
            continue
        if wins_a / total >= min_vote_ratio:
            # la dominates lb → lb is child of la
            hierarchy[lb] = la
        elif wins_b / total >= min_vote_ratio:
            # lb dominates la → la is child of lb
            hierarchy[la] = lb

    return hierarchy


def cmd_build_hierarchy(args: argparse.Namespace) -> None:
    """Entry point for the build-hierarchy subcommand."""
    hierarchy = build_hierarchy(
        scenes=args.scenes,
        masks_root=Path(args.masks_dir),
        gt_root=Path(args.gt_dir),
        min_vote_ratio=args.min_vote_ratio,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in hierarchy.items()}, f, indent=2)

    print(f"Hierarchy saved to {out_path}  ({len(hierarchy)} child→parent pairs)")
    for child, parent in sorted(hierarchy.items()):
        print(f"  {child} → {parent}")


# ---------------------------------------------------------------------------
# Phase 2: Clean masks
# ---------------------------------------------------------------------------

def resolve_frame(
    masks_with_conf: dict[int, list[tuple[np.ndarray, float]]],
    parent_set: set[int],
) -> np.ndarray:
    """
    Merge all SAM3 masks for one frame into a single label image.

    Pixel assignment priority:
        - Priority 2: labels that appear as a parent in the hierarchy
        - Priority 1: all other labels (children or ungrouped)
        - Tiebreaker within the same priority level: higher SAM3 confidence

    Args:
        masks_with_conf: Output of :func:`load_masks_with_confidence`.
        parent_set: Set of label IDs that are parents in the hierarchy.

    Returns:
        Label image of shape (H, W) with dtype uint16; pixel value is the
        assigned Replica label_id (0 = uncovered by any mask).
    """
    if not masks_with_conf:
        return np.zeros((1, 1), dtype=np.uint16)

    # Infer H, W from first mask
    first_masks = next(iter(masks_with_conf.values()))
    H, W = first_masks[0][0].shape

    label_image = np.zeros((H, W), dtype=np.uint16)
    priority_image = np.zeros((H, W), dtype=np.int8)
    confidence_image = np.zeros((H, W), dtype=np.float32)

    for label_id, instances in masks_with_conf.items():
        priority = 2 if label_id in parent_set else 1
        for mask, conf in instances:
            pixels = mask > 0
            override = pixels & (
                (priority_image < priority)
                | ((priority_image == priority) & (confidence_image < conf))
            )
            label_image[override] = label_id
            priority_image[override] = priority
            confidence_image[override] = conf

    return label_image


def process_scene(
    scene: str,
    masks_root: Path,
    hierarchy: dict[int, int],
    output_dir: Path,
) -> None:
    """
    Clean all frames for a scene and write label images to disk.

    Args:
        scene: Scene name.
        masks_root: Root of the SAM3 masks directory.
        hierarchy: Child→parent dict from :func:`build_hierarchy`.
        output_dir: Directory where cleaned label PNGs will be saved.
    """
    parent_set = set(hierarchy.values())
    scene_out = output_dir / scene
    scene_out.mkdir(parents=True, exist_ok=True)

    frame_indices = _frame_indices(masks_root, scene)
    for idx in tqdm(frame_indices, desc=f"Cleaning: {scene}", unit="frame"):
        frame_dir = masks_root / scene / "masks" / f"frame{idx:06d}"
        masks = load_masks_with_confidence(frame_dir)
        label_image = resolve_frame(masks, parent_set)
        out_path = scene_out / f"frame{idx:06d}.png"
        Image.fromarray(label_image).save(str(out_path))


def cmd_clean(args: argparse.Namespace) -> None:
    """Entry point for the clean subcommand."""
    hierarchy_path = Path(args.hierarchy)
    if hierarchy_path.exists():
        with open(hierarchy_path) as f:
            raw = json.load(f)
        hierarchy = {int(k): int(v) for k, v in raw.items()}
    else:
        print(f"Warning: hierarchy file {hierarchy_path} not found; using empty hierarchy.")
        hierarchy = {}

    process_scene(
        scene=args.scene,
        masks_root=Path(args.masks_dir),
        hierarchy=hierarchy,
        output_dir=Path(args.output_dir),
    )
    print(f"Cleaned masks written to {Path(args.output_dir) / args.scene}")


# ---------------------------------------------------------------------------
# Phase 3: Evaluate
# ---------------------------------------------------------------------------

def _confidence_only_label_image(
    masks_with_conf: dict[int, list[tuple[np.ndarray, float]]]
) -> np.ndarray:
    """
    Baseline: merge masks using confidence only (no hierarchy).

    Args:
        masks_with_conf: Output of :func:`load_masks_with_confidence`.

    Returns:
        Label image (H, W) uint16.
    """
    return resolve_frame(masks_with_conf, parent_set=set())


def evaluate_against_gt(
    label_image: np.ndarray,
    gt_image: np.ndarray,
) -> dict:
    """
    Compare a cleaned label image against the GT semantic_ids image.

    Background pixels (GT == 0) are excluded from all metrics.

    Args:
        label_image: Cleaned label image (H, W) uint16.
        gt_image: GT semantic_ids image (H, W) uint8.

    Returns:
        Dict with keys:
            - ``overall_accuracy``: fraction of foreground pixels that match GT.
            - ``per_label_accuracy``: dict mapping label_id → accuracy float.
            - ``total_fg_pixels``: count of non-background GT pixels.
    """
    fg = gt_image > 0
    if not fg.any():
        return {"overall_accuracy": 0.0, "per_label_accuracy": {}, "total_fg_pixels": 0}

    gt_fg = gt_image[fg].astype(np.int32)
    pred_fg = label_image[fg].astype(np.int32)

    overall = float((gt_fg == pred_fg).mean())

    per_label: dict[int, float] = {}
    for label_id in np.unique(gt_fg):
        mask = gt_fg == label_id
        per_label[int(label_id)] = float((pred_fg[mask] == label_id).mean())

    return {
        "overall_accuracy": overall,
        "per_label_accuracy": per_label,
        "total_fg_pixels": int(fg.sum()),
    }


def evaluate_scene(
    scene: str,
    cleaned_dir: Path,
    gt_root: Path,
    masks_root: Path | None = None,
) -> dict:
    """
    Evaluate cleaned label images for a scene against GT.

    Also computes a confidence-only baseline if ``masks_root`` is provided.

    Args:
        scene: Scene name.
        cleaned_dir: Directory containing cleaned label PNGs (output of clean subcommand).
        gt_root: Root of the Replica_semantics GT directory.
        masks_root: Optional; if given, a confidence-only baseline is computed.

    Returns:
        Dict with aggregated accuracy metrics and optional baseline comparison.
    """
    scene_dir = cleaned_dir / scene
    frame_files = sorted(scene_dir.glob("frame??????.png"))

    overall_accs: list[float] = []
    baseline_accs: list[float] = []
    per_label_totals: dict[int, list[float]] = defaultdict(list)

    for png in tqdm(frame_files, desc=f"Evaluating: {scene}", unit="frame"):
        idx = int(png.stem[5:])
        label_image = np.array(Image.open(png))

        try:
            gt = _load_gt_image(gt_root, scene, idx)
        except (FileNotFoundError, OSError):
            continue

        metrics = evaluate_against_gt(label_image, gt)
        overall_accs.append(metrics["overall_accuracy"])
        for lid, acc in metrics["per_label_accuracy"].items():
            per_label_totals[lid].append(acc)

        if masks_root is not None:
            frame_dir = masks_root / scene / "masks" / f"frame{idx:06d}"
            masks = load_masks_with_confidence(frame_dir)
            baseline_img = _confidence_only_label_image(masks)
            baseline_metrics = evaluate_against_gt(baseline_img, gt)
            baseline_accs.append(baseline_metrics["overall_accuracy"])

    results: dict = {
        "scene": scene,
        "mean_overall_accuracy": float(np.mean(overall_accs)) if overall_accs else 0.0,
        "per_label_mean_accuracy": {
            lid: float(np.mean(accs)) for lid, accs in per_label_totals.items()
        },
        "num_frames_evaluated": len(overall_accs),
    }

    if baseline_accs:
        results["baseline_mean_accuracy"] = float(np.mean(baseline_accs))
        results["accuracy_gain"] = results["mean_overall_accuracy"] - results["baseline_mean_accuracy"]

    return results


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Entry point for the evaluate subcommand."""
    masks_root = Path(args.masks_dir) if args.masks_dir else None
    results = evaluate_scene(
        scene=args.scene,
        cleaned_dir=Path(args.cleaned_dir),
        gt_root=Path(args.gt_dir),
        masks_root=masks_root,
    )

    out_path = Path(args.cleaned_dir) / args.scene / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== {args.scene} ===")
    print(f"  Hierarchy accuracy : {results['mean_overall_accuracy']:.4f}")
    if "baseline_mean_accuracy" in results:
        print(f"  Baseline accuracy  : {results['baseline_mean_accuracy']:.4f}")
        print(f"  Accuracy gain      : {results['accuracy_gain']:+.4f}")
    print(f"  Frames evaluated   : {results['num_frames_evaluated']}")
    print(f"  Metrics saved to   : {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser(
        description="SAM3 mask cleaning: build hierarchy, clean, evaluate."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build-hierarchy
    p_hier = sub.add_parser("build-hierarchy", help="Build label hierarchy from GT.")
    p_hier.add_argument("--scenes", nargs="+", required=True, help="Scene names to process.")
    p_hier.add_argument("--masks_dir", required=True, help="Root of SAM3 masks directory.")
    p_hier.add_argument("--gt_dir", required=True, help="Root of Replica_semantics GT directory.")
    p_hier.add_argument("--output", required=True, help="Output path for hierarchy.json.")
    p_hier.add_argument(
        "--min_vote_ratio",
        type=float,
        default=0.6,
        help="Minimum fraction of frames where A beats B to declare A as parent of B (default: 0.6).",
    )

    # clean
    p_clean = sub.add_parser("clean", help="Clean masks using hierarchy.")
    p_clean.add_argument("--scene", required=True, help="Scene name.")
    p_clean.add_argument("--masks_dir", required=True, help="Root of SAM3 masks directory.")
    p_clean.add_argument("--hierarchy", required=True, help="Path to hierarchy.json.")
    p_clean.add_argument("--output_dir", required=True, help="Output directory for cleaned label PNGs.")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate cleaned masks against GT.")
    p_eval.add_argument("--scene", required=True, help="Scene name.")
    p_eval.add_argument("--cleaned_dir", required=True, help="Directory containing cleaned label PNGs.")
    p_eval.add_argument("--gt_dir", required=True, help="Root of Replica_semantics GT directory.")
    p_eval.add_argument(
        "--masks_dir",
        default=None,
        help="Root of SAM3 masks directory (optional; enables baseline comparison).",
    )

    args = parser.parse_args()

    if args.command == "build-hierarchy":
        cmd_build_hierarchy(args)
    elif args.command == "clean":
        cmd_clean(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
