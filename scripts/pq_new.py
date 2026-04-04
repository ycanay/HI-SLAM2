import argparse
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from tqdm import tqdm
from torchmetrics.detection.panoptic_qualities import PanopticQuality
from torchmetrics.segmentation import MeanIoU
import matplotlib
matplotlib.use('Agg')

DATASET_PATH = Path("/storage/user/ayu/repos/HI-SLAM2/data/Replica_semantics")
OUTPUT_PATH = Path("/storage/user/ayu/repos/HI-SLAM2/outputs/semantic")

# Evaluation config JSONs (tracked in the repo under scripts/eval_configs/).
# These are small class-mapping files independent of the actual Replica dataset.
_EVAL_CONFIGS_DIR = Path(__file__).parent / "eval_configs"


def color_to_instance_id(rgb):
    """
    Convert a single RGB color to instance ID using bit-interleaving scheme.

    The encoding scheme used by Replica dataset:
    - R channel bits [7,6,5,4,3,2,1,0] correspond to instance ID bits [0,3,6,9,12,15,18,21]
    - G channel bits [7,6,5,4,3,2,1,0] correspond to instance ID bits [1,4,7,10,13,16,19,22]
    - B channel bits [7,6,5,4,3,2,1,0] correspond to instance ID bits [2,5,8,11,14,17,20,23]

    Args:
        rgb: tuple or array of (R, G, B) values, each in range [0, 255]

    Returns:
        int: instance ID decoded from the color
    """
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])

    instance_id = 0
    for bit_pos in range(8):
        # Extract bit from each channel (from MSB to LSB)
        color_bit = 7 - bit_pos  # Color channel bit position
        id_bit_base = bit_pos * 3  # Base ID bit position

        r_bit = (r >> color_bit) & 1
        g_bit = (g >> color_bit) & 1
        b_bit = (b >> color_bit) & 1

        instance_id |= (r_bit << id_bit_base)
        instance_id |= (g_bit << (id_bit_base + 1))
        instance_id |= (b_bit << (id_bit_base + 2))

    return instance_id


def color_image_to_instance_ids(color_image):
    """
    Convert an RGB color image to instance ID image using bit-interleaving scheme.

    This is a vectorized version for efficient processing of entire images.

    Args:
        color_image: numpy array of shape (H, W, 3) with RGB values in range [0, 255]

    Returns:
        numpy array of shape (H, W) with instance IDs (uint32)
    """
    # Ensure input is the correct type
    color_image = color_image.astype(np.uint32)

    r = color_image[:, :, 0]
    g = color_image[:, :, 1]
    b = color_image[:, :, 2]

    instance_ids = np.zeros(r.shape, dtype=np.uint32)

    for bit_pos in range(8):
        # Extract bit from each channel (from MSB to LSB)
        color_bit = 7 - bit_pos  # Color channel bit position
        id_bit_base = bit_pos * 3  # Base ID bit position

        r_bit = (r >> color_bit) & 1
        g_bit = (g >> color_bit) & 1
        b_bit = (b >> color_bit) & 1

        instance_ids |= (r_bit << id_bit_base)
        instance_ids |= (g_bit << (id_bit_base + 1))
        instance_ids |= (b_bit << (id_bit_base + 2))

    return instance_ids


def build_color_to_instance_mapping(scene_path, num_frames=None):
    """
    Build a mapping from RGB colors to instance IDs by scanning instance_colors 
    and instance_ids folders.

    This can be used as an alternative to the bit-interleaving conversion,
    or to verify the conversion is correct.

    Args:
        scene_path: Path to the scene directory (e.g., data/Replica_semantics/office1)
        num_frames: Number of frames to scan (None for all)

    Returns:
        dict: mapping from (R, G, B) tuple to instance ID
    """
    scene_path = Path(scene_path)
    instance_colors_dir = scene_path / "instance_colors"
    instance_ids_dir = scene_path / "instance_ids"

    color_files = sorted(instance_colors_dir.glob("instance_color*.png"))
    if num_frames is not None:
        color_files = color_files[:num_frames]

    color_to_id = {}

    for color_file in tqdm(color_files, desc="Building color mapping"):
        frame_idx = int(color_file.stem.replace("instance_color", ""))
        id_file = instance_ids_dir / f"instance_id{frame_idx:06d}.png"

        color_img = cv2.imread(str(color_file))
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        id_img = cv2.imread(str(id_file), cv2.IMREAD_UNCHANGED)

        # Get unique colors and their corresponding IDs
        for i in range(color_img.shape[0]):
            for j in range(color_img.shape[1]):
                color = tuple(color_img[i, j])
                inst_id = int(id_img[i, j])
                if color not in color_to_id:
                    color_to_id[color] = inst_id

    return color_to_id


def load_ground_truth_frame(scene_path, frame_idx):
    """
    Load ground truth data for a single frame.

    Args:
        scene_path: Path to the scene directory
        frame_idx: Frame index

    Returns:
        tuple: (instance_ids, semantic_ids) as numpy arrays
    """
    scene_path = Path(scene_path)

    # Load instance colors and convert to IDs
    instance_color_path = scene_path / "instance_colors" / \
        f"instance_color{frame_idx:06d}.png"
    instance_color_img = cv2.imread(str(instance_color_path))
    instance_color_img = cv2.cvtColor(instance_color_img, cv2.COLOR_BGR2RGB)
    instance_ids = color_image_to_instance_ids(instance_color_img)

    # Load semantic IDs
    semantic_id_path = scene_path / "semantic_ids" / \
        f"semantic_id{frame_idx:06d}.png"
    semantic_ids = cv2.imread(str(semantic_id_path), cv2.IMREAD_UNCHANGED)

    return instance_ids, semantic_ids


def load_scene_info(scene_path):
    """
    Load scene info from info_semantic.json.

    Args:
        scene_path: Path to the scene directory

    Returns:
        dict: Scene information including classes, objects, and id_to_label mapping
    """
    scene_path = Path(scene_path)
    info_path = scene_path / "info_semantic.json"

    with open(info_path, 'r') as f:
        info = json.load(f)

    return info


def load_panoptic_config(config_path=None):
    """
    Load panoptic segmentation configuration including stuff/thing classification.

    Args:
        config_path: Path to semantics_panoptic.json. If None, uses default path.

    Returns:
        dict: Mapping from semantic ID to class info (name, isStuff)
    """
    if config_path is None:
        config_path = _EVAL_CONFIGS_DIR / "semantics_panoptic.json"

    with open(config_path, 'r') as f:
        classes = json.load(f)

    # Build mapping from id to class info
    id_to_class = {}
    for cls in classes:
        id_to_class[cls['id']] = {
            'name': cls['name'],
            'isStuff': cls['isStuff']
        }

    return id_to_class


def load_panoptic_lifting_mapping(mapping_path=None):
    """
    Load the panoptic lifting class mapping.

    This maps original Replica semantic classes to a smaller set of classes
    used by Panoptic Lifting evaluation.

    Args:
        mapping_path: Path to map_panoptic_lifting.json. If None, uses default path.

    Returns:
        tuple: (semantic_mapping, target_classes)
            - semantic_mapping: dict mapping source_id -> target_id
            - target_classes: dict mapping target_id -> {name, isThing}
    """
    if mapping_path is None:
        mapping_path = _EVAL_CONFIGS_DIR / "map_panoptic_lifting.json"

    with open(mapping_path, 'r') as f:
        mapping_list = json.load(f)

    # Build source -> target mapping
    semantic_mapping = {}
    target_classes = {}

    for entry in mapping_list:
        source_id = entry['id']
        target_id = entry['mapping']['id']
        target_name = entry['mapping']['name']
        is_thing = entry['mapping']['isThing']

        semantic_mapping[source_id] = target_id
        target_classes[target_id] = {
            'name': target_name,
            'isThing': is_thing
        }

    return semantic_mapping, target_classes


def apply_semantic_mapping(semantic_ids, mapping, unmapped_value=255):
    """
    Apply semantic class mapping to an image.

    Args:
        semantic_ids: numpy array of semantic class IDs (H, W)
        mapping: dict mapping source_id -> target_id
        unmapped_value: value to use for unmapped classes (default 255 = ignore)

    Returns:
        numpy array with mapped semantic IDs
    """
    mapped_ids = np.full_like(semantic_ids, unmapped_value, dtype=np.int32)

    for source_id, target_id in mapping.items():
        mapped_ids[semantic_ids == source_id] = target_id

    return mapped_ids


def get_things_and_stuffs_from_mapping(target_classes):
    """
    Get sets of thing and stuff class IDs from panoptic lifting mapping.

    Args:
        target_classes: Mapping from target class ID to class info {name, isThing}

    Returns:
        tuple: (things_set, stuffs_set) - sets of class IDs
    """
    things = set()
    stuffs = set()

    for class_id, info in target_classes.items():
        if info['isThing']:
            things.add(class_id)
        else:
            stuffs.add(class_id)

    return things, stuffs


def load_cluster_features(output_path):
    """
    Load cluster features from cluster_features.json.

    Args:
        output_path: Path to output directory containing cluster_features.json

    Returns:
        dict: Mapping from cluster ID (int) to semantic label
    """
    output_path = Path(output_path)
    features_path = output_path / "cluster_features.json"

    with open(features_path, 'r') as f:
        features = json.load(f)

    # Build mapping from cluster ID to semantic label
    cluster_to_semantic = {}
    for cluster_id, info in features.items():
        cluster_to_semantic[int(cluster_id)] = info['label']

    return cluster_to_semantic


def load_prediction_frame(output_path, frame_idx, cluster_to_semantic):
    """
    Load prediction data for a single frame from rendered cluster image.

    Args:
        output_path: Path to output directory
        frame_idx: Frame index
        cluster_to_semantic: Mapping from cluster ID to semantic label

    Returns:
        tuple: (pred_instance_ids, pred_semantic_ids) as numpy arrays
    """
    output_path = Path(output_path)
    render_path = output_path / "renders" / \
        "cluster_after_opt" / f"{frame_idx:06d}.png"

    if not render_path.exists():
        return None, None

    # Load cluster image (grayscale, all 3 channels have same value)
    cluster_img = cv2.imread(str(render_path), cv2.IMREAD_UNCHANGED)
    if len(cluster_img.shape) == 3:
        cluster_ids = cluster_img[:, :, 0].astype(np.int32)
    else:
        cluster_ids = cluster_img.astype(np.int32)

    # Convert cluster IDs to semantic IDs
    semantic_ids = np.zeros_like(cluster_ids, dtype=np.int32)
    for cluster_id, sem_id in cluster_to_semantic.items():
        semantic_ids[cluster_ids == cluster_id] = sem_id

    return cluster_ids, semantic_ids


def get_things_and_stuffs(id_to_class):
    """
    Get sets of thing and stuff class IDs from panoptic config.

    Args:
        id_to_class: Mapping from semantic ID to class info

    Returns:
        tuple: (things_set, stuffs_set) - sets of class IDs
    """
    things = set()
    stuffs = set()

    for class_id, info in id_to_class.items():
        if info['isStuff']:
            stuffs.add(class_id)
        else:
            things.add(class_id)

    return things, stuffs


def mask_void_regions(gt_semantic_ids, pred_semantic_ids, pred_instance_ids, void_id=0):
    """
    Mask out void/unknown regions in predictions to match ground truth.

    In Replica dataset, semantic_id=0 corresponds to unknown classes (class_id=-1).
    These regions should be ignored during evaluation.

    Args:
        gt_semantic_ids: Ground truth semantic IDs (H, W)
        pred_semantic_ids: Predicted semantic IDs (H, W)
        pred_instance_ids: Predicted instance IDs (H, W)
        void_id: The ID representing void/unknown regions (default 0)

    Returns:
        tuple: (masked_pred_semantic_ids, masked_pred_instance_ids)
               where void regions in GT are also set to void in predictions
    """
    # Create mask for void regions in ground truth
    void_mask = gt_semantic_ids == void_id

    # Apply mask to predictions - set void regions to 0
    masked_pred_semantic_ids = pred_semantic_ids.copy()
    masked_pred_instance_ids = pred_instance_ids.copy()

    masked_pred_semantic_ids[void_mask] = void_id
    masked_pred_instance_ids[void_mask] = 0

    return masked_pred_semantic_ids, masked_pred_instance_ids


def prepare_panoptic_tensor(semantic_ids, instance_ids):
    """
    Prepare panoptic tensor for torchmetrics PanopticQuality.

    Args:
        semantic_ids: Semantic class IDs (H, W)
        instance_ids: Instance IDs (H, W)

    Returns:
        torch.Tensor: Shape (1, H, W, 2) with (category_id, instance_id) pairs
    """
    H, W = semantic_ids.shape
    panoptic = np.stack([semantic_ids, instance_ids], axis=-1)
    return torch.from_numpy(panoptic).unsqueeze(0).long()


def evaluate_panoptic_quality(
    scene_name,
    run_number,
    dataset_path=None,
    output_base_path=None,
    frame_indices=None,
    void_id=0,
    mapping_mode="lifting",
    verbose=True
):
    """
    Evaluate panoptic quality for a given scene and run using torchmetrics.

    Args:
        scene_name: Name of the scene (e.g., 'office1')
        run_number: Run number (e.g., 6)
        dataset_path: Path to Replica_semantics dataset. If None, uses default.
        output_base_path: Path to outputs/semantic. If None, uses default.
        frame_indices: List of frame indices to evaluate. If None, evaluates all available.
        void_id: ID representing void/unknown classes to ignore (default 0)
        verbose: Whether to print progress

    Returns:
        dict: Formatted PQ results
    """
    if mapping_mode not in {"lifting", "none"}:
        raise ValueError(
            f"mapping_mode must be one of {{'lifting', 'none'}}, got: {mapping_mode!r}"
        )

    if dataset_path is None:
        dataset_path = DATASET_PATH
    if output_base_path is None:
        output_base_path = OUTPUT_PATH

    dataset_path = Path(dataset_path)
    output_base_path = Path(output_base_path)

    scene_path = dataset_path / scene_name
    output_path = output_base_path / f"{scene_name}_{run_number}"

    # Load cluster features for predictions
    cluster_to_semantic = load_cluster_features(output_path)

    semantic_mapping = None
    if mapping_mode == "lifting":
        # Load panoptic lifting mapping (Replica -> Panoptic Lifting class set)
        semantic_mapping, target_classes = load_panoptic_lifting_mapping()

        # Get thing and stuff class sets from the mapping
        things, stuffs = get_things_and_stuffs_from_mapping(target_classes)

        # If any valid target class uses void_id (0), shift all target IDs by +1
        # so that 0 is reserved exclusively for unmapped/void pixels.
        # Without this shift, apply_semantic_mapping(..., unmapped_value=0) makes
        # truly unmapped pixels indistinguishable from valid classes that happen to
        # have target ID 0 (e.g. "wall" in the Panoptic Lifting mapping), and
        # stuffs.discard(void_id) below would silently drop that class entirely.
        if void_id in things or void_id in stuffs:
            semantic_mapping = {src: tgt + 1 for src, tgt in semantic_mapping.items()}
            things = {c + 1 for c in things}
            stuffs = {c + 1 for c in stuffs}

        # Void/unknown should never be evaluated as a class
        things.discard(void_id)
        stuffs.discard(void_id)

        if verbose:
            print("Using Panoptic Lifting mapping")
            print(f"Number of thing classes: {len(things)}")
            print(f"Number of stuff classes: {len(stuffs)}")
            print(f"Things: {things}")
            print(f"Stuffs: {stuffs}")
    else:
        # Use the original Replica semantic IDs and panoptic config to define thing/stuff
        id_to_class = load_panoptic_config()
        things, stuffs = get_things_and_stuffs(id_to_class)

        # Void/unknown should never be evaluated as a class
        things.discard(void_id)
        stuffs.discard(void_id)

        if verbose:
            print("Using original semantic IDs (no mapping)")
            print(f"Number of thing classes: {len(things)}")
            print(f"Number of stuff classes: {len(stuffs)}")
            print(f"Things: {things}")
            print(f"Stuffs: {stuffs}")

    # Initialize PanopticQuality metric
    pq_metric = PanopticQuality(
        things=things,
        stuffs=stuffs,
        allow_unknown_preds_category=True,
        return_sq_and_rq=True,
        return_per_class=False
    )

    # Initialize MeanIoU metric
    # Number of classes is max class ID + 1 (including void class 0)
    num_classes = max(things | stuffs) + 1
    miou_metric = MeanIoU(
        num_classes=num_classes,
        per_class=False,
        include_background=False  # Exclude void class (0) from mIoU
    )

    if verbose:
        print(f"Number of classes for mIoU: {num_classes}")

    # Get available frames
    render_dir = output_path / "renders" / "cluster_after_opt"
    if frame_indices is None:
        frame_files = sorted(render_dir.glob("*.png"))
        frame_indices = [int(f.stem) for f in frame_files]

    if verbose:
        print(
            f"Evaluating {len(frame_indices)} frames for {scene_name}_{run_number}")

    # Evaluate each frame
    iterator = tqdm(
        frame_indices, desc="Evaluating frames") if verbose else frame_indices
    valid_frames = 0

    for frame_idx in iterator:
        # Load ground truth
        gt_instance_ids, gt_semantic_ids = load_ground_truth_frame(
            scene_path, frame_idx)

        # Load predictions
        pred_instance_ids, pred_semantic_ids = load_prediction_frame(
            output_path, frame_idx, cluster_to_semantic
        )

        if pred_instance_ids is None or pred_semantic_ids is None:
            continue

        # Resize predictions to match GT if needed
        if pred_instance_ids.shape != gt_instance_ids.shape:
            pred_instance_ids = cv2.resize(
                pred_instance_ids.astype(np.float32),
                (gt_instance_ids.shape[1], gt_instance_ids.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)
            pred_semantic_ids = cv2.resize(
                pred_semantic_ids.astype(np.float32),
                (gt_semantic_ids.shape[1], gt_semantic_ids.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)

        # Mask void regions - predictions in void GT regions should be ignored
        pred_semantic_ids, pred_instance_ids = mask_void_regions(
            gt_semantic_ids, pred_semantic_ids, pred_instance_ids, void_id
        )

        # Optionally apply semantic class mapping to both GT and predictions.
        # Unmapped classes are set to void_id so they are ignored.
        if semantic_mapping is not None:
            gt_semantic_ids_eval = apply_semantic_mapping(
                gt_semantic_ids, semantic_mapping, unmapped_value=void_id
            )
            pred_semantic_ids_eval = apply_semantic_mapping(
                pred_semantic_ids, semantic_mapping, unmapped_value=void_id
            )
        else:
            gt_semantic_ids_eval = gt_semantic_ids.astype(np.int32, copy=False)
            pred_semantic_ids_eval = pred_semantic_ids.astype(
                np.int32, copy=False)

        # Prepare tensors for torchmetrics
        gt_tensor = prepare_panoptic_tensor(
            gt_semantic_ids_eval.astype(np.int64),
            gt_instance_ids.astype(np.int64)
        )
        pred_tensor = prepare_panoptic_tensor(
            pred_semantic_ids_eval.astype(np.int64),
            pred_instance_ids.astype(np.int64)
        )

        # Update PQ metric
        pq_metric.update(pred_tensor, gt_tensor)

        # Update mIoU metric (expects [N, H, W] tensors)
        gt_semantic_tensor = torch.from_numpy(
            gt_semantic_ids_eval.astype(np.int64)).unsqueeze(0)
        pred_semantic_tensor = torch.from_numpy(
            pred_semantic_ids_eval.astype(np.int64)).unsqueeze(0)
        miou_metric.update(pred_semantic_tensor, gt_semantic_tensor)

        valid_frames += 1

    if valid_frames == 0:
        print("No valid frames found!")
        return None

    # Compute final results
    pq_results = pq_metric.compute()
    miou_results = miou_metric.compute()

    # Format results
    formatted = format_torchmetrics_results(pq_results, miou_results)

    if verbose:
        print_pq_results(formatted, scene_name, run_number)

    return formatted


def format_torchmetrics_results(pq_results, miou_results):
    """
    Format torchmetrics PanopticQuality and MeanIoU results into a readable dict.

    Args:
        pq_results: Raw results tensor from pq_metric.compute() - shape [3] with [PQ, SQ, RQ]
        miou_results: Raw results tensor from miou_metric.compute() - scalar mIoU value

    Returns:
        dict: Formatted results with PQ, SQ, RQ, and mIoU
    """
    formatted = {
        'pq': pq_results[0].item() * 100,
        'sq': pq_results[1].item() * 100,
        'rq': pq_results[2].item() * 100,
        'miou': miou_results.item() * 100
    }

    return formatted


def print_pq_results(formatted, scene_name, run_number):
    """
    Print formatted PQ results to console.

    Args:
        formatted: Formatted results dict
        scene_name: Scene name
        run_number: Run number
    """
    print("\n" + "=" * 60)
    print(f"Panoptic Quality Results for {scene_name}_{run_number}")
    print("=" * 60)
    print(
        f"\nPQ={formatted['pq']:.2f}  SQ={formatted['sq']:.2f}  RQ={formatted['rq']:.2f}")
    print(f"mIoU={formatted['miou']:.2f}")


def save_pq_results(results, output_path, scene_name, run_number):
    """
    Save PQ results to JSON file.

    Args:
        results: Formatted PQ results
        output_path: Base output path
        scene_name: Scene name
        run_number: Run number
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / f"pq_results_{scene_name}_{run_number}.json"

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Panoptic Quality (PQ/SQ/RQ) and mIoU on Replica_semantics outputs."
    )
    parser.add_argument("--scene", required=True,
                        help="Scene name (e.g., room0, office1)")
    parser.add_argument("--run", type=int, required=True,
                        help="Run number (e.g., 109)")
    parser.add_argument(
        "--mapping",
        choices=["lifting", "none"],
        default="lifting",
        help="Semantic mapping mode: 'lifting' applies map_panoptic_lifting.json, 'none' evaluates in original semantic IDs.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(DATASET_PATH),
        help="Path to Replica_semantics root",
    )
    parser.add_argument(
        "--output-base-path",
        default=str(OUTPUT_PATH),
        help="Base path containing per-run outputs (e.g., outputs/semantic)",
    )
    parser.add_argument(
        "--void-id",
        type=int,
        default=0,
        help="Void/unknown semantic ID to ignore during evaluation (Replica default: 0)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress bars / verbose prints",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save results JSON under outputs/per_class_results_new",
    )
    args = parser.parse_args()

    results = evaluate_panoptic_quality(
        scene_name=args.scene,
        run_number=args.run,
        dataset_path=args.dataset_path,
        output_base_path=args.output_base_path,
        void_id=args.void_id,
        mapping_mode=args.mapping,
        verbose=not args.quiet,
    )

    if results and args.save_json:
        save_pq_results(
            results,
            OUTPUT_PATH.parent / "per_class_results_new",
            args.scene,
            args.run,
        )
