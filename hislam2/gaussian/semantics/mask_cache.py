from collections import OrderedDict
from pathlib import Path
import json

from hislam2.gaussian.semantics.mask_generator import MaskGenerator
from hislam2.gaussian.semantics.mask_reader import (
    filter_overlapping_masks,
    read_sam3_masks,
    resolve_sam_masks_conflicts,
    sam_masks_semantic_image,
    sam3_dict_to_tensor,
)


class MaskCache:
    """LRU cache of CPU mask tensors to avoid repeated HDD reads.

    Keys are integer frame timestamps. Values are tuples of
    (sam_masks_cpu, sem_masks_cpu, sem_mask_ids_cpu), all stored on CPU.
    GPU transfer happens in :meth:`load`, returning CUDA tensors ready for
    use in the training loop.
    """

    def __init__(self, config, save_dir, max_frames=300):
        """Initialize the cache.

        Args:
            config: Full pipeline config dict (reads ``masks`` sub-section).
            save_dir: Output directory used by :class:`MaskGenerator` when
                Mask2Former masks are generated on-demand.
            max_frames: Maximum number of frames to keep resident in RAM.
        """
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_frames
        self._config = config
        self._save_dir = save_dir
        self._mask_generator = None

        masks_cfg = config.get("masks", {})
        source = str(masks_cfg.get("source", "sam3")).lower()
        if source == "mas2former":
            source = "mask2former"
        self._source = source

        if source == "sam3":
            self._masks_dir = masks_cfg.get("sam3_masks_dir")
        elif source == "mask2former":
            self._masks_dir = masks_cfg.get("mask2former_masks_dir")
        else:
            raise ValueError(
                f"Unsupported masks.source '{source}'. Use 'sam3' or 'mask2former'."
            )

        if not self._masks_dir:
            raise ValueError(
                f"Mask directory is not configured for masks.source='{source}'."
            )

        self._hierarchy = self._load_hierarchy(masks_cfg)
        self._overlap_threshold = float(masks_cfg.get("conflict_overlap_threshold", 0.8))

    @staticmethod
    def _load_hierarchy(masks_cfg: dict) -> dict[int, int] | None:
        """Load the label hierarchy from disk if configured and enabled.

        The JSON file maps child label ids (as string keys) to their parent
        label id (integer values).  Both are converted to ``int`` so callers
        can look up integer label ids directly.

        Args:
            masks_cfg: The ``masks`` sub-section of the pipeline config.

        Returns:
            Dict mapping ``child_id -> parent_id``, or *None* when
            ``use_hierarchy`` is false or ``hierarchy_path`` is not set.
        """
        if not masks_cfg.get("use_hierarchy", False):
            return None
        hierarchy_path = masks_cfg.get("hierarchy_path")
        if not hierarchy_path:
            return None
        path = Path(hierarchy_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Mask hierarchy file not found: {path}"
            )
        with open(path, "r") as f:
            raw = json.load(f)
        return {int(k): int(v) for k, v in raw.items()}

    @staticmethod
    def is_valid(mask_tensor):
        """Return True only for non-empty mask tensors with valid spatial size."""
        return (
            mask_tensor is not None
            and mask_tensor.numel() > 0
            and mask_tensor.ndim == 3
            and mask_tensor.shape[1] > 0
            and mask_tensor.shape[2] > 0
        )

    def load(self, tstamp, viewpoints):
        """Load and GPU-transfer masks for *tstamp*.

        CPU tensors are served from the LRU cache when available, avoiding
        repeated HDD reads. Only the final ``.cuda()`` transfers happen on
        every call.

        Args:
            tstamp: Frame timestamp (int or float).
            viewpoints: Mapping of tstamp → Camera, used by Mask2Former when
                masks have not been pre-computed for this frame.

        Returns:
            sam_masks    (Tensor | None): Binary SAM instance masks [N, H, W] on CUDA.
            sem_masks    (Tensor | None): Semantically grouped masks on CUDA.
            sem_mask_ids (Tensor | None): Class label per semantic mask on CUDA.
        """
        key = int(tstamp)

        if key in self._cache:
            self._cache.move_to_end(key)
            sam_cpu, sem_cpu, ids_cpu = self._cache[key]
        else:
            sam_cpu, sem_cpu, ids_cpu = self._read_from_disk(key, viewpoints)

            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = (sam_cpu, sem_cpu, ids_cpu)

        if not self.is_valid(sam_cpu):
            sam_cpu = None
        if not self.is_valid(sem_cpu):
            sem_cpu = None
            ids_cpu = None
        if ids_cpu is not None and ids_cpu.numel() == 0:
            ids_cpu = None

        sam_masks = sam_cpu.cuda() if sam_cpu is not None else None
        sem_masks = sem_cpu.cuda() if sem_cpu is not None else None
        sem_mask_ids = ids_cpu.cuda() if ids_cpu is not None else None
        return sam_masks, sem_masks, sem_mask_ids

    def _read_from_disk(self, key, viewpoints):
        """Read raw CPU mask tensors from disk for frame *key*.

        Args:
            key: Integer frame timestamp.
            viewpoints: Mapping of tstamp → Camera (needed for Mask2Former).

        Returns:
            Tuple of (sam_cpu, sem_cpu, ids_cpu) CPU tensors (any may be None).
        """
        frame_dir = Path(self._masks_dir) / f"frame{key:06d}"
        has_precomputed = (frame_dir / "masks.json").exists()

        if self._source == "mask2former" and not has_precomputed:
            return self._generate_mask2former(key, viewpoints)

        sam_masks_dict = read_sam3_masks(key, self._masks_dir)
        sam_masks_dict = filter_overlapping_masks(
            sam_masks_dict, threshold=self._overlap_threshold
        )
        sam_masks_dict = resolve_sam_masks_conflicts(
            sam_masks_dict, key, self._masks_dir, hierarchy=self._hierarchy
        )
        sem_cpu, ids_cpu = sam_masks_semantic_image(
            sam_masks_dict, key, self._masks_dir, hierarchy=self._hierarchy
        )
        sam_cpu = sam3_dict_to_tensor(sam_masks_dict)
        return sam_cpu, sem_cpu, ids_cpu

    def _generate_mask2former(self, key, viewpoints):
        """Run Mask2Former on-demand for frame *key* and save masks to disk.

        Args:
            key: Integer frame timestamp.
            viewpoints: Mapping of tstamp → Camera.

        Returns:
            Tuple of (sam_cpu, sem_cpu, ids_cpu) CPU tensors.
        """
        if self._mask_generator is None:
            self._mask_generator = MaskGenerator(self._config, self._save_dir)

        viewpoint = viewpoints.get(key, None)
        if viewpoint is None:
            for vp in viewpoints.values():
                if int(vp.tstamp) == key:
                    viewpoint = vp
                    break

        if viewpoint is None:
            raise RuntimeError(
                f"No viewpoint found for tstamp={key} to run Mask2Former."
            )

        return self._mask_generator.generate_and_save_masks(viewpoint)
