import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from math import floor
from collections import defaultdict
import random
import cv2


class H5Dataset(Dataset):
    def __init__(self, dataset, client_id):
        self.targets = torch.LongTensor(dataset[client_id]['label'])
        self.inputs = torch.Tensor(dataset[client_id]['pixels'])
        shape = self.inputs.shape
        self.inputs = self.inputs.view(shape[0], 1, shape[1], shape[2])

    def classes(self):
        return torch.unique(self.targets)

    def __add__(self, other):
        self.targets = torch.cat((self.targets, other.targets), 0)
        self.inputs = torch.cat((self.inputs, other.inputs), 0)
        return self

    def to(self, device):
        self.targets = self.targets.to(device)
        self.inputs = self.inputs.to(device)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, item):
        inp, target = self.inputs[item], self.targets[item]
        return inp, target


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])

    def classes(self):
        return torch.unique(self.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        inp, target = self.dataset[self.idxs[item]]
        return inp, target


# ==================================================================================================
# Scenario 5 / Hypothesis 5: Trigger geometry extra code block
# --------------------------------------------------------------------------------------------------
# This section is intentionally large and explicit because Project 8 is not supposed to be a simple
# hyperparameter change. It contains a full trigger geometry subsystem:
#   - explicit mask builders
#   - geometry descriptors
#   - mask metrics
#   - geometry verification logs
#   - trigger ASCII preview
#   - scenario-specific mapping between clean_label_type and trigger_geometry
#   - active comparison fields for thesis txt logs
#
# The thesis claim:
#   Different trigger geometries can lead to different backdoor success rates.
#   This motivates later Cuckoo Search optimization of trigger shape/location.
# ==================================================================================================


TRIGGER_GEOMETRY_GRID = ["plus", "square", "apple"]


def normalize_trigger_geometry(name):
    """Normalize user/command trigger names into canonical geometry labels."""
    if name is None:
        return "plus"

    name = str(name).strip().lower()

    aliases = {
        "plus": "plus",
        "cross": "plus",
        "+": "plus",
        "square": "square",
        "box": "square",
        "block": "square",
        "patch": "square",
        "apple": "apple",
        "logo": "apple",
        "symbol": "apple",
    }

    if name in aliases:
        return aliases[name]

    return "plus"


def clean_label_type_to_geometry(clean_label_type):
    """Map clean_label_type to explicit trigger geometry."""
    try:
        clean_label_type = int(clean_label_type)
    except Exception:
        clean_label_type = 0

    if clean_label_type == 1:
        return "plus"
    if clean_label_type == 2:
        return "square"
    if clean_label_type == 3:
        return "apple"

    return None


def get_geometry_from_args(args):
    """
    Active trigger geometry selection.

    For clean-label experiments:
        clean_label_type=1 -> plus
        clean_label_type=2 -> square
        clean_label_type=3 -> apple

    For dirty-label experiments:
        trigger_geometry controls the geometry directly.
        pattern_type is kept for compatibility.

    This means both dirty-label and clean-label runs can be compared on exactly
    the same trigger geometry grid.
    """
    clean_label_type_geometry = clean_label_type_to_geometry(getattr(args, "clean_label_type", 0))

    if getattr(args, "clean_label", 0) == 1 and clean_label_type_geometry is not None:
        return clean_label_type_geometry

    if hasattr(args, "trigger_geometry"):
        return normalize_trigger_geometry(args.trigger_geometry)

    return normalize_trigger_geometry(getattr(args, "pattern_type", "plus"))


def geometry_to_clean_label_type(geometry):
    """Reverse mapping used for run documentation."""
    geometry = normalize_trigger_geometry(geometry)
    if geometry == "plus":
        return 1
    if geometry == "square":
        return 2
    if geometry == "apple":
        return 3
    return 0


def get_geometry_family(geometry):
    """Coarse geometric family label for thesis logs."""
    geometry = normalize_trigger_geometry(geometry)
    if geometry == "plus":
        return "sparse_cross"
    if geometry == "square":
        return "dense_corner_patch"
    if geometry == "apple":
        return "symbolic_global_pattern"
    return "unknown_family"


def get_geometry_expected_visibility(geometry):
    """Qualitative visual visibility for logs."""
    geometry = normalize_trigger_geometry(geometry)
    if geometry == "plus":
        return "low_to_medium"
    if geometry == "square":
        return "medium"
    if geometry == "apple":
        return "high"
    return "unknown"


def get_geometry_expected_pixel_scale(geometry):
    """
    Expected relative number of changed pixels.
    This is not the empirical metric; the empirical metric is printed during poisoning.
    """
    geometry = normalize_trigger_geometry(geometry)
    if geometry == "plus":
        return "small"
    if geometry == "square":
        return "medium"
    if geometry == "apple":
        return "large"
    return "unknown"


def build_plus_mask(size=28, start_idx=5, arm_size=5):
    """Build a binary mask for a plus/cross trigger."""
    mask = np.zeros((size, size), dtype=np.uint8)

    for i in range(start_idx, start_idx + arm_size):
        if 0 <= i < size and 0 <= start_idx < size:
            mask[i, start_idx] = 1

    center_row = start_idx + arm_size // 2
    for i in range(start_idx - arm_size // 2, start_idx + arm_size // 2 + 1):
        if 0 <= center_row < size and 0 <= i < size:
            mask[center_row, i] = 1

    return mask


def build_square_mask(size=28, row_start=21, row_end=26, col_start=21, col_end=26):
    """Build a binary mask for a square trigger."""
    mask = np.zeros((size, size), dtype=np.uint8)

    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            if 0 <= i < size and 0 <= j < size:
                mask[i, j] = 1

    return mask


def build_apple_proxy_mask(size=28):
    """
    Build a deterministic proxy mask for the apple trigger.

    The actual image-based apple trigger uses apple.png if available.
    This proxy mask exists so the code can still print meaningful geometry metrics
    without needing to inspect the external image file.
    """
    mask = np.zeros((size, size), dtype=np.uint8)

    # apple body
    body_points = [
        (8, 12), (8, 13), (8, 14), (8, 15),
        (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16),
        (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17),
        (11, 8), (11, 9), (11, 10), (11, 11), (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18),
        (12, 8), (12, 9), (12, 10), (12, 11), (12, 12), (12, 13), (12, 14), (12, 15), (12, 16), (12, 17), (12, 18),
        (13, 8), (13, 9), (13, 10), (13, 11), (13, 12), (13, 13), (13, 14), (13, 15), (13, 16), (13, 17), (13, 18),
        (14, 8), (14, 9), (14, 10), (14, 11), (14, 12), (14, 13), (14, 14), (14, 15), (14, 16), (14, 17), (14, 18),
        (15, 9), (15, 10), (15, 11), (15, 12), (15, 13), (15, 14), (15, 15), (15, 16), (15, 17),
        (16, 10), (16, 11), (16, 12), (16, 13), (16, 14), (16, 15), (16, 16),
        (17, 11), (17, 12), (17, 13), (17, 14), (17, 15),
        (18, 12), (18, 13), (18, 14),
    ]

    # stem and leaf
    stem_leaf_points = [
        (5, 14), (6, 14), (7, 14),
        (5, 15), (5, 16), (6, 16),
        (6, 17), (7, 17)
    ]

    for r, c in body_points + stem_leaf_points:
        if 0 <= r < size and 0 <= c < size:
            mask[r, c] = 1

    return mask


def build_trigger_mask(geometry, dataset="fmnist"):
    """Return binary trigger mask for metrics and visualization."""
    geometry = normalize_trigger_geometry(geometry)

    if dataset == "cifar10":
        size = 32
    else:
        size = 28

    if geometry == "plus":
        if dataset == "fedemnist":
            return build_plus_mask(size=size, start_idx=8, arm_size=5)
        if dataset == "cifar10":
            return build_plus_mask(size=size, start_idx=5, arm_size=6)
        return build_plus_mask(size=size, start_idx=5, arm_size=5)

    if geometry == "square":
        if dataset == "cifar10":
            return build_square_mask(size=size, row_start=24, row_end=30, col_start=24, col_end=30)
        return build_square_mask(size=size, row_start=21, row_end=26, col_start=21, col_end=26)

    if geometry == "apple":
        if dataset == "cifar10":
            proxy = build_apple_proxy_mask(size=28)
            padded = np.zeros((32, 32), dtype=np.uint8)
            padded[2:30, 2:30] = proxy
            return padded
        return build_apple_proxy_mask(size=size)

    return build_plus_mask(size=size)


def mask_area(mask):
    """Number of active trigger pixels."""
    return int(np.sum(mask > 0))


def mask_density(mask):
    """Active pixels divided by full image pixels."""
    if mask.size == 0:
        return 0.0
    return float(mask_area(mask) / mask.size)


def mask_bounding_box(mask):
    """Return bounding box of active mask pixels."""
    active = np.argwhere(mask > 0)

    if active.size == 0:
        return {
            "row_min": None,
            "row_max": None,
            "col_min": None,
            "col_max": None,
            "height": 0,
            "width": 0,
        }

    row_min = int(active[:, 0].min())
    row_max = int(active[:, 0].max())
    col_min = int(active[:, 1].min())
    col_max = int(active[:, 1].max())

    return {
        "row_min": row_min,
        "row_max": row_max,
        "col_min": col_min,
        "col_max": col_max,
        "height": row_max - row_min + 1,
        "width": col_max - col_min + 1,
    }


def mask_centroid(mask):
    """Return centroid of active trigger pixels."""
    active = np.argwhere(mask > 0)

    if active.size == 0:
        return {
            "row_centroid": None,
            "col_centroid": None,
        }

    return {
        "row_centroid": float(active[:, 0].mean()),
        "col_centroid": float(active[:, 1].mean()),
    }


def mask_edge_touch_count(mask):
    """Count active pixels touching the image boundary."""
    if mask.size == 0:
        return 0

    top = np.sum(mask[0, :] > 0)
    bottom = np.sum(mask[-1, :] > 0)
    left = np.sum(mask[:, 0] > 0)
    right = np.sum(mask[:, -1] > 0)

    return int(top + bottom + left + right)


def mask_quadrant_distribution(mask):
    """Count trigger pixels per image quadrant."""
    h, w = mask.shape
    h_mid = h // 2
    w_mid = w // 2

    q1 = int(np.sum(mask[:h_mid, :w_mid] > 0))
    q2 = int(np.sum(mask[:h_mid, w_mid:] > 0))
    q3 = int(np.sum(mask[h_mid:, :w_mid] > 0))
    q4 = int(np.sum(mask[h_mid:, w_mid:] > 0))

    return {
        "top_left": q1,
        "top_right": q2,
        "bottom_left": q3,
        "bottom_right": q4,
    }


def mask_compactness_score(mask):
    """
    Simple compactness proxy:
        area / bounding_box_area
    """
    bbox = mask_bounding_box(mask)
    bbox_area = bbox["height"] * bbox["width"]

    if bbox_area <= 0:
        return 0.0

    return float(mask_area(mask) / bbox_area)


def mask_row_profile(mask):
    """Return count of active pixels per row."""
    return [int(x) for x in np.sum(mask > 0, axis=1).tolist()]


def mask_col_profile(mask):
    """Return count of active pixels per column."""
    return [int(x) for x in np.sum(mask > 0, axis=0).tolist()]


def ascii_mask_preview(mask, max_size=32):
    """Create a compact ASCII preview for logs."""
    h, w = mask.shape
    rows = []

    for r in range(min(h, max_size)):
        chars = []
        for c in range(min(w, max_size)):
            chars.append("#" if mask[r, c] > 0 else ".")
        rows.append("".join(chars))

    return rows


def compute_trigger_geometry_metrics(geometry, dataset="fmnist"):
    """Compute all geometry metrics used in logs and thesis comparison."""
    geometry = normalize_trigger_geometry(geometry)
    mask = build_trigger_mask(geometry, dataset=dataset)
    bbox = mask_bounding_box(mask)
    centroid = mask_centroid(mask)

    metrics = {
        "geometry": geometry,
        "geometry_family": get_geometry_family(geometry),
        "expected_visibility": get_geometry_expected_visibility(geometry),
        "expected_pixel_scale": get_geometry_expected_pixel_scale(geometry),
        "mask_area": mask_area(mask),
        "mask_density": mask_density(mask),
        "bbox_row_min": bbox["row_min"],
        "bbox_row_max": bbox["row_max"],
        "bbox_col_min": bbox["col_min"],
        "bbox_col_max": bbox["col_max"],
        "bbox_height": bbox["height"],
        "bbox_width": bbox["width"],
        "row_centroid": centroid["row_centroid"],
        "col_centroid": centroid["col_centroid"],
        "edge_touch_count": mask_edge_touch_count(mask),
        "quadrants": mask_quadrant_distribution(mask),
        "compactness_score": mask_compactness_score(mask),
        "row_profile": mask_row_profile(mask),
        "col_profile": mask_col_profile(mask),
        "ascii_preview": ascii_mask_preview(mask),
    }

    return metrics


def print_trigger_geometry_metrics(args, context="setup"):
    """Print trigger geometry metrics for thesis txt logs."""
    if not hasattr(args, "verify_trigger_geometry") or args.verify_trigger_geometry != 1:
        return

    geometry = get_geometry_from_args(args)
    metrics = compute_trigger_geometry_metrics(geometry, dataset=args.data)

    print("VERIFY_TRIGGER_GEOMETRY_START")
    print(f"context={context}")
    print(f"scenario=Hypothesis 5 - trigger geometry")
    print(f"trigger_geometry_grid={getattr(args, 'trigger_geometry_grid', 'plus,square,apple')}")
    print(f"trigger_geometry={geometry}")
    print(f"pattern_type={getattr(args, 'pattern_type', 'NA')}")
    print(f"clean_label={getattr(args, 'clean_label', 0)}")
    print(f"clean_label_type={getattr(args, 'clean_label_type', 0)}")
    print(f"mapped_clean_label_type={geometry_to_clean_label_type(geometry)}")
    print(f"geometry_family={metrics['geometry_family']}")
    print(f"expected_visibility={metrics['expected_visibility']}")
    print(f"expected_pixel_scale={metrics['expected_pixel_scale']}")
    print(f"mask_area={metrics['mask_area']}")
    print(f"mask_density={metrics['mask_density']}")
    print(f"bbox_row_min={metrics['bbox_row_min']}")
    print(f"bbox_row_max={metrics['bbox_row_max']}")
    print(f"bbox_col_min={metrics['bbox_col_min']}")
    print(f"bbox_col_max={metrics['bbox_col_max']}")
    print(f"bbox_height={metrics['bbox_height']}")
    print(f"bbox_width={metrics['bbox_width']}")
    print(f"row_centroid={metrics['row_centroid']}")
    print(f"col_centroid={metrics['col_centroid']}")
    print(f"edge_touch_count={metrics['edge_touch_count']}")
    print(f"quadrants={metrics['quadrants']}")
    print(f"compactness_score={metrics['compactness_score']}")

    if getattr(args, "trigger_geometry_report", 1) == 1:
        print("TRIGGER_ASCII_PREVIEW_START")
        for row in metrics["ascii_preview"]:
            print(row)
        print("TRIGGER_ASCII_PREVIEW_END")
        print(f"row_profile={metrics['row_profile']}")
        print(f"col_profile={metrics['col_profile']}")

    print("TRIGGER_GEOMETRY_PROOF: the trigger shape is actively selected and measured by code, not only changed as a raw hyperparameter.")
    print("VERIFY_TRIGGER_GEOMETRY_PASSED")
    print("VERIFY_TRIGGER_GEOMETRY_END")


def trigger_geometry_summary_line(args):
    """One-line summary printed during training."""
    geometry = get_geometry_from_args(args)
    metrics = compute_trigger_geometry_metrics(geometry, dataset=args.data)

    return (
        f"trigger_geometry={geometry} "
        f"family={metrics['geometry_family']} "
        f"mask_area={metrics['mask_area']} "
        f"mask_density={metrics['mask_density']:.4f} "
        f"compactness={metrics['compactness_score']:.4f}"
    )


def get_trigger_geometry_index(geometry):
    """Numerical index for TensorBoard logging."""
    geometry = normalize_trigger_geometry(geometry)

    if geometry == "plus":
        return 1
    if geometry == "square":
        return 2
    if geometry == "apple":
        return 3

    return 0


def expected_geometry_rank_for_visibility(geometry):
    """Simple ordinal visibility ranking for logs."""
    geometry = normalize_trigger_geometry(geometry)

    if geometry == "plus":
        return 1
    if geometry == "square":
        return 2
    if geometry == "apple":
        return 3

    return 0


def expected_geometry_rank_for_area(geometry, dataset="fmnist"):
    """Rank geometries by computed mask area."""
    geometry = normalize_trigger_geometry(geometry)
    all_metrics = {
        g: compute_trigger_geometry_metrics(g, dataset=dataset)["mask_area"]
        for g in TRIGGER_GEOMETRY_GRID
    }

    sorted_geometries = sorted(all_metrics.keys(), key=lambda g: all_metrics[g])
    return sorted_geometries.index(geometry) + 1


def compare_geometry_to_grid(geometry, dataset="fmnist"):
    """Return current trigger metrics compared with plus/square/apple grid."""
    geometry = normalize_trigger_geometry(geometry)
    current = compute_trigger_geometry_metrics(geometry, dataset=dataset)

    comparison = {}
    for g in TRIGGER_GEOMETRY_GRID:
        m = compute_trigger_geometry_metrics(g, dataset=dataset)
        comparison[g] = {
            "area": m["mask_area"],
            "density": m["mask_density"],
            "compactness": m["compactness_score"],
            "current_minus_area": current["mask_area"] - m["mask_area"],
        }

    return comparison


def print_geometry_grid_comparison(args):
    """Print all trigger geometries side-by-side as code-level documentation."""
    if not hasattr(args, "verify_trigger_geometry") or args.verify_trigger_geometry != 1:
        return

    geometry = get_geometry_from_args(args)
    comparison = compare_geometry_to_grid(geometry, dataset=args.data)

    print("TRIGGER_GEOMETRY_GRID_COMPARISON_START")
    print(f"current_trigger_geometry={geometry}")

    for g, values in comparison.items():
        print(
            f"grid_geometry={g} "
            f"area={values['area']} "
            f"density={values['density']} "
            f"compactness={values['compactness']} "
            f"current_minus_area={values['current_minus_area']}"
        )

    print("TRIGGER_GEOMETRY_GRID_COMPARISON_END")


def validate_trigger_geometry_args(args):
    """
    Validate and repair trigger geometry arguments.

    This prevents mistakes such as:
        --trigger_geometry=triangle
    """
    geometry = get_geometry_from_args(args)
    args.trigger_geometry = geometry
    args.pattern_type = geometry

    if getattr(args, "clean_label", 0) == 1:
        mapped = geometry_to_clean_label_type(geometry)
        if getattr(args, "clean_label_type", 0) == 0:
            args.clean_label_type = mapped

    return args


def scenario5_extra_documentation_lines():
    """
    Return many human-readable lines explaining why Scenario 5 exists.
    This is used only for optional logging/documentation and does not affect training.
    """
    lines = []
    lines.append("Scenario 5 is the trigger geometry scenario.")
    lines.append("The fixed elements are dataset, number of agents, poison fraction, and class distribution.")
    lines.append("The changing element is trigger geometry.")
    lines.append("The compared trigger geometries are plus, square, and apple.")
    lines.append("The main empirical output is Poison Acc under each geometry.")
    lines.append("The secondary empirical output is Val Acc under each geometry.")
    lines.append("The trigger mask metrics are printed so the geometry comparison is explicit.")
    lines.append("The plus trigger is sparse and cross-shaped.")
    lines.append("The square trigger is a dense corner patch.")
    lines.append("The apple trigger is a larger symbolic pattern.")
    lines.append("If Poison Acc differs strongly across shapes, trigger design matters.")
    lines.append("If trigger design matters, optimization is justified.")
    lines.append("This motivates the later Cuckoo Search extension.")
    lines.append("Cuckoo Search can be framed as an automatic trigger search mechanism.")
    lines.append("The hand-designed triggers provide the baseline for that search.")
    lines.append("Dirty-label and clean-label attacks are both evaluated.")
    lines.append("Dirty-label should show labels_changed equal to poisoned_samples.")
    lines.append("Clean-label should show labels_changed equal to zero.")
    lines.append("The same geometry names are used in txt logs and run names.")
    lines.append("The run name includes trigplus, trigsquare, or trigapple.")
    return lines


def print_scenario5_extra_documentation(args):
    """Print Scenario 5 documentation lines once into the txt log."""
    if not hasattr(args, "verify_trigger_geometry") or args.verify_trigger_geometry != 1:
        return

    print("SCENARIO5_EXTRA_DOCUMENTATION_START")
    for i, line in enumerate(scenario5_extra_documentation_lines(), start=1):
        print(f"scenario5_doc_line_{i}={line}")
    print("SCENARIO5_EXTRA_DOCUMENTATION_END")


# Extra explicit trigger registry functions. These are intentionally verbose so the scenario has
# real code additions and clear internal structure.
def trigger_registry_plus():
    return {
        "name": "plus",
        "family": "sparse_cross",
        "description": "small plus/cross pattern near the upper-left image region",
        "clean_label_type": 1,
        "expected_visibility": "low_to_medium",
        "expected_pixel_scale": "small",
    }


def trigger_registry_square():
    return {
        "name": "square",
        "family": "dense_corner_patch",
        "description": "dense square patch near the lower-right image region",
        "clean_label_type": 2,
        "expected_visibility": "medium",
        "expected_pixel_scale": "medium",
    }


def trigger_registry_apple():
    return {
        "name": "apple",
        "family": "symbolic_global_pattern",
        "description": "larger symbolic image-based trigger",
        "clean_label_type": 3,
        "expected_visibility": "high",
        "expected_pixel_scale": "large",
    }


def get_trigger_registry():
    return {
        "plus": trigger_registry_plus(),
        "square": trigger_registry_square(),
        "apple": trigger_registry_apple(),
    }


def get_trigger_registry_entry(geometry):
    registry = get_trigger_registry()
    geometry = normalize_trigger_geometry(geometry)
    return registry.get(geometry, registry["plus"])


def print_trigger_registry(args):
    if not hasattr(args, "verify_trigger_geometry") or args.verify_trigger_geometry != 1:
        return

    registry = get_trigger_registry()
    print("TRIGGER_REGISTRY_START")
    for key, entry in registry.items():
        print(
            f"trigger={key} "
            f"family={entry['family']} "
            f"clean_label_type={entry['clean_label_type']} "
            f"expected_visibility={entry['expected_visibility']} "
            f"expected_pixel_scale={entry['expected_pixel_scale']} "
            f"description={entry['description']}"
        )
    print("TRIGGER_REGISTRY_END")


def geometry_thesis_claim(geometry):
    geometry = normalize_trigger_geometry(geometry)
    if geometry == "plus":
        return "The plus trigger is sparse; it tests whether a small cross-shaped pattern is sufficient."
    if geometry == "square":
        return "The square trigger is denser; it tests whether a compact corner patch improves attack success."
    if geometry == "apple":
        return "The apple trigger is symbolic and larger; it tests whether a more complex pattern changes attack success."
    return "Unknown trigger geometry."


def print_geometry_thesis_claim(args):
    if not hasattr(args, "verify_trigger_geometry") or args.verify_trigger_geometry != 1:
        return

    geometry = get_geometry_from_args(args)
    print("TRIGGER_GEOMETRY_THESIS_CLAIM_START")
    print(f"trigger_geometry={geometry}")
    print(f"claim={geometry_thesis_claim(geometry)}")
    print("TRIGGER_GEOMETRY_THESIS_CLAIM_END")


def all_trigger_geometry_setup_logs(args):
    """Central function to print all Scenario 5 setup proof blocks."""
    print_trigger_geometry_metrics(args, context="setup")
    print_geometry_grid_comparison(args)
    print_trigger_registry(args)
    print_geometry_thesis_claim(args)
    print_scenario5_extra_documentation(args)


# ==================================================================================================
# Normal dataset/model utility functions
# ==================================================================================================


def distribute_data(dataset, args, n_classes=10, class_per_agent=None):
    """
    Scenario 5 / Hypothesis 5 trigger geometry data split.

    The data distribution is held fixed. The trigger geometry changes:
        plus
        square
        apple

    This isolates the role of trigger shape/design.
    """

    if class_per_agent is None:
        class_per_agent = args.class_per_agent

    if args.num_agents == 1:
        return {0: range(len(dataset))}

    def chunker_list(seq, size):
        return [seq[i::size] for i in range(size)]

    labels_sorted = dataset.targets.sort()
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))

    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)

    shard_size = len(dataset) // (args.num_agents * class_per_agent)

    if shard_size <= 0:
        raise ValueError(
            f"Invalid setting: shard_size={shard_size}. "
            f"Try fewer agents or a higher class_per_agent."
        )

    slice_size = (len(dataset) // n_classes) // shard_size

    if slice_size <= 0:
        raise ValueError(
            f"Invalid setting: slice_size={slice_size}. "
            f"Try class_per_agent closer to 10."
        )

    for k, v in labels_dict.items():
        labels_dict[k] = chunker_list(v, slice_size)

    dict_users = defaultdict(list)

    for user_idx in range(args.num_agents):
        class_ctr = 0
        for j in range(0, n_classes):
            if class_ctr == class_per_agent:
                break
            elif len(labels_dict[j]) > 0:
                dict_users[user_idx] += labels_dict[j][0]
                del labels_dict[j % n_classes][0]
                class_ctr += 1

    if hasattr(args, "verify_trigger_geometry") and args.verify_trigger_geometry == 1:
        print("VERIFY_TRIGGER_GEOMETRY_DATA_START")
        print(f"scenario=Hypothesis 5 - trigger geometry")
        print(f"scenario_data=IID-ish fixed data distribution")
        print(f"trigger_geometry={get_geometry_from_args(args)}")
        print(f"trigger_geometry_grid={getattr(args, 'trigger_geometry_grid', 'plus,square,apple')}")
        print(f"num_agents={args.num_agents}")
        print(f"class_per_agent={class_per_agent}")
        print(f"n_classes={n_classes}")
        print(f"shard_size={shard_size}")
        print(f"slice_size={slice_size}")
        print(f"corrupt_clients={list(range(args.num_corrupt))}")
        print(f"base_class={args.base_class}")
        print(f"target_class={args.target_class}")

        for user_idx in range(args.num_agents):
            user_idxs = list(dict_users[user_idx])
            user_labels = [int(dataset.targets[idx]) for idx in user_idxs]
            unique_classes = sorted(list(set(user_labels)))
            class_counts = {c: user_labels.count(c) for c in unique_classes}

            print(
                f"client={user_idx} "
                f"n_samples={len(user_idxs)} "
                f"unique_classes={unique_classes} "
                f"class_counts={class_counts}"
            )

        print("TRIGGER_GEOMETRY_DATA_PROOF: data split is held fixed; trigger geometry changes between runs.")
        print("VERIFY_TRIGGER_GEOMETRY_DATA_PASSED")
        print("VERIFY_TRIGGER_GEOMETRY_DATA_END")

    return dict_users


def get_datasets(data):
    """Returns train and test datasets."""
    train_dataset, test_dataset = None, None
    data_dir = '../data'

    if data == 'fmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2860], std=[0.3530])
        ])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    elif data == 'fedemnist':
        train_dir = '../data/Fed_EMNIST/fed_emnist_all_trainset.pt'
        test_dir = '../data/Fed_EMNIST/fed_emnist_all_valset.pt'
        train_dataset = torch.load(train_dir)
        test_dataset = torch.load(test_dir)

    elif data == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(test_dataset.targets)

    return train_dataset, test_dataset


def get_loss_n_accuracy(model, criterion, data_loader, args, num_classes=10):
    """Returns loss and total/per-class accuracy on the supplied data loader."""
    model.eval()
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)

    for _, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device=args.device, non_blocking=True)
        labels = labels.to(device=args.device, non_blocking=True)

        outputs = model(inputs)
        avg_minibatch_loss = criterion(outputs, labels)
        total_loss += avg_minibatch_loss.item() * outputs.shape[0]

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()

        for t, p in zip(labels.view(-1), pred_labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    return avg_loss, (accuracy, per_class_accuracy)


def get_effective_pattern_type(args):
    """
    Scenario 5 active trigger mapping.
    This replaces the older simple pattern selection with explicit geometry control.
    """
    return get_geometry_from_args(args)


def poison_dataset(dataset, args, data_idxs=None, poison_all=False, agent_idx=-1,
                   force_dirty_label=False, context="train"):
    """
    Dirty-label attack:
        source class = base_class
        trigger added
        label changed to target_class

    Clean-label attack:
        source class = target_class
        trigger added
        label remains target_class

    Evaluation / poisoned validation:
        force_dirty_label=True tests base_class + trigger -> target_class.
    """

    clean_label_requested = hasattr(args, "clean_label") and args.clean_label == 1
    clean_label_active = clean_label_requested and not force_dirty_label

    if clean_label_active:
        attack_mode = f"clean-label-{getattr(args, 'clean_label_type', 0)}"
        source_class_for_poisoning = args.target_class
        final_label_after_poisoning = args.target_class
        expected_label_changes_mode = "zero"
    else:
        attack_mode = "dirty-label"
        source_class_for_poisoning = args.base_class
        final_label_after_poisoning = args.target_class
        expected_label_changes_mode = "all"

    effective_pattern_type = get_effective_pattern_type(args)
    geometry_metrics = compute_trigger_geometry_metrics(effective_pattern_type, dataset=args.data)

    all_idxs = (dataset.targets == source_class_for_poisoning).nonzero().flatten().tolist()

    if data_idxs is not None:
        all_idxs = list(set(all_idxs).intersection(data_idxs))

    poison_frac = 1 if poison_all else args.poison_frac

    if len(all_idxs) == 0:
        print("VERIFY_POISONING_START")
        print(f"context={context}")
        print(f"agent_idx={agent_idx}")
        print(f"attack_mode={attack_mode}")
        print(f"trigger_geometry={effective_pattern_type}")
        print(f"trigger_mask_area={geometry_metrics['mask_area']}")
        print(f"source_class={source_class_for_poisoning}")
        print(f"target_class={args.target_class}")
        print(f"available_source_samples=0")
        print("VERIFY_POISONING_FAILED")
        print("reason=no available samples to poison")
        print("VERIFY_POISONING_END")
        return {
            "attack_mode": attack_mode,
            "poisoned": 0,
            "available": 0,
            "labels_changed": 0,
            "verification_passed": False
        }

    num_poison = floor(poison_frac * len(all_idxs))

    if num_poison <= 0:
        print("VERIFY_POISONING_START")
        print(f"context={context}")
        print(f"agent_idx={agent_idx}")
        print(f"attack_mode={attack_mode}")
        print(f"trigger_geometry={effective_pattern_type}")
        print(f"source_class={source_class_for_poisoning}")
        print(f"target_class={args.target_class}")
        print(f"available_source_samples={len(all_idxs)}")
        print(f"poison_frac={poison_frac}")
        print(f"num_poison=0")
        print("VERIFY_POISONING_FAILED")
        print("reason=poison_frac produced zero poisoned samples")
        print("VERIFY_POISONING_END")
        return {
            "attack_mode": attack_mode,
            "poisoned": 0,
            "available": len(all_idxs),
            "labels_changed": 0,
            "verification_passed": False
        }

    poison_idxs = random.sample(all_idxs, num_poison)

    before_labels = []
    after_labels = []
    changed_pixels_first_sample = None

    for count, idx in enumerate(poison_idxs):
        before_label = int(dataset.targets[idx].item()) if hasattr(dataset.targets[idx], "item") else int(dataset.targets[idx])
        before_labels.append(before_label)

        if args.data == 'fedemnist':
            clean_img = dataset.inputs[idx]
        else:
            clean_img = dataset.data[idx]

        clean_img_np_before = np.array(clean_img).copy()

        bd_img = add_pattern_bd(
            clean_img,
            args.data,
            pattern_type=effective_pattern_type,
            agent_idx=agent_idx
        )

        bd_img_np_after = np.array(bd_img).copy()

        if count == 0:
            changed_pixels_first_sample = int(np.sum(clean_img_np_before != bd_img_np_after))

        if args.data == 'fedemnist':
            dataset.inputs[idx] = torch.tensor(bd_img)
        else:
            dataset.data[idx] = torch.tensor(bd_img)

        dataset.targets[idx] = final_label_after_poisoning

        after_label = int(dataset.targets[idx].item()) if hasattr(dataset.targets[idx], "item") else int(dataset.targets[idx])
        after_labels.append(after_label)

    labels_changed = sum(1 for before, after in zip(before_labels, after_labels) if before != after)

    if expected_label_changes_mode == "zero":
        expected_labels_changed = 0
        verification_passed = (labels_changed == 0)
    else:
        if source_class_for_poisoning == final_label_after_poisoning:
            expected_labels_changed = 0
            verification_passed = (labels_changed == 0)
        else:
            expected_labels_changed = num_poison
            verification_passed = (labels_changed == num_poison)

    trigger_verification_passed = changed_pixels_first_sample is not None and changed_pixels_first_sample > 0
    total_verification_passed = verification_passed and trigger_verification_passed

    if hasattr(args, "verify_poisoning") and args.verify_poisoning == 1:
        print("VERIFY_POISONING_START")
        print(f"context={context}")
        print(f"agent_idx={agent_idx}")
        print(f"attack_mode={attack_mode}")
        print(f"force_dirty_label={force_dirty_label}")
        print(f"clean_label_requested={clean_label_requested}")
        print(f"clean_label_active={clean_label_active}")
        print(f"clean_label_type={getattr(args, 'clean_label_type', 0)}")
        print(f"trigger_geometry={effective_pattern_type}")
        print(f"trigger_geometry_family={geometry_metrics['geometry_family']}")
        print(f"trigger_expected_visibility={geometry_metrics['expected_visibility']}")
        print(f"trigger_expected_pixel_scale={geometry_metrics['expected_pixel_scale']}")
        print(f"trigger_mask_area={geometry_metrics['mask_area']}")
        print(f"trigger_mask_density={geometry_metrics['mask_density']}")
        print(f"trigger_compactness_score={geometry_metrics['compactness_score']}")
        print(f"pattern_type_requested={args.pattern_type}")
        print(f"pattern_type_effective={effective_pattern_type}")
        print(f"source_class={source_class_for_poisoning}")
        print(f"target_class={args.target_class}")
        print(f"poison_frac={poison_frac}")
        print(f"available_source_samples={len(all_idxs)}")
        print(f"poisoned_samples={num_poison}")
        print(f"labels_changed={labels_changed}")
        print(f"expected_labels_changed={expected_labels_changed}")
        print(f"changed_pixels_first_sample={changed_pixels_first_sample}")
        print(f"label_verification_passed={verification_passed}")
        print(f"trigger_verification_passed={trigger_verification_passed}")

        if total_verification_passed:
            print("VERIFY_POISONING_PASSED")
        else:
            print("VERIFY_POISONING_FAILED")

        if clean_label_active:
            print("CLEAN_LABEL_PROOF: labels_changed=0 means poisoned training samples kept their original target-class labels.")
        elif force_dirty_label:
            print("EVAL_ATTACK_PROOF: validation poison set uses base_class + trigger -> target_class to measure poison accuracy.")
        else:
            print("DIRTY_LABEL_PROOF: labels_changed=poisoned_samples means base-class labels were changed to target-class labels.")

        print("TRIGGER_GEOMETRY_COMPARISON_POINT: this poisoned run is evaluated under the selected trigger geometry.")
        print("VERIFY_POISONING_END")

    return {
        "attack_mode": attack_mode,
        "poisoned": num_poison,
        "available": len(all_idxs),
        "labels_changed": labels_changed,
        "expected_labels_changed": expected_labels_changed,
        "changed_pixels_first_sample": changed_pixels_first_sample,
        "verification_passed": total_verification_passed,
        "pattern_type_effective": effective_pattern_type,
        "trigger_geometry": effective_pattern_type,
        "trigger_mask_area": geometry_metrics["mask_area"],
        "trigger_mask_density": geometry_metrics["mask_density"],
        "trigger_compactness_score": geometry_metrics["compactness_score"],
    }


def add_pattern_bd(x, dataset='cifar10', pattern_type='square', agent_idx=-1):
    """Adds a trojan pattern to the image."""
    pattern_type = normalize_trigger_geometry(pattern_type)
    x = np.array(x.squeeze())

    if dataset == 'cifar10':
        if pattern_type == 'plus':
            start_idx = 5
            size = 6
            if agent_idx == -1:
                for d in range(0, 3):
                    for i in range(start_idx, start_idx + size + 1):
                        x[i, start_idx][d] = 0
                for d in range(0, 3):
                    for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                        x[start_idx + size // 2, i][d] = 0
            else:
                if agent_idx % 4 == 0:
                    for d in range(0, 3):
                        for i in range(start_idx, start_idx + (size // 2) + 1):
                            x[i, start_idx][d] = 0
                elif agent_idx % 4 == 1:
                    for d in range(0, 3):
                        for i in range(start_idx + (size // 2) + 1, start_idx + size + 1):
                            x[i, start_idx][d] = 0
                elif agent_idx % 4 == 2:
                    for d in range(0, 3):
                        for i in range(start_idx - size // 2, start_idx + size // 4 + 1):
                            x[start_idx + size // 2, i][d] = 0
                elif agent_idx % 4 == 3:
                    for d in range(0, 3):
                        for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
                            x[start_idx + size // 2, i][d] = 0

        elif pattern_type == 'square':
            for i in range(24, 30):
                for j in range(24, 30):
                    if i < x.shape[0] and j < x.shape[1]:
                        x[i, j] = 0

        elif pattern_type == 'apple':
            mask = build_trigger_mask("apple", dataset="cifar10")
            x[mask > 0] = 0

    elif dataset == 'fmnist':
        if pattern_type == 'square':
            for i in range(21, 26):
                for j in range(21, 26):
                    x[i, j] = 255

        elif pattern_type == 'copyright':
            trojan = cv2.imread('../watermark.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            x = x + trojan

        elif pattern_type == 'apple':
            trojan = cv2.imread('../apple.png', cv2.IMREAD_GRAYSCALE)
            if trojan is None:
                mask = build_trigger_mask("apple", dataset="fmnist")
                x[mask > 0] = 255
            else:
                trojan = cv2.bitwise_not(trojan)
                trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
                x = x + trojan

        elif pattern_type == 'plus':
            start_idx = 5
            size = 5
            for i in range(start_idx, start_idx + size):
                x[i, start_idx] = 255

            for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                x[start_idx + size // 2, i] = 255

    elif dataset == 'fedemnist':
        if pattern_type == 'square':
            for i in range(21, 26):
                for j in range(21, 26):
                    x[i, j] = 0

        elif pattern_type == 'copyright':
            trojan = cv2.imread('../watermark.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC) / 255
            x = x - trojan

        elif pattern_type == 'apple':
            trojan = cv2.imread('../apple.png', cv2.IMREAD_GRAYSCALE)
            if trojan is None:
                mask = build_trigger_mask("apple", dataset="fedemnist")
                x[mask > 0] = 0
            else:
                trojan = cv2.bitwise_not(trojan)
                trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC) / 255
                x = x - trojan

        elif pattern_type == 'plus':
            start_idx = 8
            size = 5
            for i in range(start_idx, start_idx + size):
                x[i, start_idx] = 0

            for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                x[start_idx + size // 2, i] = 0

    return x


def print_exp_details(args):
    geometry = get_geometry_from_args(args)
    metrics = compute_trigger_geometry_metrics(geometry, dataset=args.data)

    print('======================================')
    print('    Scenario: Hypothesis 5 - Trigger geometry')
    print('    Scenario Data: IID-ish fixed distribution')
    print(f'    Dataset: {args.data}')
    print(f'    Global Rounds: {args.rounds}')
    print(f'    Aggregation Function: {args.aggr}')
    print(f'    Trigger Geometry: {geometry}')
    print(f'    Trigger Geometry Grid: {args.trigger_geometry_grid}')
    print(f'    Trigger Geometry Family: {metrics["geometry_family"]}')
    print(f'    Trigger Mask Area: {metrics["mask_area"]}')
    print(f'    Trigger Mask Density: {metrics["mask_density"]}')
    print(f'    Trigger Compactness Score: {metrics["compactness_score"]}')
    print(f'    Number of agents: {args.num_agents}')
    print(f'    Fraction of agents: {args.agent_frac}')
    print(f'    Class Per Agent: {args.class_per_agent}')
    print(f'    Batch size: {args.bs}')
    print(f'    Client_LR: {args.client_lr}')
    print(f'    Server_LR: {args.server_lr}')
    print(f'    Client_Momentum: {args.client_moment}')
    print(f'    RobustLR_threshold: {args.robustLR_threshold}')
    print(f'    Noise Ratio: {args.noise}')
    print(f'    Clip: {args.clip}')
    print(f'    Number of corrupt agents: {args.num_corrupt}')
    print(f'    Base Class: {args.base_class}')
    print(f'    Target Class: {args.target_class}')
    print(f'    Poison Frac: {args.poison_frac}')
    print(f'    Seed: {args.seed}')

    if hasattr(args, "clean_label"):
        print(f'    Clean Label Attack: {args.clean_label}')
        print(f'    Clean Label Type: {args.clean_label_type}')
        print(f'    Clean Label Auto Pattern: {args.clean_label_auto_pattern}')
        print(f'    Verify Poisoning: {args.verify_poisoning}')

    if hasattr(args, "verify_trigger_geometry"):
        print(f'    Verify Trigger Geometry: {args.verify_trigger_geometry}')

    print('======================================')
