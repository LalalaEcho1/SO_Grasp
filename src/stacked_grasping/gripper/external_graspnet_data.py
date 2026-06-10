from __future__ import annotations

import io
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image


DEFAULT_FACTOR_DEPTH = 1000


@dataclass(frozen=True)
class RealSenseFrame:
    frame: str
    color: np.ndarray
    depth_raw: np.ndarray
    intrinsic_matrix: np.ndarray
    label: np.ndarray | None = None
    camera_pose: np.ndarray | None = None
    cam0_wrt_table: np.ndarray | None = None
    factor_depth: int = DEFAULT_FACTOR_DEPTH

    @property
    def depth_meters(self) -> np.ndarray:
        return self.depth_raw.astype(np.float32) / float(self.factor_depth)


@dataclass(frozen=True)
class AnnotationObject:
    object_id: int
    label_id: int
    name: str
    position: np.ndarray
    model_path: str | None = None
    orientation_quat_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))


class GraspNetRealSenseSource:
    """Read a GraspNet-style RealSense folder from a directory or zip file."""

    def __init__(self, path: str | Path, *, zip_handle: zipfile.ZipFile | None = None):
        self.path = Path(path)
        self._zip = zip_handle
        self._zip_names = set(zip_handle.namelist()) if zip_handle is not None else set()
        if self._zip is None:
            self._root_dir = _find_realsense_directory(self.path)
            self.root_prefix = self._root_dir.name
        else:
            self._root_dir = None
            self.root_prefix = _find_zip_realsense_prefix(self._zip_names)

    @classmethod
    def open(cls, path: str | Path) -> "GraspNetRealSenseSource":
        source_path = Path(path)
        if source_path.suffix.lower() == ".zip":
            return cls(source_path, zip_handle=zipfile.ZipFile(source_path))
        return cls(source_path)

    def __enter__(self) -> "GraspNetRealSenseSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def list_frames(self) -> list[str]:
        rgb = self._frame_stems("rgb", ".png")
        depth = self._frame_stems("depth", ".png")
        return sorted(rgb & depth)

    def load_frame(self, frame: int | str) -> RealSenseFrame:
        frame_name = normalize_frame_id(frame)
        color = self._read_png_array(f"rgb/{frame_name}.png")
        depth = self._read_png_array(f"depth/{frame_name}.png")
        intrinsic = self._read_npy_array("camK.npy")
        label = self._read_png_array(f"label/{frame_name}.png") if self.exists(f"label/{frame_name}.png") else None

        camera_pose = None
        if self.exists("camera_poses.npy"):
            poses = self._read_npy_array("camera_poses.npy")
            frame_index = int(frame_name)
            if poses.ndim == 3 and frame_index < poses.shape[0]:
                camera_pose = poses[frame_index]

        cam0_wrt_table = self._read_npy_array("cam0_wrt_table.npy") if self.exists("cam0_wrt_table.npy") else None
        return RealSenseFrame(
            frame=frame_name,
            color=_ensure_rgb(color),
            depth_raw=np.asarray(depth, dtype=np.uint16),
            label=np.asarray(label) if label is not None else None,
            intrinsic_matrix=np.asarray(intrinsic, dtype=float),
            camera_pose=np.asarray(camera_pose, dtype=float) if camera_pose is not None else None,
            cam0_wrt_table=np.asarray(cam0_wrt_table, dtype=float) if cam0_wrt_table is not None else None,
        )

    def load_annotation_objects(self, frame: int | str) -> list[AnnotationObject]:
        frame_name = normalize_frame_id(frame)
        return parse_graspnet_annotation_xml(self._read_bytes(f"annotations/{frame_name}.xml").decode("utf-8"))

    def exists(self, relative_path: str) -> bool:
        if self._zip is not None:
            return self._zip_name(relative_path) in self._zip_names
        assert self._root_dir is not None
        return (self._root_dir / relative_path).exists()

    def _frame_stems(self, folder: str, suffix: str) -> set[str]:
        if self._zip is not None:
            prefix = f"{self.root_prefix}/{folder}/"
            return {
                PurePosixPath(name).stem
                for name in self._zip_names
                if name.startswith(prefix) and name.lower().endswith(suffix)
            }
        assert self._root_dir is not None
        target = self._root_dir / folder
        if not target.exists():
            return set()
        return {path.stem for path in target.glob(f"*{suffix}")}

    def _read_png_array(self, relative_path: str) -> np.ndarray:
        data = self._read_bytes(relative_path)
        return np.array(Image.open(io.BytesIO(data)))

    def _read_npy_array(self, relative_path: str) -> np.ndarray:
        data = self._read_bytes(relative_path)
        return np.load(io.BytesIO(data), allow_pickle=False)

    def _read_bytes(self, relative_path: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(self._zip_name(relative_path))
        assert self._root_dir is not None
        return (self._root_dir / relative_path).read_bytes()

    def _zip_name(self, relative_path: str) -> str:
        return f"{self.root_prefix}/{relative_path}".replace("\\", "/")


def normalize_frame_id(frame: int | str) -> str:
    if isinstance(frame, int):
        return f"{frame:04d}"
    text = str(frame)
    if text.isdigit():
        return f"{int(text):04d}"
    return Path(text).stem


def depth_to_point_cloud(
    depth_raw: np.ndarray,
    intrinsic_matrix: np.ndarray,
    *,
    factor_depth: int = DEFAULT_FACTOR_DEPTH,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_raw)
    intrinsic = np.asarray(intrinsic_matrix, dtype=float)
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic_matrix must have shape (3, 3).")

    depth_m = depth.astype(np.float32) / float(factor_depth)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    rows, cols = np.nonzero(valid)
    z = depth_m[rows, cols]
    x = (cols.astype(np.float32) - float(intrinsic[0, 2])) * z / float(intrinsic[0, 0])
    y = (rows.astype(np.float32) - float(intrinsic[1, 2])) * z / float(intrinsic[1, 1])
    return np.stack([x, y, z], axis=1).astype(np.float32), valid


def summarize_realsense_frame(frame: RealSenseFrame, *, min_boundary_pixels: int = 50) -> dict[str, object]:
    points, valid_mask = depth_to_point_cloud(
        frame.depth_raw,
        frame.intrinsic_matrix,
        factor_depth=frame.factor_depth,
    )
    valid_depth = frame.depth_raw[frame.depth_raw > 0]
    summary: dict[str, object] = {
        "frame": frame.frame,
        "rgb_shape": list(frame.color.shape),
        "depth_shape": list(frame.depth_raw.shape),
        "depth_dtype": str(frame.depth_raw.dtype),
        "depth_raw_min": int(valid_depth.min()) if valid_depth.size else None,
        "depth_raw_max": int(frame.depth_raw.max()) if frame.depth_raw.size else None,
        "valid_depth_pixels": int(valid_mask.sum()),
        "point_cloud_points": int(points.shape[0]),
        "point_cloud_bounds_m": _point_bounds(points),
    }
    if frame.label is not None:
        label_counts = visible_label_counts(frame.label)
        edges = visible_boundary_edges(frame.label, frame.depth_meters, min_boundary_pixels=min_boundary_pixels)
        summary.update(
            {
                "visible_label_count": len(label_counts),
                "top_visible_labels": label_counts[:10],
                "visible_boundary_edge_count": len(edges),
                "top_visible_boundary_edges": edges[:10],
            }
        )
    return summary


def parse_graspnet_annotation_xml(xml_text: str) -> list[AnnotationObject]:
    root = ET.fromstring(xml_text)
    objects: list[AnnotationObject] = []
    for obj_node in root.findall("obj"):
        object_id = int(_required_text(obj_node, "obj_id"))
        name = _optional_text(obj_node, "obj_name", default=f"object_{object_id}")
        model_path = _optional_text(obj_node, "obj_path", default=None)
        position = np.fromstring(_required_text(obj_node, "pos_in_world"), sep=" ", dtype=float)
        if position.shape != (3,):
            raise ValueError(f"Annotation object {object_id} pos_in_world must contain 3 values.")
        orientation = _parse_annotation_quat_wxyz(obj_node, object_id)
        objects.append(
            AnnotationObject(
                object_id=object_id,
                label_id=object_id + 1,
                name=name,
                position=position,
                model_path=model_path,
                orientation_quat_wxyz=orientation,
            )
        )
    return objects


def assess_single_view_od_sufficiency(
    frame: RealSenseFrame,
    annotation_objects: Sequence[AnnotationObject],
    *,
    min_boundary_pixels: int = 50,
) -> dict[str, object]:
    if frame.label is None:
        raise ValueError("Single-view OD sufficiency assessment requires a label image.")

    complete_label_ids = sorted({obj.label_id for obj in annotation_objects})
    complete_pairs = {tuple(pair) for pair in combinations(complete_label_ids, 2)}
    visible_label_ids = sorted({item["label"] for item in visible_label_counts(frame.label) if item["label"] in complete_label_ids})
    visible_set = set(visible_label_ids)
    hidden_label_ids = sorted(set(complete_label_ids) - visible_set)

    boundary_edges = visible_boundary_edges(
        frame.label,
        frame.depth_meters,
        min_boundary_pixels=min_boundary_pixels,
    )
    direct_boundary_pairs = {
        tuple(edge["pair"])
        for edge in boundary_edges
        if tuple(edge["pair"]) in complete_pairs
    }
    hidden_object_pairs = {pair for pair in complete_pairs if pair[0] in hidden_label_ids or pair[1] in hidden_label_ids}
    visible_nonboundary_pairs = {
        pair
        for pair in complete_pairs
        if pair[0] in visible_set and pair[1] in visible_set and pair not in direct_boundary_pairs
    }
    unobservable_pairs = complete_pairs - direct_boundary_pairs
    support_or_contact_not_fully_observable = bool(complete_pairs)

    reasons = []
    if hidden_label_ids:
        reasons.append("hidden_objects_present")
    if visible_nonboundary_pairs:
        reasons.append("visible_objects_without_direct_boundary")
    if support_or_contact_not_fully_observable:
        reasons.append("bottom_backside_contact_not_observable_from_single_view")

    complete_pair_count = len(complete_pairs)
    direct_pair_count = len(direct_boundary_pairs)
    return {
        "frame": frame.frame,
        "complete_object_count": len(complete_label_ids),
        "visible_object_count": len(visible_label_ids),
        "hidden_object_count": len(hidden_label_ids),
        "complete_label_ids": complete_label_ids,
        "visible_label_ids": visible_label_ids,
        "hidden_label_ids": hidden_label_ids,
        "complete_pair_count": complete_pair_count,
        "direct_visible_boundary_pair_count": direct_pair_count,
        "hidden_object_pair_count": len(hidden_object_pairs),
        "visible_nonboundary_pair_count": len(visible_nonboundary_pairs),
        "unobservable_pair_count": len(unobservable_pairs),
        "direct_pair_observability_ratio": float(direct_pair_count / complete_pair_count) if complete_pair_count else 1.0,
        "direct_visible_boundary_pairs": [list(pair) for pair in sorted(direct_boundary_pairs)],
        "hidden_object_pairs": [list(pair) for pair in sorted(hidden_object_pairs)],
        "visible_nonboundary_pairs": [list(pair) for pair in sorted(visible_nonboundary_pairs)],
        "single_view_sufficient_for_complete_od": not reasons,
        "insufficiency_reasons": reasons,
    }


def visible_label_counts(label: np.ndarray) -> list[dict[str, int]]:
    labels, counts = np.unique(np.asarray(label)[np.asarray(label) > 0], return_counts=True)
    pairs = sorted(zip(labels.tolist(), counts.tolist()), key=lambda item: (-item[1], item[0]))
    return [{"label": int(label_id), "pixels": int(count)} for label_id, count in pairs]


def visible_boundary_edges(
    label: np.ndarray,
    depth_meters: np.ndarray,
    *,
    min_boundary_pixels: int = 50,
) -> list[dict[str, object]]:
    label_array = np.asarray(label)
    depth = np.asarray(depth_meters, dtype=float)
    if label_array.shape != depth.shape:
        raise ValueError("label and depth_meters must have the same shape.")

    stats: dict[tuple[int, int], dict[str, object]] = {}
    _accumulate_label_edges(stats, label_array[:, :-1], label_array[:, 1:], depth[:, :-1], depth[:, 1:])
    _accumulate_label_edges(stats, label_array[:-1, :], label_array[1:, :], depth[:-1, :], depth[1:, :])

    edges: list[dict[str, object]] = []
    for (a, b), item in stats.items():
        count = int(item["count"])
        if count < min_boundary_pixels:
            continue
        votes = item["near_votes"]
        near_id = None
        near_votes = 0
        if votes:
            near_id, near_votes = max(votes.items(), key=lambda pair: pair[1])
        edges.append(
            {
                "pair": [a, b],
                "boundary_pixels": count,
                "nearer_label_by_boundary_depth": near_id,
                "near_vote_ratio": float(near_votes / count) if count else 0.0,
            }
        )
    return sorted(edges, key=lambda edge: (-int(edge["boundary_pixels"]), edge["pair"]))


def summarize_graspnet_prediction_package(path: str | Path) -> dict[str, object]:
    arrays = list(_iter_prediction_arrays(Path(path)))
    if not arrays:
        raise ValueError(f"No .npy GraspNet prediction arrays found in {path}.")

    normalized = [_ensure_grasp_array(array, source) for source, array in arrays]
    all_grasps = np.concatenate([array for _, array in normalized], axis=0)
    shape_each_unique = sorted({tuple(array.shape) for _, array in normalized})
    object_ids = sorted({_format_object_id(value) for value in all_grasps[:, 16]})
    rotation = all_grasps[:, 4:13].reshape(-1, 3, 3)
    rotation_error = np.linalg.norm(np.matmul(rotation.transpose(0, 2, 1), rotation) - np.eye(3), axis=(1, 2))

    return {
        "path": str(path),
        "file_count": len(normalized),
        "total_grasps": int(all_grasps.shape[0]),
        "shape_each_unique": [list(shape) for shape in shape_each_unique],
        "score": _range_summary(all_grasps[:, 0]),
        "width": _range_summary(all_grasps[:, 1]),
        "height": _range_summary(all_grasps[:, 2]),
        "depth": _range_summary(all_grasps[:, 3]),
        "translation_bounds_m": {
            "min": all_grasps[:, 13:16].min(axis=0).astype(float).tolist(),
            "mean": all_grasps[:, 13:16].mean(axis=0).astype(float).tolist(),
            "max": all_grasps[:, 13:16].max(axis=0).astype(float).tolist(),
        },
        "object_ids": object_ids,
        "rotation_orthonormal_error": {
            "max": float(rotation_error.max()),
            "mean": float(rotation_error.mean()),
        },
        "top_scores": [float(score) for score in np.sort(all_grasps[:, 0])[-5:][::-1]],
    }


def _find_realsense_directory(path: Path) -> Path:
    if path.name == "realsense" and path.is_dir():
        return path
    nested = path / "realsense"
    if nested.is_dir():
        return nested
    raise FileNotFoundError(f"Cannot find a GraspNet realsense directory under {path}.")


def _find_zip_realsense_prefix(names: Iterable[str]) -> str:
    prefixes = set()
    for name in names:
        parts = PurePosixPath(name).parts
        if "realsense" in parts:
            index = parts.index("realsense")
            prefixes.add("/".join(parts[: index + 1]))
    if not prefixes:
        raise FileNotFoundError("Cannot find a GraspNet realsense folder in zip file.")
    return sorted(prefixes, key=lambda item: (item.count("/"), item))[0]


def _required_text(node: ET.Element, child_name: str) -> str:
    text = _optional_text(node, child_name, default="")
    if text == "":
        raise ValueError(f"Annotation object is missing {child_name}.")
    return text


def _optional_text(node: ET.Element, child_name: str, *, default: str | None) -> str | None:
    child = node.find(child_name)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _parse_annotation_quat_wxyz(node: ET.Element, object_id: int) -> np.ndarray:
    raw = _optional_text(node, "ori_in_world", default="")
    if raw == "":
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    quat = np.fromstring(raw, sep=" ", dtype=float)
    if quat.shape != (4,):
        raise ValueError(f"Annotation object {object_id} ori_in_world must contain 4 values.")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-9:
        raise ValueError(f"Annotation object {object_id} ori_in_world must be a non-zero quaternion.")
    return quat / norm


def _ensure_rgb(color: np.ndarray) -> np.ndarray:
    array = np.asarray(color)
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] == 4:
        return array[:, :, :3].astype(np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("RGB image must have shape (height, width, 3).")
    return array.astype(np.uint8)


def _point_bounds(points: np.ndarray) -> dict[str, list[float] | None]:
    if points.size == 0:
        return {"x": None, "y": None, "z": None}
    return {
        "x": [float(points[:, 0].min()), float(points[:, 0].max())],
        "y": [float(points[:, 1].min()), float(points[:, 1].max())],
        "z": [float(points[:, 2].min()), float(points[:, 2].max())],
    }


def _accumulate_label_edges(
    stats: dict[tuple[int, int], dict[str, object]],
    left_label: np.ndarray,
    right_label: np.ndarray,
    left_depth: np.ndarray,
    right_depth: np.ndarray,
) -> None:
    mask = (
        (left_label > 0)
        & (right_label > 0)
        & (left_label != right_label)
        & np.isfinite(left_depth)
        & np.isfinite(right_depth)
        & (left_depth > 0)
        & (right_depth > 0)
    )
    for a, b, da, db in zip(left_label[mask], right_label[mask], left_depth[mask], right_depth[mask]):
        pair = (int(min(a, b)), int(max(a, b)))
        item = stats.setdefault(pair, {"count": 0, "near_votes": defaultdict(int)})
        item["count"] = int(item["count"]) + 1
        if da < db:
            item["near_votes"][int(a)] += 1
        elif db < da:
            item["near_votes"][int(b)] += 1


def _iter_prediction_arrays(path: Path) -> Iterable[tuple[str, np.ndarray]]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in sorted(n for n in zf.namelist() if n.lower().endswith(".npy")):
                yield name, np.load(io.BytesIO(zf.read(name)), allow_pickle=False)
        return
    if path.is_file():
        yield str(path), np.load(path, allow_pickle=False)
        return
    for file_path in sorted(path.rglob("*.npy")):
        yield str(file_path), np.load(file_path, allow_pickle=False)


def _ensure_grasp_array(array: np.ndarray, source: str) -> tuple[str, np.ndarray]:
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 17:
        raise ValueError(f"GraspNet prediction array {source} must have shape (N, 17).")
    return source, arr


def _range_summary(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def _format_object_id(value: float) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number
