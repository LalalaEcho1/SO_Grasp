from __future__ import annotations

import io
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import numpy as np

from stacked_grasping.gripper.candidate_io import records_from_graspnet_array
from stacked_grasping.gripper.external_graspnet_data import (
    AnnotationObject,
    RealSenseFrame,
    depth_to_point_cloud,
    normalize_frame_id,
)
from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate, graspnet_outputs_to_candidates

BINDING_MODES = ("pixel", "3d")


@dataclass(frozen=True)
class BoundGraspNetCandidate:
    record: Mapping[str, object]
    frame: str
    status: str
    pixel: tuple[int, int] | None
    label_id: int | None
    object_id: int | None
    object_name: str | None
    depth_error_m: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "status": self.status,
            "pixel": list(self.pixel) if self.pixel is not None else None,
            "label_id": self.label_id,
            "object_id": self.object_id,
            "object_name": self.object_name,
            "score": float(self.record.get("score", 0.0)),
            "width": float(self.record.get("width", 0.0)),
            "translation": np.asarray(self.record.get("translation", [0.0, 0.0, 0.0]), dtype=float).tolist(),
            "depth_error_m": self.depth_error_m,
        }


class GraspNetPredictionSource:
    """Read GraspNet prediction frames from a .npy file, directory, or zip package."""

    def __init__(self, path: str | Path, *, zip_handle: zipfile.ZipFile | None = None):
        self.path = Path(path)
        self._zip = zip_handle
        self._frame_to_source = self._index_frames()

    @classmethod
    def open(cls, path: str | Path) -> "GraspNetPredictionSource":
        source_path = Path(path)
        if source_path.suffix.lower() == ".zip":
            return cls(source_path, zip_handle=zipfile.ZipFile(source_path))
        return cls(source_path)

    def __enter__(self) -> "GraspNetPredictionSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def list_frames(self) -> list[str]:
        return sorted(self._frame_to_source)

    def load_records(self, frame: int | str) -> list[dict[str, object]]:
        frame_name = normalize_frame_id(frame)
        if frame_name not in self._frame_to_source:
            raise FileNotFoundError(f"No GraspNet prediction found for frame {frame_name}.")
        source = self._frame_to_source[frame_name]
        if self._zip is not None:
            assert isinstance(source, str)
            array = np.load(io.BytesIO(self._zip.read(source)), allow_pickle=False)
        else:
            array = np.load(Path(source), allow_pickle=False)
        return records_from_graspnet_array(array)

    def _index_frames(self) -> dict[str, str | Path]:
        if self._zip is not None:
            names = sorted(name for name in self._zip.namelist() if name.lower().endswith(".npy"))
            return {PurePosixPath(name).stem: name for name in names}
        if self.path.is_file():
            return {self.path.stem: self.path}
        return {path.stem: path for path in sorted(self.path.rglob("*.npy"))}


def project_camera_points_to_pixels(points: np.ndarray, intrinsic_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    intrinsic = np.asarray(intrinsic_matrix, dtype=float)
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic_matrix must have shape (3, 3).")

    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > 0.0)
    pixels = np.full((pts.shape[0], 2), np.nan, dtype=float)
    pixels[valid, 0] = intrinsic[0, 0] * pts[valid, 0] / z[valid] + intrinsic[0, 2]
    pixels[valid, 1] = intrinsic[1, 1] * pts[valid, 1] / z[valid] + intrinsic[1, 2]
    return pixels, valid


def bind_graspnet_records_to_frame_labels(
    records: Sequence[Mapping[str, object]],
    frame: RealSenseFrame,
    annotation_objects: Sequence[AnnotationObject] = (),
    *,
    pixel_radius: int = 2,
    depth_tolerance_m: float | None = 0.08,
) -> list[BoundGraspNetCandidate]:
    if frame.label is None:
        return [_unbound(record, frame.frame, "no-label") for record in records]

    translations = np.array([record.get("translation", [np.nan, np.nan, np.nan]) for record in records], dtype=float)
    pixels, valid = project_camera_points_to_pixels(translations, frame.intrinsic_matrix)
    label_to_object = {obj.label_id: obj for obj in annotation_objects}

    bound: list[BoundGraspNetCandidate] = []
    for record, point, pixel, is_valid in zip(records, translations, pixels, valid):
        if not is_valid:
            bound.append(_unbound(record, frame.frame, "invalid-translation"))
            continue
        u = int(round(float(pixel[0])))
        v = int(round(float(pixel[1])))
        if not _inside_image(u, v, frame.label.shape):
            bound.append(_unbound(record, frame.frame, "out-of-frame", pixel=(u, v)))
            continue

        match = _best_label_match(
            label=np.asarray(frame.label),
            depth_meters=frame.depth_meters,
            u=u,
            v=v,
            candidate_depth_m=float(point[2]),
            pixel_radius=pixel_radius,
        )
        if match is None:
            bound.append(_unbound(record, frame.frame, "background", pixel=(u, v)))
            continue

        label_id, depth_error = match
        if depth_tolerance_m is not None and depth_error is not None and depth_error > depth_tolerance_m:
            bound.append(_unbound(record, frame.frame, "depth-mismatch", pixel=(u, v), depth_error_m=depth_error))
            continue

        annotation = label_to_object.get(label_id)
        object_id = annotation.object_id if annotation is not None else label_id - 1
        object_name = annotation.name if annotation is not None else None
        bound.append(
            BoundGraspNetCandidate(
                record=record,
                frame=frame.frame,
                status="bound",
                pixel=(u, v),
                label_id=label_id,
                object_id=object_id,
                object_name=object_name,
                depth_error_m=depth_error,
            )
        )
    return bound


def bind_graspnet_records_to_objects_3d(
    records: Sequence[Mapping[str, object]],
    frame: RealSenseFrame,
    annotation_objects: Sequence[AnnotationObject] = (),
    *,
    max_distance_m: float = 0.05,
    point_stride: int = 4,
) -> list[BoundGraspNetCandidate]:
    """Bind each grasp to the nearest labeled 3D point instead of its pixel projection.

    Motivated by the 2026-07-26 diagnosis: pixel projection suffers from parallax —
    grasp centers frequently project onto a surface *behind* the intended object
    (signed depth offsets clustered around -0.17 m), so a single-pixel lookup binds
    to the wrong label or to background. Nearest-3D-point association is immune to
    that failure mode. ``depth_error_m`` carries the 3D distance for bound results.
    """
    if frame.label is None:
        return [_unbound(record, frame.frame, "no-label") for record in records]

    points, valid_mask = depth_to_point_cloud(
        frame.depth_raw,
        frame.intrinsic_matrix,
        factor_depth=frame.factor_depth,
    )
    labels = np.asarray(frame.label).reshape(-1)[np.asarray(valid_mask).reshape(-1)]
    labeled = labels > 0
    labeled_points = np.asarray(points, dtype=float).reshape(-1, 3)[labeled]
    labeled_ids = labels[labeled]
    stride = max(int(point_stride), 1)
    labeled_points = labeled_points[::stride]
    labeled_ids = labeled_ids[::stride]
    if labeled_points.shape[0] == 0:
        return [_unbound(record, frame.frame, "no-label") for record in records]

    label_to_object = {obj.label_id: obj for obj in annotation_objects}
    translations = np.array([record.get("translation", [np.nan, np.nan, np.nan]) for record in records], dtype=float)
    pixels, _ = project_camera_points_to_pixels(translations, frame.intrinsic_matrix)

    bound: list[BoundGraspNetCandidate] = []
    for record, point, pixel in zip(records, translations, pixels):
        if not np.isfinite(point).all():
            bound.append(_unbound(record, frame.frame, "invalid-translation"))
            continue
        distances = np.linalg.norm(labeled_points - point[None, :], axis=1)
        nearest = int(np.argmin(distances))
        nearest_distance = float(distances[nearest])
        pixel_uv = (
            (int(round(float(pixel[0]))), int(round(float(pixel[1]))))
            if np.isfinite(pixel).all()
            else None
        )
        if nearest_distance > float(max_distance_m):
            bound.append(
                _unbound(
                    record,
                    frame.frame,
                    "no-nearby-points",
                    pixel=pixel_uv,
                    depth_error_m=nearest_distance,
                )
            )
            continue
        label_id = int(labeled_ids[nearest])
        annotation = label_to_object.get(label_id)
        bound.append(
            BoundGraspNetCandidate(
                record=record,
                frame=frame.frame,
                status="bound",
                pixel=pixel_uv,
                label_id=label_id,
                object_id=annotation.object_id if annotation is not None else label_id - 1,
                object_name=annotation.name if annotation is not None else None,
                depth_error_m=nearest_distance,
            )
        )
    return bound


def bind_graspnet_records(
    records: Sequence[Mapping[str, object]],
    frame: RealSenseFrame,
    annotation_objects: Sequence[AnnotationObject] = (),
    *,
    mode: str = "pixel",
    pixel_radius: int = 2,
    depth_tolerance_m: float | None = 0.08,
    max_distance_m: float = 0.05,
    point_stride: int = 4,
) -> list[BoundGraspNetCandidate]:
    """Dispatch to pixel-projection or nearest-3D-point binding."""
    if mode == "pixel":
        return bind_graspnet_records_to_frame_labels(
            records,
            frame,
            annotation_objects,
            pixel_radius=pixel_radius,
            depth_tolerance_m=depth_tolerance_m,
        )
    if mode == "3d":
        return bind_graspnet_records_to_objects_3d(
            records,
            frame,
            annotation_objects,
            max_distance_m=max_distance_m,
            point_stride=point_stride,
        )
    valid = ", ".join(BINDING_MODES)
    raise ValueError(f"Unknown binding mode {mode!r}. Valid modes: {valid}")


def bound_candidates_to_grasp_poses_by_object(
    bindings: Sequence[BoundGraspNetCandidate],
    *,
    pregrasp_distance: float = 0.12,
    generator: str = "graspnet-bound",
) -> dict[str, list[GraspPoseCandidate]]:
    poses_by_object: dict[str, list[GraspPoseCandidate]] = {}
    for binding in bindings:
        if binding.status != "bound":
            continue
        object_name = _binding_object_name(binding)
        if object_name is None:
            continue

        record = dict(binding.record)
        record["object_name"] = object_name
        if binding.object_id is not None:
            record["object_id"] = binding.object_id
        record.setdefault("closing_axis", "6d")

        poses = graspnet_outputs_to_candidates(
            [record],
            pregrasp_distance=pregrasp_distance,
            generator=generator,
        )
        poses_by_object.setdefault(object_name, []).extend(poses)
    return poses_by_object


def summarize_bound_graspnet_candidates(
    bindings: Sequence[BoundGraspNetCandidate],
    annotation_objects: Sequence[AnnotationObject] = (),
) -> dict[str, object]:
    status_counts = dict(Counter(binding.status for binding in bindings))
    bound_items = [binding for binding in bindings if binding.status == "bound" and binding.label_id is not None]
    by_label: dict[int, list[BoundGraspNetCandidate]] = {}
    for binding in bound_items:
        by_label.setdefault(int(binding.label_id), []).append(binding)

    objects = []
    label_ids = sorted(set(by_label) | {obj.label_id for obj in annotation_objects})
    annotation_by_label = {obj.label_id: obj for obj in annotation_objects}
    for label_id in label_ids:
        items = by_label.get(label_id, [])
        scores = [float(item.record.get("score", 0.0)) for item in items]
        annotation = annotation_by_label.get(label_id)
        objects.append(
            {
                "label_id": int(label_id),
                "object_id": annotation.object_id if annotation is not None else int(label_id - 1),
                "object_name": annotation.name if annotation is not None else None,
                "candidate_count": len(items),
                "best_score": max(scores) if scores else None,
                "mean_score": float(np.mean(scores)) if scores else None,
            }
        )

    return {
        "total_candidates": len(bindings),
        "bound_count": len(bound_items),
        "unbound_count": len(bindings) - len(bound_items),
        "status_counts": status_counts,
        "objects": objects,
    }


def _best_label_match(
    *,
    label: np.ndarray,
    depth_meters: np.ndarray,
    u: int,
    v: int,
    candidate_depth_m: float,
    pixel_radius: int,
) -> tuple[int, float | None] | None:
    height, width = label.shape
    radius = max(int(pixel_radius), 0)
    candidates = []
    for yy in range(max(0, v - radius), min(height, v + radius + 1)):
        for xx in range(max(0, u - radius), min(width, u + radius + 1)):
            label_id = int(label[yy, xx])
            if label_id <= 0:
                continue
            depth_value = float(depth_meters[yy, xx])
            depth_error = abs(depth_value - candidate_depth_m) if np.isfinite(depth_value) and depth_value > 0 else None
            pixel_distance = float((xx - u) ** 2 + (yy - v) ** 2)
            candidates.append((pixel_distance, float("inf") if depth_error is None else depth_error, label_id, depth_error))
    if not candidates:
        return None
    _, _, label_id, depth_error = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return label_id, depth_error


def _binding_object_name(binding: BoundGraspNetCandidate) -> str | None:
    if binding.object_name:
        return binding.object_name
    if binding.object_id is not None:
        return f"object_{binding.object_id}"
    if binding.label_id is not None:
        return f"label_{binding.label_id}"
    return None


def _inside_image(u: int, v: int, shape: tuple[int, ...]) -> bool:
    return 0 <= v < int(shape[0]) and 0 <= u < int(shape[1])


def _unbound(
    record: Mapping[str, object],
    frame: str,
    status: str,
    *,
    pixel: tuple[int, int] | None = None,
    depth_error_m: float | None = None,
) -> BoundGraspNetCandidate:
    return BoundGraspNetCandidate(
        record=record,
        frame=frame,
        status=status,
        pixel=pixel,
        label_id=None,
        object_id=None,
        object_name=None,
        depth_error_m=depth_error_m,
    )
