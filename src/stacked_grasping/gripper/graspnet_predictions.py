from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from stacked_grasping.gripper.candidate_io import load_graspnet_candidates, load_graspnet_records
from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate, assign_candidates_to_objects, graspnet_outputs_to_candidates
from stacked_grasping.relations.geometry import ObjectState


PREDICTION_SUFFIXES = (".npy", ".npz", ".json")


def scene_key_from_path(scene_path: str | Path) -> str:
    return Path(scene_path).stem


def prediction_file_candidates(
    prediction_root: str | Path,
    scene_path: str | Path,
    *,
    camera: str = "realsense",
    view_id: int | str = 0,
) -> list[Path]:
    root = Path(prediction_root)
    scene_key = scene_key_from_path(scene_path)
    view_name = _view_filename(view_id)
    candidates = []
    for suffix in PREDICTION_SUFFIXES:
        candidates.extend(
            [
                root / scene_key / camera / f"{view_name}{suffix}",
                root / scene_key / f"{view_name}{suffix}",
                root / f"{scene_key}{suffix}",
                root / scene_key / f"grasps{suffix}",
            ]
        )
    return candidates


def find_scene_prediction_file(
    prediction_root: str | Path,
    scene_path: str | Path,
    *,
    camera: str = "realsense",
    view_id: int | str = 0,
) -> Path:
    for path in prediction_file_candidates(prediction_root, scene_path, camera=camera, view_id=view_id):
        if path.exists():
            return path
    searched = "\n  ".join(str(path) for path in prediction_file_candidates(prediction_root, scene_path, camera=camera, view_id=view_id))
    raise FileNotFoundError(f"No GraspNet prediction file found. Searched:\n  {searched}")


def load_scene_prediction_candidates(
    prediction_root: str | Path,
    scene_path: str | Path,
    *,
    objects: Sequence[ObjectState] | None = None,
    metadata_root: str | Path | None = None,
    object_id_to_name: Mapping[int | str, str] | None = None,
    camera: str = "realsense",
    view_id: int | str = 0,
    assign_margin: float = 0.0,
    pregrasp_distance: float = 0.12,
    generator: str = "graspnet-prediction",
) -> Dict[str, list[GraspPoseCandidate]]:
    prediction_file = find_scene_prediction_file(prediction_root, scene_path, camera=camera, view_id=view_id)
    records = load_graspnet_records(prediction_file, object_id_to_name=object_id_to_name)
    camera_to_world = load_camera_to_world_matrix(metadata_root, scene_path) if metadata_root is not None else None
    if camera_to_world is not None:
        records = transform_graspnet_records(records, camera_to_world)

    if objects is None:
        candidates = load_graspnet_candidates(
            prediction_file,
            object_id_to_name=object_id_to_name,
            pregrasp_distance=pregrasp_distance,
            generator=generator,
        )
        if camera_to_world is not None:
            candidates = graspnet_outputs_to_candidates(
                records,
                pregrasp_distance=pregrasp_distance,
                generator=generator,
            )
        return _group_candidates_by_object(candidates)

    candidates = graspnet_outputs_to_candidates(
        records,
        pregrasp_distance=pregrasp_distance,
        generator=generator,
    )
    return assign_candidates_to_objects(objects, candidates, margin=assign_margin)


def load_camera_to_world_matrix(metadata_root: str | Path, scene_path: str | Path) -> np.ndarray | None:
    metadata_path = Path(metadata_root) / scene_key_from_path(scene_path) / "metadata.json"
    if not metadata_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    matrix = payload.get("camera_to_world_matrix")
    if matrix is None:
        return None
    camera_to_world = np.asarray(matrix, dtype=float)
    if camera_to_world.shape != (4, 4):
        raise ValueError(f"camera_to_world_matrix in {metadata_path} must have shape (4, 4).")
    return camera_to_world


def transform_graspnet_records(records: Sequence[Mapping[str, object]], camera_to_world_matrix: np.ndarray) -> list[Dict[str, object]]:
    transform = np.asarray(camera_to_world_matrix, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("camera_to_world_matrix must have shape (4, 4).")

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    transformed = []
    for record in records:
        item = dict(record)
        local_rotation = np.asarray(item["rotation_matrix"], dtype=float).reshape(3, 3)
        local_translation = np.asarray(item["translation"], dtype=float).reshape(3)
        item["rotation_matrix"] = (rotation @ local_rotation).tolist()
        item["translation"] = (rotation @ local_translation + translation).tolist()
        transformed.append(item)
    return transformed


def _group_candidates_by_object(candidates: Sequence[GraspPoseCandidate]) -> Dict[str, list[GraspPoseCandidate]]:
    grouped: Dict[str, list[GraspPoseCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.object_name, []).append(candidate)
    return grouped


def _view_filename(view_id: int | str) -> str:
    if isinstance(view_id, int):
        return f"{view_id:04d}"
    text = str(view_id)
    return Path(text).stem
