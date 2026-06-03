from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate, graspnet_outputs_to_candidates


def load_graspnet_candidates(
    path: str | Path,
    object_id_to_name: Mapping[int | str, str] | None = None,
    pregrasp_distance: float = 0.12,
    generator: str = "graspnet",
) -> List[GraspPoseCandidate]:
    records = load_graspnet_records(path, object_id_to_name=object_id_to_name)
    return graspnet_outputs_to_candidates(
        records,
        pregrasp_distance=pregrasp_distance,
        generator=generator,
    )


def load_graspnet_records(
    path: str | Path,
    object_id_to_name: Mapping[int | str, str] | None = None,
) -> List[Dict[str, object]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        records = _records_from_json_payload(payload)
    elif suffix == ".npy":
        records = _records_from_graspnet_array(np.load(source, allow_pickle=False))
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            key = "grasps" if "grasps" in payload.files else payload.files[0]
            records = _records_from_graspnet_array(payload[key])
    else:
        raise ValueError(f"Unsupported GraspNet candidate file: {source}")

    return [_with_object_name(record, object_id_to_name) for record in records]


def records_from_graspnet_array(
    array: np.ndarray,
    object_id_to_name: Mapping[int | str, str] | None = None,
) -> List[Dict[str, object]]:
    records = _records_from_graspnet_array(array)
    return [_with_object_name(record, object_id_to_name) for record in records]


def _records_from_json_payload(payload: object) -> List[Dict[str, object]]:
    if isinstance(payload, list):
        raw_records = payload
    elif isinstance(payload, dict):
        raw_records = payload.get("grasps", payload.get("candidates", payload.get("records", [])))
    else:
        raise ValueError("GraspNet JSON must be a list or an object containing grasps/candidates.")

    return [_normalize_json_record(record) for record in raw_records]


def _normalize_json_record(record: object) -> Dict[str, object]:
    if not isinstance(record, dict):
        raise ValueError("Each GraspNet JSON record must be an object.")
    rotation = record.get("rotation_matrix", record.get("rotation"))
    translation = record.get("translation", record.get("center"))
    if rotation is None or translation is None or "width" not in record:
        raise ValueError("GraspNet record requires rotation_matrix, translation, and width.")

    normalized: Dict[str, object] = {
        "score": float(record.get("score", 1.0)),
        "width": float(record["width"]),
        "rotation_matrix": np.array(rotation, dtype=float).reshape(3, 3).tolist(),
        "translation": np.array(translation, dtype=float).reshape(3).tolist(),
    }
    for key in ("height", "depth", "object_id", "object_name", "closing_axis"):
        if key in record:
            normalized[key] = record[key]
    return normalized


def _records_from_graspnet_array(array: np.ndarray) -> List[Dict[str, object]]:
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 16:
        raise ValueError("GraspNet npy array must have shape (N, >=16).")

    records: List[Dict[str, object]] = []
    for row in arr:
        record: Dict[str, object] = {
            "score": float(row[0]),
            "width": float(row[1]),
            "height": float(row[2]) if row.shape[0] > 2 else 0.0,
            "depth": float(row[3]) if row.shape[0] > 3 else 0.0,
            "rotation_matrix": row[4:13].reshape(3, 3).tolist(),
            "translation": row[13:16].tolist(),
        }
        if row.shape[0] > 16:
            object_id = row[16]
            record["object_id"] = int(object_id) if float(object_id).is_integer() else float(object_id)
        records.append(record)
    return records


def _with_object_name(
    record: Dict[str, object],
    object_id_to_name: Mapping[int | str, str] | None,
) -> Dict[str, object]:
    if object_id_to_name is None or "object_id" not in record or "object_name" in record:
        return record
    object_id = record["object_id"]
    object_name = object_id_to_name.get(object_id)
    if object_name is None:
        object_name = object_id_to_name.get(str(object_id))
    if object_name is None and isinstance(object_id, str) and object_id.isdigit():
        object_name = object_id_to_name.get(int(object_id))
    if object_name is None:
        return record
    copied = dict(record)
    copied["object_name"] = object_name
    return copied
