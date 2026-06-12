from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from stacked_grasping.gripper.external_graspnet_data import AnnotationObject
from stacked_grasping.gripper.franka_hand import (
    FRANKA_HAND_BACKEND,
    FrankaHandConfig,
    append_franka_hand_to_mujoco_root,
    franka_hand_config_from_graspnet_candidate,
    resolve_franka_hand_xml,
)
from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate
from stacked_grasping.gripper.robotiq_2f85_lite import Robotiq2F85LiteConfig, build_gripper_body


DEFAULT_MESH_FILE = "textured.obj"
DEFAULT_TABLE_FRICTION = "2.0 0.02 0.001"
DEFAULT_OBJECT_FRICTION = "3.0 0.05 0.001"
DEFAULT_CONTACT_CONDIM = "6"
DEFAULT_CONTACT_SOLREF = "0.004 1"
DEFAULT_CONTACT_SOLIMP = "0.95 0.99 0.001"
LITE_GRIPPER_BACKEND = "lite"


def resolve_graspnet_model_mesh(
    annotation: AnnotationObject,
    dataset_root: str | Path,
    *,
    mesh_file: str = DEFAULT_MESH_FILE,
) -> Path:
    """Resolve a GraspNet annotation object to the official models/NNN mesh."""

    root = Path(dataset_root)
    candidates = [
        root / "models" / f"{int(annotation.object_id):03d}" / mesh_file,
    ]
    if annotation.model_path:
        annotated_path = Path(annotation.model_path)
        if annotated_path.is_absolute():
            candidates.append(annotated_path)
        else:
            candidates.append(root / annotated_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No GraspNet model mesh found for object {annotation.object_id}. Searched:\n  {searched}")


def build_graspnet_mujoco_scene_xml(
    annotation_objects: Sequence[AnnotationObject],
    *,
    dataset_root: str | Path,
    output_path: str | Path | None = None,
    selected_grasp: GraspPoseCandidate | None = None,
    gripper_config: Robotiq2F85LiteConfig | None = None,
    gripper_backend: str = LITE_GRIPPER_BACKEND,
    franka_hand_config: FrankaHandConfig | None = None,
    franka_hand_xml: str | Path | None = None,
    selected_grasp_controls_gripper_pose: bool = True,
    include_freejoints: bool = True,
    mesh_file: str = DEFAULT_MESH_FILE,
) -> str:
    root = ET.Element("mujoco", {"model": "graspnet_annotation_scene"})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    ET.SubElement(root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81"})
    asset = ET.SubElement(root, "asset")
    _append_materials(asset)
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "support_plane",
            "type": "plane",
            "size": "0.70 0.55 0.01",
            "material": "table_mat",
            "friction": DEFAULT_TABLE_FRICTION,
            "condim": DEFAULT_CONTACT_CONDIM,
            "solref": DEFAULT_CONTACT_SOLREF,
            "solimp": DEFAULT_CONTACT_SOLIMP,
        },
    )

    for annotation in annotation_objects:
        mesh_path = resolve_graspnet_model_mesh(annotation, dataset_root, mesh_file=mesh_file)
        mesh_name = f"obj_{int(annotation.object_id):03d}_mesh"
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": _mesh_file_for_xml(mesh_path, output_path)})

        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": mujoco_body_name_for_annotation(annotation),
                "pos": _fmt_vec(annotation.position),
                "quat": _fmt_quat(annotation.orientation_quat_wxyz),
            },
        )
        if include_freejoints:
            ET.SubElement(body, "freejoint", {"name": f"obj_{int(annotation.object_id):03d}_freejoint"})
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"obj_{int(annotation.object_id):03d}_geom",
                "type": "mesh",
                "mesh": mesh_name,
                "material": "object_mat",
                "density": "500",
                "friction": DEFAULT_OBJECT_FRICTION,
                "condim": DEFAULT_CONTACT_CONDIM,
                "solref": DEFAULT_CONTACT_SOLREF,
                "solimp": DEFAULT_CONTACT_SOLIMP,
            },
        )

    if selected_grasp is not None and gripper_backend == LITE_GRIPPER_BACKEND:
        cfg = gripper_config or Robotiq2F85LiteConfig()
        if selected_grasp_controls_gripper_pose:
            gripper_cfg = replace(
                cfg,
                pos=_tuple3(selected_grasp.position),
                quat=_tuple4(selected_grasp.orientation_quat_wxyz),
            )
        else:
            gripper_cfg = cfg
        worldbody.append(build_gripper_body(gripper_cfg))
    elif selected_grasp is not None and gripper_backend == FRANKA_HAND_BACKEND:
        hand_xml = resolve_franka_hand_xml(explicit_path=franka_hand_xml)
        cfg = franka_hand_config or franka_hand_config_from_graspnet_candidate(
            selected_grasp,
            include_freejoint=True,
        )
        append_franka_hand_to_mujoco_root(
            root,
            hand_xml_path=hand_xml,
            output_path=output_path,
            config=cfg,
        )
    elif selected_grasp is not None:
        raise ValueError(f"Unsupported gripper_backend: {gripper_backend}")

    _indent(root)
    return ET.tostring(root, encoding="unicode")


def mujoco_body_name_for_annotation(annotation: AnnotationObject) -> str:
    stem = Path(annotation.name).stem
    safe_stem = re.sub(r"[^0-9A-Za-z_]+", "_", stem).strip("_") or "object"
    return f"obj_{int(annotation.object_id):03d}_{safe_stem}"


def transform_annotation_objects(
    annotation_objects: Sequence[AnnotationObject],
    transform_matrix: np.ndarray,
) -> list[AnnotationObject]:
    transform = np.asarray(transform_matrix, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("transform_matrix must have shape (4, 4).")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    transformed: list[AnnotationObject] = []
    for annotation in annotation_objects:
        object_rotation = _quat_wxyz_to_rotation(annotation.orientation_quat_wxyz)
        transformed.append(
            replace(
                annotation,
                position=rotation @ np.asarray(annotation.position, dtype=float).reshape(3) + translation,
                orientation_quat_wxyz=_rotation_matrix_to_quat_wxyz(rotation @ object_rotation),
            )
        )
    return transformed


def robotiq_lite_config_from_graspnet_candidate(
    grasp: GraspPoseCandidate,
    *,
    include_freejoint: bool = False,
    opening_margin: float = 0.004,
    max_opening: float = 0.085,
    min_opening: float = 0.02,
    grasp_center_local: tuple[float, float, float] = (0.0, 0.0, -0.143),
) -> Robotiq2F85LiteConfig:
    grasp_rotation = _quat_wxyz_to_rotation(grasp.orientation_quat_wxyz)
    approach = _normalized(np.asarray(grasp.approach_direction, dtype=float).reshape(3))
    if float(np.linalg.norm(approach)) <= 1e-9:
        approach = _normalized(grasp_rotation[:, 0])
    closing = grasp_rotation[:, 1] - approach * float(np.dot(grasp_rotation[:, 1], approach))
    closing = _normalized(closing)
    if float(np.linalg.norm(closing)) <= 1e-9:
        closing = _any_perpendicular(approach)
    span = _normalized(np.cross(approach, closing))
    body_rotation = np.column_stack((span, closing, -approach))
    local_grasp_center = np.asarray(grasp_center_local, dtype=float).reshape(3)
    body_position = np.asarray(grasp.position, dtype=float).reshape(3) - body_rotation @ local_grasp_center
    opening = min(max(float(grasp.required_opening) + float(opening_margin), float(min_opening)), float(max_opening))
    return Robotiq2F85LiteConfig(
        pos=_tuple3(body_position),
        quat=_tuple4(_rotation_matrix_to_quat_wxyz(body_rotation)),
        include_freejoint=include_freejoint,
        opening=opening,
    )


def write_graspnet_mujoco_scene_xml(
    path: str | Path,
    annotation_objects: Sequence[AnnotationObject],
    *,
    dataset_root: str | Path,
    selected_grasp: GraspPoseCandidate | None = None,
    gripper_config: Robotiq2F85LiteConfig | None = None,
    gripper_backend: str = LITE_GRIPPER_BACKEND,
    franka_hand_config: FrankaHandConfig | None = None,
    franka_hand_xml: str | Path | None = None,
    selected_grasp_controls_gripper_pose: bool = True,
    include_freejoints: bool = True,
    mesh_file: str = DEFAULT_MESH_FILE,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    xml_text = build_graspnet_mujoco_scene_xml(
        annotation_objects,
        dataset_root=dataset_root,
        output_path=output,
        selected_grasp=selected_grasp,
        gripper_config=gripper_config,
        gripper_backend=gripper_backend,
        franka_hand_config=franka_hand_config,
        franka_hand_xml=franka_hand_xml,
        selected_grasp_controls_gripper_pose=selected_grasp_controls_gripper_pose,
        include_freejoints=include_freejoints,
        mesh_file=mesh_file,
    )
    output.write_text(xml_text, encoding="utf-8")
    return output


def _append_materials(asset: ET.Element) -> None:
    ET.SubElement(asset, "material", {"name": "table_mat", "rgba": "0.55 0.56 0.55 1"})
    ET.SubElement(asset, "material", {"name": "object_mat", "rgba": "0.82 0.78 0.68 1"})
    ET.SubElement(asset, "material", {"name": "robotiq_dark", "rgba": "0.08 0.08 0.09 1"})
    ET.SubElement(asset, "material", {"name": "robotiq_pad", "rgba": "0.02 0.02 0.02 1"})


def _mesh_file_for_xml(mesh_path: Path, output_path: str | Path | None) -> str:
    absolute = mesh_path.resolve()
    if output_path is None:
        return absolute.as_posix()
    output_dir = Path(output_path).resolve().parent
    return Path(os.path.relpath(absolute, start=output_dir)).as_posix()


def _fmt_vec(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.6f}" for value in values)


def _fmt_quat(values: Iterable[float]) -> str:
    quat = np.asarray(list(values), dtype=float).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-9:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        quat = quat / norm
    return _fmt_vec(quat)


def _quat_wxyz_to_rotation(values: Iterable[float]) -> np.ndarray:
    w, x, y, z = _normalized_quat(values)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _rotation_matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=float,
        )
    else:
        diag = np.diag(matrix)
        axis = int(np.argmax(diag))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=float,
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=float,
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=float,
            )
    return _normalized_quat(quat)


def _normalized_quat(values: Iterable[float]) -> np.ndarray:
    quat = np.asarray(list(values), dtype=float).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


def _tuple3(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float).reshape(3)
    return (float(array[0]), float(array[1]), float(array[2]))


def _tuple4(values: Iterable[float]) -> tuple[float, float, float, float]:
    array = np.asarray(list(values), dtype=float).reshape(4)
    norm = float(np.linalg.norm(array))
    if norm > 1e-9:
        array = array / norm
    return (float(array[0]), float(array[1]), float(array[2]), float(array[3]))


def _normalized(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return np.zeros_like(arr)
    return arr / norm


def _any_perpendicular(axis: np.ndarray) -> np.ndarray:
    vector = _normalized(np.asarray(axis, dtype=float))
    reference = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(vector, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    return _normalized(np.cross(vector, reference))


def _indent(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            _indent(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent
