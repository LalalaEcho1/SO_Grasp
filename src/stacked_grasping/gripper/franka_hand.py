from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate
from stacked_grasping.utils.paths import project_root


FRANKA_HAND_BACKEND = "franka-hand"
FRANKA_HAND_BODY_NAME = "franka_hand"
FRANKA_HAND_ROOT_JOINT_NAME = "franka_hand_freejoint"
FRANKA_LEFT_FINGER_JOINT_NAME = "finger_joint1"
FRANKA_RIGHT_FINGER_JOINT_NAME = "finger_joint2"
FRANKA_HAND_XML_ENV = "SO_GRASP_FRANKA_HAND_XML"

_PROJECT_FRANKA_HAND_XML = Path("external") / "senior_graspnet" / "simulation" / "dataset" / "franka" / "hand.xml"
_SERVER_FRANKA_HAND_XML_CANDIDATES = (
    Path("/home/bobo/99-students/2023_2026_gxl/Python_Program/graspnet-baseline/graspnet数据集/simulation/franka/hand.xml"),
    Path("/home/bobo/99-students/2023_2026_gxl/Python_Program/graspnet-baseline/mujoco/simulation/dataset/franka/hand.xml"),
)


@dataclass(frozen=True)
class FrankaHandConfig:
    pos: tuple[float, float, float] = (0.0, 0.0, 0.85)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    include_freejoint: bool = True


def resolve_franka_hand_xml(
    explicit_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    root: str | Path | None = None,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    else:
        values = os.environ if env is None else env
        env_path = values.get(FRANKA_HAND_XML_ENV)
        if env_path:
            candidates.append(Path(env_path))
        base = Path(root) if root is not None else project_root()
        candidates.append(base / _PROJECT_FRANKA_HAND_XML)
        candidates.extend(_SERVER_FRANKA_HAND_XML_CANDIDATES)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "No Franka/Panda hand MuJoCo XML found. "
        f"Pass --franka-hand-xml or set {FRANKA_HAND_XML_ENV}.\nSearched:\n  {searched}"
    )


def franka_hand_config_from_graspnet_candidate(
    grasp: GraspPoseCandidate,
    *,
    include_freejoint: bool = True,
    grasp_center_local: tuple[float, float, float] = (0.0, 0.0, 0.103),
) -> FrankaHandConfig:
    grasp_rotation = _quat_wxyz_to_rotation(grasp.orientation_quat_wxyz)
    approach = _normalized(np.asarray(grasp.approach_direction, dtype=float).reshape(3))
    if float(np.linalg.norm(approach)) <= 1e-9:
        approach = _normalized(grasp_rotation[:, 0])

    closing = grasp_rotation[:, 1] - approach * float(np.dot(grasp_rotation[:, 1], approach))
    closing = _normalized(closing)
    if float(np.linalg.norm(closing)) <= 1e-9:
        closing = _any_perpendicular(approach)

    span = _normalized(np.cross(closing, approach))
    body_rotation = np.column_stack((span, closing, approach))
    local_grasp_center = np.asarray(grasp_center_local, dtype=float).reshape(3)
    body_position = np.asarray(grasp.position, dtype=float).reshape(3) - body_rotation @ local_grasp_center
    return FrankaHandConfig(
        pos=_tuple3(body_position),
        quat=_tuple4(_rotation_matrix_to_quat_wxyz(body_rotation)),
        include_freejoint=include_freejoint,
    )


def append_franka_hand_to_mujoco_root(
    root: ET.Element,
    *,
    hand_xml_path: str | Path,
    output_path: str | Path | None,
    config: FrankaHandConfig | None = None,
) -> None:
    cfg = config or FrankaHandConfig()
    source_path = Path(hand_xml_path)
    source_root = ET.parse(source_path).getroot()
    source_meshdir = _source_meshdir(source_root)

    _merge_default(root, source_root)
    _merge_assets(root, source_root, source_path=source_path, source_meshdir=source_meshdir, output_path=output_path)
    _append_franka_hand_body(root, source_root, cfg)
    _append_reference_sections(root, source_root)


def _merge_default(root: ET.Element, source_root: ET.Element) -> None:
    source_default = source_root.find("default")
    if source_default is None:
        return
    destination_default = root.find("default")
    if destination_default is None:
        root.insert(_top_level_insert_index(root), copy.deepcopy(source_default))
        return
    for child in list(source_default):
        destination_default.append(copy.deepcopy(child))


def _merge_assets(
    root: ET.Element,
    source_root: ET.Element,
    *,
    source_path: Path,
    source_meshdir: str,
    output_path: str | Path | None,
) -> None:
    source_asset = source_root.find("asset")
    if source_asset is None:
        return
    destination_asset = root.find("asset")
    if destination_asset is None:
        destination_asset = ET.Element("asset")
        root.insert(_top_level_insert_index(root), destination_asset)

    for child in list(source_asset):
        copied = copy.deepcopy(child)
        if "file" in copied.attrib:
            copied.attrib["file"] = _asset_file_for_xml(
                copied.attrib["file"],
                source_path=source_path,
                source_meshdir=source_meshdir,
                output_path=output_path,
            )
        destination_asset.append(copied)


def _append_franka_hand_body(root: ET.Element, source_root: ET.Element, config: FrankaHandConfig) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Scene XML does not contain <worldbody>.")
    source_worldbody = source_root.find("worldbody")
    if source_worldbody is None:
        raise ValueError("Franka hand XML does not contain <worldbody>.")

    source_body = source_worldbody.find("body[@name='hand']")
    if source_body is None:
        source_body = source_worldbody.find("body")
    if source_body is None:
        raise ValueError("Franka hand XML does not contain a hand body.")

    hand_body = copy.deepcopy(source_body)
    _rename_body_references(hand_body, {"hand": FRANKA_HAND_BODY_NAME})
    hand_body.attrib["name"] = FRANKA_HAND_BODY_NAME
    hand_body.attrib["pos"] = _fmt_vec(config.pos)
    hand_body.attrib["quat"] = _fmt_vec(config.quat)
    if config.include_freejoint:
        _ensure_freejoint(hand_body)
    worldbody.append(hand_body)


def _append_reference_sections(root: ET.Element, source_root: ET.Element) -> None:
    for tag in ("contact", "tendon", "equality", "actuator"):
        source_section = source_root.find(tag)
        if source_section is None:
            continue
        destination_section = root.find(tag)
        if destination_section is None:
            destination_section = ET.Element(tag)
            root.append(destination_section)
        for child in list(source_section):
            copied = copy.deepcopy(child)
            _rename_body_references(copied, {"hand": FRANKA_HAND_BODY_NAME})
            destination_section.append(copied)


def _ensure_freejoint(hand_body: ET.Element) -> None:
    existing = hand_body.find("freejoint")
    if existing is not None:
        existing.attrib["name"] = FRANKA_HAND_ROOT_JOINT_NAME
        return
    freejoint = ET.Element("freejoint", {"name": FRANKA_HAND_ROOT_JOINT_NAME})
    insert_at = 0
    if len(hand_body) and hand_body[0].tag == "inertial":
        insert_at = 1
    hand_body.insert(insert_at, freejoint)


def _rename_body_references(element: ET.Element, mapping: Mapping[str, str]) -> None:
    for key, value in list(element.attrib.items()):
        if key in {"body1", "body2", "name"} and value in mapping:
            element.attrib[key] = mapping[value]
    for child in list(element):
        _rename_body_references(child, mapping)


def _source_meshdir(source_root: ET.Element) -> str:
    compiler = source_root.find("compiler")
    if compiler is None:
        return ""
    return compiler.attrib.get("meshdir", "")


def _asset_file_for_xml(
    file_value: str,
    *,
    source_path: Path,
    source_meshdir: str,
    output_path: str | Path | None,
) -> str:
    file_path = Path(file_value)
    if file_path.is_absolute():
        absolute = file_path
    else:
        absolute = (source_path.parent / source_meshdir / file_path).resolve()

    if output_path is None:
        return absolute.as_posix()
    output_dir = Path(output_path).resolve().parent
    try:
        return Path(os.path.relpath(absolute, start=output_dir)).as_posix()
    except ValueError:
        return absolute.as_posix()


def _top_level_insert_index(root: ET.Element) -> int:
    for index, child in enumerate(list(root)):
        if child.tag in {"asset", "worldbody"}:
            return index
    return len(root)


def _fmt_vec(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.6f}" for value in values)


def _tuple3(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float).reshape(3)
    return (float(array[0]), float(array[1]), float(array[2]))


def _tuple4(values: Iterable[float]) -> tuple[float, float, float, float]:
    array = np.asarray(list(values), dtype=float).reshape(4)
    return (float(array[0]), float(array[1]), float(array[2]), float(array[3]))


def _normalized(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return np.zeros_like(vector, dtype=float)
    return vector / norm


def _any_perpendicular(axis: np.ndarray) -> np.ndarray:
    axis = _normalized(axis)
    trial = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(axis, trial))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0], dtype=float)
    return _normalized(np.cross(axis, trial))


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
