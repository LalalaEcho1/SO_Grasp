from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple
import xml.etree.ElementTree as ET


Vec3 = Tuple[float, float, float]
QuatWxyz = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Robotiq2F85LiteConfig:
    pos: Vec3 = (0.0, 0.0, 0.85)
    quat: QuatWxyz | None = None
    include_freejoint: bool = False
    opening: float = 0.085
    approach_height: float = 0.20
    palm_size: Vec3 = (0.055, 0.020, 0.018)
    finger_size: Vec3 = (0.012, 0.010, 0.055)
    fingertip_size: Vec3 = (0.016, 0.012, 0.012)
    rgba_dark: str = "0.08 0.08 0.09 1"
    rgba_pad: str = "0.02 0.02 0.02 1"


def build_gripper_body(config: Robotiq2F85LiteConfig | None = None) -> ET.Element:
    cfg = config or Robotiq2F85LiteConfig()
    attributes = {"name": "robotiq_2f85_lite", "pos": _fmt_vec(cfg.pos)}
    if cfg.quat is not None:
        attributes["quat"] = _fmt_vec(cfg.quat)
    body = ET.Element("body", attributes)
    if cfg.include_freejoint:
        ET.SubElement(body, "freejoint", {"name": "robotiq_root_freejoint"})

    ET.SubElement(
        body,
        "geom",
        {
            "name": "robotiq_palm",
            "type": "box",
            "size": _fmt_vec(cfg.palm_size),
            "pos": "0 0 0",
            "material": "robotiq_dark",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    half_opening = cfg.opening / 2.0
    _append_finger(
        body,
        name="left",
        y=half_opening,
        slide_range=(0.0, half_opening),
        cfg=cfg,
    )
    _append_finger(
        body,
        name="right",
        y=-half_opening,
        slide_range=(-half_opening, 0.0),
        cfg=cfg,
    )
    return body


def attach_gripper_to_scene_xml(
    xml_text: str,
    target_object: str,
    config: Robotiq2F85LiteConfig | None = None,
) -> str:
    cfg = config or Robotiq2F85LiteConfig()
    root = ET.fromstring(xml_text)
    _ensure_gripper_materials(root, cfg)

    target_pos, target_half_extents = _target_pose_and_size(root, target_object)
    gripper_pos = (
        target_pos[0],
        target_pos[1],
        target_pos[2] + target_half_extents[2] + cfg.approach_height,
    )
    gripper = build_gripper_body(_replace_pos(cfg, gripper_pos))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Scene XML does not contain <worldbody>.")
    _remove_existing_gripper(worldbody)
    worldbody.append(gripper)
    _indent(root)
    return ET.tostring(root, encoding="unicode")


def rewrite_asset_file_paths(
    xml_text: str,
    source_xml_path: str | Path,
    output_path: str | Path | None = None,
    mode: str = "relative-to-output",
) -> str:
    root = ET.fromstring(xml_text)
    source_dir = Path(source_xml_path).resolve().parent
    output_dir = Path(output_path).resolve().parent if output_path is not None else source_dir
    if mode not in {"relative-to-output", "absolute"}:
        raise ValueError(f"Unsupported asset path mode: {mode}")

    for element in root.findall(".//*[@file]"):
        file_value = element.attrib["file"]
        file_path = Path(file_value)
        absolute_path = file_path if file_path.is_absolute() else (source_dir / file_path).resolve()
        if mode == "absolute":
            element.attrib["file"] = absolute_path.as_posix()
        else:
            element.attrib["file"] = _relative_posix_path(absolute_path, output_dir)
    _indent(root)
    return ET.tostring(root, encoding="unicode")


def _append_finger(
    parent: ET.Element,
    name: str,
    y: float,
    slide_range: Tuple[float, float],
    cfg: Robotiq2F85LiteConfig,
) -> None:
    sign = 1.0 if y >= 0 else -1.0
    finger = ET.SubElement(parent, "body", {"name": f"robotiq_{name}_finger", "pos": f"0 {y:.6f} -0.045000"})
    ET.SubElement(
        finger,
        "joint",
        {
            "name": f"robotiq_{name}_slide",
            "type": "slide",
            "axis": "0 1 0",
            "range": f"{slide_range[0]:.6f} {slide_range[1]:.6f}",
            "damping": "1.0",
        },
    )
    ET.SubElement(
        finger,
        "geom",
        {
            "name": f"robotiq_{name}_finger_link",
            "type": "box",
            "size": _fmt_vec(cfg.finger_size),
            "pos": "0 0 -0.045000",
            "material": "robotiq_dark",
        },
    )
    ET.SubElement(
        finger,
        "geom",
        {
            "name": f"robotiq_{name}_finger_pad",
            "type": "box",
            "size": _fmt_vec(cfg.fingertip_size),
            "pos": f"0 {-sign * 0.004:.6f} -0.098000",
            "material": "robotiq_pad",
        },
    )


def _ensure_gripper_materials(root: ET.Element, cfg: Robotiq2F85LiteConfig) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)

    material_names = {material.attrib.get("name") for material in asset.findall("material")}
    if "robotiq_dark" not in material_names:
        ET.SubElement(asset, "material", {"name": "robotiq_dark", "rgba": cfg.rgba_dark})
    if "robotiq_pad" not in material_names:
        ET.SubElement(asset, "material", {"name": "robotiq_pad", "rgba": cfg.rgba_pad})


def _target_pose_and_size(root: ET.Element, target_object: str) -> Tuple[Vec3, Vec3]:
    target = root.find(f".//body[@name='{target_object}']")
    if target is None:
        raise ValueError(f"Target object not found: {target_object}")

    pos = _parse_vec(target.attrib.get("pos", "0 0 0"))
    geom = target.find("geom")
    if geom is None:
        raise ValueError(f"Target object has no geom: {target_object}")

    geom_type = geom.attrib.get("type", "box")
    size = _parse_vec(geom.attrib.get("size", "0 0 0"))
    if geom_type == "cylinder":
        half_extents = (size[0], size[0], size[1])
    else:
        half_extents = size
    return pos, half_extents


def _remove_existing_gripper(worldbody: ET.Element) -> None:
    for child in list(worldbody):
        if child.attrib.get("name") == "robotiq_2f85_lite":
            worldbody.remove(child)


def _parse_vec(value: str) -> Vec3:
    parts = [float(part) for part in value.split()]
    if len(parts) < 3:
        raise ValueError(f"Expected 3D vector, got: {value}")
    return (parts[0], parts[1], parts[2])


def _fmt_vec(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.6f}" for value in values)


def _replace_pos(config: Robotiq2F85LiteConfig, pos: Vec3) -> Robotiq2F85LiteConfig:
    return Robotiq2F85LiteConfig(
        pos=pos,
        quat=config.quat,
        include_freejoint=config.include_freejoint,
        opening=config.opening,
        approach_height=config.approach_height,
        palm_size=config.palm_size,
        finger_size=config.finger_size,
        fingertip_size=config.fingertip_size,
        rgba_dark=config.rgba_dark,
        rgba_pad=config.rgba_pad,
    )


def _relative_posix_path(path: Path, start: Path) -> str:
    import os

    return Path(os.path.relpath(path, start=start)).as_posix()


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
