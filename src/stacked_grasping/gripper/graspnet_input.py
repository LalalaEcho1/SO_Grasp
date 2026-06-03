from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


def depth_meters_to_uint16(depth_meters: np.ndarray, factor_depth: int = 1000) -> np.ndarray:
    """Encode metric depth as GraspNet-style uint16 PNG values."""
    if factor_depth <= 0:
        raise ValueError("factor_depth must be positive.")

    depth = np.asarray(depth_meters, dtype=float)
    encoded = np.zeros(depth.shape, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0.0)
    encoded[valid] = np.rint(depth[valid] * float(factor_depth))
    return np.clip(encoded, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def workspace_mask_from_depth(
    depth_meters: np.ndarray,
    *,
    min_depth_m: float = 0.0,
    max_depth_m: float | None = None,
) -> np.ndarray:
    """Build a boolean workspace mask from metric depth limits."""
    depth = np.asarray(depth_meters, dtype=float)
    mask = np.isfinite(depth) & (depth > float(min_depth_m))
    if max_depth_m is not None:
        mask &= depth <= float(max_depth_m)
    return mask


def graspnet_input_dir_for_scene(out_root: str | Path, scene_path: str | Path) -> Path:
    return Path(out_root) / Path(scene_path).stem


def write_mat_v4_numeric(path: str | Path, variables: Mapping[str, np.ndarray | float | int]) -> None:
    """Write numeric variables in the simple MATLAB level-4 MAT format.

    GraspNet's demo only needs ``intrinsic_matrix`` and ``factor_depth``.
    Keeping this writer local avoids adding SciPy as a hard dependency.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("wb") as handle:
        for name, value in variables.items():
            if not name:
                raise ValueError("MAT variable names must be non-empty.")

            matrix = np.asarray(value, dtype="<f8")
            if matrix.ndim == 0:
                matrix = matrix.reshape(1, 1)
            elif matrix.ndim == 1:
                matrix = matrix.reshape(1, matrix.shape[0])
            elif matrix.ndim != 2:
                raise ValueError(f"MAT variable {name!r} must be scalar, vector, or 2-D.")

            name_bytes = name.encode("ascii") + b"\x00"
            header = struct.pack(
                "<5i",
                0,  # little-endian, double precision, numeric matrix
                int(matrix.shape[0]),
                int(matrix.shape[1]),
                0,  # real-valued matrix
                len(name_bytes),
            )
            handle.write(header)
            handle.write(name_bytes)
            handle.write(np.asfortranarray(matrix, dtype="<f8").tobytes(order="F"))


def write_graspnet_input_bundle(
    out_dir: str | Path,
    *,
    color: np.ndarray,
    depth_meters: np.ndarray,
    intrinsic_matrix: np.ndarray,
    factor_depth: int = 1000,
    workspace_mask: np.ndarray | None = None,
    camera: str | None = None,
    scene: str | None = None,
    camera_to_world_matrix: np.ndarray | None = None,
    camera_frame: str = "opencv",
) -> dict[str, Any]:
    """Write a GraspNet demo-compatible RGB-D input folder."""
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    color_array = _as_uint8_rgb(color)
    depth_array = np.asarray(depth_meters, dtype=float)
    if depth_array.shape != color_array.shape[:2]:
        raise ValueError("depth_meters must have the same height and width as color.")

    intrinsic = np.asarray(intrinsic_matrix, dtype=float)
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic_matrix must have shape (3, 3).")
    camera_to_world = None
    if camera_to_world_matrix is not None:
        camera_to_world = np.asarray(camera_to_world_matrix, dtype=float)
        if camera_to_world.shape != (4, 4):
            raise ValueError("camera_to_world_matrix must have shape (4, 4).")

    depth_png = depth_meters_to_uint16(depth_array, factor_depth=factor_depth)
    mask_png = _workspace_mask_to_uint8(workspace_mask, depth_array)

    Image.fromarray(color_array, mode="RGB").save(target_dir / "color.png")
    Image.fromarray(depth_png, mode="I;16").save(target_dir / "depth.png")
    Image.fromarray(mask_png, mode="L").save(target_dir / "workspace_mask.png")
    write_mat_v4_numeric(
        target_dir / "meta.mat",
        {
            "intrinsic_matrix": intrinsic,
            "factor_depth": np.array([[float(factor_depth)]], dtype=float),
        },
    )

    metadata = {
        "camera": camera,
        "scene": scene,
        "width": int(color_array.shape[1]),
        "height": int(color_array.shape[0]),
        "factor_depth": int(factor_depth),
        "intrinsic_matrix": intrinsic.tolist(),
        "camera_frame": camera_frame,
        "camera_to_world_matrix": camera_to_world.tolist() if camera_to_world is not None else None,
        "files": {
            "color": "color.png",
            "depth": "depth.png",
            "workspace_mask": "workspace_mask.png",
            "meta": "meta.mat",
        },
    }
    (target_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metadata


def _as_uint8_rgb(color: np.ndarray) -> np.ndarray:
    array = np.asarray(color)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("color must have shape (height, width, 3).")
    if array.dtype == np.uint8:
        return array
    return np.clip(array, 0, 255).astype(np.uint8)


def _workspace_mask_to_uint8(workspace_mask: np.ndarray | None, depth_meters: np.ndarray) -> np.ndarray:
    if workspace_mask is None:
        mask = workspace_mask_from_depth(depth_meters)
    else:
        mask = np.asarray(workspace_mask).astype(bool)
        if mask.shape != depth_meters.shape:
            raise ValueError("workspace_mask must have the same height and width as depth_meters.")
    return np.where(mask, 255, 0).astype(np.uint8)
