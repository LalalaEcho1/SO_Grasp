from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stacked_grasping.assets.ycb_starter import STARTER_OBJECTS, YcbObjectSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a MuJoCo scene using official YCB visual meshes.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "assets" / "objects" / "ycb",
        help="Directory containing normalized YCB object folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "assets" / "scenes" / "ycb_mesh_stacked.xml",
        help="Output MuJoCo XML scene.",
    )
    return parser.parse_args()


def read_obj_bounds(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    vertex_count = 0

    with path.open("r", encoding="utf-8", errors="ignore") as obj_file:
        for line in obj_file:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            values = [float(parts[1]), float(parts[2]), float(parts[3])]
            for idx, value in enumerate(values):
                mins[idx] = min(mins[idx], value)
                maxs[idx] = max(maxs[idx], value)
            vertex_count += 1

    if vertex_count == 0:
        raise RuntimeError(f"No vertices found in OBJ: {path}")
    return tuple(mins), tuple(maxs)


def orientation_hint_half_extents(spec: YcbObjectSpec) -> tuple[float, float, float]:
    if spec.geom_type == "cylinder":
        radius, half_height, _ = spec.size
        return radius, radius, half_height
    return spec.size


def rotate_z(vec: tuple[float, float, float], yaw: float) -> tuple[float, float, float]:
    x, y, z = vec
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * x - s * y, s * x + c * y, z


def quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)


def mesh_unit_scale(raw_half: tuple[float, float, float]) -> float:
    # Official YCB Google 16k meshes are normally in meters. This guard keeps the
    # generator usable if a millimeter-scaled mesh is introduced later.
    return 0.001 if max(raw_half) > 1.0 else 1.0


def best_visual_alignment(
    raw_mins: tuple[float, float, float],
    raw_maxs: tuple[float, float, float],
    hint: tuple[float, float, float],
) -> tuple[float, float, tuple[float, float, float], tuple[float, float, float], float]:
    raw_center = tuple((raw_mins[i] + raw_maxs[i]) * 0.5 for i in range(3))
    raw_half = tuple((raw_maxs[i] - raw_mins[i]) * 0.5 for i in range(3))
    scale = mesh_unit_scale(raw_half)

    best: tuple[float, float, tuple[float, float, float], tuple[float, float, float], float] | None = None
    for yaw in (0.0, math.pi / 2.0, -math.pi / 2.0, math.pi):
        rotated_half = tuple(abs(value) * scale for value in rotate_z(raw_half, yaw))
        error = math.sqrt(sum((rotated_half[i] - hint[i]) ** 2 for i in range(3)))
        normalizer = math.sqrt(sum(value * value for value in hint)) or 1.0
        normalized_error = error / normalizer
        rotated_center = rotate_z(tuple(scale * value for value in raw_center), yaw)
        visual_pos = tuple(-value for value in rotated_center)
        candidate = (scale, yaw, visual_pos, rotated_half, normalized_error)
        if best is None or normalized_error < best[4]:
            best = candidate

    if best is None:
        raise RuntimeError("Could not compute mesh alignment.")
    return best


def fmt_vec(values: tuple[float, ...], precision: int = 6) -> str:
    return " ".join(f"{value:.{precision}f}" for value in values)


def relpath_for_xml(path: Path, xml_path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), xml_path.parent.resolve())).as_posix()


def collision_geom(spec: YcbObjectSpec, half_extents: tuple[float, float, float]) -> str:
    rgba = "0.12 0.12 0.12 0.0"
    if spec.geom_type == "cylinder":
        radius = (half_extents[0] + half_extents[1]) * 0.5
        half_height = half_extents[2]
        size = f"{radius:.6f} {half_height:.6f}"
        return (
            f'      <geom name="obj_{spec.name}_geom" type="cylinder" size="{size}" '
            f'mass="{spec.mass:.6f}" rgba="{rgba}" group="3"/>\n'
        )

    return (
        f'      <geom name="obj_{spec.name}_geom" type="box" size="{fmt_vec(half_extents)}" '
        f'mass="{spec.mass:.6f}" rgba="{rgba}" group="3"/>\n'
    )


def object_body(
    spec: YcbObjectSpec,
    body_pos: tuple[float, float, float],
    half_extents: tuple[float, float, float],
    yaw: float,
    visual_pos: tuple[float, float, float],
) -> str:
    quat = quat_from_yaw(yaw)
    return (
        f'    <body name="obj_{spec.name}" pos="{fmt_vec(body_pos)}">\n'
        f"      <freejoint/>\n"
        f"{collision_geom(spec, half_extents)}"
        f'      <geom name="obj_{spec.name}_visual" type="mesh" mesh="mesh_{spec.name}" '
        f'material="mat_{spec.name}" pos="{fmt_vec(visual_pos)}" quat="{fmt_vec(quat, 8)}" '
        f'contype="0" conaffinity="0" group="2"/>\n'
        f"    </body>\n"
    )


def generate_scene(
    dataset_root: Path,
    output: Path,
    specs: tuple[YcbObjectSpec, ...] = STARTER_OBJECTS,
    model_name: str = "ycb_mesh_stacked",
) -> str:
    asset_lines: list[str] = [
        '    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.82 0.82 0.82" rgb2="0.68 0.68 0.68"/>',
        '    <material name="floor_mat" texture="grid" texrepeat="2 2" reflectance="0.15"/>',
        '    <material name="table_mat" rgba="0.35 0.31 0.27 1"/>',
    ]
    body_lines: list[str] = []
    report_lines: list[str] = []
    body_positions: dict[str, tuple[float, float, float]] = {}
    collision_half_extents: dict[str, tuple[float, float, float]] = {}
    table_top_z = 0.35 + 0.035

    for spec in specs:
        mesh_dir = dataset_root / spec.name / "google_16k"
        obj_path = mesh_dir / "textured.obj"
        texture_path = mesh_dir / "texture_map.png"
        if not obj_path.exists():
            raise FileNotFoundError(
                f"Missing official YCB mesh for {spec.name}: {obj_path}\n"
                "Run: python scripts/download_ycb_meshes.py --objects starter"
            )

        raw_mins, raw_maxs = read_obj_bounds(obj_path)
        hint = orientation_hint_half_extents(spec)
        scale, yaw, visual_pos, mesh_half, error = best_visual_alignment(raw_mins, raw_maxs, hint)
        half_extents = mesh_half
        if spec.geom_type == "cylinder":
            radius = (mesh_half[0] + mesh_half[1]) * 0.5
            half_extents = (radius, radius, mesh_half[2])

        if spec.support:
            support_pos = body_positions[spec.support]
            support_half = collision_half_extents[spec.support]
            z_pos = support_pos[2] + support_half[2] + half_extents[2] + 0.001
        else:
            z_pos = table_top_z + half_extents[2] + 0.001
        body_pos = (spec.pos[0], spec.pos[1], z_pos)
        body_positions[spec.name] = body_pos
        collision_half_extents[spec.name] = half_extents

        mesh_file = relpath_for_xml(obj_path, output)
        asset_lines.append(f'    <mesh name="mesh_{spec.name}" file="{mesh_file}" scale="{scale:.9f} {scale:.9f} {scale:.9f}"/>')

        if texture_path.exists():
            texture_file = relpath_for_xml(texture_path, output)
            asset_lines.append(f'    <texture name="tex_{spec.name}" type="2d" file="{texture_file}"/>')
            asset_lines.append(f'    <material name="mat_{spec.name}" texture="tex_{spec.name}" specular="0.25" shininess="0.35"/>')
        else:
            asset_lines.append(f'    <material name="mat_{spec.name}" rgba="{fmt_vec(spec.rgba)}"/>')

        body_lines.append(object_body(spec, body_pos, half_extents, yaw, visual_pos))
        report_lines.append(
            f"     {spec.name}: scale={scale:.9f}, yaw={yaw:.6f}, "
            f"mesh_half=({fmt_vec(mesh_half, 4)}), collision_half=({fmt_vec(half_extents, 4)}), "
            f"hint_half=({fmt_vec(hint, 4)}), error={error:.4f}"
        )

    report = "\n".join(report_lines)
    assets = "\n".join(asset_lines)
    bodies = "\n\n".join(body_lines)
    return f"""<mujoco model="{model_name}">
  <compiler angle="radian" autolimits="true"/>

  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <visual>
    <global offwidth="1280" offheight="900"/>
  </visual>

  <default>
    <joint damping="0.01"/>
    <geom condim="4" friction="1.05 0.006 0.0001" solref="0.01 1" solimp="0.9 0.95 0.001"/>
  </default>

  <!-- Generated by scripts/generate_ycb_mesh_scene.py.
{report}
  -->
  <asset>
{assets}
  </asset>

  <worldbody>
    <light name="key_light" pos="0.25 -0.55 1.2" dir="-0.2 0.45 -1"/>
    <camera name="overview" mode="targetbody" target="table" pos="0.82 -0.92 1.08" fovy="62"/>
    <camera name="topdown" mode="targetbody" target="table" pos="0.00 0.00 1.45" fovy="54"/>
    <camera name="front" mode="targetbody" target="table" pos="0.00 -1.20 0.72" fovy="55"/>
    <geom name="floor" type="plane" size="1.2 1.2 0.02" material="floor_mat"/>

    <body name="table" pos="0 0 0.35">
      <geom name="table_top" type="box" size="0.50 0.38 0.035" material="table_mat"/>
    </body>

{bodies}
  </worldbody>
</mujoco>
"""


def main() -> None:
    args = parse_args()
    scene_xml = generate_scene(args.dataset_root, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(scene_xml, encoding="utf-8")
    print(f"saved MuJoCo scene: {args.output}")


if __name__ == "__main__":
    main()
