from __future__ import annotations

import math
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np

from stacked_grasping.relations.geometry import ObjectState


class MujocoStackedScene:
    """Thin MuJoCo wrapper for reading object-level scene state."""

    def __init__(self, xml_path: str | Path, object_prefix: str = "obj_") -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "The mujoco Python package is not installed. "
                "Create a virtual environment and run: pip install -r requirements.txt"
            ) from exc

        self.mujoco = mujoco
        self.xml_path = Path(xml_path)
        self.object_prefix = object_prefix
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.removed_object_names: Set[str] = set()
        self._removal_count = 0

    def reset_and_settle(self, steps: int = 1500) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        self.removed_object_names.clear()
        self._removal_count = 0
        self.settle(steps)

    def settle(self, steps: int = 500) -> None:
        for _ in range(steps):
            self.mujoco.mj_step(self.model, self.data)

    def read_objects(self) -> List[ObjectState]:
        objects: List[ObjectState] = []
        for body_id in range(1, self.model.nbody):
            body_name = self._body_name(body_id)
            if not body_name or not body_name.startswith(self.object_prefix):
                continue
            if body_name in self.removed_object_names:
                continue

            geom_id = self._first_geom_for_body(body_id)
            if geom_id is None:
                continue

            objects.append(
                ObjectState(
                    name=body_name,
                    body_id=body_id,
                    geom_id=geom_id,
                    geom_type=self._geom_type_name(geom_id),
                    position=np.array(self.data.xpos[body_id], dtype=float),
                    half_extents=self._geom_half_extents(geom_id),
                )
            )
        return objects

    def read_object_contact_pairs(self) -> Set[Tuple[str, str]]:
        body_to_name = {obj.body_id: obj.name for obj in self.read_objects()}
        pairs: Set[Tuple[str, str]] = set()

        for contact_idx in range(self.data.ncon):
            contact = self.data.contact[contact_idx]
            body_a = int(self.model.geom_bodyid[contact.geom1])
            body_b = int(self.model.geom_bodyid[contact.geom2])
            if body_a == body_b:
                continue
            if body_a not in body_to_name or body_b not in body_to_name:
                continue
            name_a = body_to_name[body_a]
            name_b = body_to_name[body_b]
            pairs.add(tuple(sorted((name_a, name_b))))

        return pairs

    def remove_object(self, name: str) -> None:
        """Move a successfully grasped free body out of the workspace."""
        if name in self.removed_object_names:
            return

        body_id = self._body_id(name)
        free_joint_id = self._free_joint_for_body(body_id)
        if free_joint_id is None:
            raise ValueError(f"Body {name!r} does not have a free joint and cannot be removed as a grasped object.")

        park_position = np.array([8.0 + 0.5 * self._removal_count, 8.0, 1.0], dtype=float)
        qpos_adr = int(self.model.jnt_qposadr[free_joint_id])
        qvel_adr = int(self.model.jnt_dofadr[free_joint_id])

        self.data.qpos[qpos_adr : qpos_adr + 7] = np.array([*park_position, 1.0, 0.0, 0.0, 0.0], dtype=float)
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        self.removed_object_names.add(name)
        self._removal_count += 1
        self.mujoco.mj_forward(self.model, self.data)

    def render_rgb(self, width: int = 1280, height: int = 900, camera: str | None = "overview") -> np.ndarray:
        renderer = self.mujoco.Renderer(self.model, height=height, width=width)
        try:
            if camera:
                renderer.update_scene(self.data, camera=camera)
            else:
                renderer.update_scene(self.data)
            return np.asarray(renderer.render(), dtype=np.uint8)
        finally:
            renderer.close()

    def render_depth(self, width: int = 1280, height: int = 720, camera: str | None = "overview") -> np.ndarray:
        renderer = self.mujoco.Renderer(self.model, height=height, width=width)
        try:
            renderer.enable_depth_rendering()
            if camera:
                renderer.update_scene(self.data, camera=camera)
            else:
                renderer.update_scene(self.data)
            return np.asarray(renderer.render(), dtype=float)
        finally:
            renderer.close()

    def render_rgbd(
        self,
        width: int = 1280,
        height: int = 720,
        camera: str | None = "overview",
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.render_rgb(width=width, height=height, camera=camera), self.render_depth(
            width=width,
            height=height,
            camera=camera,
        )

    def camera_intrinsic_matrix(
        self,
        width: int = 1280,
        height: int = 720,
        camera: str | None = "overview",
    ) -> np.ndarray:
        fovy_degrees = self._camera_fovy_degrees(camera)
        focal_y = 0.5 * float(height) / math.tan(math.radians(fovy_degrees) * 0.5)
        focal_x = focal_y
        return np.array(
            [
                [focal_x, 0.0, (float(width) - 1.0) * 0.5],
                [0.0, focal_y, (float(height) - 1.0) * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def camera_to_world_matrix(self, camera: str = "overview", frame: str = "opencv") -> np.ndarray:
        """Return a homogeneous camera-to-world transform.

        GraspNet point clouds use an OpenCV-style camera frame: x right,
        y down, z forward. MuJoCo cameras use an OpenGL-style local frame,
        so the OpenCV transform flips local y and z before applying cam_xmat.
        """
        if frame != "opencv":
            raise ValueError("Only the opencv camera frame is supported.")
        camera_id = self._camera_id(camera)
        self.mujoco.mj_forward(self.model, self.data)

        mujoco_camera_to_world = np.array(self.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
        opencv_to_mujoco_camera = np.diag([1.0, -1.0, -1.0])
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = mujoco_camera_to_world @ opencv_to_mujoco_camera
        transform[:3, 3] = np.array(self.data.cam_xpos[camera_id], dtype=float)
        return transform

    def _body_name(self, body_id: int) -> str:
        return self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id) or ""

    def _body_id(self, name: str) -> int:
        body_id = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name))
        if body_id < 0:
            raise KeyError(name)
        return body_id

    def _camera_fovy_degrees(self, camera: str | None) -> float:
        if camera:
            camera_id = self._camera_id(camera)
            fovy = float(self.model.cam_fovy[camera_id])
            if fovy > 0.0:
                return fovy

        return float(self.model.vis.global_.fovy)

    def _camera_id(self, camera: str) -> int:
        camera_id = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_CAMERA, camera))
        if camera_id < 0:
            raise KeyError(camera)
        return camera_id

    def _geom_name(self, geom_id: int) -> str:
        return self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""

    def _first_geom_for_body(self, body_id: int) -> int | None:
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_bodyid[geom_id]) == body_id:
                return geom_id
        return None

    def _free_joint_for_body(self, body_id: int) -> int | None:
        joint_start = int(self.model.body_jntadr[body_id])
        joint_count = int(self.model.body_jntnum[body_id])
        for joint_id in range(joint_start, joint_start + joint_count):
            if int(self.model.jnt_type[joint_id]) == self.mujoco.mjtJoint.mjJNT_FREE:
                return joint_id
        return None

    def _geom_type_name(self, geom_id: int) -> str:
        geom_type = int(self.model.geom_type[geom_id])
        geom_enum = self.mujoco.mjtGeom(geom_type)
        return geom_enum.name.replace("mjGEOM_", "").lower()

    def _geom_half_extents(self, geom_id: int) -> np.ndarray:
        geom_type = int(self.model.geom_type[geom_id])
        size = np.array(self.model.geom_size[geom_id], dtype=float)

        if geom_type == self.mujoco.mjtGeom.mjGEOM_BOX:
            return size[:3]
        if geom_type == self.mujoco.mjtGeom.mjGEOM_CYLINDER:
            radius, half_height = size[0], size[1]
            return np.array([radius, radius, half_height], dtype=float)
        if geom_type == self.mujoco.mjtGeom.mjGEOM_SPHERE:
            radius = size[0]
            return np.array([radius, radius, radius], dtype=float)

        fallback_radius = float(np.max(size[:3]))
        return np.array([fallback_radius, fallback_radius, fallback_radius], dtype=float)
