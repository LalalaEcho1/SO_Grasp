from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from stacked_grasping.gripper.candidate_io import load_graspnet_candidates, load_graspnet_records
from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate
from stacked_grasping.utils.paths import project_root


CURRENT_MAIN_V1_OBJECT_IDS = ("003", "004", "005", "007", "008", "009", "010", "061")


@dataclass(frozen=True)
class SeniorGraspNetValidation:
    checks: Dict[str, bool]
    missing_coacd_models: List[str]
    missing_official_models: List[str]

    @property
    def ok(self) -> bool:
        return all(self.checks.values()) and not self.missing_coacd_models and not self.missing_official_models

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "checks": dict(self.checks),
            "missing_coacd_models": list(self.missing_coacd_models),
            "missing_official_models": list(self.missing_official_models),
        }


@dataclass(frozen=True)
class SeniorGraspNetPaths:
    root: Path | str | None = None

    @property
    def base(self) -> Path:
        if self.root is None:
            return project_root() / "external" / "senior_graspnet"
        return Path(self.root)

    @property
    def baseline_dir(self) -> Path:
        return self.base / "graspnet-baseline-main"

    @property
    def api_dir(self) -> Path:
        return self.base / "graspnetAPI-master"

    @property
    def checkpoints_dir(self) -> Path:
        return self.base / "checkpoints"

    @property
    def dataset_dir(self) -> Path:
        return self.base / "simulation" / "dataset"

    @property
    def coacd_models_dir(self) -> Path:
        return self.dataset_dir / "coacd_models"

    @property
    def official_models_dir(self) -> Path:
        return self.dataset_dir / "offical_models"

    @property
    def franka_dir(self) -> Path:
        return self.dataset_dir / "franka"

    @property
    def grasp_poses_dir(self) -> Path:
        return self.dataset_dir / "grasp_poses"

    @property
    def mujoco_script(self) -> Path:
        return self.base / "mujoco" / "mujoco_sim_final.py"

    @property
    def utils_py(self) -> Path:
        return self.base / "utils" / "utils.py"

    def checkpoint_path(self, camera: str = "realsense") -> Path:
        suffix = "rs" if camera.lower() in {"realsense", "rs"} else "kn"
        return self.checkpoints_dir / f"checkpoint-{suffix}.tar"

    def grasp_pose_path(
        self,
        split: str,
        scene_id: int | str,
        view_id: int | str,
        camera: str = "realsense",
    ) -> Path:
        return self.grasp_poses_dir / split / _scene_name(scene_id) / camera / _view_filename(view_id)

    def validate(self, required_object_ids: Sequence[str] = CURRENT_MAIN_V1_OBJECT_IDS) -> SeniorGraspNetValidation:
        checks = {
            "baseline_demo": (self.baseline_dir / "demo.py").exists(),
            "api_package": (self.api_dir / "graspnetAPI" / "__init__.py").exists(),
            "checkpoint_rs": self.checkpoint_path("realsense").exists(),
            "checkpoint_kn": self.checkpoint_path("kinect").exists(),
            "mujoco_script": self.mujoco_script.exists(),
            "utils_py": self.utils_py.exists(),
            "franka_panda": (self.franka_dir / "panda.xml").exists(),
            "grasp_poses": self.grasp_poses_dir.exists(),
        }
        required = tuple(_object_id_text(item) for item in required_object_ids)
        missing_coacd = [item for item in required if not (self.coacd_models_dir / item).exists()]
        missing_official = [item for item in required if not (self.official_models_dir / item).exists()]
        return SeniorGraspNetValidation(
            checks=checks,
            missing_coacd_models=missing_coacd,
            missing_official_models=missing_official,
        )


def load_reference_grasp_candidates(
    split: str,
    scene_id: int | str,
    view_id: int | str,
    root: Path | str | None = None,
    camera: str = "realsense",
    object_id_to_name: Mapping[int | str, str] | None = None,
) -> List[GraspPoseCandidate]:
    paths = SeniorGraspNetPaths(root)
    return load_graspnet_candidates(
        paths.grasp_pose_path(split=split, scene_id=scene_id, view_id=view_id, camera=camera),
        object_id_to_name=object_id_to_name,
        generator="graspnet-reference",
    )


def summarize_reference_grasp_file(path: Path | str) -> Dict[str, object]:
    records = load_graspnet_records(path)
    scores = [float(record["score"]) for record in records]
    widths = [float(record["width"]) for record in records]
    object_ids = sorted({record["object_id"] for record in records if "object_id" in record})
    return {
        "path": str(path),
        "candidate_count": len(records),
        "score_min": round(min(scores) if scores else 0.0, 6),
        "score_max": round(max(scores) if scores else 0.0, 6),
        "width_min": round(min(widths) if widths else 0.0, 6),
        "width_max": round(max(widths) if widths else 0.0, 6),
        "object_ids": object_ids,
    }


def _scene_name(scene_id: int | str) -> str:
    if isinstance(scene_id, int):
        return f"scene_{scene_id:04d}"
    text = str(scene_id)
    if text.startswith("scene_"):
        return text
    return f"scene_{int(text):04d}"


def _view_filename(view_id: int | str) -> str:
    if isinstance(view_id, int):
        return f"{view_id:04d}.npy"
    text = str(view_id)
    if text.endswith(".npy"):
        return text
    return f"{int(text):04d}.npy"


def _object_id_text(value: str | int) -> str:
    return f"{int(value):03d}" if isinstance(value, int) or str(value).isdigit() else str(value)
