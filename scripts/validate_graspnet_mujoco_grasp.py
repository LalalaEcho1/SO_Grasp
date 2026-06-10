from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stacked_grasping.gripper.candidate_io import load_graspnet_records  # noqa: E402
from stacked_grasping.gripper.external_graspnet_data import AnnotationObject, GraspNetRealSenseSource, normalize_frame_id  # noqa: E402
from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate, graspnet_outputs_to_candidates  # noqa: E402
from stacked_grasping.gripper.graspnet_predictions import transform_graspnet_records  # noqa: E402
from stacked_grasping.gripper.graspnet_mujoco_scene import (  # noqa: E402
    mujoco_body_name_for_annotation,
    transform_annotation_objects,
    write_graspnet_mujoco_scene_xml,
)
from stacked_grasping.gripper.mujoco_grasp_validation import (  # noqa: E402
    LiteGraspValidationConfig,
    validate_lite_grasp_xml,
)
from stacked_grasping.gripper.robotiq_2f85_lite import Robotiq2F85LiteConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one GraspNet prediction in a MuJoCo scene with a Robotiq 2F-85 Lite gripper."
    )
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "graspnet_mujoco_validation")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--mesh-file", default="textured.obj")
    parser.add_argument("--candidate-rank", type=int, default=0, help="0 selects the highest-score prediction.")
    parser.add_argument("--no-align-to-table", dest="align_to_table", action="store_false", help="Keep raw camera-frame coordinates.")
    parser.set_defaults(align_to_table=True)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--approach-steps", type=int, default=40)
    parser.add_argument("--close-steps", type=int, default=40)
    parser.add_argument("--lift-steps", type=int, default=60)
    parser.add_argument("--hold-steps", type=int, default=20)
    parser.add_argument("--pregrasp-distance", type=float, default=0.06)
    parser.add_argument("--lift-distance", type=float, default=0.08)
    parser.add_argument("--lift-success-threshold-m", type=float, default=0.02)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_graspnet_mujoco_grasp(
        scene_root=args.scene_root,
        dataset_root=args.dataset_root,
        prediction_root=args.prediction_root,
        out_dir=args.out_dir,
        scene=args.scene,
        frame=args.frame,
        camera=args.camera,
        mesh_file=args.mesh_file,
        candidate_rank=args.candidate_rank,
        align_to_table=args.align_to_table,
        validation_config=LiteGraspValidationConfig(
            settle_steps=args.settle_steps,
            approach_steps=args.approach_steps,
            close_steps=args.close_steps,
            lift_steps=args.lift_steps,
            hold_steps=args.hold_steps,
            pregrasp_distance=args.pregrasp_distance,
            lift_distance=args.lift_distance,
            lift_success_threshold_m=args.lift_success_threshold_m,
        ),
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    validation = summary["validation"]
    print("GraspNet MuJoCo grasp validation finished")
    print(f"  xml_path: {summary['xml_path']}")
    print(f"  summary_path: {summary['summary_path']}")
    print(f"  target: {summary['target_object_name']} ({summary['target_body_name']})")
    print(f"  selected_grasp_score: {summary['selected_grasp_score']}")
    print(f"  compile_success: {validation['compile_success']}")
    print(f"  lift_success: {validation['lift_success']}")
    print(f"  target_contact_step_count: {validation['target_contact_step_count']}")
    print(f"  target_lift_delta_m: {validation['target_lift_delta_m']}")
    if validation.get("failure_reason"):
        print(f"  failure_reason: {validation['failure_reason']}")


def validate_graspnet_mujoco_grasp(
    *,
    scene_root: str | Path,
    dataset_root: str | Path,
    prediction_root: str | Path,
    out_dir: str | Path,
    scene: str,
    frame: int | str,
    camera: str = "realsense",
    mesh_file: str = "textured.obj",
    candidate_rank: int = 0,
    align_to_table: bool = True,
    validation_config: LiteGraspValidationConfig | None = None,
) -> dict[str, object]:
    frame_id = normalize_frame_id(frame)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / f"{scene}_{camera}_{frame_id}_candidate{candidate_rank:03d}_dynamic.xml"
    summary_path = output_dir / f"{scene}_{camera}_{frame_id}_candidate{candidate_rank:03d}_dynamic_summary.json"

    realsense_path = _realsense_path_for_scene(scene_root, scene, camera)
    with GraspNetRealSenseSource.open(realsense_path) as source:
        annotations = source.load_annotation_objects(frame_id)

    prediction_file = _prediction_file_for_scene_frame(prediction_root, scene, camera, frame_id)
    records = sorted(load_graspnet_records(prediction_file), key=lambda item: float(item.get("score", 0.0)), reverse=True)
    if not records:
        raise ValueError(f"No GraspNet prediction records found in {prediction_file}.")
    if candidate_rank < 0 or candidate_rank >= len(records):
        raise ValueError(f"candidate_rank {candidate_rank} is out of range for {len(records)} candidates.")

    align_transform = _load_table_alignment_transform(scene_root, scene, camera, frame_id) if align_to_table else None
    if align_transform is not None:
        annotations = transform_annotation_objects(annotations, align_transform)
        records_for_candidates = transform_graspnet_records(records, align_transform)
    else:
        records_for_candidates = records

    selected_grasp = graspnet_outputs_to_candidates(
        [records_for_candidates[candidate_rank]],
        generator="graspnet-dynamic-validation",
    )[0]
    target_annotation = select_target_annotation_for_grasp(annotations, selected_grasp)
    target_body_name = mujoco_body_name_for_annotation(target_annotation)

    write_graspnet_mujoco_scene_xml(
        xml_path,
        annotations,
        dataset_root=dataset_root,
        selected_grasp=selected_grasp,
        gripper_config=Robotiq2F85LiteConfig(include_freejoint=True),
        mesh_file=mesh_file,
    )
    validation = validate_lite_grasp_xml(
        xml_path,
        target_body_name=target_body_name,
        config=validation_config,
    )
    summary = {
        "scene": scene,
        "camera": camera,
        "frame": frame_id,
        "scene_root": str(scene_root),
        "dataset_root": str(dataset_root),
        "prediction_root": str(prediction_root),
        "prediction_file": str(prediction_file),
        "xml_path": str(xml_path),
        "summary_path": str(summary_path),
        "candidate_rank": int(candidate_rank),
        "candidate_count": len(records),
        "coordinate_frame": "table_aligned" if align_transform is not None else "camera",
        "selected_grasp_score": round(float(selected_grasp.score), 6),
        "selected_grasp_position": selected_grasp.position.round(6).tolist(),
        "selected_grasp_object_id": selected_grasp.object_id,
        "target_object_id": int(target_annotation.object_id),
        "target_object_name": target_annotation.name,
        "target_position": np.asarray(target_annotation.position, dtype=float).round(6).tolist(),
        "target_body_name": target_body_name,
        "validation": validation.to_dict(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def select_target_annotation_for_grasp(
    annotations: Sequence[AnnotationObject],
    grasp: GraspPoseCandidate,
) -> AnnotationObject:
    if not annotations:
        raise ValueError("Cannot select target annotation from an empty scene.")

    object_id = _candidate_object_id(grasp)
    if object_id is not None and object_id >= 0:
        for annotation in annotations:
            if int(annotation.object_id) == object_id:
                return annotation

    grasp_position = np.asarray(grasp.position, dtype=float).reshape(3)
    return min(
        annotations,
        key=lambda annotation: float(np.linalg.norm(grasp_position - np.asarray(annotation.position, dtype=float))),
    )


def _candidate_object_id(grasp: GraspPoseCandidate) -> int | None:
    raw = grasp.object_id
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return int(value) if value.is_integer() else None


def _realsense_path_for_scene(scene_root: str | Path, scene: str, camera: str) -> Path:
    root = Path(scene_root)
    camera_path = root / scene / camera
    return camera_path if camera_path.exists() else root / scene


def _prediction_file_for_scene_frame(prediction_root: str | Path, scene: str, camera: str, frame: str) -> Path:
    root = Path(prediction_root)
    candidates = [
        root / scene / camera / f"{frame}.npy",
        root / scene / f"{frame}.npy",
        root / f"{scene}_{frame}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No GraspNet prediction file found. Searched:\n  {searched}")


def _load_table_alignment_transform(scene_root: str | Path, scene: str, camera: str, frame: str) -> np.ndarray:
    camera_path = Path(scene_root) / scene / camera
    if not camera_path.exists():
        camera_path = Path(scene_root) / scene
    camera_poses = np.load(camera_path / "camera_poses.npy")
    align_mat = np.load(camera_path / "cam0_wrt_table.npy")
    frame_index = int(frame)
    if camera_poses.ndim != 3 or camera_poses.shape[1:] != (4, 4):
        raise ValueError(f"camera_poses.npy under {camera_path} must have shape (N, 4, 4).")
    if frame_index >= camera_poses.shape[0]:
        raise ValueError(f"Frame {frame} is out of range for camera_poses with {camera_poses.shape[0]} frames.")
    if align_mat.shape != (4, 4):
        raise ValueError(f"cam0_wrt_table.npy under {camera_path} must have shape (4, 4).")
    return np.asarray(align_mat, dtype=float) @ np.asarray(camera_poses[frame_index], dtype=float)


if __name__ == "__main__":
    main()
