from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stacked_grasping.gripper.candidate_io import load_graspnet_records  # noqa: E402
from stacked_grasping.gripper.external_graspnet_data import GraspNetRealSenseSource, normalize_frame_id  # noqa: E402
from stacked_grasping.gripper.grasp_pose import GraspPoseCandidate, graspnet_outputs_to_candidates  # noqa: E402
from stacked_grasping.gripper.graspnet_mujoco_scene import write_graspnet_mujoco_scene_xml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one official GraspNet annotation frame as a MuJoCo XML scene with a Robotiq 2F-85 Lite gripper."
    )
    parser.add_argument("--scene-root", type=Path, required=True, help="Official GraspNet scenes root containing scene_xxxx.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Official GraspNet dataset root containing models/NNN.")
    parser.add_argument("--prediction-root", type=Path, help="Optional GraspNet dump root containing scene_xxxx/realsense/NNNN.npy.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "graspnet_mujoco_scenes")
    parser.add_argument("--scene", required=True, help="Scene id, for example scene_0009.")
    parser.add_argument("--frame", required=True, help="Frame id, for example 0255.")
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--mesh-file", default="textured.obj")
    parser.add_argument("--compile-mujoco", action="store_true", help="Compile the exported XML with mujoco.MjModel.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_graspnet_mujoco_scene(
        scene_root=args.scene_root,
        dataset_root=args.dataset_root,
        prediction_root=args.prediction_root,
        out_dir=args.out_dir,
        scene=args.scene,
        frame=args.frame,
        camera=args.camera,
        mesh_file=args.mesh_file,
        compile_mujoco=args.compile_mujoco,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print("GraspNet MuJoCo scene exported")
    print(f"  xml_path: {summary['xml_path']}")
    print(f"  summary_path: {summary['summary_path']}")
    print(f"  objects: {summary['object_count']}")
    print(f"  selected_grasp_score: {summary['selected_grasp_score']}")
    print(f"  compile_success: {summary['compile_success']}")
    if summary.get("compile_error"):
        print(f"  compile_error: {summary['compile_error']}")


def export_graspnet_mujoco_scene(
    *,
    scene_root: str | Path,
    dataset_root: str | Path,
    out_dir: str | Path,
    scene: str,
    frame: int | str,
    prediction_root: str | Path | None = None,
    camera: str = "realsense",
    mesh_file: str = "textured.obj",
    compile_mujoco: bool = False,
) -> dict[str, object]:
    frame_id = normalize_frame_id(frame)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / f"{scene}_{camera}_{frame_id}_robotiq2f85_lite.xml"
    summary_path = output_dir / f"{scene}_{camera}_{frame_id}_robotiq2f85_lite_summary.json"

    realsense_path = _realsense_path_for_scene(scene_root, scene, camera)
    with GraspNetRealSenseSource.open(realsense_path) as source:
        annotations = source.load_annotation_objects(frame_id)

    prediction_file = None
    selected_grasp = None
    selected_grasp_score = None
    if prediction_root is not None:
        prediction_file = _prediction_file_for_scene_frame(prediction_root, scene, camera, frame_id)
        records = load_graspnet_records(prediction_file)
        selected_grasp = _top_prediction_candidate(records)
        selected_grasp_score = round(float(selected_grasp.score), 6) if selected_grasp is not None else None

    write_graspnet_mujoco_scene_xml(
        xml_path,
        annotations,
        dataset_root=dataset_root,
        selected_grasp=selected_grasp,
        mesh_file=mesh_file,
    )
    compile_success, compile_error = _compile_mujoco_xml(xml_path) if compile_mujoco else (None, None)

    summary = {
        "scene": scene,
        "camera": camera,
        "frame": frame_id,
        "scene_root": str(scene_root),
        "dataset_root": str(dataset_root),
        "prediction_root": str(prediction_root) if prediction_root is not None else None,
        "prediction_file": str(prediction_file) if prediction_file is not None else None,
        "xml_path": str(xml_path),
        "summary_path": str(summary_path),
        "object_count": len(annotations),
        "object_ids": [int(obj.object_id) for obj in annotations],
        "selected_grasp_score": selected_grasp_score,
        "selected_grasp_generator": selected_grasp.generator if selected_grasp is not None else None,
        "compile_success": compile_success,
        "compile_error": compile_error,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _top_prediction_candidate(records: list[dict[str, object]]) -> GraspPoseCandidate | None:
    if not records:
        return None
    top_record = max(records, key=lambda item: float(item.get("score", 0.0)))
    return graspnet_outputs_to_candidates([top_record], generator="graspnet-sim-preview")[0]


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


def _compile_mujoco_xml(xml_path: Path) -> tuple[bool, str | None]:
    try:
        import mujoco

        mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as exc:  # pragma: no cover - depends on local MuJoCo mesh support.
        return False, str(exc)
    return True, None


if __name__ == "__main__":
    main()
