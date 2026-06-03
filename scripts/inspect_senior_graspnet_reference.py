from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stacked_grasping.gripper.senior_graspnet import (
    CURRENT_MAIN_V1_OBJECT_IDS,
    SeniorGraspNetPaths,
    summarize_reference_grasp_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the imported senior GraspNet reference bundle.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "external" / "senior_graspnet")
    parser.add_argument("--object-ids", nargs="+", default=list(CURRENT_MAIN_V1_OBJECT_IDS))
    parser.add_argument("--split", default="train")
    parser.add_argument("--scene-id", default=0)
    parser.add_argument("--view-id", default=0)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = SeniorGraspNetPaths(args.root)
    validation = paths.validate(required_object_ids=args.object_ids)
    sample_path = paths.grasp_pose_path(args.split, args.scene_id, args.view_id, camera=args.camera)
    sample_summary = summarize_reference_grasp_file(sample_path) if sample_path.exists() else {"path": str(sample_path), "missing": True}
    payload = {
        "root": str(paths.base),
        "validation": validation.to_dict(),
        "sample": sample_summary,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("Senior GraspNet reference bundle")
    print(f"  root: {payload['root']}")
    print(f"  ok: {validation.ok}")
    print("  checks:")
    for key, value in validation.checks.items():
        print(f"    {key}: {value}")
    print(f"  missing coacd models: {validation.missing_coacd_models}")
    print(f"  missing official models: {validation.missing_official_models}")
    print("  sample:")
    for key, value in sample_summary.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
