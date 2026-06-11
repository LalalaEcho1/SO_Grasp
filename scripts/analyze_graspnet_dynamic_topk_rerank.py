from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stacked_grasping.gripper.external_graspnet_data import GraspNetRealSenseSource, RealSenseFrame  # noqa: E402
from stacked_grasping.gripper.external_graspnet_scene import build_external_graspnet_episode_inputs  # noqa: E402
from stacked_grasping.planning.adaptive_score_v2 import rank_objects_v2  # noqa: E402
from stacked_grasping.relations.graph import build_relation_graph  # noqa: E402


PolicyFn = Callable[[Sequence[dict]], object]
ObjectPriorProvider = Callable[[dict], Mapping[str, Mapping[str, object]]]


POLICY_ORDER = (
    "top1",
    "graspnet-safe-rerank",
    "graspnet-score",
    "object-consensus",
    "object-consensus-score",
    "pointcloud-feasible-score",
    "pointcloud-soft-score",
    "pointcloud-low-collision",
    "od-pointcloud-compact",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze non-oracle reranking policies on saved GraspNet dynamic top-K results."
    )
    parser.add_argument("--summary", type=Path, required=True, help="Path to split_dynamic_topk_summary.json.")
    parser.add_argument("--out-dir", type=Path, help="Defaults to <summary parent>/rerank_analysis.")
    parser.add_argument("--scene-root", type=Path, help="Optional GraspNet scenes root used to attach OD object priors.")
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--factor-depth", type=int, default=1000)
    parser.add_argument("--min-points-per-object", type=int, default=20)
    parser.add_argument("--min-half-extent", type=float, default=0.01)
    parser.add_argument("--object-padding", type=float, default=0.002)
    parser.add_argument("--min-boundary-pixels", type=int, default=50)
    parser.add_argument("--no-save", action="store_true", help="Run analysis without writing JSON/CSV outputs.")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    object_prior_provider = (
        build_external_graspnet_object_prior_provider(
            scene_root=args.scene_root,
            camera=args.camera,
            factor_depth=args.factor_depth,
            min_points_per_object=args.min_points_per_object,
            min_half_extent=args.min_half_extent,
            object_padding=args.object_padding,
            min_boundary_pixels=args.min_boundary_pixels,
        )
        if args.scene_root is not None
        else None
    )
    summary = analyze_dynamic_topk_rerank(
        summary_path=args.summary,
        out_dir=args.out_dir or args.summary.parent / "rerank_analysis",
        save_outputs=not args.no_save,
        object_prior_provider=object_prior_provider,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print("GraspNet dynamic top-K rerank analysis finished")
    print(f"  output_dir: {summary['output_dir']}")
    print(f"  frames: {summary['frame_count']}")
    print("  policy success rates:")
    for policy in POLICY_ORDER:
        aggregate = summary["policy_aggregate"][policy]
        print(f"   {policy}: {aggregate['success_count']}/{aggregate['frame_count']} = {aggregate['success_rate']}")


def analyze_dynamic_topk_rerank(
    *,
    summary_path: str | Path,
    out_dir: str | Path,
    save_outputs: bool = True,
    object_prior_provider: ObjectPriorProvider | None = None,
) -> dict[str, object]:
    source = Path(summary_path)
    topk_summary = json.loads(source.read_text(encoding="utf-8"))
    target = Path(out_dir)
    if save_outputs:
        target.mkdir(parents=True, exist_ok=True)

    policy_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for frame in topk_summary.get("frame_results", []):
        candidates = sorted(
            [dict(candidate) for candidate in frame.get("candidate_results", [])],
            key=lambda item: int(item.get("candidate_rank", 10**6)),
        )
        if object_prior_provider is not None:
            attach_object_prior_features(candidates, object_prior_provider(frame))
        for candidate in candidates:
            candidate_rows.append(candidate_label_row(frame, candidate))
        for policy_name, policy_fn in _policy_functions().items():
            selected = policy_fn(candidates)
            policy_rows.append(policy_result_row(frame, selected, policy_name))

    aggregate = aggregate_policy_rows(policy_rows)
    label_analysis = aggregate_candidate_labels(candidate_rows)
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "summary_path": str(source),
        "output_dir": str(target),
        "output_saved": bool(save_outputs),
        "top_k": topk_summary.get("top_k"),
        "frame_count": len(topk_summary.get("frame_results", [])),
        "candidate_count": len(candidate_rows),
        "policy_aggregate": aggregate,
        "candidate_label_analysis": label_analysis,
        "policy_results": policy_rows,
    }

    if save_outputs:
        (target / "dynamic_topk_rerank_analysis.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_policy_results_csv(target / "dynamic_topk_rerank_policy_results.csv", policy_rows)
        write_candidate_label_analysis_csv(
            target / "dynamic_topk_candidate_label_analysis.csv",
            label_analysis["by_rank"],
        )
    return result


def choose_top1_candidate(candidates: Sequence[dict]) -> dict | None:
    if not candidates:
        return None
    return min(candidates, key=lambda item: int(item.get("candidate_rank", 10**6)))


def choose_graspnet_score_candidate(candidates: Sequence[dict]) -> dict | None:
    if not candidates:
        return None
    return max(candidates, key=lambda item: (_float_value(item.get("selected_grasp_score")), -_rank(item)))


def choose_graspnet_safe_rerank_candidate(candidates: Sequence[dict]) -> dict | None:
    if not candidates:
        return None
    ordered = sorted(candidates, key=_rank)
    top1 = ordered[0]
    if _is_graspnet_safe_candidate(top1):
        return top1
    safe_candidates = [candidate for candidate in ordered if _is_graspnet_safe_candidate(candidate)]
    if not safe_candidates:
        return top1
    return max(safe_candidates, key=lambda item: (_safe_rerank_score(item), -_rank(item)))


def choose_object_consensus_candidate(candidates: Sequence[dict]) -> dict | None:
    if not candidates:
        return None
    counts = Counter(_target_key(candidate) for candidate in candidates)
    return max(
        candidates,
        key=lambda item: (
            counts[_target_key(item)],
            _float_value(item.get("selected_grasp_score")),
            -_rank(item),
        ),
    )


def choose_object_consensus_score_candidate(candidates: Sequence[dict]) -> dict | None:
    if not candidates:
        return None
    counts = Counter(_target_key(candidate) for candidate in candidates)
    max_count = max(counts.values()) if counts else 1
    scores = [_float_value(candidate.get("selected_grasp_score"), default=0.0) for candidate in candidates]
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 1.0
    score_span = max(max_score - min_score, 1e-9)
    max_rank = max(_rank(candidate) for candidate in candidates) if candidates else 0

    def key(item: dict) -> tuple[float, float]:
        consensus = float(counts[_target_key(item)]) / float(max_count)
        score_norm = (_float_value(item.get("selected_grasp_score"), default=0.0) - min_score) / score_span
        rank_norm = 1.0 - (float(_rank(item)) / float(max(max_rank, 1)))
        blended = 0.45 * consensus + 0.40 * score_norm + 0.15 * rank_norm
        return (blended, -_rank(item))

    return max(candidates, key=key)


def choose_pointcloud_feasible_score_candidate(candidates: Sequence[dict]) -> dict | None:
    if not candidates:
        return None
    feasible = [candidate for candidate in candidates if bool(candidate.get("pointcloud_feasible"))]
    pool = feasible if feasible else list(candidates)
    return max(pool, key=lambda item: (_float_value(item.get("selected_grasp_score")), -_rank(item)))


def choose_pointcloud_soft_score_candidate(candidates: Sequence[dict]) -> dict | None:
    if not candidates:
        return None
    return max(candidates, key=lambda item: (_pointcloud_soft_score(item), -_rank(item)))


def choose_pointcloud_low_collision_candidate(candidates: Sequence[dict]) -> dict | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            1.0 if bool(item.get("pointcloud_feasible")) else 0.0,
            -_float_value(item.get("pointcloud_collision_iou"), default=1.0),
            _float_value(item.get("pointcloud_empty_ratio"), default=0.0),
            _float_value(item.get("selected_grasp_score")),
            -_rank(item),
        ),
    )


def choose_od_pointcloud_compact_candidate(candidates: Sequence[dict]) -> dict | None:
    if not candidates:
        return None
    return max(candidates, key=lambda item: (_od_pointcloud_compact_score(item), -_rank(item)))


def attach_object_prior_features(
    candidates: Sequence[dict],
    prior_by_object: Mapping[str, Mapping[str, object]],
) -> None:
    for candidate in candidates:
        prior = prior_by_object.get(str(candidate.get("target_object_name")))
        if prior is not None:
            candidate.update(dict(prior))


def build_external_graspnet_object_prior_provider(
    *,
    scene_root: str | Path,
    camera: str = "realsense",
    factor_depth: int = 1000,
    min_points_per_object: int = 20,
    min_half_extent: float = 0.01,
    object_padding: float = 0.002,
    min_boundary_pixels: int = 50,
) -> ObjectPriorProvider:
    root = Path(scene_root)
    cache: dict[tuple[str, str], dict[str, dict[str, object]]] = {}

    def provider(frame_result: dict) -> Mapping[str, Mapping[str, object]]:
        scene = str(frame_result["scene"])
        frame_id = str(frame_result["frame"])
        key = (scene, frame_id)
        if key not in cache:
            cache[key] = compute_external_graspnet_object_priors(
                scene_root=root,
                scene=scene,
                frame=frame_id,
                camera=camera,
                factor_depth=factor_depth,
                min_points_per_object=min_points_per_object,
                min_half_extent=min_half_extent,
                object_padding=object_padding,
                min_boundary_pixels=min_boundary_pixels,
            )
        return cache[key]

    return provider


def compute_external_graspnet_object_priors(
    *,
    scene_root: Path,
    scene: str,
    frame: str,
    camera: str,
    factor_depth: int,
    min_points_per_object: int,
    min_half_extent: float,
    object_padding: float,
    min_boundary_pixels: int,
) -> dict[str, dict[str, object]]:
    source_path = scene_root / scene / camera
    if not source_path.is_dir():
        source_path = scene_root / scene
    with GraspNetRealSenseSource.open(source_path) as source:
        loaded = source.load_frame(frame)
        real_frame = RealSenseFrame(
            frame=loaded.frame,
            color=loaded.color,
            depth_raw=loaded.depth_raw,
            label=loaded.label,
            intrinsic_matrix=loaded.intrinsic_matrix,
            camera_pose=loaded.camera_pose,
            cam0_wrt_table=loaded.cam0_wrt_table,
            factor_depth=factor_depth,
        )
        annotations = source.load_annotation_objects(real_frame.frame)

    episode_inputs = build_external_graspnet_episode_inputs(
        real_frame,
        annotations,
        [],
        min_points_per_object=min_points_per_object,
        min_half_extent=min_half_extent,
        padding=object_padding,
        min_boundary_pixels=min_boundary_pixels,
    )
    graph = build_relation_graph(
        episode_inputs.scene.read_objects(),
        episode_inputs.scene.read_object_contact_pairs(),
    )
    ranking = rank_objects_v2(graph)
    raw_scores = [float(item.adaptive_v2_score) for item in ranking]
    min_score = min(raw_scores) if raw_scores else 0.0
    max_score = max(raw_scores) if raw_scores else 1.0
    span = max(max_score - min_score, 1e-9)
    return {
        item.name: {
            "object_rank": index,
            "object_adaptive_v2_score": round(float(item.adaptive_v2_score), 6),
            "object_adaptive_v2_score_norm": round((float(item.adaptive_v2_score) - min_score) / span, 6),
            "object_grasp_risk": round(float(item.grasp_risk), 6),
            "object_high_risk": bool(item.high_risk),
            "object_height_priority": round(float(item.height_priority), 6),
        }
        for index, item in enumerate(ranking)
    }


def policy_result_row(frame: dict, selected: dict | None, policy_name: str) -> dict[str, object]:
    return {
        "policy": policy_name,
        "group_id": frame.get("group_id"),
        "split": frame.get("split"),
        "scene": frame.get("scene"),
        "frame": frame.get("frame"),
        "role": frame.get("role"),
        "candidate_count": len(frame.get("candidate_results", [])),
        "selected_candidate_rank": selected.get("candidate_rank") if selected else None,
        "selected_grasp_score": selected.get("selected_grasp_score") if selected else None,
        "selected_target_object_id": selected.get("target_object_id") if selected else None,
        "selected_target_object_name": selected.get("target_object_name") if selected else None,
        "lift_success": bool(selected.get("lift_success")) if selected else False,
        "failure_reason": selected.get("failure_reason") if selected else "missing_candidate",
        "simulation_unstable": bool(selected.get("simulation_unstable")) if selected else False,
        "target_lift_delta_m": selected.get("target_lift_delta_m") if selected else None,
        "max_target_lift_delta_m": selected.get("max_target_lift_delta_m") if selected else None,
        "pointcloud_feasible": selected.get("pointcloud_feasible") if selected else None,
        "pointcloud_failure_reason": selected.get("pointcloud_failure_reason") if selected else None,
        "pointcloud_collision_iou": selected.get("pointcloud_collision_iou") if selected else None,
        "pointcloud_empty_ratio": selected.get("pointcloud_empty_ratio") if selected else None,
        "grasp_width_m": selected.get("grasp_width_m") if selected else None,
        "opening_over_limit_m": selected.get("opening_over_limit_m") if selected else None,
        "object_rank": selected.get("object_rank") if selected else None,
        "object_adaptive_v2_score_norm": selected.get("object_adaptive_v2_score_norm") if selected else None,
        "object_grasp_risk": selected.get("object_grasp_risk") if selected else None,
        "object_high_risk": selected.get("object_high_risk") if selected else None,
    }


def candidate_label_row(frame: dict, candidate: dict) -> dict[str, object]:
    return {
        "group_id": frame.get("group_id"),
        "split": frame.get("split"),
        "scene": frame.get("scene"),
        "frame": frame.get("frame"),
        "candidate_rank": candidate.get("candidate_rank"),
        "selected_grasp_score": candidate.get("selected_grasp_score"),
        "target_object_id": candidate.get("target_object_id"),
        "target_object_name": candidate.get("target_object_name"),
        "lift_success": bool(candidate.get("lift_success")),
        "failure_reason": candidate.get("failure_reason"),
        "simulation_unstable": bool(candidate.get("simulation_unstable")),
        "target_lift_delta_m": candidate.get("target_lift_delta_m"),
        "max_target_lift_delta_m": candidate.get("max_target_lift_delta_m"),
        "pointcloud_feasible": candidate.get("pointcloud_feasible"),
        "pointcloud_failure_reason": candidate.get("pointcloud_failure_reason"),
        "pointcloud_collision_iou": candidate.get("pointcloud_collision_iou"),
        "pointcloud_empty_ratio": candidate.get("pointcloud_empty_ratio"),
        "grasp_width_m": candidate.get("grasp_width_m"),
        "opening_over_limit_m": candidate.get("opening_over_limit_m"),
        "object_rank": candidate.get("object_rank"),
        "object_adaptive_v2_score_norm": candidate.get("object_adaptive_v2_score_norm"),
        "object_grasp_risk": candidate.get("object_grasp_risk"),
        "object_high_risk": candidate.get("object_high_risk"),
    }


def aggregate_policy_rows(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    by_policy: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy"])].append(dict(row))
    return {
        policy: {
            "frame_count": len(items),
            "success_count": sum(1 for item in items if item.get("lift_success")),
            "success_rate": _rate(sum(1 for item in items if item.get("lift_success")), len(items)),
            "mean_selected_rank": _mean(
                [float(item["selected_candidate_rank"]) for item in items if item.get("selected_candidate_rank") is not None]
            ),
            "simulation_unstable_count": sum(1 for item in items if item.get("simulation_unstable")),
            "failure_reason_counts": dict(Counter(str(item.get("failure_reason")) for item in items)),
        }
        for policy, items in sorted(by_policy.items())
    }


def aggregate_candidate_labels(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_rank: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_rank[int(row.get("candidate_rank", -1))].append(dict(row))
    rank_rows = []
    for rank, items in sorted(by_rank.items()):
        rank_rows.append(
            {
                "candidate_rank": rank,
                "candidate_count": len(items),
                "success_count": sum(1 for item in items if item.get("lift_success")),
                "success_rate": _rate(sum(1 for item in items if item.get("lift_success")), len(items)),
                "mean_graspnet_score": _mean(
                    [float(item["selected_grasp_score"]) for item in items if item.get("selected_grasp_score") is not None]
                ),
                "mean_target_lift_delta_m": _mean(
                    [float(item["target_lift_delta_m"]) for item in items if item.get("target_lift_delta_m") is not None]
                ),
                "simulation_unstable_count": sum(1 for item in items if item.get("simulation_unstable")),
            }
        )
    return {
        "overall_success_count": sum(1 for row in rows if row.get("lift_success")),
        "overall_candidate_count": len(rows),
        "overall_success_rate": _rate(sum(1 for row in rows if row.get("lift_success")), len(rows)),
        "by_rank": rank_rows,
    }


def write_policy_results_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "policy",
        "group_id",
        "split",
        "scene",
        "frame",
        "role",
        "candidate_count",
        "selected_candidate_rank",
        "selected_grasp_score",
        "selected_target_object_id",
        "selected_target_object_name",
        "lift_success",
        "failure_reason",
        "simulation_unstable",
        "target_lift_delta_m",
        "max_target_lift_delta_m",
        "pointcloud_feasible",
        "pointcloud_failure_reason",
        "pointcloud_collision_iou",
        "pointcloud_empty_ratio",
        "grasp_width_m",
        "opening_over_limit_m",
        "object_rank",
        "object_adaptive_v2_score_norm",
        "object_grasp_risk",
        "object_high_risk",
    ]
    _write_csv(path, fieldnames, rows)


def write_candidate_label_analysis_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "candidate_rank",
        "candidate_count",
        "success_count",
        "success_rate",
        "mean_graspnet_score",
        "mean_target_lift_delta_m",
        "simulation_unstable_count",
    ]
    _write_csv(path, fieldnames, rows)


def _policy_functions() -> dict[str, PolicyFn]:
    return {
        "top1": choose_top1_candidate,
        "graspnet-safe-rerank": choose_graspnet_safe_rerank_candidate,
        "graspnet-score": choose_graspnet_score_candidate,
        "object-consensus": choose_object_consensus_candidate,
        "object-consensus-score": choose_object_consensus_score_candidate,
        "pointcloud-feasible-score": choose_pointcloud_feasible_score_candidate,
        "pointcloud-soft-score": choose_pointcloud_soft_score_candidate,
        "pointcloud-low-collision": choose_pointcloud_low_collision_candidate,
        "od-pointcloud-compact": choose_od_pointcloud_compact_candidate,
    }


def _target_key(candidate: dict) -> str:
    if candidate.get("target_object_id") is not None:
        return str(candidate.get("target_object_id"))
    return str(candidate.get("target_object_name"))


def _rank(candidate: dict) -> int:
    return int(candidate.get("candidate_rank", 10**6))


def _float_value(value: object, *, default: float = float("-inf")) -> float:
    if value is None:
        return default
    return float(value)


def _pointcloud_soft_score(candidate: dict) -> float:
    score = _float_value(candidate.get("selected_grasp_score"), default=0.0)
    reason = str(candidate.get("pointcloud_failure_reason") or "")
    opening_over_limit = max(0.0, _float_value(candidate.get("opening_over_limit_m"), default=0.0))

    if bool(candidate.get("pointcloud_feasible")):
        return score + 0.03
    if reason == "opening-too-small":
        return score - min(opening_over_limit / 0.02, 1.0) * 0.35
    if reason == "pointcloud-collision":
        collision_iou = max(0.0, _float_value(candidate.get("pointcloud_collision_iou"), default=1.0))
        return score - min(collision_iou / 0.05, 1.0) * 0.35
    if reason == "empty-grasp":
        empty_ratio = max(0.0, _float_value(candidate.get("pointcloud_empty_ratio"), default=0.0))
        return score - max(0.0, 0.05 - empty_ratio) / 0.05 * 0.25
    return score - 0.08


def _is_graspnet_safe_candidate(candidate: dict) -> bool:
    if _has_unbound_status(candidate):
        return False
    reason = str(candidate.get("pointcloud_failure_reason") or "")
    return reason not in {"binding-background", "no-bound-object"}


def _safe_rerank_score(candidate: dict) -> float:
    score = _float_value(candidate.get("selected_grasp_score"), default=0.0)
    object_score = _float_value(candidate.get("object_adaptive_v2_score_norm"), default=0.5)
    feasible_bonus = 0.03 if bool(candidate.get("pointcloud_feasible")) else 0.0
    rank_penalty = 0.03 * float(_rank(candidate))
    return score + 0.08 * object_score + feasible_bonus - rank_penalty


def _od_pointcloud_compact_score(candidate: dict) -> float:
    score = _float_value(candidate.get("selected_grasp_score"), default=0.0)
    object_score = _float_value(candidate.get("object_adaptive_v2_score_norm"), default=0.5)
    empty_ratio = max(0.0, _float_value(candidate.get("pointcloud_empty_ratio"), default=0.0))
    collision_iou = max(0.0, _float_value(candidate.get("pointcloud_collision_iou"), default=1.0))
    opening_over_limit = max(0.0, _float_value(candidate.get("opening_over_limit_m"), default=0.0))
    opening_penalty = min(opening_over_limit / 0.02, 1.0)
    feasible_bonus = 0.05 if bool(candidate.get("pointcloud_feasible")) else 0.0
    high_risk_penalty = 0.30 if bool(candidate.get("object_high_risk")) else 0.0
    binding_penalty = 0.70 if _has_unbound_status(candidate) else 0.0
    return (
        0.20 * score
        + 0.60 * object_score
        + feasible_bonus
        - 1.20 * empty_ratio
        - 0.50 * collision_iou
        - 0.50 * opening_penalty
        - high_risk_penalty
        - binding_penalty
    )


def _has_unbound_status(candidate: dict) -> bool:
    status = candidate.get("binding_status")
    return status is not None and str(status) != "bound"


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _rate(count: int, total: int) -> float | None:
    return round(float(count) / float(total), 6) if total else None


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


if __name__ == "__main__":
    main()
