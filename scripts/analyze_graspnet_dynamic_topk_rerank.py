from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PolicyFn = Callable[[Sequence[dict]], object]


POLICY_ORDER = (
    "top1",
    "graspnet-score",
    "object-consensus",
    "object-consensus-score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze non-oracle reranking policies on saved GraspNet dynamic top-K results."
    )
    parser.add_argument("--summary", type=Path, required=True, help="Path to split_dynamic_topk_summary.json.")
    parser.add_argument("--out-dir", type=Path, help="Defaults to <summary parent>/rerank_analysis.")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_dynamic_topk_rerank(
        summary_path=args.summary,
        out_dir=args.out_dir or args.summary.parent / "rerank_analysis",
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
) -> dict[str, object]:
    source = Path(summary_path)
    topk_summary = json.loads(source.read_text(encoding="utf-8"))
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    policy_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for frame in topk_summary.get("frame_results", []):
        candidates = sorted(
            [dict(candidate) for candidate in frame.get("candidate_results", [])],
            key=lambda item: int(item.get("candidate_rank", 10**6)),
        )
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
        "top_k": topk_summary.get("top_k"),
        "frame_count": len(topk_summary.get("frame_results", [])),
        "candidate_count": len(candidate_rows),
        "policy_aggregate": aggregate,
        "candidate_label_analysis": label_analysis,
        "policy_results": policy_rows,
    }

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
        "graspnet-score": choose_graspnet_score_candidate,
        "object-consensus": choose_object_consensus_candidate,
        "object-consensus-score": choose_object_consensus_score_candidate,
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
