from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the abstract grasp-risk score against physically validated grasp outcomes "
            "(MuJoCo dynamic lift validation), and suggest a data-backed risk threshold. "
            "Semantics: a grasp is predicted successful when risk < threshold."
        )
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        help="CSV with columns risk,success (success in {0,1,true,false}); optional frame,object columns.",
    )
    parser.add_argument(
        "--frame-results",
        type=Path,
        help="frame_results.csv from a point-cloud episode run (frame, selected_object, grasp_risk).",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        help="Directory scanned recursively for dynamic validation summary JSONs (validation.lift_success).",
    )
    parser.add_argument("--out-dir", type=Path, help="Output directory for CSV/JSON/PNG results.")
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG plot generation.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    return parser.parse_args()


def compute_threshold_table(
    risks: Sequence[float],
    successes: Sequence[bool],
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, object]]:
    risk_arr = np.asarray(risks, dtype=float)
    success_arr = np.asarray(successes, dtype=bool)
    if risk_arr.shape != success_arr.shape or risk_arr.ndim != 1:
        raise ValueError("risks and successes must be 1-D sequences of equal length.")
    if thresholds is None:
        candidates = np.unique(np.concatenate([risk_arr, [0.0, 1.0]]))
        # Midpoints between observed risks make decisions unambiguous.
        midpoints = (candidates[:-1] + candidates[1:]) / 2.0
        thresholds = np.unique(np.concatenate([[0.0], midpoints, [candidates.max() + 1e-6]]))

    positives = int(success_arr.sum())
    negatives = int((~success_arr).sum())
    table = []
    for threshold in thresholds:
        predicted = risk_arr < float(threshold)
        tp = int((predicted & success_arr).sum())
        fp = int((predicted & ~success_arr).sum())
        fn = int((~predicted & success_arr).sum())
        tn = int((~predicted & ~success_arr).sum())
        tpr = tp / positives if positives else 0.0
        fpr = fp / negatives if negatives else 0.0
        precision = tp / (tp + fp) if (tp + fp) else None
        accuracy = (tp + tn) / len(risk_arr) if len(risk_arr) else None
        table.append(
            {
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "tpr": float(tpr),
                "fpr": float(fpr),
                "precision": None if precision is None else float(precision),
                "accuracy": None if accuracy is None else float(accuracy),
                "youden_j": float(tpr - fpr),
            }
        )
    return table


def compute_auc(risks: Sequence[float], successes: Sequence[bool]) -> float | None:
    """Rank-based AUC: probability that a successful grasp has lower risk than a failed one."""
    risk_arr = np.asarray(risks, dtype=float)
    success_arr = np.asarray(successes, dtype=bool)
    pos = risk_arr[success_arr]
    neg = risk_arr[~success_arr]
    if pos.size == 0 or neg.size == 0:
        return None
    wins = (pos[:, None] < neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((wins + 0.5 * ties) / (pos.size * neg.size))


def reliability_bins(risks: Sequence[float], successes: Sequence[bool], bins: int = 10) -> list[dict[str, object]]:
    risk_arr = np.asarray(risks, dtype=float)
    success_arr = np.asarray(successes, dtype=bool)
    edges = np.linspace(0.0, max(1.0, float(risk_arr.max()) if risk_arr.size else 1.0), int(bins) + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (risk_arr >= low) & (risk_arr < high) if high < edges[-1] else (risk_arr >= low) & (risk_arr <= high)
        count = int(mask.sum())
        rows.append(
            {
                "risk_low": float(low),
                "risk_high": float(high),
                "count": count,
                "success_rate": float(success_arr[mask].mean()) if count else None,
            }
        )
    return rows


def suggest_thresholds(table: Sequence[dict[str, object]], target_precision: float = 0.9) -> dict[str, object]:
    best_youden = None
    for row in table:
        if best_youden is None or float(row["youden_j"]) > float(best_youden["youden_j"]):
            best_youden = row
    precise = [
        row
        for row in table
        if row["precision"] is not None and float(row["precision"]) >= float(target_precision) and int(row["tp"]) > 0
    ]
    # The largest threshold still meeting the precision target accepts the most grasps.
    best_precise = max(precise, key=lambda row: float(row["threshold"])) if precise else None
    return {
        "max_youden": None if best_youden is None else dict(best_youden),
        "target_precision": float(target_precision),
        "max_threshold_at_target_precision": None if best_precise is None else dict(best_precise),
    }


def load_pairs_csv(path: Path) -> list[dict[str, object]]:
    pairs = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pairs.append(
                {
                    "risk": float(row["risk"]),
                    "success": _parse_bool(row["success"]),
                    "frame": row.get("frame"),
                    "object": row.get("object"),
                }
            )
    return pairs


def collect_pairs_from_validation(
    frame_results_path: Path,
    validation_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Join episode risk scores with dynamic validation lift outcomes.

    Join key is (frame, object) when the validation summary carries a frame id,
    otherwise object name alone (skipped as ambiguous when several episode rows share it).
    """
    episode_rows = []
    with Path(frame_results_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            risk = row.get("grasp_risk")
            if risk in (None, "", "None"):
                continue
            episode_rows.append(
                {
                    "frame": str(row.get("frame") or ""),
                    "object": str(row.get("selected_object") or ""),
                    "risk": float(risk),
                }
            )

    pairs: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for summary_path in sorted(Path(validation_dir).rglob("*.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped.append({"path": str(summary_path), "reason": "unreadable-json"})
            continue
        validation = payload.get("validation") if isinstance(payload, dict) else None
        if not isinstance(validation, dict) or "lift_success" not in validation:
            skipped.append({"path": str(summary_path), "reason": "no-lift-success"})
            continue
        object_name = payload.get("target_object_name")
        frame = payload.get("frame") or payload.get("frame_id")
        matches = [
            row
            for row in episode_rows
            if row["object"] == str(object_name)
            and (frame is None or str(row["frame"]).lstrip("0") == str(frame).lstrip("0") or str(row["frame"]) == str(frame))
        ]
        if not matches:
            skipped.append({"path": str(summary_path), "reason": "no-episode-match", "object": object_name})
            continue
        if frame is None and len(matches) > 1:
            skipped.append({"path": str(summary_path), "reason": "ambiguous-object-match", "object": object_name})
            continue
        pairs.append(
            {
                "risk": float(matches[0]["risk"]),
                "success": bool(validation["lift_success"]),
                "frame": matches[0]["frame"],
                "object": matches[0]["object"],
            }
        )
    return pairs, skipped


def run_calibration(
    pairs: Sequence[dict[str, object]],
    *,
    out_dir: Path,
    target_precision: float = 0.9,
    bins: int = 10,
    make_plot: bool = True,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    risks = [float(pair["risk"]) for pair in pairs]
    successes = [bool(pair["success"]) for pair in pairs]
    table = compute_threshold_table(risks, successes)
    bins_rows = reliability_bins(risks, successes, bins=bins)
    summary = {
        "output_dir": str(out_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pair_count": len(pairs),
        "success_count": int(sum(successes)),
        "base_success_rate": float(np.mean(successes)) if pairs else None,
        "auc": compute_auc(risks, successes),
        "suggestions": suggest_thresholds(table, target_precision=target_precision),
        "reliability_bins": bins_rows,
    }
    _write_table_csv(out_dir / "calibration_table.csv", table)
    _write_bins_csv(out_dir / "reliability_bins.csv", bins_rows)
    (out_dir / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if make_plot and pairs:
        plot_path = _plot_calibration(out_dir, table, bins_rows, summary)
        if plot_path is not None:
            summary["plot_path"] = str(plot_path)
    return summary


def _plot_calibration(
    out_dir: Path,
    table: Sequence[dict[str, object]],
    bins_rows: Sequence[dict[str, object]],
    summary: dict[str, object],
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, (ax_roc, ax_rel) = plt.subplots(1, 2, figsize=(10, 4))
    fprs = [row["fpr"] for row in table]
    tprs = [row["tpr"] for row in table]
    order = np.argsort(fprs)
    ax_roc.plot(np.asarray(fprs)[order], np.asarray(tprs)[order], marker="o", markersize=3)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax_roc.set_xlabel("FPR (accepted failures)")
    ax_roc.set_ylabel("TPR (accepted successes)")
    ax_roc.set_title(f"ROC (AUC={summary['auc']:.3f})" if summary.get("auc") is not None else "ROC")

    centers = [(row["risk_low"] + row["risk_high"]) / 2.0 for row in bins_rows]
    rates = [row["success_rate"] if row["success_rate"] is not None else np.nan for row in bins_rows]
    ax_rel.bar(centers, rates, width=(centers[1] - centers[0]) * 0.9 if len(centers) > 1 else 0.08, color="#4878a8")
    ax_rel.set_xlabel("grasp risk")
    ax_rel.set_ylabel("actual lift success rate")
    ax_rel.set_ylim(0, 1.05)
    ax_rel.set_title("Risk reliability")

    fig.tight_layout()
    plot_path = out_dir / "calibration.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def _write_table_csv(path: Path, table: Sequence[dict[str, object]]) -> None:
    fieldnames = ["threshold", "tp", "fp", "tn", "fn", "tpr", "fpr", "precision", "accuracy", "youden_j"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)


def _write_bins_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = ["risk_low", "risk_high", "count", "success_rate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean from {value!r}.")


def _resolve_output_dir(out_dir: Path | None) -> Path:
    if out_dir is None:
        return PROJECT_ROOT / "results" / "risk_threshold_calibration" / (
            "calibration_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    return out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir


def main() -> None:
    args = parse_args()
    skipped: list[dict[str, object]] = []
    if args.pairs_csv is not None:
        pairs = load_pairs_csv(args.pairs_csv)
    elif args.frame_results is not None and args.validation_dir is not None:
        pairs, skipped = collect_pairs_from_validation(args.frame_results, args.validation_dir)
    else:
        raise SystemExit("Provide either --pairs-csv, or both --frame-results and --validation-dir.")
    if not pairs:
        raise SystemExit("No (risk, success) pairs collected; nothing to calibrate.")

    summary = run_calibration(
        pairs,
        out_dir=_resolve_output_dir(args.out_dir),
        target_precision=args.target_precision,
        bins=args.bins,
        make_plot=not args.no_plot,
    )
    summary["skipped_validation_files"] = skipped
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print("Risk threshold calibration finished")
    print(f"  output_dir: {summary['output_dir']}")
    print(f"  pairs: {summary['pair_count']} (base success rate {summary['base_success_rate']})")
    print(f"  AUC: {summary['auc']}")
    best = summary["suggestions"]["max_youden"]
    if best is not None:
        print(f"  max-Youden threshold: {best['threshold']:.4f} (TPR={best['tpr']:.3f}, FPR={best['fpr']:.3f})")
    precise = summary["suggestions"]["max_threshold_at_target_precision"]
    if precise is not None:
        print(
            f"  max threshold at precision>={summary['suggestions']['target_precision']}: "
            f"{precise['threshold']:.4f} (precision={precise['precision']:.3f})"
        )
    if skipped:
        print(f"  skipped validation files: {len(skipped)} (see JSON for reasons)")


if __name__ == "__main__":
    main()
