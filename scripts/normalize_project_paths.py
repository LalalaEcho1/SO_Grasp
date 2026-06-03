from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stacked_grasping.utils.paths import to_project_relative


DEFAULT_JSON_PATHS = [
    PROJECT_ROOT / "assets" / "scenes" / "generated_main_v1" / "manifest.json",
    PROJECT_ROOT / "assets" / "scenes" / "generated_main_v1" / "difficulty_splits.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite project-local absolute paths in JSON files as relative paths.")
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_JSON_PATHS)
    return parser.parse_args()


def normalize_json_project_paths(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized, changed = _normalize_value(payload)
    if changed:
        path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


def main() -> None:
    args = parse_args()
    total = 0
    for path in args.paths:
        json_path = path if path.is_absolute() else PROJECT_ROOT / path
        changed = normalize_json_project_paths(json_path)
        total += changed
        print(f"{json_path}: normalized {changed} path value(s)")
    print(f"total normalized: {total}")


def _normalize_value(value: Any) -> tuple[Any, int]:
    if isinstance(value, dict):
        changed = 0
        result = {}
        for key, item in value.items():
            normalized_item, item_changed = _normalize_value(item)
            result[key] = normalized_item
            changed += item_changed
        return result, changed

    if isinstance(value, list):
        changed = 0
        result = []
        for item in value:
            normalized_item, item_changed = _normalize_value(item)
            result.append(normalized_item)
            changed += item_changed
        return result, changed

    if isinstance(value, str):
        normalized = to_project_relative(value)
        if normalized != value:
            return normalized, 1
    return value, 0


if __name__ == "__main__":
    main()
