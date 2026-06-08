from __future__ import annotations

import re
from pathlib import Path


_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_LEGACY_PROJECT_NAMES = (
    "Stacked-Object Grasping",
    "SO_Grasp",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def is_windows_absolute_path(value: str | Path) -> bool:
    return bool(_WINDOWS_ABSOLUTE_RE.match(str(value)))


def resolve_project_path(value: str | Path, root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else project_root()
    text = str(value)
    project_relative = _legacy_project_relative_part(text, base.name)
    if project_relative is not None:
        return base / Path(project_relative)

    normalized = text.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        return path
    return base / path


def to_project_relative(value: str | Path, root: str | Path | None = None) -> str:
    base = Path(root) if root is not None else project_root()
    project_relative = _legacy_project_relative_part(str(value), base.name)
    if project_relative is not None:
        return project_relative

    path = Path(str(value).replace("\\", "/"))
    if path.is_absolute():
        try:
            return path.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix().lstrip("./")


def path_key(value: str | Path, root: str | Path | None = None) -> str:
    return to_project_relative(value, root=root).casefold()


def _legacy_project_relative_part(value: str, project_name: str) -> str | None:
    normalized = value.replace("\\", "/")
    for name in _project_name_candidates(project_name):
        marker = f"/{name}/"
        if marker in normalized:
            return normalized.split(marker, 1)[1]
        prefix = f"{name}/"
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return None


def _project_name_candidates(project_name: str) -> tuple[str, ...]:
    names = [project_name]
    names.extend(name for name in _LEGACY_PROJECT_NAMES if name != project_name)
    return tuple(names)
