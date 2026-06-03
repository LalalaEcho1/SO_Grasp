from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stacked_grasping.assets.ycb_starter import resolve_object_names


DEFAULT_BASE_URL = "https://ycb-benchmarks.s3.amazonaws.com/data/google"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official YCB Google 16k meshes.")
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["starter"],
        help="YCB object names, or 'starter' for the project starter set.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "assets" / "objects" / "ycb",
        help="Directory where normalized YCB meshes will be stored.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for official YCB Google scanner archives.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-download and replace existing object folders.")
    parser.add_argument("--keep-archives", action="store_true", help="Keep downloaded .tgz files under _downloads.")
    parser.add_argument(
        "--no-generate-scene",
        action="store_true",
        help="Only download meshes; do not generate assets/scenes/ycb_mesh_stacked.xml.",
    )
    return parser.parse_args()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as out_file:
        total = int(response.headers.get("content-length", "0") or 0)
        copied = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)
            copied += len(chunk)
            if total:
                percent = copied / total * 100
                print(f"    {copied / 1_000_000:7.1f} MB / {total / 1_000_000:7.1f} MB ({percent:5.1f}%)", end="\r")
        if total:
            print()


def safe_extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (dest / member.name).resolve()
            if dest_resolved not in member_path.parents and member_path != dest_resolved:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
        tar.extractall(dest)


def normalize_extracted_object(tmp_root: Path, object_name: str, target_root: Path, overwrite: bool) -> Path:
    obj_files = sorted(tmp_root.rglob("textured.obj"))
    if not obj_files:
        raise RuntimeError(f"{object_name}: textured.obj was not found in the extracted archive.")

    google_16k_dir = obj_files[0].parent
    target_dir = target_root / object_name / "google_16k"
    if target_dir.exists():
        if not overwrite:
            return target_dir
        shutil.rmtree(target_dir)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(google_16k_dir, target_dir)
    return target_dir


def run_scene_generator() -> None:
    generator = PROJECT_ROOT / "scripts" / "generate_ycb_mesh_scene.py"
    subprocess.run([sys.executable, str(generator)], check=True)


def main() -> None:
    args = parse_args()
    object_names = resolve_object_names(args.objects)
    download_dir = args.dataset_root / "_downloads"
    args.dataset_root.mkdir(parents=True, exist_ok=True)

    print("Downloading official YCB Google 16k meshes")
    for object_name in object_names:
        target_dir = args.dataset_root / object_name / "google_16k"
        if target_dir.exists() and not args.overwrite:
            print(f"  - {object_name}: already exists")
            continue

        archive_name = f"{object_name}_google_16k.tgz"
        archive = download_dir / archive_name
        url = f"{args.base_url.rstrip('/')}/{archive_name}"
        print(f"  - {object_name}: {url}")

        if not archive.exists() or args.overwrite:
            download_file(url, archive)

        with tempfile.TemporaryDirectory(prefix=f"ycb_{object_name}_") as tmp:
            tmp_root = Path(tmp)
            safe_extract(archive, tmp_root)
            normalized_dir = normalize_extracted_object(tmp_root, object_name, args.dataset_root, args.overwrite)
            print(f"    saved: {normalized_dir}")

        if not args.keep_archives and archive.exists():
            archive.unlink()

    if not args.keep_archives and download_dir.exists() and not any(download_dir.iterdir()):
        download_dir.rmdir()

    if not args.no_generate_scene:
        run_scene_generator()


if __name__ == "__main__":
    main()

