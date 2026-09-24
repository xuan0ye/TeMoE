"""Build a small, auditable TeMoE release bundle from an explicit allowlist."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


ROOT_FILES = (
    "ARTIFACT.md",
    "run_release_audit_gated.sh",
    "run_mosei_source_control.sh",
    "run_zero_shot_gated.sh",
    "run_raw_transcript_control_gated.sh",
    "run_text_teacher_control_gated.sh",
    "requirements-server.txt",
)

RESULT_FILES = (
    "outputs/mmsa/official/control_caches.json",
    "outputs/mmsa/official/equivalence_tests.json",
    "outputs/mmsa/official/equivalence_tests.md",
    "outputs/mmsa/official/official_baselines.json",
    "outputs/mmsa/official/qwen25vl_zeroshot_summary.json",
    "outputs/mmsa/official/raw_transcript_control.json",
    "outputs/mmsa/official/text_teacher_control.json",
    "outputs/mmsa/official/env_provenance.txt",
    "outputs/mmsa/official/env_release_audit_wjhenv.txt",
    "outputs/mmsa/official/env_release_audit_mmsa.txt",
    "outputs/mmsa/official/official_crossing_mosi_full.json",
    "outputs/mmsa/official/official_crossing_mosi_full.md",
    "outputs/mmsa/condition_audit.json",
    "outputs/mmsa/condition_audit.md",
    "outputs/mmsa/crossing_paired_stats.md",
    "outputs/mmsa/mosei/efficiency/end_to_end.json",
    "outputs/mmsa/mosei/efficiency/end_to_end_v2_full.json",
    "outputs/mmsa/mosei/analysis/mosei_source_control.json",
    "outputs/mmsa/mosei/analysis/mosei_source_control.md",
)

EXCLUDED_NAMES = {"__pycache__", ".pytest_cache"}
MAX_FILE_BYTES = 50 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ("mmsa/**/*.py", "mmsa/configs/*.yaml"):
        files.extend(path for path in root.glob(pattern) if path.is_file())
    for name in ROOT_FILES + RESULT_FILES:
        path = root / name
        if path.is_file():
            files.append(path)
    unique = sorted(set(files))
    return [
        path for path in unique
        if not any(part in EXCLUDED_NAMES for part in path.relative_to(root).parts)
    ]


def destination(relative: Path) -> Path:
    parts = relative.parts
    if len(parts) >= 3 and parts[0] == "outputs" and parts[1] == "mmsa":
        return Path("results") / Path(*parts[2:])
    if relative == Path("ARTIFACT.md"):
        return Path("README.md")
    return relative


def build(output_dir: str, archive: str) -> dict:
    root = Path.cwd().resolve()
    output = Path(output_dir).resolve()
    archive_path = Path(archive).resolve()
    if output == root:
        raise ValueError("output directory must not be the repository root")
    output.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for source in source_files(root):
        if source.stat().st_size > MAX_FILE_BYTES:
            raise RuntimeError(f"refusing to publish file over 50 MiB: {source}")
        relative = source.relative_to(root)
        target_relative = destination(relative)
        target = output / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest_rows.append({
            "path": target_relative.as_posix(),
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        })

    manifest = {
        "schema_version": 1,
        "file_count": len(manifest_rows),
        "files": sorted(manifest_rows, key=lambda row: row["path"]),
        "excluded_categories": [
            "datasets and raw video",
            "model weights and trained checkpoints",
            "per-sample rationale text and embedding caches",
            "logs, PID files and historical drafts",
        ],
    }
    manifest_path = output / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(output, arcname=output.name)
    report = {
        "output_dir": str(output),
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "file_count": len(manifest_rows),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="release/temoe_artifact")
    parser.add_argument("--archive", default="release/temoe_artifact.tgz")
    args = parser.parse_args()
    build(args.output_dir, args.archive)


if __name__ == "__main__":
    main()
