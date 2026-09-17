"""Verify the public release without requiring datasets, weights, or a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.json"
REPORT = ROOT / "SELF_CHECK_REPORT.json"
MAX_BYTES = 50 * 1024 * 1024

IGNORED_PARTS = {".git", ".pytest_cache", "__pycache__"}
UNMANIFESTED = {"MANIFEST.json", "SELF_CHECK_REPORT.json"}
FORBIDDEN_SUFFIXES = {
    ".7z", ".avi", ".ckpt", ".doc", ".docx", ".gz", ".jpeg", ".jpg",
    ".jsonl", ".log", ".m4a", ".mkv", ".mov", ".mp3", ".mp4", ".npy",
    ".npz", ".pdf", ".pid", ".pickle", ".pkl", ".pt", ".pth", ".tar",
    ".tgz", ".wav", ".zip",
}
TEXT_SUFFIXES = {
    ".cfg", ".csv", ".ini", ".json", ".md", ".py", ".sh", ".txt", ".yaml", ".yml",
}
PRIVATE_PATTERNS = {
    "unix_home": re.compile(r"/home/[^/\s]+/", re.IGNORECASE),
    "windows_user": re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\s]+", re.IGNORECASE),
    "known_private_marker": re.compile(r"sylai|gzk8s|administrator", re.IGNORECASE),
    "credential_assignment": re.compile(
        r"(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*[\"'][^\"']+",
        re.IGNORECASE,
    ),
}
REQUIRED = {
    "README.md",
    "REPRODUCIBILITY.md",
    "RELEASE_POLICY.md",
    "ENVIRONMENT.md",
    "LICENSE-NOTICE.md",
    "requirements.lock.txt",
    "requirements-mmsa-official.lock.txt",
    "results/RESULTS.md",
    "mmsa/models/model.py",
    "mmsa/training/train.py",
    "run_release_audit_gated.sh",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_files() -> list[Path]:
    rows: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if relative.as_posix() in UNMANIFESTED:
            continue
        rows.append(relative)
    return sorted(rows, key=lambda item: item.as_posix())


def write_manifest() -> None:
    rows = []
    for relative in relative_files():
        path = ROOT / relative
        rows.append({
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    payload = {
        "schema_version": 1,
        "artifact": "TeMoE controlled-study release candidate",
        "file_count": len(rows),
        "files": rows,
        "excluded_categories": [
            "datasets and raw media",
            "model weights and trained checkpoints",
            "per-sample rationales, predictions, and embedding caches",
            "logs, archives, paper files, and private provenance",
        ],
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def verify_manifest(errors: list[str], checks: dict[str, object]) -> None:
    if not MANIFEST.is_file():
        errors.append("MANIFEST.json is missing")
        return
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"MANIFEST.json cannot be parsed: {exc}")
        return
    listed = {row["path"]: row for row in manifest.get("files", [])}
    actual = {path.as_posix() for path in relative_files()}
    if set(listed) != actual:
        missing = sorted(set(listed) - actual)
        extra = sorted(actual - set(listed))
        errors.append(f"manifest set mismatch: missing={missing}, extra={extra}")
    for relative, row in listed.items():
        path = ROOT / relative
        if not path.is_file():
            continue
        size = path.stat().st_size
        digest = sha256(path)
        if size != row.get("bytes"):
            errors.append(f"size mismatch: {relative}")
        if digest != row.get("sha256"):
            errors.append(f"sha256 mismatch: {relative}")
    if manifest.get("file_count") != len(listed):
        errors.append("manifest file_count does not match files array")
    checks["manifest_entries"] = len(listed)


def inspect_payload(errors: list[str], checks: dict[str, object]) -> None:
    files = relative_files()
    actual = {path.as_posix() for path in files}
    absent = sorted(REQUIRED - actual)
    if absent:
        errors.append(f"required files missing: {absent}")

    parsed_json = 0
    compiled_python = 0
    scanned_text = 0
    for relative in files:
        path = ROOT / relative
        if path.stat().st_size > MAX_BYTES:
            errors.append(f"file exceeds 50 MiB: {relative.as_posix()}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden payload type: {relative.as_posix()}")
        if path.suffix.lower() == ".sh" and b"\r" in path.read_bytes():
            errors.append(f"shell script is not LF-only: {relative.as_posix()}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            errors.append(f"non-UTF-8 text file: {relative.as_posix()}: {exc}")
            continue
        scanned_text += 1
        # The checker necessarily contains the literal signatures it searches
        # for, so scan every text payload except this signature-definition file.
        if relative.as_posix() != "scripts/release_check.py":
            for name, pattern in PRIVATE_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"private pattern {name}: {relative.as_posix()}")
        if path.suffix.lower() == ".json":
            try:
                json.loads(text)
                parsed_json += 1
            except json.JSONDecodeError as exc:
                errors.append(f"invalid JSON: {relative.as_posix()}: {exc}")
        if path.suffix.lower() == ".py":
            try:
                compile(text, relative.as_posix(), "exec")
                compiled_python += 1
            except SyntaxError as exc:
                errors.append(f"Python syntax error: {relative.as_posix()}: {exc}")
    checks.update({
        "payload_files": len(files),
        "json_files_parsed": parsed_json,
        "python_files_compiled": compiled_python,
        "text_files_scanned": scanned_text,
    })


def run_tests(errors: list[str], checks: dict[str, object]) -> None:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "mmsa/tests",
    ]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    checks["pytest_command"] = "python -m pytest -q -p no:cacheprovider mmsa/tests"
    checks["pytest_exit_code"] = completed.returncode
    checks["pytest_output"] = completed.stdout[-4000:]
    if completed.returncode != 0:
        errors.append("pytest failed; see SELF_CHECK_REPORT.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()

    if args.write_manifest:
        write_manifest()

    errors: list[str] = []
    checks: dict[str, object] = {}
    verify_manifest(errors, checks)
    inspect_payload(errors, checks)
    if args.run_tests:
        run_tests(errors, checks)

    report = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "checks": checks,
        "errors": errors,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
