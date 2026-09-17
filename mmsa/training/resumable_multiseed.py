"""Resumable multi-seed training with a cache/config provenance manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

from mmsa.training.multiseed import _aggregate, _round
from mmsa.training.train import load_config, train


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _manifest(config_path: Path, arch: str, seeds: list[int], interp_dir: Path) -> dict:
    cache_files = {}
    for name in ("interpretation_metadata.json", "train.npy", "valid.npy", "test.npy"):
        path = interp_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required cache artifact missing: {path}")
        cache_files[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return {
        "schema_version": 1,
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "arch": arch,
        "seeds": seeds,
        "interpretation_dir": str(interp_dir.resolve()),
        "cache_files": cache_files,
    }


def _load_finished(path: Path, dataset: str, arch: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("dataset") != dataset or report.get("arch") != arch:
        raise RuntimeError(
            f"Stale report at {path}: dataset/arch="
            f"{report.get('dataset')}/{report.get('arch')}, expected {dataset}/{arch}"
        )
    if not report.get("use_interpretation"):
        raise RuntimeError(f"Stale report at {path}: interpretation branch was disabled")
    test = report.get("test") or {}
    if "mae" not in test or "corr" not in test:
        raise RuntimeError(f"Incomplete report at {path}")
    if float(test.get("nonfinite_preds", 0.0)) != 0.0:
        raise RuntimeError(f"Diverged report at {path}: nonfinite predictions present")
    return report


def run(
    config_path: str,
    arch: str,
    seeds: list[int],
    output_dir: str,
    condition: str = "raw_transcript_direct_embedding",
) -> dict:
    interp_value = os.environ.get("TEMOE_INTERP_DIR")
    if not interp_value:
        raise RuntimeError("TEMOE_INTERP_DIR must point to the controlled interpretation cache")
    config_file = Path(config_path)
    interp_dir = Path(interp_value)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    manifest = _manifest(config_file, arch, seeds, interp_dir)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError(
                f"Resume provenance mismatch at {manifest_path}; do not mix cache/config variants"
            )
        print(f"[resume] manifest verified: {manifest_path}", flush=True)
    else:
        _atomic_json(manifest_path, manifest)
        print(f"[manifest] wrote {manifest_path}", flush=True)

    base = load_config(str(config_file))
    dataset = str(base["data"].get("name"))
    per_seed: list[dict] = []
    started = time.monotonic()
    trained_this_session = 0
    for position, seed in enumerate(seeds, 1):
        seed_dir = output / f"seed_{seed}"
        report_path = seed_dir / "temoe_report.json"
        if report_path.is_file():
            report = _load_finished(report_path, dataset, arch)
            action = "reused"
        else:
            config = copy.deepcopy(base)
            config["seed"] = seed
            config["model"]["arch"] = arch
            config["training"]["output_dir"] = str(seed_dir)
            print(f"\n===== arch={arch} seed={seed} ({position}/{len(seeds)}) =====", flush=True)
            report = train(config)
            report = _load_finished(report_path, dataset, arch)
            trained_this_session += 1
            action = "trained"

        per_seed.append({"seed": seed, **report["test"]})
        elapsed = time.monotonic() - started
        if trained_this_session:
            remaining = len(seeds) - position
            eta = elapsed / trained_this_session * remaining
            eta_text = f"{eta / 60:.1f} min"
        else:
            eta_text = "pending first new seed"
        print(
            f"[progress] arch={arch} completed={position}/{len(seeds)} seed={seed} "
            f"action={action} mae={report['test']['mae']:.6f} eta={eta_text}",
            flush=True,
        )

    aggregate = _aggregate(per_seed)
    result = {
        "arch": arch,
        "dataset": dataset,
        "condition": condition,
        "seeds": seeds,
        "per_seed": per_seed,
        "mean": aggregate["mean"],
        "std": aggregate["std"],
        "manifest": str(manifest_path),
    }
    result_path = output / f"multiseed_{arch}.json"
    _atomic_json(result_path, result)
    print(
        json.dumps(
            {"arch": arch, "mean": _round(aggregate["mean"]), "std": _round(aggregate["std"])},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--condition",
        default="raw_transcript_direct_embedding",
        help="condition identity stored in the aggregate report",
    )
    args = parser.parse_args()
    run(args.config, args.arch, args.seeds, args.output_dir, args.condition)


if __name__ == "__main__":
    main()
