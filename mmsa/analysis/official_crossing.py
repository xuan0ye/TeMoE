"""Official MMSA cache crossing on one dataset.

This module keeps MMSA 2.2.1's released model and trainer code unmodified.
The only adapter is at the feature boundary: a fixed rationale embedding is
concatenated to each valid text timestep.  The matched no-cache arm concatenates
zeros of the same dimension, so feature shapes, model parameter counts, seeds,
trainer and hyperparameters are identical across arms.

Typical use (MOSI, run each arm/model in its own process):

    python -m mmsa.analysis.official_crossing prepare \
      --source-pkl "MSA Datasets/MOSI/Processed/aligned_50.pkl" \
      --cache-dir data/interpretation/mosi \
      --output-dir outputs/mmsa/official_crossing/mosi/data

    python -m mmsa.analysis.official_crossing run --dataset mosi --model LMF \
      --arm no_cache --manifest outputs/mmsa/official_crossing/mosi/data/manifest.json \
      --run-root outputs/mmsa/official_crossing/mosi/runs

    python -m mmsa.analysis.official_crossing summarize \
      --run-root outputs/mmsa/official_crossing/mosi/runs \
      --output outputs/mmsa/official/official_crossing_mosi.json
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import pickle
import re
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

SPLITS = {"train": ("train",), "valid": ("valid", "val", "dev"), "test": ("test",)}
ARMS = ("no_cache", "cached")
RESULT = re.compile(r"Result for seed\s+(\d+):\s*(\{.*\})")
KV = re.compile(r"'([A-Za-z0-9_]+)':\s*(?:np\.float64\()?([-+0-9.eE]+)")
PARAMS = re.compile(r"model has\s+(\d+)\s+trainable parameters", re.IGNORECASE)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), len(files)


def _load_pickle(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        payload = pickle.loads(raw)
    except UnicodeDecodeError:
        payload = pickle.loads(raw, encoding="latin1")
    if not isinstance(payload, dict):
        raise TypeError(f"expected dict pickle root, got {type(payload).__name__}")
    return payload


def _split(payload: dict[str, Any], aliases: tuple[str, ...]) -> tuple[str, dict[str, Any]]:
    for name in aliases:
        value = payload.get(name)
        if isinstance(value, dict):
            return name, value
    raise KeyError(f"none of split aliases {aliases} found")


def _lengths(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"expected a [N,T,D] feature array, got {array.shape}")
    return np.full(array.shape[0], array.shape[1], dtype=np.int64)


def prepare_crossing(
    source_pkl: str,
    cache_dir: str,
    output_dir: str,
) -> dict[str, Any]:
    """Create dimension-matched cached/no-cache pkls and a cryptographic manifest."""
    source = Path(source_pkl)
    cache_root = Path(cache_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(source)

    cache_files = {split: cache_root / f"{split}.npy" for split in SPLITS}
    for path in cache_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "official MMSA unmodified model/trainer; rationale appended to text features",
        "source_pkl": str(source.resolve()),
        "source_pkl_sha256": _sha256(source),
        "cache_dir": str(cache_root.resolve()),
        "arms": {},
        "invariants": [
            "original text/audio/vision/labels are unchanged",
            "cached and no_cache pkls have identical shapes",
            "no_cache appends zeros with the same dimension as the rationale cache",
            "MMSA model and trainer source are not patched",
            "both arms use identical seeds and official hyperparameters",
        ],
    }

    for arm in ARMS:
        payload = _load_pickle(source)
        split_report: dict[str, Any] = {}
        for canonical, aliases in SPLITS.items():
            split_name, block = _split(payload, aliases)
            text = np.asarray(block["text"], dtype=np.float32)
            audio = np.asarray(block["audio"], dtype=np.float32)
            vision = np.asarray(block["vision"], dtype=np.float32)
            cache = np.load(cache_files[canonical]).astype(np.float32)
            if text.ndim != 3:
                raise ValueError(f"{canonical}.text must be [N,T,D], got {text.shape}")
            if cache.ndim != 2 or cache.shape[0] != text.shape[0]:
                raise ValueError(
                    f"{canonical} cache/text mismatch: cache={cache.shape}, text={text.shape}"
                )

            # The source aligned_50 features use a fixed 50-step contract in
            # this paper. Repeating the clip-level cache across those steps is
            # the least invasive way to expose it to an unmodified official
            # baseline. Both arms receive the same added dimensionality.
            repeated = np.broadcast_to(cache[:, None, :], (len(cache), text.shape[1], cache.shape[1]))
            injected = repeated if arm == "cached" else np.zeros_like(repeated)
            block["text"] = np.concatenate([text, injected], axis=-1).astype(np.float32)
            block.setdefault("text_lengths", _lengths(text))
            block.setdefault("audio_lengths", _lengths(audio))
            block.setdefault("vision_lengths", _lengths(vision))

            split_report[canonical] = {
                "source_split_key": split_name,
                "n": int(text.shape[0]),
                "seq_len": int(text.shape[1]),
                "source_text_dim": int(text.shape[2]),
                "cache_dim": int(cache.shape[1]),
                "output_text_dim": int(block["text"].shape[2]),
                "audio_dim": int(audio.shape[-1]),
                "vision_dim": int(vision.shape[-1]),
                "cache_sha256": _sha256(cache_files[canonical]),
            }

        out_path = output / f"{source.stem}_official_crossing_{arm}.pkl"
        with out_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=4)
        manifest["arms"][arm] = {
            "feature_path": str(out_path.resolve()),
            "feature_sha256": _sha256(out_path),
            "bytes": out_path.stat().st_size,
            "splits": split_report,
        }

    for split in SPLITS:
        a = manifest["arms"]["cached"]["splits"][split]
        b = manifest["arms"]["no_cache"]["splits"][split]
        for key in ("n", "seq_len", "output_text_dim", "audio_dim", "vision_dim"):
            if a[key] != b[key]:
                raise AssertionError(f"arm mismatch for {split}.{key}: {a[key]} != {b[key]}")

    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"wrote {manifest_path}")
    return manifest


def _config_for(get_config_regression, model: str, dataset: str):
    last: Exception | None = None
    for name in (model, model.lower(), model.upper()):
        try:
            return get_config_regression(name, dataset), name
        except Exception as exc:  # official API is case-sensitive across releases
            last = exc
    assert last is not None
    raise last


def run_arm(
    *,
    dataset: str,
    model: str,
    arm: str,
    manifest_path: str,
    run_root: str,
    seeds: list[int],
) -> dict[str, Any]:
    """Run one official MMSA (model, arm) pair in an isolated process."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    import importlib.metadata
    import MMSA
    from MMSA import MMSA_run, get_config_regression

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    feature_path = Path(manifest["arms"][arm]["feature_path"])
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    split_meta = manifest["arms"][arm]["splits"]["train"]
    feature_dims = [
        split_meta["output_text_dim"],
        split_meta["audio_dim"],
        split_meta["vision_dim"],
    ]

    supported = list(getattr(MMSA, "SUPPORTED_MODELS", []))
    canonical = {value.lower(): value for value in supported}
    if canonical and model.lower() not in canonical:
        raise ValueError(f"{model} is not supported by this MMSA build: {sorted(supported)}")
    official_name = canonical.get(model.lower(), model)

    pair_root = Path(run_root) / arm / official_name
    record_path = pair_root / "run_record.json"
    if record_path.is_file():
        prior = json.loads(record_path.read_text(encoding="utf-8"))
        if prior.get("status") == "ok" and prior.get("seeds") == seeds:
            print(f"[resume] verified completed run: {record_path}")
            return prior

    result_dir = pair_root / "results"
    log_dir = pair_root / "logs"
    model_dir = pair_root / "saved_models"
    for directory in (result_dir, log_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cfg, lookup_name = _config_for(get_config_regression, official_name, dataset)
    cfg["featurePath"] = str(feature_path)
    cfg["feature_dims"] = feature_dims
    cfg["seeds"] = seeds
    cfg["use_bert"] = False

    package_file = Path(inspect.getfile(MMSA))
    package_tree_hash, package_source_files = _source_tree_sha256(package_file.parent)
    normalized_cfg = dict(cfg)
    normalized_cfg["featurePath"] = "<ARM_FEATURE_PATH>"
    config_hash = hashlib.sha256(
        json.dumps(normalized_cfg, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    record: dict[str, Any] = {
        "schema_version": 1,
        "dataset": dataset,
        "model": official_name,
        "config_lookup_name": lookup_name,
        "arm": arm,
        "seeds": seeds,
        "feature_path": str(feature_path),
        "feature_sha256": _sha256(feature_path),
        "feature_dims": feature_dims,
        "use_bert": False,
        "mmsa_version": importlib.metadata.version("MMSA"),
        "mmsa_package_file": str(package_file),
        "mmsa_package_file_sha256": _sha256(package_file),
        "mmsa_python_source_tree_sha256": package_tree_hash,
        "mmsa_python_source_file_count": package_source_files,
        "official_config_sha256_except_feature_path": config_hash,
        "model_and_trainer_modified": False,
    }
    start = time.time()
    try:
        signature = inspect.signature(MMSA_run)
        kwargs = {
            "config": cfg,
            "seeds": seeds,
            "res_save_dir": str(result_dir),
            "log_dir": str(log_dir),
            "model_save_dir": str(model_dir),
            "gpu_ids": [0],
            "num_workers": 4,
            "verbose_level": 1,
        }
        accepted = set(signature.parameters)
        kwargs = {key: value for key, value in kwargs.items() if key in accepted}
        returned = MMSA_run(lookup_name, dataset, **kwargs)
    except Exception as exc:
        traceback.print_exc()
        record.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            seconds=round(time.time() - start, 1),
        )
    else:
        record.update(
            status="ok",
            returned=str(returned)[:500],
            seconds=round(time.time() - start, 1),
        )
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if record["status"] != "ok":
        raise RuntimeError(record["error"])
    return record


def _per_seed(log_path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for line in log_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        match = RESULT.search(line)
        if not match:
            continue
        rows[int(match.group(1))] = {key: float(value) for key, value in KV.findall(match.group(2))}
    return rows


def _trainable_params(log_path: Path) -> int:
    values = {
        int(match.group(1))
        for match in PARAMS.finditer(log_path.read_text(encoding="utf-8-sig", errors="replace"))
    }
    if len(values) != 1:
        raise RuntimeError(f"expected one trainable-parameter count in {log_path}, found {sorted(values)}")
    return values.pop()


def _holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, p_values[index] * (total - rank)))
        adjusted[index] = running
    return adjusted


def summarize_crossing(
    *,
    run_root: str,
    output_path: str,
    expected_models: list[str],
    expected_seeds: list[int],
) -> dict[str, Any]:
    """Pair official cached/no-cache seed results and test the MAE change."""
    from scipy import stats

    root = Path(run_root)
    comparisons: list[dict[str, Any]] = []
    for requested in expected_models:
        found: dict[str, tuple[str, Path, dict[int, dict[str, float]]]] = {}
        for arm in ARMS:
            arm_root = root / arm
            candidates = [
                directory for directory in arm_root.iterdir()
                if directory.is_dir() and directory.name.lower() == requested.lower()
            ] if arm_root.is_dir() else []
            if len(candidates) != 1:
                raise RuntimeError(f"expected one {arm}/{requested} directory, found {candidates}")
            model_dir = candidates[0]
            logs = sorted((model_dir / "logs").glob("*.log"))
            if len(logs) != 1:
                raise RuntimeError(f"expected one log under {model_dir / 'logs'}, found {logs}")
            rows = _per_seed(logs[0])
            missing = sorted(set(expected_seeds) - set(rows))
            if missing:
                raise RuntimeError(f"{arm}/{requested} is missing seed results: {missing}")
            found[arm] = (model_dir.name, logs[0], rows)

        official_name = found["cached"][0]
        no_rows = found["no_cache"][2]
        cache_rows = found["cached"][2]
        no_params = _trainable_params(found["no_cache"][1])
        cached_params = _trainable_params(found["cached"][1])
        if no_params != cached_params:
            raise RuntimeError(
                f"parameter mismatch for {official_name}: no_cache={no_params}, cached={cached_params}"
            )
        no_mae = np.asarray([no_rows[seed]["MAE"] for seed in expected_seeds], dtype=np.float64)
        cached_mae = np.asarray([cache_rows[seed]["MAE"] for seed in expected_seeds], dtype=np.float64)
        delta = cached_mae - no_mae
        test = stats.ttest_rel(cached_mae, no_mae)
        if len(delta) > 1:
            half = float(stats.t.ppf(0.975, len(delta) - 1) * stats.sem(delta))
        else:
            half = float("nan")
        comparisons.append({
            "model": official_name,
            "seeds": expected_seeds,
            "no_cache_mae": no_mae.tolist(),
            "cached_mae": cached_mae.tolist(),
            "no_cache_mean_mae": float(no_mae.mean()),
            "no_cache_std_mae_ddof1": float(no_mae.std(ddof=1)) if len(no_mae) > 1 else 0.0,
            "cached_mean_mae": float(cached_mae.mean()),
            "cached_std_mae_ddof1": float(cached_mae.std(ddof=1)) if len(cached_mae) > 1 else 0.0,
            "paired_delta_cached_minus_no_cache": float(delta.mean()),
            "delta_95ci": [float(delta.mean() - half), float(delta.mean() + half)],
            "paired_t": float(test.statistic),
            "paired_p_two_sided": float(test.pvalue),
            "cached_better_seed_count": int((delta < 0).sum()),
            "trainable_params_both_arms": no_params,
            "no_cache_log": str(found["no_cache"][1]),
            "cached_log": str(found["cached"][1]),
        })

    adjusted = _holm([row["paired_p_two_sided"] for row in comparisons])
    for row, p_holm in zip(comparisons, adjusted):
        row["holm_p_across_official_models"] = p_holm
    multiple_models = len(comparisons) > 1

    report = {
        "schema_version": 1,
        "protocol": {
            "dataset": "mosi",
            "implementation": "MMSA 2.2.1 released model and trainer, unmodified",
            "cache_adapter": "384-D rationale embedding appended to every aligned text timestep",
            "control": "same added dimensions filled with zeros",
            "paired_seeds": expected_seeds,
            "multiplicity": (
                "Holm correction across official models"
                if multiple_models
                else "not applicable (one official-model comparison)"
            ),
        },
        "comparisons": comparisons,
        "all_models_cache_better_on_mean": all(
            row["paired_delta_cached_minus_no_cache"] < 0 for row in comparisons
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = output.with_suffix(".md")
    lines = [
        "# Official MMSA cache crossing (MOSI)",
        "",
        "MMSA 2.2.1's released model and trainer are unmodified. The cached arm appends the",
        "384-D rationale embedding to each aligned text timestep; the control appends zeros",
        "of the same size. Both arms therefore have identical parameters and training budgets.",
        "",
        "| official model | params/arm | no cache MAE | cached MAE | paired delta | 95% CI | "
        + ("Holm p" if multiple_models else "two-sided paired p")
        + " | cached better seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        low, high = row["delta_95ci"]
        lines.append(
            f"| {row['model']} | {row['trainable_params_both_arms']} "
            f"| {row['no_cache_mean_mae']:.4f}+-{row['no_cache_std_mae_ddof1']:.4f} "
            f"| {row['cached_mean_mae']:.4f}+-{row['cached_std_mae_ddof1']:.4f} "
            f"| {row['paired_delta_cached_minus_no_cache']:+.4f} "
            f"| [{low:+.4f}, {high:+.4f}] | "
            f"{(row['holm_p_across_official_models'] if multiple_models else row['paired_p_two_sided']):.4g} "
            f"| {row['cached_better_seed_count']}/{len(expected_seeds)} |"
        )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {output} and {markdown}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="build dimension-matched cached/no-cache feature pkls")
    prep.add_argument("--source-pkl", required=True)
    prep.add_argument("--cache-dir", required=True)
    prep.add_argument("--output-dir", required=True)

    run = sub.add_parser("run", help="run one official MMSA model/arm in an isolated process")
    run.add_argument("--dataset", default="mosi")
    run.add_argument("--model", required=True)
    run.add_argument("--arm", required=True, choices=ARMS)
    run.add_argument("--manifest", required=True)
    run.add_argument("--run-root", required=True)
    run.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4])

    summary = sub.add_parser("summarize", help="aggregate paired official runs")
    summary.add_argument("--run-root", required=True)
    summary.add_argument("--output", required=True)
    summary.add_argument("--models", nargs="+", default=["MISA", "Self_MM", "MMIM", "LMF"])
    summary.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4])

    args = parser.parse_args()
    if args.command == "prepare":
        prepare_crossing(args.source_pkl, args.cache_dir, args.output_dir)
    elif args.command == "run":
        run_arm(
            dataset=args.dataset,
            model=args.model,
            arm=args.arm,
            manifest_path=args.manifest,
            run_root=args.run_root,
            seeds=args.seeds,
        )
    else:
        summarize_crossing(
            run_root=args.run_root,
            output_path=args.output,
            expected_models=args.models,
            expected_seeds=args.seeds,
        )


if __name__ == "__main__":
    main()
