from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from mmsa.training.train import load_config, train

_METRIC_KEYS = ["mae", "corr", "acc7", "acc5", "acc2_has0", "f1_has0", "acc2_non0", "f1_non0"]


def run_multiseed(config_path: str, arch: str, seeds: list[int], output_dir: str) -> dict:
    base = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict] = []
    for seed in seeds:
        config = copy.deepcopy(base)
        config["seed"] = seed
        config["model"]["arch"] = arch
        config["training"]["output_dir"] = str(output / f"seed_{seed}")
        print(f"\n===== arch={arch} seed={seed} =====")
        report = train(config)
        per_seed.append({"seed": seed, **report["test"]})

    aggregate = _aggregate(per_seed)
    result = {
        "arch": arch,
        "dataset": base["data"].get("name"),
        "seeds": seeds,
        "per_seed": per_seed,
        "mean": aggregate["mean"],
        "std": aggregate["std"],
    }
    with (output / f"multiseed_{arch}.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps({"arch": arch, "mean": _round(aggregate["mean"]), "std": _round(aggregate["std"])}, ensure_ascii=False))
    return result


def _aggregate(per_seed: list[dict]) -> dict:
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in _METRIC_KEYS:
        values = np.array([row[key] for row in per_seed if _is_number(row.get(key))], dtype=np.float64)
        if values.size == 0:
            mean[key] = float("nan")
            std[key] = float("nan")
        else:
            mean[key] = float(values.mean())
            std[key] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return {"mean": mean, "std": std}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not (isinstance(value, float) and np.isnan(value))


def _round(report: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 4) for key, value in report.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one architecture across multiple seeds and aggregate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--arch", default="temoe")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or f"outputs/mmsa/multiseed/{args.arch}"
    run_multiseed(args.config, args.arch, args.seeds, output_dir)


if __name__ == "__main__":
    main()
