"""Deployment latency: component timing and matched-boundary request timing.

Motivation (reviewer-facing): the paper's efficiency claim is only meaningful
against the alternative it replaces -- running the video MLLM online for every
inference call. This script measures that comparison directly and derives the
one-off cache-build cost, so the deployment story rests on measured numbers
rather than on the wall-clock of the small TeMoE forward pass alone.

The legacy component comparison is retained for backwards compatibility.
Its online arm includes video decoding, preprocessing, transfer and generation;
its cached arm is only a CUDA-synchronised forward with GPU-resident tensors.
It is therefore a cost-component ratio, not an end-to-end speed-up.

With --checkpoint, --interpretation-dir and --encoder-name the script also
measures both arms at one matched request boundary. The online request generates
and encodes a rationale before scoring; the cached request fetches the resident
embedding. Both arms then construct/transfer the same real MMSA tensors and run
the same trained TeMoE checkpoint. One-time model and dataset loading is outside
both timed regions, and every inclusion/exclusion is written to the JSON.

Usage (GPU, needs the local VL model and the raw videos):

    python -m mmsa.efficiency.end_to_end \
        --pkl "MSA Datasets/MOSI/Processed/aligned_50.pkl" \
        --raw-dir "MSA Datasets/MSA-Datasets/CMU-MOSI/Raw" \
        --vl-model models/Qwen2.5-VL-7B-Instruct \
        --samples 30 --output outputs/mmsa/mosi/efficiency/end_to_end.json

Use --skip-online to profile only the cached path (runs anywhere).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from mmsa.efficiency.profile import profile_model


def measure_cached_path(
    seq_len: int,
    interp_dim: int,
    d_model: int,
    warmup: int,
    iters: int,
) -> dict:
    """Latency of the deployed TeMoE forward pass (explanations already cached)."""
    per_sample = profile_model(
        interp_dim=interp_dim,
        seq_len=seq_len,
        batch_size=1,
        d_model=d_model,
        warmup=warmup,
        iters=iters,
    )
    per_batch = profile_model(
        interp_dim=interp_dim,
        seq_len=seq_len,
        batch_size=32,
        d_model=d_model,
        warmup=warmup,
        iters=max(10, iters // 5),
    )
    return {"per_sample": per_sample, "per_batch32": per_batch}


def measure_online_path(
    pkl: str,
    raw_dir: str,
    vl_model: str,
    samples: int,
    split: str,
    fps: float,
    max_frames: int,
    max_pixels: int,
    max_new_tokens: int,
    warmup: int,
    blacklist: str | None,
) -> dict:
    """Median generator-component latency for one video rationale."""
    from mmsa.data.mmsa_pkl_dataset import read_ids, read_raw_text
    from mmsa.data.precompute_interpretation_video import VIDEO_PROMPT, _build_vl_generator
    from mmsa.data.raw_video_index import build_video_index, resolve_video

    blocked: set[str] = set()
    if blacklist:
        payload = json.loads(Path(blacklist).read_text(encoding="utf-8"))
        blocked = set(payload.get("bad_paths", payload if isinstance(payload, list) else []))

    ids = read_ids(pkl)[split]
    texts = read_raw_text(pkl)[split]
    index = build_video_index(raw_dir)

    # load the model outside the timed region: the cache-once design pays this cost
    # once per dataset, never per inference call
    generate = _build_vl_generator(vl_model, max_new_tokens, fps, max_pixels, max_frames, VIDEO_PROMPT)

    candidates = []
    for sample_id, text in zip(ids, texts):
        path = resolve_video(sample_id, index)
        if path is not None and path not in blocked:
            candidates.append((path, text))
        if len(candidates) >= samples + warmup:
            break

    latencies: list[float] = []
    for i, (path, text) in enumerate(candidates):
        start = time.perf_counter()
        generate(path, text)
        elapsed = time.perf_counter() - start
        if i < warmup:
            print(f"[info] online warmup {i + 1}/{warmup}: {elapsed:.2f}s")
            continue
        latencies.append(elapsed)
        print(f"[info] online sample {len(latencies)}/{samples}: {elapsed:.3f}s")

    if not latencies:
        raise RuntimeError("no usable clips found for the online latency probe")

    return {
        "n_samples": len(latencies),
        "median_s": round(statistics.median(latencies), 3),
        "mean_s": round(statistics.mean(latencies), 3),
        "min_s": round(min(latencies), 3),
        "max_s": round(max(latencies), 3),
        "fps": fps,
        "max_frames": max_frames,
        "max_pixels": max_pixels,
        "max_new_tokens": max_new_tokens,
        "measurement_boundary": {
            "includes": [
                "video file read and decode",
                "chat-template construction",
                "vision/text processor preprocessing",
                "host-to-device transfer of processor outputs",
                "Qwen2.5-VL generation",
                "generated-token decoding and cleanup",
            ],
            "excludes": [
                "Qwen2.5-VL model loading",
                "rationale sentence-encoder inference",
                "downstream TeMoE feature preparation and scoring",
            ],
        },
    }


def _latency_summary(seconds: list[float]) -> dict[str, Any]:
    if not seconds:
        raise RuntimeError("latency summary received no observations")
    ordered = sorted(seconds)
    return {
        "n_samples": len(seconds),
        "median_ms": round(statistics.median(seconds) * 1000.0, 4),
        "mean_ms": round(statistics.mean(seconds) * 1000.0, 4),
        "min_ms": round(ordered[0] * 1000.0, 4),
        "max_ms": round(ordered[-1] * 1000.0, 4),
        "per_sample_ms": [round(value * 1000.0, 4) for value in seconds],
    }


def measure_matched_request_paths(
    *,
    config_path: str,
    pkl: str,
    raw_dir: str,
    interpretation_dir: str,
    checkpoint: str,
    vl_model: str,
    encoder_name: str,
    samples: int,
    split: str,
    fps: float,
    max_frames: int,
    max_pixels: int,
    max_new_tokens: int,
    warmup: int,
    blacklist: str | None,
) -> dict[str, Any]:
    """Measure online and cached requests at the same application boundary."""
    import numpy as np
    import torch

    from mmsa.analysis.common import _load_state
    from mmsa.data.mmsa_pkl_dataset import load_mmsa_splits, read_ids, read_raw_text
    from mmsa.data.precompute_interpretation import _build_encoder
    from mmsa.data.precompute_interpretation_video import VIDEO_PROMPT, _build_vl_generator
    from mmsa.data.raw_video_index import build_video_index, resolve_video
    from mmsa.models.builder import build_model
    from mmsa.training.train import _forward, load_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("matched request timing requires CUDA")

    config = load_config(config_path)
    datasets = load_mmsa_splits(pkl, interpretation_dir=interpretation_dir)
    if split not in datasets:
        raise KeyError(f"split '{split}' not found in {pkl}")
    dataset = datasets[split]
    dims = dataset.dims
    use_interpretation = bool(config["model"].get("use_interpretation", True)) and dims["interp_dim"] > 0
    if not use_interpretation:
        raise RuntimeError("the selected cache/config does not enable the interpretation branch")

    model = build_model("temoe", dims, config["model"], use_interpretation).to(device)
    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    _load_state(model, torch.load(ckpt, map_location="cpu"), ckpt)
    model.eval()

    generate = _build_vl_generator(vl_model, max_new_tokens, fps, max_pixels, max_frames, VIDEO_PROMPT)
    encode = _build_encoder(encoder_name)
    ids = read_ids(pkl)[split]
    texts = read_raw_text(pkl)[split]
    video_index = build_video_index(raw_dir)

    blocked: set[str] = set()
    if blacklist:
        payload = json.loads(Path(blacklist).read_text(encoding="utf-8"))
        blocked = set(payload.get("bad_paths", payload if isinstance(payload, list) else []))

    candidates: list[tuple[int, str, str, str]] = []
    for index, (sample_id, text) in enumerate(zip(ids, texts)):
        path = resolve_video(sample_id, video_index)
        if path is None or path in blocked:
            continue
        candidates.append((index, str(sample_id), path, text))
        if len(candidates) >= samples + warmup:
            break
    if len(candidates) < samples + warmup:
        raise RuntimeError(
            f"only {len(candidates)} usable videos found; need {samples + warmup} for measurement"
        )

    def sync() -> None:
        torch.cuda.synchronize(device)

    def make_batch(item: dict[str, torch.Tensor], interpretation: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = {key: value.unsqueeze(0) for key, value in item.items()}
        batch["interpretation"] = interpretation.reshape(1, -1).to(dtype=torch.float32)
        return batch

    cached_seconds: list[float] = []
    online_seconds: list[float] = []
    measured_ids: list[str] = []
    with torch.no_grad():
        for position, (index, sample_id, video_path, text) in enumerate(candidates):
            sync()
            start = time.perf_counter()
            cached_item = dataset[index]
            cached_batch = make_batch(cached_item, cached_item["interpretation"])
            _forward(model, cached_batch, device, use_interpretation)
            sync()
            cached_elapsed = time.perf_counter() - start

            sync()
            start = time.perf_counter()
            rationale = generate(video_path, text)
            online_vector = np.asarray(encode([rationale])[0], dtype=np.float32)
            online_item = dataset[index]
            online_batch = make_batch(online_item, torch.from_numpy(online_vector))
            _forward(model, online_batch, device, use_interpretation)
            sync()
            online_elapsed = time.perf_counter() - start

            if position < warmup:
                print(
                    f"[info] matched-boundary warmup {position + 1}/{warmup}: "
                    f"cached={cached_elapsed * 1000:.3f}ms online={online_elapsed:.3f}s"
                )
                continue
            cached_seconds.append(cached_elapsed)
            online_seconds.append(online_elapsed)
            measured_ids.append(sample_id)
            print(
                f"[info] matched-boundary sample {len(measured_ids)}/{samples}: "
                f"cached={cached_elapsed * 1000:.3f}ms online={online_elapsed:.3f}s"
            )

    cached = _latency_summary(cached_seconds)
    online = _latency_summary(online_seconds)
    return {
        "boundary": "resident MMSA features to one trained TeMoE prediction",
        "device": str(device),
        "split": split,
        "sample_ids": measured_ids,
        "cached_request": cached,
        "online_request": online,
        "speedup_online_over_cached": round(online["median_ms"] / cached["median_ms"], 1),
        "cached_includes": [
            "resident feature/cache lookup",
            "numpy-to-torch tensor construction",
            "host-to-device transfer",
            "trained TeMoE forward",
            "CUDA synchronization",
        ],
        "online_includes": [
            "raw video file read and decode",
            "chat-template and multimodal preprocessing",
            "host-to-device transfer",
            "Qwen2.5-VL rationale generation and decoding",
            "rationale sentence encoding",
            "the same MMSA tensor construction/transfer and trained TeMoE forward as the cached arm",
            "CUDA synchronization",
        ],
        "both_exclude": [
            "one-time model/checkpoint loading",
            "one-time MMSA pickle/cache loading",
            "network transport outside the host",
        ],
        "checkpoint": str(ckpt),
        "interpretation_dir": interpretation_dir,
        "encoder_name": encoder_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Online-MLLM vs cache-once end-to-end latency.")
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--vl-model", default=None)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--split", default="test")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--interp-dim", type=int, default=384)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--blacklist", default=None)
    parser.add_argument("--skip-online", action="store_true", help="profile only the cached path")
    parser.add_argument("--config", default=None, help="TeMoE yaml for matched request timing")
    parser.add_argument("--interpretation-dir", default=None, help="cache used by the trained TeMoE checkpoint")
    parser.add_argument("--checkpoint", default=None, help="trained TeMoE checkpoint for matched request timing")
    parser.add_argument("--encoder-name", default=None, help="sentence encoder used to build the rationale cache")
    parser.add_argument("--output", default="outputs/mmsa/efficiency/end_to_end.json")
    args = parser.parse_args()

    cached = measure_cached_path(args.seq_len, args.interp_dim, args.d_model, args.warmup, args.iters)
    per_sample_ms = cached["per_sample"]["latency_ms_per_batch"]

    report: dict = {
        "schema_version": 2,
        "cached_path": cached,
        "cached_path_per_sample_ms": round(per_sample_ms, 4),
        "component_ratio_warning": (
            "cached_path is a GPU-resident model forward, while online_path includes video preprocessing "
            "and transfer; speedup_online_over_cached is a cost-component ratio, not end-to-end speed-up"
        ),
        "settings": {
            "seq_len": args.seq_len,
            "interp_dim": args.interp_dim,
            "d_model": args.d_model,
            "batch_size_small": 1,
            "batch_size_large": 32,
        },
    }

    if not args.skip_online:
        if not (args.raw_dir and args.vl_model):
            raise SystemExit("--raw-dir and --vl-model are required unless --skip-online is given")
        online = measure_online_path(
            args.pkl, args.raw_dir, args.vl_model, args.samples, args.split,
            args.fps, args.max_frames, args.max_pixels, args.max_new_tokens,
            args.warmup, args.blacklist,
        )
        online_ms = online["median_s"] * 1000.0
        report["online_path"] = online
        report["speedup_online_over_cached"] = round(online_ms / per_sample_ms, 1)

        # one-off cache build cost for the whole dataset (all splits)
        from mmsa.data.mmsa_pkl_dataset import read_ids

        n_clips = sum(len(v) for v in read_ids(args.pkl).values())
        total_s = online["median_s"] * n_clips
        report["cache_build"] = {
            "n_clips": n_clips,
            "one_off_seconds": round(total_s, 1),
            "one_off_gpu_hours": round(total_s / 3600.0, 2),
            "note": "one-off cost paid once offline; zero MLLM cost thereafter",
        }

        matched_args = (args.config, args.interpretation_dir, args.checkpoint, args.encoder_name)
        if any(matched_args) and not all(matched_args):
            raise SystemExit(
                "matched request timing requires all of --config, --interpretation-dir, "
                "--checkpoint and --encoder-name"
            )
        if all(matched_args):
            matched = measure_matched_request_paths(
                config_path=args.config,
                pkl=args.pkl,
                raw_dir=args.raw_dir,
                interpretation_dir=args.interpretation_dir,
                checkpoint=args.checkpoint,
                vl_model=args.vl_model,
                encoder_name=args.encoder_name,
                samples=args.samples,
                split=args.split,
                fps=args.fps,
                max_frames=args.max_frames,
                max_pixels=args.max_pixels,
                max_new_tokens=args.max_new_tokens,
                warmup=args.warmup,
                blacklist=args.blacklist,
            )
            report["matched_request_paths"] = matched
            matched_online_s = matched["online_request"]["median_ms"] / 1000.0
            matched_cached_s = matched["cached_request"]["median_ms"] / 1000.0
            n_clips = report["cache_build"]["n_clips"]
            build_s = matched_online_s * n_clips
            report["matched_request_amortization"] = {
                "estimated_build_gpu_hours": round(build_s / 3600.0, 2),
                "break_even_queries": round(build_s / max(1e-12, matched_online_s - matched_cached_s)),
                "formula": "N * online_request / (online_request - cached_request)",
            }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
