"""Batch-1 latency probe for 1.5B text-only and 7B VL text-only teachers."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np

from mmsa.data.mmsa_pkl_dataset import read_raw_text
from mmsa.data.precompute_interpretation_video import NO_VIDEO, VIDEO_PROMPT, _build_vl_generator
from mmsa.data.precompute_text_teacher_cache import load_generator, prompt_sha256


def _summary(latencies: list[float]) -> dict:
    values = np.asarray(latencies, dtype=np.float64)
    return {
        "n": len(latencies),
        "median_s": float(statistics.median(latencies)),
        "mean_s": float(values.mean()),
        "min_s": float(values.min()),
        "p25_s": float(np.percentile(values, 25)),
        "p75_s": float(np.percentile(values, 75)),
        "max_s": float(values.max()),
        "per_sample_s": [float(value) for value in latencies],
    }


def _measure(name: str, texts: list[str], warmup: int, call) -> tuple[dict, list[str]]:
    import torch

    outputs: list[str] = []
    for text in texts[:warmup]:
        value = call(text)
        if not value:
            raise RuntimeError(f"{name} produced an empty warm-up output")
    latencies: list[float] = []
    for index, text in enumerate(texts[warmup:], 1):
        torch.cuda.synchronize()
        started = time.perf_counter()
        value = call(text)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if not value:
            raise RuntimeError(f"{name} produced an empty output at measured sample {index}")
        outputs.append(value)
        latencies.append(elapsed)
        eta = statistics.mean(latencies) * (len(texts) - warmup - index)
        print(
            f"[latency-progress] model={name} completed={index}/{len(texts) - warmup} "
            f"last={elapsed:.4f}s eta={eta:.1f}s",
            flush=True,
        )
    return _summary(latencies), outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--vl-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the controlled latency probe")
    test = read_raw_text(args.pkl)["test"]
    needed = args.samples + args.warmup
    if len(test) < needed:
        raise RuntimeError(f"Need {needed} test transcripts, found {len(test)}")
    texts = test[:needed]

    small = load_generator(args.text_model, args.max_new_tokens)
    small_stats, small_outputs = _measure(
        "qwen2.5-1.5b-text", texts, args.warmup, lambda text: small([text])[0][0]
    )
    del small
    gc.collect()
    torch.cuda.empty_cache()

    large = _build_vl_generator(
        args.vl_model,
        args.max_new_tokens,
        fps=1.0,
        max_pixels=100352,
        max_frames=8,
        prompt_template=VIDEO_PROMPT,
    )
    large_stats, large_outputs = _measure(
        "qwen2.5-vl-7b-text-only", texts, args.warmup, lambda text: large(NO_VIDEO, text)
    )
    ratio = large_stats["median_s"] / small_stats["median_s"]
    report = {
        "schema_version": 1,
        "question": "How much cheaper is a 1.5B text teacher than the 7B VL model in text-only mode?",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "batch_size": 1,
        "warmup_samples": args.warmup,
        "measured_samples": args.samples,
        "prompt_sha256": prompt_sha256(),
        "prompt_template": VIDEO_PROMPT,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "model_load_excluded": True,
        "text_1p5b": {"model": args.text_model, **small_stats, "sample_outputs": small_outputs[:3]},
        "vl_7b_textonly": {"model": args.vl_model, **large_stats, "sample_outputs": large_outputs[:3]},
        "median_speedup_1p5b_over_7b": ratio,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[latency] 1.5B={small_stats['median_s']:.4f}s "
        f"7B-text-only={large_stats['median_s']:.4f}s speedup={ratio:.2f}x",
        flush=True,
    )
    print(f"[done] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
