"""Qwen2.5-VL zero-shot direct-score baseline on the MMSA test split.

The baseline uses the same Qwen2.5-VL-7B checkpoint and video loading path as
the explanation-cache generator, but asks for a numeric sentiment score and
does not train a fusion model. Static blacklist/missing-video cases, and the
rare runtime video failure, are retried with the same 7B model in text-only
mode instead of being silently defaulted to neutral.

Every sample is appended to a JSONL checkpoint. Re-running the same command
continues from that checkpoint after verifying a hash of the run configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mmsa.data.mmsa_pkl_dataset import SPLIT_ALIASES, _find_split, _labels, _load_pkl
from mmsa.data.precompute_interpretation_video import (
    NO_VIDEO,
    _HardTimeoutVLWorker,
    _load_blacklist,
)
from mmsa.data.raw_video_index import build_video_index, resolve_video
from mmsa.training.metrics import msa_regression_metrics

SCORE_PROMPT_TEMPLATE = (
    "You are a multimodal sentiment analysis system. Watch this short clip and predict its "
    "sentiment intensity as a single number between {low} (strongly negative) and {high} "
    "(strongly positive), where 0 is neutral. Transcript: {text}\n"
    'Respond with ONLY the number, e.g. "1.4" or "-0.8". Score:'
)

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_score(text: str, low: float, high: float) -> tuple[float, bool]:
    """Parse the first numeric token, clipping it to the dataset label range."""
    match = _NUMBER_RE.search(text or "")
    if match is None:
        return 0.0, False
    value = float(match.group())
    return float(np.clip(value, low, high)), True


def _read_test_records(pkl_path: str) -> tuple[list[str], list, np.ndarray]:
    """Load the large pickle once and extract aligned test text, ids and labels."""
    payload = _load_pkl(pkl_path)
    raw = _find_split(payload, SPLIT_ALIASES["test"])
    if raw is None:
        raise ValueError(f"No test split found in {pkl_path}")
    texts_raw = raw.get("raw_text")
    if texts_raw is None:
        texts_raw = raw.get("text_raw")
    if texts_raw is None:
        raise ValueError("Test split has no raw_text/text_raw")
    ids_raw = raw.get("id")
    if ids_raw is None:
        raise ValueError("Test split has no id field")
    texts = [str(item) for item in np.asarray(texts_raw).reshape(-1).tolist()]
    ids = list(np.asarray(ids_raw).tolist())
    labels = _labels(raw)
    if not (len(texts) == len(ids) == len(labels)):
        raise ValueError(
            f"Mismatched test records: texts={len(texts)}, ids={len(ids)}, labels={len(labels)}"
        )
    return texts, ids, labels


def _stable_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_checkpoint(path: Path, config_hash: str, n_samples: int) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("config_hash") != config_hash:
                raise RuntimeError(
                    f"Checkpoint config mismatch at {path}:{line_no}; use a new --output path "
                    "or move the stale checkpoint aside."
                )
            index = int(row["i"])
            if not 0 <= index < n_samples:
                raise RuntimeError(f"Checkpoint index {index} outside [0, {n_samples})")
            rows[index] = row
    return rows


def _format_seconds(value: float) -> str:
    seconds = max(0, int(round(value)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def run_zero_shot_baseline(
    pkl_path: str,
    raw_dir: str,
    vl_model: str,
    output_path: str,
    label_range: tuple[float, float],
    fps: float,
    max_pixels: int,
    max_frames: int,
    max_new_tokens: int,
    blacklist_path: str | None = None,
    sample_timeout_seconds: int = 90,
    max_samples: int | None = None,
    progress_every: int = 25,
) -> dict:
    texts, ids, labels = _read_test_records(pkl_path)
    if max_samples is not None:
        texts = texts[:max_samples]
        ids = ids[:max_samples]
        labels = labels[:max_samples]

    index = build_video_index(raw_dir)
    blacklist = _load_blacklist(blacklist_path)
    video_paths = [
        (None if (path := resolve_video(sample_id, index)) in blacklist else path)
        for sample_id in ids
    ]

    low, high = label_range
    prompt_template = SCORE_PROMPT_TEMPLATE.format(low=low, high=high, text="{text}")
    config = {
        "pkl": str(pkl_path),
        "raw_dir": str(raw_dir),
        "vl_model": str(vl_model),
        "label_range": [float(low), float(high)],
        "fps": float(fps),
        "max_pixels": int(max_pixels),
        "max_frames": int(max_frames),
        "max_new_tokens": int(max_new_tokens),
        "sample_timeout_seconds": int(sample_timeout_seconds),
        "blacklist": str(blacklist_path) if blacklist_path else None,
        "max_samples": max_samples,
        "prompt_template": prompt_template,
    }
    config_hash = _stable_hash(config)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.parent / f"{output.stem}_predictions.jsonl"
    done = _load_checkpoint(checkpoint, config_hash, len(texts))
    pending = [i for i in range(len(texts)) if i not in done]
    print(
        f"[resume] output={output} config_sha256={config_hash} "
        f"done={len(done)} pending={len(pending)} total={len(texts)}",
        flush=True,
    )

    predictions = np.zeros(len(texts), dtype=np.float32)
    parsed_ok = np.zeros(len(texts), dtype=bool)
    replies: list[str] = [""] * len(texts)
    rows: dict[int, dict] = dict(done)

    for i, row in done.items():
        predictions[i] = float(row["score"])
        parsed_ok[i] = bool(row["parsed_ok"])
        replies[i] = str(row.get("reply", ""))

    worker = None
    started = time.monotonic()
    completed_this_session = 0
    try:
        if pending:
            worker = _HardTimeoutVLWorker(
                vl_model,
                max_new_tokens,
                fps,
                max_pixels,
                max_frames,
                prompt_template,
            )
        with checkpoint.open("a", encoding="utf-8", newline="\n") as stream:
            iterator = tqdm(
                pending,
                desc="qwen2.5-vl zero-shot",
                initial=len(done),
                total=len(texts),
            )
            for i in iterator:
                assert worker is not None
                video_path = video_paths[i]
                source = "video"
                video_status = "not_attempted"
                text_fallback_status = "not_needed"

                if video_path is None:
                    source = "text_fallback"
                    video_status = "missing_or_blacklisted"
                    status, payload = worker.generate(NO_VIDEO, texts[i], sample_timeout_seconds)
                    text_fallback_status = status
                else:
                    status, payload = worker.generate(video_path, texts[i], sample_timeout_seconds)
                    video_status = status
                    if status != "ok":
                        source = "text_fallback"
                        print(
                            f"[warn] test[{i}] video {status}; retrying the same 7B model text-only",
                            flush=True,
                        )
                        status, payload = worker.generate(NO_VIDEO, texts[i], sample_timeout_seconds)
                        text_fallback_status = status

                reply = payload if status == "ok" and payload is not None else ""
                score, ok = _parse_score(reply, low, high)
                row = {
                    "config_hash": config_hash,
                    "i": i,
                    "sample_id": str(ids[i]),
                    "source": source,
                    "video_status": video_status,
                    "text_fallback_status": text_fallback_status,
                    "reply": reply,
                    "score": score,
                    "parsed_ok": ok,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                rows[i] = row
                predictions[i] = score
                parsed_ok[i] = ok
                replies[i] = reply

                completed_this_session += 1
                total_done = len(done) + completed_this_session
                if total_done % max(1, progress_every) == 0 or total_done == len(texts):
                    elapsed = time.monotonic() - started
                    rate = completed_this_session / max(elapsed, 1e-6)
                    eta = (len(texts) - total_done) / max(rate, 1e-9)
                    print(
                        f"[progress] {total_done}/{len(texts)} "
                        f"session_rate={rate:.3f} clip/s eta={_format_seconds(eta)} "
                        f"parse_success={int(parsed_ok.sum())}/{total_done}",
                        flush=True,
                    )
    finally:
        if worker is not None:
            worker.close()

    ordered = [rows[i] for i in range(len(texts))]
    metrics = msa_regression_metrics(predictions, labels, label_range=label_range)
    input_counts = {
        "video_success": sum(row["source"] == "video" and row["video_status"] == "ok" for row in ordered),
        "planned_text_fallback": sum(row["video_status"] == "missing_or_blacklisted" for row in ordered),
        "runtime_text_fallback": sum(
            row["source"] == "text_fallback" and row["video_status"] in {"error", "timeout"}
            for row in ordered
        ),
        "text_fallback_failures": sum(
            row["source"] == "text_fallback" and row["text_fallback_status"] != "ok"
            for row in ordered
        ),
        "parse_failures_defaulted_to_neutral": int((~parsed_ok).sum()),
    }
    report = {
        "model": vl_model,
        "mode": "zero-shot-direct-score",
        "config_sha256": config_hash,
        "config": config,
        "n_samples": len(texts),
        "video_available": sum(path is not None for path in video_paths),
        "video_available_rate": round(sum(path is not None for path in video_paths) / max(1, len(texts)), 4),
        "parse_success_rate": round(float(parsed_ok.mean()), 4),
        "input_counts": input_counts,
        "prediction_summary": {
            "mean": float(predictions.mean()),
            "std": float(predictions.std()),
            "min": float(predictions.min()),
            "max": float(predictions.max()),
            "unique_rounded_4dp": int(len(np.unique(np.round(predictions, 4)))),
        },
        "test": metrics,
        "checkpoint": str(checkpoint),
    }

    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output.parent / f"{output.stem}_replies.txt").write_text(
        "\n".join(replies) + "\n", encoding="utf-8", newline="\n"
    )
    print("[done] " + json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL zero-shot direct-score baseline (test only).")
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--vl-model", required=True)
    parser.add_argument("--label-range", type=float, nargs=2, default=[-3.0, 3.0], metavar=("LOW", "HIGH"))
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--blacklist", default=None)
    parser.add_argument("--sample-timeout-seconds", type=int, default=90)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_zero_shot_baseline(
        pkl_path=args.pkl,
        raw_dir=args.raw_dir,
        vl_model=args.vl_model,
        output_path=args.output,
        label_range=tuple(args.label_range),
        fps=args.fps,
        max_pixels=args.max_pixels,
        max_frames=args.max_frames,
        max_new_tokens=args.max_new_tokens,
        blacklist_path=args.blacklist,
        sample_timeout_seconds=args.sample_timeout_seconds,
        max_samples=args.max_samples,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()