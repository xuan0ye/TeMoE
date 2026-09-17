from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

from mmsa.data.mmsa_pkl_dataset import read_ids
from mmsa.data.raw_video_index import build_video_index, resolve_video

# Some raw clips are truncated/malformed and cause decord to hang indefinitely
# on open() rather than raising. A hard-killed subprocess is the only reliable
# way to bound that: signal-based timeouts inside the main process cannot
# interrupt a blocking call in a C extension that never releases the GIL.


def _probe(path: str, queue: "mp.Queue[tuple[bool, str]]", max_frames: int = 8) -> None:
    """Decode frame 0 *and* a handful of frames spread across the whole clip.

    A frame-0-only probe is not representative of the real failure mode: the
    actual precompute pipeline (qwen_vl_utils.process_vision_info) samples
    up to `max_frames` frames spread across the clip's full duration (~1fps
    sampling), which seeks decord to arbitrary, possibly-non-keyframe
    positions. Some MP4s have a corrupted/partial seek index that lets
    sequential decode of frame 0 succeed instantly while a seek to a later
    frame hangs indefinitely -- exactly the case a frame-0-only probe misses
    (observed on MOSEI: this probe reported `bad_count: 0` while the real
    run hung on the very same file). Seeking a spread of frames here makes
    this probe representative of what actually happens downstream.
    """
    try:
        import numpy as np
        import decord

        vr = decord.VideoReader(path)
        frame_count = len(vr)
        if frame_count <= 0:
            raise ValueError("zero-length video")
        sample_idx = np.linspace(0, frame_count - 1, num=min(max_frames, frame_count), dtype=int)
        sample_idx = sorted(set(int(i) for i in sample_idx))
        _ = vr.get_batch(sample_idx).asnumpy()  # force real seeks + decodes, not just header parsing
        queue.put((True, f"ok frames={frame_count}"))
    except Exception as error:  # noqa: BLE001
        queue.put((False, f"error: {error}"))


def validate_videos(paths: list[str], timeout_seconds: float, workers: int, max_frames: int = 8) -> dict:
    ctx = mp.get_context("spawn")
    pending = list(dict.fromkeys(paths))  # de-dup, keep order
    results: dict[str, str] = {}
    bad: list[str] = []
    running: list[tuple[object, "mp.Queue", str, float]] = []

    idx = 0
    while idx < len(pending) or running:
        while len(running) < workers and idx < len(pending):
            path = pending[idx]
            idx += 1
            queue: "mp.Queue" = ctx.Queue()
            proc = ctx.Process(target=_probe, args=(path, queue, max_frames), daemon=True)
            proc.start()
            running.append((proc, queue, path, time.monotonic()))

        still_running = []
        for proc, queue, path, started in running:
            if not proc.is_alive():
                ok, detail = queue.get() if not queue.empty() else (False, "no result (crashed)")
                results[path] = detail
                if not ok:
                    bad.append(path)
                proc.join(timeout=1)
                continue
            if time.monotonic() - started > timeout_seconds:
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=5)
                results[path] = f"timeout after {timeout_seconds}s (killed)"
                bad.append(path)
                continue
            still_running.append((proc, queue, path, started))
        running = still_running
        if running:
            time.sleep(0.05)

    return {
        "total": len(pending),
        "bad_count": len(bad),
        "bad_paths": bad,
        "details": {path: detail for path, detail in results.items() if path in bad},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe raw video files for hangs/corruption before a long VL precompute run."
    )
    parser.add_argument("--pkl", help="restrict validation to videos referenced by this pkl's ids")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=8,
        help="frames sampled per clip, spread across its full duration -- match the "
        "--max-frames you pass to precompute_interpretation_video so this probe actually "
        "exercises the same decord seek pattern the real run will hit",
    )
    parser.add_argument("--output", required=True, help="output json with a 'bad_paths' blacklist")
    args = parser.parse_args()

    index = build_video_index(args.raw_dir)
    if args.pkl:
        ids = read_ids(args.pkl)
        paths = [
            path
            for split_ids in ids.values()
            for path in (resolve_video(sample_id, index) for sample_id in split_ids)
            if path is not None
        ]
    else:
        paths = [path for clips in index.values() for path in clips.values()]

    print(
        f"[info] validating {len(paths)} video files (workers={args.workers}, "
        f"timeout={args.timeout_seconds}s, max_frames={args.max_frames})"
    )
    report = validate_videos(
        paths, timeout_seconds=args.timeout_seconds, workers=args.workers, max_frames=args.max_frames
    )
    print(f"[info] done: {report['bad_count']} / {report['total']} bad")
    for path, detail in report["details"].items():
        print(f"  BAD {path}: {detail}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
