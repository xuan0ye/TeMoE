from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmsa.data.mmsa_pkl_dataset import read_ids
from mmsa.data.raw_video_index import build_video_index, coverage


def check_coverage(pkl_path: str, raw_dir: str, output: str | None = None) -> dict:
    ids = read_ids(pkl_path)
    index = build_video_index(raw_dir)
    total_videos = sum(len(clips) for clips in index.values())

    report = {
        "pkl": pkl_path,
        "raw_dir": raw_dir,
        "indexed_video_folders": len(index),
        "indexed_clips": total_videos,
        "splits": {split: coverage(split_ids, index) for split, split_ids in ids.items()},
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Report id->raw-video match rate before running VL inference.")
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    check_coverage(args.pkl, args.raw_dir, args.output)


if __name__ == "__main__":
    main()
