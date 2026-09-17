from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

_SEPARATORS = ("$_$", "@", "$", "/", "\\")


def build_video_index(raw_dir: str | Path) -> dict[str, dict[str, str]]:
    """Index a ``Raw/<video_id>/<clip>.mp4`` tree as video_id -> {clip_stem: path}."""
    root = Path(raw_dir)
    index: dict[str, dict[str, str]] = {}
    if not root.exists():
        return index
    for video_dir in root.iterdir():
        if not video_dir.is_dir():
            continue
        clips: dict[str, str] = {}
        for mp4 in video_dir.glob("*.mp4"):
            clips[mp4.stem] = str(mp4)
        if clips:
            index[video_dir.name] = clips
    return index


def resolve_video(sample_id, index: dict[str, dict[str, str]]) -> str | None:
    """Best-effort map a pkl sample id to an mp4 path.

    Handles the common MMSA id encodings:
    ``[video_id, clip_id]``, ``video_id$_$clip``, ``video_id_clip``,
    ``video_id/clip``, with clip matched exactly, as int, or zero-padded.
    """
    video_id, clip_id = _split_id(sample_id)
    if video_id is None:
        return None

    clips = index.get(video_id)
    if clips is None:
        # some datasets prefix/suffix the folder differently; try a relaxed match
        for key in index:
            if key == video_id or key.endswith(video_id) or video_id.endswith(key):
                clips = index[key]
                break
    if not clips:
        return None
    return _match_clip(clip_id, clips)


def coverage(sample_ids: Iterable, index: dict[str, dict[str, str]]) -> dict:
    total = 0
    matched = 0
    misses: list[str] = []
    for sample_id in sample_ids:
        total += 1
        if resolve_video(sample_id, index) is not None:
            matched += 1
        elif len(misses) < 20:
            misses.append(str(sample_id))
    rate = matched / total if total else 0.0
    return {"total": total, "matched": matched, "rate": round(rate, 4), "miss_examples": misses}


def _split_id(sample_id) -> tuple[str | None, str | None]:
    if isinstance(sample_id, (list, tuple)) and len(sample_id) >= 2:
        return str(sample_id[0]), str(sample_id[1])

    text = str(sample_id).strip()
    for sep in _SEPARATORS:
        if sep in text:
            head, _, tail = text.rpartition(sep)
            if head:
                return head, tail

    bracket = re.match(r"^(.*)\[(\d+)\]$", text)
    if bracket:
        return bracket.group(1), bracket.group(2)

    if "_" in text:
        head, _, tail = text.rpartition("_")
        if head and tail.isdigit():
            return head, tail

    return text, None


def _match_clip(clip_id: str | None, clips: dict[str, str]) -> str | None:
    if clip_id is None:
        # single-clip video: return the only file if unambiguous
        return next(iter(clips.values())) if len(clips) == 1 else None
    if clip_id in clips:
        return clips[clip_id]
    if clip_id.isdigit():
        as_int = str(int(clip_id))
        if as_int in clips:
            return clips[as_int]
        for width in (2, 3, 4, 5):
            padded = clip_id.zfill(width)
            if padded in clips:
                return clips[padded]
        for stem, path in clips.items():
            if stem.isdigit() and int(stem) == int(clip_id):
                return path
    return None
