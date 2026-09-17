from __future__ import annotations
import os

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

SPLIT_ALIASES = {
    "train": ["train"],
    "valid": ["valid", "val", "dev"],
    "test": ["test"],
}

_MODALITIES = ("audio", "vision", "text")
_LABEL_KEYS = ("regression_labels", "regression_labels_M", "labels", "classification_labels")


class MMSASequenceDataset(Dataset):
    """Sequence-level MSA dataset built from already-sliced modality arrays.

    Modalities are kept as ``(N, T, D)`` sequences (no pooling) so the temporal
    alignment module can model them. ``disabled_modalities`` zeroes a modality
    for ablation studies.
    """

    def __init__(
        self,
        audio: np.ndarray,
        vision: np.ndarray,
        text: np.ndarray,
        labels: np.ndarray,
        interpretation: np.ndarray | None = None,
        interp_channels: dict[str, np.ndarray] | None = None,
        disabled_modalities: set[str] | None = None,
    ) -> None:
        self.audio = _clean(audio)
        self.vision = _clean(vision)
        self.text = _clean(text)
        self.labels = np.asarray(labels, dtype=np.float32).reshape(-1)
        self.interpretation = None if interpretation is None else _clean(interpretation)
        # Plan B: per-channel clue embeddings (face/content/bg + bg importance flag).
        # Keys: "face", "content", "bg", "bg_flag". Kept separate from the legacy
        # single-vector `interpretation` so old caches keep working unchanged.
        self.interp_channels: dict[str, np.ndarray] = {}
        if interp_channels:
            self.interp_channels = {key: _clean(value) for key, value in interp_channels.items()}
        self.disabled_modalities = disabled_modalities or set()
        self._check_lengths()

    def _check_lengths(self) -> None:
        lengths = {
            "audio": len(self.audio),
            "vision": len(self.vision),
            "text": len(self.text),
            "labels": len(self.labels),
        }
        if self.interpretation is not None:
            lengths["interpretation"] = len(self.interpretation)
        for key, value in self.interp_channels.items():
            lengths[f"interp_{key}"] = len(value)
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Mismatched sample counts: {lengths}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "audio": _pick(self.audio[index], "audio", self.disabled_modalities),
            "vision": _pick(self.vision[index], "vision", self.disabled_modalities),
            "text": _pick(self.text[index], "text", self.disabled_modalities),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
        }
        if self.interpretation is not None:
            item["interpretation"] = _pick(self.interpretation[index], "interpretation", self.disabled_modalities)
        for key, value in self.interp_channels.items():
            item[f"interp_{key}"] = _pick(value[index], f"interp_{key}", self.disabled_modalities)
        return item

    @property
    def dims(self) -> dict[str, int]:
        out = {
            "audio_dim": self.audio.shape[-1],
            "vision_dim": self.vision.shape[-1],
            "text_dim": self.text.shape[-1],
        }
        out["interp_dim"] = 0 if self.interpretation is None else self.interpretation.shape[-1]
        for key, value in self.interp_channels.items():
            out[f"interp_{key}_dim"] = 0 if value.size == 0 else (value.shape[-1] if value.ndim > 1 else 1)
        return out


def load_mmsa_splits(
    pkl_path: str | Path,
    interpretation_dir: str | Path | None = None,
    disabled_modalities: set[str] | None = None,
) -> dict[str, MMSASequenceDataset]:
    """Load a whole MMSA pkl once and build train/valid/test datasets."""
    payload = _load_pkl(pkl_path)
    datasets: dict[str, MMSASequenceDataset] = {}
    for split, aliases in SPLIT_ALIASES.items():
        raw = _find_split(payload, aliases)
        if raw is None:
            continue
        interp, channels = _load_interpretation(interpretation_dir, split)
        datasets[split] = MMSASequenceDataset(
            audio=_ensure_sequence(raw["audio"]),
            vision=_ensure_sequence(raw["vision"]),
            text=_ensure_sequence(raw["text"]),
            labels=_labels(raw),
            interpretation=interp,
            interp_channels=channels,
            disabled_modalities=disabled_modalities,
        )
    if not datasets:
        raise ValueError(f"No train/valid/test splits found in {pkl_path}")
    return datasets


def read_raw_text(pkl_path: str | Path) -> dict[str, list[str]]:
    """Extract raw_text per split for offline MLLM interpretation generation."""
    payload = _load_pkl(pkl_path)
    result: dict[str, list[str]] = {}
    for split, aliases in SPLIT_ALIASES.items():
        raw = _find_split(payload, aliases)
        if raw is None:
            continue
        texts = raw.get("raw_text")
        if texts is None:
            texts = raw.get("text_raw")
        if texts is None:
            raise ValueError(f"'{split}' split has no raw_text; cannot build interpretation.")
        result[split] = [str(item) for item in np.asarray(texts).reshape(-1).tolist()]
    return result


def read_ids(pkl_path: str | Path) -> dict[str, list]:
    """Extract the per-sample id per split (for mapping to raw video files)."""
    payload = _load_pkl(pkl_path)
    result: dict[str, list] = {}
    for split, aliases in SPLIT_ALIASES.items():
        raw = _find_split(payload, aliases)
        if raw is None:
            continue
        ids = raw.get("id")
        if ids is None:
            raise ValueError(f"'{split}' split has no 'id' field; cannot map to videos.")
        result[split] = list(np.asarray(ids).tolist())
    return result


def _load_pkl(pkl_path: str | Path) -> dict[str, Any]:
    with Path(pkl_path).open("rb") as file:
        payload = pickle.load(file, encoding="latin1")
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected pkl root type: {type(payload).__name__}")
    return payload


def _find_split(payload: dict[str, Any], aliases: list[str]) -> dict[str, Any] | None:
    for name in aliases:
        value = payload.get(name)
        if isinstance(value, dict):
            return value
    return None


def _labels(split: dict[str, Any]) -> np.ndarray:
    for key in _LABEL_KEYS:
        if key in split and split[key] is not None:
            return np.asarray(split[key], dtype=np.float32).reshape(-1)
    raise ValueError("No regression/classification labels found in split")


def _ensure_sequence(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, None, :]
    if array.ndim != 3:
        raise ValueError(f"Expected 2D/3D modality array, got shape {array.shape}")
    return array


def _maybe_mismatch(arr: np.ndarray | None) -> np.ndarray | None:
    """Control experiment: hand every sample *another* sample's explanation.

    Enabled by the environment variable ``TEMOE_SHUFFLE_INTERP``. A deterministic
    cyclic shift (``np.roll`` by 1) is used rather than a random permutation so
    that (a) the control is exactly reproducible and (b) no sample can keep its
    own explanation -- the accuracy drop then bounds how much the *content* of
    the cache matters, as opposed to its mere presence (a zeros/absent branch is
    already covered by the no-interpretation ablation).
    """
    if arr is None or not os.environ.get("TEMOE_SHUFFLE_INTERP"):
        return arr
    return np.roll(arr, 1, axis=0)


def _load_interpretation(    interpretation_dir: str | Path | None, split: str
) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    """Load interpretation features for one split.

    Returns ``(legacy_single, channels)`` where ``channels`` is a dict with
    keys ``face``/``content``/``bg``/``bg_flag`` when the Plan-B structured
    cache exists, otherwise ``{}`` (legacy single-vector cache or none).

    ``_load_interpretation_legacy`` (the previous single-return version) is
    kept for backwards compatibility with callers that only need one vector.
    """
    if interpretation_dir is None:
        return None, {}
    root = Path(interpretation_dir)
    path = root / f"{split}.npy"
    channels: dict[str, np.ndarray] = {}
    # Plan B structured cache: per-channel npy files.
    for channel, suffix in (("face", "_face"), ("content", "_content"), ("bg", "_bg"), ("bg_flag", "_bg_flag")):
        channel_path = root / f"{split}{suffix}.npy"
        if channel_path.exists():
            channels[channel] = np.load(channel_path).astype(np.float32)
    if channels:
        # Structured cache: keep legacy `interpretation` too (concatenation of
        # face/content/bg) so old training code paths that read only the
        # single vector keep working.
        legacy = None
        if path.exists():
            legacy = np.load(path).astype(np.float32)
        if legacy is None and {"face", "content", "bg"} <= set(channels):
            legacy = np.concatenate([channels[k] for k in ("face", "content", "bg")], axis=-1)
        return _maybe_mismatch(legacy), channels
    if not path.exists():
        raise FileNotFoundError(
            f"Interpretation features not found: {path}. Run mmsa.data.precompute_interpretation first."
        )
    return _maybe_mismatch(np.load(path).astype(np.float32)), {}


def _clean(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _pick(array: np.ndarray, modality: str, disabled: set[str]) -> torch.Tensor:
    value = np.zeros_like(array) if modality in disabled else array
    return torch.tensor(value, dtype=torch.float32)
