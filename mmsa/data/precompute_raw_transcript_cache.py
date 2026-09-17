"""Build a no-LLM control cache by directly encoding each raw transcript.

This isolates the contribution of Qwen's sentiment rewrite from the contribution
of adding a second, frozen sentence-encoder view of the transcript. The output
layout intentionally matches the ordinary interpretation cache: train.npy,
valid.npy and test.npy can therefore be consumed by the unchanged training
pipeline through TEMOE_INTERP_DIR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mmsa.data.mmsa_pkl_dataset import read_raw_text


MODE = "raw-transcript-direct-embedding-no-llm"


def _text_hash(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        payload = text.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.save(stream, value)
    tmp.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def build_cache(
    pkl_path: str,
    output_dir: str,
    encoder_name: str,
    batch_size: int = 128,
    device: str | None = None,
    max_samples: int | None = None,
) -> dict:
    from sentence_transformers import SentenceTransformer

    raw = read_raw_text(pkl_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    encoder = SentenceTransformer(encoder_name, device=device)

    summary: dict = {
        "schema_version": 1,
        "source": str(pkl_path),
        "mode": MODE,
        "encoder": encoder_name,
        "encoder_device": str(encoder.device),
        "normalize_embeddings": False,
        "splits": {},
    }
    for split, all_texts in raw.items():
        texts = all_texts if max_samples is None else all_texts[:max_samples]
        features = np.asarray(
            encoder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ),
            dtype=np.float32,
        )
        if features.ndim != 2 or len(features) != len(texts):
            raise RuntimeError(
                f"Unexpected {split} feature shape {features.shape}; expected ({len(texts)}, D)"
            )
        if not np.isfinite(features).all():
            raise RuntimeError(f"Non-finite value in {split} transcript embeddings")

        _atomic_npy(output / f"{split}.npy", features)
        records = "\n".join(
            json.dumps({"i": i, "r": text}, ensure_ascii=False)
            for i, text in enumerate(texts)
        )
        _atomic_text(output / f"{split}_rationales.jsonl", records + ("\n" if records else ""))
        summary["splits"][split] = {
            "count": len(texts),
            "feature_dim": int(features.shape[1]),
            "dtype": str(features.dtype),
            "finite": True,
            "raw_text_sha256": _text_hash(texts),
        }
        print(
            f"[cache] split={split} count={len(texts)} dim={features.shape[1]} "
            f"mean_norm={np.linalg.norm(features, axis=1).mean():.6f}",
            flush=True,
        )

    _atomic_text(
        output / "interpretation_metadata.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--encoder-name",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Use the exact sentence encoder used for the Qwen rationale cache.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    report = build_cache(
        pkl_path=args.pkl,
        output_dir=args.output_dir,
        encoder_name=args.encoder_name,
        batch_size=args.batch_size,
        device=args.device,
        max_samples=args.max_samples,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
