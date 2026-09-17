"""Validate the cached MLLM explanations as data, not just as an input dimension.

The crossing experiments show that *some* cached rationale helps every
architecture, and the mispaired-cache control shows the *pairing* matters.  What
neither shows is whether the rationale text itself is meaningful.  This script
measures five things about the cache directly, with no additional MLLM calls:

  A. text statistics        -- length and diversity (not one template repeated)
  B. visual grounding       -- does the text describe appearance at all?
  C. polarity agreement     -- does the rationale's affect agree with the label?
  D. transcript leakage     -- is the rationale merely a paraphrase of the text?
  E. linear probe           -- how much of the label is readable from the
                               rationale embedding alone (train -> test)?

C and D are the two honest failure modes of this design: a rationale with no
affective content adds nothing, and one that duplicates the transcript is not an
external prior.  E bounds the opposite worry, that the cache is a shortcut which
replaces the multimodal model.

Pass ``--interp-dir auto`` (the default) to auto-detect the cache directory for
each dataset: candidates under ``data/interpretation/*`` are ranked by split
length match, name affinity with the dataset, and the presence of the
``{split}_rationales.jsonl`` checkpoint this script needs.  The chosen directory
is printed and recorded in the report, so the pairing is always verifiable.

Usage:
  python -m mmsa.analysis.rationale_quality \
      --pkl "MSA Datasets/MOSI/Processed/aligned_50.pkl" \
      --interp-dir auto --splits valid test --probe-train-split train \
      --output outputs/mmsa/mosi/analysis/rationale_quality.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np

POS_ANCHOR = "The speaker is happy, cheerful and positive."
NEG_ANCHOR = "The speaker is sad, upset and negative."

# Words that indicate the rationale is grounded in what is visible in the clip
# (appearance / expression / scene) rather than in the transcript alone.
VISUAL_PATTERNS = re.compile(
    r"\b(face|facial|expression|smil|frown|eye|gaze|look|appear|hair|wear|shirt|"
    r"cloth|dress|background|scene|setting|posture|gesture|body|hand|head|nod|"
    r"camera|light|indoor|outdoor|beard|glasses)\w*",
    re.IGNORECASE,
)

_TOKEN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")

_SPLIT_ALIASES = {"train": ["train"], "valid": ["valid", "val", "dev"], "test": ["test"]}


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text or "")


def _read_jsonl(path: Path) -> dict[int, str]:
    records: dict[int, str] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[int(item["i"])] = str(item.get("r", ""))
    return records


def _load_rationales(path: Path, n_expected: int | None = None) -> list[str]:
    """Read a ``{split}_rationales.jsonl`` checkpoint into index-ordered texts."""
    if not path.exists():
        raise FileNotFoundError(f"rationale checkpoint not found: {path}")
    records = _read_jsonl(path)
    size = n_expected if n_expected is not None else (max(records) + 1 if records else 0)
    return [records.get(i, "") for i in range(size)]


def _load_pkl(pkl_path: str | Path) -> dict:
    with Path(pkl_path).open("rb") as file:
        return pickle.load(file, encoding="latin1")


def _split_payload(payload: dict, split: str) -> dict:
    for name in _SPLIT_ALIASES[split]:
        value = payload.get(name)
        if isinstance(value, dict):
            return value
    raise KeyError(f"split '{split}' not found")


def _load_labels(pkl_path: str | Path, split: str) -> np.ndarray:
    raw = _split_payload(_load_pkl(pkl_path), split)
    for key in ("regression_labels", "regression_labels_M", "labels", "classification_labels"):
        if raw.get(key) is not None:
            return np.asarray(raw[key], dtype=np.float32).reshape(-1)
    raise KeyError(f"no labels in split '{split}'")


def _load_raw_text(pkl_path: str | Path, split: str) -> list[str]:
    raw = _split_payload(_load_pkl(pkl_path), split)
    texts = raw.get("raw_text", raw.get("text_raw"))
    if texts is None:
        raise KeyError(f"no raw_text in split '{split}'")
    return [str(item) for item in np.asarray(texts).reshape(-1).tolist()]


def _candidate_size(path: Path) -> int | None:
    """Sample count of a cache artifact, or None if it cannot be read cheaply."""
    try:
        if path.suffix == ".npy":
            return int(np.load(path, mmap_mode="r").shape[0])
        if path.suffix == ".jsonl":
            return len(_read_jsonl(path))
    except Exception:
        return None
    return None


def _dataset_key(pkl_path: str | Path) -> str:
    name = str(pkl_path).lower()
    for key in ("mosei", "mosi", "sims"):
        if key in name:
            return key
    return ""


def _resolve_interp_dir(pkl_path: str | Path, split: str, root: str | Path = "data/interpretation") -> Path | None:
    """Pick the cached-explanation directory that belongs to this dataset/split.

    Both a single-rationale cache and a Plan-B structured cache can hold a
    ``{split}.npy`` of the right length, so candidates are scored rather than
    taken in directory order: the ``{split}_rationales.jsonl`` this script needs
    is decisive, dataset-name affinity breaks remaining ties, and Plan-B/smoke
    directories are demoted so that a structured run never masquerades as the
    single-rationale cache under test.
    """
    try:
        target = len(_load_labels(pkl_path, split))
    except (KeyError, FileNotFoundError) as exc:
        # A dataset/split without labels cannot be validated; let the caller skip it
        # rather than aborting the whole run.
        print(f"[skip] {split}: {exc}")
        return None
    key = _dataset_key(pkl_path)
    root = Path(root)
    if not root.is_dir():
        return None
    scored: list[tuple[int, Path]] = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        jsonl = candidate / f"{split}_rationales.jsonl"
        npy = candidate / f"{split}.npy"
        probe = jsonl if jsonl.exists() else (npy if npy.exists() else None)
        if probe is None:
            continue
        size = _candidate_size(probe)
        if size is None or size != target:
            continue
        score = 0
        if jsonl.exists():
            score += 10
        if key and key in candidate.name.lower():
            score += 5
        if "planb" in candidate.name.lower() or "smoke" in candidate.name.lower():
            score -= 5
        scored.append((score, candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1].name))
    return scored[0][1]


def _encode(encoder, texts: list[str], batch_size: int = 128) -> np.ndarray:
    return np.asarray(
        encoder.encode(texts, batch_size=batch_size, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=True),
        dtype=np.float32,
    )


def _distinct(texts: list[str], n: int) -> float:
    total, seen = 0, set()
    for text in texts:
        toks = _tokens(text)
        for i in range(len(toks) - n + 1):
            seen.add(tuple(toks[i : i + n]))
        total += max(len(toks) - n + 1, 0)
    return float(len(seen) / total) if total else 0.0


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(len(x), dtype=np.float64)
        ranks[order] = np.arange(len(x), dtype=np.float64)
        return ranks

    return _pearson(rank(a), rank(b))


def _ridge_probe(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
                 test_y: np.ndarray, alpha: float = 1.0) -> dict:
    """Closed-form ridge regression on raw embeddings (no tuning, no test peeking)."""
    x = np.concatenate([train_x, np.ones((len(train_x), 1), dtype=np.float32)], axis=1)
    xt = np.concatenate([test_x, np.ones((len(test_x), 1), dtype=np.float32)], axis=1)
    gram = x.T @ x + alpha * np.eye(x.shape[1], dtype=np.float32)
    weights = np.linalg.solve(gram, x.T @ train_y)
    pred = xt @ weights
    lo, hi = float(train_y.min()), float(train_y.max())
    pred_clipped = np.clip(pred, lo, hi)
    truth = test_y[: len(pred_clipped)]
    return {
        "n_train": int(len(train_y)),
        "n_test": int(len(truth)),
        "mae": float(np.mean(np.abs(pred_clipped - truth))),
        "mae_unclipped": float(np.mean(np.abs(pred - truth))),
        "corr": _pearson(pred, truth),
        "label_range_train": [lo, hi],
    }


def analyse_split(pkl_path, interp_dir: Path, split: str, encoder, anchors: dict, args) -> dict:
    labels = _load_labels(pkl_path, split)
    texts = _load_rationales(interp_dir / f"{split}_rationales.jsonl", n_expected=len(labels))
    # prefix slicing (not a spread-out subsample) so that --max-samples keeps the
    # rationale/label/index alignment the linear probe relies on.
    if args.max_samples and len(texts) > args.max_samples:
        texts = texts[: args.max_samples]
        labels = labels[: args.max_samples]

    lengths = np.array([len(_tokens(t)) for t in texts], dtype=np.float64)
    empty = int(np.sum(lengths == 0))
    visual = np.array([bool(VISUAL_PATTERNS.search(t)) for t in texts], dtype=bool)

    emb = _encode(encoder, texts)
    polarity = emb @ anchors["pos"] - emb @ anchors["neg"]  # cosine difference, in [-2, 2]

    mask = np.abs(labels) >= args.neutral_band
    if mask.sum() >= 10:
        sign_agreement = float(np.mean(np.sign(polarity[mask]) == np.sign(labels[mask])))
        sign_n = int(mask.sum())
    else:
        sign_agreement, sign_n = float("nan"), 0

    result = {
        "split": split,
        "cache_dir": str(interp_dir),
        "n": len(texts),
        "text": {
            "tokens_mean": float(lengths.mean()),
            "tokens_median": float(np.median(lengths)),
            "tokens_p10": float(np.percentile(lengths, 10)),
            "tokens_p90": float(np.percentile(lengths, 90)),
            "empty_count": empty,
            "unique_ratio": float(len(set(texts)) / len(texts)) if texts else 0.0,
            "distinct_1": _distinct(texts, 1),
            "distinct_2": _distinct(texts, 2),
        },
        "visual_grounding": {
            "rate": float(visual.mean()) if len(visual) else 0.0,
            "count": int(visual.sum()),
        },
        "polarity_agreement": {
            "pearson_vs_label": _pearson(polarity, labels),
            "spearman_vs_label": _spearman(polarity, labels),
            "sign_accuracy": sign_agreement,
            "sign_n": sign_n,
            "neutral_band": args.neutral_band,
        },
    }

    if args.with_transcript:
        try:
            raw = _load_raw_text(pkl_path, split)[: len(texts)]
            raw_emb = _encode(encoder, raw)
            own = np.sum(emb * raw_emb, axis=1)
            # shifted pairing gives the "unrelated text" baseline for the same metric
            other = np.sum(emb * np.roll(raw_emb, 1, axis=0), axis=1)
            result["transcript_leakage"] = {
                "cos_own_transcript_mean": float(own.mean()),
                "cos_other_transcript_mean": float(other.mean()),
                "gap": float(own.mean() - other.mean()),
                "pearson_with_label": _pearson(own, labels),
            }
        except KeyError as exc:
            print(f"[skip] transcript leakage for {split}: {exc}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cached MLLM explanation quality.")
    parser.add_argument("--pkl", required=True, help="MMSA pkl (for labels/raw_text)")
    parser.add_argument("--interp-dir", default="auto",
                        help="cache dir holding *_rationales.jsonl, or 'auto' (default)")
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--probe-train-split", default="train",
                        help="split used to fit the linear probe; '' disables it")
    parser.add_argument("--cache-root", default="data/interpretation",
                        help="where --interp-dir auto looks for candidates")
    parser.add_argument("--encoder-name", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--neutral-band", type=float, default=0.5,
                        help="|label| below this is excluded from sign accuracy")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--no-transcript", dest="with_transcript", action="store_false")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(args.encoder_name, device=args.device)
    anchors = {"pos": _encode(encoder, [POS_ANCHOR])[0], "neg": _encode(encoder, [NEG_ANCHOR])[0]}

    needed = list(dict.fromkeys(list(args.splits) + ([args.probe_train_split] if args.probe_train_split else [])))
    resolved: dict[str, Path] = {}
    report: dict = {"pkl": args.pkl, "encoder": args.encoder_name, "splits": {}}
    for split in needed:
        if args.interp_dir in ("auto", "", None):
            found = _resolve_interp_dir(args.pkl, split, args.cache_root)
            if found is None:
                print(f"[skip] {split}: no cache under {args.cache_root} matching this dataset")
                continue
            resolved[split] = found
            print(f"[auto] {split}: {found}")
        else:
            resolved[split] = Path(args.interp_dir)
    report["resolved_cache_dirs"] = {k: str(v) for k, v in resolved.items()}

    for split in args.splits:
        if split not in resolved:
            continue
        try:
            report["splits"][split] = analyse_split(args.pkl, resolved[split], split, encoder, anchors, args)
        except (FileNotFoundError, KeyError) as exc:
            print(f"[skip] {split}: {exc}")

    if args.probe_train_split and args.probe_train_split in resolved and report["splits"]:
        train_labels = _load_labels(args.pkl, args.probe_train_split)
        train_texts = _load_rationales(resolved[args.probe_train_split] / f"{args.probe_train_split}_rationales.jsonl",
                                       n_expected=len(train_labels))
        if args.max_samples and len(train_texts) > args.max_samples:
            train_texts = train_texts[: args.max_samples]
            train_labels = train_labels[: args.max_samples]
        train_emb = _encode(encoder, train_texts)
        report["linear_probe_on_rationale_only"] = {}
        for split in report["splits"]:
            if split == args.probe_train_split:
                continue
            n = report["splits"][split]["n"]
            test_labels = _load_labels(args.pkl, split)[:n]
            test_emb = _encode(encoder, _load_rationales(
                resolved[split] / f"{split}_rationales.jsonl", n_expected=n))[:n]
            report["linear_probe_on_rationale_only"][split] = _ridge_probe(
                train_emb, train_labels, test_emb, test_labels)

    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"\n[ok] wrote {out}")


if __name__ == "__main__":
    main()