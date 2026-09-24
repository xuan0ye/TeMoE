#!/usr/bin/env bash
# MOSEI transcript-only Qwen2.5-VL-7B control. Run from the TeMoE release root.
set -euo pipefail
cd "$(dirname "$0")"

MODE=$(printenv MODE 2>/dev/null || printf all)        # cache | train | report | all
ARCHS=$(printenv ARCHS 2>/dev/null || printf temoe)   # later: "temoe lmf late_fusion"
PKL=$(printenv PKL 2>/dev/null || printf 'MSA Datasets/MOSEI/Processed/aligned_50.pkl')
RAW_DIR=$(printenv RAW_DIR 2>/dev/null || printf 'MSA Datasets/MSA-Datasets/CMU-MOSEI/Raw')
VL_MODEL=$(printenv VL_MODEL 2>/dev/null || printf 'models/Qwen2.5-VL-7B-Instruct')
ENCODER=$(printenv ENCODER 2>/dev/null || printf 'paraphrase-multilingual-MiniLM-L12-v2')
CACHE=$(printenv CACHE 2>/dev/null || printf 'data/interpretation/mosei_textonly')
MATCHED_CACHE=$(printenv MATCHED_CACHE 2>/dev/null || printf 'data/interpretation/mosei_video')
OUT=$(printenv OUT 2>/dev/null || printf 'outputs/mmsa/mosei')
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

need_file() { if [ ! -f "$1" ]; then echo "ERROR missing file: $1" >&2; exit 1; fi; }
need_cache() { for split in train valid test; do need_file "$1/$split.npy"; done; }
need_file "$PKL"
mkdir -p "$OUT/logs"

case "$MODE" in
  cache|all)
    need_file "$VL_MODEL/config.json"
    if [ ! -d "$RAW_DIR" ]; then echo "ERROR missing raw video directory: $RAW_DIR" >&2; exit 1; fi
    echo "[source-control] building Qwen2.5-VL-7B transcript-only cache; JSONL checkpoints resume"
    python -m mmsa.data.precompute_interpretation_video \
      --pkl "$PKL" --raw-dir "$RAW_DIR" --output-dir "$CACHE" \
      --vl-model "$VL_MODEL" --encoder-name "$ENCODER" \
      --ignore-video --sample-timeout-seconds 90 \
      > "$OUT/logs/source_textonly_cache.log" 2>&1
    need_cache "$CACHE"
    need_file "$CACHE/interpretation_metadata.json"
    tail -8 "$OUT/logs/source_textonly_cache.log"
    ;;
esac

case "$MODE" in
  train|all)
    need_cache "$CACHE"
    need_cache "$MATCHED_CACHE"
    python - "$CACHE" "$MATCHED_CACHE" "$ENCODER" <<'PY'
import json
from pathlib import Path
import sys
import numpy as np
cache, matched, encoder = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
meta = json.loads((cache / "interpretation_metadata.json").read_text(encoding="utf-8"))
if meta.get("encoder") != encoder:
    raise SystemExit(f"ERROR encoder mismatch: {meta.get('encoder')} != {encoder}")
for split in ("train", "valid", "test"):
    a = np.load(cache / f"{split}.npy", mmap_mode="r")
    b = np.load(matched / f"{split}.npy", mmap_mode="r")
    if a.shape != b.shape or not np.isfinite(a).all():
        raise SystemExit(f"ERROR bad {split} control cache: {a.shape} vs matched {b.shape}")
    print(f"[source-control] {split}: {a.shape} finite, matched shape")
PY
    for arch in $ARCHS; do
      dir="$OUT/control_textonly/$arch"
      json="$dir/multiseed_$arch.json"
      if [ -s "$json" ]; then echo "[source-control] skip complete $arch: $json"; continue; fi
      mkdir -p "$dir"
      echo "[source-control] training $arch, seeds 42 1 2 3 4"
      TEMOE_INTERP_DIR="$CACHE" python -m mmsa.training.multiseed \
        --config mmsa/configs/mosei.yaml --arch "$arch" --seeds 42 1 2 3 4 \
        --output-dir "$dir" > "$OUT/logs/source_textonly_$arch.log" 2>&1
      need_file "$json"
      tail -3 "$OUT/logs/source_textonly_$arch.log"
    done
    ;;
esac

case "$MODE" in
  report|all)
    python -m mmsa.analysis.mosei_source_control --root "$OUT" --archs $ARCHS \
      --output "$OUT/analysis/mosei_source_control.json" \
      --markdown "$OUT/analysis/mosei_source_control.md"
    ;;
  cache|train) ;;
  *) echo "ERROR MODE must be cache, train, report, or all" >&2; exit 1 ;;
esac
