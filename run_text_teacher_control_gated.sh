#!/usr/bin/env bash
# SCRIPT_REV=1
# MOSI control: 1.5B transcript-only rationale teacher vs 7B VL text-only.
set -Eeuo pipefail
cd "$(dirname "$0")"

STAGE=${STAGE:-smoke}
SMOKE_N=${SMOKE_N:-8}
SEEDS=${SEEDS:-"42 1 2 3 4"}
ARCHS=${ARCHS:-"temoe lmf late_fusion"}
PKL="MSA Datasets/MOSI/Processed/aligned_50.pkl"
TEXT_MODEL=${TEXT_MODEL:-models/Qwen2.5-1.5B-Instruct}
VL_MODEL=${VL_MODEL:-models/Qwen2.5-VL-7B-Instruct}
ENC=${ENC:-paraphrase-multilingual-MiniLM-L12-v2}
CACHE="data/interpretation/mosi_text_teacher_1p5b"
SMOKE_DIR="/tmp/temoe_text_teacher_smoke"
LOG_DIR="outputs/mmsa/logs"
LATENCY="outputs/mmsa/mosi/efficiency/text_teacher_latency.json"
SUMMARY="outputs/mmsa/official/text_teacher_control.json"
mkdir -p "$LOG_DIR" "outputs/mmsa/official" "outputs/mmsa/mosi/efficiency"

on_error () {
  rc=$?
  echo "[teacher] FAILED stage=$STAGE line=${BASH_LINENO[0]} exit=$rc $(date '+%F %T')" >&2
  exit "$rc"
}
trap on_error ERR

echo "[teacher] SCRIPT_REV=1 stage=$STAGE $(date '+%F %T')"
echo "[teacher] python=$(command -v python) text_model=$TEXT_MODEL vl_model=$VL_MODEL"
test -f "$PKL"
test -d "$TEXT_MODEL"
test -d "$VL_MODEL"
python - <<'PY'
import torch
import transformers
import sentence_transformers
print(
    f"[preflight] torch={torch.__version__} transformers={transformers.__version__} "
    f"cuda={torch.cuda.is_available()} "
    f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
PY

if [ "$STAGE" = "smoke" ]; then
  rm -rf "$SMOKE_DIR"
  python -m mmsa.data.precompute_text_teacher_cache \
    --pkl "$PKL" --output-dir "$SMOKE_DIR" --model-name "$TEXT_MODEL" \
    --encoder-name "$ENC" --max-new-tokens 64 --batch-size 4 --max-samples "$SMOKE_N"
  python - "$SMOKE_DIR" "$SMOKE_N" "$TEXT_MODEL" "$ENC" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
from mmsa.data.precompute_text_teacher_cache import MODE, prompt_sha256

root, n, model, encoder = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
meta = json.loads((root / "interpretation_metadata.json").read_text())
assert meta["mode"] == MODE, meta
assert meta["generator"] == model, meta
assert meta["encoder"] == encoder, meta
assert meta["prompt_sha256"] == prompt_sha256(), meta
assert meta["max_new_tokens"] == 64 and meta["do_sample"] is False, meta
for split in ("train", "valid", "test"):
    arr = np.load(root / f"{split}.npy")
    assert arr.shape == (n, 384), (split, arr.shape)
    assert np.isfinite(arr).all(), split
    rows = [json.loads(line) for line in (root / f"{split}_rationales.jsonl").read_text().splitlines()]
    assert len(rows) == n and [row["i"] for row in rows] == list(range(n)), split
    assert all(row["r"].strip() for row in rows), split
print(f"[smoke] VERIFIED splits=3 n={n} dim=384 prompt_match=true finite=true")
print("[smoke] sample=" + json.loads((root / "test_rationales.jsonl").read_text().splitlines()[0])["r"])
PY
  echo "[teacher] COMPLETE stage=smoke $(date '+%F %T')"
  exit 0
fi

if [ "$STAGE" != "full" ]; then
  echo "Unknown STAGE=$STAGE (expected smoke or full)" >&2
  exit 2
fi

echo "[cache] START 1.5B MOSI cache $(date '+%F %T')"
python -m mmsa.data.precompute_text_teacher_cache \
  --pkl "$PKL" --output-dir "$CACHE" --model-name "$TEXT_MODEL" \
  --encoder-name "$ENC" --max-new-tokens 64 --batch-size 16 \
  2>&1 | tee "$LOG_DIR/text_teacher_cache.log"

python - "$PKL" "$CACHE" "$TEXT_MODEL" "$ENC" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
from mmsa.data.mmsa_pkl_dataset import read_raw_text
from mmsa.data.precompute_text_teacher_cache import MODE, prompt_sha256

pkl, root, model, encoder = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
texts = read_raw_text(pkl)
meta = json.loads((root / "interpretation_metadata.json").read_text())
assert meta["mode"] == MODE and meta["generator"] == model, meta
assert meta["encoder"] == encoder and meta["prompt_sha256"] == prompt_sha256(), meta
assert meta["max_new_tokens"] == 64 and meta["do_sample"] is False, meta
for split, raw in texts.items():
    arr = np.load(root / f"{split}.npy", mmap_mode="r")
    assert arr.shape == (len(raw), 384), (split, arr.shape, len(raw))
    assert np.isfinite(arr).all(), split
    assert meta["splits"][split]["count"] == len(raw), split
print("[cache] VERIFIED " + " ".join(f"{split}={len(raw)}" for split, raw in texts.items()) + " dim=384")
PY

echo "[latency] START matched batch-1 probe (30 measured + 3 warm-up per model) $(date '+%F %T')"
python -m mmsa.efficiency.text_teacher_latency \
  --pkl "$PKL" --text-model "$TEXT_MODEL" --vl-model "$VL_MODEL" \
  --output "$LATENCY" --samples 30 --warmup 3 --max-new-tokens 64 \
  2>&1 | tee "$LOG_DIR/text_teacher_latency.log"

for ARCH in $ARCHS; do
  echo "[train] START arch=$ARCH $(date '+%F %T')"
  TEMOE_INTERP_DIR="$CACHE" python -m mmsa.training.resumable_multiseed \
    --config mmsa/configs/mosi.yaml --arch "$ARCH" --seeds $SEEDS \
    --condition text_teacher_1p5b \
    --output-dir "outputs/mmsa/mosi/control_text_teacher_1p5b/$ARCH" \
    2>&1 | tee "$LOG_DIR/text_teacher_train_$ARCH.log"
  echo "[train] END arch=$ARCH $(date '+%F %T')"
done

python -m mmsa.analysis.text_teacher_control \
  --dataset mosi --root outputs/mmsa --latency "$LATENCY" --output "$SUMMARY" \
  2>&1 | tee "$LOG_DIR/text_teacher_analysis.log"

python - "$SUMMARY" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1]))
assert set(data["rows"]) == {"temoe", "lmf", "late_fusion"}
for arch, row in data["rows"].items():
    assert row["teacher_1p5b"]["n"] == 5, (arch, row["teacher_1p5b"])
    assert len(data["paired_tests"][arch]) == 5, arch
assert data["latency"]["measured_samples"] == 30
print("[summary] VERIFIED architectures=3 seeds=5 paired_tests=15 latency_n=30x2")
PY
echo "[teacher] COMPLETE stage=full $(date '+%F %T')"
