#!/usr/bin/env bash
# SCRIPT_REV=1
# Gated raw-transcript embedding control for the MOSI R3 attribution study.
set -Eeuo pipefail
cd "$(dirname "$0")"

STAGE=${STAGE:-smoke}
SMOKE_N=${SMOKE_N:-8}
SEEDS=${SEEDS:-"42 1 2 3 4"}
ARCHS=${ARCHS:-"temoe lmf late_fusion"}
PKL="MSA Datasets/MOSI/Processed/aligned_50.pkl"
ENC=${ENC:-paraphrase-multilingual-MiniLM-L12-v2}
CACHE="data/interpretation/mosi_rawtext"
SMOKE_DIR="/tmp/temoe_rawtext_smoke"
LOG_DIR="outputs/mmsa/logs"
mkdir -p "$LOG_DIR" "outputs/mmsa/official"

on_error () {
  rc=$?
  echo "[rawtext] FAILED stage=$STAGE line=${BASH_LINENO[0]} exit=$rc $(date '+%F %T')" >&2
  exit "$rc"
}
trap on_error ERR

echo "[rawtext] SCRIPT_REV=1 stage=$STAGE $(date '+%F %T')"
echo "[rawtext] python=$(command -v python) encoder=$ENC seeds=$SEEDS archs=$ARCHS"
test -f "$PKL"
python - <<'PY'
import torch
import numpy
import sentence_transformers
print(
    f"[preflight] torch={torch.__version__} cuda={torch.cuda.is_available()} "
    f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
    f"numpy={numpy.__version__}"
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the controlled training run")
PY

if [ "$STAGE" = "smoke" ]; then
  echo "[smoke] building $SMOKE_N direct transcript embeddings per split"
  python -m mmsa.data.precompute_raw_transcript_cache     --pkl "$PKL" --output-dir "$SMOKE_DIR" --encoder-name "$ENC"     --device cuda --batch-size 32 --max-samples "$SMOKE_N"
  python - "$SMOKE_DIR" "$SMOKE_N" "$ENC" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
n = int(sys.argv[2])
encoder_name = sys.argv[3]
meta = json.loads((root / "interpretation_metadata.json").read_text())
assert meta["mode"] == "raw-transcript-direct-embedding-no-llm", meta
assert meta["encoder"] == encoder_name, meta
assert meta["normalize_embeddings"] is False, meta
for split in ("train", "valid", "test"):
    arr = np.load(root / f"{split}.npy")
    assert arr.shape == (n, 384), (split, arr.shape)
    assert np.isfinite(arr).all(), split
    rows = [json.loads(line) for line in (root / f"{split}_rationales.jsonl").read_text().splitlines()]
    assert len(rows) == n and [r["i"] for r in rows] == list(range(n)), split
print(f"[smoke] VERIFIED splits=3 n={n} dim=384 finite=true")
PY
  echo "[rawtext] COMPLETE stage=smoke $(date '+%F %T')"
  exit 0
fi

if [ "$STAGE" != "full" ]; then
  echo "Unknown STAGE=$STAGE (expected smoke or full)" >&2
  exit 2
fi

if [ -f "$CACHE/interpretation_metadata.json" ] &&
   [ -f "$CACHE/train.npy" ] &&
   [ -f "$CACHE/valid.npy" ] &&
   [ -f "$CACHE/test.npy" ]; then
  echo "[cache] existing complete artifact set found; validating and reusing it"
else
  echo "[cache] building full MOSI direct transcript cache $(date '+%F %T')"
  python -m mmsa.data.precompute_raw_transcript_cache     --pkl "$PKL" --output-dir "$CACHE" --encoder-name "$ENC"     --device cuda --batch-size 128 2>&1 | tee "$LOG_DIR/rawtext_cache.log"
fi

python - "$PKL" "$CACHE" "$ENC" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
from mmsa.data.mmsa_pkl_dataset import read_raw_text

pkl, root = sys.argv[1], Path(sys.argv[2])
encoder_name = sys.argv[3]
texts = read_raw_text(pkl)
meta = json.loads((root / "interpretation_metadata.json").read_text())
assert meta["mode"] == "raw-transcript-direct-embedding-no-llm", meta
assert meta["encoder"] == encoder_name, meta
assert meta["normalize_embeddings"] is False, meta
for split, raw in texts.items():
    arr = np.load(root / f"{split}.npy", mmap_mode="r")
    assert arr.shape == (len(raw), 384), (split, arr.shape, len(raw))
    assert np.isfinite(arr).all(), split
    assert meta["splits"][split]["count"] == len(raw), split
print("[cache] VERIFIED " + " ".join(f"{s}={len(v)}" for s, v in texts.items()) + " dim=384")
PY

for ARCH in $ARCHS; do
  echo "[train] START arch=$ARCH $(date '+%F %T')"
  TEMOE_INTERP_DIR="$CACHE" python -m mmsa.training.resumable_multiseed     --config mmsa/configs/mosi.yaml --arch "$ARCH" --seeds $SEEDS     --output-dir "outputs/mmsa/mosi/control_rawtext/$ARCH"     2>&1 | tee -a "$LOG_DIR/rawtext_train_$ARCH.log"
  echo "[train] END arch=$ARCH $(date '+%F %T')"
done

python -m mmsa.analysis.raw_transcript_control   --dataset mosi --root outputs/mmsa   --output outputs/mmsa/official/raw_transcript_control.json   2>&1 | tee "$LOG_DIR/rawtext_analysis.log"

python - <<'PY'
import json
from pathlib import Path

summary = Path("outputs/mmsa/official/raw_transcript_control.json")
data = json.loads(summary.read_text())
assert set(data["rows"]) == {"temoe", "lmf", "late_fusion"}
for arch, row in data["rows"].items():
    assert row["rawtext"]["n"] == 5, (arch, row["rawtext"])
    assert len(data["paired_tests"][arch]) == 4, arch
print("[summary] VERIFIED architectures=3 seeds=5 paired_tests=12")
PY
echo "[rawtext] COMPLETE stage=full $(date '+%F %T')"
