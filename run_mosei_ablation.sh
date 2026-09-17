#!/usr/bin/env bash
# ============================================================================
# MOSEI component ablation, trained WITH the cached explanation.
#
# Why this file exists: `run_mosei_suite.sh MODE=ablation` does not export
# TEMOE_INTERP_DIR, and mmsa/training/train.py derives
#     use_interpretation = model flag AND interp_dim > 0
# so without the cache every variant trains explanation-free and `full` comes
# out identical to `no_interpretation`.  That ablation would be worthless.  Run
# this instead: it exports the cache, so the rows are directly comparable to the
# MOSI ablation table in the paper.
#
# Default variant list = exactly the seven rows of that table.  ABL_ONLY can be
# widened (e.g. add no_audio vision_only high_aux) at roughly 1 GPU-hour per
# extra variant per seed set.
#
# Usage:  nohup bash run_mosei_ablation.sh > mosei_ablation.log 2>&1 &
#         ABL_ONLY="full no_interpretation" SEEDS="42" bash run_mosei_ablation.sh
# ============================================================================
set -u
cd "$(dirname "$0")"
CFG=${CFG:-mmsa/configs/mosei.yaml}
OUT=${OUT:-outputs/mmsa/mosei}
CACHE=${CACHE:-data/interpretation/mosei_video}
SEEDS=${SEEDS:-42 1 2 3 4}
ABL_ONLY=${ABL_ONLY:-"full no_mamba ta_gated_conv_text no_interpretation dense_moe no_vision no_text"}

if [ "${SKIP_CACHE_CHECK:-0}" != "1" ]; then
  for split in train valid test; do
    if [ ! -f "$CACHE/$split.npy" ]; then
      echo "ERROR: $CACHE/$split.npy is missing; the ablation needs the cache."
      exit 1
    fi
  done
  echo "cache OK: $CACHE"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
echo "===== MOSEI ablation WITH cached explanation ====="
echo "variants: $ABL_ONLY"
echo "seeds   : $SEEDS"
echo "start   : $(date '+%F %T')"

TEMOE_INTERP_DIR="$CACHE" python -m mmsa.training.ablation \
  --config "$CFG" --output-dir "$OUT/ablation" --seeds $SEEDS --only $ABL_ONLY \
  > "$OUT/ablation_interp.log" 2>&1
rc=$?
echo "exit=$rc  $(date '+%F %T')   (log: $OUT/ablation_interp.log)"
tail -12 "$OUT/ablation_interp.log" 2>/dev/null

# Every variant must actually have been trained with the explanation; verify it
# from the run reports rather than assuming the env var took effect.
echo "----- condition check (use_interpretation per variant) -----"
python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/mmsa/mosei/ablation")
if not root.is_dir():
    raise SystemExit("no ablation directory")
for variant in sorted(p for p in root.iterdir() if p.is_dir()):
    reports = sorted(variant.glob("seed_*/*_report.json"))
    if not reports:
        continue
    flags = {json.loads(r.read_text(encoding="utf-8")).get("use_interpretation") for r in reports}
    print(f"{variant.name:24s} seeds={len(reports)} use_interpretation={flags}")
PY
echo "===== ablation finished $(date '+%F %T') ====="
