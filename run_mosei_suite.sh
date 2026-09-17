#!/usr/bin/env bash
# ============================================================================
# MOSEI suite.  Two independent knobs:
#   MODE=crossing | ablation | efficiency
#   COND=nointerp | interp        (crossing only; which cache condition to run)
#
# Directory convention, identical to MOSI/CH-SIMS so the cross-dataset audit
# picks the runs up without special cases:
#   COND=interp    -> outputs/mmsa/mosei/multiseed/<arch>            (w/ cached explanation)
#   COND=nointerp  -> outputs/mmsa/mosei/multiseed_nointerp/<arch>   (w/o explanation)
#
# The explanation-free condition does NOT need the MLLM cache, so it can run
# while the cache is still being built; the interp condition is guarded and
# refuses to start against an incomplete cache.
#
# Usage:
#   bash run_mosei_suite.sh                          # both conditions, all archs
#   COND=nointerp bash run_mosei_suite.sh            # only the cache-free half
#   MODE=ablation ABL_SEEDS="42 1 2" bash run_mosei_suite.sh
#   ABL_ONLY=all MODE=ablation bash run_mosei_suite.sh    # every classic variant
#   SEEDS="42 1 2" ARCHS="temoe lmf" bash run_mosei_suite.sh
# ============================================================================
set -u
cd "$(dirname "$0")"
MODE=${MODE:-crossing}
COND=${COND:-both}
SEEDS=${SEEDS:-42 1 2 3 4}
ABL_SEEDS=${ABL_SEEDS:-$SEEDS}
CFG=${CFG:-mmsa/configs/mosei.yaml}
OUT=${OUT:-outputs/mmsa/mosei}
CACHE=${CACHE:-data/interpretation/mosei_video}
# priority order: our model first, then the two best MOSI baselines and the two
# simplest ones, so a truncated sweep still leaves a complete, usable table row
# set rather than a scatter of half-finished architectures.
ARCHS=${ARCHS:-"temoe almt kuda late_fusion lmf misa selfmm magbert cross_modal cormult"}
# the seven variants reported in the paper's MOSI ablation table; use
# ABL_ONLY=all to run every classic variant (12, roughly twice the wall clock)
ABL_ONLY=${ABL_ONLY:-"full no_mamba ta_gated_conv_text no_interpretation dense_moe no_vision no_text"}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

require_cache () {
  for split in train valid test; do
    if [ ! -f "$CACHE/$split.npy" ]; then
      echo "ERROR: $CACHE/$split.npy is missing -- the MOSEI explanation precompute"
      echo "       has not finished this split.  Check:  tail -4 mosei_video_precompute.log"
      echo "       (train=16326, valid=1871, test=4659 samples expected)"
      exit 1
    fi
  done
  echo "cache OK: $CACHE has train/valid/test"
}

run_arch () {  # $1 = arch, $2 = condition
  local arch=$1 cond=$2 dir json log rc
  dir="$OUT/multiseed_$cond/$arch"
  json="$dir/multiseed_$arch.json"
  log="$OUT/logs/${cond}_${arch}.log"
  mkdir -p "$OUT/logs"
  if [ -f "$json" ] && [ "${RESUME:-1}" = "1" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "----- skip $cond/$arch (finished: $json)"
    return 0
  fi
  echo "----- $cond/$arch  seeds=[$SEEDS]  $(date '+%F %T')"
  if [ "$cond" = "interp" ]; then
    TEMOE_INTERP_DIR="$CACHE" python -m mmsa.training.multiseed --config "$CFG" \
      --arch "$arch" --seeds $SEEDS --output-dir "$dir" > "$log" 2>&1
  else
    # env -u so a leftover TEMOE_INTERP_DIR in the shell cannot leak the cache in
    env -u TEMOE_INTERP_DIR python -m mmsa.training.multiseed --config "$CFG" \
      --arch "$arch" --seeds $SEEDS --output-dir "$dir" > "$log" 2>&1
  fi
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "!!!!! $cond/$arch FAILED rc=$rc (log: $log)"
  fi
  grep -E '^\{' "$log" 2>/dev/null | tail -1
  return 0
}

run_condition () {  # $1 = condition
  local cond=$1
  if [ "$cond" = "interp" ]; then require_cache; fi
  echo "===== MODE=crossing COND=$cond  archs=[$ARCHS]  $(date '+%F %T') ====="
  for a in $ARCHS; do run_arch "$a" "$cond"; done
  if [ "$cond" = "interp" ]; then
    python -m mmsa.training.summarize --results-dir "$OUT/multiseed" \
      --reference-arch temoe --output-dir "$OUT/summary"
  fi
}

run_ablation () {
  local only_arg="" rc
  require_cache
  if [ "$ABL_ONLY" != "all" ]; then only_arg="--only $ABL_ONLY"; fi
  echo "===== MODE=ablation WITH cached explanation ====="
  echo "variants: $ABL_ONLY"
  echo "seeds   : $ABL_SEEDS   start: $(date '+%F %T')"
  mkdir -p "$OUT"
  # TEMOE_INTERP_DIR is essential here: without it interp_dim is 0, train.py
  # auto-disables the explanation branch, and `full` collapses onto
  # `no_interpretation` -- a silently worthless ablation.
  TEMOE_INTERP_DIR="$CACHE" python -m mmsa.training.ablation --config "$CFG" \
    --output-dir "$OUT/ablation" --seeds $ABL_SEEDS $only_arg > "$OUT/ablation_interp.log" 2>&1
  rc=$?
  echo "ablation rc=$rc (log: $OUT/ablation_interp.log)"
  tail -8 "$OUT/ablation_interp.log" 2>/dev/null
  # verify from the run reports that the explanation really was active
  python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/mmsa/mosei/ablation")
if root.is_dir():
    for variant in sorted(p for p in root.iterdir() if p.is_dir()):
        reports = sorted(variant.glob("seed_*/*_report.json"))
        if not reports:
            continue
        flags = {json.loads(r.read_text(encoding="utf-8")).get("use_interpretation") for r in reports}
        print(f"{variant.name:24s} seeds={len(reports)} use_interpretation={flags}")
PY
}

run_efficiency () {
  echo "===== MODE=efficiency (5 runs each) ====="
  mkdir -p "$OUT/efficiency"
  for i in 1 2 3 4 5; do
    python -m mmsa.efficiency.profile --output "$OUT/efficiency/mamba_run$i.json" >/dev/null 2>&1
    python -m mmsa.efficiency.profile --no-mamba --output "$OUT/efficiency/gru_run$i.json" >/dev/null 2>&1
  done
  echo "===== end-to-end deployment (online MLLM vs cache) ====="
  python -m mmsa.efficiency.end_to_end \
    --pkl "MSA Datasets/MOSEI/Processed/aligned_50.pkl" \
    --raw-dir "MSA Datasets/MSA-Datasets/CMU-MOSEI/Raw" \
    --vl-model models/Qwen2.5-VL-7B-Instruct --samples 30 \
    --blacklist outputs/mmsa/mosei/video_blacklist.json \
    --output "$OUT/efficiency/end_to_end.json" 2>&1 | tail -12
}

case "$MODE" in
  crossing)
    case "$COND" in
      both)     run_condition nointerp; run_condition interp ;;
      interp|nointerp) run_condition "$COND" ;;
      *) echo "unknown COND=$COND"; exit 1 ;;
    esac
    ;;
  ablation)   run_ablation ;;
  efficiency) run_efficiency ;;
  *) echo "unknown MODE=$MODE"; exit 1 ;;
esac
echo "MODE=$MODE COND=$COND done  $(date '+%F %T')"
