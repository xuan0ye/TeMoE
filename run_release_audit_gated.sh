#!/usr/bin/env bash
# Matched-boundary latency plus an official MMSA 2.2.1 cache crossing.
# Smoke: STAGE=smoke bash run_release_audit_gated.sh
# Full:  nohup env STAGE=full bash run_release_audit_gated.sh > release_audit_full.log 2>&1 &

set -euo pipefail

STAGE=${STAGE:-smoke}
PYTHON=${PYTHON:-python}
OFFICIAL_PYTHON=${OFFICIAL_PYTHON:-"$HOME/.mmsa_official_env/bin/python"}
ENCODER=${ENCODER:-paraphrase-multilingual-MiniLM-L12-v2}
VL_MODEL=${VL_MODEL:-models/Qwen2.5-VL-7B-Instruct}
MOSEI_PKL=${MOSEI_PKL:-"MSA Datasets/MOSEI/Processed/aligned_50.pkl"}
MOSEI_RAW=${MOSEI_RAW:-"MSA Datasets/MSA-Datasets/CMU-MOSEI/Raw"}
MOSEI_CACHE=${MOSEI_CACHE:-}
MOSEI_CKPT=${MOSEI_CKPT:-outputs/mmsa/mosei/multiseed_interp/temoe/seed_42/temoe_model.pt}
MOSEI_BLACKLIST=${MOSEI_BLACKLIST:-outputs/mmsa/mosei/video_blacklist.json}
MOSI_PKL=${MOSI_PKL:-"MSA Datasets/MOSI/Processed/aligned_50.pkl"}
MOSI_CACHE=${MOSI_CACHE:-}

resolve_cache_dir() {
  local label=$1
  local configured=$2
  shift 2
  local candidates=()
  if [ -n "$configured" ]; then
    candidates+=("$configured")
  fi
  candidates+=("$@")
  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -f "$candidate/train.npy" ] && [ -f "$candidate/valid.npy" ] && [ -f "$candidate/test.npy" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo "[fatal] cannot locate $label cache with train.npy, valid.npy, and test.npy" >&2
  echo "[fatal] checked: ${candidates[*]}" >&2
  echo "[hint] set ${label}_CACHE=/actual/cache/directory and rerun" >&2
  return 2
}

case "$STAGE" in
  smoke)
    SAMPLES=3
    WARMUP=1
    MODELS=(LMF)
    SEEDS=(42)
    TAG=smoke
    ;;
  full)
    SAMPLES=30
    WARMUP=2
    # LMF is the released MMSA model that natively accepts continuous text
    # features. MISA/Self-MM/MMIM require use_bert=True and therefore cannot
    # receive an appended 384-D cache without patching the official model.
    MODELS=(LMF)
    SEEDS=(42 1 2 3 4)
    TAG=full
    ;;
  *)
    echo "[fatal] STAGE must be smoke or full, got: $STAGE" >&2
    exit 2
    ;;
esac

# Main video+transcript caches are named *_video. The legacy unsuffixed
# candidates are retained only as a compatibility fallback.
MOSEI_CACHE=$(resolve_cache_dir MOSEI "$MOSEI_CACHE" data/interpretation/mosei_video data/interpretation/mosei)
MOSI_CACHE=$(resolve_cache_dir MOSI "$MOSI_CACHE" data/interpretation/mosi_video data/interpretation/mosi)

echo "[release-audit] stage=$STAGE start=$(date '+%F %T')"
echo "[release-audit] python=$($PYTHON -c 'import sys; print(sys.executable)')"
echo "[release-audit] official_python=$OFFICIAL_PYTHON"
echo "[release-audit] mosei_cache=$MOSEI_CACHE"
echo "[release-audit] mosi_cache=$MOSI_CACHE"

for path in "$MOSEI_PKL" "$MOSEI_RAW" "$MOSEI_CACHE/train.npy" "$MOSEI_CACHE/valid.npy" "$MOSEI_CACHE/test.npy" "$MOSEI_CKPT" "$VL_MODEL" "$MOSI_PKL" "$MOSI_CACHE/train.npy" "$MOSI_CACHE/valid.npy" "$MOSI_CACHE/test.npy" "$OFFICIAL_PYTHON"; do
  if [ ! -e "$path" ]; then
    echo "[fatal] required path is missing: $path" >&2
    exit 2
  fi
done

$PYTHON - <<'PY'
import torch
print(f"[preflight] torch={torch.__version__} cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
print(f"[preflight] device={torch.cuda.get_device_name(0)}")
PY

"$OFFICIAL_PYTHON" - <<'PY'
import importlib.metadata
import MMSA
version = importlib.metadata.version("MMSA")
print(f"[preflight] MMSA={version} module={MMSA.__file__}")
if version != "2.2.1":
    raise SystemExit(f"expected MMSA 2.2.1, got {version}")
PY

mkdir -p outputs/mmsa/official
$PYTHON -m pip freeze > outputs/mmsa/official/env_release_audit_wjhenv.txt
"$OFFICIAL_PYTHON" -m pip freeze > outputs/mmsa/official/env_release_audit_mmsa.txt

EFF_OUT="outputs/mmsa/mosei/efficiency/end_to_end_v2_${TAG}.json"
BLACKLIST_ARGS=()
if [ -f "$MOSEI_BLACKLIST" ]; then
  BLACKLIST_ARGS=(--blacklist "$MOSEI_BLACKLIST")
fi

echo "[latency] matched-boundary probe start=$(date '+%F %T') samples=$SAMPLES"
$PYTHON -m mmsa.efficiency.end_to_end --pkl "$MOSEI_PKL" --raw-dir "$MOSEI_RAW" --vl-model "$VL_MODEL" --config mmsa/configs/mosei.yaml --interpretation-dir "$MOSEI_CACHE" --checkpoint "$MOSEI_CKPT" --encoder-name "$ENCODER" --samples "$SAMPLES" --warmup "$WARMUP" --fps 1 --max-frames 8 --max-pixels 100352 --max-new-tokens 64 "${BLACKLIST_ARGS[@]}" --output "$EFF_OUT"

OFFICIAL_ROOT="outputs/mmsa/official_crossing/mosi_${TAG}"
DATA_ROOT="$OFFICIAL_ROOT/data"
RUN_ROOT="$OFFICIAL_ROOT/runs"
MANIFEST="$DATA_ROOT/manifest.json"
SUMMARY="outputs/mmsa/official/official_crossing_mosi_${TAG}.json"

echo "[official-crossing] prepare start=$(date '+%F %T')"
$PYTHON -m mmsa.analysis.official_crossing prepare --source-pkl "$MOSI_PKL" --cache-dir "$MOSI_CACHE" --output-dir "$DATA_ROOT"

RUN_RECORD_DIRS=()
for arm in no_cache cached; do
  for model in "${MODELS[@]}"; do
    echo "[official-crossing] start arm=$arm model=$model $(date '+%F %T')"
    "$OFFICIAL_PYTHON" -m mmsa.analysis.official_crossing run --dataset mosi --model "$model" --arm "$arm" --manifest "$MANIFEST" --run-root "$RUN_ROOT" --seeds "${SEEDS[@]}"
    echo "[official-crossing] end arm=$arm model=$model $(date '+%F %T')"
    RUN_RECORD_DIRS+=("$RUN_ROOT/$arm/$model")
  done
done

$PYTHON -m mmsa.analysis.official_crossing summarize --run-root "$RUN_ROOT" --output "$SUMMARY" --models "${MODELS[@]}" --seeds "${SEEDS[@]}"

$PYTHON - "$EFF_OUT" "$SUMMARY" <<'PY'
import json, sys
eff = json.load(open(sys.argv[1], encoding="utf-8"))
cross = json.load(open(sys.argv[2], encoding="utf-8"))
matched = eff.get("matched_request_paths")
assert matched and matched["cached_request"]["n_samples"] > 0
assert cross["comparisons"]
print(f"[verified] matched speedup={matched['speedup_online_over_cached']}x official_models={len(cross['comparisons'])}")
PY

RESULT_TAR="release_audit_results_${TAG}.tgz"
tar --exclude='*/saved_models' --exclude='*/saved_models/*' -czf "$RESULT_TAR" "$EFF_OUT" "$SUMMARY" "${SUMMARY%.json}.md" "$MANIFEST" "${RUN_RECORD_DIRS[@]}" outputs/mmsa/official/env_release_audit_wjhenv.txt outputs/mmsa/official/env_release_audit_mmsa.txt

echo "[release-audit] COMPLETE stage=$STAGE $(date '+%F %T')"
echo "[release-audit] return: $RESULT_TAR"
