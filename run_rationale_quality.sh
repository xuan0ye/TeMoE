#!/usr/bin/env bash
# ============================================================================
# run_rationale_quality.sh -- validate the cached explanations as data.
#
# Cheap (minutes, no MLLM calls): reads the *_rationales.jsonl checkpoints plus
# the dataset labels and reports text statistics, visual grounding, polarity
# agreement, transcript leakage, and a linear probe on rationale embeddings.
#
# It answers the reviewer question "is the cached explanation real signal, or
# just an extra 384-d input / a paraphrase of the transcript?" -- and it needs
# no GPU while training is running, because it defaults to DEVICE=cpu.
#
# The cache directory for each dataset is auto-detected (see --interp-dir auto),
# so the env-var names you used during precompute do not have to be repeated.
#
# Usage:  bash run_rationale_quality.sh
#         SPLITS="test" DEVICE=cuda bash run_rationale_quality.sh
# ============================================================================
set -u
cd "$(dirname "$0")"
SPLITS=${SPLITS:-"valid test"}
DEVICE=${DEVICE:-cpu}
PROBE=${PROBE:-train}

run_one () {  # $1 = label, $2 = pkl
  local label=$1 pkl=$2
  if [ ! -f "$pkl" ]; then
    echo "[skip] $label: missing pkl $pkl"
    return 0
  fi
  echo "===== $label ====="
  echo "      pkl: $pkl   (cache: auto under data/interpretation/)"
  mkdir -p "outputs/mmsa/$label/analysis"
  python -m mmsa.analysis.rationale_quality \
    --pkl "$pkl" --interp-dir auto --cache-root data/interpretation \
    --splits $SPLITS --probe-train-split "$PROBE" --device "$DEVICE" \
    --output "outputs/mmsa/$label/analysis/rationale_quality.json" 2>&1 | tail -80
  echo "[done] $label -> outputs/mmsa/$label/analysis/rationale_quality.json"
}

run_one mosi  "MSA Datasets/MOSI/Processed/aligned_50.pkl"
run_one sims  "MSA Datasets/SIMS/Processed/unaligned_39.pkl"
run_one mosei "MSA Datasets/MOSEI/Processed/aligned_50.pkl"
# Rebuild the condition audit and the supplementary document so that the tables
# are always regenerated from the raw run reports in the same pass.
echo "===== condition audit (from raw run reports) ====="
python -m mmsa.analysis.audit_conditions --root outputs/mmsa --datasets mosi sims mosei \
  --markdown outputs/mmsa/condition_audit.md --output outputs/mmsa/condition_audit.json
echo "===== supplementary document ====="
python -m mmsa.analysis.make_supplementary --results-dir outputs/mmsa \
  --output outputs/mmsa/supplementary.md
echo "all checks finished"
