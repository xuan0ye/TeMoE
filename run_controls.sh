#!/usr/bin/env bash
# ============================================================================
# Control experiments that answer specific reviewer questions.
#   C1  baseline + cached explanation  -> is the explanation the decisive factor
#                                         for ANY architecture? (R3: "weak baselines")
#   C2  stronger load balancing        -> R1: "why not fix expert collapse?"
#   C3  mismatched explanation cache   -> R1: "what if the cache goes stale?"
# Usage:  bash run_controls.sh            (C1+C2+C3 on MOSI, ~30-45 min total)
#         MODE=c2 bash run_controls.sh    (single experiment)
# ============================================================================
set -u
cd "$(dirname "$0")"
MODE=${MODE:-all}
SEEDS=${SEEDS:-42 1 2 3 4}
MOSI_INTERP=data/interpretation/mosi_video
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$MODE" = all ] || [ "$MODE" = c1 ]; then
  echo "===== C1: baselines augmented with the cached explanation (MOSI) ====="
  for a in lmf late_fusion cross_modal misa kuda; do
    echo "--- $a + interp ---"
    TEMOE_INTERP_DIR=$MOSI_INTERP python -m mmsa.training.multiseed \
      --config mmsa/configs/mosi.yaml --arch $a --seeds $SEEDS \
      --output-dir outputs/mmsa/mosi/multiseed_interp/$a 2>&1 | tail -2
  done
  python -m mmsa.training.summarize --results-dir outputs/mmsa/mosi/multiseed_interp \
    --reference-arch lmf --output-dir outputs/mmsa/mosi/summary_interp
fi

if [ "$MODE" = all ] || [ "$MODE" = c2 ]; then
  echo "===== C2: 10x load-balancing weight (full vs high_aux, MOSI) ====="
  python -m mmsa.training.ablation --config mmsa/configs/mosi.yaml \
    --output-dir outputs/mmsa/mosi/ablation_aux --seeds $SEEDS --only full high_aux 2>&1 | tail -5
fi

if [ "$MODE" = all ] || [ "$MODE" = c3 ]; then
  echo "===== C3: mismatched explanation cache (MOSI) ====="
  TEMOE_SHUFFLE_INTERP=1 TEMOE_INTERP_DIR=$MOSI_INTERP python -m mmsa.training.multiseed \
    --config mmsa/configs/mosi.yaml --arch temoe --seeds $SEEDS \
    --output-dir outputs/mmsa/mosi/multiseed_shuffled/temoe 2>&1 | tail -3
  python -m mmsa.training.summarize --results-dir outputs/mmsa/mosi/multiseed_shuffled \
    --reference-arch temoe --output-dir outputs/mmsa/mosi/summary_shuffled
fi
echo "controls done ($MODE)"