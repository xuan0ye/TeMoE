#!/usr/bin/env bash
# ============================================================================
# run_mosei_chain.sh -- the whole MOSEI plan, unattended, ordered by value.
#
# Stage 1 starts IMMEDIATELY: the explanation-free runs do not touch the MLLM
# cache, so the GPU is never idle while the precompute finishes the valid/test
# splits.  Everything after that waits for the cache.
#
#   stage 1  crossing / nointerp   top-5 archs   (no cache needed)
#   ---- wait for data/interpretation/mosei_video/{train,valid,test}.npy ----
#   stage 2  crossing / interp     top-5 archs   -> 5 complete table rows
#   stage 3  crossing / nointerp   remaining archs
#   stage 4  crossing / interp     remaining archs
#   stage 5  ablation (classic variants)
#   stage 6  efficiency + end-to-end deployment
#   stage 7  condition audit + supplementary regeneration
#
# Every stage is resumable: a finished arch is skipped, a phase failure does not
# stop the chain, and re-running this script is always safe.
#
# Usage:  nohup bash run_mosei_chain.sh > mosei_chain.log 2>&1 &
#         tail -f mosei_chain.log
#
# Overrides: SEEDS="42 1 2"  ABL_SEEDS="42 1 2"  SKIP_WAIT=1
#            TOP5="temoe almt kuda late_fusion lmf"  MAXWAIT=21600  POLL=60
#            STAGES="1 2 3 4 5 6 7"   (run a subset)
# ============================================================================
set -u
cd "$(dirname "$0")"
CACHE=${CACHE:-data/interpretation/mosei_video}
SEEDS=${SEEDS:-42 1 2 3 4}
ABL_SEEDS=${ABL_SEEDS:-$SEEDS}
TOP5=${TOP5:-"temoe almt kuda late_fusion lmf"}
REST=${REST:-"misa selfmm magbert cross_modal cormult"}
LOGDIR=${LOGDIR:-logs}
POLL=${POLL:-60}
MAXWAIT=${MAXWAIT:-21600}
STAGES=${STAGES:-"1 2 3 4 5 6 7"}
mkdir -p "$LOGDIR" outputs/mmsa/mosei/logs

stamp () { date '+%F %T'; }
stage_on () { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

run_stage () {  # $1 = tag, $2.. = env assignments and command
  local tag=$1; shift
  echo "[chain] ---- $tag START $(stamp)"
  "$@" > "$LOGDIR/mosei_$tag.log" 2>&1
  local rc=$?
  echo "[chain] ---- $tag EXIT=$rc $(stamp)  (log: $LOGDIR/mosei_$tag.log)"
  tail -3 "$LOGDIR/mosei_$tag.log" 2>/dev/null | sed 's/^/[chain]    /'
  return 0
}

wait_for_cache () {
  if [ "${SKIP_WAIT:-0}" = "1" ]; then return 0; fi
  echo "[chain] waiting for $CACHE/test.npy  (poll ${POLL}s, cap ${MAXWAIT}s)"
  local waited=0 warned=0 v t alive
  while [ ! -f "$CACHE/test.npy" ]; do
    if [ "$waited" -ge "$MAXWAIT" ]; then
      echo "[chain] TIMEOUT after ${MAXWAIT}s: $CACHE/test.npy still missing; stopping"
      echo "[chain] before starting any interp run.  Re-run this script later."
      exit 1
    fi
    sleep "$POLL"
    waited=$((waited + POLL))
    if [ $((waited % 600)) -lt "$POLL" ]; then
      v=$(wc -l < "$CACHE/valid_rationales.jsonl" 2>/dev/null || echo 0)
      t=$(wc -l < "$CACHE/test_rationales.jsonl" 2>/dev/null || echo 0)
      alive=$(pgrep -fc precompute_interpretation_video 2>/dev/null || echo 0)
      echo "[chain] waited ${waited}s | valid ${v}/1871 | test ${t}/4659 | precompute procs: ${alive}"
      if [ "$alive" = "0" ] && [ "$warned" = "0" ]; then
        warned=1
        echo "[chain] WARNING: the precompute process is gone but the cache is incomplete."
        echo "[chain]          Resume it (safe, checkpoints are kept):"
        echo "[chain]          nohup python -m mmsa.data.precompute_interpretation_video \\"
        echo "[chain]            --pkl 'MSA Datasets/MOSEI/Processed/aligned_50.pkl' \\"
        echo "[chain]            --raw-dir 'MSA Datasets/MSA-Datasets/CMU-MOSEI/Raw' \\"
        echo "[chain]            --output-dir data/interpretation/mosei_video \\"
        echo "[chain]            --vl-model models/Qwen2.5-VL-7B-Instruct \\"
        echo "[chain]            --encoder-name paraphrase-multilingual-MiniLM-L12-v2 \\"
        echo "[chain]            --text-llm-model models/Qwen2.5-1.5B-Instruct \\"
        echo "[chain]            --blacklist outputs/mmsa/mosei/video_blacklist.json \\"
        echo "[chain]            >> mosei_video_precompute.log 2>&1 &"
      fi
    fi
  done
  echo "[chain] cache COMPLETE $(stamp)"
}

echo "[chain] ================ MOSEI chain start $(stamp) seeds=[$SEEDS] ================"
echo "[chain] stages: $STAGES | top archs: $TOP5"

if stage_on 1; then
  run_stage 1_nointerp_top env SEEDS="$SEEDS" COND=nointerp ARCHS="$TOP5" bash run_mosei_suite.sh
fi

if stage_on 2 || stage_on 3 || stage_on 4; then
  wait_for_cache
fi

if stage_on 2; then
  run_stage 2_interp_top env SEEDS="$SEEDS" COND=interp ARCHS="$TOP5" bash run_mosei_suite.sh
fi
if stage_on 3; then
  run_stage 3_nointerp_rest env SEEDS="$SEEDS" COND=nointerp ARCHS="$REST" bash run_mosei_suite.sh
fi
if stage_on 4; then
  run_stage 4_interp_rest env SEEDS="$SEEDS" COND=interp ARCHS="$REST" bash run_mosei_suite.sh
fi
if stage_on 5; then
  run_stage 5_ablation env ABL_SEEDS="$ABL_SEEDS" MODE=ablation bash run_mosei_suite.sh
fi
if stage_on 6; then
  run_stage 6_efficiency env MODE=efficiency bash run_mosei_suite.sh
fi
if stage_on 7; then
  echo "[chain] ---- 7_audit START $(stamp)"
  python -m mmsa.analysis.audit_conditions --root outputs/mmsa \
    --datasets mosi sims mosei \
    --markdown outputs/mmsa/condition_audit.md \
    --output outputs/mmsa/condition_audit.json > "$LOGDIR/mosei_7_audit.log" 2>&1
  echo "[chain] ---- 7_audit EXIT=$? $(stamp)"
  sed -n '/^## MOSEI/,/^## /p' outputs/mmsa/condition_audit.md 2>/dev/null | head -30 | sed 's/^/[chain]    /'
  python -m mmsa.analysis.make_supplementary --results-dir outputs/mmsa \
    --output outputs/mmsa/supplementary.md >> "$LOGDIR/mosei_7_audit.log" 2>&1
  echo "[chain] ---- 7_supplement EXIT=$? $(stamp)"
fi

echo "[chain] ================ MOSEI chain done $(stamp) ================"
ls -1 outputs/mmsa/mosei/multiseed/*/multiseed_*.json 2>/dev/null | sed 's/^/[chain] w\/   /'
ls -1 outputs/mmsa/mosei/multiseed_nointerp/*/multiseed_*.json 2>/dev/null | sed 's/^/[chain] w\/o  /'
