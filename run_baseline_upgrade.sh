#!/usr/bin/env bash
# ============================================================================
# run_baseline_upgrade.sh -- re-run the two baselines whose implementations just
# became faithful, so their rows in the main table are produced by the new code.
#
#   MISA    now carries all four auxiliary losses (similarity + difference +
#           reconstruction + CMD); the previous version had only two.
#   Self-MM now builds modality-specific pseudo-labels
#           L_m = alpha_m*y + (1-alpha_m)*y_hat_m with alpha re-estimated each
#           epoch from the unimodal errors; the previous version merely distilled
#           the fused prediction and its docstring said so.
#
# Both conditions are re-run, on all three datasets, over the same 5 seeds, so the
# crossing table stays one protocol.  Existing results are overwritten in place.
#
# Usage:  nohup bash run_baseline_upgrade.sh > baseline_upgrade.log 2>&1 &
# ============================================================================
set -u
cd "$(dirname "$0")"
SEEDS=${SEEDS:-42 1 2 3 4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run () {  # $1 dataset  $2 config  $3 condition  $4 arch  $5 interp dir
  local ds=$1 cfg=$2 cond=$3 arch=$4 cache=$5 dir
  dir="outputs/mmsa/$ds/multiseed_$cond/$arch"
  echo "----- $ds/$cond/$arch  $(date '+%H:%M:%S')"
  if [ "$cond" = interp ]; then
    TEMOE_INTERP_DIR="$cache" python -m mmsa.training.multiseed --config "$cfg" \
      --arch "$arch" --seeds $SEEDS --output-dir "$dir" 2>&1 | tail -1
  else
    env -u TEMOE_INTERP_DIR python -m mmsa.training.multiseed --config "$cfg" \
      --arch "$arch" --seeds $SEEDS --output-dir "$dir" 2>&1 | tail -1
  fi
}

for spec in "mosi mmsa/configs/mosi.yaml data/interpretation/mosi_video" \
            "sims mmsa/configs/sims.yaml data/interpretation/sims_video" \
            "mosei mmsa/configs/mosei.yaml data/interpretation/mosei_video"; do
  set -- $spec
  for arch in misa selfmm; do
    run "$1" "$2" nointerp "$arch" "$3"
    run "$1" "$2" interp   "$arch" "$3"
  done
done

echo "===== refresh summary, audit, paired stats, supplement  $(date '+%F %T')"
python -m mmsa.training.summarize --results-dir outputs/mmsa/mosei/multiseed_interp \
  --reference-arch temoe --output-dir outputs/mmsa/mosei/summary
python -m mmsa.analysis.audit_conditions --root outputs/mmsa --datasets mosi sims mosei \
  --markdown outputs/mmsa/condition_audit.md --output outputs/mmsa/condition_audit.json
python -m mmsa.analysis.crossing_stats
python -m mmsa.analysis.make_supplementary --results-dir outputs/mmsa \
  --output outputs/mmsa/supplementary.md

echo "===== the two upgraded rows, before vs after ====="
python - <<'PY'
import json
j = json.load(open("outputs/mmsa/condition_audit.json", encoding="utf-8"))
before = {"mosi": {"misa": (0.937, 0.722), "selfmm": (0.941, 0.714)},
          "sims": {"misa": (0.436, 0.391), "selfmm": (0.438, 0.391)},
          "mosei": {"misa": (0.558, 0.515), "selfmm": (0.568, 0.517)}}
for ds in ("mosi", "sims", "mosei"):
    for arch in ("misa", "selfmm"):
        w = j[ds]["conditions"].get(f"{arch}|nointerp")
        i = j[ds]["conditions"].get(f"{arch}|interp")
        if not w or not i:
            print(f"  {ds:6}{arch:8} incomplete"); continue
        b = before[ds][arch]
        print(f"  {ds:6}{arch:8} w/o {b[0]:.3f} -> {w['mae_mean']:.3f}   w/ {b[1]:.3f} -> {i['mae_mean']:.3f}"
              f"   (n={w['n_seeds']}/{i['n_seeds']})")
PY
echo "===== done $(date '+%F %T') ====="
