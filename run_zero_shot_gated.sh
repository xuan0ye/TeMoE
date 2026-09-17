#!/usr/bin/env bash
# SCRIPT_REV=2
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

STAGE="${STAGE:-smoke}"
DATASETS="${DATASETS:-mosi}"
SMOKE_N="${SMOKE_N:-8}"
MODEL="${MODEL:-models/Qwen2.5-VL-7B-Instruct}"
LOG_DIR="outputs/mmsa/logs"
mkdir -p "$LOG_DIR" outputs/mmsa/official

echo "[zero-shot] SCRIPT_REV=2 stage=$STAGE datasets=$DATASETS $(date '+%F %T')"
echo "[zero-shot] python=$(command -v python) model=$MODEL"

case "$STAGE" in
  smoke|full) ;;
  *) echo "STAGE must be smoke or full" >&2; exit 2 ;;
esac

ACTIVE="$(pgrep -af 'python.*mmsa.data.mllm_zero_shot_baseline|python.*mmsa.data.precompute_interpretation_video' || true)"
if [ -n "$ACTIVE" ]; then
  echo "Another Qwen video job is already running; refusing to compete for the GPU:" >&2
  echo "$ACTIVE" >&2
  exit 3
fi

for path in "$MODEL" mmsa/data/mllm_zero_shot_baseline.py; do
  [ -e "$path" ] || { echo "MISSING: $path" >&2; exit 2; }
done

python - <<'PY'
import torch
from mmsa.data.mllm_zero_shot_baseline import _parse_score
assert _parse_score("Score: -0.8", -3, 3) == (-0.8, True)
print(f"[preflight] torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required; refusing to run the 7B baseline on CPU")
PY

run_one() {
  local ds="$1" pkl raw low high blacklist output log
  case "$ds" in
    mosi)
      pkl="MSA Datasets/MOSI/Processed/aligned_50.pkl"
      raw="MSA Datasets/MSA-Datasets/CMU-MOSI/Raw"
      low=-3; high=3; blacklist=""
      ;;
    sims)
      pkl="MSA Datasets/SIMS/Processed/unaligned_39.pkl"
      raw="MSA Datasets/MSA-Datasets/CH-SIMS/Raw"
      low=-1; high=1; blacklist=""
      ;;
    mosei)
      pkl="MSA Datasets/MOSEI/Processed/aligned_50.pkl"
      raw="MSA Datasets/MSA-Datasets/CMU-MOSEI/Raw"
      low=-3; high=3; blacklist="outputs/mmsa/mosei/video_blacklist.json"
      ;;
    *) echo "unknown dataset: $ds" >&2; return 2 ;;
  esac

  [ -f "$pkl" ] || { echo "MISSING: $pkl" >&2; return 2; }
  [ -d "$raw" ] || { echo "MISSING: $raw" >&2; return 2; }
  if [ -n "$blacklist" ] && [ ! -f "$blacklist" ]; then
    echo "MISSING: $blacklist" >&2
    return 2
  fi

  mkdir -p "outputs/mmsa/$ds/baselines"
  if [ "$STAGE" = smoke ]; then
    output="outputs/mmsa/$ds/baselines/qwen25vl_zeroshot_smoke.json"
    log="$LOG_DIR/zeroshot_${ds}_smoke.log"
  else
    output="outputs/mmsa/$ds/baselines/qwen25vl_zeroshot.json"
    log="$LOG_DIR/zeroshot_${ds}.log"
  fi

  args=(python -u -m mmsa.data.mllm_zero_shot_baseline
    --pkl "$pkl" --raw-dir "$raw" --vl-model "$MODEL"
    --label-range "$low" "$high" --output "$output" --progress-every 25)
  if [ -n "$blacklist" ]; then args+=(--blacklist "$blacklist"); fi
  if [ "$STAGE" = smoke ]; then args+=(--max-samples "$SMOKE_N" --progress-every 1); fi

  echo "[dataset] START ds=$ds output=$output $(date '+%F %T')"
  "${args[@]}" 2>&1 | tee "$log"
  python - "$output" "$STAGE" "$SMOKE_N" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
expected = int(sys.argv[3]) if sys.argv[2] == "smoke" else None
assert d["mode"] == "zero-shot-direct-score"
assert expected is None or d["n_samples"] == expected
assert d["parse_success_rate"] >= 0.75, d
assert d["input_counts"]["text_fallback_failures"] == 0, d
print(f"[dataset] VERIFIED {p} n={d['n_samples']} MAE={d['test']['mae']:.4f} corr={d['test']['corr']:.4f} parse={d['parse_success_rate']:.3f}")
PY
  echo "[dataset] END ds=$ds $(date '+%F %T')"
}

for ds in $DATASETS; do run_one "$ds"; done

if [ "$STAGE" = full ]; then
  python - <<'PY'
import json
from pathlib import Path
rows = {}
for ds in ("mosi", "sims", "mosei"):
    p = Path(f"outputs/mmsa/{ds}/baselines/qwen25vl_zeroshot.json")
    if p.exists():
        rows[ds] = json.loads(p.read_text(encoding="utf-8"))
out = Path("outputs/mmsa/official/qwen25vl_zeroshot_summary.json")
out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
print(f"[summary] wrote {out} datasets={sorted(rows)}")
PY
fi

echo "[zero-shot] COMPLETE stage=$STAGE $(date '+%F %T')"