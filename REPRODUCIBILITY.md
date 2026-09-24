# Reproducibility Guide

## 1. Scope

There are three reproduction levels:

1. **Integrity and unit tests** require no datasets, model weights, or GPU.
2. **Aggregate-analysis reproduction** uses the included JSON summaries.
3. **End-to-end reproduction** requires licensed datasets, local model weights,
   generated caches, checkpoints, and an NVIDIA GPU.

All commands below run from the repository root.

## 2. Hardware and software

Primary experiments were run on Ubuntu 20.04 with one NVIDIA A800 80 GB GPU,
CUDA 12.1, Python 3.10, and PyTorch 2.4.0+cu121. CPU-only integrity checks are
platform independent.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python scripts/release_check.py --run-tests
```

For the optional Mamba backend:

```bash
python -m pip install -r requirements-mamba.lock.txt
```

For the official MMSA sensitivity study, create an isolated environment and
install `requirements-mmsa-official.lock.txt`. Do not replace the TeMoE
environment with it.

## 3. Licensed resources

The default gated scripts expect the following repository-relative layout:

```text
MSA Datasets/MOSI/Processed/aligned_50.pkl
MSA Datasets/MOSEI/Processed/aligned_50.pkl
MSA Datasets/SIMS/Processed/unaligned_39.pkl
MSA Datasets/MSA-Datasets/CMU-MOSI/Raw/
MSA Datasets/MSA-Datasets/CMU-MOSEI/Raw/
models/Qwen2.5-VL-7B-Instruct/
models/Qwen2.5-1.5B-Instruct/
data/interpretation/mosi_video/{train,valid,test}.npy
data/interpretation/mosei_video/{train,valid,test}.npy
```

The MOSI main cache is `mosi_video` (video plus transcript), not the distinct
`mosi_videonly` modality-control cache. The release-audit script auto-detects
the main `*_video` directories and prints the resolved paths before running.

## 4. Core training and controls

The repository includes gated entry points rather than one monolithic command:

```bash
bash run_mosei_chain.sh
bash run_mosei_suite.sh
bash run_mosei_ablation.sh
bash run_controls.sh
bash run_baseline_upgrade.sh
bash run_rationale_quality.sh
```

Inspect each script's environment-variable block before launching. Five-seed
experiments use seeds 42, 1, 2, 3, and 4. Aggregate claims must be read from
machine-generated JSON rather than terminal output.

## 5. MOSEI rationale-source control

The released summary is results/mosei/analysis/mosei_source_control.json.
It contains the test MAE for each of five paired seeds under no cache,
Qwen2.5-VL-7B transcript-only rationales, and video-plus-transcript
rationales for TeMoE, LMF, and Late Fusion. The summary contains no raw
clips, transcript text, generated rationales, or embedding vectors.

Recompute the declared +/-0.06 MAE and stricter +/-0.01 MAE TOSTs from the
released JSON without datasets, model weights, or a GPU:

    python -m mmsa.analysis.verify_mosei_source_control --report results/mosei/analysis/mosei_source_control.json

To regenerate the underlying transcript-only cache and train the three
consumers, obtain the licensed MOSEI data and model weights listed above,
prepare the matched video-plus-transcript cache, then run on the GPU host:

    MODE=all ARCHS="temoe lmf late_fusion" bash run_mosei_source_control.sh

The script logs fallback behavior and validates finite, shape-matched
384-dimensional cache arrays before training. It resumes cache generation
from JSONL checkpoints. The full run is expensive; if all condition runs
already exist, regenerate only the report with MODE=report and the same
ARCHS value. The generated JSON and Markdown land under
outputs/mmsa/mosei/analysis/; only aggregate summaries are published.
Source equivalence is conditional on the fixed test split, the five seeds,
and the stated margin. MOSEI clip-only and other consumers were not tested.

## 6. Release audits

Run the smoke gate first:

```bash
STAGE=smoke bash run_release_audit_gated.sh
```

Expected smoke completion markers are:

```text
[verified] matched speedup=... official_models=1
[release-audit] COMPLETE stage=smoke
```

Then launch the full audit:

```bash
nohup env STAGE=full bash run_release_audit_gated.sh \
  > release_audit_full.log 2>&1 &
echo $! > release_audit_full.pid
```

The full run uses 30 timing samples and the released MMSA LMF model under two
conditions, with five paired seeds per condition. LMF natively accepts the
continuous text features needed by the data-only adapter. The released MISA,
Self-MM, and MMIM paths require `use_bert=True`; adding a continuous cache to
them would require modifying the official model and would no longer be an
unmodified-code crossing. On one A800 the complete audit is expected to take
about 15--30 minutes.

Progress:

```bash
tail -f release_audit_full.log
grep -E "\[latency\]|\[official-crossing\]|\[verified\]|COMPLETE|fatal|Traceback" \
  release_audit_full.log | tail -80
find outputs/mmsa/official_crossing/mosi_full/runs \
  -name run_record.json | wc -l
nvidia-smi
```

There are two final `run_record.json` files: LMF times two arms.
Successful completion produces:

```text
outputs/mmsa/mosei/efficiency/end_to_end_v2_full.json
outputs/mmsa/official/official_crossing_mosi_full.json
outputs/mmsa/official/official_crossing_mosi_full.md
release_audit_results_full.tgz
```

## 7. Timing boundaries

The legacy 1.351 s online measurement includes video read/decode, processor
preprocessing, host-to-device transfer, Qwen2.5-VL generation, and decoding. It
excludes model loading, rationale sentence encoding, and downstream scoring.

The legacy 3.8123 ms cached measurement is a CUDA-synchronized TeMoE forward on
already-resident GPU tensors. It excludes cache lookup and host-to-device
transfer. Their 354.4x ratio is therefore a component-cost ratio, not a matched
end-to-end speedup.

The v2 full audit measures both arms at the same boundary: real tensor
construction, host-to-device transfer, the same trained TeMoE forward, and CUDA
synchronization. The online arm additionally performs raw-video rationale
generation and sentence encoding; the cached arm performs cache lookup. Model
and dataset loading remain outside both measurements.

## 8. Result validation

```bash
python scripts/release_check.py
python -m mmsa.analysis.equivalence_tests
```

The first command verifies package integrity. The second recalculates paired
equivalence decisions when the original multiseed run directories are present;
the included formal outputs are preserved under `results/official/` for
data-free inspection.
