# TeMoE Controlled-Study Artifact

This anonymous-ready artifact accompanies **Cache-Once MLLM Rationales for
Multimodal Sentiment Analysis: A Controlled Study**. It contains the TeMoE
implementation, cache-generation and control scripts, experiment
configurations, aggregate result records, and release-verification utilities.

It intentionally does **not** redistribute licensed datasets, raw video,
foundation-model weights, trained checkpoints, per-clip rationale text, or
rationale embedding caches.

## What this artifact supports

- controlled explanation/no-explanation crossings on CMU-MOSI, CH-SIMS, and
  CMU-MOSEI;
- modality and teacher controls for cached rationales;
- paired equivalence tests for temporal and routing alternatives;
- zero-shot Qwen2.5-VL baselines;
- a matched-boundary latency audit;
- a sensitivity study using MMSA 2.2.1's released models and trainer.

The original aggregate evidence is under [`results/`](results/). A compact
map from paper claims to evidence files is in
[`results/RESULTS.md`](results/RESULTS.md).

## Quick integrity check

The integrity check uses only Python's standard library:

```bash
python scripts/release_check.py
```

To repeat the lightweight unit tests as well:

```bash
python scripts/release_check.py --run-tests
```

The check verifies the SHA-256 manifest, rejects data/model/archive formats,
scans for private paths and identity markers, parses all JSON, compiles all
Python sources, and optionally runs the data-free tests.

## Environment

The principal GPU runs used Python 3.10, PyTorch 2.4.0+cu121, and an NVIDIA
A800 80 GB GPU. Install the pinned primary environment with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
```

Mamba is an optional CUDA extension. If it is unavailable, the code exposes a
GRU fallback; install the exact optional versions from
`requirements-mamba.lock.txt` when reproducing Mamba timings.

The independent official-code check uses a separate environment described by
`requirements-mmsa-official.lock.txt`. See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for data placement, commands,
expected runtimes, progress checks, and output files.

## Repository layout

```text
mmsa/                         TeMoE code, analyses, tests, and YAML configs
results/                      aggregate, non-personal result artifacts
scripts/release_check.py      integrity/privacy/test gate
run_*                         gated experiment entry points
requirements*.lock.txt        pinned environments
MANIFEST.json                 SHA-256 and byte size for each payload file
RELEASE_POLICY.md             explicit inclusion/exclusion policy
```

## Data and model access

Obtain CMU-MOSI, CMU-MOSEI, and CH-SIMS from their official providers. Obtain
Qwen2.5-VL, Qwen2.5, and SentenceTransformers weights under their respective
licenses. Paths can be overridden through the environment variables documented
in the gated shell scripts.

## Anonymity and current status

This candidate contains no author names, email addresses, account names,
machine hostnames, or absolute local paths. The new five-seed official-code
cache crossing and matched-boundary full timing audit were still running when
this snapshot was assembled; only a smoke run existed, so no smoke number is
presented as a formal result. The eventual full JSON/Markdown reports can be
added under `results/official/` without changing the protocol.

See [`LICENSE-NOTICE.md`](LICENSE-NOTICE.md) before redistributing or reusing
the source code.
