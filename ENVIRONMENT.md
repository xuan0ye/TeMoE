# Environment Provenance

## Primary TeMoE environment

- Ubuntu 20.04
- Python 3.10
- CUDA 12.1
- PyTorch 2.4.0+cu121
- NVIDIA A800 80 GB PCIe
- optional selective-scan Mamba backend, with bidirectional-GRU fallback

`requirements.lock.txt` is the reproducibility lock selected for the public
artifact. It pins the critical runtime and analysis packages to a mutually
compatible Python 3.10/CUDA 12.1 set. `requirements-mamba.lock.txt` is kept
separate because these CUDA extensions are platform-sensitive.

## Official MMSA environment

The independent check ran in a separate environment with MMSA 2.2.1. Its
captured package versions are recorded in
`requirements-mmsa-official.lock.txt`. Keeping it separate prevents the
official package's dependency choices from changing the primary training
environment.

## Provenance limitation

The complete primary-environment `pip freeze` is produced automatically by the
full release audit. That full audit had not yet been synchronized into this
snapshot, so the public lock is a prescribed reproduction environment rather
than a byte-for-byte retrospective freeze. Once available, the freeze should be
added under `results/environment/` without replacing these readable locks.
