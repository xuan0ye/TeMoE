# Environment Provenance

## Primary TeMoE environment

- Ubuntu 20.04
- Python 3.10
- CUDA 12.1
- PyTorch 2.4.0+cu121
- NVIDIA A800 80 GB PCIe
- optional selective-scan Mamba backend, with bidirectional-GRU fallback

`requirements.lock.txt` pins the critical packages to the versions captured
from the completed run. The full byte-for-byte package listing is preserved in
`results/environment/wjhenv.freeze.txt`. `requirements-mamba.lock.txt` is kept
separate because these CUDA extensions are platform-sensitive.

## Official MMSA environment

The independent check ran in a separate environment with MMSA 2.2.1. Its
captured package versions are recorded in
`requirements-mmsa-official.lock.txt`, with the complete freeze at
`results/environment/mmsa_official.freeze.txt`. Keeping it separate prevents
the official package's dependency choices from changing the primary training
environment.
