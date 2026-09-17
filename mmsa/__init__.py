"""MLLM-Augmented Temporal-Aligned Mixture-of-Experts for Multimodal Sentiment Analysis.

Package layout:
- data: MMSA pkl sequence dataset + MLLM interpretation precompute
- models: temporal alignment (Mamba + time cross-attention) + text-centric sparse MoE
- training: training loop, MSA metrics, ablation runner
- efficiency: params / FLOPs / latency / throughput profiling
"""

import torch

# cuDNN is disabled package-wide on this deployment: the A800 server's cuDNN
# state is corrupted (every fresh CUDA context fails with
# CUDNN_STATUS_NOT_INITIALIZED -- see mmsa/README.md). Disabling cuDNN lets
# PyTorch fall back to its native kernels; Qwen2.5-VL and the TeMoE training
# loop both work correctly without it (only slightly slower), and disabling
# it also removes cuDNN's non-determinism, which matches the paper's
# multi-seed protocol.
torch.backends.cudnn.enabled = False

__all__ = ["data", "models", "training", "efficiency"]
