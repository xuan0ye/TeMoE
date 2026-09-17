r"""Expert-routing interpretability analysis for TeMoE's text-routed sparse
Mixture-of-Experts (mmsa.models.text_moe.TextGuidedSparseMoE).

TEXT (Rao et al. 2026) asserts, but does not measure, that its analogous
text-routed SMoE is "keyword-sensitive" -- i.e. that individual experts
specialize for particular sentiment content, and that this specialization
"may foster cross-modal consistency for improved interpretability" (their
Discussion, \S4.6). This script tests that claim empirically for TeMoE
instead of only asserting it: it runs the trained router over the *entire*
test set and reports

1. a contingency table of the top-1-routed expert vs.\ a coarse
   negative/neutral/positive sentiment-polarity bucket (from the ground-truth
   label, not the prediction, so this is a property of the router alone);
2. the normalized mutual information (NMI) between top-1 expert and polarity
   bucket -- 0 means routing is independent of sentiment polarity, 1 means
   the expert id fully determines (or is fully determined by) polarity;
3. mean routing entropy (nats, and normalized by log(num_experts) into
   [0, 1]) -- low entropy means the router commits confidently to a small
   expert subset per sample, high entropy means it near-uniformly spreads
   weight across all candidate experts;
4. per-expert utilization (fraction of test samples routed to each expert
   as their top-1 choice), to check for the "always picks the same expert"
   degenerate failure mode sparse MoE training is known to risk.

Running this for both the `full` (top-k=2, sparse) and `dense_moe`
(top-k=num_experts, i.e.\ every expert always active) ablation variants
additionally tests whether *sparsity itself* -- not just text-conditioning --
is what produces any specialization observed, tying this analysis back to
the efficiency narrative (\S~sec:efficiency in the paper): if entropy is
similarly low for both, specialization is a property of the text-conditioned
router regardless of sparsity; if `dense_moe` has much higher entropy, sparse
top-k selection is itself necessary to force specialization.

Usage (after mmsa.training.ablation has produced `full`/`dense_moe`
checkpoints for the given seed -- no additional training needed):

    python -m mmsa.analysis.expert_routing \
        --config mmsa/configs/mosi.yaml \
        --ablation-dir outputs/mmsa/mosi/ablation \
        --seed 42 --variants full dense_moe \
        --output outputs/mmsa/mosi/analysis/expert_routing.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score

from mmsa.analysis.common import load_variant


@torch.no_grad()
def _collect(model: torch.nn.Module, test_set, use_interp: bool, device: torch.device):
    top1: list[int] = []
    entropy: list[float] = []
    labels: list[float] = []
    n_experts = model.moe.num_experts  # type: ignore[union-attr]

    for index in range(len(test_set)):
        item = test_set[index]
        interpretation = item.get("interpretation")
        interpretation = interpretation.unsqueeze(0).to(device) if (use_interp and interpretation is not None) else None
        # Plan-B channel path: without the real per-channel clues the model would
        # zero-fill them (see TeMoE.forward), and the resulting routing would not
        # reflect the trained model at all -- so pass them whenever the rebuilt
        # architecture consumes channels.
        channel_kwargs: dict[str, torch.Tensor] = {}
        if getattr(model, "use_channels", False):
            for key in ("interp_face", "interp_content", "interp_bg", "interp_bg_flag"):
                value = item.get(key)
                if value is not None:
                    channel_kwargs[key] = value.unsqueeze(0).to(device)
        model(
            audio=item["audio"].unsqueeze(0).to(device),
            vision=item["vision"].unsqueeze(0).to(device),
            text=item["text"].unsqueeze(0).to(device),
            interpretation=interpretation,
            **channel_kwargs,
        )
        probs = model.moe.last_probs[0].cpu().numpy()  # type: ignore[union-attr]
        top1.append(int(model.moe.last_topk_idx[0, 0].item()))  # type: ignore[union-attr]
        entropy.append(float(-(probs * np.log(probs + 1e-12)).sum()))
        labels.append(float(item["label"].item()))

    return np.asarray(top1), np.asarray(entropy), np.asarray(labels), n_experts


def _polarity_bucket(labels: np.ndarray) -> np.ndarray:
    """0 = negative, 1 = neutral (label==0), 2 = positive. Deliberately the
    same has0-style sign convention as msa_regression_metrics, so this bucket
    is consistent with the rest of the paper's binary-metric definitions."""
    bucket = np.ones_like(labels, dtype=int)
    bucket[labels < 0] = 0
    bucket[labels > 0] = 2
    return bucket


def analyze_variant(
    config_path: str,
    variant: str,
    ablation_dir: str,
    seed: int,
    device: torch.device,
    ckpt_variant: str | None = None,
) -> dict:
    model, test_set, use_interp = load_variant(config_path, variant, ablation_dir, seed, device, ckpt_variant)
    top1, entropy, labels, n_experts = _collect(model, test_set, use_interp, device)
    bucket = _polarity_bucket(labels)

    contingency = np.zeros((n_experts, 3), dtype=int)
    for expert_id, pol in zip(top1, bucket):
        contingency[expert_id, pol] += 1

    nmi = float(normalized_mutual_info_score(bucket, top1)) if len(set(top1.tolist())) > 1 else 0.0
    max_entropy = float(np.log(n_experts)) if n_experts > 1 else 0.0
    utilization = np.bincount(top1, minlength=n_experts) / max(1, len(top1))

    return {
        "variant": variant,
        "n_experts": int(n_experts),
        "n_samples": int(len(top1)),
        "top1_vs_polarity_nmi": round(nmi, 4),
        "mean_routing_entropy_nats": round(float(entropy.mean()), 4),
        "mean_routing_entropy_normalized": round(float(entropy.mean() / max_entropy), 4) if max_entropy > 0 else 0.0,
        "expert_utilization_top1_share": [round(float(u), 4) for u in utilization],
        "contingency_expert_x_polarity_neg_neu_pos": contingency.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Expert-routing interpretability analysis for TeMoE's sparse MoE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ablation-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["full", "dense_moe"],
        help="ablation variants to analyze (must have use_mamba/temoe arch with a MoE router)",
    )
    parser.add_argument("--output", default="outputs/mmsa/analysis/expert_routing.json")
    parser.add_argument(
        "--checkpoint-variant",
        default=None,
        help="checkpoint sub-directory to load when it differs from the config variant name "
        "(e.g. --variants single_channel --checkpoint-variant full for a pre-Plan-B checkpoint)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [
        analyze_variant(args.config, variant, args.ablation_dir, args.seed, device, args.checkpoint_variant)
        for variant in args.variants
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
