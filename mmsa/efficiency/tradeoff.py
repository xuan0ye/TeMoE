from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmsa.efficiency.profile import profile_model


def run_tradeoff(
    audio_dim: int,
    vision_dim: int,
    text_dim: int,
    interp_dim: int,
    seq_len: int,
    batch_size: int,
    top_ks: list[int],
    num_experts: int,
    output: str | None,
) -> dict:
    rows: list[dict] = []

    # 1. temporal backend: Mamba vs GRU fallback (linear-time modeling claim)
    for use_mamba in (True, False):
        rows.append(
            _label(
                profile_model(
                    audio_dim=audio_dim,
                    vision_dim=vision_dim,
                    text_dim=text_dim,
                    interp_dim=interp_dim,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    num_experts=num_experts,
                    top_k=2,
                    use_mamba=use_mamba,
                    use_interpretation=True,
                ),
                variant=f"backend={'mamba' if use_mamba else 'gru'}",
            )
        )

    # 2. MoE sparsity: top-k sweep (dense == top_k = num_experts)
    for top_k in top_ks:
        rows.append(
            _label(
                profile_model(
                    audio_dim=audio_dim,
                    vision_dim=vision_dim,
                    text_dim=text_dim,
                    interp_dim=interp_dim,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    num_experts=num_experts,
                    top_k=top_k,
                    use_mamba=True,
                    use_interpretation=True,
                ),
                variant=f"top_k={top_k}",
            )
        )

    # 3. interpretation branch on/off (offline-cached vs no MLLM)
    for use_interp in (True, False):
        rows.append(
            _label(
                profile_model(
                    audio_dim=audio_dim,
                    vision_dim=vision_dim,
                    text_dim=text_dim,
                    interp_dim=interp_dim,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    num_experts=num_experts,
                    top_k=2,
                    use_mamba=True,
                    use_interpretation=use_interp,
                ),
                variant=f"interpretation={'cached' if use_interp else 'off'}",
            )
        )

    result = {"batch_size": batch_size, "seq_len": seq_len, "num_experts": num_experts, "rows": rows}
    _write_markdown(rows, output)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text, encoding="utf-8")
    return result


def _label(profile: dict, variant: str) -> dict:
    return {"variant": variant, **profile}


def _write_markdown(rows: list[dict], output: str | None) -> None:
    if not output:
        return
    md_path = Path(output).with_suffix(".md")
    columns = ["variant", "temporal_backend", "params_million", "latency_ms_per_batch", "throughput_samples_per_s", "flops_gflops"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Efficiency-accuracy trade-off sweep for TeMoE.")
    parser.add_argument("--audio-dim", type=int, default=5)
    parser.add_argument("--vision-dim", type=int, default=20)
    parser.add_argument("--text-dim", type=int, default=768)
    parser.add_argument("--interp-dim", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-ks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output", default="outputs/mmsa/efficiency/tradeoff.json")
    args = parser.parse_args()

    run_tradeoff(
        audio_dim=args.audio_dim,
        vision_dim=args.vision_dim,
        text_dim=args.text_dim,
        interp_dim=args.interp_dim,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        top_ks=args.top_ks,
        num_experts=args.num_experts,
        output=args.output,
    )


if __name__ == "__main__":
    main()
