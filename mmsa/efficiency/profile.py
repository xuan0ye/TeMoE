from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mmsa.models.model import TeMoE


def profile_model(
    audio_dim: int = 5,
    vision_dim: int = 20,
    text_dim: int = 768,
    interp_dim: int = 256,
    seq_len: int = 50,
    batch_size: int = 32,
    d_model: int = 128,
    num_experts: int = 8,
    top_k: int = 2,
    use_mamba: bool = True,
    use_interpretation: bool = True,
    temporal_variant: str = "hybrid",
    channels: bool = False,
    warmup: int = 5,
    iters: int = 50,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TeMoE(
        audio_dim=audio_dim,
        vision_dim=vision_dim,
        text_dim=text_dim,
        interp_dim=interp_dim if use_interpretation else 0,
        channel_dim=interp_dim if (channels and use_interpretation) else 0,
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        use_mamba=use_mamba,
        use_interpretation=use_interpretation,
        temporal_variant=temporal_variant,
        use_channels=channels,
        use_face=True,
        use_bg_gate=True,
        use_consistency=True,
    ).to(device).eval()

    audio = torch.randn(batch_size, seq_len, audio_dim, device=device)
    vision = torch.randn(batch_size, seq_len, vision_dim, device=device)
    text = torch.randn(batch_size, seq_len, text_dim, device=device)
    if channels and use_interpretation:
        face = torch.randn(batch_size, interp_dim, device=device)
        content = torch.randn(batch_size, interp_dim, device=device)
        bg = torch.randn(batch_size, interp_dim, device=device)
        bg_flag = torch.rand(batch_size, 1, device=device)
        call = lambda: model(audio, vision, text, interp_face=face, interp_content=content,
                             interp_bg=bg, interp_bg_flag=bg_flag)
    else:
        interpretation = torch.randn(batch_size, interp_dim, device=device) if use_interpretation else None
        call = lambda: model(audio, vision, text, interpretation)

    with torch.no_grad():
        for _ in range(warmup):
            call()
        _sync(device)
        start = time.perf_counter()
        for _ in range(iters):
            call()
        _sync(device)
        elapsed = time.perf_counter() - start

    latency_ms = elapsed / iters * 1000.0
    throughput = batch_size * iters / elapsed
    if channels and use_interpretation:
        flops = _try_flops_channels(model, audio, vision, text, face, content, bg)
    else:
        flops = _try_flops(model, audio, vision, text, interpretation)
    result = {
        "device": str(device),
        "temporal_backend": model.temporal_backend,
        "use_interpretation": use_interpretation,
        "use_channels": bool(channels),
        "params": sum(p.numel() for p in model.parameters()),
        "params_million": round(sum(p.numel() for p in model.parameters()) / 1e6, 4),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "latency_ms_per_batch": round(latency_ms, 4),
        "throughput_samples_per_s": round(throughput, 2),
        "flops_gflops": flops,
    }
    return result


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _try_flops(model, audio, vision, text, interpretation) -> float | None:
    try:
        from thop import profile as thop_profile  # type: ignore

        inputs = (audio, vision, text, interpretation)
        macs, _ = thop_profile(model, inputs=inputs, verbose=False)
        return round(macs * 2 / 1e9, 4)
    except Exception:
        return None


def _try_flops_channels(model, audio, vision, text, face, content, bg) -> float | None:
    try:
        from thop import profile as thop_profile  # type: ignore

        inputs = (audio, vision, text, face, content, bg)
        macs, _ = thop_profile(model, inputs=inputs, verbose=False)
        return round(macs * 2 / 1e9, 4)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile TeMoE params / latency / throughput / FLOPs.")
    parser.add_argument("--audio-dim", type=int, default=5)
    parser.add_argument("--vision-dim", type=int, default=20)
    parser.add_argument("--text-dim", type=int, default=768)
    parser.add_argument("--interp-dim", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--no-mamba", action="store_true")
    parser.add_argument("--no-interpretation", action="store_true")
    parser.add_argument("--temporal-variant", default="hybrid", choices=["hybrid", "gated_conv"])
    parser.add_argument(
        "--channels",
        action="store_true",
        help="Plan B: profile the structured three-channel (face/content/bg) variant",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = profile_model(
        audio_dim=args.audio_dim,
        vision_dim=args.vision_dim,
        text_dim=args.text_dim,
        interp_dim=args.interp_dim,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_experts=args.num_experts,
        top_k=args.top_k,
        use_mamba=not args.no_mamba,
        use_interpretation=not args.no_interpretation,
        temporal_variant=args.temporal_variant,
        channels=args.channels,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
