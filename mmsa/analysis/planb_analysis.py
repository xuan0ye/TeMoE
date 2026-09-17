"""Plan B analysis: bg-gate distribution + cross-modal consistency case study.

Run on the server after `run_planb.sh <dataset>` finishes:
    python -m mmsa.analysis.planb_analysis --dataset mosi \
        --cache data/interpretation/mosi_planb \
        --ablation outputs/mmsa/mosi/planb_ablation

Outputs (into --out-dir, default outputs/mmsa/<dataset>/planb_analysis/):
  - bg_gate_report.md    : MLLM bg_important stats + learned gate distribution
                           (from the best full_pb checkpoint if present)
  - consistency_report.md: sim(face,content) bucketed by label sign
                           (irony/misleading detector)
  - case_study.md        : top-20 samples where face and content disagree
                           (sarcasm/misleading candidates) with raw clues
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from mmsa.data.mmsa_pkl_dataset import _load_interpretation, load_mmsa_splits, read_ids
from mmsa.models.builder import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan B: bg-gate + consistency analysis")
    parser.add_argument("--dataset", required=True, choices=["mosi", "sims", "mosei"])
    parser.add_argument("--cache", required=True, help="structured cache dir (e.g. data/interpretation/mosi_planb)")
    parser.add_argument("--ablation", required=True, help="planb_ablation output dir")
    parser.add_argument("--pkl", default=None, help="dataset pkl (defaults to mmsa/configs/<dataset>.yaml data.pkl)")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(open(f"mmsa/configs/{args.dataset}.yaml", encoding="utf-8"))
    pkl_path = args.pkl or cfg["data"]["pkl"]

    out_dir = Path(args.out_dir or f"outputs/mmsa/{args.dataset}/planb_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load structured cache (test split) ----
    _, channels = _load_interpretation(args.cache, "test")
    if not channels:
        raise SystemExit("no structured cache found at %s (run cache generation first)" % args.cache)
    face = channels["face"]
    content = channels["content"]
    bg_flag = channels.get("bg_flag")

    ids = read_ids(pkl_path)["test"]
    with Path(pkl_path).open("rb") as f:
        payload = pickle.load(f, encoding="latin1")
    labels = np.asarray(payload["test"]["regression_labels"], dtype=np.float32).reshape(-1)

    # ---- cosine similarity face vs content ----
    def cos(a, b):
        a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
        b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
        return (a * b).sum(-1)

    sim_tc = cos(face, content)

    # ---- bg gate report ----
    gate_report = []
    if bg_flag is not None:
        n_important = int(bg_flag.sum())
        gate_report.append(
            f"MLLM bg_important=true: {n_important}/{len(bg_flag)} "
            f"({100 * n_important / len(bg_flag):.1f}%)"
        )

    # learned gate values from the best full_pb checkpoint, if present
    ckpt_dir = Path(args.ablation) / "full_pb" / "seed_42"
    ckpt = ckpt_dir / "temoe_model.pt"
    if ckpt.exists():
        try:
            splits = load_mmsa_splits(pkl_path, interpretation_dir=args.cache)
            test = splits["test"]
            model = build_model(
                "temoe",
                test.dims,
                {**cfg.get("model", {}), "use_channels": True, "use_face": True,
                 "use_bg_gate": True, "use_consistency": True},
                use_interpretation=True,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            gates = []
            with torch.no_grad():
                for i in range(len(test)):
                    item = test[i]
                    def _to(x):
                        return None if x is None else x.unsqueeze(0).to(device)
                    out = model(
                        audio=_to(item["audio"]),
                        vision=_to(item["vision"]),
                        text=_to(item["text"]),
                        interp_face=_to(item.get("interp_face")) if "interp_face" in item else None,
                        interp_content=_to(item.get("interp_content")) if "interp_content" in item else None,
                        interp_bg=_to(item.get("interp_bg")) if "interp_bg" in item else None,
                    )
                    if model.last_bg_gate is not None:
                        gates.append(float(model.last_bg_gate.item()))
            gates = np.asarray(gates)
            gate_report.append(f"learned bg gate (full_pb seed_42): mean={gates.mean():.3f} "
                               f"median={np.median(gates):.3f} p90={np.percentile(gates, 90):.3f}")
            if bg_flag is not None:
                hi = gates > np.percentile(gates, 75)
                gate_report.append(
                    f"gate>p75 samples: bg_important rate = {bg_flag[hi].mean():.3f} "
                    f"(vs overall {bg_flag.mean():.3f}) -- does the model learn to open the "
                    f"gate when the MLLM says background matters?"
                )
        except Exception as exc:  # noqa: BLE001 - analysis must not crash the pipeline
            gate_report.append(f"(learned-gate analysis skipped: {exc})")
    else:
        gate_report.append("(no full_pb checkpoint yet at %s -- rerun after training)" % ckpt_dir)

    # ---- consistency vs label sign ----
    sign = np.sign(labels)
    consistency_md = [
        "# Cross-modal consistency (test split)",
        "",
        "sim(face_clue, content_clue) = cosine between the MLLM's facial-expression",
        "description and its event/content description. Low values indicate the face",
        "and the content disagree -- sarcasm/misleading candidates.",
        "",
        "| label sign | n | sim(face,content) |",
        "|---|---|---|",
    ]
    for name, mask in (("negative", sign < 0), ("neutral", sign == 0), ("positive", sign > 0)):
        if mask.sum() == 0:
            continue
        consistency_md.append(
            f"| {name} | {mask.sum()} | {sim_tc[mask].mean():.4f}+-{sim_tc[mask].std():.4f} |"
        )

    # ---- case study: top disagreement ----
    disagreement = np.argsort(sim_tc)[:20]
    structured_path = Path(args.cache) / "test_structured.txt"
    raw_records: list[dict] = []
    if structured_path.exists():
        for line in structured_path.read_text(encoding="utf-8").splitlines():
            try:
                raw_records.append(json.loads(line))
            except Exception:
                raw_records.append({})
    case_lines = ["# Case study: top-20 face-vs-content disagreement (test split)", ""]
    for rank, i in enumerate(disagreement, 1):
        rec = raw_records[i] if i < len(raw_records) else {}
        case_lines.append(f"## {rank}. id={ids[i]} label={labels[i]:+.2f} sim={sim_tc[i]:.3f}")
        case_lines.append(f"- face:    {rec.get('face', '(n/a)')}")
        case_lines.append(f"- content: {rec.get('content', '(n/a)')}")
        case_lines.append(f"- bg:      {rec.get('bg', '(n/a)')} (important={rec.get('bg_important', 'n/a')})")
        case_lines.append("")

    (out_dir / "bg_gate_report.md").write_text("\n".join(gate_report) + "\n", encoding="utf-8")
    (out_dir / "consistency_report.md").write_text("\n".join(consistency_md) + "\n", encoding="utf-8")
    (out_dir / "case_study.md").write_text("\n".join(case_lines), encoding="utf-8")
    print("wrote:", out_dir / "bg_gate_report.md", out_dir / "consistency_report.md", out_dir / "case_study.md")


if __name__ == "__main__":
    main()
