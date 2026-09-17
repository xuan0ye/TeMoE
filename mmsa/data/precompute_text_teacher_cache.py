"""Build a resumable text-only rationale cache with a small causal LLM.

The prompt and decoding budget deliberately match the Qwen2.5-VL-7B
transcript-only control. Only the generator family/size changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mmsa.data.mmsa_pkl_dataset import read_raw_text
from mmsa.data.precompute_interpretation import _build_encoder, _clean_rationale
from mmsa.data.precompute_interpretation_video import VIDEO_PROMPT


MODE = "text-teacher-rationale-cache"


def prompt_sha256() -> str:
    return hashlib.sha256(VIDEO_PROMPT.encode("utf-8")).hexdigest()


def load_checkpoint(path: Path) -> dict[int, str]:
    rows: dict[int, str] = {}
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        index = int(row["i"])
        rationale = str(row["r"])
        if index in rows:
            raise RuntimeError(f"Duplicate index {index} in {path}:{line_number}")
        rows[index] = rationale
    return rows


def chat_prompt(tokenizer, text: str) -> str:
    content = VIDEO_PROMPT.format(text=(text or "").strip() or "(empty)")
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return content


def load_generator(model_name: str, max_new_tokens: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the controlled text-teacher cache")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda").eval()

    def generate(texts: list[str]) -> tuple[list[str], float]:
        prompts = [chat_prompt(tokenizer, text) for text in texts]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        prefix = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(generated[:, prefix:], skip_special_tokens=True)
        return [_clean_rationale(item) for item in decoded], elapsed

    return generate


def build(
    pkl_path: str,
    output_dir: str,
    model_name: str,
    encoder_name: str,
    max_new_tokens: int,
    batch_size: int,
    max_samples: int | None,
) -> dict:
    import torch

    raw_text = read_raw_text(pkl_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generate = load_generator(model_name, max_new_tokens)
    split_stats: dict[str, dict] = {}

    for split, all_texts in raw_text.items():
        texts = all_texts[:max_samples] if max_samples is not None else all_texts
        checkpoint = output / f"{split}_rationales.jsonl"
        done = load_checkpoint(checkpoint)
        if any(index < 0 or index >= len(texts) for index in done):
            raise RuntimeError(f"Checkpoint indices do not match requested {split} size={len(texts)}")
        rationales = [done.get(index, "") for index in range(len(texts))]
        pending = [index for index in range(len(texts)) if index not in done]
        generated_seconds = 0.0
        started = time.perf_counter()
        with checkpoint.open("a", encoding="utf-8") as stream:
            progress = tqdm(total=len(texts), initial=len(done), desc=f"{split} 1.5B teacher")
            for start in range(0, len(pending), batch_size):
                indices = pending[start : start + batch_size]
                batch, elapsed = generate([texts[index] for index in indices])
                generated_seconds += elapsed
                for index, rationale in zip(indices, batch):
                    if not rationale:
                        raise RuntimeError(f"Empty rationale for {split}[{index}]")
                    rationales[index] = rationale
                    stream.write(json.dumps({"i": index, "r": rationale}, ensure_ascii=False) + "\n")
                stream.flush()
                progress.update(len(indices))
                completed = len(done) + start + len(indices)
                wall = time.perf_counter() - started
                rate = (start + len(indices)) / max(wall, 1e-9)
                remaining = len(pending) - start - len(indices)
                eta = remaining / max(rate, 1e-9)
                print(
                    f"[cache-progress] split={split} completed={completed}/{len(texts)} "
                    f"session_rate={rate:.3f} clip/s eta={eta / 60:.1f} min",
                    flush=True,
                )
            progress.close()

        if not all(rationales):
            raise RuntimeError(f"Incomplete rationale set for {split}")
        split_stats[split] = {
            "count": len(texts),
            "reused": len(done),
            "generated_this_session": len(pending),
            "generation_seconds_this_session": generated_seconds,
            "generation_ms_per_new_clip": (
                1000.0 * generated_seconds / len(pending) if pending else 0.0
            ),
        }

    del generate
    torch.cuda.empty_cache()
    encode = _build_encoder(encoder_name)
    for split, all_texts in raw_text.items():
        texts = all_texts[:max_samples] if max_samples is not None else all_texts
        done = load_checkpoint(output / f"{split}_rationales.jsonl")
        rationales = [done[index] for index in range(len(texts))]
        features = np.asarray(encode(rationales), dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != len(texts) or not np.isfinite(features).all():
            raise RuntimeError(f"Invalid encoded features for {split}: shape={features.shape}")
        np.save(output / f"{split}.npy", features)
        (output / f"{split}_rationales.txt").write_text("\n".join(rationales), encoding="utf-8")
        split_stats[split]["feature_dim"] = int(features.shape[1])

    metadata = {
        "schema_version": 1,
        "mode": MODE,
        "source": pkl_path,
        "generator": model_name,
        "generator_parameters": "1.5B",
        "input_modality": "transcript-only",
        "prompt_template": VIDEO_PROMPT,
        "prompt_sha256": prompt_sha256(),
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "cache_build_batch_size": batch_size,
        "encoder": encoder_name,
        "normalize_embeddings": False,
        "device": "cuda",
        "torch_version": torch.__version__,
        "splits": split_stats,
    }
    (output / "interpretation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--encoder-name", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    build(
        args.pkl,
        args.output_dir,
        args.model_name,
        args.encoder_name,
        args.max_new_tokens,
        args.batch_size,
        args.max_samples,
    )


if __name__ == "__main__":
    main()
