from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from mmsa.data.mmsa_pkl_dataset import read_raw_text

PROMPT_TEMPLATE = (
    "You are an affective-computing analyst. Read the utterance and explain, in one short "
    "sentence, the speaker's sentiment and the linguistic cues behind it.\n"
    "Utterance: {text}\nSentiment interpretation:"
)

# Lightweight bilingual cue lexicon used only by the deterministic fallback.
_POSITIVE = {
    "good", "great", "love", "happy", "amazing", "excellent", "nice", "best", "enjoy", "wonderful",
    "喜欢", "开心", "很好", "不错", "棒", "满意", "喜爱", "赞",
}
_NEGATIVE = {
    "bad", "hate", "sad", "terrible", "awful", "worst", "angry", "boring", "disappointed", "poor",
    "讨厌", "难过", "糟糕", "失望", "差", "生气", "无聊", "烦",
}


def precompute_interpretation(
    pkl_path: str,
    output_dir: str,
    model_name: str | None,
    encoder_name: str,
    use_llm: bool,
    max_new_tokens: int,
    batch_size: int,
    max_samples: int | None = None,
) -> dict:
    raw_text = read_raw_text(pkl_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if use_llm:
        generate = _build_llm_generator(model_name, max_new_tokens, batch_size)
        mode = f"llm:{model_name}"
    else:
        generate = _template_generator
        mode = "template-fallback"

    encode = _build_encoder(encoder_name)

    summary: dict = {"source": pkl_path, "mode": mode, "encoder": encoder_name, "splits": {}}
    for split, texts in raw_text.items():
        if max_samples is not None:
            texts = texts[:max_samples]
        rationales = generate(texts)
        features = encode(rationales).astype(np.float32)
        np.save(output / f"{split}.npy", features)
        (output / f"{split}_rationales.txt").write_text("\n".join(rationales), encoding="utf-8")
        summary["splits"][split] = {"count": len(texts), "feature_dim": int(features.shape[1])}

    with (output / "interpretation_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def _build_llm_generator(model_name: str | None, max_new_tokens: int, batch_size: int):
    if not model_name:
        raise ValueError("--model-name is required when --use-llm is set")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate(texts: list[str]) -> list[str]:
        rationales: list[str] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            prompts = [_chat_prompt(tokenizer, text) for text in chunk]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for row, prompt_ids in zip(generated, encoded["input_ids"]):
                completion = row[prompt_ids.shape[0]:]
                text = tokenizer.decode(completion, skip_special_tokens=True)
                rationales.append(_clean_rationale(text))
        return rationales

    return generate


def _chat_prompt(tokenizer, text: str) -> str:
    content = PROMPT_TEMPLATE.format(text=text.strip() or "(empty)")
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
    return content


def _template_generator(texts: list[str]) -> list[str]:
    rationales = []
    for text in texts:
        lowered = (text or "").lower()
        pos = sum(1 for word in _POSITIVE if word in lowered)
        neg = sum(1 for word in _NEGATIVE if word in lowered)
        if pos > neg:
            polarity = "positive"
        elif neg > pos:
            polarity = "negative"
        else:
            polarity = "neutral"
        snippet = (text or "").strip()[:80]
        rationales.append(f"The utterance expresses {polarity} sentiment. Cue: {snippet}")
    return rationales


def _clean_rationale(text: str) -> str:
    text = text.strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    # 256 chars silently truncated structured VL JSON (face/content/bg/flag
    # descriptions routinely exceed 256 chars), breaking json.loads and
    # forcing template fallback for ~73% of samples. 2048 is generous for
    # any single rationale or structured JSON while still bounding junk.
    return text[:2048] if text else "neutral sentiment"


def _build_encoder(encoder_name: str):
    if encoder_name == "hashing":
        from sklearn.feature_extraction.text import HashingVectorizer

        vectorizer = HashingVectorizer(
            n_features=256, analyzer="char_wb", ngram_range=(2, 4), alternate_sign=False, norm="l2"
        )

        def encode(texts: list[str]) -> np.ndarray:
            return vectorizer.transform(texts).toarray().astype(np.float32)

        return encode

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(encoder_name)

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray(encoder.encode(texts, batch_size=64, show_progress_bar=False), dtype=np.float32)

    return encode


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute MLLM sentiment interpretation features offline.")
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-llm", action="store_true", help="use a local causal LLM instead of the template fallback")
    parser.add_argument("--model-name", default=None, help="local LLM path, e.g. models/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--encoder-name",
        default="hashing",
        help="'hashing' (no download) or a sentence-transformers model, e.g. paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None, help="debug: cap samples per split")
    args = parser.parse_args()

    summary = precompute_interpretation(
        pkl_path=args.pkl,
        output_dir=args.output_dir,
        model_name=args.model_name,
        encoder_name=args.encoder_name,
        use_llm=args.use_llm,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
