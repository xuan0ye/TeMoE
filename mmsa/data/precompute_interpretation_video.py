from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mmsa.data.mmsa_pkl_dataset import read_ids, read_raw_text
from mmsa.data.precompute_interpretation import (
    _build_encoder,
    _build_llm_generator,
    _clean_rationale,
    _template_generator,
)
from mmsa.data.raw_video_index import build_video_index, resolve_video


class _LazyTextFallbackGenerator:
    """Defers loading the fallback text LLM until it is actually needed.

    Datasets typically have ~100% video->id coverage (see
    mmsa.data.check_video_coverage), so this fallback may never execute in a
    given run. Loading it eagerly at startup means a missing/broken fallback
    model (e.g. not yet downloaded on this node) crashes the entire
    multi-hour video pipeline before it processes a single sample, even when
    the fallback would never have been used. If loading fails even when
    needed, degrade to the template generator rather than crashing the run.
    """

    def __init__(self, model_name: str, max_new_tokens: int) -> None:
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._generate = None
        self._load_failed = False

    def __call__(self, texts: list[str]) -> list[str]:
        if self._generate is None and not self._load_failed:
            print(f"[info] text fallback needed -- lazily loading LLM {self._model_name}")
            try:
                self._generate = _build_llm_generator(self._model_name, self._max_new_tokens, batch_size=16)
            except Exception as exc:  # noqa: BLE001 - network/missing-model errors must not kill the run
                print(f"[warn] failed to load text-LLM fallback ({exc}); using template fallback instead")
                self._load_failed = True
        if self._generate is None:
            return _template_generator(texts)
        return self._generate(texts)

# Modality-control sentinel (--ignore-video): the clip is dropped but the same model
# and the same prompt are used, so "visual grounding matters" is separable from
# "any generative text helps".
NO_VIDEO = "__no_video__"

VIDEO_PROMPT = (
    "You are an affective-computing analyst. Watch this short clip and, in one sentence, "
    "explain the speaker's sentiment using facial expression, tone of voice, and words. "
    "Transcript: {text}\nSentiment interpretation:"
)

# Plan B: structured person-centric clue extraction. The VL model returns a
# JSON object with three clue channels plus a background-importance flag:
#   face    -- the speaker's facial expression / gaze / micro-movements
#   content -- what is actually happening in the clip (actions, events)
#   bg      -- background / environment / other people in the scene
#   bg_important -- whether the background materially affects the sentiment
# Each channel is encoded separately and cached, so the online model can gate
# the background channel and compute cross-modal consistency signals.
STRUCTURED_VIDEO_PROMPT = (
    "You are an affective-computing analyst. Watch this short video clip and extract "
    "three separate clues relevant to the speaker's sentiment.\n"
    "Transcript: {text}\n"
    "Respond with ONLY a JSON object of exactly this form (no other text):\n"
    "{{\n"
    '  "face": "<one short sentence: the speaker\'s facial expression, gaze, or micro-movements>",\n'
    '  "content": "<one short sentence: what is happening in the clip (actions/events)>",\n'
    '  "bg": "<one short sentence: the background, environment, or other people in the scene>",\n'
    '  "bg_important": <true or false: whether the background materially affects the sentiment>\n'
    "}}\n"
    "If a channel is not visible or irrelevant, still give a brief factual description; "
    "set bg_important to false when the background is irrelevant."
)


def _try_parse_structured(payload: str) -> dict | None:
    """Strict JSON parse of the VL structured output; None if not valid JSON.

    Tries the raw payload first, then with code fences stripped. Only returns
    a dict when the model actually produced JSON with at least one usable
    field -- used for the parse-rate statistic (a low parse rate means the
    prompt needs adjustment, not that the fallback silently saved bad data).
    """
    text = (payload or "").strip()
    candidates = [text, re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()]
    # the model sometimes wraps the JSON in prose ("Here is the result: {...}"); slice
    # the outermost {...} span as a last-resort candidate.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and any(str(obj.get(k) or "").strip() for k in ("face", "content", "bg")):
                return obj
        except (ValueError, TypeError):
            continue
    return None


def _parse_structured_rationale(payload: str) -> dict:
    """Parse the VL model's JSON output into a {face, content, bg, bg_important} dict.

    Falls back gracefully: if the payload is not valid JSON (models sometimes
    wrap the object in prose or markdown), strip code fences and retry; as a
    last resort, treat the whole payload as the content channel and provide
    neutral templates for the other channels so the pipeline never crashes.
    """
    text = (payload or "").strip()
    obj = _try_parse_structured(text)
    if obj is not None:
        return {
            "face": str(obj.get("face") or "").strip() or "The speaker's face is not clearly visible.",
            "content": str(obj.get("content") or "").strip() or "No clear event is visible in the clip.",
            "bg": str(obj.get("bg") or "").strip() or "The background is neutral and unremarkable.",
            "bg_important": bool(obj.get("bg_important", False)),
        }
    return {
        "face": "The speaker's facial expression is neutral.",
        "content": text or "No clear event is visible in the clip.",
        "bg": "The background is neutral and unremarkable.",
        "bg_important": False,
    }


def _vl_worker_entry(
    req_queue: "mp.Queue",
    resp_queue: "mp.Queue",
    vl_model: str,
    max_new_tokens: int,
    fps: float,
    max_pixels: int,
    max_frames: int,
    prompt_template: str,
) -> None:
    """Subprocess entry point: loads the VL model once, then serves requests
    from `req_queue` until it receives the `None` sentinel or is killed.

    Runs as an isolated OS process (not a thread) specifically so the parent
    can `kill()` it if a single sample never returns -- see
    `_HardTimeoutVLWorker` for why this replaced a SIGALRM-based timeout.
    """
    try:
        generate = _build_vl_generator(vl_model, max_new_tokens, fps, max_pixels, max_frames, prompt_template)
    except Exception as exc:  # noqa: BLE001 - report back instead of a silent dead process
        resp_queue.put(("fatal", repr(exc)))
        return
    resp_queue.put(("ready", None))
    while True:
        item = req_queue.get()
        if item is None:
            return
        video_path, text = item
        try:
            rationale = generate(video_path, text)
            resp_queue.put(("ok", rationale))
        except Exception as exc:  # noqa: BLE001 - report back; caller restarts the worker regardless
            resp_queue.put(("error", repr(exc)))


class _HardTimeoutVLWorker:
    """Runs VL generation in a persistent, hard-killable subprocess.

    Why this exists: a SIGALRM-based per-sample timeout only interrupts code
    that eventually returns control to the Python interpreter loop or hits a
    syscall that raises EINTR. It does NOT interrupt a call stuck inside a
    single C-extension call (decord's video decode, an OpenMP-parallel
    resize, a CUDA kernel launch that never completes, ...). This was not
    theoretical: on the MOSEI run, a 90s SIGALRM timeout never fired on a
    sample that was still alive and burning ~270% CPU past two minutes.

    IMPORTANT: killing a process that is actively holding a CUDA context
    must go through SIGTERM first, not straight to SIGKILL. SIGKILL bypasses
    CUDA's normal context teardown and can leave the GPU driver in a bad
    state (observed in practice on this exact server: after the first
    SIGKILL, *every* subsequent worker -- fresh process, fresh CUDA context
    -- immediately failed with `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`,
    silently degrading ~99% of a multi-day MOSEI run to text-only fallback).
    SIGTERM lets Python's normal interpreter shutdown/atexit path run and
    release the GPU context cleanly; SIGKILL is now only a last resort if
    the process doesn't exit within a grace period (mirrors the same
    terminate-then-kill escalation `mmsa.data.validate_videos` already uses,
    which never touches CUDA and was therefore never at risk).

    Any exception from a sample (not just a timeout) also now triggers a
    worker restart -- a cuDNN/CUDA error can poison the whole context, so
    the safe assumption is that the worker is no longer trustworthy, not
    that only this one sample was bad.

    The model is loaded once inside the worker and reused across samples
    (reloading it per-sample would be far too slow for a 7B model); a fresh
    worker (paying one model-load cost) is only spawned after an actual
    failure, which should be rare.
    """

    def __init__(
        self,
        vl_model: str,
        max_new_tokens: int,
        fps: float,
        max_pixels: int,
        max_frames: int,
        prompt_template: str = VIDEO_PROMPT,
        startup_timeout_seconds: float = 900.0,
    ) -> None:
        self._args = (vl_model, max_new_tokens, fps, max_pixels, max_frames, prompt_template)
        self._startup_timeout = startup_timeout_seconds
        self._ctx = mp.get_context("spawn")
        self._proc: "mp.Process | None" = None
        self._req_q: "mp.Queue | None" = None
        self._resp_q: "mp.Queue | None" = None

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._req_q = self._ctx.Queue()
        self._resp_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_vl_worker_entry,
            args=(self._req_q, self._resp_q, *self._args),
            daemon=True,
        )
        self._proc.start()
        try:
            status, payload = self._resp_q.get(timeout=self._startup_timeout)
        except queue.Empty:
            self._hard_kill()
            raise RuntimeError(f"VL worker did not finish loading the model within {self._startup_timeout}s")
        if status != "ready":
            self._hard_kill()
            raise RuntimeError(f"VL worker failed to load the model: {payload}")

    def _hard_kill(self, grace_seconds: float = 10.0) -> None:
        """Terminate (SIGTERM) first so CUDA can tear down its context
        cleanly; only escalate to kill (SIGKILL) if it doesn't exit in time.
        Going straight to SIGKILL on a CUDA-active process is what corrupted
        the GPU driver state in practice -- see the class docstring.
        """
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=grace_seconds)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=grace_seconds)
        self._proc = None
        self._req_q = None
        self._resp_q = None

    def generate(self, video_path: str, text: str, timeout_seconds: float) -> tuple[str, str | None]:
        """Returns (status, value) with status in {"ok", "error", "timeout"}.

        Restarts the worker after ANY non-"ok" outcome, not just a timeout:
        a CUDA/cuDNN exception can leave the context in a state where every
        subsequent call fails the same way, so a worker that just errored
        cannot be trusted for the next sample either.
        """
        self._ensure_worker()
        assert self._req_q is not None and self._resp_q is not None
        self._req_q.put((video_path, text))
        try:
            status, payload = self._resp_q.get(timeout=timeout_seconds)
        except queue.Empty:
            self._hard_kill()
            return "timeout", None
        if status == "error":
            self._hard_kill(grace_seconds=5.0)  # worker is still responsive here; a quick graceful restart
            return "error", payload
        return "ok", payload

    def close(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            try:
                assert self._req_q is not None
                self._req_q.put(None)
                self._proc.join(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=10)
        if self._proc is not None and self._proc.is_alive():
            self._proc.kill()
            self._proc.join(timeout=10)
        self._proc = None


def _load_blacklist(path: str | None) -> set[str]:
    if not path:
        return set()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return set(payload.get("bad_paths", payload if isinstance(payload, list) else []))


def precompute_video_interpretation(
    pkl_path: str,
    raw_dir: str,
    output_dir: str,
    vl_model: str,
    encoder_name: str,
    max_new_tokens: int,
    fps: float,
    max_pixels: int,
    max_frames: int,
    text_llm_model: str | None,
    max_samples: int | None = None,
    ignore_video: bool = False,
    ignore_transcript: bool = False,
    blacklist_path: str | None = None,
    sample_timeout_seconds: int = 90,
    structured: bool = False,
) -> dict:
    texts = read_raw_text(pkl_path)
    ids = read_ids(pkl_path)
    index = build_video_index(raw_dir)
    blacklist = _load_blacklist(blacklist_path)
    if blacklist:
        print(f"[info] {len(blacklist)} videos blacklisted; those samples fall back to text.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    prompt = STRUCTURED_VIDEO_PROMPT if structured else VIDEO_PROMPT
    vl_worker = _HardTimeoutVLWorker(vl_model, max_new_tokens, fps, max_pixels, max_frames, prompt_template=prompt)
    hard_timeout_paths: set[str] = set()
    hard_timeout_log = output / "hard_timeout_videos.json"
    if hard_timeout_log.exists():
        try:
            hard_timeout_paths |= set(json.loads(hard_timeout_log.read_text(encoding="utf-8")).get("bad_paths", []))
        except Exception:  # noqa: BLE001
            pass
    if text_llm_model:
        text_generate = _LazyTextFallbackGenerator(text_llm_model, max_new_tokens)
        text_mode = f"llm:{text_llm_model} (lazy)"
    else:
        text_generate = _template_generator
        text_mode = "template-fallback"
    encode = _build_encoder(encoder_name)

    summary: dict = {
        "source": pkl_path,
        "raw_dir": raw_dir,
        "vl_model": vl_model,
        "text_fallback": text_mode,
        "encoder": encoder_name,
        "splits": {},
    }

    for split, split_texts in texts.items():
        split_ids = ids[split]
        if max_samples is not None:
            split_texts = split_texts[:max_samples]
            split_ids = split_ids[:max_samples]

        video_paths = [
            (None if (path := resolve_video(sample_id, index)) in blacklist else path) for sample_id in split_ids
        ]
        if ignore_video:
            video_paths = [NO_VIDEO] * len(split_ids)
        if ignore_transcript:
            split_texts = [""] * len(split_texts)

        # resume from a per-sample checkpoint so a multi-hour run survives crashes
        checkpoint = output / f"{split}_rationales.jsonl"
        done = _load_checkpoint(checkpoint)
        rationales: list[str] = [""] * len(split_texts)
        for i in done:
            if i < len(rationales):
                rationales[i] = done[i]

        # structured mode: also accumulate per-channel texts (face/content/bg)
        structured_records: list[dict] = [{} for _ in split_texts]
        parse_ok = 0
        parse_total = 0
        for i in done:
            if i < len(structured_records):
                structured_records[i] = _parse_structured_rationale(done[i])
                if _try_parse_structured(done[i]) is not None:
                    parse_ok += 1
                parse_total += 1

        matched = sum(1 for path in video_paths if path is not None)
        runtime_fallback_count = 0  # samples that HAD a video but still degraded to text at runtime
        pending = [i for i in range(len(split_texts)) if i not in done]
        with checkpoint.open("a", encoding="utf-8") as ckpt:
            for i in tqdm(pending, desc=f"{split} interp", initial=len(done), total=len(split_texts)):
                if video_paths[i] is not None:
                    status, payload = vl_worker.generate(video_paths[i], split_texts[i], sample_timeout_seconds)
                    if status == "ok":
                        rationale = payload
                    elif status == "timeout":
                        print(
                            f"[warn] sample {split}[{i}] hard-timeout after {sample_timeout_seconds}s "
                            f"(worker killed & will restart); path={video_paths[i]}; using text fallback"
                        )
                        hard_timeout_paths.add(video_paths[i])
                        hard_timeout_log.write_text(
                            json.dumps({"bad_paths": sorted(hard_timeout_paths)}, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        runtime_fallback_count += 1
                        rationale = text_generate([split_texts[i]])[0]
                    else:  # "error" -- worker restarted, safe to continue
                        print(f"[warn] sample {split}[{i}] VL error ({payload}); using text fallback")
                        runtime_fallback_count += 1
                        rationale = text_generate([split_texts[i]])[0]
                else:
                    rationale = text_generate([split_texts[i]])[0]
                rationales[i] = rationale
                if structured:
                    structured_records[i] = _parse_structured_rationale(rationale)
                    parse_total += 1
                    if _try_parse_structured(rationale) is not None:
                        parse_ok += 1
                ckpt.write(json.dumps({"i": i, "r": rationale}, ensure_ascii=False) + "\n")
                ckpt.flush()
                # loud, immediate warning instead of a stat silently buried until the run finishes --
                # this exact blind spot let a GPU-corruption bug degrade ~99% of a run to text-only
                # fallback for two days before anyone noticed
                if runtime_fallback_count and runtime_fallback_count % 200 == 0:
                    print(
                        f"[warn] {split}: {runtime_fallback_count} runtime VL failures so far in this split "
                        "-- if this keeps climbing, something (not just a few bad video files) is wrong; "
                        "check GPU health (nvidia-smi) rather than letting the run continue unattended"
                    )

        features = encode(rationales).astype(np.float32)
        np.save(output / f"{split}.npy", features)
        (output / f"{split}_rationales.txt").write_text("\n".join(rationales), encoding="utf-8")
        channel_dims: dict[str, int] = {}
        if structured:
            # Plan B: save per-channel embeddings + background-importance flag.
            face_texts = [rec.get("face", "") for rec in structured_records]
            content_texts = [rec.get("content", "") for rec in structured_records]
            bg_texts = [rec.get("bg", "") for rec in structured_records]
            bg_flags = np.asarray(
                [1.0 if rec.get("bg_important") else 0.0 for rec in structured_records], dtype=np.float32
            ).reshape(-1, 1)
            face_feats = encode(face_texts).astype(np.float32)
            content_feats = encode(content_texts).astype(np.float32)
            bg_feats = encode(bg_texts).astype(np.float32)
            np.save(output / f"{split}_face.npy", face_feats)
            np.save(output / f"{split}_content.npy", content_feats)
            np.save(output / f"{split}_bg.npy", bg_feats)
            np.save(output / f"{split}_bg_flag.npy", bg_flags)
            (output / f"{split}_structured.txt").write_text(
                "\n".join(
                    json.dumps(rec, ensure_ascii=False) for rec in structured_records
                ),
                encoding="utf-8",
            )
            channel_dims = {
                "face": int(face_feats.shape[1]),
                "content": int(content_feats.shape[1]),
                "bg": int(bg_feats.shape[1]),
                "bg_flag": 1,
            }
            if parse_total:
                parse_rate = round(parse_ok / parse_total, 4)
                print(f"[info] {split}: structured JSON parse rate = {parse_ok}/{parse_total} ({100 * parse_rate:.1f}%)")
                channel_dims["json_parse_rate"] = parse_rate
                if parse_rate < 0.8:
                    print(
                        f"[WARNING] {split}: parse rate {100 * parse_rate:.1f}% < 80% -- the VL model is "
                        "often not returning the requested JSON; check the prompt and sample "
                        "`{split}_rationales.txt` before trusting these features."
                    )
        summary["splits"][split] = {
            "count": len(split_texts),
            "video_matched": matched,
            "video_coverage": round(matched / max(1, len(split_texts)), 4),
            "text_fallback_used": len(split_texts) - matched,
            "runtime_vl_failures": runtime_fallback_count,
            "runtime_vl_failure_rate_of_matched": round(runtime_fallback_count / max(1, matched), 4),
            "feature_dim": int(features.shape[1]),
            **({"channels": channel_dims} if channel_dims else {}),
        }
        print(json.dumps({split: summary["splits"][split]}, ensure_ascii=False))
        if matched and runtime_fallback_count / matched > 0.05:
            print(
                f"[WARNING] {split}: {runtime_fallback_count}/{matched} "
                f"({100 * runtime_fallback_count / matched:.1f}%) of video-matched samples fell back to "
                "text at runtime -- this is far more than a few corrupt files; do NOT trust these "
                "interpretation features until you've checked GPU health (nvidia-smi) and grepped the "
                "log for the actual error, e.g. `grep -c 'VL error' <log>`."
            )

    vl_worker.close()
    summary["hard_timeout_videos"] = sorted(hard_timeout_paths)
    with (output / "interpretation_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    if hard_timeout_paths:
        print(
            f"[info] {len(hard_timeout_paths)} video(s) hard-timed-out this run; "
            f"see {hard_timeout_log} -- merge into your static blacklist for future runs."
        )
    return summary


def _load_checkpoint(path: Path) -> dict[int, str]:
    done: dict[int, str] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                done[int(record["i"])] = str(record["r"])
            except (ValueError, KeyError):
                continue
    return done


def _build_vl_generator(
    vl_model: str,
    max_new_tokens: int,
    fps: float,
    max_pixels: int,
    max_frames: int,
    prompt_template: str = VIDEO_PROMPT,
):
    """Build a (video_path, text) -> str generator around a Qwen2.5-VL model.

    `prompt_template` must accept a single `{text}` placeholder (via
    `str.format`); it defaults to the sentiment-rationale prompt used by the
    offline explanation cache (`precompute_video_interpretation`), but
    `mmsa.data.mllm_zero_shot_baseline` reuses this same loader/OOM-degrade
    machinery with a different prompt that asks for a direct numeric score
    instead of a free-text rationale.
    """
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
    except Exception:  # older transformers
        from transformers import Qwen2VLForConditionalGeneration as VLModel

    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(vl_model, trust_remote_code=True)
    model = VLModel.from_pretrained(
        vl_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    ).eval()

    def _run(video_path: str, text: str, frames_cap: int, pixels_cap: int) -> str:
        content: list[dict] = []
        if video_path != NO_VIDEO:  # --ignore-video keeps the model and prompt, drops the clip
            content.append(
                {
                    "type": "video",
                        "video": video_path,
                        "fps": fps,
                        "min_frames": 2,
                        "max_frames": frames_cap,
                    "max_pixels": pixels_cap,
                }
            )
        content.append({"type": "text", "text": prompt_template.format(text=(text or "").strip() or "(no transcript)")})
        messages = [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        try:
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            trimmed = generated[:, inputs["input_ids"].shape[1]:]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return _clean_rationale(decoded[0] if decoded else "")
        finally:
            del inputs
            if "generated" in dir():
                del generated

    # progressively shrink frames/resolution on OOM instead of killing a multi-hour job
    _DEGRADE_STEPS = [(max_frames, max_pixels), (max(4, max_frames // 2), max_pixels // 2), (2, max_pixels // 4)]

    def generate(video_path: str, text: str) -> str:
        last_error: Exception | None = None
        for frames_cap, pixels_cap in _DEGRADE_STEPS:
            try:
                return _run(video_path, text, frames_cap, pixels_cap)
            except torch.cuda.OutOfMemoryError as error:  # type: ignore[attr-defined]
                last_error = error
                torch.cuda.empty_cache()
                continue
            finally:
                torch.cuda.empty_cache()
        print(f"[warn] VL OOM on all degrade steps for {video_path}: {last_error}")
        return "sentiment unclear (video too large to process)"

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute video multimodal-LLM interpretation features (Path B).")
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--raw-dir", required=True, help="e.g. 'MSA Datasets/MSA-Datasets/CMU-MOSEI/Raw'")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vl-model", required=True, help="local Qwen2.5-VL path, e.g. models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--encoder-name", default="hashing")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-pixels", type=int, default=100352, help="per-frame pixel cap (default ~316x316)")
    parser.add_argument("--max-frames", type=int, default=8, help="hard cap on sampled frames per clip (memory-critical)")
    parser.add_argument("--text-llm-model", default=None, help="fallback text LLM for samples with no video")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ignore-video", action="store_true",
                        help="modality control: send no clip, same prompt and model")
    parser.add_argument("--ignore-transcript", action="store_true",
                        help="modality control: send no transcript, same prompt and model")
    parser.add_argument(
        "--blacklist", default=None, help="json from mmsa.data.validate_videos; blacklisted videos use text fallback"
    )
    parser.add_argument(
        "--sample-timeout-seconds",
        type=int,
        default=90,
        help="real (subprocess-kill) per-sample hard timeout; run validate_videos first as a cheaper "
        "pre-filter, but this is the actual safety net that reliably stops a hung sample",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Plan B: emit structured face/content/bg clue channels + bg-importance flag "
        "(JSON-prompted VL output, encoded per channel into {split}_{face,content,bg}.npy and "
        "{split}_bg_flag.npy)",
    )
    args = parser.parse_args()

    summary = precompute_video_interpretation(
        pkl_path=args.pkl,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        vl_model=args.vl_model,
        encoder_name=args.encoder_name,
        max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else (256 if args.structured else 64),
        fps=args.fps,
        max_pixels=args.max_pixels,
        max_frames=args.max_frames,
        text_llm_model=args.text_llm_model,
        max_samples=args.max_samples,
        ignore_video=args.ignore_video,
        ignore_transcript=args.ignore_transcript,
        blacklist_path=args.blacklist,
        sample_timeout_seconds=args.sample_timeout_seconds,
        structured=args.structured,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
