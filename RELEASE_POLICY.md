# Release Policy and Allowlist

The candidate is assembled from an explicit allowlist. Inclusion is based on
reproducibility value, not on directory-wide copying.

## Included

- Python source under `mmsa/`;
- YAML experiment configurations under `mmsa/configs/`;
- selected gated shell entry points in the repository root;
- aggregate JSON/Markdown result summaries listed in `results/RESULTS.md`;
- pinned environment descriptions, documentation, tests, and integrity tools.

## Excluded

- CMU-MOSI, CMU-MOSEI, CH-SIMS, and all other dataset files;
- raw video, audio, images, transcripts, user-level examples, and labels;
- Qwen, SentenceTransformers, and other model weights;
- TeMoE checkpoints and optimizer state;
- per-clip rationale text, prediction JSONL, and embedding caches;
- logs, PID files, temporary files, archives, and historical drafts;
- paper source/PDFs, reviews, handoff notes, and author correspondence;
- absolute filesystem paths, hostnames, usernames, emails, tokens, and keys.

## Mechanical safeguards

`scripts/release_check.py` rejects common data/model/archive extensions, files
larger than 50 MiB, CRLF shell scripts, unexpected files outside the manifest,
and known private path/identity patterns. `MANIFEST.json` records SHA-256 and
byte size for every payload file.
