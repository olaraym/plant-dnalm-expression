# AI assistance log

This project used AI coding tools. This file records what they did and what I checked myself,
so a reader can judge the work.

## Tools used

| Tool | Used for |
| --- | --- |
| Claude (Anthropic) | initial scaffold: data loader, k-mer features, ridge head, embedding extraction, scripts, tests, Colab notebook |

## Log

| Date | Task | What the AI produced | What I verified by hand |
| --- | --- | --- | --- |
| 2026-11-19 | Initial scaffold | `src/`, `scripts/`, `tests/` (18 tests, including an end-to-end run on synthetic motif-driven data), Colab notebook | _to fill:_ read every module; confirmed the PGB header layout on the real files; checked k-mer counts on a hand-made sequence; confirmed train/valid/test sizes match the dataset card |
| | | | |

## Rules I follow

- I run and read every AI-generated piece of code before committing it.
- Scientific choices (models, subset sizes, metrics) are mine and are justified in the README.
- Results are checked for consistency: identical splits and head for every feature set; published results quoted as a reference.
