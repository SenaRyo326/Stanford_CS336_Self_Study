# Stanford CS336: Language Modeling from Scratch

**Self-study** implementations for [Stanford CS336 (Spring 2026)](https://cs336.stanford.edu/), using the official staff test suite as the quality bar. Many thanks to the CS336 instructors and course staff for open-sourcing such a well-crafted, first-principles curriculum—it is an outstanding resource for learning how language models work under the hood.

Maintained by a Year 1 Electrical Engineering student at The Hong Kong Polytechnic University (not enrolled for credit; no Gradescope or leaderboard submissions). Implementations live in this repository, while large datasets, checkpoints, and official assignment PDFs stay local.

## Progress

| Assignment | Topic | Progress |
| :--- | :--- | :---: |
| Assignment 1 | Basics (Tokenizer, Transformer, Optimizer, Training) | 40% |
| Assignment 2 | Systems (FlashAttention-2, Distributed Training) | 0% |
| Assignment 3 | Scaling (Scaling Laws) | 0% |
| Assignment 4 | Data (Pretraining Data Filtering & Deduplication) | 0% |
| Assignment 5 | Alignment (SFT, RL, DPO) | 0% |

## Log

- **2026-11-16** — Implemented `bpe.py`, `linear.py`, `embedding.py`, `RMSNorm.py`, and `SwiGLU.py` (all unit tests passing). 
- **2026-11-18** — Implemented `RoPE.py`. RoPE encodes positional information into q and k, took lots of time to understand how it works. Method `unsqueeze` starts from -1 lol.