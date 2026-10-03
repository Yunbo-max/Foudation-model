# Foundation Model Tutorial Implementation Plan

> **For agentic workers:** Use the project's independent task boundaries and report verification evidence. User has requested autonomous creation and upload; no additional design approval is required.

**Goal:** Publish an original comprehensive Chinese foundation-model tutorial with working practice code and verified public-source guides.

**Architecture:** Markdown textbook plus small Python package. Independent chapter groups and implementation modules share the interfaces in `docs/design.md`; root integrates, reviews and publishes one coherent snapshot.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, unittest, GitHub Markdown; optional Hugging Face Hub + HTTPX downloader and external Triton experiments.

**Spec:** `docs/design.md`.

## Global Constraints

- Original prose and implementations; source attribution with direct URLs.
- Actual data/benchmark measurements only; no invented results or synthetic training benchmark.
- Chinese explanations begin with the underlying concepts and a concrete scenario.
- No third-party unlicensed-code redistribution.
- Preserve existing GitHub content if it appears before publish; no force push.

## Review Focus

UTF-8 and vocabulary identity; document deduplication before splitting; causal and padding masks; alignment completion masks and group validity; commands/links and accurately limited completion claims.

## Tasks

- [x] Task 1: Chapters 00–05 and assignments 01–02. Explain foundations, architectures, scaling, systems and inference; references to primary docs; no unsupported performance claims.
- [x] Task 2: Chapters 06–10 and assignments 03–04. Explain real data, SFT, preference methods and RLVR; mathematical masking details and real benchmarks.
- [x] Task 3: Chapters 11–16, assignment 05 and official-project source guide. Explain agent RL, memory, continual learning, self-evaluation and safety; cover Nemotron/NeMo, OLMo/OLMo-core, DCLM and Pile.
- [x] Task 4: Tests first, then tokenizer/data/scaling + preparation and download CLIs. Validate Unicode, split leakage, persistence and held-out curve fitting.
- [x] Task 5: Tests first, then model/train/generate and two configs. Validate causal independence, next-token loss, gradients and checkpoint resume.
- [x] Task 6: Tests first, then alignment losses, bounded agent harness and usable mathematical CLI example. Validate masked gradients, DPO direction, GRPO validity and termination.
- [x] Integration: README, source registry, coverage map, dependencies, CI and link checker; review every cross-module interface and relevant command.
- [x] Verification: full tests, CLI help, package compilation, local links; report actual environment limitations in `docs/validation.md`.

Publication uses the reviewed snapshot: initialize the empty remote, publish one complete tutorial commit via the GitHub plugin, and verify remote file hashes and commit. The GitHub branch history records the actual published version.
