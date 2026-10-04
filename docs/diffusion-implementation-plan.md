# Diffusion Tutorial Implementation Plan

> **For agentic workers:** Use task-scoped parallel agents for independent modules, then an independent whole-change review.

**Goal:** Deliver the authorized diffusion chapter with explanations and runnable core algorithms across the requested modalities.

**Architecture:** Shared noise processes are independent of denoiser architecture and data representation. Three connected tutorial files map equations to original CPU implementations, with a separate primary-source guide.

**Tech Stack:** Python 3.10+, PyTorch 2.3+, NumPy, unittest, Markdown.

**Spec:** [diffusion-design.md](diffusion-design.md).

## Global constraints

- Preserve existing 16-week tutorial and code; no new runtime dependencies.
- Continuous array times 0..T-1; D3PM times 1..T; document clean endpoints.
- Small synthetic runs are algorithm demonstrations, never benchmark evidence.
- Use primary papers and author repositories for technical claims.

## Review focus

- Reverse-time integration uses a negative time increment; the ODE score factor is one half.
- DDIM skipping uses the selected previous alpha_bar, with alpha_bar=1 at the clean endpoint.
- Categorical posterior marginalizes normalized posteriors for each predicted clean class.
- Fully masked text cannot access clean targets or condition tokens through an unintended path.
- Image/video/4D layouts preserve geometry and time semantics; shape correctness is not quality evidence.

## Tasks

1. Continuous module (`continuous.py`, `models.py`, `tests/test_diffusion_continuous.py`): test corruption, oracle x0 reconstruction, posterior variance, DDPM/DDIM endpoints, VP score and reverse integration before implementation; run targeted tests.
2. Discrete module (`discrete.py`, `tests/test_diffusion_discrete.py`): test stochastic matrices and posterior against exact enumeration; test masked loss, prompt preservation, all-mask sampling and bidirectional model; run targeted tests.
3. Tutorials and sources: write 17/17a/17b from primary references; connect formulas, scenes, shapes and exact code APIs; independently audit equations and modality boundaries.
4. Demos and integration: implement CLI Gaussian/masked training and multi-modality shape checks with bounded CPU runs; write assignment; add README and reference navigation; preserve existing 71 tests.
5. Verification and publish: run complete unittest suite, alignment checks, Markdown links, compileall and demo commands; record actual measurements and unexecuted work; independent review; update remote main with a fast-forward commit and verify remote content/CI.
