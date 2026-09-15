# eval-awareness-metrology

Pre-registered test of whether the reported power-law scaling of probe-based **evaluation awareness**
in open-weight LLMs (Chaudhary et al. 2025, arXiv:2509.13333) survives

1. a null distribution that includes the best-layer and sign selection step,
2. layer selection independent of the reported score, and
3. probes whose training decorrelates prompt format from evaluation context (Devbunova 2026, arXiv:2603.19426).

15 instruct models, 4 families (SmolLM2, Qwen2.5, Gemma-3, Llama-3.x), 0.135B–14B, RTX 3070 8 GB.

- `prereg/PRE_REGISTRATION.md` — hypotheses, estimators, analysis plan, frozen before confirmatory runs (SHA-256 in `prereg/SHA256.txt`; this commit is the public timestamp).
- `code/` — `fetch_data.py`, `data.py`, `extract.py` (resid_pre via forward pre-hooks, bf16, batch size 1), `g0.py` (validation gate), `analyze.py`, `run_all.py`.

Data: SAD stages-oversight texts are kept inside the password-protected zip and never written in plaintext (SAD terms).
The contrastive set is downloaded from its original repository. The 2×2 dataset (MIT) is fetched from the Hugging Face Hub.

Author: Pablo L. Rainieri Blasco (independent). Pipeline code drafted with an LLM assistant; see prereg §0.
License: code MIT.
