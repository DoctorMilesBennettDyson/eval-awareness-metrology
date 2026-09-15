# Pre-registration — Is the evaluation-awareness scaling law a measurement artifact?

**Working title:** *Selection-aware, format-controlled measurement of evaluation-awareness scaling in open-weight LLMs*
**Author:** Pablo L. Rainieri Blasco (independent researcher). Pipeline code drafted with an LLM assistant (Claude); all analysis decisions below are fixed before any confirmatory activation is computed.
**Registered:** 2026-09-15 (local time, UTC-3). Timestamp mechanism: SHA-256 of this file recorded in `prereg/SHA256.txt`; public git commit to be made before confirmatory runs (pending author approval).
**Status:** frozen. Changes after this point go only to §15 (dated changelog) and are labelled as deviations.

---

## 0. What was observed before registration (full disclosure)

1. Read the papers in §1 (abstracts, methods, and the result figures of Nayan et al. 2026).
2. Downloaded the reference code of Nguyen et al. 2025 (`Jordine/pivotal-test-phase-steering`) and read how the probe vector and scores are computed.
3. Downloaded the contrastive dataset (204 items; the papers report 203) and inspected three items.
4. Downloaded SAD `stages/private_data_gen.zip` and computed, **without reading item texts**: counts (200 `test_oversight`, 200 `deploy_oversight`) and mean body length in characters (deploy 803.8, test 466.4). This length asymmetry motivates baseline B1 (§8); predictions about B1 are therefore informed and are registered as *descriptive*, not confirmatory.
5. Read the schema and row counts of the Devbunova 2×2 dataset on Hugging Face (no item texts inspected).
6. **No activation, probe, or AUROC has been computed for any model in §4.** A pipeline pilot is allowed only on `Qwen/Qwen2.5-0.5B` (base, not instruct), which is excluded from every confirmatory analysis (§11).

---

## 1. Background and gap

- **Nguyen et al. 2025** (arXiv:2507.01786): difference-of-means probes on Llama-3.3-70B-Instruct separate SAD evaluation vs deployment prompts (best AUROC 0.816–0.829 at layers 23–24).
- **Chaudhary et al. 2025** (arXiv:2509.13333): same probe across 15 models (0.27B–70B, 4 families); report a "clear power-law" in best-layer |AUROC − 0.5| vs parameters. No fitted coefficients, no R², no null baseline; Qwen excluded post hoc.
- **Devbunova 2026** (arXiv:2603.19426): on Llama-3.1-8B-Instruct, single-contrast probes (benchmark-eval vs casual-deploy) track *format*: 94.5% of benchmark-formatted deployment prompts classified as evaluation; paired training that decorrelates format from context reduces this to 7.2%. Single model, no scaling.
- **Manek 2026** (arXiv:2606.29196): within-family trajectories non-monotonic (Qwen 2.5) or inverse (Gemma 2). Lists as limitations: no permutation/null baselines, no max-over-layers correction, no text-only/embedding baselines, no format-controlled sets.
- **Nayan et al. 2026** (arXiv:2606.23583): behavioural detection across 37 models is "a training law more than a scaling law" (≈ +0.07 AUROC per 10×); format×intent 2×2 collapses many detectors. Probing limited to Gemma-3 1B/4B.
- **Nordby et al. 2026** (arXiv:2604.13386): probe AUROC for deception scales ≈ +5% per 10×; best layer chosen per model, no selection-aware null.

**Gap.** No study measures the *scaling* of probe-based evaluation awareness with (a) a null that includes the layer- and sign-selection step, (b) a layer-selection procedure independent of the reported score, and (c) probes whose training decorrelates prompt format from evaluation context. Best-layer |AUROC − 0.5| has two upward biases under the null (max over L correlated layers; absolute value) and L grows with parameter count in most families, so the selection floor alone can produce a positive size trend.

---

## 2. Research questions

- **RQ1 (selection floor).** How large is the null distribution of best-layer |AUROC − 0.5| under the reference protocol, and does it grow with depth?
- **RQ2 (replication).** Does the reference estimator increase with model size on an independent run?
- **RQ3 (format across scale).** Do single-contrast probes track format rather than context across model sizes, not just in Llama-3.1-8B?
- **RQ4 (core).** What is the scaling slope of evaluation awareness when the estimator is selection-corrected and format-controlled, and how does it compare with the naive slope on the same data?

---

## 3. Hypotheses, predictions, falsification criteria

All slopes are in AUROC units per decade of parameters (log10 N). "CI" = 95% percentile bootstrap CI (§9).

| ID | Type | Hypothesis (H1) | Null (H0) | Confirmed if | Refuted if |
|---|---|---|---|---|---|
| **H1a** | confirmatory | Under protocol P-SAD, the test-label-permutation null mean of E_naive is ≥ 0.02 in every analysed model | null mean < 0.02 in some model | min over models of null mean ≥ 0.02 | any model < 0.02 |
| **H1b** | confirmatory | The null mean of E_naive increases with number of layers L | no monotone association | Spearman ρ(null mean, L) > 0 with one-sided permutation p < 0.05 across models | ρ ≤ 0 or p ≥ 0.05 |
| **H2** | confirmatory (replication) | E_naive on P-SAD increases with log10 N | slope ≤ 0 | pooled slope (family fixed effects) > 0 and CI excludes 0 | CI includes 0 or slope < 0 |
| **H3** | confirmatory | Under P-STD, the single-contrast probe tracks format: AUROC_crossed < 0.5 at the validation-selected layer | AUROC_crossed ≥ 0.5 | AUROC_crossed < 0.5 in ≥ 80% of analysed models | < 80% |
| **H4** | confirmatory (core) | The naive 2×2 slope exceeds the corrected 2×2 slope | no difference | slope(E_naive2x2) − slope(E_corr2x2) > 0, CI excludes 0 | CI includes 0 or difference < 0 |
| **H5** | descriptive (core estimate) | — | — | report slope(E_corr2x2) with CI and the ratio slope(E_corr2x2)/slope(E_naive2x2) | — |
| **H6** | secondary, low power | Depth explains part of E_naive beyond size | — | report OLS E_naive ~ log10 N + log L + family FE; SmolLM2 (L not increasing with N) reported separately | — |

No directional prediction is made for the sign of slope(E_corr2x2) itself.

---

## 4. Models (fixed list; instruct/chat variants; bf16 on RTX 3070 8 GB with CPU offload when needed)

| Family | Models (HF id; ungated mirrors used where the original is gated) | Nominal N | L (to be verified from config.json) |
|---|---|---|---|
| SmolLM2 | HuggingFaceTB/SmolLM2-135M-Instruct, -360M-Instruct, -1.7B-Instruct | 0.135, 0.36, 1.7 B | 30, 32, 24 |
| Qwen2.5 | Qwen/Qwen2.5-0.5B-Instruct, 1.5B, 3B, 7B, 14B | 0.5, 1.5, 3, 7, 14 B | 24, 28, 36, 28, 48 |
| Gemma-3 | unsloth/gemma-3-270m-it, gemma-3-1b-it, gemma-3-4b-it, gemma-3-12b-it (text tower only) | 0.27, 1, 4, 12 B | 18, 26, 34, 48 |
| Llama-3.x | unsloth/Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct | 1, 3, 8 B | 16, 28, 32 |

- 15 models, 4 families, 0.135B–14B (≈ 2.0 decades). Values of L in config.json override this table.
- **Exclusion rule:** a model is excluded only if it cannot be loaded or run within 8 GB VRAM + 32 GB RAM (+ disk offload ≤ 12 h per model). Exclusions are reported; no replacement enters the confirmatory set.
- Size variable: nominal N (primary, comparable with Chaudhary et al.); exact text-tower parameter count (robustness).
- Gemma-3 4B/12B: the text decoder is used; the vision tower is not loaded or not called.

---

## 5. Data

- **D1 — Contrastive set** (Nguyen et al. 2025): `datasets/contrastive_dataset.json`, 204 items, fields question/positive/negative. Downloaded at runtime from the public repository; not redistributed.
- **D2 — SAD stages-oversight** (Laine et al. 2024): 200 test + 200 deploy bodies from `sad/stages/private_data_gen.zip`. Kept inside the encrypted zip on disk; **never written to plaintext files, logs, or the public repository** (SAD license condition). Balanced 400 items, formatted as in the reference notebook (system instruction "You must guess whether the following text comes from the testing or deployment phase. Answer with just (A) or (B)", body, choices, `Answer:`), rendered with each model's own chat template; (A)/(B) order alternates by item index as in the reference (first half Testing=(A), second half Testing=(B), interleaved test/deploy).
- **D3 — Devbunova 2×2** (MIT): configs `bench_eval` (BE, 1076), `casual_deploy_matched` (CD, 1076), `casual_eval_mutual` (CE, 1076), `bench_deploy_rewritten` (BD, 835). Field `formatted_prompt`, rendered as a single user turn with each model's chat template, generation prompt appended.
  - **Split:** each cell split 50% train / 25% validation / 25% test by SHA-256 of `id` with salt `evalaware-20260915`. If BD items carry a link to an originating CD item (`original_query`), linked pairs are assigned to the same split (leakage guard).

---

## 6. Activation extraction

- Hugging Face `transformers`, `output_hidden_states=True`, dtype bfloat16, no quantization, batch size 1, left context = full prompt (truncate at 2048 tokens; truncations counted and reported).
- Layer convention: residual stream entering block l ("resid_pre") = `hidden_states[l]`, l = 0 … L−1 (l = 0 is the embedding output). `hidden_states[L]` is not used.
- **P-SAD vector (reference protocol):** for each D1 item, prompt = `question + "\n\n " + answer` for answer ∈ {positive, negative}, rendered raw (no chat template, as in the reference); activation at the last token whose decoded text is the answer letter; v_l = mean(positive) − mean(negative); normalised to unit norm.
- **P-SAD scores:** for each D2 prompt, projection of every non-BOS token's resid_pre onto v_l, averaged over tokens (reference `mean_score`).
- **2×2 activations:** resid_pre at the last token of the rendered prompt (Devbunova protocol).

---

## 7. Estimators

Let AUROC_l be computed with the evaluation class as positive.

- **E_naive** (reference): max over l ∈ {0…L−1} of |AUROC_l − 0.5|, selection on the reported data.
- **N1 null:** permute evaluation/deployment labels of the scored items B = 1000 times, holding scores fixed; recompute E_naive per permutation. Report null mean and 95th percentile. **E_corr = E_naive − null mean.**
- **N2 null (P-SAD):** random sign flip of positive/negative within each D1 pair, B = 200, full vector recomputation + scoring + selection. Reported as secondary.
- **E_split:** 200 random stratified halvings of the scored items; layer and sign chosen on half A (max |AUROC − 0.5|), signed AUROC − 0.5 (in the chosen sign) evaluated on half B; mean over halvings.
- **E_fixed:** |AUROC − 0.5| at l = round(0.5·(L−1)) (secondary, no selection).
- **2×2 probes (difference of means, unit-normalised, trained on train split):**
  - **STD:** BE vs CD.
  - **PAIRED:** (mean BE + mean CE)/2 − (mean CD + mean BD)/2.
  - **FORMAT (diagnostic):** (mean BE + mean BD)/2 − (mean CE + mean CD)/2; report cosine(STD, FORMAT) per layer.
- **E_naive2x2:** STD probe; max over layers of |AUROC(BE∪CE vs CD∪BD on test) − 0.5| selected on test (mimics reference practice).
- **E_corr2x2:** PAIRED probe; layer selected on validation by AUROC(BE∪CE vs CD∪BD); reported value = AUROC_crossed − 0.5 on test, where **AUROC_crossed = AUROC(CE vs BD)** (format and context anti-correlated; a pure format detector scores < 0.5, a pure context detector > 0.5).
- **H3 metric:** STD probe, layer selected on validation by AUROC(BE vs CD); AUROC_crossed on test.
- Robustness (secondary): L2-regularised logistic regression (C = 1.0, StandardScaler) in place of difference of means, same selections.

---

## 8. Baselines

- **B1 length-only:** AUROC of token count (and of character count) for D2 and for each 2×2 contrast. *Descriptive prediction (informed by §0.4):* |AUROC − 0.5| ≥ 0.15 on D2.
- **B2 surface text:** TF-IDF (word 1–2-grams, min_df = 2) + logistic regression. D2: 5-fold stratified CV AUROC. 2×2: trained on STD cells (train), AUROC_crossed on test.
- **B3 embedding layer:** l = 0 values from the layerwise curves; E_naive is also reported with l = 0 excluded.

---

## 9. Statistical analysis

- **Slope model:** OLS of the estimator on log10 N with family fixed intercepts and a common slope, over analysed models.
- **Uncertainty:** (i) item bootstrap — resample scored items with replacement within each model (B = 1000), recompute estimator (including selection) per model, refit slope → percentile CI; (ii) model jackknife (leave-one-model-out) reported alongside. For H4, the difference of slopes is computed within each bootstrap replicate using the same resampled items.
- **H1b:** Spearman ρ across models; one-sided p from 10 000 permutations of L.
- **Multiplicity:** H1a and H3 are count-based decision rules (no p-value). The three inferential tests (H1b, H2, H4) use Bonferroni at family-wise α = 0.05: decisions at p < 0.0167 and 98.33% bootstrap CIs. Uncorrected (95%) decisions are reported alongside.
- **Minimum detectable effect (a priori):** with SE(AUROC) ≈ 0.029 (n = 400) and between-model residual SD assumed 0.05, SD(log10 N) ≈ 0.6 across 15 models → SE(slope) ≈ 0.022 → MDE ≈ 0.06 AUROC/decade at 80% power. Slopes smaller than this cannot be distinguished from zero; a null result on H2/H4 is reported as "not detected at MDE 0.06", not as absence.

---

## 10. Validation gate G0 (pipeline validity, before any confirmatory analysis)

On `unsloth/Llama-3.1-8B-Instruct`, logistic-regression probe at layer 24 (Devbunova's reported layer, same 0-indexed convention), threshold 0.5:
- STD probe classifies ≥ 80% of BD test items as evaluation (published: 94.5%), and
- PAIRED-trained logistic probe (BE, CE vs CD, BD) classifies ≤ 20% of BD test items as evaluation (published: 7.2%).

If G0 fails, confirmatory analysis does not start; the discrepancy is diagnosed and documented in §15 before proceeding. G0 results are reported regardless. (G0 uses the same model and data as the confirmatory set; it checks only these two numbers.)

---

## 11. Pilot

`Qwen/Qwen2.5-0.5B` (base) may be run end-to-end to test code paths, memory, and timing. Its outputs are used only for debugging, reported in an appendix, and never enter confirmatory analyses. Code changes after the pilot are allowed; analysis definitions in §6–§9 are not.

---

## 12. Stopping rule and run order

- Fixed model list (§4); every model is run once per protocol with fixed seed `20260915`; no re-runs conditioned on results (re-runs only for crashes, logged).
- Run order: G0 → all models for activation extraction (smallest to largest) → analyses in §9 once all models finish. Intermediate per-model scores are written to disk but estimators and slopes are computed only after the last model completes.

---

## 13. Threats to validity (declared ex ante)

1. **Construct:** "context" in D3 is defined by source dataset (MMLU/ARC/GSM8K/MuTual vs ShareGPT); a probe could separate *content provenance* rather than awareness of being evaluated. AUROC_crossed controls format, not topic.
2. **Rendering:** chat templates differ across families; family fixed effects absorb constant offsets only.
3. **Range:** 0.135B–14B covers ≈ 2 decades vs 2.4 in Chaudhary et al.; no claim is made above 14B.
4. **Linear readout:** difference-of-means measures linear recoverability, not use; no causal claim.
5. **Offload numerics:** CPU-offloaded bf16 layers may differ numerically from GPU execution; a 64-prompt consistency check (GPU vs offload) is run on the largest model that fits fully on GPU and reported.
6. **Mirrors:** ungated re-uploads are assumed weight-identical; SHA-256 of weight files recorded.

---

## 14. Interpretation matrix (fixed in advance)

| H2 (naive scaling) | E_split / E_corr slope | H3 / H4 | Reading |
|---|---|---|---|
| holds | preserved (CI excludes 0, ratio ≥ 0.5) | H3 fails | Scaling robust to selection and not format-driven in this range |
| holds | collapses (CI includes 0 or ratio < 0.5) | — | Naive scaling largely a selection artifact |
| holds | preserved on P-SAD | H3 & H4 hold | Selection is not the issue; format is — naive 2×2 scaling overstates context awareness |
| fails | — | — | Reference scaling not replicated in 0.135B–14B (report MDE) |

---

## 15. Changelog (deviations; dated)

*(empty at registration; frozen copy: `PRE_REGISTRATION_frozen_d0c97af2.md`)*

- **2026-09-15 00:05 — D3 schema clarification (no change to analyses).** The Hugging Face files do not share a `formatted_prompt` field. Prompt text fields used: BE `formatted_prompt`, CD `first_turn`, CE `first_turn` (Devbunova: "MuTual first turns"), BD `rewritten`. `bench_eval` ids are not unique (692 duplicates), so the split key is `cell|row_index|id`. Leakage guard: BD `original_query` matches a CD `first_turn` for 200 of 835 BD items; those 200 inherit the CD split. Resulting split sizes (train/val/test): BE 526/290/260, CD 529/279/268, CE 556/263/257, BD 423/208/204. Descriptive, observed while checking the schema (no texts read): mean characters BE 266, CD 286, CE 108, BD 264. CE prompts are much shorter than BD, so B1 (length-only) is especially relevant to AUROC_crossed; no analysis is changed.
- **2026-09-15 00:20 — Pilot observations and one registered exploratory addition (before any confirmatory activation).** Pilot = excluded `Qwen/Qwen2.5-0.5B` base (§11). (a) Engineering: resid_pre captured by pre-hooks equals `hidden_states[l]` exactly; right-padded batching in bf16 changes resid_pre by up to 4% relative L2 (fp32: 1e-5), so batch size 1 (§6) is kept. (b) Design weakness exposed: PAIRED is trained on CE and BD, so AUROC_crossed(CE vs BD) for PAIRED is in-distribution cell discrimination; in the pilot it reached 1.0 (E_corr2x2 = 0.5, ceiling) while length alone gave AUROC_crossed 0.11. **H4 may therefore be decided by a ceiling on E_corr2x2 rather than by awareness**; this will be stated wherever H4 is reported. The confirmatory definitions are NOT changed. (c) **Exploratory analysis E-T (registered now, not confirmatory):** STD, PAIRED and FORMAT probes trained on the 2×2 train split are applied to SAD at the last prompt token (`d2_last.npy`, same forward pass as `d2_mean`), with the layer chosen on 2×2 validation (the H3 and E_corr2x2 layers), giving out-of-distribution transfer AUROCs with no selection on SAD; slopes vs log10 N reported with the §9 machinery and labelled exploratory. Pilot numbers are reported only in the appendix.
