# Candidate Foundation-Model Baselines for the Context-Length Study

Written in response to the supervisor's question: why SleepFM specifically, and
why not compare against recent general-purpose time-series foundation models
(TSFMs, e.g. Chronos) or other pretrained physiological-signal backbones? This
document surveys real, currently-available candidates, with verified repo/
checkpoint links, and a first-pass assessment of how hard each would be to
integrate into our existing pipeline and how each fits the study's two planned
usage modes (frozen embeddings, and LoRA fine-tuning), for both context-length
regimes (native long-context vs. needs-a-sequence-head-on-top).

**This document does not implement anything.** It is a selection aid: read it,
pick 2-3 models, and the next pass will clone those specific repos and write a
detailed, code-level integration plan for each.

**Everything below was verified via web search/fetch against the actual repo,
paper, or model card as of writing (2026-08).** Where something could not be
confirmed from public documentation, it is marked explicitly rather than
guessed — several entries have real gaps (e.g. PhysioOmni's exact modality
list and pretraining corpus are not fully documented in its README/abstract).
Treat unconfirmed items as "verify once we clone the repo," not as settled
facts.

---

## 0. What "integration difficulty" means for our pipeline, concretely

Every candidate below is scored against the same fixed target: our current
preprocessing pipeline produces channel-harmonised, 128 Hz, z-scored signal
split into 4 SleepFM modality groups (BAS = EEG+EOG, RESP, EKG, EMG), stored
as float16 HDF5, and SleepFM patches it into 5-second (640-sample) segments.
Any alternative backbone will differ from this in at least one of: sampling
rate, expected channel names/count, segment/epoch length, and modality
grouping — so *some* adapter code is unavoidable regardless of which model(s)
we pick. The question is how much.

**The one caveat that applies to every general-purpose TSFM on this list, not
just some of them:** "long context" in the TSFM literature means long in
*timesteps*, not long in *wall-clock time at 128 Hz*. Chronos-2's 8,192-token
context, applied to raw 128 Hz signal, covers 64 seconds — not hours. None of
the general TSFMs below can ingest raw hours-long high-frequency PSG directly.
To use one of these as a true "long-context, no sequence head needed" backbone
(the user's Plan A for long-native-context models), the signal has to be
downsampled hard first (e.g. to ~1 Hz derived features) or fed in as our
existing per-patch embeddings treated as a multivariate series (in which case
the "foundation model" is really acting as a pretrained sequence head over
SleepFM features, not a raw-signal encoder — a genuinely interesting
comparison, but a different experiment than "replace SleepFM"). This is
flagged per-model below where it matters most.

---

## 1. Recommended shortlist (read this first)

If picking only 2-3 to start, in priority order:

1. **OSF** — required by your supervisor's framing (only other sleep-PSG FM
   with a public checkpoint), multimodal, but has a real contamination risk
   (see §2.1) that must be handled honestly in the paper regardless of
   whether it "wins."
2. **PhysioOmni** — the strongest genuinely-multimodal *general physiological*
   (not sleep-specific) FM found, explicitly designed for missing-modality
   robustness, which conceptually parallels our own reduced/full-channel
   framing. Weakest point: undocumented respiratory-channel support (a real
   problem for the apnea task specifically) and undocumented pretraining
   corpus — needs verification once cloned.
3. **MOMENT** — the strongest *general-purpose* TSFM for this use case, not
   Chronos. It is classification-native (Chronos/TimesFM/Moirai are
   forecasting-native and would need more adaptation), MIT-licensed, ships 3
   checkpoint sizes, and has a documented multichannel-classification API and
   a PEFT/LoRA ECG tutorial already in its own repo.

If you want a fourth, add **Chronos-2** specifically because your supervisor
named it — even though MOMENT is the better technical fit, having a direct
answer to "did you try Chronos" is likely worth the extra run.

For the single-modality + majority-vote plan (Plan B), **CBraMod** (EEG) and
**ECG-FM** (ECG) are the cleanest picks — both MIT-licensed, both have a
documented frozen-embedding extraction path, both trained on real clinical
data at meaningful scale.

---

## 2. Tier 1 — primary candidates

### 2.1 OSF (On Pre-training and Scaling of Sleep Foundation Models)

- **Paper:** ICML 2026. arXiv:2603.00190.
- **Code:** https://github.com/yang-ai-lab/OSF-Open-Sleep-FM (public)
- **Checkpoint:** `yang-ai-lab/OSF-Base` on HuggingFace (12-channel ViT-Base).
- **License:** MIT.
- **Input format (confirmed from repo):** exactly 12 PSG channels at 64 Hz,
  30-second epochs (1,920 samples/channel), input shape `[B, 12, 1920]`.
  Required channels: `ECG, EMG_Chin, EMG_LLeg, EMG_RLeg, ABD, THX, NP, SN,
  EOG_E1_A2, EOG_E2_A1, EEG_C3_A2, EEG_C4_A1` (NP = nasal pressure/airflow,
  SN = snore).
- **Embedding extraction:** documented — `backbone.forward_encoding(x,
  return_sequence=False)` returns a 768-dim global (`cls`) embedding and
  patch-level embeddings.
- **Fine-tuning:** full fine-tuning script (`main_finetune.py`) is in the
  repo; **no LoRA/PEFT example documented**. The paper's core claim (channel
  masking during pretraining makes the model robust to missing channels at
  inference) is not demonstrated in the top-level README's basic usage
  example, which shows all 12 channels — whether there's a documented
  partial-channel inference path needs checking in the code directly, not
  assumed from the paper's abstract.

**Integration effort: moderate.** Two real mismatches with our pipeline:
(a) 64 Hz / 30-second epochs vs. our 128 Hz / 5-second patches — needs a
resampling + re-chunking adapter, not a simple slice; (b) OSF wants snore
(`SN`) and full thoracic+abdominal+airflow (`THX`,`ABD`,`NP`), which our
**reduced**-channel configuration does not carry (reduced RESP = airflow
only). Our **full**-channel configuration should cover it (RESP priority
list already includes Thor, ABD, SpO2, Snore), so OSF experiments would need
to run against the full-channel preprocessed data, not the fast/reduced
path used for the paper's primary results.

**The contamination caveat (already flagged in
`docs/SOTA_COMPARISON_AND_ABLATIONS.md`, repeating here because it's
critical for this specific use):** OSF's pretraining data includes SHHS and
MrOS, which are two of our four test cohorts. Any AUROC comparison against
SleepFM on SHHS/MrOS-derived test subjects would not be a fair backbone
comparison — OSF has plausibly seen these exact recordings (or close
neighbors from the same cohort) during pretraining. This needs to be stated
explicitly in the paper if OSF is included, and ideally the comparison
should be restricted to, or at least reported separately for, APPLES and
STAGES (cohorts OSF was not pretrained on).

### 2.2 PhysioOmni

- **Paper:** "Towards Robust Multimodal Physiological Foundation Models:
  Handling Arbitrary Missing Modalities." arXiv:2504.19596 (latest v3 dated
  2026-03).
- **Code:** https://github.com/935963004/PhysioOmni (same author group as
  NeuroLM, below; code and weights are released, not just promised).
- **Checkpoint:** referenced as available via a HuggingFace org
  ("Weibang"/related) — exact checkpoint name not resolved from the README
  alone; needs a direct HuggingFace search once we commit to this model.
- **Modalities (confirmed from abstract/search):** EEG, ECG, EOG, EMG.
  **Not confirmed: respiratory/airflow support.** This is a real gap for our
  purposes — our own modality ablation shows apnea detection is the task
  most dependent on the RESP channel group, and if PhysioOmni has no
  respiratory pathway, it cannot be a fair backbone comparison for that task
  specifically without a workaround (e.g. feeding a derived respiratory
  feature stream through whichever generic channel slot it exposes, if any).
- **Architecture:** decoupled multimodal tokenizer, masked pretraining with
  modality-invariant and modality-specific objectives, "resilient
  fine-tuning with prototype alignment" for adapting to incomplete modality
  combinations at downstream time — this last part is exactly the
  missing-modality-robustness property that would make it a natural
  comparison point for our own channel-ablation results.
- **Downstream tasks reported in the paper:** emotion recognition, sleep
  stage classification, motor prediction, mental workload detection. Sleep
  staging specifically being one of their four validated downstream tasks is
  a good sign for relevance, though it does not confirm PSG-scale (full
  night) pretraining data was used — could be epoch-level EEG-sleep data
  from a BCI-style dataset rather than NSRR-style PSG.
- **Pretraining corpus size, exact sampling rate/segment length, and license:
  not documented** in the README or abstract as fetched. **Needs verification
  from the full paper PDF or HuggingFace model card before scoping
  integration effort precisely** — flagging this rather than guessing.

**Integration effort: unknown, likely moderate-to-high until the above gaps
are resolved.** The missing-modality-robust design is architecturally the
best conceptual fit of anything on this list to our own reduced/full-channel
story, which is why it's ranked highly despite the documentation gaps — but
those gaps need to close before committing engineering time.

### 2.3 MOMENT (A Family of Open Time-series Foundation Models)

- **Paper:** ICML 2024.
- **Code:** https://github.com/moment-timeseries-foundation-model/moment
- **Checkpoints:** `AutonLab/MOMENT-1-small`, `-base`, `-large` on
  HuggingFace (three sizes, pick based on compute budget).
- **License:** MIT.
- **Classification support:** native and multichannel — the repo explicitly
  notes multi-channel classification was fixed/supported, with a documented
  API:
  ```python
  from momentfm import MOMENTPipeline
  model = MOMENTPipeline.from_pretrained(
      "AutonLab/MOMENT-1-large",
      model_kwargs={"task_name": "classification", "n_channels": 1, "num_class": 2},
  )
  ```
  (channel count is a constructor argument, so wiring up our 4 modality
  groups or a flattened multichannel input is plausible without deep
  surgery).
- **PEFT/LoRA:** the repo includes an ECG classification tutorial
  specifically demonstrating parameter-efficient fine-tuning — i.e. this is
  a documented, not hypothetical, usage path for exactly the LoRA experiment
  the supervisor asked about.
- **Max context length: not confirmed from available documentation** (the
  paper uses patch-based tokenization; exact max input length needs checking
  in code once cloned). Given MOMENT's general design lineage (patch length
  ~8, comparable to PatchTST-family models), expect a context in the
  low-thousands of timesteps, which — per §0 — is on the order of tens of
  seconds at 128 Hz, not hours, unless fed pre-extracted patch embeddings.

**Integration effort: low-to-moderate.** This is the most "just call the
API" option of the general-purpose TSFMs: pip-installable, HuggingFace
checkpoints, explicit classification + multichannel + PEFT support already
demonstrated by the authors on a biosignal (ECG) task, which is unusually
favorable precedent among general TSFMs.

### 2.4 Chronos-2 (named explicitly by the supervisor)

- **Family:** Amazon Chronos, latest generation.
- **Context length:** 8,192 tokens (confirmed).
- **Design:** forecasting-native (T5-style encoder-decoder over binned
  values), extended in Chronos-2 with group-attention for cross-series
  sharing and in-context learning; supports multivariate input and external
  covariates.
- **Classification usage:** not native — Chronos is built and evaluated as a
  *forecaster*. Using it for classification means taking frozen embeddings
  (there is public precedent for this, e.g. Chronos-2 frozen embeddings used
  as a baseline on UCR classification benchmarks) and training a separate
  classification head on top, i.e. exactly our existing LSTM/Transformer/
  MeanPool head pattern, just swapping the frozen backbone.
- **License and checkpoint:** Amazon Science, `amazon-science/chronos-forecasting`
  on GitHub; check the specific license terms before use in a paper (Amazon
  models have historically used Apache-2.0 for Chronos-1; verify Chronos-2's
  license explicitly when cloning, don't assume it carried over).

**Integration effort: moderate.** Not classification-native, so this
effectively becomes "SleepFM-style frozen embedding + our own sequence head"
regardless of context length, i.e. it cannot deliver the user's Plan A
(native long context, no sequence head) — only Plan B/C style usage. Include
it specifically because it answers the supervisor's named example directly,
not because it's the best technical fit.

---

## 3. Tier 2 — single-modality fallbacks (for the fine-tune-per-modality + majority-vote plan)

These pair naturally with our existing modality groups: EEG models pair with
BAS, ECG models pair with EKG. There is no single-modality RESP or EMG
foundation model of comparable maturity found in this search — for those two
groups, SleepFM (or a from-scratch small encoder) would remain the source of
truth even under the multi-backbone plan.

### 3.1 EEG

| Model | Checkpoint | License | Pretraining scale | Notes |
|---|---|---|---|---|
| **CBraMod** | HuggingFace `weighting666/CBraMod`, GitHub `wjq-learning/cbramod` | MIT | 9,000 hrs, TUEG, 19-channel 10-20 system | ICLR 2025. Documented frozen-embedding extraction (swap output layer for `nn.Identity()`). No LoRA example, but MIT + simple API makes adding one low-effort. **Top EEG pick.** |
| **LaBraM** | GitHub `935963004/LaBraM` (checkpoint `labram-base.pth` linked directly) | check repo | ~2,500 hrs, ~20 datasets, up to 137 channels | First EEG FM (broadly cited baseline in the field); very flexible channel count, useful if we want to feed more than the reduced 4-channel BAS set. |
| **BIOT** | GitHub `ycq091044/BIOT`, HuggingFace `braindecode/BIOT` | check repo | resamples to 200 Hz internally; handles variable channel count/missing values by design | Also integrated into the `braindecode` library, which could simplify plugging into an existing PyTorch pipeline. |
| **NeuroLM** | GitHub `935963004/NeuroLM` | check repo | 25,000 hrs | ICLR 2025. Treats EEG as a "foreign language" fed into an LLM backbone — most architecturally novel, but likely the highest integration effort of the four (LLM tokenization/prompting machinery on top of the signal encoder). |

### 3.2 ECG

| Model | Checkpoint | License | Pretraining scale | Notes |
|---|---|---|---|---|
| **ECG-FM** | HuggingFace (`mimic_iv_ecg_physionet_pretrained.pt`, `mimic_iv_ecg_finetuned.pt`), GitHub `bowang-lab/ECG-FM` | MIT | 1.5M 12-lead ECGs (MIMIC-IV-ECG + PhysioNet 2021) | Built on the `fairseq_signals` framework — powerful but its own ecosystem (not plain PyTorch/HF `transformers`), so expect more plumbing to wire into our HDF5-based pipeline than a native-PyTorch model. Ships an inference quickstart notebook. **Top ECG pick if a clean tutorial matters more than raw scale.** |
| **HuBERT-ECG** | GitHub `Edoar-do/HuBERT-ECG`, HuggingFace `Edoardo-BS/hubert-ecg-base` (+ SSL-pretrained variant) | check repo | 9.1M 12-lead ECGs, 164 cardiovascular conditions | Larger pretraining corpus than ECG-FM; HuBERT-style (masked cluster prediction) architecture, single-lead-benchmark results reported (AUROC 88-92%), suggesting it may tolerate our single-lead EKG input reasonably well without needing all 12 leads. **Top ECG pick if raw scale/robustness to fewer leads matters more.** |

Plan: fine-tune (or frozen-probe) one EEG model on the BAS channel group and
one ECG model on the EKG channel group per task, then combine the two
modality-specific predictions via majority vote or mean-probability
averaging — directly reusing the MP/MV convention already established
in the main study (Supplementary Section on Inference-Time Aggregation)
rather than inventing a new fusion rule.

---

## 4. Considered and deprioritized (with reasons, not silently dropped)

| Model | Why deprioritized |
|---|---|
| **SleepMaMi** | Already documented in `docs/SOTA_COMPARISON_AND_ABLATIONS.md`. No confirmed public code/checkpoint as of that review — cannot be used as a runnable baseline, only cited qualitatively. |
| **SleepFounder** | Same file, same issue: code/checkpoint availability "unclear," medRxiv preprint only. Cardiorespiratory-only by design (no EEG) also makes it a narrower comparison than OSF or PhysioOmni. |
| **Mantis** | Classification-native, lightweight (8M params), real checkpoint (`paris-noah/Mantis-8M`) — genuinely easy to integrate. Deprioritized because it is pretrained **exclusively on synthetic data** via contrastive learning, not on real physiological (or even real-world) signals, so it carries no physiological prior — closer to a strong generic classifier architecture than a "foundation model" in the sense your supervisor's question is really asking about. Worth a footnote mention, not a headline comparison. |
| **TimesFM, Moirai-2** | Same structural issue as Chronos-2 (§2.4): forecasting-native, context length in the low thousands of *tokens*, not classification-native. Chronos-2 is the more prominent/citable representative of this family and is already included; adding both would be redundant unless reviewers specifically ask for it. |
| **PPG foundation model (arXiv:2606.07365)**, **QualityFM** | Both very recent (mid-2026), both use respiratory signal as auxiliary/contrastive supervision (conceptually interesting), but **no confirmed public checkpoint found** for either as of this search. Worth re-checking closer to submission — flagged as a watch-list item, not a current candidate. |

---

## 5. Full comparison table

| Model | Type | Modalities | Checkpoint | License | Native ctx. (raw sec @128Hz equiv.) | Classification-native | LoRA/PEFT documented | Integration effort |
|---|---|---|---|---|---|---|---|---|
| OSF | Sleep-PSG FM | 12-ch PSG (EEG/EOG/EMG/ECG/RESP/snore) | ✓ HF `yang-ai-lab/OSF-Base` | MIT | 30s (per-epoch design; full-night = many epochs) | Epoch-level, not sequence-level | ✗ (full FT only) | Moderate (64Hz/30s vs. our 128Hz/5s; needs full-channel config) |
| PhysioOmni | Multimodal physio FM | EEG/ECG/EOG/EMG (RESP unconfirmed) | Referenced, exact HF repo TBD | Unconfirmed | Unconfirmed | Unconfirmed | Unconfirmed | Unknown pending doc verification |
| MOMENT | General TSFM | Any (channel-count is a constructor arg) | ✓ HF `AutonLab/MOMENT-1-{small,base,large}` | MIT | Low thousands of timesteps (TBD exact) | ✓ | ✓ (own ECG PEFT tutorial) | Low-moderate |
| Chronos-2 | General TSFM (forecaster) | Multivariate (generic) | ✓ `amazon-science/chronos-forecasting` | Verify at clone time | 8,192 tokens | ✗ (embeddings + external head) | Not documented for classification | Moderate |
| CBraMod | EEG FM | EEG (up to 19ch, 10-20) | ✓ HF `weighting666/CBraMod` | MIT | Patch-based; max ctx. TBD | Via frozen-embedding + head | Not documented (easy to add) | Low-moderate |
| LaBraM | EEG FM | EEG (up to 137ch) | ✓ GitHub direct `.pth` | Verify | TBD | Via frozen-embedding + head | Not documented | Moderate |
| BIOT | EEG FM (cross-dataset) | EEG, variable channels | ✓ HF `braindecode/BIOT` | Verify | TBD (200Hz internal resample) | Via frozen-embedding + head | Not documented | Low-moderate (braindecode ecosystem) |
| NeuroLM | EEG-as-language FM | EEG | ✓ GitHub | Verify | TBD | Via LLM prompting | Not documented | High (LLM tokenization layer) |
| ECG-FM | ECG FM | ECG (12-lead trained) | ✓ HF, 2 checkpoints | MIT | TBD | Via frozen-embedding + head | Not documented | Moderate (fairseq_signals ecosystem) |
| HuBERT-ECG | ECG FM | ECG (12-lead trained, single-lead tolerant per benchmarks) | ✓ HF `Edoardo-BS/*` | Verify | TBD | Via frozen-embedding + head | Not documented | Moderate |

"TBD"/"Unconfirmed"/"Verify" entries are exactly that — not yet checked
against the actual code, and should not be treated as known quantities when
scoping the next phase.

---

## 6. Suggested next step

Pick 2-3 from §1's shortlist (or override with your own priorities from the
tables above). For each chosen model, the next pass will: clone the repo,
confirm the exact input/output contract against real code (not just READMEs),
identify the minimal preprocessing adapter needed from our existing HDF5
pipeline, confirm LoRA/PEFT feasibility (via `peft` library compatibility or
a documented native path), and produce a concrete experiment plan mirroring
the existing context-length sweep design (which context lengths are even
reachable per model, given each one's real max sequence length once verified
in code).
