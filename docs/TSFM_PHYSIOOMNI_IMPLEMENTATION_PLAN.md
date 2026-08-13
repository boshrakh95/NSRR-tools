# PhysioOmni Implementation Plan

> **Purpose**: Authoritative record of what would be built for PhysioOmni as
> TSFM baseline #2 (of 3 — OSF is #1 and is being implemented/run first;
> MOMENT is #3 and comes later), why, and the go/no-go decision behind it.
> **Nothing in this plan has been implemented — this is planning only, per
> explicit instruction.** OSF must finish first (Stage 1 sweep + LoRA +
> results write-up) before any of this is started. Format mirrors
> `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` — read that doc's top section for
> the pattern this one follows (a live checklist plus an Appendix of
> verification detail), adapted here to a pre-implementation, decision-first
> document since no code exists yet.

---

## 0. Should we build this? — decision: **yes, build it, with caveats stated up front**

The user asked explicitly: read what's known about PhysioOmni's pretraining
data/tasks, decide if it's good enough, and if not, stop here and suggest
looking elsewhere instead. This section is that decision, made directly
rather than deferred.

**Verdict: proceed with the plan below.** PhysioOmni is a real, working,
code-and-weights-released multimodal physiological foundation model that
fits our comparison's "how does a pretrained encoder handle a reduced/
missing channel set" question at least as well as OSF conceptually (its
whole paper thesis is "arbitrary missing modality" robustness — the same
shape as our own reduced/full-channel framing). But it is a **visibly
weaker candidate than OSF** on several honest, quantifiable axes, and the
paper's own reported numbers are a genuine reason for modest expectations,
not just risk-management prose:

**Reasons this is weaker than OSF, found by reading the code and paper (not
guessed):**

1. **Never peer-reviewed.** arXiv:2504.19596, v1 April 2025 → v3 March 2026,
   no venue acceptance found anywhere (checked the paper's own submission
   history). OSF is ICML 2026 (accepted, peer-reviewed). PhysioOmni is not
   necessarily wrong for being a preprint, but it means the downstream
   numbers below haven't been through review.
2. **Pretraining corpus is real but small and not sleep-PSG-scale for the
   sleep-relevant slice of it.** Per §1 (paper text, fetched directly):
   TUEG 26,846 recordings (clinical epilepsy-monitoring EEG, not sleep),
   CAP 108 polysomnographic recordings, Sleep-EDF 197 whole-night
   recordings, DEAP 32 participants (emotion, not sleep), a private set of
   54 recordings/19 subjects. **The actual overnight-PSG-relevant slice of
   pretraining data is CAP+Sleep-EDF ≈ 305 recordings** — two to three
   orders of magnitude smaller than our own ~16,000-subject cohort and far
   smaller than OSF's pretraining scale (which already includes SHHS and
   MrOS at NSRR scale). This doesn't disqualify the model, but it means
   "pretrained sleep foundation model" is a much bigger claim for OSF than
   for PhysioOmni, whose pretraining is dominated by non-sleep clinical EEG.
3. **On its own best-fit downstream task (HMC sleep staging), PhysioOmni
   does not clearly beat a non-foundation-model baseline.** Fetched directly
   from the paper: PhysioOmni scores **0.7377±0.0056 balanced accuracy**
   (all 3 modalities) on HMC 5-class sleep staging; the paper's own
   `FeatFusion` baseline (a hand-engineered-feature method, not a
   foundation model) scores **0.7478±0.0038** — higher. This is the single
   most important fact behind this section's honest framing: the paper's
   own numbers, on the task most like ours, do not show the pretrained
   model beating a much simpler baseline. That is a legitimate, informative
   result to include in a baseline-comparison paper (arguably it's exactly
   the kind of finding this paper is designed to surface), but it means
   expectations for PhysioOmni's absolute performance should be modest
   going in — this is not being buried, it's the headline reason this
   section exists.
4. **No RESP/airflow pathway anywhere** — confirmed independently by four
   code locations (§2 below, unchanged from the earlier candidates-doc
   finding). **Apnea is excluded from the PhysioOmni comparison**, same as
   already decided in `CLAUDE.md`.
5. **No LoRA/PEFT code anywhere in the repo** (`grep -rniE "peft|lora"` →
   zero hits) — Stage 2 needs LoRA wiring built from scratch, whereas OSF's
   equivalent work was "find the two Linear layer names and pass them to
   `LoraConfig`." More implementation effort for less validated upside.
6. **Native per-forward-pass context ceiling is shorter and less uniform
   than the original candidates-doc estimate implied** — see §3's
   correction: it is not a flat "512 seconds" across all four modalities;
   it depends on which resample rate is used per modality, and for
   ECG/EMG at the rate PhysioOmni's own downstream scripts use, it's closer
   to 100 seconds. Not a blocker for our 30-second-epoch Plan B design (see
   §5), but a real correction to how "8.5-minute ceiling" should be stated
   if this shows up in the paper's methods section.
7. **License is split and partially undocumented**: the HuggingFace weights
   repo (`Weibang/PhysioOmni`) declares **CC-BY-4.0** explicitly (verified
   directly via the HF API, see §2 — this *corrects* the earlier
   candidates-doc claim of "no LICENSE file anywhere," which was checking
   the GitHub code repo, not the weights repo) — a real, usable, permissive
   license, just attribution-required rather than MIT's no-strings-attached.
   The GitHub code repo itself still has no LICENSE file — a genuine open
   question for how the *code* (not the weights) may be reused/modified,
   worth flagging in the paper if PhysioOmni is included, same spirit as
   OSF's own license-tracking discipline.

**Reasons to proceed anyway:**

- It is still one of very few genuinely multimodal (not single-signal)
  physiological foundation models with both code and weights actually
  released and downloadable (verified: `PhysioOmni.pt` 267.8MB,
  `VQ.pt` 238.4MB, both resolve and download from HuggingFace — see §2).
- The channel mapping (§4) turns out to need **no raw-EDF reprocessing, and
  — better than OSF — no full-channel HDF5 tree at all.** PhysioOmni's
  needed channels (EEG C3/C4, EOG LOC/ROC, ECG, EMG-chin) are already kept
  by the **fast-channel `psg/` tree's own priority-order channel caps**
  (`configs/preprocessing_params.yaml`'s `"fast"` strategy: BAS=4 keeps
  exactly `C3-M2, C4-M1, LOC, ROC`; EKG=1 keeps `EKG`; EMG=2 keeps `CHIN` —
  confirmed directly against `configs/modality_groups.yaml`'s priority-order
  lists and against a real fast-tree HDF5 file's channel keys, §4). This
  means PhysioOmni can be compared against SleepFM's **paper-primary**
  `phase0_v3` numbers, not the secondary `phase0_v3_full` numbers OSF has
  to use — a more directly relevant comparison for the paper's headline
  results. None of the three gaps `docs/OSF_CHANNEL_REPROCESSING_PLAN.md`
  already found and deferred (`ABD`, leg EMG, airflow) matter here either,
  same as before. This makes the implementation cost mostly about
  model/training code, not about a new preprocessing campaign.
- Even a result where PhysioOmni underperforms SleepFM (and possibly even
  underperforms OSF) is legitimate, reportable content for a paper whose
  explicit purpose is "why SleepFM and not some other pretrained encoder" —
  a negative result here is not wasted effort, provided it's measured fairly
  and reported honestly (same standard already applied to OSF's
  contamination caveats).
- This mirrors the selection already made and recorded in `CLAUDE.md` on
  2026-08-05 (PhysioOmni picked as baseline #2 specifically for its
  missing-modality framing) — nothing found in this deeper pass overturns
  that choice, it just sharpens the caveats that should accompany it.

**Practical recommendation for sequencing, not part of the yes/no
decision**: if compute or time becomes the binding constraint once OSF and
PhysioOmni Stage 1 are both running, **PhysioOmni Stage 2 (LoRA) is the
most defensible thing to deprioritize or drop first** among the six
frozen/LoRA × three-model cells in the overall TSFM comparison — it has the
weakest a priori case for showing something new (§0.3 above) and the
highest relative implementation cost (§0.5). This is a sequencing note for
whoever picks up Phase 2, not a reason to skip Phase 1.

---

## 1. Overview

PhysioOmni (arXiv:2504.19596, NTU + SJTU) is a masked-signal-pretrained,
per-modality-tokenized encoder for EEG/EOG/ECG/EMG, explicitly designed for
robustness to arbitrary missing modalities via a "decoupled multimodal
tokenizer" (private + shared VQ codebooks per modality) and a
"resilient fine-tuning with prototype alignment" downstream procedure. Code
and weights are both released (GitHub `935963004/PhysioOmni`, HF
`Weibang/PhysioOmni`). Same overall comparison question as OSF and SleepFM:
does more temporal context help Tier-1 clinical prediction, and how does
PhysioOmni's encoder compare at matched context lengths/subjects/splits —
minus the apnea task, which is architecturally out of scope (§0.4).

**Repos** (read-only reference clones, not modified):
- Local: `/Users/boshra/NSRR-workspace/PhysioOmni`
- Cluster: `/home/boshra95/PhysioOmni` (confirmed present, sibling to
  `NSRR-tools/`, `OSF-Open-Sleep-FM/`, `moment/` — same layout convention
  as `CLAUDE.md`'s "Cluster Execution Guidance" section describes)

Implementation, if/when started, would live in **`NSRR-tools`** (this repo,
presumably on a `physioomni-implementation` branch, mirroring OSF's
`osf-implementation` branch convention) — `PhysioOmni/` itself stays a
read-only reference clone.

**Reference materials**:
- **arXiv:2504.19596** (fetched directly for this plan; not yet saved
  locally as a PDF the way `related_work/OSF.pdf` was for OSF — do that
  first if implementation starts, mirroring the OSF precedent).
- `docs/TSFM_BASELINE_CANDIDATES.md` §2.2 — the original PhysioOmni research
  pass this plan extends and corrects in a few places (§0 above, §3 below).
- `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` — the sibling plan this mirrors in
  structure; its Channel Mapping audit table is reused directly in §4 below
  (same source HDF5s, overlapping channel needs).
- `NSRR-tools/output/channel_analysis/{apples,shhs,mros,stages}_channels.csv`
  — same raw per-subject channel dumps OSF's plan used.
- `CLAUDE.md`'s "TSFM Baseline Model Comparison" section — the three-model
  program, Plan A/B/C definitions, and the frozen/LoRA staged-training
  procedure this plan follows.

## 2. Status (2026-08-13)

**Planning only — nothing built.** This document is the result of a single
research/planning pass: reading the PhysioOmni GitHub repo's code directly
(`dataset.py`, `model/{FT,MSM,VQ,neural_transformer,transformer}.py`,
`prepare_dataset/*.py`, `train_finetune.py`, `train_msm.py`, `train_vq.py`,
`utils.py`), fetching the arXiv paper text and the HuggingFace weights-repo
metadata over the network, and cross-referencing our own `psg/`
(fast-channel) HDF5 channel keys against what PhysioOmni needs
(spot-checked directly against one real subject per cohort — see §4). **OSF must finish first** (per
`CLAUDE.md`'s existing instruction not to start the next model unprompted,
and the user's explicit "we should finish osf first" in this session) —
this plan is not to be acted on until that happens and the user says to
proceed.

**Checkpoint availability, verified over the network (not downloaded
locally)**:
- `https://huggingface.co/Weibang/PhysioOmni` — public, not gated. Two
  files: `PhysioOmni.pt` (267,795,410 bytes ≈ 267.8MB) and `VQ.pt`
  (238,423,693 bytes ≈ 238.4MB), both resolve with HTTP 200 via the
  `resolve/main/` redirect (checked with `curl -IL`, 2026-08-13).
- **License: CC-BY-4.0**, declared directly in the HF repo's model card
  metadata (`cardData: {"license":"cc-by-4.0"}` via the HF API) — this
  corrects `docs/TSFM_BASELINE_CANDIDATES.md`'s "no LICENSE file anywhere"
  finding, which was checking the GitHub *code* repo (still true — no
  LICENSE file there) rather than the HF *weights* repo (does have a
  declared license). Both facts are true simultaneously and should both be
  stated if this ships in the paper: weights are CC-BY-4.0, code license is
  undocumented.
- **`VQ.pt` is very likely not needed for our use case.** Traced
  `train_finetune.py`'s model-loading path end to end: `FT.__init__`
  filters `pretrained_ckpt['model']` for keys starting with
  `EEG_encoder.`/`EOG_encoder.`/`ECG_encoder.`/`EMG_encoder.` and loads only
  those into the four `NeuralTransformer` instances — no reference to VQ
  codebooks anywhere in `FT.py` or `train_finetune.py`. `model/MSM.py`,
  `model/VQ.py`, and `model/FT.py` all name their per-modality encoders
  identically (`self.EEG_encoder = NeuralTransformer(...)`, confirmed by
  grepping all three files), so `PhysioOmni.pt` — whichever pretraining
  stage actually produced it — should strict-or-partial-load into `FT.py`'s
  encoders via this exact mechanism regardless of which stage it came from.
  **This needs to be confirmed by actually downloading and inspecting the
  checkpoint's `state_dict` keys once implementation starts** (same
  "download and strict-load-verify" step OSF did in its own Phase 0/§5) —
  flagged as the very first checklist item below, not assumed.

## 3. Encoder architecture (code-verified, all four `.py` files read directly)

- **Four independent per-modality encoders**, each a `NeuralTransformer`
  (`model/neural_transformer.py`) — **not** one shared backbone. No
  cross-modal attention or fusion exists in the pretrained/frozen weights
  themselves; fusion is a downstream-training-time construct (see below).
- **Per-modality config** (from `train_finetune.py:267-274`, identical in
  `train_msm.py`/`train_vq.py`):

  | Modality | `n_embd` | `patch_size` | `n_layer` | `n_head` |
  |---|---|---|---|---|
  | EEG | 200 | 200 samples | 12 | 10 |
  | EOG | 100 | 100 samples | 12 | 10 |
  | ECG | 100 | 100 samples | 12 | 10 |
  | EMG | 100 | 100 samples | 12 | 10 |

- **Patch embedding**: `TemporalConv` — 3-layer `Conv2d` stack over the
  patch axis (kernel `(1,15)`→`(1,3)`→`(1,3)`, stride `(1,8)` on the first
  layer only) — this is a real temporal feature extractor per patch, not a
  bare linear projection like OSF's `Conv2d` patchify.
- **Position/time embeddings**: `pos_embed = nn.Embedding(256, n_embd)`
  (channel-identity index, looked up by exact string match against a
  132-entry `standard_1020` name list in `dataset.py:8-23`) and
  `time_embed = nn.Embedding(512, n_embd)` (sequential patch-index within
  the modality stream, **not** a real-world-time offset).
- **⚠️ Correction to the earlier candidates-doc estimate — the native
  per-forward-pass ceiling is not a uniform "512 seconds" across
  modalities.** `time_embed`'s cap of 512 indices is a cap on *patch count*,
  not seconds — the real-world duration one 512-patch forward pass can span
  depends on the resample rate used to produce the raw signal fed in, which
  is dataset/script-specific, not architecturally fixed. Confirmed directly
  from the two real prep scripts that write pretraining/downstream data:
  `prepare_dataset/prepare_CAP.py` and `prepare_dataset/prepare_HMC_downstream.py`
  both resample **EEG to 200Hz** (patch=200 samples → 1.0s/patch, so
  512×1.0s ≈ 512s ≈ 8.5min *is* correct for EEG specifically) but
  **EOG to 200Hz** (patch=100 samples → 0.5s/patch, 512×0.5s = 256s ≈
  4.3min) and **ECG/EMG to 500Hz** (patch=100 samples → 0.2s/patch,
  512×0.2s = 102.4s ≈ 1.7min). The "8.5-minute ceiling" from the earlier
  candidates-doc pass is only accurate for the EEG branch; ECG/EMG's real
  ceiling under PhysioOmni's own reference preprocessing is roughly 5× 
  shorter. **Not a blocker for this plan** — our own extraction design
  (§5) uses 30-second epochs, an order of magnitude below even the
  shortest (ECG/EMG) ceiling — but worth getting right if this shows up in
  a methods section, since "512 seconds" as a single flat number is wrong.
- **Attention: bidirectional, non-causal**
  (`scaled_dot_product_attention(..., is_causal=False)`,
  `model/transformer.py:55`) — masked-pretraining style, confirming Plan A
  (native long-context, no sequence head) is unavailable, same conclusion
  as OSF/MOMENT.
- **Pooled output per modality per forward pass**: `forward_features(...,
  return_all_tokens=False)` returns `x[:, 0]` (the CLS token) when
  `use_mean_pooling=False` (`NTConfig`'s default, unchanged in any of the
  three training scripts) — i.e. each modality's `NeuralTransformer`
  produces one `n_embd`-dim CLS vector per forward pass, directly usable as
  a frozen embedding, same shape of decision OSF's CLS output represents.
- **No unified multimodal fusion in the pretrained weights.** `FT.py`'s
  fusion machinery (`EEG_embedding`/`EOG_embedding`/.../`EEG_head`/.../
  `EEG_Linear`/.../`X_transformer`/`alignment_module`/`lm_head`) is
  constructed fresh in `FT.__init__` every time, and `pretrained_ckpt_path`
  loading only ever populates the four `*_encoder` submodules (§2's
  key-prefix trace) — **the released checkpoint is the four per-modality
  tokenizers/encoders only, not a fusion model.** This is a real
  architectural difference from OSF (a single ViT with one CLS output) —
  see §5 for how this shapes our own embedding-extraction design.
- **LoRA target modules, if Stage 2 is built**: `model/transformer.py`'s
  `Attention` class has `self.c_attn` (fused QKV, `nn.Linear(n_embd,
  3*n_embd)`) and `self.c_proj` (output projection, `nn.Linear(n_embd,
  n_embd)`) — the natural PEFT `target_modules=["c_attn", "c_proj"]`
  choice, analogous to OSF's `to_qkv`/`to_out.0`. Applied per encoder (up to
  4 encoders × 12 blocks each = up to 48 attention modules total). No
  existing LoRA code in the repo to reference or copy — this would be
  genuinely new code, unlike OSF where the target-module names could be
  lifted straight from a working `peft.get_peft_model()` pattern already
  implicit in the checkpoint's own naming.

## 4. Channel mapping — the **fast-channel `psg/` tree already suffices**,
no reprocessing and no full-channel tree needed

**PhysioOmni needs a strict subset of the channels OSF already needed**:
EEG (any 10-20 electrode name), EOG (derived from LOC/ROC), ECG (single
lead), EMG (chin). It does **not** need any respiratory/thoracic/abdominal/
snore channel — `apnea` is excluded from the comparison for exactly this
reason (§0.4), so none of the RESP-group channel gaps matter here.

**This is a real difference from OSF, not just a restatement of OSF's own
conclusion.** OSF needed the full-channel `psg_full/` tree specifically
because its 12-channel input includes RESP-group signals (snore, thoracic/
abdominal effort, nasal airflow) that the fast-channel tree's `RESP` cap
(1 channel) doesn't fully carry. PhysioOmni has no RESP input at all, so
that constraint doesn't apply — **the question is whether the
fast-channel tree's much smaller `BAS`/`EKG`/`EMG` caps still keep the
*specific* EEG/EOG/ECG/EMG channels PhysioOmni needs, not whether RESP is
present.** Checked directly, not assumed:

`configs/preprocessing_params.yaml`'s `channel_selection.strategy: "fast"`
(the paper-primary strategy, used for `phase0_v3`) caps `BAS=4, EKG=1,
EMG=2, RESP=1` (8 channels total). Cross-referencing
`configs/modality_groups.yaml`'s exact priority-order lists (the order caps
are applied in):

- `sleepfm_modalities.BAS.priority_order`: `C3-M2, C4-M1, LOC, ROC, O1-M2,
  O2-M1, F3-M2, F4-M1, A1, A2` — **with cap=4, the top 4 kept are exactly
  `C3-M2, C4-M1, LOC, ROC`**, i.e. precisely PhysioOmni's EEG(C3,C4) + the
  two channels the derived EOG(HEO) needs. Nothing PhysioOmni needs from
  this group is cut by the fast cap.
- `sleepfm_modalities.EKG.priority_order`: `EKG, ECG-L` — cap=1 keeps
  `EKG` (PhysioOmni's ECG input), same fallback order already used for OSF.
- `sleepfm_modalities.EMG.priority_order`: `CHIN, LLEG, RLEG, EMG` — cap=2
  keeps `CHIN` plus one leg channel; **`CHIN` (PhysioOmni's only EMG need)
  survives the cap in first position**, the leg channel that also gets kept
  is simply unused by PhysioOmni.

**Spot-checked against a real fast-channel HDF5 file this session**
(`/scratch/boshra95/psg/apples/derived/hdf5_signals/APL1373.h5`): actual
stored keys are `['Airflow', 'C3-M2', 'C4-M1', 'EKG', 'EMG', 'LOC', 'ROC']`
— confirms the priority-order math above exactly (this subject's EMG group
kept the generic `EMG` fallback rather than `CHIN` specifically, since
APPLES doesn't have a literal `CHIN`-named channel — same fallback pattern
already established for OSF, not a new finding).

**Conclusion: use the fast-channel `psg/` tree, not `psg_full/`, as the
source for PhysioOmni's embedding extraction, and compare against
SleepFM's paper-primary `phase0_v3` results, not `phase0_v3_full`.** This
corrects an earlier draft of this plan (and matches independent analysis
from a peer session working the same question) that defaulted to
mirroring OSF's full-channel choice without checking whether it was
actually necessary here — it isn't.

### 4.1 Channel-name vocabulary (real constraint, code-verified)

`dataset.py`'s `standard_1020` list (132 entries, `dataset.py:8-23`) is a
**closed vocabulary** — `get_chans()` does
`standard_1020.index(ch_name)`, which raises `ValueError` on any name not
in the list. Unlike OSF's fixed-but-generic slot names
(`EEG_C3_A2`, `ECG`, etc.), PhysioOmni's channel "names" are used as
*position-embedding lookup keys*, not literal referencing claims — the
model's own downstream prep scripts confirm this directly:
`prepare_dataset/prepare_HMC_downstream.py:116` takes a raw label like
`'EEG C4-M1'` and reduces it to `'C4'` (`name.split(' ')[-1].split('-')[0]`)
purely for the position-embedding lookup, while the *signal itself* stays
whatever it physically was (M1-referenced). **This means our own `C3-M2`/
`C4-M1` HDF5 channels can be fed in with the reference-electrode suffix
stripped for the lookup key (`'C3'`/`'C4'`), exactly mirroring what
PhysioOmni's own reference code does** — not an approximation we're
inventing, the pattern already exists in their prep scripts.

### 4.2 Per-modality mapping plan

| PhysioOmni branch | Our HDF5 source | Mapping notes |
|---|---|---|
| EEG (up to 2 channels) | `C3-M2` → label `'C3'`, `C4-M1` → label `'C4'` | Both fed in the same forward pass — the model natively supports multiple named EEG channels per sample (`CAP`'s own pretraining used up to 16 EEG channels at once, per `prepare_CAP.py`'s `channels` lists) |
| EOG (1 derived channel) | `LOC` − `ROC` → label `'HEO'` | Matches `prepare_HMC_downstream.py:99`'s own derivation exactly (`EOG_signals[[0],:] - EOG_signals[[1],:]`, named `'HEO'`) — not an invented approximation |
| ECG (1 channel) | `EKG` → fallback `ECG-L` → label `'ECG'` | Same fallback order already established for OSF |
| EMG (1 channel) | `CHIN` → fallback generic `EMG` → label `'EMG'` | Same fallback order already established for OSF |

### 4.3 Real per-cohort availability — verified directly against the
fast-channel tree this session (not inferred from OSF's full-channel audit)

Spot-checked one real subject per cohort directly against
`/scratch/boshra95/psg/{dataset}/derived/hdf5_signals/*.h5` (the
fast-channel tree, 2026-08-13):

| Cohort | Real HDF5 keys found | PhysioOmni-relevant channels present |
|---|---|---|
| APPLES (`APL1373.h5`) | `Airflow, C3-M2, C4-M1, EKG, EMG, LOC, ROC` | EEG ✅, EOG ✅, ECG ✅, EMG ✅ (generic fallback) |
| SHHS (`203805_v2.h5`) | `Airflow, EEG, EKG, EMG, LOC, ROC` | EEG ❌ (generic only, no C3/C4), EOG ✅, ECG ✅, EMG ✅ (generic fallback) |
| MrOS (`AA1449_v2.h5`) | `Airflow, C3-M2, C4-M1, CHIN, EKG, LLEG, LOC, ROC` | EEG ✅, EOG ✅, ECG ✅, EMG ✅ (real `CHIN`) |
| STAGES (`STNF00032.h5`) | `Airflow, C3-M2, C4-M1, CHIN, EKG, LOC, ROC` | EEG ✅, EOG ✅, ECG ✅, EMG ✅ (real `CHIN`) |

This directly confirms §4's priority-order math against real extracted
data, not just the config file's stated caps, and matches
`docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s own (larger, 50-subject-per-cohort)
full-channel-tree audit pattern exactly — same structural SHHS gap, same
fallback behavior for APPLES/SHHS's generic `EMG`. Since the underlying
raw-channel-name alias resolution (`channel_definitions.yaml`) is identical
between the fast and full trees — only the *how many channels survive the
per-modality cap* differs, and §4 already showed the cap doesn't cut
anything PhysioOmni needs — OSF's larger 50-subject-per-cohort percentages
transfer directly as the best available completeness estimate:

| Channel | APPLES | SHHS | MrOS | STAGES |
|---|---|---|---|---|
| `C3-M2`/`C4-M1` (EEG) | 100% | **0%** | 100% | 100% |
| `LOC`/`ROC` (EOG, both needed for the derived HEO) | 100% | 100% | 100% | 100% |
| `EKG`→`ECG-L` (ECG) | 100% | 100% | 100% | 90% |
| `CHIN`→generic `EMG` (EMG) | 100%† | 100%† | 100% | 100% |

†generic `EMG` fallback for APPLES/SHHS, real `CHIN` channel for MrOS/STAGES
(exactly matching this session's real per-cohort spot-check above).

**The one real gap: SHHS has no distinguishable C3/C4 EEG**, identical root
cause to OSF's own SHHS finding, confirmed on the fast tree independently
this session (SHHS's fast-tree file has only a generic `EEG` key). **Decision,
mirroring OSF's own resolved SHHS decision**: duplicate SHHS's single
generic `EEG` channel into both `'C3'` and `'C4'` position-lookup slots
(same approximation, same justification — no distinguishable-electrode data
exists in our SHHS HDF5s to do better without reprocessing). This should be
revisited together with OSF's own SHHS caveat if results look degraded, not
decided independently per model.

### 4.4 Reprocessing decision: **no raw EDF reprocessing needed, and no
full-channel tree needed either**

Every channel PhysioOmni needs is already present in the **existing
fast-channel `psg/` HDF5s** — the same tree SleepFM's paper-primary
`phase0_v3` results already use — except the one structural SHHS EEG gap
above (which reprocessing can't fix either — SHHS's source recordings
genuinely don't preserve a C3/C4 distinction in this repo's current
extraction, an upstream question separate from re-running the same
extraction logic, and separate from which channel-cap strategy is used).

**Importantly, `docs/OSF_CHANNEL_REPROCESSING_PLAN.md`'s three already-
identified, already-deferred gaps (MrOS `ABD`, STAGES leg EMG `LAT`/`RAT`,
SHHS `NEW AIR`/airflow) do not apply to PhysioOmni at all** — none of those
three channels (`ABD`, leg EMG, airflow) are in PhysioOmni's input set. If
that reprocessing pass is ever run for OSF's benefit, it would not change
anything for PhysioOmni's channel completeness. **No new reprocessing plan
doc is needed for PhysioOmni** — this section is the complete answer.

### 4.5 Sample-rate handling — resample per PhysioOmni's own native rates,
not our own convenience rate

Our HDF5s (both the fast and full-channel trees — confirmed identical
source rate in the fast-tree file's own `sampling_rate` attribute, §5) are
uniformly 128Hz. PhysioOmni's reference prep scripts
(`prepare_CAP.py`, `prepare_HMC_downstream.py`, `prepare_tuh.py`, all
read directly) consistently resample to **200Hz for EEG and EOG, 500Hz for
ECG and EMG** before chunking into patches. **Recommendation: replicate
these exact per-modality resample targets** (128→200Hz for EEG/EOG,
128→500Hz for ECG/EMG — all upsampling, since our source is 128Hz and every
PhysioOmni-native rate is higher) rather than feeding 128Hz signal directly
into 200/100-sample patches, which would silently shift the real-world time
each patch represents away from what the frozen encoder was pretrained on
(§3's ceiling-correction section already shows this ratio matters). This is
the same "match the reference pipeline's resample choice exactly" principle
already applied to OSF (128→64Hz linear interpolation, chosen specifically
to match OSF's own reference resampling). Use `scipy.signal.resample` or
linear interpolation (either is a defensible choice for upsampling by a
non-integer ratio like 128→200; note this ratio, unlike OSF's clean 2:1
128→64Hz decimation, has no simple exact-decimation shortcut — pick
whichever `mne`/`scipy` resampling function is used elsewhere in this
codebase's preprocessing for consistency, verify empirically in the Step 0
pilot per §7 item 4).

## 5. Embedding design — Plan B, similar shape to OSF but no shared fusion
to reuse

Because the released checkpoint has **no unified fusion module** (§3), our
own frozen embedding is naturally "the four per-modality CLS/pooled outputs,
concatenated" — there is no PhysioOmni-native single-vector-per-epoch output
to just save, unlike OSF's single ViT with one CLS token.

**Per-epoch embedding definition** (mirrors OSF's "one 30-second epoch, one
forward pass per available branch" pattern):

1. For each 30-second epoch, and for each of the up to 4 modality branches
   with data available for that subject: resample to the branch's native
   rate (§4.5), chunk into 1-second EEG / 0.5-second EOG / 0.2-second
   ECG-and-EMG patches (§3's table), run through that modality's
   `NeuralTransformer.forward_features(..., return_all_tokens=False)` to
   get one `n_embd`-dim CLS vector (200-dim for EEG, 100-dim for
   EOG/ECG/EMG).
2. **Concatenate the four vectors into one flat 500-dim vector per epoch**
   (200 EEG + 100 EOG + 100 ECG + 100 EMG), **zero-filling any entirely
   absent modality's 100/200-dim slice** (not per-patch — per whole
   modality, since a modality is either present or absent for a whole
   subject in our channel-availability picture, §4.3). This is simpler than
   OSF's `[T, 2, 768]` two-subtoken design (no shared dimension across
   sub-tokens to preserve) — a flat `[T_epochs, 500]` array per subject is
   sufficient; there's no need to fork `ContextWindowDataset`'s
   `N_MODALITIES`/`EMBED_DIM` split into a 3D shape at all. **Recommended
   dataset-class constants** (forking `ContextWindowDataset` the same way
   OSF's plan did): `N_SUBTOKENS = 1` (not 4 or 2 — the four modalities are
   already concatenated into one flat vector before saving, so from the
   dataset class's perspective there is exactly one "sub-token" per epoch),
   `EMBED_DIM = 500`, `FLAT_DIM = 500`, `PATCH_SECONDS = 30`,
   `PATCHES_PER_EPOCH = 1` — genuinely simpler to fork than OSF's version
   precisely because there's no per-slot dimension mismatch to work around.
3. **Log which modality was zero-filled per subject** (same
   `_channel_fill_log.jsonl` convention as OSF's extraction script) — the
   SHHS EEG-duplication case (§4.3) and any true per-subject absence both
   need to be visible in this log before trusting results.

**Output**: `{output_dir}/{dataset}/{subject_id}.npy`, dtype float16, shape
`[T_epochs, 500]`. Suggested output dir, mirroring OSF's `osf_30sec`
naming convention but under the **fast-channel** unified tree (§4's
correction — PhysioOmni's source HDF5s are `psg/`, not `psg_full/`):
`/scratch/boshra95/psg/unified/embeddings/physioomni_30sec/`. Results dir,
mirroring `phase0_v3`'s own location under the same tree:
`/scratch/boshra95/psg/unified/results/phase0_physioomni/`.

**Context-length → epoch-count mapping**: identical to OSF's, since both
use 30-second epochs (`30s`→1, `10m`→20, `40m`→80, `80m`→160, `120m`→240,
`240m`→480) — the exact same table already in
`docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` §3.2 applies verbatim, no
recomputation needed.

**Normalization — concrete method identified, still needs empirical
validation before trusting extraction output.** PhysioOmni's `dataset.py`
divides raw signal by 100 (`EEG/100`, `EOG/100`, etc., not z-score) before
feeding the model — **different from OSF's expectation of already-z-scored
input.** Our `signal_processor.py` z-scores per-channel during EDF→HDF5
preprocessing, but **every HDF5 file stores the exact per-channel
pre-normalization statistics needed to invert this**, confirmed by reading
a real file's root attributes directly this session
(`/scratch/boshra95/psg/apples/derived/hdf5_signals/APL1373.h5`): a
`normalization_stats` JSON attribute with `{mean, std, min, max}` per
channel, e.g. `"C3-M2": {"mean": -0.00155, "std": 15.25, "min": -221.6,
"max": 219.9}`. **Recommended approach**: `x_original = x_zscored * std +
mean` per channel (using each channel's own stored stats), then scale to
match whatever raw unit PhysioOmni's own prep scripts assume, then divide
by 100.

**One real wrinkle found while reading these stats, worth flagging
explicitly**: the stored pre-normalization scales are **not uniform across
channels** — `C3-M2`/`C4-M1`/`EKG`/`EMG`/`Airflow` have `std` in the
~15-25 range (consistent with already-µV-scale raw amplitudes), but
`LOC`/`ROC` have `std` around `1.6e-5` and min/max around `±0.0002`
(consistent with raw **volts**, not µV — roughly 4-5 orders of magnitude
smaller). **A single flat "convert V→µV" rule applied uniformly would be
wrong** — the unit-inversion step needs to check each channel's actual
recovered scale (or track it explicitly per channel from the original EDF
units) rather than assuming every channel was in the same unit before
z-scoring. This is a real, previously-undocumented preprocessing detail
worth confirming with a source-code read of `signal_processor.py`'s unit
handling before writing the extraction script, not just inferred from the
stored stats' magnitudes alone.

**Still flagged as a Step 0 pilot check (§7 item 0.5)**, not assumed to
work correctly on the first attempt: after inverting to raw scale and
applying the `/100` convention, run a handful of real epochs through the
frozen encoder and check for NaNs/degenerate (near-zero or saturated) CLS
outputs before trusting any extraction at scale — same "verify
empirically, don't just assume normalization compatibility" discipline
OSF's own plan already applied (and confirmed correct, for OSF's simpler
case) before trusting its extraction pipeline.

## 6. Stage 2 (LoRA) — architecturally different from OSF, more new code

Same staged LP-FT procedure as OSF (`CLAUDE.md`'s "Frozen vs. LoRA-fine-
tuned conditions" section applies unchanged: Stage 1 frozen-embedding head
training first, then Stage 2 wraps the warmed model with LoRA and continues
training LoRA+head together) — but Stage 2's *mechanics* differ more from
OSF's here than they did for OSF-vs-SleepFM, because of §3's "no unified
fusion in the pretrained weights" finding:

- **What gets LoRA-wrapped**: up to 4 separate `NeuralTransformer`
  instances (`target_modules=["c_attn", "c_proj"]`, §3), each wrapped
  independently or all together in one `get_peft_model()` call over a
  parent module holding all four — needs a real design decision when this
  phase starts (not resolved here, flagged as an open item in §8).
- **What sits on top**: since there's no pretrained fusion module to warm-
  start from (unlike OSF, which had nothing extra to build at all — a
  single ViT feeds directly into our sequence head), Stage 2 needs **our
  own fusion step** (the simple concatenation from §5, or a small learned
  fusion layer) between the four per-modality LoRA-adapted encoders and our
  sequence head — this is genuinely new code, not a fork of anything
  PhysioOmni ships, and not something OSF's `train_osf_lora.py` design
  (§6.1 of the OSF plan) has an equivalent for.
- **Memory/compute profile**: up to 4× the per-step forward/backward cost
  of a single-encoder LoRA setup (OSF has 1 ViT in the trainable graph per
  step; PhysioOmni could have up to 4 `NeuralTransformer`s, each smaller
  individually — net compute unclear without a pilot). Apply the same
  memory-mitigation ladder already agreed for OSF (`CLAUDE.md` §0.1:
  gradient checkpointing → larger GPU allocation → cap max context length
  as a last resort, with the frozen condition still reported at all context
  lengths) — no reason to design a different ladder for this model.

**Wall-time/compute budget: unknown, same caveat as OSF's own Stage 2
section** — run a short pilot before committing to a full sweep's `--time`
budget, don't extrapolate from OSF's or SleepFM's numbers (different
architecture, different per-step cost shape entirely, given the up-to-4x
multi-encoder structure above).

## 7. Implementation checklist (not started — sequencing only, for when
OSF is done and the user says to proceed)

**Do not start any of this until OSF's Stage 1 + Stage 2 + results
write-up are done and the user explicitly asks for PhysioOmni to begin** —
per `CLAUDE.md`'s standing instruction and this session's explicit "we
should finish osf first."

### Phase 0 — Setup
- [ ] 0.1 Build `physioomni_env` (fresh venv, mirroring `osf_env`'s
      rationale — do not reuse `sleepfm_env`/`osf_env`, PhysioOmni's own
      README pins `torch==2.5.1`/`torchvision==0.20.1`/`torchaudio==2.5.1`
      via conda, which may or may not conflict with `osf_env`'s exact pins;
      check before assuming reuse is safe)
- [ ] 0.2 Download `PhysioOmni.pt` from `https://huggingface.co/Weibang/PhysioOmni`
      and **strict-or-partial-load-verify** it against `FT.py`'s four
      `NeuralTransformer` encoders (§2's key-prefix hypothesis needs
      confirming against the real file, exactly like OSF's checkpoint-
      filename resolution in its own §5) — do **not** download `VQ.pt`
      unless the load-verification in this step reveals it's actually
      needed (§2 already traces the code path showing it shouldn't be)
- [ ] 0.3 Save the paper PDF locally (mirrors `related_work/OSF.pdf`)
- [ ] 0.4 Confirm the SHHS EEG-duplication decision (§4.3) with the user
      before writing extraction code (same checkpoint OSF's plan required)
- [ ] 0.5 Empirically validate the normalization approach (§5 — invert
      each channel's stored `normalization_stats`, correcting for the
      per-channel unit inconsistency found there, then `/100`) on a
      handful of real epochs before trusting it at scale
- [ ] 0.6 Create the implementation branch (mirrors `osf-implementation`)

### Phase 1 — Stage 1 (frozen encoders)
- [ ] 1.1 Implement `scripts/extract_physioomni_embeddings.py` +
      `configs/phase0_physioomni_config.yaml` (§5's per-epoch, per-modality
      forward-pass + concatenate-and-zero-fill design)
- [ ] 1.2 Smoke-test on real APPLES + SHHS subjects (small `--limit`, CPU),
      verify no NaNs, verify SHHS's EEG-duplication and any true zero-fills
      match §4.3's expected pattern — **user checkpoint** before continuing
- [ ] 1.3 Implement `src/nsrr_tools/datasets/physioomni_context_window_dataset.py`
      (§5's simplified constants: `N_SUBTOKENS=1, EMBED_DIM=500,
      FLAT_DIM=500, PATCH_SECONDS=30, PATCHES_PER_EPOCH=1`) — fork
      `ContextWindowDataset` the same way OSF's plan did, following the
      same "recompute `min_recording_patches` in epoch units, not
      5-second-patch units" discipline (`480`, not `2880`, same value OSF
      already uses since both are 30s-epoch-based)
- [ ] 1.4 Smoke-test the dataset class at 30s/10m/full_night contexts —
      **user checkpoint**
- [ ] 1.5 Implement `scripts/train_physioomni_context_sweep.py` (fork of
      `train_context_sweep.py`, same pattern as
      `train_osf_context_sweep.py` — drop `--zero-modalities`, no
      modality-group ablation concept applies here either) + its job script
      — **user checkpoint**
- [ ] 1.6 Implement `scripts/infer_physioomni_subject_windows.py` + job
      script (fork of `infer_subject_windows.py`, recalibrate the
      batch-size auto-scaling reference point the same way OSF's plan did
      — `_ref_N=480` for PhysioOmni too, same 30s-epoch units) —
      **user checkpoint**
- [ ] 1.7 Implement `experiments/v2_physioomni_registry.yaml` +
      `scripts/gen_commands_physioomni.py` (same 4-task-not-5 scope — Tier-1
      minus `apnea_binary`: `sex_binary`, `sleep_efficiency_binary`,
      `bmi_binary`, `age_class` — × 3 heads, following the exact
      `gen_commands_osf.py` structural precedent, same deliberate exclusion
      of figure/table subcommands per `CLAUDE.md`'s reuse-assessment
      reasoning)
- [ ] 1.8 Implement `jobs/extract_physioomni_embeddings_gpu.sh`, test via a
      small real GPU allocation before trusting it for the full run
- [ ] 1.9 Run full embedding extraction, all 4 datasets
- [ ] 1.10 Run the Stage 1 sweep (4 tasks × 3 heads × 6 contexts = up to 72
      training runs), then inference, then analysis
- [ ] 1.11 Re-run the channel-completeness audit against real extraction
      output, confirm it matches §4.3's expectations

### Phase 2 — Stage 2 (LoRA)
- [ ] 2.1 Resolve the open multi-encoder LoRA-wrapping design question
      (§6, "wrap each `NeuralTransformer` separately or all four together")
      with the user before writing code
- [ ] 2.2 Implement `scripts/train_physioomni_lora.py` (§6 — genuinely new
      code, more so than OSF's equivalent, since a fusion step has to be
      built here that OSF didn't need)
- [ ] 2.3 Short wall-time pilot at the smallest context length —
      **user checkpoint** before the full sweep
- [ ] 2.4 Run the full Stage 2 sweep, applying the same memory-mitigation
      ladder as OSF's Stage 2 (§6)

### Phase 3 — Results
- [ ] 3.1 Compile Stage 1 + Stage 2 results against `phase0_v3`
      (paper-primary — §4's correction, not `phase0_v3_full`), stating §0's
      caveats plainly (small sleep-specific pretraining scale, arXiv-only/
      unreviewed, own-paper HMC numbers below a non-FM baseline, no apnea)
      — same "no silently-incomplete cells" discipline as OSF
- [ ] 3.2 Report back before starting MOMENT — do not start the next
      model's plan unprompted

---

## 8. Key Decisions (recorded here for a future session, not yet all
finalized — several need a user checkpoint before Phase 1 starts)

| Decision | Choice | Rationale |
|---|---|---|
| Go/no-go | **Go**, with caveats stated in §0 | Weaker case than OSF on several honest axes, but real released weights + code, no blocking technical gap, and a mixed/negative result is still informative for a baseline-comparison paper |
| Task scope | 4 tasks: `sex_binary`, `sleep_efficiency_binary`, `bmi_binary`, `age_class` — **not** `apnea_binary` | No RESP/airflow pathway exists anywhere in the model (§3/§0.4), confirmed by 4 independent code locations, unchanged from the original candidates-doc finding |
| Source HDF5 tree | **Fast-channel `psg/`** (paper-primary), **not** full-channel `psg_full/` | §4: the fast strategy's own priority-order caps (BAS=4, EKG=1, EMG=2) already keep `C3-M2, C4-M1, LOC, ROC, EKG, CHIN` — everything PhysioOmni needs, since it has no RESP input to require the full-channel tree the way OSF does. Confirmed both from config priority-order math and real HDF5 key listings for all 4 cohorts, not assumed by analogy to OSF |
| Comparison baseline | `phase0_v3` (paper-primary SleepFM) | Follows directly from the source-tree decision above — a more directly relevant comparison than OSF's `phase0_v3_full` since it's the paper's own headline numbers |
| Embedding storage | Concatenated per-modality CLS vectors, `[T, 500]` per epoch, flat (not 3D like OSF's `[T,2,768]`) | No unified fusion module exists in the pretrained weights (§3) — there is no single PhysioOmni-native per-epoch vector to save, so concatenation is our own design choice, made as simple as possible |
| Channel mapping | Verified directly against real fast-channel HDF5s for all 4 cohorts this session (§4.3), cross-checked against OSF's larger 50-subject-per-cohort audit for completeness percentages | PhysioOmni's channel needs are a strict subset of OSF's; same underlying alias-resolution logic regardless of fast/full tree; direct spot-check + OSF's audit agree exactly |
| Reprocessing | **No** — reuse existing fast-channel `psg/` HDF5s | Every PhysioOmni-needed channel already exists except SHHS's structural EEG gap, which reprocessing wouldn't fix anyway; the already-deferred `OSF_CHANNEL_REPROCESSING_PLAN.md` gaps don't overlap with PhysioOmni's channel needs at all |
| SHHS EEG handling | Duplicate generic `EEG` into both `'C3'`/`'C4'` lookup slots | Same approximation, same justification as OSF's already-user-confirmed decision — **needs its own explicit re-confirmation before Phase 1** (checklist 0.4), not assumed to carry over automatically just because the pattern matches |
| Sample rate | Resample to PhysioOmni's own native per-modality rates (200Hz EEG/EOG, 500Hz ECG/EMG), not a single convenience rate | Matches the reference prep scripts exactly (§4.5); patch duration is a fraction of real time that depends on this choice (§3), so getting it right matters for staying in-distribution with the frozen encoder |
| Normalization | Invert each channel's stored `normalization_stats` (mean/std, real HDF5 attribute, §5) back to raw scale, then `/100` per PhysioOmni's own convention — **not** a per-channel-uniform unit assumption, since `LOC`/`ROC`'s stored scale (§5) implies volts while other channels imply µV | `/100` raw-scale expectation vs. our already-z-scored HDF5 data is a real mismatch, unlike OSF (which happened to already be compatible); the per-channel stats needed to invert it are confirmed present in every HDF5 file — still needs an empirical Step 0 check (§7 checklist 0.5) before trusting extraction output |
| LoRA target modules | `c_attn`, `c_proj` (per encoder, up to 4 encoders) | The only two Linear layers in PhysioOmni's `Attention` block (§3), same kind of finding as OSF's `to_qkv`/`to_out.0` — but genuinely new code since no LoRA precedent exists anywhere in PhysioOmni's own repo |
| Checkpoint needed | `PhysioOmni.pt` only, not `VQ.pt` | Traced `FT.py`'s loading code directly — only `*_encoder.`-prefixed keys are ever loaded from `pretrained_ckpt_path`; VQ codebooks are pretraining-stage-only and never referenced by `FT.py`/`train_finetune.py` (§2) — **still needs empirical confirmation once actually downloaded** (checklist 0.2), this is a strong hypothesis from code reading, not a verified fact yet |

---

## 9. Known open questions (flagged, not blocking the plan's existence,
but needed before Phase 1 starts)

- **Checkpoint key-prefix hypothesis (§2, §8) — not yet verified against
  the real file.** Needs downloading `PhysioOmni.pt` and inspecting
  `torch.load(...)['model'].keys()` directly, same discipline as OSF's own
  checkpoint-filename resolution.
- **Normalization mismatch (§5, §8)** — a concrete inversion method is
  identified (`normalization_stats` attribute, real and present in every
  HDF5 file) and the per-channel unit inconsistency (µV vs. volts) is
  flagged, but neither is empirically validated yet — needs a pilot check
  before extraction code is trusted.
- **Multi-encoder LoRA-wrapping design (§6, §8)** — genuinely undecided,
  needs a design conversation with the user before Phase 2 starts (this is
  new territory PEFT's own docs don't directly cover — wrapping 4 separate
  submodules under one `get_peft_model()` call vs. 4 independent calls has
  real implications for the optimizer/checkpoint structure).
- **Whether `physioomni_env` can share `osf_env`'s environment or needs its
  own fresh build** — not checked yet; PhysioOmni's README pins are close
  to but not identical to OSF's (`torch==2.5.1` matches; `torchvision`/
  `torchaudio` versions and the conda-vs-pip install path differ) — treat
  as "probably needs its own env" until checked, don't assume compatibility.
- **GitHub code repo's missing LICENSE** (§0.7, §2) — the weights are
  CC-BY-4.0 but the training/inference *code* has no stated license; if
  PhysioOmni ships in the paper, this should be flagged the same way OSF's
  own license was tracked, not silently assumed permissive by association
  with the weights' license.

---

## Appendix: source citations for this plan's factual claims

Everything above was verified directly against one of these three sources
(not inferred, not carried over from the earlier candidates-doc pass
without re-checking) — listed here so a future session can re-verify
quickly rather than re-deriving from scratch:

1. **PhysioOmni GitHub repo, read directly at `/home/boshra95/PhysioOmni`
   (2026-08-13)**: `README.md`, `dataset.py`, `dataset.yaml`,
   `model/neural_transformer.py`, `model/FT.py`, `model/transformer.py`,
   `model/MSM.py`, `model/VQ.py` (grep only, for encoder-attribute-naming
   consistency), `train_finetune.py`, `prepare_dataset/prepare_HMC_downstream.py`,
   `prepare_dataset/prepare_CAP.py`, `prepare_dataset/prepare_tuh.py`,
   `utils.py` (grep only, for dataset-split helper signatures).
2. **HuggingFace API, fetched live (2026-08-13)**:
   `https://huggingface.co/api/models/Weibang/PhysioOmni` (license,
   filenames), `curl -IL` against both `resolve/main/PhysioOmni.pt` and
   `resolve/main/VQ.pt` (real file sizes via redirect-then-200).
3. **arXiv paper, fetched live (2026-08-13)**: `arxiv.org/abs/2504.19596`
   and `arxiv.org/html/2504.19596v3` (pretraining corpus composition, HMC
   downstream numbers, author affiliations, submission history/venue
   status).
4. **Our own cluster data, spot-checked live (2026-08-13)**: one real
   fast-channel HDF5 file's channel-key listing and `normalization_stats`
   attribute per cohort
   (`/scratch/boshra95/psg/{apples,shhs,mros,stages}/derived/hdf5_signals/*.h5`),
   cross-referenced against `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s
   already-completed 50-subject-per-cohort audit (against the full-channel
   tree) for the overlapping channel set. Also read
   `configs/preprocessing_params.yaml` and `configs/modality_groups.yaml`
   directly to confirm the fast-channel strategy's priority-order caps
   (§4) via the config, not just the sampled files.
