# PhysioOmni — Supplementary Findings (research notes for drafting)

**Purpose of this file**: raw, cited research findings on PhysioOmni as TSFM
baseline #2 (of 3: OSF, PhysioOmni, Mantis), collected for a later
supplementary-material drafting pass in the npj Digital Medicine submission.
Every concrete claim below is either read directly from code/config in this
session or copied verbatim from an existing, already-code-verified project
doc (never guessed). Section 8 lists every file:line source used.

---

## 1. Model identity & checkpoint

- **Name / paper**: PhysioOmni, "Towards Robust Multimodal Physiological
  Foundation Models: Handling Arbitrary Missing Modalities," arXiv:2504.19596
  (v1 April 2025 → v3 dated 2026-03). Authors from NTU + SJTU (same author
  group as NeuroLM).
- **Peer-review status**: **arXiv-only, never peer-reviewed** — no venue
  acceptance found anywhere in the paper's own submission history, as of the
  last check. Contrast: OSF (baseline #1) is ICML 2026, accepted/peer-reviewed.
- **Params per encoder**: four independent `NeuralTransformer` encoders, no
  shared backbone. Strict-load-verified against the real checkpoint
  (`scripts/verify_physioomni_checkpoint.py`, checklist 0.2): **total 13,871,304
  params across all 4 encoders** — EEG alone 7,839,576; EOG/ECG/EMG
  ~2,010,576 each (3×2,010,576 + 7,839,576 = 13,871,304, checks out exactly).
- **License situation — split, not simply "missing"**:
  - GitHub **code** repo (`935963004/PhysioOmni`): **no LICENSE file anywhere**
    in the repo tree — confirmed live in this session (`ls -la` on the local
    clone at `/Users/boshra/NSRR-workspace/PhysioOmni/` shows no
    LICENSE/COPYING file).
  - HuggingFace **weights** repo (`Weibang/PhysioOmni`): declares
    **CC-BY-4.0** explicitly in the model-card metadata
    (`cardData: {"license":"cc-by-4.0"}`, verified live via the HF API,
    2026-08-13/17). A real, usable, permissive license for the *weights*,
    attribution-required.
  - **Both facts are true simultaneously** and should both be stated if this
    ships in the paper: the code's reuse/modification terms are genuinely
    undocumented even though the weights themselves are clearly and
    permissively licensed.
- **Pretraining corpus / domain**: general clinical/BCI-scale physiological
  signal, **not** sleep-PSG-scale. Composition: TUEG (26,846 recordings,
  clinical epilepsy-monitoring EEG, not sleep), CAP (108 polysomnographic
  recordings), Sleep-EDF (197 whole-night recordings), DEAP (32 participants,
  emotion task, not sleep), a private set (54 recordings / 19 subjects).
  **The sleep-PSG-relevant slice of pretraining is CAP + Sleep-EDF ≈ 305
  recordings** — two to three orders of magnitude smaller than this project's
  own ~16,000-subject cohort, and far smaller than OSF's pretraining scale
  (which already includes SHHS and MrOS at NSRR scale). `dataset.yaml` gives
  dataset *names* only, no subject counts/hours — pretraining corpus size is
  not exhaustively documented by the authors.
- **Checkpoint source**: `https://huggingface.co/Weibang/PhysioOmni`, public,
  not gated. Two files: `PhysioOmni.pt` (267,795,410 bytes ≈ 267.8MB) and
  `VQ.pt` (238,423,693 bytes ≈ 238.4MB), both resolve HTTP 200 via
  `resolve/main/`. **Only `PhysioOmni.pt` is needed** — traced end-to-end
  (`FT.__init__` filters `pretrained_ckpt['model']` for keys starting with
  `EEG_encoder.`/`EOG_encoder.`/`ECG_encoder.`/`EMG_encoder.` only; no VQ
  codebook reference anywhere in `FT.py`/`train_finetune.py`) and empirically
  strict-load-verified: **zero missing keys on all 4 encoders**, one expected
  `mask_token` unexpected-key per encoder (an MSM-pretraining-only component
  `FT.py`'s own `strict=False` load already accounts for). `VQ.pt` was never
  downloaded.
- **A key downstream-comparison fact, load-bearing for the "honest
  expectations" framing**: on PhysioOmni's own best-fit downstream task (HMC
  5-class sleep staging), the paper reports PhysioOmni at **0.7377±0.0056**
  balanced accuracy (all 3 modalities), while the paper's own `FeatFusion`
  baseline (a hand-engineered-feature method, **not** a foundation model)
  scores **0.7478±0.0038** — higher. PhysioOmni does not beat its own paper's
  simplest baseline on the task most similar to this project's use case.

---

## 2. Data adaptation / preprocessing pipeline

- **HDF5 tree used**: the **fast-channel `psg/` tree** (the same,
  paper-primary tree SleepFM's headline `phase0_v3` results use) — **not**
  the full-channel `psg_full/` tree OSF needed. PhysioOmni needs no
  respiratory/thoracic/abdominal/snore channel at all (apnea is excluded
  entirely, see §7), so none of the RESP-group channel gaps that forced OSF
  onto the full-channel tree apply here. Verified both from
  `configs/preprocessing_params.yaml`'s fast-strategy caps (`BAS=4, EKG=1,
  EMG=2, RESP=1`) and `configs/modality_groups.yaml`'s priority-order lists:
  the top-4 BAS channels kept under the fast cap are exactly `C3-M2, C4-M1,
  LOC, ROC` — precisely what PhysioOmni needs — and `EKG`/`CHIN` both survive
  their respective caps in first priority position. **Consequence for the
  paper: PhysioOmni is compared against `phase0_v3` (paper-primary), not
  `phase0_v3_full`** — a more directly relevant baseline than OSF gets.
- **Channel mapping** (`src/nsrr_tools/datasets/physioomni_channel_loader.py`,
  `PHYSIOOMNI_CHANNEL_MAPPING` / `configs/phase0_physioomni_config.yaml`'s
  `data.channel_candidates`):

  | PhysioOmni branch | Our HDF5 source | Notes |
  |---|---|---|
  | EEG (up to 2 ch.) | `C3-M2`→`'C3'`, `C4-M1`→`'C4'` | reference-electrode suffix stripped for the position-embedding lookup key, mirroring PhysioOmni's own `prepare_HMC_downstream.py:116` (`name.split(' ')[-1].split('-')[0]`) |
  | EOG (1 derived ch.) | `LOC − ROC` → `'HEO'` | matches `prepare_HMC_downstream.py:99`'s own derivation exactly |
  | ECG (1 ch.) | `EKG` → fallback `ECG-L` → `'ECG'` | same fallback order already used for OSF |
  | EMG (1 ch.) | `CHIN` → fallback generic `EMG` → `'EMG'` | same fallback order already used for OSF |

  Real per-cohort coverage (full raw population, `output/channel_analysis/*.csv`
  cross-referenced against `configs/channel_definitions.yaml`): C3/C4 ≈99.4–100%
  in APPLES/MrOS/STAGES but **0% in SHHS** (SHHS has no split C3/C4, only a
  generic `EEG` key at 100% coverage); LOC/ROC ≈99.4–100% all cohorts; ECG
  ≈99.9–100% except **STAGES 90.8%** (173/1,879 subjects have only a derived
  `Heartrate` value, not a raw waveform — genuinely absent data, not a fixable
  alias-list gap); EMG ≈99.9–100% all cohorts.
- **SHHS's single-EEG-channel handling (final decision, no reprocessing)**:
  SHHS's `EEG`/`EEG(sec)`-family channels were investigated and found to be
  genuinely distinct (100% of 8,444 SHHS subjects carry both; a real raw EDF
  correlation check gave **r = 0.18**, far from a duplicate's ~1.0). Despite
  this, the plan explicitly chose **not** to duplicate SHHS's one available
  `EEG` channel into both the C3 and C4 slots (unlike OSF, which correctly
  *does* duplicate, for its own different fixed-tensor architectural reason).
  PhysioOmni's `NeuralTransformer` takes a variable-length token sequence per
  modality (CAP pretraining itself used anywhere from a few to 16 EEG
  channels per subject), so **SHHS is fed exactly 1 real, non-duplicated EEG
  channel** — a legitimate native input, not a workaround. This is live-tested
  and confirmed working through the actual encoder
  (`scripts/test_physioomni_channel_loader.py`; extraction smoke test on
  `200001_v1`: `eeg_channel_count: 1`, CLS-output std ~0.89–1.12, in the same
  non-degenerate range as APPLES's 2-channel EEG). A lightweight future fix
  (additively extracting the currently-discarded `EEG(sec)` channel) is
  documented but deliberately not pursued unless SHHS results look degraded.
- **The `/100` raw-amplitude normalization inversion — self-calibrating, not
  a uniform per-channel-type rule.** PhysioOmni's own `dataset.py` divides
  raw signal by 100 (not z-scoring) before feeding the model. This project's
  HDF5s store per-channel z-scored data plus the exact pre-normalization
  `{mean, std, min, max}` in a `normalization_stats` root attribute, so the
  inversion is `x_original = x_zscored * std + mean` per channel, then a unit
  correction, then `/100`. **The unit correction is not a fixed per-channel
  rule** — traced `signal_processor.py` directly and found **no explicit unit
  conversion anywhere** (`raw[ch_idx, :]` plain MNE indexing, no `units=`
  argument, no scaling in `_process_channel`/`_normalize_signal`), so the raw
  unit MNE returns depends on the source EDF file's own header declarations,
  which turned out to be **cohort/file-dependent, not per-channel-name
  fixed**. Real measured evidence (loading real EDFs directly with
  `mne.io.read_raw_edf`): APPLES's `ECG` is µV-scale (std≈18.7) but SHHS's
  `ECG` is volts-scale (std≈1.3–2.2e-4) — **the same canonical channel name,
  two different units depending on cohort.** The original hypothesis ("LOC/ROC
  = volts, everything else = µV") was tested and found wrong for exactly this
  reason. Final self-calibrating detector, implemented in
  `invert_normalization()`:
  ```python
  def invert_normalization(x_zscored, stats):
      x = x_zscored * stats["std"] + stats["mean"]
      if abs(stats["std"]) < 1.0:      # this channel's raw scale was volts
          x = x * 1e6                  # -> uV
      return x / 100.0                 # PhysioOmni's own dataset.py convention
  ```
  The `1.0` threshold exploits a ~5-order-of-magnitude gap between the
  µV-scale (std ~1–100s) and volts-scale (std ~1e-6–1e-3) regimes — reasoned,
  not exhaustively swept, but empirically validated (see below).
- **Sample rate**: our HDF5s are uniformly 128Hz; PhysioOmni's own reference
  prep scripts (`prepare_CAP.py`, `prepare_HMC_downstream.py`, `prepare_tuh.py`)
  consistently resample to **200Hz for EEG/EOG, 500Hz for ECG/EMG** before
  chunking into patches — replicated exactly (all upsampling, via
  `scipy.signal.resample`/polyphase, no exact-decimation shortcut the way
  OSF's clean 128→64Hz 2:1 ratio has).
- **Windowing/patching at 30s granularity**: for each 30-second epoch and
  each of the up to 4 present modality branches: resample to native rate,
  chunk into patches (EEG 200-sample/1.0s patches, EOG 100-sample/0.5s,
  ECG/EMG 100-sample/0.2s — all at native Hz), run through that modality's
  `NeuralTransformer.forward_features(..., return_all_tokens=False)` to get
  one CLS vector (200-dim EEG, 100-dim each other). Real per-modality config
  (`train_finetune.py:267-274`, confirmed identical in `train_msm.py`/`train_vq.py`,
  re-confirmed directly in this session):

  | Modality | `n_embd` | `patch_size` | `n_layer` | `n_head` |
  |---|---|---|---|---|
  | EEG | 200 | 200 samples | 12 | 10 |
  | EOG | 100 | 100 samples | 12 | 10 |
  | ECG | 100 | 100 samples | 12 | 10 |
  | EMG | 100 | 100 samples | 12 | 10 |

- **Embedding extraction pipeline**: `scripts/extract_physioomni_embeddings.py`,
  driven by `configs/phase0_physioomni_config.yaml`. Per subject: concatenate
  the four modality CLS vectors into one flat 500-dim vector per epoch
  (200 EEG + 100 EOG + 100 ECG + 100 EMG, fixed order), zero-filling any
  entirely-absent modality's whole 100/200-dim slice (never per-patch —
  a modality is present-or-absent for a whole subject). **Output shape:
  `[T_epochs, 500]`, dtype float16**, at
  `/scratch/boshra95/psg/unified/embeddings/physioomni_30sec/{dataset}/{subject_id}.npy`.
  This is genuinely 2D — no sub-token axis the way OSF's `[T,2,768]` needs,
  because there is no unified fusion module in the released checkpoint to
  preserve any shared structure across. A `_channel_fill_log.jsonl`-style log
  records which modality was zero-filled per subject and the real EEG
  channel count (1 for SHHS, 2 elsewhere).
- **The missing-respiratory-pathway fact, confirmed at 4 independent code
  locations** (re-verified directly in this session, not just cited from
  prior notes):
  1. `dataset.yaml` — every pretraining source declares only
     `contain_EEG/EOG/ECG/EMG` booleans; no RESP/SpO2 field exists in the
     schema at all (confirmed live: `grep -n "contain_" dataset.yaml` returns
     only these four booleans across all listed sources).
  2. `dataset.py` (`PickleLoader`, `DownstreamLoader`) only ever reads
     `sample["EEG"|"EOG"|"ECG"|"EMG"]` keys.
  3. `prepare_dataset/prepare_tuh.py` — its `drop_channels` list (line
     35-37, confirmed live) explicitly discards
     `'EEG RESP1-REF', 'EEG RESP2-REF', 'RESP ABDOMEN-REF', 'PULSE RATE',
     'RESP THORAX-REF'` from raw TUH EDFs before any processing.
  4. `prepare_dataset/prepare_HMC_downstream.py` — HMC is a real PSG dataset
     with SpO2/airflow channels available in the source EDF, but the prep
     script only defines EEG/EOG/ECG/EMG channel lists; respiratory/SpO2
     channels are never touched even though present in the source data.
  `model/FT.py`'s `forward()` (confirmed live, `FT.py:86-89` and `:215-218`)
  hard-codes exactly 4 encoder branches
  (`self.EEG_encoder`/`self.EOG_encoder`/`self.ECG_encoder`/`self.EMG_encoder`)
  — there is no generic N-modality loop a 5th channel group could be slotted
  into.

---

## 3. Architecture summary

- **Four independent per-modality encoders**, each a `NeuralTransformer`
  (`model/neural_transformer.py`) — **not** one shared backbone, and **no
  cross-modal attention or fusion exists in the pretrained/frozen weights
  themselves.** This is the single biggest structural difference from OSF
  (one ViT) and from SleepFM (contrastive alignment across 4 modality-group
  encoders during pretraining, but SleepFM's design is otherwise more unified
  than PhysioOmni's).
- **Per-encoder params**: EEG 7,839,576; EOG/ECG/EMG 2,010,576 each — total
  13,871,304 (strict-load-verified, §1).
- **Hidden dims**: EEG `n_embd=200` (a multiple of 8); EOG/ECG/EMG
  `n_embd=100` each (**not** a multiple of 8 — flagged explicitly in
  `docs/TSFM_MODEL_COMPARISON.md` as a minor additional GPU-tensor-core
  misalignment stacked on top of the more fundamental "just too small"
  problem, see §6).
- **Patch embedding**: `TemporalConv`, a 3-layer `Conv2d` stack over the
  patch axis (kernel `(1,15)`→`(1,3)`→`(1,3)`, stride `(1,8)` on the first
  layer only) — a real temporal feature extractor per patch, not a bare
  linear projection.
- **Position/time embeddings**: `pos_embed = nn.Embedding(256, n_embd)`
  (channel-identity index, exact-string match against a 132-entry
  `standard_1020` name list — a closed vocabulary, `ValueError` on any
  unrecognized name) and `time_embed = nn.Embedding(512, n_embd)`
  (sequential patch-index within the modality stream, not a real-world-time
  offset — capping any single forward pass at 512 *tokens*, not 512 seconds).
- **Attention**: bidirectional, non-causal
  (`scaled_dot_product_attention(..., is_causal=False)`,
  `model/transformer.py:55`) — masked-pretraining style, confirming Plan A
  (native long-context, no sequence head) is architecturally unavailable,
  same conclusion as for OSF and MOMENT.
- **Pooled output**: `forward_features(..., return_all_tokens=False)` returns
  the CLS token (`x[:, 0]`) per modality per forward pass — one `n_embd`-dim
  vector, directly usable as a frozen embedding.
- **No unified multimodal fusion in the checkpoint.** `FT.py`'s fusion
  machinery (`*_embedding`/`*_head`/`*_Linear`/`X_transformer`/
  `alignment_module`/`lm_head`) is constructed fresh at `FT.__init__` time
  every run; `pretrained_ckpt_path` loading only ever populates the four
  `*_encoder` submodules. **The released checkpoint is four per-modality
  tokenizers/encoders only — any concatenation/fusion used in this project's
  own pipeline (the flat 500-dim vector) is entirely this project's own
  construction, not something the checkpoint provides.**
- **LoRA target modules**: `model/transformer.py`'s `Attention` class has
  exactly two Linear layers — `self.c_attn` (fused QKV) and `self.c_proj`
  (output projection) — the natural `target_modules=["c_attn", "c_proj"]`
  choice, applied per encoder (up to 4 encoders × 12 blocks = 48 attention
  modules). No existing LoRA/PEFT code exists anywhere in the PhysioOmni
  repo (`grep -rniE "peft|lora"` → zero hits) — all LoRA wiring for this
  project is new code, unlike OSF where the target-module names came "for
  free" from the checkpoint's own naming.

---

## 4. Stage 1 — Frozen encoder + downstream head training

- **Frozen vs. trained**: all 4 `NeuralTransformer` encoders are frozen
  (embeddings precomputed once, offline); only a new sequence head
  (LSTM/Transformer/MeanPool, reused unmodified from
  `src/nsrr_tools/models/sequence_head.py`) is trained from scratch on top
  of the concatenated 500-dim embeddings.
- **New files, exact paths** (nothing shared with OSF/SleepFM is ever
  edited — hard file-isolation constraint, verified):
  - `src/nsrr_tools/datasets/physioomni_channel_loader.py` — shared
    channel-loading/resampling/normalization utility.
  - `scripts/extract_physioomni_embeddings.py` — embedding extraction.
  - `jobs/extract_physioomni_embeddings_gpu.sh`,
    `jobs/extract_physioomni_embeddings_cpu.sh` — SLURM jobs.
  - `src/nsrr_tools/datasets/physioomni_context_window_dataset.py` —
    `PhysioOmniContextWindowDataset` (fork of `ContextWindowDataset`,
    simplified to a flat `[T,500]` shape — no 3D sub-token dimension, no
    `--zero-modalities` ablation feature since PhysioOmni's 4 "modalities"
    are already the finest granularity).
  - `scripts/train_physioomni_context_sweep.py` +
    `jobs/train_physioomni_context_sweep_gpu.sh` — training loop, forked
    from `scripts/train_context_sweep.py` with identical
    `run_epoch`/`compute_metrics`/checkpoint/resume/early-stopping machinery.
  - `scripts/infer_physioomni_subject_windows.py` +
    `jobs/infer_physioomni_subject_windows_gpu.sh` — inference.
  - `experiments/v2_physioomni_registry.yaml` +
    `scripts/gen_commands_physioomni.py` — registry/command generator.
  - `configs/phase0_physioomni_config.yaml` — master config.
- **Real hyperparameters** (`configs/phase0_physioomni_config.yaml`, verified
  by direct read, not approximated):
  - `embedding.chunk_batch_size: 16` (kept at the default — an A/B against
    64 on matched 20-subject SHHS batches found **no meaningful difference**,
    4.05 vs. 4.15 s/subject; unlike OSF, where this knob was the real
    bottleneck).
  - `model.input_dim: 500`, `hidden_dim: 128`, `num_layers: 1`, `num_heads: 8`,
    `dropout: 0.3`, `num_classes: 2` — architecture held identical to
    SleepFM's/OSF's seq2label head; only `input_dim` changes (500 vs.
    SleepFM's 512, OSF's 1536), preserving the "only the encoder/channels
    change" comparison principle.
  - `training.epochs: 40`, `lr: 1.0e-4`, `weight_decay: 1.0e-3`,
    `scheduler: cosine`, `early_stopping_patience: 10`,
    `mixed_precision: false`, `class_weights: auto`, `weighted_sampler: false`
    — untuned OSF/SleepFM placeholder starting values, never revised for
    Stage 1 (no evidence a real pilot required a change, unlike Stage 2).
  - `context_lr_overrides: {"120m": 5.0e-5, "240m": 5.0e-5}`.
  - `dataset.min_recording_patches: 480` (240m in 30s-epoch units — **not**
    SleepFM's 2880, a documented "single easiest place to introduce a silent
    bug" warning carried over from OSF's own plan).
  - `dataset.split_seed: 42`, `train/val/test: 0.70/0.15/0.15` — same seed
    and ratios as SleepFM's `phase0_v3` and OSF, for a fair comparison
    (though PhysioOmni's own split-mismatch-vs-SleepFM has **not** been
    investigated the way OSF's was — flagged explicitly in
    `TSFM_MODEL_COMPARISON.md` §2.3 as an open item, "not because it was
    checked and found fine, but because it hasn't been checked").
- **Compute characteristics (frozen extraction, real measured)**: ~4.1
  s/subject on a single H100 MIG `1g.10gb` slice (15–100× faster than the
  CPU path's 50–450 s/subject) — full ~14,994-subject population ≈17h
  serial on one GPU, sharded into ~6 parallel jobs of ~2,500 subjects
  (~2.85h each).
- **Coverage completed**: **4 Tier-1 tasks** (`sex_binary`,
  `sleep_efficiency_binary`, `bmi_binary`, `age_class`) × **2 heads**
  (lstm, transformer — `mean_pool` not run) × **6 contexts** (30s, 10m, 40m,
  80m, 120m, 240m) = complete, monotonic-with-context, no anomalies (646
  analysis rows, zero NaN `seg_auroc`, per `PHYSIOOMNI_CLAUDE.md`). **`apnea_binary`
  is deliberately excluded** — no respiratory pathway anywhere in the model
  (§2/§7). Embedding extraction itself: 14,993/14,994 subjects (99.99%),
  the 1 gap (`stages/STLK00096`) is a known pre-existing data-quality
  outlier with zero PhysioOmni-relevant channels, not a pipeline bug (this
  exact subject is also flagged as an outlier in OSF's own population).
  Real headline numbers (test `seg_auroc`, k=all, 2026-09-06):

  | task | head | 30s | 240m |
  |---|---|---|---|
  | sex_binary | lstm | 0.673 | 0.847 |
  | sex_binary | transformer | 0.670 | **0.863** |
  | sleep_efficiency_binary | lstm | 0.657 | 0.768 |
  | sleep_efficiency_binary | transformer | 0.657 | 0.771 |
  | bmi_binary | lstm | 0.665 | 0.727 |
  | bmi_binary | transformer | 0.660 | 0.736 |
  | age_class | lstm | 0.794 | **0.860** |
  | age_class | transformer | 0.790 | 0.856 |

  Compared against SleepFM's `phase0_v3` at the same tasks/contexts,
  **PhysioOmni's frozen encoder underperforms SleepFM at every context, on
  every comparable task** — e.g. `sex_binary`, 30s: SleepFM 0.832 vs.
  PhysioOmni 0.754 (−7.8pp); 240m: SleepFM 0.910 vs. PhysioOmni 0.877
  (−3.3pp). The gap is largest at short context and narrows but never
  closes at long context, with no contamination confound available to
  explain it away (PhysioOmni's pretraining corpus contains none of the
  four NSRR cohorts).

---

## 5. Stage 2 — LoRA fine-tuning

- **LoRA config** (`configs/phase0_physioomni_lora_config.yaml`, `lora:`
  section, real values):
  ```yaml
  lora:
    r: 8
    lora_alpha: 16
    lora_dropout: 0.05
    target_modules: ["c_attn", "c_proj"]
    modules_to_save: ["sequence_head"]
  ```
  `r`/`lora_alpha`/`lora_dropout`/`lr` are explicitly flagged in the config's
  own comments as **not yet calibrated against a real pilot** — starting
  values match OSF's own original (pre-pilot) LoRA starting point, not
  independently tuned for PhysioOmni.
- **Multi-encoder wrapping (resolved, live-verified 2026-08-19)**: all 4
  encoders + the sequence head are wrapped as ONE `CombinedPhysioOmniLoRAModel`
  submodule tree, then a **single** `peft.get_peft_model(combined,
  LoraConfig(target_modules=["c_attn","c_proj"],
  modules_to_save=["sequence_head"]))` call — `peft`'s target-module matching
  is name-suffix-based across the whole module tree, so one call correctly
  wraps all 4×12=48 attention blocks. Live-verified against the real
  checkpoint:
  ```
  trainable params: 289,002 || all params: 14,161,308 || trainable%: 2.04
  LoRA modules per encoder: {'eeg_encoder': 24, 'eog_encoder': 24, 'ecg_encoder': 24, 'emg_encoder': 24}
  ```
  24 = 12 blocks × 2 target modules per encoder × 4 encoders = 96 total
  LoRA-wrapped Linear layers, exactly as expected.
- **Missing-modality handling**: a batch-level present/absent mask (not
  zero-input-forward) preserves Stage 1's exact zero-fill contract — an
  absent modality's embedding slice stays exactly zero with **no** encoder
  forward pass run for it, rather than running the encoder on zero-valued
  raw signal (which would produce a non-zero learned output, silently
  diverging from Stage 1's frozen embeddings).
- **Warm-start strategy**: 30s always warm-starts the sequence head from
  Stage 1's frozen-backbone checkpoint; every other context length
  warm-starts LoRA+head **together** from that same (task, head)'s own
  already-converged **30s** LoRA checkpoint — a branch from 30s, not a
  chain (30s→10m→...→240m) — because compute scales ~linearly with raw
  epochs per window, so chaining would make every length's result depend on
  arbitrary sweep order.
- **Training budget (revised from OSF's placeholder, grounded in
  PhysioOmni's own pilot curve, not copied)**: `epochs: 40→25`,
  `early_stopping_patience: 10→5`, `lr` deliberately left at `1.0e-4`
  (**not** halved to OSF's `5.0e-5`). PhysioOmni's own 30s pilot
  (`sex_binary`/lstm) showed **no OSF-style overfitting** (still improving
  every epoch through epoch 4: val_auroc 0.6615→0.6679→0.6721→0.6743, all
  new-bests, patience 0/10; only a small first decline at epoch 5, 0.6723,
  patience 1/10) — the revision was driven by **wall-clock time** (~1
  hr/epoch observed at 30s, the fastest context), not by an overfitting
  signature.
- **Coverage completed: 9 of 48 (task, head, context) cells, as of
  2026-09-06** — confirmed exact number from both the implementation plan's
  checklist 2.8 and `PHYSIOOMNI_CLAUDE.md`'s Stage 2 results section:

  | task | head | contexts done | test seg_auroc (k=all) |
  |---|---|---|---|
  | sex_binary | lstm | 30s, 10m, 40m | 0.720 / 0.794 / 0.825 |
  | sex_binary | transformer | 30s, 10m | 0.703 / 0.782 |
  | sleep_efficiency_binary | lstm | 30s, 10m | 0.664 / 0.676 |
  | sleep_efficiency_binary | transformer | 30s, 10m | 0.656 / 0.677 |
  | bmi_binary | lstm, transformer | none | — |
  | age_class | lstm, transformer | none | — |

  (9 cells = 3+2+2+2.) `bmi_binary` and `age_class` have **zero** LoRA
  cells. Where measured, LoRA is a real but modest win over frozen at every
  matched context (e.g. `sex_binary` lstm: frozen 0.673/0.748/0.790 at
  30s/10m/40m vs. LoRA 0.720/0.794/0.825).
- **Exact stopping reasons — a compound of three distinct causes, explicitly
  not to be described with the same "clean stopping point" language used for
  OSF's own (deliberate, uniform, 120m) LoRA cutoff:**
  1. **A genuine architectural compute ceiling** — PhysioOmni's Stage 2
     measured **~0.69 TFLOP/s, ~3.6% of a realistic fp32 ceiling on a
     `3g.40gb` MIG slice** (see §6 for the full breakdown) — not fixable
     from the training script, a property of the released checkpoint's
     architecture (4 separate small-hidden-dim encoders).
  2. **A real 15-day operational GPU-billing incident, dated and
     quantified**: on **2026-08-22**, `--gpus=h100:1` (a whole H100 card)
     was requested instead of the usual MIG slice, on the strength of a
     `sbatch --test-only` estimate showing no obvious queue penalty. The
     resulting 80m job then sat **PENDING for 15 real days** because
     whole-card jobs bill at **2.3×** and starve under this account's
     fairshare policy — caught only when the user asked why nothing had
     progressed. Reverted to `3g.40gb` on **2026-09-06**.
  3. **Specific, dated engineering bugs, found while trying to get the raw-
     signal cache and first real training run working (2026-08-20/21)**:
     - **OOM on the `[9000:13481]` shard (mostly MrOS)**: 16 concurrent
       `scipy.signal.resample` worker buffers exceeded the job's 32GB
       memory request, confirmed even running alone on its node (ruling
       out cross-job contention). Fixed: `--mem` raised 32000M → 64000M.
     - **Non-atomic `meta.json` write**: workers killed by the OOM/SIGTERM
       events above left **81 zero-byte `meta.json` files** with fully
       intact `.npy` siblings; `cache_exists()` treated mere file existence
       as "done," so these silently blocked reprocessing and broke the
       first real LoRA training attempt with a `JSONDecodeError`. Fixed:
       `save_signal_cache()` now writes `meta.json` via temp-file +
       `os.replace` (atomic rename). The 81 corrupt files were deleted for
       reprocessing.
     - **A `chunk_batch_size`-independent OOM, requiring gradient
       checkpointing**: `chunk_batch_size` only bounds each individual
       encoder call's size, not peak memory — every chunk's activations
       stay in the *same* autograd graph until one shared `backward()`
       call, so a 240m window (480 epochs ÷ 16/chunk = 30 sequential calls)
       retained 30 chunks' worth of full 12-block activations
       simultaneously. First real OOM at `sex_binary_lstm`/10m with
       `micro_batch=32` (32×20=640 "epoch-units," right at the ~19.6GB
       ceiling observed for a `2g.20gb` slice at the time); after reducing
       `micro_batch` to the floor of 1, **240m still OOM'd** — the actual
       fix applied was `torch.utils.checkpoint.checkpoint(...,
       use_reentrant=False)` wrapping each chunk's encoder call, opt-in and
       default-off (verified via a monkeypatched call-count check that it
       never activates unless requested), and verified **bit-identical**
       loss/gradients between checkpointed and non-checkpointed runs on
       identical input (max abs diff: 0.0) — a real correctness guarantee,
       not just "doesn't crash."

---

## 6. Honest computational drawbacks

- **Measured throughput: ~0.69 TFLOP/s**, reported as **~3.6% of this
  account's realistic fp32 ceiling on a `3g.40gb` MIG slice**
  (`PHYSIOOMNI_CLAUDE.md`, `TSFM_MODEL_COMPARISON.md` §3.2/§4). A
  second, independently-computed framing of the same measurement in
  `docs/TSFM_THIRD_MODEL_DECISION.md` §5.3, calibrated against the real
  PhysioOmni 80m transformer run (18h/epoch, 57,195 items, N=160 raw
  epochs/window): **9.2M epoch-forwards → ~44.6 PFLOP per training epoch
  (fwd+bwd) → 0.69 TFLOP/s effective → 0.14% of a whole H100's ~989 TFLOP/s
  bf16 peak.** These two percentage figures (3.6% vs. 0.14%) are **not
  contradictory** — they are the same ~0.69 TFLOP/s measurement expressed
  against two different denominators (a realistic fp32 ceiling on a
  `3g.40gb` slice vs. a whole-card bf16 peak); state the denominator
  explicitly if either number is quoted in the paper.
- **The specific architectural reason, per `TSFM_THIRD_MODEL_DECISION.md`
  §5.3's multiplicative breakdown** (of the ~989 TFLOP/s whole-card bf16
  peak, only some factors are PhysioOmni-fixable):

  | Factor | Divisor | Resulting ceiling |
  |---|---|---|
  | H100 bf16 peak, whole card | — | 989 TFLOP/s |
  | fp32 with TF32 disabled (PyTorch 2.5 default) | ÷8 | 124 TFLOP/s |
  | MIG slice (2/7 or 3/7 of the card, depending on run) | ÷3.5 | ~35 TFLOP/s |
  | Hidden dim 100–200 — too small for tensor cores | ÷5–10 | 3.5–7 TFLOP/s |

  The first two factors (TF32, MIG-vs-whole-card) were investigated and
  partly addressed on 2026-08-22 (see the billing incident below — the
  whole-card attempt was exactly this fix, and it backfired operationally).
  **The third factor — small hidden dims — is architectural and
  unfixable from the training script**: PhysioOmni's EOG/ECG/EMG encoders
  use `hidden_dim=100`, not even a multiple of 8, spread across **four
  separate small encoders** rather than one joint tensor. `TSFM_MODEL_COMPARISON.md`
  §4 states this as the most fragmented of the three fine-tuned-backbone
  designs compared in the paper (SleepFM, OSF, PhysioOmni): "even a
  perfectly-tuned batching scheme still pays for 4 separate kernel-launch
  groups per window instead of 1."
- **The batching fix that worked for OSF did not work for PhysioOmni** — a
  real, diagnostic finding, not an assumption. OSF's `chunk_batch_size`
  16→64 A/B gave a confirmed 3.28× speedup (overhead-bound at short
  contexts). The *identical* A/B for PhysioOmni (run during embedding
  extraction, on matched 20-subject SHHS batches) found **no meaningful
  difference** (4.05 vs. 4.15 s/subject) — evidence that PhysioOmni's
  bottleneck is the matmul size itself (compute-shape-bound), not
  call-launch overhead, and is therefore structurally harder to engineer
  around than OSF's problem.
- **The operational incident, in full detail**: on **2026-08-22**, the
  Stage 2 job script's GPU request was changed from the MIG slice size in
  use (`2g.20gb`, confirmed the exact figure appearing in the OOM logs) to
  a whole `h100:1` card, motivated by a `sbatch --test-only` snapshot
  showing no apparent queue-time penalty for the bigger request. The
  resulting **80m LoRA job sat PENDING for 15 real days** — whole-card jobs
  bill at 2.3× and starve under this Compute Canada account's fairshare
  policy, a scheduling-economics fact that a one-off `--test-only` snapshot
  does not reveal (it is a heuristic snapshot of one moment, not a
  guarantee across hours/days of real scheduling pressure). This was
  caught only when the user asked why progress had stalled — not by design
  or by any automated check. Reverted to `3g.40gb` on **2026-09-06**.
  `docs/LORA_GPU_THROUGHPUT_INVESTIGATION.md` documents that this exact
  proposal (TF32 + whole-card request) originated from "a session working
  on PhysioOmni" that had found its own LoRA training at a tiny fraction of
  peak throughput — i.e., the throughput investigation and the billing
  incident are directly causally linked, not two independent facts.
- **Other real, dated bugs** (see §5 for full detail): (1) OOM on the
  `[9000:13481]` (mostly-MrOS) precompute shard — fixed by raising `--mem`
  32000M→64000M; (2) non-atomic `meta.json` cache-write leaving 81
  zero-byte files that silently blocked reprocessing and broke the first
  real training attempt with a `JSONDecodeError` — fixed via temp-file +
  `os.replace` atomic write; (3) a `chunk_batch_size`-independent OOM at
  240m (root cause: `chunk_batch_size` bounds per-call size, not peak
  memory, since all chunks share one autograd graph until one `backward()`
  call) — fixed via opt-in, default-off gradient checkpointing, verified
  bit-identical to non-checkpointed gradients.
- **A separate lesson the project explicitly does not want conflated with
  the above**: `docs/TSFM_MODEL_COMPARISON.md` §3.2 states directly, "Don't
  let that incident get folded into 'the architecture is inefficient'; it's
  a distinct, non-architectural cause that happened to compound with the
  real architectural one." The billing incident is an operational mistake;
  the ~0.69 TFLOP/s ceiling is a property of the checkpoint.

---

## 7. Other reviewer-relevant caveats

- **Apnea exclusion, restated**: PhysioOmni has no respiratory/airflow
  pathway anywhere in the model or its pretraining data, confirmed at 4
  independent code locations (§2). Adding one would require a new tokenizer
  branch plus retraining the VQ/MSM pretraining stage — a pretraining-scale
  change, not a fine-tuning adapter — so apnea is excluded from the
  PhysioOmni comparison with a stated reason rather than attempted as a
  workaround. This reduces PhysioOmni's task scope to 4 of the paper's 5
  Tier-1 tasks (sex, sleep efficiency, BMI, age).
- **License split**: code repo has no LICENSE file; HF weights repo
  declares CC-BY-4.0 explicitly (§1). Flag both facts if this ships.
- **arXiv-only, never peer-reviewed**, contrasted directly with OSF's ICML
  2026 acceptance — the downstream numbers cited from the PhysioOmni paper
  have not been through review.
- **PhysioOmni's own paper's best-fit downstream result did not beat its
  own non-foundation-model baseline**: 0.7377 (PhysioOmni, HMC 5-class
  sleep staging, all 3 modalities) vs. 0.7478 (`FeatFusion`, hand-engineered
  features, no foundation model). This is presented in this project's own
  docs as "the single most important fact behind this section's honest
  framing" — a legitimate reason to keep absolute-performance expectations
  modest going in, stated up front rather than discovered as a surprise.
- **No contamination confound available** (unlike OSF, whose pretraining
  corpus includes SHHS/MrOS): PhysioOmni's pretraining corpus (TUEG, CAP,
  Sleep-EDF, DEAP, a small private set) contains none of this project's
  four NSRR cohorts, so PhysioOmni's underperformance relative to SleepFM
  cannot be attributed to a held-out-data advantage either way.
- **PhysioOmni's split-vs-SleepFM has not been checked for the OSF-style
  subject-list mismatch** — OSF and SleepFM were found to use subtly
  different train/val/test splits due to differing per-embedding-directory
  subject coverage (a documented, real, already-found issue for OSF). The
  analogous check has simply not been run for PhysioOmni — flagged as an
  open item in `TSFM_MODEL_COMPARISON.md` §2.3, "not because it was checked
  and found fine, but because it hasn't been checked." State this
  explicitly if the PhysioOmni-vs-SleepFM gap is reported to the same
  apparent precision as the OSF-vs-SleepFM one.
- **PhysioOmni is the only one of the three baselines (OSF, PhysioOmni,
  Mantis) where "Plan B" (short-segment embedder + external sequence head,
  as opposed to native long-context ingestion) was a genuine *choice* rather
  than an architectural inevitability.** Its `NeuralTransformer` could
  theoretically ingest up to ~512s (EEG) / ~102–256s (EOG/ECG/EMG) natively
  per forward pass — unlike SleepFM's hard 300s and OSF's hard 30s ceilings,
  which leave no alternative at all — but even that best case falls short
  of every context-length sweep point except the shortest (30s), since 10m
  (600s) already exceeds even EEG's 512s theoretical maximum. The 30-second
  epoch unit ultimately chosen also happens to match PhysioOmni's *own* HMC
  downstream fine-tuning convention (`prepare_HMC_downstream.py`'s hard
  `Duration==30` filter) — not an arbitrary convenience choice. Worth
  stating in the paper as a materially more favorable ("we gave this model
  a fair shot") situation than SleepFM's/OSF's genuinely forced Plan B.
- **Downstream fine-tuning convention nuance**: the model's *typical*
  pretraining exposure was tens of seconds of real time across many (8–26)
  simultaneous EEG channels (`time = 512 // len(eegCh)`, confirmed in
  `prepare_CAP.py`/`prepare_tuh.py`/`prepare_DEAP.py`), not minutes of real
  time on one channel — the "8.5-minute ceiling" figure is a valid
  theoretical upper bound but not representative of what the encoder
  actually learned to expect. Worth stating precisely rather than the
  simpler-but-misleading flat framing if this appears in the Methods
  section.

---

## 8. Source citations

**Primary planning/status docs (already code-verified in prior sessions, re-verified spot-checks noted below):**
- `/Users/boshra/NSRR-workspace/NSRR-tools/CLAUDE.md:267-693` — "TSFM
  Baseline Model Comparison" section, PhysioOmni-specific status notes.
- `/Users/boshra/NSRR-workspace/NSRR-tools/docs/PHYSIOOMNI_PLANNING_HANDOFF.md:1-91`
  — full file, review/rewrite history context.
- `/Users/boshra/NSRR-workspace/NSRR-tools/docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md:1-2744`
  — full file (all sections used; §0 go/no-go decision, §2 checkpoint
  status, §3 architecture, §4-§5 channel mapping/normalization, §6-§14
  Stage 1 file map/config, §15 Stage 2 design + checklist 2.6/2.8 (billing
  incident, bugs), §17 key-decisions table, §19-§20 native-context-ceiling
  and 3-way comparison table, Appendix source list).
- `/Users/boshra/NSRR-workspace/NSRR-tools/docs/PHYSIOOMNI_EXPERIMENTS_GUIDE.md:1-665`
  — full file (run-identity table, Step 0-8 operational detail, real
  measured throughput numbers).
- `/Users/boshra/NSRR-workspace/NSRR-tools/docs/TSFM_BASELINE_CANDIDATES.md:271-385`
  — §2.2 PhysioOmni technical survey (architecture, LoRA absence, license,
  respiratory-pathway 4-location evidence, adapter effort).
- `/Users/boshra/NSRR-workspace/NSRR-tools/docs/TSFM_MODEL_COMPARISON.md:238-321,482-821`
  — §2.2 PhysioOmni-vs-SleepFM results tables, §2.3 split-mismatch caveat,
  §3.1-3.3 computational-cost narrative and stopping-criteria table, §4
  PhysioOmni architectural-inefficiency subsection.
- `/Users/boshra/NSRR-workspace/NSRR-tools/docs/TSFM_THIRD_MODEL_DECISION.md:396-490`
  — §5 compute section: verified param counts, GFLOP/epoch table, the
  0.69 TFLOP/s / 0.14%-of-peak calibration and its multiplicative-cause
  breakdown.
- `/Users/boshra/NSRR-workspace/NSRR-tools/docs/LORA_GPU_THROUGHPUT_INVESTIGATION.md:1-191`
  — full file; §1 documents that the TF32/whole-card proposal originated
  from the PhysioOmni session, directly linking it to the 15-day billing
  incident.
- `/Users/boshra/NSRR-workspace/NSRR-tools/PHYSIOOMNI_CLAUDE.md:257-464`
  — Stage 1 results table, Stage 2 results table (9/48 cells) and full
  chronological bug/incident log with dates.

**Code and config, read/verified directly in this session:**
- `/Users/boshra/NSRR-workspace/PhysioOmni/` — `ls -la` confirms no
  LICENSE/COPYING file in the repo root.
- `/Users/boshra/NSRR-workspace/PhysioOmni/dataset.yaml:4-31` — `contain_EEG/EOG/ECG/EMG`
  booleans only, no RESP field, across every listed pretraining source.
- `/Users/boshra/NSRR-workspace/PhysioOmni/prepare_dataset/prepare_tuh.py:35-37`
  — `drop_channels` list explicitly discarding RESP-named channels.
- `/Users/boshra/NSRR-workspace/PhysioOmni/model/FT.py:86-89,99-114,215-218`
  — four hard-coded `*_encoder` submodules, checkpoint key-prefix filtering,
  four-branch `forward()`.
- `/Users/boshra/NSRR-workspace/PhysioOmni/train_finetune.py:267-280`
  — real per-modality `NTConfig` kwargs (`n_layer=12, n_head=10`, EEG
  `n_embd=200/patch_size=200`, EOG/ECG/EMG `n_embd=100/patch_size=100`).
- `/Users/boshra/NSRR-workspace/NSRR-tools/configs/phase0_physioomni_config.yaml`
  — full file read; Stage 1 real hyperparameters.
- `/Users/boshra/NSRR-workspace/NSRR-tools/configs/phase0_physioomni_lora_config.yaml`
  — full file read; Stage 2 real hyperparameters (`lora:` section, revised
  `training:` budget with dated justification comments).
- `/Users/boshra/NSRR-workspace/NSRR-tools/scripts/extract_physioomni_embeddings.py:1-120`
  — module docstring and header confirming output shape `[T,500]` float16,
  channel-loader delegation, SHHS single-channel handling.
- File existence for the full new-file set (dataset classes, training/
  inference/precompute scripts, configs, registries, job scripts) confirmed
  via `find ... -iname "*physioomni*"` against
  `/Users/boshra/NSRR-workspace/NSRR-tools/` — 29 PhysioOmni-specific files
  present in the current working tree (not just referenced in docs),
  confirming the user's statement that this work is merged into the main
  working copy, not stranded on an unmerged branch.

**Note on staleness handled per task instructions**: several passages in
`TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` (e.g. §1's "Implementation now
lives in `/home/boshra95/NSRR-tools-omni`," a separate git worktree,
`physioomni-implementation` branch "not yet merged to `main`" as of an
earlier date) describe a cluster-side worktree/branch state from mid-project
that predates the user's 2026-09-19 statement that PhysioOmni is merged into
this repo's main working copy. This file's findings are drawn from the
technical content of those docs (architecture, config values, results,
incident timeline), not from their now-superseded branch/worktree-location
claims — the file-existence check above independently confirms the merged
state directly, rather than relying on any doc's location claims.
