# OSF Supplementary Findings (for npj Digital Medicine supplementary material)

> Research notes only, feeding a later drafting step. Every concrete claim
> below is cited to a specific file:line (or a specific measured number in a
> tracked doc) in `docs/TSFM_SOURCE_CITATIONS` style at the end (§8). OSF
> implementation lives on `main` in `NSRR-tools` as of 2026-09-19 (merged
> from the `osf-implementation` branch; do not trust any older note in these
> docs claiming it is "not yet merged" — that was true only as of 2026-09-07).

---

## 1. Model identity & checkpoint

- **Name**: OSF ("On Pre-training and Scaling of Sleep Foundation Models"),
  backbone variant `vit_base`, DINO-style self-supervised pretraining
  objective (checkpoint metadata field `model_name: 'dino'`).
- **Paper**: ICML 2026, arXiv:2603.00190. **Peer-reviewed** (ICML 2026
  acceptance) — the only one of the three additional TSFM baselines
  (OSF/PhysioOmni/Mantis) that is peer-reviewed; PhysioOmni is arXiv-only
  (v3, 2026-03) and Mantis is arXiv-only (arXiv:2502.15637), neither
  peer-reviewed.
- **Params**: 85,325,568, strict-load-verified against the released
  checkpoint (zero missing/unexpected `state_dict` keys).
- **License**: MIT, confirmed by reading the `LICENSE` file directly in the
  reference clone (`OSF-Open-Sleep-FM/LICENSE`, Copyright 2026 Health
  Intelligence Lab @ UCLA, standard unmodified MIT boilerplate, no
  additional restrictions). This is the cleanest license of the three
  additional TSFM baselines (PhysioOmni's code repo has no LICENSE file at
  all, though its HF weights repo separately declares CC-BY-4.0; Mantis is
  Apache-2.0).
- **Checkpoint source**: HuggingFace Hub, `yang-ai-lab/OSF-Base`, file
  `osf_backbone.pth` (325 MB / 341,360,652 bytes). Note: the file was
  originally `torch.save`'d under the name `dino_vit_base_backbone.pth`
  (confirmed by inspecting the archive's internal folder name,
  `dino_vit_base_backbone/`) and later renamed for the public HF upload —
  `demo.ipynb` in the upstream repo still references the old name, which is
  stale; `snapshot_download`/`hf_hub_download` actually produce
  `osf_backbone.pth`.
- **Pretraining corpus / domain**: sleep PSG, self-supervised (DINO-style),
  across 9 named cohorts split into an in-domain/pretrain group (**SHHS**,
  NCHSDB, WSC, CCSHS, CFS) and an out-of-domain/downstream-eval group
  (MrOS, MESA, CHAT, SOF). Of our 4 test cohorts (APPLES, SHHS, MrOS,
  STAGES): **SHHS is severely contaminated** (87.7% of our SHHS test
  subjects — 1,114/1,271, apnea_binary/80m — are directly in OSF's own
  pretrain train+valid splits, exact subject-ID match against OSF's shipped
  `osf/splits/patient_pretrain_{train,valid,test}_ids.csv`); **MrOS is
  downstream/eval-only in OSF's own pretraining** (lower risk, not
  pretraining exposure); **STAGES and APPLES are both confirmed clean**
  (zero exact-ID overlap; STAGES also confirmed by a zero-match search of
  its real site-code naming, e.g. `STNF`, `MSTR`, `GSDV`, `MAYO`, across all
  6 of OSF's shipped split files). Full detail and mitigation strategy in
  §7 below (brief, since this is covered at length elsewhere in the
  project's docs and, presumably, in the paper itself).
- **Checkpoint metadata** (read directly from the loaded checkpoint dict,
  not guessed): `{'model_name': 'dino', 'encoder_name': 'vit_base',
  'num_leads': 12, 'patch_size_time': 64, 'patch_size_ch': 4, 'lead_wise':
  1, 'sample_rate': 64, 'window_size_sec': 30, 'seq_len': 1920, 'width':
  768, 'depth': 12}`.

---

## 2. Data adaptation / preprocessing pipeline

- **HDF5 tree used**: the **full-channel** tree
  (`/scratch/boshra95/psg_full/{dataset}/derived/hdf5_signals/`), NOT the
  paper-primary reduced/fast-channel tree. Decision reasoning: OSF's
  12-channel input requires snore and full thoracic/abdominal/airflow
  channels that only the full-channel preprocessing carries. Consequence:
  **OSF is compared against SleepFM's `phase0_v3_full` results, not the
  paper's primary `phase0_v3` (fast-channel) headline numbers** — this must
  be stated explicitly wherever OSF numbers appear next to SleepFM numbers.
  (PhysioOmni and Mantis, by contrast, both use the fast/reduced-channel
  tree and compare against `phase0_v3`.)
- **Channel mapping** (our HDF5 name → OSF's fixed 12-channel input, in the
  exact order OSF requires — order matters, it's fed positionally into a
  `Conv2d` patch layer): `ECG, EMG_Chin, EMG_LLeg, EMG_RLeg, ABD, THX, NP,
  SN, EOG_E1_A2, EOG_E2_A1, EEG_C3_A2, EEG_C4_A1`.

  | OSF slot | Primary source (our HDF5) | Fallback |
  |---|---|---|
  | `ECG` | `EKG` | `ECG-L` |
  | `EMG_Chin` | `CHIN` | generic `EMG` |
  | `EMG_LLeg` | `LLEG` | — |
  | `EMG_RLeg` | `RLEG` | — |
  | `ABD` | `ABD` | — |
  | `THX` | `Thor` | — |
  | `NP` (airflow) | `Airflow` | — |
  | `SN` (snore) | `Snore` | — |
  | `EOG_E1_A2` | `LOC` | — |
  | `EOG_E2_A1` | `ROC` | — |
  | `EEG_C3_A2` | `C3-M2` | — |
  | `EEG_C4_A1` | `C4-M1` | — |

  Confirmed exactly in `configs/phase0_osf_config.yaml`'s
  `data.channel_candidates` block (lines 51-63), which matches the plan
  doc's original 50-subject-audit table.

- **Real per-cohort channel completeness** (50-subject-per-cohort audit,
  code-verified against the real HDF5s, later confirmed structural — not
  per-subject noise — against the full extraction run's
  `_channel_fill_log.jsonl` files): APPLES has **0% `EMG_RLeg`**; SHHS has
  **0% `EMG_LLeg`/`EMG_RLeg`/`SN`**, only 68% `NP`, and **0% distinguishable
  `EEG_C3_A2`/`EEG_C4_A1`** (SHHS's HDF5s carry only a single generic `EEG`
  channel — no left/right central-electrode distinction survives our
  preprocessing for this cohort); MrOS has **0% `ABD`** and **0% `SN`**
  (otherwise the cleanest cohort, 100% on 9 of 12 slots); STAGES has only
  56% `EMG_LLeg`/`EMG_RLeg` and 90% `ECG`.
- **SHHS-specific approximation (provisional, flagged, not walked back)**:
  because SHHS has no distinguishable C3/C4, its single generic `EEG`
  channel is **duplicated into both `EEG_C3_A2` and `EEG_C4_A1` slots**,
  and `EMG_LLeg`/`EMG_RLeg`/`SN` are zero-filled. This means **5 of OSF's
  12 input slots are either always-zero or a duplicated approximation for
  every SHHS subject** — a material degradation of SHHS's input quality
  independent of, and compounding, SHHS's separate pretraining-contamination
  problem. A separate investigation
  (`docs/OSF_CHANNEL_REPROCESSING_PLAN.md`) found and root-caused three
  additional, currently-**unfixed** channel-coverage gaps in the shared
  upstream preprocessing (not OSF-specific code): MrOS's 0% `ABD` is a
  confirmed **code bug** (raw data has it in ~100% of subjects; a
  `raw.pick()` call chokes on a duplicate-channel-name collision and
  silently drops it), STAGES's `EMG_LLeg`/`EMG_RLeg` gap is a simple
  missing-alias bug (`LAT`/`RAT`, standard PSG terminology for the leg-EMG
  electrode site, present in 23.2% of STAGES subjects and not in the alias
  list at all), and SHHS's `NP` gap is likewise a missing-alias bug (a
  `NEW AIR` channel-name family, present in up to ~62% of subjects,
  unrecognized). Fixing all three would require re-preprocessing ~14,000
  subjects (MrOS+STAGES+SHHS) from raw EDF — judged out of scope for this
  round and **deliberately deferred**, not silently dropped; a full
  fork-not-edit implementation plan exists and is ready to execute if a
  reviewer's concern (or a degraded-result investigation) makes it worth
  the compute.
- **Normalization**: no additional per-channel normalization is applied by
  OSF beyond a `clamp(-6, 6)` on 3 specific vital-sign channels (HR, SpO2,
  OX) not present in our 12-channel list — meaning OSF expects its 12 input
  channels to arrive already roughly zero-mean/unit-variance. Our own
  preprocessing already z-scores per channel, so this is **compatible by
  construction**, not inverted or rescaled (unlike PhysioOmni, which
  required inverting our z-scoring back to raw amplitude).
- **Windowing / patching**:
  - Raw signal resampled 128 Hz → 64 Hz via exact 2:1 decimation (`x[::2]`)
    — chosen as mathematically identical to OSF's own linear-interpolation
    resample for this exact ratio, replicating OSF's own preprocessing
    pipeline order (resample → select/reorder/zero-fill 12 channels →
    pad/truncate to 1920 samples → clamp).
  - **Epoch length**: fixed 30 seconds (1,920 samples at 64 Hz) — this is
    OSF's architecturally forced unit; there is no cross-epoch attention
    anywhere in the model (confirmed: the positional-embedding table
    `pos_embedding` is sized to exactly `N_max + 1` = 91 tokens = one
    epoch's worth, `osf/backbone/vit1d_cls.py:184`).
  - **Patchify geometry**: `lead_wise=1` mode (2D `Conv2d` patchify, not the
    1D path) with `patch_size_ch=4`, `patch_size_time=64` →
    `Lr = 12/4 = 3` channel-groups × `Nt = 1920/64 = 30` time-patches = 90
    tokens + 1 CLS token = 91 total, exactly matching the checkpoint's
    `pos_embedding` shape `(1, 91, 768)` (`osf/backbone/vit1d_cls.py:172-186`).
  - **No architecture surgery was needed** — the checkpoint's own
    architecture parameters (read from its metadata dict) were used
    directly to instantiate `vit_base(...)`; no positional-buffer
    regeneration or resizing was required (contrast with Mantis, whose
    released 512-sample positional buffer had to be regenerated to cover a
    full 30s epoch at 128 Hz).
- **Embedding extraction pipeline**:
  - Script: `scripts/extract_osf_embeddings.py` (359 lines).
  - Per-epoch forward pass: `ViT.forward_encoding(x, return_sequence=False)`
    → `(cls: [B,768], patches: [B,90,768])`
    (`scripts/extract_osf_embeddings.py:200`); patches are then mean-pooled
    over the 90-token axis (`extract_osf_embeddings.py:201`,
    `patches.mean(dim=1)`) and stacked as `[cls, mean_pooled_patches]` →
    `[B, 2, 768]` per epoch. **Not** the raw 91-token undivided sequence and
    **not** CLS-only — a deliberate choice made to preserve some patch-level
    information beyond the CLS summary token without saving the full,
    much-larger token sequence.
  - Epochs are batched for GPU throughput via a config-driven
    `chunk_batch_size` (default 16 for Stage 1 extraction —
    `configs/phase0_osf_config.yaml:16`; raised to 64 for Stage 2, see §5).
  - Output: `{output_dir}/{dataset}/{subject_id}.npy`, dtype **float16**,
    shape **`[T_epochs, 2, 768]`** (`T_epochs` = number of complete
    non-overlapping 30s epochs in the recording; incomplete trailing epochs
    dropped). Output location:
    `/scratch/boshra95/psg_full/unified/embeddings/osf_30sec/`.
  - A per-dataset `_channel_fill_log.jsonl` records which OSF input
    channels were zero-filled/substituted per subject, used to regenerate
    the real-population channel-completeness numbers in §2 above.

---

## 3. Architecture summary

- **Params**: 85,325,568 (backbone only; strict-load-verified, zero
  missing/unexpected keys).
- **Fusion approach**: **single joint tensor** across all 12 channels per
  forward call — not a per-modality/per-channel independent-encoder design
  (contrast with PhysioOmni's 4 fully independent per-modality encoders and
  Mantis's channel-independent `Conv1d(in_channels=1)` design). This is the
  same "one fused backbone call" pattern SleepFM uses, structurally
  advantaged for GPU utilization compared to a multi-encoder split (see §6).
- **Hidden dim**: `width=768` (a well-shaped, tensor-core-friendly power of
  2 — favorable compared to PhysioOmni's d=100/200).
- **Depth**: 12 transformer blocks (`depth=12`).
- **Tokenizer / patchify**: 2D `Conv2d` patch embedding (`lead_wise=1`
  mode), `patch_size_ch=4` × `patch_size_time=64` → 3 channel-groups × 30
  time-patches = 90 patch tokens + 1 CLS token = 91 tokens per 30-second
  epoch (`osf/backbone/vit1d_cls.py:174-187`, `:207-221`).
- **Attention block internals** (relevant to §5's LoRA target modules): each
  block's `Attention` submodule (`osf/backbone/vit1d_cls.py:68-106`) has
  exactly two `Linear` layers, `self.to_qkv` (line 86) and
  `self.to_out[0]` (inside a `Sequential`, line 89) — these are the only
  two Linear layers in the attention path, and are wrapped in a `PreNorm`
  module (`self.attn = PreNorm(dim=..., fn=attn)`), so the
  fully-qualified module path at runtime is `block{i}.attn.fn.to_qkv` /
  `block{i}.attn.fn.to_out.0`, not `block{i}.attn.to_qkv` — PEFT matches by
  name suffix, so this doesn't change the LoRA config but is worth knowing
  if inspecting `named_modules()` directly.
- **No cross-epoch attention or aggregation of any kind** anywhere in
  `osf/backbone/`, `osf/models/`, or `main_pipelines/` — confirmed by
  keyword grep (`hierarchical|long_sequence|full_night|multi_epoch|
  sequence_model`, zero hits) and by the positional-embedding table being
  sized to exactly one epoch. This means OSF is Plan-B-only in this
  project's Plan A/B/C framework (short-segment embedder + our own sequence
  head on top) — there is no "native long context" condition to report for
  OSF, exactly the same situation as PhysioOmni and Mantis.

---

## 4. Stage 1 — Frozen encoder + downstream head training

- **Frozen vs. trained**: the entire 85.3M-parameter OSF backbone is
  **frozen** — embeddings are precomputed once (§2) and never touched again
  during Stage 1. Only a lightweight sequence head (LSTM/Transformer/
  MeanPool, from `src/nsrr_tools/models/sequence_head.py`, reused
  unmodified — dim-agnostic factory, no fork needed) is trained from
  random initialization.
- **New files created** (all new, nothing from the existing SleepFM
  pipeline edited):
  - `configs/phase0_osf_config.yaml`
  - `scripts/extract_osf_embeddings.py`, `jobs/extract_osf_embeddings_gpu.sh`
  - `src/nsrr_tools/datasets/osf_context_window_dataset.py`
    (`OSFContextWindowDataset`)
  - `scripts/train_osf_context_sweep.py`, `jobs/train_osf_context_sweep_gpu.sh`
  - `scripts/infer_osf_subject_windows.py`, `jobs/infer_osf_subject_windows_gpu.sh`
  - `experiments/v2_osf_registry.yaml`, `scripts/gen_commands_osf.py`
- **Real hyperparameters** (`configs/phase0_osf_config.yaml`, verified
  directly, not approximated):
  - `model.input_dim = 1536` (= 2 subtokens × 768; only value that differs
    from SleepFM's own seq2label head config, which uses `input_dim=512`),
    `hidden_dim = 128`, `num_layers = 1`, `num_heads = 8`, `dropout = 0.3`.
    Architecture otherwise **held constant** vs. SleepFM's own head config —
    a deliberate "only the encoder/channels change" comparison principle.
  - `training.epochs = 40`, `lr = 1.0e-4`, `weight_decay = 1.0e-3`,
    `early_stopping_patience = 10`, `early_stopping_monitor = "val_auroc"`,
    `class_weights = "auto"`, `mixed_precision = false`,
    `weighted_sampler = false`.
  - `context_lr_overrides`: `120m` and `240m` both use `lr = 5.0e-5` (half
    the base rate).
  - `dataset.windows_per_subject = 5` (K=5 windows sampled per subject at
    train/val time), `train_split/val_split/test_split = 0.70/0.15/0.15`,
    `split_seed = 42` — the **same** seed/ratios as the SleepFM
    `phase0_v3_full` runs (required, not optional, for a fair comparison on
    matched subjects — though see §7 for a real, documented split-mismatch
    caveat between the two pipelines' actual subject populations).
  - `min_recording_patches = 480` (the cohort-inclusion floor, in OSF's
    30-second-epoch units; equivalent to SleepFM's `2880` in its own
    5-second-patch units — both represent the same 240-minute minimum
    recording length).
  - Two config fields (`training.optimizer: "adamw"`,
    `training.scheduler: "cosine"`, `training.device: "cuda"`) are **dead
    config keys, never actually read** by `train_osf_context_sweep.py` —
    the optimizer is hardcoded to plain `torch.optim.Adam` (not decoupled
    AdamW) and the scheduler to `CosineAnnealingLR`; this is pre-existing
    behavior inherited from the SleepFM config template, not an
    OSF-specific bug, but worth knowing before assuming these fields do
    anything.
- **Compute characteristics for Stage 1: cheap.** Because embeddings are
  precomputed once and the backbone never enters the trainable graph,
  Stage 1's compute profile is essentially identical to SleepFM's own
  frozen-embedding sequence-head training — no measured TFLOP/s number
  exists for Stage 1 specifically because it was never expensive enough to
  warrant profiling (contrast sharply with Stage 2, §6). Wall-time lookup
  tables in `gen_commands_osf.py` remain uncalibrated placeholder copies of
  SleepFM's own tables (never GPU-recalibrated), but this was judged low
  risk since an underestimate only costs one auto-requeue, not lost work.
- **Coverage completed** (as of the last update to
  `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`, 2026-08-13): **all 5 Tier-1
  tasks** (`sex_binary`, `sleep_efficiency_binary`, `apnea_binary`,
  `bmi_binary`, `age_class`) **× 2 of 3 heads** (`lstm`, `transformer`) **×
  all 6 context lengths** (30s, 10m, 40m, 80m, 120m, 240m) — trained,
  inferred (test split, K-dense, bootstrap CIs), and collected into
  `results/collected/phase0_osf/{training,analysis}.csv`. **`mean_pool`
  head was never run for any task** — an explicitly flagged, still-open
  gap, not a silent omission.

---

## 5. Stage 2 — LoRA fine-tuning

- **Frozen vs. trained**: the OSF backbone is wrapped with LoRA adapters
  and fine-tuned jointly with a sequence head that is **warm-started** from
  the matching Stage 1 checkpoint (LP-FT procedure, Kumar et al. 2022
  justification) rather than randomly initialized — this is a staged, not
  joint, training procedure by explicit design.
- **LoRA config** (`configs/phase0_osf_lora_config.yaml`'s `lora:` section,
  live-verified against the real checkpoint, not just read from config):
  - `target_modules: ["to_qkv", "to_out.0"]` — the only two Linear layers
    in OSF's `Attention` block (§3); PEFT matches by name suffix, correctly
    hitting all 12 transformer blocks.
  - `r = 8`, `lora_alpha = 16`, `lora_dropout = 0.05`.
  - `modules_to_save = ["sequence_head"]`.
  - **Exact trainable-parameter count, live-verified**: `peft.get_peft_model`
    injects **96 LoRA-parameter submodules** across all 12 transformer
    blocks, **442,368 / 85,767,936 base params trainable ≈ 0.52%** (before
    counting the sequence head's own ~2.1M parameters, which are fully
    trainable as `modules_to_save`).
  - `r`/`lora_alpha`/`lora_dropout` are explicitly flagged in the config
    itself as **standard-practice defaults, not calibrated against a real
    pilot** — never revisited/tuned in this project.
- **New files created** (all new; nothing from Stage 1 or SleepFM edited):
  - `configs/phase0_osf_lora_config.yaml`
  - `src/nsrr_tools/datasets/osf_channel_loader.py` (shared
    channel-mapping/resampling utility, factored out of the Stage 1
    extraction script so the logic isn't duplicated)
  - `src/nsrr_tools/datasets/osf_raw_epoch_dataset.py`
    (`OSFRawEpochWindowDataset`)
  - `scripts/precompute_osf_raw_signal_cache.py`,
    `jobs/precompute_osf_raw_signal_cache.sh`
  - `scripts/train_osf_lora.py`, `jobs/train_osf_lora_gpu.sh`
  - `scripts/infer_osf_lora_subject_windows.py`,
    `jobs/infer_osf_lora_subject_windows_gpu.sh`
  - `experiments/v2_osf_lora_registry.yaml`,
    `scripts/gen_commands_osf_lora.py`
- **Key Stage 2 hyperparameters, real values**
  (`configs/phase0_osf_lora_config.yaml`, revised 2026-08-15 from Stage
  1's original placeholder values after observing real overfitting):
  `epochs = 18` (was 40), `lr = 5.0e-5` (was 1.0e-4), `early_stopping_
  patience = 5` (was 10), `context_lr_overrides` at `2.5e-5` for `120m`/
  `240m` (rescaled to stay half of the new base). Justification: the first
  completed real 30s LoRA run at the old `lr=1e-4` showed clear overfitting
  — best `val_auroc` at epoch 9 (0.7193), declining/noisy through epoch 16
  (0.701–0.713) while `train_bal_acc` kept climbing (0.677→0.771). `lr` was
  halved partly because LoRA fine-tuning generally needs gentler updates
  than head-only training from random init, and partly because the same
  learning rate was otherwise being applied uniformly to both the
  freshly-injected LoRA matrices and the already-converged,
  Stage-1-warm-started head — a real, **unresolved** simplification (a
  discriminative per-parameter-group learning rate would be more correct
  but was not implemented).
- **`chunk_batch_size` tuning history, with before/after numbers**: raised
  from **16 → 64** (`configs/phase0_osf_lora_config.yaml:38`), a confirmed,
  measured **3.28× speedup** (18.8 vs. 61.6 min/epoch at 30s context) —
  purely a GPU-batching/scheduling knob (how many raw 30s epochs get
  batched into one backbone forward call within a training window), safe
  to change without affecting `batch_size`/`effective_batch_size` or the
  forward pass's mathematical result, since OSF's backbone uses only
  per-sample-independent `nn.LayerNorm` (zero `nn.BatchNorm` anywhere).
- **Mixed-precision / TF32 decisions, and why**: `mixed_precision` (AMP)
  was tried and explicitly **reverted** — measured **zero speedup** on a
  `1g.10gb` pilot (61.7 vs. 59.9 min/epoch), not worth the fp32-vs-mixed-
  precision numerical asymmetry against Stage 1/SleepFM's plain-fp32
  training for no measured benefit. **TF32** (`torch.backends.cuda.matmul.
  allow_tf32`) was investigated separately (2026-08-27,
  `docs/LORA_GPU_THROUGHPUT_INVESTIGATION.md`) and confirmed **off** by
  PyTorch's own default (`allow_tf32=False`, `float32_matmul_precision=
  "highest"`) in both training scripts — never enabled. The investigation's
  verdict: TF32 is **likely real but context-length-dependent** — expected
  close to zero benefit at 30s/10m (the same overhead-bound regime where
  AMP and a 3× MIG-slice GPU upgrade both already measured zero speedup),
  with an honest, unmeasured guess of 15-40% at 40m and beyond where
  compute genuinely dominates wall time — **not applied, a deliberate
  "decide later, don't default into it" state**, not an oversight.
- **GPU allocation history**: started on `1g.10gb` MIG slice, upgraded to
  `3g.40gb` on 2026-08-15 (`jobs/train_osf_lora_gpu.sh`, the largest MIG
  slice available on this cluster — no full unpartitioned H100 exists here)
  — this upgrade (3× more compute/memory) measured **zero speedup** at 30s
  context (~61.6 min/epoch both ways over 12 real epochs), confirming the
  bottleneck was per-call batching granularity (`chunk_batch_size`), not
  raw compute availability, in the overhead-bound short-context regime.
- **Warm-start / staging strategy across context lengths** (real design
  decision, made necessary by a hard compute-scaling finding, not a
  convenience): **compute scales linearly with context length N** (raw
  30-second epochs per window), because `CombinedOSFLoRAModel.forward()`
  runs every epoch in a window through the full LoRA-adapted backbone
  individually — there is no cross-epoch attention to exploit, so this is
  architectural, not a tuning artifact. Measured: 30s ≈ 18.8 min/epoch
  (post-`chunk_batch_size` fix); linear extrapolation put 240m (480× more
  epochs/window than 30s) at roughly 20 days/epoch on the pre-fix hardware
  — **independently fine-tuning every context length from scratch is not
  achievable within a normal project timeline.** The chosen fix:
  **every context length other than 30s warm-starts its LoRA adapters +
  head from that same (task, head)'s own fully-converged 30s LoRA
  checkpoint** — a **branch**, not a sequential chain (30s→10m→40m→...).
  30s itself still warm-starts from Stage 1's frozen-backbone head, as
  before. Branching (not chaining) was chosen because OSF's backbone has no
  per-length internal state (it processes one 30-second epoch at a time
  regardless of which window it belongs to), so there is no structural
  reason to expect a chained lineage to transfer better than one common
  reference point, and a chain would make every length's result depend on
  the arbitrary sweep order — an uncontrolled variable. **A genuine,
  explicitly-flagged limitation, distinct from Stage 1's purity**: for the
  frozen condition (Stage 1), context length N really is the only variable
  — every head trains from the same random init against the same frozen
  backbone. For the LoRA condition (Stage 2), the *backbone's starting
  point* is now shared/inherited across lengths (all descending from the
  30s fine-tune), not independently random-initialized per length the way
  Stage 1's heads are — a deliberate departure made necessary by compute
  constraints, and this should be stated plainly as a Stage-2-specific
  methodological limitation, not implied to have the same "N is the only
  variable" purity Stage 1 has. (The alternative of "train once at one
  length, evaluate everywhere with no further tuning" was considered and
  explicitly rejected — it would have confounded "does LoRA help at length
  L" with "does a backbone tuned elsewhere transfer to length L," and every
  context length in the reported results really is independently
  fine-tuned to its own convergence, just from a shared starting point.)
- **Effective batch size / gradient accumulation**: `effective_batch = 32`
  throughout (same convention as Stage 1/SleepFM, restored after an
  initial, user-rejected decision to skip it). `context_micro_batch`
  (`experiments/v2_osf_lora_registry.yaml`) is **not flat** — `30s: 32,
  10m: 32, 40m: 8, 80m: 4, 120m: 2, 240m: 1`, with `accum_steps` rising to
  compensate and keep `effective_batch=32` exactly. This is not a
  placeholder guess for the shorter contexts: **`40m`/`80m`/`120m`/`240m`
  all confirmed-OOM'd at `micro_batch=32`** on real GPU jobs (38.05/39.5
  GiB allocated, deep inside the backbone's attention layer — retained
  activations scale with `batch_size × raw_epochs`, not just
  `chunk_batch_size`), so the lower values target the same ~640-raw-epoch
  memory ceiling observed as safe at 10m — itself an extrapolation, not
  independently re-measured at every context.
- **Coverage completed and exact stopping point/reason**: **all 5 Tier-1
  tasks × 2 heads (lstm, transformer) — 10 runs total — trained fully
  through `30s → 10m → 40m → 80m → 120m`.** **`240m` was never started for
  any of the 10 runs** — a **deliberate, uniform, cost-curve-informed
  stopping point** (running it for all 10 would cost roughly 2× the 120m
  wall-time per run, judged not worth it for a first LoRA-vs-frozen
  comparison pass), applied identically to every task/head combination —
  not an arbitrary or partial gap. `mean_pool` head and val-split
  threshold-tuning were also never run for Stage 2, matching the same
  outstanding gaps already present in Stage 1. Everything that *was* run
  went all the way through: train → resumable full-window inference (test
  split) → `analyze --k-dense` → `collect`, landing in
  `results/collected/phase0_osf_lora/{training,analysis}.csv` in the same
  schema as Stage 1's and SleepFM's collected CSVs. Headline result: every
  single task/head combination improves **monotonically** with context
  length through 120m, e.g. `apnea_binary`/lstm goes 0.719 (30s) → 0.761 →
  0.812 → 0.840 → 0.870 (120m) on test.

---

## 6. Honest computational drawbacks

**The core mechanism**: LoRA fine-tuning here means running the *raw
signal*, not pre-extracted embeddings, through the LoRA-adapted backbone
for every 30-second epoch inside a training window (a frozen-encoder-only
pipeline cannot backpropagate into the backbone). `CombinedOSFLoRAModel.
forward()` runs every epoch in a window through the full backbone
individually, so a 240-minute window (480 raw epochs) costs roughly 480×
what a 30-second window (1 epoch) costs per training step. **This scaling
is architectural** (no cross-epoch attention anywhere in OSF to amortize
against), not an implementation inefficiency, and the same mechanism would
apply to any similarly-fine-tuned model (explicitly flagged as relevant to
Mantis's own eventual LoRA stage too).

**Measured throughput, real, single-uninterrupted-training-segment data
only** (cross-checked via `resume.pt` + `sacct`, not the buggy
`metrics.json` aggregate — see the unresolved bug below):

| Context | Raw epochs/window | min/epoch | Est. TFLOP/s |
|---|---|---|---|
| 30s | 1 | 18.8 | ~2.0 |
| 10m | 20 | 58.9 | ~12.8 |
| 40m | 80 | 112.3 | ~18.5 |
| 80m | 160 | 217.2 | ~19.2 |

(TFLOP/s from a hand FLOP estimate for one ViT block — d=768, 90 tokens per
30s epoch, 12 blocks, ~3× for forward+backward ≈ 46.8 GFLOP per raw epoch,
ignoring the sequence head and patch embedding — so absolute values are
order-of-magnitude only; the *trend* is the trustworthy part, since formula
errors cancel in the ratio.)

**The specific mechanism: short contexts are overhead-bound, long contexts
are compute-bound.** GPU utilization jumps **~6.4×** from 30s to 10m, then
**plateaus** (+3% only) from 40m to 80m — meaning generic fixes that seem
obviously helpful ("cost scales with context, so more compute/precision
should help") measured **zero speedup when tried at the wrong regime**: the
`1g.10gb → 3g.40gb` MIG upgrade (3× more compute) and enabling
`mixed_precision` were both tested at 30s and both measured **zero**
speedup there, because at 30s a single, mostly-empty backbone call's fixed
per-call overhead (kernel launch, the Python-side chunking loop) dominates
wall time regardless of how much raw compute is available. The fix that
*did* work was purely a GPU-scheduling knob: raising `chunk_batch_size`
16→64 gave a confirmed **3.28× speedup** (18.8 vs. 61.6 min/epoch at 30s)
by amortizing that same fixed per-call overhead across more epochs per
forward call. Against a ~28.7 TFLOP/s FP32-dense peak for a `3g.40gb` MIG
slice, the 40m/80m plateau (~19 TFLOP/s) is roughly 65% of that peak — a
high-enough estimate that the absolute number shouldn't be over-trusted,
but consistent with the mechanism.

**What was tried and didn't help (with numbers)**:
- `mixed_precision` (AMP): 61.7 vs. 59.9 min/epoch at 30s on `1g.10gb` —
  effectively zero speedup, reverted.
- `1g.10gb → 3g.40gb` GPU upgrade (3× compute/memory): ~61.6 min/epoch both
  ways at 30s over 12 real epochs — zero speedup, kept anyway since it was
  a free/no-cost change once made, but not the fix that mattered.

**What was tried and did help**: `chunk_batch_size` 16→64, a confirmed
3.28× speedup at 30s (above) — the single most consequential lever found in
this investigation.

**Not yet applied (deliberately, evaluated but deferred)**: enabling TF32
(`torch.backends.cuda.matmul.allow_tf32=True`) — investigated 2026-08-27,
verdict "likely real but small at 30s/10m, honestly-guessed 15-40% at
40m+, not multiples," low numerical risk (TF32 keeps full FP32
storage/exponent range, only truncates mantissa bits) but still introduces
*some* asymmetry with Stage 1/SleepFM's plain-fp32 training. **Decided
against applying without more evidence** — a cheap `torch.profiler` run at
a longer context (40m, not 30s) to directly separate matmul-kernel time
from overhead was proposed as the cheapest way to settle this, but **was
never run**. Requesting a whole (unpartitioned) H100 card instead of the
`3g.40gb` MIG slice was also considered and **recommended against** — the
same overhead-bound-at-30s finding that made the 1g→3g upgrade a
zero-speedup change argues a further 2.33× jump to a full card is unlikely
to help for the same reason, and it costs real shared-account allocation
for an unproven benefit.

**Unresolved bug, explicitly flagged, not fixed**: `metrics.json`'s
`training_time_min` field is **wrong for any run that needed more than one
resume/requeue cycle**. Root cause in `train_osf_lora.py`: the wall-clock
timer `t0 = time.time()` resets on every script (re)invocation, including
mid-training resumes, but `history` (and therefore `n_epochs_run`) correctly
accumulates across all resume segments. So `training_time_min` only ever
covers the *final* resume segment's elapsed time while `n_epochs_run`
counts every epoch ever run — silently **undercounting** true wall-clock
cost for any multi-resume context. This was discovered while trying to
build a clean per-epoch timing table from completed `metrics.json` files
(the numbers looked nonsensical: 10m showing as faster than 30s, some
zeros) — the fix is straightforward (record cumulative time in `resume.pt`
and carry it forward) but has **not been implemented**; all real throughput
numbers reported above and in the paper's supplementary material were
therefore deliberately drawn only from single-uninterrupted-segment data
(`resume.pt` + `sacct` cross-checked), not from `metrics.json`'s own
`training_time_min / n_epochs_run` ratio, to avoid this bug.

**Architectural source of the inefficiency (not a driving-quality
problem)**: OSF's single joint 12-channel tensor per call and well-shaped
`d=768` hidden dimension are both *favorable* properties — the actual
issue is pure call **granularity**: one native call spans only 30 seconds
(~46.8 GFLOP, a hand estimate), small enough that fixed per-call overhead
dominates wall time unless many epochs are batched into one call. This is
a real gap in OSF's own reference implementation, not something the model
could avoid on its own — nothing in OSF's released code surfaces
`chunk_batch_size`-style batching as a tunable a user should check, and
the default value (16) left a measured 3.28× of throughput on the table
until a human noticed. Because there is no cross-epoch attention anywhere
in the backbone, all temporal aggregation over a long context is 100%
external (our own sequence head) — the backbone itself never does anything
smarter with more context than "run once per epoch, N times," so every
doubling of context is a literal doubling of backbone calls with zero
internal amortization the architecture provides on its own.

---

## 7. Other reviewer-relevant caveats

- **Contamination**: covered in full in §1 and elsewhere in the project's
  docs (`CLAUDE.md`'s "Honest comparison framing," `docs/
  TSFM_OSF_IMPLEMENTATION_PLAN.md`'s "Stage 1 Results"/"Precise
  contamination quantification" sections) — brief pointer here only.
  Headline: SHHS severely contaminated (87.7% exact-ID pretraining overlap,
  must never be blended into a pooled/headline AUROC number without a
  caveat); STAGES and APPLES confirmed clean; MrOS downstream/eval-only in
  OSF's own splits (lower risk, not pretraining exposure). Even
  individually-unseen SHHS subjects (the 157/1,271 not in any OSF
  pretraining split) show OSF's advantage over SleepFM **not shrinking**
  when recomputed on just that clean subset — evidence that
  subject-level filtering alone doesn't fully rescue SHHS as a fair
  comparison cohort, since the encoder was still trained on ~1,114 *other*
  SHHS subjects and may have deeply learned that cohort's recording/device
  characteristics at a cohort level, not just a subject level.
- **License**: MIT, confirmed (§1) — the cleanest of the three additional
  TSFM baselines.
- **Peer-review status**: published, ICML 2026 — the only one of the three
  additional baselines that is peer-reviewed (PhysioOmni and Mantis are
  both arXiv-only).
- **Split mismatch vs. SleepFM (real, documented, NOT fixed)**: SleepFM's
  `phase0_v3_full` results and OSF's Stage 1 results were computed on
  **subtly different train/val/test splits**, even though both use the
  identical code pattern (`np.random.default_rng(split_seed).shuffle()`
  over a filtered subject list) and the identical `split_seed=42`. The
  difference arises because each pipeline filters subjects by "has an
  embedding file" against a *different* embedding directory
  (`sleepfm_5sec` vs. `osf_30sec`), and their extraction success
  populations differ by a handful of subjects per cohort — enough to
  scramble `shuffle()`'s entire output permutation. Live-confirmed exact
  differences: APPLES has 1 subject (`APL0419`) with an OSF embedding but
  no SleepFM embedding; STAGES has 1 subject each way (`STLK00099`
  OSF-only, `STLK00096` SleepFM-only). SHHS and MrOS populations are
  identical between the two. **This means individual subjects can land in
  different train/val/test splits between the already-published SleepFM
  results and the already-published OSF Stage 1 results** — not a
  rounding-level discrepancy. **Deliberately not fixed** — fixing it would
  mean re-deriving one of the two already-completed, already-analyzed
  splits, invalidating existing results, which was judged out of scope
  unless explicitly requested. **OSF Stage 2 (LoRA) is, by contrast,
  deliberately anchored to match OSF Stage 1 exactly** (not SleepFM) via an
  explicit existence-check against Stage 1's own embedding directory before
  computing Stage 2's split — since Stage 2's own comparison target is
  Stage 1, not SleepFM directly.
- **N-count mismatches, not fully root-caused**: subject counts are close
  but not always identical between OSF and SleepFM at the same
  context/task even setting the split-mismatch issue aside (e.g.
  `apnea_binary`/80m: OSF N=2,077 vs. SleepFM N=2,054, driven mostly by
  STAGES: 227 vs. 200, a ~13% difference for that cohort). Most likely
  explained by embedding-extraction coverage differences rather than a
  deeper split problem, but this was **never root-caused** — flagged as
  worth verifying before over-trusting comparisons for STAGES-heavy tasks,
  though unlikely to flip the direction/rough magnitude of any per-cohort
  finding.
- **Channel-completeness caveats** (§2): SHHS's approximated/zero-filled
  input (5 of 12 slots degraded) is a real, quantified data-quality
  concern layered on top of, and separate from, its contamination problem
  — its actual downstream impact on OSF's SHHS results has **never been
  isolated** from the contamination effect. Three additional real,
  root-caused, currently-unfixed upstream channel-coverage bugs exist
  (MrOS `ABD` code bug, STAGES `LAT`/`RAT` alias gap, SHHS `NEW AIR` alias
  gap) with a ready-to-execute fix plan not yet run.
- **Task-category pattern worth flagging to a reviewer**: OSF's frozen
  encoder shows a real, credible, contamination-survives-scrutiny advantage
  specifically on tasks tied to static/structural subject characteristics
  (`sex_binary`, `bmi_binary`, `age_class` — the clean APPLES cohort shows
  a *larger* gap than the contaminated SHHS cohort in all three, the
  opposite of what contamination alone would produce), but is
  inconclusive-to-mixed on tasks tied to dynamic physiological events
  (`sleep_efficiency_binary`, `apnea_binary`) — not explainable by
  contamination alone, since STAGES (confirmed clean) also favors OSF on
  apnea while APPLES and MrOS (also clean) favor SleepFM. LoRA fine-tuning
  narrows or reverses part of this picture task-by-task (e.g. LoRA clearly
  helps apnea, +2 to +5pp at matched context, but *consistently
  underperforms the frozen encoder* for `sleep_efficiency_binary` at every
  context tested) — a genuinely non-uniform, non-cherry-picked finding
  worth reporting plainly rather than smoothing into a single verdict.
- **Comparison baseline choice**: OSF is compared against SleepFM's
  full-channel `phase0_v3_full` results, not the paper's primary
  fast-channel `phase0_v3` headline numbers (§2) — this must never be
  conflated in any table/caption, since the reduced-vs-full-channel gap is
  itself non-trivial (paper-documented, +0.03-0.05 AUROC for
  apnea/BMI-type tasks).

---

## 8. Source citations

**Documentation (read in full or in the cited ranges)**:
- `NSRR-tools/CLAUDE.md:267-736` — "TSFM Baseline Model Comparison" section:
  Plan A/B/C framework, code-reuse assessment, frozen-vs-LoRA staging
  rationale, contamination facts, status log.
- `NSRR-tools/docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` (full file, 2867
  lines) — primary source for architecture (`Encoder: OSF vit_base`
  section), channel mapping, Stage 1/Stage 2 results tables, full
  Implementation Checklist (Phase 0-3), Key Decisions table, Appendix §0-12
  (environment build log, checkpoint verification, per-item technical
  detail for every script/config/registry file).
- `NSRR-tools/docs/OSF_EXPERIMENTS_GUIDE.md:79-834` — Steps 0-8, operational
  commands/paths/output shapes for both Stage 1 and Stage 2, including
  Step 8.2b (raw signal cache precompute) and Step 8.3 (training, LoRA
  warm-start policy, GPU sizing history).
- `NSRR-tools/docs/OSF_CHANNEL_REPROCESSING_PLAN.md` (full file) — the
  MrOS `ABD` code bug, STAGES `LAT`/`RAT` alias gap, SHHS `NEW AIR` alias
  gap; deferred fix plan.
- `NSRR-tools/docs/TSFM_BASELINE_CANDIDATES.md:158-269` — §2.1, OSF
  candidate research: code-verified input format, checkpoint/license,
  classification-head API, LoRA/PEFT dependency status, contamination
  split breakdown (pretrain vs. downstream/eval).
- `NSRR-tools/docs/TSFM_MODEL_COMPARISON.md:1-241` (§0-2.1), `:482-781`
  (§3-4) — architecture/results comparison table, per-cohort OSF-vs-
  SleepFM AUROC tables (frozen + LoRA), honest computational-cost
  narrative (§3.1-3.3), intrinsic architectural inefficiency sources (§4,
  OSF subsection).
- `NSRR-tools/docs/LORA_GPU_THROUGHPUT_INVESTIGATION.md` (full file, 190
  lines) — TF32/GPU-allocation investigation, the `metrics.json` timing
  bug, real measured throughput table, verdict on TF32/whole-card
  requests.

**Code (read directly, not inferred from docs)**:
- `NSRR-tools/configs/phase0_osf_config.yaml` (full file, 172 lines) —
  Stage 1 real hyperparameters, channel mapping, `min_recording_patches`,
  `chunk_batch_size=16`.
- `NSRR-tools/configs/phase0_osf_lora_config.yaml` (full file, 220 lines)
  — Stage 2 real hyperparameters, `lora:` section, `chunk_batch_size=64`
  and its tuning-history comment, revised `epochs/lr/patience`.
- `NSRR-tools/experiments/v2_osf_lora_registry.yaml:1-104` — gradient
  accumulation/`context_micro_batch` real values and the OOM history
  comment explaining why they're not flat.
- `NSRR-tools/scripts/extract_osf_embeddings.py:14,31,155-206,268-318` —
  embedding extraction pipeline: `forward_encoding` call, mean-pooling,
  output shape/dtype/path, `chunk_batch_size` read.
- `NSRR-tools/scripts/train_osf_lora.py:190-241` —
  `build_combined_lora_model`, `LoraConfig` construction, `chunk_batch_size`
  read, `warm_start_head_from_stage1`'s `ModulesToSaveWrapper` handling.
- `NSRR-tools/src/nsrr_tools/datasets/osf_context_window_dataset.py:1-84`
  — module docstring (input file shape `[T,2,768]`), constants
  (`N_SUBTOKENS=2, EMBED_DIM=768, FLAT_DIM=1536, PATCH_SECONDS=30,
  PATCHES_PER_EPOCH=1`).
- `/Users/boshra/NSRR-workspace/OSF-Open-Sleep-FM/osf/backbone/vit1d_cls.py:68-106,145-270`
  — `Attention`/`PreNorm`/`ViT` classes: `to_qkv`/`to_out` module names,
  `pos_embedding` sizing (`N_max+1`), `to_tokens_2d`/`forward_encoding`/
  `forward_avg_pool` implementations, patch geometry math.
- `/Users/boshra/NSRR-workspace/OSF-Open-Sleep-FM/LICENSE:1-3` — MIT
  license confirmation, UCLA Health Intelligence Lab copyright.

**Repo state verification**:
- `git -C NSRR-tools branch -a` / `git log --oneline -15` (run
  2026-09-19) — confirmed all OSF files (`configs/phase0_osf*.yaml`,
  `scripts/*osf*.py`, `src/nsrr_tools/datasets/osf_*.py`,
  `experiments/v2_osf*.yaml`, `jobs/*osf*.sh`) exist on `main`, confirming
  the user's statement that the `osf-implementation` branch has been
  merged (superseding `CLAUDE.md`'s and the implementation plan's own
  "still not merged as of 2026-09-07" notes, which are now stale).

**Explicitly unverified / could not confirm from available sources**:
- Whether `torch.profiler`-based settlement of the TF32
  compute-vs-overhead-bound question (proposed in
  `LORA_GPU_THROUGHPUT_INVESTIGATION.md` §6) was ever actually run — the
  doc states it was "not yet run" as of 2026-08-27 and no later doc update
  or code artifact was found confirming it happened since.
- The exact current cluster-side completion state of the channel-fill-log
  re-audit against the *real, full-population* extraction run (the plan
  doc's checklist item 1.11 is marked unchecked) — this write-up reports
  the last-documented completeness numbers, not a live re-verification.
- Whether `docs/npj_paper_md_files/TSFM_SUPP_IMPLEMENTATION_PLAN.md` (an
  untracked file found alongside this one during research, `git status`)
  contains newer information than what's synthesized here — it was not
  part of the assigned reading list and was deliberately not read or relied
  upon, per the instruction to read only the specified files.
