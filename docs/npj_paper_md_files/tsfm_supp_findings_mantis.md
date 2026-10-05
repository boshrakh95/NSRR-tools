# TSFM Supplementary Findings: Mantis

> **Scope note.** Mantis's implementation lives entirely on the
> `mantis-implementation` branch / `/Users/boshra/NSRR-workspace/NSRR-tools-mantis/`
> worktree, not yet merged into `NSRR-tools` `main`. All file:line citations
> below marked "(mantis worktree)" are from that repo; citations marked
> "(main repo)" are from `/Users/boshra/NSRR-workspace/NSRR-tools/`. As of
> this writing (2026-09-19), the Mantis worktree is still actively running
> Stage 2 (LoRA) experiments — several facts below (registry placeholder
> values, in-progress cells) are current-state snapshots, not final results,
> and are flagged as such throughout.
>
> Two copies of the implementation plan exist
> (`NSRR-tools/docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md` and
> `NSRR-tools-mantis/docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md`) — spot-checked
> and they are the same document (the main-repo copy is a stale mirror, not a
> divergent draft); the worktree copy was used as the citation source per the
> task instructions, and no content differences were found in the sections
> read.

---

## 1. Model identity & checkpoint

**Checkpoint actually used for all reported Stage 1 results: `paris-noah/Mantis-8M`** — the real-time-series-pretrained checkpoint, *not* the synthetic-only `MantisPlus` variant. Confirmed in the live config (`configs/phase0_mantis_config.yaml:11`, `embedding.repo_id: "paris-noah/Mantis-8M"`) and restated in `TSFM_MODEL_COMPARISON.md` §2.4 ("The checkpoint is **Mantis-8M** ... not the synthetic-only `MantisPlus` ablation (deferred, not run)").

Four checkpoints exist upstream: `Mantis-8M` (real pretraining), `MantisPlus` (CauKer-2M, purely synthetic, architecturally identical to Mantis-8M), `MantisV2` (CauKer-2M synthetic, architecturally different — 2× cheaper per token, SwiGLU MLP, different LoRA target names), and `Utica` (self-distillation, not investigated). Only `Mantis-8M` has been run for the paper's actual Stage 1 sweep; `MantisPlus` is a planned-but-deferred internal ablation (real-vs-synthetic pretraining, a perfectly controlled contrast since the two checkpoints differ by exactly 2 buffer tensors / 18 params — `tokgen_unit.scalar_encoders.{0,1}.scales`); `MantisV2` was documented but never run.

**Parameter count — the 8.11M vs 8.037M distinction is real and matters for any "model size" claim.** The checkpoint-file total (`model.safetensors`, read via HTTP range request) is **8,112,384** for Mantis-8M / **8,112,402** for MantisPlus — the number usually quoted in the literature and in early planning docs. But this includes a non-trainable, deterministic sinusoidal positional buffer (`pos_encoder.pe`, 8,448 elements) and a `prj` projector head that is dead weight at inference (only invoked when `pre_training=True`). **The number that matters for model-size claims and for the LoRA-adapted-fraction denominator is 8,037,632 live parameters**, identical for both checkpoints (`MANTIS_CLAUDE.md:225-229`, `TSFM_MANTIS_IMPLEMENTATION_PLAN.md` §1.0 item 6 and §1.1). This was found only by actually running `scripts/verify_mantis_checkpoint.py` against the real checkpoint (2026-09-06), not by reading the source more carefully — dropping `pos_encoder.pe` alone raised a second, independent `RuntimeError: size mismatch` from `prj`, caused by `output_token='combined'` doubling `self.hidden_dim`.

**License: Apache-2.0**, confirmed from the repo's own `LICENSE` file and all three HuggingFace model cards' `cardData` (`MANTIS_CLAUDE.md:201-202`, `TSFM_MANTIS_IMPLEMENTATION_PLAN.md:76-78`) — the cleanest license of the three TSFM baselines (OSF: MIT; PhysioOmni: split — no LICENSE in the code repo, CC-BY-4.0 on the HF weights repo).

**Pretraining corpus: general time series, explicitly NOT physiological.** Mantis-8M's pretraining mix is described (in `TSFM_THIRD_MODEL_DECISION.md` §4.1) as containing "a small portion of EEG data" within an otherwise generic UCR/UEA-style time-series archive — this is why the synthetic `MantisPlus`/CauKer ablation exists as a documented option, to sidestep even that small EEG exposure. No NSRR cohort, or PSG data of any kind, appears in Mantis's pretraining corpus by construction (`TSFM_MODEL_COMPARISON.md` §1: "provably zero" — "no ID-level check is possible (or needed) when the pretraining data was never PSG in the first place").

**Peer-review status: arXiv only.** Huawei Noah's Ark Lab, `arXiv:2502.15637`. `TSFM_MODEL_COMPARISON.md` §1 states plainly: "peer-review status not otherwise confirmed in our own docs — treat as unreviewed until directly checked, same category as PhysioOmni." (OSF is the only one of the three TSFM baselines with a peer-reviewed venue, ICML 2026.)

---

## 2. Data adaptation / preprocessing pipeline

**HDF5 tree used: fast/reduced-channel, same tree as PhysioOmni (`phase0_v3`), NOT OSF's full-channel tree.** Confirmed in `configs/phase0_mantis_config.yaml:31` (`hdf5_dir: "/scratch/boshra95/psg"`, the fast tree) and stated explicitly in `TSFM_MANTIS_IMPLEMENTATION_PLAN.md` §2.3 ("Comparison baseline: `phase0_v3`... same as PhysioOmni, not OSF's `phase0_v3_full`"). No EDF reprocessing and no resampling are needed — every needed channel already exists at 128 Hz in the fast tree, and `3840 = 240 × 16` exactly.

**The 6-slot canonical channel map is NOT a uniform 6-channel set across cohorts** — this is an explicit correction to an earlier planning-doc claim. Measured across 250-subject random samples per cohort (`TSFM_MANTIS_IMPLEMENTATION_PLAN.md` §2.1, "Five skeleton claims were wrong"): APPLES/MrOS/STAGES carry 8 raw channels under *different names* (`C3-M2`, `C4-M1`, `CHIN`, `LLEG`, etc.), while SHHS carries only 6 and its RESP slot is `Airflow` for ~75% of subjects and `Thor` for ~25%. STAGES has genuine per-subject gaps (~10% missing `EKG`, ~22% missing a chin channel). The actual canonical map, live-verified in code (`src/nsrr_tools/datasets/mantis_channel_loader.py:63-78`):

```
SLOT_ORDER = ["EEG", "EOG_L", "EOG_R", "ECG", "EMG", "RESP"]
DEFAULT_CHANNEL_CANDIDATES = {
    "EEG":   ["C3-M2", "EEG", "C4-M1", "O1-M2"],
    "EOG_L": ["LOC"],
    "EOG_R": ["ROC"],
    "ECG":   ["EKG", "ECG-L"],
    "EMG":   ["CHIN", "EMG", "LLEG", "RLEG"],
    "RESP":  ["Airflow", "Thor", "ABD"],
}
```

Each slot resolves via a priority-ordered candidate list per cohort, not a single fixed name (`mantis_channel_loader.py:103-151`, `load_subject_channels`). A slot with no resolvable candidate is left as **exact zero**, not skipped or interpolated — an explicit "absent-slot contract" (`mantis_channel_loader.py:117-121`; matches `TSFM_MANTIS_IMPLEMENTATION_PLAN.md` §2.2's stated reasoning: running the backbone on zeros would *not* actually produce a zero output, since `ts_scaler` gives `0/(0+1e-5)=0` but the conv adds its bias and the scalar encoders emit a nonzero constant — so Stage 1 explicitly skips the backbone forward for missing slots and writes zeros directly into the embedding).

**Only one EEG derivation, not two, deliberately.** APPLES/MrOS/STAGES all have both `C3-M2` and `C4-M1`, but SHHS has only a single generic `EEG` key. A second EEG slot would be structurally absent for 8,444 of 14,994 subjects (56%, all of SHHS) — judged the same failure mode PhysioOmni's own plan flagged and rejected (`TSFM_MANTIS_IMPLEMENTATION_PLAN.md` §2.2).

**The critical architecture adaptation: regenerating the sinusoidal positional buffer for `num_patches=240`, `seq_len=3840`, to feed a full native 30-second epoch at 128 Hz — Option D, not interpolation down to Mantis's released 512-sample window.** `TSFM_THIRD_MODEL_DECISION.md` §3.2 gives the Nyquist argument directly: interpolating our 3,840-sample epoch down to Mantis's native 512 samples gives an effective sample rate of 17.07 Hz, a Nyquist frequency of 8.5 Hz — below sleep spindles (11–16 Hz), beta activity, and essentially all EMG/ECG morphology. For a sleep paper this "is not a defensible preprocessing choice — it would hand a reviewer an easy objection that the baseline was crippled before it started." Instead, `seq_len=3840, num_patches=240` keeps `patch_window_size = seq_len/num_patches = 16` and the conv tokenizer's `kernel_size=17` **identical to pretraining** (`TSFM_MANTIS_IMPLEMENTATION_PLAN.md` §1.2, §3.1) — only the *number* of tokens changes (240 vs. the pretrained 32), not the tokenizer's own operations. The positional buffer (`register_buffer('pe', ...)`, shape `(num_patches+1, 1, d_model)`) must be manually regenerated rather than loaded, because `from_pretrained` cannot build a 240-patch model (it reconstructs from the repo's `config.json`, which is hardcoded to `seq_len:512, num_patches:32`, and a naive `load_state_dict(..., strict=False)` still raises `RuntimeError: size mismatch` on a shape-mismatched buffer even under `strict=False`, reproduced empirically on torch 2.5.1). The manual load sequence is implemented once in `load_mantis_backbone()` (`mantis_channel_loader.py:241`, plan §3.4) and imported by every stage. **Honest caveat carried in the plan itself**: 241 tokens is 7.3× longer than anything Mantis saw in pretraining (33 tokens); the conv tokenizer is length-agnostic and the sinusoidal PE is defined at every position, but the transformer's learned attention behavior at that length was explicitly flagged as untested and as "the single largest scientific risk in the plan" before the empirical pilots (§13.1/§13.2 below) ran.

**Windowing decision confirmed empirically, not just reasoned about.** Three candidate windowing schemes were piloted head-to-head on 100 real subjects (50 APPLES + 50 SHHS) via a single-epoch sleep-staging probe (`scripts/pilot_mantis_windowing_layer.py`, `MANTIS_CLAUDE.md:538-593`): Option D (full epoch, regenerated PE), Option D-interp (full epoch, rescaled PE), and Option B (8 sub-windows of the pretrained 512-sample size, mean-pooled). All 12 (windowing × layer × token) combinations spanned only 0.0415 weighted F1 (0.7029–0.7444) — none of the implementation choices moved the needle much, and **Option D was confirmed** (gap vs. Option B was −0.0096, within the pre-specified tie-break band; an escape hatch would have required >0.15).

**Embedding extraction: output shape `[T, 6, 512]` per subject.** `T` = number of 30-second epochs, 6 = canonical channel slots, 512 = the "combined" per-channel embedding dimension. The **combined token = concat(CLS, mean-pooled patch tokens)** from the model's **last** transformer layer, giving `embed_dim=512` per channel and a flattened head input of `input_dim = 6 × 512 = 3072` (`configs/phase0_mantis_config.yaml:22-30`).

**This choice is an empirical, cross-model-comparability decision, NOT the authors' own per-checkpoint recipe.** The Mantis README and `getting_started/intermediate_layers.ipynb` recommend, for best frozen performance, `return_transf_layer=layer_idx` (Mantis-8M's own claimed optimum: layer 2, out of 6) together with `output_token='combined'`. This plan explicitly rejected the layer-2 half of that recipe for two reasons documented in `TSFM_MANTIS_IMPLEMENTATION_PLAN.md` §3.3:
1. **Empirically, it doesn't reproduce.** The authors' own notebook's stored outputs (GestureMidAirD1, 130 test samples) show Mantis-8M's *own* best layer is 1 (cls) or 3 (combined), not 2 — with a sample-count-implied margin of only 3–8 samples across all 6 layers. What *does* reproduce robustly is the token choice: `combined` beats `cls` at all 6 layers for Mantis-8M.
2. **Methodologically, truncating at layer 2 would make Mantis the only truncated encoder among the four models** (SleepFM, OSF, and PhysioOmni all harvest their last layer), would halve Mantis's LoRA-adaptable depth (3/6 vs. OSF's/PhysioOmni's 12/12), and would turn the Mantis-8M-vs-MantisPlus ablation into a two-variable comparison (differs in both pretraining data and truncation depth, since MantisPlus's own claimed optimum layer is 1, not 2).

Decision: **`combined @ last`** for both Stage 1 and Stage 2 — `return_transf_layer: -1`, `output_token: "combined"`, `embed_dim: 512`, `input_dim: 3072` (`configs/phase0_mantis_config.yaml:20-28`). This was also confirmed empirically in the same 100-subject pilot: the layer-2-vs-last gap was 0.0033, below the pre-specified 0.08 escape-hatch threshold. The explicit cost of this decision, stated in the plan (§3.3): giving up a compute/memory saving (`@2` would have roughly halved backbone FLOPs and Stage 2 activation memory) and making Mantis's head input the widest of any model in the paper (3072, vs. OSF's 1536, SleepFM's 512, PhysioOmni's 500).

---

## 3. Architecture summary

**Channel-independent by construction, same structural pattern as PhysioOmni but architecturally simpler.** Mantis's tokenizer applies `Conv1d(in_channels=1, out_channels=256, kernel_size=17, padding=8)` (`same`-padding, confirmed against the real checkpoint tensor shape `tokgen_unit.convs.0.conv.weight [256,1,17]`) — every channel is a fully independent forward pass through the identical weights, with **no channel-identity embedding of any kind** (`TSFM_MANTIS_IMPLEMENTATION_PLAN.md` §1.2, §2.2). This differs from PhysioOmni's channel-independence in one structural way worth noting: PhysioOmni is channel-independent *across four separate encoders* (EEG/EOG/ECG/EMG, each its own small network with its own weights, `d=100–200`), whereas Mantis is channel-independent through **one single, well-shaped encoder** (`hidden_dim=256`, attention inner dim 1024) applied identically to all 6 channels via a single batched forward call. This is what gives Mantis its favorable GPU-throughput profile relative to PhysioOmni (§6 below).

**No cross-modal fusion exists anywhere in the released checkpoint.** `TSFM_MODEL_COMPARISON.md` §1 states this directly: "None — channel-independent by construction, the same structural pattern as PhysioOmni (not a fused single-tensor design like SleepFM/OSF); channels are combined only downstream, by our own sequence heads." All cross-channel structure in the final prediction is learned entirely by the downstream LSTM/Transformer head, not by the frozen backbone.

**Why apnea/RESP IS in scope for Mantis, unlike PhysioOmni.** PhysioOmni's checkpoint has no respiratory pathway anywhere (confirmed at 4 independent code locations in the main-repo investigation) and apnea is excluded from its comparison entirely. Mantis has no such architectural gap: because every channel — including the RESP slot (`Airflow`/`Thor`/`ABD`) — is processed by the exact same channel-independent encoder as EEG/EOG/ECG/EMG, there is no modality-specific pretraining requirement to satisfy. `TSFM_MODEL_COMPARISON.md` §1: "channel-independent by construction (`Conv1d(in_channels=1)` per channel), so the RESP slot is processed by the identical encoder as every other channel; no architectural exclusion needed, unlike PhysioOmni." Apnea (`apnea_binary`) is therefore included in all 7 Mantis Stage 1 tasks, matching OSF's scope rather than PhysioOmni's reduced 4-task scope.

**Architecture constants** (`TSFM_MANTIS_IMPLEMENTATION_PLAN.md` §1.2): `hidden_dim=256`, `num_patches=32` (pretrained) / `240` (adapted), `transf_depth=6`, `heads=8`, `dim_head=128` → attention inner dim `8×128=1024`, `mlp_dim=512`. **Zero `BatchNorm`** in the backbone (confirmed by an exhaustive `grep` — the only two `BatchNorm1d` hits are in the library's own default fine-tuning head, which is never used here) — meaning `chunk_batch_size` is mathematically inert on gradient correctness, purely a throughput-scheduling knob (§6 below).

---

## 4. Stage 1 — Frozen encoder + downstream head training

**What's frozen vs. trained**: the entire Mantis backbone (tokenizer + 6-layer transformer) is frozen; only a downstream `LSTMHead`/`TransformerHead` (from `src/nsrr_tools/models/sequence_head.py`, imported unmodified, never edited per the worktree-isolation rule) is trained, exactly as for SleepFM/OSF/PhysioOmni.

**New files, real paths (mantis worktree)**:
- Channel loading + backbone manual-load + Stage 2 cache helpers: `src/nsrr_tools/datasets/mantis_channel_loader.py`
- Context-window dataset (Stage 1): `src/nsrr_tools/datasets/mantis_context_window_dataset.py`
- Raw-signal-cache dataset (Stage 2): `src/nsrr_tools/datasets/mantis_raw_epoch_dataset.py`
- Embedding extraction: `scripts/extract_mantis_embeddings.py`, job scripts `jobs/extract_mantis_embeddings_gpu{,_rorqual}.sh`
- Training (Stage 1): `scripts/train_mantis_context_sweep.py`, `jobs/train_mantis_context_sweep_gpu{,_nibi,_rorqual}.sh`
- Inference (Stage 1): `scripts/infer_mantis_subject_windows.py`, `jobs/infer_mantis_subject_windows_gpu{,_nibi,_rorqual}.sh`
- Config: `configs/phase0_mantis_config.yaml`
- Registry + command generator: `experiments/v2_mantis_registry.yaml`, `scripts/gen_commands_mantis.py`
- Checkpoint verification: `scripts/verify_mantis_checkpoint.py`

**Real hyperparameters** (`configs/phase0_mantis_config.yaml`), held identical to OSF's/PhysioOmni's own Stage 1 values where architecture allows:
- `epochs: 40`, `lr: 1.0e-4`, `weight_decay: 1.0e-3`, cosine LR schedule (hardcoded in the training-loop fork, the config's `scheduler`/`optimizer` fields are not actually read), `early_stopping_patience: 10` on `val_auroc`.
- Sequence head: `hidden_dim: 128`, `num_layers: 1`, `num_heads: 8`, `dropout: 0.3`, `input_dim: 3072` — the head architecture itself (`hidden_dim`, `num_layers`) is held identical to the other three baselines; only `input_dim` changes per encoder, preserving the paper's "architecture held constant, only the encoder changes" claim.
- Extraction `chunk_batch_size: 192` (= 32 epochs × 6 channels per backbone forward call) — measured to make **no meaningful throughput difference** vs. `chunk_batch_size=48` in a real GPU pilot (4.370 vs. 4.341 TFLOP/s, a 0.7% difference), unlike OSF's confirmed 3.28× sensitivity to the same knob (`MANTIS_CLAUDE.md:490-506`). 192 was kept as the default anyway since it showed no downside.
- `context_lr_overrides`: `120m`/`240m` use a reduced `5.0e-5`.
- `min_recording_patches: 480` (= 240 minutes in 30-second-epoch units — a Mantis-specific value, explicitly *not* SleepFM's own `2880` convention, since Mantis's epoch unit is 30s not 5s).
- `bootstrap_samples: 0` — **no bootstrap CIs are computed for Stage 1 results** (see §7).

**Compute characteristics**: extraction runs under `torch.no_grad()` on a `1g.10gb` MIG slice (a deliberate, user-directed choice — Stage 1 has no backward-pass memory pressure, so there is no memory argument for a bigger allocation, unlike Stage 2; `MANTIS_CLAUDE.md:839-853`). Real measured extraction throughput: **6.18–6.20% of a 1/7 MIG slice's peak** (job 58534992, live sbatch verification), ~44× PhysioOmni's historical 0.14%-of-full-card extraction efficiency. Per-subject extraction cost ~8.0 seconds (Option D, from Pilot 3/Pilots 1-2).

**Coverage: COMPLETE for all 7 tasks × lstm/transformer heads × 6 contexts.** Confirmed in two independent places: `MANTIS_CLAUDE.md`'s 2026-09-08 entry ("Full embedding extraction finished... 14,993/14,994 subjects (99.99%)... User is clear to proceed to the Stage 1 training sweep") and `TSFM_MODEL_COMPARISON.md` §3.3's stopping-criteria table ("Complete: 7 tasks (apnea included) × 2 heads × 6 contexts (`mean_pool` not run)"). The 7 tasks are the 5 Tier-1 tasks (sex_binary, sleep_efficiency_binary, apnea_binary, age_class, bmi_binary) plus 2 Tier-2 secondary tasks added 2026-09-09 (depression_extreme_binary, osa_binary_apples_postqc). `mean_pool` head was never run for any task, matching OSF's and PhysioOmni's own status — no MeanPool-vs-temporal-head comparison can be made from this data.

**The headline result: frozen Mantis beats SleepFM outright on sex, age, and the secondary OSA task — contradicting a real, pre-registered "expect a weak frozen result" prediction.** Numbers (Transformer head, `mean_prob_auroc`, test split, K=Kmax, from `TSFM_MODEL_COMPARISON.md` §2.4, pulled directly from `results/collected/phase0_mantis/analysis.csv`):

| Task | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| sex_binary — SleepFM | 0.832 | 0.851 | 0.872 | 0.897 | 0.905 | 0.910 |
| sex_binary — **Mantis-frozen** | **0.863** | **0.891** | **0.916** | **0.927** | **0.935** | **0.923** |
| age_class — SleepFM | 0.854 | 0.870 | 0.877 | 0.900 | 0.902 | 0.905 |
| age_class — **Mantis-frozen** | **0.856** | **0.885** | **0.912** | **0.918** | **0.923** | **0.919** |
| osa_binary_apples_postqc — SleepFM | 0.789 | 0.804 | 0.853 | 0.888 | 0.856 | 0.861 |
| osa_binary_apples_postqc — **Mantis-frozen** | **0.818** | **0.837** | **0.872** | 0.887 | **0.879** | **0.903** |

Mantis wins every context on sex (+1.3 to +4.4pp) and age (+0.2 to +3.5pp), and 5 of 6 contexts on OSA (+1.9 to +4.2pp, essential tie at 80m: −0.1pp), with the largest single margin in the entire cross-model comparison document at 240m OSA (+4.2pp). Apnea, sleep efficiency, and BMI are much closer calls — apnea is within 2.0pp either direction with no consistent winner; sleep efficiency is nearly tied through 40m and runs slightly behind SleepFM from 80m on; BMI has no consistent direction. None of these three should be folded into the headline win claim.

**Why this contradicts the pre-registered prediction, stated precisely**: `TSFM_THIRD_MODEL_DECISION.md` §6 explicitly committed, *before any Mantis code was written*, to "expect — and pre-commit to reporting — a weak frozen result," citing a real published study (Gnassounou et al. 2025, arXiv:2510.27522) that found "freezing the encoder for EEG data leads to a huge decrease in performance" for this class of generic TSFM. This was not a strawman or a post-hoc framing — it was a documented, dated prediction that the actual Stage 1 results falsified for 3 of 7 tasks. `TSFM_MODEL_COMPARISON.md` §6.5 calls this "the single most attention-worthy number this document now contains, not a footnote," precisely because a non-physiological, general-time-series encoder outright beating a 585,000-hour PSG-specific pretrained encoder — with a completely clean, "provably zero" contamination story (§7 below) — is a genuinely surprising, falsifiable, and falsified prediction.

**Context-sensitivity pattern replicates almost exactly under this completely different encoder** (`TSFM_MODEL_COMPARISON.md` §2.4.1): sleep efficiency and apnea remain the two most context-sensitive tasks under Mantis (same as under SleepFM, exact rank order swaps by a hair); sex and age form a moderately context-sensitive middle tier under both; BMI is the least context-sensitive primary task under both (Δ 30s→peak: +0.036 Mantis vs. +0.030 SleepFM). This is treated as the strongest generalization evidence in the whole three-model comparison, because it comes from the encoder with the *least* physiological relatedness to the task, making a coincidental match less likely.

---

## 5. Stage 2 — LoRA fine-tuning

**LoRA config, live-verified against the real checkpoint** (`scripts/verify_mantis_checkpoint.py`, 2026-09-06): `target_modules=["to_qkv", "to_out.0"]`, `r=8`, `lora_alpha=16`, `lora_dropout=0.05`, `modules_to_save=["sequence_head"]` (`configs/phase0_mantis_lora_config.yaml`). This injects **exactly 12** LoRA-wrapped Linears (6 transformer blocks × 2 modules — half of OSF's 24, since OSF has 12 blocks) and **exactly 221,184** trainable LoRA parameters ≈ **2.75%** of the 8,037,632 live backbone parameters — matching a hand-derived arithmetic check exactly, identical for both checkpoints since Mantis-8M and MantisPlus share the same architecture.

**Warm-start strategy**: every context length other than 30s warm-starts its LoRA+head weights from that same (task, head)'s own already-converged **30s** LoRA checkpoint (a branch structure, not a chained 30s→10m→...→240m progression) — 30s itself warm-starts from Stage 1's frozen-backbone head. This mirrors OSF's own documented Stage 2 warm-start convention.

**The `context_micro_batch` OOM story, in full, including a real fix found mid-stream**:
1. **Root cause**: `experiments/v2_mantis_lora_registry.yaml`'s `gradient_accumulation.context_micro_batch` was — and, verified directly against the current registry file as of this writing (commit `c3f6939`), **still is** — a flat, uncalibrated `32` at *every* context from 30s through 240m (`v2_mantis_lora_registry.yaml:52-61`), with the registry's own comment already flagging this as "NOT YET GPU-CALIBRATED FOR STAGE 2." The memory-calibration pilot intended to replace these placeholder values has not yet landed in the registry.
2. **Real production impact**: submitting `sex_binary_lstm` at 30s/10m/40m/80m on a **whole 80GB H100** on the Nibi cluster, only 30s succeeded (12.8 min, real achieved 33.99 TFLOP/s — confirmed genuinely running LoRA-adapted backbone compute, not a shortcut, since trainable params = 3,500,546 exactly matches head (3,279,362) + LoRA (221,184)). **10m, 40m, and 80m all failed with `CUDA out of memory`, every one topping out at 78.6–79.18 GiB of a full 80GB card** (`MANTIS_CLAUDE.md:1166-1188`). This is not a long-context-only problem — 10m (20 epochs/window) already exceeds capacity at `micro_batch=32`, the same as 80m (160 epochs/window), because Stage 2 memory scales with `micro_batch × epochs-per-window`, not context length directly.
3. **A wrong first fix, corrected**: the user's initial instinct was to shrink the GPU allocation (`--gpus=h100:1` → `h100_3g.40gb:1`) to preserve queue priority — this made the actual problem *worse*, since the bottleneck was memory capacity and a 40GB slice has less of it than the 80GB card that had already failed.
4. **A second, independently discovered bug (2026-09-12, not yet in `MANTIS_CLAUDE.md`'s status log, found in `configs/phase0_mantis_lora_config.yaml`'s own inline comments, commit `c3f6939`, "Fix Stage 2 LoRA cuDNN-LSTM crash"): `checkpoint_tokgen` (the cheap first memory-mitigation rung, gradient-checkpointing only the tokenizer conv) actively CRASHES the LSTM head.** `torch.utils.checkpoint` (tried with both `use_reentrant=True/False`, and an explicit `requires_grad_(True)` fix) corrupts cuDNN's bidirectional-LSTM+`PackedSequence` backward pass when a checkpointed segment appears earlier in the same computation graph, on this cluster's torch/cuDNN build — confirmed not a PEFT/head-wrapping issue by separate testing. The fix actually shipped is **`checkpoint_chunks: true`** (checkpointing the whole per-chunk backbone call, a coarser granularity that also saves *more* memory than `checkpoint_tokgen` would have) with **`checkpoint_tokgen: false`**, verified working for both LSTM and Transformer heads. Current config state: `checkpoint_tokgen: false, checkpoint_chunks: true` (`configs/phase0_mantis_lora_config.yaml`, training section).

**The Nibi-cluster 44-hour queue-time finding, and why it contradicts the Fir-cluster assumption**: this project's Fir-derived assumption (from OSF's own investigation) was that requesting a whole H100 (`--gpus=h100:1`) costs nothing extra in queue time versus a MIG slice, verified via `sbatch --test-only` showing identical estimated start times. **This does not transfer to Nibi.** Real, live `squeue`/`sinfo` telemetry at the time of submission showed whole `h100:1` requests **772 pending vs. 90 running (8.6× oversubscribed)** cluster-wide, with `3g.40gb` slices even worse (242 pending / 8 running, ~30×), while `1g.10gb` (54/69) and `2g.20gb` (3/14) slices had real headroom (`MANTIS_CLAUDE.md:1207-1224`). A whole-H100 memory-calibration pilot job was first submitted and projected to queue ~44 hours before start; it was cancelled and resubmitted against a `2g.20gb` slice instead (job 21777147), chosen for queue headroom rather than maximum memory. **Corrected takeaway, stated explicitly in the source**: "on Nibi, the right GPU size is whichever slice actually has queue headroom right now, not reflexively 'whole card' — check `squeue` before choosing, don't assume Fir's precedent holds."

**Exact current coverage, checked directly against the results directory as of 2026-09-13** (`TSFM_MODEL_COMPARISON.md` §3.4, cross-checked against the training-completed-vs-in-progress table there):

| Task | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| age_class (lstm) | done | in progress | — | — | — | — |
| apnea_binary (lstm) | done | in progress | — | — | — | — |
| bmi_binary (lstm) | done | in progress | — | — | — | — |
| sex_binary (lstm) | done | in progress | — | — | — | — |
| depression_extreme_binary (lstm) | done | done | in progress | in progress | — | — |
| sleep_efficiency_binary | — | — | — | — | — | — |
| osa_binary_apples_postqc | — | — | — | — | — | — |

No Transformer-head LoRA run has been started at all. `mean_pool` is deferred (matching PhysioOmni's own LoRA precedent). This snapshot is explicitly non-final — several cells were "in progress" (running on-cluster) at the time this table was recorded, and the memory-fix (`checkpoint_chunks`) landed *after* this snapshot, so coverage should be re-checked directly against `/scratch/boshra95/psg/unified/results/phase0_mantis_lora/` rather than trusted as current.

**The critical honest caveat: NO frozen-vs-LoRA comparison is possible for Mantis yet, and this is a metric-incompatibility problem, not just a coverage gap.** `infer_mantis_lora_subject_windows.py` — the subject-level, K-aggregated inference script that produces the `mean_prob_auroc, k="all"` metric every other number in this document uses — **has never been run against any Mantis LoRA checkpoint, at any context** (`TSFM_MODEL_COMPARISON.md` §3.4). The `test_auroc` values currently sitting in each Stage 2 `summary.csv` come from the *training* script's own held-out evaluation over a small, fixed set of sampled windows per subject (the same `windows_per_subject=5`-style convention used during training/validation), **not** the full, non-overlapping, subject-aggregated K_max protocol. These numbers are explicitly stated as not comparable to Mantis's own Stage 1 numbers in §4 above, or to any other number in this document — and this document deliberately does not reproduce them side-by-side with the Stage 1 table, to avoid exactly the mistake the source document itself warns against elsewhere (comparing numbers computed under different metrics as if they were the same measurement).

---

## 6. Honest computational drawbacks

**Measured per-training-epoch cost at 30s** (`TSFM_MODEL_COMPARISON.md` §3.4, from each completed run's own `summary.csv`, `training_time_min / n_epochs_run` — a real measurement):

| Task | Epochs run | Wall time (min) | Min/epoch | Achieved TFLOP/s |
|---|---|---|---|---|
| apnea_binary | 16 | 24.72 | **1.55** | 6.63 (1.34% of a whole-H100 TF32 peak) |
| bmi_binary | 10 | 11.06 | 1.11 | 6.63 |
| sex_binary | 17 | 17.03 | 1.00 | 6.62 |
| depression_extreme_binary | 8 | 11.52 | 1.44 | 6.62 |
| age_class | 15 | 165.77 | **11.05** | 6.64 |

**`age_class`'s 11.05 min/epoch is 7–10× every other task's wall-clock time, despite essentially identical achieved TFLOP/s across all five rows (~6.6, all within 0.3%).** This is flagged explicitly as **unexplained, not resolved**: the anomaly is in wall-clock time, not compute throughput, which points toward something external (all four 30s jobs ran concurrently on the same shared node, `g34`) rather than a genuine per-task compute difference — but this was **not investigated further, per an explicit decision** recorded in the source. `apnea_binary`'s clean 1.55 min/epoch is used as the representative 30s baseline for any extrapolation; `age_class`'s number is reported as a real measurement but excluded from any estimate.

**The achieved ~6.6 TFLOP/s figure and what it represents**: this is **1.34% of a whole H100's TF32 peak** (~495 TFLOP/s). For context against the other two TSFM baselines' own measured LoRA numbers: OSF measured ~2.0 TFLOP/s at 30s rising to ~19.2 TFLOP/s at 80m (on a smaller MIG slice, so not directly percentage-comparable); PhysioOmni measured ~0.69 TFLOP/s (~3.6% of a `3g.40gb` slice's smaller peak) and never improved much beyond that, attributed to a genuine architectural ceiling (four separate small encoders, hidden dims not multiples of 8). Mantis's raw ~6.6 TFLOP/s at 30s is **roughly an order of magnitude above PhysioOmni's own measured 30s-adjacent regime**, and in the same rough range as OSF's own 30s number — read the raw TFLOP/s values, not the percentages, when comparing across models, since the percentages are against different-sized GPU allocations.

**ESTIMATED (not measured) per-context cost table for the four larger Tier-1 tasks**, explicitly labeled as an estimate in the source (`TSFM_MODEL_COMPARISON.md` §3.4), built using the same naive linear-in-raw-epochs assumption applied to OSF, anchored to `apnea_binary`'s measured 30s baseline (1.55 min/epoch):

| Context | Raw epochs/window (vs. 30s) | **ESTIMATED** min/epoch |
|---|---|---|
| 10m | 20× | **~31** |
| 40m | 80× | **~124** |
| 80m | 160× | **~248** |
| 120m | 240× | **~371** |
| 240m | 480× | **~742** |

**Stated reason this estimate is likely an overstatement**: the one real longer-context data point that exists — `depression_extreme_binary`'s completed 10m run (9 epochs, 85.48 min → 9.50 min/epoch) — shows a **6.6× cost increase over its own 30s baseline**, well below the ~20× naive raw-epoch-count scaling used to build the table above. A plausible (not yet controlled-A/B-verified) explanation: Mantis had TF32 and a large `chunk_batch_size` (192, confirmed insensitive) built in from day one, unlike OSF and PhysioOmni, which discovered these optimizations only after weeks of running unoptimized — so Mantis's 30s baseline may already sit closer to its own compute-bound ceiling, making the jump to 10m proportionally smaller. This correction factor is explicitly **not applied** to the estimate table above, because it comes from the smallest task in the comparison (5,615 training windows vs. 43,000–48,000 for the four larger Tier-1 tasks) and has not been confirmed on any of those larger tasks specifically. Treat the table as a conservative upper bound pending a real 10m measurement on a larger task.

**Overall diagnosis, per the source's own explicit framing**: "Mantis, so far, looks like neither [OSF's granularity/batching problem nor PhysioOmni's structural architecture ceiling] — but the evidence so far is thin." Its architecture is the best-shaped of the three fine-tuned backbones (one joint tensor per channel call, `hidden_dim=256` — a clean power of 2, unlike PhysioOmni's `d=100`), and it was engineered with every efficiency lesson (TF32, `chunk_batch_size`, achieved-TFLOP/s logging) built in from the start rather than discovered reactively — but the sweep is far less complete than either OSF's or PhysioOmni's LoRA stage was at a comparable point in their own timelines, so this apparent efficiency should be read as encouraging, not settled.

---

## 7. Other reviewer-relevant caveats

**Provably zero pretraining contamination — the strongest contamination story of the three TSFM baselines.** OSF's contamination is real, severe, and precisely quantified (87.7% of the SHHS test split was in OSF's own pretrain train/valid splits, by exact subject-ID match). PhysioOmni's contamination check found none of the 4 test cohorts in its pretraining corpus (TUH/CAP/Sleep-EDF/DEAP — none are NSRR), but this required an actual ID-level check to establish. **Mantis's case is categorically different and stronger**: its pretraining corpus is generic real-world time-series archives (UCR/UEA-style), never physiological signal of any kind, so **no NSRR cohort could appear in it by construction** — "no ID-level check is possible (or needed) when the pretraining data was never PSG in the first place" (`TSFM_MODEL_COMPARISON.md` §1). This is the cleanest possible contamination narrative of the three models, and it directly strengthens the credibility of Mantis's headline sex/age/OSA wins (§4 above) — a stronger, cleaner-provenance model beating SleepFM is a more surprising and more trustworthy result than a contamination-confounded one.

**No bootstrap confidence intervals exist for any Mantis number, Stage 1 or Stage 2.** Confirmed directly, not assumed: `bootstrap_samples: 0` in `configs/phase0_mantis_config.yaml`, and every `mean_prob_auroc_ci_lo`/`_hi` cell in `results/collected/phase0_mantis/analysis.csv` is `NaN` (`TSFM_MODEL_COMPARISON.md` §2.4.1, §5). Every Mantis point estimate in this document — including the headline sex/age/OSA wins — should be read as provisional pending a CI pass. This is a real, checked fact about Mantis specifically, not a general statement extended from OSF/PhysioOmni's own (not independently re-verified here) CI status.

**A small test-population mismatch vs. SleepFM exists, similar to OSF's documented split-mismatch issue but smaller and less rigorously checked.** SleepFM and Mantis Stage 1 filter subjects by "has embedding file" using the same `np.random.default_rng(split_seed).shuffle()` pattern, but against different embedding directories with slightly different per-subject extraction-success populations. Measured test-subject-count differences are single-digit-to-a-few-dozen per task (e.g. `apnea_binary`: 2,077 vs. 2,054; `depression_extreme_binary`: 241 vs. 229) — consistent with the same shuffle-permutation-sensitivity mechanism OSF's own investigation found and quantified exactly (OSF: APPLES had exactly 1 subject differing; STAGES had exactly 1 subject each way). **Mantis's version of this issue has not been independently investigated to OSF's level of rigor** — no exact-ID cross-check was run, so this is flagged as an open item, not a resolved non-issue (`TSFM_MODEL_COMPARISON.md` §2.4.2).

**`mean_pool` head was never run**, for either Stage 1 or Stage 2, matching OSF's and PhysioOmni's own status — no H3-style "does the temporal-head-vs-MeanPool advantage replicate under a different encoder" claim can be made from Mantis's data.

**Val-split threshold-tuning was never run**, for either stage — no balanced-accuracy/threshold-based comparison should be built from Mantis's numbers without first running it, per the paper's standing rule against reporting incomplete analyses as complete.

**The sex-AUROC-declining-at-240m divergence from SleepFM's pattern — a real, unresolved, flagged-not-explained anomaly.** SleepFM's sex-classification AUROC rises monotonically through every context tested, still climbing at 240m (the paper's own basis for calling sex "still rising at the longest context evaluated"). **Mantis's sex AUROC instead peaks at 120m (0.935, Transformer) and then declines at 240m (0.923)** — the same shape SleepFM shows for no primary task (`TSFM_MODEL_COMPARISON.md` §2.4.1). Two facts are both true and both stated in the source: (1) no bootstrap CI exists for either the 120m or 240m Mantis point, so there is no statistical basis yet to call this dip real rather than sampling noise; (2) the test-subject count is stable and large at both contexts (1,431 subjects at every context for sex — no cohort-dropout confound explains it). This is explicitly reported as "an open, unresolved divergence, not a claim that Mantis's context-value story differs from SleepFM's," since the other six tasks' context-sensitivity pattern replicates cleanly — but it is a genuine anomaly that a CI pass has not yet resolved, and should not be smoothed over if this material ships.

**A cross-model "Mantis vs. OSF" leaderboard is not supportable from the existing data** — Mantis and PhysioOmni share the same SleepFM baseline (`phase0_v3`, reduced-channel), but OSF is compared against a different SleepFM baseline (`phase0_v3_full`, full-channel). Any apparent "Mantis beats OSF" or "OSF beats Mantis" framing built from the existing tables would be comparing each model against a different reference point, not against each other under matched conditions.

---

## 8. Source citations

**Main repo (`/Users/boshra/NSRR-workspace/NSRR-tools/`):**
- `docs/TSFM_THIRD_MODEL_DECISION.md` (full file, 568 lines) — the Mantis-vs-MOMENT decision document; §1 (TL;DR comparison table), §2 (the decisive Gnassounou et al. 2025 evidence and the "expect a weak frozen result" pre-registration), §3.1–3.2 (channel-independence, the 512-sample wall, Option D windowing and the Nyquist argument), §3.2 (Mantis-specific windowing note, lines 286–293), §4.1 (Mantis-8M candidate profile, lines 318–344), §5 (compute/FLOP comparison, lines 396–489), §6 (paper-framing implications, lines 493–522), §7/decision (lines 526–555).
- `docs/TSFM_MODEL_COMPARISON.md` (full file, 1181 lines) — §0 (executive summary), §1 (architecture/input-handling table, lines 93–139, Mantis column), §2.4 and §2.4.1–2.4.2 (Stage 1 results tables and context-sensitivity replication, lines 309–478), §3 (LoRA cost mechanism), §3.3–3.4 (stopping-criteria table and Mantis Stage 2 honest accounting, lines 563–697), §4 Mantis subsection (lines 797–860), §5 (what not to over-interpret, lines 864–893), §6.3–6.5 (caveat table and updated framing, lines 948–1058).
- `docs/LORA_GPU_THROUGHPUT_INVESTIGATION.md` (full file, 191 lines) — the overhead-bound-at-short-context / compute-bound-at-long-context mechanism, explicitly flagged (line 8–17) as relevant to Mantis's own eventual LoRA stage; real OSF measured timing table (§4, lines 101–128).
- `CLAUDE.md` — repository map and general TSFM-comparison context (background only, no Mantis-specific facts cited directly).

**Mantis worktree (`/Users/boshra/NSRR-workspace/NSRR-tools-mantis/`):**
- `MANTIS_CLAUDE.md` (full file, 1243 lines) — checkpoint verification and parameter-count correction (lines 195–249), environment build notes (lines 131–182), status log entries throughout (Phase 0–2 progress, real bugs found, Pilot 1/2/3 results at lines 464–593, Nibi setup at lines 998–1165, the Stage 2 OOM/queue-time incident at lines 1166–1243).
- `docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md` (read in full to line 856 of 2995; remaining sections not read directly but cross-verified via `MANTIS_CLAUDE.md`'s summaries, which quote the plan's section numbers and content directly) — §1.0 (corrections to the 2026-08-22 skeleton, lines 80–94), §1.1–1.5 (checkpoint/architecture/LoRA-target/BatchNorm/channel facts, lines 95–259), §2.1–2.3 (real per-cohort channel measurements and the 6-slot map decision, lines 262–350), §3.1–3.4 (windowing options, normalization decision, output-token/layer decision and manual-load sequence, lines 353–641), §4.1–4.5 (performance lessons: TF32, memory scaling, chunk_batch_size, whole-card-vs-MIG, lines 644–856).
- `src/nsrr_tools/datasets/mantis_channel_loader.py` — `SLOT_ORDER`/`DEFAULT_CHANNEL_CANDIDATES` (lines 63–78), `_SAFE_TO_DROP`/`_ALLOWED_MISSING_AFTER_DROP`/`_OPTIONAL_MISSING` checkpoint-surgery constants (lines 82–96), `load_subject_channels()` (lines 103–151), `get_epoch_count()` (lines 154–164), `load_mantis_backbone()` (line 241), `epochs_to_model_input()` (line 171), `sinusoidal_pe()` (line 218).
- `configs/phase0_mantis_config.yaml` (full file) — Stage 1 real hyperparameters, embedding-extraction config, channel-candidate map, sequence-head config.
- `configs/phase0_mantis_lora_config.yaml` (full file) — Stage 2 config, including the LoRA block (`r=8, lora_alpha=16, target_modules=["to_qkv","to_out.0"]`), the `checkpoint_tokgen`-crashes-LSTM finding and `checkpoint_chunks` fix (training section comments), the raw-signal-cache path.
- `experiments/v2_mantis_lora_registry.yaml` (lines 1–75) — the still-flat `context_micro_batch: 32` at every context (lines 52–61), confirmed current via `git log`/direct read against commit `c3f6939`.
- `git log --oneline` on `experiments/v2_mantis_lora_registry.yaml`, `configs/phase0_mantis_lora_config.yaml`, `MANTIS_CLAUDE.md` — used to confirm the `checkpoint_tokgen`/cuDNN-LSTM fix (commit `c3f6939`, "Fix Stage 2 LoRA cuDNN-LSTM crash; revise memory/GPU config") postdates `MANTIS_CLAUDE.md`'s own last written status-log entry, i.e. is a real, more-current finding not yet reflected in that file's prose.
- File existence/listing: `find . -iname "*mantis*"` in the worktree root, confirming all named scripts/configs/jobs exist as real files (`scripts/extract_mantis_embeddings.py`, `scripts/train_mantis_context_sweep.py`, `scripts/train_mantis_lora.py`, `scripts/infer_mantis_subject_windows.py`, `scripts/infer_mantis_lora_subject_windows.py`, `scripts/gen_commands_mantis{,_lora}.py`, `scripts/verify_mantis_checkpoint.py`, `scripts/pilot_mantis_windowing_layer.py`, `scripts/probe_mantis_staging.py`, `scripts/precompute_mantis_raw_signal_cache.py`, and the full `jobs/*mantis*.sh` set including `_nibi`/`_rorqual` cluster variants).

**Not independently verified in this pass** (relied on the cited docs' own statements rather than re-deriving): the exact real safetensors byte counts and tensor-shape reads (taken from `TSFM_MANTIS_IMPLEMENTATION_PLAN.md`'s own reported HTTP-range-request results, not re-fetched from HuggingFace in this session); the full contents of `TSFM_MANTIS_IMPLEMENTATION_PLAN.md` past line 856 (sections §5 onward, e.g. §5.1 MantisV2 detail, §5.5/§5.8 task-scope discussion, §13 pilot specifications) — cross-checked only indirectly via `MANTIS_CLAUDE.md`'s status-log summaries of those sections' outcomes, which is why this document cites `MANTIS_CLAUDE.md` as the primary source for pilot results rather than the plan document directly; the vendored/pip-installed status of the `mantis` package on the actual cluster (`/home/boshra95/mantis`, `pip install mantis-tsfm`) was not independently re-verified locally, since no cluster filesystem access exists from this environment — taken as stated in `MANTIS_CLAUDE.md`.
