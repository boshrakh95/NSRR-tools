# Mantis Implementation Plan

> **Purpose**: authoritative plan for Mantis as TSFM baseline #3 of 3 (OSF #1,
> PhysioOmni #2). Created 2026-08-22 as a skeleton on the
> `physioomni-implementation` branch.
>
> **Expanded from skeleton to full, cluster-runnable plan on 2026-08-27**,
> after reading `/home/boshra95/mantis` in detail, reading the three Mantis
> checkpoints' real safetensors headers over the network, and re-checking
> every data claim against real HDF5 files. **§1.0 lists the five claims in
> the skeleton that turned out to be wrong.** Nothing below is inherited on
> trust: every factual statement is either marked with how it was verified,
> or explicitly marked as unverified.
>
> **Why Mantis and not MOMENT**: see
> `docs/TSFM_THIRD_MODEL_DECISION.md`. Short version — a 2025 study ran
> nearly this experiment on eight NSRR-family sleep datasets, found Mantis
> beat CBraMod (the leading EEG-specific FM) on all eight, and explicitly
> excluded MOMENT as impractical to fine-tune. Mantis is also 8.1 M params
> against MOMENT-large's 346 M.

---

## 0. Hard constraints — read before touching anything

### 0.1 Worktree / branch isolation

| Directory | Branch | Owner |
|---|---|---|
| `/home/boshra95/NSRR-tools` | `osf-implementation` | OSF session |
| `/home/boshra95/NSRR-tools-omni` | `physioomni-implementation` | PhysioOmni session |
| **`/home/boshra95/NSRR-tools-mantis`** | **`mantis-implementation`** | **this session** |

**Write only inside `/home/boshra95/NSRR-tools-mantis` on
`mantis-implementation`.** Never write to the other worktrees, never
`git checkout` another branch here. All worktrees share one `.git`, so a
stray write or checkout breaks another live session's work.

**Reading the other worktrees is unnecessary** — this branch forked from
`physioomni-implementation` and already carries a full copy of every OSF and
PhysioOmni file in its own checkout. Read them *here*.

**One file outside this worktree is in scope**: `/home/boshra95/.vscode/launch.json`,
the shared VSCode debug config all three sessions already use (it already
contains `🧬 OSF …` and `🫀 PhysioOmni …` entries pointing at their own
worktrees). Mantis entries are **appended** with a `🦗 Mantis …` prefix and
absolute `/home/boshra95/NSRR-tools-mantis/...` paths. **Never edit or
reorder an existing entry** — append only.

### 0.2 Total file isolation — never edit shared files

Everything Mantis needs must be a **new, Mantis-specific file**.

**Never edit**: `CLAUDE.md` (status goes in `MANTIS_CLAUDE.md`), the SleepFM
pipeline (`scripts/gen_commands.py`, `train_context_sweep.py`,
`infer_subject_windows.py`, `src/nsrr_tools/datasets/context_window_dataset.py`),
anything `osf_*` / `train_osf*` / `infer_osf*`, anything `physioomni_*`, or
any config/registry/job script belonging to another model.

**Always create parallel `*_mantis_*` files** — the full list is §6.1.

Results → `/scratch/boshra95/psg/unified/results/phase0_mantis{,_lora}`.
Embeddings → `/scratch/boshra95/psg/unified/embeddings/mantis_30sec{,_plus}`.
Raw cache → `/scratch/boshra95/psg/unified/mantis_raw_signal_128hz`.
Logs → `logs_mantis/`, `logs_mantis_lora/` (separate per stage — sharing one
corrupted OSF's and PhysioOmni's status files; see §4.10).

**The one documented exception**: `src/nsrr_tools/models/sequence_head.py` is
dim-agnostic (`input_dim` is a constructor arg). **Import it, don't edit it.**

---

## 1. Model facts — re-verified against the real repo and checkpoints 2026-08-27

Repo cloned read-only at **`/home/boshra95/mantis`** (commit `9018b98`,
2026-08-17). Install: `pip install mantis-tsfm` (v1.1.0).
**License: Apache-2.0** (repo `LICENSE`; all three HF model cards declare
`apache-2.0` — confirmed live via the HF API).

### 1.0 Corrections to the 2026-08-22 skeleton — read these first

Six claims in the skeleton (or in this plan's own earlier drafts) were wrong
or incomplete. They are corrected in place below; listed here so nobody
re-derives them from the old text.

| # | Skeleton claimed | Reality (how verified) |
|---|---|---|
| 1 | Frozen embedding is `(N, C, 256)`; `FLAT_DIM = 6×256 = 1536` | Only for `output_token='cls_token'` at the last layer. **The Mantis authors' own documented recipe for frozen feature extraction is `return_transf_layer=2, output_token='combined'`** → **512** per channel (README's own table + `getting_started/intermediate_layers.ipynb`). This is a live decision, not a detail — see §3.3. |
| 2 | `MantisV1(seq_len=3840, num_patches=240).from_pretrained(...)` gives a 240-patch model | **False, and it crashes.** `PyTorchModelHubMixin.from_pretrained` *constructs a new model* from the repo's `config.json` (`seq_len:512, num_patches:32` — fetched and read directly); `MantisV1.from_pretrained` copies back only `return_transf_layer`/`output_token`/`pre_training`/`hidden_dim`/`device`. Passing `num_patches=240` as a kwarg *does* override the config (verified in `huggingface_hub/hub_mixin.py:540`) but then the load raises `RuntimeError: size mismatch for …pos_encoder.pe`. Verified empirically: **`torch.nn.Module.load_state_dict` raises on a shape mismatch even with `strict=False`** (reproduced on torch 2.5.1). **We must load manually — §3.4.** |
| 3 | `MantisV2` has "the same FLOPs" as `Mantis-8M` | **False.** Read from the real safetensors headers: V2's attention is `wQKV [768,256]` / `wO [256,256]` — attention inner dim **256**, not V1's 1024 — and its MLP is SwiGLU `[1024,256]`/`[256,512]`. **V2 is ≈2× cheaper per token than V1**, not equal. Its conv kernel is 41, not 17. Its LoRA target names are `wQKV`/`wO`, not `to_qkv`/`to_out.0`. |
| 4 | "Our data is 128 Hz, 6 fast channels (`EEG, LOC, ROC, EKG, EMG, Airflow`)" | **True for SHHS only.** That statement came from one SHHS file. Measured across all four cohorts (250-subject random samples each, §2.1): APPLES/MrOS/STAGES carry **8** channels with *different names* (`C3-M2`,`C4-M1`,`CHIN`,`LLEG`…), SHHS carries 6 and **its 6th is `Airflow` for ~75 % of subjects and `Thor` for ~25 %**. A fixed canonical slot map with per-slot candidate lists is mandatory — §2.2. |
| 5 | The synthetic (CauKer) checkpoint is a separate thing to go find | **It is `paris-noah/MantisPlus`**, already in the checkpoint table. README: *"We have open-sourced CauKer 2M, the synthetic data set we used to pre-train the two versions of Mantis, resulting in MantisPlus and MantisV2 checkpoints."* MantisPlus is **byte-identical in architecture to Mantis-8M** (key-for-key diff of the two safetensors headers: only `tokgen_unit.scalar_encoders.{0,1}.scales`, a deterministic 9-element constant buffer, differs). **This makes real-vs-synthetic a perfectly controlled ablation costing one config line.** See §5.1. |
| 6 | Dropping `pos_encoder.pe` is the only surgery needed to load the checkpoint | **False — a second shape mismatch exists, found only by actually running the load (2026-09-06, `scripts/verify_mantis_checkpoint.py`).** `output_token='combined'` doubles `self.hidden_dim` in `MantisV1.__init__`, which also resizes `self.prj` (the pretraining-only contrastive projector) — the checkpoint's 256-dim `prj.{0,1}.{weight,bias}` then collides with our combined-mode model's 512-dim `prj`, raising the identical `RuntimeError: size mismatch` even under `strict=False`. **`prj` must be dropped too.** It is dead weight regardless of shape — `MantisV1.forward` only calls it when `pre_training=True`, never our case. Live-verified against both real checkpoints. See §3.4. |

### 1.1 Checkpoints — real byte sizes and parameter counts

Read directly from each repo's `model.safetensors` header over HTTP range
requests on 2026-08-27 (not from the papers, not from the README):

| Checkpoint | Module | safetensors bytes | Tensors | **Params** | Pretraining data |
|---|---|---:|---:|---:|---|
| `paris-noah/Mantis-8M` | `MantisV1` | 32,466,928 | 156 | **8,112,384** | real time series (mixed corpus; contains "a small portion of EEG") |
| `paris-noah/MantisPlus` | `MantisV1` | 32,467,192 | 158 | **8,112,402** | **CauKer 2M — purely synthetic** |
| `paris-noah/MantisV2` | `MantisV2` | 16,771,648 | 151 | **4,188,690** | CauKer 2M — purely synthetic |
| `fegounna/Utica` | `MantisV1` | — | — | — | self-distillation; not fetched |

- **`Mantis-8M` and `MantisPlus` differ by exactly 2 tensors / 18 params** —
  `tokgen_unit.scalar_encoders.{0,1}.scales`, the constant
  `[1e-4 … 1e4]` scale grid, which MantisPlus persists and Mantis-8M does
  not. Architecturally identical, same FLOPs, same LoRA surface.
- **Both use the legacy `vit_unit.` state-dict prefix.** `MantisV1` registers
  `rename_vit_unit_weights_hook` as a `_load_state_dict_pre_hook`
  (`architecture/version1.py:157-168`) which rewrites `vit_unit.` →
  `transf_unit.` at load time. **The hook only fires through
  `load_state_dict()`** — so load that way, don't hand-rename keys.
- Loading `Mantis-8M` into current `MantisV1` therefore produces
  **exactly two legitimate missing keys** (`…scalar_encoders.{0,1}.scales`,
  deterministically reconstructed by `register_buffer`) plus whatever we
  deliberately drop (§3.4). Assert on that exact set; anything else is a bug.

### 1.2 Architecture (`src/mantis/architecture/version1.py`) — code-verified

```
MantisV1(seq_len=512, hidden_dim=256, num_patches=32, ...)
   assert (seq_len % num_patches) == 0
   patch_window_size = seq_len // num_patches            #  = 16
   transf_depth=6  transf_num_heads=8  transf_dim_head=128  transf_mlp_dim=512
                                        -> attention inner_dim = 8*128 = 1024
```

**`self.seq_len` is stored but never used in `forward()`.** It exists only for
the assert and to derive `patch_window_size`. This is what makes the
240-patch surgery possible at all — verified by reading `MantisV1.forward`,
`TokenGeneratorUnit.forward`, `TransformerUnit.forward` end to end.

**Token generator (`TokenGeneratorUnit`)** — the part that actually matters:

1. Two `Convolution` modules (`tokgen_utils/convolution.py`), each
   `nn.Conv1d(in_channels=1, out_channels=256, kernel_size=17, padding=8)`.
   **`padding = receptive_field // 2` makes this `same`-padding: output
   length == input length**, at full sample resolution. Confirmed by the
   real checkpoint tensor shape `tokgen_unit.convs.0.conv.weight [256,1,17]`.
   `kernel_size = patch_window_size + 1` when `patch_window_size` is even —
   16 → 17.
2. One conv sees `ts_scaler(x)`, the other `ts_scaler(diff(x))`.
   **`ts_scaler` z-scores each input series over its whole length**
   (`(x - mean(x,axis=2)) / (std(x,axis=2)+1e-5)`). So **the convolutional
   path is completely scale-invariant.**
3. Patching is a plain **mean over `patch_window_size` consecutive
   positions**: `reshape(b, num_patches, -1, hidden) -> mean(dim=2)`. There
   is no strided/learned pooling.
4. **In parallel, per-patch `mean` and `std` of the *raw* input** go through
   `MultiScaledScalarEncoder` (`tokgen_utils/encoders.py`), whose scale grid
   is `[1e-4, 1e-3, … , 1e4]`. **This is the only path that sees absolute
   amplitude** — decisive for the normalization question, §3.2.
5. `LinearEncoder(32*2 + 256*2 -> 256)` produces the final 256-d token.

**Consequence, and it is the load-bearing fact of this whole plan**: keeping
`patch_window_size = 16` keeps the conv kernel, the pooling window, and the
per-patch statistics *bit-for-bit the operations pretraining used*. Only the
**number** of tokens changes. `seq_len=3840, num_patches=240` does exactly
that.

**Positional encoding** (`transformer_v1_utils/positional_encoding.py`):
deterministic sinusoidal, `register_buffer('pe', ...)` (persistent → in the
state dict), shape `(max_len, 1, d_model)` with `max_len = num_patches + 1`.
Real checkpoint: `vit_unit.pos_encoder.pe [33, 1, 256]`. `forward` does
`x + self.pe[:x.size(0)]`, so an oversized buffer is harmless; a shape
mismatch on *load* is not (§1.0 #2).

**Transformer** (`transformer_v1_utils/transformer.py`): lucidrains-style
pre-norm ViT blocks, `PreNorm(LayerNorm) -> Attention -> +x`,
`PreNorm(LayerNorm) -> FeedForward -> +x`.

### 1.3 LoRA targets — confirmed against the real checkpoint

Real tensor names in `Mantis-8M/model.safetensors`:

```
vit_unit.transformer.layers.{0..5}.0.fn.to_qkv.weight     [3072, 256]   (bias=False)
vit_unit.transformer.layers.{0..5}.0.fn.to_out.0.weight   [256, 1024]
vit_unit.transformer.layers.{0..5}.0.fn.to_out.0.bias     [256]
vit_unit.transformer.layers.{0..5}.1.fn.net.0.{weight,bias}   [512,256]
vit_unit.transformer.layers.{0..5}.1.fn.net.3.{weight,bias}   [256,512]
```

`target_modules = ["to_qkv", "to_out.0"]` — the **same names OSF uses**, so
`configs/phase0_osf_lora_config.yaml`'s LoRA block transfers verbatim.
`3072 = 3 × 1024` confirms `inner_dim = 1024`.

**Live-verified against both real checkpoints, 2026-09-06**
(`scripts/verify_mantis_checkpoint.py`, checklist 0.3): injecting
`get_peft_model(net, LoraConfig(target_modules=["to_qkv","to_out.0"], r=8))`
produces **exactly 12 LoRA-wrapped Linears** (6 blocks × 2 modules — OSF has
24, since OSF has 12 blocks) and **exactly 221,184 trainable LoRA
parameters ≈ 2.75 % of the 8,037,632 live params** (§1.1). Matches the
hand-derived arithmetic (`6 × (8×256 + 3072×8 + 8×1024 + 256×8)`) exactly —
both checkpoints give identical numbers, as expected from their identical
architecture.

`modules_to_save=["sequence_head"]` — same mechanism, same
`ModulesToSaveWrapper` two-copy gotcha as OSF checklist 2.3 / PhysioOmni
§15.6 (§14.5 below).

### 1.4 Zero `BatchNorm` in the backbone — confirmed

`grep -rn "BatchNorm" src/` over the whole Mantis package returns **only two
hits, both in `trainer/trainer.py`** (the library's own default fine-tuning
head — `nn.BatchNorm1d` — which **we do not use**; we use our own
`sequence_head.py`). The backbone uses `nn.LayerNorm` only. Also relevant:
`MantisTrainer.pretrain()` calls `convert_sync_batchnorm`, a no-op here.

**Therefore `chunk_batch_size` is mathematically inert** (§4.4) — the
per-call batch size cannot change the result. Re-assert this in checklist
0.5 rather than relying on this paragraph.

### 1.5 Channel handling — channel-independent, and we must batch it ourselves

`MantisTrainer.transform()` (`trainer/trainer.py:290-320`), read in full:

```python
if three_dim:
    return concat([ self._transform(x[:, [i], :], ...)[:, None, :]
                    for i in range(x.shape[1]) ], axis=1)
```

Docstring: *"In the multivariate case, each channel is sent independently to
the foundation model."* Input `(n_samples, n_channels, seq_len)`, output
`(N, C*256)` or `(N, C, 256)` with `three_dim=True`.

**Do not use this method.** It runs one Python-loop iteration *and one whole
`DataLoader`* per channel — 6 sequential passes for our 6 channels. Because
`MantisV1.forward` takes `(n, 1, seq_len)` and is channel-independent by
construction, reshaping `(B, C, L) -> (B*C, 1, L)` and doing **one** forward
is exactly equivalent and 6× fewer launches. §4.6.

**Do not average or majority-vote across channels.** Keep `[T, C, D]`, flatten
to `[T, C*D]`, feed `(B, N, C*D)` to the head. That is what all three existing
models already do — verified in code, every one of them calls
`x = w.reshape(N, FLAT_DIM)` and the head never sees a channel axis:

| Model | Per-timestep structure | head `input_dim` |
|---|---|---|
| SleepFM | 4 modality embeddings × 128 | 512 |
| OSF | 2 subtokens × 768 | 1536 |
| PhysioOmni | EEG 200 + EOG 100 + ECG 100 + EMG 100 | 500 |
| **Mantis** | **6 channels × D** | **6×256 = 1536, or 6×512 = 3072** (§3.3) |

**One real architectural difference to state in the paper** (inherent, not a
choice we make): how much cross-channel mixing happens *inside* the backbone.
OSF attends across all 12 leads jointly; SleepFM fuses 6 physical channels
into 4 modality groups; PhysioOmni fuses within each modality but never
across them; **Mantis does none at all** — every channel is encoded in
isolation, so all cross-channel structure must be learned by the head. The
Mantis-on-EEG study found this design *outperformed* CBraMod's multivariate
pretraining in exactly the low-channel-count regime sleep PSG occupies, so it
is not obviously a handicap — but it is a real difference and belongs in
Methods.

---

## 2. Our data — re-measured 2026-08-27, and it is not what the skeleton said

### 2.1 Real per-cohort channel availability (measured, 250-subject random sample per cohort)

Source tree: **fast-channel `/scratch/boshra95/psg/{cohort}/derived/hdf5_signals/*.h5`**.
All files: `sampling_rate=128`, `float16`, one 1-D dataset per channel at the
file root, plus root attrs `channel_names`, `normalization_stats`,
`duration_seconds`, `original_sfreq`. **Datasets are gzip-compressed with
chunk size 38,400 samples (= 300 s = 10 epochs)** — this matters for Stage 2
(§14.3).

| Cohort | Files | Most common key set (count / 250) |
|---|---:|---|
| **apples** | 1,104 | `Airflow, C3-M2, C4-M1, EKG, EMG, LLEG, LOC, ROC` (200); same minus `LLEG` (43); 4 + 3 outliers |
| **shhs** | 8,444 | `Airflow, EEG, EKG, EMG, LOC, ROC` (188) **or** `EEG, EKG, EMG, LOC, ROC, Thor` (62) — nothing else |
| **mros** | 3,933 | `Airflow, C3-M2, C4-M1, CHIN, EKG, LLEG, LOC, ROC` (**250/250, perfectly uniform**) |
| **stages** | 1,513 | same as mros (91); minus `LLEG` (69); `…EKG, LLEG, LOC, RLEG, ROC` with no chin (55); no `EKG` (26); `EMG` instead of `CHIN` (7); 2 others |

Three facts the skeleton missed, each with a design consequence:

1. **Channel *names* differ by cohort.** SHHS's single EEG is literally
   `EEG`; everyone else has `C3-M2` + `C4-M1`. Chin EMG is `EMG` in
   APPLES/SHHS but `CHIN` in MrOS/STAGES. → per-slot candidate lists, §2.2.
2. **SHHS's RESP channel is `Airflow` for ~75 % and `Thor` for ~25 %** —
   the `fast` strategy caps `RESP: 1` and applies
   `modality_groups.yaml`'s priority `Airflow → Thor → ABD → …`, so
   subjects without a nasal-pressure channel fall back to thoracic effort.
   → real, reportable heterogeneity in the RESP slot, §5.5.
3. **STAGES has genuine per-subject gaps** — ~10 % of the sample has no
   `EKG` at all, ~22 % has no chin channel. → the loader must handle
   absent slots, and the fill log must make it visible, §7.

`configs/preprocessing_params.yaml` confirms the mechanism: strategy `fast`
caps `BAS=4, EKG=1, EMG=2, RESP=1` (8 channels max), selected by
`modality_groups.yaml` priority order.

### 2.2 The canonical 6-slot Mantis channel map — DECIDED

Mantis is channel-agnostic (no channel-identity embedding of any kind — the
conv is `in_channels=1` and every channel is a separate forward). So the only
requirements are (a) a **fixed** slot count, so `[T, C, D]` and `input_dim`
are constant across subjects and cohorts, and (b) that each slot carries a
consistent *kind* of signal.

| # | Slot | Candidates (priority order) | Coverage in the sample |
|---|---|---|---|
| 0 | `EEG` | `C3-M2` → `EEG` → `C4-M1` → `O1-M2` | ~100 % all cohorts |
| 1 | `EOG_L` | `LOC` | ~100 % |
| 2 | `EOG_R` | `ROC` | ~99 % |
| 3 | `ECG` | `EKG` → `ECG-L` | ~100 % except STAGES ~90 % |
| 4 | `EMG` | `CHIN` → `EMG` → `LLEG` → `RLEG` | ~98 %; STAGES ~78 % on chin, ~100 % with the leg fallback |
| 5 | `RESP` | `Airflow` → `Thor` → `ABD` | ~99 % |

**Why 6 slots and not 7 (i.e. why no second EEG derivation).** APPLES, MrOS
and STAGES all have both `C3-M2` and `C4-M1`; SHHS has only one generic
`EEG`. A 7th slot would be **permanently absent for 8,444 of 14,994 subjects
(56 %)**. That is exactly the failure mode PhysioOmni's plan §8 flagged — a
structurally zero slice for the majority cohort, which the head will learn to
ignore and which makes SHHS's representation categorically different from the
other three. Duplicating SHHS's single EEG into both slots is worse still
(PhysioOmni §4.5 rejected it for the right reason: it fabricates an r = 1.0
"second channel" unlike anything in pretraining). **One EEG derivation for
every subject in every cohort. Zero-filled slots become rare (≈1–2 %) rather
than the norm.** Documented as a deferred option (§17) exactly as PhysioOmni
did, revisited only if EEG-dependent tasks look degraded.

**Absent-slot contract**: if no candidate resolves, **do not run the backbone
on zeros** — skip the forward and write **exact zeros** into that slot's
`[T, slot, :]` slice, and log it per subject. (Running zeros through Mantis
would *not* produce zeros: `ts_scaler` gives `0/(0+1e-5)=0`, the conv adds its
bias, and the scalar encoders emit the constant `k·b` — a well-defined but
arbitrary non-zero vector. Zeroing the slice is cheaper and gives Stage 1 and
Stage 2 an identical, trivially-reproducible contract.)

### 2.3 What we do NOT have to do

- **No EDF reprocessing.** Every needed channel already exists in the
  fast tree.
- **No resampling.** Mantis has no native sample rate — it sees an array of
  numbers. 128 Hz × 30 s = 3840 samples, and `3840 = 240 × 16` exactly. This
  is a genuine advantage over both previous baselines (OSF needed 128→64,
  PhysioOmni needed 128→200 and 128→500 by FFT).
- **No denormalization.** See §3.2.

**Comparison baseline: `phase0_v3` (fast-channel, paper-primary)** — same as
PhysioOmni, not OSF's `phase0_v3_full`. Follows directly from using the fast
tree, and matches the user's explicit instruction that reduced-channel files
are sufficient.

---

## 3. Feeding a 30-second epoch to Mantis

### 3.1 Windowing — recommended Option D, but **pilot it** (§13.1)

```
our 30 s epoch @ 128 Hz = 3840 samples
Mantis pretrained on       512 samples (32 patches × 16)
```

**Option A — interpolate 3840 → 512. Rejected outright.** Effective 17.07 Hz,
**Nyquist 8.5 Hz**. Destroys sleep spindles (11–16 Hz), beta, and essentially
all EMG/ECG morphology. This is what Mantis's own README recommends
generically (`F.interpolate(..., 512)`); it is wrong for sleep PSG and would
hand a reviewer a free objection. Not the primary, not a fallback.

**Option D — `seq_len=3840, num_patches=240`.** ***Recommended.***
`patch_window_size = 16` and `kernel_size = 17` stay **identical to
pretraining** (§1.2), the full 128 Hz bandwidth is preserved, self-attention
spans the whole epoch so an event straddling any internal boundary is modelled
continuously, and there is exactly one embedding per 30 s epoch with no
pooling heuristic to defend. The 30 s epoch stays the atomic unit, so the
context sweep is directly comparable to OSF, PhysioOmni and SleepFM.

> **The honest caveat**: 241 tokens is **7.3× longer than anything Mantis saw
> in pretraining** (33). The conv tokenizer is length-agnostic and the
> sinusoidal PE is defined at every position, but the transformer's learned
> attention behaviour at that length is untested. This is the single largest
> scientific risk in the plan.

**Option D-interp — Option D with a rescaled positional buffer.** Same as D,
except the regenerated sinusoidal buffer uses
`position = arange(241) * (32/240)` instead of `arange(241)`. The 241 tokens
then occupy exactly the same arc of the sinusoidal PE that 33 tokens occupied
in pretraining. This is the standard ViT position-embedding-interpolation
trick for changing input resolution, it costs one line, and it directly
targets D's one known out-of-distribution axis.

**Option B — 8 sub-windows of 512, embedded independently, mean-pooled.**
Fully in-distribution (33 tokens, kernel 17, patch 16 — everything exactly as
pretrained). Needs `3840 → 4096` resampling to get 8 clean windows (128 →
136.53 Hz, a 6.7 % stretch; benign, far above any band of interest). Costs
8 × 33 = 264 token-positions vs D's 241 — essentially the same compute.
Drawbacks: no attention across sub-window boundaries, and the mean-pool is an
arbitrary heuristic. **This is the fallback if D and D-interp both look bad.**

**Decision: Option D**, on cross-model fairness grounds — D matches the
*interface* every other backbone uses (one forward pass over one 30 s epoch →
one embedding), whereas B adds a **second aggregation stage no other model in
the comparison has**, in the exact place the paper makes its claims. **§13.1
gives the full option-by-option comparison, the reasoning, and the one
staging-probe result that would overturn it.**

Implement all three behind **two** config keys, not one — D and D-interp
produce **identical model input** (both are the plain 3840-sample epoch);
they differ only in the backbone's positional buffer, not in how the epoch is
sliced. Conflating them into one "windowing" enum would make two of its three
values do the same tensor-shaping work for no reason:

- `embedding.windowing: full_epoch | subwindow` — controls
  `epochs_to_model_input()`, i.e. what tensor shape the epoch becomes.
- `embedding.pe_mode: extrapolate | interpolate` — controls
  `load_mantis_backbone()`'s positional buffer (§3.4's `sinusoidal_pe`
  `stride` argument); only meaningful when `windowing: full_epoch`, since
  `subwindow` never changes `num_patches` away from the pretrained 32.

So there are three *runnable* configurations (`full_epoch`+`extrapolate` = D,
`full_epoch`+`interpolate` = D-interp, `subwindow`+`extrapolate` = B), not
three windowing values — matching the two-key split already in §9's config
template. Each differs only in the epoch→model-input step and the backbone's
buffer, ~30 lines total combined, so the confirming measurement is nearly
free and the escape hatch is a config change, not a rewrite. **Do not mix
windowings between Stage 1 and Stage 2**; the
frozen-vs-LoRA comparison depends on the backbone input being identical.

### 3.2 Normalization — feed our z-scored data AS-IS. DECIDED.

Unlike PhysioOmni (which needed the stored `normalization_stats` inverted back
to raw µV and then `/100`), **Mantis needs nothing**. Three independent
reasons, all from code:

1. **The convolutional path is scale-invariant by construction.**
   `TokenGeneratorUnit.ts_scaler` z-scores each input series over its own
   length before either conv. Whatever we multiply the signal by, that path
   sees the same thing.
2. **The only scale-sensitive path is the per-patch `mean`/`std` scalar
   encoder**, whose scale grid is `[1e-4 … 1e4]`. Our night-level z-scored
   data puts per-patch mean/std at **O(1)** — dead centre of that grid. Raw
   µV would put EEG std at O(10) (still fine) but LOC/ROC at O(1e-5) in the
   cohorts where MNE returns volts (PhysioOmni §5.2 measured this: the unit is
   *cohort- and file-dependent*, not a per-channel rule) — i.e. **restoring
   "raw" scale would actively introduce a 5-order-of-magnitude inconsistency
   Mantis's scalar encoder would then have to absorb.** Feeding z-scored data
   removes that problem instead of creating it.
3. **Mantis's own pretraining corpus is z-normalized.** `getting_started/pretrain.py`
   and every notebook apply only `F.interpolate` — no normalization step
   anywhere — because the UCR/UEA archives are z-normalized per series by
   convention, and CauKer is synthetic.

**Do NOT re-normalize per epoch either.** Night-level z-scoring means a quiet
epoch has per-patch std ≪ 1 and a movement/arousal epoch has ≫ 1 — real,
useful relative-amplitude information that the scalar encoders are built to
read. Per-epoch renormalization would destroy it. Pass the stored values
straight through, cast `float16 → float32`.

**Verify, don't assume** (checklist 0.6 / §13.1): on the pilot subset, assert
zero NaN/Inf, non-degenerate per-dimension embedding std, and — cheaply, since
the code already exists in `physioomni_channel_loader.invert_normalization` —
compare against a µV-restored variant on the same subjects. If µV wins
clearly, this decision flips; the plan just changes one config flag
(`data.denormalize: false`).

### 3.3 Which output token / which layer — **DECIDED: `combined @ last`**

Mantis lets you pick *where* to read the encoder (`return_transf_layer`, one
of 6 blocks) and *what* to read (`output_token` ∈ `cls_token` | `mean_token` |
`combined`, where `combined = cat(cls, mean-of-non-cls-tokens)`).

The README recommends, for frozen feature extraction:

> *"the superior performance of the frozen encoder is achieved by using one of
> the intermediate representations together with the aggregated output-token
> strategy … pass `return_transf_layer=layer_idx` and `output_token='combined'`"*
>
> optimal `layer_idx`: **Mantis-8M → 2**, MantisPlus → 1, MantisV2 → 2.

**An earlier draft of this plan defaulted to `combined @ 2` on the strength of
that sentence. That was wrong, for two independent reasons — one empirical,
one methodological.**

#### The empirical reason: the layer recommendation does not reproduce

`getting_started/intermediate_layers.ipynb` ships with stored outputs. Read
directly (GestureMidAirD1, 130 test samples, RandomForest on the frozen
features):

```
Mantis-8M  cls@0..5:      .654  .677  .654  .669  .662  .669     best = layer 1
Mantis-8M  combined@0..5: .677  .700  .700  .715  .685  .692     best = layer 3
MantisPlus cls@0..5:      .654  .692  .631  .669  .677  .669     best = layer 1
MantisPlus combined@0..5: .669  .692  .654  .700  .662  .700     best = layers 3 / 5
MantisV2   cls@0..5:      .677  .677  .715  .700  .708  .685     best = layer 2
MantisV2   combined@0..5: .638  .700  .731  .700  .731  .662     best = layers 2 / 4
```

The authors' own demo **does not select layer 2 for Mantis-8M** — it selects
layer 1 (cls) or layer 3 (combined). With 130 test samples one sample is
0.0077, so the entire layer-to-layer spread is 3–8 samples wide. The README's
table is presumably a UCR-128 average; the *specific layer* is a low-margin,
dataset-dependent pick that does not survive contact with a different dataset.

**What does reproduce is the token choice.** `combined` beats `cls` at **all
six layers** for Mantis-8M (6/6), and at most layers for MantisPlus and
MantisV2 — **including at the last layer** (.692 vs .669 and .700 vs .669).

So the recipe splits into a **robust half** (`combined`) and a **fragile half**
(a specific intermediate layer). Take the robust half.

#### The methodological reason: comparability with the other three backbones

All three other models in the paper harvest the **last layer** — verified in
their own extraction code:

| Model | What is harvested | Layer |
|---|---|---|
| SleepFM | 60 per-patch token embeddings from a native 300 s chunk (`extract_sleepfm_embeddings.py`, `patch_emb`) | last |
| OSF | CLS + mean of patch tokens (`forward_encoding`) | last |
| PhysioOmni | CLS (`forward_features(..., return_all_tokens=False)`) | last |
| **Mantis (decided)** | **CLS + mean of patch tokens (`output_token='combined'`)** | **last** |

Three consequences of truncating Mantis at layer 2 that the earlier draft
under-weighted:

1. **Mantis would be the only truncated encoder in the paper** — a visible,
   easily-challenged asymmetry, in exchange for a benefit the authors' own
   notebook does not reproduce.
2. **Stage 2 would be systematically handicapped.** OSF's LoRA adapts 12/12
   transformer blocks; PhysioOmni's adapts 12/12 per encoder. Truncating at
   layer 2 would let Mantis's LoRA adapt only **3 of 6** — a weaker LoRA
   condition for a reason unrelated to the model.
3. **It would break the MantisPlus ablation.** Mantis-8M's published optimal
   layer is 2; MantisPlus's is 1. Following per-checkpoint layer
   recommendations would make the real-vs-synthetic contrast differ in **two**
   variables (pretraining corpus *and* extraction depth) instead of one —
   destroying the single property that made that pair worth running (§5.1).
   Using the last layer for both keeps it a clean one-variable ablation.

#### Decision

| Config | dim/ch | `input_dim` | Backbone cost | Layer matches other 3? | LoRA depth | Authors' recipe? |
|---|---:|---:|---|---|---|---|
| `cls @ last` | 256 | 1536 | 100 % | ✅ | 6/6 | ❌ |
| **`combined @ last`** ← **chosen** | **512** | **3072** | **100 %** | **✅** | **6/6** | **token yes, layer no** |
| `cls @ 2` | 256 | 1536 | ~55 % | ❌ | 3/6 | layer yes, token no |
| `combined @ 2` | 512 | 3072 | ~55 % | ❌ | 3/6 | ✅ |

**`combined @ last`** — `return_transf_layer: -1`, `output_token: "combined"`,
`embed_dim: 512`, `model.input_dim: 3072`.

**What this choice costs, stated plainly:**
- **We give up the compute/memory saving.** `@ 2` would have halved backbone
  FLOPs and roughly halved Stage 2 activation memory (§4.3) — the single
  biggest lever on whether 240m LoRA fits. We are paying that in order to keep
  the comparison clean, and §4.5's mitigation ladder has to carry the
  difference instead.
- **Mantis's head input becomes the widest in the paper** — 3072, versus
  OSF 1536, SleepFM 512, PhysioOmni 500. This asymmetry already exists and is
  documented in `CLAUDE.md`'s code-reuse assessment (`input_dim` is the
  encoder's output dim and necessarily varies); Mantis extends it to 2× OSF.
  The head *architecture* — `hidden_dim=128, num_layers=1` — stays identical
  across all four models, which is what the paper's "architecture held
  constant" claim actually asserts. **One sentence in Methods.**
- **We are not following the authors' layer recommendation.** Pilot 2 (§13.2)
  measures exactly what that cost, and the number goes in the supplement:
  *"the authors' recommended layer-2 extraction scored X higher on the
  single-epoch staging probe; we used the last layer for comparability with
  the other backbones."* Reporting it is what makes the choice defensible.

**Hard constraint**: Stage 1 and Stage 2 must use the **same** setting.
Stage 2 warm-starts Stage 1's head, so `input_dim` must match, and the
frozen-vs-LoRA contrast is only clean if adaptation is the sole difference.
This value is baked into every saved embedding file — changing it later means
re-extracting the entire population.

### 3.4 How to actually load the checkpoint — manual, not `from_pretrained`

**`from_pretrained` cannot be used for the 240-patch model** (§1.0 #2). The
required sequence, to be implemented once in
`src/nsrr_tools/datasets/mantis_channel_loader.py::load_mantis_backbone()`
and imported by both stages:

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from mantis.architecture import MantisV1

def load_mantis_backbone(repo_id, seq_len, num_patches, return_layer,
                         output_token, device, pe_mode="extrapolate"):
    net = MantisV1(seq_len=seq_len, num_patches=num_patches,
                   return_transf_layer=return_layer,
                   output_token=output_token, device=device)   # builds pe [num_patches+1,1,256]
    path = hf_hub_download(repo_id, "model.safetensors")       # or a local snapshot dir
    sd = load_file(path, device="cpu")
    for key in ["vit_unit.pos_encoder.pe",                     # regenerate, never load
                "prj.0.weight", "prj.0.bias",                  # dead at inference (pre_training=False,
                "prj.1.weight", "prj.1.bias"]:                 #   §1.0 #6) AND shape-mismatched in
        sd.pop(key, None)                                      #   'combined' mode (hidden_dim doubles)
    missing, unexpected = net.load_state_dict(sd, strict=False)  # rename hook fires here
    allowed_missing = {"transf_unit.pos_encoder.pe",
                       "prj.0.weight", "prj.0.bias", "prj.1.weight", "prj.1.bias",
                       "tokgen_unit.scalar_encoders.0.scales",
                       "tokgen_unit.scalar_encoders.1.scales"}   # Mantis-8M only; MantisPlus HAS these
    assert not unexpected, unexpected
    assert set(missing) <= allowed_missing, set(missing) - allowed_missing
    if pe_mode == "interpolate":
        net.transf_unit.pos_encoder.pe.copy_(
            _sinusoidal_pe(num_patches + 1, 256, stride=32 / num_patches))
    return net.eval().to(device)
```

Notes, each verified (`scripts/verify_mantis_checkpoint.py`, both real
checkpoints, 2026-09-06):
- `sd.pop` before `load_state_dict` is what avoids the `RuntimeError` — the
  constructor already built the correct `[num_patches+1, 1, 256]` sinusoidal
  buffer, and the sinusoid is fully determined by position and `d_model`, so
  regenerating is exactly equal to what a 241-position pretrained buffer would
  have been.
- **`prj` must be dropped too — a second, independent shape mismatch**
  (§1.0 #6). `output_token='combined'` doubles `self.hidden_dim`, resizing
  `self.prj`'s `LayerNorm`/`Linear` to 512-dim, which collides with the
  checkpoint's native 256-dim `prj`. It raises the same hard `RuntimeError`
  under `strict=False` that the positional buffer does. `prj` is dead weight
  at inference regardless of shape (`MantisV1.forward` only calls it when
  `pre_training=True`), so dropping it costs nothing.
- Pop the raw `vit_unit.…` spelling: the checkpoint's own key is
  `vit_unit.pos_encoder.pe`, and the rename pre-hook rewrites it to
  `transf_unit.…` *during* the `load_state_dict()` call itself — popping
  after the fact is impossible. `allowed_missing` lists the post-rename
  (`transf_unit.…`) name because that is what `load_state_dict` reports back.
- `allowed_missing` is exactly the set live-verified in §1.1/§1.3. Fail
  loudly on anything else — that is precisely how the `prj` mismatch above
  was caught, not by reading the source more carefully.
- `_sinusoidal_pe(..., stride=32/num_patches)` implements Option D-interp
  (§3.1); with `stride=1.0` it reproduces the constructor's own buffer.

**Download once to a fixed local path** (`/home/boshra95/mantis_checkpoints/`,
via `snapshot_download`) so compute nodes never need network access — same
discipline OSF and PhysioOmni used. Record real byte sizes in
`MANTIS_CLAUDE.md`.

---

## 4. Performance — read this BEFORE writing any training code

Everything here is a **real, measured lesson** from OSF and PhysioOmni, with
Mantis-specific arithmetic added. The single most expensive mistake in this
project so far was spending weeks tuning batch sizes and GPU allocations
around a slowness whose actual cause was never checked. **Do not repeat it.**

### 4.1 Establish achieved FLOP/s before optimizing anything

PhysioOmni's 80m LoRA run was measured at **0.69 TFLOP/s — 0.14 % of an
H100's ~989 TFLOP/s peak.** That number was not computed until weeks of
tuning had gone by, and it made the diagnosis obvious in one step.

> **But "0.14 % of peak" is not a uniform property of this training pattern —
> OSF measured otherwise, and the difference matters.** From
> `docs/LORA_GPU_THROUGHPUT_INVESTIGATION.md` (on `osf-implementation`, real
> single-segment timings cross-checked against `resume.pt` + `sacct`):
>
> | context | N (raw epochs/window) | min/epoch | est. TFLOP/s |
> |---|---:|---:|---:|
> | 30s | 1 | 18.77 | ~2.0 |
> | 10m | 20 | 58.9 | ~12.8 |
> | 40m | 80 | 112.26 | ~18.5 |
> | 80m | 160 | 217.22 | ~19.2 |
>
> Utilization jumps **~6.4× from 30s to 10m**, then **plateaus** (40m→80m is
> +3 %), reaching roughly **65 % of the FP32-dense peak of that MIG slice**.
> So OSF is **overhead-bound at 30s/10m and substantially compute-utilizing at
> 40m+** — one answer does not cover the sweep. This also explains why AMP and
> the `1g.10gb → 3g.40gb` upgrade both measured *zero* speedup: **both were
> tested at 30s, exactly where overhead dominates.**
>
> **Two consequences for Mantis, both actionable now:**
> 1. **Never benchmark a throughput change at 30s.** Measure at 40m+, where
>    most of the sweep's compute budget actually lives (§13.3 item 3).
> 2. **Also note what the table says about wall-time scaling.** 30s→10m is 20×
>    the raw epochs for only 3.1× the time; 10m→40m is 4× for ~2.8×; 40m→80m is
>    2× for ~1.9×. Cost is **strongly sub-linear at short contexts and
>    approaches linear only once compute-bound.** The plan's "compute scales
>    ~linearly with N" shorthand is the *long-context* limit, not the whole
>    curve — do not budget 240m by multiplying 30s by 480.

**First real training run: compute achieved FLOP/s and compare to peak.**
`(items × epochs_per_window × channels × FLOP_per_channel_epoch × 3) / seconds`.
Use §4.8's per-channel-epoch numbers. If it is under ~5 % of peak, stop tuning
batch sizes and find out why.

**Build this into the code, not into a one-off notebook**: `train_mantis_lora.py`
and `train_mantis_context_sweep.py` both log
`achieved_TFLOPs` and `pct_of_h100_tf32_peak` (495 TFLOP/s) once per epoch.
It costs nothing and makes §4.1 impossible to skip.

### 4.2 TF32 — enable it in line one of the training script

**PyTorch 2.5 ships with matmul TF32 OFF by default** (confirmed:
`physioomni_env` runs torch 2.5.1). `torch.backends.cuda.matmul.allow_tf32` is
`False` and `torch.get_float32_matmul_precision()` is `"highest"`, so every
matmul runs at true FP32: **~67 TFLOP/s instead of ~495 on H100.** OSF and
PhysioOmni both ran that way for weeks.

> **Correction, from OSF's own live check** (`docs/LORA_GPU_THROUGHPUT_INVESTIGATION.md`
> §2 on `osf-implementation`, verified in `osf_env`): **`torch.backends.cudnn.allow_tf32`
> is already `True` by default in torch 2.5.1** — the "both flags default to the
> slow path" claim that propagated from the PhysioOmni session is wrong. Only
> the *matmul* flag actually needs changing. Setting the cudnn one is harmless
> and self-documenting, so keep it in the block, but do not describe it as a fix.

```python
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
```

Put it at the top of **every** Mantis script that touches a GPU:
`extract_mantis_embeddings.py`, `train_mantis_context_sweep.py`,
`infer_mantis_subject_windows.py`, `train_mantis_lora.py`,
`infer_mantis_lora_subject_windows.py`.

TF32 keeps FP32's exponent range, needs no `GradScaler`, changes no stored
dtype, leaves checkpoints byte-compatible. **Prefer it to `torch.autocast`** —
autocast on CUDA defaults to float16, adding casts, a scaler, and numerics
that differ from Stage 1's.

**Honest caveat**: OSF measured *zero* speedup from AMP/fp16 on a `1g.10gb`
pilot (61.7 vs 59.9 min/epoch). AMP ≠ TF32, and that pilot was on 1/7 of a
card — but **measure, don't assume**.

**Do not use `dtype=float32` in the log to check whether TF32 is on.** That
line reports storage dtype, which is unchanged. Use per-epoch time, or the
achieved-TFLOP/s number from §4.1.

### 4.3 Memory scales with `micro_batch × epochs-per-window`, not chunk size

The architectural OOM that hit **both** previous models. Stage 2 runs every
raw epoch through the backbone. Even with the forward pass internally chunked,
**all chunks' activations stay live for the shared `backward()`** — one
autograd graph. So peak memory ∝ `micro_batch × N`, where N = 30 s epochs per
context window. Chunking bounds each *call*; it does **not** reduce peak
memory.

Real failures: **OSF OOM'd at 40m and up; PhysioOmni at 10m, 40m and 240m.**
Both fixed it with a per-context gradient-accumulation schedule holding
`effective_batch = 32` constant; PhysioOmni's is additionally **head-aware**
(the transformer head's own O(N²) attention adds a second cost the LSTM's
O(N) doesn't).

**Mantis's numbers do not transfer from either. Here is the arithmetic to
start from, and it must be validated per context (§13.3).**

Per **channel-epoch** activation footprint retained for backward, fp32,
Option D (241 tokens, d=256, 6 layers):

| Component | Tensor | Bytes |
|---|---|---:|
| tokgen: 2 convs + 2 LayerNorms, each `[3840, 256]` | 4 × 3.93 MB | **15.7 MB** |
| transformer, per layer: attn probs `8×241²` + qkv/out/ff | ~4 MB | |
| transformer, 6 layers | | **~24 MB** |
| **total per channel-epoch** | | **~40 MB** |
| **× 6 channels = per 30 s epoch-unit** | | **~240 MB** |

On a whole 80 GB H100 with ~70 GB usable for activations that is a ceiling of
roughly **290 epoch-units** (`micro_batch × N`). For calibration: PhysioOmni
observed a real ceiling near **640 units on 19.6 GB** — i.e. ~30 MB/unit. So
Mantis is ~8× heavier per unit but gets 4× the memory. **The numbers really
are different; do not copy PhysioOmni's table.**

**Two Mantis-specific facts that change the mitigation ladder:**

- **The tokenizer, not the transformer, is the biggest single memory term at
  low depth** — 15.7 MB of the 40 MB, for only 67 MFLOP of the 5.29 GFLOP
  (§4.8). **Gradient-checkpointing just the `tokgen_unit` call costs ~1.3 %
  extra compute and removes ~39 % of activation memory.** That is a far better
  trade than whole-model checkpointing and should be the **first**
  checkpointing rung, not the last.
- **`return_transf_layer=2` (§3.3) removes 3 of 6 transformer layers**,
  dropping ~12 MB/channel-epoch → ~28 MB, ~168 MB/epoch-unit, ceiling ≈ **500
  units**. If Pilot 2 picks `@ layer 2`, the memory problem is materially
  smaller.

**First-pass schedule to validate** (Option D, `combined @ 2`, whole H100,
targeting ≲250 units with ≥2× margin under the ~500-unit estimate):

| context | N | lstm `micro_batch` | units | transformer `micro_batch` | units |
|---|---:|---:|---:|---:|---:|
| 30s | 1 | 32 | 32 | 32 | 32 |
| 10m | 20 | 12 | 240 | 8 | 160 |
| 40m | 80 | 3 | 240 | 2 | 160 |
| 80m | 160 | 1 | 160 | 1 | 160 |
| 120m | 240 | 1 | 240 | 1 | 240 |
| 240m | 480 | 1 | 480 | 1 | 480 |

**240m is over the estimate at the `micro_batch=1` floor in both columns.**
Expect it to need tokgen checkpointing. Build that in from day one (opt-in,
default off) rather than discovering it mid-sweep. `effective_batch=32`
throughout via `accum_steps = 32 // micro_batch`.

### 4.4 Choosing `micro_batch` and `chunk_batch_size`

| | `micro_batch` | `chunk_batch_size` |
|---|---|---|
| What it is | windows per optimizer micro-step | **channel-epochs** per backbone forward call |
| Affects peak memory? | **Yes**, ∝ `micro_batch × N` | Barely |
| Affects gradient math? | No, if `micro_batch × accum = 32` | **No** — pure scheduling knob |
| Affects speed? | Some (occupancy) | **Yes** — fewer, bigger calls amortize overhead |

- **`chunk_batch_size` is the throughput knob.** OSF measured a **3.28×
  speedup** raising it 16→64. PhysioOmni measured **no difference** at all
  (4.2 vs 4.05 s/subject) — the lesson is *measure it*, not *copy the number*.
  It is safe here because Mantis's backbone has **zero `BatchNorm`** (§1.4).
- **The mechanism behind OSF's 3.28×, made explicit** (from
  `LORA_GPU_THROUGHPUT_INVESTIGATION.md` §4): a training step's backbone work
  is `micro_batch × N` items split into `ceil(items / chunk_batch_size)` calls.
  At 30s with `micro_batch=32, N=1, chunk=64`, OSF got **one half-empty call**
  — per-call launch and Python overhead dominated. At 40m it got 40 full calls
  and became compute-bound. **`chunk_batch_size` only helps once there are
  enough items to fill several chunks**, which is exactly why it did nothing
  for PhysioOmni's *extraction* (whole nights, always plenty of items) and
  3.28× for OSF's *short-context training*.
- **For Mantis, `chunk_batch_size` counts channel-epochs, not epochs — and
  this is a real structural advantage.** One window-epoch is **6** items,
  because all 6 channels go through the same encoder in one call (§4.6). At
  30s with `micro_batch=32`: OSF has 32 items, **Mantis has 192**. So Mantis
  enters the compute-bound regime at a *much shorter context than OSF did*,
  and should be far less overhead-bound at 30s — the regime where every
  previous throughput fix on this project measured nothing. **Predicted, not
  measured: confirm in Pilot 3 (§13.3).**
- Set `chunk_batch_size` to a multiple of 6 so chunks land on epoch boundaries
  and no call is ragged. **Start at 192** (= 32 epochs × 6 channels) for
  extraction, and at `min(192, micro_batch × N × 6)` for Stage 2.
- **Aim for few, large, evenly-sized calls.** At `micro_batch=1, N=160` there
  are 960 channel-epochs; chunk 192 → 5 even calls. Good.
- **`micro_batch=1` is a last resort**, not a default. On a whole 80 GB H100
  the short contexts should sit well above it.

### 4.5 Ask for a whole H100, not a MIG slice

`sinfo` confirms Fir has whole-card nodes (`gpu:h100:4` on `fc[10101-10120,
10201,10203,10405,10611-10620]`, 48 cores / 1.15 TB each) alongside the MIG
nodes — re-checked 2026-08-27. `--gpus=h100:1` gives 80 GB and 7/7 of the SMs.
**`sbatch --test-only` previously showed identical queue-start estimates for
MIG vs whole card**, so the extra compute and memory appeared free.

**`--time` dominates queue position far more than GPU type.** Measured on the
same job: `--time=36:00:00` queued 4 days out; `--time=12:00:00` started the
same day. With per-epoch checkpointing plus auto-resume, a short wall-time
request is nearly free. **Default to `--time=4:00:00` for training** (what
PhysioOmni's LoRA job script settled on) and let auto-resume handle the rest.

> **Two sessions disagree about the whole card, and the disagreement is worth
> resolving explicitly rather than silently picking one.**
> `LORA_GPU_THROUGHPUT_INVESTIGATION.md` §5 **recommends against**
> `--gpus=h100:1` for OSF: since `1g.10gb → 3g.40gb` (3× the compute) gave
> *zero* speedup, a further 2.33× jump is unlikely to help, and it spends real
> shared-account allocation for an unproven benefit. That reasoning is sound —
> **but it is a throughput argument, and it was measured at 30s, in the
> overhead-bound regime.**
>
> **For Mantis the justification is memory, not throughput, and it is not
> optional.** §4.3's arithmetic puts the activation cost at ~240 MB per
> epoch-unit. A `2g.20gb` slice (~19.6 GB usable — the exact figure in
> PhysioOmni's OOM logs) holds roughly **80 epoch-units**; 240m needs 480 at
> the `micro_batch=1` floor. It is not a question of speed: **the long contexts
> cannot run at all on a MIG slice.** Request the whole card because we need
> the 80 GB, treat any throughput gain as a bonus, and **measure it at 40m, not
> 30s** (§13.3). PhysioOmni reached the same allocation for the same
> memory-driven reason on 2026-08-22.

**Mitigation ladder, in order (revised for Mantis, §4.3):**
1. Whole H100 (`--gpus=h100:1`) — zero speed cost.
2. Lower `micro_batch`, raise `accum_steps` to hold `effective_batch=32`.
3. **Gradient-checkpoint `tokgen_unit` only** — ~1.3 % compute for ~39 %
   memory. Mantis-specific; strictly better than rung 4.
4. Full per-chunk gradient checkpointing — real compute cost, opt-in,
   default off.
5. Cap the LoRA condition at the longest tractable context and **report the
   ceiling explicitly** rather than silently omitting points.

### 4.6 Batch channels — do NOT use Mantis's own `transform()` loop

`MantisTrainer.transform()` calls `self._transform(x[:, [i], :])` **once per
channel in a Python loop**, each with its own `DataLoader` — 6 sequential
passes for our 6 channels (§1.5).

**In our code, reshape `(B, C, L) → (B*C, 1, L)`, do ONE forward, reshape back
to `(B, C, D)`.** Legitimate because the model is channel-independent by
construction (`Conv1d(in_channels=1)`), and it turns 6 small launches into 1
large one. This is the "one encoder, all channels batchable" advantage that
makes Mantis attractive versus PhysioOmni's four separate tiny encoders.

### 4.7 Why Mantis should use the GPU well

PhysioOmni's efficiency ceiling is architectural: its EOG/ECG/EMG encoders
have `hidden_dim=100`, **not even a multiple of 8**, so tensor cores can't be
used cleanly at any batch size — four separate small encoders make it worse.

**Mantis's `hidden_dim=256` (attention `inner_dim=1024`, qkv `[3072,256]`) is
tensor-core friendly**, it is one encoder, and all channels batch into a
single call. So although Mantis costs ~19× PhysioOmni's FLOPs per epoch, it
should achieve *far* higher utilization — plausibly **faster in wall-clock
despite more FLOPs**. **Verify with the Pilot-3 throughput measurement
(§13.3); do not take it on faith.**

### 4.8 Cost per 30-second epoch — recomputed 2026-08-27 from real tensor shapes

Per **channel-epoch**, Option D (241 tokens, `d=256`, `inner=1024`, `mlp=512`,
6 layers):

| Term | MACs | GFLOP |
|---|---:|---:|
| tokenizer: 2 × Conv1d(1→256, k=17) over 3840 samples | 33.4 M | 0.067 |
| projections (qkv 786 k + out 262 k + ff 262 k) × 241 tok × 6 layers | 1.895 G | 3.79 |
| attention (QKᵀ + AV), 8 heads × 241² × 128 × 2 × 6 layers | 714 M | 1.43 |
| **total forward, one channel, one 30 s epoch** | | **5.29** |
| **× 6 channels** | | **31.7** |
| same at `return_transf_layer=2` (3 of 6 layers) | | **~16.4** |

Cross-model (updated — the skeleton's MantisV2 row was wrong, §1.0 #3):

| Model | Tokens/epoch | GFLOP/epoch (6 ch) | vs PhysioOmni |
|---|---|---:|---|
| PhysioOmni | 30+60+150+150, **4 separate** encoders | 1.62 | 1.0× |
| OSF | 90 (12 leads, one pass) | 15.6 | 9.6× |
| **Mantis V1, full depth** | 241 × 6 ch | **31.7** | 19.6× |
| **Mantis V1, `@ layer 2`** | 241 × 6 ch | **16.4** | 10.1× |
| **MantisV2** (inner 256, SwiGLU, k=41) | 241 × 6 ch | **~14.4** | ~8.9× |
| MOMENT-small | 480 × 6 ch | 143 | 88× |
| MOMENT-large | 480 × 6 ch | 1912 | 1177× |

**Derived budgets** (to be replaced by measurements, §13.3):
- **Stage 1 extraction, whole population**: 14,994 subjects × ~1,150 epochs ×
  31.7 GFLOP ≈ **547 PFLOP**. At 50 TFLOP/s ≈ 3 h serial; at 10 TFLOP/s ≈
  15 h. Shard into ~6 GPU jobs like PhysioOmni did.
- **Stage 1 embedding storage**: `[T, 6, 512]` fp16 ≈ **7.1 MB/subject** →
  **~106 GB** for the full population (`[T, 6, 256]` → ~53 GB). Scratch is at
  8,436 GiB of 19 TiB — fine.
- **Stage 2, one training epoch at 240m**: 57 k items × 480 epochs × 6 ch ×
  5.29 GFLOP × 3 (fwd+bwd) ≈ **2.6 EFLOP**. At 100 TFLOP/s ≈ 7.3 h; at
  30 TFLOP/s ≈ 24 h. **240m LoRA is the binding constraint of the whole
  plan** — resolve its real cost in Pilot 3 before committing.

### 4.9 Data loading — latency-bound, not bandwidth-bound

Measured on `/scratch` (Lustre), disjoint cold subject groups:

```
stat only (5 × exists())        11.6 ms/subject
+ open & parse npy header      104.5 ms/subject   (~20 ms per open!)
+ 5 window reads, N=1  (30s)   374.0 ms/subject
+ 5 window reads, N=160 (80m)  399.8 ms/subject
full np.load of all channels   366.9 ms/subject
```

1. **`mmap_mode="r"` is SLOWER than a full read on Lustre** — 0.68× at 30s,
   0.32× at 80m. Use **seek + one contiguous read** (PhysioOmni's
   `_NpySliceReader`).
2. **Per-operation latency bound** (~20 ms/open, ~12 ms/read-op), not
   bandwidth: N=1 and N=160 cost nearly the same despite ~170× the bytes.
   **Fewer files and fewer read *operations* per subject is the real lever.**
3. **I/O is ~50 % of a 30s epoch but only ~3.5 % of an 80m epoch.** Long
   contexts are ~96 % compute — no DataLoader tuning helps them.

**Mantis-specific design consequence (§14.3): store the Stage 2 raw cache
epoch-major, `[T_epochs, 6, 3840]` float16, one `.npy` per subject.** A window
of N consecutive epochs across all 6 channels is then **exactly one contiguous
byte range** — 1 open + 1 seek + 1 read. OSF's cache is channel-major
(`[12, n_samples]`), which needs 12 strided reads for the same window. This is
a real improvement, available for free because we design it now instead of
retrofitting.

Latency-bound I/O parallelizes near-linearly: derive `num_workers` from
`SLURM_CPUS_PER_TASK` (don't hardcode 2), and request 8 CPUs.

### 4.10 Other real bugs worth not repeating

- **Non-atomic `meta.json` writes.** OOM-killed workers left 81 zero-byte
  `meta.json` files with intact `.npy` siblings; an existence-only
  `cache_exists()` treated them as done and training died with
  `JSONDecodeError`. **Write cache metadata via temp file + `os.replace`.**
- **Stale shape caches.** A 16-subject debug-era `shape_cache.json` silently
  capped every later dataset build. Invalidate automatically or document
  loudly.
- **`torch.save()` does not create parent directories.**
- **Separate `logs_dir` per stage.** Stage 1 and Stage 2 sharing one log
  directory corrupted each other's `.log` and status `.jsonl` files.
- **Warm-start readiness must gate on `metrics.json`, not `best_model.pt`.**
  `best_model.pt` is written from epoch 1; PhysioOmni had a 120m run branch
  off an unconverged 30s checkpoint because of this.
- **`metrics.json`'s `training_time_min` is wrong for any multi-resume run**
  (found on OSF, `LORA_GPU_THROUGHPUT_INVESTIGATION.md` §3, **not fixed there**).
  `t0 = time.time()` resets on every script (re)start, but `history` — and so
  `n_epochs_run` — correctly persists across resumes via `resume.pt`. The
  numerator covers only the final segment while the denominator counts every
  epoch ever run, so per-epoch cost is silently **undercounted**. It made OSF's
  own timing table nonsensical (10m appearing faster than 30s, some zeros).
  **Fix this in Mantis's scripts from the start**: carry cumulative elapsed
  time inside `resume.pt` and add to it, rather than measuring from the current
  process's start. Cheap now, and it is the number every wall-time and
  throughput decision downstream depends on.
- **Build a dedicated `mantis_env`.** Do not reuse `osf_env`/`physioomni_env` —
  each has an `nsrr_tools_src.pth` pointing at ONE worktree's `src/`.
- **`nsrr_tools.core` eagerly imports `pyedflib`**, which is not in
  `osf_env`/`physioomni_env`. Put the channel loader under
  `src/nsrr_tools/datasets/`, not `core/`. Re-verify for `mantis_env`.
- **Compute Canada ships a stub `pyarrow` wheel.** Copy `physioomni_env`'s
  `pyarrow_arrow_module.pth` into `mantis_env` or `df.to_parquet()` fails at
  inference time.
- **Never launch multiple concurrent CPU debug/smoke-test subjects on the
  login node — even 2-3.** Found 2026-09-06 running checklist 1.3's smoke
  test: three single-subject extractions launched at once (APPLES, SHHS,
  STAGES) each ran ~40+ minutes with **zero** progress, not because of a
  code bug but because the shared login node had `uptime` load average
  13–19 across **87 other users**, and each of our own processes was
  independently spawning ~34 threads — real oversubscription on top of real
  external load. Killed all three and re-ran **one at a time**, with
  `OMP_NUM_THREADS=8`/`MKL_NUM_THREADS=8` set, and got a clean ~2.6–3.2
  min/subject. **CPU debug runs must be serialized and thread-limited**,
  never run in parallel "for speed" on a login node — that is exactly the
  "don't run anything long here" rule this project already has, just
  triggered by concurrency rather than a single long job.

---

## 5. The skeleton's open questions — resolved

### 5.1 Which checkpoint? — `Mantis-8M` primary, `MantisPlus` as the contamination ablation

**Primary: `paris-noah/Mantis-8M`** (`MantisV1`, 8,112,384 params).
- It is the checkpoint the decisive sleep-staging evidence
  (`docs/TSFM_THIRD_MODEL_DECISION.md` §2) was produced with.
- Its documented frozen-optimal layer is 2 (§3.3).
- Real-data pretrained, so it is the fair "generic TSFM" arm.

**Ablation: `paris-noah/MantisPlus`** (`MantisV1`, 8,112,402 params, **CauKer
2M synthetic only**).
- **Architecturally identical to Mantis-8M** — the two safetensors headers
  differ by exactly one constant buffer (§1.1). Same FLOPs, same LoRA surface,
  same `input_dim`, same code path. **Swapping it in is one config line:
  `embedding.repo_id`.**
- This is the cleanest contamination story available to this paper: a model
  that has **provably never seen a physiological signal**, against OSF's
  quantified 87.7 % SHHS pretraining overlap. And because the architecture is
  held exactly constant, "does physiological/real pretraining data matter at
  all?" becomes a controlled one-variable ablation rather than a model swap.
- Its own documented optimal frozen layer is **1**, where Mantis-8M's is 2 —
  which is precisely why §3.3 rejects per-checkpoint layer selection and uses
  the **last layer for both**. Following the per-checkpoint recommendations
  would make this ablation differ in *two* variables (pretraining corpus AND
  extraction depth) and destroy the only property that makes it worth running.
- **Scope: Stage 1 only**, at the winning windowing, all 6 contexts,
  `lstm` + `transformer`, all 5 tasks. Separate embedding dir
  (`mantis_30sec_plus`), separate results dir (`phase0_mantis_plus`),
  separate registry. Roughly doubles Stage 1 extraction and Stage 1 training
  cost; Stage 1 is the cheap stage, so this is affordable.

**`MantisV2` — documented, not run.** Genuinely attractive on two axes
(≈2× cheaper per token, and RoPE means **no positional surgery at all** for
3840 samples — `kernel_size=41` is decoupled from patch size, so
`num_patches=240` keeps patch=16 exactly as pretrained). But: it is a
*different architecture* (RMSNorm/SwiGLU/xPos-RoPE), its LoRA target names are
`wQKV`/`wO` not `to_qkv`/`to_out.0`, it is **synthetic-only** so it cannot
carry the real-vs-synthetic contrast on its own, and no sleep evidence exists
for it. Adding it would be a third model family, not a config swap. Note it in
the paper as considered; run only if asked.

**`fegounna/Utica` — not run.** Third-party, needs a pinned revision, no
sleep evidence, adds nothing the Mantis-8M/MantisPlus pair doesn't.

### 5.2 Option D vs Option A? — Never A. Pilot D / D-interp / B. (§3.1, §13.1)

### 5.3 Fast-channel or full-channel? — **Fast (`/scratch/boshra95/psg`)**

Confirmed by measurement (§2.1): every channel the 6-slot map needs is in the
fast tree, for all four cohorts. No `psg_full` dependency, no reprocessing.
Comparison baseline is therefore **`phase0_v3`** (paper-primary), matching
PhysioOmni, not OSF's `phase0_v3_full`. Also matches the user's explicit
instruction that reduced-channel files are sufficient.

### 5.4 Normalization? — **Feed our z-scored data as-is.** (§3.2)

### 5.5 Apnea? — **IN SCOPE. 5 Tier-1 tasks, same as OSF.**

PhysioOmni had to drop apnea because it has **no respiratory pathway at all**
— an architectural absence, confirmed at four independent code locations.
Mantis has no modality concept whatsoever: the RESP slot is just another
1-channel time series through the same encoder. So apnea is back.

**This makes Mantis the only one of the two general baselines directly
comparable to OSF on `apnea_binary`** — worth stating as a positive.

**Two honest caveats to report, both measured (§2.1):**
1. The RESP slot is `Airflow` for ~99 % of APPLES/MrOS/STAGES but for SHHS it
   is `Airflow` (~75 %) or `Thor` (~25 %) — nasal pressure vs thoracic
   effort, physiologically different signals in one slot.
2. **This is not a Mantis handicap** — it is a property of the shared
   fast-channel preprocessing (`RESP: 1`, priority `Airflow → Thor → …`), so
   SleepFM's own RESP modality group has exactly the same composition. The
   comparison stays matched. Say so explicitly rather than leaving it to be
   discovered.

The extraction fill log must record which key filled the RESP slot per
subject, so this can be quantified exactly rather than sampled (§7).

### 5.6 Heads? — Stage 1 all three; Stage 2 `lstm` + `transformer` only

Stage 1 (`lstm`, `transformer`, `mean_pool`) matches OSF's and PhysioOmni's
Stage 1 scope and is cheap (embeddings are precomputed). Stage 2 defers
`mean_pool` — the same decision both previous models made, for the same
reason (real LoRA compute cost, not worth tripling the sweep before the first
two heads' numbers exist).

### 5.7 Full fine-tuning instead of LoRA? — **No. LoRA, for method parity.**

At 8.11 M params full FT is genuinely cheap and Mantis's own tooling supports
it. But the paper's claim is a **matched-protocol** comparison across four
backbones; changing the adaptation method for one of them confounds
"backbone" with "adaptation method" and hands a reviewer a real objection.
LoRA it is, with `["to_qkv", "to_out.0"]` exactly as OSF.

Documented as a **supplementary** ablation, not a substitute: at 30s only,
one task, one head, full FT vs LoRA — cheap, and it would answer "was LoRA
the bottleneck?" directly. Do it only if the main sweep finishes with time to
spare, and report it as supplementary.

### 5.8 Task scope and expected run counts

**5 Tier-1 tasks** — `sex_binary`, `sleep_efficiency_binary`, `bmi_binary`,
`age_class`, `apnea_binary` — exactly OSF's set (registry fields, dataset
lists, `n_size`, and notes copy verbatim from `experiments/v2_osf_registry.yaml`;
only the config/results/logs paths change). Sleep staging (seq2seq) is out of
scope for all three baselines.

| Sweep | Runs |
|---|---:|
| Stage 1, Mantis-8M | 5 tasks × 3 heads × 6 contexts = **90** |
| Stage 1, MantisPlus (ablation) | 5 × 2 × 6 = **60** |
| Stage 2, Mantis-8M LoRA | 5 × 2 × 6 = **60** (only the 30s of each pair trained from scratch; the other 5 branch from it, §14.5) |

---

## 6. Stage 1 — file map

### 6.1 File isolation table

| Purpose | SleepFM (ref only) | OSF (ref only) | PhysioOmni (ref only) | **Mantis (new)** |
|---|---|---|---|---|
| Backbone loader / channel loader | *(inline)* | `osf_channel_loader.py` | `physioomni_channel_loader.py` | **`src/nsrr_tools/datasets/mantis_channel_loader.py`** |
| Embedding config | `phase0_v3_config.yaml` | `phase0_osf_config.yaml` | `phase0_physioomni_config.yaml` | **`configs/phase0_mantis_config.yaml`** |
| Extraction script | `extract_sleepfm_embeddings.py` | `extract_osf_embeddings.py` | `extract_physioomni_embeddings.py` | **`scripts/extract_mantis_embeddings.py`** |
| Extraction job (GPU / CPU) | `extract_embeddings_gpu.sh` | `extract_osf_embeddings_gpu.sh` | `extract_physioomni_embeddings_{gpu,cpu}.sh` | **`jobs/extract_mantis_embeddings_{gpu,cpu}.sh`** |
| Dataset class | `context_window_dataset.py` | `osf_context_window_dataset.py` | `physioomni_context_window_dataset.py` | **`src/nsrr_tools/datasets/mantis_context_window_dataset.py`** |
| Training script | `train_context_sweep.py` | `train_osf_context_sweep.py` | `train_physioomni_context_sweep.py` | **`scripts/train_mantis_context_sweep.py`** |
| Training job | `train_context_sweep_gpu.sh` | `train_osf_context_sweep_gpu.sh` | `train_physioomni_context_sweep_gpu.sh` | **`jobs/train_mantis_context_sweep_gpu.sh`** |
| Inference script | `infer_subject_windows.py` | `infer_osf_subject_windows.py` | `infer_physioomni_subject_windows.py` | **`scripts/infer_mantis_subject_windows.py`** |
| Inference job | `infer_subject_windows_gpu.sh` | `infer_osf_subject_windows_gpu.sh` | `infer_physioomni_subject_windows_gpu.sh` | **`jobs/infer_mantis_subject_windows_gpu.sh`** |
| Registry | `v2_registry.yaml` | `v2_osf_registry.yaml` | `v2_physioomni_registry.yaml` | **`experiments/v2_mantis_registry.yaml`** (+ `v2_mantis_plus_registry.yaml`) |
| Command generator | `gen_commands.py` | `gen_commands_osf.py` | `gen_commands_physioomni.py` | **`scripts/gen_commands_mantis.py`** |
| Checkpoint verifier | — | — | `verify_physioomni_checkpoint.py` | **`scripts/verify_mantis_checkpoint.py`** |
| Dataset smoke test | `test_context_window_dataset.py` | `test_osf_context_window_dataset.py` | `test_physioomni_context_window_dataset.py` | **`scripts/test_mantis_context_window_dataset.py`** |
| Loader smoke test | — | — | `test_physioomni_channel_loader.py` | **`scripts/test_mantis_channel_loader.py`** |
| Results dir | `.../results/phase0_v3/` | `.../phase0_osf/` | `.../phase0_physioomni/` | **`/scratch/boshra95/psg/unified/results/phase0_mantis/`** |
| Logs dir | `logs_v3/` | `logs_osf/` | `logs_physioomni/` | **`logs_mantis/`** |
| Embeddings dir | `.../embeddings/sleepfm_5sec/` | `.../osf_30sec/` | `.../physioomni_30sec/` | **`.../embeddings/mantis_30sec/`** |
| **`sequence_head.py`** | shared, unmodified | shared, unmodified | shared, unmodified | **shared, unmodified — import it** |

**Everything under "Mantis (new)" is a brand-new file. Nothing under the other
columns is ever opened in write mode.**

### 6.2 Stage-2-only additions (§14)

| File | Role |
|---|---|
| `src/nsrr_tools/datasets/mantis_channel_loader.py` (extended, same file) | `save/load_signal_cache`, `cache_exists`, `get_cached_t_epochs`, `_NpySliceReader` |
| `scripts/precompute_mantis_raw_signal_cache.py` | offline, CPU-only, builds `[T,6,3840]` fp16 per subject |
| `jobs/precompute_mantis_raw_signal_cache.sh` | CPU SLURM job, `--account=def-forouzan` |
| `src/nsrr_tools/datasets/mantis_raw_epoch_dataset.py` | `MantisRawEpochWindowDataset` + `SubjectGroupedSampler` |
| `configs/phase0_mantis_lora_config.yaml` | fork of Stage 1's + `lora:` + `data.raw_signal_cache_dir` + `dataset.stage1_embedding_dir` |
| `scripts/train_mantis_lora.py` | `CombinedMantisLoRAModel` + warm-start + sweep `main()` |
| `scripts/infer_mantis_lora_subject_windows.py` | live-backbone inference |
| `jobs/train_mantis_lora_gpu.sh`, `jobs/infer_mantis_lora_subject_windows_gpu.sh` | GPU jobs |
| `experiments/v2_mantis_lora_registry.yaml` | 10 `(task, head)` entries |
| `scripts/gen_commands_mantis_lora.py` | fork of `gen_commands_physioomni_lora.py` |
| `docs/MANTIS_EXPERIMENTS_GUIDE.md` | operational how-to-run, written incrementally |

### 6.3 Embedding definition

**Output**: `{output_dir}/{dataset}/{subject_id}.npy`, dtype **float16**,
shape **`[T, 6, D]`** where `D = 512` (`combined`) or `256` (`cls_token`),
per §3.3. `T` = number of complete 30 s epochs =
`floor(n_samples_128hz / 3840)`, identical across the 6 slots because every
channel in one HDF5 shares one sample count.

Structurally identical to OSF's `[T, 2, 768]` — so the dataset class is OSF's
fork with three constants changed, and the head's first linear layer sees
`x.reshape(N, 6*D)` exactly as every other model in this project.

**Per-subject fill log**, one JSON line to
`{output_dir}/{dataset}/_channel_fill_log.jsonl`:
`{subject_id, slots_found: {slot: key}, slots_missing: [...], fallback_used:
{slot: key}, resp_source: "Airflow"|"Thor"|..., t_epochs}`. The `resp_source`
field is what turns §5.5's sampled estimate into an exact population number.

**Context-length → epoch-count mapping**: identical to OSF's and
PhysioOmni's, since all three use 30 s epochs —
`30s→1, 10m→20, 40m→80, 80m→160, 120m→240, 240m→480`.
`min_recording_patches: 480` (**not** SleepFM's 2880, which is in 5-second
units — the single easiest place to introduce a silent bug).

---

## 7. `src/nsrr_tools/datasets/mantis_channel_loader.py`

Shared from day one (PhysioOmni's §7 lesson: OSF factored this out only after
Stage 2 needed it, then had to regression-test the refactor). Placed under
`datasets/`, not `core/`, because `nsrr_tools.core.__init__` eagerly imports
`pyedflib` (§4.10) — re-verify for `mantis_env`, don't assume it transfers.

```python
EPOCH_SECONDS  = 30
SOURCE_HZ      = 128
EPOCH_SAMPLES  = EPOCH_SECONDS * SOURCE_HZ          # 3840
N_SLOTS        = 6

SLOT_ORDER = ["EEG", "EOG_L", "EOG_R", "ECG", "EMG", "RESP"]
DEFAULT_CHANNEL_CANDIDATES = {                       # §2.2, measured
    "EEG":   ["C3-M2", "EEG", "C4-M1", "O1-M2"],
    "EOG_L": ["LOC"],
    "EOG_R": ["ROC"],
    "ECG":   ["EKG", "ECG-L"],
    "EMG":   ["CHIN", "EMG", "LLEG", "RLEG"],
    "RESP":  ["Airflow", "Thor", "ABD"],
}

def load_subject_channels(h5_path, candidates) -> tuple[np.ndarray, dict]:
    """-> (x [6, n_samples] float32, fill_info).
    NO resampling (already 128 Hz), NO denormalization (§3.2) — just read,
    cast float16->float32, and place into the fixed 6-slot order. An
    unresolved slot is left as exact zeros AND recorded in
    fill_info['slots_missing'] so the caller can skip its forward pass."""

def get_epoch_count(h5_path) -> int:
    """floor(n_samples / 3840) from a metadata-only read. Mirrors OSF's and
    PhysioOmni's get_epoch_count()."""

def epochs_to_model_input(x, windowing, epoch_start, n_epochs) -> torch.Tensor:
    """[6, n_samples] -> model input for n_epochs epochs (§3.1). Takes only
       `windowing` (NOT `pe_mode` — that's a load_mantis_backbone() concern,
       §3.1's two-key split): D and D-interp produce IDENTICAL tensors here,
       differing only in the backbone's positional buffer.
       'full_epoch' -> (n_epochs*6, 1, 3840)               (D and D-interp)
       'subwindow'  -> (n_epochs*6*8, 1, 512), after a
                       3840->4096 linear interpolation       (Option B)
       Returns epoch-major, channel-minor order so the caller can reshape
       the backbone's output to [n_epochs, 6, D] with no transpose."""

def load_mantis_backbone(repo_id, seq_len, num_patches, return_layer,
                         output_token, device, pe_mode, local_dir) -> nn.Module:
    """§3.4 — manual safetensors load, never from_pretrained. Asserts the
    exact allowed-missing-key set. Used by BOTH stages."""

def sinusoidal_pe(max_len, d_model, stride=1.0) -> torch.Tensor:
    """Regenerates PositionalEncoding's buffer. stride=1.0 reproduces it
    exactly; stride=32/num_patches implements Option D-interp (§3.1)."""

# ── Stage 2 raw-signal cache (§14.3) ──────────────────────────────────────
def cache_path_for(cache_dir, dataset, subject_id) -> Path      # {ds}/{sid}.npy
def save_signal_cache(cache_dir, dataset, subject_id, x, meta)  # atomic meta write
def load_signal_cache_window(cache_dir, dataset, subject_id, e0, n) -> np.ndarray
def get_cached_t_epochs(cache_dir, dataset, subject_id) -> int
def cache_exists(cache_dir, dataset, subject_id) -> bool        # meta parses, not just exists
```

Two things this file must get right, both from real prior failures:
- **`save_signal_cache` writes `meta.json` via temp file + `os.replace`**, and
  `cache_exists()` **parses** the JSON rather than checking existence (§4.10).
- **`load_signal_cache_window` uses seek + one contiguous read**, not
  `mmap_mode="r"` (§4.9). The `[T,6,3840]` epoch-major layout makes this a
  single `f.seek(header + e0*6*3840*2); f.read(n*6*3840*2)`.

---

## 8. `src/nsrr_tools/datasets/mantis_context_window_dataset.py`

Fork of `osf_context_window_dataset.py` — the closest match of the three,
since Mantis's embeddings are 3-D `[T, C, D]` exactly like OSF's `[T, 2, 768]`.
All the K-sampling logic, `SubjectGroupedSampler`, window index math, padding
and `collate_fn` are pure integer arithmetic over `T`/`N` and copy unchanged.

**Only the module-level constants change:**

```python
PATCH_SECONDS       = 30      # each Mantis embedding row = one 30-second epoch
PATCHES_PER_EPOCH   = 1       # epoch and patch are the same unit (as OSF)
EMBED_DIM           = 512     # 'combined' output (§3.3). MUST match model.input_dim/6
N_SUBTOKENS         = 6       # the 6 channel slots (§2.2)
FLAT_DIM            = N_SUBTOKENS * EMBED_DIM     # 3072
FULL_NIGHT_SENTINEL = -1
```

`EMBED_DIM` is fixed by §3.3's decision (512, `combined @ last`). Read it from the config
(`model.input_dim // 6`) and **assert** it against the first loaded `.npy`'s
real shape at dataset-build time, so a mismatched config fails immediately
rather than producing silent garbage.

**Drop** `zero_modality_indices` / `--zero-modalities` and
`_apply_modality_zeroing()` — same as OSF's and PhysioOmni's forks. There is
no 4-modality-group structure to ablate. (A *channel* ablation would be
possible here, since the 6 slots are individually addressable, but it is out
of scope: the user scoped Mantis to two rounds — frozen and LoRA — with no
channel ablation.)

`min_recording_patches: 480`. Same cohort filter, same units as OSF.

---

## 9. `configs/phase0_mantis_config.yaml`

```yaml
# ── Embedding extraction ─────────────────────────────────────────────────────
embedding:
  repo_id:        "paris-noah/Mantis-8M"      # MantisPlus ablation: swap this line
  local_dir:      "/home/boshra95/mantis_checkpoints/Mantis-8M"
  seq_len:        3840        # 30s @ 128Hz  (Option D, §3.1)
  num_patches:    240         # -> patch_window_size 16, kernel 17, as pretrained
  windowing:      "full_epoch"        # full_epoch | subwindow  (tensor shape, §13.1)
  pe_mode:        "extrapolate"       # extrapolate | interpolate  (backbone PE buffer only,
                                      # meaningful only when windowing=full_epoch; §3.1/§3.4)
  return_transf_layer: -1     # LAST layer — matches SleepFM/OSF/PhysioOmni (§3.3).
                              # Deliberately NOT the authors' per-checkpoint
                              # intermediate layer (2 for Mantis-8M, 1 for
                              # MantisPlus): that does not reproduce in their own
                              # notebook, would make Mantis the only truncated
                              # encoder, would halve its LoRA depth vs OSF's 12/12,
                              # and would make the MantisPlus ablation two-variable.
  output_token:   "combined"  # cls_token | mean_token | combined. 'combined' =
                              # cat(cls, mean) — the ROBUST half of the authors'
                              # recipe (beats cls at 6/6 layers in their notebook).
  embed_dim:      512         # 512 for 'combined', 256 otherwise. ASSERTED at runtime.
  chunk_batch_size: 192       # CHANNEL-epochs per forward call = 32 epochs x 6 (§4.4)
  output_dir:     "/scratch/boshra95/psg/unified/embeddings/mantis_30sec"
  datasets: [apples, shhs, mros, stages]

# ── HDF5 signal data (fast-channel tree) ─────────────────────────────────────
data:
  hdf5_dir: "/scratch/boshra95/psg"
  epoch_seconds: 30
  source_hz: 128
  denormalize: false          # §3.2 — feed the stored z-scored values as-is
  # Fixed 6-slot canonical map, measured per cohort 2026-08-27 (§2.1/§2.2).
  # Order here IS the channel axis order of the saved [T, 6, D] array.
  channel_candidates:
    EEG:   [C3-M2, EEG, C4-M1, O1-M2]
    EOG_L: [LOC]
    EOG_R: [ROC]
    ECG:   [EKG, ECG-L]
    EMG:   [CHIN, EMG, LLEG, RLEG]
    RESP:  [Airflow, Thor, ABD]

# ── Context-window dataset ───────────────────────────────────────────────────
dataset:
  embedding_dir: "/scratch/boshra95/psg/unified/embeddings/mantis_30sec"
  label_source:  "/scratch/boshra95/psg/unified/targets_v2/master_targets.parquet"
  task_subject_dir: "/scratch/boshra95/psg/unified/targets_v2/task_subjects"
  sleep_stage_dir:  "/scratch/boshra95/psg"
  context_lengths: ["30s", "10m", "40m", "80m", "120m", "240m"]
  task: "sex_binary"
  task_type: "seq2label"
  datasets: [apples, shhs, mros, stages]
  train_split: 0.70
  val_split:   0.15
  test_split:  0.15
  split_seed:  42            # SAME seed as SleepFM phase0_v3
  windows_per_subject: 5
  # seq2seq params carried for config-shape parity; inert for the seq2label scope
  seq2seq_context_mode: "centered"
  seq2seq_padding_policy: "complete_only"
  seq2seq_max_padding_fraction: 0.5
  min_past_denom: 8
  max_min_past_patches: 40
  min_recording_patches: 480   # 240m in 30s-epoch units — NOT SleepFM's 2880

# ── Sequence head ────────────────────────────────────────────────────────────
# hidden_dim/num_layers held at phase0_v3's seq2label values. Only input_dim
# changes (6 x embed_dim), preserving "architecture held constant, only the
# encoder changes".
model:
  input_dim: 3072            # 6 x 512 (combined @ last, §3.3).
                             # MUST equal 6 * embedding.embed_dim — asserted at
                             # dataset-build time against a real .npy's shape.
  head_type: "lstm"
  hidden_dim: 128
  num_layers: 1
  num_heads: 8
  dropout: 0.3
  num_classes: 2

# ── Training — Stage 1 values held identical to OSF/PhysioOmni Stage 1 ───────
training:
  epochs: 40
  lr: 1.0e-4
  weight_decay: 1.0e-3
  optimizer: "adamw"    # NOTE: not read — hardcoded to Adam in the fork's run loop.
  scheduler: "cosine"   # NOTE: not read — hardcoded CosineAnnealingLR.
  early_stopping_patience: 10
  device: "cuda"        # NOTE: not read — comes from --cpu / torch.cuda.is_available().
  mixed_precision: false        # TF32 instead (§4.2), not autocast/fp16
  class_weights: "auto"
  early_stopping_monitor: "val_auroc"
  weighted_sampler: false
  windows_strategy: "fixed"
  token_budget_minutes: 240
  k_max: 50
  context_lr_overrides:
    "120m": 5.0e-5
    "240m": 5.0e-5
  overfit_epochs: 0
  save_snapshots: true
  snapshot_interval: 5

analysis:
  bootstrap_samples: 0

# REQUIRED — the train_*_context_sweep.py family reads
# cfg["logging"]["results_dir"] via bare bracket access, no default.
logging:
  results_dir: "/scratch/boshra95/psg/unified/results/phase0_mantis"
```

Two dead-config footnotes carried from OSF/PhysioOmni, to **confirm** rather
than assume when the training script is written: `training.optimizer` /
`scheduler` / `device` are likely unread; `dataset.label_source` is
documentation-only (`task_subject_dir` + `split_seed` are what actually
determine the split).

---

## 10. Training and inference scripts

### `scripts/train_mantis_context_sweep.py`
Fork of `train_physioomni_context_sweep.py` (which is itself a fork of
`train_osf_context_sweep.py`), which is the cleanest of the three. Only the
dataset import and the `--wandb-project` default change. **Plus one addition
none of them have: TF32 at the top (§4.2) and the achieved-TFLOP/s log line
(§4.1).**

CLI: `--config --task --task-type --head --context (nargs+) --datasets
(nargs+) --limit --max-items --full-night-epochs --cpu --wandb-project
(default "nsrr-phase0-mantis") --wandb-entity --no-wandb --batch-size
--accum-steps --lr --run-tag`. **No `--zero-modalities`.**

**Keep the function boundaries clean from the start** — `run_epoch`,
`compute_metrics`, `compute_monitor_metric`, `append_to_summary`,
`_classify_failure`, `train_one_context`, `main`. `train_mantis_lora.py`
imports the first five directly, exactly as `train_osf_lora.py` and
`train_physioomni_lora.py` do. `run_epoch()` only ever calls `x.to(device)`,
`x.size(0)` and `model(x, mask)`, which is why Stage 2 reuses it unmodified.

**Output**: `{results_dir}/{task}_{head}/context_{L}/{best_model.pt,
metrics.json,training_curves.csv}` + `{results_dir}/{task}_{head}/summary.csv`
— identical schema to OSF's/PhysioOmni's/SleepFM's, so `analyze`, `collect`,
`build-heatmap` and the plotting/table code work unchanged.

### `scripts/infer_mantis_subject_windows.py`
Fork of `infer_physioomni_subject_windows.py`. Batch-size auto-scaling
reference: `_ref_bs=64, _ref_N=480` — same 30s-epoch unit as OSF and
PhysioOmni, so the same values apply. **Carry their open caveat honestly**:
`_ref_bs=64` has never been GPU-verified for any of the three. Verify in
checklist 1.7 rather than inheriting the assumption a third time.

**Output**: `{results_dir}/inference/{task}_{head}/context_{L}/{split}_windows.parquet`
with the 7-column schema `subject_id, dataset, window_idx, true_label,
pred_label, prob_class0…N`.

---

## 11. Registry and command generator

`experiments/v2_mantis_registry.yaml` — schema copied field-for-field from
`v2_osf_registry.yaml`, all **5 Tier-1 tasks × 3 heads = 15 experiments**,
same `datasets`/`contexts`/`n_size`/`tier`/`notes` per task.

```yaml
config: configs/phase0_mantis_config.yaml
results_dir:   /scratch/boshra95/psg/unified/results/phase0_mantis
inference_dir: /scratch/boshra95/psg/unified/results/phase0_mantis/inference
logs_dir:      /home/boshra95/NSRR-tools-mantis/logs_mantis
python_bin:    /home/boshra95/mantis_env/bin/python

gradient_accumulation:
  enabled: true
  effective_batch: 32
  context_micro_batch: {"30s": 32, "10m": 32, "40m": 32, "80m": 32, "120m": 32, "240m": 32}
```

**Both `inference_dir` and `python_bin` are load-bearing.** `inference_dir` is
read via bare bracket access (`KeyError` without it). `python_bin` has a
fallback — and **the fallback is `/home/boshra95/sleepfm_env/bin/python`**,
hardcoded at every call site in the base `gen_commands.py` — so omitting it
would silently run Mantis jobs in SleepFM's environment.

`experiments/v2_mantis_plus_registry.yaml` — same, `lstm`/`transformer` only,
pointed at `configs/phase0_mantis_plus_config.yaml` (identical to the Stage 1
config except `repo_id`, `return_transf_layer: 1`, `output_dir`,
`embedding_dir`, `results_dir`, `logs_dir`).

`scripts/gen_commands_mantis.py` — structural fork of
`gen_commands_physioomni.py`; same subcommands (`list, probe-batch, train,
infer, analyze, build-heatmap, collect, threshold-tuning, status, runs`), same
deliberate exclusion of figure/table subcommands. `--registry` selects the
Mantis-8M or MantisPlus registry.

**Wall-time tables**: start from `gen_commands_physioomni.py`'s structure but
**do not copy its numbers** — Mantis's compute profile is different again
(one encoder, 19× the FLOPs, better shapes). Seed them from the Pilot-3
measurement (§13.3) and mark them as calibrated-from-one-point until the first
full sweep confirms.

---

## 12. Job scripts

Fork `jobs/train_physioomni_lora_gpu.sh`'s SLURM header — it is the most
current of the three and already encodes §4.5's conclusions:

```bash
#SBATCH --account=def-forouzan_gpu
#SBATCH --time=4:00:00               # short beats big — §4.5
#SBATCH --gpus=h100:1                # WHOLE card, not a MIG slice — §4.5
#SBATCH --cpus-per-task=8            # latency-bound I/O parallelizes — §4.9
#SBATCH --mem=64000M                 # PhysioOmni OOM'd its CPU job at 32000M
#SBATCH --exclude=fc11006,fc11013,fc11010
#SBATCH --signal=B:USR1@120
```

Same two-layer auto-resume: (1) `--signal=B:USR1@120` + a bash trap that kills
Python cleanly and resubmits `sbatch "$0"`, picking up from `resume.pt`;
(2) node-failure `--requeue`, supplied by `gen_commands_mantis.py` at the
*initial* `sbatch`, not baked into the `.sh`. Same per-job JSONL status log
under `logs_mantis/status/`. Same `cd /home/boshra95/NSRR-tools-mantis`,
`source /home/boshra95/mantis_env/bin/activate`, and CUDA-availability
fail-fast check.

CPU jobs (`extract_mantis_embeddings_cpu.sh`,
`precompute_mantis_raw_signal_cache.sh`) use `--account=def-forouzan` (no
`_gpu`), 16 CPUs, `--mem=64000M`.

---

## 13. Step 0 — the pilots, and how each choice was decided

**Framing, because it changed once the cross-model comparison was done
properly (§13.0): Pilots 1 and 2 are no longer open choices.** Fairness with
the other three backbones constrains both of them, and both are decided below
with reasons. What remains is a **confirmatory measurement with a named
escape hatch** — a specific number that, if it came back, would overturn the
decision. Pilot 3 was never a choice at all; it is a measurement that fills in
config values.

### 13.0 The fairness frame — what every model in this paper actually gets fed

Verified by reading each model's own extraction code in this repo, not
inferred:

| | Pretrained on | What we feed it | In distribution? | Harvested | Layer | `input_dim` |
|---|---|---|---|---|---|---:|
| **SleepFM** | 300 s chunk = 60 × 5 s patches | 300 s chunk, **exact** | ✅ exact | 60 per-patch tokens | last | 512 (4 × 128) |
| **OSF** | one 30 s epoch, 12 ch @ 64 Hz, 90 tokens | one 30 s epoch, **exact** | ✅ exact | CLS + mean(patches) | last | 1536 (2 × 768) |
| **PhysioOmni** | patch streams, ≤512 patches | 30 s = 30–150 patches | ✅ well inside | CLS | last | 500 |
| **Mantis** | **512 samples = 32 patches** | **3840 samples = 240 patches** | ❌ **7.3× beyond** | CLS + mean (§3.3) | last | 3072 (6 × 512) |

Two facts follow, and they drive everything below:

1. **Mantis is the only one of the four forced out of distribution at all.**
   We cannot engineer that away — only choose *where to put it*. It belongs in
   the paper's Methods whichever option we pick.
2. **All three others harvest the last layer.** §3.3 already used this to
   settle the output question. It also shapes Pilot 1.

A third, pre-existing asymmetry worth naming so it isn't mistaken for
something Mantis introduces: **SleepFM's atomic unit is 5 s, everyone else's
is 30 s** (`PATCH_SECONDS=5`, `min_recording_patches=2880` for SleepFM vs
`30`/`480` for the three baselines). That was accepted before Mantis existed.
It is also why "Option C" — making Mantis's atomic unit its native 4 s window
— was rejected outright: it would add a *fourth* granularity to the sweep.

---

### 13.1 Pilot 1 — how do we hand Mantis a 30-second epoch?

#### The problem

Mantis was pretrained on series of exactly **512 numbers**, cut into 32
patches of 16 samples. Our 30 s epoch at 128 Hz is **3840 numbers**. Something
must give. Four options were considered; three survive.

#### Option A — interpolate 3840 → 512. **Rejected outright, not piloted.**

What Mantis's own README suggests generically
(`F.interpolate(..., size=512)`). Effective sampling rate 17.07 Hz, **Nyquist
8.5 Hz**.

*Comparison with the others*: none of SleepFM, OSF or PhysioOmni is
bandwidth-reduced anywhere near this. OSF's 128→64 Hz decimation keeps Nyquist
at 32 Hz, above the whole EEG band of interest; PhysioOmni is *up*sampled.

*Why rejected*: destroys sleep spindles (11–16 Hz), beta, and essentially all
EMG and ECG morphology. It would cripple the baseline before it starts and
hand a reviewer a free objection. **Not a fallback either.**

#### Option D — `seq_len=3840, num_patches=240`. ***CHOSEN.***

Tell Mantis the series is 3840 long with 240 patches. `patch_window_size` is
still `3840/240 = 16`, so `kernel_size` is still 17, the patch-mean pooling
window is still 16 samples, and the per-patch mean/std statistics are computed
over the same span. **Every learned weight performs exactly the operation it
performed in pretraining.** The only change is that the transformer sees
**241 tokens instead of 33**.

*Comparison with the others*: this is structurally the *same operation* every
model in the comparison performs — tokenise the epoch into patches, attend
across them. Token counts: SleepFM 60 per chunk, OSF 90, PhysioOmni 30–150,
**Mantis-D 241**. Relative to the comparison set, 241 is unremarkable. It is
only unusual relative to *Mantis's own pretraining*.

*Cost*: **7.3× beyond Mantis's pretraining sequence length.** The conv
tokenizer is length-agnostic and the sinusoidal PE is defined at every
position, but the transformer's learned attention behaviour at that length is
untested. **This is the single largest scientific risk in the plan.**

#### Option D-interp — Option D with a rescaled positional buffer.

Identical to D except the regenerated sinusoidal buffer uses
`position = arange(241) * (32/240)` instead of `arange(241)`, so the 241
tokens occupy exactly the arc of the sinusoid that 33 tokens occupied in
pretraining. The standard ViT position-embedding-interpolation trick for
feeding a bigger input than the model was trained on. One line.

*Fairness*: **identical to D.** Both regenerate a **non-learned** buffer
(`PositionalEncoding.pe` is deterministic sinusoid, `register_buffer`), so
neither is surgery on pretrained parameters. There is no fairness distinction
between them, only an empirical one:

| | local spacing between adjacent tokens | total positional span |
|---|---|---|
| **D** | 1.0 — **matches pretraining** | 241 — **7.3× OOD** |
| **D-interp** | 0.133 — **OOD** | 33-equivalent — **matches pretraining** |

Neither dominates. Genuinely empirical, which is why both are measured.

#### Option B — 8 sub-windows of 512, embedded independently, mean-pooled.

Resample 3840 → 4096 (128 → 136.53 Hz, a 6.7 % stretch, far above any band of
interest) to get exactly 8 clean 512-sample windows; run each at native
settings (33 tokens, kernel 17, patch 16 — **fully in distribution**); average
the 8 embeddings into one per epoch. Compute cost 8 × 33 = 264 token-positions
vs D's 241 — essentially identical.

*Comparison with the others*: **this is where B loses.** It adds a **second
aggregation stage — a mean over 8 sub-windows — that no other model in the
comparison has.** SleepFM, OSF and PhysioOmni each produce their per-timestep
embedding from a single forward pass; only Mantis would have a hand-designed
pooling step inside the backbone. And **aggregation is exactly what this paper
makes claims about** — the entire study is "how does performance change as we
aggregate more context." Introducing a bespoke aggregation stage inside the
one backbone that most needs to look comparable is the worst possible place
for an asymmetry.

*Secondary costs*: Mantis never attends across a sub-window boundary, so an
event straddling one (a spindle, a K-complex, an arousal) is split; and the
128 → 136.53 Hz resample is a second small departure the others don't need.

#### Decision and reasoning

**Option D, with D-interp measured alongside it as a free variant.**

| | D | D-interp | B |
|---|---|---|---|
| Interface matches SleepFM/OSF/PhysioOmni (one forward → one embedding) | ✅ | ✅ | ❌ extra pooling stage |
| Token count unremarkable vs the comparison set (60/90/30–150) | ✅ 241 | ✅ 241 | ✅ 33 |
| In distribution for **Mantis's own** pretraining | ❌ | partly | ✅ |
| Introduces an aggregation step no other model has | ✅ no | ✅ no | ❌ yes |
| Full 128 Hz bandwidth | ✅ | ✅ | ✅ |
| 30 s atomic unit preserved (comparability of the sweep) | ✅ | ✅ | ✅ |
| Extra resampling step | ✅ none | ✅ none | ❌ 3840→4096 |

**The reason, in one sentence**: *D matches on interface and breaks on
distribution; B matches on distribution and breaks on interface — and it
breaks it in the load-bearing place, by adding an aggregation stage to the one
component the paper's claims are about.*

#### Escape hatch — the one result that would overturn this

If the staging probe (§13.4) shows **Option D scoring far below Option B** —
concretely, weighted F1 below ~0.60 where B reaches ~0.75+, i.e. a gap several
times larger than the between-variant noise — then "the model is not working
at this sequence length" beats "the interface is tidier," and we switch to B
and say so explicitly in Methods. Anything smaller than that, take D.

D vs D-interp: take whichever scores higher; if within ~0.01, take plain **D**
(no PE modification at all is the simpler thing to describe).

---

### 13.2 Pilot 2 — which layer, which output token?

**Already decided in §3.3: `combined @ last` (`return_transf_layer: -1`,
`output_token: "combined"`, 512/channel, `input_dim: 3072`).** Read §3.3 for
the full argument; the short version:

- The authors' recipe splits into a **robust half** (`combined` beats `cls` at
  6/6 layers in their own notebook, including the last) and a **fragile half**
  (a specific intermediate layer, which their own demo fails to reproduce —
  it picks layer 1 or 3, not the README's 2, with the whole spread only 3–8
  test samples wide).
- **All three other backbones harvest the last layer.** Truncating Mantis at
  layer 2 would make it the only truncated encoder, would let its LoRA adapt
  only 3 of 6 blocks against OSF's and PhysioOmni's 12/12, and — decisively —
  would break the MantisPlus ablation into a two-variable comparison, since
  Mantis-8M's published optimal layer is 2 and MantisPlus's is 1.

**So Pilot 2 is not a decision. It is the supplementary number.** The same
extraction pass that produces `combined @ last` also produces layer-2 output
for free (capture the intermediate activation while iterating
`transf_unit.transformer.layers` — ~10 lines), so all four cells of §3.3's
table come from one job at zero extra GPU cost:

```
                     dim/ch   input_dim   weighted F1   kappa
cls      @ last        256       1536        ?            ?
combined @ last  ←     512       3072        ?            ?    (chosen)
cls      @ 2           256       1536        ?            ?
combined @ 2           512       3072        ?            ?
```

What we do with it: report it in the supplement as *"the authors' recommended
layer-2 extraction scored X higher on the single-epoch staging probe; we used
the last layer for comparability with the other backbones, and because
per-checkpoint layer selection would have made the Mantis-8M vs MantisPlus
contrast two-variable."* Showing the number we didn't take is what makes the
choice auditable rather than arbitrary.

**Escape hatch**: if `@ 2` beats `@ last` by a very large margin — say >0.08
weighted F1, far beyond anything the authors' own ±0.02 spread suggests — that
would mean the last layer is genuinely broken at 241 tokens rather than merely
suboptimal, and it becomes a *finding about sequence-length extrapolation*
worth acting on. Then we would switch, use layer 2 for **both** checkpoints
(not per-checkpoint layers), and state the comparability cost plainly.

---

### 13.3 Pilot 3 — measurement, not a decision

Nothing to choose. Four numbers that go straight into config files, and one
gate.

1. **Achieved TFLOP/s vs the H100's ~495 TF32 peak** (§4.1), logged by the
   script itself. **This is a gate**: if extraction comes back under ~5 % of
   peak, stop and diagnose before launching a 90-run sweep. That single check
   would have saved weeks on PhysioOmni.
2. **`chunk_batch_size` A/B**: 192 vs 48, matched fresh subject batches, same
   cohort. OSF measured **3.28×** from this knob; PhysioOmni measured
   **nothing**. Ours is unknown, and §4.4 predicts Mantis should be *less*
   sensitive than OSF at short contexts because the 6-channel axis already
   gives 6× more items per forward call.
3. **`torch.profiler` over a handful of batches at 40m** (not 30s — OSF's
   `docs/LORA_GPU_THROUGHPUT_INVESTIGATION.md` §6 is explicit that profiling at
   30s measures the overhead-bound regime and answers the wrong question).
   Gives the actual matmul-vs-everything-else split, settling
   compute-bound-vs-overhead-bound directly instead of via a FLOP estimate.
4. **One training step at each of 30s/10m/40m/80m/120m/240m**, recording
   `torch.cuda.max_memory_allocated()` → rewrites §4.3's estimated
   `context_micro_batch` table into a measured one. **Do not extrapolate from
   one point** — that is exactly what PhysioOmni did and had to redo.

---

### 13.4 The instrument for Pilots 1 and 2 — a single-epoch sleep-staging probe

**Why not val AUROC on a 30 s `sex_binary` run** (what an earlier draft of this
section proposed): the pilot val split is ~165 subjects, where AUROC noise is
roughly ±0.04. A decision threshold of 0.01 sits **well inside** that noise —
it would be picking on coin flips. It is also expensive: three extraction
passes over ~1,600 subjects, twice.

**The instrument instead**: per-30 s-epoch sleep-stage labels, which already
exist for every subject — verified:
`/scratch/boshra95/psg/{cohort}/derived/annotations/{subject}_stages.npy`,
e.g. `APL0001_stages.npy`, shape `(1142,)`, `int8`, classes
`{0:W, 1:N1, 2:N2, 3:N3, 5:REM}` (5→4 via the existing `_remap_stages`).

Protocol:
- **~100 subjects** (50 APPLES + 50 SHHS — SHHS also exercises the `EEG`
  name fallback and the `Thor` RESP fallback, §2.2) → roughly **100,000
  labelled epochs**.
- For each variant, fit a plain multinomial logistic regression on the
  `[6 × D]` per-epoch embeddings of the train subjects; score **weighted F1**
  and **Cohen's κ** on held-out *subjects* (never held-out epochs from a
  training subject).
- No sequence head, no context sweep, no training loop. Seconds of CPU per
  variant after extraction.

Why this is the right instrument:

| | val AUROC, `sex_binary` 30s | single-epoch staging probe |
|---|---|---|
| Effective n | ~165 subjects | ~20,000 held-out epochs |
| Detectable difference | ~0.04 (useless) | ~0.005 |
| Extraction cost | 3 × 1,600 subjects, twice | **3 × 100 subjects, once** |
| Wall time | many hours | one ~30 min GPU job |
| External reference point | none | **Gnassounou et al. report Mantis at 75–89 weighted F1 on exactly this task, on NSRR-family cohorts** |

That last row is the real value: it turns "which variant is bigger" into "does
our variant land where the published Mantis-on-sleep numbers land, or far
below" — which is precisely the question Option D's 7.3× extrapolation raises.

**Honest caveat**: staging is a *proxy*. It measures whether the encoder
resolves the physiology, which is what is at stake here, but the paper's real
tasks are subject-level (sex, BMI, age, apnea, sleep efficiency). So the winner
is confirmed with **one** real 30 s `sex_binary` training run before the full
sweep — one job, not six.

**This probe also subsumes the standalone embedding-sanity check** that used
to be §13.5 item 8: a degenerate or collapsed embedding cannot reach a
plausible staging F1, so a sane probe result is stronger evidence than a
NaN-and-variance check alone. Keep the NaN/variance assertions inside the
probe script; drop the separate step.

**Both pilots run from one job.** Three extraction passes (D, D-interp, B),
each capturing layer-2 *and* layer-6 output, each in both `cls` and `combined`
form ⇒ **3 × 4 = 12 variants from 3 forward passes**, all scored by the same
probe.

---

### 13.5 The rest of the Step-0 checklist

1. **Checkpoint load** (§3.4) — construct at `num_patches=240`, load
   `Mantis-8M`, assert missing keys are exactly
   `{transf_unit.pos_encoder.pe, tokgen_unit.scalar_encoders.{0,1}.scales}`,
   assert zero unexpected, assert 8,112,384 params. Repeat for `MantisPlus`
   (8,112,402).
2. **LoRA injection count** — `get_peft_model` on the combined module; assert
   **12** LoRA-wrapped Linears (6 blocks × `to_qkv` + `to_out.0`), **221,184**
   LoRA params, and that `sequence_head` is trainable via `modules_to_save`.
3. **Zero `BatchNorm`** — machine-check
   `not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in
   backbone.modules())`, so §4.4's `chunk_batch_size` safety claim is asserted,
   not merely prose.
4. **Absent-slot path** — force a slot missing on a real subject; confirm the
   slice is exactly zero, that no forward ran for it, and that the fill log
   records it.
5. **Cohort filter units** — confirm `min_recording_patches: 480` is applied
   in 30 s-epoch units, not SleepFM's 5 s units.
6. **`mantis_env` sanity** — `import nsrr_tools` resolves to
   `/home/boshra95/NSRR-tools-mantis/src`; `nsrr_tools.datasets` imports
   without `pyedflib`; `pyarrow` writes a parquet.
7. **Split-population audit** — after full extraction, compare the
   "has embedding" population against PhysioOmni's 14,993/14,994. Any
   difference changes `rng.shuffle()`'s entire permutation and therefore the
   splits (§14.4).

## 14. Stage 2 (LoRA) — design

Materially simpler than PhysioOmni's Stage 2 and close to OSF's, for one
reason: **every subject has the same fixed `[6, ...]` tensor shape**, so
there is no ragged per-modality batching, no present-mask grouping, no custom
`collate_fn`.

### 14.1 What we're building (one paragraph, for the paper)

Stage 2 fine-tunes Mantis's single 8.11 M-parameter encoder via LoRA adapters
on `to_qkv` and `to_out.0` in every transformer block, jointly with the
sequence head, warm-started from Stage 1's converged head (LP-FT staging,
Kumar et al. 2022). The encoder, the 6-channel batching, and the sequence head
are wrapped as one `nn.Module` and passed through a **single**
`get_peft_model()` call with `modules_to_save=["sequence_head"]`, so PEFT's
save/load and gradient-freezing treat the head as a first-class trainable
submodule. Every context length other than 30 s warm-starts from that same
(task, head)'s own converged 30 s Stage 2 checkpoint (a branch, not a chain).

### 14.2 `CombinedMantisLoRAModel`

```python
class CombinedMantisLoRAModel(nn.Module):
    def __init__(self, backbone, sequence_head, chunk_batch_size=192,
                 checkpoint_tokgen=False, checkpoint_chunks=False): ...

    def forward(self, x, mask):
        # x: [B, N, 6, 3840] float32 raw signal;  mask: [B, N] bool
        B, N, C, L = x.shape
        flat = x.reshape(B * N * C, 1, L)              # ONE channel axis -> batch (§4.6)
        outs = [self.backbone(flat[i:i + self.chunk_batch_size])
                for i in range(0, flat.shape[0], self.chunk_batch_size)]
        emb = torch.cat(outs, 0).reshape(B, N, C * D)  # [B, N, 6*D]
        emb = emb * self.present.view(1, 1, C, 1).reshape(1, 1, C * D)  # zero absent slots
        return self.sequence_head(emb, mask)
```

Three notes:
- `forward(x, mask) -> logits` matches `run_epoch()`'s `model(x, mask)`
  contract exactly, which is why `run_epoch` is reused unmodified.
- **Absent slots**: unlike PhysioOmni, we keep the uniform `[6, ...]` tensor
  and run all six through the backbone, then **zero the absent slices before
  the head**. Absences are ~1–2 % (§2.1), so the wasted compute is negligible,
  the code stays branch-free and batch-uniform, and — the actual point — the
  contract is **bit-identical to Stage 1's** (§2.2), where those slices are
  also exact zeros. Gradients through zeroed slices are exactly zero, so
  nothing leaks. `present` comes from the cache's `meta.json`, per subject.
- `chunk_batch_size` counts **channel-epochs** (§4.4); set it to a multiple of
  6.

### 14.3 Raw-signal cache — `[T, 6, 3840]` float16, one file per subject

**Not optional. Build it before any Stage 2 training job.** OSF's first real
Stage 2 GPU job stalled 2+ hours with an idle GPU because its dataset loaded
and resampled from raw HDF5 on every `__getitem__`.

For Mantis the motivation is different but just as real: **the fast-tree HDF5
datasets are gzip-compressed with 38,400-sample chunks** (measured, §2.1), so
every window read decompresses at least one 300-second chunk per channel, on
the critical path, in every training job, for every task/head/context. The
cache is uncompressed and pre-sliced.

```
{raw_cache_dir}/{dataset}/{subject_id}.npy     # [T_epochs, 6, 3840] float16
{raw_cache_dir}/{dataset}/{subject_id}.meta.json
      {"t_epochs": int, "slots_found": {...}, "slots_missing": [...],
       "present": [1,1,1,1,1,1], "resp_source": "Airflow"}
```

- **Epoch-major layout is the whole point** (§4.9): a window of N consecutive
  epochs across all 6 channels is one contiguous byte range — 1 open, 1 seek,
  1 read. OSF's channel-major `[12, n_samples]` needs 12 strided reads for the
  same window.
- **Size**: ~48 MB/subject (identical to OSF's measured
  `52,669,568` bytes for `APL0001`, because 12 ch × 64 Hz and 6 ch × 128 Hz
  are the same sample count). **~720 GB for all 14,994 subjects.** Scratch is
  at 8,436 GiB of 19 TiB and 273 K of 1 M files — this fits, but **check
  `diskusage_report` again immediately before building it**, and build
  cohort by cohort so a quota surprise is recoverable.
- **Scope: all four cohorts**, because Mantis includes `apnea_binary`, which
  uses STAGES. (PhysioOmni could skip STAGES; we cannot.)
- **CPU-only, offline, sharded SLURM jobs.** `load_subject_channels` is pure
  `h5py` + `numpy` — no GPU, no resampling, no `scipy`. It should be
  *substantially* faster than PhysioOmni's FFT-resampling precompute; measure
  the first shard before budgeting the rest.
- **Atomic `meta.json`** via temp file + `os.replace`; `cache_exists()`
  parses it (§4.10).

### 14.4 `MantisRawEpochWindowDataset`

Near-verbatim copy of `OSFRawEpochWindowDataset` — the windowing arithmetic
(`_build_seq2label_index`, `SubjectGroupedSampler`, K-sampling by split) is
pure integer math over `T`/`N` and is unrelated to what is stored per epoch.
`__getitem__` returns a plain `([N, 6, 3840] float32, [N] bool mask)`, so
`torch.utils.data.default_collate` works and **no custom `collate_fn` is
needed** (PhysioOmni needed one only because its per-subject structure was
ragged).

**Split-matching discipline — a real, previously-live bug, not a theoretical
concern.** Filter subjects by **"has a Stage 1 Mantis embedding file"** first
(existence check only, contents never read), *then* separately check the raw
cache for actually reading data. `np.random.default_rng(split_seed).shuffle()`
produces a completely different permutation if the filtered population differs
by even one subject; OSF silently scrambled several cohorts' splits this way
before it was caught. Config key: `dataset.stage1_embedding_dir`.

### 14.5 Warm-start / LP-FT staging

- **30s** warm-starts the sequence head from Stage 1's checkpoint
  (`phase0_mantis/{task}_{head}/context_30s/best_model.pt`).
- **Every other context** warm-starts LoRA + head *together* from that same
  (task, head)'s own converged **30s Stage 2** checkpoint — a **branch**, not
  a 30s→10m→…→240m chain. Branching keeps each length independent of sweep
  order and is consistent with the backbone having no per-length state.
- **Readiness gates on `metrics.json`, not `best_model.pt`** — the latter is
  written from epoch 1 and caused a real PhysioOmni bug where a 120m run
  branched off an unconverged 30s checkpoint (§4.10).
- **`modules_to_save` wraps `sequence_head` in a `ModulesToSaveWrapper`** with
  two copies (`.original_module`, frozen; `.modules_to_save["default"]`, the
  trainable one used in forward). A naive `load_state_dict()` fails on the key
  prefix — **load into both copies explicitly**, exactly as
  `train_osf_lora.py` and `train_physioomni_lora.py` already do. The
  Stage-2-30s warm-start needs no special handling
  (`set_peft_model_state_dict` round-trips PEFT's own format).

**State the limitation honestly in the paper**: the LoRA condition's backbone
starting point is shared across context lengths, a genuine departure from the
frozen condition's "N is the only variable" purity, forced by compute.

### 14.6 `configs/phase0_mantis_lora_config.yaml` — deltas from Stage 1

```yaml
data:
  raw_signal_cache_dir: "/scratch/boshra95/psg/unified/mantis_raw_signal_128hz"
dataset:
  stage1_embedding_dir: "/scratch/boshra95/psg/unified/embeddings/mantis_30sec"
lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules: ["to_qkv", "to_out.0"]      # §1.3, checkpoint-verified
  modules_to_save: ["sequence_head"]
training:
  epochs: 25                    # PhysioOmni's revised budget; NOT OSF's 18
  lr: 1.0e-4                    # NOT halved to OSF's 5e-5 — see below
  early_stopping_patience: 5
logging:
  results_dir: "/scratch/boshra95/psg/unified/results/phase0_mantis_lora"
```

On `epochs`/`lr`: OSF cut `40→18` and halved `lr` because *its own* pilot
curve showed overfitting from epoch 9. PhysioOmni's curve showed no such
signature and it kept `lr=1e-4`, cutting only `epochs 40→25` and
`patience 10→5` on wall-clock grounds. **Neither is evidence about Mantis.**
Start at PhysioOmni's values (the more conservative read of the evidence) and
revise from Mantis's own 30s curve — record the curve in `MANTIS_CLAUDE.md`
when it exists, the way both previous plans did.

### 14.7 `experiments/v2_mantis_lora_registry.yaml`

5 tasks × `lstm`/`transformer` = **10 experiments**, `mean_pool` deferred
(§5.6). `gradient_accumulation.context_micro_batch` **nested by head**, seeded
from §4.3's table and **replaced by Pilot 3's real measurements** before the
sweep. `effective_batch: 32` throughout. Separate `logs_dir:
/home/boshra95/NSRR-tools-mantis/logs_mantis_lora` — sharing Stage 1's
corrupted both stages' status files for PhysioOmni (§4.10).

### 14.8 Memory-mitigation ladder — see §4.5

Ordered for Mantis specifically. The Mantis-specific rung is **#3:
gradient-checkpoint `tokgen_unit` only** — ~1.3 % compute for ~39 % of
activation memory, because the `same`-padded conv at 3840-sample resolution is
the single largest activation term while being 1.3 % of the FLOPs (§4.3).
Implement it as `checkpoint_tokgen=True` alongside PhysioOmni's whole-chunk
`checkpoint_chunks=True`; **both opt-in, both default off**, both verified
bit-identical against a non-checkpointed run on the same seed (PhysioOmni
proved this is achievable: max abs diff 0.0).

### 14.9 Full three-way config/argparse audit

Before trusting the Stage 2 config, side-by-side-diff it against Stage 1's
config **and** script for *every* training-loop option. OSF's Stage 2 shipped
three separate times with something Stage 1 had wired correctly and Stage 2
silently didn't (`context_lr_overrides` not applied;
`mixed_precision`/`weighted_sampler`/`persistent_workers` not wired despite
being supported by the imported `run_epoch`).

---

## 15. Implementation checklist — incremental, one user checkpoint per step

**Every step below stops for the user to debug it via `launch.json` before
the next one starts. Do not chain steps.**

### Phase 0 — setup
- [x] **0.1** Build `/home/boshra95/mantis_env` — **done 2026-09-06.** NOT a
      plain `-r osf_env_requirements.txt` install (aborts atomically on any
      unsatisfiable pin — `accelerate==1.2.1`/`scikit-learn==1.7.2` aren't in
      the CC wheelhouse, and installing them from PyPI silently upgraded
      `torch` to 2.6.0). Built in small batches, verified after each one,
      `torch` re-pinned to `2.5.1+computecanada` last. `pyarrow`'s dummy CC
      stub fixed via the same `.pth` trick `physioomni_env` already found.
      `mantis-tsfm` installed with `--no-deps` (its `datasets>=4.0` dep is
      never imported by `mantis.architecture`, which is all we use). Own
      `nsrr_tools_src.pth` → `/home/boshra95/NSRR-tools-mantis/src`,
      confirmed. `nsrr_tools.datasets` imports; `nsrr_tools.core` correctly
      fails on `pyedflib` (re-confirms the channel-loader placement decision,
      §7). Full detail: `MANTIS_CLAUDE.md`.
- [x] **0.2** `snapshot_download` `paris-noah/Mantis-8M` and
      `paris-noah/MantisPlus` to `/home/boshra95/mantis_checkpoints/` —
      **done 2026-09-06.** Byte sizes match the remote header read exactly
      (32,466,928 / 32,467,192).
- [x] **0.3** `scripts/verify_mantis_checkpoint.py` — **done and PASSING on
      both checkpoints, 2026-09-06.** Manual 240-patch load (never
      `.from_pretrained()`), exact allowed-missing-key assertion, LoRA
      injection count = 12 = 221,184 trainable params, `modules_to_save`
      (tested with a stand-in head, real `sequence_head.py` wiring is
      Phase 1) leaves the head trainable while non-LoRA backbone params stay
      frozen, zero `BatchNorm`, 6-channel-batched forward pass is NaN/Inf-free
      and non-degenerate. **A real second checkpoint-loading bug was found
      running this, not anticipated by reading the source** — see §1.0 #6:
      `prj` must be dropped too, not just `pos_encoder.pe`. **Parameter count
      corrected**: 8,037,632 live params (excl. the dead `prj` head), not
      8,112,384/8,112,402 (those are checkpoint-file totals including a
      non-trainable buffer and `prj`) — see §1.1. **User checkpoint.**
- [x] **0.4** Save the Mantis papers to `/home/boshra95/related_work/` —
      **done 2026-09-06**: `Mantis.pdf` (arXiv 2502.15637, 1.01 MB) and
      `Mantis_EEG_Study.pdf` (arXiv 2510.27522, 273 KB), same shared location
      as `OSF.pdf` / `PhysioOmni.pdf`.
- [x] **0.5** Append `🦗 Mantis …` entries to
      `/home/boshra95/.vscode/launch.json` — **done 2026-09-06**: one entry
      so far ("🦗 Mantis Phase0 Step3: Verify Checkpoint"), appended after the
      last PhysioOmni entry, JSON validated (51 total configs, all existing
      entries untouched).

### Phase 1 — Stage 1 (frozen)
- [x] **1.1** `mantis_channel_loader.py` (§7) + `test_mantis_channel_loader.py`
      — **done and PASSING, 2026-09-06.** Contains the full §7 design
      (`load_subject_channels`, `get_epoch_count`, `epochs_to_model_input`
      for both `full_epoch`/`subwindow`, and `load_mantis_backbone`/
      `sinusoidal_pe` from §3.4 — built as one shared module from day one,
      not incrementally). Default 2-subject × 4-cohort smoke test: `[6, n]`
      shape exact, zero NaN, `epochs_to_model_input` correct shapes for both
      windowing modes on real signal. **Both SHHS fallbacks confirmed on
      real data, not just reasoned about**: the very first SHHS subject
      (`200001_v1`) used both the generic `EEG` key AND the `Thor` RESP
      fallback simultaneously. **Went further than the checklist's own
      2-subject scope**: random-sampled STAGES to find and directly test a
      subject with a genuinely absent slot (`STLK00151`, no ECG candidate at
      all — confirmed zero-filled, not skipped) and a subject exercising the
      EMG candidate list's 3rd fallback tier (`MSTR00178`, no
      `CHIN`/`EMG`, resolved via `LLEG`) — the default 2-subject sample
      never hit either case, and the fallback chain beyond the first
      alternate was otherwise untested. **User checkpoint** — test via
      VSCode launch.json "🦗 Mantis Phase1 Step1: Test Channel Loader".
- [x] **1.2** `extract_mantis_embeddings.py` + `configs/phase0_mantis_config.yaml`
      (§9) — **done 2026-09-06.** Wires `load_mantis_backbone`/
      `epochs_to_model_input` into a real extraction loop; both
      `windowing`/`pe_mode` config keys wired through (§3.1's two-key
      split); TF32 on (§4.2); channels batched into one forward (§4.6).
      **The absent-slot contract is Stage-1-specific and different from
      Stage 2's** (§2.2 vs §14.2, stated explicitly in the script's
      docstring): only *present* slots are batched into the backbone call
      per chunk (`chunk_batch_size // n_present`), absent ones are never
      forwarded, not run-then-zeroed. Correctness verified two ways before
      any real run: a synthetic ordering test (distinct per-slot/per-epoch
      constants through `epochs_to_model_input`, both windowing modes) and
      an end-to-end run through `extract_subject_embeddings`'s own
      present-slot-selection/scatter logic using a fake backbone, with a
      genuinely partial `present_idxs` — confirmed absent slots land as
      exact zero and present ones land in the correct position, independent
      of the real (slow) backbone. **User checkpoint.**
- [x] **1.3** CPU smoke test — **done 2026-09-06, real numbers, not a dry
      run.** Ran APPLES (`APL0001`), STAGES (`STLK00151` — the same
      genuinely-ECG-absent subject 1.1 found), and SHHS (`200001_v1` — the
      same Thor-fallback subject 1.1 found), one at a time. **A real
      resource-contention incident happened first**: three subjects were
      launched concurrently and none finished after ~40 minutes; `uptime`
      showed the shared login node at load average 13–19 across 87 other
      users, and each process was independently spawning ~34 threads — not
      a code bug, but a real violation of "don't run anything long on the
      login node." Killed all three, re-ran one at a time with
      `OMP_NUM_THREADS=8`/`MKL_NUM_THREADS=8`, ~2.6–3.2 min/subject.
      Results: `APL0001` → `(1143, 6, 512)`, zero missing slots, zero
      NaN/Inf, per-slot std 1.68–2.23 (float32 — computing `.std()` directly
      on the saved float16 array overflows its accumulator and prints `inf`,
      a display artifact of the verification script, not a data problem).
      `STLK00151` → `(1148, 6, 512)`, `slots_missing: ['ECG']`, that slot
      **exactly** 0.0 (mean and std both 0.0), every other slot
      non-degenerate. `200001_v1` → `(1084, 6, 512)`, `resp_source: Thor`,
      `fallback_used: {"EEG": "EEG", "EMG": "EMG", "RESP": "Thor"}`, zero
      NaN/Inf. All three fill logs and shapes match step 1.1's loader-only
      predictions for these exact subjects. **User checkpoint.**
- [ ] **1.4** `jobs/extract_mantis_embeddings_{gpu,cpu}.sh`; run **Pilot 3**
      (§13.3) — throughput, achieved TFLOP/s vs peak, `chunk_batch_size` A/B.
      **Stop here if under ~5 % of peak.** **User checkpoint.**
- [ ] **1.5** `scripts/probe_mantis_staging.py` (§13.4) — the single-epoch
      sleep-staging probe: ~100 subjects (50 APPLES + 50 SHHS), multinomial
      logistic regression on `[6 × D]` epoch embeddings, subject-wise held-out
      split, weighted F1 + Cohen's κ. Includes the NaN / per-dimension-variance
      assertions (this subsumes the old standalone sanity step).
      **User checkpoint.**
- [ ] **1.6** Run **Pilots 1 and 2 in one job** (§13.1/§13.2): three extraction
      passes (D, D-interp, B), each capturing layer-2 *and* layer-6 output in
      both `cls` and `combined` form → 12 variants scored by one probe.
      Confirm Option D and `combined @ last`, or trigger an escape hatch.
      Record the full 12-row table in `MANTIS_CLAUDE.md` either way — the
      not-taken rows are the paper's supplementary number (§13.2).
      **User checkpoint — this confirms the embedding format for both stages.**
- [ ] **1.7** `mantis_context_window_dataset.py` (§8) +
      `test_mantis_context_window_dataset.py` — on ≥10 subjects/cohort (the
      3-subject population PhysioOmni used was too small to exercise the
      padding branch or K-sampling). **User checkpoint.**
- [ ] **1.8** `train_mantis_context_sweep.py` (§10) + job script; CPU smoke
      test on the pilot subset, run to `Status: SUCCESS`. **User checkpoint.**
- [ ] **1.9** `infer_mantis_subject_windows.py` + job script; CPU smoke test
      against 1.8's checkpoint; verify the parquet schema and zero NaN.
      **User checkpoint.**
- [ ] **1.10** `v2_mantis_registry.yaml` + `gen_commands_mantis.py` (§11);
      verify `list`/`train`/`infer`/`status` against the real checkpoint from
      1.8. **User checkpoint.**
- [ ] **1.11** Full extraction, all 4 cohorts, sharded GPU jobs. Then the
      §13.4-12 split-population audit and the §5.5 `resp_source` tally from
      the fill logs.
- [ ] **1.12** Full Stage 1 sweep (90 runs) → inference → analysis.
- [ ] **1.13** MantisPlus ablation: `phase0_mantis_plus_config.yaml` +
      `v2_mantis_plus_registry.yaml`, extraction, 60 runs (§5.1).
- [ ] **1.14** `docs/MANTIS_EXPERIMENTS_GUIDE.md`, written incrementally as
      each step lands — real commands, real measured numbers, real paths.

### Phase 2 — Stage 2 (LoRA)
- [ ] **2.1** Extend `mantis_channel_loader.py` with the cache functions
      (§7, §14.3); round-trip test. **User checkpoint.**
- [ ] **2.2** `precompute_mantis_raw_signal_cache.py` + CPU job. Check
      `diskusage_report` first. Build cohort by cohort; time the first shard
      before budgeting the rest. **User checkpoint.**
- [ ] **2.3** `mantis_raw_epoch_dataset.py` (§14.4) + smoke test, including
      the Stage-1-embedding-existence split-match assertion.
      **User checkpoint.**
- [ ] **2.4** `train_mantis_lora.py` (§14.2/14.5) — synthetic
      forward+backward against the real checkpoint, asserting LoRA **and**
      `sequence_head` gradients are finite and non-zero, and that a zeroed
      slot produces exactly-zero gradient. **User checkpoint.**
- [ ] **2.5** `infer_mantis_lora_subject_windows.py`, `v2_mantis_lora_registry.yaml`,
      `gen_commands_mantis_lora.py`, job scripts. **User checkpoint.**
- [ ] **2.6** **Real GPU pilot at EVERY context length** (§13.3 item 3) —
      one training step each, record `max_memory_allocated()`, rewrite
      `context_micro_batch` from measurements. Then a real multi-epoch 30s
      pilot for the wall-clock and lr/epochs budget (§14.6).
      **User checkpoint.**
- [ ] **2.7** Full three-way config/argparse audit (§14.9).
- [ ] **2.8** Full Stage 2 sweep (60 runs), applying §4.5's ladder as needed.

### Phase 3 — results
- [ ] **3.1** Stage 1 + Stage 2 vs `phase0_v3`, both checkpoints. Same
      "no silently-incomplete cells" discipline as OSF and PhysioOmni.
- [ ] **3.2** Four-way comparison writeup (SleepFM / OSF / PhysioOmni /
      Mantis), including the pretraining-domain gradient table from
      `docs/TSFM_THIRD_MODEL_DECISION.md` §6.

---

## 16. Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Checkpoint | **`Mantis-8M`** primary, **`MantisPlus`** as the ablation | Byte-identical architecture (verified from both safetensors headers) → real-vs-synthetic pretraining is a controlled one-variable ablation costing one config line (§5.1) |
| `MantisV2` | Documented, not run | Different architecture (RoPE/RMSNorm/SwiGLU), different LoRA names (`wQKV`/`wO`), synthetic-only, no sleep evidence — a third model family, not a swap |
| Loading | **Manual safetensors load, never `from_pretrained`** | `from_pretrained` rebuilds from the repo `config.json` and then hard-raises on the `pos_encoder.pe` shape mismatch — verified empirically on torch 2.5.1 (§1.0 #2, §3.4) |
| Windowing | **Option D** (`seq_len=3840, num_patches=240`); D-interp measured alongside | D matches on *interface* (one forward → one embedding, 241 tokens vs OSF's 90 / PhysioOmni's 30–150 — unremarkable for the comparison set); B matches on *distribution* but adds an aggregation stage **no other model has**, in the exact place the paper makes claims. Option A (→512) destroys spindles/beta/EMG. The 7.3× extrapolation is the one real risk → §13.1's escape hatch |
| Output token / layer | **`combined @ last`** (`return_transf_layer: -1`, `input_dim: 3072`) | Takes the **robust** half of the authors' recipe (`combined` beats `cls` at 6/6 layers in their own notebook) and rejects the **fragile** half (their demo picks layer 1/3, not the README's 2). Last layer matches SleepFM/OSF/PhysioOmni, keeps LoRA depth at 6/6 like OSF's 12/12, and keeps the MantisPlus ablation one-variable (§3.3) |
| Source tree | **Fast-channel `/scratch/boshra95/psg`** | Measured: every needed channel is present in all 4 cohorts. No reprocessing, no resampling |
| Comparison baseline | **`phase0_v3`** (paper-primary) | Follows from the fast tree; matches PhysioOmni |
| Channel map | **6 fixed slots**, per-slot candidate lists, absent → exact-zero slice | Measured cohort-dependent names; a 7th (second-EEG) slot would be permanently absent for 56 % of subjects, and duplicating SHHS's single EEG would fabricate an r=1.0 channel (§2.2) |
| Normalization | **None — feed the stored z-scored values as-is** | The conv path is scale-invariant (`ts_scaler`); the only scale-sensitive path is the per-patch scalar encoder whose grid is centred on O(1); Mantis's own corpus is z-normalized. Restoring µV would *introduce* a 5-order-of-magnitude cross-cohort inconsistency (§3.2) |
| Apnea | **In scope** — 5 Tier-1 tasks | Mantis is modality-agnostic and the RESP slot exists. Makes Mantis the only general baseline comparable to OSF on apnea. Report the Airflow/Thor heterogeneity, and that it is shared with SleepFM (§5.5) |
| Heads | Stage 1 all 3; Stage 2 `lstm`+`transformer` | Matches OSF's and PhysioOmni's own scoping |
| Adaptation | **LoRA**, `["to_qkv","to_out.0"]` — not full FT | Method parity across the four backbones; full FT documented as a supplementary 30s ablation only (§5.7) |
| Channel batching | Reshape `(B,C,L)→(B·C,1,L)`, one forward | `transform()` loops channels with a `DataLoader` each — 6 launches instead of 1 (§4.6) |
| Stage 2 cache | **`[T,6,3840]` fp16 epoch-major, one `.npy`/subject** | Makes an N-epoch window one contiguous read (§4.9). The HDF5s are gzip-chunked at 300 s, so reading them live is worse than it looks (§14.3) |
| Absent slots in Stage 2 | Uniform tensor, zero the *output* slice | ~1–2 % of subjects; keeps batches uniform and the contract bit-identical to Stage 1 (§14.2) |
| Whole H100 vs MIG | **Whole card, justified on MEMORY not throughput** | OSF's throughput investigation argues against it *for OSF at 30s*; but 240m needs ~480 epoch-units × ~240 MB, which a 19.6 GB slice cannot hold at any batch size. Not a speed choice (§4.5) |
| Pilot instrument | **Single-epoch sleep-staging probe**, not val AUROC | ~20,000 held-out labelled epochs vs ~165 subjects → detects 0.005 instead of 0.04; 16× cheaper; and comparable to published Mantis-on-NSRR numbers (§13.4) |
| Memory ladder | whole H100 → micro_batch → **tokgen-only checkpointing** → full checkpointing → cap and report | The `same`-padded 3840-sample conv is 39 % of activation memory for 1.3 % of FLOPs — a Mantis-specific rung neither previous model had (§4.3/§4.5) |
| Env | Dedicated `mantis_env` | `osf_env`/`physioomni_env` each pin `nsrr_tools_src.pth` to another worktree |

---

## 17. Known open questions

- **Windowing (§13.1)** — **decided (Option D)** on cross-model fairness
  grounds, with a named escape hatch: staging-probe weighted F1 below ~0.60
  where Option B reaches ~0.75+ would mean the 241-token extrapolation is
  genuinely broken, and we switch to B and say so in Methods. D vs D-interp is
  purely empirical and fairness-neutral.
- **Output token / layer (§3.3, §13.2)** — **decided (`combined @ last`)** on
  cross-model fairness plus the authors' own non-reproducing layer table. Not
  an open question; the pilot produces the supplementary number showing what
  layer-2 would have given.
- **Achieved GPU utilization (§4.7)** — §4.8's per-context wall-clock budgets
  are FLOP arithmetic, not measurements. The Mantis-vs-PhysioOmni "faster
  despite more FLOPs" hypothesis is an *estimate from tensor shapes*, and it
  is the dominant uncertainty in the whole plan. §4.4 adds a second, sharper
  prediction to test: because all 6 channels batch into one call, Mantis has
  **6× more items per forward call than OSF at the same `micro_batch`**, so it
  should escape the overhead-bound regime at a much shorter context than OSF
  did. Both are predictions; Pilot 3 measures them, at 40m not 30s.
- **Whether 240m LoRA is reachable at all.** §4.3's estimate puts it over the
  ceiling at the `micro_batch=1` floor. Expect to need tokgen checkpointing;
  be prepared to cap the LoRA condition and report the ceiling explicitly
  rather than silently omitting the point.
- **Second EEG derivation for the three cohorts that have it** — deferred by
  the §2.2 decision, documented, revisit only if EEG-dependent tasks look
  degraded. (SHHS's discarded `EEG(sec)` channel could also be recovered by a
  lightweight additive patch job, per PhysioOmni §4.5 — same deferral.)
- **`chunk_batch_size` sensitivity** — OSF measured 3.28×, PhysioOmni measured
  nothing. Ours is unknown until Pilot 3.
- **Raw cache disk** — ~720 GB estimated from OSF's real per-subject sizes.
  Re-check quota immediately before building (§14.3).
- **Expected weak frozen result.** The Mantis-on-EEG study found freezing
  "leads to a huge decrease in performance" on EEG. If Stage 1 is poor and
  Stage 2 rescues it, **that is a finding** — general-purpose pretraining
  transfers to sleep PSG only with adaptation — and it must be reported as
  such, never left looking like a completed unremarkable table cell.

---

## 18. Reading order

1. `MANTIS_CLAUDE.md` — living status, environment, gotchas.
2. **§1.0 of this file** — the five skeleton claims that were wrong.
3. `docs/TSFM_THIRD_MODEL_DECISION.md` — why Mantis, the multi-channel and
   windowing reasoning, and the pre-commitment to reporting a weak frozen
   result honestly.
4. `docs/TSFM_BASELINE_CANDIDATES.md` §6 — the staged frozen/LoRA design.
5. `CLAUDE.md` — repo map, Plan A/B/C framing, honest-comparison rules.
   **Read only; never edit** (§0.2).
6. `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` and
   `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` — PhysioOmni's §15 (Stage 2)
   and checklist 2.6 (real failures) transfer most directly.
7. `docs/EXPERIMENTS_GUIDE.md` — the SleepFM pipeline all baselines mirror.
8. `/home/boshra95/mantis` — the model repo (read-only, never modify).
