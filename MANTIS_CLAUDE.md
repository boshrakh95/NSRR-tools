# MANTIS_CLAUDE.md

**This file exists because `CLAUDE.md` is not safe to edit for live status.**
`CLAUDE.md` is tracked and shared across `osf-implementation`,
`physioomni-implementation` and `mantis-implementation`, and all of them edit
it — a permanent merge-conflict surface. Same reasoning that produced
`PHYSIOOMNI_CLAUDE.md`. This is Mantis's equivalent, scoped only to the
`mantis-implementation` branch / `NSRR-tools-mantis` worktree.

**⚠️ Unlike `CLAUDE.md`, this file is NOT auto-loaded into context.** Read it
explicitly at the start of any work here.

Technical plan: [`docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md`](docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md).
Why Mantis and not MOMENT: [`docs/TSFM_THIRD_MODEL_DECISION.md`](docs/TSFM_THIRD_MODEL_DECISION.md).
This file is the short, living "current state" companion, not a replacement.

> **Created 2026-08-22** from `physioomni-implementation` as an initial
> skeleton. No environment, no checkpoint, no code yet. Keep it updated the
> way `PHYSIOOMNI_CLAUDE.md` was: record what was actually verified, with
> dates, not what seemed likely.

---

## ⚠️ Hard constraints — not suggestions

### Only ever edit this worktree

| Directory | Branch | Owner |
|---|---|---|
| `/home/boshra95/NSRR-tools` | `osf-implementation` | OSF session |
| `/home/boshra95/NSRR-tools-omni` | `physioomni-implementation` | PhysioOmni session |
| **`/home/boshra95/NSRR-tools-mantis`** | **`mantis-implementation`** | **this session** |

**Write only inside `/home/boshra95/NSRR-tools-mantis`.** Never write to the
other worktrees, never `git checkout` another branch here, never intend a
change for another branch. All worktrees share one `.git` — a stray write or
checkout breaks another live session's work.

**Reading the other worktrees is encouraged.** Their finished code is the best
structural reference available, and their real failures are what the
performance section below exists to stop you repeating.

### Never touch files shared between branches

Everything Mantis needs must be a **new `*_mantis_*` file**. Never edit
`CLAUDE.md` (status goes here), the SleepFM pipeline (`gen_commands.py`,
`train_context_sweep.py`, `infer_subject_windows.py`,
`context_window_dataset.py`), anything `osf_*`, anything `physioomni_*`, or
another model's config/registry/job script.

Results → `/scratch/boshra95/psg/unified/results/phase0_mantis{,_lora}`.
Logs → `logs_mantis/`, `logs_mantis_lora/` (separate per stage — sharing one
corrupted OSF's and PhysioOmni's status files).

**One documented exception**: `src/nsrr_tools/models/sequence_head.py` is
dim-agnostic. **Import it. Do not edit it.**

---

## ⚠️ Performance — the expensive lessons, already paid for

**Full detail: plan doc §4. Read it before writing training code.** The single
most costly mistake on this project was spending weeks tuning batch sizes and
GPU allocations around a slowness whose actual cause was never checked.

**1. Measure achieved FLOP/s on your first real run.** PhysioOmni's 80m LoRA
run turned out to be at **0.69 TFLOP/s — 0.14 % of an H100's ~989 peak**. That
number wasn't computed until weeks of tuning had passed, and it made the
diagnosis obvious in one step. If you're under ~5 % of peak, stop tuning and
find out why.

**2. TF32 is OFF by default in PyTorch 2.5.** Both
`torch.backends.cuda.matmul.allow_tf32` and `set_float32_matmul_precision`
default to the slow path, so every matmul ran at true FP32 — **~67 TFLOP/s
instead of ~495, a ~7× penalty for nothing.** OSF and PhysioOmni both ran that
way for weeks. Put this at the top of the training script:
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```
Prefer it to `torch.autocast` (which defaults to fp16 on CUDA, adds casts and
a GradScaler). *Caveat*: OSF measured **zero** speedup from AMP on a `1g.10gb`
pilot — AMP ≠ TF32, but measure rather than assume. And don't use
`dtype=float32` in the logs to check whether it's on; that reports storage
dtype and won't change.

**3. Stage 2 memory scales with `micro_batch × epochs-per-window`, not chunk
size.** All chunks' activations stay live for the shared `backward()`. This
OOM'd **OSF at 40m+** and **PhysioOmni at 10m/40m/240m**. Both fixed it with a
per-context gradient-accumulation schedule holding `effective_batch=32`
(PhysioOmni's is head-aware — transformer heads need more conservative values
than LSTM). **Budget for the same fix; the numbers will not transfer; validate
with a real pilot at each context length.**

**4. `micro_batch` and `chunk_batch_size` are different knobs.** `micro_batch`
drives memory; `chunk_batch_size` (epochs per backbone forward call) is a pure
throughput knob invisible to the gradient math — OSF measured a **3.28×
speedup** raising it 16→64. Scale them together; aim for few, large,
evenly-sized calls. `micro_batch=1` is a last resort, not a default.

**5. Ask for a whole H100 (`--gpus=h100:1`), not a MIG slice** — `--test-only`
showed identical queue estimates. And **keep `--time` short**: 36 h queued 4
days out, 12 h started same-day, and auto-resume makes short requests nearly
free.

**6. Batch channels yourself.** Mantis's own `transform()` loops channels
sequentially (6 passes for 6 channels). Reshape `(B,C,L) → (B*C,1,L)`, one
forward, reshape back — valid because the model is channel-independent.

**7. Lustre I/O is per-operation latency bound** (~20 ms/open, ~12 ms/read),
not bandwidth bound. **`mmap` is SLOWER than a full read here** (0.32× at
80m). Use seek + one contiguous read. **Fewer files per subject is the real
lever** — design the cache as one file per subject from the start.

---

## Why Mantis should use the GPU well

PhysioOmni's efficiency ceiling is architectural: `hidden_dim=100` on three of
its four encoders — **not even a multiple of 8**, so tensor cores can't be
used cleanly at any batch size, across four separate small encoders.

**Mantis is `hidden_dim=256` (attention `inner_dim=1024`), one encoder, all
channels batchable into a single call.** It costs ~19× PhysioOmni's FLOPs per
30 s epoch but should achieve far higher utilization — plausibly **faster in
wall-clock despite more FLOPs**. Verify with a pilot; don't take it on faith.

---

## Environment

**Built and verified 2026-09-06.** `/home/boshra95/mantis_env`, Python
3.10.13. Real gotchas hit while building it, worth knowing before rebuilding:

- **`pip install -r osf_env_requirements.txt` cannot be reused wholesale.**
  `accelerate==1.2.1` and `scikit-learn==1.7.2` are not in the CC wheelhouse
  (they were installed into `osf_env`/`physioomni_env` from PyPI originally);
  a plain `--no-index -r requirements.txt` aborts the **entire batch** on the
  first unsatisfiable pin, silently dropping everything else in that command
  (this cost two rebuild cycles — `pandas`/`h5py`/`einops` were missing
  because `PyYAML==6.0.3` doesn't exist in the wheelhouse either, only
  `6.0.2`). **Install in small batches and verify each one**, not one big
  `-r` file.
- **Installing `accelerate`/`peft` from PyPI silently upgraded `torch` to
  2.6.0**, overwriting the wheelhouse's pinned 2.5.1. Caught by re-checking
  `torch.__version__` after every batch, not assumed. Fixed with
  `pip install --no-index --no-deps --force-reinstall torch==2.5.1
  torchvision==0.20.1 torchaudio==2.5.1` run *after* peft/accelerate/transformers.
- **`pyarrow` needs the same `.pth` fix `physioomni_env` already found** — CC
  ships a dummy stub wheel; the real compiled package lives under the `arrow`
  environment module. Copied `physioomni_env`'s
  `pyarrow_arrow_module.pth` verbatim (same absolute path, no `module load`
  needed at runtime). `df.to_parquet()` verified working.
- **`mantis-tsfm` was installed with `--no-deps`** — every real dependency
  (`torch`, `einops`, `safetensors`, `huggingface_hub`) was already pinned
  above; installing normally would have let it pull `datasets>=4.0` from
  PyPI, which we don't need (see below).
- `pip` correctly complains that `datasets>=4.0` (a `mantis-tsfm` declared
  dependency) is missing — **left uninstalled, deliberately.** Verified:
  `mantis/__init__.py` only lists submodule names, it doesn't import them;
  `from mantis.architecture import MantisV1` imports cleanly with zero
  `datasets` dependency. `datasets` is only needed by `mantis.trainer`
  (`MantisTrainer.pretrain`/`.fit`), which we never use — we call the
  architecture directly and load weights manually (plan §3.4).

**Versions** (torch/numpy/scipy/pandas/h5py/einops match `osf_env`/
`physioomni_env` exactly, for numerical parity across the three baselines):

```
torch 2.5.1+computecanada    numpy 2.1.1+computecanada    scipy 1.14.1+computecanada
pandas 2.2.3+computecanada   h5py 3.12.0+computecanada    einops 0.8.0+computecanada
safetensors 0.8.0            huggingface_hub 0.36.2       peft 0.14.0+computecanada
transformers 4.47.0+computecanada   accelerate 1.2.1      scikit-learn 1.7.2
mantis-tsfm 1.1.0            pyarrow 18.1.0 (via .pth)
```
(`safetensors`/`huggingface_hub`/`scikit-learn` are newer than
`osf_env`'s — pulled in transitively resolving `peft`/`transformers`/
`mantis-tsfm`; none of these affect training numerics, only checkpoint I/O
and sklearn API, both exercised by real code below.)

51 packages installed.

**Confirmed** (all live-checked, not assumed):
- `import nsrr_tools` resolves to `/home/boshra95/NSRR-tools-mantis/src/nsrr_tools`
  (own `nsrr_tools_src.pth`, not shared with `osf_env`/`physioomni_env`).
- `nsrr_tools.datasets` imports with **zero** errors.
- `nsrr_tools.core` correctly **fails** on `import pyedflib` — same gotcha
  `physioomni_env` found, re-verified here rather than assumed. Confirms the
  channel-loader placement decision (`datasets/`, not `core/`, plan §7).
- `torch.cuda.is_available()` is `False` on the login node, as expected —
  real GPU checks wait for a job.
- `from mantis.architecture import MantisV1, MantisV2` imports cleanly.

## Checkpoint

**Downloaded and strict-load-verified 2026-09-06** — `scripts/verify_mantis_checkpoint.py`,
PASSING on both `Mantis-8M` and `MantisPlus` at
`/home/boshra95/mantis_checkpoints/{Mantis-8M,MantisPlus}/`, byte sizes
matching the remote header read exactly (32,466,928 / 32,467,192). License
**Apache-2.0** (confirmed from the repo's `LICENSE` and all three HF model
cards' `cardData`) — the cleanest of the three baselines.

Checkpoint-file totals read directly from each `model.safetensors` header
over HTTP range requests, 2026-08-27 (not from the papers, not from the
README):

| Checkpoint | Module | File total (incl. buffer + `prj`) | Pretraining data | Optimal frozen layer (not used, see below) |
|---|---|---:|---|---|
| `paris-noah/Mantis-8M` | `MantisV1` | 8,112,384 | real time series | 2 |
| `paris-noah/MantisPlus` | `MantisV1` | 8,112,402 | **CauKer 2M — synthetic only** | 1 |
| `paris-noah/MantisV2` | `MantisV2` | 4,188,690 | CauKer 2M — synthetic only | 2 |

**⚠️ A real second checkpoint-loading bug, found only by running the load
(not by reading the source more carefully) — 2026-09-06.** Dropping
`pos_encoder.pe` is not the only surgery needed. `output_token='combined'`
(our decided setting, §3.3 below) doubles `self.hidden_dim`, which also
resizes `self.prj` — the pretraining-only contrastive projector — to 512-dim,
colliding with the checkpoint's native 256-dim `prj` and raising the same
hard `RuntimeError: size mismatch` even under `strict=False`. `prj` is dead
weight at inference regardless (`MantisV1.forward` only calls it when
`pre_training=True`), so it must be dropped alongside `pos_encoder.pe`. See
plan §1.0 #6 and §3.4.

**Correction to the "8.11M" headline figure**: it is the checkpoint-file
total, including the non-trainable `pos_encoder.pe` buffer (8,448 elements)
and the dead `prj` head (66,304 elements at its native shape). **The number
that matters for model-size claims and the LoRA-adapted fraction is
8,037,632 live parameters, identical for both checkpoints.**

**LoRA — live-verified against both real checkpoints**: `target_modules=
["to_qkv","to_out.0"]` injects **exactly 12** LoRA-wrapped Linears (6 blocks
× 2) and **exactly 221,184** trainable params ≈ **2.75 %** of 8,037,632.
`modules_to_save` correctly leaves a stand-in head trainable while freezing
every non-LoRA backbone param.

**`Mantis-8M` and `MantisPlus` differ by exactly 2 tensors / 18 params** —
`tokgen_unit.scalar_encoders.{0,1}.scales`, a deterministic constant buffer.
Architecturally identical (confirmed again by the missing-key sets above:
Mantis-8M is missing those 2 keys entirely, MantisPlus is not). **So the
synthetic-pretraining ("CauKer") ablation the plan wanted is `MantisPlus`,
and running it is one config line** — a perfectly controlled contrast
against OSF's quantified 87.7 % SHHS overlap.

**Correction to the 2026-08-22 skeleton**: `MantisV2` does **not** have the
same FLOPs as Mantis-8M. Its attention inner dim is 256 (`wQKV [768,256]`),
not V1's 1024, and its MLP is SwiGLU — it is roughly **2× cheaper per token**.
Its conv kernel is 41, not 17, and its LoRA targets are `wQKV`/`wO`, not
`to_qkv`/`to_out.0`. Documented, not run (plan §5.1).

**On the "optimal frozen layer" column above**: decided AGAINST using it
(plan §3.3, checklist decision, not a pilot). The authors' own
`getting_started/intermediate_layers.ipynb` does not reproduce its own
README table (Mantis-8M's best in that notebook is layer 1 or 3, not 2), and
per-checkpoint layer truncation would make Mantis the only truncated encoder
in the paper, halve its LoRA depth vs OSF's/PhysioOmni's 12/12, and break the
Mantis-8M-vs-MantisPlus ablation into two variables. **Decided: `combined @
last` for both checkpoints** — `input_dim=3072`.

## Reference materials

- Model repo: `/home/boshra95/mantis` (commit `9018b98`) — read-only, never modify.
- `docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md` — the plan (skeleton).
- `docs/TSFM_THIRD_MODEL_DECISION.md` — why Mantis; multi-channel + windowing.
- `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`, `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md`.
- `docs/EXPERIMENTS_GUIDE.md` — the SleepFM pipeline all baselines mirror.

---

## Key facts to keep front-of-mind

**All re-verified 2026-08-27 against real code, real checkpoint headers and
real HDF5 files. Five skeleton claims were wrong — see plan §1.0.**

- **Architecture**: `hidden_dim=256`, `num_patches=32`, `transf_depth=6`,
  `heads=8`, `dim_head=128` (inner 1024), `mlp_dim=512`. Tokenizer is a
  Conv1d with `kernel_size = patch_window_size + 1` (17), **`same`-padded at
  full sample resolution**, followed by a plain mean over `patch_window_size`
  positions. `self.seq_len` is never used in `forward()`.
- **30 s epoch = 3840 samples; Mantis pretrained on 512.** Fix:
  `seq_len=3840, num_patches=240`, which keeps `patch_window_size=16` and
  `kernel_size=17` **identical to pretraining**. Regenerate the sinusoidal
  positional buffer (`num_patches+1` = 241 positions). **Do NOT interpolate
  3840→512** — 17 Hz effective, Nyquist 8.5 Hz, destroys spindles (11–16 Hz),
  beta, and EMG/ECG morphology.
- **⚠️ `from_pretrained` CANNOT be used for the 240-patch model.**
  `PyTorchModelHubMixin.from_pretrained` rebuilds the model from the repo's
  `config.json` (`seq_len:512, num_patches:32`), and passing `num_patches=240`
  as a kwarg then hard-raises `RuntimeError: size mismatch for
  …pos_encoder.pe`. **Verified empirically**: `load_state_dict` raises on a
  shape mismatch **even with `strict=False`** (torch 2.5.1). Load manually:
  `hf_hub_download` → `safetensors.load_file` → `sd.pop("vit_unit.pos_encoder.pe")`
  → `net.load_state_dict(sd, strict=False)` (the `vit_unit`→`transf_unit`
  rename pre-hook fires here) → assert missing keys ⊆
  `{pos_encoder.pe, scalar_encoders.{0,1}.scales}`. Plan §3.4.
- **⚠️ The frozen embedding is 512-dim per channel, not 256** — if we follow
  the authors' own documented recipe. README + `intermediate_layers.ipynb`:
  frozen extraction is best with `return_transf_layer=2, output_token='combined'`
  → `cat(cls, mean)` = 512. `FLAT_DIM = 6×512 = 3072`, not 1536. This is a
  live decision (plan §3.3), pilot-confirmed before anything is locked.
- **⚠️ Our fast-channel data is NOT a uniform 6 channels.** Measured across
  all four cohorts (250-subject samples): APPLES/MrOS/STAGES carry **8**
  channels with different names (`C3-M2`, `C4-M1`, `CHIN`, `LLEG`…); SHHS
  carries 6, and **its RESP channel is `Airflow` for ~75 % and `Thor` for
  ~25 %**. STAGES has real gaps (~10 % no `EKG`, ~22 % no chin). The skeleton's
  "6 fast channels (EEG, LOC, ROC, EKG, EMG, Airflow)" was true of one SHHS
  file only. → fixed 6-slot canonical map with per-slot candidate lists, plan
  §2.2.
- **Channel-independent by construction** (`Conv1d(in_channels=1)`).
  `transform(three_dim=True)` returns `(N, C, D)` — structurally identical to
  OSF's `[T, 2, 768]`, so the dataset fork is a three-constant change.
  **Keep per-channel embeddings; do not average or vote.**
- **Normalization: feed our z-scored data as-is.** The conv path is
  scale-invariant (`ts_scaler` z-scores each series); the only scale-sensitive
  path is the per-patch mean/std `MultiScaledScalarEncoder`, whose grid
  (`1e-4…1e4`) is centred on the O(1) values night-level z-scoring produces.
  Restoring µV would *introduce* the volts-vs-µV cross-cohort inconsistency
  PhysioOmni had to fight. Plan §3.2.
- **LoRA targets `["to_qkv", "to_out.0"]`** — confirmed against the real
  checkpoint tensor names (`…layers.{0..5}.0.fn.to_qkv.weight [3072,256]`,
  `…to_out.0.weight [256,1024]`). Expect **12** wrapped Linears (6 blocks × 2).
- **Zero `BatchNorm` in the backbone** — confirmed by grep; the only
  `BatchNorm1d` is in the library's default fine-tuning head, which we don't
  use. So `chunk_batch_size` is mathematically inert.
- **Apnea is IN SCOPE** (unlike PhysioOmni) — Mantis is modality-agnostic and
  the RESP slot exists. 5 Tier-1 tasks, same as OSF.
- **The fast-tree HDF5 datasets are gzip-compressed, chunked at 38,400
  samples (300 s)** — so a Stage 2 raw-signal cache is warranted, and it
  should be `[T, 6, 3840]` fp16 **epoch-major** so an N-epoch window is one
  contiguous read.
- **Expect a weak frozen (Stage 1) result.** The published Mantis-on-EEG study
  found freezing the encoder "leads to a huge decrease in performance" on EEG.
  If Stage 1 is poor and Stage 2 rescues it, that is a *finding* — general
  pretraining transfers to sleep PSG only with adaptation. Report it honestly;
  never leave it looking like a completed unremarkable table cell.

## Status log

Append dated entries here as work happens (newest last).

- **2026-08-22** — `MANTIS_CLAUDE.md` and
  `docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md` created as skeletons from the
  `physioomni-implementation` branch, after
  `docs/TSFM_THIRD_MODEL_DECISION.md` recommended Mantis over MOMENT. Repo
  cloned to `/home/boshra95/mantis`. No environment, no checkpoint, no code.
  Everything in the plan's §6 checklist is open; first task is reading the
  Mantis repo in detail and expanding the plan.

- **2026-08-27** — **Mantis repo, checkpoints and our own data re-verified;
  `docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md` expanded from skeleton (526 lines)
  to a full cluster-runnable plan (~1,760 lines).** Still **no environment, no
  checkpoint downloaded, no code**. What was actually verified, and how:
  - Read `/home/boshra95/mantis` `architecture/version1.py`, `version2.py`,
    `tokgen_utils/*`, `transformer_v1_utils/*`, `transformer_v2_utils/*`,
    `trainer/trainer.py`, README and the `getting_started/` notebooks in full.
  - Read all three checkpoints' `model.safetensors` headers over HTTP range
    requests → real param counts, real tensor names/shapes, and the
    Mantis-8M-vs-MantisPlus 2-tensor diff.
  - Fetched all three `config.json` files → confirmed `seq_len:512,
    num_patches:32` is baked into the repo config.
  - Reproduced the `load_state_dict(..., strict=False)` shape-mismatch raise
    locally on torch 2.5.1, and read `huggingface_hub/hub_mixin.py` to confirm
    `from_pretrained` rebuilds from config.
  - Measured real per-cohort HDF5 channel availability (250-subject random
    samples × 4 cohorts), plus HDF5 chunking/compression and scratch quota.
  - Confirmed Fir has whole-card `gpu:h100:4` nodes.
  **Five skeleton claims were wrong** (embedding dim, `from_pretrained`,
  MantisV2 FLOPs, our channel set, where the CauKer checkpoint lives) — all
  corrected above and tabulated in plan §1.0. All seven of the skeleton's §5
  open questions are resolved in plan §5. Three pilots (windowing, output
  token/layer, throughput+memory) are specified in plan §13 and must run
  before any full sweep. Next action: user reviews the plan, then checklist
  0.1 (build `mantis_env`).

- **2026-09-06** — Plan §3.3/§13 rewritten after user pushback that the
  pilots were unexplained and the fairness comparison with the other three
  backbones was under-argued. Decided (not piloted): **`combined @ last`**
  for output token/layer (rejects the authors' per-checkpoint intermediate
  layer — their own notebook doesn't reproduce it, and following it would
  make Mantis the only truncated encoder and break the MantisPlus ablation
  into two variables) and **Option D** for windowing (matches the interface
  every other backbone uses; Option B's extra pooling stage was judged the
  worse asymmetry). Both pilots reframed as confirmatory, scored by a
  single-epoch sleep-staging probe (existing per-subject annotation `.npy`
  files) instead of 30s `sex_binary` val AUROC — ~20,000 held-out epochs vs
  ~165 subjects, detects 0.005 not 0.04. Folded in OSF's
  `docs/LORA_GPU_THROUGHPUT_INVESTIGATION.md` (from `osf-implementation`):
  corrected the "TF32 both flags default off" claim (cudnn's is already
  True), and replaced "0.14% of peak" as a universal Mantis expectation with
  OSF's real measured overhead-bound-at-30s / compute-bound-at-40m+ pattern
  — Pilot 3 now profiles at 40m, not 30s.

  **Then implementation started, Phase 0.** `mantis_env` built (0.1) — not a
  trivial `pip install -r`, see "Environment" section above for the real
  gotchas (wheelhouse gaps, a silent torch 2.6.0 upgrade, the pyarrow `.pth`
  fix). Both checkpoints downloaded (0.2), byte-exact to the earlier remote
  read. `scripts/verify_mantis_checkpoint.py` written and PASSING on both
  (0.3) — and it caught a **second real checkpoint-loading bug** no amount
  of reading the source had surfaced: `prj` collides in `combined` mode
  exactly the way `pos_encoder.pe` does, for a different reason (`hidden_dim`
  doubling, not sequence length). Also corrected the "8.11M" headline
  parameter count — that number includes a non-trainable buffer and a
  dead-at-inference head; the number that matters is 8,037,632. LoRA
  injection (12 modules, 221,184 params, 2.75%) and `modules_to_save`
  behavior are now real-checkpoint-verified, not reasoned about. **User
  confirmed the Phase 0 verify output matches — proceeded to Phase 1.**

  **Phase 1.1 — `mantis_channel_loader.py` + its test, done and PASSING.**
  Built as one complete shared module from day one per plan §7 (not just
  the channel-reading piece): `load_subject_channels`, `get_epoch_count`,
  `epochs_to_model_input` (both `full_epoch`/`subwindow` windowing modes),
  and `load_mantis_backbone`/`sinusoidal_pe` (the manual-load logic from
  checklist 0.3, factored into the shared module this time). One real fix
  made along the way to the plan's own §13.1 text: it described three
  windowing values behind one config key, but Option D and D-interp produce
  **identical model input** — they differ only in the backbone's positional
  buffer — so `epochs_to_model_input` takes just `windowing:
  full_epoch|subwindow`, and D-vs-D-interp is `load_mantis_backbone`'s
  separate `pe_mode` argument, matching what `configs/phase0_mantis_config.yaml`'s
  template already had. Corrected in the plan doc, not just the code.

  Real-data test (`scripts/test_mantis_channel_loader.py`, launch.json "🦗
  Mantis Phase1 Step1"): default 2-subject × 4-cohort run passed everything
  — `[6,n]` shape exact, zero NaN, both windowing modes correct-shaped on
  real signal — and, by luck, the very first SHHS subject (`200001_v1`)
  exercised BOTH documented SHHS fallbacks at once (generic `EEG` key AND
  `Thor` RESP source). **Went beyond the checklist's own scope**: since the
  default sample never hit an actually-absent slot or a fallback beyond the
  1st alternate, random-sampled STAGES directly to find and test
  `STLK00151` (genuinely no ECG candidate — confirmed exact-zero, not
  skipped) and `MSTR00178` (no `CHIN`/`EMG` — correctly fell through to the
  3rd-tier `LLEG` candidate). Both real, both previously untested paths.
  **User confirmed, proceeded to 1.2/1.3.**

  **Phase 1.2/1.3 — `extract_mantis_embeddings.py` + `configs/phase0_mantis_config.yaml`,
  done and real-data-verified.** Stage 1's absent-slot contract (§2.2:
  skip the forward, write exact zero) is deliberately different from Stage
  2's (§14.2: run uniformly then zero the output, needed only for Stage 2's
  batch-uniformity constraint) — the script only batches *present* slots per
  subject, `chunk_batch_size // n_present`. Verified twice before trusting
  any real run: a synthetic ordering test through `epochs_to_model_input`
  (distinct constants per slot/epoch, both windowing modes) and an
  end-to-end run of the real selection/scatter logic against a fake
  backbone with a genuinely partial present-set — confirms the logic itself
  independent of the real (slow) model.

  **Real incident during the CPU smoke test**: launched three single-subject
  extractions (APPLES/SHHS/STAGES) concurrently and none finished after
  ~40 minutes. Root cause was NOT a code bug — `uptime` showed the shared
  login node at load average 13–19 across 87 other users, plus each of our
  own processes independently spawning ~34 threads. Killed all three,
  re-ran serially with `OMP_NUM_THREADS=8`/`MKL_NUM_THREADS=8`: clean
  ~2.6–3.2 min/subject. **Lesson recorded in plan §4.10 — never run
  concurrent CPU debug subjects on the login node, even 2–3 of them.**

  Real results, all matching step 1.1's loader-only predictions for the
  same exact subjects: `APL0001` → `(1143,6,512)`, zero missing, zero
  NaN/Inf, per-slot std 1.68–2.23; `STLK00151` → `(1148,6,512)`,
  `slots_missing:['ECG']`, that slot's mean AND std both exactly 0.0, every
  other slot non-degenerate; `200001_v1` → `(1084,6,512)`, `resp_source:
  Thor`, correct fallback log, zero NaN/Inf. **User checkpoint — test via
  VSCode launch.json "🦗 Mantis Phase1 Step2: Extract Embeddings" before
  continuing to 1.4 (job scripts + Pilot 3).**
