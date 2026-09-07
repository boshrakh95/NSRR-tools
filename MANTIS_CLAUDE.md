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

**Not built yet.** Plan: `/home/boshra95/mantis_env`, `pip install mantis-tsfm`,
plus `peft`.

Do **not** reuse `osf_env` or `physioomni_env` — each has an
`nsrr_tools_src.pth` pointing at ONE worktree's `src/`, so reusing one
silently imports another branch's `nsrr_tools`, and repointing it breaks that
branch's live environment. PhysioOmni hit exactly this.

Record here once built: Python/torch versions, package count, and confirmation
that `import nsrr_tools` resolves to `/home/boshra95/NSRR-tools-mantis/src`.

## Checkpoint

**Not downloaded yet.** License **Apache-2.0** (confirmed from the repo's
`LICENSE`) — the cleanest of the three baselines.

| Checkpoint | Params (verified) | Notes |
|---|---|---|
| `paris-noah/Mantis-8M` | 8.11 M | headline model, optimal layer 2 |
| `paris-noah/MantisPlus` | 8.11 M | optimal layer 1 |
| `paris-noah/MantisV2` | 4.19 M | **same FLOPs** — pick on quality, not speed |

Also run the **synthetic-pretrained (CauKer)** checkpoint: provably zero
physiological or NSRR exposure, which is a uniquely clean contamination story
against OSF's quantified 87.7 % SHHS overlap, and a free ablation on whether
physiological pretraining data matters at all.

## Reference materials

- Model repo: `/home/boshra95/mantis` (commit `9018b98`) — read-only, never modify.
- `docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md` — the plan (skeleton).
- `docs/TSFM_THIRD_MODEL_DECISION.md` — why Mantis; multi-channel + windowing.
- `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`, `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md`.
- `docs/EXPERIMENTS_GUIDE.md` — the SleepFM pipeline all baselines mirror.

---

## Key facts to keep front-of-mind

- **Architecture**: `hidden_dim=256`, `num_patches=32`, `transf_depth=6`,
  `heads=8`, `dim_head=128` (inner 1024), `mlp_dim=512`. Tokenizer is a Conv1d
  with `kernel_size = patch_window_size + 1`.
- **Our data is 128 Hz**, 6 fast channels (`EEG, LOC, ROC, EKG, EMG, Airflow`),
  already z-scored, `float16`. **No EDF reprocessing needed.**
- **30 s epoch = 3840 samples; Mantis pretrained on 512.** Recommended fix
  (plan §2): `seq_len=3840, num_patches=240`, which keeps `patch_window_size=16`
  and the conv kernel identical to pretraining. Regenerate the sinusoidal
  positional buffer (sized `num_patches+1`, so it size-mismatches on load).
  **Do NOT interpolate 3840→512** — that's 17 Hz effective, Nyquist 8.5 Hz,
  and destroys spindles (11–16 Hz), beta, and EMG/ECG morphology.
- **Channel-independent by construction.** `transform(three_dim=True)` returns
  `(N, C, 256)` — structurally identical to SleepFM's `[T, 4, 128]`, so the
  dataset fork is a three-constant change. **Keep per-channel embeddings; do
  not average or vote** — let the sequence head combine them, as the SleepFM
  pipeline already does.
- **LoRA targets `["to_qkv", "to_out.0"]`** — same lucidrains-style ViT blocks
  as OSF, so its config transfers. Live-verify anyway.
- **No contamination concern** (general/synthetic pretraining, not NSRR) —
  verify rather than assume.
- **Expect a weak frozen (Stage 1) result.** The published Mantis-on-EEG study
  found freezing the encoder "leads to a huge decrease in performance" on EEG.
  If Stage 1 is poor and Stage 2 rescues it, that is a *finding* — general
  pretraining transfers to sleep PSG only with adaptation. Report it honestly;
  never leave it looking like a completed unremarkable table cell.

---

## Status log

Append dated entries here as work happens (newest last).

- **2026-08-22** — `MANTIS_CLAUDE.md` and
  `docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md` created as skeletons from the
  `physioomni-implementation` branch, after
  `docs/TSFM_THIRD_MODEL_DECISION.md` recommended Mantis over MOMENT. Repo
  cloned to `/home/boshra95/mantis`. No environment, no checkpoint, no code.
  Everything in the plan's §6 checklist is open; first task is reading the
  Mantis repo in detail and expanding the plan.
