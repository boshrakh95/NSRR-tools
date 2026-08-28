# Mantis Implementation Plan

> **Purpose**: authoritative plan for Mantis as TSFM baseline #3 of 3 (OSF #1,
> PhysioOmni #2). Created 2026-08-22 on the `physioomni-implementation`
> branch as the starting skeleton for `mantis-implementation`.
>
> **This is a SKELETON.** Expanding it into the full, cluster-runnable plan —
> at the level of detail `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` and
> `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` reached — is the Mantis
> session's first job, after reading the Mantis repo properly.
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

**Reading the other worktrees is encouraged** — their finished code is the
best structural reference available, and §4's hard-won performance lessons
came from them.

### 0.2 Total file isolation — never edit shared files

Everything Mantis needs must be a **new, Mantis-specific file**. This is what
lets an eventual merge to `main` stay tractable.

**Never edit**: `CLAUDE.md` (status goes in `MANTIS_CLAUDE.md`), the SleepFM
pipeline (`scripts/gen_commands.py`, `train_context_sweep.py`,
`infer_subject_windows.py`, `src/nsrr_tools/datasets/context_window_dataset.py`),
anything `osf_*` / `train_osf*` / `infer_osf*`, anything `physioomni_*`, or
any config/registry/job script belonging to another model.

**Always create parallel `*_mantis_*` files:**

```
src/nsrr_tools/datasets/mantis_channel_loader.py
src/nsrr_tools/datasets/mantis_context_window_dataset.py
src/nsrr_tools/datasets/mantis_raw_epoch_dataset.py     # Stage 2
scripts/extract_mantis_embeddings.py
scripts/train_mantis_context_sweep.py                   # Stage 1
scripts/infer_mantis_subject_windows.py
scripts/train_mantis_lora.py                            # Stage 2
scripts/gen_commands_mantis.py  /  gen_commands_mantis_lora.py
configs/phase0_mantis_config.yaml  /  phase0_mantis_lora_config.yaml
experiments/v2_mantis_registry.yaml  /  v2_mantis_lora_registry.yaml
jobs/*_mantis_*.sh
logs_mantis/ , logs_mantis_lora/
```

Results → `/scratch/boshra95/psg/unified/results/phase0_mantis{,_lora}`.

**The one documented exception**: `src/nsrr_tools/models/sequence_head.py` is
dim-agnostic (`input_dim` is a constructor arg). **Import it, don't edit it.**

---

## 1. Model facts — verified against the real repo/checkpoints 2026-08-22

Repo cloned read-only at **`/home/boshra95/mantis`** (commit `9018b98`).
Install: `pip install mantis-tsfm`. **License: Apache-2.0.**

### Checkpoints (real param counts, read from safetensors headers)

| Checkpoint | Params | Module | Notes |
|---|---|---|---|
| `paris-noah/Mantis-8M` | **8.11 M** | `MantisV1` | the headline model, optimal layer 2 |
| `paris-noah/MantisPlus` | **8.11 M** | `MantisV1` | optimal layer 1 |
| `paris-noah/MantisV2` | **4.19 M** | `MantisV2` | half the params, **same FLOPs** (§4.7) |
| `fegounna/Utica` | — | `MantisV1` | different filename; not verified here |

A **purely synthetic-pretrained variant (CauKer)** also exists and matches
real-data pretraining performance. For our paper this is unusually valuable:
provably zero physiological or NSRR exposure — the cleanest possible
contamination story, against OSF's quantified 87.7 % SHHS overlap. **Run both
real- and synthetic-pretrained checkpoints** as an internal ablation on
whether physiological pretraining data matters at all. No other candidate
offers this.

### Architecture (`src/mantis/architecture/version1.py:218-247`)

```
seq_len=512   hidden_dim=256   num_patches=32   ->  patch_window_size = 16
transf_depth=6   transf_num_heads=8   transf_dim_head=128   transf_mlp_dim=512
                                     -> attention inner_dim = 8*128 = 1024
assert (seq_len % num_patches) == 0
```

- **Tokenizer is a Conv1d** with `kernel_size = patch_window_size + 1` (17
  when even). So `patch_window_size` is baked into the pretrained conv
  weights — see §2.
- **Positional encoding is sinusoidal**, `register_buffer`
  (`transformer_v1_utils/positional_encoding.py`), constructed with
  `max_len=num_patches+1`. Because it *is* in the state dict, changing
  `num_patches` causes a size mismatch on load — it is deterministic, so
  **regenerate it rather than loading it** (§2).
- **`hidden_dim=256` is a multiple of 8** — matters more than it sounds
  (§4.7).

### Channel handling — already channel-independent

`MantisTrainer.transform()` (`src/mantis/trainer/trainer.py:290-320`)
docstring: *"In the multivariate case, each channel is sent independently to
the foundation model."*

```
input   (n_samples, n_channels, seq_len)
output  (n_samples, n_channels * 256)         # default
        (n_samples, n_channels, 256)          # three_dim=True
```

**`three_dim=True` gives `(N, C, 256)` — structurally identical to SleepFM's
`[T, 4, 128]`.** Our `ContextWindowDataset` fork therefore only needs its
three constants changed (`N_MODALITIES→n_channels`, `EMBED_DIM=256`,
`FLAT_DIM=n_channels*256`). This is by far the smallest adapter of the three
baselines.

**Do not average or majority-vote across channels.** Keep per-channel
embeddings and let our existing `LSTMHead`/`TransformerHead` combine them —
exactly what the SleepFM pipeline already does with its 4 modality
embeddings. Averaging discards information the head is designed to use and
breaks the paper's "architecture held constant, only the encoder changes"
claim. Published evidence supports this: the Mantis-on-EEG study found its
channel-independent design *outperformed* CBraMod's multivariate
pretraining in the low-channel-count regime that sleep PSG occupies.

### LoRA targets — same names as OSF, verified

`transformer_v1_utils/transformer.py:38-71`:
```python
self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), ...)
```
Mantis uses the same lucidrains-style ViT blocks OSF does, so
`target_modules=["to_qkv", "to_out.0"]` transfers directly from
`configs/phase0_osf_lora_config.yaml`. **Still live-verify against the real
checkpoint before trusting it** — both OSF and PhysioOmni did this and both
found it worthwhile. Also confirm `modules_to_save=["sequence_head"]`
actually leaves the head trainable.

---

## 2. Our data, and how to feed it

Verified from a real file
(`/scratch/boshra95/psg/shhs/derived/hdf5_signals/203805_v2.h5`):

- **128 Hz**, already resampled and z-scored, `float16`, one 1-D array per channel
- fast-channel `psg/` tree: **6 channels** — `EEG, LOC, ROC, EKG, EMG, Airflow`
- full-channel `psg_full/`: 9 — adds `ABD, HR, SpO2, Thor`
- **No EDF reprocessing needed.** Mantis takes a plain float array.

The mismatch:
```
our 30 s epoch @ 128 Hz = 3840 samples
Mantis pretrained on       512 samples (32 patches x 16)
```

**Option A — interpolate 3840 → 512.** Effective 17 Hz, **Nyquist 8.5 Hz**.
Destroys sleep spindles (11–16 Hz), beta, and most EMG/ECG morphology. This is
the path Mantis's own README recommends generically (`F.interpolate(..., 512)`)
and it is **wrong for sleep PSG**. Do not use it as the primary.

**Option D — feed the full 3840 samples with `seq_len=3840, num_patches=240`.**
***(recommended)*** That keeps `patch_window_size = 3840/240 = 16` and hence
`kernel_size=17`, **identical to pretraining**, so the conv tokenizer weights
stay valid. Full 128 Hz bandwidth preserved; one embedding per 30 s epoch;
the 30 s epoch stays the atomic unit so the context sweep is directly
comparable to OSF and PhysioOmni. Requires regenerating the sinusoidal
positional buffer for 241 positions instead of 33 (deterministic — do not
load it from the checkpoint).

**Caveat to verify in a pilot, not assume**: 240 patches is 7.5× longer than
anything Mantis saw in pretraining. The conv tokenizer and sinusoidal
positions extrapolate cleanly in principle, but the transformer's learned
behaviour at that length is untested. **Compare Option D against Option A on
a small pilot before committing the full sweep** — if Option D turns out to
be badly out of distribution, the fallback is Option A′: split the epoch into
non-overlapping 512-sample windows, embed each at native settings, and pool.

**Comparison baseline**: fast-channel `phase0_v3` (paper-primary), matching
PhysioOmni, since Mantis needs no special channels. Confirm deliberately.

---

## 3. Two stages (same protocol as OSF and PhysioOmni)

**Stage 1 — frozen backbone + our sequence head.** Extract per-epoch
embeddings `[T, C, 256]` offline once, then train `LSTMHead`/`TransformerHead`
over context windows. This *is* the frozen-embedding condition.

**Stage 2 — LoRA + sequence head.** Wrap backbone+head in one module, single
`get_peft_model()` call, `target_modules=["to_qkv","to_out.0"]`,
`modules_to_save=["sequence_head"]`. Warm-start from Stage 1 (LP-FT staging).
Every context other than 30s warm-starts from that same (task, head)'s own
**converged** 30s Stage 2 checkpoint — gate on `metrics.json` existing, **not**
`best_model.pt`, which is written from epoch 1 and caused a real bug in
PhysioOmni (long contexts branching off an unconverged 30s run).

⚠️ **Expect a Stage 2 OOM at long contexts** — see §4.3. It hit OSF and
PhysioOmni independently.

---

## 4. Performance — read this BEFORE writing any training code

Everything here is a **real, measured lesson** from OSF and PhysioOmni. The
single most expensive mistake in this project so far was spending weeks
tuning batch sizes and GPU allocations around a slowness whose actual cause
was never checked. **Do not repeat it.**

### 4.1 Establish achieved FLOP/s before optimizing anything

PhysioOmni's 80m LoRA run was measured at **0.69 TFLOP/s — 0.14 % of an
H100's ~989 TFLOP/s peak.** That number was not computed until weeks of
tuning had already gone by, and it made the diagnosis obvious in one step.

**First real training run: compute achieved FLOP/s and compare to peak.**
`(items × epochs_per_window × FLOP_per_epoch × 3) / seconds`. If it is under
~5 % of peak, stop tuning batch sizes and find out why. This one check would
have saved weeks.

### 4.2 TF32 — enable it in line one of the training script

**PyTorch 2.5 ships with TF32 OFF by default.** Both
`torch.backends.cuda.matmul.allow_tf32` and
`torch.set_float32_matmul_precision` default to the slow path, so every
matmul runs at true FP32: **~67 TFLOP/s instead of ~495 on H100, a ~7×
penalty for nothing.** OSF and PhysioOmni both ran this way for weeks.

```python
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
```

TF32 keeps FP32's exponent range, needs no `GradScaler`, changes no stored
dtype, and leaves checkpoints byte-compatible. **Prefer it to
`torch.autocast`** — autocast on CUDA defaults to float16, which adds cast
overhead and a scaler and changes numerics relative to Stage 1.

**Honest caveat**: OSF measured *zero* speedup from AMP/fp16 on a `1g.10gb`
pilot (61.7 vs 59.9 min/epoch). That is real evidence the gain may
disappoint. AMP ≠ TF32 (no casts, no scaler), and that pilot was on 1/7 of a
card where something else may have dominated — but **measure, don't assume**.

**Do not use `dtype=float32` in the log to check whether TF32 is on.** That
line reports tensor storage dtype, which is unchanged. Use per-epoch time.

### 4.3 Memory scales with `micro_batch × epochs-per-window`, not chunk size

The architectural OOM that hit **both** previous models. Stage 2 runs every
raw epoch through the backbone. Even with the forward pass internally chunked,
**all chunks' activations stay live for the shared `backward()`** — one
autograd graph. So peak memory ∝ `micro_batch × N`, where N = 30 s epochs per
context window. Chunking bounds each *call*, it does **not** reduce peak memory.

Real failures: **OSF OOM'd at 40m and up; PhysioOmni at 10m, 40m and 240m.**

Both fixed it with a per-context gradient-accumulation schedule holding
`effective_batch = 32` constant:

- **OSF** (`experiments/v2_osf_lora_registry.yaml`): ~640-unit
  (`micro_batch × N`) ceiling — `30s:32, 10m:32, 40m:8, 80m:4, 120m:2, 240m:1`.
- **PhysioOmni** (more refined): **head-aware**, separate schedules for `lstm`
  vs `transformer`, because the transformer head's own O(N²) attention adds a
  second cost the LSTM's O(N) doesn't.

**Budget for the same fix from the start.** The numbers will **not** transfer
— they depend on each backbone's memory footprint, and Mantis's is different
again. **Validate with a real GPU pilot at each context length.** PhysioOmni's
first schedule was a one-point extrapolation and needed revision.

### 4.4 Choosing `micro_batch` and `chunk_batch_size`

Two different knobs, routinely confused:

| | `micro_batch` | `chunk_batch_size` |
|---|---|---|
| What it is | windows per optimizer micro-step | raw epochs per backbone forward call |
| Affects peak memory? | **Yes**, ∝ `micro_batch × N` | Barely — only transient attention buffers |
| Affects gradient math? | No, if `micro_batch × accum = 32` | **No** — pure scheduling knob |
| Affects speed? | Some (occupancy) | **Yes** — fewer, bigger calls amortize overhead |

- **`chunk_batch_size` is the throughput knob.** OSF measured a **3.28×
  speedup** raising it 16→64 (61.6 → 18.8 min/epoch at 30s). It is invisible
  to `effective_batch` and to the paper's "context length is the only
  variable" claim. Verified safe because these backbones use only
  per-sample-independent `LayerNorm` and **zero `BatchNorm`** — **re-verify
  this for Mantis before relying on it.**
- **Scale the two together.** At `micro_batch=1, N=160, chunk=64` you get 3
  calls where the last holds only 32 items — ragged and wasteful. Raise
  `micro_batch` to 8 and `n_items` becomes 1280 → 20 calls at chunk 64; set
  chunk ≈ 320 to keep it at 4 big calls. **Aim for few, large, evenly-sized
  calls.**
- **`micro_batch=1` is a last resort**, not a default. It was forced on
  PhysioOmni by a 19.6 GB MIG slice; on a whole 80 GB H100 it is far too
  conservative and wastes occupancy.

### 4.5 Ask for a whole H100, not a MIG slice

`sinfo` shows Fir has whole-card nodes (`gpu:h100:4`) alongside MIG nodes.
`--gpus=h100:1` gives 80 GB and 7/7 of the SMs; `2g.20gb` is 2/7.
**`sbatch --test-only` showed identical queue-start estimates for MIG vs
whole card**, so the extra compute and memory appeared free.

Also: **`--time` dominates queue position far more than GPU type.** Measured
on the same job — `--time=36:00:00` queued until Aug 26; `--time=12:00:00`
started the same day. With per-epoch checkpointing plus auto-resume, a
shorter wall-time request is nearly free. **Don't request 36 h.**

Mitigation ladder, in order: (1) bigger GPU allocation — zero speed cost;
(2) lower `micro_batch` with compensating `accum_steps`; (3) gradient
checkpointing — real compute cost, keep it **opt-in, default off**; (4) cap
the LoRA condition at the longest tractable context and **report the ceiling
explicitly** rather than silently omitting points.

### 4.6 Batch channels — do NOT use Mantis's own `transform()` loop

`MantisTrainer.transform()` calls `self._transform(x[:, [i], :])` **once per
channel in a Python loop** — 6 sequential passes for our 6 channels, each with
its own DataLoader.

**In our extraction code, reshape `(B, C, L) → (B*C, 1, L)` and do ONE
forward**, then reshape back to `(B, C, 256)`. Legitimate because the model is
channel-independent by construction, and it turns 6 small launches into 1
large one. This is exactly the "one encoder, channels batched" advantage that
makes Mantis attractive versus PhysioOmni's four separate tiny encoders.

### 4.7 Why Mantis should use the GPU well

PhysioOmni's real efficiency ceiling is architectural: its EOG/ECG/EMG
encoders have `hidden_dim=100`, **not even a multiple of 8**, so tensor cores
can't be used cleanly at any batch size — four separate small encoders make it
worse. Nothing in the training code can fix that.

**Mantis's `hidden_dim=256` (attention `inner_dim=1024`) is tensor-core
friendly**, and it is one encoder with all channels batchable into a single
call. So although Mantis costs ~19× PhysioOmni's FLOPs per epoch (§4.8), it
should achieve *far* higher utilization — plausibly **faster in wall-clock
despite more FLOPs**. Verify with a pilot; do not take this on faith.

Note `MantisV2` (4.19 M) has **the same FLOPs** as Mantis-8M (8.11 M) — the
difference is in the token generator, not the transformer. Pick on quality,
not speed.

### 4.8 Cost per 30-second epoch, all six channels

| Model | Tokens/epoch | GFLOP/epoch | vs PhysioOmni |
|---|---|---|---|
| PhysioOmni | 30+60+150+150, **4 separate** encoders | 1.62 | 1.0× |
| OSF | 90 (12 leads, one pass) | 15.6 | 9.6× |
| **Mantis (all versions)** | **240 × 6 ch** | **31.1** | **19×** |
| MOMENT-small | 480 × 6 ch | 143 | 88× |
| MOMENT-large | 480 × 6 ch | 1912 | 1177× |

### 4.9 Data loading — latency-bound, not bandwidth-bound

Measured on `/scratch` (Lustre), disjoint cold subject groups:

```
stat only (5 x exists())        11.6 ms/subject
+ open & parse npy header      104.5 ms/subject   (~20 ms per open!)
+ 5 window reads, N=1  (30s)   374.0 ms/subject
+ 5 window reads, N=160 (80m)  399.8 ms/subject
full np.load of all channels   366.9 ms/subject
```

Three lessons:
1. **`mmap_mode="r"` is SLOWER than a full read on Lustre** — 0.68× at 30s,
   0.32× at 80m. Lustre turns page faults into small RPCs with no readahead.
   Use **seek + one contiguous read** instead (see PhysioOmni's
   `_NpySliceReader`).
2. **This is per-operation latency bound** (~20 ms/open, ~12 ms/read-op), not
   bandwidth bound: N=1 and N=160 cost nearly the same despite ~170× the
   bytes. Cutting bytes buys only ~1.1–1.3×. **Fewer FILES per subject is the
   real lever** — consolidating per-channel `.npy` into one file per subject
   would save ~80 ms/subject of open latency. **Design the cache that way from
   the start** rather than retrofitting.
3. **Keep it in proportion**: I/O is ~50 % of a 30s epoch but only ~3.5 % of
   an 80m epoch. Long contexts are ~96 % compute — no DataLoader tuning helps
   them.

Latency-bound I/O parallelizes near-linearly: derive `num_workers` from
`SLURM_CPUS_PER_TASK` (don't hardcode 2), and request 8 CPUs.

### 4.10 Other real bugs worth not repeating

- **Non-atomic `meta.json` writes.** OOM-killed workers left 81 zero-byte
  `meta.json` files with intact `.npy` siblings; an existence-only
  `cache_exists()` treated them as done and training died with
  `JSONDecodeError`. **Write cache metadata via temp-file + `os.replace`.**
- **Stale shape caches.** A 16-subject debug-era `shape_cache.json` silently
  capped every later dataset build. If you cache derived shapes, either
  invalidate automatically or document loudly that it's manual.
- **`torch.save()` does not create parent directories.** If a results dir is
  deleted while a job runs, the next checkpoint save raises `FileNotFoundError`.
- **Separate `logs_dir` per stage.** Stage 1 and Stage 2 sharing one log
  directory corrupted each other's `.log` and status `.jsonl` files.
- **Build a dedicated `mantis_env`.** Do not reuse `osf_env`/`physioomni_env`
  — each has an `nsrr_tools_src.pth` pointing at ONE worktree's `src/`, so
  reusing one silently imports another branch's code and repointing it breaks
  that branch.

---

## 5. Open questions for the Mantis session

1. **Which checkpoint** — `Mantis-8M`, `MantisPlus`, or `MantisV2`? Same FLOPs,
   so decide on quality. Plus: run the **synthetic (CauKer)** checkpoint as the
   zero-contamination ablation (§1).
2. **Option D vs Option A** (§2) — pilot both before committing.
3. **Fast-channel (6) or full-channel (9)?** Fast-channel matches PhysioOmni
   and the paper-primary baseline.
4. **Normalization** — our HDF5 is already z-scored, and Mantis's token
   generator does its own per-patch scaling (`ts_scaler`, mean/std
   statistics). Check whether our pre-z-scoring is redundant or harmful.
   PhysioOmni needed de-normalization to raw amplitude; **verify empirically,
   don't assume either way.**
5. **Apnea**: PhysioOmni excluded it (no respiratory pathway). Mantis is
   modality-agnostic and we have `Airflow`, so apnea is probably back in
   scope — confirm.
6. **Heads**: OSF and PhysioOmni both scoped Stage 2 to `lstm`/`transformer`,
   deferring `mean_pool`, on compute grounds. Likely right here too.
7. **Full fine-tuning instead of LoRA?** At 8 M params it is genuinely viable
   and Mantis's own tooling supports it — but it breaks method-parity with the
   other two baselines. Think it through; don't do it silently.

---

## 6. Implementation checklist — all open

### Phase 0 — setup
- [ ] 0.1 Build `mantis_env`; `pip install mantis-tsfm`; pin `peft`; verify
      `import nsrr_tools` resolves to **this** worktree's `src/`.
- [ ] 0.2 Read `/home/boshra95/mantis` in detail; re-verify §1 against real
      code; expand this plan to full detail.
- [ ] 0.3 Download checkpoint(s); record real byte size + param count.
- [ ] 0.4 Live-verify LoRA targets `["to_qkv","to_out.0"]` against the real
      checkpoint, and that `modules_to_save` leaves the head trainable.
- [ ] 0.5 Confirm **zero `BatchNorm`** in the backbone (§4.4).
- [ ] 0.6 Resolve §5's open questions; pilot Option D vs Option A.

### Phase 1 — Stage 1 (frozen)
- [ ] 1.1 `mantis_channel_loader.py` — **one cache file per subject** (§4.9),
      atomic metadata writes (§4.10).
- [ ] 1.2 `mantis_context_window_dataset.py` — `[T, C, 256]`.
- [ ] 1.3 `extract_mantis_embeddings.py` — **channels batched** (§4.6), TF32
      on (§4.2).
- [ ] 1.4 `configs/phase0_mantis_config.yaml`.
- [ ] 1.5 `train_mantis_context_sweep.py` + `infer_mantis_subject_windows.py`.
- [ ] 1.6 Registry + `gen_commands_mantis.py`.
- [ ] 1.7 **Measure achieved FLOP/s on the first real run** (§4.1).
- [ ] 1.8 Full Stage 1 sweep.

### Phase 2 — Stage 2 (LoRA)
- [ ] 2.1 Design section — **read §4 first**.
- [ ] 2.2 Raw-signal cache + `mantis_raw_epoch_dataset.py`.
- [ ] 2.3 `train_mantis_lora.py`; readiness gated on `metrics.json` (§3).
- [ ] 2.4 Registry with a context-scaled `context_micro_batch` schedule (§4.3),
      `effective_batch` constant.
- [ ] 2.5 **Real GPU pilot at EACH context length** before the sweep. Do not
      extrapolate from one point.
- [ ] 2.6 Full Stage 2 sweep.

### Phase 3 — results
- [ ] 3.1 Stage 1 + Stage 2 vs `phase0_v3`; both checkpoints (§1).
- [ ] 3.2 Four-way comparison writeup (SleepFM / OSF / PhysioOmni / Mantis).

---

## 7. Reading order

1. `MANTIS_CLAUDE.md` — living status, environment, gotchas.
2. `docs/TSFM_THIRD_MODEL_DECISION.md` — why Mantis, and the multi-channel
   and windowing reasoning in full.
3. `docs/TSFM_BASELINE_CANDIDATES.md` — the survey; §6 on staged frozen/LoRA.
4. `CLAUDE.md` — repo map, Plan A/B/C framing, honest-comparison rules.
   **Read only; never edit** (§0.2).
5. `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` and
   `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` — the two finished plans.
   PhysioOmni's §15 (Stage 2 design) and checklist 2.6 (real failures) are the
   most directly transferable.
6. `docs/EXPERIMENTS_GUIDE.md` — the SleepFM pipeline all baselines mirror.
7. `/home/boshra95/mantis` — the model repo.
