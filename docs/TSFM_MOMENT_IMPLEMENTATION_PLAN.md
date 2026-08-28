# MOMENT Implementation Plan

> **Purpose**: Authoritative record of what will be built for MOMENT as
> TSFM baseline #3 (of 3 — OSF is #1, PhysioOmni is #2). **This is the
> INITIAL SKELETON only.** It was created on the
> `physioomni-implementation` branch (2026-08-22) purely so the
> `moment-implementation` branch has something structured to start from —
> it is deliberately *not* the detailed, cluster-runnable plan that
> `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` and
> `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` grew into.
>
> **Writing that detailed plan is the first job of the MOMENT session**,
> after reading the MOMENT repo in detail. Follow the same
> research-then-plan discipline the other two plans document: verify every
> claim against real code, never assume a number transfers from another
> model, and record what was actually checked rather than what seemed
> likely.
>
> Format mirrors the other two plan docs — read either of them for the
> pattern this one should grow into (a live checklist plus verification
> detail recorded inline as it's established).

---

## 0. Hard constraints — read before touching anything

### 0.1 Worktree / branch isolation

On Compute Canada Fir this repository is checked out as **several `git
worktree`s at once**, one per active TSFM baseline, so each can be worked
on from its own VSCode window / Claude Code session without clobbering the
others:

| Directory | Branch | What it's for |
|---|---|---|
| `/home/boshra95/NSRR-tools` | `osf-implementation` | OSF baseline (#1) |
| `/home/boshra95/NSRR-tools-omni` | `physioomni-implementation` | PhysioOmni baseline (#2) |
| `/home/boshra95/NSRR-tools-moment` | `moment-implementation` | **MOMENT baseline (#3) — this plan** |

**A session working on MOMENT operates *only* inside
`/home/boshra95/NSRR-tools-moment`, on the `moment-implementation` branch.
It never writes anything under `/home/boshra95/NSRR-tools` or
`/home/boshra95/NSRR-tools-omni`, and never intends a change for any branch
other than `moment-implementation`.**

Reading the other worktrees is allowed and encouraged — OSF's and
PhysioOmni's finished implementations are the best available structural
reference, and their real engineering lessons (§4 below especially) are
what stop MOMENT rediscovering the same problems a third time. **Reading
is fine; writing is not.** All worktrees share one underlying `.git`, so a
stray write or a `git checkout` in the wrong directory affects real work
in another live session.

### 0.2 Total file isolation — never edit shared files

**Everything MOMENT needs must be a NEW, MOMENT-specific file.** This is
not a style preference — it is what let OSF coexist with the original
SleepFM pipeline, and what let PhysioOmni coexist with both, while all
three branches kept moving independently. It is also what makes an
eventual merge to `main` tractable instead of a conflict minefield.

**Never edit a file that another branch also edits.** In particular, never
edit:

- `CLAUDE.md` — shared with every branch and edited by all of them. This is
  exactly why `MOMENT_CLAUDE.md` exists as a separate, branch-local status
  file (same reason `PHYSIOOMNI_CLAUDE.md` exists). Put MOMENT status
  there, never in `CLAUDE.md`.
- `scripts/gen_commands.py`, `scripts/train_context_sweep.py`,
  `scripts/infer_subject_windows.py` — the SleepFM pipeline.
- Anything `osf_*` / `train_osf*` / `infer_osf*` — OSF's files.
- Anything `physioomni_*` — PhysioOmni's files.
- `src/nsrr_tools/datasets/context_window_dataset.py`.
- Existing `configs/*.yaml`, `experiments/*.yaml`, `jobs/*.sh` belonging to
  another model.

**Instead, create parallel MOMENT-specific files**, following the naming
pattern the other two established:

```
src/nsrr_tools/datasets/moment_context_window_dataset.py
src/nsrr_tools/datasets/moment_channel_loader.py
scripts/extract_moment_embeddings.py
scripts/train_moment_context_sweep.py
scripts/infer_moment_subject_windows.py
scripts/train_moment_lora.py
scripts/gen_commands_moment.py
scripts/gen_commands_moment_lora.py
configs/phase0_moment_config.yaml
configs/phase0_moment_lora_config.yaml
experiments/v2_moment_registry.yaml
experiments/v2_moment_lora_registry.yaml
jobs/*_moment_*.sh
logs_moment/ , logs_moment_lora/
```

Results go to a **sibling** results directory, never into another model's
tree: `/scratch/boshra95/psg/unified/results/phase0_moment{,_lora}`.

**The one genuine exception**: `src/nsrr_tools/models/sequence_head.py` is
dim-agnostic and reusable *as-is* (`input_dim` is a constructor arg). Reuse
it by **importing** it. Do not edit it.

---

## 1. Overview

MOMENT is the only *general-purpose* time-series foundation model in this
comparison — OSF and PhysioOmni are both physiological/sleep-specific. That
is the point of including it: it tests whether a general TSFM, with no
physiological pretraining at all, can compete on our tasks.

- **Paper**: ICML 2024, *MOMENT: A Family of Open Time-series Foundation Models*.
- **Repo**: `/home/boshra95/moment` (cloned; read-only reference, never modified).
- **Checkpoints**: `AutonLab/MOMENT-1-{small,base,large}` on HuggingFace
  (~40M/125M/385M per the paper — **not stated in the repo's own code**;
  pull each `config.json` to code-verify before quoting a number).
- **License**: MIT, confirmed (`LICENSE`: "Copyright (c) 2024 Auton Lab,
  Carnegie Mellon University"). Cleanest license of the three baselines.

**Comparison baseline**: to be decided by the MOMENT session — depends on
which channels MOMENT is fed. PhysioOmni compares against `phase0_v3`
(fast-channel, paper-primary); OSF compares against `phase0_v3_full`. Decide
this deliberately and record the reasoning, rather than inheriting either.

---

## 2. Status

**Nothing implemented. No environment built. No code written.** This
document is a skeleton created from another branch. Every item in §7's
checklist is open.

---

## 3. Model facts — carried over from `docs/TSFM_BASELINE_CANDIDATES.md` §2.3

These were code-verified when that document was written. **Re-verify
anything load-bearing against the repo directly** before building on it —
the MOMENT session's first task is reading the repo in detail, and this
section is a starting point, not a substitute.

- **Context length: T=512 timesteps, HARD-FIXED.** Confirmed in five
  independent places: `PatchEmbedding.__init__(..., seq_len: int = 512,
  patch_len: int = 8, stride: int = 8, ...)`
  (`momentfm/models/layers/embed.py:181-192`);
  `classification_dataset.py:16` hardcodes `self.seq_len = 512` and
  pads/truncates to it; `tutorials/finetune_demo/classification.py:334`
  argparse default `--seq_len 512`, help text *"currently only support 512
  for MOMENT"*; every tutorial notebook; `tests/test_inference.py`.
  Patch length 8, stride 8 (non-overlapping) → **64 patches per input**.
  - **At 128 Hz this is ≈4 seconds** — shorter than even our own
    5-second SleepFM patches. The backbone is a `T5EncoderModel` with
    relative-position attention, so a different `seq_len` would not
    necessarily raise a shape error, but **zero tutorial/script/test in
    the repo exercises anything but 512** — treat longer as unverified,
    not "should work".
- **This kills Plan A** (native long context, no sequence head). Same
  conclusion as OSF and PhysioOmni, for a different reason.
- **Plan C is also unavailable.** `embed()` (`TASKS.EMBED`,
  `moment.py:229-294`) uses the identical 512-timestep, patch-8 pipeline as
  classification. MOMENT's `Patching`/`TokenEmbedding` expects raw scalar
  amplitude per timestep with its own RevIN normalization, not pre-embedded
  vectors. Feeding SleepFM embeddings in would mean bypassing MOMENT's
  patch/value embedding and reusing only its T5 stack — a nontrivial
  adaptation, not a drop-in.
- **⇒ Plan B only** (short-segment embedder + our own sequence heads),
  exactly like OSF and PhysioOmni. State this plainly in the paper.
- **`n_channels` is NOT checkpoint-locked.** The backbone patch-tokenizes
  each channel independently and reshapes to
  `(batch*n_channels, n_patches, d_model)` before the shared encoder
  (`classify()`, `moment.py:536-546`), so pretrained weights never see a
  channel dimension. `n_channels` only sizes the freshly-initialized
  `ClassificationHead`: `reduction="concat"` (default) →
  `nn.Linear(n_channels * d_model, n_classes)`; `reduction="mean"` →
  `nn.Linear(d_model, n_classes)`. No architectural barrier to our
  multi-channel input.
- **`MOMENTPipeline` requires `.init()` after `from_pretrained()`** to swap
  in the task head (`moment.py:594-624`).
- **`enable_gradient_checkpointing` defaults to `True`** (`moment.py:220`)
  — **must be manually disabled** for frozen-encoder / linear-probe
  (Stage 1) runs, where there is no backward pass through the backbone to
  recompute for and it is pure wasted compute.
- **RevIN normalization is applied automatically per-channel** inside
  `embed`/`classify`/`reconstruction` (`RevIN(num_features=1, ...)`,
  `moment.py:105-107`). Our pipeline's own pre-z-scoring is therefore
  redundant. Decide deliberately what to feed MOMENT and record it —
  PhysioOmni needed raw-amplitude de-normalization for a similar reason and
  that was one of its fiddlier steps.
- **LoRA config from MOMENT's own tutorial**
  (`tutorials/finetune_demo/classification.py:66-74`):
  ```python
  lora_config = LoraConfig(r=64, lora_alpha=32,
                           target_modules=["q", "v"], lora_dropout=0.05)
  self.model = get_peft_model(self.model, lora_config)
  ```
  targeting the T5 encoder's `q`/`v` projections.
  **⚠ REAL GAP — do not copy this recipe as-is.** It passes no
  `modules_to_save=["head"]`. Standard PEFT behavior freezes all non-LoRA
  base-model parameters, so **the classification head is likely frozen
  too**. This reads as a genuine bug/omission in MOMENT's reference
  tutorial, not a validated recipe. Add `modules_to_save` ourselves — the
  same fix OSF and PhysioOmni both needed (both use
  `modules_to_save=["sequence_head"]`).
- **`peft` / `accelerate` are NOT declared dependencies** —
  `requirements.txt`/`pyproject.toml` list only `huggingface-hub`, `numpy`,
  `torch>=2.0.1`, `transformers>=4.54.1`. Both are imported ad hoc in the
  tutorial only. Pin them ourselves, and check the `transformers>=4.54.1`
  requirement against whatever base environment we build from.

---

## 4. ⚠️ EXPECT A STAGE 2 (LoRA) OOM — budget for it from the start

**This is the single most valuable lesson transferred from the other two
baselines. It is a general architectural problem, not an OSF or PhysioOmni
quirk, and MOMENT should expect the same class of failure.** Read this
before designing Stage 2, not after the first OOM.

### The mechanism

Stage 2 LoRA fine-tuning for both OSF and PhysioOmni processes **every raw
signal epoch through the full backbone**. Even when the backbone forward
pass is internally chunked (to limit how many epochs go through one forward
call), **all chunks' resulting activations for a training batch still have
to stay in memory for the backward pass** — they remain part of the same
autograd graph until one shared `backward()`.

**So GPU memory scales with `batch_size × raw-epochs-per-context-window`,
not just the chunk size.** A chunk-size knob (`chunk_batch_size` in both
implementations) bounds each individual forward call but does **not** reduce
peak memory. This is counterintuitive enough that both previous
implementations hit it independently.

### What actually happened

Real OOM failures for **both** models at longer context lengths, despite
working fine at short ones:

- **OSF**: failed at 40m and up (30s and 10m trained fine).
- **PhysioOmni**: failed even at 10m, and again at 40m and 240m in some
  configurations.

### The fixes both landed on

- **OSF** — a per-context-length gradient-accumulation schedule
  (`context_micro_batch` in `experiments/v2_osf_lora_registry.yaml`, on the
  `osf-implementation` branch), targeting roughly a **640-unit
  (`batch_size × context_length`) ceiling** while keeping
  `effective_batch=32` via `accum_steps`:
  ```yaml
  context_micro_batch:
    "30s":  32     # 32 x 1   = 32 units
    "10m":  32     # 32 x 20  = 640 units  (known-good ceiling)
    "40m":  8      # 8  x 80  = 640
    "80m":  4      # 4  x 160 = 640
    "120m": 2      # 2  x 240 = 480
    "240m": 1      # 1  x 480 = 480
  ```
- **PhysioOmni** — a **more refined, head-aware** version of the same idea
  (`experiments/v2_physioomni_lora_registry.yaml`, this branch), with
  **separate micro-batch schedules for `lstm` vs `transformer` heads**,
  because the transformer head's own O(N²) self-attention adds a second,
  head-specific memory cost on top of the shared backbone cost that the
  LSTM head's O(N) does not:
  ```yaml
  context_micro_batch:
    lstm:          # targets ~150-250 units
      "30s": 32   "10m": 8   "40m": 2   "80m": 1   "120m": 1   "240m": 1
    transformer:   # ~half that — more conservative for the O(N^2) head
      "30s": 32   "10m": 4   "40m": 1   "80m": 1   "120m": 1   "240m": 1
  ```

### What MOMENT should do

**Budget from the start for the same fix pattern: a context-scaled,
gradient-accumulation-based micro-batch schedule, with `effective_batch`
held constant so training dynamics stay comparable across contexts.**

**The exact numeric values above will NOT transfer.** They depend on each
backbone's own memory footprint — OSF's ViT1D, PhysioOmni's four parallel
encoders, and MOMENT's T5 encoder are all different, and MOMENT's
512-timestep/64-patch tokenization gives it a different per-epoch cost than
either. **Validate with a real GPU pilot at each context length before
trusting a schedule for a full sweep** — do not extrapolate from one data
point (PhysioOmni's first schedule was a one-point extrapolation and needed
revision).

Two further notes from the other two implementations' hard-won experience:

1. **Prefer a larger GPU allocation before gradient checkpointing.** On
   Fir, MIG partitions scale *compute proportionally to memory*, so a
   bigger slice is strictly more of both — not a tradeoff. Gradient
   checkpointing trades real compute for memory and should be opt-in, last
   resort. (Note MOMENT's own `enable_gradient_checkpointing=True` default,
   §3 — that interacts with this decision, check it deliberately.)
   Whole, non-MIG H100s exist on Fir (`--gpus=h100:1`, `sinfo` shows
   `gpu:h100:4` nodes) and, measured 2026-08-22, cost no extra queue time
   versus a MIG slice.
2. **Ordering matters in the mitigation ladder**: (1) bigger GPU
   allocation, (2) lower micro-batch with compensating `accum_steps`,
   (3) gradient checkpointing, (4) only then cap the LoRA condition at the
   longest tractable context and **report the compute ceiling explicitly**
   in the paper rather than silently omitting points.

---

## 5. Open questions for the MOMENT session to resolve

None of these are decided. Resolve each deliberately, record the reasoning
inline in this doc, and never inherit an answer from OSF/PhysioOmni without
checking it applies.

1. **Which checkpoint** — `small`, `base`, or `large`? Trades compute
   against capability; `large` (~385M) is far bigger than OSF (~86M) or
   PhysioOmni (~15M per encoder). Consider what a fair comparison means.
2. **What is a "segment" for Plan B?** MOMENT's 512 timesteps must map onto
   *something* in our 30-second-epoch protocol. At what sampling rate? One
   512-sample segment per 30s epoch (i.e. ~17 Hz — heavy downsampling), or
   several segments per epoch, or a different epoch definition entirely?
   **This is the central design decision of the whole MOMENT integration**
   and it has no obvious right answer. It also determines the comparison
   baseline (§1).
3. **Which channels / how many `n_channels`?** Fast-channel `psg/` tree
   (like PhysioOmni) or full-channel `psg_full/` (like OSF)?
4. **Normalization** — MOMENT applies RevIN internally. Do we feed our
   z-scored data, or de-normalize to raw amplitude first (as PhysioOmni
   needed)? Verify empirically; do not assume.
5. **Environment** — build a dedicated `moment_env` venv, following the
   precedent that PhysioOmni needed its own separate from `osf_env`
   (a shared venv's `nsrr_tools_src.pth` points at ONE worktree's `src/`,
   so sharing silently imports the wrong branch's code). Pin `peft` and
   `accelerate` explicitly (§3).
6. **Apnea task** — PhysioOmni excluded it (no respiratory pathway). MOMENT
   is modality-agnostic, so apnea is probably back in scope; confirm.
7. **Which tasks/heads for the first pass** — OSF and PhysioOmni both
   scoped Stage 2 to `lstm`/`transformer` only, deferring `mean_pool`, on
   real compute-cost grounds. Likely correct here too.

---

## 6. Honest comparison framing (for the paper)

Carried from `CLAUDE.md`'s existing framing rules — apply the same
discipline:

- MOMENT is Plan-B-only. The comparison is "SleepFM vs. MOMENT embeddings +
  our sequence head", **not** "SleepFM vs. MOMENT's own native context
  handling". Say so plainly.
- Report which context lengths were actually reachable and via which plan.
  If a context OOMs or times out, **report the ceiling explicitly** — never
  leave a table cell that reads like a completed, unremarkable result.
- MOMENT has **no physiological pretraining at all**. Modest expectations
  are appropriate; this is a genuine general-vs-specialist test, and a loss
  is an informative result, not a failed experiment.
- **No contamination concern** — MOMENT's pretraining corpus is
  general time-series, not NSRR cohorts. Unlike OSF (severe, quantified
  SHHS contamination), all four cohorts should be fair comparison ground.
  Verify rather than assume.

---

## 7. Implementation checklist

Mirrors the phase structure of the other two plans. **All open.**

### Phase 0 — Setup and verification
- [ ] 0.1 Create the `moment-implementation` branch and
      `/home/boshra95/NSRR-tools-moment` worktree.
- [ ] 0.2 Read the MOMENT repo in detail; re-verify every claim in §3
      against real code; expand this doc into a full, cluster-runnable plan.
- [ ] 0.3 Build `moment_env`; pin `peft`/`accelerate`; verify
      `import nsrr_tools` resolves to **this** worktree's `src/`.
- [ ] 0.4 Download/verify the chosen checkpoint; record real byte size and
      parameter count (do not quote the paper's numbers unverified).
- [ ] 0.5 Resolve §5's open questions, especially Q2 (segment definition).
- [ ] 0.6 Live-verify the LoRA target modules against the real checkpoint
      (both OSF and PhysioOmni did this and both found it worthwhile) —
      including that `modules_to_save` actually unfreezes the head.

### Phase 1 — Stage 1 (frozen backbone + our sequence heads)
- [ ] 1.1 `moment_channel_loader.py` / dataset class.
- [ ] 1.2 `extract_moment_embeddings.py` + job script.
- [ ] 1.3 `configs/phase0_moment_config.yaml`.
- [ ] 1.4 `train_moment_context_sweep.py`.
- [ ] 1.5 `infer_moment_subject_windows.py`.
- [ ] 1.6 Registry + `gen_commands_moment.py`.
- [ ] 1.7 Small-scale pilot, then the full sweep.

### Phase 2 — Stage 2 (LoRA)
- [ ] 2.1 Design doc section — **read §4 first**.
- [ ] 2.2 Raw-signal cache / dataset (if the design needs one).
- [ ] 2.3 `train_moment_lora.py` with `modules_to_save` fixed (§3).
- [ ] 2.4 Registry with a **context-scaled `context_micro_batch`
      schedule** (§4), `effective_batch` held constant.
- [ ] 2.5 **Real GPU pilot at EACH context length** to validate the
      schedule before the full sweep (§4). Do not extrapolate.
- [ ] 2.6 Full Stage 2 sweep.

### Phase 3 — Results
- [ ] 3.1 Compile Stage 1 + Stage 2 results against the chosen baseline.
- [ ] 3.2 Three-way (four-way, with SleepFM) comparison writeup.

---

## 8. Reference material — read in this order

1. `MOMENT_CLAUDE.md` (this branch) — living status, environment, gotchas.
2. `docs/TSFM_BASELINE_CANDIDATES.md` §2.3 — the code-verified MOMENT
   findings §3 above summarizes, plus §6 on the staged frozen/LoRA
   procedure and the LP-FT justification.
3. `CLAUDE.md` — repo map, Plan A/B/C framing, honest-comparison rules.
   **Read only. Never edit** (§0.2).
4. `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` and
   `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` — the two finished plans
   this one should grow to match. PhysioOmni's §15 (Stage 2 design) and its
   checklist 2.6 (real-world failures and fixes) are the most directly
   transferable.
5. `docs/EXPERIMENTS_GUIDE.md` — the original SleepFM pipeline every
   baseline mirrors for comparability.
6. `/home/boshra95/moment` — the model repo itself.
