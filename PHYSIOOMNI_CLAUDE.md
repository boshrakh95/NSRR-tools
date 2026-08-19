# PHYSIOOMNI_CLAUDE.md

**This file exists because `CLAUDE.md` is not safe to keep editing for
live status right now** — it's a tracked file shared with `osf-implementation`'s
history, and both branches keep editing it independently, which would mean
a recurring merge-conflict surface (see
`docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md`'s "Hard constraint:
worktree/directory isolation" section for the full reasoning — this is an
explicitly open, undecided question about the right long-term fix). Until
that's resolved, **this file is where PhysioOmni's own "keep this updated
with progress" status lives instead**, scoped only to this
`physioomni-implementation` branch / `NSRR-tools-omni` worktree.

**⚠️ Important: unlike `CLAUDE.md`, this file is NOT auto-loaded into
context.** A session working on this branch should read it explicitly at
the start of any work here (the same way `CLAUDE.md` would normally be
read automatically) — don't assume its contents are already known.

For the actual technical plan (architecture, channel mapping, file specs,
checklist), read `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` — this file
is the short, living "what's the current state" companion to that plan,
not a replacement for it.

---

## Environment

- **`/home/boshra95/physioomni_env`** — dedicated Python 3.10.13 venv,
  built 2026-08-17. **Deliberately separate from `osf_env`**, not a reuse —
  found during setup that `osf_env`'s `nsrr_tools_src.pth` points at
  `/home/boshra95/NSRR-tools/src` (the OSF worktree); reusing `osf_env`
  as-is would have silently imported OSF's branch's copy of `nsrr_tools`
  instead of this worktree's, and repointing that `.pth` file would have
  broken OSF's own environment (a shared venv, still in active use for
  Stage 2). Built from `/home/boshra95/osf_env_requirements.txt` (OSF's
  already-CC-wheelhouse-proven, relaxed requirements — installed cleanly,
  no version relaxation needed this time). 147 packages.
  - `torch==2.5.1`, `torchvision==0.20.1`, `torchaudio==2.5.1` — exact
    match to PhysioOmni's own README pins.
  - `einops`, `pandas`, `scikit-learn`, `huggingface_hub` all present
    (PhysioOmni's own light pip deps beyond the conda/torch stack).
  - `wandb` **not installed** — same known Compute Canada Go-toolchain
    build issue OSF already hit and worked around; PhysioOmni's own
    `train_finetune.py` has `--wandb_log` default `False` (opt-in), so
    this doesn't block anything. Revisit only if W&B tracking parity
    becomes a real requirement later.
  - `nsrr_tools_src.pth` → `/home/boshra95/NSRR-tools-omni/src` (verified:
    `import nsrr_tools` resolves to this worktree's `__init__.py`, not
    OSF's).
  - `nsrr_tools.core` still fails to import (`pyedflib` missing, same as
    `osf_env`) — confirms `physioomni_channel_loader.py` must live under
    `src/nsrr_tools/datasets/`, not `core/`, exactly as the plan already
    specifies.
  - `PhysioOmni`'s own `model/` code imports cleanly with
    `PYTHONPATH=/home/boshra95/PhysioOmni`.

## Checkpoint

- **`/home/boshra95/PhysioOmni/checkpoints/PhysioOmni.pt`** — downloaded
  2026-08-17, 267,795,410 bytes (matches the HF-API-reported size exactly).
  `VQ.pt` deliberately **not** downloaded (confirmed unneeded, see below).
- **Strict-load-verified** via `scripts/verify_physioomni_checkpoint.py`
  (real script, not a throwaway snippet — VSCode debug config: "🫀
  PhysioOmni Phase0 Step2: Verify Checkpoint"). Result: **all 4 encoders
  load with zero missing keys**, one harmless `unexpected` key each
  (`mask_token` — an MSM-pretraining-only component, absent from `FT.py`'s
  plainer `NeuralTransformer`, correctly ignored by the `strict=False`
  load `FT.py` itself uses).
- **New facts this resolves/adds, not previously confirmed**:
  - The checkpoint's top-level dict has `EEG_encoder_args` /
    `EOG_encoder_args` / `ECG_encoder_args` / `EMG_encoder_args` — the
    exact `NTConfig` kwargs needed per modality are stored *in* the
    checkpoint, so the extraction script/channel loader should read them
    from there rather than hardcoding, mirroring how OSF reads its own
    `metadata` dict. Confirmed identical to what `train_finetune.py`'s
    source already stated (n_layer=12, n_head=10, n_embd=200/100,
    patch_size=200/100, `emb_after_conv_size=104` for EOG/ECG/EMG).
  - `epoch=49`, `iter_num=95050` — matches `train_finetune.py`'s own
    expected filename convention (`ckpt-49.pt`) exactly, strong
    independent confirmation this is genuinely the MSM-stage-49 pretrained
    checkpoint, not some other stage.
  - **Total real encoder parameters: 13,871,304** (~13.9M) — EEG alone is
    7.84M (n_embd=200, roughly 4x the per-block param count of the
    100-dim encoders due to the ~n_embd² scaling in attention/MLP), EOG/
    ECG/EMG are ~2.01M each. Worth remembering when interpreting relative
    results later: PhysioOmni's total encoder capacity (~13.9M) is over 6x
    smaller than OSF's single ViT (85.3M).
  - `ckpt['model']` also contains `{modality}_shared_lm_head` /
    `{modality}_private_lm_head` keys (32 total, 4 per modality) — MSM's
    masked-prediction heads, pretraining-only, correctly never touched by
    `FT.py`'s loading filter.

## Reference materials

- Paper PDF saved: `/home/boshra95/related_work/PhysioOmni.pdf` (arXiv
  2504.19596v3, 15 pages) — same shared, non-git-tracked location OSF's
  own paper PDF lives in.

## SHHS EEG channel — decision (2026-08-17)

**Full investigation and reasoning: `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md`
§4.5 — this is the short version.** SHHS's `psg/` HDF5s carry one generic
`EEG` channel (100% coverage — SHHS isn't missing EEG, just the C3/C4
split the other 3 cohorts have). Followed up on
`docs/OSF_CHANNEL_REPROCESSING_PLAN.md` §4's unverified lead: **100% of
SHHS subjects (8,444/8,444) have both `EEG` and an `EEG(sec)`-family
channel in the raw file**, and a real-EDF correlation check gave **r=0.18**
— confirming these are genuinely distinct electrodes, not a duplicate.

**Decision**: feed SHHS's EEG branch **one real channel, not duplicated
into two** (reversing this plan's earlier draft, which mirrored OSF's own
duplication approach). PhysioOmni's variable-length per-modality token
sequence makes this a legitimate input, unlike OSF's fixed-tensor ViT
where duplication was the right (and still correct, for OSF) choice.
**Zero reprocessing involved either way** — uses the exact `EEG` channel
already in the existing HDF5s.

A lightweight future option (additive patch job to recover the currently-
discarded `EEG(sec)` channel, cheaper than a full SHHS reprocessing,
benefits OSF too) is documented but **not pursued now** — revisit only if
SHHS results look degraded.

## Phase 0 status (per `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` §16)

- [x] 0.1 `physioomni_env` built and verified
- [x] 0.2 Checkpoint downloaded, strict-load-verified
- [x] 0.3 Paper PDF saved locally
- [x] 0.4 SHHS EEG decision — **resolved, see section above**
- [x] 0.5 Normalization approach — **fully resolved 2026-08-18** (3 real
      subjects through the actual frozen encoder, zero NaNs, non-degenerate
      CLS output, see Phase 1 status below). **Real correction found along
      the way**: the original per-channel-type unit table (LOC/ROC=volts,
      rest=µV) was wrong — traced `signal_processor.py` directly, confirmed
      the unit is file/cohort-dependent (APPLES's ECG is µV-scale, SHHS's
      ECG is volts-scale). Fixed with a self-calibrating per-channel check
      instead of a hardcoded table — see plan §5.2.
- [x] 0.6 Sample-rate resampling approach — **fully resolved 2026-08-18**,
      same real-encoder-forward-pass evidence as 0.5
- [x] 0.7 Branch created — **and now also has its own worktree**
      (`/home/boshra95/NSRR-tools-omni`), a step further than the plan's
      original checklist envisioned

## Phase 1 status

- [x] 1.1 `src/nsrr_tools/datasets/physioomni_channel_loader.py` — **done
      2026-08-18**, smoke-tested via `scripts/test_physioomni_channel_loader.py`
      (VSCode debug config "🫀 PhysioOmni Phase1 Step1") against 2 real
      subjects × all 4 cohorts. Zero NaNs, correct resampled lengths,
      SHHS confirmed getting exactly 1 real EEG channel (not 2, not 0) —
      the core §4.5 decision, verified working, not just designed. Caught
      and fixed a real bug in the *test itself* (wrong expected-length
      formula, surfaced only on STAGES) — see the plan doc's checklist 1.1
      entry for the full story.
- [x] 1.2 `scripts/extract_physioomni_embeddings.py` + `configs/phase0_physioomni_config.yaml`
      — **done 2026-08-18**, smoke-tested on real data (APPLES ×2 + SHHS
      ×1, CPU). Runs each of the 4 frozen encoders independently per
      subject (no unified fusion model exists in the checkpoint) and
      concatenates CLS outputs into `[T, 500]`. Real results: APPLES
      shapes `(1143,500)`/`(970,500)` (2 EEG channels), SHHS shape
      `(1084,500)` (**1 EEG channel, confirmed** — §4.5's decision working
      end-to-end through the real encoder, not just the loader). Zero
      NaNs, non-degenerate CLS std (~0.8-1.3) across every modality slice
      in all 3 subjects — resolves 0.5/0.6 above. CPU timing:
      ~584-938s/subject (~10-16 min) — GPU needed for any real-scale run
      (checklist 1.9). VSCode debug configs: "🫀 PhysioOmni Phase1 Step2"
      (APPLES 2-subject and SHHS 1-subject variants).
- [x] 1.3 Smoke test — folded into 1.2's entry above (same real-data run
      covers both)
- [x] 1.4/1.5 `src/nsrr_tools/datasets/physioomni_context_window_dataset.py`
      + smoke test — **done 2026-08-18.** Genuinely simpler fork than
      OSF's, not just renamed: since embeddings are 2D `[T,500]` (no
      sub-token dimension), every 3D pad-block shape and reshape call in
      OSF's version is dropped. Tested against the 3 real subjects
      extracted so far: correct 2/0/1 train/val/test split (val=0 is
      arithmetic at this population size, not a bug), correct
      `(N,500)` shapes at `30s`/`10m`/`full_night`, zero NaN, zero
      unexpected padding. **Known gap, flagged not hidden**: this small a
      population doesn't exercise the padding branch or realistic
      K-sampling — re-test with more extracted subjects before trusting
      at full-sweep scale. VSCode debug config: "🫀 PhysioOmni Phase1
      Step3".
- [x] 1.6 `scripts/train_physioomni_context_sweep.py` + `jobs/
      train_physioomni_context_sweep_gpu.sh` — **done 2026-08-18.** Fork of
      OSF's training script/job with identical function boundaries, only
      the dataset import and `wandb_project` default changed. Needed more
      extracted subjects first (val split empty at 3 subjects) — found the
      minimal sufficient population (8 apples + 8 shhs) by simulating the
      split logic directly rather than guessing, then extracted the rest
      via a new **CPU-only** sbatch job,
      `jobs/extract_physioomni_embeddings_cpu.sh` (mirrors
      `jobs/precompute_osf_raw_signal_cache.sh`'s `def-forouzan`/16-CPU
      pattern — login-node CPU usage is not okay for sustained work, use
      this for any future pilot/debug extraction). CPU smoke test
      (`--context 30s --datasets apples shhs --batch-size 2 --cpu
      --no-wandb`) ran end-to-end to `Status: SUCCESS`, `best_model.pt`
      saved correctly (val AUROC=0.52, no longer NaN), checkpoint resume
      exercised live too. Test-split metrics are degenerate (tiny
      population) but expected — not a bug, revisit at full scale
      (checklist 1.9).
- [x] 1.7 `scripts/infer_physioomni_subject_windows.py` + `jobs/
      infer_physioomni_subject_windows_gpu.sh` — **done 2026-08-18.** Fork
      of OSF's infer script/job, same structure, dataset import + batch-
      size reference kept as OSF's (same 30s-epoch token unit, so no
      re-derivation needed). **Found and fixed a `physioomni_env`
      environment gap along the way**: `pyarrow` was a non-functional CC
      "dummy" stub wheel — fixed by copying `osf_env`'s working
      `pyarrow_arrow_module.pth` (points at the `arrow/18.1.0` module's own
      site-packages) into `physioomni_env`. This fixes `pyarrow` for the
      whole env, not just this script. CPU smoke test against checklist
      1.6's checkpoint ran end-to-end: `Dataset items: 1,796` → parquet
      saved, correct 7-column schema, zero NaNs, `Segment accuracy: 50.84%`.
- [x] 1.8 `experiments/v2_physioomni_registry.yaml` +
      `scripts/gen_commands_physioomni.py` — **done 2026-08-18.** Registry
      mirrors `v2_registry.yaml` (fast-channel/paper-primary) for 4 of the
      5 Tier-1 tasks — sex, sleep efficiency, BMI, age — **apnea
      deliberately excluded** (no respiratory pathway in PhysioOmni).
      Generator is a structural fork of `gen_commands_osf.py`, same
      pipeline logic, pointed at the new registry/job scripts/env.
      Verified live against checklist 1.6's real checkpoint: `list` shows
      `sex_binary_lstm` correctly as `trained (1/6)`, `train`/`infer`
      generate correct sbatch commands.
- [x] 1.9 `jobs/extract_physioomni_embeddings_gpu.sh` — **done 2026-08-18,
      real GPU-verified.** Fork of `jobs/extract_osf_embeddings_gpu.sh`
      (same sharding/auto-resume pattern). **Real measured throughput:
      ~4.1s/subject** on an H100 MIG `1g.10gb` slice — 15-100x faster than
      the CPU path. **Ran a controlled `chunk_batch_size` A/B (16 vs 64) on
      matched shhs batches — found no meaningful difference here**, unlike
      OSF where this knob was the real bottleneck (16→64 gave 3.28x there).
      Kept at 16 (original default). ~14,994 subjects total → ~17h serial
      on one GPU; will shard into parallel jobs for checklist 1.10.
- [x] 1.13 `docs/PHYSIOOMNI_EXPERIMENTS_GUIDE.md` — **initial version done
      2026-08-18**, covering Steps 0-7 with real commands/paths/measured
      numbers (not placeholders). Written now rather than fully
      incrementally since 1.1-1.9 were already all done — same shape as
      `docs/OSF_EXPERIMENTS_GUIDE.md`. Step 8 (LoRA) is a placeholder.
      **Living document — keep updated as 1.10/1.11/1.12 progress.**
- [x] 1.10 Run full embedding extraction, all 4 datasets — **done
      2026-08-19.** Final: apples 1104/1104, shhs 8444/8444, mros
      3933/3933, stages 1512/1513 = 14,993/14,994 (99.99%), zero errors
      elsewhere. The 1 gap (`stages/STLK00096`) has no PhysioOmni-relevant
      channels at all — a known outlier already flagged for OSF too, not
      a bug. Ready for 1.11 (the real Stage 1 sweep).
- [ ] 1.11 Run the Stage 1 sweep — next step

## Native context ceiling / Plan A decision (2026-08-18)

**Full reasoning, exact numbers, and the 3-way SleepFM/OSF/PhysioOmni
comparison table: plan doc §19-§20 — this is paper-facing content, read
it directly rather than a summary here.** Short version: PhysioOmni
*could* architecturally support more than 30s per native call (unlike
SleepFM/OSF, which are hard-fixed at 300s/30s) — up to ~512s for EEG,
~102s for ECG/EMG at its own reference resample rates — but even that
best case falls short of every sweep point except 30s itself (10m alone
is already 600s > EEG's 512s ceiling). **Decision: kept 30-second epochs
(Option 1)** — no pipeline change — which turns out to match not just
SleepFM's/OSF's own epoch unit but PhysioOmni's *own* HMC downstream
fine-tuning convention too (verified: `prepare_HMC_downstream.py` hard-
filters to exactly 30s samples). Nothing in the already-implemented
Phase 1.1/1.2/1.4 code changes because of this.

## Open questions carried over from the plan doc, still open

See `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` §18 for the full list —
not duplicating it here. The `physioomni_env`-vs-`osf_env` question and the
SHHS EEG decision are now resolved (§§ above); everything else in that
section is still open.
