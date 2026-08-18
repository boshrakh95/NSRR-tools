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
- [ ] 0.5 Normalization approach — not yet empirically validated (needs
      real HDF5 data + a forward pass; natural to fold into Phase 1's
      first smoke test rather than a standalone step)
- [ ] 0.6 Sample-rate resampling approach — same as above
- [x] 0.7 Branch created — **and now also has its own worktree**
      (`/home/boshra95/NSRR-tools-omni`), a step further than the plan's
      original checklist envisioned

## Open questions carried over from the plan doc, still open

See `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` §18 for the full list —
not duplicating it here. The `physioomni_env`-vs-`osf_env` question and the
SHHS EEG decision are now resolved (§§ above); everything else in that
section is still open.
