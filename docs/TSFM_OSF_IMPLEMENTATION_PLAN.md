# OSF Implementation Plan — Code-Level, Cluster-Runnable

Detailed step-by-step plan for adding **OSF** (On Pre-training and Scaling of
Sleep Foundation Models, ICML 2026) as the first TSFM baseline compared
against SleepFM. Written for an agent picking this up on the Compute Canada
cluster with `NSRR-tools`, `OSF-Open-Sleep-FM`, and
`npj_digital_medicine_submission` all cloned as sibling directories (see
`CLAUDE.md` → "Cluster Execution Guidance").

**Status (updated 2026-08-10): environment setup and checkpoint download
are done and verified (§4, §5) — this plan was then re-verified
line-by-line against the real OSF and NSRR-tools source (not just prose
review), which surfaced several corrections, most importantly a bug that
would have broken the extraction script immediately (§3.1's
`return_sequence` argument) and a missing config section that would have
crashed the first training run (§3.4's `logging:` block). No pipeline code
(extraction/dataset/training/inference scripts, configs, registries, job
scripts) has been implemented yet — that's still all ahead, per §10.** This
is model #1 of 3; PhysioOmni and MOMENT will get their own plan docs
later, reusing whatever this pass validates.

**Read before this doc:** `CLAUDE.md` (repo map + code reuse assessment),
`docs/TSFM_BASELINE_CANDIDATES.md` §2.1 (OSF's public research findings),
`docs/EXPERIMENTS_GUIDE.md` (the SleepFM pipeline this mirrors).

**Reference materials (added 2026-08-10) — check these whenever unsure
about a model detail, before guessing:**
- **`/home/boshra95/related_work/OSF.pdf`** — the OSF paper itself. The
  code (`osf/backbone/vit1d_cls.py`, `train_config.py`, `demo.ipynb`) is
  the primary source of truth for anything implementation-level (exact
  tensor shapes, argument names, checkpoint format), but the paper is the
  right place to check when a code-level detail doesn't fully explain the
  *reasoning* behind a design choice (e.g. why `lead_wise=1` patchification,
  why 30s epochs, pretraining objective details). A duplicate copy also
  exists at `NSRR-tools/papers/2603.00190v1_OSF.pdf` (same paper, arXiv ID
  in the filename).
- **`NSRR-tools/output/channel_analysis/{apples,shhs,mros,stages}_channels.csv`**
  — per-subject raw EDF channel-label dumps (dataset, subject_id, channel
  list, sampling freq, source EDF path) from an early preprocessing pass
  (2026-02-23). This is the raw-label ground truth behind
  `configs/channel_definitions.yaml`'s alias tables — use it to check what
  a cohort's *original* channel names looked like before our
  standardization, which is more informative than the standardized HDF5
  keys alone when debugging a channel-mapping question. Companion files:
  `all_unique_channels.txt`, `channel_frequency.json`,
  `CHANNEL_EXTRACTION_SUMMARY.md` (human-readable summary). **Use this
  alongside, not instead of, directly sampling the real full-channel HDF5s
  at `/scratch/boshra95/psg_full/{dataset}/derived/hdf5_signals/*.h5`** —
  the CSVs are raw pre-standardization labels from a 5-10-file preview per
  cohort (Feb 2026); §1's channel-completeness table below is a larger
  (50-subject), post-standardization audit against the actual HDF5s
  extraction will read from, and is the more authoritative source for
  "will this channel be there."

---

## Master Implementation Checklist

**This is the authoritative, actively-maintained progress tracker for
this plan — check items off here as work completes, in this repo, on the
`osf-implementation` branch.** The narrative sections below (§1-§12)
contain the supporting research/reasoning each checklist item draws on;
§10's "Suggested execution order" predates this checklist and is now
superseded by it (kept for narrative context only — don't maintain both).

**Workflow**: work through a few unchecked items, commit after each one
(one commit per completed item, referencing this checklist), update this
list (check the box, add a one-line "done — see §X" note), then stop and
hand off for the user to debug/verify using the VSCode configs in §12
before continuing to the next batch. Do not batch many steps together
without a checkpoint in between.

### Phase 0 — Setup
- [x] **0.1** Build `osf_env`, verify `import nsrr_tools` +
      `from osf.backbone.vit1d_cls import ViT, vit_base` — done 2026-08-10, §4.
- [x] **0.2** Download + strict-load-verify the OSF checkpoint — done
      2026-08-10, §5.
- [x] **0.3** Re-verify the whole plan doc against real OSF/NSRR-tools
      source + a real per-cohort channel audit — done 2026-08-10, this
      whole doc.
- [x] **0.4** Locate reference materials (channel CSVs, OSF paper) — done
      2026-08-10, see "Reference materials" above.
- [x] **0.5** Resolve the SHHS channel-completeness decision — **done
      2026-08-10: duplicate SHHS's single generic `EEG` channel into both
      `EEG_C3_A2`/`EEG_C4_A1`, zero-fill `EMG_LLeg`/`EMG_RLeg`/`SN`.
      Explicitly flagged by the user as provisional** — if the
      approximation turns out to hurt SHHS's OSF results too much once
      Stage 1 numbers are in, the fix under consideration is a **targeted
      re-preprocessing pass for SHHS specifically** (revisit
      `signal_processor.py`/`channel_mapper.py` to see whether a
      distinguishable C3/C4 or leg-EMG signal can be recovered from SHHS's
      raw EDFs — the raw `EEG(sec)`/`EMG` labels are genuinely
      undifferentiated per `channel_analysis/shhs_channels.csv`, so this
      would need new preprocessing logic, not just a config change). Not
      committed to doing this yet — a future decision point, not a task on
      this checklist.
- [x] **0.6** Confirm the SHHS/EOG-referencing questions using raw channel
      labels — see §1's updated findings below. EOG referencing: **STAGES's
      dominant raw label (`EOG_LOC-A2`, `EOG_ROC-A1`) confirms OSF's exact
      expected contralateral-mastoid convention**, but
      `channel_definitions.yaml`'s full alias table also folds
      non-contralateral variants (`E1:M1`, `E1-Cz`, `E1:E2`) into the same
      "LOC" bucket for less-common cases across cohorts — so this is
      "likely correct for most subjects," not fully closed; the empirical
      no-NaN/sanity check in §9 item 2 is still the final confirmation
      step, now lower-risk than before.
- [x] **0.7** Create the `osf-implementation` branch in `NSRR-tools` — done
      2026-08-10.

### Phase 1 — Stage 1 frozen-encoder pipeline
- [x] **1.1** Implement `scripts/extract_osf_embeddings.py` (§3.1) — done
      2026-08-10. Also implemented `configs/phase0_osf_config.yaml` (§3.4)
      alongside it since the script needs a real config to run against
      (originally checklist item 1.5 — moved up).
- [x] **1.2** Add its VSCode debug config to `~/.vscode/launch.json` (§12
      item 1, plus an extra SHHS-specific config to exercise the
      EEG-duplication special case) and smoke-test — **done 2026-08-10,
      by Claude, ahead of the user checkpoint**: ran 2 APPLES + 2 SHHS
      subjects for real (CPU, `--limit 2`) before handing off. Both
      cohorts produced correct `[T, 2, 768]` shapes, zero NaNs, non-zero
      variance (mean≈0, std≈0.32 for all 4 subjects — not degenerate),
      and fill-logs matching the §1 audit exactly (APPLES: `EMG_RLeg`
      zero-filled, `EMG_Chin`→generic `EMG`; SHHS: `EMG_LLeg`/`EMG_RLeg`/
      `SN` zero-filled, `EMG_Chin`→generic `EMG`, and — the case that
      mattered most to verify — `EEG_C3_A2`/`EEG_C4_A1` did *not* show up
      as zero-filled/fallback, confirming the SHHS EEG-duplication path
      fired correctly as the designed primary source, not as a fallback).
      This also serves as real evidence for §9 item 2 (EOG referencing) —
      no NaNs/degenerate output across 4 real subjects is a good sign,
      though not a substitute for checking a larger sample.
      **USER CHECKPOINT — please re-verify independently** using the
      `🧬 OSF Step1: Extract Embeddings` configs in `~/.vscode/launch.json`
      (5-subject versions for both `apples` and `shhs`) before continuing
      to item 1.3.
- [ ] **1.3** Implement `src/nsrr_tools/datasets/osf_context_window_dataset.py`
      (`OSFContextWindowDataset`, §3.2).
- [ ] **1.4** Implement `scripts/test_osf_context_window_dataset.py` (forked
      from `test_context_window_dataset.py`, §12 item 2), add its debug
      config, smoke-test — **USER CHECKPOINT**.
- [ ] **1.5** Implement `configs/phase0_osf_config.yaml` (§3.4 — template
      already fully drafted, including the previously-missing `logging:`
      section).
- [ ] **1.6** Implement `scripts/train_osf_context_sweep.py` (§3.3), add its
      debug config, smoke-test a tiny CPU run — **USER CHECKPOINT**.
- [ ] **1.7** Implement `scripts/infer_osf_subject_windows.py` (§3.3), add
      its debug config, smoke-test — **USER CHECKPOINT**.
- [ ] **1.8** Implement `experiments/v2_osf_registry.yaml` +
      `scripts/gen_commands_osf.py` (§3.5 — remember `inference_dir` and
      `python_bin: /home/boshra95/osf_env/bin/python` explicitly).
- [ ] **1.9** Implement `jobs/extract_osf_embeddings_gpu.sh`,
      `jobs/train_osf_context_sweep_gpu.sh`,
      `jobs/infer_osf_subject_windows_gpu.sh` (§3.6).
- [ ] **1.10** Run full embedding extraction for all 4 datasets (GPU job) —
      **USER CHECKPOINT before submitting** (this is a real cluster job,
      confirm readiness first).
- [ ] **1.11** Run the Stage 1 sweep (5 tasks × 3 heads × 6 contexts = 90
      training runs), then inference, then analysis.
- [ ] **1.12** Re-run the §1 channel-completeness audit against the real
      extraction output (all subjects, not the 50-per-cohort preview) and
      update §1's table with final numbers.

### Phase 2 — Stage 2 LoRA fine-tuning
- [ ] **2.1** Implement `scripts/train_osf_lora.py` (§6.1).
- [ ] **2.2** Add its debug config (§12 item 5), run the short wall-time
      pilot (§6.3) — **USER CHECKPOINT**.
- [ ] **2.3** Run the full Stage 2 sweep across context lengths, applying
      the memory-mitigation ladder (§6.2) as needed.

### Phase 3 — Results
- [ ] **3.1** Compile Stage 1 + Stage 2 results against `phase0_v3_full`
      (§0.2), applying the contamination caveats (§8) and the SHHS
      channel-completeness caveat (checklist item 0.5) honestly.
- [ ] **3.2** Report back before starting PhysioOmni or MOMENT — do not
      start the next model's plan unprompted.

---

## 0. Decisions already made (do not re-litigate without asking)

Confirmed with the user on 2026-08-10:

1. **LoRA fine-tuning at long context: attempt full end-to-end first.** If it
   hits GPU memory limits, the fallback ladder (in order) is: (a) enable
   gradient checkpointing, (b) request a larger memory allocation on
   Compute Canada, (c) only if both of those fail, cap the LoRA condition
   at a shorter max context length and report the frozen-only condition at
   the longer ones with the limitation stated explicitly. **Do not skip
   straight to (c)** — try (a) and (b) first.
2. **Comparison baseline is `phase0_v3_full`** (the existing full-channel
   SleepFM run), not `phase0_v3` (paper-primary, fast-channel) — because
   OSF needs channels (snore, full thoracic/abdominal/airflow) that only
   the full-channel config carries. State this explicitly in any results
   writeup: OSF is compared against SleepFM's full-channel numbers, not the
   paper's primary fast-channel headline numbers.
3. **First pass covers Tier 1 tasks only**: `sex_binary`,
   `sleep_efficiency_binary`, `bmi_binary`, `age_class`, `apnea_binary`.
   Sleep staging and Tier 2 tasks (depression, PSQI, etc.) are a later
   pass, after this pipeline is validated end-to-end.
4. **Embedding storage: CLS token + mean-pooled patch tokens**, both
   768-dim, saved per epoch (not the full 90-token patch sequence, and not
   CLS-only). See §3 for the exact shape.

---

## 1. Channel mapping (our full-channel HDF5 → OSF's 12-channel input)

OSF's `ViT` expects exactly 12 channels, in this order (from
`train_config.py` `TRAIN_EDF_COLS_UNI_ENC` in the OSF repo — **code-verified
2026-08-10, exact match confirmed against `train_config.py`, `demo.ipynb`'s
`CHANNEL_NAMES`, and `README.md`'s "Input Format" section, all three
independently agree**):
`ECG, EMG_Chin, EMG_LLeg, EMG_RLeg, ABD, THX, NP, SN, EOG_E1_A2, EOG_E2_A1,
EEG_C3_A2, EEG_C4_A1`. Order matters — it's fed positionally into a
`Conv2d` patch-embedding layer indexed by channel position, not by name.

**Channel-name mapping (which raw HDF5 key fills which OSF slot) — no
changes from the original draft, still accurate:**

| OSF channel | Our channel (full-channel HDF5) |
|---|---|
| `EEG_C3_A2` | `C3-M2` |
| `EEG_C4_A1` | `C4-M1` |
| `EOG_E1_A2` | `LOC` |
| `EOG_E2_A1` | `ROC` |
| `ECG` | `EKG` (fallback `ECG-L` where `EKG` absent, MrOS/STAGES only) |
| `EMG_Chin` | `CHIN` (fallback generic `EMG` where `CHIN` absent — SHHS, APPLES) |
| `EMG_LLeg` | `LLEG` |
| `EMG_RLeg` | `RLEG` |
| `ABD` | `ABD` |
| `THX` | `Thor` |
| `NP` (nasal pressure/airflow) | `Airflow` |
| `SN` (snore) | `Snore` |

**What changed since the original draft: per-cohort channel *availability*
is far less uniform than "clean, channel-for-channel, all High confidence"
implied.** The original table's confidence ratings were not backed by
measurement — they were plausibility judgments. **Code-verified 2026-08-10**
by sampling 50 random subjects per cohort from the real full-channel HDF5s
at `/scratch/boshra95/psg_full/{dataset}/derived/hdf5_signals/` and checking
raw key presence:

| OSF slot | APPLES (n=50) | SHHS (n=50) | MrOS (n=50) | STAGES (n=50) |
|---|---|---|---|---|
| `ECG` | 100% | 100% | 100% | 90% |
| `EMG_Chin` | 100%† | 100%† | 100% | 100% |
| `EMG_LLeg` | 78% | **0%** | 100% | 56% |
| `EMG_RLeg` | **0%** | **0%** | 100% | 56% |
| `ABD` | 100% | 100% | **0%** | 100% |
| `THX` | 100% | 100% | 100% | 100% |
| `NP` (Airflow) | 100% | 68% | 100% | 100% |
| `SN` (Snore) | 100% | **0%** | **0%** | 100% |
| `EOG_E1_A2` (LOC) | 100% | 100% | 100% | 100% |
| `EOG_E2_A1` (ROC) | 100% | 100% | 100% | 100% |
| `EEG_C3_A2` | 100% | **0%** | 100% | 100% |
| `EEG_C4_A1` | 100% | **0%** | 100% | 100% |

†APPLES's and SHHS's 100% "EMG_Chin" is the generic `EMG` fallback, not a
channel literally named `CHIN` — see the SHHS caveat below.

**This is structural, not per-subject noise** — the 0%/100% entries are
consistent across the full 50-subject sample per cohort, meaning these are
cohort-wide recording/preprocessing characteristics, not something a larger
sample would smooth out. Three findings change the plan materially:

1. **SHHS cannot supply distinct C3/C4 EEG at all.** SHHS's full-channel
   HDF5s store a single generic `EEG` channel (0% `C3-M2`, 0% `C4-M1`, in
   every sampled subject) — SHHS's original PSG montage does not preserve a
   left/right central-electrode distinction in this repo's preprocessing.
   Combined with SHHS also having 0% `LLEG`/`RLEG`/`Snore` and only 68%
   `Airflow`, **5 of OSF's 12 input slots are either always zero or
   majority-zero for every SHHS subject** (`EMG_LLeg`, `EMG_RLeg`, `SN`,
   and effectively both `EEG_C3_A2`/`EEG_C4_A1` unless the generic `EEG`
   channel is duplicated into both slots as an approximation).
   **✅ DECIDED 2026-08-10 (confirmed with the user): duplicate SHHS's
   single generic `EEG` channel into both `EEG_C3_A2` and `EEG_C4_A1`
   slots** (same spirit as the already-accepted `Airflow`-for-`NP`
   approximation), **zero-fill `EMG_LLeg`/`EMG_RLeg`/`SN`.** Explicitly
   flagged by the user as provisional — track in a future results pass
   whether SHHS's OSF numbers look meaningfully degraded by this
   approximation, and revisit with a **targeted re-preprocessing pass for
   SHHS specifically** if so (see Master Implementation Checklist item 0.5
   for what that would involve). Not implementing that reprocessing now —
   only the duplication logic in `extract_osf_embeddings.py`.
2. **MrOS has zero `ABD` channels** (0/50, not partial) despite `ABD` being
   rated "High confidence" in the original table and despite MrOS otherwise
   having excellent coverage (100% on 9 of the other 11 slots, including
   `EMG_LLeg`/`EMG_RLeg`/`CHIN` — better than APPLES or STAGES on those).
   MrOS also has 0% `Snore`. **2 of 12 slots always zero-filled for MrOS**;
   otherwise MrOS is the cleanest cohort for OSF's specific 12-channel
   requirement, which matters since the plan's §0.2 baseline-choice
   reasoning ("OSF needs channels... abdominal... that only the full-channel
   config carries") assumed `ABD` availability that MrOS does not actually
   have.
3. **APPLES has zero `EMG_RLeg`** (0/50, structural — no right-leg EMG
   channel exists in APPLES's full-channel HDF5s at all) and only 78%
   `EMG_LLeg`. **STAGES** has only 56% `EMG_LLeg`/`EMG_RLeg` and 90% `ECG`
   (not the 100% "High confidence" the original table implied for either).

**Missing-channel handling (unchanged decision, now with real numbers to
apply it to):** OSF's own demo code zero-fills any channel absent from the
input — do the same in the extraction script, and **log which channels
were zero-filled per subject** (not just per dataset) so the per-cohort
completeness table above can be regenerated from real extraction-run data,
not just this 50-subject preview sample, before results are reported.
Given the numbers above, no cohort is "mostly zeros" in the sense that
would justify outright exclusion on channel-completeness grounds alone
(APPLES/MrOS/STAGES are all ≥83% complete by slot-count; SHHS is the
outlier at ~58% complete, 7/12 slots at ≥68%, but that's a judgment call,
not an automatic exclusion — see decision point above).

**Conclusion: no raw EDF reprocessing needed** — this part of the original
draft holds. All 12 channels are either present, present via a documented
approximation (generic `EMG`→chin, `Airflow`→`NP`), or absent and
zero-filled, using data already in the existing full-channel HDF5s at
`/scratch/boshra95/psg_full/{dataset}/derived/hdf5_signals/`. The
remaining work is a resample + reorder + rename + rechunk adapter, not new
preprocessing from raw signal.

**Normalization compatibility:** OSF's own `SleepEpochDataset` applies no
additional per-channel normalization to any of these 12 channels beyond a
final `clamp(-6, 6)` (`NEED_NORM_COL = [HR, SPO2, OX]` in OSF's
`train_config.py` — none of which are in our 12-channel list; **code-verified
2026-08-10** directly in `pretrain_dataset.py`'s `SleepEpochDataset.__getitem__`,
both the pretrain and downstream branches). This means OSF expects its 12
input channels to already be roughly zero-mean, unit-variance (z-scored)
before being fed in. Our own preprocessing (`signal_processor.py`'s
`_normalize_signal`) already does per-channel z-score normalization —
**compatible by construction**, but confirm empirically (§9 sanity checks)
rather than assuming.

**EOG referencing — encouraging evidence found 2026-08-10, not yet fully
closed.** The channel-presence audit above only checked whether the
`LOC`/`ROC` *keys* exist, not what they're referenced to. Checking
`configs/channel_definitions.yaml`'s alias tables and the raw per-subject
channel labels in `output/channel_analysis/*_channels.csv` (see "Reference
materials" at the top of this doc) gives a real (if partial) answer:
**STAGES's dominant raw alias is literally `EOG_LOC-A2`/`EOG_ROC-A1`** —
i.e. the raw EDF label itself spells out contralateral-mastoid referencing
(A2 for the left/LOC channel, A1 for the right/ROC channel), which is
*exactly* OSF's expected `EOG_E1_A2`/`EOG_E2_A1` convention, not just a
plausible guess. **However, `channel_definitions.yaml`'s full `LOC` alias
list also folds in non-contralateral variants for other cohorts/subjects**
(`E1:M1` — ipsilateral if M1 is the same-side mastoid, `E1-Cz`, `E1:E2` —
referenced to the other EOG channel, not a mastoid at all), meaning the
referencing convention behind our generic "LOC" standard channel name is
**not uniformly guaranteed across every subject/cohort**, just confirmed
correct for STAGES's dominant case and very likely correct for SHHS/APPLES
(both use `EOG(L)`/`EOG(R)` or bare `LOC`/`ROC` raw labels, which by
standard AASM/clinical PSG convention also imply contralateral mastoid
referencing, though this wasn't verified with the same "the raw label
itself says A2" level of certainty as STAGES). **Net: likely fine for the
large majority of subjects, worth the empirical no-NaN/no-degenerate-CLS
sanity check in §9 item 2 as final confirmation rather than treating this
as fully closed.**

---

## 2. Preprocessing decision (confirmed)

**Reuse existing full-channel HDF5s. Do not reprocess raw EDFs.** Sampling
rate and epoch length are the only real mismatches, and both are handled
in the extraction script (§3), not by re-running the EDF→HDF5 pipeline:

| | Our full-channel HDF5 | OSF's expected input |
|---|---|---|
| Sampling rate | 128 Hz | 64 Hz |
| Chunking | 5-second patches (SleepFM convention) | 30-second epochs |
| Channel set | Harmonized names, already referenced/z-scored | 12 specific channels, same referencing convention |

Resampling 128→64 Hz: OSF's own `_resample_df` does linear interpolation
to the target rate; replicate this exactly (or use `scipy.signal.resample`
/ simple 2:1 decimation with an anti-alias filter — linear interpolation
is what OSF's authors used on their own pretraining data, so matching it
exactly is the lower-risk choice for staying in-distribution).

**Code-verified 2026-08-10 — exact OSF-native pipeline order to replicate**
(from `osf/datasets/pretrain_dataset.py`'s `SleepEpochDataset`, confirmed
identical in `demo.ipynb`'s inline reimplementation): **resample (pandas
`reindex` + `.interpolate(method="linear", limit_direction="both")`,
`_resample_df` at `pretrain_dataset.py:251-263`) → select/reorder to the
fixed 12-channel list, zero-filling any missing → apply `to_pm1` only to
`NEED_NORM_COL` channels (none of our 12, so this step is a no-op for us)
→ pad/truncate to exactly `sample_rate * window_size` = 1920 samples →
`clamp(-6, 6)`.** There is no separate "chunk" step distinct from this —
chunking into 30s epochs *is* windowing at read time, one epoch per forward
pass (see §6.1's per-epoch-only confirmation). `_resample_df`'s literal
implementation, for exact replication:
```python
def _resample_df(self, df, target_hz):
    if not np.issubdtype(df.index.dtype, np.number):
        t = np.arange(len(df)) / float(target_hz)
        df = df.copy(); df.index = t
    t0, t1 = float(df.index.min()), float(df.index.max())
    t_target = np.arange(t0, t0 + self.window_size, 1.0 / target_hz)
    if t_target[-1] > t1:
        t_target = t_target[t_target <= t1 + 1e-9]
    return df.reindex(t_target).interpolate(method="linear", limit_direction="both").fillna(0.0)
```
Note `torch.load(..., weights_only=False)` is required when loading the
checkpoint in §5/§3.1 — the payload is a plain dict of `state_dict` +
`metadata`, which newer torch's default `weights_only=True` safety check
can reject depending on version; this is confirmed in both `demo.ipynb`
and `README.md`'s own loading snippet.

---

## 3. Stage 1 (frozen encoder) — components

### 3.1 New script: `scripts/extract_osf_embeddings.py`

**Environment/checkpoint status (done 2026-08-10, ahead of this section
being implemented): `osf_env` is fully built and verified, and the
checkpoint is downloaded and verified-loadable — see §4/§5, both now
"done" not "planned."**

Mirror `scripts/extract_sleepfm_embeddings.py`'s structure, but **not
uniformly** — one piece of its structure (SIGTERM handling) should
specifically *not* be copied verbatim; see below. Code-verified 2026-08-10
against the real file (`scripts/extract_sleepfm_embeddings.py`):

- **Model import — exact `sys.path` pattern to match** (not a literal
  `"../OSF-Open-Sleep-FM"` string): the existing script computes the
  sibling-repo path via `_REPO = Path(__file__).resolve().parent.parent.parent
  / "sleepfm-clinical"` (three `.parent` calls: `scripts/` → `NSRR-tools/` →
  sibling level) then does **two** insertions —
  `sys.path.insert(0, str(_REPO))` and
  `sys.path.insert(0, str(_REPO / "sleepfm"))`. For OSF, mirror this exactly
  with `_REPO = Path(__file__).resolve().parent.parent.parent /
  "OSF-Open-Sleep-FM"` and a single `sys.path.insert(0, str(_REPO))` (OSF
  has no nested-package layer analogous to `sleepfm-clinical/sleepfm`).
  Import the `ViT` class from `osf.backbone.vit1d_cls` and its
  `vit_base(...)` factory.
- **SIGTERM handling — mirror `extract_sleepfm_embeddings.py`'s pattern,
  NOT `train_context_sweep.py`'s.** These two existing scripts use
  *different* patterns and the embedding-extraction script should follow
  its own sibling, not the training script: `extract_sleepfm_embeddings.py`
  sets a module-level flag (`_stop_requested = False`) in the SIGTERM
  handler and the per-subject loop checks it *after* finishing the current
  subject (graceful finish-current-item, relying on `out_path.exists()`
  skip-logic for resumability — there's no per-subject `resume.pt`).
  `train_context_sweep.py` instead calls `sys.exit(0)` immediately from the
  handler (safe there because it flushes `resume.pt` every epoch). OSF's
  extraction script has no epoch-level checkpoint either, so it needs the
  *finish-current-subject-then-stop* pattern, exactly like
  `extract_sleepfm_embeddings.py` — copy that one's SIGTERM code, not
  `train_context_sweep.py`'s.
- **Checkpoint loading — resolved, no longer an open question.** The real
  HuggingFace Hub filename is **`osf_backbone.pth`** (confirmed both by
  downloading it — see §5 — and by disassembling the archive: its internal
  folder name is `dino_vit_base_backbone/`, meaning the file was originally
  `torch.save`'d as `dino_vit_base_backbone.pth` — the name `demo.ipynb`
  still references — and later renamed to `osf_backbone.pth` for the public
  HF upload. One checkpoint, two historical names; use `osf_backbone.pth`
  since that's what `snapshot_download`/`hf_hub_download` actually produce).
  Load with `torch.load(path, map_location=..., weights_only=False)` — the
  payload is a plain `{"state_dict": ..., "metadata": ...}` dict (**verified
  by actually loading it**: `metadata = {'model_name': 'dino',
  'encoder_name': 'vit_base', 'num_leads': 12, 'patch_size_time': 64,
  'patch_size_ch': 4, 'lead_wise': 1, 'sample_rate': 64,
  'window_size_sec': 30, 'seq_len': 1920, 'width': 768, 'depth': 12}`).
  Instantiate with **all five of these kwargs read from `metadata`, not
  guessed** — `vit_base(num_leads=meta['num_leads'], seq_len=meta['seq_len'],
  patch_size=meta['patch_size_time'], patch_size_ch=meta['patch_size_ch'],
  lead_wise=meta['lead_wise'])` — then `model.load_state_dict(state_dict,
  strict=True)`. **Verified 2026-08-10: this loads with zero missing/unexpected
  keys, 85,325,568 params.** `vit_base(...)`'s own defaults
  (`patch_size=50`, no `lead_wise`/`patch_size_ch`) do NOT match the released
  checkpoint and will raise `assert seq_len % patch_size == 0` if you omit
  `patch_size`/call with defaults — this bit us once already when testing.
  **Patch geometry, now pinned down** (previously left implicit): the
  checkpoint uses `lead_wise=1` (2D `Conv2d` patchify, not the 1D `Conv1d`
  path), giving `Lr = num_leads/patch_size_ch = 12/4 = 3` channel-groups ×
  `Nt = seq_len/patch_size_time = 1920/64 = 30` time-patches = **90 tokens**
  (matches the checkpoint's `pos_embedding` shape `(1, 91, 768)` = 90+CLS
  exactly).
- **Per-subject processing loop**:
  1. Load the 12 mapped channels (§1) from the full-channel HDF5 at 128 Hz.
  2. Resample each channel to 64 Hz (§2's `_resample_df`-equivalent).
  3. Zero-fill any missing channel (log which channels were zero-filled,
     per subject — §1's per-cohort completeness table should be
     regenerated from these logs once extraction actually runs, not left
     as the 50-subject preview sample it currently is).
  4. Chunk into contiguous, non-overlapping 30-second (1920-sample)
     epochs; drop any incomplete trailing epoch (same convention as
     SleepFM's incomplete-chunk handling).
  5. Batch epochs through **`ViT.forward_encoding(x, return_sequence=False)`**
     (⚠️ **corrected from the original draft, which had this boolean
     backwards** — `return_sequence=False` is both the default and the
     value that actually returns the `(cls, patches)` 2-tuple the next step
     needs; `return_sequence=True` returns the raw undivided `[B, 91, 768]`
     sequence with CLS and patches still concatenated together, which would
     silently break the next line's unpacking. Verified directly against
     `demo.ipynb`'s own inference cell, which calls with `return_sequence=False`.)
     → `cls: [B, 768]`, `patches: [B, 90, 768]`.
  6. Mean-pool `patches` over the 90-token axis → `[B, 768]`.
  7. Stack `[cls, mean_pooled_patches]` → `[B, 2, 768]` per epoch.
- **Output**: `{output_dir}/{dataset}/{subject_id}.npy`, dtype float16,
  shape `[T_epochs, 2, 768]` (`T_epochs` = number of complete 30s epochs in
  the recording). Output dir:
  `/scratch/boshra95/psg_full/unified/embeddings/osf_30sec/` (naming
  convention mirrors the existing `sleepfm_5sec/`).
- **GPU batching**: batch across epochs (not subjects), same pattern as
  the SleepFM script's chunk batching, for GPU utilization. Note
  `extract_sleepfm_embeddings.py` has **no `--batch-size` CLI flag** — chunk
  batch size comes only from a config field (`embedding.chunk_batch_size`,
  default 16). Mirror that (config-driven, not CLI-driven) rather than
  adding a new flag.

### 3.2 New dataset class: `src/nsrr_tools/datasets/osf_context_window_dataset.py`

Per the Code Reuse Assessment in `CLAUDE.md`, `ContextWindowDataset` is
SleepFM-shape-hardcoded and not reusable unmodified. **Code-verified
2026-08-10** directly against `context_window_dataset.py`: the real
module-level constants (`:111-118`) are `PATCH_SECONDS = 5`,
`PATCHES_PER_EPOCH = 6`, `EMBED_DIM = 128`, `N_MODALITIES = 4`,
`FLAT_DIM = N_MODALITIES * EMBED_DIM`, `FULL_NIGHT_SENTINEL = -1` — one
more constant (`PATCHES_PER_EPOCH`) than the original draft listed. Its
exact role wasn't fully traced during this verification pass — check its
call sites when actually forking the file, since it may or may not need an
OSF-side equivalent (OSF has no sub-epoch patch/epoch distinction the way
SleepFM does 6×5s-patches-per-30s-epoch, so this constant may simply not
apply, or may need to become `1`; don't assume either way without reading
its usage first). Create `OSFContextWindowDataset` as a **parallel class**,
copied from `ContextWindowDataset` with these constants changed:

```python
N_SUBTOKENS = 2       # was N_MODALITIES = 4  (index 0 = CLS, index 1 = mean-pooled patches)
EMBED_DIM   = 768     # was 128
FLAT_DIM    = 1536    # was 512  (= N_SUBTOKENS * EMBED_DIM)
PATCH_SECONDS = 30    # was 5
```

These are used directly (not shape-introspected from the loaded array) in
every `np.zeros((..., N_MODALITIES, EMBED_DIM))` pad-block allocation and
every `.reshape(N, FLAT_DIM)` call across the three window-builder methods
(causal/centered/seq2label) — confirmed 6 call sites total. If a real
`.npy` had OSF's `(T, 2, 768)` shape but the constants were left at
SleepFM's values, the mismatched pad-block concatenation would crash
loudly (shape mismatch) rather than silently misbehave — a reasonably
safe failure mode if the constants are missed, but still worth getting
right the first time.

**Context-length → epoch-count mapping (recompute, do not reuse
`parse_context_length`'s 5-second-patch arithmetic as-is):**

| Context | SleepFM (5s patches) | OSF (30s epochs) |
|---|---|---|
| 30s | 6 | **1** |
| 10m | 120 | **20** |
| 40m | 480 | **80** |
| 80m | 960 | **160** |
| 120m | 1440 | **240** |
| 240m | 2880 | **480** |

**Cohort consistency filter — recompute `min_recording_patches`.** The
existing filter (`dataset.min_recording_patches: 2880` in
`phase0_v3_full_config.yaml`) is calibrated to 5-second patches (240m ×
60s/m ÷ 5s = 2880). For OSF's 30-second epochs the equivalent value is
**480** (240m × 60s/m ÷ 30s = 480), not 2880. Using 2880 unmodified against
epoch counts would incorrectly exclude almost every subject. **This is
the single easiest place to introduce a silent bug — double check it.**
**Code-verified 2026-08-10: the threshold comparison itself
(`context_window_dataset.py:395-413`) is genuinely unit-agnostic** — it
reads `T` directly from the `.npy`'s on-disk shape (`np.load(...,
mmap_mode="r").shape[0]`) and compares it to `min_recording_patches` as a
plain integer, with no embedded assumption about what a "patch" is, so
`480` will work correctly as-is. **One cosmetic bug to fix while forking,
not a logic bug:** the warning message printed when a subject is excluded
computes a "minutes" figure via `min_min = self._min_recording_patches * 5
// 60` (hardcoded `5`, i.e. seconds-per-patch, baked into the display
string only) — this will print a wrong number for OSF (e.g. `480*5//60=40`
displayed instead of the correct `480*30//60=240`) unless updated to `* 30`
in the fork. Purely cosmetic (doesn't affect which subjects get excluded),
but worth fixing so debug output isn't misleading.

There is no `zero_modality_indices` equivalent needed — OSF has no
4-modality-group structure to ablate. **Code-verified 2026-08-10: this is
a dataset-class-level feature (not training-script-level), cleanly
separable but not centralized** — it's the constructor param
`zero_modality_indices` (stored as `self._zero_modality_indices`), the
`_apply_modality_zeroing()` method, and **three separate call sites** (one
in each of the causal/centered/seq2label window-builder methods). Forking
means dropping the constructor param, the stored attribute, the method,
and all three call sites — mechanically simple, but four distinct edits,
not one.

Everything else (K-sampling logic — overlapping train/val/test vs.
non-overlapping inference at K_max>100, `SubjectGroupedSampler`, seq2label
window building, padding/collate) is shape-parametric once the above
constants are updated — copy as-is. **Code-verified 2026-08-10**: the
window-index builder (`_build_seq2label_index`) and `SubjectGroupedSampler`
operate purely on integers (`T` from the shape cache, `N` from
`parse_context_length`) with zero references to `N_MODALITIES`/`EMBED_DIM`
anywhere in their logic — confirming this reuse claim holds exactly as
stated, not just approximately.

### 3.3 New training/inference scripts

Per the Code Reuse Assessment, `train_context_sweep.py` and
`infer_subject_windows.py` are mostly backbone-agnostic (they delegate
embedding I/O entirely to the dataset class and only import
`build_head`/`ContextWindowDataset`). Fork them as:

- `scripts/train_osf_context_sweep.py` — same as `train_context_sweep.py`
  except: import `OSFContextWindowDataset` instead of
  `ContextWindowDataset`; drop the `--zero-modalities` CLI flag and
  `_MODALITY_INDICES` dict (not applicable, no modality groups); otherwise
  identical (checkpoint/resume, early stopping, overfit-phase, snapshots,
  bootstrap-CI-adjacent flags all carry over unchanged since they're
  head/optimizer-level, not backbone-level). **Code-verified 2026-08-10:
  `_MODALITY_INDICES = {"BAS":0,"RESP":1,"EKG":2,"EMG":3}` is real
  (`train_context_sweep.py:117`, `infer_subject_windows.py:70`) and is
  correctly the only *shape/channel-semantic* SleepFM hardcode in either
  file** — but not literally the only SleepFM-*calibrated* thing worth
  knowing about (see the batch-size note below). Checkpoint/resume for
  reference when replicating in the fork: `resume.pt` (full training state
  — epoch, step, model/optimizer/scheduler state, history, etc.) is
  rewritten every epoch; a `SIGTERM` handler calls `sys.exit(0)`
  immediately (safe because `resume.pt` is already flushed); `best_model.pt`
  (lighter, state_dict only) saves whenever the monitored val metric
  improves; `resume.pt` is deleted on successful completion so a later
  resubmit isn't mistaken for a mid-run resume.
- `scripts/infer_osf_subject_windows.py` — same relationship to
  `infer_subject_windows.py`. **One SleepFM-calibrated (not
  SleepFM-hardcoded) detail worth checking rather than blindly trusting**:
  `infer_subject_windows.py` auto-scales its inference batch size against a
  reference point tuned for SleepFM's context arithmetic (`_ref_bs=64`,
  `_ref_N=2880`, i.e. "training used batch=32 at 240m/2880 patches") via
  `eff_bs = min(args.batch_size, max(_ref_bs, int(_ref_bs*_ref_N/N_patches)))`.
  This is generic index arithmetic (won't crash or misbehave for OSF's much
  smaller epoch counts and much larger 1536-dim vectors) but it also isn't
  *tuned* for OSF's memory profile — don't assume the auto-scaled batch
  size is well-chosen for OSF without checking GPU memory headroom
  empirically (§9's pilot run is the place to catch this).

**Model architecture: keep `hidden_dim=128, num_layers=1` (matching
`phase0_v3_full_config.yaml`'s seq2label head), only `input_dim` changes**
(1536 instead of 512). This preserves the existing paper's "architecture
held constant, only the encoder/channels change" comparison principle —
do not tune the head size differently for OSF without a specific reason.

### 3.4 New config: `configs/phase0_osf_config.yaml`

Copy `configs/phase0_v3_full_config.yaml` as the template. **⚠️ Correction
(code-verified 2026-08-10): the template below was originally missing a
required `logging:` section — `train_context_sweep.py:998` and
`infer_subject_windows.py:215` both do `Path(cfg["logging"]["results_dir"])`
via plain bracket access with no default, and the job scripts also read
`['logging']['results_dir']` for failure-reason lookup. Without this
section, the very first invocation of either script would `KeyError`
immediately.** Now added below. Changes from the real
`phase0_v3_full_config.yaml`:

```yaml
embedding:
  checkpoint_dir: "/home/boshra95/OSF-Open-Sleep-FM/pretrained_weights/osf_backbone.pth"  # resolved filename, see §3.1/§5
  output_dir: "/scratch/boshra95/psg_full/unified/embeddings/osf_30sec"
  # chunk_batch_size: tune empirically — start at 16, matching SleepFM's default

data:
  hdf5_dir: "/scratch/boshra95/psg_full"     # same source HDF5s as SleepFM full-channel
  sampling_freq: 64                            # OSF's expected rate, not 128
  epoch_seconds: 30                            # was chunk_seconds: 300 / patch_size: 640
  channel_order: [ECG, EMG_Chin, EMG_LLeg, EMG_RLeg, ABD, THX, NP, SN,
                   EOG_E1_A2, EOG_E2_A1, EEG_C3_A2, EEG_C4_A1]
  channel_mapping:                              # our name -> OSF name; primary alternative only, see §1's
                                                 # per-cohort table for fallback names (e.g. generic EMG for
                                                 # CHIN, ECG-L for EKG) and the SHHS EEG-duplication decision
    C3-M2: EEG_C3_A2
    C4-M1: EEG_C4_A1
    LOC: EOG_E1_A2
    ROC: EOG_E2_A1
    EKG: ECG
    CHIN: EMG_Chin
    LLEG: EMG_LLeg
    RLEG: EMG_RLeg
    ABD: ABD
    Thor: THX
    Airflow: NP
    Snore: SN

dataset:
  embedding_dir: "/scratch/boshra95/psg_full/unified/embeddings/osf_30sec"
  label_source: "/scratch/boshra95/psg/unified/targets_v2/master_targets.parquet"   # documentation only, see note below
  task_subject_dir: "/scratch/boshra95/psg/unified/targets_v2/task_subjects"         # SAME as SleepFM — this one is load-bearing
  context_lengths: ["30s", "10m", "40m", "80m", "120m", "240m"]
  datasets: [apples, shhs, mros, stages]
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
  split_seed: 42          # SAME seed as SleepFM runs — required for a fair comparison on identical splits
  windows_per_subject: 5
  min_recording_patches: 480    # NOTE: 480, not 2880 — see §3.2 cohort-filter recompute

model:
  input_dim: 1536     # 2 * 768, not 512
  head_type: "lstm"   # sweep lstm/transformer/mean_pool same as SleepFM
  hidden_dim: 128
  num_layers: 1
  num_heads: 8
  dropout: 0.3
  num_classes: 2       # per-task, same as existing registries

training:
  epochs: 40
  lr: 1.0e-4
  weight_decay: 1.0e-3
  optimizer: "adamw"   # NOTE: not actually read — see caveat below
  scheduler: "cosine"  # NOTE: not actually read — see caveat below
  early_stopping_patience: 10
  device: "cuda"        # NOTE: not actually read — see caveat below

logging:                # ⚠️ REQUIRED — missing from the original draft, would KeyError without it
  results_dir: "/scratch/boshra95/psg_full/unified/results/phase0_osf"
```

**Two footnotes, both code-verified 2026-08-10, that materially change how
to read this template — neither is an OSF-specific bug, both are
pre-existing behavior in the SleepFM config/scripts too, just easy to miss
when copying the template for a new backbone:**

1. **`training.optimizer`/`training.scheduler`/`training.device` are dead
   config keys — `train_context_sweep.py` never reads any of the three.**
   The optimizer is hardcoded to plain `torch.optim.Adam` (not AdamW,
   despite the field commonly being set to `"adamw"` in existing configs —
   weight decay is applied Adam-style, not decoupled), the scheduler is
   hardcoded to `CosineAnnealingLR`, and device selection comes from the
   `--cpu` CLI flag / `torch.cuda.is_available()`, not this config field.
   This is pre-existing cruft already present in
   `phase0_v3_full_config.yaml` itself (not introduced by this template) —
   don't assume changing these three keys in the OSF config does anything;
   if a different optimizer/scheduler is ever wanted for OSF specifically,
   it requires an actual code change to `train_osf_context_sweep.py`, not
   a config edit.
2. **`dataset.label_source` is documentation-only — never read anywhere in
   `src/` or `scripts/` (grepped, zero hits).** The actual runtime
   label/subject source is `{task_subject_dir}/{task}_subjects.csv`
   (already-materialized per-task CSVs), plus `train_split`/`val_split`/
   `test_split`/`split_seed` (all read directly by `ContextWindowDataset`).
   The "reuse the same labels/splits as SleepFM" requirement below is still
   correct and still matters — it just actually depends on
   `task_subject_dir` + `split_seed` (+ the split ratios), not
   `label_source`, which can be left pointing at the same file purely for
   human documentation value.

**Label/split reuse is a hard requirement, not just an optimization**: OSF
must be trained/evaluated on the **exact same subjects and the exact same
train/val/test split** as the SleepFM `phase0_v3_full` runs for the
comparison to mean anything — hence pointing `task_subject_dir` and
`split_seed` (the fields that actually matter at runtime, see footnote 2
above) at the identical existing files rather than regenerating them.

### 3.5 New registry + command generation

Per the Code Reuse Assessment, `gen_commands.py` has no backbone hook and
retrofitting it is higher-risk than a parallel generator. For this first
pass (5 tasks × 3 heads × 6 contexts = 90 training runs, manageable),
create:

- `experiments/v2_osf_registry.yaml` — same schema as `v2_registry.yaml`,
  restricted to the 5 Tier-1 tasks × 3 heads, pointing at
  `configs/phase0_osf_config.yaml`, `results_dir:
  /scratch/boshra95/psg_full/unified/results/phase0_osf`,
  `logs_dir: /home/boshra95/NSRR-tools/logs_osf`. **Two more top-level
  fields the plan's original sketch didn't mention, code-verified
  2026-08-10 as required/load-bearing:**
  - **`inference_dir`** — required, not optional: `gen_commands.py:199`
    does `Path(registry["inference_dir"])` via plain bracket access (no
    default), so `infer`-subcommand generation will `KeyError` without it.
    Set to `/scratch/boshra95/psg_full/unified/results/phase0_osf/inference`
    (mirrors `v2_full_registry.yaml`'s `.../phase0_v3_full/inference`
    convention).
  - **`python_bin`** — technically optional (has a fallback default), but
    **the fallback default is `/home/boshra95/sleepfm_env/bin/python`**
    (hardcoded in `gen_commands.py` at every one of its ~15 call sites via
    `registry.get("python_bin", "/home/boshra95/sleepfm_env/bin/python")`).
    **If the OSF registry omits `python_bin`, any command generated by the
    unmodified `gen_commands.py` (or a careless fork that keeps this
    default) would silently use `sleepfm_env`'s interpreter — which has
    none of OSF's `torch`/`peft`/`transformers` pins — rather than
    `osf_env`.** Set explicitly to `/home/boshra95/osf_env/bin/python` in
    the registry, and if `gen_commands_osf.py` is forked (next bullet),
    update its own copy of that fallback default too, not just the
    registry.
- `scripts/gen_commands_osf.py` — fork of `gen_commands.py` trimmed to
  the subcommands actually needed for this pass (`train`, `infer`,
  `analyze`, `status`, `runs`, `collect` — the plotting/table
  subcommands can wait, or better: once results land in the same
  `metrics.json`/`summary.csv`/parquet schema, try pointing the *existing*
  `analyze`/`collect`/plotting code at the new results dir directly rather
  than forking those too — they're plausibly reusable as-is per the Code
  Reuse Assessment). New wall-time lookup tables (`_TRAIN_HOURS`,
  `_INFER_HOURS_PER_CTX`) will need fresh calibration — OSF's per-epoch
  ViT forward pass has a different cost profile than SleepFM's frozen
  512-dim embeddings, so do **not** copy SleepFM's wall-time numbers
  as a starting assumption. Start with generous `--time` estimates for the
  first few runs, then tighten the table once actual wall-clock times are
  observed (mirrors how the original SleepFM table was calibrated —
  see `docs/EXPERIMENTS_GUIDE.md` §"Expected Runtimes").

### 3.6 New job scripts

`jobs/extract_osf_embeddings_gpu.sh`, `jobs/train_osf_context_sweep_gpu.sh`,
`jobs/infer_osf_subject_windows_gpu.sh` — copy the existing
`*_gpu.sh`/`*_gpu_rorqual.sh` pairs, pointing at the new Python scripts and
`osf_env` (§4, already built) instead of `sleepfm_env`.

**Auto-resume mechanism — two distinct mechanisms, at two different
levels, both real (code-verified 2026-08-10 directly against
`jobs/train_context_sweep_gpu.sh` and `scripts/gen_commands.py`, after an
initial pass of this verification incorrectly concluded `--requeue` was
unused — correcting that here since it was double-checked and found
wrong):**
1. **Wall-time-triggered resume** (bash-level, baked into the `.sh` file
   itself): `#SBATCH --signal=B:USR1@120` sends `SIGUSR1` to the **bash**
   script (the `B:` prefix) 120s before the wall-time limit; Python runs in
   the background (`eval "$CMD" & ; _PYTHON_PID=$! ; wait $_PYTHON_PID`) so
   `wait` returns immediately when the trap fires instead of blocking until
   Python exits on its own; a `trap '_timeout_handler' USR1` handler writes
   a `TIMEOUT_REQUEUED` status line, `SIGTERM`s the Python process, reads
   the job's original time limit via `scontrol show job`, and **calls
   `sbatch` again on its own script path** (`sbatch --export=ALL
   --time="$_TIME_LIMIT" --output=... --error=... "$_SCRIPT_PATH"` — this
   particular resubmit call does **not** include `--requeue`, confirmed by
   reading the trap's exact `sbatch` invocation) — i.e. it resubmits itself
   as a brand-new SLURM job, which then finds `resume.pt` on disk and
   continues.
2. **Node-failure requeue** (SLURM-native, supplied at the *initial*
   `sbatch` invocation, not inside the `.sh` file): `scripts/gen_commands.py`
   appends a literal `--requeue` flag when it constructs the `sbatch`
   command line for both `train`/`infer` subcommands (confirmed at two call
   sites). This is what handles a node failure/preemption mid-run (SLURM's
   own requeue mechanism) — a genuinely different failure mode from #1
   (wall-time approaching), and it's why `CLAUDE.md`'s existing one-line
   summary ("Auto-requeue via `--signal=B:USR1@120` + `--requeue`") is
   correct as a list of the two mechanisms in play, even though neither
   `--signal` nor `--requeue` alone would be a complete description — they
   apply at different levels (script pragma vs. `sbatch` CLI flag) for
   different failure modes, which is easy to miss from a `jobs/*.sh`-only
   grep (that's the mistake this verification pass initially made and is
   correcting here).

**Practical implication for a new `jobs/train_osf_context_sweep_gpu.sh`**:
copy the `#SBATCH --signal=B:USR1@120` + bash trap pattern verbatim into
the `.sh` file (mechanism #1); separately, whatever generates the `sbatch`
invocation for OSF runs (§3.5's `gen_commands_osf.py`, or manual
submission during early debugging) needs to append `--requeue` itself at
submission time (mechanism #2) — don't expect the `.sh` file alone to
provide node-failure resilience. Other structural details to copy exactly
from the existing job scripts: SLURM directives (`--account=def-forouzan_gpu`,
`--gpus=nvidia_h100_80gb_hbm3_1g.10gb:1`, `--cpus-per-task=4`,
`--mem=32000M`, `--exclude=fc11006[,...]`), a CUDA fail-fast check
(`python -c "import torch; assert torch.cuda.is_available()..."`), and a
per-job JSONL status log under `{logs_dir}/status/*.jsonl` with
`STARTED/TIMEOUT_REQUEUED/SUCCESS/FAILED` events. **One nuance specific to
the new scripts**: `train_context_sweep.py`/`infer_subject_windows.py`
never import SleepFM code directly (only the embedding-extraction script
does), so the `PYTHONPATH` export pointing at the sibling model repo is
only actually needed in `extract_osf_embeddings_gpu.sh`, not in
`train_osf_context_sweep_gpu.sh`/`infer_osf_subject_windows_gpu.sh` — copy
it there too for consistency if it's harmless, but don't assume it's
load-bearing in the train/infer job scripts.

---

## 4. Environment — ✅ DONE (2026-08-10), built and verified on-cluster

**Do not reuse `sleepfm_env`.** OSF's `requirements.txt` pins
`torch==2.5.1`, `transformers==4.47.0`, `peft==0.14.0`,
`pytorch-lightning==2.4.0`, `timm==1.0.12`, plus several git-based
dependencies — risking a version conflict with whatever `sleepfm_env` has
pinned for the existing pipeline. `/home/boshra95/osf_env` (Python 3.10.13,
via the cluster's `python/3.10.13` module) now exists and has all 187
packages installed and verified: `import nsrr_tools` and
`from osf.backbone.vit1d_cls import ViT, vit_base` both import cleanly, and
the real checkpoint loads with zero missing/unexpected `state_dict` keys
(§5). **Several real obstacles came up that the original draft's simple
`pip install -r requirements.txt` command doesn't survive unmodified —
recorded here so no one repeats the debugging:**

1. **This cluster's Python doesn't self-report as `manylinux`-compatible
   to pip** (`pip debug --verbose` shows only plain `linux_x86_64` tags,
   no `manylinux*` variants) — standard PyPI wheels for many packages are
   therefore unusable here, and pip falls back to building from source
   (or fails outright, e.g. for `onnxruntime`, which ships no sdist).
   Compute Canada's own wheel cache (`/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/`,
   auto-configured via `$PIP_CONFIG_FILE`) has plain-`linux_x86_64`-tagged
   builds and is the actual working path — but it doesn't carry every
   exact version OSF's `requirements.txt` pins. **Practical rule that
   worked:** for any package CC's wheelhouse has *some* version of, relax
   the exact pin to the closest version CC carries rather than fighting to
   get the literal pinned version from PyPI; only fall back to building
   from source (needs a `module load` of a matching toolchain — `gcc`,
   `rust/1.85.0`+ for Rust extensions, etc.) for packages CC's wheelhouse
   doesn't have at all. From-source C++ builds are also memory-constrained
   on this node — `matplotlib` and `xgboost`'s parallel C++ compiles both
   got OOM-killed (`cc1plus`/`cython` subprocess `SIGKILL`) before being
   relaxed to CC wheelhouse versions instead.
2. **~20 packages in `requirements.txt` are genuinely unused by OSF's own
   code** (confirmed via `grep -rl "import X" --include="*.py"` across the
   whole repo, zero hits) — an artifact of a shared team requirements file,
   exactly as suspected below. Safely dropped rather than fought:
   `kornia`/`kornia_rs` (also: broken sdist packaging bug on PyPI,
   confirmed even with Rust installed), `opencv-python`/`opencv-python-headless`,
   `albumentations`/`albucore`, `insightface` (the actual source of the
   `albumentations`/`opencv` chain — dropping `albumentations` alone wasn't
   enough since `insightface` pulls it back in transitively), `wandb`
   (needs a Go toolchain to build `wandb-core` from source; also confirmed
   used only in OSF's own `main_pretrain.py`/`main_finetune.py`, which this
   integration never runs), `xgboost`, `streamlit` (the source of a
   `pyarrow` build failure — CC deliberately ships a dummy `pyarrow`
   package on this cluster instructing `module load arrow` instead of pip;
   dropping `streamlit`/`datasets`/`pyarrow` — all confirmed unused —
   avoided that entirely rather than fighting the module-load requirement),
   `datasets`, `scikit-image`, `onnx`/`onnx2torch`, `numba`/`llvmlite`
   (came back transitively via `pynndescent`, a real dependency — but
   resolved fine from CC's wheelhouse once not exact-pinned), `ml_dtypes`.
   A working copy of the trimmed requirements lives at
   `/home/boshra95/osf_env_requirements.txt` (the original
   `OSF-Open-Sleep-FM/requirements.txt` is untouched).
3. **Version-relaxed (CC wheelhouse's closest available, not the exact
   pin), all confirmed working:** `numpy` 2.1.2→2.1.1, `h5py` 3.14.0→3.12.0,
   `onnxruntime` 1.23.1→1.17.3 (also unused by OSF's own code, kept anyway
   since it installs cleanly), `safetensors` 0.6.2→0.4.5, `matplotlib`
   3.9.3→3.9.2, `grpcio` — dropped as an exact top-level pin (unused, only
   a transitive dependency of `tensorboard`/`wandb`, which don't need the
   specific version) and resolved fine from CC's wheelhouse once
   unconstrained.
4. **`pip install -e /home/boshra95/NSRR-tools` does not work as the
   original draft assumed — NSRR-tools' `pyproject.toml` declares
   `requires-python = ">=3.11"`**, incompatible with `osf_env`'s Python 3.10
   (deliberately chosen to match OSF's own pinned stack). Modern pip has no
   flag to bypass a `requires-python` gate for an editable install.
   **Actual working substitute**: a `.pth` file
   (`osf_env/lib/python3.10/site-packages/nsrr_tools_src.pth`, containing
   the single line `/home/boshra95/NSRR-tools/src`) achieves the identical
   practical effect (`import nsrr_tools` works) without needing to satisfy
   the packaging metadata gate. Use this instead of `pip install -e` when
   setting up `osf_env`.

```bash
module load python/3.10.13
python3.10 -m venv /home/boshra95/osf_env
source /home/boshra95/osf_env/bin/activate
pip install -r /home/boshra95/osf_env_requirements.txt   # the trimmed/relaxed copy, not the repo's own requirements.txt
echo "/home/boshra95/NSRR-tools/src" > "$(python -c 'import site; print(site.getsitepackages()[0])')/nsrr_tools_src.pth"
python -c "import nsrr_tools; from osf.backbone.vit1d_cls import ViT, vit_base" \
  # (run from /home/boshra95/OSF-Open-Sleep-FM, or with it on sys.path — both import cleanly once there)
```

Note: `environment.yml` in the OSF repo is an **aarch64 (ARM) conda lock
file** — not usable on a Compute Canada x86_64 cluster. Use
`requirements.txt` (pip, architecture-independent) instead, as above.
**Also code-verified 2026-08-10**: `README.md`'s own "Dependencies"
section states `PyTorch >= 2.9.0` and `PyTorch Lightning >= 2.5.5` — both
*above* what `requirements.txt` actually pins (`torch==2.5.1`,
`pytorch-lightning==2.4.0`). Trust `requirements.txt` (the real, working,
pinned environment), not the README prose.

The dependencies that looked unrelated to the sleep encoder itself turned
out to genuinely be unrelated (see point 2 above) — the original
"install everything first, trim only on a real conflict" instinct was
right in spirit, and every trim made was backed by both a grep-confirmed
zero-usage check and an actual install failure, not a guess.

---

## 5. Checkpoint download — ✅ DONE (2026-08-10), downloaded and verified-loadable

```bash
source /home/boshra95/osf_env/bin/activate
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='yang-ai-lab/OSF-Base',
                   local_dir='/home/boshra95/OSF-Open-Sleep-FM/pretrained_weights')
"
```

**Filename resolved: it's `osf_backbone.pth`** (325MB / 341,360,652 bytes),
not `dino_vit_base_backbone.pth` — confirmed both by actually downloading
it and by disassembling the archive (its internal folder name is
`dino_vit_base_backbone/`, meaning the checkpoint was originally
`torch.save`'d under that name and later renamed to `osf_backbone.pth` for
the public HF upload; one file, two historical names — `demo.ipynb` just
has the stale one). Use `osf_backbone.pth`, matching what
`snapshot_download`/`hf_hub_download` actually produce.

**End-to-end load verified**, not just downloaded:
```python
import torch
from osf.backbone.vit1d_cls import vit_base
ckpt = torch.load('pretrained_weights/osf_backbone.pth', map_location='cpu', weights_only=False)
meta = ckpt['metadata']   # {'model_name':'dino','encoder_name':'vit_base','num_leads':12,
                           #  'patch_size_time':64,'patch_size_ch':4,'lead_wise':1,
                           #  'sample_rate':64,'window_size_sec':30,'seq_len':1920,'width':768,'depth':12}
model = vit_base(num_leads=meta['num_leads'], seq_len=meta['seq_len'],
                  patch_size=meta['patch_size_time'], patch_size_ch=meta['patch_size_ch'],
                  lead_wise=meta['lead_wise'])
missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
# missing: [], unexpected: [] — full strict match, 85,325,568 params
```
**License: MIT** (verified — `LICENSE` file, UCLA Health Intelligence Lab,
standard unmodified boilerplate, no additional restrictions) — a clean,
permissive baseline, notably better-documented than PhysioOmni's situation
(no LICENSE file at all, per `CLAUDE.md`'s existing note on that model).

---

## 6. Stage 2 (LoRA fine-tuning) — components

This is architecturally different from Stage 1, not just a flag flip:
Stage 1 precomputes embeddings once and reuses them across all training
runs; Stage 2 needs the OSF encoder inside the trainable graph, so
embeddings can no longer be precomputed — raw signal has to be loaded and
encoded on the fly, every training step.

### 6.1 New script: `scripts/train_osf_lora.py`

A genuinely new end-to-end training script, not a fork of
`train_context_sweep.py`. Structure:

- **New raw-epoch dataset** (e.g. `OSFRawEpochWindowDataset`): same
  windowing/K-sampling logic as `OSFContextWindowDataset` (§3.2), but
  `__getitem__` returns the raw `[N_epochs, 12, 1920]` signal tensor for
  the window (built from the full-channel HDF5 via the §1 channel mapping
  + resample, same as the extraction script) instead of a precomputed
  embedding array.
- **Combined model**: wrap OSF's `ViT` with LoRA
  (`peft.get_peft_model(vit, LoraConfig(target_modules=["to_qkv", "to_out.0"], r=..., lora_alpha=...))`
  — `to_qkv` and `to_out.0` are the actual Linear-layer attribute names in
  OSF's `Attention` class, `osf/backbone/vit1d_cls.py`, confirmed by
  reading the source directly, **and independently confirmed 2026-08-10
  from the real checkpoint's own `state_dict` key strings** (e.g.
  `block0.attn.fn.to_qkv.weight`, `block0.attn.fn.to_out.0.weight`, through
  `block11...`). **One nuance PEFT's config hides but worth knowing if
  anyone ever needs the exact dotted path** (e.g. manual freezing/inspection
  outside PEFT): `Attention` is wrapped in a `PreNorm` module
  (`self.attn = PreNorm(dim=..., fn=attn)`), so the *fully-qualified*
  module path is actually `block{i}.attn.fn.to_qkv` /
  `block{i}.attn.fn.to_out.0`, not `block{i}.attn.to_qkv`. This doesn't
  change the `LoraConfig` above — PEFT matches `target_modules` by name
  *suffix*, so plain `"to_qkv"`/`"to_out.0"` still correctly hits all 12
  blocks — but don't be surprised by the `.fn.` segment if inspecting
  `named_modules()` directly.), then wrap `(lora_vit, sequence_head)`
  together in one `nn.Module` so a single optimizer and a single
  `modules_to_save`-style mechanism cover both. Concretely: build the
  combined module first, *then* call `get_peft_model` on the whole thing
  with `modules_to_save=["sequence_head"]`, rather than LoRA-wrapping the
  ViT in isolation and bolting the head on after — this way PEFT's
  save/load and gradient-freezing logic treats the head as a first-class
  fully-trainable submodule, matching the `modules_to_save=["classifier"]`
  pattern already documented in `docs/TSFM_BASELINE_CANDIDATES.md` §6.
- **Warm start from Stage 1**: load the Stage-1-trained sequence head's
  weights into the combined module's head submodule before starting Stage
  2 training (per the staged LP-FT procedure in `CLAUDE.md` → "Frozen vs.
  LoRA-fine-tuned conditions") — don't start LoRA fine-tuning from a
  randomly-initialized head.
- **Forward pass per training step**: for each window, run all `N_epochs`
  raw epochs through the LoRA-adapted ViT (batched across epochs, same
  batching pattern as the extraction script, calling
  `forward_encoding(x, return_sequence=False)` per §3.1's corrected call —
  same fix applies here) to get `(cls, patches)` → mean-pool patches →
  stack to `[N_epochs, 2, 768]` embeddings, flatten to `[N_epochs, 1536]`,
  feed through the sequence head, compute loss, backprop through **both**
  the LoRA adapters and the head.
- **Checkpoint/resume**: replicate `train_context_sweep.py`'s
  `resume.pt`/`best_model.pt` pattern (save combined-module + optimizer +
  scheduler state every epoch; SIGUSR1/timeout handling via the same job
  script pattern) — don't skip this, LoRA runs at long context will be the
  slowest, most timeout-prone jobs in the whole project.

### 6.2 Memory mitigation ladder (apply in this order — per §0 decision)

1. **Gradient checkpointing** (`torch.utils.checkpoint`) through the ViT's
   transformer blocks — standard, should be the first thing tried, cheap
   to implement, no change to results.
2. **Request a larger GPU memory allocation** on Compute Canada (bigger
   MIG slice or full H100) if checkpointing alone isn't enough.
3. **Only if both of the above are insufficient**: cap the LoRA condition
   at the longest context length that fits (e.g. stop at 80m or 120m
   instead of 240m), keep the frozen-embedding condition (Stage 1) at all
   6 context lengths as usual, and state the compute ceiling explicitly in
   any results table/caption — do not silently omit the long-context LoRA
   points.

### 6.3 Wall-time / compute budget — unknown, do not assume

Unlike Stage 1 (which reuses the well-calibrated SleepFM-style
checkpoint/resume/wall-time infrastructure), Stage 2's per-step cost is
new and uncalibrated: it does a full ViT forward+backward pass per epoch
in the window, at every training step, versus Stage 1's cheap
precomputed-embedding lookup. **Run a short pilot (a handful of epochs at
the smallest context length, e.g. 30s or 10m) before submitting the full
sweep**, to get real wall-clock numbers and set realistic `--time` values
— do not extrapolate from SleepFM's training-time table.

---

## 7. Task/label reuse recap

No new target extraction is needed. `master_targets.parquet` and
`task_subjects/*.csv` under `/scratch/boshra95/psg/unified/targets_v2/`
already exist from the SleepFM pipeline and cover all 7 tasks (including
the 5 Tier-1 tasks used here) — point the new config at them directly
(§3.4). The only new label-adjacent work would be if a future pass needs a
task OSF can't fairly support (not the case for any of the 5 Tier-1 tasks
— the apnea/no-respiratory-pathway problem is specific to PhysioOmni, not
OSF, which does have a respiratory channel).

---

## 8. Honest reporting reminders (recap from `CLAUDE.md`, OSF-specific)

- **SHHS is confirmed in OSF's pretraining set** — any SHHS-inclusive
  AUROC comparison against SleepFM needs an explicit contamination
  caveat. **STAGES is very likely also in pretraining** (numeric-ID
  pattern match) — confirm by cross-checking a handful of numeric IDs
  from OSF's `osf/splits/patient_pretrain_*.csv` against our local STAGES
  `subject_code` list before treating as certain (this is a 10-minute
  cluster task, do it early). **MrOS is downstream/eval-only in OSF's own
  splits** (lower risk). **APPLES is clean.**
- Report APPLES (and, once confirmed, STAGES-excluded) results as the
  primary honest comparison; report the full 4-cohort numbers too but with
  the contamination caveat stated alongside, not buried in a footnote.
- If a run doesn't complete (frozen or LoRA, any context length), say so
  explicitly in the results table rather than leaving a blank/implied-zero
  cell.

---

## 9. Step 0 — verification checklist (run these before the full sweep)

1. **Channel availability per cohort — ✅ DONE (2026-08-10), see §1.**
   50-subject-per-cohort audit complete, with exact percentages per OSF
   slot per cohort. Headline results: SHHS has 0% `EEG_C3_A2`/`EEG_C4_A1`/
   `EMG_LLeg`/`EMG_RLeg`/`SN` and 68% `NP` — **a decision on how to handle
   SHHS's missing EEG-channel distinction is still needed before
   implementation** (§1 lists 4 options; not yet chosen). MrOS has 0%
   `ABD`/`SN`. APPLES has 0% `EMG_RLeg`. STAGES has only 56%
   `EMG_LLeg`/`EMG_RLeg` and 90% `ECG`. **Re-run this audit against the
   real extraction output once `extract_osf_embeddings.py` exists and has
   run on all subjects** (not just the 50-per-cohort preview sample) to
   confirm the full-population numbers match this preview.
2. **EOG referencing check — still open, not resolved by the channel-count
   audit above** (that only checked whether `LOC`/`ROC` *keys* exist, not
   their reference electrode). Confirm `LOC`/`ROC` in our HDF5s are
   referenced the way OSF expects (contralateral mastoid) — check
   `channel_mapper.py`/original NSRR channel-label provenance, or inspect
   embedding output sanity (no NaN, no degenerate all-zero CLS vectors)
   once extraction runs on a small pilot batch.
3. **Checkpoint filename resolution — ✅ DONE (2026-08-10), see §5.**
   Confirmed `osf_backbone.pth`; checkpoint downloaded and
   strict-load-verified (85.3M params, zero missing/unexpected keys).
4. **Small-scale pilot** — still to do once `extract_osf_embeddings.py`
   exists: run `--limit 5` on one dataset, inspect the output `.npy`
   shape/values, then a tiny Stage-1 training run (`--context 30s`, one
   task, one head) end-to-end before submitting the full 90-run sweep.
   This is where a VSCode debug config earns its keep — see §12.
5. **STAGES-in-pretraining confirmation** (§8) — cross-check numeric IDs.
   Not yet done.
6. **Cohort filter unit check** (§3.2) — confirm `min_recording_patches:
   480` is actually being applied in epoch units. **Partially resolved by
   code reading** (2026-08-10): the threshold comparison itself is
   genuinely unit-agnostic (reads `T` from the `.npy`'s on-disk shape, no
   embedded patch-duration assumption) — so `480` will work correctly
   once the file exists. Still needs an *empirical* check once real OSF
   `.npy` files exist (confirm the printed subject-exclusion counts look
   sane, not that every subject gets excluded or none do).
7. **NEW — Environment/checkpoint sanity, already covered by §4/§5's own
   verification, listed here for completeness**: `osf_env` package install
   ✅, `import nsrr_tools` + `from osf.backbone.vit1d_cls import ViT,
   vit_base` ✅, checkpoint strict-load ✅ — all done 2026-08-10, ahead of
   the rest of this checklist.

---

## 10. Suggested execution order

**Superseded by the "Master Implementation Checklist" near the top of this
doc (added 2026-08-10) — that's the actively-maintained, checkable list;
this section is kept only for the narrative reasoning behind the
ordering.**

1. ~~Set up `osf_env` (§4), download the checkpoint (§5).~~ **✅ DONE
   2026-08-10.**
2. **Decide the SHHS channel-completeness question (§1)** — this blocks a
   concrete decision inside `extract_osf_embeddings.py`'s channel-mapping
   logic, so resolve it before writing that script, not during.
3. Run the remaining Step 0 verification checklist items (§9: EOG
   referencing, STAGES-in-pretraining ID cross-check) — cheap, catch
   expensive mistakes early.
4. Implement `extract_osf_embeddings.py` (§3.1) with a VSCode debug config
   (§12) from the start, run on a small pilot, then the full extraction
   for all 4 datasets.
5. Implement `OSFContextWindowDataset` (§3.2), `phase0_osf_config.yaml`
   (§3.4), forked train/infer scripts (§3.3), registry + generator (§3.5),
   job scripts (§3.6) — each with a debug config (§12) added as it's
   written, mirroring the existing NSRR-tools pattern.
6. Run Stage 1 (frozen) for all 5 Tier-1 tasks × 3 heads × 6 contexts = 90
   training runs, then inference, then analysis (reuse existing
   `analyze_windows.py`/`collect_results_v2.py` pointed at the new results
   dir if the schema lines up, per the Code Reuse Assessment — plausible,
   not yet verified against real OSF results).
7. Implement `train_osf_lora.py` (§6), pilot at short context first (§6.3),
   then run Stage 2 (LoRA) across context lengths per the memory
   mitigation ladder (§6.2).
8. Compile results against `phase0_v3_full` (§0.2), applying the
   contamination caveats (§8) honestly.
9. Report back before starting PhysioOmni or MOMENT — do not start the
   next model's plan unprompted.

---

## 11. Open items not fully resolved (flagged, not blocking)

- ~~Exact checkpoint filename~~ **✅ RESOLVED 2026-08-10 — `osf_backbone.pth`,
  see §5.**
- **SHHS channel-completeness handling — new, code-verified 2026-08-10,
  needs a decision before `extract_osf_embeddings.py` is written.** See §1
  for the 4 options (duplicate generic EEG into both slots / zero-fill one
  slot / report with an explicit caveat / exclude SHHS entirely). Not
  something to decide unilaterally given it stacks on top of SHHS's
  existing contamination caveat.
- Whether LOC/ROC referencing exactly matches OSF's E1-A2/E2-A1 convention
  — still open.
- ~~Per-cohort Snore/EOG channel availability~~ **✅ largely resolved
  2026-08-10 — see §1's full per-cohort, per-slot table** (50 subjects/
  cohort). What remains open specifically: re-confirming these percentages
  against the *full* population once real extraction runs (the table is
  currently a 50-subject preview, not exhaustive), and the EOG-referencing
  question above (a different kind of check — reference electrode, not
  presence).
- Real wall-clock cost of Stage 2 (LoRA) training — no calibrated estimate
  exists yet, unlike Stage 1 which can reuse SleepFM-style estimation
  logic once its own pilot numbers are in.
- Whether `analyze`/`collect`/plotting code can be pointed at the new
  results directory unmodified, or needs its own fork — plausible per the
  Code Reuse Assessment (their code has zero hardcoded references to
  SleepFM's embedding shape, confirmed 2026-08-10) but not verified
  against real OSF results yet.
- **NEW — `ContextWindowDataset`'s `PATCHES_PER_EPOCH` constant** (§3.2):
  its exact role wasn't traced during this verification pass; check
  whether `OSFContextWindowDataset` needs an equivalent before assuming it
  doesn't.

---

## 12. VSCode debug workflow — a launch.json config per implementation step

**New section, added 2026-08-10** per explicit request: every script below
should get a debug config added to `/home/boshra95/.vscode/launch.json`
(the **workspace-root** launch.json — `~/.vscode/launch.json`, not a
per-repo one; **code-verified 2026-08-10: `NSRR-tools/.vscode/` itself has
never contained a `launch.json`** — the actual established pattern lives
at the home-directory workspace level, since it already contains configs
spanning multiple sibling repos/interpreters, e.g. `sleepfm_env` vs.
`NSRR-tools/.venv`) at the point that script is written, using a small
`--limit`/`--max-items`/`--cpu` debug run — mirroring the existing pattern
already used for every SleepFM-pipeline script (e.g. "Test: Extract
Channels (5 files)", "🎾 Phase0 Step4: Train Sweep DEBUG (apnea_binary,
lstm, CPU)"). **Do not implement any of these scripts yet** — this section
is the plan for the configs, to be added incrementally as each script is
actually written (§10's execution order), not all at once now.

**Interpreter/env-var convention to follow** (matches the existing
pattern's per-python-env grouping):

| Script | `python` | `PYTHONPATH` needed? |
|---|---|---|
| `extract_osf_embeddings.py` | `/home/boshra95/osf_env/bin/python` | Yes — `/home/boshra95/OSF-Open-Sleep-FM` (only script that imports OSF directly) |
| `test_osf_context_window_dataset.py` | `/home/boshra95/osf_env/bin/python` | No |
| `train_osf_context_sweep.py` | `/home/boshra95/osf_env/bin/python` | No (confirmed 2026-08-10: neither the SleepFM nor the OSF train/infer scripts import the backbone repo directly — only the extraction script does) |
| `infer_osf_subject_windows.py` | `/home/boshra95/osf_env/bin/python` | No |
| `train_osf_lora.py` | `/home/boshra95/osf_env/bin/python` | Yes — same reason as extraction (loads the raw ViT for the trainable graph) |

All configs need `"env": {"SCRATCH": "/scratch/boshra95", "HOME":
"/home/boshra95"}` (matches every existing entry) and `"justMyCode":
false` (needed to step into `nsrr_tools`/`osf` library code, not just the
top-level script).

**Planned configs, one per script, added when that script exists:**

1. **`extract_osf_embeddings.py`** — two configs mirroring the existing
   "👽 Phase0 Step1: Extract Embeddings" pair (CPU-debug + GPU-full):
   ```jsonc
   {
     "name": "🧬 OSF Step1: Extract Embeddings (5 subjects, CPU debug)",
     "type": "debugpy", "request": "launch",
     "program": "/home/boshra95/NSRR-tools/scripts/extract_osf_embeddings.py",
     "args": ["--config", "/home/boshra95/NSRR-tools/configs/phase0_osf_config.yaml",
               "--datasets", "apples", "--limit", "5", "--cpu"],
     "console": "integratedTerminal", "justMyCode": false,
     "python": "/home/boshra95/osf_env/bin/python",
     "cwd": "/home/boshra95/NSRR-tools",
     "env": {"SCRATCH": "/scratch/boshra95", "HOME": "/home/boshra95",
              "PYTHONPATH": "/home/boshra95/OSF-Open-Sleep-FM"}
   }
   ```
   This is the very first thing to debug once the script exists — it's
   also where §9 item 1's real-population channel-completeness re-check
   and item 2's EOG-referencing sanity check both happen in practice
   (inspect the saved `.npy` for NaNs / degenerate all-zero CLS vectors).
2. **`test_osf_context_window_dataset.py`** — a genuinely new debug script
   (there is no existing one to copy the launch.json entry from, since
   `test_context_window_dataset.py` itself predates the workspace
   `launch.json`'s current entries — see the correction below). Fork
   `scripts/test_context_window_dataset.py`'s CLI/structure (it's a solid,
   reusable template — instantiates the dataset for train/val/test,
   pulls one batch, asserts dtypes/shapes) with **one required
   edit**: the hardcoded `x.shape[-1] == 512` assertion must become
   `1536` for OSF. Two configs mirroring "🈂️ Phase0 Step2: Test
   ContextWindowDataset":
   ```jsonc
   {
     "name": "🧪 OSF Step2: Test OSFContextWindowDataset (apnea_binary, apples)",
     "type": "debugpy", "request": "launch",
     "program": "/home/boshra95/NSRR-tools/scripts/test_osf_context_window_dataset.py",
     "args": ["--config", "/home/boshra95/NSRR-tools/configs/phase0_osf_config.yaml",
               "--task", "apnea_binary", "--task-type", "seq2label",
               "--context", "30s", "10m", "--datasets", "apples"],
     "console": "integratedTerminal", "justMyCode": false,
     "python": "/home/boshra95/osf_env/bin/python",
     "cwd": "/home/boshra95/NSRR-tools"
   }
   ```
3. **`train_osf_context_sweep.py`** — mirrors "🎾 Phase0 Step4: Train Sweep
   DEBUG", using `--max-items`/`--cpu` for a fast smoke test:
   ```jsonc
   {
     "name": "🎯 OSF Step4: Train Sweep DEBUG (apnea_binary, lstm, CPU)",
     "type": "debugpy", "request": "launch",
     "program": "/home/boshra95/NSRR-tools/scripts/train_osf_context_sweep.py",
     "args": ["--config", "/home/boshra95/NSRR-tools/configs/phase0_osf_config.yaml",
               "--task", "apnea_binary", "--task-type", "seq2label", "--head", "lstm",
               "--context", "30s", "10m", "--datasets", "apples",
               "--max-items", "200", "--cpu"],
     "console": "integratedTerminal", "justMyCode": false,
     "python": "/home/boshra95/osf_env/bin/python",
     "cwd": "/home/boshra95/NSRR-tools"
   }
   ```
4. **`infer_osf_subject_windows.py`** — **new**, no existing SleepFM
   equivalent currently in `launch.json` to mirror (worth adding one for
   *both* pipelines while at it, since inference debugging has no config
   today even for the SleepFM path):
   ```jsonc
   {
     "name": "🎯 OSF Step5: Infer DEBUG (apnea_binary, lstm, one context, CPU)",
     "type": "debugpy", "request": "launch",
     "program": "/home/boshra95/NSRR-tools/scripts/infer_osf_subject_windows.py",
     "args": ["--config", "/home/boshra95/NSRR-tools/configs/phase0_osf_config.yaml",
               "--task", "apnea_binary", "--task-type", "seq2label", "--head", "lstm",
               "--context", "30s", "--datasets", "apples", "--split", "val", "--cpu"],
     "console": "integratedTerminal", "justMyCode": false,
     "python": "/home/boshra95/osf_env/bin/python",
     "cwd": "/home/boshra95/NSRR-tools"
   }
   ```
5. **`train_osf_lora.py`** — Stage 2's own pilot config (§6.3 already calls
   for a short pilot at the smallest context before the full sweep; this
   is that pilot, runnable under the debugger too):
   ```jsonc
   {
     "name": "🔬 OSF Step6: LoRA Pilot DEBUG (apnea_binary, 30s, few epochs)",
     "type": "debugpy", "request": "launch",
     "program": "/home/boshra95/NSRR-tools/scripts/train_osf_lora.py",
     "args": ["--config", "/home/boshra95/NSRR-tools/configs/phase0_osf_config.yaml",
               "--task", "apnea_binary", "--task-type", "seq2label", "--head", "lstm",
               "--context", "30s", "--datasets", "apples",
               "--max-items", "50", "--epochs", "2", "--cpu"],
     "console": "integratedTerminal", "justMyCode": false,
     "python": "/home/boshra95/osf_env/bin/python",
     "cwd": "/home/boshra95/NSRR-tools",
     "env": {"SCRATCH": "/scratch/boshra95", "HOME": "/home/boshra95",
              "PYTHONPATH": "/home/boshra95/OSF-Open-Sleep-FM"}
   }
   ```
   (`--epochs`/`--max-items` flag names here are provisional — `train_osf_lora.py`
   is genuinely new code, not a fork, so its actual CLI will be whatever
   is implemented; update this config to match once written, don't treat
   these flag names as settled.)

**Correction to a premise worth flagging explicitly**: earlier in this
planning process it was assumed `test_context_window_dataset.py` already
had an established `launch.json` debug-config precedent to point at inside
the NSRR-tools *git repo itself*. **Code-verified 2026-08-10: that's not
where the precedent lives** — `NSRR-tools/.vscode/launch.json` was briefly
added and then deleted early in the project (2026-02-23, well before
`test_context_window_dataset.py` existed) and was never re-created inside
the repo; the real, current, working set of ~30 debug configs lives at
`/home/boshra95/.vscode/launch.json` (workspace root), which does already
include working entries for `test_context_window_dataset.py`
("🈂️ Phase0 Step2: Test ContextWindowDataset..."). The plan above targets
that same file, correctly.
