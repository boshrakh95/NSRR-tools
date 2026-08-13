# OSF Channel Reprocessing Plan (deferred — not started)

**Status: investigation complete, fix NOT implemented, re-preprocessing NOT
run.** This doc exists so that **if OSF Stage 1 results look degraded for
MrOS, STAGES, or SHHS**, there's a concrete, ready-to-follow plan to come
back to instead of re-deriving this investigation from scratch. It was
written 2026-08-12 while the first full-population OSF embedding extraction
(checklist item 1.9) was already running against the *existing*
`psg_full/` HDF5s — that run was **not paused or cancelled** for this;
these are known, quantified gaps in the current embeddings, not blockers.

**Read this doc if:** OSF's frozen-encoder results for MrOS/STAGES/SHHS
come back meaningfully worse than APPLES's, and channel completeness is a
plausible explanation worth ruling in/out before concluding OSF just
performs worse on those cohorts.

**Do not read this doc if:** you're doing routine OSF implementation work
— see `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` and
`docs/OSF_EXPERIMENTS_GUIDE.md` instead.

---

## TL;DR

Three real, quantified gaps were found in `psg_full/`'s channel coverage,
all affecting channels OSF needs. All three are fixable **without touching
any file the existing SleepFM pipeline depends on** — the fix is to fork
(copy, not edit) the relevant config/code files into OSF-specific
versions, re-run preprocessing into a **new** output directory, and point
`configs/phase0_osf_config.yaml` at it. Nothing about SleepFM's existing
`psg_full/` HDF5s, results, or code changes.

| Cohort | OSF slot | Current coverage | Root cause | After fix (estimate) | Gain |
|---|---|---|---|---|---|
| MrOS | `ABD` | **0%** (0/3,933) | Code bug — see §1 | ~100% (pending validation) | **+100 pp** |
| STAGES | `EMG_LLeg` | 55%† / 58.2%‡ | Alias-list gap — see §2 | ~81.4%‡ | **+23.2 pp** |
| STAGES | `EMG_RLeg` | 55%† / 58.2%‡ | Alias-list gap — see §2 | ~81.4%‡ | **+23.2 pp** |
| SHHS | `NP` (Airflow) | 84.9%† / 77.0%‡ | Alias-list gap — see §3 | ~99.9%‡ | **+22.9 pp** |

† = measured from the real `_channel_fill_log.jsonl` files during the
2026-08-12 extraction run (716–936 subjects/dataset at time of writing).
‡ = computed directly from the raw per-subject channel-name CSVs in
`output/channel_analysis/` (full populations: 8,444 SHHS / 3,933 MrOS /
1,879 STAGES / 1,104 APPLES raw files), simulating the alias-matching logic
before vs. after the proposed fix. The two measurement methods differ
slightly (real HDF5 fill-rate vs. raw-alias-presence) — treat the raw-CSV
numbers as the best available *estimate* of the post-fix outcome, not a
guarantee; **validate with a small re-preprocessing smoke test before
trusting them for a full re-run** (same "smoke test before trusting"
practice used throughout this implementation).

APPLES needs no fix — none of the three gaps affect it.

---

## Why these weren't fixed immediately

All three fixes require **re-running the upstream EDF→HDF5 preprocessing**
(`scripts/preprocess_signals.py` / `src/nsrr_tools/core/signal_processor.py`)
for STAGES (1,879 subjects), SHHS (8,444 subjects), and MrOS (3,933
subjects) — a ~14,000-subject cluster job, not a change to
`extract_osf_embeddings.py`. That preprocessing code is **shared with the
existing, already-used SleepFM pipeline** (`phase0_v3`/`phase0_v3_full`,
whose results are already reported elsewhere). Editing it in place and
re-running it would risk changing SleepFM's own already-established HDF5s
and results — a much bigger, more consequential decision than anything
else done in the OSF implementation so far, and one that needs an explicit
decision to spend the re-preprocessing compute, not something to do
unilaterally mid-investigation.

---

## §1. MrOS `ABD` — confirmed code bug (not an alias-completeness gap)

### The evidence chain

1. **Raw data has it.** Loaded a raw MrOS EDF directly
   (`mros-visit1-aa0001.edf`) with `mne.io.read_raw_edf` — its 22-channel
   header includes `Abdominal`, with real (non-flat) signal data at 16 Hz
   native rate (`std≈0.00045`, not zero). Confirmed at scale: **100% of
   3,933 raw MrOS subjects** have a valid `ABD`-aliasable channel in their
   raw CSV inventory (`Abdominal` for visit-1-naming subjects, `ABD` for
   visit-2-naming subjects — two different recording-era naming
   conventions, both already listed as aliases).
2. **The config already wants it, correctly, with high priority.**
   `configs/channel_definitions.yaml`'s `ABD:` alias list already includes
   both `Abdominal` and `ABD`. `configs/modality_groups.yaml`'s `RESP`
   channel list already has `ABD` as Priority 3, well within the RESP
   modality's 7-channel cap (MrOS only ever has ~3-4 RESP-modality raw
   channels detected, nowhere near the cap).
3. **Ground truth: it's 0% in the actual HDF5s.** Sampled 200 random
   `psg_full/mros/derived/hdf5_signals/*.h5` files — zero have an `ABD`
   key. Traced two specific naming-era subjects
   (`mros-visit1-aa0001` → `AA0001_v1.h5`, and several
   `mros-visit2-*` subjects) directly — confirmed `ABD` is absent from
   their real HDF5 output despite point 1 above. Broadened to 100 subjects
   from each visit era: **0/100 both times.**
4. **Traced the actual pipeline code live**, calling the real classes in
   sequence (`ChannelMapper.detect_channels_from_list` →
   `ModalityDetector.group_channels_by_modality` →
   `SignalProcessor._apply_sleepfm_limits`) against the real raw channel
   list. **`ABD` survives all three stages correctly** — it's detected,
   correctly grouped into `RESP`, and correctly survives the channel-limit
   filtering (RESP only has 5 detected channels, cap is 7, nothing gets
   dropped).
5. **Found where it actually breaks**: `signal_processor.py`'s
   `process_edf()` (around line 162-169) tries to batch-preload only the
   selected channels via `raw.pick(list(channel_mapping.values()))`. For
   this file, `channel_mapping.values()` is:
   ```
   ['C3', 'C4', 'A1', 'A2', 'LOC', 'ROC', 'ECG L', 'ECG L', 'L Chin',
    'Leg L', 'Leg R', 'Thoracic', 'Abdominal', 'SaO2', 'Airflow', 'HR']
   ```
   Note **`'ECG L'` appears twice** — both the `EKG` canonical slot and
   the `ECG-L` canonical slot resolve to the same physical raw channel
   (this file only has one usable ECG lead by the time
   `_apply_sleepfm_limits`'s EKG-group priority selection picks
   `['EKG', 'ECG-L']` from `sleepfm_modalities.EKG.priority_order`, and
   both happen to have matched the same raw name for this file).
   `raw.pick()` throws:
   ```
   ValueError: Found 15 / 16 unique names, sel is not unique
   ```
   This is caught by a `try/except` that just logs a warning and "falls
   back to lazy loading" — **but empirically, calling `_process_channel()`
   on the *same* `raw` object afterward (even though `raw.ch_names` and
   `raw.preload` look unchanged) now raises `AssertionError` for channels
   that worked fine moments earlier on a freshly-loaded `raw` object.**
   Verified this directly, twice, reproducibly. Each of these per-channel
   failures is *also* individually caught-and-logged (`process_edf`'s
   per-channel loop, line ~188), so nothing crashes — the subject's HDF5
   just silently ends up missing whichever channels failed.

### One loose end (documented, not resolved)

The *exact* mechanism above (failed `.pick()` → some kind of internal MNE
state issue → `AssertionError` on subsequent channel access) is confirmed
**reproducible today**, but doesn't perfectly explain why the *real*,
already-generated `AA0001_v1.h5` has 13 channels including both `EKG` and
`ECG-L` (i.e., not the broad failure my live reproduction showed) while
still missing only `ABD`. Plausible explanations, not yet distinguished:
- `channel_definitions.yaml` may have been edited (aliases added) *after*
  the original `psg_full/mros` preprocessing run, meaning the EKG/ECG-L
  collision didn't exist at the time and `ABD` failed for a narrower,
  not-yet-identified reason specific to just that channel.
- MNE's `.pick()` failure-state behavior could be non-deterministic or
  version-sensitive.

**This doesn't change the recommended fix** (see §4) — deduplicating the
raw-name list before calling `.pick()` is safe and correct regardless of
which exact failure mode caused the historical drop, and it directly
addresses the reproducible collision found today.

---

## §2. STAGES `EMG_LLeg`/`EMG_RLeg` — alias-list gap (`LAT`/`RAT` not recognized)

`LAT`/`RAT` (Left/Right **A**nterior **T**ibialis — standard clinical PSG
terminology for the leg-EMG electrode site) appear in STAGES's raw channel
inventory at **23.2% each (435/1,879 subjects)** —
`output/channel_analysis/stages_channels.csv`, confirmed via direct
per-subject parsing. Grepped `configs/channel_definitions.yaml` precisely
(`^\s*-\s*(LAT|RAT)\s*$`) — **zero matches**. Neither `LLEG:` nor `RLEG:`'s
alias list includes them, despite both lists otherwise being fairly
thorough (dozens of other STAGES-specific naming variants like `Leg_1`,
`L-LEG1`, `LegsL-Leg1` etc. are already covered — this looks like a simple
oversight, not a deliberate exclusion, since `LAT`/`RAT` is unambiguous PSG
terminology for exactly this signal).

Simulated the fix directly against all 1,879 raw STAGES subjects: adding
`LAT` to `LLEG`'s alias list and `RAT` to `RLEG`'s raises raw-alias
coverage from **58.2% → 81.4%** for both sides — a clean +23.2 percentage
points, recovering exactly the 435 subjects who have `LAT`/`RAT` and
nothing else already-aliased. No collision risk (unlike §1 — `LAT`/`RAT`
don't share raw channel names with any other canonical slot).

---

## §3. SHHS `NP` (Airflow) — alias-list gap (`NEW AIR` family not recognized)

SHHS's raw channel inventory has an `AIRFLOW`/`AIRFLOW-0`/`AIRFLOW-1`
family (present in ~77% of subjects, `output/channel_analysis/shhs_channels.csv`)
**and separately** a `NEW AIR`/`NEWAIR`/`New A/F`/`New AIR`/`New Air`
family (present in up to ~62% combined) — these look like an
alternate/secondary nasal-flow sensor used across different SHHS
recording batches, not noise. `configs/channel_definitions.yaml`'s
`Airflow:` alias list is otherwise thorough (includes `Cannula Flow`,
`NASAL_PRESSURE`, `Flow`, `FLOW`, etc.) but does **not** include any
`NEW AIR` variant.

Simulated the fix against all 8,444 raw SHHS subjects: adding the 5
`NEW AIR` variants to `Airflow`'s alias list raises raw-alias coverage
from **77.0% → 99.9%** — a +22.9 percentage-point gain, recovering 1,931
subjects. (The real measured extraction fill-rate as of 2026-08-12 was
84.9%, slightly higher than the 77.0% raw-alias baseline — likely
explained by `AIRFLOW-0`/`AIRFLOW-1` or minor sample differences between
the CSV snapshot and the subjects actually extracted so far; the *gain
from the fix* should still be close to the 22.9 pp figure regardless of
which baseline it's measured from.)

---

## §4. SHHS `EEG(sec)` — open research question, NOT a recommended fix

Flagging for completeness, explicitly **not** part of the recommended fix
in §5 below.

Up to ~98% of raw SHHS subjects have *both* an `EEG` channel and an
`EEG(sec)` channel. Currently, `channel_definitions.yaml` treats
`EEG(sec)` as just another alias of the *same single* `EEG` canonical
slot — architecturally there is only one EEG channel to fill, so if both
are present in a file, only the first-matching one (`EEG`) gets used and
`EEG(sec)` is ignored entirely. If `EEG`/`EEG(sec)` actually correspond to
two *different* electrode sites (plausible — this is exactly the kind of
generic-secondary-channel naming a harmonized/de-identified NSRR release
might use to hide the original C3/C4 labels), this could in principle let
SHHS use **real, distinct C3/C4 channels** instead of the current
generic-`EEG`-duplicated-into-both-slots approximation already documented
in `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s Channel Mapping section.

**Why this isn't in the recommended fix:** unlike §2/§3 (simple alias
additions to an existing single-channel slot), this would require a
**schema change** — a new canonical channel (e.g. `EEG_sec`) distinct from
`EEG`, plus updated extraction logic to route it to `EEG_C4_A1` instead of
duplicating `EEG` into both slots. It's also not verified whether
`EEG`/`EEG(sec)` truly are different electrode sites, vs. e.g. a redundant
backup channel of the same site — that would need checking SHHS's own PSG
manual/documentation, or checking signal correlation/distinctness between
the two channels across several real files. **If pursued, do it as a
separate follow-up after §5's simpler fixes are validated**, not bundled
with them.

---

## §5. The fix: fork, don't edit — proposed file layout

Per instruction: none of the shared SleepFM-pipeline files get edited.
Everything OSF-specific lives in new, parallel files, mirroring how
`configs/phase0_osf_config.yaml`, `experiments/v2_osf_registry.yaml`, and
`scripts/gen_commands_osf.py` already sit alongside their SleepFM
counterparts elsewhere in this implementation.

### New files to create (none of these exist yet)

1. **`configs_osf_channels/channel_definitions.yaml`** — copy of
   `configs/channel_definitions.yaml`, plus:
   - Add `LAT` to the `LLEG:` alias list, `RAT` to the `RLEG:` alias list
     (§2).
   - Add `NEW AIR`, `NEWAIR`, `New A/F`, `New AIR`, `New Air` to the
     `Airflow:` alias list (§3).
2. **`configs_osf_channels/modality_groups.yaml`** — unchanged copy of
   `configs/modality_groups.yaml`. No edits needed (both `ABD` and
   `LLEG`/`RLEG` are already correctly declared with the right priority),
   but copied anyway so the OSF preprocessing config directory is fully
   self-contained and immune to any future edits to the shared version.
3. **`configs_osf_channels/preprocessing_params.yaml`** — unchanged copy
   of `configs/preprocessing_params.yaml`. This is the file
   `src/nsrr_tools/utils/config.py`'s `Config` class actually reads for
   `channel_selection.strategy`/`channel_limits` (see the gotcha below) —
   copied for self-containment, not because it needs edits.
4. **`configs/preprocessing_params_osf_full.yaml`** — copy of
   `configs/preprocessing_params_full.yaml`, with only
   `paths.base_output` changed from `/scratch/boshra95/psg_full` to a
   **new** directory, e.g. `/scratch/boshra95/psg_full_osf_channels`.
   This is the file passed via `--config` to control *where output gets
   written* — see the gotcha below for why this is a separate file from
   #1-3.
5. **`src/nsrr_tools/core/signal_processor_osf.py`** — copy of
   `src/nsrr_tools/core/signal_processor.py`, with one fix in
   `process_edf()`: deduplicate the raw-name list before calling
   `raw.pick()`, e.g.
   ```python
   pick_names = list(dict.fromkeys(channel_mapping.values()))  # dedupe, preserve order
   raw.pick(pick_names)
   ```
   This directly fixes the `ValueError('Found N / M unique names, sel is
   not unique')` from §1 — picking a channel once is sufficient even when
   two canonical slots (e.g. `EKG` and `ECG-L`) alias to the same physical
   channel; both slots will just read the same already-loaded data
   afterward, which is correct (they *are* the same signal).
6. **`scripts/preprocess_signals_osf.py`** — copy of
   `scripts/preprocess_signals.py`, with two changes:
   - Import `SignalProcessor` from `signal_processor_osf.py` instead of
     `signal_processor.py`.
   - Instantiate `Config(config_dir=Path(__file__).parent.parent /
     "configs_osf_channels")` instead of bare `Config()`.

### The `Config()`-loading gotcha this design works around

`src/nsrr_tools/utils/config.py`'s `Config.__init__` **hardcodes** the
three filenames it loads (`channel_definitions.yaml`, `modality_groups.yaml`,
`preprocessing_params.yaml`) from whatever `config_dir` is passed in (default:
`configs/`). Separately, `preprocess_signals.py`'s `--config` CLI flag
loads a *different* config object (`self.preprocess_config`, used only for
output *paths*) — **not** the one `SignalProcessor` reads for channel
selection. This means **`configs/preprocessing_params_full.yaml` is
currently NOT used for its `channel_selection.strategy` field at all** —
only for path routing — which is why file #4 above only needs its
`base_output` path changed, and why files #1-3 need to live in a
dedicated `config_dir` rather than being passed via `--config`. (This
double-config-object gotcha is itself a pre-existing quirk of the shared
codebase, documented here for whoever implements this fix — not something
introduced by OSF work, and not being fixed in the shared code per the
fork-don't-edit instruction.)

### Execution steps (checklist form, for whoever picks this up)

- [ ] Create the 6 new files above.
- [ ] **Smoke-test on a handful of subjects first** (mirrors this whole
      implementation's established practice): run
      `scripts/preprocess_signals_osf.py --dataset mros --max-subjects 5
      --config configs/preprocessing_params_osf_full.yaml`, then check the
      output HDF5s have `ABD` where expected. Do the same for `stages`
      (check `LLEG`/`RLEG` on subjects known to have `LAT`/`RAT`) and
      `shhs` (check `Airflow` on subjects known to only have `NEW AIR`).
- [ ] If smoke tests pass, run the full re-preprocessing for `mros`
      (3,933 subjects), `stages` (1,879), and `shhs` (8,444) into
      `/scratch/boshra95/psg_full_osf_channels/`. **Do not touch
      `apples`** — none of the three gaps affect it, no need to
      reprocess. **This is a real multi-day-scale cluster undertaking at
      SHHS's size — budget accordingly and consider sharding, same as
      the embedding-extraction jobs.**
- [ ] Re-run the real per-subject channel-fill audit (same method as
      `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s Known Issues entry, i.e.
      aggregate `_channel_fill_log.jsonl`) against the **new** HDF5s to
      confirm the actual gain matches the §1-3 estimates before trusting
      them.
- [ ] Update `configs/phase0_osf_config.yaml`'s `data.hdf5_dir` (and
      `embeddings.output_dir`/`dataset.embedding_dir`, since those should
      also point to a fresh embeddings directory, e.g.
      `.../unified/embeddings/osf_30sec_v2/`, to avoid mixing
      old-channel-coverage and new-channel-coverage embeddings under the
      same directory) — or create `configs/phase0_osf_config_v2.yaml` if
      keeping the old config/results around for comparison is useful.
- [ ] Re-run `scripts/extract_osf_embeddings.py` /
      `jobs/extract_osf_embeddings_gpu.sh` (unchanged — no code changes
      needed here, just pointed at the new HDF5s via the updated config)
      for the affected datasets.
- [ ] Re-run Stage 1 training/inference for the affected tasks/cohorts and
      compare against the original results to see whether channel
      completeness was actually a meaningful factor.

### What this does *not* touch

- `configs/channel_definitions.yaml`, `configs/modality_groups.yaml`,
  `configs/preprocessing_params.yaml`, `configs/preprocessing_params_full.yaml`
  — all unchanged.
- `src/nsrr_tools/core/signal_processor.py`, `channel_mapper.py`,
  `modality_detector.py` — all unchanged.
- `scripts/preprocess_signals.py` — unchanged.
- `/scratch/boshra95/psg_full/` (existing SleepFM HDF5s) — untouched;
  the new output lands in a sibling directory.
- Any existing SleepFM results (`phase0_v3`, `phase0_v3_full`) — fully
  isolated from this work, same as the rest of the OSF implementation
  (see the path-isolation audit in the implementation plan's history).

---

## Cross-references

- `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` — Known Issues section links
  here; see also its Channel Mapping section for the original 50-subject
  audit this investigation was cross-checked against.
- `output/channel_analysis/` — the raw per-subject channel CSVs this
  investigation's numbers are computed from (`apples_channels.csv`,
  `shhs_channels.csv`, `mros_channels.csv`, `stages_channels.csv`,
  `CHANNEL_EXTRACTION_SUMMARY.md`). Note the summary `.md` in that
  directory understates the real sample sizes (says "5-10 files/dataset,
  Feb 2026") — the actual CSVs cover the **full populations**
  (1,104/8,444/3,933/1,879 subjects) as of whenever they were last
  regenerated; trust the CSVs' row counts over the summary doc's prose.
