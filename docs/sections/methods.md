# Methods Section — Extended Reference Document
*Status: DRAFT — ready for user review. Questions marked **[QUESTION]**.*
*Corresponding LaTeX section: `generic-color.tex` → `\section{Methods}`*
*Last updated: 2026-06-03 (major revision: corrected sleep staging context mode to centered; added STAGES exclusion, REM remapping, v2 K-bug fix, cohort filter per-task numbers, sleep staging architecture details)*

---

## How to use this file

Every sub-section below contains more detail than the final paper will print.
The paper text (10-page TBME budget) will pull from this selectively.
Everything here is also the source for supplementary material.
Do NOT edit the LaTeX before this file is approved.

---

## III-A. Datasets

### Overview

We use four publicly available PSG cohorts from the National Sleep Research Resource (NSRR), selected to maximise task label diversity and demographic breadth.

**[QUESTION 1]** Please confirm the correct full citation/access URL for each NSRR dataset. The NSRR homepage is https://sleepdata.org. Each dataset has a DOI — do you want to cite the dataset directly (DOI) or cite the original published paper? Typically TBME reviewers expect both.

### SHHS — Sleep Heart Health Study

- **Study design:** Large, community-based, prospective epidemiological cohort. Adults recruited from existing cardiovascular cohorts; no sleep-disorder enrichment.
- **Visits:** 2 (SHHS1 and SHHS2). SHHS1: ~6,441 subjects; SHHS2: ~3,295 subjects (subset who returned ~5 years later). We treat each visit independently (different PSG recording, same subject may contribute to both).
- **Age range:** 40–98 years at enrolment. Mean ~63 years.
- **Sex:** ~53% female, ~47% male.
- **PSG equipment:** In-home unattended PSG (Compumedics devices). 2-channel EEG (C3/A2, C4/A1), 2-channel EOG, chin EMG, ECG, airflow (thermistor), thoracic and abdominal respiratory effort (RIP belts), SpO2, body position.
- **Sleep scoring:** Centralized human scoring at Brigham and Women's Hospital, using Rechtschaffen & Kales (R&K) criteria for SHHS1 and AASM 2007 for SHHS2. **[QUESTION 2]** We currently pool SHHS1 and SHHS2. SHHS1 uses R&K scoring (4 NREM stages: S1, S2, S3, S4) which we map to 5-class AASM staging (W, N1, N2, N3, REM) following the standard mapping (S3+S4 → N3). This mapping is imperfect. Should we mention this in the paper, or exclude SHHS1 from sleep staging?
- **Labels available:** AHI (AHI ≥ 5 per hour), ESS (Epworth Sleepiness Scale), cardiovascular disease history (any_cvd composite), BMI, age, sex.
- **Usable N after preprocessing:** ~[PENDING — to be filled from actual parquet] subjects with valid PSG and ≥1 task label.
- **Key exclusions:** Subjects with recording length < 240 min (cohort consistency filter); subjects with NaN embeddings (none in SHHS per nan_blocklist).
- **NSRR access:** https://sleepdata.org/datasets/shhs

### MrOS — Osteoporotic Fractures in Men Sleep Study

- **Study design:** Prospective cohort of community-dwelling men aged ≥65. Primary study objective: risk factors for osteoporotic fractures. Sleep study was ancillary.
- **Visits:** 2 (Visit 1 and Visit 2, ~3.4 years apart). Visit 1: ~2,911 subjects; Visit 2: ~2,560 subjects. Each visit contributes an independent PSG recording.
- **Age range:** 65–90+ years. Mean ~76 years. **Entirely male cohort** — MrOS is excluded from the sex classification task.
- **PSG equipment:** In-home unattended PSG (VITAPORT-3 device). 2-channel EEG (C3/A2, C4/A1), 2-channel EOG, chin EMG, ECG, airflow (thermistor), thoracic and abdominal RIP effort, SpO2.
- **Sleep scoring:** Centralized scoring using AASM 2007 criteria.
- **Labels available:** AHI, PSQI (Pittsburgh Sleep Quality Index; PSQI > 5 = poor sleep), cardiovascular disease (cvchd = coronary heart disease only — narrower than SHHS's any_cvd composite), BMI, age.
- **Multi-visit handling:** For task labels that are visit-specific (PSQI), both visits contribute independently. For BMI and AHI, Visit 1 labels are primary; Visit 2 used if Visit 1 is missing. **[QUESTION 3]** Confirm this is correct — does MrOS Visit 2 have BMI and AHI available? The memory notes say "MrOS: visit-1 labels only" for BMI. Please confirm.
- **Usable N:** ~[PENDING] subjects across both visits.
- **Key exclusions:** Recording < 240 min (cohort filter); all-male cohort noted for sex task.
- **NSRR access:** https://sleepdata.org/datasets/mros

### APPLES — Apnea Positive Pressure Long-term Efficacy Study

- **Study design:** Randomised controlled trial of CPAP therapy for OSA. OSA-enriched sample (required clinical PSG-confirmed OSA for enrolment). Rich baseline questionnaire and cognitive battery.
- **Visits:** Single baseline visit (pre-CPAP randomisation). Some subjects have follow-up PSG at 6 months — we use baseline only.
- **Age range:** 18–85 years. Mean ~52 years.
- **Sex:** ~75% male (consistent with OSA clinical enrichment).
- **PSG equipment:** In-lab attended PSG (varies by site). Full montage typically available: multiple EEG channels, 2-channel EOG, chin EMG, ECG, airflow (thermistor + nasal pressure), thoracic + abdominal RIP, SpO2, leg EMG.
- **Sleep scoring:** AASM 2007 criteria.
- **Labels available:** AHI (clinician-adjudicated, post-QC; variable "ahi_ahi30" for AHI per 30-min window), BDI (Beck Depression Inventory; BDI ≥ 14 = depression threshold; we use extreme group: BDI ≤ 9 vs ≥ 20), ESS, BMI, age, sex.
- **Important caveat:** 412 of 1,516 enrolled subjects have no PSG file on NSRR (withdrawn consent or failed recordings). Actual usable N ≈ 1,103 subjects with valid embeddings.
- **Key exclusions:** Same 240-min filter; 1 subject with NaN embeddings (in nan_blocklist).
- **NSRR access:** https://sleepdata.org/datasets/apples

### STAGES — Stanford Technology Analytics and Genomics in Sleep

- **Study design:** Multi-site clinical and community sleep cohort. Subjects recruited from sleep clinic patients and the general community across multiple Stanford-affiliated sites (Santa Cruz, Palo Alto, etc.). Explicitly designed for ML research on PSG.
- **Visits:** Single visit per subject.
- **Age range:** Wide (18–90+).
- **Sex:** ~50/50.
- **PSG equipment:** In-lab attended PSG. Full montage. EDF files include multiple EEG channels, EOG, chin EMG, ECG, respiratory channels.
- **Sleep scoring:** AASM 2012 criteria (the most current at time of collection).
- **Labels available:** AHI (from XLSX: `STAGESPSGKeySRBDVariables2020-08-29 Deidentified.xlsx`, column `ahi`, subject ID `s_code`), PHQ-9 (Patient Health Questionnaire-9; PHQ-9 ≥ 10 = moderate-severe depression; extreme group ≤4 vs ≥15), GAD-7 (Generalised Anxiety Disorder-7; GAD-7 ≥ 10 = moderate-severe anxiety), ISI (Insomnia Severity Index; ISI ≥ 15 = clinical insomnia), ESS (ess_0900 column; ESS ≥ 11 = sleepiness), sex, age.
- **Unique features:** STAGES has the richest psychiatric questionnaire coverage of all four cohorts (PHQ-9, GAD-7, ISI in the same dataset). It is the only dataset contributing to depression_extreme, anxiety_binary, and insomnia_binary tasks.
- **Usable N:** ~1,500 subjects. 152 subjects with NaN embeddings excluded via nan_blocklist (STLK cohort — suspected signal processing issue).
- **Key exclusions:** NaN blocklist (152); 240-min cohort filter.
- **NSRR access:** https://sleepdata.org/datasets/stages

### Cohort Summary Table (for paper)

| Dataset | Population | N (approx) | Age | Sex | PSG type | Scoring | Tasks |
|---|---|---|---|---|---|---|---|
| SHHS | Community, general | ~9,000 (v1+v2) | 40–98 | 53% F | In-home | R&K / AASM 2007 | AHI, ESS, CVD, BMI, age, sex, staging |
| MrOS | Community, elderly men | ~5,400 (v1+v2) | 65–90+ | 0% F | In-home | AASM 2007 | AHI, PSQI, CVD, BMI, age, staging |
| APPLES | OSA clinical trial | ~1,103 usable | 18–85 | 25% F | In-lab | AASM 2007 | AHI, BDI, ESS, BMI, age, sex, staging |
| STAGES | Multi-site clinical/community | ~1,350 usable | 18–90+ | ~50% F | In-lab | AASM 2012 | AHI, PHQ-9, GAD-7, ISI, ESS, sex, age, staging |

**[QUESTION 4]** Can you confirm the exact numbers once the master_targets.parquet is finalised? The N values above are approximations. For the paper table, we need the exact per-task N (post all filters) for each dataset. This comes from `targets_v2/task_subjects/` — one CSV per task.

### Dataset splits

- 70% train / 15% validation / 15% test, stratified at the **subject level** (not window level)
- Split seed = 42, applied identically across all tasks and context lengths
- Subjects with recordings < 240 minutes are excluded from all context lengths (cohort consistency filter, see §III-D)

---

## III-B. Clinical Task Definitions

All tasks are framed as classification. Labels are derived from validated instruments or PSG-derived metrics — no raw physiological signals are used as labels.

### Tier 1 tasks (all three heads, all 6 context lengths)

| Task | Type | N (all datasets) | Label source | Threshold | Notes |
|---|---|---|---|---|---|
| `sex_binary` | 2-class | ~13,000 | Self-report | Female=1, Male=0 | MrOS excluded (all-male) |
| `sleep_efficiency_binary` | 2-class | ~13,600 | PSG-derived TST/TIB | < 0.85 = poor (1) | STAGES excluded (no SE score) |
| `bmi_binary` | 2-class | ~15,500 | Self-report / clinical measurement | ≥ 30 kg/m² = obese (1) | WHO definition |
| `age_class` | 3-class | ~16,000 | Self-report | <50=0, 50–64=1, ≥65=2 | MrOS subjects all class 2 |
| `apnea_binary` | 2-class | ~14,100 | PSG-derived AHI | ≥ 15 events/hr = mod-severe OSA | Standard clinical threshold |
| `sleep_staging` | 5-class (seq2seq) | ~15,000 epochs × subjects | PSG-scored | W=0, N1=1, N2=2, N3=3, REM=4 | Primary metric: Cohen's κ |

### Tier 2 tasks (LSTM head only, initially)

| Task | Type | N | Label source | Threshold | Notes |
|---|---|---|---|---|---|
| `psqi_binary` | 2-class | ~3,900 | PSQI questionnaire | > 5 = poor sleep quality | MrOS only |
| `depression_extreme_binary` | 2-class | ~1,760 | BDI (APPLES), PHQ-9 (STAGES) | APPLES: BDI ≤9→0, ≥20→1; STAGES: PHQ-9 ≤4→0, ≥15→1 | Extreme-group design; middle excluded |
| `osa_binary_apples_postqc` | 2-class | ~1,516 | Clinician-adjudicated AHI | Non-rand+Mild→0, Mod+Severe→1 | APPLES only; more conservative than AHI ≥15 |
| `osa_severity_apples` | 4-class | ~1,516 | Clinician-adjudicated AHI | <5=0, 5–15=1, 15–30=2, ≥30=3 | APPLES only |
| `cvd_binary` | 2-class | ~13,000 | Medical history | SHHS: any_cvd (composite); MrOS: cvchd (CHD only) | **Mixed definitions — note in paper** |
| `sleepiness_binary` | 2-class | ~16,400 | ESS questionnaire | ≥ 11 = excessive daytime sleepiness | All 4 datasets |

### Deferred tasks (run last; poor phase0 signal)

| Task | Type | N | Threshold | Phase0 AUROC |
|---|---|---|---|---|
| `insomnia_binary` | 2-class | ~1,710 | ISI ≥ 15 | 0.56–0.60 |
| `anxiety_binary` | 2-class | ~1,698 | GAD-7 ≥ 10 | 0.56–0.58 |
| `rested_morning` | 2-class | ~3,934 | Single-item questionnaire | 0.52–0.54 (near chance) |

**[QUESTION 5]** For the CVD task: should we note the definition mismatch (SHHS any_cvd composite vs MrOS cvchd coronary only) as a limitation in the Methods section itself, or only in Discussion? I lean toward a footnote in the task table (Methods) and a fuller discussion in Discussion.

**[QUESTION 6]** For the extreme-group depression design: the paper should explain why we use extreme groups rather than the full BDI/PHQ-9 range. The argument is that the middle range introduces label noise (subthreshold depression is clinically ambiguous). Is this your reasoning, or is there another motivation (e.g., small class sizes without extreme groups)?

---

## III-C. Signal Preprocessing

### Pipeline overview

Raw EDF files → standardised HDF5 → SleepFM embeddings (.npy)

This pipeline runs once per subject and per dataset. The outputs are cached; all downstream training reads from the cache.

### Step 1: EDF → HDF5 (`preprocess_signals.py`)

**Tool:** MNE-Python for EDF loading.

**Channel selection:**
1. Each EDF is scanned for available channel names.
2. Channels are mapped to standardised names via `ChannelMapper` (handles dataset-specific naming conventions, e.g., SHHS uses "C3-M2", STAGES may use "EEG C3-A2").
3. For each modality group, channels are selected by priority order (defined in `configs/phase0_v3_config.yaml`):
   - BAS (EEG + EOG): C3-M2, C4-M1, LOC, ROC, O1-M2, O2-M1, F3-M2, F4-M1, A1, A2 (up to 10)
   - RESP: Airflow, Thor, ABD, SpO2, HR, Snore, RespRate (up to 7)
   - EKG: EKG, ECG-L, ECG-R (up to 2)
   - EMG: CHIN, LLEG, RLEG, EMG (up to 4)
4. If a channel is unavailable in a recording, its slot is zero-padded and a boolean mask records its absence (mask=True → padded/missing).

**Signal processing (per channel):**
1. **Bandpass filtering** (Butterworth, 4th order, MNE FFT-based for long signals):
   - EEG: 0.3–35 Hz
   - EOG: 0.3–35 Hz
   - ECG/EKG: 0.5–45 Hz
   - EMG (chin/leg): 10–100 Hz
   - Respiratory (all RESP channels): 0.05–2.0 Hz
2. **Resampling to 128 Hz** (SleepFM requirement). Done via `resample_poly` (rational ratio) when feasible, scipy `interp1d` fallback.
3. **Per-channel z-score normalization**: subtract per-recording mean, divide by per-recording standard deviation.

**Output format:**
- One `.h5` file per subject: `{dataset}/derived/hdf5_signals/{subject_id}.h5`
- Each HDF5 key = standardised channel name; value = float16 array of shape `(n_samples,)` at 128 Hz
- Compression: gzip level 4

**[QUESTION 7]** Were any additional artifact rejection steps applied (e.g., amplitude clipping, epoch-level rejection, EMG interference detection)? The signal_processor.py code I read only shows filtering, resampling, and z-score. If no artifact rejection was applied, we should acknowledge this in the Methods as a limitation ("we apply no artifact rejection beyond bandpass filtering; SleepFM's tokenizer and masking are relied upon to handle residual artifacts").

**[QUESTION 8]** Were all four datasets successfully preprocessed and embedded? Any datasets with unusual channel naming that required special handling beyond the standard mapper?

### Step 2: HDF5 → SleepFM patch embeddings (`extract_sleepfm_embeddings.py`)

**Model used:** SleepFM `model_base` checkpoint (`sleepfm-clinical/sleepfm/checkpoints/model_base/`). Frozen (no gradient updates, no fine-tuning).

**Processing unit:** 5-minute chunks (38,400 samples at 128 Hz). The recording is divided into non-overlapping 5-minute chunks. Any trailing signal shorter than 5 minutes is discarded.

**Per-chunk processing (per modality group):**
1. Build tensor of shape `(B, C_max, 38400)` where B = chunk_batch_size (16), C_max = max channels for modality.
   - Each of the C_actual present channels fills one row; remaining C_max − C_actual rows are zeros.
2. Run frozen SetTransformer forward pass: `model(x, mask)` → returns `(pooled, patch_embeddings)`.
   - `mask` is boolean shape `(B, C_max)`, True for zero-padded (absent) channels.
   - We use `patch_embeddings` (the second return value): shape `(B, 60, 128)`.
     - 60 = patches per chunk = 38,400 / 640 samples/patch
     - 128 = SleepFM embed_dim per modality
3. Reshape and store into output array: `out[patch_start:patch_end, mi, :] = patch_embeddings.reshape(-1, 128)`.

**SleepFM architecture brief (for paper):**
- **Tokenizer**: 6-stage Conv1D cascade (Conv1d → BN → ELU → LayerNorm, stride=2 at each stage) → AdaptiveAvgPool1d → Linear. Input: 640 samples (5 sec at 128 Hz) per channel. Output: 128-dim vector per channel per patch.
- **Spatial pooling (SetTransformer)**: for each 5-sec patch, a Transformer encoder layer with AttentionPooling aggregates across all channels within a modality group → one 128-dim vector per modality per patch.
- **Temporal encoder**: sinusoidal positional encoding → Transformer encoder → temporal AttentionPooling → 128-dim modality-level representation (used during pre-training; we discard this and use only the per-patch embeddings `patch_embeddings`).
- **Pre-training**: contrastive learning with a leave-one-out objective among modality groups — each modality's embedding must predict the others. Trained on ~100,000+ hours of PSG.

**Output per subject:** `.npy` file, shape `[T, 4, 128]`, dtype float16.
- T = total 5-second patches = (n_full_5min_chunks × 60)
- 4 = modalities in order: BAS, RESP, EKG, EMG
- 128 = embed_dim per modality

**Storage:** ~5.6 MB per subject (float16). ~22 GB total across ~4,000 subjects.

**At training time:** The `[T, 4, 128]` array is memory-mapped (never fully loaded). Context windows are sliced as `emb[t_start:t_start+N, :, :]` then flattened to `[N, 512]` (= 4 × 128) as input to the downstream head.

**[QUESTION 9]** What was the total GPU compute time for embedding extraction across all 4 datasets? (Approximate, for Methods or supplementary — reviewers of TBME often ask about compute costs.)

**[QUESTION 10]** How many subjects had to be discarded due to being too short to form even 1 full 5-minute chunk? (These would have been excluded before training, before the 240-min cohort filter.)

---

## III-D. Context-Window Dataset

### Input representation

After embedding extraction, each subject is represented as a matrix `E ∈ ℝ^{T × 512}` where T is the number of 5-second patches and each row is the flattened 4-modality embedding.

For a context length L (in minutes), the corresponding number of patches is:
```
N = L × 60 / 5 = L × 12
```
Specifically: 30s → 6, 10m → 120, 40m → 480, 80m → 960, 120m → 1440, 240m → 2880 patches.

### seq2label mode (all clinical prediction tasks)

- **Index unit:** (subject, window_start_patch_index)
- **Input:** `E[window_start : window_start + N, :]` — N consecutive patches
- **Label:** scalar (same for all windows of the same subject; night-level label)
- **Padding:** if `window_start + N > T` (window extends past the end), trailing positions are zero-padded and flagged mask=True. In practice, rare after the 240-min cohort filter.

**Training window sampling (K_train = 5) — OVERLAPPING POOL:**
- Draw 5 start positions uniformly at random from ALL valid positions `[0, T − N]` (any start in the range; overlapping windows are explicitly allowed)
- n_valid = T − N + 1 valid positions. For an 8h night at 240m context (N=2880): n_valid = 960 − 2880 + 1 = 481 → K=5 is achievable.
- Positions re-drawn each epoch (different random subset each time)
- **Why overlapping matters:** The v2 protocol used NON-overlapping windows only (stride-N positions), giving n_windows = T // N. For a 240m context on an 8h night: only 2 non-overlapping windows exist → K=min(5,2)=2. This meant the 240m model trained on 40% fewer gradient steps per epoch than the 30s model — a hidden training asymmetry. The v3 overlapping-pool fix ensures K=5 at ALL context lengths, making the gradient update count identical. This was the most important protocol fix before paper submission. (See `docs/TRAINING_PROTOCOL_FIXES.md` Issue 1 for full analysis.)

**Validation/test window sampling during training (K_val = 5) — also overlapping pool:**
- 5 evenly-spaced positions across `[0, T − N]`, deterministic (same each epoch)
- Also uses the overlapping pool: `K=5` is achievable at 240m (vs only 2 with the old stride-N val pool)
- Provides a stable early-stopping signal at ALL context lengths

**Inference (K_infer = T // N) — NON-overlapping, stride-N:**
- K_max = 99,999 is passed to the dataset → triggers the non-overlapping stride-N path
- All non-overlapping stride-N windows: positions `{0, N, 2N, ..., ⌊T/N⌋·N}`
- This ensures systematic, non-redundant coverage of the full night
- Per-window predictions saved as parquet; subject-level AUROC computed by aggregating

**Shape cache:** a JSON shape cache (`{embedding_dir}/shape_cache.json`) stores T (recording length in patches) for each subject. Built on first run, loaded on subsequent runs. This avoids loading the full float16 array just to determine recording length during dataset index construction.

### seq2seq mode (sleep staging only)

**Context mode — CENTERED (NOT causal):**
This is the active design decision as of v3 (`seq2seq_context_mode: "centered"`, `seq2seq_padding_policy: "complete_only"` in `configs/phase0_v3_config.yaml`). Previous documentation erroneously described it as "causal/past-only" — that was the *archived* v1 design (see `sleep_staging_lstm_old_arch128`). The current implementation is symmetric/centered.

- **Index unit:** (subject, anchor_epoch_index) where anchor = one 6-patch (30-sec) epoch
- **Context window composition (centered):**
  - Let `half_past = (N − 6) // 2` and `half_future = N − 6 − half_past`
  - Input = `E[anchor_start − half_past : anchor_end + half_future, :]` — total N patches
  - The anchor's 6 patches are centred in the window; `half_past` patches precede and `half_future` patches follow
  - This is symmetric: both past and future sleep structure inform the prediction
- **Label:** sleep stage of the anchor epoch remapped to 5-class (see REM remapping below)
- **Padding policy (`complete_only`):** only anchors where the full N-patch symmetric window fits entirely within the recording are included. This means anchors in the first `half_past` patches and the last `half_future` patches of each recording are excluded from both training and evaluation. At 240m context, this excludes approximately the first and last ~120 minutes of each recording (~50% excluded).
- **Why complete_only:** avoids zero-padded batches → Flash attention fires at all context lengths; also avoids the confound of comparing models trained on different fractions of padded signal.
- **Training vs causal:** the centered design is used because (a) human sleep scorers also use bidirectional context and (b) the research question is "does more context about the night improve staging?" not "can you stage in real time?" The causal design is reserved as a sensitivity analysis (see `sleep_staging_design.md` §2).

**REM stage remapping:**
NSRR raw annotations encode REM as stage 5. The code remaps: stage 5 → stage 4, producing the 5-class scheme W=0, N1=1, N2=2, N3=3, REM=4. This remapping is applied in `_remap_stages()` in `context_window_dataset.py` and is done once when annotations are loaded.

**STAGES excluded from sleep staging:**
The STAGES dataset is NOT used for sleep staging despite being used for all other tasks. Reason (documented in `sleep_staging_design.md` §10): STAGES subjects have ~847 epochs/subject vs ~80 for SHHS/MrOS/APPLES, so STAGES alone contributes ~54% of all training items despite being only 10% of subjects. This causes the model to be trained primarily on STAGES scoring conventions and generalises poorly. Final sleep staging cohort: SHHS + MrOS + APPLES only.

**Subject-level aggregation:** each anchor epoch has its own label. There is no subject-level majority vote or mean-probability aggregation for staging. Cohen's κ and per-stage F1 are computed across all test anchor epochs.

**Common evaluation set issue:** Different context lengths evaluate on different anchor subsets (longer contexts exclude more edge epochs). This means kappa at 30s and 240m are computed on slightly different epoch populations — 30s includes all epochs, 240m excludes the first/last ~120 min per recording. The bias direction: short contexts include harder sleep-onset N1 epochs → kappa is slightly deflated at short contexts, making the context-length benefit an *underestimate*. A common-evaluation-set supplementary analysis (restricting all models to anchors valid at 240m) is planned to verify that the qualitative trend is unchanged (`analyze_common_eval_set.py`).

**Architecture exception:** for sleep staging, the LSTMHead uses `hidden_dim=256, num_layers=2` (~3.16M params) instead of the 128/1 configuration used for all binary/multiclass tasks. Motivation: the 5-class seq2seq task requires substantially more model capacity; prior work (phase0 runs) showed kappa dropped from ~0.62 to ~0.54 with the smaller architecture. See §III-E for full details.

### Cohort consistency filter

Subjects with total recording T < 2,880 patches (< 240 minutes) are excluded from **all** context lengths.

**Motivation:** without this filter, as L increases, shorter-recording subjects are progressively dropped (they cannot provide any valid windows at long L). This would mean the set of subjects changes as L grows, confounding context-length comparisons with population differences (shorter recordings may belong to a specific demographic or have different disease severity).

**Implementation:** `dataset.min_recording_patches = 2880` in `configs/phase0_v3_config.yaml`. Applied before split assignment — the same subjects are excluded from train/val/test at every L.

**Impact (from `docs/cohort_filter.md`):**

| Task | Total subjects | Excluded | % lost |
|---|---|---|---|
| sex_binary (APPLES+SHHS) | 9,547 | 20 | 0.21% |
| sleep_efficiency_binary (APPLES+SHHS+MrOS) | 13,480 | 20 | 0.15% |
| bmi_binary (APPLES+SHHS+MrOS) | 12,385 | 20 | 0.16% |
| age_class (APPLES+SHHS+MrOS) | 12,410 | 20 | 0.16% |
| psqi_binary (MrOS only) | 3,929 | 0 | 0% |
| depression_extreme_binary (APPLES) | 874 | 15 | 1.72% |
| osa_binary_apples_postqc (APPLES) | 1,103 | 19 | 1.72% |
| osa_severity_apples (APPLES) | 1,103 | 19 | 1.72% |

The 19–20 excluded subjects are from APPLES (recordings ranging from 5 to 230 minutes — clearly truncated acquisitions rather than full overnight studies) and 1 from SHHS (180-min recording). Full list: `docs/excluded_subjects_T_lt_2880.csv`.

**Paper language (from cohort_filter.md):**
> "To ensure a fair comparison across context lengths, subjects whose full-night PSG recording was shorter than the longest context window (240 min, 2,880 × 5-second patches) were excluded from all context lengths. This affected 20 of 9,547–13,480 subjects (≤ 0.2%) for Tier 1 tasks and 15–19 of 874–1,103 subjects (≤ 1.7%) for Tier 2 tasks. The excluded recordings ranged from 5 to 230 minutes and appear to be truncated acquisitions rather than full-night studies."

**Secondary benefit:** eliminates CUDA OOM at L=240m for the Transformer head. Short recordings produce partially-padded batches; even a single all-zeros mask forces PyTorch to use O(N²) Math attention instead of O(N) Flash attention. With the filter, all masks are all-False, and Flash attention is used throughout.

---

## III-E. Downstream Sequence Heads

Three architectures are compared, ranging from a parameter-free temporal baseline to a full self-attention model. All receive `x ∈ ℝ^{B × N × 512}` and a boolean padding mask `mask ∈ {True,False}^{B × N}` (True = padded position, excluded from computation).

### MeanPool (baseline)

```
x_valid = x * (~mask).unsqueeze(-1).float()          # zero out padded
pooled  = x_valid.sum(1) / (~mask).float().sum(1, keepdim=True)  # masked mean
logits  = Linear(512, C)(Dropout(p=0.3)(pooled))
```

- **Parameters:** 512 × C + C ≈ 1K (C = num_classes, typically 2 or 3)
- **Temporal sensitivity:** zero — treats the context window as a bag of patches; order and duration are irrelevant
- **Role in experiment:** non-temporal baseline. If MeanPool ≈ LSTM/Transformer at all L, context-length effects are driven by feature averaging, not temporal integration.

### LSTMHead

Two configurations exist depending on task type:

**For binary/multiclass seq2label tasks (hidden=128, num_layers=1):**
```
packed = pack_padded_sequence(x, lengths, enforce_sorted=False)
_, (h_n, _) = BiLSTM(input=512, hidden=128, layers=1, bidirectional=True)(packed)
h = cat([h_n[-2], h_n[-1]], dim=-1)   # (B, 256)
logits = Linear(256, C)(Dropout(p=0.3)(h))
```
- **Parameters:** 4 × (512 + 128) × 128 × 2 (BiLSTM L1) + 256 × C ≈ 655K + head

**For sleep staging seq2seq (hidden=256, num_layers=2):**
```
_, (h_n, _) = BiLSTM(input=512, hidden=256, layers=2, bidirectional=True)(packed)
h = cat([h_n[-2], h_n[-1]], dim=-1)   # (B, 512)
logits = Linear(512, C)(Dropout(p=0.3)(h))
```
- **Parameters:** ~3.16M (Layer 1: 4×(512+256)×256×2; Layer 2: 4×(512+256)×256×2; head)
- **Rationale:** Phase 0 experiments showed a substantial performance drop (kappa: 0.62 → 0.54 at 10m) when using the smaller architecture for the 5-class seq2seq task. The 5-class problem requires higher model capacity than binary tasks.

**Shared properties of both configurations:**
- **`pack_padded_sequence`:** skips padded patches; LSTM never processes zero-padded inputs
- **BiLSTM:** final hidden concatenates last valid forward and backward states → full window access
- **For centered seq2seq:** both forward and backward LSTM passes see different halves of the symmetric window around the anchor — the forward pass processes from past to future, the backward from future to past. This is appropriate for the centered (non-causal) staging design.

**[QUESTION 11 — UPDATED]** The current design uses centered context (future + past both visible). BiLSTM is fully appropriate here. The earlier concern about "future leakage" was based on a misunderstanding of the design — the centered symmetric window explicitly includes future patches, and the BiLSTM is designed to exploit both directions. Bidirectionality is well-suited to the centered context mode.

### TransformerHead

```
x = Linear(512, 128)(x)                              # project to hidden dim
cls = cls_token.expand(B, -1, -1)
x = cat([cls, x], dim=1)                             # prepend CLS; shape (B, N+1, 128)
x = x + sinusoidal_PE[:N+1, :]                       # positional encoding
mask_full = cat([zeros(B,1,bool), mask], dim=1)      # CLS never masked
out = TransformerEncoder(d_model=128, nhead=8, FF_dim=512, Pre-LN, layers=1)(x,
        src_key_padding_mask=mask_full if mask_full.any() else None)
logits = Linear(128, C)(Dropout(p=0.3)(out[:, 0, :]))  # CLS output
```

- **Parameters:** 512×128 (proj) + 128 (CLS) + 1 × TransformerLayer(128, 8, 512) + 128×C ≈ 264K
- **Flash attention:** when `src_key_padding_mask=None` (all-False, which is guaranteed by the cohort filter), PyTorch selects Flash attention → O(N) memory. Confirmed in training logs: `[Attn] Flash (mask=None, O(N) memory) | N=2880`.
- **Pre-LN:** `norm_first=True` in `TransformerEncoderLayer` — more stable than Post-LN for long sequences.
- **CLS token:** the class token aggregates sequence-level information; its output after the transformer is the subject representation.
- **Sinusoidal PE:** fixed (non-trainable), max_len=4096 to accommodate 240m (2880 patches + CLS + margin).

### Full-channel run: matched architectures for clean comparison

The full-channel run (`phase0_v3_full_config.yaml`) uses **the same head configs as the
fast-channel baseline**, by design:

| Run | seq2label tasks | Sleep staging |
|---|---|---|
| Fast-channel (v3) | hidden=128, layers=1 (~658K LSTM, ~264K Transformer) | hidden=256, layers=2 (~3.16M LSTM) |
| Full-channel (v3_full) | hidden=128, layers=1 (~658K LSTM, ~264K Transformer) | hidden=256, layers=2 (~3.16M LSTM) |

The config key `model.hidden_dim` is a **shared** value: the same number drives the LSTM
hidden-state width and the Transformer d_model.

Sleep staging uses a larger head in both runs because phase0 showed kappa dropping 0.62 → 0.54
at 10m with the 128/1 config — the 5-class seq2seq task requires substantially more capacity
than binary seq2label tasks (see §III-E LSTMHead section above). Sleep staging uses a separate
config file (`phase0_v3_full_staging_config.yaml`) with `hidden_dim: 256, num_layers: 2`.

**Paper-ready statement:** All full-channel experiments use identical head architectures to
their fast-channel counterparts. Any performance difference reflects the richer channel set alone.

Note: the TransformerHead for sleep staging at `d_model=256, ff=1024, layers=2` has ~1.7M
parameters — the original config comment stating "~1M" was incorrect.

### Design rationale (paper paragraph draft)

> We compare three head architectures of increasing temporal sophistication: a non-temporal
> pooling baseline (MeanPool, ~1K parameters), a bidirectional LSTM (BiLSTM, ~658K parameters),
> and a Transformer encoder with CLS token (~264K parameters). All receive the same
> 512-dimensional flattened SleepFM embeddings as input (`input_dim = 4 modalities × 128 dims`).
> For sleep staging only, we use a larger BiLSTM configuration (hidden=256, 2 layers, ~3.16M
> parameters) and Transformer (d_model=256, ~1.7M parameters), motivated by a significant
> capacity gap observed in phase 0 experiments (kappa: 0.62 vs 0.54 at 10m context). All
> fast-channel and full-channel runs use identical head architectures, ensuring that any
> performance difference reflects the channel set and not model capacity.

---

## III-F. Training Protocol

### Optimiser and schedule

- **Optimiser:** AdamW, weight decay = 1e-3
- **Learning rate:** 1e-4 for L ∈ {30s, 10m, 40m, 80m}; 5e-5 for L ∈ {120m, 240m}
  - Rationale for LR reduction at long contexts: overlapping training windows drawn from a 240-minute recording are highly correlated (they differ by only one stride-N shift); the effective gradient diversity per step is lower than at short contexts where windows can span the entire night. A halved LR compensates.
- **LR schedule:** cosine annealing over the total number of training epochs (warm restarts disabled; single cosine cycle to epoch limit)
- **Epoch limit:** 40 epochs for seq2label tasks; **60 epochs for sleep staging** (120m/240m require more epochs with the larger 256/2 model to converge)
- **Early stopping:** patience = 10 epochs. Monitor differs by task type:
  - **Seq2label (all binary/multiclass tasks):** `val_auroc` (macro OvR). Threshold-free; robust to class imbalance; better than val_loss for imbalanced tasks (val_loss is noisy for tasks like OSA where a single poorly-calibrated batch causes large spikes).
  - **Sleep staging (seq2seq, 5-class):** `val_kappa` (Cohen's κ at argmax threshold). Directly optimises the primary reported metric. val_auroc for 5-class OvR is slow to plateau, causing 120m to hit the epoch limit without converging; val_balanced_accuracy is dangerous for staged predictions (can peak spuriously at epoch 1–2 due to 5-class imbalance, observed in transformer 80m). val_kappa converges at the same rate as val_loss while aligning checkpoint selection with the paper metric.
  - **[Paper note]:** "Models were trained with early stopping (patience 10 epochs) monitoring validation AUROC for classification tasks and validation Cohen's κ for sleep staging."

### Batch size and gradient accumulation

- **Batch size = 32** at all context lengths, on all heads.
- **Gradient accumulation = 1** (no accumulation needed). This was established after:
  1. Cohort consistency filter (min_recording_patches = 2880) ensures all padding masks are all-False, so no memory is wasted on attention over padding.
  2. Mask fix in TransformerHead: `src_key_padding_mask=None` when mask is all-False → Flash attention (O(N) memory), not O(N²) Math attention.
- **Paper claim:** "All models were trained with batch size 32, identical across all context lengths."

### Class imbalance handling

- **Loss function:** CrossEntropyLoss with inverse-frequency class weights. Weights computed from training labels as `w_c = N_total / (C × N_c)` then normalised so mean weight = 1.
- **No WeightedRandomSampler:** tested but caused recall collapse (model predicting only minority class) at the 2–3:1 imbalance ratios present in most tasks. Loss weighting alone was sufficient.
- **Post-hoc threshold tuning** (for imbalanced binary tasks): after inference, the decision threshold is swept over the validation set `[0.01, 0.99]` to maximise balanced accuracy; the optimal threshold is applied to the test set. This improves balanced accuracy without affecting AUROC. Confirmed gain: +0.015 balanced accuracy for bmi_binary (32.8% minority). See §III-G.

### Training K (windows per subject per epoch)

K_train = 5 windows per subject per epoch, fixed across all context lengths.

**Justification (two-column table for paper):**

| Criterion | K=5 fixed (our choice) | Token budget (K × L = const) |
|---|---|---|
| Equal gradient updates/subject/epoch | ✅ identical at all L | ✗ 160× more at 30s than 240m |
| Equal total signal/subject/epoch | ✗ short contexts see less | ✅ |
| Agreement at long contexts (≥80m) | ✅ both give K≈1 | ✅ |

K=5 is the correct criterion for an unconfounded comparison of context-length effects. The token-budget approach (K × L = constant) introduces an asymmetric effective learning rate — the 30s model receives 160× more gradient updates per epoch than the 120m model, confounding the comparison.

K=5 controls per-epoch exposure, not total data seen across training. With random window sampling over 40 epochs, the 30s model covers a diverse fraction of the available ~960 windows by training completion.

**Sensitivity validation:** a one-task token-budget ablation (`sex_binary` or `bmi_binary`, `windows_strategy: "token_budget"`, `token_budget_minutes: 80`) will be reported in Supplementary as Table SX. If the saturation curve shape is unchanged (expected, since both methods agree at L ≥ 80m), this closes the reviewer concern definitively.

### Checkpointing and SLURM infrastructure

- **Best checkpoint:** saved whenever the monitor metric improves (as `best_model.pt`). Monitor is `val_auroc` for seq2label tasks and `val_kappa` for sleep staging.
- **Resume checkpoint:** per-epoch `resume.pt` for crash/timeout recovery on SLURM. Deleted on successful completion.
- **Auto-requeue:** SLURM `--requeue` flag + USR1 signal handler (120s before wall-time) enables seamless job restart with W&B run continuation.
- **[Note for paper]:** infrastructure details go in supplementary or omitted; mention only: "models were trained on NVIDIA H100 GPUs on Compute Canada infrastructure."

**[QUESTION 12]** What is the approximate total GPU compute time for all v3 experiments (Groups A–D, ~114 context runs)? This is useful for the supplementary "computational cost" section. Rough estimate from the timing table: large/LSTM ~12h per 240m run × 6 contexts × ~12 experiments ≈ ballpark numbers. Do you have the actual wall-time from SLURM logs?

---

## III-G. Evaluation Protocol

### Primary metric: AUROC

AUROC (Area Under the Receiver Operating Characteristic Curve) is the primary metric for all binary and multi-class tasks.

- **Binary tasks:** standard AUROC using class-1 probability scores.
- **Multi-class tasks (age_class, osa_severity_apples):** macro-averaged one-vs-rest AUROC.
- **Sleep staging:** AUROC is logged but NOT the primary metric (see below).
- **Why AUROC:** threshold-free; robust to class imbalance; used for early stopping. All comparisons across context lengths use AUROC — it is not affected by the choice of decision threshold.

### Secondary metric: balanced accuracy at optimal threshold

For binary tasks with significant class imbalance (minority class < 40%), we additionally report balanced accuracy at the decision threshold selected to maximise balanced accuracy on the validation set.

```
t_opt = argmax_{t ∈ [0.01, 0.99]} balanced_accuracy(val_labels, (val_scores > t))
test_bal_acc = balanced_accuracy(test_labels, (test_scores > t_opt))
```

This prevents the misleadingly high "accuracy" of always predicting the majority class.

Tasks where threshold tuning is applied: bmi_binary (+0.015 confirmed), osa_binary_apples_postqc (est. +0.06–0.09), depression_extreme_binary (est. +0.10–0.15 if AUROC > 0.60), cvd_binary.

### Sleep staging metrics: Cohen's κ and per-stage F1

For sleep staging (seq2seq), the primary metrics are:
- **Cohen's κ (kappa):** chance-corrected inter-rater agreement; standard metric in the sleep staging literature. κ = 0 = chance, κ = 1 = perfect.
- **Per-stage F1:** F1 score computed separately for each stage (W, N1, N2, N3, REM). N1 is the minority stage (~5–8% of epochs) and typically has the lowest F1.

AUROC is also reported for reference (multi-class OvR) but is not used for model selection.

### Subject-level prediction aggregation

After inference over all non-overlapping windows (K_infer = T//N):

**Mean-probability (primary):** average softmax probabilities across K_infer windows → single subject-level score → AUROC.
```
score_subj = mean(prob_class1 for all windows of subject)
```

**Majority vote (secondary):** mode of per-window hard predictions.

Mean-probability is preferred for AUROC (requires soft scores) and tends to outperform majority vote at small K because it preserves confidence information.

### K-window sweep (post-hoc, CPU-only)

From saved inference parquets, we sweep K ∈ {1, 5, 10, 20, 50, all} (sparse) and K ∈ {1, 2, 3, ..., 500, all} (dense) per context length. For each K:
1. Subsample K windows per subject (evenly spaced within the inference pool)
2. Compute mean-probability AUROC and majority-vote AUROC

This answers H3 (aggregation saturation): how many windows per patient are needed before performance saturates at a given context length?

### Iso-compute analysis

The 2D grid of (L, K) pairs, where each cell gives AUROC achieved by a model trained at context L and aggregated from K windows at inference.

**Total signal budget per patient:** `compute_budget = L_minutes × K` (minutes of PSG used per patient at inference).

**Iso-compute contour:** all (L, K) pairs satisfying `L × K ≈ constant` represent different trade-offs (many short windows vs few long windows) at the same total inference cost. Comparing cells on the same iso-compute line answers H2.

### Confidence intervals

95% bootstrap confidence intervals via subject-level resampling (1,000 bootstrap samples). Resampling at the subject level (not window level) is correct because subjects are the independent unit; window-level resampling would underestimate variance.

Used for: saturation curve error bands, comparison of L* vs L*±1 context results.

### L* (saturation threshold)

For each task and head, L* is defined as the smallest context length L such that AUROC at L is within 0.005 (0.5 percentage points) of the best AUROC achieved across all context lengths.

```
L* = min{L : AUROC(L) ≥ max_L(AUROC) − 0.005}
```

**[QUESTION 13]** Is 0.005 the right threshold? In some tasks, 0.5pp is very small relative to the confidence interval width. An alternative is to define L* as the point where the bootstrap CI at L overlaps with the CI at the best-L. This is more statistically principled but harder to explain in one sentence.

---

## III-H. Context-Length Sweep Design

### The sweep

One model is trained per (task, head, context length). Six context lengths:

| Context | Patches N | Wall clock L |
|---|---|---|
| 30s | 6 | 0.5 min |
| 10m | 120 | 10 min |
| 40m | 480 | 40 min |
| 80m | 960 | 80 min |
| 120m | 1440 | 120 min |
| 240m | 2880 | 240 min (= 4 hours) |

These cover ~3 decades of temporal context (0.5 min → 240 min) on a roughly log-spaced grid.

The gap between 30s and 10m (a 20× jump) is the largest uninformative gap — this is where most tasks show the steepest AUROC improvement. Acknowledged as a limitation.

### The only variable between models

For a fixed (task, head), the context window length N is the **only variable** between experiments:
- Same SleepFM encoder (frozen, identical weights)
- Same head architecture, hidden dimensions, and random initialisation strategy
- Same optimiser, LR schedule, and epoch limit
- Same subject population (cohort filter ensures this)
- Same train/val/test splits (same subjects, same seed)

This allows direct attribution of AUROC differences to context length.

### Four hypotheses tested

| Hypothesis | What it tests | How answered |
|---|---|---|
| **H1 (context saturation):** AUROC increases with L and saturates at task-specific L* | Context length effect | Saturation curve (Fig 2): AUROC vs L at K=all |
| **H2 (aggregation substitution):** At matched total budget L×K, AUROC is the same regardless of L/K split | Whether long-context training is necessary or averaging substitutes | Iso-compute heatmap (Fig 3): cells on same diagonal |
| **H3 (aggregation saturation):** For fixed L, AUROC saturates at some small K | Minimum windows needed per patient at inference | K-sweep curves (Fig S3): AUROC vs K per context |
| **H4 (head comparison):** LSTM/Transformer outperform MeanPool at long L, not short L | Whether temporal modeling matters | Saturation curves per head (Fig 2) |

---

## Open questions summary

| # | Question | Blocking what? |
|---|---|---|
| Q1 | Dataset DOI vs paper citation preference | Reference list |
| Q2 | SHHS1 R&K → AASM staging mapping: include SHHS1 in staging or exclude? | Sleep staging cohort |
| Q3 | MrOS Visit 2 BMI/AHI availability | Task label table |
| Q4 | Exact per-task N (from task_subjects/ CSVs) | Dataset table in paper |
| Q5 | CVD definition mismatch: footnote in Methods or only in Discussion? | Methods framing |
| Q6 | Extreme-group depression design rationale | Methods framing |
| Q7 | Any artifact rejection steps applied? | Methods accuracy |
| Q8 | Channel naming issues during preprocessing? | Methods accuracy |
| Q9 | Total GPU compute for embedding extraction | Supplementary compute table |
| Q10 | How many subjects too short for even 1 full 5-min chunk? | Dataset N reporting |
| Q11 | BiLSTM for staging: backward pass is into past context only? | Methods accuracy |
| Q12 | Total GPU compute for all v3 training runs | Supplementary compute table |
| Q13 | L* threshold: 0.005 absolute, or CI-overlap criterion? | Results framing |

---

## Notes on paper vs supplementary allocation

### In the 10-page TBME main text:
- Datasets: compact table (one row per cohort) + 2–3 sentences per unique aspect. ~0.5 columns.
- Preprocessing: 2–3 sentences covering the key steps (filter, resample, normalize, embed). Implementation details → supplementary.
- SleepFM: 1 short paragraph (frozen, SetTransformer, 512-dim output, cite original paper).
- Context-window dataset: 3–4 sentences covering seq2label vs seq2seq, cohort filter, K=5 justification.
- Heads: compact table (architecture, params, key property) + 1–2 sentences per head.
- Training: 4–5 sentences covering optimizer, batch size, early stopping, K=5 justification.
- Evaluation: 2–3 sentences covering AUROC primary, threshold tuning secondary, Cohen's κ for staging, bootstrap CIs.

### In Supplementary:
- Full channel priority lists per dataset
- Detailed bandpass filter parameters
- NaN blocklist rationale
- Exact timing and compute costs
- K=5 sensitivity ablation table
- Full cohort filter exclusion list
