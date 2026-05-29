# Cohort Consistency Filter — `min_recording_patches`

## Problem

The context-length sweep compares model performance at 30s, 10m, 40m, 80m, 120m, and 240m. For this comparison to be valid, each point on the curve must answer exactly the same prediction question on exactly the same subjects — only the amount of context given as input should differ.

Without a recording-length filter, subjects whose PSG recording is shorter than the longest context window (240m = 2880 patches × 5 s/patch) would:

- be **included** at short contexts (30s–120m), where their full recording fits  
- be **excluded** at 240m, where they produce one zero-padded window  

This causes a silent cohort shift: the 240m model is evaluated on a different (harder? easier?) subject pool than the shorter-context models. Any observed performance difference at 240m could partly reflect population differences rather than context-length effects.

There was also a GPU memory consequence: zero-padded windows produce non-zero (`−∞`) entries in the Transformer's `src_key_padding_mask`. PyTorch's fused C++ kernel (`_transformer_encoder_layer_fwd`) falls back to O(N²) Math attention whenever any mask entry is non-zero, instead of Flash/Efficient Attention which is O(N). At 240m (N = 2881 with CLS token), one padded sample in a batch of 168 caused a 45 GB allocation on a 9.75 GB GPU slice — CUDA OOM. This was the root cause of the OOM during training/validation that the synthetic batch-size probe failed to detect (the probe always used all-False masks).

## Fix

`dataset.min_recording_patches: 2880` in `configs/phase0_v3_config.yaml`.

`ContextWindowDataset.__init__` reads this value and, after the train/val/test split, drops any subject with `T < min_recording_patches` from that split's subject list — before building the flat index. This applies at **every context length**, not just at 240m, so the cohort is identical across the entire sweep.

Setting `min_recording_patches: 0` disables the filter (backward compatible with older configs).

## Excluded subjects

**30 subjects total** across all embedding files (global shape cache); 20 affect active experiments.

Full list: [`excluded_subjects_T_lt_2880.csv`](excluded_subjects_T_lt_2880.csv)

| Dataset | Excluded | Notes |
|---------|----------|-------|
| APPLES  | 19 | Clearly truncated recordings: 5 min → 3.8 h |
| SHHS    | 1  | 180 min recording |
| STAGES  | 10 | Not used in active experiments (registry excludes STAGES) |

## Impact per active experiment

| Task | Total subjects | Excluded | % lost |
|------|---------------|----------|--------|
| sex_binary (APPLES+SHHS) | 9,547 | 20 | 0.21% |
| sleep_efficiency_binary (APPLES+SHHS+MrOS) | 13,480 | 20 | 0.15% |
| bmi_binary (APPLES+SHHS+MrOS) | 12,385 | 20 | 0.16% |
| age_class (APPLES+SHHS+MrOS) | 12,410 | 20 | 0.16% |
| psqi_binary (MrOS only) | 3,929 | **0** | 0% |
| depression_extreme_binary (APPLES only) | 874 | 15 | 1.72% |
| osa_binary_apples_postqc (APPLES only) | 1,103 | 19 | 1.72% |
| osa_severity_apples (APPLES only) | 1,103 | 19 | 1.72% |

The APPLES-only Tier 2 tasks lose ~1.7% of subjects, but those tasks are capped at 120m in the registry (not at 240m), so the filter is conservative for them — it removes subjects that would be perfectly valid at their maximum context. This is intentional: the filter enforces a single consistent cohort across all context lengths for a given experiment, so the paper can state unambiguously that "every evaluation point uses the same N subjects."

## Paper language

> "To ensure a fair comparison across context lengths, subjects whose full-night PSG recording was shorter than the longest context window (240 min, 2880 × 5-second patches) were excluded from all context lengths. This affected 20 of 9,547–13,480 subjects (≤ 0.2%) for Tier 1 tasks and 15–19 of 874–1,103 subjects (≤ 1.7%) for Tier 2 tasks. The excluded recordings ranged from 5 to 230 minutes and appear to be truncated acquisitions rather than full-night studies."

## Relation to batch-size probe

With `min_recording_patches=2880`, real training and validation data at every context length are guaranteed to have all-False padding masks (no sample is shorter than its context window). The batch-size probe in `scripts/find_batch_size.py` uses synthetic all-False masks by design; after this filter these masks accurately represent real data conditions, making the probe's recommendations reliable for the Transformer head.

Previously (before this filter), the probe gave over-optimistic batch sizes: it probed with all-False masks (→ Flash/Efficient Attention, O(N) memory), but real data occasionally contained padded samples (→ Math Attention, O(N²) memory, ~170× more at N=2881), causing CUDA OOM at runtime despite the probe passing.
