# Training Protocol Fixes

This document records methodological issues in the current training setup and the planned fixes. None of these changes have been implemented yet. Ask Claude to implement them when ready.

**Why this matters:** These fixes ensure that when you compare models trained at different context lengths (30s vs 10m vs 240m), the only variable is the context length L — not the amount of training data per epoch, not the number of gradient steps, not the number of windows per subject. Without these fixes, any observed performance difference could partly reflect training-data asymmetry rather than context quality.

---

## Issue 1 — Fixed K is not actually fixed at long context lengths

### The problem

`configs/phase0_v2_config.yaml` sets `windows_per_subject: 5`, which is supposed to give every subject K=5 training windows per epoch. But the dataset index builder (`_build_seq2label_index` in `src/nsrr_tools/datasets/context_window_dataset.py`) only samples from **non-overlapping, N-aligned** window positions:

```python
n_windows = T // N          # integer division → non-overlapping positions only
K = min(self._K_max, n_windows)
starts = [s * N for s in rng.choice(n_windows, size=K, replace=False)]
```

For a **240m context** (N = 480 patches):
- A typical 8h night (T ≈ 960 patches): `n_windows = 960 // 480 = 2` → K = min(5, 2) = **2**
- A 5h night (T ≈ 600 patches): `n_windows = 1` → K = **1**

For a **30s context** (N = 1 patch):
- Any night: `n_windows = T` (thousands) → K = **5** (always achieves the target)

**The result:** The 240m model trains on ~40% of the gradient steps per epoch compared to the 30s model. "Fixed K=5" is only fixed for short contexts. This is a hidden training-data asymmetry: any performance gap between L=30s and L=240m could partly reflect the 30s model having more training data, not better context quality.

### The fix — overlapping window sampling

Allow window starts at any valid position, not just N-aligned ones. A valid start is any integer `s ∈ [0, T-N]`. The number of valid starts is `n_valid = T - N + 1`.

For 240m (N=480) with an 8h night (T=960): `n_valid = 960 - 480 + 1 = 481` valid starts. Sampling K=5 without replacement from 481 positions is trivially achievable. The windows may overlap (adjacent window starts could be, e.g., 10 patches apart), but this is fine — the model sees slightly different subsets of the same night.

**Key property preserved:** every subject with T ≥ N contributes exactly K=5 training examples per epoch at all context lengths. The only remaining asymmetry is intentional: longer-context windows have more patches, so each gradient step is more expensive (more FLOPs) — this is what you measure with the scaling law analysis.

### What changes in the code

**File:** `src/nsrr_tools/datasets/context_window_dataset.py`  
**Function:** `_build_seq2label_index`

Replace:
```python
n_windows = T // N                    # non-overlapping windows
K = min(self._K_max, n_windows)

if self.split == "train":
    starts = sorted(
        rng.choice(n_windows, size=K, replace=False).tolist()
    )
    starts = [s * N for s in starts]
else:
    positions = np.linspace(0, n_windows - 1, K, dtype=int)
    starts = [int(p) * N for p in positions]
```

With:
```python
n_valid = T - N + 1                   # all valid (possibly overlapping) start positions
K = min(self._K_max, n_valid)

if self.split == "train":
    # K random starts without replacement from all valid positions
    starts = sorted(
        rng.choice(n_valid, size=K, replace=False).tolist()
    )
else:
    # K evenly spaced starts across [0, T-N] (deterministic, for val/test during training)
    starts = np.linspace(0, n_valid - 1, K, dtype=int).tolist()

for s in starts:
    index.append((row_idx, int(s), label))
```

**No other files need to change.** The `_get_seq2label_window` function already handles arbitrary `window_start` values correctly (it slices `emb[s : s+N]` and pads if needed). Since all starts now satisfy `s + N ≤ T`, no padding will occur.

### What this does NOT change

- The val/test split datasets used for training-time metrics still use K evenly-spaced windows, just with arbitrary (not N-aligned) spacing — same logic, different pool of positions
- The inference script (`infer_subject_windows.py`) is unaffected — it already uses all possible non-overlapping windows for evaluation
- Subjects with T < N (night shorter than the context window): still get one zero-padded window. This is correct; they can't provide a full context window regardless

### Impact on existing results

If you have already-trained models using the old non-overlapping strategy, their `training_curves.csv` files are valid as-is (they reflect the actual training that happened). **Do not mix** old-protocol and new-protocol runs in the same comparison figure — rerun all context lengths with the new protocol if you switch.

This is a **breaking change to training** — all experiments should be rerun after this fix. It does not affect inference or analysis scripts.

---

## Issue 2 — `windows_per_subject_train` in metrics.json reflects config, not actual K

### The problem

In `train_context_sweep.py`:
```python
_windows_per_subject_train = int(cfg["dataset"].get("windows_per_subject", 5))
```

This reads the configured value (5), not the actual K per subject (which is min(5, n_valid)). So `metrics.json` always reports `windows_per_subject_train: 5` even when the model trained with K=1–2 per subject.

After Issue 1 is fixed, the actual K will be 5 for almost all subjects, making this accurate. But it should ideally reflect the true per-subject average (for the few short-night subjects that are capped by T < N).

### The fix

After building the training dataset index, compute the actual average K:

```python
# After train_ds is built and _index is set:
from collections import Counter
# Count windows per subject in training split
subject_window_counts = Counter(row_idx for row_idx, _, _ in train_ds._index)
_windows_per_subject_train = (
    sum(subject_window_counts.values()) / max(len(subject_window_counts), 1)
    if subject_window_counts else 0
)
```

Then save this actual average as `windows_per_subject_train` in `metrics.json`. This gives you a precise record of how much data each model actually trained on.

**File:** `scripts/train_context_sweep.py`  
**Location:** just after `train_ds = make_ds("train")`, before the model is built

---

## Issue 3 — Token budget strategy should be deprecated / clearly labelled as secondary

### The problem

The config has `windows_strategy: "fixed"` (correct default) and a commented `"token_budget"` alternative. The token budget sets K ∝ 1/L:
- At 30s: K = 240/0.5 = 480 (capped at k_max=50 → K=50)
- At 240m: K = 240/240 = 1

This is a completely different training regime and should NOT be used as a substitute for fixed K. It answers a different question: "what is the optimal compute allocation across context lengths?" — not "does context length improve performance?"

### The fix

No code change needed. Just add a comment in the config making the purpose explicit:

In `configs/phase0_v2_config.yaml`, update the `windows_strategy` comment:

```yaml
# K-selection strategy for training windows per subject:
#   "fixed"        — RECOMMENDED for context-length comparison experiments.
#                    Use dataset.windows_per_subject as K for all context lengths.
#                    With overlapping window sampling (see dataset code), this
#                    gives exactly K windows per subject at every context length.
#   "token_budget" — SECONDARY ANALYSIS ONLY. Sets K = floor(budget/ctx_min) so
#                    total night coverage per epoch is constant across L. Useful
#                    for §10 (head architecture at equal compute) but NOT suitable
#                    as the primary training protocol — it confounds K and L.
windows_strategy: "fixed"
```

---

## Issue 4 — Training step count initialisation on resume is approximate

### The problem (minor)

When resuming from `resume.pt`, `global_step` is now read directly from the checkpoint. But old checkpoints (written before the `global_step` field was added) fall back to:

```python
_global_step = sum(
    1 for h in history if not h.get("is_overfit_epoch", False)
) * _steps_per_epoch
```

This approximation is exact only if `steps_per_epoch` did not change between the original run and the resumed run (e.g., if the dataset size is exactly reproducible). In practice, `steps_per_epoch` is determined by `len(train_loader)` which depends on dataset size, so this should be stable. No fix strictly needed, but note it for bookkeeping.

---

## Summary of changes needed

| Issue | File | Change | Requires retraining? |
|-------|------|--------|----------------------|
| 1 (overlapping windows) | `src/nsrr_tools/datasets/context_window_dataset.py` | Change `_build_seq2label_index` to sample from all valid starts | **Yes — rerun all experiments** |
| 2 (actual K in metrics.json) | `scripts/train_context_sweep.py` | Compute average K from index after dataset is built | Yes (minor; only affects metrics.json reporting) |
| 3 (token budget comment) | `configs/phase0_v2_config.yaml` | Update comment to clarify intended use | No |
| 4 (global_step resume) | Already implemented — backwards-compatible fallback exists | None | No |

## What to keep unchanged

- **Batch size**: keep fixed at 32 (or per-experiment override). Do not equalize FLOPs per step across context lengths — the difference is inherent and scientifically interesting.
- **Epochs, LR, patience**: same for all context lengths. The only variable is L.
- **Val/test inference**: `infer_subject_windows.py` is unaffected — it already uses all non-overlapping windows for evaluation, which is correct and should remain so.
- **`windows_per_subject: 5`**: keep this as the target K. After Issue 1 is fixed, it will be achieved for essentially all subjects with T ≥ N.

## Claim you can make after these fixes

> "All models were trained with the same protocol: K=5 randomly sampled context windows per subject per epoch (overlapping positions allowed), fixed batch size of 32, and the same optimizer, LR schedule, and early stopping criterion. The only variable between experiments is the context length L. Per-step FLOPs differ across L (longer windows require more computation per gradient step); this is documented and used in the scaling-law analysis (§1B), not corrected for."
