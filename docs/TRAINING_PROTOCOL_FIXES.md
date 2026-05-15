# Training Protocol Fixes

This document records methodological issues in the v2 training setup and their fixes.

**Implementation status (as of v3):**
- Issue 1 (overlapping windows) — ✅ Implemented in `context_window_dataset.py`
- Issue 2 (actual K in metrics.json) — ✅ Implemented in `train_context_sweep.py`
- Issue 3 (token budget comment) — ✅ Implemented in `phase0_v2_config.yaml` and `phase0_v3_config.yaml`
- Issue 4 (global_step resume) — ✅ Previously implemented
- Issue 5 (batch size variation) — ✅ Already recorded via metrics.json; no code change needed

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

### Window overlap in practice

For a representative 8h night (T = 960 patches), the table below shows n_valid and the expected spacing between the K=5 sampled windows, to give intuition for how much overlap exists at each context length:

| Context L | N (patches) | n_valid (8h night) | K achieved | Avg. spacing between windows |
|-----------|-------------|---------------------|------------|------------------------------|
| 30 s      | 1           | 960                 | 5          | ~192 patches = ~96 min       |
| 5 min     | 10          | 951                 | 5          | ~190 patches = ~95 min       |
| 30 min    | 60          | 901                 | 5          | ~180 patches = ~90 min       |
| 2 h       | 240         | 721                 | 5          | ~144 patches = ~72 min       |
| 4 h       | 480         | 481                 | 5          | ~96 patches = ~48 min        |

Avg. spacing = (n_valid − 1) / (K − 1). All windows share the same subject night, so the K=5 windows at 4h context are ~48 min apart on average — meaningful overlap but far from identical training examples.

For a short 5h night (T = 600 patches):

| Context L | N (patches) | n_valid (5h night) | K achieved |
|-----------|-------------|---------------------|------------|
| 30 s      | 1           | 600                 | 5          |
| 2 h       | 240         | 361                 | 5          |
| 4 h       | 480         | 121                 | 5          |

K=5 is always achieved for any T ≥ N + 4. Only nights shorter than the context window itself (T < N) fall back to K=1 with zero-padding, which is correct regardless.

### Choosing K_max

Any K_max works with the formula `K = min(K_max, n_valid)`. Guidelines:

- **K=5**: conservative, faster epochs. Acceptable if training time is the primary constraint. Each epoch sees 5 random views per subject. Effectively discards data for long contexts where n_valid ≫ 5, but this is symmetric across all L so it does not introduce bias.
- **K=10**: safer. Doubles epoch time but provides twice the gradient signal. Reduces epoch-to-epoch variance in the sampled windows. Recommended if compute allows.
- **K=20+**: approaching diminishing returns unless subjects have highly variable nights. Not needed for the scaling law argument.

The paper's claim ("K windows per subject per epoch") holds for any fixed K_max after this fix. The value should be the same for all context lengths in the comparison — do not vary K_max across experiments.

### What this does NOT change

- The val/test split datasets used for training-time metrics still use K evenly-spaced windows, just with arbitrary (not N-aligned) spacing — same logic, different pool of positions
- The inference script (`infer_subject_windows.py`) is unaffected — it already uses all possible non-overlapping windows for evaluation
- Subjects with T < N (night shorter than the context window): still get one zero-padded window. This is correct; they can't provide a full context window regardless

### Impact on existing results

If you have already-trained models using the old non-overlapping strategy, their `training_curves.csv` files are valid as-is (they reflect the actual training that happened). **Do not mix** old-protocol and new-protocol runs in the same comparison figure — rerun all context lengths with the new protocol if you switch.

This is a **breaking change to training** — all experiments should be rerun after this fix. It does not affect inference or analysis scripts.

---

## Issue 5 — Batch size must vary across context lengths (OOM) but must be recorded

### The problem

At longer context lengths (2h, 4h), the GPU runs out of memory with the same batch size used at 30s or 5min. The training script currently has a single `batch_size` setting in the config. If you silently reduce the batch size to avoid OOM, the FLOPs comparison across context lengths is wrong unless you record what was actually used.

This is not a methodological flaw in itself — varying batch size is standard practice. The flaw is in not recording or accounting for it.

### What NOT to do — gradient accumulation

Gradient accumulation (accumulate gradients over N micro-batches before stepping the optimizer) simulates a larger effective batch size but does not fix the memory problem: it runs N forward passes sequentially, which is just as slow and still OOMs on the backward pass of each micro-batch if the micro-batch itself is too large.

More importantly for this paper:
- Gradient accumulation adds code complexity and a new hyperparameter.
- The scaling law plots (§1B) use total FLOPs = `batch_size × steps × per_window_FLOPs`. With accumulation, the effective batch size is `micro_batch × accumulation_steps`, which is easy to get wrong.
- For a paper whose primary claim is final AUROC vs context length (not per-step training dynamics), identical total gradient steps matter more than identical per-step batch sizes.

**Do not use gradient accumulation.**

### The fix — different batch sizes, always recorded

Use different `batch_size` values for different context lengths. The value is already stored in `metrics.json` via the config. The critical requirement is that the value recorded is the actual batch size used, not a default.

**Rule:** when you change batch size to avoid OOM, update the config (or the per-experiment override) before launching. Verify `metrics.json` contains the correct value after training.

**FLOPs formula using recorded values:**

```
total_FLOPs = batch_size × total_steps × per_window_FLOPs(N, model_arch)
```

where `batch_size` comes from `metrics.json` (per-experiment), `total_steps` from `training_curves.csv` (`global_step` at final epoch), and `per_window_FLOPs` from the architecture-specific formula (§1B).

**Suggested batch sizes by context length (adjust for your GPU):**

| Context L | N (patches) | Suggested batch_size |
|-----------|-------------|----------------------|
| 30 s      | 1           | 32                   |
| 5 min     | 10          | 32                   |
| 30 min    | 60          | 16                   |
| 2 h       | 240         | 8                    |
| 4 h       | 480         | 4–8                  |

**Learning rate scaling (optional but recommended):** if batch size drops by 4× (e.g., from 32 to 8), consider scaling LR by √(1/4) = 0.5 (square-root rule) or by 1/4 (linear rule). Document whichever you choose in `metrics.json` via the config. For exploratory runs, keeping LR fixed is acceptable if you document it.

### What changes in the code

No code change strictly needed if the config already records `batch_size` correctly. The actual `batch_size` logged in `metrics.json` comes from:

```python
# train_context_sweep.py — already present:
"batch_size": int(cfg["training"]["batch_size"]),
```

This is correct as long as the per-experiment config override sets the right value. The `windows_per_subject_train` fix (Issue 2) is independent and should be done alongside this.

### What to verify before a run

```bash
grep batch_size configs/your_experiment_config.yaml   # check value is correct
# after training:
python -c "import json; d=json.load(open('results/your_run/metrics.json')); print(d['batch_size'])"
```

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

## What to keep unchanged

- **Batch size**: will vary by context length to avoid OOM (see Issue 5). Record actual value in metrics.json. Do not use gradient accumulation. Do not equalize FLOPs per step — the per-step compute difference across L is inherent and scientifically meaningful.
- **K_max (windows_per_subject)**: keep the same value for all context lengths in a given comparison. K=5 is acceptable; K=10 is safer. After Issue 1 is fixed, K=K_max is achieved for essentially all subjects with T ≥ N.
- **Epochs, LR, patience**: same for all context lengths. The only primary variable is L.
- **Val/test inference**: `infer_subject_windows.py` is unaffected — it already uses all non-overlapping windows for evaluation, which is correct and should remain so.

## Summary of changes (all implemented in v3)

| Issue | File | Change | Status |
|-------|------|--------|--------|
| 1 (overlapping windows) | `src/nsrr_tools/datasets/context_window_dataset.py` | Sampling from all valid starts — K=K_max for all subjects | ✅ Done |
| 2 (actual K in metrics.json) | `scripts/train_context_sweep.py` | Compute average K from index after dataset is built | ✅ Done |
| 3 (token budget comment) | `configs/phase0_v2_config.yaml`, `phase0_v3_config.yaml` | Warning comment; token_budget labelled SECONDARY ANALYSIS ONLY | ✅ Done |
| 4 (global_step resume) | `scripts/train_context_sweep.py` | global_step saved in history/resume.pt and training_curves.csv | ✅ Done (prior session) |
| 5 (batch size variation) | `experiments/v2_registry.yaml` | Per-experiment batch_size; smaller for longer contexts; recorded in metrics.json | ✅ Done (config-level; no code change) |
| — (new directories) | `configs/phase0_v3_config.yaml`, `experiments/v2_registry.yaml`, shell scripts | Results → phase0_v3, logs → logs_v3; prevents mixing v2/v3 data | ✅ Done |
| — (context LR overrides) | `scripts/train_context_sweep.py`, `configs/phase0_v3_config.yaml` | LR halved at 120m and 240m (5e-5) relative to short contexts | ✅ Done |

## Claim you can make after these fixes (v3 protocol)

> "All models were trained with the same protocol: K=5 randomly sampled context windows per subject per epoch, sampled from all valid start positions (overlapping windows allowed), giving exactly K=5 windows per subject at all context lengths for subjects with T ≥ N. The optimizer (AdamW), LR schedule (cosine), and early stopping criterion (AUROC patience=10) were identical across context lengths, except that a context-specific LR of 5×10⁻⁵ was used for 120m and 240m (vs 10⁻⁴ for shorter contexts) to compensate for higher gradient noise from overlapping windows and reduced batch size. Batch size was reduced for longer context lengths to avoid GPU OOM (32 for L ≤ 80min, smaller for L ≥ 120m); the actual batch size is recorded in each experiment's metrics.json and used in all FLOPs calculations. The only primary variable between experiments is the context length L."
