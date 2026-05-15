# Batch Size, Gradient Accumulation, and Fair Comparison Across Context Lengths

This document records the analysis of how batches are filled at each training step, what differs across context lengths, and what changes are needed to make the comparison fair for the paper's main claim: **does longer context improve clinical prediction from PSG?**

---

## 1. How a Batch Is Filled

The dataset index is a flat list of `n_subjects × K` items. With 13,000 subjects and K=5 windows per subject, there are **65,000 items** in the training set. Each item is one `(subject, window_start, label)` tuple.

At the start of each epoch, PyTorch shuffles all 65,000 items randomly. At each training step, the DataLoader grabs the next `batch_size` items from this shuffled list. Each item becomes one tensor of shape `[N, 512]`, where N is the number of 5-second patches in the context window.

**Concrete example with K=5, batch_size=32:**

```
Epoch starts → shuffle all 65,000 items
Batch 1:  items [41823, 7, 62001, 19433, 1002, 58877, ... (32 total)]
           → 32 tensors of shape [N, 512] stacked into [32, N, 512]
Batch 2:  next 32 items
...
Batch 2031: last chunk  → epoch complete, all 65,000 items seen once
```

Each of the 32 items almost certainly comes from a **different subject**, because each subject contributes only 5 items out of 65,000. The probability of two items sharing a subject in one batch is ~0.004 — negligible for large datasets.

The K=5 window positions per subject are **fixed for the entire training run** — chosen randomly once when the dataset is initialised. Each epoch sees the same 5 windows per subject, just in a different random order relative to other subjects. This is standard ML practice (same as ImageNet images seen every epoch) and is not a problem.

---

## 2. What Differs Between Short and Long Context — Without Gradient Accumulation

With the current OOM-driven batch sizes (no accumulation):

| Context | N patches | batch_size | Subjects/batch | Total context/step | FLOPs/step (relative) | Steps/epoch |
|---------|-----------|-----------|----------------|-------------------|----------------------|-------------|
| 30s     | 1         | 32        | ~32            | 16 min            | 1×                   | 2,031       |
| 10m     | 20        | 32        | ~32            | 320 min           | 20×                  | 2,031       |
| 40m     | 80        | 16        | ~16            | 1,280 min         | 80×                  | 4,063       |
| 80m     | 160       | 8         | ~8             | 1,280 min         | 160×                 | 8,125       |
| 120m    | 240       | 8         | ~8             | 1,920 min         | 240×                 | 8,125       |
| 240m    | 480       | 4         | ~4             | 960 min           | 480×                 | 16,250      |

**Problems with this setup:**

- The 240m model takes 8× more gradient steps per epoch than the 30s model
- The 240m model sees only ~4 subjects per batch vs ~32 for 30s
- The 240m model's each gradient update costs 480× more compute
- Everything is different simultaneously — subjects per batch, steps per epoch, compute per step — making comparison confounded

---

## 3. The Two Design Options

There are two philosophies for making the comparison fair, and they are mutually exclusive.

### Option A — Equal compute per gradient step (same total context per batch)

Set `batch_size ∝ 1/N` so that total patches processed per step is constant across L. Anchor to 240m with batch=4 (total patches per step = 4 × 480 = 1,920):

| Context | N patches | batch_size needed | Subjects/batch | Steps/epoch |
|---------|-----------|-------------------|----------------|-------------|
| 30s     | 1         | 1,920             | ~1,920         | 34          |
| 10m     | 20        | 96                | ~96            | 677         |
| 40m     | 80        | 24                | ~24            | 2,708       |
| 120m    | 240       | 8                 | ~8             | 8,125       |
| 240m    | 480       | 4                 | ~4             | 16,250      |

**Memory feasibility:** Yes — 30s windows are tiny ([1, 512] tensors), so batch=1920 fits easily on any GPU.

**Problems:**
- 30s model does only 34 gradient steps per epoch; 240m does 16,250 — 478× more
- The 30s model would converge differently (very large batches → smoother but potentially worse generalisation)
- This answers the question: *"given the same compute budget per step, how does context length perform?"* — not the same as *"does longer context improve prediction?"*

### Option B — Equal subjects per effective batch (variable compute per step)

Fix effective batch size = 32 at all context lengths. Use gradient accumulation at long contexts to compensate for OOM:

| Context | micro_batch | accum_steps | Effective batch | Subjects/eff. batch | Steps/epoch (effective) | FLOPs/eff. step (relative) |
|---------|------------|-------------|-----------------|---------------------|------------------------|---------------------------|
| 30s     | 32         | 1           | 32              | ~32                 | 2,031                  | 1×                        |
| 10m     | 32         | 1           | 32              | ~32                 | 2,031                  | 20×                       |
| 40m     | 16         | 2           | 32              | ~32                 | 2,031                  | 80×                       |
| 80m     | 8          | 4           | 32              | ~32                 | 2,031                  | 160×                      |
| 120m    | 8          | 4           | 32              | ~32                 | 2,031                  | 240×                      |
| 240m    | 4          | 8           | 32              | ~32                 | 2,031                  | 480×                      |

**What accumulation fixes:**
- ✅ Same subjects per effective gradient update at all L
- ✅ Same number of effective gradient updates per epoch at all L
- ✅ Same K=5 windows per subject per epoch at all L (already fixed by overlapping-window change)

**What accumulation cannot fix:**
- ❌ FLOPs per effective step — 240m still costs 480× more than 30s. This is **unavoidable and intentional**: it is the cost of processing a longer context window, which is exactly the variable being studied.

---

## 4. Why Option B Is Correct for This Paper

The research question is: **does giving the model more context about the night improve clinical prediction?**

To answer this cleanly, every dimension of training should be identical across L except the context window size itself:

| Dimension | Option A | Option B (with accum) |
|-----------|----------|-----------------------|
| Subjects per gradient update | ❌ varies 480× | ✅ constant (~32) |
| Gradient updates per epoch | ❌ varies 478× | ✅ constant (~2,031) |
| K windows per subject per epoch | ✅ constant (5) | ✅ constant (5) |
| FLOPs per effective step | ✅ constant | ❌ varies 480× |

Option A controls for compute but destroys uniformity in training dynamics. Option B controls for training dynamics and lets compute vary — and the compute variation is the scientifically meaningful cost of the longer context. If the 240m model achieves better AUROC despite costing 480× more per step, that is a meaningful and interpretable result.

---

## 5. A Note on "Same Total Night Time Per Batch"

A suggested alternative was: *ensure each batch contains the same total minutes of PSG coverage regardless of context length — many short windows or fewer long ones, but always the same total time.*

This is Option A restated. The constraint `batch_size × L = constant` gives:

- Anchor to 30s × 32 = 16 minutes: at 240m, batch_size = 16/240 = **0.067** — impossible even before accumulation
- Anchor to 240m × 4 = 960 minutes: at 30s, batch_size = 960/0.5 = **1,920** — feasible but gives 478× fewer steps per epoch at 30s

Even with gradient accumulation the total-context-per-batch constraint cannot be satisfied simultaneously at both ends of the context range, because a single 240m window already contains more total time than any practical 30s batch.

The proposal conflates "the model sees more data per step" with "training is fairer." What matters for fairness is that each subject contributes equally to training — which is controlled by K and effective batch size, not by total minutes per step.

---

## 6. Recommended Implementation

Use gradient accumulation with `micro_batch × accum_steps = 32` (effective batch) at all context lengths:

```yaml
# In experiments/v2_registry.yaml — per-experiment overrides
sex_binary_lstm:        # 30s–10m: no accumulation needed
  batch_size: 32
  accum_steps: 1

# For longer contexts, use per-job overrides:
# 40m:  micro_batch=16, accum=2
# 80m:  micro_batch=8,  accum=4
# 120m: micro_batch=8,  accum=4
# 240m: micro_batch=4,  accum=8
```

Record in `metrics.json` for each run:
- `batch_size` — micro-batch (what actually runs on GPU)
- `accum_steps` — gradient accumulation factor
- `effective_batch_size = batch_size × accum_steps` — what to use in all FLOPs calculations

Total FLOPs formula:
```
total_FLOPs = effective_batch_size × effective_steps_per_epoch × n_epochs × per_window_FLOPs(N, head)
```

---

## 7. Paper Claim After These Changes

> "All models were trained with effective batch size 32 (achieved via gradient accumulation for longer context lengths where GPU memory is limiting), K=5 randomly sampled overlapping context windows per subject per epoch, and identical optimizer, LR schedule, and early stopping criterion across all L. The only variable between experiments is the context length L. Per-step FLOPs increase with L (a 240m window requires 480× more computation per gradient update than a 30s window); this is documented in metrics.json per experiment and used in the scaling-law analysis."
