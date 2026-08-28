# Choosing the Third TSFM Baseline — MOMENT vs. the alternatives

> **Purpose**: decide which general-purpose time-series foundation model
> fills baseline slot #3, before the `moment-implementation` branch starts
> real work. Written 2026-08-22 in response to a direct question: *is MOMENT
> actually the best choice, given that it must work with our existing 128 Hz
> HDF5 data, be tractable to implement, and satisfy a supervisor who asked
> about "recent general-purpose TSFMs, e.g. Chronos"?*
>
> **Recommendation: switch the primary pick from MOMENT to Mantis, and keep
> MOMENT-small as the documented secondary.** The reasoning is below; §2 and
> §6 are the parts that actually decide it. Nothing here is committed — this
> document exists so the decision is made on evidence rather than by
> inheriting the first draft's pick.
>
> Companion to `docs/TSFM_BASELINE_CANDIDATES.md`, which stays the general
> survey. This file is narrower: it answers "which one, and how exactly do we
> feed it our data".

---

## 1. TL;DR

| | **Mantis-8M** (recommended) | **MOMENT** (current plan) | Chronos-2 / TimesFM / TiRex |
|---|---|---|---|
| Task it was pretrained for | **Classification** (contrastive) | Reconstruction, classification-capable | **Forecasting only** |
| Native window | 512 samples (32 patches × 16) | 512 samples (64 patches × 8) | 1000s of steps, but forecasting-shaped |
| Can ingest a full 30 s epoch? | Yes — `num_patches=240` (regen pos buffer) | **Yes — 480 patches, zero surgery** | n/a |
| Multi-channel | **Channel-independent, built in** | Channel-independent, built in | Varies; Chronos-2 group attention (forecasting) |
| Frozen-embedding API | `transform(X, three_dim=True)` → `(N, C, 256)` | `embed(reduction="none")` → `(N, C, P, D)` | **None first-class** — research-grade only |
| Size | **8.1 M** | 40 M / 125 M / **385 M** | 28–120 M+ |
| Est. extraction cost (§5) | **~1 GPU-h** | 10 / 32 / **100 GPU-h** | n/a |
| License | Apache-2.0 | MIT | Apache-2.0 |
| **Published sleep-staging evidence** | **Yes — 8 NSRR-family datasets, beat CBraMod + EEGNet** | **No — explicitly excluded from that study** | No |
| LoRA precedent | None (but tiny → full FT viable) | Own tutorial, **has a real bug** | TimesFM has the best first-party support |

**The single most decisive fact** (§2): a 2025 paper tested exactly our
question — generic TSFMs on sleep staging, on eight datasets including
**SHHS, MrOS, CHAT, SOF, CCSHS, CFS** — and it (a) found Mantis beat the
best EEG-specific foundation model, and (b) **explicitly excluded MOMENT as
impractical**, while reporting that *freezing* the encoder "leads to a huge
decrease in performance" on EEG.

That second half is a direct warning about **our paper's primary condition**,
which is the frozen encoder. It does not invalidate the plan — but it means
we should expect a weak frozen result from *any* generic TSFM, budget for
the LoRA condition to be the one that carries the comparison, and report the
frozen result honestly rather than as a surprise.

---

## 2. The decisive evidence

**Gnassounou, Moakher, Xie, Feofanov & Redko (2025), "Leveraging Generic
Time Series Foundation Models for EEG Classification"**
([arXiv:2510.27522](https://arxiv.org/abs/2510.27522)) is almost exactly our
experiment, one model narrower.

What they did:
- Evaluated **Mantis** (generic TSFM) against **CBraMod** (the most recent
  EEG-specific foundation model) and **EEGNet** (classical CNN baseline).
- **Eight sleep-staging datasets: ABC, CCSHS, CFS, CHAT, HOMEPAP, MASS,
  PhysioNet, SOF** — plus SHHS and MrOS named in their setup. These are
  NSRR cohorts, i.e. **the same data family as ours**.
- 30-second epochs, five sleep stages, subject-wise 60/20/20 splits. Our
  protocol, essentially.

What they found (their Table 3, weighted F1, sleep staging):

| Dataset | EEGNet | CBraMod (EEG-pretrained) | **Mantis (real pretrain)** | **Mantis (synthetic pretrain)** |
|---|---|---|---|---|
| ABC | 67.94 | 74.90 | **75.50** | **75.74** |
| CCSHS | 83.13 | 88.04 | **88.85** | 88.80 |
| CFS | 78.60 | 84.30 | **85.35** | 85.06 |
| CHAT | 78.91 | 85.01 | **85.94** | 85.72 |
| HOMEPAP | 69.43 | 72.56 | 73.14 | **73.53** |
| MASS | 79.85 | 81.12 | **84.09** | 82.49 |
| PhysioNet | 75.73 | 78.97 | **79.82** | 78.83 |
| SOF | 78.74 | 83.39 | **84.69** | 84.31 |

**Mantis beat the EEG-specific foundation model on all eight.** A generic,
8M-parameter, non-physiological model outperformed a purpose-built EEG FM on
sleep staging. That is a genuinely strong result for the "generic TSFM"
arm of our paper — and it is the arm the supervisor asked for.

Three quotes that matter more than the table:

1. > *"we have found that freezing the encoder for EEG data leads to a huge
   > decrease in performance, so fine-tuning is necessary in this context.
   > This is why we have not considered MOMENT, which is very difficult to
   > fine-tune due to its large model size compared to CBraMod and Mantis."*

   **Both halves are about us.** Our Stage 1 *is* the frozen condition, and
   our Stage 2 (LoRA) exists precisely because full fine-tuning is
   impractical. This says: expect Stage 1 to be weak for a generic TSFM, and
   expect MOMENT specifically to be the hardest one to rescue in Stage 2.

2. > *"Unlike CBraMod, which relies on multivariate pretraining, Mantis
   > processes channels univariately and only models their inter-dependencies
   > at the final classification layer. The fact that Mantis still outperforms
   > the multivariate approach suggests its architecture offers a more
   > efficient method for preserving and leveraging spatial information."*

   This is **exactly our architecture** — per-channel embeddings, combined by
   a downstream head. It is evidence that the design we already use is not a
   handicap.

3. > *"in scenarios with limited spatial information, CBraMod's multivariate
   > architecture is less effective, whereas Mantis's more general,
   > channel-independent design holds a distinct advantage."*

   Sleep PSG is exactly the low-channel-count regime (they say "typically
   1-7"). Our fast-channel tree has **6**.

**Caveat, stated plainly**: Feofanov and Redko are authors on *both* this
paper and the Mantis paper — this is the model's own authors evaluating it.
That weakens it as independent evidence. It is partly mitigated by their
CBraMod and EEGNet numbers being taken from *CBraMod's own paper* rather than
re-run by them, but treat the margin as optimistic and the direction as
credible, not the exact deltas.

---

## 3. Answering your actual question: how do these models take our data?

This is the part that was genuinely unclear, so here it is concretely.

### 3.1 They do not "understand" channels at all — and that is fine

Both MOMENT and Mantis are **channel-independent**. Neither has any notion of
"EEG" vs "ECG", and neither learns cross-channel structure. Verified in code:

- **Mantis** (`src/mantis/trainer/trainer.py:290-320`) — docstring says it
  outright: *"In the multivariate case, each channel is sent independently to
  the foundation model."* Input `(n_samples, n_channels, seq_len)`, output
  `(n_samples, n_channels * 256)` or, with `three_dim=True`,
  **`(n_samples, n_channels, 256)`**.
- **MOMENT** (`momentfm/models/moment.py:264-274`) — reshapes to
  `(batch * n_channels, n_patches, d_model)` before the shared encoder, so
  *"pretrained weights never see a channel dimension"*. With
  `reduction="none"` you get `(batch, n_channels, n_patches, d_model)`.

**So you do not need to do majority voting or averaging yourself, and you
should not.** The "one channel at a time" you were imagining is already what
these models do internally. The question is only what you do with the
per-channel embeddings afterwards — and the right answer is:

> **Keep them separate and let our existing sequence head combine them.**

That is precisely what our SleepFM pipeline already does: it concatenates 4
modality embeddings of 128 dims into a 512-dim per-patch vector and feeds
that to `LSTMHead`/`TransformerHead`. Averaging or voting across channels
would *throw away* information the head is designed to use, and would break
the paper's "architecture held constant, only the encoder changes" claim.

**Why this is lucky**: `transform(X, three_dim=True)` returns
`(N, C, 256)`, which is structurally identical to SleepFM's per-patch
`[T, 4, 128]`. Our `ContextWindowDataset` is hardcoded to `[T, 4, 128]`
(`N_MODALITIES=4, EMBED_DIM=128, FLAT_DIM=512`) — the MOMENT/Mantis fork only
has to change three constants to `[T, 6, 256]` / `FLAT_DIM=1536`. That is the
smallest adapter of the three baselines by a wide margin.

### 3.2 The real problem is not channels — it is the 512-sample wall

Our data, confirmed from a real file
(`/scratch/boshra95/psg/shhs/derived/hdf5_signals/203805_v2.h5`):

- **128 Hz**, already resampled and z-scored, `float16`, one 1-D array per channel
- fast-channel tree: **6 channels** — `EEG, LOC, ROC, EKG, EMG, Airflow`
- full-channel tree: 9 — adds `ABD, HR, SpO2, Thor`
- **No EDF reprocessing needed for any candidate here.** Every model in this
  document takes a plain float array. This constraint does not discriminate
  between them at all — it is already satisfied.

The mismatch that *does* matter:

```
our 30 s epoch @ 128 Hz  = 3840 samples
MOMENT / Mantis window   =  512 samples  = 4.0 s
                           -> 7.5 windows per epoch
```

**⚠️ CORRECTION (2026-08-22, after review).** An earlier version of this
section called 512 a "hard wall" for both models and told the implementer to
"treat 512 as a hard constraint in practice". **That was an overstatement and
it is wrong.** `docs/TSFM_BASELINE_CANDIDATES.md` §2.3 was actually more
careful than I was — it said longer sequences were *untested*, not
*impossible*, and I hardened that into a constraint it never claimed.

Verified against the real `AutonLab/MOMENT-1-large` checkpoint (safetensors
header read directly, not inferred):

```
patch_embedding.value_embedding.weight   shape=[1024, 8]      <- patch_len IS baked in at 8
patch_embedding.position_embedding.pe    shape=[1, 5000, 1024] <- sinusoidal buffer, 5000 positions
```

Two consequences, and they point in opposite directions:

- **Sequence length is NOT architecturally capped at 64 patches.** The
  positional embedding is a non-learned sinusoidal buffer covering 5000
  positions (`register_buffer`, `require_grad=False`,
  `momentfm/models/layers/embed.py:10-28`), and the backbone is T5 with
  *relative* attention bias. Feeding many more than 64 patches is fine.
  The "512" in the repo's `classification_dataset.py` and the tutorial's
  *"currently only support 512"* argparse help are statements about the
  **tested path**, not the architecture.
- **But `patch_len` cannot be changed on a pretrained checkpoint.**
  `PatchEmbedding` builds `nn.Linear(patch_len, d_model)`
  (`embed.py:200`), so the pretrained weight is literally `[1024, 8]`.
  Setting `patch_len=64` makes that layer `[1024, 64]` — a shape mismatch,
  so those weights cannot load. You would get a **randomly initialised input
  projection feeding a pretrained 24-layer T5 encoder in a basis it has never
  seen**, which discards the tokenizer and almost certainly destroys frozen
  performance. Any advice to "set `patch_len = 64`" is wrong for this reason,
  even though its underlying intuition (let MOMENT read a whole 30 s epoch and
  attend across it) is right.

The correct way to get that intuition is **Option D** below.

Four ways to bridge it:

**Option A — downsample the 30 s epoch to 512 samples.**
Effective rate 17.07 Hz, **Nyquist 8.5 Hz**. This destroys sleep spindles
(11–16 Hz), beta, and essentially all EMG and ECG morphology. For a sleep
paper this is not a defensible preprocessing choice — it would hand a
reviewer an easy objection that the baseline was crippled before it started.
**Do not do this.**

**Option B — split each 30 s epoch into 512-sample windows, embed each, pool
→ one 30 s embedding.**
Keeps the full 128 Hz bandwidth and the 30-second epoch as the atomic unit.
Resample 3840 → 4096 for exactly 8 clean windows. Workable, but strictly
worse than Option D: the model never sees across window boundaries, so an
event spanning two windows is split, and the pooling step is an arbitrary
heuristic we would have to justify.

**Option C — abandon the 30 s epoch; treat the 512-sample (4 s) window as the
atomic unit.**
Closest to SleepFM's *own* design (5 s / 640-sample patches at 128 Hz — only
20 % longer than MOMENT's 4 s). Arguably the most faithful "let each model use
its native granularity" framing. But it makes the context sweep's units differ
from OSF and PhysioOmni, so cross-baseline comparison needs care. Worth
considering as a secondary analysis, not the primary.

**Option D — feed the whole 3840-sample epoch at native `patch_len=8`,
giving 480 patches.** ***(recommended)***
This is the correct implementation of the "let MOMENT read the whole epoch"
idea. **Every pretrained weight stays valid** — `patch_len` is untouched at 8,
so `value_embedding` loads normally, and 480 < 5000 so the sinusoidal
positional buffer already covers it with no surgery at all. Self-attention
then spans the entire 30 s epoch, so an event that straddles what would have
been a window boundary in Option B (a spindle, a K-complex, an arousal) is
modelled as one continuous waveform. It yields **one embedding per epoch
naturally**, with no pooling heuristic to defend.

**And it is free.** Measured by FLOP count for MOMENT-large
(`8·n·d² + 4·n²·d + 6·n·d·d_ff`, 24 layers, d=1024, d_ff=2816):

| | per channel-epoch |
|---|---|
| Option B — 8 passes × 64 patches | 318.9 GFLOP |
| **Option D — 1 pass × 480 patches** | **318.6 GFLOP (1.00×)** |

Identical, because the linear-in-`n` projection and MLP terms dominate over
the quadratic attention term at these lengths. So Option D is a strictly
better design at the same cost — §5's model-size comparison is unaffected.

**One real caveat, worth stating in the paper**: `flan-t5-large` has
`relative_attention_num_buckets=32`, `relative_attention_max_distance=128`
(verified from the checkpoint config). At `patch_len=8` and 128 Hz, 128
patches = **8 seconds**. So relative position is resolved finely *within*
~8 s and bucket-saturates beyond it — the model can tell "2 patches apart"
from "10 patches apart", but not "300 apart" from "400 apart". For the
events that matter at this scale (spindles 0.5–2 s, K-complexes, arousals)
that is comfortably fine. It does mean the intra-epoch attention is
genuinely local, not global, and 480 patches is 7.5× longer than anything
MOMENT saw in pretraining — an out-of-distribution risk that a pilot should
check rather than assume away.

**Mantis can do the same thing**, with one extra step: set `seq_len=3840,
num_patches=240`, which keeps `patch_window_size = 3840/240 = 16` and hence
`kernel_size=17`, identical to pretraining, so the conv tokenizer weights
stay valid (`architecture/version1.py:25-40`). Its positional encoding is
also sinusoidal (`transformer_v1_utils/positional_encoding.py`), but it is a
`register_buffer` sized `num_patches+1`, so it appears in the state dict and
will size-mismatch on load — it is deterministic, so regenerate rather than
load it. Slightly more surgery than MOMENT needs, but sound.

**Recommendation: Option D**, for whichever model is chosen, with Option C
noted in the paper as the alternative framing that was considered.

### 3.3 Why the forecasting-native models are worse for this specifically

Beyond the missing embedding API, there is a signal-processing reason:
Chronos-2 and TimesFM use **fixed patching schemes biased toward low
frequencies** — good for long-horizon forecasting, bad for high-frequency
physiological content. Our EEG's discriminative information lives in exactly
the band those schemes de-emphasise. Combined with having no first-class
frozen-embedding API, they are a poor fit despite being the models the
supervisor named.

**If the supervisor specifically wants Chronos answered**, the honest answer
is a paragraph in the paper, not a fourth implementation: Chronos-2 is
forecasting-only with no representation-extraction API, and its patching is
biased against the frequency band that carries sleep information. That is a
real, defensible reason — much stronger than "we ran out of time".

---

## 4. Candidate-by-candidate

### 4.1 Mantis-8M — **recommended**
- [github.com/vfeofanov/mantis](https://github.com/vfeofanov/mantis) · [paris-noah/Mantis-8M](https://huggingface.co/paris-noah/Mantis-8M) · `pip install mantis-tsfm` · Apache-2.0
- Huawei Noah's Ark Lab, [arXiv:2502.15637](https://arxiv.org/abs/2502.15637).
- **Classification-native** (contrastive pretraining), not a repurposed forecaster.
- Architecture (`src/mantis/architecture/version1.py:218-241`): `seq_len=512`,
  `num_patches=32` → `patch_window_size=16`, `hidden_dim=256`.
  `assert (seq_len % num_patches) == 0`.
  **Note**: `num_patches` is fixed at construction, so naively raising
  `seq_len` alone just makes each patch bigger (3840 → patch size 120 vs. the
  pretrained 16 — badly out of distribution). The fix is to raise *both*:
  `seq_len=3840, num_patches=240` keeps `patch_window_size=16` and the conv
  `kernel_size=17` exactly as pretrained. See §3.2 Option D — an earlier
  version of this doc wrongly called 512 a hard constraint here.
- Four checkpoints: `Mantis-8M`, `MantisPlus`, `Utica`, `MantisV2`.
- **A purely synthetic-pretrained variant exists** (CauKer, 1M synthetic
  samples) that matches real-data pretraining performance. For our paper this
  is unusually attractive: **provably zero physiological or NSRR exposure** —
  the cleanest possible contamination story, in direct contrast to OSF's
  quantified 87.7% SHHS overlap.
- Trivial frozen API; adapters (`MultichannelProjector`,
  `LinearChannelCombiner`) exist if we ever want channel reduction.
- **Weaknesses, honestly**: no LoRA/PEFT precedent (though at 8M, *full*
  fine-tuning is cheap — arguably better than LoRA here, but it breaks
  method-parity with the other two baselines, so think this through);
  its real-data pretraining mix contains "a small portion of EEG data" (use
  the synthetic checkpoint to sidestep, or report both); 8M parameters may
  read as "not SOTA scale" to a reviewer.

### 4.2 MOMENT — **keep as documented secondary**
- Everything in `docs/TSFM_BASELINE_CANDIDATES.md` §2.3 still holds and was
  re-verified here: `seq_len=512`, `patch_len=8`, `stride=8` → 64 patches;
  `MOMENT-1-large` is `google/flan-t5-large`, `d_model=1024`, 24 layers.
- Real advantages: MIT, three sizes, an actual `embed()`/`classify()` API, a
  documented (if buggy) LoRA recipe, and the strongest name recognition of
  the classification-capable options.
- Real disadvantages: **10–100× the extraction cost** (§5); the LoRA tutorial
  is missing `modules_to_save` so the head is likely frozen as written; and
  it was *explicitly excluded* from the one study that ran this exact
  experiment, for reasons that apply to us.
- **If chosen, use `MOMENT-1-small`, not `-large`.** `-large` is not
  tractable under Option B (~100 GPU-h for extraction alone, before any
  training), and there is no evidence the extra capacity helps on this task.

### 4.3 Chronos-2 — **do not implement; answer in prose**
- 120M / 28M, Apache-2.0, [arXiv:2510.15821](https://arxiv.org/abs/2510.15821).
- Genuinely SOTA at what it does, and multivariate-capable via group
  attention — but **forecasting-only, with no classification or
  embedding-extraction API anywhere in the repo**. Building one means
  bypassing its output head and reusing the encoder in a way its authors
  never validated.
- Plus the low-frequency patching bias (§3.3).

### 4.4 TimesFM / TiRex / Moirai-2 / Time-MoE — **no**
Same objection as Chronos-2: forecasting-native, no first-class
representation API. TimesFM has the best first-party LoRA support of the
group; TiRex (35M, xLSTM, [arXiv:2505.23719](https://arxiv.org/abs/2505.23719))
is the current zero-shot forecasting SOTA and there is a NeurIPS-2025-workshop
result on using forecasters as frozen classification feature extractors — but
that is a research finding, not a supported API, and it is exactly the kind
of bespoke adaptation you said you want to avoid.

### 4.5 UniShape — **no, on availability grounds**
AAAI-26, classification-pretrained on 1.89M samples, which is conceptually
the right kind of model. But checked directly: **no license file, no
confirmed downloadable checkpoint, no documented multivariate handling, no
feature-extraction API in the README.** Too much unverified risk for a
paper-critical baseline.

### 4.6 OTiS — **interesting, but wrong slot**
45M, pretrained across ECG / EEG / audio / weather / engineering, and
explicitly designed for heterogeneous sampling rates and channel counts —
which is a genuine engineering fit. But its pretraining is **~62% ECG by
sample count**, which makes it a *physiological* model wearing general-purpose
clothes. That collides with PhysioOmni's slot and weakens the domain-gradient
framing in §6. Note it as considered; don't pick it.

---

## 5. Compute — the practical argument

Order-of-magnitude estimate for Stage 1 embedding extraction over the full
cohort. **These are FLOP-counting estimates (`2 · params · tokens`), not
measurements** — treat them as ratios, not absolute wall-times.

Scale: ~15,000 subjects × ~1,080 epochs × 6 channels = **97 M channel-epochs**.
Under Option B (8 × 512-sample windows per epoch) that is ~729 M forward
passes. **Option D (§3.2) costs the same to within 0.1 %** — 1 pass of 480
patches ≈ 8 passes of 64 — so this table stands unchanged either way, and the
model-size ratios below are what actually matter.

| Model | GFLOP / window | Total | ≈ GPU-h @ 100 TFLOP/s eff. |
|---|---|---|---|
| **Mantis-8M** | 0.52 | 0.38 EFLOP | **~1** |
| MOMENT-small | 5.12 | 3.7 EFLOP | ~10 |
| MOMENT-base | 16.0 | 11.7 EFLOP | ~32 |
| MOMENT-large | 49.3 | 35.9 EFLOP | ~100 |

You have already lived the consequences of underestimating this: PhysioOmni's
Stage 2 is currently running ~18 h per epoch at 80m context. Mantis's 8M
parameters are not a compromise here — given that the published evidence says
it *outperforms* a 385M-parameter-class alternative's peer group on our exact
task, the small size is close to free.

---

## 6. What this does to the paper's story

Slot #3 should stay **domain-general**, and this choice sharpens rather than
weakens that. The four-model progression becomes a clean **pretraining-domain
gradient**:

| | Pretraining domain | Cohort contamination |
|---|---|---|
| **SleepFM** | Sleep PSG (ours) | n/a — our own protocol |
| **OSF** | Sleep PSG (external) | **Severe** — 87.7% of our SHHS test split |
| **PhysioOmni** | General physiological (no sleep focus, no respiratory pathway) | None found |
| **Mantis** | **General time series — or purely synthetic** | **Provably zero** |

That last row is the strongest version of the argument, and it is only
available with Mantis: the CauKer checkpoint is pretrained on **synthetic data
only**, so it has demonstrably never seen a physiological signal, let alone an
NSRR cohort. Against OSF's quantified contamination problem, that is a
genuinely valuable contrast to be able to draw — and running *both* Mantis
checkpoints (real-pretrained and synthetic-pretrained) gives a clean internal
ablation on "does physiological pretraining data matter at all", which no
other candidate here offers.

**Expect — and pre-commit to reporting — a weak frozen result.** §2's finding
that freezing degrades badly on EEG is the single most likely outcome to be
surprised by. If Stage 1 is poor for Mantis and Stage 2 (LoRA, or full
fine-tuning given 8M params) rescues it, that is a *finding*: it says
general-purpose pretraining transfers to sleep PSG only with adaptation. That
is a publishable, honest result and it directly answers the supervisor's
question. It is not a failed experiment, and it must not be presented as a
completed unremarkable cell in a table.

---

## 7. Recommendation

1. **Primary: Mantis-8M**, Option B windowing (§3.2), per-channel embeddings
   kept separate and combined by our existing sequence heads (§3.1).
2. **Run both Mantis checkpoints** — real-pretrained and synthetic
   (CauKer) — as an internal ablation on pretraining-domain relevance.
   Cheap at 8M parameters; unique among the candidates.
3. **Keep MOMENT-small** as the documented secondary, to be run if a reviewer
   asks for a larger / better-known general TSFM. Note in the paper *why*
   `-large` was not run (compute, §5) rather than omitting it silently.
4. **Answer Chronos in prose**, not in code (§3.3, §4.3).
5. **Still expect the Stage 2 OOM** — that warning in
   `MOMENT_CLAUDE.md` / the plan doc's §4 is architecture-independent and
   applies to Mantis exactly as much as to MOMENT. Mantis's small size buys
   headroom, but the memory still scales with
   `batch_size × raw-epochs-per-context-window`.

### If you accept this
The two skeleton files need renaming and revising, and the branch name no
longer matches. Least disruptive path: **keep the `moment-implementation`
branch and worktree names** (renaming a pushed branch and a worktree is more
disruptive than the mismatch is worth) and treat them as "third TSFM
baseline". Rename only the *files* — `MOMENT_CLAUDE.md` →
`TSFM3_CLAUDE.md` (or `MANTIS_CLAUDE.md`), likewise the plan doc — and record
the decision at the top of both. Alternatively, if you would rather the names
match exactly, delete and recreate the branch/worktree now, before any
MOMENT-specific code exists; that is cheapest today and only gets more
expensive.

**This is your call, not mine to make** — §2's evidence is strong but comes
with the authorship caveat noted there, and "MOMENT is the better-known name
to a reviewer" is a legitimate reason to overrule it that I can't weigh for
you.

---

## Sources

- [Leveraging Generic Time Series Foundation Models for EEG Classification (arXiv:2510.27522)](https://arxiv.org/abs/2510.27522) — the decisive evidence, §2
- [Mantis: Lightweight Foundation Model for Time Series Classification (arXiv:2502.15637)](https://arxiv.org/abs/2502.15637)
- [github.com/vfeofanov/mantis](https://github.com/vfeofanov/mantis) · [paris-noah/Mantis-8M](https://huggingface.co/paris-noah/Mantis-8M)
- [MOMENT (arXiv:2402.03885)](https://arxiv.org/abs/2402.03885) · [github.com/moment-timeseries-foundation-model/moment](https://github.com/moment-timeseries-foundation-model/moment) · [AutonLab/MOMENT-1-large config](https://huggingface.co/AutonLab/MOMENT-1-large)
- [Chronos-2 (arXiv:2510.15821)](https://arxiv.org/abs/2510.15821) · [github.com/amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
- [TiRex (arXiv:2505.23719)](https://arxiv.org/abs/2505.23719) · [github.com/NX-AI/tirex](https://github.com/NX-AI/tirex)
- [UniShape, AAAI-26 (arXiv:2601.06429)](https://arxiv.org/abs/2601.06429) · [github.com/qianlima-lab/UniShape](https://github.com/qianlima-lab/UniShape)
- [Towards Generalisable Time Series Understanding Across Domains — OTiS (arXiv:2410.07299)](https://arxiv.org/pdf/2410.07299)
- [TSFM-Bench (arXiv:2410.11802)](https://arxiv.org/html/2410.11802v6)
