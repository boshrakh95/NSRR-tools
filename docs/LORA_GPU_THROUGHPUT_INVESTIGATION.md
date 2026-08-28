# LoRA Stage 2 GPU Throughput Investigation (OSF, 2026-08-27)

**Status: investigated, verdict given, nothing applied yet.** This documents
a GPU-throughput question raised while OSF's Stage 2 (LoRA) sweep was
running, what was verified, what the real measured data shows, and the
recommendation — to try later, on OSF, deliberately.

**Relevance beyond OSF:** the root cause here (per-epoch chunked backbone
forward pass through a frozen model, LoRA gradients flowing through it) is
architectural, not OSF-specific. Any model fine-tuned the same way —
including MOMENT/Mantis — will very likely hit the same overhead-bound
pattern at short contexts and the same open question about TF32/GPU-type at
long contexts. Worth reading before assuming a GPU-allocation or
mixed-precision change will or won't help a new model's LoRA stage; the
right answer depends on where THAT model's per-epoch time actually goes,
which needs its own version of the profiling proposed below (§5), not a
copy of OSF's numbers.

**Note on file location:** this lives on `osf-implementation`. If a branch
built on top of `physioomni-implementation` (e.g. a `mantis-implementation`
branch) wants it, either merge `osf-implementation`'s latest into
`physioomni-implementation` first (confirmed clean, zero file overlap — see
prior session), or just copy this one file over.

---

## 1. The claim that triggered this

A session working on PhysioOmni found its own LoRA training running at a
tiny fraction of H100 peak throughput and attributed it to three
multiplicative causes, proposing to fix all three for OSF too:

1. PyTorch defaults `torch.backends.cuda.matmul.allow_tf32=False` and
   `torch.set_float32_matmul_precision("highest")`, so matmuls run at true
   FP32 (~67 TFLOP/s on H100) instead of TF32 tensor cores (~495 TFLOP/s).
2. OSF requests `nvidia_h100_80gb_hbm3_3g.40gb:1` (3/7 of a card) when
   whole `h100:1` cards exist on Fir, and `sbatch --test-only` showed no
   queue-time penalty for the bigger request.
3. Small hidden dimensions underutilise tensor cores — flagged as less of a
   concern for OSF specifically, since OSF's backbone is d=768 (well-shaped)
   vs. PhysioOmni's d=100 (not even a multiple of 8).

Proposed fix: 3 lines of `torch.backends`/`torch.set_float32_matmul_precision`
at the top of the training script, plus `--gpus=h100:1` and
`--cpus-per-task=8` in the job script.

**Counter-evidence already in this project**, from `configs/phase0_osf_lora_config.yaml`:
`mixed_precision` was tried and reverted 2026-08-15 — measured **zero**
speedup on a 1g.10gb pilot (61.7 vs 59.9 min/epoch). That's real evidence
against the same class of fix, from OSF's own runs.

## 2. What was verified (not taken on trust)

Checked live in `osf_env` and by reading the actual scripts:

- `torch.backends.cuda.matmul.allow_tf32` = **False** (confirmed).
- `torch.get_float32_matmul_precision()` = **"highest"** (confirmed).
- **Correction to the original claim**: `torch.backends.cudnn.allow_tf32`
  is actually **True** by default in torch 2.5.1, not False as claimed —
  only the matmul flag needs changing, not both.
- `grep` for `allow_tf32`/`float32_matmul` in `train_osf_lora.py` and
  `train_osf_context_sweep.py`: nothing. Confirmed correct.
- Job script requests `nvidia_h100_80gb_hbm3_3g.40gb:1`, `--cpus-per-task=4`.
  Confirmed correct.
- `sbatch --test-only` for MIG (4 cpus) vs. whole-card `h100:1` (8 cpus),
  same `--time`: **identical estimated start time.** Supports "no obvious
  queue penalty" — but `--test-only` is a heuristic snapshot of one moment,
  not a guarantee across hours of real scheduling.
- Checked whether Stage 1/SleepFM secretly uses AMP (the shared
  `train_osf_context_sweep.py` defaults `mixed_precision` to `True`) — no:
  `phase0_osf_config.yaml`, `phase0_v3_config.yaml`, and
  `phase0_v3_full_config.yaml` all explicitly override it to `false`. So
  the original rationale for reverting Stage 2's AMP ("numerical asymmetry
  vs. Stage 1/SleepFM's fp32 training") is factually sound — Stage 1 really
  does train in plain fp32.

## 3. A bug found along the way: `metrics.json`'s `training_time_min` is wrong for multi-resume runs

While building a per-epoch timing table from completed contexts'
`metrics.json`, the numbers were nonsensical (10m faster than 30s, 40m
faster than 80m, some zeros). Root cause, in `train_osf_lora.py`:

```python
t0 = time.time()          # reset on EVERY invocation, including resumes
...
"training_time_min": elapsed / 60,   # only covers the LAST resume segment
"n_epochs_run": len(history),        # correctly counts ALL epochs across all segments
```

`t0` resets every time the script (re)starts, but `history` (and therefore
`n_epochs_run`) correctly persists across resumes via `resume.pt`. So
`training_time_min` silently **undercounts** true wall-clock cost for any
context that needed more than one resume cycle — the numerator only covers
the final segment while the denominator counts every epoch ever run.

**Not fixed, not blocking any decision here** — just don't trust
`metrics.json`'s `training_time_min / n_epochs_run` for a multi-resume run
until this is fixed (record cumulative time in `resume.pt` and carry it
forward). Worth a small fix later, independent of the throughput question.

## 4. Real measured throughput (from verified, single-segment data only)

Because of §3, only numbers cross-checked via `resume.pt` + `sacct` during
a run's *first, uninterrupted* segment were used (not the buggy
`metrics.json` aggregate):

| context | n_train | N (raw epochs/window) | min/epoch | est. TFLOP/s |
|---|---|---|---|---|
| 30s | 48,380 (apnea) | 1 | 18.77 | ~2.0 |
| 10m | 48,380 (apnea) | 20 | 58.9 | ~12.8 |
| 40m | 33,335 (sex) | 80 | 112.26 | ~18.5 |
| 80m | 33,335 (sex) | 160 | 217.22 | ~19.2 |

TFLOP/s from a hand FLOP estimate for one ViT block (d=768, 90 tokens per
30s epoch, 12 blocks, ~3x forward for fwd+bwd ≈ 46.8 GFLOP per raw epoch —
ignores the sequence head and patch embedding, so treat absolute values as
order-of-magnitude only). **The trend is the trustworthy part** (formula
errors cancel in a ratio): utilization jumps **~6.4x from 30s to 10m**,
then **plateaus** 40m→80m (+3% only). Against a ~28.7 TFLOP/s FP32-dense
peak for a 3g.40gb slice (67 × 3/7), the 40m/80m plateau is ~65% of that
peak — high enough that the absolute number shouldn't be over-trusted, but
consistent with a clear mechanism: `chunk_batch_size=64` amortizes
per-call overhead far better once a window has enough raw epochs to fill
several full chunks (10+ chunks/batch at 40m+, vs. one partially-empty
chunk at 30s). This is the same mechanism that made the earlier
`chunk_batch_size` fix (16→64) give a real 3.28x speedup, and it directly
explains why AMP and the 1g→3g GPU upgrade both measured **zero** speedup
— both were tested at 30s, exactly where overhead dominates.

**Conclusion: OSF is heavily overhead-bound at 30s/10m, and substantially
more compute-utilizing at 40m and beyond.** Not a uniform answer across the
sweep — this is the key thing any other model's own investigation should
check for itself before assuming one answer covers all context lengths.

## 5. Verdict

**TF32 (the 3-line change):** likely real but smaller than "0.14%-of-peak,
~500x headroom" implies, and context-length-dependent:
- At 30s/10m: expect close to nothing — same regime where AMP and the MIG
  upgrade already measured zero, for a mechanistic reason (overhead-bound)
  that should transfer to TF32 too.
- At 40m/80m/120m/240m — where most of the sweep's *remaining* compute
  budget actually lives — compute is a much larger share of wall time, so
  there's real room. Honest guess, not a measurement: something like a
  15-40% speedup at these contexts, not multiples, if the ~65%-of-peak
  estimate is roughly right.
- Risk: low. TF32 keeps full FP32 storage/exponent range, only truncates
  mantissa bits during the matmul — a much smaller numerical departure than
  AMP's fp16+GradScaler (which was reverted specifically over a
  Stage-1-consistency concern). Still introduces *some* asymmetry with
  Stage 1 (confirmed plain fp32, §2) — smaller in degree than AMP's, but
  not zero. Decide deliberately, don't default into it.

**Whole-card GPU request:** recommend against without more evidence. The
overhead-bound finding at 30s is exactly the regime where 1g.10gb→3g.40gb
(3x more compute) already gave zero speedup — the same logic argues a
further 2.33x jump to a full card is unlikely to help much for the same
reason, and it costs real shared-account allocation for an unproven
benefit. Don't bundle it with the TF32 decision — evaluate independently,
and only reconsider at the longest contexts (240m) if TF32 alone doesn't
close the gap there.

**`--cpus-per-task` 4→8:** no evidence either way. Free to try (no
compute/precision implications), could help if any overhead is CPU-side
(data loading) rather than pure kernel-launch overhead. Untested.

## 6. Cheapest measurement to settle it (not yet run)

Most direct and cheapest: a short `torch.profiler` run over a handful of
batches **at a longer context (40m, not 30s)** — a few minutes of GPU time,
not a full epoch — measuring the actual time breakdown (matmul kernels vs.
everything else) directly instead of relying on the FLOP estimate above.
Settles compute-bound-vs-overhead-bound definitively. A second, more
expensive but more decision-relevant check: a real 1-epoch A/B at 40m with
TF32 on, compared against the known 112.26 min/epoch baseline — same
pattern as every other validation this session (`chunk_batch_size`,
`context_micro_batch`). Do the profiler run first; only do the A/B if that
doesn't fully settle it.

## 7. Constraints when actually trying this

- ~7 `osf_lora_sweep` jobs were running at investigation time (one ~19.5h
  in, several 120m runs near epoch 17-18/18). Running jobs only pick up
  SBATCH/script edits on their *next* auto-resubmit — leave currently
  running jobs alone regardless of what gets decided; apply any change only
  to fresh job submissions or contexts not yet started.
- Any OSF-side change must stay OSF-only (no edits to files shared with
  other branches) — same discipline that's kept `osf-implementation`,
  `physioomni-implementation`, and any branch built on top of them
  conflict-free at merge time so far.
