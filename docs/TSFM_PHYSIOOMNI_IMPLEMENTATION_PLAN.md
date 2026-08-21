# PhysioOmni Implementation Plan

> **Purpose**: Authoritative record of what will be built for PhysioOmni as
> TSFM baseline #2 (of 3 — OSF is #1, with Stage 1 done and Stage 2 (LoRA)
> in progress on `osf-implementation`; MOMENT is #3, no plan doc yet).
> **Nothing in this plan has been implemented — this is still planning
> only.** Written 2026-08-13, revised 2026-08-17 per
> `docs/PHYSIOOMNI_PLANNING_HANDOFF.md`'s explicit request for a
> fresh, code-verified, implementation-informed pass — this revision adds
> concrete Stage 1 file specs (config schema, dataset-class constants,
> script CLI shapes, job-script pattern) at the same level of detail as
> `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s finished Phase 1, and folds in
> real engineering lessons learned from OSF's now-finished Stage 1 and
> in-progress Stage 2. Format mirrors that same doc — read its top section
> for the pattern this one follows (a live checklist plus an Appendix of
> verification detail).
>
> **This revision's scope, per the hand-off's explicit instruction**:
> Stage 1 (frozen backbone → embeddings → our sequence heads) is planned in
> full, cluster-runnable detail. Stage 2 (LoRA) stays at outline level —
> target modules, staging approach, and the real lessons OSF's own Stage 2
> already surfaced — to be refined once Stage 1 is actually running, not
> before. **No code was written, no branch was created, and no cluster
> command was run to produce this revision** — every claim below is either
> carried over from the 2026-08-13 pass (already code-verified then) or
> newly verified this session by reading `PhysioOmni`'s repo directly (file
> mtimes checked — unchanged since 2026-08-10, nothing to re-verify there)
> and by reading OSF's actual finished implementation files in this repo as
> a structural reference (never modified — see the hard constraint in §1).

---

## 0. Should we build this? — decision: **yes, build it, with caveats stated up front**

The user asked explicitly: read what's known about PhysioOmni's pretraining
data/tasks, decide if it's good enough, and if not, stop here and suggest
looking elsewhere instead. This section is that decision, made directly
rather than deferred, and left unchanged by this revision — nothing found
while reading OSF's finished implementation bears on this judgment.

**Verdict: proceed with the plan below.** PhysioOmni is a real, working,
code-and-weights-released multimodal physiological foundation model that
fits our comparison's "how does a pretrained encoder handle a reduced/
missing channel set" question at least as well as OSF conceptually (its
whole paper thesis is "arbitrary missing modality" robustness — the same
shape as our own reduced/full-channel framing). But it is a **visibly
weaker candidate than OSF** on several honest, quantifiable axes, and the
paper's own reported numbers are a genuine reason for modest expectations,
not just risk-management prose:

**Reasons this is weaker than OSF, found by reading the code and paper (not
guessed):**

1. **Never peer-reviewed.** arXiv:2504.19596, v1 April 2025 → v3 March 2026,
   no venue acceptance found anywhere (checked the paper's own submission
   history). OSF is ICML 2026 (accepted, peer-reviewed). PhysioOmni is not
   necessarily wrong for being a preprint, but it means the downstream
   numbers below haven't been through review.
2. **Pretraining corpus is real but small and not sleep-PSG-scale for the
   sleep-relevant slice of it.** TUEG 26,846 recordings (clinical
   epilepsy-monitoring EEG, not sleep), CAP 108 polysomnographic
   recordings, Sleep-EDF 197 whole-night recordings, DEAP 32 participants
   (emotion, not sleep), a private set of 54 recordings/19 subjects. **The
   actual overnight-PSG-relevant slice of pretraining data is CAP+Sleep-EDF
   ≈ 305 recordings** — two to three orders of magnitude smaller than our
   own ~16,000-subject cohort and far smaller than OSF's pretraining scale
   (which already includes SHHS and MrOS at NSRR scale). This doesn't
   disqualify the model, but it means "pretrained sleep foundation model"
   is a much bigger claim for OSF than for PhysioOmni, whose pretraining is
   dominated by non-sleep clinical EEG.
3. **On its own best-fit downstream task (HMC sleep staging), PhysioOmni
   does not clearly beat a non-foundation-model baseline.** Fetched
   directly from the paper: PhysioOmni scores **0.7377±0.0056 balanced
   accuracy** (all 3 modalities) on HMC 5-class sleep staging; the paper's
   own `FeatFusion` baseline (a hand-engineered-feature method, not a
   foundation model) scores **0.7478±0.0038** — higher. This is the single
   most important fact behind this section's honest framing: the paper's
   own numbers, on the task most like ours, do not show the pretrained
   model beating a much simpler baseline. That is a legitimate, informative
   result to include in a baseline-comparison paper (arguably it's exactly
   the kind of finding this paper is designed to surface), but it means
   expectations for PhysioOmni's absolute performance should be modest
   going in — this is not being buried, it's the headline reason this
   section exists.
4. **No RESP/airflow pathway anywhere** — confirmed independently by four
   code locations (§3 below). **Apnea is excluded from the PhysioOmni
   comparison**, same as already decided in `CLAUDE.md`.
5. **No LoRA/PEFT code anywhere in the repo** (`grep -rniE "peft|lora"` →
   zero hits) — Stage 2 needs LoRA wiring built from scratch, whereas OSF's
   equivalent work was "find the two Linear layer names and pass them to
   `LoraConfig`." More implementation effort for less validated upside —
   confirmed still true, and now sharpened by §15's read of what OSF's own
   from-scratch LoRA work actually took (a real, multi-day engineering
   effort even *with* a single-backbone architecture; PhysioOmni's
   four-encoder structure is a harder version of the same problem, see
   §15).
6. **Native per-forward-pass context ceiling is shorter and less uniform
   than the original candidates-doc estimate implied** — see §3's
   correction: it is not a flat "512 seconds" across all four modalities;
   it depends on which resample rate is used per modality, and for
   ECG/EMG at the rate PhysioOmni's own downstream scripts use, it's closer
   to 100 seconds. Not a blocker for our 30-second-epoch Plan B design (see
   §6), but a real correction to how "8.5-minute ceiling" should be stated
   if this shows up in the paper's methods section.
7. **License is split and partially undocumented**: the HuggingFace weights
   repo (`Weibang/PhysioOmni`) declares **CC-BY-4.0** explicitly (verified
   directly via the HF API) — a real, usable, permissive license, just
   attribution-required rather than MIT's no-strings-attached. The GitHub
   code repo itself still has no LICENSE file — a genuine open question for
   how the *code* (not the weights) may be reused/modified, worth flagging
   in the paper if PhysioOmni is included, same spirit as OSF's own
   license-tracking discipline.

**Reasons to proceed anyway:**

- It is still one of very few genuinely multimodal (not single-signal)
  physiological foundation models with both code and weights actually
  released and downloadable (verified: `PhysioOmni.pt` 267.8MB,
  `VQ.pt` 238.4MB, both resolve and download from HuggingFace — see §2).
- The channel mapping (§4) needs **no raw-EDF reprocessing, and — better
  than OSF — no full-channel HDF5 tree at all.** PhysioOmni's needed
  channels (EEG C3/C4, EOG LOC/ROC, ECG, EMG-chin) are already kept by the
  fast-channel `psg/` tree's own priority-order channel caps, confirmed
  both from config math and real HDF5 key listings for all 4 cohorts. This
  means PhysioOmni is compared against SleepFM's **paper-primary**
  `phase0_v3` numbers, not the secondary `phase0_v3_full` numbers OSF has
  to use — a more directly relevant comparison for the paper's headline
  results, and the implementation cost is mostly about model/training
  code, not a new preprocessing campaign.
- Even a result where PhysioOmni underperforms SleepFM (and possibly even
  underperforms OSF) is legitimate, reportable content for a paper whose
  explicit purpose is "why SleepFM and not some other pretrained encoder" —
  a negative result here is not wasted effort, provided it's measured fairly
  and reported honestly (same standard already applied to OSF's
  contamination caveats, and to OSF's own genuinely-mixed apnea/
  sleep-efficiency Stage 1 results — see `CLAUDE.md`'s Status section).
- This mirrors the selection already made and recorded in `CLAUDE.md` on
  2026-08-05 (PhysioOmni picked as baseline #2 specifically for its
  missing-modality framing) — nothing found in this deeper pass, or in this
  revision's read of OSF's finished implementation, overturns that choice.

**Practical recommendation for sequencing, not part of the yes/no
decision**: if compute or time becomes the binding constraint, **PhysioOmni
Stage 2 (LoRA) is the most defensible thing to deprioritize or drop first**
among the six frozen/LoRA × three-model cells in the overall TSFM
comparison — weakest a priori case for showing something new (§0.3 above)
and, per §15's expanded reading of OSF's real Stage 2 engineering cost,
the highest relative implementation effort of any cell in that grid (OSF's
own single-backbone Stage 2 took a multi-day, multi-bug-fix effort even
with LoRA target modules handed to it for free by the checkpoint's own
naming; PhysioOmni's four-separate-encoders structure has no such
shortcut). This is a sequencing note for whoever picks up Phase 2, not a
reason to skip Phase 1.

---

## 1. Overview

PhysioOmni (arXiv:2504.19596, NTU + SJTU) is a masked-signal-pretrained,
per-modality-tokenized encoder for EEG/EOG/ECG/EMG, explicitly designed for
robustness to arbitrary missing modalities via a "decoupled multimodal
tokenizer" (private + shared VQ codebooks per modality) and a
"resilient fine-tuning with prototype alignment" downstream procedure. Code
and weights are both released (GitHub `935963004/PhysioOmni`, HF
`Weibang/PhysioOmni`). Same overall comparison question as OSF and SleepFM:
does more temporal context help Tier-1 clinical prediction, and how does
PhysioOmni's encoder compare at matched context lengths/subjects/splits —
minus the apnea task, which is architecturally out of scope (§0.4).

**Repos** (read-only reference clones, not modified — file mtimes checked
2026-08-17, unchanged since the 2026-08-10 clone date, nothing to
re-verify):
- Local: `/Users/boshra/NSRR-workspace/PhysioOmni`
- Cluster: `/home/boshra95/PhysioOmni` (sibling to `NSRR-tools/`,
  `OSF-Open-Sleep-FM/`, `moment/`, same layout convention as `CLAUDE.md`'s
  "Cluster Execution Guidance" section describes)

**Implementation now lives in `/home/boshra95/NSRR-tools-omni`** (a
separate `git worktree`, set up 2026-08-17, checked out to the
`physioomni-implementation` branch, forked from `osf-implementation` — not
`main` — so it inherits the OSF context this plan references throughout).
`/home/boshra95/NSRR-tools` remains a **separate worktree of the same
repository**, checked out to `osf-implementation`, where OSF's own
implementation continues in parallel. `PhysioOmni/` itself stays a
read-only reference clone, sibling to both worktrees under
`/home/boshra95/`.

### Hard constraint: worktree/directory isolation (added 2026-08-17,
**not yet mirrored in `CLAUDE.md` — pending a decision on how to keep
cross-branch status in sync without `CLAUDE.md` diverging per-branch; for
now this is recorded here only**)

**On Compute Canada Fir, this repository is checked out as two separate
`git worktree`s at once**, one per active TSFM baseline implementation, so
each can be worked on from its own VSCode window/Claude Code session
without the two clobbering each other:

| Directory | Branch | What it's for |
|---|---|---|
| `/home/boshra95/NSRR-tools` | `osf-implementation` | OSF baseline (Stage 1 done, Stage 2/LoRA in progress) |
| `/home/boshra95/NSRR-tools-omni` | `physioomni-implementation` | PhysioOmni baseline (this plan) |

**A session working on PhysioOmni operates *only* inside
`/home/boshra95/NSRR-tools-omni`, on the `physioomni-implementation`
branch. It never reads or writes anything under `/home/boshra95/NSRR-tools`
(the `osf-implementation` worktree), and never intends a change for any
branch other than `physioomni-implementation`.** Symmetrically, a session
working on OSF should operate only inside `/home/boshra95/NSRR-tools` on
`osf-implementation`, and never touch `/home/boshra95/NSRR-tools-omni` —
**this expectation currently exists only in this plan doc, not (yet) as an
equivalent note in the `osf-implementation` branch's own `CLAUDE.md`**,
since that would require editing a file this session cannot touch; flag
this to the user/other session directly if it matters before assuming it's
already known there.

**Why this needs care, not just a directory convention**: both worktrees
share the same underlying git history/objects (same `.git`), but each has
its own independent checkout and currently-checked-out branch — editing
files in the wrong worktree either silently targets the wrong branch or
creates conflicts neither session can see coming. **Do not `cd` out of your
own worktree directory to peek at or edit the other one.** If reference
material from OSF's branch is needed (e.g. reusing OSF's finished pipeline
as a structural template, §6-§15 below do this throughout), read it from
*within this worktree's own checkout* — it's already present here, since
this branch forked from `osf-implementation` and inherited its file tree at
that point — not by crossing into the other worktree's directory.

**Why this note isn't also in `CLAUDE.md` right now, and what that
implies**: `CLAUDE.md` exists independently in both branches' history, so
an edit made here would not automatically appear in the `osf-implementation`
worktree's copy until an explicit merge — and since `CLAUDE.md` is a living
status doc both branches keep independently updating, inlining a growing
"current status of both efforts" section into it would create a real,
recurring merge-conflict surface, not a one-time cost. **This is an
explicitly open, undecided question** (options discussed: a shared file
outside git version control, a symlinked `CLAUDE.md`, a separate small
git-tracked status repo) — until it's resolved, **this plan doc is the one
place all of the above is recorded**; don't assume `CLAUDE.md` reflects any
of it.

**Practical consequence for this plan's own references**: every mention
below of "OSF's finished file X" (§6-§15 read OSF's implementation as a
structural template throughout) means "read `X` as it exists in this
`physioomni-implementation` branch's own checkout," not "go look in the
`NSRR-tools` worktree." The file-isolation rule below (which files are
read-only reference material vs. new PhysioOmni-specific files) still
applies exactly as written; the worktree rule above just adds *which
directory* that discipline is enforced in.

### Hard constraint: total file isolation from OSF and SleepFM

**Everything PhysioOmni needs is a new file. Nothing in the existing OSF or
SleepFM pipeline is edited, ever.** This is not a style preference — it's
what let OSF's own implementation coexist cleanly alongside the original
SleepFM pipeline while both kept moving, and it's what will let
`physioomni-implementation` (forked from `osf-implementation`) stay
mergeable in both directions while `osf-implementation` continues its own
Stage 2 sweep. Concretely, **never edit**:

- Any of `scripts/train_osf*.py`, `scripts/train_context_sweep.py`,
  `scripts/infer_subject_windows.py`, `scripts/infer_osf*.py`
- Any `src/nsrr_tools/datasets/osf_*.py` or `context_window_dataset.py`
- `configs/phase0_osf*.yaml`, `configs/phase0_v3*.yaml`
- `experiments/v2_osf*.yaml`, `experiments/v2_registry.yaml`
- Any `jobs/*osf*.sh` or `*context_sweep*.sh`
- `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`, `docs/OSF_EXPERIMENTS_GUIDE.md`,
  or the OSF-specific parts of `CLAUDE.md`

Every one of these is read **as a reference template**, never as something
to modify. §6-§13 below name the exact new, PhysioOmni-specific file each
piece of Stage 1 needs — the same isolation table OSF itself used going
into its own Stage 2 (`docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s Phase 2
checklist header) is reproduced for PhysioOmni-vs-OSF/SleepFM in §6.1
below.

**Reference materials**:
- **arXiv:2504.19596** (fetched directly for this plan; not yet saved
  locally as a PDF the way `related_work/OSF.pdf` was for OSF — do that
  first when implementation starts, mirroring the OSF precedent).
- `docs/TSFM_BASELINE_CANDIDATES.md` §2.2 — the original PhysioOmni research
  pass this plan extends and corrects in a few places (§0, §3).
- `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` — the sibling plan this mirrors in
  structure and, as of this revision, in concrete file-spec detail for
  Stage 1 (§6-§13 below are new this revision, modeled directly on that
  doc's finished Phase 1 sections and File Map).
- `docs/OSF_EXPERIMENTS_GUIDE.md` — the operational how-to-run counterpart
  to the OSF plan; its Step 0-7 (Stage 1) and Step 8 (Stage 2) structure is
  the template this plan's own eventual `docs/PHYSIOOMNI_EXPERIMENTS_GUIDE.md`
  (not written yet — a Phase 1 checklist item, §16) should follow.
- `NSRR-tools/output/channel_analysis/{apples,shhs,mros,stages}_channels.csv`
  — raw per-subject channel dumps behind `configs/channel_definitions.yaml`.
- `CLAUDE.md`'s "TSFM Baseline Model Comparison" section — the three-model
  program, Plan A/B/C definitions, and the frozen/LoRA staged-training
  procedure this plan follows.

## 2. Status (revised 2026-08-17)

**Implementation started 2026-08-17, on its own worktree
(`/home/boshra95/NSRR-tools-omni`).** Phase 0 (env + checkpoint) is real
and verified, not just planned — see checklist 0.1/0.2 below and
`PHYSIOOMNI_CLAUDE.md` (this branch's live status file, not auto-loaded —
read it explicitly) for the full detail. Phase 1 (channel loader,
extraction script) starts next, one step at a time with a VSCode debug
config per step, mirroring the workflow OSF itself used.

Two research/planning passes preceded implementation:

1. **2026-08-13** — reading the PhysioOmni GitHub repo's code directly,
   fetching the arXiv paper and HuggingFace weights-repo metadata over the
   network, and cross-referencing our own `psg/` (fast-channel) HDF5
   channel keys against what PhysioOmni needs. Produced §0, §3, §4, and the
   core of §5/§6's design.
2. **2026-08-17 (this revision)** — per
   `docs/PHYSIOOMNI_PLANNING_HANDOFF.md`'s explicit request: re-verified
   the 2026-08-13 findings are still accurate (PhysioOmni repo file mtimes
   unchanged), then read OSF's actual **finished** Stage 1 implementation
   files (`scripts/extract_osf_embeddings.py`,
   `src/nsrr_tools/datasets/osf_channel_loader.py`,
   `src/nsrr_tools/datasets/osf_context_window_dataset.py`,
   `scripts/train_osf_context_sweep.py`,
   `scripts/infer_osf_subject_windows.py`, `configs/phase0_osf_config.yaml`,
   `experiments/v2_osf_registry.yaml`,
   `jobs/train_osf_context_sweep_gpu.sh`) and OSF's **in-progress** Stage 2
   implementation/checklist (`docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s Phase
   2 section, `docs/OSF_EXPERIMENTS_GUIDE.md`'s Step 8) as concrete
   structural references — not to copy blindly (PhysioOmni's architecture,
   input format, and per-modality context ceilings genuinely differ, see
   §3), but to give this plan's Stage 1 section the same file-level,
   cluster-runnable concreteness OSF's plan has, and to fold in real
   engineering lessons OSF's Stage 1→Stage 2 transition already paid for
   (offline signal caching, split-matching discipline, `peft` gotchas,
   compute-scaling-with-context, effective-batch-size parity — see §15).

**No longer gated on OSF finishing entirely** — per `CLAUDE.md`'s current
Status section, PhysioOmni implementation can start in parallel with OSF's
remaining long-context Stage 2 sweep (which is mostly cluster wait-time
from here). This plan is still not to be acted on (no branch, no code)
until the user explicitly says to proceed past this planning pass.

**Checkpoint availability, verified over the network 2026-08-13 (not
downloaded locally — still true as of this revision)**:
- `https://huggingface.co/Weibang/PhysioOmni` — public, not gated. Two
  files: `PhysioOmni.pt` (267,795,410 bytes ≈ 267.8MB) and `VQ.pt`
  (238,423,693 bytes ≈ 238.4MB), both resolve with HTTP 200 via the
  `resolve/main/` redirect.
- **License: CC-BY-4.0**, declared directly in the HF repo's model card
  metadata (`cardData: {"license":"cc-by-4.0"}` via the HF API) — this
  corrects `docs/TSFM_BASELINE_CANDIDATES.md`'s "no LICENSE file anywhere"
  finding, which was checking the GitHub *code* repo (still true — no
  LICENSE file there) rather than the HF *weights* repo (does have a
  declared license). Both facts are true simultaneously and should both be
  stated if this ships in the paper.
- **`VQ.pt` is very likely not needed for our use case.** Traced
  `train_finetune.py`'s model-loading path end to end: `FT.__init__`
  filters `pretrained_ckpt['model']` for keys starting with
  `EEG_encoder.`/`EOG_encoder.`/`ECG_encoder.`/`EMG_encoder.` and loads only
  those into the four `NeuralTransformer` instances — no reference to VQ
  codebooks anywhere in `FT.py` or `train_finetune.py`. `model/MSM.py`,
  `model/VQ.py`, and `model/FT.py` all name their per-modality encoders
  identically (`self.EEG_encoder = NeuralTransformer(...)`, confirmed by
  grepping all three files), so `PhysioOmni.pt` — whichever pretraining
  stage actually produced it — should strict-or-partial-load into `FT.py`'s
  encoders via this exact mechanism regardless of which stage it came from.
  **Still needs empirical confirmation by actually downloading and
  inspecting the checkpoint's `state_dict` keys once implementation
  starts** — flagged as checklist item 0.2, same discipline OSF used to
  resolve its own checkpoint-filename ambiguity (§5 of the OSF plan).

## 3. Encoder architecture (code-verified 2026-08-13, unchanged this
revision — repo file mtimes confirm nothing to re-check)

- **Four independent per-modality encoders**, each a `NeuralTransformer`
  (`model/neural_transformer.py`) — **not** one shared backbone. No
  cross-modal attention or fusion exists in the pretrained/frozen weights
  themselves; fusion is a downstream-training-time construct (see below).
  **This is the single biggest structural difference from OSF that shapes
  every design choice below** — OSF is one ViT, PhysioOmni is four small
  transformers with no shared weights and no native way to combine them.
- **Per-modality config** (from `train_finetune.py:267-274`, identical in
  `train_msm.py`/`train_vq.py`):

  | Modality | `n_embd` | `patch_size` | `n_layer` | `n_head` |
  |---|---|---|---|---|
  | EEG | 200 | 200 samples | 12 | 10 |
  | EOG | 100 | 100 samples | 12 | 10 |
  | ECG | 100 | 100 samples | 12 | 10 |
  | EMG | 100 | 100 samples | 12 | 10 |

- **Patch embedding**: `TemporalConv` — 3-layer `Conv2d` stack over the
  patch axis (kernel `(1,15)`→`(1,3)`→`(1,3)`, stride `(1,8)` on the first
  layer only) — this is a real temporal feature extractor per patch, not a
  bare linear projection like OSF's `Conv2d` patchify.
- **Position/time embeddings**: `pos_embed = nn.Embedding(256, n_embd)`
  (channel-identity index, looked up by exact string match against a
  132-entry `standard_1020` name list in `dataset.py:8-23`) and
  `time_embed = nn.Embedding(512, n_embd)` (sequential patch-index within
  the modality stream, **not** a real-world-time offset).
- **The native per-forward-pass ceiling is not a uniform "512 seconds"
  across modalities.** `time_embed`'s cap of 512 indices is a cap on
  *patch count*, not seconds — the real-world duration one 512-patch
  forward pass can span depends on the resample rate used to produce the
  raw signal fed in, which is dataset/script-specific, not
  architecturally fixed. Confirmed directly from the two real prep scripts
  that write pretraining/downstream data: `prepare_dataset/prepare_CAP.py`
  and `prepare_dataset/prepare_HMC_downstream.py` both resample **EEG to
  200Hz** (patch=200 samples → 1.0s/patch, so 512×1.0s ≈ 512s ≈ 8.5min *is*
  correct for EEG specifically) but **EOG to 200Hz** (patch=100 samples →
  0.5s/patch, 512×0.5s = 256s ≈ 4.3min) and **ECG/EMG to 500Hz** (patch=100
  samples → 0.2s/patch, 512×0.2s = 102.4s ≈ 1.7min). **Not a blocker for
  this plan** — our own extraction design (§6) uses 30-second epochs, an
  order of magnitude below even the shortest (ECG/EMG) ceiling.
- **Attention: bidirectional, non-causal**
  (`scaled_dot_product_attention(..., is_causal=False)`,
  `model/transformer.py:55`) — masked-pretraining style, confirming Plan A
  (native long-context, no sequence head) is unavailable, same conclusion
  as OSF/MOMENT.
- **Pooled output per modality per forward pass**: `forward_features(...,
  return_all_tokens=False)` returns `x[:, 0]` (the CLS token) when
  `use_mean_pooling=False` (`NTConfig`'s default, unchanged in any of the
  three training scripts) — i.e. each modality's `NeuralTransformer`
  produces one `n_embd`-dim CLS vector per forward pass, directly usable as
  a frozen embedding.
- **No unified multimodal fusion in the pretrained weights.** `FT.py`'s
  fusion machinery (`EEG_embedding`/`EOG_embedding`/.../`EEG_head`/.../
  `EEG_Linear`/.../`X_transformer`/`alignment_module`/`lm_head`) is
  constructed fresh in `FT.__init__` every time, and `pretrained_ckpt_path`
  loading only ever populates the four `*_encoder` submodules — **the
  released checkpoint is the four per-modality tokenizers/encoders only,
  not a fusion model.** See §6 for how this shapes our own
  embedding-extraction design.
- **LoRA target modules, if Stage 2 is built**: `model/transformer.py`'s
  `Attention` class has `self.c_attn` (fused QKV, `nn.Linear(n_embd,
  3*n_embd)`) and `self.c_proj` (output projection, `nn.Linear(n_embd,
  n_embd)`) — the natural PEFT `target_modules=["c_attn", "c_proj"]`
  choice, analogous to OSF's `to_qkv`/`to_out.0`. Applied per encoder (up
  to 4 encoders × 12 blocks each = up to 48 attention modules total). No
  existing LoRA code in the repo to reference — genuinely new code, see
  §15.

## 4. Channel mapping — the **fast-channel `psg/` tree already suffices**,
no reprocessing and no full-channel tree needed

**PhysioOmni needs a strict subset of the channels OSF already needed**:
EEG (any 10-20 electrode name), EOG (derived from LOC/ROC), ECG (single
lead), EMG (chin). It does **not** need any respiratory/thoracic/abdominal/
snore channel — `apnea` is excluded from the comparison for exactly this
reason (§0.4), so none of the RESP-group channel gaps matter here.

**This is a real difference from OSF, not just a restatement of OSF's own
conclusion.** OSF needed the full-channel `psg_full/` tree specifically
because its 12-channel input includes RESP-group signals (snore, thoracic/
abdominal effort, nasal airflow) that the fast-channel tree's `RESP` cap
(1 channel) doesn't fully carry. PhysioOmni has no RESP input at all, so
that constraint doesn't apply — **the question is whether the
fast-channel tree's much smaller `BAS`/`EKG`/`EMG` caps still keep the
*specific* EEG/EOG/ECG/EMG channels PhysioOmni needs, not whether RESP is
present.** Checked directly, not assumed:

`configs/preprocessing_params.yaml`'s `channel_selection.strategy: "fast"`
(the paper-primary strategy, used for `phase0_v3`) caps `BAS=4, EKG=1,
EMG=2, RESP=1` (8 channels total). Cross-referencing
`configs/modality_groups.yaml`'s exact priority-order lists (the order caps
are applied in):

- `sleepfm_modalities.BAS.priority_order`: `C3-M2, C4-M1, LOC, ROC, O1-M2,
  O2-M1, F3-M2, F4-M1, A1, A2` — **with cap=4, the top 4 kept are exactly
  `C3-M2, C4-M1, LOC, ROC`**, i.e. precisely PhysioOmni's EEG(C3,C4) + the
  two channels the derived EOG(HEO) needs. Nothing PhysioOmni needs from
  this group is cut by the fast cap.
- `sleepfm_modalities.EKG.priority_order`: `EKG, ECG-L` — cap=1 keeps
  `EKG` (PhysioOmni's ECG input), same fallback order already used for OSF.
- `sleepfm_modalities.EMG.priority_order`: `CHIN, LLEG, RLEG, EMG` — cap=2
  keeps `CHIN` plus one leg channel; **`CHIN` (PhysioOmni's only EMG need)
  survives the cap in first position**, the leg channel that also gets kept
  is simply unused by PhysioOmni.

**Conclusion: use the fast-channel `psg/` tree, not `psg_full/`, as the
source for PhysioOmni's embedding extraction, and compare against
SleepFM's paper-primary `phase0_v3` results, not `phase0_v3_full`.**

### 4.1 Channel-name vocabulary (real constraint, code-verified)

`dataset.py`'s `standard_1020` list (132 entries, `dataset.py:8-23`) is a
**closed vocabulary** — `get_chans()` does
`standard_1020.index(ch_name)`, which raises `ValueError` on any name not
in the list. Unlike OSF's fixed-but-generic slot names
(`EEG_C3_A2`, `ECG`, etc.), PhysioOmni's channel "names" are used as
*position-embedding lookup keys*, not literal referencing claims — the
model's own downstream prep scripts confirm this directly:
`prepare_dataset/prepare_HMC_downstream.py:116` takes a raw label like
`'EEG C4-M1'` and reduces it to `'C4'` (`name.split(' ')[-1].split('-')[0]`)
purely for the position-embedding lookup, while the *signal itself* stays
whatever it physically was (M1-referenced). **This means our own `C3-M2`/
`C4-M1` HDF5 channels can be fed in with the reference-electrode suffix
stripped for the lookup key (`'C3'`/`'C4'`), exactly mirroring what
PhysioOmni's own reference code does** — not an approximation we're
inventing, the pattern already exists in their prep scripts.

### 4.2 Per-modality mapping plan

| PhysioOmni branch | Our HDF5 source | Mapping notes |
|---|---|---|
| EEG (up to 2 channels) | `C3-M2` → label `'C3'`, `C4-M1` → label `'C4'` | Both fed in the same forward pass — the model natively supports multiple named EEG channels per sample (`CAP`'s own pretraining used up to 16 EEG channels at once, per `prepare_CAP.py`'s `channels` lists) |
| EOG (1 derived channel) | `LOC` − `ROC` → label `'HEO'` | Matches `prepare_HMC_downstream.py:99`'s own derivation exactly (`EOG_signals[[0],:] - EOG_signals[[1],:]`, named `'HEO'`) — not an invented approximation |
| ECG (1 channel) | `EKG` → fallback `ECG-L` → label `'ECG'` | Same fallback order already established for OSF |
| EMG (1 channel) | `CHIN` → fallback generic `EMG` → label `'EMG'` | Same fallback order already established for OSF |

### 4.3 Real per-cohort availability — verified directly against the
fast-channel tree

Spot-checked one real subject per cohort directly against
`/scratch/boshra95/psg/{dataset}/derived/hdf5_signals/*.h5` (the
fast-channel tree, 2026-08-13):

| Cohort | Real HDF5 keys found | PhysioOmni-relevant channels present |
|---|---|---|
| APPLES (`APL1373.h5`) | `Airflow, C3-M2, C4-M1, EKG, EMG, LOC, ROC` | EEG ✅, EOG ✅, ECG ✅, EMG ✅ (generic fallback) |
| SHHS (`203805_v2.h5`) | `Airflow, EEG, EKG, EMG, LOC, ROC` | EEG ⚠️ (has data — generic `EEG` only, no C3/C4 split, see §4.5), EOG ✅, ECG ✅, EMG ✅ (generic fallback) |
| MrOS (`AA1449_v2.h5`) | `Airflow, C3-M2, C4-M1, CHIN, EKG, LLEG, LOC, ROC` | EEG ✅, EOG ✅, ECG ✅, EMG ✅ (real `CHIN`) |
| STAGES (`STNF00032.h5`) | `Airflow, C3-M2, C4-M1, CHIN, EKG, LOC, ROC` | EEG ✅, EOG ✅, ECG ✅, EMG ✅ (real `CHIN`) |

**Recomputed 2026-08-17 directly against the full raw population** (not
OSF's smaller 50-subject-per-cohort sample) — parsed
`output/channel_analysis/{apples,shhs,mros,stages}_channels.csv` (1,104 /
8,444 / 3,933 / 1,879 raw subjects respectively) against
`configs/channel_definitions.yaml`'s real alias lists:

| Channel | APPLES (n=1,104) | SHHS (n=8,444) | MrOS (n=3,933) | STAGES (n=1,879) |
|---|---|---|---|---|
| C3 (`C3-M2`) | 99.4% | **0%** | 100% | 99.9% |
| C4 (`C4-M1`) | 99.4% | **0%** | 100% | 99.9% |
| LOC (EOG) | 100% | 100% | 100% | 99.9% |
| ROC (EOG) | 99.4% | 100% | 100% | 99.9% |
| ECG (`EKG`→`ECG-L`) | 99.9% | 100% | 100% | **90.8%** |
| EMG (`CHIN`→generic `EMG`) | 100% | 100% | 100% | 99.9% |

STAGES's ECG gap (173/1,879 subjects, 9.2%) was traced directly, not left
unexplained: those subjects have a channel named `Heartrate` where an ECG
waveform would be — a derived beats-per-minute value, not a usable raw
signal, and not fixable via any alias-list addition (genuinely absent
data, same category as SHHS's C3/C4 gap below, not the same category as
OSF's fixable STAGES/SHHS alias-list gaps). Not worth reprocessing over.

**The one real gap: SHHS has no distinguishable C3/C4 EEG.** To be precise
about what this means (a source of real confusion the first time this was
reported, worth stating carefully): SHHS is **not** missing EEG signal —
its `psg/` HDF5s carry a channel literally named `EEG` at 100% coverage,
the exact channel OSF already reads today. What SHHS lacks is the *split*
into two separately-labeled electrodes (`C3-M2`/`C4-M1`) the other three
cohorts have. §4.5 below covers the full investigation and the final
decision on how to handle this — **no reprocessing is needed either way.**

### 4.4 Reprocessing decision: **no raw EDF reprocessing needed, and no
full-channel tree needed either**

Every channel PhysioOmni needs is already present in the fast-channel
`psg/` HDF5s — the same tree SleepFM's paper-primary `phase0_v3` results
already use — including SHHS's EEG (§4.5 below covers exactly how it's
used). `docs/OSF_CHANNEL_REPROCESSING_PLAN.md`'s three already-identified,
already-deferred gaps (MrOS `ABD`, STAGES leg EMG `LAT`/`RAT`, SHHS
`NEW AIR`/airflow) **do not apply to PhysioOmni at all** — none of those
three channels are in PhysioOmni's input set. **No new reprocessing plan
doc is needed for PhysioOmni.**

### 4.5 SHHS's single EEG channel — investigation, a documented future
option, and the final decision (2026-08-17)

**The problem, precisely.** SHHS's raw EDFs are recorded with a generic
`EEG`/`EEG(sec)` naming convention instead of clinical `C3`/`C4` labels
(a harmonized/de-identified NSRR release quirk). `configs/channel_definitions.yaml`
currently aliases **both** `EEG` and every `EEG(sec)`-family variant
(`EEG 2`, `EEG sec`, `EEG(SEC)`, `EEG(sec2)`, `EEG2`) to the **same single
canonical `EEG` slot** — architecturally there is only one EEG channel to
fill, so whichever name appears first in a given raw file wins and the
other is silently discarded. This is exactly the same class of issue
`docs/OSF_CHANNEL_REPROCESSING_PLAN.md` §4 already flagged for OSF, where
it was left as an **explicitly unverified** "open research question, NOT a
recommended fix" — worth checking whether `EEG`/`EEG(sec)` are genuinely
two different electrode sites or just a redundant backup of the same one,
before deciding whether pursuing a fix is worthwhile.

**That follow-up was done this session, with real data — the question is
no longer unverified:**
1. **100% of SHHS subjects (8,444/8,444)** have *both* an `EEG` channel
   and an `EEG(sec)`-family channel present in the same raw file
   (`output/channel_analysis/shhs_channels.csv`, all 5 naming variants
   checked) — not a naming difference across different files/recording
   eras, genuinely two channels recorded in essentially every SHHS file.
2. **Loaded a real raw SHHS EDF directly**
   (`shhs1-203279.edf`, via `mne.io.read_raw_edf`) and computed the
   correlation between its `EEG` and `EEG(sec)` channels: **r = 0.18.**
   A duplicate/backup of the same electrode would correlate near 1.0; a
   weak correlation is exactly what two genuinely different EEG
   derivations (e.g. C3 vs. C4 — same brain, different hemisphere, some
   shared global signal, mostly independent local activity) would produce.

**Conclusion: `EEG(sec)` is very likely a real, informative second EEG
channel that the current pipeline silently discards for essentially all of
SHHS** — not a naming duplicate. This is a materially stronger finding
than OSF's own doc had when it deferred the question.

**A possible future fix exists, deliberately not pursued now — documented
here so it doesn't need re-deriving.** Unlike `OSF_CHANNEL_REPROCESSING_PLAN.md`
§5's fix (a full fork of the EDF→HDF5 pipeline, re-processing *every*
channel for the affected cohorts, because those three fixes changed
channel *selection* more broadly), this case only requires *adding* one
already-well-understood channel: extract `EEG(sec)` per SHHS subject using
the exact same `processing_params.eeg` block (bandpass/resample/z-score)
the existing `EEG` channel already gets, and write it as a small,
additive, non-destructive companion artifact — the existing SHHS HDF5s
would not need to be touched or regenerated at all. This should be
substantially cheaper than a full SHHS reprocessing (one channel × 8,444
subjects, not seven-or-eight channels × 8,444), though it hasn't been
timed. It would benefit **both** OSF and PhysioOmni, since both currently
approximate SHHS's second EEG channel rather than having a real one — but
wiring it into OSF's own extraction script is a change on the
`osf-implementation` branch this session cannot make (worktree isolation,
§1). **Revisit this only if SHHS's actual results (either model) come back
meaningfully degraded on EEG-dependent tasks relative to the other
cohorts** — not before, and not as a prerequisite for starting Phase 1.

**Final decision for PhysioOmni, superseding this plan's earlier "duplicate
into both slots" draft**: **feed SHHS's existing single `EEG` channel to
the EEG branch as one real channel — not duplicated into two.** No
reprocessing of any kind is involved; this uses the exact `EEG` channel
already sitting in the existing `psg/` HDF5s.

Why this is better than duplication here specifically (unlike OSF, where
duplication was the right call — see below): PhysioOmni's `NeuralTransformer`
takes a **variable-length token sequence per modality**, not a fixed
channel-count tensor — there is no architectural requirement to supply
exactly 2 EEG channels. `CAP`'s own pretraining data used anywhere from a
few to 16 EEG channels per subject depending on the source recording, so a
1-channel EEG branch for SHHS is a legitimate, natively-supported input,
not a workaround. Duplicating the single channel would instead fabricate a
second token stream at **r=1.0** correlation with its twin — nothing like
the r=0.18 real channel pairs the encoder was pretrained on, contributing
zero genuine information while risking looking like nothing in the
pretraining distribution. **OSF's own duplication choice is unaffected by
this reasoning and stays correct for OSF** — OSF's ViT takes a *fixed*
`[B, 12, 1920]` tensor with no mechanism for a variably-sized channel set,
so duplicating (real data in every slot) was already the better choice
there over leaving a slot all-zero; PhysioOmni just doesn't have that
constraint in the first place.

**Mechanical consequence, not a design change**: this means SHHS's EEG
branch produces a shorter token sequence than the other three cohorts (30
one-second patches from 1 real channel, vs. 60 from 2) before CLS-pooling.
This does **not** change the pooled output dimension — the CLS token is
still 200-dim regardless of how many real tokens fed the forward pass — so
§6.3's flat `[T, 500]` embedding design is entirely unaffected; only the
channel loader's per-subject token-sequence-building step (§7) needs to
branch on "does this subject have a second EEG channel," the same kind of
per-subject conditional it already needs for any other missing channel.

**On the broader "avoid repeating preprocessing per training job" question
this ties into**: none of the above requires resolving separately from
what's already planned. §6.3's embedding-extraction step (`scripts/extract_physioomni_embeddings.py`,
Phase 1.2) is exactly the one-time, offline place where resampling (§5.1),
normalization inversion (§5.2), and this SHHS single-vs-duplicated-channel
handling all happen — **once per subject, producing a small `.npy`
embedding file**. `scripts/train_physioomni_context_sweep.py` and
`scripts/infer_physioomni_subject_windows.py` (§10/§11) never touch raw
signal, never resample, never re-run the frozen encoder — they only read
these precomputed embeddings, exactly mirroring OSF's own Stage 1 pattern
(`extract_osf_embeddings.py` once → `train_osf_context_sweep.py`/
`infer_osf_subject_windows.py` read cheaply, many times, across the full
task × head × context sweep). There is no risk of this preprocessing being
repeated inside a training job's GPU time — that would only become a
concern for Stage 2 (LoRA), where the backbone runs live every training
step on raw signal (§15.2 already covers why an offline raw-signal cache
is planned for that stage specifically, for exactly this reason).

## 5. Sample-rate and normalization handling

### 5.1 Sample rate — resample per PhysioOmni's own native rates, not our
own convenience rate

Our HDF5s (fast-channel tree, confirmed via a real file's
`sampling_rate` attribute, §6) are uniformly 128Hz. PhysioOmni's reference
prep scripts (`prepare_CAP.py`, `prepare_HMC_downstream.py`,
`prepare_tuh.py`, all read directly) consistently resample to **200Hz for
EEG and EOG, 500Hz for ECG and EMG** before chunking into patches.
**Recommendation: replicate these exact per-modality resample targets**
(128→200Hz for EEG/EOG, 128→500Hz for ECG/EMG — all upsampling) rather than
feeding 128Hz signal directly into 200/100-sample patches, which would
silently shift the real-world time each patch represents away from what
the frozen encoder was pretrained on (§3's ceiling-correction section
already shows this ratio matters). This is the same "match the reference
pipeline's resample choice exactly" principle already applied to OSF
(128→64Hz exact decimation). Unlike OSF's clean 2:1 ratio, 128→200 and
128→500 have no exact-decimation shortcut — use `scipy.signal.resample`
(FFT-based) or `mne`-style polyphase resampling, whichever this codebase's
existing preprocessing already uses elsewhere for non-integer-ratio
resampling, for consistency; verify empirically in the Step 0 pilot (§14).

### 5.2 Normalization — method confirmed, self-calibrating by design
(revised 2026-08-18, corrects the earlier per-channel-type assumption)

PhysioOmni's `dataset.py` divides raw signal by 100 (`EEG/100`, `EOG/100`,
etc., not z-score) before feeding the model — **different from OSF's
expectation of already-z-scored input.** Our `signal_processor.py`
z-scores per-channel during EDF→HDF5 preprocessing, but **every HDF5 file
stores the exact per-channel pre-normalization statistics needed to invert
this**: a `normalization_stats` JSON root attribute with `{mean, std, min,
max}` per channel. **Method**: `x_original = x_zscored * std + mean` per
channel (using that channel's own stored stats), then a scale correction
(below) to reach µV, then divide by 100.

**Traced the actual source of the unit inconsistency directly in
`signal_processor.py`, not just inferred from stats magnitudes** —
`_process_channel()` reads raw amplitude via plain MNE indexing
(`raw[ch_idx, :]`, `signal_processor.py:378`), with **no explicit unit
conversion anywhere in the file** (confirmed: no `units=` argument to
`mne.io.read_raw_edf`, no scaling call in `_process_channel`/
`_normalize_signal`). `stats['std']` is the std of this raw-indexed,
bandpass-filtered, resampled signal — i.e., **whatever unit MNE's default
EDF reader happens to return for that specific channel**, which turns out
to depend on the source EDF file's own header declarations, not a fixed
system-wide convention.

**This was verified with real data across two cohorts, and the original
"LOC/ROC = volts, everything else = µV" hypothesis (this section's
2026-08-17 draft) turned out to be wrong — the real pattern is
cohort/file-dependent, not a fixed per-channel-type rule:**
- **APPLES** (`apples-560232.edf`, loaded directly with `mne.io.read_raw_edf`,
  same bandpass params `signal_processor.py` uses): `C3_M2` std≈15.6→14.5
  (before→after bandpass), `ECG` std≈18.7, `EMG` std≈34.9 — all µV-scale —
  but `LOC`/`ROC` std≈1.9-2.1e-5 — volts-scale. Matches the original
  hypothesis, for this cohort.
- **SHHS** (two real files, `shhs1-203279.edf`/`shhs1-200709.edf`): `EEG`
  std≈1.4-3.3e-5 (volts-scale, as expected) **but `ECG` std≈1.3-2.2e-4 —
  also volts-scale**, unlike APPLES's µV-scale `ECG`. **The same canonical
  channel (`ECG`) is in a different unit depending on cohort.**

**Design consequence: don't hardcode a per-channel-name unit table — detect
and correct per subject/channel at runtime instead.** Real physiological
signal amplitudes are µV-scale (std roughly 1-100s, confirmed above) or,
equivalently, volts-scale (std roughly 1e-6-1e-3) — the two regimes are
separated by several orders of magnitude with a wide, safe gap between
them, so a simple runtime check on each channel's own stored `stats['std']`
is a robust, self-calibrating detector, no hardcoded table needed:

```python
def invert_normalization(x_zscored, stats):
    x = x_zscored * stats["std"] + stats["mean"]   # undo z-score
    if abs(stats["std"]) < 1.0:      # this channel's raw scale was volts, not uV
        x = x * 1e6                  # -> uV, matching PhysioOmni's own convention
    return x / 100.0                 # PhysioOmni's own dataset.py convention
```

**Still a Step 0 pilot check (§14), not assumed correct on the first
attempt**: run a handful of real epochs through the frozen encoder after
this conversion and check for NaNs/degenerate (near-zero or saturated) CLS
outputs before trusting any extraction at scale — the self-calibrating
threshold (`1.0`) is a reasoned choice given the observed ~5-order-of-
magnitude gap between regimes, not empirically swept against many
channels/cohorts yet.

---

## 6. Stage 1 — File Map

Mirrors `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s own early "File Map"
section (before any of it was built) — same columns, all rows currently
`⬜ TODO` since this revision writes no code.

### 6.1 File isolation table (PhysioOmni vs. OSF/SleepFM — nothing shared
is ever edited)

| Purpose | SleepFM (reference only) | OSF (reference only) | **PhysioOmni (new)** |
|---|---|---|---|
| Embedding config | `configs/phase0_v3_config.yaml` | `configs/phase0_osf_config.yaml` | `configs/phase0_physioomni_config.yaml` |
| Channel-loader utility | *(inline in `extract_sleepfm_embeddings.py`)* | `src/nsrr_tools/datasets/osf_channel_loader.py` | `src/nsrr_tools/datasets/physioomni_channel_loader.py` |
| Extraction script | `scripts/extract_sleepfm_embeddings.py` | `scripts/extract_osf_embeddings.py` | `scripts/extract_physioomni_embeddings.py` |
| Extraction job script | `jobs/extract_embeddings_gpu.sh` | `jobs/extract_osf_embeddings_gpu.sh` | `jobs/extract_physioomni_embeddings_gpu.sh` |
| Dataset class | `context_window_dataset.py` | `osf_context_window_dataset.py` | `physioomni_context_window_dataset.py` |
| Training script | `train_context_sweep.py` | `train_osf_context_sweep.py` | `train_physioomni_context_sweep.py` |
| Training job script | `train_context_sweep_gpu.sh` | `train_osf_context_sweep_gpu.sh` | `train_physioomni_context_sweep_gpu.sh` |
| Inference script | `infer_subject_windows.py` | `infer_osf_subject_windows.py` | `infer_physioomni_subject_windows.py` |
| Inference job script | `infer_subject_windows_gpu.sh` | `infer_osf_subject_windows_gpu.sh` | `infer_physioomni_subject_windows_gpu.sh` |
| Registry | `v2_registry.yaml` / `v2_full_registry.yaml` | `v2_osf_registry.yaml` | `v2_physioomni_registry.yaml` |
| Command generator | `gen_commands.py` | `gen_commands_osf.py` | `gen_commands_physioomni.py` |
| Results dir | `.../results/phase0_v3/` | `.../results/phase0_osf/` | `.../results/phase0_physioomni/` |
| Logs dir | `logs_v3/` | `logs_osf/` | `logs_physioomni/` |
| Embeddings dir | `.../embeddings/sleepfm_5sec/` | `.../embeddings/osf_30sec/` | `.../embeddings/physioomni_30sec/` |
| **Sequence head (`sequence_head.py`)** | shared, unmodified | shared, unmodified | **shared, unmodified** — dim-agnostic per `CLAUDE.md`'s Code Reuse Assessment, `input_dim=500` is just a constructor arg |

**Everything under "PhysioOmni (new)" is a brand-new file. Nothing under
"SleepFM"/"OSF" columns is ever opened in write mode.**

### 6.2 File Map by phase

| File | Purpose | Status |
|---|---|---|
| `/home/boshra95/physioomni_env` | Python venv, PhysioOmni's trimmed/relaxed dependencies | ⬜ TODO (0.1) |
| `/home/boshra95/PhysioOmni/checkpoints/PhysioOmni.pt` | Downloaded checkpoint | ⬜ TODO (0.2) |
| `configs/phase0_physioomni_config.yaml` | Master config | ⬜ TODO (§9) |
| `src/nsrr_tools/datasets/physioomni_channel_loader.py` | Shared channel-loading/resampling/normalization utility | ⬜ TODO (§7) |
| `scripts/extract_physioomni_embeddings.py` | Stage 1 Step 1 — extract frozen embeddings | ⬜ TODO (§6.3) |
| `jobs/extract_physioomni_embeddings_gpu.sh` | SLURM job for embedding extraction | ⬜ TODO (§13) |
| `src/nsrr_tools/datasets/physioomni_context_window_dataset.py` | Stage 1 Step 2 — PyTorch dataset | ⬜ TODO (§8) |
| `src/nsrr_tools/models/sequence_head.py` | LSTM/Transformer/MeanPool heads | **Reused unmodified — no new file needed** |
| `scripts/train_physioomni_context_sweep.py` | Stage 1 Step 4 — training loop | ⬜ TODO (§10) |
| `jobs/train_physioomni_context_sweep_gpu.sh` | SLURM job for training | ⬜ TODO (§13) |
| `scripts/infer_physioomni_subject_windows.py` | Stage 1 Step 5 — inference | ⬜ TODO (§11) |
| `jobs/infer_physioomni_subject_windows_gpu.sh` | SLURM job for inference | ⬜ TODO (§13) |
| `experiments/v2_physioomni_registry.yaml` | Experiment registry (4 tasks × 3 heads × 6 contexts) | ⬜ TODO (§12) |
| `scripts/gen_commands_physioomni.py` | Command generator | ⬜ TODO (§12) |
| `docs/PHYSIOOMNI_EXPERIMENTS_GUIDE.md` | Operational how-to-run counterpart (mirrors `OSF_EXPERIMENTS_GUIDE.md`) | ⬜ TODO (Phase 1, low priority — fill in as each step is built, same incremental pattern OSF's guide used) |
| `scripts/train_physioomni_lora.py` | Stage 2 — new end-to-end script | ⬜ TODO, outline only (§15) |

### 6.3 Embedding definition (Plan B, no shared fusion to reuse)

Because the released checkpoint has **no unified fusion module** (§3), our
own frozen embedding is naturally "the four per-modality CLS outputs,
concatenated" — there is no PhysioOmni-native single-vector-per-epoch
output to just save, unlike OSF's single ViT with one CLS token.

**Per-epoch embedding definition** (mirrors OSF's "one 30-second epoch, one
forward pass per available branch" pattern from
`scripts/extract_osf_embeddings.py`):

1. For each 30-second epoch, and for each of the up to 4 modality branches
   with data available for that subject: resample to the branch's native
   rate (§5.1), chunk into 1-second EEG / 0.5-second EOG / 0.2-second
   ECG-and-EMG patches (§3's table), run through that modality's
   `NeuralTransformer.forward_features(..., return_all_tokens=False)` to
   get one `n_embd`-dim CLS vector (200-dim for EEG, 100-dim for
   EOG/ECG/EMG).
2. **Concatenate the four vectors into one flat 500-dim vector per epoch**
   (200 EEG + 100 EOG + 100 ECG + 100 EMG), **zero-filling any entirely
   absent modality's 100/200-dim slice** (not per-patch — per whole
   modality, since a modality is either present or absent for a whole
   subject in our channel-availability picture, §4.3). This is simpler than
   OSF's `[T, 2, 768]` two-subtoken design (no shared dimension across
   sub-tokens to preserve) — a flat `[T_epochs, 500]` array per subject is
   sufficient; there's no need for a 3D shape at all.
3. **Log which modality was zero-filled per subject, and how many real
   channels fed the EEG branch** (same `_channel_fill_log.jsonl` convention
   as OSF's extraction script) — SHHS's single-real-EEG-channel case (§4.5)
   and any true per-subject absence both need to be visible in this log
   before trusting results.

**Output**: `{output_dir}/{dataset}/{subject_id}.npy`, dtype float16, shape
`[T_epochs, 500]`, under
`/scratch/boshra95/psg/unified/embeddings/physioomni_30sec/`. Results dir:
`/scratch/boshra95/psg/unified/results/phase0_physioomni/`.

**Context-length → epoch-count mapping**: identical to OSF's, since both
use 30-second epochs (`30s`→1, `10m`→20, `40m`→80, `80m`→160, `120m`→240,
`240m`→480).

---

## 7. Shared channel-loader module: `src/nsrr_tools/datasets/physioomni_channel_loader.py`

**New this revision, modeled directly on OSF's finished
`src/nsrr_tools/datasets/osf_channel_loader.py`** — a real, working pattern
found while reading OSF's code, not present in this plan's 2026-08-13
draft. OSF originally had this channel-loading logic inline in
`extract_osf_embeddings.py`, then **factored it out into a shared module
specifically so Stage 1 (precomputed embeddings) and Stage 2 (raw signal
loaded live) never drift out of sync** — the refactor was regression-tested
against a real already-extracted subject (`np.allclose` on old vs. new
output) before being trusted.

**Build this as a shared module from day one for PhysioOmni**, not as an
inline-then-refactor-later step — the OSF precedent shows exactly where the
refactor pays for itself (Stage 2's `precompute_osf_raw_signal_cache.py`
and `OSFRawEpochWindowDataset` both import from the same module Stage 1's
extraction script uses), and starting there avoids a second regression-test
pass later.

**Planned contents** (functions, not full implementations — this is still
a plan, not code):

```
PHYSIOOMNI_CHANNEL_MAPPING = {   # PhysioOmni branch -> our HDF5 candidates (§4.2)
    "EEG":  {"C3": ["C3-M2"], "C4": ["C4-M1"]},   # up to 2 channels — SHHS only has 1, see below
    "EOG":  {"HEO": ["LOC", "ROC"]},               # derived: LOC - ROC
    "ECG":  {"ECG": ["EKG", "ECG-L"]},
    "EMG":  {"EMG": ["CHIN", "EMG"]},
}
EPOCH_SECONDS = 30
NATIVE_HZ = {"EEG": 200, "EOG": 200, "ECG": 500, "EMG": 500}   # §5.1
PATCH_SAMPLES = {"EEG": 200, "EOG": 100, "ECG": 100, "EMG": 100}   # §3's table

def build_channel_candidates(dataset: str, cfg_candidates: dict) -> dict:
    """SHHS special case (§4.5, final decision — NOT duplication): SHHS has
    no C3-M2/C4-M1 keys at all, only a single generic 'EEG'. Returns EEG
    candidates for SHHS as {"C3": ["EEG"]} ONLY — no "C4" entry at all, so
    the token-sequence builder naturally feeds just 1 real EEG channel for
    SHHS instead of 2, rather than duplicating one real channel into both
    slots (§4.5 explains why duplication is the wrong choice here,
    architecturally different from OSF's own — correct — choice to
    duplicate)."""

def load_and_resample_modality(h5_path, modality: str, candidates: dict) -> tuple[np.ndarray | None, dict]:
    """Load + resample ONE modality's channel(s) to its native rate (§5.1).
    Returns None (not zero-array) if entirely absent for this subject — the
    caller decides whether/how to zero-fill, matching OSF's per-subject,
    per-modality (not per-channel) fill-log granularity (§6.3 item 3)."""

def invert_normalization(raw_zscored: np.ndarray, stats: dict, channel_name: str) -> np.ndarray:
    """x_original = x_zscored * stats['std'] + stats['mean'], per channel
    (§5.2) — reads stats from the HDF5's own normalization_stats attribute,
    does NOT assume a uniform unit across channels (the LOC/ROC-in-volts
    finding, §5.2)."""

def chunk_into_patches(x: np.ndarray, patch_samples: int) -> np.ndarray:
    """[n_samples] -> [n_patches, patch_samples], dropping any incomplete
    trailing patch — same convention as OSF's epoch-chunking."""

def get_epoch_count(h5_path) -> int:
    """Fast metadata-only read of complete-30s-epoch count, mirrors OSF's
    get_epoch_count() — used to build the dataset-class shape cache without
    loading channel data."""
```

**Placement note, a real gotcha found while reading OSF's code, not
something to rediscover**: OSF's channel-loader module is **deliberately
placed under `src/nsrr_tools/datasets/`, not `src/nsrr_tools/core/`**,
because `nsrr_tools.core.__init__.py` eagerly imports `channel_mapper.py`,
which imports `pyedflib` — not installed in `osf_env` (confirmed live via
`ModuleNotFoundError`). **Verify this same constraint holds for whatever
`physioomni_env` ends up containing** (checklist 0.1) rather than assuming
it transfers automatically, but plan on the same placement
(`src/nsrr_tools/datasets/physioomni_channel_loader.py`) as the safe
default either way.

---

## 8. Dataset class fork: `src/nsrr_tools/datasets/physioomni_context_window_dataset.py`

Per `CLAUDE.md`'s Code Reuse Assessment, `ContextWindowDataset` is
SleepFM-shape-hardcoded and not reusable unmodified — fork it, following
the exact pattern OSF's `osf_context_window_dataset.py` already
established (K-sampling logic, `SubjectGroupedSampler`, window-building
index math are pure integer arithmetic and copy as-is; only the
embedding-shape constants and the modality-ablation feature change).

**Module-level constants** (mirroring OSF's own, simplified further per
§6.3's flat-500-dim design — genuinely simpler to fork than OSF's version,
since there's no per-slot dimension mismatch to work around):

```python
PATCH_SECONDS       = 30    # each PhysioOmni embedding row = one 30-second epoch
PATCHES_PER_EPOCH   = 1     # same reasoning as OSF: epoch and patch are the same unit here
EMBED_DIM           = 500   # 200 (EEG) + 100 (EOG) + 100 (ECG) + 100 (EMG)
N_SUBTOKENS         = 1     # already-concatenated flat vector, no sub-token split needed
FLAT_DIM            = N_SUBTOKENS * EMBED_DIM   # 500
FULL_NIGHT_SENTINEL = -1    # same convention as OSF/SleepFM
```

**No `zero_modality_indices`/`--zero-modalities` equivalent** — PhysioOmni
has no 4-modality-*group* structure the way SleepFM's BAS/RESP/EKG/EMG
grouping does (its "4 modalities" are literally EEG/EOG/ECG/EMG, already
the finest granularity, and ablating one is equivalent to just excluding it
at extraction time via the channel-fill-log mechanism, §6.3 item 3) — drop
the constructor param, stored attribute, `_apply_modality_zeroing()`
method, and all call sites, same as OSF's fork already did.

**Context-length → epoch-count mapping**: identical to OSF's table (§6.3),
recompute nothing — both use 30-second epochs.

**Cohort consistency filter**: `min_recording_patches: 480` (240m × 60s/m
÷ 30s), the exact same value OSF's config already uses, for the same
reason (both are 30s-epoch-granularity embeddings). **Do not reuse
SleepFM's `2880`** — same "single easiest place to introduce a silent bug"
warning OSF's own plan flagged, still applicable here.

**Real per-cohort spot-check needed before trusting this dataset class**:
unlike OSF (where every subject's embedding file always has exactly 2
subtokens regardless of channel completeness, since OSF zero-fills at the
*channel* level before ever running the ViT), PhysioOmni's flat 500-dim
vector has a **structurally different failure mode worth testing
explicitly** — a subject missing an entire modality (not just one channel
within a modality) produces a genuinely zero 100-or-200-dim *slice* of an
otherwise-real embedding, not a zero-filled *channel* feeding into an
otherwise-normal forward pass. Confirm during the Step 0 pilot (§14) that
this doesn't produce degenerate downstream training (e.g. the sequence
head learning to ignore the zero slice entirely). SHHS's EEG slice is a
**third, different case from either of those** (§4.5's final decision): its
200-dim EEG slice is neither zero-filled nor duplicated — it's a genuine,
real CLS output, just computed from a shorter (1-channel, 30-token) input
sequence than the other three cohorts' (2-channel, 60-token) EEG branch.
Worth keeping in mind when interpreting SHHS's results (a real, if
information-poorer, EEG representation — not a degenerate one), distinct
from both the zero-fill and duplication failure modes above.

---

## 9. Config file: `configs/phase0_physioomni_config.yaml`

Full template, following `configs/phase0_osf_config.yaml`'s exact section
shape (`embedding:`, `data:`, `dataset:`, `model:`, `training:`,
`analysis:`, `logging:`) so the two configs read side-by-side cleanly:

```yaml
# ── Embedding Extraction ─────────────────────────────────────────────────────
embedding:
  checkpoint_dir: "/home/boshra95/PhysioOmni/checkpoints/PhysioOmni.pt"
  output_dir: "/scratch/boshra95/psg/unified/embeddings/physioomni_30sec"
  chunk_batch_size: 16   # epochs-per-forward-pass batch, per modality —
                          # same knob as OSF's, start at 16, tune empirically
                          # (OSF found this was its real Stage-2 GPU
                          # bottleneck, not raw compute — §15's lesson)
  datasets: [apples, shhs, mros, stages]

# ── HDF5 Signal Data ─────────────────────────────────────────────────────────
data:
  hdf5_dir: "/scratch/boshra95/psg"     # FAST-CHANNEL tree, not psg_full (§4)
  epoch_seconds: 30
  native_hz: {EEG: 200, EOG: 200, ECG: 500, EMG: 500}   # §5.1
  patch_samples: {EEG: 200, EOG: 100, ECG: 100, EMG: 100}   # §3
  # PhysioOmni channel -> our HDF5 channel name(s), in priority order (§4.2).
  # SHHS handled specially in code (build_channel_candidates(), §7): EEG's
  # C3/C4 both resolve to the generic "EEG" key instead of C3-M2/C4-M1.
  channel_candidates:
    EEG_C3: [C3-M2]
    EEG_C4: [C4-M1]
    EOG_LOC: [LOC]   # combined into HEO = LOC - ROC at load time
    EOG_ROC: [ROC]
    ECG: [EKG, ECG-L]
    EMG: [CHIN, EMG]
  # normalization_stats inversion (§5.2) reads directly from each HDF5's own
  # root attribute — no config key needed, but flagging here as a real
  # per-subject-file dependency, not a hardcoded constant.

# ── Context-Window Dataset ───────────────────────────────────────────────────
dataset:
  embedding_dir: "/scratch/boshra95/psg/unified/embeddings/physioomni_30sec"

  # SAME labels/splits as SleepFM phase0_v3 — required for a fair comparison
  # on identical subjects/splits, not just an optimization. NOTE (learned
  # from OSF's own footnote): label_source is documentation-only, never read
  # at runtime — task_subject_dir + split_seed + the split ratios below are
  # what actually matter.
  label_source: "/scratch/boshra95/psg/unified/targets_v2/master_targets.parquet"
  task_subject_dir: "/scratch/boshra95/psg/unified/targets_v2/task_subjects"

  sleep_stage_dir: "/scratch/boshra95/psg"

  context_lengths: ["30s", "10m", "40m", "80m", "120m", "240m"]

  task: "sex_binary"
  task_type: "seq2label"

  datasets: [apples, shhs, mros, stages]

  train_split: 0.70
  val_split:   0.15
  test_split:  0.15
  split_seed:  42          # SAME seed as SleepFM phase0_v3 runs

  windows_per_subject: 5

  # seq2seq params carried for config-shape parity — NOT used by the first
  # pass (Tier-1 tasks are all seq2label, apnea already excluded, §0.4).
  seq2seq_context_mode: "centered"
  seq2seq_padding_policy: "complete_only"
  seq2seq_max_padding_fraction: 0.5
  min_past_denom: 8
  max_min_past_patches: 40   # 20-minute cap in 30s-epoch units, same as OSF

  min_recording_patches: 480   # 240m in 30s-epoch units, same as OSF — NOT 2880

# ── Sequence Head Model ──────────────────────────────────────────────────────
# hidden_dim/num_layers held constant at phase0_v3's seq2label values — only
# input_dim changes (500 = concatenated per-modality CLS dims, vs SleepFM's
# 512 = 4*128 and OSF's 1536 = 2*768). Preserves the "architecture held
# constant, only encoder/channels change" comparison principle.
model:
  input_dim: 500
  head_type: "lstm"
  hidden_dim: 128
  num_layers: 1
  num_heads: 8
  dropout: 0.3
  num_classes: 2

# ── Training — held identical to OSF/SleepFM's placeholder values as a
# starting point (§15's lesson: OSF's Stage 1 values needed no revision;
# only Stage 2/LoRA's did, once real training curves existed to justify it —
# don't pre-emptively tune before a real pilot shows a reason to) ──────────
training:
  epochs: 40
  lr: 1.0e-4
  weight_decay: 1.0e-3
  optimizer: "adamw"    # NOTE: not read if the training script mirrors
                         # OSF's — hardcoded to plain Adam there. Confirm
                         # when train_physioomni_context_sweep.py is written
                         # whether to keep that hardcode or actually wire
                         # AdamW; don't assume either way without checking.
  scheduler: "cosine"
  early_stopping_patience: 10
  device: "cuda"
  mixed_precision: false
  class_weights: "auto"
  early_stopping_monitor: "val_auroc"
  weighted_sampler: false

  windows_strategy: "fixed"
  token_budget_minutes: 240
  k_max: 50

  context_lr_overrides:
    "120m": 5.0e-5
    "240m": 5.0e-5

  overfit_epochs: 0
  save_snapshots: true
  snapshot_interval: 5

# ── Analysis ─────────────────────────────────────────────────────────────────
analysis:
  bootstrap_samples: 0

# ── Logging & Output ─────────────────────────────────────────────────────────
# REQUIRED — train_context_sweep.py-pattern scripts read
# cfg["logging"]["results_dir"] via bare bracket access, no default. OSF's
# plan draft originally missed this section and it would have KeyError'd on
# the very first run — don't repeat that, include it from the start.
logging:
  results_dir: "/scratch/boshra95/psg/unified/results/phase0_physioomni"
```

**Two footnotes worth carrying over from OSF's own config, likely to apply
identically here** (confirm when the training script is actually written,
don't assume blindly): (1) `training.optimizer`/`scheduler`/`device` may
turn out to be dead config keys if `train_physioomni_context_sweep.py` is
forked closely enough from `train_context_sweep.py`'s hardcoded-Adam
pattern — check this explicitly rather than assuming the config controls
anything; (2) `dataset.label_source` is likely documentation-only, not
read at runtime — `task_subject_dir` + `split_seed` are what actually
matter.

---

## 10. Training script: `scripts/train_physioomni_context_sweep.py`

Fork of `scripts/train_context_sweep.py`, following the exact relationship
`scripts/train_osf_context_sweep.py` already has to it — same
checkpoint/resume, early stopping, overfit-phase, snapshot, and
bootstrap-CI machinery carries over unchanged (head/optimizer-level, not
backbone-level), only the dataset import and one CLI flag change.

**CLI arguments — mirror `train_osf_context_sweep.py`'s exact flag set**
(confirmed via direct read, `scripts/train_osf_context_sweep.py:916-950`),
so a future Stage 2 script can import this one's functions the same way
`train_osf_lora.py` imports from `train_osf_context_sweep.py` (§15):

```
--config          (required)  path to phase0_physioomni_config.yaml
--task                        override dataset.task
--task-type                   seq2label | seq2seq
--head                        lstm | transformer | mean_pool
--context          nargs+     context length(s) to train
--datasets         nargs+
--limit            int        debug: subject cap
--max-items        int        debug: item cap
--full-night-epochs int       (kept for config-shape parity — seq2seq only,
                               inert for the Tier-1 seq2label scope)
--cpu
--wandb-project    default "nsrr-phase0-physioomni" (own project, kept
                    separate from both nsrr-phase0 and nsrr-phase0-osf —
                    same reasoning OSF used for its own separate project)
--wandb-entity
--no-wandb
--batch-size       int
--accum-steps      int        default 1
--lr               float
--run-tag
```

**Dropped relative to `train_context_sweep.py`**: `--zero-modalities` and
the `_MODALITY_INDICES` dict — same reasoning as OSF, no 4-modality-*group*
structure to ablate.

**Function-boundary design principle, learned directly from reading OSF's
Stage 2 code**: `train_osf_lora.py` imports `run_epoch`, `compute_metrics`,
`compute_monitor_metric`, `append_to_summary`, and `_classify_failure`
**directly from** `train_osf_context_sweep.py` rather than duplicating
them — this worked because `run_epoch()` has no knowledge of what's inside
`model`, it just calls `model(x, mask)`. **Write
`train_physioomni_context_sweep.py` with the same clean function
boundaries from the start** (even though Stage 2 is out of scope for this
revision, §15) — it costs nothing now and saves a real refactor later,
exactly the value OSF's Stage 2 got for free by having Stage 1 already
structured this way.

**Output**: `{results_dir}/{task}_{head_type}/context_{L}/{best_model.pt,metrics.json,training_curves.csv}`,
`{results_dir}/{task}_{head_type}/summary.csv` — identical schema to
OSF's/SleepFM's, minus any `zero_modality_indices` field.

**Architecture parity**: `hidden_dim=128, num_layers=1` (matching
`phase0_v3`'s seq2label head), only `input_dim` changes (500, not 512 or
1536) — preserves the "architecture held constant, only encoder/channels
change" comparison principle.

---

## 11. Inference script: `scripts/infer_physioomni_subject_windows.py`

Fork of `scripts/infer_subject_windows.py`, same relationship
`scripts/infer_osf_subject_windows.py` already has to it.

**Batch-size auto-scaling reference point** — confirmed via direct read of
`infer_osf_subject_windows.py:281-296`, OSF's formula is:

```python
_ref_bs = 64     # starting assumption, NOT GPU-verified even for OSF yet
_ref_N  = 480    # OSF's 240m in its own (30s-epoch) units
eff_bs  = min(args.batch_size, max(_ref_bs, int(_ref_bs * _ref_N / N_patches)))
```

**Recalibrate `_ref_N=480` for PhysioOmni too** — same 30s-epoch units,
same value as OSF (not SleepFM's `2880`, which is 5-second-patch units).
`_ref_bs=64` carries over as the same unverified starting assumption OSF
itself still has — **do not treat it as GPU-tested just because it's
copied from OSF's config**, both need independent GPU verification (§14).

**Output**: `{results_dir}/inference/{task}_{head_type}/context_{L}/{split}_windows.parquet`
— identical schema to OSF's/SleepFM's inference output (`subject_id,
dataset, window_idx, true_label, pred_label, prob_class0…prob_classN`, plus
`anchor_patch_end` for seq2seq only — inert for the Tier-1 seq2label scope).

---

## 12. Registry + command generator

`experiments/v2_physioomni_registry.yaml` — same schema as
`v2_osf_registry.yaml` (confirmed field-for-field via direct read of
`experiments/v2_osf_registry.yaml:1-80`), restricted to **4 Tier-1 tasks**
(sex_binary, sleep_efficiency_binary, bmi_binary, age_class — apnea
excluded, §0.4) × 3 heads = **12 experiments, up to 72 training runs** (12
× 6 contexts).

```yaml
config: configs/phase0_physioomni_config.yaml
results_dir: /scratch/boshra95/psg/unified/results/phase0_physioomni
inference_dir: /scratch/boshra95/psg/unified/results/phase0_physioomni/inference
logs_dir: /home/boshra95/NSRR-tools/logs_physioomni
python_bin: /home/boshra95/physioomni_env/bin/python

gradient_accumulation:
  enabled: true
  effective_batch: 32
  context_micro_batch:
    "30s": 32
    "10m": 32
    "40m": 32
    "80m": 32
    "120m": 32
    "240m": 32
```

**Two fields confirmed load-bearing (not optional), same gotcha OSF's plan
already documented and worth restating rather than rediscovering**:
`inference_dir` — `gen_commands.py`-pattern scripts do
`Path(registry["inference_dir"])` via plain bracket access, `KeyError`
without it; `python_bin` — has a fallback default, but **the fallback is
`/home/boshra95/sleepfm_env/bin/python`**, hardcoded at every call site in
the base `gen_commands.py` this pattern derives from. **Set explicitly** —
if omitted, a forked `gen_commands_physioomni.py` that keeps this default
would silently run PhysioOmni jobs in the wrong Python environment.

`scripts/gen_commands_physioomni.py` — fork of `gen_commands_osf.py`,
same subcommand scope (`list, probe-batch, train, infer, analyze,
build-heatmap, collect, threshold-tuning, status, runs`), same deliberate
exclusion of figure/table subcommands (superseded by notebooks for the
current paper, per `CLAUDE.md`'s Code Reuse Assessment). **Wall-time
lookup tables**: OSF's own approach was to start with a scaled-up
placeholder and refine after a real pilot — do the same here, but flag
explicitly that PhysioOmni's per-context cost profile is **unknown and
architecturally different from OSF's** (four small transformers × up to 4
forward passes per epoch during extraction, vs. OSF's one ViT) — don't
copy OSF's numbers even as a starting guess without first checking the
extraction script's own real wall-clock time on a pilot batch (§14).

---

## 13. Job scripts

`jobs/extract_physioomni_embeddings_gpu.sh`,
`jobs/train_physioomni_context_sweep_gpu.sh`,
`jobs/infer_physioomni_subject_windows_gpu.sh` — copy the SLURM directive
pattern confirmed directly from `jobs/train_osf_context_sweep_gpu.sh`:

```bash
#SBATCH --account=def-forouzan_gpu
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1   # start here, see GPU-sizing note below
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --exclude=fc11006,fc11013,fc11010
#SBATCH --signal=B:USR1@120
```

Same two-layer auto-resume mechanism as OSF/SleepFM: (1) `--signal=B:USR1@120`
+ a bash trap that kills Python cleanly and resubmits via `sbatch "$0"`,
picking up from `resume.pt`; (2) node-failure `--requeue`, supplied by
`gen_commands_physioomni.py` at the *initial* `sbatch` call, not baked into
the `.sh` file itself. Same per-job JSONL status log convention under
`logs_physioomni/status/`.

**GPU-sizing lesson from OSF's Stage 2, worth applying proactively rather
than rediscovering**: OSF found that upgrading from `1g.10gb` to `3g.40gb`
(3× more compute/memory) measured **zero** speedup, because the real
bottleneck was `chunk_batch_size` (how many epochs get batched per forward
call), not raw GPU headroom — MIG partitions compute proportionally to
memory, but kernel-launch/Python-loop overhead per chunk is roughly fixed
regardless of available compute. **For PhysioOmni's extraction script,
treat `embedding.chunk_batch_size` (§9) as the primary throughput lever to
tune first**, before assuming a bigger MIG slice is needed — cheaper to
test and directly informed by OSF's own measured result, not a guess.

---

## 14. Step 0 — verification checklist (run before any real sweep)

Mirrors `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s own §9, adapted to
PhysioOmni's open items:

1. **Checkpoint key-prefix hypothesis** (§2) — download `PhysioOmni.pt`,
   `torch.load(..., weights_only=False)['model'].keys()`, confirm
   `EEG_encoder.`/`EOG_encoder.`/`ECG_encoder.`/`EMG_encoder.` prefixes
   exist and strict-or-partial-load into `FT.py`'s four `NeuralTransformer`
   instances with zero unexpected keys.
2. **Normalization pilot** (§5.2) — invert a handful of real epochs'
   `normalization_stats`, apply the (per-channel-aware) unit correction,
   divide by 100, run through the frozen encoder, check for NaN/degenerate
   (near-zero or saturated) CLS outputs before trusting extraction at
   scale.
3. **Sample-rate resampling sanity check** (§5.1) — confirm the chosen
   128→200Hz/128→500Hz resampling method (FFT-based or polyphase) produces
   sane-looking waveforms on a real subject, not just correct shapes.
4. ~~**SHHS EEG-duplication decision**~~ — **✅ RESOLVED 2026-08-17, see
   §4.5**: not duplication — SHHS feeds the EEG branch one real,
   non-duplicated channel. No reprocessing involved, no further
   confirmation needed before writing extraction code.
5. **Small-scale pilot, end-to-end** — `--limit 5` extraction on one
   dataset, inspect `.npy` shape/values, then a tiny Stage 1 training run
   (`--context 30s`, one task, one head) before submitting a full sweep.
   Include SHHS specifically, to confirm its shorter (1-channel) EEG token
   sequence runs cleanly through the encoder (§4.5).
6. **`physioomni_env` compatibility check** — **✅ RESOLVED 2026-08-17**:
   dedicated venv, does not share `osf_env` (checklist 0.1); the
   `nsrr_tools.core` eager-import gotcha (§7) confirmed to apply the same
   way in `physioomni_env`.
7. **Cohort filter unit check** — confirm `min_recording_patches: 480` is
   applied in epoch units, same discipline OSF's own Step 0 checklist used.
8. **Per-modality-missing degeneracy check** (§8) — confirm a subject with
   an entire modality zero-filled doesn't produce a pathological (not just
   suboptimal) training signal. Distinct from SHHS's real-but-shorter EEG
   case (§4.5) — that one isn't a degeneracy risk in the same sense, but
   still worth confirming it trains sensibly.

---

## 15. Stage 2 (LoRA) — detailed design, resolved and live-verified 2026-08-19

**This revision replaces the earlier outline-only version.** Every
previously-open design question (§15.1's "genuinely undecided" multi-
encoder wrapping question, in particular) is now resolved below, with the
resolution live-verified against the real checkpoint (not just reasoned
about) before being written down. Written after re-reading
`scripts/train_osf_lora.py`/`osf_raw_epoch_dataset.py`/
`osf_channel_loader.py` in full (not just their docstrings) and PhysioOmni's
own `model/neural_transformer.py`/`model/transformer.py` source directly —
see the per-decision "verified" notes below for exactly what was checked
and how.

### 15.0 What we're building — one-paragraph summary (for the paper later)

Stage 2 fine-tunes PhysioOmni's 4 frozen encoders via LoRA adapters
(`c_attn`, `c_proj` in every attention block — the same staged LP-FT
procedure used for OSF: Stage 1's already-trained sequence head warm-starts
Stage 2, then LoRA-adapted encoders and the head are fine-tuned jointly).
Architecturally, PhysioOmni's 4 independent encoders are wrapped as ONE
combined `nn.Module` and passed through a SINGLE `peft.get_peft_model()`
call — `peft`'s target-module matching is name-suffix-based across the
whole module tree, so one call correctly finds and wraps all 4×12=48
attention blocks (96 `c_attn`/`c_proj` Linear layers total) without any
per-encoder special-casing. This was live-verified against the real
checkpoint, not assumed. The one genuine engineering complication unique to
PhysioOmni (not present in OSF's single-backbone case) is that different
subjects have different *sets* of present modalities/channels (e.g. some
STAGES subjects are missing EMG or ECG) — Stage 2 preserves Stage 1's exact
zero-fill contract (a missing modality's encoder is never called for that
subject; its embedding slice stays exactly zero) via a batch-level
present/absent mask, rather than running the encoder on zero-padded input
(which would silently change the model's behavior relative to Stage 1).

### 15.1 Multi-encoder LoRA wrapping — RESOLVED, live-verified 2026-08-19

**Decision: one `CombinedPhysioOmniLoRAModel(nn.Module)` holding all 4
encoders (`self.eeg_encoder`, `self.eog_encoder`, `self.ecg_encoder`,
`self.emg_encoder`) + `self.sequence_head` as submodules, wrapped by a
SINGLE `peft.get_peft_model(combined, LoraConfig(target_modules=["c_attn",
"c_proj"], modules_to_save=["sequence_head"], ...))` call** — direct
extension of OSF's `CombinedOSFLoRAModel(backbone, sequence_head)` pattern
to 4 backbones instead of 1, not a different mechanism.

**Why this works (verified, not assumed):** `peft`'s `target_modules`
matching operates on each submodule's full dotted name, matching by suffix
— `"c_attn"` matches `eeg_encoder.blocks.3.attn.c_attn` exactly as readily
as it matches `eog_encoder.blocks.7.attn.c_attn`. There's no
backbone-count-awareness in the matching logic, so a single call over a
module tree containing 4 parallel encoder branches naturally wraps all of
them. Live-verified 2026-08-19 by constructing the real 4-encoder combined
module from the actual `PhysioOmni.pt` checkpoint and calling
`get_peft_model()` once:
```
trainable params: 289,002 || all params: 14,161,308 || trainable%: 2.04
LoRA modules per encoder: {'eeg_encoder': 24, 'eog_encoder': 24, 'ecg_encoder': 24, 'emg_encoder': 24}
```
24 = 12 blocks × 2 target modules (`c_attn`, `c_proj`) per encoder, ×4
encoders = 96 total LoRA-wrapped Linear layers — exactly the expected
count, confirmed live. (`n_layer=12` for all 4 encoders, also confirmed
directly from the checkpoint's `{mod}_encoder_args` — not assumed to match
EEG's value.) Four independent `get_peft_model()` calls would produce the
same wrapped modules with more bookkeeping (4 separate wrapped submodels,
4 state dicts to merge) for zero functional benefit — rejected.

**No separate learned fusion layer** — considered and rejected. PhysioOmni
ships no native cross-modality fusion (the flat 500-dim concatenation from
§6.3 is entirely our own construction, not something the checkpoint
provides), which might suggest Stage 2 should add one. But the sequence
head (LSTM/Transformer/MeanPool) already sits on the concatenated 500-dim
vector and already performs cross-modality fusion at every timestep before
temporal aggregation — a separate fusion layer would be redundant and
would blur what "the LoRA condition" means (adapting the pretrained model
vs. adding a new randomly-initialized component). Concatenation stays
exactly as-is from Stage 1.

### 15.2 Missing-modality handling — batch-level present-mask, not zero-input-forward

**The one genuinely new engineering problem PhysioOmni's Stage 2 has that
OSF's didn't**: OSF's raw signal is always exactly `[12, n_samples]` per
subject (missing channels are zero-filled at the raw level, see
`osf_channel_loader.load_and_resample_channels`), so every subject in a
batch has an identical tensor shape and the combined model's `forward()`
can naively reshape `[B,N,C,T] -> [B*N,C,T]` and run everything through one
backbone call. PhysioOmni's raw channels are NOT zero-filled at Stage 1 —
a modality with zero real channels is skipped entirely (`if not
channel_list: ... continue`, `extract_physioomni_embeddings.py` line ~248),
leaving that modality's embedding slice at literal zero with no encoder
forward pass ever run. Different subjects have genuinely different
*channel-presence patterns* (confirmed in the real extraction logs —
e.g. some STAGES subjects logged `zero-filled: ['EMG']` or `['ECG']`),
which a batched `[B,...]` tensor can't represent directly the way OSF's
uniform 12-channel case can.

**Decision: preserve Stage 1's exact zero-fill contract via a batch-level
presence mask, not by feeding zero-valued raw signal through the encoder.**
For each modality, `CombinedPhysioOmniLoRAModel.forward()`:
1. Builds a boolean `present[b]` for each subject `b` in the batch (True if
   that subject has ≥1 real channel for this modality — a per-subject,
   whole-recording property, constant across all of that subject's windows,
   so it's attached once per subject by the raw-signal cache/dataset, not
   recomputed per window).
2. Selects only the present subjects' epochs (`x[present]`), runs them
   through that modality's LoRA-adapted encoder (chunk-batched exactly like
   Stage 1's `chunk_batch_size` pattern), and scatters the resulting CLS
   outputs into the correct rows/columns of the `[B, N, 500]` embedding
   tensor.
3. Leaves `emb[~present, :, slot_start:slot_end]` at exactly zero — no
   forward pass, no gradient, identical behavior to Stage 1's embeddings
   for that modality/subject.

This is more engineering than OSF's case needed (a real instance of what
§15.1's old text flagged as "genuinely harder here"), but it's the
correct thing to do — feeding zero-valued raw signal through the encoder
instead would produce a *non-zero* learned output (transformers with
position embeddings and a CLS token don't map all-zero input to all-zero
output), silently diverging from what Stage 1's embeddings — and every
frozen-condition result already computed — actually represent for that
subject.

### 15.3 Raw signal cache — per-subject, per-slot files (not a single unified matrix)

**OSF's raw signal cache is one `[12, n_samples_64]` array per subject**
(`osf_channel_loader.save_signal_cache`/`load_signal_cache`) because every
subject has the same 12 channels (zero-filled if actually missing) at the
same 64Hz rate. **This doesn't transfer to PhysioOmni directly**: channels
are genuinely present-or-absent (not zero-filled) per subject, channel
*count* varies (EEG has 1 or 2 channels depending on cohort — SHHS gets 1,
per §4.5's final decision), and different modalities run at different
native rates (EEG/EOG at 200Hz, ECG/EMG at 500Hz, per §5.1) — no single
fixed-shape array covers all of that.

**Decision: one cache directory per subject, one `.npy` file per present
channel-slot, plus a small `meta.json`.**
```
{raw_cache_dir}/{dataset}/{subject_id}/
    EEG_C3.npy      (float16, 200Hz, only if present)
    EEG_C4.npy      (float16, 200Hz, only if present — omitted for SHHS, §4.5)
    EOG_HEO.npy     (float16, 200Hz, only if both LOC and ROC were present)
    ECG.npy         (float16, 500Hz, only if present)
    EMG.npy         (float16, 500Hz, only if present)
    meta.json       {"t_epochs": <int>, "slots_found": {...}, "slots_missing": [...]}
```
Directly mirrors `physioomni_channel_loader.load_subject_signals()`'s
existing return structure (`{"EEG": [(label, arr), ...], ...}`) — the
cache IS that structure, persisted, so building/reading it needs no new
data-shape design, just serialization. `t_epochs` (the same `min(...)`
computation Stage 1's extraction already does) is stored directly in
`meta.json` so shape lookups never need to open an array (cheaper than
even OSF's own `get_cached_epoch_count`, which still does a shape-only
`np.load(mmap_mode="r")`).

**Scope: apples + shhs + mros only (13,481 subjects), NOT stages.** None
of PhysioOmni's 4 Tier-1 tasks (`v2_physioomni_registry.yaml`) use STAGES
— apnea (the only task that would need it) is already excluded for
PhysioOmni entirely (no respiratory pathway). Building the cache for
STAGES' 1,513 subjects would be pure wasted compute. This is a real,
concrete scope reduction relative to OSF's own cache (which covers all 4
datasets since OSF's apnea task does use STAGES).

**Precompute is CPU-only, offline, built BEFORE any training job** — direct
answer to the standing instruction not to bury slow preprocessing inside
GPU training jobs (many training jobs would each redo it otherwise).
`load_subject_signals()` is pure `h5py`/`numpy`/`scipy.signal.resample` —
no GPU dependency at all, same as OSF's precompute script. Sharded via CPU
SLURM jobs (`--account=def-forouzan`, matching every other CPU-only job in
this project), NOT run on the login node.
**Wall-time is not assumed to match OSF's own precompute numbers** (OSF:
1104 APPLES subjects in 5.1 min at `--num-workers 8`, CPU-only) —
PhysioOmni's FFT-based `scipy.signal.resample` (needed since 128→200Hz and
128→500Hz have no exact-decimation shortcut, unlike OSF's clean 128→64Hz
2:1 case) is more expensive per sample than OSF's array-slice
decimation, so this could plausibly be slower. Time the first real shard
before assuming a total wall-time budget for the rest.

### 15.4 Raw-signal dataset class

`PhysioOmniRawEpochWindowDataset` — same split-matching discipline as
`OSFRawEpochWindowDataset` (§15.7 below), same windowing arithmetic
(`_build_seq2label_index`, `SubjectGroupedSampler`, K-sampling by split),
copied near-verbatim from that file since it's pure integer arithmetic
over `T`/`N`, unrelated to what's actually stored per epoch. The real
difference from OSF's version: `__getitem__` returns a dict of per-modality
raw windows (not one `[N,12,1920]` tensor), since channel presence varies
per subject:
```python
{
  "EEG": (x_eeg, chan_labels_eeg),   # x_eeg: [n_eeg_chans, N, 200] or None if absent
  "EOG": (x_eog, ["HEO"]) or None,
  "ECG": (x_ecg, ["ECG"]) or None,
  "EMG": (x_emg, ["EMG"]) or None,
  "mask": bool[N],                    # True = right-padded position (recording too short)
}, label
```
`collate_fn` (custom, like `PhysioOmniContextWindowDataset`'s own) batches
this into the `present[b]`-masked structure §15.2's `forward()` expects —
NOT `torch.utils.data.default_collate`, which can't handle the
per-subject-varying tuple/`None` structure. Reads only from the §15.3
cache — never the raw HDF5 — same as `OSFRawEpochWindowDataset`.

**Split-matching discipline — copied from OSF's real, previously-live bug
fix (§15.7), not theoretical here either**: filter subjects by "has a
Stage 1 PhysioOmni embedding file" FIRST (existence check only, contents
never read), before separately checking the Stage 2 raw-signal cache for
actually reading data. `np.random.default_rng(split_seed).shuffle()`
produces a completely different permutation if the filtered population
differs by even one subject — this must exactly match
`PhysioOmniContextWindowDataset`'s (Stage 1's) subject pool at
split-computation time, for the frozen-vs-LoRA comparison to be valid.

### 15.5 Combined model forward() — position-ID construction is duplicated, not shared

`CombinedPhysioOmniLoRAModel.forward()` needs the same per-modality
patch-reshape + `input_chans`/`input_times` position-ID construction
`extract_physioomni_embeddings.py`'s `_modality_forward()` already does —
but that function wraps its encoder call in `with torch.no_grad():`,
which is wrong for Stage 2 (LoRA gradients must flow through it).

**Decision: duplicate the ~20-line patch/position-ID logic into
`train_physioomni_lora.py` directly, rather than factor it into the shared
channel-loader module.** This deliberately follows OSF's own stated
precedent (`train_osf_lora.py`'s `load_osf_backbone()` docstring: "single
~15-line function... duplicated rather than imported... unlike the
channel-loading logic, which was genuinely at risk of drifting out of
sync") — a small, single-purpose function used in exactly two places (one
`no_grad`, one not) is lower total risk to duplicate carefully than to
factor out and thread a `grad_enabled` flag through, especially since
`extract_physioomni_embeddings.py` is an already-verified, already-in-
production script actively feeding real completed training runs
(checklist 1.10) — not touching it at all is the safest choice here.
**Also chunk-batches epochs the identical way Stage 1 does** (`chunk_batch_size`
epochs at a time as the batch dimension, each epoch independently
patchified with its own position-ID sequence reset per channel) — this is
load-bearing, not a style choice: PhysioOmni's positional-embedding tables
(`pos_embed = nn.Embedding(256, ...)`, `time_embed = nn.Embedding(512,
...)`) are sized for one epoch's own token layout, never a multi-epoch
concatenated sequence (confirmed directly from `neural_transformer.py`,
consistent with §19's already-documented native-context-ceiling finding).
Concatenating multiple epochs into one long per-item token sequence
instead of batching them would both exceed these tables' sizes for longer
contexts AND not match how the pretrained model was ever used.

### 15.6 Warm-start / LP-FT staging — identical to OSF's resolved design

Same mechanism as OSF's `warm_start_head_from_stage1()` /
`warm_start_from_stage2_30s()`, reused conceptually (own PhysioOmni
implementation, not imported — different checkpoint format): **30s always
warm-starts the sequence head from Stage 1's frozen-backbone checkpoint**
(`configs/phase0_physioomni_config.yaml`'s `results_dir`); **every other
context length warm-starts LoRA+head together from that same (task,
head)'s own already-converged 30s Stage 2 checkpoint** (branch, not a
chain — compute scales ~linearly with context length here too, likely
worse than OSF's given up to 4 encoder forward passes per epoch instead of
1, not yet measured). `peft`'s `modules_to_save` wraps `sequence_head` in a
`ModulesToSaveWrapper` (`.original_module` + `.modules_to_save["default"]`)
— the Stage-1-checkpoint warm-start must load into both copies explicitly,
exactly the gotcha OSF's checklist 2.3 already found and fixed; the
Stage-2-30s warm-start is already in `peft`'s own state-dict format
(`get_peft_model_state_dict`/`set_peft_model_state_dict` round-trip), no
special handling needed, same as OSF.

### 15.7 Lessons from OSF's real Stage 2 build, applicable here

Read directly from `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s Phase 2
checklist (items 2.1-2.5d) and `docs/OSF_EXPERIMENTS_GUIDE.md`'s Step 8 —
concrete, dated findings, not speculation:

- **Build the combined model (backbone(s) + sequence head) BEFORE peft
  injection**, then call `get_peft_model(combined, LoraConfig(...,
  modules_to_save=["sequence_head"]))` on the whole thing — this way
  `peft`'s own save/load and gradient-freezing logic treats the head as a
  first-class trainable submodule alongside the LoRA deltas, rather than
  bolting the head on after wrapping the backbone in isolation.
- **`peft`'s `modules_to_save` wraps the target submodule in a
  `ModulesToSaveWrapper`** holding two copies (`.original_module`, frozen,
  and `.modules_to_save["default"]`, the trainable copy actually used in
  the forward pass). A naive `load_state_dict()` for a Stage-1 warm-start
  will fail with a key-prefix mismatch against the wrapper — load into
  both inner copies explicitly, verified against the real wrapper
  structure (`peft.utils.other.ModulesToSaveWrapper`) before writing the
  fix, not guessed.
- **An offline raw-signal cache is not optional — build it from day one,
  not after discovering a stalled job.** OSF's first real Stage 2 GPU job
  stalled for 2+ hours (GPU essentially idle) because its raw-epoch
  dataset resampled/loaded signal live, from raw HDF5, on every
  `__getitem__`, redone from scratch for every task/head/context
  combination — the fix was a CPU-only offline precompute script
  (`precompute_osf_raw_signal_cache.py`) writing a per-subject `.npy`
  cache, read via `mmap`. **Plan PhysioOmni's Stage 2 with this cache from
  the start** — the shared channel-loader module (§7) should expose
  `save/load_signal_cache`-equivalent functions from the beginning, the
  same way OSF's `osf_channel_loader.py` does, so this isn't a mid-project
  surprise.
- **Split-matching discipline is a real, previously-live bug category, not
  a theoretical concern.** OSF's raw-epoch dataset originally filtered
  subjects by raw-HDF5 existence, while Stage 1's dataset filters by
  Stage-1-embedding-file existence — since
  `np.random.default_rng(split_seed).shuffle()` produces a completely
  different permutation if the filtered population differs even by one
  subject, this silently scrambled the entire train/val/test split for
  several cohorts before being caught. **PhysioOmni's Stage 2 raw-signal
  dataset must filter by "has a Stage-1 PhysioOmni embedding file" first**
  (existence check only, contents never read), exactly mirroring OSF's
  fix, before separately checking its own raw-signal cache for actually
  reading data.
- **Compute scales ~linearly with context length, not sub-linearly** — a
  window's full backbone forward+backward pass happens once per raw epoch
  in the window, every training step, since LoRA gradients must flow
  through the frozen layers to reach the adapters. OSF measured 30s ≈ 60
  min/epoch on `1g.10gb`; 240m (480× more epochs/window) would be ≈480×
  that — not independently-fine-tunable at every context length within a
  normal project timeline. **Expect this to apply to PhysioOmni's Stage 2
  too, likely worse** given up to 4 encoders instead of 1 (net effect on
  wall-clock unclear without a pilot — the individual encoders are smaller
  than OSF's ViT, so this isn't a guaranteed 4× multiplier, but don't
  assume it's better either without measuring).
- **Warm-start every context length other than the shortest from that same
  (task, head)'s own shortest-context LoRA checkpoint, not a chain and not
  independent per-length runs.** OSF's resolution (branch from 30s, not
  30s→10m→...→240m) is a direct, load-bearing design pattern to reuse:
  branching avoids making every length's result depend on the arbitrary
  sweep order, and is consistent with the architectural fact that the
  backbone has no per-length internal state. **State the same honest
  limitation in the paper if adopted for PhysioOmni**: the LoRA
  condition's backbone starting point becomes shared/inherited across
  lengths, a genuine departure from the frozen condition's "N is the only
  variable" purity, made necessary by compute constraints.
- **Effective-batch-size parity with Stage 1/SleepFM matters even when
  per-item compute is dominated by something else.** OSF's Stage 2 was
  initially shipped with a flat, ungrounded `batch_size=4` and no gradient
  accumulation, reasoned as "doesn't obviously matter since the backbone
  forward pass dominates cost" — this was correctly rejected: gradient
  noise/optimization dynamics depend on effective batch size regardless of
  what dominates per-item wall-clock. **Default to matching Stage 1's
  `effective_batch=32` via gradient accumulation for PhysioOmni's Stage 2
  too**, adjusting `context_micro_batch` down (not `effective_batch`) if
  memory is tight.
- **Full three-way argparse/config audits catch real, silent gaps.** OSF's
  Stage 2 shipped three times with something Stage 1 already had wired up
  correctly but Stage 2 silently didn't: `context_lr_overrides` not
  applied, `mixed_precision`/`weighted_sampler`/`persistent_workers` not
  wired despite being fully supported by the (already-imported) `run_epoch`
  function. **Before trusting any PhysioOmni Stage 2 config/script,
  side-by-side-diff it against Stage 1's own config/script for every
  training-loop option**, not just the ones that seem architecture-relevant.

### 15.8 Memory mitigation ladder (unchanged from OSF's, apply in this order)

1. Gradient checkpointing through each encoder's transformer blocks.
2. Request a larger GPU memory allocation.
3. Only if both fail: cap the LoRA condition at the longest tractable
   context length, keep the frozen condition (Stage 1) at all 6 lengths,
   state the compute ceiling explicitly.

### 15.9 Wall-time/compute budget — unknown, do not assume, do not
extrapolate from OSF's numbers

OSF's own Stage 2 wall-time tables are themselves still uncalibrated
placeholders as of this revision (checklist 2.6, not yet done) — there is
no trustworthy OSF number to scale from even qualitatively yet, let alone
one that accounts for PhysioOmni's up-to-4-encoder structure. Run a short
real pilot (few epochs, smallest context, actual GPU allocation) before
committing to any `--time` budget, same discipline OSF's own plan already
states for itself.

### 15.10 Scope for this round

Same 4 Tier-1 tasks as Stage 1 (sex, sleep efficiency, BMI, age — apnea
still excluded, no respiratory pathway). **`lstm`/`transformer` heads only
for the first pass, `mean_pool` deferred** — matches OSF's own Stage 2
scoping decision (real LoRA compute cost, not worth tripling the sweep
before the first two heads' numbers exist). All 6 context lengths per
(task, head), but only 30s is independently trained from Stage 1's
warm-start — 10m/40m/80m/120m/240m all branch from that same (task,
head)'s own 30s LoRA checkpoint (§15.6), so the "12 experiments" from
Stage 1's registry becomes effectively "8 (task,head) pairs × 6 contexts =
48 training runs" here, not 12×6=72 — `mean_pool`'s absence and the
warm-start branching both reduce real compute relative to a naive reading
of the registry.

**New files this phase** (mirrors the file list style already in this
plan's earlier sections):

| File | Role |
|---|---|
| `src/nsrr_tools/datasets/physioomni_channel_loader.py` (extended, not replaced) | Add `cache_path_for`/`save_signal_cache`/`load_signal_cache`/`load_meta`-equivalent functions (§15.3) alongside the existing Stage 1 functions — same file, since both stages' loading logic must never drift apart |
| `scripts/precompute_physioomni_raw_signal_cache.py` | Offline, CPU-only, builds §15.3's per-subject-per-slot cache — run before any Stage 2 training job |
| `jobs/precompute_physioomni_raw_signal_cache.sh` | CPU-only SLURM job (`--account=def-forouzan`, mirrors `jobs/precompute_osf_raw_signal_cache.sh` / this repo's own `extract_physioomni_embeddings_cpu.sh`) |
| `src/nsrr_tools/datasets/physioomni_raw_epoch_dataset.py` | `PhysioOmniRawEpochWindowDataset` + `SubjectGroupedSampler` + custom `collate_fn` (§15.4) |
| `configs/phase0_physioomni_lora_config.yaml` | Forked from `phase0_physioomni_config.yaml`, adds `lora:` section + `data.raw_signal_cache_dir` + `dataset.stage1_embedding_dir` |
| `scripts/train_physioomni_lora.py` | `CombinedPhysioOmniLoRAModel` (§15.1/15.2/15.5) + warm-start logic (§15.6) + sweep `main()`, reusing `run_epoch`/`compute_metrics`/`compute_monitor_metric`/`append_to_summary`/`_classify_failure` imported from `train_physioomni_context_sweep.py` (same reuse pattern as `train_osf_lora.py`) |
| `jobs/train_physioomni_lora_gpu.sh` | GPU job script, same auto-resume mechanism as every other job script in this project |
| `scripts/infer_physioomni_lora_subject_windows.py` | Live-backbone inference (raw signal in, not precomputed embeddings) |
| `jobs/infer_physioomni_lora_subject_windows_gpu.sh` | GPU job script |
| `experiments/v2_physioomni_lora_registry.yaml` | 8 `(task, head)` entries (4 tasks × lstm/transformer) |
| `scripts/gen_commands_physioomni_lora.py` | Structural fork of `gen_commands_osf_lora.py`, pointed at the new registry/job scripts |

---

## 16. Implementation checklist

**Do not start any of this until the user explicitly says to proceed past
this planning pass** — per `CLAUDE.md`'s current instruction ("implementation
starting now, in parallel with OSF's remaining sweep — no longer gated on
OSF finishing first," but still gated on an explicit go-ahead for actual
code/branch work, which this revision is not).

### Phase 0 — Setup — ✅ mostly done 2026-08-17 (0.4-0.6 remain)
- [x] 0.1 Build `physioomni_env` — **resolved: dedicated venv, not shared
      with `osf_env`.** Package-wise `osf_env` was a near-perfect match
      (`torch==2.5.1`/`torchvision==0.20.1`/`torchaudio==2.5.1` already
      exactly right, plus `einops`/`pandas`/`scikit-learn`/`huggingface_hub`
      already present), but `osf_env`'s `nsrr_tools_src.pth` points at
      `/home/boshra95/NSRR-tools/src` (the OSF worktree) — reusing it
      as-is would silently import OSF's branch's `nsrr_tools`, and
      repointing that `.pth` would break OSF's own still-active
      environment. Built `/home/boshra95/physioomni_env` fresh from
      `/home/boshra95/osf_env_requirements.txt` (147 packages, no version
      relaxation needed this time — already CC-wheelhouse-proven), own
      `.pth` → `/home/boshra95/NSRR-tools-omni/src`, verified
      `import nsrr_tools` resolves to this worktree. Confirmed
      `nsrr_tools.core` still fails (`pyedflib` missing, same as
      `osf_env`) — `nsrr_tools.datasets` placement (§7) confirmed correct.
      `wandb` not installed (same known CC Go-toolchain issue OSF hit;
      inert since our own training scripts will make it opt-in like
      OSF's). Full detail: `PHYSIOOMNI_CLAUDE.md`.
- [x] 0.2 Download `PhysioOmni.pt`, strict-load-verify — **done, real
      script not a throwaway snippet**: `scripts/verify_physioomni_checkpoint.py`
      (VSCode debug config "🫀 PhysioOmni Phase0 Step2: Verify Checkpoint").
      **Result: zero missing keys on all 4 encoders** (one expected
      `mask_token` unexpected-key per encoder, an MSM-pretraining-only
      component `FT.py`'s own `strict=False` load already accounts for).
      Resolves the previously-open checkpoint-key-prefix hypothesis (§2,
      §18) — confirmed, not just inferred from code reading. **Bonus
      finding**: the checkpoint's own `{modality}_encoder_args` dicts
      contain the exact `NTConfig` kwargs per modality — read these at
      runtime (mirroring OSF's `metadata` dict pattern) instead of
      hardcoding §3's table when writing the extraction script (§7).
      Total real encoder params: 13,871,304. `VQ.pt` confirmed **not**
      downloaded/needed — the verification only ever touches
      `*_encoder.`-prefixed keys, exactly as hypothesized.
- [x] 0.3 Save the paper PDF locally — done,
      `/home/boshra95/related_work/PhysioOmni.pdf` (arXiv 2504.19596v3, 15
      pages), same shared non-git-tracked location as `OSF.pdf`.
- [x] 0.4 SHHS EEG channel decision — **✅ RESOLVED 2026-08-17, see §4.5.**
      Investigated the dual-EEG-channel question `docs/OSF_CHANNEL_REPROCESSING_PLAN.md`
      §4 had left open (confirmed live: 100% of SHHS subjects carry both
      `EEG` and an `EEG(sec)`-family channel; correlation between them in a
      real file is r=0.18, confirming they're genuinely distinct
      electrodes, not a duplicate). **Decided against duplicating** SHHS's
      single available `EEG` channel into two slots (unlike OSF, which
      correctly does duplicate, for its own different architectural
      reasons) — PhysioOmni's variable-length token sequence per modality
      means SHHS can legitimately be fed **one real EEG channel, not two**,
      requiring zero reprocessing. A lightweight future option (recover the
      currently-discarded `EEG(sec)` channel via an additive, non-destructive
      patch job, not a full SHHS reprocessing) is documented in §4.5 but
      deliberately not pursued now — revisit only if SHHS results look
      degraded.
- [x] 0.5 Empirically validate the normalization approach (§5.2, §14 item 2)
      — **✅ RESOLVED 2026-08-18 via checklist 1.3's real smoke test**: 3
      real subjects (2 APPLES + 1 SHHS) through the actual frozen encoder,
      zero NaNs, non-degenerate CLS output std (~0.8-1.3) across every
      modality slice.
- [x] 0.6 Empirically validate the sample-rate resampling approach (§5.1,
      §14 item 3) — **✅ RESOLVED 2026-08-18**, same evidence as 0.5 (both
      checks share the same real-data pipeline).
- [x] 0.7 Create the `physioomni-implementation` branch, forked from
      `osf-implementation` (§1) — not `main`. **Went further than
      originally planned**: this branch now has its own `git worktree` at
      `/home/boshra95/NSRR-tools-omni`, running in its own VSCode window,
      completely isolated from the `osf-implementation` worktree at
      `/home/boshra95/NSRR-tools` — see this section's own "Hard
      constraint: worktree/directory isolation" note above.

### Phase 1 — Stage 1 (frozen encoders)
- [x] 1.1 Implement `src/nsrr_tools/datasets/physioomni_channel_loader.py`
      (§7) — **done 2026-08-18.** Built as the shared utility from day
      one, per §7's design. Smoke-tested via a new
      `scripts/test_physioomni_channel_loader.py` (VSCode debug config
      "🫀 PhysioOmni Phase1 Step1: Test Channel Loader") against 2 real
      subjects × all 4 cohorts: zero NaNs, all resampled lengths exactly
      match `round(n_samples_128hz * native_hz / source_hz)`, and —
      the one thing checked explicitly, not just implicitly — **SHHS gets
      exactly 1 real EEG channel, non-SHHS cohorts get 2**, confirming
      §4.5's final decision is correctly implemented. **A real bug was
      caught and fixed during this step, not in the loader itself but in
      its own test**: the test's first draft computed "expected length"
      via `get_epoch_count()*30*native_hz`, which truncates to whole 30s
      epochs — wrong, since `load_subject_signals()` deliberately returns
      the *full* resampled night (epoch truncation is a later, separate
      step, matching OSF's own `load_and_resample_channels()` design).
      This only surfaced on STAGES (`BOGN00004.h5`: 4,884,352 raw samples
      at 128Hz isn't an exact 30s-epoch multiple), not the other 3
      cohorts' sampled subjects, whose sample counts happened to divide
      evenly — a reminder that a 2-subject smoke test can miss real
      test-logic gaps a slightly different sample would have caught.
      Fixed by computing the expected length the same way the loader
      itself does, not via the (correctly epoch-truncating, but not the
      right comparison here) `get_epoch_count()`.
      **Also resolved during this step, a real correction to §5.2's own
      2026-08-17 draft**: traced `signal_processor.py` directly (not
      inferred) and found the earlier "LOC/ROC = volts, everything else =
      µV" hypothesis was wrong — the actual unit depends on the source
      EDF file's own header, confirmed cohort-dependent (APPLES's `ECG` is
      µV-scale, SHHS's `ECG` is volts-scale, same canonical channel name).
      `invert_normalization()` self-calibrates per channel via its own
      stored `std` instead of a hardcoded table — see §5.2's revised text.
- [x] 1.2 Implement `scripts/extract_physioomni_embeddings.py` +
      `configs/phase0_physioomni_config.yaml` (§6.3, §9) — **done
      2026-08-18.** Runs each of the 4 frozen encoders independently per
      subject (no unified fusion model exists, §3) and concatenates their
      CLS outputs into the flat `[T, 500]` embedding. VSCode debug configs
      added: "🫀 PhysioOmni Phase1 Step2" (APPLES 2-subject and SHHS
      1-subject variants).
- [x] 1.3 Smoke-test on real APPLES + SHHS subjects (CPU) — **done
      2026-08-18, real results not just a dry run**:
      - APPLES (`APL0001`, `APL0003`): shapes `(1143, 500)`/`(970, 500)`
        — exactly matching checklist 1.1's channel-loader-only epoch
        counts for the same subjects — zero NaNs, `eeg_channel_count: 2`,
        `modalities_zero_filled: []`. Non-degenerate CLS output std across
        every 100-dim block (~0.8–1.3), confirming the encoder isn't
        producing collapsed/saturated output.
      - SHHS (`200001_v1`): shape `(1084, 500)`, again exactly matching
        checklist 1.1's prediction, zero NaNs, **`eeg_channel_count: 1`**
        — confirms §4.5's final decision (one real EEG channel, not
        duplicated) works correctly all the way through the actual frozen
        encoder forward pass, not just the channel loader in isolation.
        EEG block std (~0.89–1.12) is in the same range as APPLES's,
        not degenerate despite the shorter (30 vs. 60 token) input
        sequence.
      - **This also resolves checklist 0.5/0.6** (normalization +
        resampling empirical validation) — sane, non-NaN, non-degenerate
        encoder output on real data is exactly the check those items
        called for.
      - **CPU timing, for planning purposes**: ~584-938s/subject
        (~10-16 min) — confirms CPU is debug-only, real extraction needs
        a GPU job (checklist 1.9); no GPU timing yet.
      - **User checkpoint** — re-verify via the "🫀 PhysioOmni Phase1
        Step2" debug configs before continuing to 1.4.
- [x] 1.4 Implement `src/nsrr_tools/datasets/physioomni_context_window_dataset.py`
      (§8) — **done 2026-08-18.** Forked from
      `osf_context_window_dataset.py`. **One real simplification beyond a
      rename, not just cosmetic**: since the saved embeddings are
      genuinely 2D `[T, 500]` (§6.3's design — PhysioOmni has no
      meaningful sub-token dimension the way OSF's CLS+mean-pooled-patch
      pair does), every 3D `(N_SUBTOKENS, EMBED_DIM)` pad-block shape and
      the post-hoc `.reshape(N, FLAT_DIM)` call in OSF's fork are dropped
      entirely — pad blocks are plain `(n, EMBED_DIM)` and windows are
      already the right shape with no reshape needed. `PATCH_SECONDS=30`,
      `PATCHES_PER_EPOCH=1`, `min_recording_patches=480` all carried over
      unchanged (same 30s-epoch units as OSF, no recomputation needed).
- [x] 1.5 Smoke-test the dataset class — **done 2026-08-18, with an honest
      population-size caveat.** Formalized as
      `scripts/test_physioomni_context_window_dataset.py` (VSCode debug
      config "🫀 PhysioOmni Phase1 Step3"). Real results against the 3
      subjects extracted so far (checklist 1.3): train/val/test correctly
      split to 2/0/1 subjects (val landing empty is arithmetic —
      `int(3*0.15)==0` — not a bug, confirmed by inspection); item
      retrieval at `30s`/`10m`/`full_night` all produced correctly-shaped
      `(N, 500)` float32 tensors, zero NaNs, correct dtypes, zero
      unexpected padding. **What this does NOT yet cover, flagged
      explicitly rather than glossed over**: with only 3 subjects (all
      with T well above every non-full_night context tested), no padding
      branch was exercised, and K-sampling wasn't tested at a realistic
      pool size the way OSF's own dataset-class smoke test used (10
      subjects/cohort specifically to get this right). The
      per-modality-missing degeneracy check (§14 item 8) is really a
      downstream sequence-head-training concern, not something the
      dataset class itself behaves differently for (it reads whatever
      values are in the `.npy`, zero-filled or not, uniformly) — deferred
      to Phase 1.6+'s real training smoke test, not applicable here.
      **Re-run this test with more extracted subjects before trusting it
      at full-sweep scale** — a real, open follow-up, not resolved by this
      pass. **User checkpoint.**
- [x] 1.6 Implement `scripts/train_physioomni_context_sweep.py` (§10) +
      job script (§13) — **done 2026-08-18, verified with a real CPU smoke
      test, not just a code read.** Forked structurally from
      `scripts/train_osf_context_sweep.py` with identical
      `run_epoch`/`compute_metrics`/`compute_monitor_metric`/
      `append_to_summary`/`_classify_failure`/`train_one_context`/`main`
      boundaries; only the dataset import (`PhysioOmniContextWindowDataset`
      family) and the `wandb_project` default changed — no
      `--zero-modalities` flag, matching OSF's own fork (no 4-modality-group
      structure to ablate). Job script:
      `jobs/train_physioomni_context_sweep_gpu.sh`, forked from
      `jobs/train_osf_context_sweep_gpu.sh` with paths/env/job-name updated
      for this repo (`physioomni_env`, `logs_physioomni`,
      `physioomni_ctx_sweep`), same SIGUSR1 auto-resume + `--requeue`
      mechanism.

      **Population expansion needed first.** Checklist 1.5's 3-subject
      smoke test was too small for a real training run (val split empty).
      Simulated `PhysioOmniContextWindowDataset`'s exact split logic
      (`np.random.default_rng(42).shuffle`) against candidate population
      sizes to find the minimal sufficient count *before* spending
      extraction time, rather than guessing: **8 apples + 8 shhs = 16
      total** reliably gives both classes in val for `sex_binary`.
      Extracted the additional subjects (apples 5→8, shhs 4→8) via a new
      **CPU-only** SLURM job, `jobs/extract_physioomni_embeddings_cpu.sh`
      (`--account=def-forouzan`, no `_gpu` suffix, 16 CPUs/32GB, forked from
      `jobs/precompute_osf_raw_signal_cache.sh`'s CPU-job pattern) —
      submitted via `sbatch`, not run on the login node. Both jobs
      completed with 0 errors (2 apples + 4 shhs newly extracted, rest
      skipped as already present).

      **Smoke test**: `--config configs/phase0_physioomni_config.yaml
      --context 30s --datasets apples shhs --batch-size 2 --no-wandb --cpu`.
      Result: `Items — train: 55 | val: 10 | test: 15`, val AUROC a real
      number (0.52, no longer NaN) from the first eligible epoch, so
      `best_model.pt` saved correctly this time — the NaN-AUROC /
      `best_model.pt`-never-saved failure mode from checklist 1.5's
      3-subject population (a known, pre-existing pattern, already seen in
      OSF's own Stage 2 build) is resolved by having enough val-split
      subjects, not by a code change. Checkpoint resume was also exercised
      live (a stale 11-epoch checkpoint from an earlier run was correctly
      picked up and continued to early-stop at epoch 21). Ran to
      `Status: SUCCESS — all context lengths completed.` end-to-end. Train/
      val metrics are non-degenerate (val bal_acc 0.5, auroc 0.52); test
      metrics are degenerate (bal_acc 0.0, auroc NaN) — expected and
      unconcerning at this population size (2-3 subjects in test, plausibly
      single-class or a saturated single-class prediction), not a script
      bug — this is a mechanics smoke test, not a real result, and will be
      revisited naturally once the real extraction (checklist 1.9) runs at
      full population. **User checkpoint.**
- [x] 1.7 Implement `scripts/infer_physioomni_subject_windows.py` (§11) +
      job script (§13) — **done 2026-08-18, verified with a real CPU smoke
      test.** Fork of `scripts/infer_osf_subject_windows.py` with identical
      structure; only the dataset import changed and the batch-size
      auto-scaling reference (`_ref_bs=64`, `_ref_N=480`) was kept
      unchanged rather than re-derived, since PhysioOmni's dataset uses the
      exact same token unit as OSF's (one row per 30s epoch,
      `PATCHES_PER_EPOCH=1`) — not a re-verified-on-GPU number, same open
      caveat OSF's own script already carries. Job script:
      `jobs/infer_physioomni_subject_windows_gpu.sh`, forked from
      `jobs/infer_osf_subject_windows_gpu.sh` with paths/env/job-name
      updated.

      **Environment fix needed along the way**: `physioomni_env` had no
      working `pyarrow`, so `df.to_parquet()` failed
      (`ImportError: ... pyarrow or fastparquet`) even though the rest of
      the pipeline (dataset build, checkpoint auto-detection from state-dict
      shapes, forward pass) ran correctly. Root cause: Compute Canada
      ships `pyarrow` as a "dummy" stub wheel that only registers metadata
      — the real compiled package lives under the `arrow` environment
      module's own site-packages and has to be exposed via a `.pth` file
      (found the working pattern already present in `osf_env`:
      `pyarrow_arrow_module.pth` pointing at
      `.../easybuild/software/2023/x86-64-v4/Compiler/gcccore/arrow/18.1.0/lib/python3.10/site-packages`).
      Copied the same `.pth` into `physioomni_env` — fixes `pyarrow` for
      every script in this env going forward, not just this one, with no
      `module load` needed at runtime (the `.pth` path is absolute).

      **Smoke test**: `--config configs/phase0_physioomni_config.yaml
      --task sex_binary --task-type seq2label --head lstm --context 30s
      --datasets apples shhs --split val --cpu`, against checklist 1.6's
      trained checkpoint. Result: `Dataset items: 1,796 (subjects: 2)` →
      `Saved 1,796 rows → .../val_windows.parquet`,
      `Segment accuracy: 50.84%`. Verified the parquet directly: correct
      7-column schema (`subject_id, dataset, true_label, pred_label,
      prob_class0, prob_class1, window_idx`), zero NaNs. Ran to
      `All contexts processed successfully.` **User checkpoint.**
- [x] 1.8 Implement `experiments/v2_physioomni_registry.yaml` +
      `scripts/gen_commands_physioomni.py` (§12) — **done 2026-08-18,
      verified against the real checklist-1.6 checkpoint.**
      `v2_physioomni_registry.yaml` mirrors `experiments/v2_registry.yaml`
      (fast-channel, paper-primary — not `v2_full_registry.yaml`, since
      PhysioOmni only needs EEG/EOG/ECG/EMG and the fast-channel `psg/`
      tree already carries all of it, unlike OSF) for the same 4 tasks
      (sex, sleep efficiency, BMI, age) × 3 heads = 12 experiments.
      **apnea_binary is deliberately absent** — no respiratory pathway in
      PhysioOmni, confirmed at 4 independent code locations (see
      CLAUDE.md's PhysioOmni section). `gen_commands_physioomni.py` is a
      structural fork of `gen_commands_osf.py` (itself unmodified in
      pipeline logic — `list`/`train`/`infer`/`analyze`/`build-heatmap`/
      `collect`/`threshold-tuning`/`status`/`runs`), pointed at the new
      registry, `jobs/{train,infer}_physioomni_*_gpu.sh`, and
      `physioomni_env`; wall-time tables are placeholder copies of OSF's
      own (themselves not yet calibrated), to revisit after the first real
      GPU sweep (checklist 1.11) — same open caveat OSF's generator already
      carries. Verified live: `list` correctly shows `sex_binary_lstm` as
      `trained (1/6 contexts)` (picking up checklist 1.6's 30s checkpoint);
      `train`/`infer`/`status` all produce correct sbatch commands and
      paths against the real registry and job scripts.
- [x] 1.9 Implement `jobs/extract_physioomni_embeddings_gpu.sh` — **done
      2026-08-18, verified with real GPU allocations, including an
      empirical `chunk_batch_size` A/B test.** Fork of
      `jobs/extract_osf_embeddings_gpu.sh` — same `--start-idx`/`--end-idx`
      sharding, SIGUSR1 auto-resume, skip-if-exists convention.

      **Test 1** (job 55499713, `END=20 DATASETS="apples shhs"`): 20
      subjects (12 new, 8 already-cached skipped correctly), completed in
      3m40s wall / 1.5 min extraction time on a real H100 MIG `1g.10gb`
      slice — first real signal that GPU is dramatically faster than the
      CPU path (checklist 1.6's extraction logs: 50-450s/subject on CPU).

      **Test 2** (the `chunk_batch_size` question): rather than assuming
      OSF's own finding (`chunk_batch_size` 16→64 gave a measured 3.28x
      speedup — CLAUDE.md's OSF section) transfers here, ran a controlled
      A/B on matched, same-dataset, same-size batches — 20 fresh shhs
      subjects at `chunk_batch_size=64` (job 55500731, 1.4 min = 84s →
      4.2s/subject) vs. the next 20 fresh shhs subjects at
      `chunk_batch_size=16` (job 55500915, 1.4 min = 81s → 4.05s/subject).
      **Result: no meaningful difference** — unlike OSF, PhysioOmni's
      per-subject extraction here is not `chunk_batch_size`-bound (likely
      because per-subject fixed overhead dominates at these window sizes,
      not per-call GPU compute). Kept `chunk_batch_size: 16` (the original
      default, no reason to change it) — documented directly in
      `configs/phase0_physioomni_config.yaml`'s comment and the job
      script's own header, both with the real measured numbers rather than
      an assumption borrowed from OSF.

      **Real measured throughput: ~4.1s/subject** on a single H100 MIG
      `1g.10gb` slice (apples + shhs; mros/stages not yet tested, expected
      similar — same channel-loading path) — 15-100x faster than CPU.
      Across the ~14,994-subject population (apples 1104 + shhs 8444 +
      mros 3933 + stages 1513), that's ~17 hours serial on one GPU;
      checklist 1.10 will shard into parallel jobs (~2500 subjects/job,
      ~2.85h each) rather than one long run, mirroring OSF's own sharding
      convention. **User checkpoint.**
- [x] 1.10 Run full embedding extraction, all 4 datasets — **done
      2026-08-19.** Sharded across 6 GPU jobs (`--start-idx`/`--end-idx`
      ranges spanning the full 14,994-subject concatenated list) plus the
      earlier small test-batch jobs. Final counts: apples 1104/1104, shhs
      8444/8444, mros 3933/3933, stages 1512/1513 — **14,993/14,994
      (99.99%)**, zero errors across 8 of 9 extraction jobs. The 1 gap is
      `stages/STLK00096`: "No PhysioOmni-relevant channels found at all"
      — not a pipeline bug, this exact subject is already flagged in
      `CLAUDE.md`'s OSF section as a known data-quality outlier
      (SleepFM-only, missing from OSF's own population too). No further
      action needed — proceed to 1.11 with this population as final.
- [ ] 1.11 Run the Stage 1 sweep (4 tasks × 3 heads × 6 contexts = up to 72
      training runs), then inference, then analysis
- [ ] 1.12 Re-run the channel-completeness audit against real extraction
      output, confirm it matches §4.3's expectations
- [x] 1.13 Write `docs/PHYSIOOMNI_EXPERIMENTS_GUIDE.md` incrementally as
      each step above is built and verified, mirroring
      `docs/OSF_EXPERIMENTS_GUIDE.md`'s Step 0-7 structure — the concrete
      "how to actually run it" counterpart to this plan doc's "why" —
      **initial version done 2026-08-18**, covering Steps 0-7 (all real,
      verified content — real commands, real measured numbers, real
      output paths, not placeholders) since checklist 1.1-1.9 were all
      already done and verified by the time this was written. Step 8
      (LoRA) is a placeholder pointing at OSF's own Step 8 by analogy,
      since Phase 2 hasn't started. Includes an explicit scope note
      (per the user, 2026-08-18): unlike SleepFM (the main model, full
      protocol including channel ablation), PhysioOmni gets exactly two
      rounds — frozen backbone + seq head (this guide), then LoRA + seq
      head (Step 8) — no channel ablation, no full-channel round, no
      sleep staging. **Living document — keep updated alongside this plan
      doc as 1.10/1.11/1.12 progress and as problems get found/fixed**,
      same convention as `docs/OSF_EXPERIMENTS_GUIDE.md`.

### Phase 2 — Stage 2 (LoRA), detailed design per §15 (2026-08-19)
- [x] 2.1 Resolve the multi-encoder LoRA-wrapping design question (§15.1)
      — **done 2026-08-19, live-verified** against the real checkpoint
      (single `CombinedPhysioOmniLoRAModel` + single `get_peft_model()`
      call, 96 LoRA-wrapped Linear layers across all 4 encoders confirmed
      by direct construction, not reasoned about in the abstract). No user
      design conversation was needed in the end — `peft`'s name-suffix
      matching resolved it mechanically once actually tested.
- [x] 2.2 Implement the shared raw-signal-cache extension to the channel
      loader (§15.3) and an offline precompute script + CPU-only job —
      **done 2026-08-19.** Added `cache_subject_dir`/`save_signal_cache`/
      `load_signal_cache`/`load_meta`/`get_cached_t_epochs`/`cache_exists`
      to `physioomni_channel_loader.py`, live round-trip tested (synthetic
      signals dict → save → load, shapes/values/dtypes verified byte-exact
      modulo float16 precision). `scripts/precompute_physioomni_raw_signal_cache.py`
      + `jobs/precompute_physioomni_raw_signal_cache.sh` (CPU-only,
      `--account=def-forouzan`, mirrors `precompute_osf_raw_signal_cache.sh`).
      Scope: apples+shhs+mros only (13,481 subjects), not stages (§15.3) —
      not yet run for real (that's checklist 2.6's prerequisite).
- [x] 2.3 Implement `src/nsrr_tools/datasets/physioomni_raw_epoch_dataset.py`
      (§15.4) — **done 2026-08-19.** `PhysioOmniRawEpochWindowDataset` +
      `SubjectGroupedSampler` + `physioomni_lora_collate_fn` +
      `PhysioOmniLoRABatch` (a tiny wrapper implementing only `.to(device)`/
      `.size(0)`, so Stage 2's per-modality-grouped raw batches — genuinely
      not a fixed-shape tensor, §15.2 — flow through
      `train_physioomni_context_sweep.py`'s `run_epoch()` completely
      unmodified: that function's only two touches on its `x` argument are
      exactly those two calls). Filters subjects by Stage-1-embedding-file
      existence first (§15.7's split-matching lesson). `collate_fn`'s
      channel-count grouping (EEG: 1 vs. 2 channels, §4.5) live-tested with
      synthetic batches — verified correct grouping/shapes/batch_idx
      assignment for a 3-subject batch with mixed EEG channel counts and
      mixed EOG/ECG/EMG presence. **The subject-list/split logic itself
      also live-verified against the real config and real `sex_binary`
      task CSV** (not synthetic): 11,400 total subjects in the CSV,
      correctly filtered to 9,548 by Stage-1-embedding existence (exactly
      matching apples 1104 + shhs 8444, checklist 1.10's real population),
      correct 70% train split (6,683) — then correctly raised
      `FileNotFoundError` with a clear, actionable message once it checked
      for the (not-yet-built) raw-signal cache, exactly the designed
      fail-loud behavior rather than a silent wrong-data bug.
- [x] 2.4 Implement `scripts/train_physioomni_lora.py` (§15.5/15.6) —
      **done 2026-08-19, live-verified end-to-end against the real
      checkpoint** (not just unit-tested pieces). `CombinedPhysioOmniLoRAModel`
      with batch-level present-mask handling (§15.2: absent modalities get
      zero embedding slices with no encoder forward pass, matching Stage
      1's exact contract) and warm-start logic (§15.6). A full synthetic
      forward+backward pass (3 subjects, mixed 1-/2-channel EEG, mixed
      EOG/ECG/EMG presence, real 4-encoder `peft`-wrapped model) produced
      correctly-shaped logits, a finite loss, **and confirmed LoRA
      gradients flowed into all target modules and sequence_head
      gradients flowed too** — the hardest, most novel part of this
      design (reshape/position-ID construction + gradient-preserving
      scatter-write into the embedding tensor) is real-checkpoint-verified,
      not just reasoned about.
- [x] 2.5 `experiments/v2_physioomni_lora_registry.yaml` +
      `scripts/gen_commands_physioomni_lora.py` (§15.10 scope: 8
      `(task,head)` pairs, lstm/transformer only, mean_pool deferred) —
      **done 2026-08-19**, verified live: `list` shows all 8 experiments
      correctly (all `pending`, as expected — no Stage 2 training has run
      yet), `train`/`status` produce correct commands/output.
- [ ] 2.6 Short wall-time pilot at the smallest context length (§15.9) —
      **user checkpoint** before the full sweep. Prerequisite: run
      `jobs/precompute_physioomni_raw_signal_cache.sh` for real first
      (checklist 2.2's script) — **first real attempt in progress
      2026-08-20, two real bugs found and fixed along the way:**
      1. **OOM on the `[9000:13481]` shard** (mostly MrOS — longer
         recordings than apples/shhs): 16 workers' concurrent
         `scipy.signal.resample` buffers exceeded the job's 32GB request,
         confirmed even running completely alone on its node (ruling out
         cross-job contention as the sole cause). Fixed: `--mem` raised
         32000M → 64000M (node capacity is 768GB, plenty of room).
      2. **Non-atomic `meta.json` write**: workers killed by the OOM/
         SIGTERM events above left 81 zero-byte `meta.json` files (their
         `.npy` siblings were all fully intact) — `cache_exists()` treats
         mere existence as "done," so these silently blocked reprocessing
         and broke the first real LoRA training attempt with a
         `JSONDecodeError` the moment one was read. Fixed:
         `save_signal_cache()` now writes `meta.json` via temp-file +
         `os.replace` (same pattern already used elsewhere in this
         project, e.g. `infer_osf_lora_subject_windows.py`'s resume
         checkpoint). The 81 corrupt files were deleted so they get
         reprocessed on the next run.
      3. **Separately, `v2_physioomni_lora_registry.yaml`'s `logs_dir` was
         pointed at the same directory as Stage 1's** (unlike OSF's clean
         `logs_osf`/`logs_osf_lora` split) — real consequence, not just a
         findability annoyance: a Stage 1 job and the first Stage 2 LoRA
         job both trained `sex_binary_lstm`/30s and appended to the exact
         same persistent `.log` and status `.jsonl` files, corrupting both
         stages' job-history tracking. Fixed: Stage 2 now gets its own
         `logs_physioomni_lora/` directory (registry + all three Stage 2
         job scripts).
- [ ] 2.7 Full three-way config/argparse audit against Stage 1 and OSF's
      Stage 2 (§15.7's lesson) before trusting the config's stated options
- [ ] 2.8 Run the full Stage 2 sweep (§15.10: 48 training runs — 8
      `(task,head)` × 6 contexts, only 30s independently trained per
      pair), applying the memory-mitigation ladder (§15.8) if needed

### Phase 3 — Results
- [ ] 3.1 Compile Stage 1 + Stage 2 results against `phase0_v3`
      (paper-primary), stating §0's caveats plainly (small sleep-specific
      pretraining scale, arXiv-only/unreviewed, own-paper HMC numbers below
      a non-FM baseline, no apnea) — same "no silently-incomplete cells"
      discipline as OSF
- [ ] 3.2 Report back before starting MOMENT — do not start the next
      model's plan unprompted

---

## 17. Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Go/no-go | **Go**, with caveats stated in §0 | Weaker case than OSF on several honest axes, but real released weights + code, no blocking technical gap, and a mixed/negative result is still informative |
| Task scope | 4 tasks: `sex_binary`, `sleep_efficiency_binary`, `bmi_binary`, `age_class` — **not** `apnea_binary` | No RESP/airflow pathway exists anywhere in the model (§3/§0.4) |
| Source HDF5 tree | **Fast-channel `psg/`** (paper-primary), **not** full-channel `psg_full/` | The fast strategy's own priority-order caps already keep everything PhysioOmni needs, confirmed both from config math and real HDF5 key listings (§4) |
| Comparison baseline | `phase0_v3` (paper-primary SleepFM) | Follows from the source-tree decision — more directly relevant than OSF's `phase0_v3_full` |
| Embedding storage | Concatenated per-modality CLS vectors, `[T, 500]` per epoch, flat | No unified fusion module exists in the pretrained weights (§3); simpler than a 3D shape since there's no shared dim across modalities to preserve |
| Channel loader placement | `src/nsrr_tools/datasets/`, not `src/nsrr_tools/core/` | Avoids `nsrr_tools.core`'s eager `pyedflib` import, a real gotcha confirmed in `osf_env` (§7) — verify holds for `physioomni_env` too, don't assume |
| Channel loader design | Shared module from day one, built before the extraction script depends on it | OSF only factored this out after Stage 2 needed it, then had to regression-test the refactor; building it shared from the start avoids that extra step (§7) |
| Reprocessing | **No** — reuse existing fast-channel `psg/` HDF5s, including SHHS's | Every PhysioOmni-needed channel already exists; SHHS's single `EEG` channel is used as-is, not reprocessed (§4.5) |
| SHHS EEG handling | **One real, non-duplicated channel** — not OSF's duplicate-into-both-slots approximation | **Decided 2026-08-17, see §4.5.** Verified `EEG`/`EEG(sec)` are genuinely distinct (100% co-occurrence, r=0.18 correlation) but not pursuing the fix to recover the second channel now. PhysioOmni's variable-length token sequence per modality (unlike OSF's fixed-tensor ViT) makes 1-channel EEG a legitimate input, not a workaround — duplicating would fabricate an r=1.0 fake "second channel" unlike anything in pretraining |
| SHHS `EEG(sec)` recovery (deferred) | **Documented, not implemented** — a lightweight additive patch job (not full SHHS reprocessing) could recover a genuine second EEG channel for SHHS, benefiting both OSF and PhysioOmni | Not needed for correctness now that duplication is off the table; revisit only if SHHS results look degraded on EEG-dependent tasks (§4.5) |
| Sample rate | Resample to PhysioOmni's own native per-modality rates (200Hz EEG/EOG, 500Hz ECG/EMG) | Matches the reference prep scripts exactly (§5.1); patch duration is a fraction of real time that depends on this choice |
| Normalization | Invert each channel's stored `normalization_stats`, then a **self-calibrating** V→µV correction (`if abs(std) < 1.0: ×1e6`) based on that channel's own stored `std`, then `/100` | Traced `signal_processor.py` directly: no explicit unit conversion anywhere, so the unit MNE returns is file/cohort-dependent, not a fixed per-channel-name rule (confirmed: APPLES's `ECG` is µV-scale, SHHS's `ECG` is volts-scale) — a hardcoded per-channel table would be wrong; still needs an empirical Step 0 check (§14) |
| LoRA target modules | `c_attn`, `c_proj` (per encoder, up to 4 encoders) | The only two Linear layers in PhysioOmni's `Attention` block (§3) — genuinely new code, no existing LoRA precedent in the repo |
| Multi-encoder LoRA wrapping | **Undecided** — one `get_peft_model()` call over all 4 vs. 4 independent calls | Needs a design conversation before Phase 2 starts (§15.1) |
| Stage 2 raw-signal caching | Offline, from day one, not discovered after a stalled job | OSF's real Stage 2 lost 2+ hours to exactly this before fixing it (§15.2) |
| Stage 2 split-matching | Filter by Stage-1-embedding-file existence, not raw-HDF5 existence | A live, previously-real bug in OSF's own Stage 2 — same fix applies here (§15.2) |
| Stage 2 warm-start | Every context length other than the shortest branches from that (task, head)'s shortest-context LoRA checkpoint | Compute scales linearly with context length (§15.2); direct reuse of OSF's own resolved design |
| Checkpoint needed | `PhysioOmni.pt` only, not `VQ.pt` | **Confirmed 2026-08-17** — strict-load-verified via `scripts/verify_physioomni_checkpoint.py`: all 4 encoders load with zero missing keys from `PhysioOmni.pt` alone (checklist 0.2) |
| `physioomni_env` vs. `osf_env` | **Dedicated venv**, not shared | `osf_env`'s `nsrr_tools_src.pth` points at the OSF worktree; sharing would either silently import OSF's branch's code or require repointing a still-in-use shared environment — neither acceptable (checklist 0.1, `PHYSIOOMNI_CLAUDE.md`) |
| Per-modality `NTConfig` kwargs | Read from the checkpoint's own `{modality}_encoder_args` dicts at runtime | **Found 2026-08-17** while verifying the checkpoint — the exact kwargs are stored in the checkpoint itself, no need to hardcode §3's table (mirrors OSF's own `metadata`-dict pattern) |

---

## 18. Known open questions

- ~~**Checkpoint key-prefix hypothesis**~~ — **✅ RESOLVED 2026-08-17**, see
  checklist 0.2 and §17.
- ~~**Whether `physioomni_env` can share `osf_env`'s environment**~~ — **✅
  RESOLVED 2026-08-17: no, dedicated venv**, see checklist 0.1 and §17.
- ~~**Normalization mismatch**~~ — **✅ RESOLVED 2026-08-18**, see §5.2's
  revised self-calibrating method and checklist 0.5/1.3.
- ~~**Sample-rate resampling method**~~ — **✅ RESOLVED 2026-08-18**, see
  checklist 0.6/1.3.
- ~~**SHHS EEG-duplication decision**~~ — **✅ RESOLVED 2026-08-17, see
  §4.5 and checklist 0.4**: not duplication, one real channel.
- **Multi-encoder LoRA-wrapping design (§15.1)** — genuinely undecided,
  needs a design conversation with the user before Phase 2 starts.
- **GitHub code repo's missing LICENSE** — the weights are CC-BY-4.0 but
  the training/inference *code* has no stated license.
- **Per-modality-missing degeneracy** (§8) — flagged as a real failure mode
  (whole modality zero-filled), distinct from SHHS's real-but-shorter EEG
  case (§4.5), not yet empirically checked.
- **SHHS `EEG(sec)` recovery** (§4.5) — a lightweight, additive future fix
  is documented and estimated cheaper than a full SHHS reprocessing, but
  not timed or implemented — deliberately deferred until/unless SHHS
  results look degraded enough to justify it.
- **PhysioOmni Stage 1's own wall-time/compute profile** — genuinely
  unknown, architecturally different from OSF's (four small transformers ×
  up to 4 forward passes per epoch during extraction vs. one ViT) — do not
  reuse OSF's wall-time tables even as a starting guess without a real
  pilot (§12).

---

## 19. Native context ceiling and the Plan A/B/C decision (2026-08-18 —
write straight into the paper's Methods/Limitations, this section exists
for that purpose)

### What we wanted

The project's standing preference, stated in `CLAUDE.md`'s "three usage
modes" section: **Plan A (native long context, no sequence head) is the
fairest comparison whenever a backbone can actually support it** — it
tests the backbone's own long-context capability rather than our own
aggregation machinery. Plan B (short-segment embedder + our sequence head)
is the documented fallback, used for SleepFM and OSF only because their
architectures leave no other option (SleepFM: hard 300s chunk requirement;
OSF: hard 30s chunk requirement, zero cross-epoch attention in the model).
PhysioOmni's `NeuralTransformer` is architecturally different from both —
a variable-length token sequence per modality, no fixed native window —
so before defaulting to Plan B again, it was worth asking concretely: how
far could Plan A actually go here, and is it far enough to matter for a
30s→240m sweep?

### The ceiling, precisely (not the flat "512 seconds" first estimated)

The `time_embed` table caps any single forward pass at 512 *tokens*, not
512 seconds — the real-world duration one call can span depends on that
modality's own patch size (samples) and resample rate (Hz), both fixed by
the pretrained checkpoint (changing either would require retraining the
`TemporalConv` weights). Using PhysioOmni's own reference resample rates
(§5.1 — the rates its own prep scripts use, not a rate we invented):

| Modality | Patch size | Resample rate | Seconds/patch | **Native ceiling (512 patches)** |
|---|---|---|---|---|
| EEG | 200 samples | 200Hz | 1.0s | **512s ≈ 8.5 min** |
| EOG | 100 samples | 200Hz | 0.5s | **256s ≈ 4.3 min** |
| ECG | 100 samples | 500Hz | 0.2s | **102.4s ≈ 1.7 min** |
| EMG | 100 samples | 500Hz | 0.2s | **102.4s ≈ 1.7 min** |

**These are theoretical single-channel maxima, not what pretraining
typically used — a real, verified nuance worth stating precisely in the
paper rather than the simpler-but-misleading "8.5-minute ceiling"
framing.** Traced the actual pretraining window-construction logic
directly (`prepare_dataset/prepare_CAP.py:178`, identical formula in
`prepare_tuh.py`/`prepare_DEAP.py`):
```python
time = 512 // len(eegCh)   # seconds per sample, applied to ALL 4 modalities
```
i.e. pretraining **deliberately saturates the 512-token EEG budget every
time**, splitting it across however many EEG channels a given source
montage has (CAP/TUH montages typically carry anywhere from ~8 to ~26 EEG
channels — exact tally not exhaustively counted, but multi-channel is the
norm, not the single-channel edge case). A 16-channel montage gets
`512/16=32` real seconds per sample; a 26-channel montage gets `512/26≈19`
seconds. **The model's typical pretraining exposure was tens of seconds
of real time across many simultaneous channels, not minutes of real time
on one channel** — the 8.5-minute figure is a real, valid upper bound
(useful for stating "how far could this architecture go at all"), but not
representative of what the encoder actually learned to expect.

**Downstream fine-tuning (HMC, the model's own validated sleep-staging
task) uses a different, fixed convention**: exactly 30 real seconds per
sample regardless of channel count
(`prepare_dataset/prepare_HMC_downstream.py`, `row[' Duration'] != 30:
continue` — a hard AASM-epoch filter, not a computed value) — the *same*
30-second unit this plan already uses for embedding extraction (§6.3).
**This is a genuine point of confidence for Option 1 below, not just an
arbitrary convenience choice**: 30 seconds is not an unfamiliar duration
to this encoder — it's literally the unit PhysioOmni's own authors used to
fine-tune and validate it on a real sleep-staging task.

### Why 240m (or even most of the sweep) is not reachable via Plan A

240 minutes = 14,400 seconds. Even EEG's best-case theoretical ceiling
(512s) is short by ~28×. Closing that gap would require resampling EEG
down to ~7Hz and ECG/EMG down to ~3.6Hz to fit 14,400s into 512 patches —
not "more efficient use of capacity," but destruction of the signal (EEG
below ~30Hz loses essentially all clinically meaningful frequency content;
3.6Hz can't resolve an ECG waveform). No resample-rate choice makes 240m
native. **More consequentially for the sweep as a whole**: converting the
six sweep points to seconds (30s, 600s, 2400s, 4800s, 7200s, 14400s)
against the ceiling table above shows **10m (600s) already exceeds even
EEG's theoretical 512s maximum** — so every sweep point except the
trivial 30s one already requires aggregation beyond a single native
forward pass, for every modality, regardless of resample-rate tuning.
Plan A is architecturally inapplicable to this study's sweep beyond its
very first point.

### Options considered

1. **Keep 30-second epochs** (matches SleepFM's and OSF's own epoch-unit
   convention exactly, and — per the finding above — matches PhysioOmni's
   *own* HMC downstream fine-tuning convention too, not just our other two
   backbones'). Simplest, most directly comparable across all three
   models on the same epoch unit. Zero rework of already-built/tested
   Phase 1.1/1.2/1.4 code.
2. **Grow the native chunk size toward PhysioOmni's own ceiling** (up to
   ~90-100s, bound by ECG/EMG — the shortest of the four, since a single
   chunk spanning all 4 modalities over the same real-world duration is
   capped by whichever modality's ceiling is smallest). Uses more of the
   architecture's per-chunk attention span, at the cost of moving away
   from the SleepFM/OSF-shared 30-second convention and requiring real
   rework across the already-implemented pipeline (`EPOCH_SECONDS` is a
   module constant threaded through the channel loader, extraction
   script, and dataset class).
3. **A standalone supplementary experiment** (not a sweep point): feed
   each modality its own true maximum native window as a single item, no
   sequence head, straight to a classification head — a "PhysioOmni's own
   best native representation" reference number, analogous in spirit to
   OSF's Plan A discussion but capped at ~1.7-8.5 min instead of a long
   context. Not a substitute for the sweep (one fixed condition, not a
   function of context length), and not decided against permanently —
   just not pursued now.

### What we chose, and why (for the paper)

**Option 1 — 30-second epochs, unchanged from the rest of this plan.**
State this plainly in the Methods/Limitations section: PhysioOmni is
evaluated via Plan B (short-segment embedder + our sequence head) at
every context length in the sweep, **the same as SleepFM and OSF** — not
because Plan A wasn't considered or because PhysioOmni's architecture is
as rigid as the other two (it genuinely isn't — see the ceiling table
above), but because even its own best-case native ceiling (8.5 minutes
for EEG, materially less for the other three modalities) falls short of
every sweep point except the shortest. The 30-second unit chosen is not
arbitrary relative to PhysioOmni itself, either — it matches the model's
*own* downstream fine-tuning convention (HMC), a fact worth citing
directly if this comparison is challenged in review. Growing the native
chunk size (Option 2) was considered and set aside: the achievable gain
(30s → ~100s, bound by ECG/EMG) is small relative to the sweep's own span
(up to 240m) and would break the "same epoch unit across all three
backbones" comparison principle for a benefit that doesn't change the
fundamental Plan B conclusion. The standalone native-max-window reference
experiment (Option 3) remains a live option for later, not implemented as
part of this plan.

---

## 20. Three-way model comparison: SleepFM vs. OSF vs. PhysioOmni (for
the paper — everything here is code-verified in this repo, not recalled
from memory, except where explicitly flagged as unverified)

| | SleepFM | OSF | PhysioOmni |
|---|---|---|---|
| **Native per-call window — hard requirement** | Exactly 300s (5 min); incomplete trailing chunks dropped (`scripts/extract_sleepfm_embeddings.py`: `chunk_size = sampling_freq*300`, "model requires full 5-min chunks") | Exactly 30s; no cross-epoch attention exists anywhere in the model (`osf/backbone/vit1d_cls.py`'s positional table is sized to exactly one epoch) | **No fixed requirement** — variable-length token sequence; real ceiling is per-modality (§19 table: 512s EEG / 256s EOG / 102s ECG / 102s EMG at PhysioOmni's own reference rates) |
| **Internal sub-patch (within one native call)** | 5s (640 samples @ 128Hz) — 60 sub-patches per 300s call | 1s (64 samples @ 64Hz) — 90 tokens (30 time-steps × 3 channel-groups) per 30s call | Per-modality: EEG 1s/patch, EOG 0.5s/patch, ECG/EMG 0.2s/patch (all at PhysioOmni's own reference resample rates) |
| **Channel handling per call** | All 4 modality groups (BAS/RESP/EKG/EMG) in one joint tensor | All 12 channels in one joint tensor | **Each of the 4 modalities is a fully separate forward call** — no shared tensor, no cross-modal attention anywhere in the pretrained weights |
| **What pretraining actually saw (real-world duration)** | 300s fixed, always (same as inference — no separate shorter-window pretraining regime found) | 30s fixed, always | **Variable, saturating a 512-EEG-token budget per sample**: `time = 512 // n_eeg_channels` (`prepare_CAP.py:178`, same formula in `prepare_tuh.py`/`prepare_DEAP.py`) — typically tens of seconds across many (8-26) simultaneous EEG channels, not minutes on one channel (§19) |
| **What the model's own downstream fine-tuning used** | N/A in this repo's scope (SleepFM is used frozen throughout) | N/A (OSF's own downstream benchmarks not the focus here) | **Fixed 30 real seconds**, matching AASM epoch length exactly (`prepare_HMC_downstream.py`, hard `Duration==30` filter) — the *same* unit this plan uses (§19) |
| **What our own extraction uses** | 5s (our stored granularity) inside 300s calls, batched | 30s (fixed by the architecture itself) | **30s** — a deliberate choice (§19 Option 1), not an architectural requirement; matches SleepFM's/OSF's own epoch unit *and* PhysioOmni's own HMC fine-tuning unit |
| **Output per native call** | 4 × 128-dim (one per modality group), at 5s granularity within the call | 1 × [CLS(768) ⊕ mean-pooled-patches(768)] — fully fused, no per-channel structure survives | 4 × independent CLS (200-dim EEG, 100-dim each EOG/ECG/EMG) — **no fusion in the pretrained weights at all**, concatenation happens only in our own extraction script |
| **Saved embedding shape (this project's convention)** | `[T, 4, 128]`, T = 5s patches/night, flat dim 512 | `[T, 2, 768]`, T = 30s epochs/night, flat dim 1536 | `[T, 500]`, T = 30s epochs/night — genuinely 2D, no sub-token axis needed (§6.3/§8) |
| **Sequence head input_dim** | 512 | 1536 | 500 |
| **Encoder parameter count** | Not verified in this repo this session (would need reading SleepFM's own model-loading code with the same rigor OSF/PhysioOmni got — flagged as a gap, not asserted) | 85,325,568 (strict-load-verified, `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` §5) | 13,871,304 total across all 4 encoders (strict-load-verified, checklist 0.2) — EEG alone 7,839,576; EOG/ECG/EMG ~2,010,576 each |
| **Fusion mechanism in released weights** | Contrastive alignment across the 4 modality-group encoders during pretraining, but each still outputs its own 128-dim vector — not fused into one shared vector | Single unified ViT — full fusion, one representation | **None** — four independent tokenizers/encoders, no cross-modal attention or shared parameters anywhere; fusion (if any) is entirely downstream-constructed, by us or by PhysioOmni's own `FT.py` machinery (not part of the released checkpoint) |
| **Plan actually used for the sweep** | B (only option — architecture leaves no other choice) | B (only option — architecture leaves no other choice) | **B** (chosen — §19; architecture could theoretically support Plan A up to ~1.7-8.5 min, but that's short of every sweep point except 30s) |

**One clarifying note for the paper, since the table above could be
read as implying otherwise**: PhysioOmni is the only one of the three
where Plan B was a *choice* rather than an architectural inevitability —
worth stating explicitly, since it's a materially different (and more
favorable, from a "we gave this model a fair shot" standpoint) situation
than SleepFM's and OSF's genuinely-forced Plan B.

---

## Appendix: source citations for this plan's factual claims

**2026-08-13 pass** — verified directly against three sources:

1. **PhysioOmni GitHub repo, read directly at `/home/boshra95/PhysioOmni`**:
   `README.md`, `dataset.py`, `dataset.yaml`,
   `model/{neural_transformer,FT,transformer,MSM,VQ}.py`,
   `train_finetune.py`, `prepare_dataset/{prepare_HMC_downstream,prepare_CAP,prepare_tuh}.py`,
   `utils.py`.
2. **HuggingFace API, fetched live**:
   `https://huggingface.co/api/models/Weibang/PhysioOmni` (license,
   filenames), `curl -IL` against both checkpoint files (real file sizes
   via redirect-then-200).
3. **arXiv paper, fetched live**: `arxiv.org/abs/2504.19596` and
   `arxiv.org/html/2504.19596v3` (pretraining corpus composition, HMC
   downstream numbers, author affiliations, submission history/venue
   status).
4. **Our own cluster data, spot-checked live**: one real fast-channel HDF5
   file's channel-key listing and `normalization_stats` attribute per
   cohort, cross-referenced against `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s
   50-subject-per-cohort audit; `configs/preprocessing_params.yaml` and
   `configs/modality_groups.yaml` read directly to confirm the
   fast-channel strategy's priority-order caps.

**2026-08-17 pass (this revision)** — additionally verified directly
against:

5. **PhysioOmni repo file mtimes** (`ls -la --time-style=full-iso`) —
   confirmed unchanged since the 2026-08-10 clone date, nothing from
   sources 1-4 needed re-checking.
6. **OSF's finished/in-progress implementation, read directly in this
   repo (never modified — see §1's hard constraint)**:
   `scripts/extract_osf_embeddings.py`,
   `src/nsrr_tools/datasets/osf_channel_loader.py`,
   `src/nsrr_tools/datasets/osf_context_window_dataset.py` (header +
   constants),
   `scripts/train_osf_context_sweep.py` (CLI args + function structure),
   `scripts/infer_osf_subject_windows.py` (CLI args + batch-scaling
   formula), `configs/phase0_osf_config.yaml`,
   `configs/phase0_osf_lora_config.yaml`,
   `experiments/v2_osf_registry.yaml`,
   `jobs/train_osf_context_sweep_gpu.sh` (SLURM header).
7. **`CLAUDE.md`'s current "Status" subsection and
   `docs/OSF_EXPERIMENTS_GUIDE.md`'s Step 8** — OSF's Stage 1 results
   framing and Stage 2's operational how-to-run detail.
8. **`docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s Phase 2 checklist (items
   2.1-2.5d)** — the dated, concrete engineering lessons folded into §15
   above (offline raw-signal caching, split-matching, `peft`
   `ModulesToSaveWrapper`, compute-scaling-with-context, warm-start design,
   effective-batch-size parity, three-way config audits).
9. **`docs/PHYSIOOMNI_PLANNING_HANDOFF.md`** — the prompt this revision
   follows; its "hard constraints" section is reproduced verbatim as this
   plan's §1 file-isolation constraint.

**2026-08-17, Phase 0 execution pass** — real cluster actions, not
research reading, all outputs captured in `PHYSIOOMNI_CLAUDE.md`:

10. **`physioomni_env` built live** — `pip install -r
    /home/boshra95/osf_env_requirements.txt` into a fresh venv, package
    list and `.pth` contents inspected directly (not assumed) both before
    (`osf_env`) and after (`physioomni_env`) the build.
11. **`PhysioOmni.pt` downloaded live** from
    `https://huggingface.co/Weibang/PhysioOmni/resolve/main/PhysioOmni.pt`
    (267,795,410 bytes, matches the 2026-08-13 pass's HF-API size
    exactly) and **strict-load-verified against the real
    `NeuralTransformer`/`NTConfig` classes** from
    `/home/boshra95/PhysioOmni/model/neural_transformer.py` via
    `scripts/verify_physioomni_checkpoint.py` — the checkpoint's own
    `torch.load(...)` output (top-level keys, per-modality
    `*_encoder_args`, `load_state_dict(strict=False)` missing/unexpected
    key lists, parameter counts) is the direct source for every claim in
    checklist 0.2, not inference from code reading alone.
12. **Paper PDF fetched live** from `arxiv.org/pdf/2504.19596v3` to
    `/home/boshra95/related_work/PhysioOmni.pdf`, verified as a real
    15-page PDF (`file` command), not just a successful HTTP status.

**2026-08-17, SHHS EEG channel investigation (§4.5)** — a direct follow-up
on `docs/OSF_CHANNEL_REPROCESSING_PLAN.md` §4's explicitly-unverified
question, this time with real measurements:

13. **`output/channel_analysis/shhs_channels.csv` parsed directly**
    (pandas, all 8,444 rows) — checked for co-occurrence of `EEG` and each
    of the 5 `EEG(sec)`-family alias variants
    (`configs/channel_definitions.yaml`'s own `EEG:` list) within the same
    row's `channels` field. Result: 8,444/8,444 (100%).
14. **A real raw SHHS EDF loaded directly** (`mne.io.read_raw_edf`,
    `shhs1-203279.edf`) — both channels extracted via `raw.get_data()`,
    correlation computed via `np.corrcoef`. Result: r=0.18, not
    numerically identical (`np.allclose` → `False`).
15. **Full-population channel coverage recomputed for every channel
    PhysioOmni needs, all 4 cohorts**, directly against
    `output/channel_analysis/{apples,shhs,mros,stages}_channels.csv` using
    the real alias lists from `configs/channel_definitions.yaml` — the
    §4.3 table's numbers (this revision) are computed this way, not
    carried over from OSF's smaller 50-subject-per-cohort sample. Also
    traced STAGES's ECG gap (9.2% of subjects) to a genuine cause (a
    `Heartrate`-only channel, not a raw ECG waveform) rather than a
    fixable alias-list oversight.

**2026-08-18, native-context-ceiling investigation (§19-§20)**:

16. **`scripts/extract_sleepfm_embeddings.py` read directly** — confirmed
    SleepFM's hard 300s chunk requirement (`chunk_size =
    sampling_freq*300 = 38400` samples, "model requires full 5-min chunks"
    stated in the script's own docstring) and the 4-modality-group output
    structure (`[T,4,128]`, "BAS, RESP, EKG, EMG").
17. **`prepare_dataset/{prepare_CAP,prepare_tuh,prepare_DEAP}.py` read
    directly** — confirmed the exact pretraining window-construction
    formula (`time = 512 // len(eegCh)`, identical across all three
    scripts) behind §19's "pretraining typically saw tens of seconds
    across many channels, not minutes on one channel" finding.
18. **`prepare_dataset/prepare_HMC_downstream.py` re-read for this
    specific question** (already read once for §4.2's EOG-derivation
    finding) — confirmed the hard `row[' Duration'] != 30: continue`
    filter, establishing that PhysioOmni's own downstream fine-tuning
    (not just our own extraction choice) uses a fixed 30-second epoch.
