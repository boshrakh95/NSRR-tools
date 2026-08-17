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

Implementation, when it starts, lives in **`NSRR-tools`** (this repo, on a
`physioomni-implementation` branch forked from `osf-implementation` — not
`main` — so it inherits the OSF context this plan references throughout,
per `CLAUDE.md`'s current instruction) — `PhysioOmni/` itself stays a
read-only reference clone.

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

**Planning only — nothing built.** Two research/planning passes so far:

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
| SHHS (`203805_v2.h5`) | `Airflow, EEG, EKG, EMG, LOC, ROC` | EEG ❌ (generic only, no C3/C4), EOG ✅, ECG ✅, EMG ✅ (generic fallback) |
| MrOS (`AA1449_v2.h5`) | `Airflow, C3-M2, C4-M1, CHIN, EKG, LLEG, LOC, ROC` | EEG ✅, EOG ✅, ECG ✅, EMG ✅ (real `CHIN`) |
| STAGES (`STNF00032.h5`) | `Airflow, C3-M2, C4-M1, CHIN, EKG, LOC, ROC` | EEG ✅, EOG ✅, ECG ✅, EMG ✅ (real `CHIN`) |

Cross-referenced against `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s larger
50-subject-per-cohort audit for the overlapping channel set:

| Channel | APPLES | SHHS | MrOS | STAGES |
|---|---|---|---|---|
| `C3-M2`/`C4-M1` (EEG) | 100% | **0%** | 100% | 100% |
| `LOC`/`ROC` (EOG, both needed for the derived HEO) | 100% | 100% | 100% | 100% |
| `EKG`→`ECG-L` (ECG) | 100% | 100% | 100% | 90% |
| `CHIN`→generic `EMG` (EMG) | 100%† | 100%† | 100% | 100% |

†generic `EMG` fallback for APPLES/SHHS, real `CHIN` channel for MrOS/STAGES.

**The one real gap: SHHS has no distinguishable C3/C4 EEG**, identical root
cause to OSF's own SHHS finding. **Decision, mirroring OSF's own resolved
SHHS decision**: duplicate SHHS's single generic `EEG` channel into both
`'C3'` and `'C4'` position-lookup slots. This should be revisited together
with OSF's own SHHS caveat if results look degraded, not decided
independently per model. **Needs its own explicit re-confirmation with the
user before Phase 1** (checklist 0.4) — the pattern matching OSF's decision
is not the same as the user having actually confirmed it for PhysioOmni.

### 4.4 Reprocessing decision: **no raw EDF reprocessing needed, and no
full-channel tree needed either**

Every channel PhysioOmni needs is already present in the fast-channel
`psg/` HDF5s — the same tree SleepFM's paper-primary `phase0_v3` results
already use — except the one structural SHHS EEG gap above (which
reprocessing can't fix either). `docs/OSF_CHANNEL_REPROCESSING_PLAN.md`'s
three already-identified, already-deferred gaps (MrOS `ABD`, STAGES leg
EMG `LAT`/`RAT`, SHHS `NEW AIR`/airflow) **do not apply to PhysioOmni at
all** — none of those three channels are in PhysioOmni's input set. **No
new reprocessing plan doc is needed for PhysioOmni.**

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

### 5.2 Normalization — concrete method identified, needs empirical
validation before trusting extraction output

PhysioOmni's `dataset.py` divides raw signal by 100 (`EEG/100`, `EOG/100`,
etc., not z-score) before feeding the model — **different from OSF's
expectation of already-z-scored input.** Our `signal_processor.py`
z-scores per-channel during EDF→HDF5 preprocessing, but **every HDF5 file
stores the exact per-channel pre-normalization statistics needed to invert
this**, confirmed by reading a real file's root attributes directly
(`/scratch/boshra95/psg/apples/derived/hdf5_signals/APL1373.h5`): a
`normalization_stats` JSON attribute with `{mean, std, min, max}` per
channel, e.g. `"C3-M2": {"mean": -0.00155, "std": 15.25, "min": -221.6,
"max": 219.9}`. **Recommended approach**: `x_original = x_zscored * std +
mean` per channel (using each channel's own stored stats), then scale to
match whatever raw unit PhysioOmni's own prep scripts assume, then divide
by 100.

**One real wrinkle found while reading these stats, worth flagging
explicitly**: the stored pre-normalization scales are **not uniform across
channels** — `C3-M2`/`C4-M1`/`EKG`/`EMG`/`Airflow` have `std` in the
~15-25 range (consistent with already-µV-scale raw amplitudes), but
`LOC`/`ROC` have `std` around `1.6e-5` and min/max around `±0.0002`
(consistent with raw **volts**, not µV — roughly 4-5 orders of magnitude
smaller). **A single flat "convert V→µV" rule applied uniformly would be
wrong** — the unit-inversion step needs to check each channel's actual
recovered scale (or track it explicitly per channel from the original EDF
units) rather than assuming every channel was in the same unit before
z-scoring. Confirm against `signal_processor.py`'s unit handling directly
before writing the extraction script, not just from the stored stats'
magnitudes alone.

**Still flagged as a Step 0 pilot check (§14), not assumed to work
correctly on the first attempt**: after inverting to raw scale and
applying the `/100` convention, run a handful of real epochs through the
frozen encoder and check for NaNs/degenerate (near-zero or saturated) CLS
outputs before trusting any extraction at scale.

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
3. **Log which modality was zero-filled per subject** (same
   `_channel_fill_log.jsonl` convention as OSF's extraction script) — the
   SHHS EEG-duplication case (§4.3) and any true per-subject absence both
   need to be visible in this log before trusting results.

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
    "EEG":  {"C3": ["C3-M2"], "C4": ["C4-M1"]},   # up to 2 channels, both fed
    "EOG":  {"HEO": ["LOC", "ROC"]},               # derived: LOC - ROC
    "ECG":  {"ECG": ["EKG", "ECG-L"]},
    "EMG":  {"EMG": ["CHIN", "EMG"]},
}
EPOCH_SECONDS = 30
NATIVE_HZ = {"EEG": 200, "EOG": 200, "ECG": 500, "EMG": 500}   # §5.1
PATCH_SAMPLES = {"EEG": 200, "EOG": 100, "ECG": 100, "EMG": 100}   # §3's table

def build_channel_candidates(dataset: str, cfg_candidates: dict) -> dict:
    """Same SHHS-special-case pattern as OSF's build_channel_candidates()
    (§4.3) — for SHHS, EEG's C3/C4 candidates both resolve to the single
    generic 'EEG' key instead of C3-M2/C4-M1."""

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
head learning to ignore the zero slice entirely in a way that's
mathematically fine but worth being aware of when interpreting SHHS's
results, since SHHS's EEG slice will be the *duplicated-generic-channel*
case, not the *zero-filled* case — two different degeneracy risks, not one).

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
4. **SHHS EEG-duplication decision** (§4.3) — explicit user re-confirmation
   before writing extraction code, not assumed to carry over from OSF's
   already-confirmed decision just because the pattern matches.
5. **Small-scale pilot, end-to-end** — `--limit 5` extraction on one
   dataset, inspect `.npy` shape/values, then a tiny Stage 1 training run
   (`--context 30s`, one task, one head) before submitting a full sweep.
6. **`physioomni_env` compatibility check** — confirm whether it can share
   `osf_env` or needs its own build (checklist 0.1); confirm the
   `nsrr_tools.core` eager-import gotcha (§7) applies the same way.
7. **Cohort filter unit check** — confirm `min_recording_patches: 480` is
   applied in epoch units, same discipline OSF's own Step 0 checklist used.
8. **Per-modality-missing degeneracy check** (§8) — confirm a subject with
   an entire modality zero-filled doesn't produce a pathological (not just
   suboptimal) training signal, distinct from SHHS's duplicated-channel
   case.

---

## 15. Stage 2 (LoRA) — outline, informed by OSF's real Stage 2 lessons

**Per the hand-off's explicit scope, this section stays at outline level —
target modules and staging approach, not full file specs.** What's new
this revision is folding in real, hard-won lessons from OSF's own
(still-in-progress) Stage 2, so PhysioOmni's eventual Stage 2 doesn't
rediscover the same problems from scratch.

### 15.1 What's architecturally harder here than for OSF

- **No single backbone to wrap.** OSF's Stage 2 wraps one ViT; PhysioOmni's
  needs up to 4 separate `NeuralTransformer` instances LoRA-wrapped (either
  independently or together under one `get_peft_model()` call over a
  parent module holding all four) — **genuinely undecided, needs a design
  conversation before Phase 2 starts**, not resolved by this plan.
- **No pretrained fusion to warm-start from.** OSF's combined model is
  `(ViT, sequence_head)` — nothing else to build. PhysioOmni needs its own
  fusion step (the flat-concatenation from §6.3, or a small learned fusion
  layer) between the four LoRA-adapted encoders and the sequence head —
  genuinely new code, not a fork of anything PhysioOmni ships.
- **LoRA target modules**: `c_attn`, `c_proj` (§3) — the natural PEFT
  choice, but applied across up to 4×12=48 attention modules instead of
  OSF's 12, with no existing `get_peft_model()` call anywhere in
  PhysioOmni's repo to reference (OSF at least had the checkpoint's own
  naming conventions as an implicit hint; PhysioOmni has nothing analogous).

### 15.2 Lessons from OSF's real Stage 2 build, applicable here

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

### 15.3 Memory mitigation ladder (unchanged from OSF's, apply in this order)

1. Gradient checkpointing through each encoder's transformer blocks.
2. Request a larger GPU memory allocation.
3. Only if both fail: cap the LoRA condition at the longest tractable
   context length, keep the frozen condition (Stage 1) at all 6 lengths,
   state the compute ceiling explicitly.

### 15.4 Wall-time/compute budget — unknown, do not assume, do not
extrapolate from OSF's numbers

OSF's own Stage 2 wall-time tables are themselves still uncalibrated
placeholders as of this revision (checklist 2.6, not yet done) — there is
no trustworthy OSF number to scale from even qualitatively yet, let alone
one that accounts for PhysioOmni's up-to-4-encoder structure. Run a short
real pilot (few epochs, smallest context, actual GPU allocation) before
committing to any `--time` budget, same discipline OSF's own plan already
states for itself.

---

## 16. Implementation checklist

**Do not start any of this until the user explicitly says to proceed past
this planning pass** — per `CLAUDE.md`'s current instruction ("implementation
starting now, in parallel with OSF's remaining sweep — no longer gated on
OSF finishing first," but still gated on an explicit go-ahead for actual
code/branch work, which this revision is not).

### Phase 0 — Setup
- [ ] 0.1 Build `physioomni_env` — check whether it can share `osf_env` or
      needs its own venv (§14 item 6); verify the `nsrr_tools.core` eager
      `pyedflib` import gotcha (§7) applies the same way before deciding
      module placement
- [ ] 0.2 Download `PhysioOmni.pt`, strict-or-partial-load-verify against
      `FT.py`'s four `NeuralTransformer` encoders (§14 item 1) — do **not**
      download `VQ.pt` unless this reveals it's actually needed
- [ ] 0.3 Save the paper PDF locally (mirrors `related_work/OSF.pdf`)
- [ ] 0.4 Confirm the SHHS EEG-duplication decision (§4.3) with the user
- [ ] 0.5 Empirically validate the normalization approach (§5.2, §14 item 2)
- [ ] 0.6 Empirically validate the sample-rate resampling approach (§5.1,
      §14 item 3)
- [ ] 0.7 Create the `physioomni-implementation` branch, forked from
      `osf-implementation` (§1) — not `main`

### Phase 1 — Stage 1 (frozen encoders)
- [ ] 1.1 Implement `src/nsrr_tools/datasets/physioomni_channel_loader.py`
      (§7) — the shared utility, built first so both the extraction script
      and (eventually) Stage 2 import from it
- [ ] 1.2 Implement `scripts/extract_physioomni_embeddings.py` +
      `configs/phase0_physioomni_config.yaml` (§6.3, §9)
- [ ] 1.3 Smoke-test on real APPLES + SHHS subjects (small `--limit`, CPU),
      verify no NaNs, verify SHHS's EEG-duplication and any true zero-fills
      match §4.3's expected pattern — **user checkpoint**
- [ ] 1.4 Implement `src/nsrr_tools/datasets/physioomni_context_window_dataset.py`
      (§8) — fork `ContextWindowDataset`, following OSF's fork pattern
- [ ] 1.5 Smoke-test the dataset class at 30s/10m/full_night contexts,
      including the per-modality-missing degeneracy check (§14 item 8) —
      **user checkpoint**
- [ ] 1.6 Implement `scripts/train_physioomni_context_sweep.py` (§10) +
      job script (§13), written with clean `run_epoch`/`compute_metrics`
      function boundaries from the start (§10's reuse principle) —
      **user checkpoint**
- [ ] 1.7 Implement `scripts/infer_physioomni_subject_windows.py` (§11) +
      job script (§13) — **user checkpoint**
- [ ] 1.8 Implement `experiments/v2_physioomni_registry.yaml` +
      `scripts/gen_commands_physioomni.py` (§12)
- [ ] 1.9 Implement `jobs/extract_physioomni_embeddings_gpu.sh`, test via a
      small real GPU allocation before trusting it for the full run —
      apply the `chunk_batch_size`-first tuning lesson (§13) if throughput
      looks off
- [ ] 1.10 Run full embedding extraction, all 4 datasets
- [ ] 1.11 Run the Stage 1 sweep (4 tasks × 3 heads × 6 contexts = up to 72
      training runs), then inference, then analysis
- [ ] 1.12 Re-run the channel-completeness audit against real extraction
      output, confirm it matches §4.3's expectations
- [ ] 1.13 Write `docs/PHYSIOOMNI_EXPERIMENTS_GUIDE.md` incrementally as
      each step above is built and verified, mirroring
      `docs/OSF_EXPERIMENTS_GUIDE.md`'s Step 0-7 structure — the concrete
      "how to actually run it" counterpart to this plan doc's "why"

### Phase 2 — Stage 2 (LoRA), outline only per §15 — refine once Phase 1 runs
- [ ] 2.1 Resolve the multi-encoder LoRA-wrapping design question (§15.1)
      with the user before writing code
- [ ] 2.2 Implement the shared raw-signal-cache extension to the channel
      loader (§15.2) and an offline precompute script, from the start —
      do not wait for a stalled-job discovery the way OSF did
- [ ] 2.3 Implement `scripts/train_physioomni_lora.py`, filtering subjects
      by Stage-1-embedding-file existence first (§15.2's split-matching
      lesson), reusing Stage 1's `run_epoch`/etc. functions per §10
- [ ] 2.4 Short wall-time pilot at the smallest context length —
      **user checkpoint** before the full sweep
- [ ] 2.5 Full three-way config/argparse audit against Stage 1 and OSF's
      Stage 2 (§15.2's lesson) before trusting the config's stated options
- [ ] 2.6 Run the full Stage 2 sweep, applying the memory-mitigation
      ladder (§15.3) and warm-start-from-shortest-context pattern (§15.2)

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
| Reprocessing | **No** — reuse existing fast-channel `psg/` HDF5s | Every PhysioOmni-needed channel already exists except SHHS's structural EEG gap |
| SHHS EEG handling | Duplicate generic `EEG` into both `'C3'`/`'C4'` lookup slots | Same approximation as OSF's already-user-confirmed decision — **needs its own explicit re-confirmation before Phase 1** (checklist 0.4) |
| Sample rate | Resample to PhysioOmni's own native per-modality rates (200Hz EEG/EOG, 500Hz ECG/EMG) | Matches the reference prep scripts exactly (§5.1); patch duration is a fraction of real time that depends on this choice |
| Normalization | Invert each channel's stored `normalization_stats`, correcting for the per-channel unit inconsistency (µV vs. volts), then `/100` | `/100` raw-scale expectation vs. our already-z-scored HDF5 data is a real mismatch; still needs an empirical Step 0 check (§14) |
| LoRA target modules | `c_attn`, `c_proj` (per encoder, up to 4 encoders) | The only two Linear layers in PhysioOmni's `Attention` block (§3) — genuinely new code, no existing LoRA precedent in the repo |
| Multi-encoder LoRA wrapping | **Undecided** — one `get_peft_model()` call over all 4 vs. 4 independent calls | Needs a design conversation before Phase 2 starts (§15.1) |
| Stage 2 raw-signal caching | Offline, from day one, not discovered after a stalled job | OSF's real Stage 2 lost 2+ hours to exactly this before fixing it (§15.2) |
| Stage 2 split-matching | Filter by Stage-1-embedding-file existence, not raw-HDF5 existence | A live, previously-real bug in OSF's own Stage 2 — same fix applies here (§15.2) |
| Stage 2 warm-start | Every context length other than the shortest branches from that (task, head)'s shortest-context LoRA checkpoint | Compute scales linearly with context length (§15.2); direct reuse of OSF's own resolved design |
| Checkpoint needed | `PhysioOmni.pt` only, not `VQ.pt` | Traced `FT.py`'s loading code directly — only `*_encoder.`-prefixed keys are ever loaded; **still needs empirical confirmation once downloaded** (checklist 0.2) |

---

## 18. Known open questions

- **Checkpoint key-prefix hypothesis (§2, §17)** — not yet verified against
  the real file.
- **Normalization mismatch (§5.2)** — a concrete inversion method is
  identified and the per-channel unit inconsistency is flagged, but neither
  is empirically validated yet.
- **Sample-rate resampling method (§5.1)** — recommended (FFT-based or
  polyphase resampling for the non-integer 128→200/128→500 ratios) but not
  yet empirically validated against real signal.
- **Multi-encoder LoRA-wrapping design (§15.1)** — genuinely undecided,
  needs a design conversation with the user before Phase 2 starts.
- **Whether `physioomni_env` can share `osf_env`'s environment or needs its
  own fresh build** — not checked yet; PhysioOmni's README pins are close
  to but not identical to OSF's.
- **GitHub code repo's missing LICENSE** — the weights are CC-BY-4.0 but
  the training/inference *code* has no stated license.
- **Per-modality-missing degeneracy** (§8) — flagged as a real, distinct
  failure mode from SHHS's duplicated-channel case, not yet empirically
  checked.
- **PhysioOmni Stage 1's own wall-time/compute profile** — genuinely
  unknown, architecturally different from OSF's (four small transformers ×
  up to 4 forward passes per epoch during extraction vs. one ViT) — do not
  reuse OSF's wall-time tables even as a starting guess without a real
  pilot (§12).

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
