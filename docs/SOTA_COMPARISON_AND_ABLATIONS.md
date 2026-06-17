# SOTA Sleep Foundation Models — Comparison, Gaps, and Planned Ablations

*Written May 2026. Use as a reference for paper writing, related-work framing, and experiment planning.*

---

## 1. The Three Papers

### 1.1 OSF: On Pre-training and Scaling of Sleep Foundation Models
**Venue:** ICML 2026 (top ML venue)  
**Authors:** Shuai, Xu, Yang, Wang, Yang  
**arXiv:** 2603.00190  
**Code/weights:** Public — https://github.com/yang-ai-lab/OSF-Open-Sleep-FM  
**Benchmark:** SleepBench (166,500 hours, 9 cohorts, fully open-source)

**What they did:**
- Systematic comparison of 4 SSL pre-training families: contrastive (SimCLR), reconstruction (MAE, VQ-VAE), autoregressive (AR), and self-distillation (DINO)
- Three model scales: 1M, 5M, 85M parameters — demonstrate consistent scaling laws (performance does not plateau unlike prior models)
- **Key innovation: channel masking during pre-training.** Randomly drop modality channels during SSL training, forcing the model to be channel-invariant. Critical finding: existing models fail catastrophically when any channel is absent at inference. OSF's channel masking fixes this.
- Multi-source data mixing experiments (9 datasets, various proportions)
- Downstream tasks: 4-class sleep staging, arousal detection, hypopnea detection, O2 desaturation detection, central apnea detection, coronary disease, diabetes, hypertension (patient-level)
- Evaluation: in-domain (SHHS) and out-of-domain (MrOS)
- Best pre-training: DINO + channel masking

**Key numbers:** Sleep staging AUC = 97.3 (linear probe); coronary disease AUC = 68.1; best across all 9 datasets

**Relation to our work:**
- OSF pre-trains a new model from scratch. We use **frozen SleepFM embeddings** and train only the downstream head. Different problem statement.
- OSF evaluates patient-level clinical tasks directly comparable to ours (coronary disease, diabetes). Our task set is broader and more clinically nuanced (PSQI, ESS, OSA severity, depression).
- Their **scaling law result** is relevant: they show larger models and more data consistently help. SleepFM (which we use) was trained on ~100k hours — OSF uses 166k. This is a comparison point we should acknowledge.
- Their **channel masking finding** is the most important for us: it frames why channel ablation matters. We can cite this directly when motivating our own channel experiments.
- **Potential conflict:** OSF was pre-trained on SHHS and MrOS, which are our test cohorts. A direct AUROC comparison would be contaminated. State this in the paper.

**Paper writing notes:**
- Cite OSF when discussing robustness to channel availability.
- Cite the scaling law result when discussing SleepFM's scale relative to SOTA.
- Do NOT directly compare AUROC numbers on SHHS/MrOS between our model and OSF — different training data.
- Can argue: our framework is backbone-agnostic; replacing SleepFM with OSF embeddings is a one-line change (future work).

---

### 1.2 SleepMaMi: A Universal Sleep Foundation Model for Integrating Macro- and Micro-structures
**Venue:** arXiv 2026 (cs.AI), preprint  
**Authors:** Park, Na, Choi, Ryu, Shin, Kim  
**arXiv:** 2602.07628  
**Pre-training data:** 20,964 PSG recordings, 158,028 hours (SHHS, KISS, KVSS, PhysioNet, MESA, MrOS)

**What they did:**
- **Hierarchical dual-encoder:** Micro-Encoder (Transformer + MoE) processes within-epoch fine-grained events; Macro-Encoder (bidirectional **Mamba**) processes full-night temporal sequences.
- **DGCL (Demographic-Guided Contrastive Learning):** Uses patient demographics (age, sex, BMI) as free self-supervised alignment signals for the Macro-Encoder, avoiding reliance on noisy manual stage labels.
- Downstream tasks: sleep staging, sleep-disordered breathing (SDB) segmentation (apnea-hypopnea events), clinical disease prediction, few-shot evaluation
- Ablations: Micro-only vs. full model; individual demographic factor contribution (age alone, sex alone, BMI alone)
- Sleep staging accuracy: 81.9% (5-class, SHHS1); SDB F1 = 60.6%; few-shot at 30 samples = 72.6%

**Key numbers:** 30-sample few-shot accuracy = 72.6%

**Relation to our work:**
- SleepMaMi's Macro-Encoder (bidirectional Mamba over full-night sequences) is conceptually doing what our LSTM/Transformer head does: aggregating temporal structure across the night. **Their Mamba macro-encoder is the architecture-level analogue of adding Mamba as a head in our context sweep.**
- Their finding that the Macro-Encoder consistently outperforms the Micro-only baseline directly supports our hypothesis that long-context temporal modeling matters (our H1/H4).
- DGCL is orthogonal to our work (pre-training level). No need to implement it.
- Few-shot evaluation: they test label efficiency. We could add a data-efficiency curve at low training cost (see ablation section).

**Paper writing notes:**
- Cite SleepMaMi when arguing that long-context temporal modeling is clinically important (supports our H1).
- Cite Mamba results as motivation if we add Mamba as a 4th head: "SleepMaMi uses bidirectional Mamba for macro-structure encoding; we test Mamba as a downstream aggregation head under our context-length framework."
- Their DGCL finding (demographics improve sleep macrostructure learning) is consistent with our age_class and sex_binary tasks showing strong context-length effects.
- Can argue our context-sweep is a more systematic study of the macro-structure question: instead of using a fixed full-night context, we sweep 6 context lengths to quantify when diminishing returns set in.

---

### 1.3 SleepFounder: A Zero-Burden Sleep Foundation Model Built on Cardiorespiratory Signals from 800,000+ Hours of Multi-Ethnic Sleep Recordings
**Venue:** medRxiv September 2025 (preprint)  
**DOI:** 10.1101/2025.09.06.25335216  
**Pre-training data:** 800,000+ hours, 35 cohorts (11 Chinese + 3 BCG + 21 US PSG cohorts)

**What they did:**
- Largest pre-training dataset of the three by far (~5× SleepBench).
- **Key constraint:** Only cardiorespiratory signals (respiration + heartbeat). No EEG. Designed for zero-burden deployment on contactless devices (BCG = ballistocardiography).
- Architecture: dual ResNet branches + RoFormer encoder (6 layers, 8 heads, 512-dim hidden)
- Pre-training objective: cross-modal reconstruction — learn to predict EEG spectrograms and SpO2 from cardiorespiratory signals alone
- Downstream tasks: sleep staging (κ=0.65, 5-class), OSA detection (AUROC=0.917 moderate-severe), disease prediction (Parkinson's 0.943, heart failure 0.881, coronary CHD 0.81, atrial fibrillation 0.77), age MAE=6.90, sex AUROC=0.85
- External validation on contactless BCG recordings (maintained 94% of PSG-based performance)

**Key numbers:** κ=0.65 (staging), OSA AUROC=0.917, Parkinson's AUROC=0.943

**Relation to our work:**
- SleepFounder's claim: cardiorespiratory signals alone (no EEG) are sufficient for clinical prediction. **This directly motivates our channel ablation:** if we show that SleepFM's performance on clinical tasks holds after zeroing out BAS (EEG) embeddings, we support this claim from the downstream-evaluation side.
- Conversely, if EEG removal degrades performance, this shows SleepFounder's zero-burden approach has real costs.
- Their disease prediction results (heart failure 0.88, CHD 0.81) are directly comparable to our cvd_binary (AUROC ~0.67) and other clinical tasks. The gap likely reflects: (a) their much larger training set, (b) their cross-modal reconstruction objective, (c) different task definitions.
- Their sex classification (AUROC=0.85) and age prediction (MAE=6.9 years) are comparable to our sex_binary and age_class tasks. Good reference numbers.
- **Important caveat for paper:** Their numbers come from fine-tuning the full pre-trained model; ours come from training only a lightweight head on frozen SleepFM embeddings. Different evaluation protocol — do not compare directly without noting this.

**Paper writing notes:**
- Cite SleepFounder when discussing cardiorespiratory-only performance and the zero-burden deployment scenario.
- Use their disease prediction numbers as upper-bound reference (they fine-tune the full model on much more data).
- Cite their κ=0.65 as a reference for sleep staging performance from non-EEG signals.
- If our channel ablation shows RESP+EKG-only performance, compare to their numbers as a zero-burden baseline.

---

## 2. Summary Comparison Table

| | SleepFounder | SleepMaMi | OSF | **Our work** |
|---|---|---|---|---|
| Pre-training data | 800K hours, 35 cohorts | 158K hours, 20,964 PSG | 166.5K hours, 9 sources | — (frozen SleepFM) |
| Backbone training | From scratch | From scratch | From scratch | Frozen SleepFM |
| Input channels | Cardiorespiratory only (2 modalities) | Full PSG (EEG, EOG, EMG, etc.) | Full PSG (8+ channels) | Full PSG via SleepFM (BAS/RESP/EKG/EMG) |
| Head | Fine-tuned full model | Fine-tuned full model | Linear probe / fine-tune | Lightweight head only (LSTM/Transformer/MeanPool) |
| Context length | Fixed (full night) | Fixed micro + full-night macro | Fixed (epoch-level) | **Swept: 30s → 240m** |
| Sleep staging | κ=0.65 | 81.9% acc | AUC=97.3 | TBD (v3 sweep) |
| OSA detection | AUROC=0.917 | Incl. in SDB | Incl. in 9 tasks | TBD (apnea_binary) |
| Disease prediction | 10+ diseases, up to 0.943 | Not primary focus | Coronary 0.681 | 10 tasks (bmi, sex, age, cvd, depression, etc.) |
| Channel ablation | N/A (2 channels by design) | Not studied | ✓ (key contribution) | Planned (see below) |
| Context-length study | ✗ | Partial (micro vs. macro) | ✗ | ✓ (our main contribution) |
| Code/model public | Unclear | Not confirmed | ✓ | — |

---

## 3. What Our Paper Genuinely Lacks

Ordered by how likely a reviewer is to ask about it:

**3.1 Channel/modality contribution analysis** (OSF directly motivates this)
Reviewers familiar with OSF will ask: which modalities drive your clinical task performance? What happens if EEG is unavailable? OSF makes channel robustness a central result; we need to address this.

**3.2 Comparison to recent SOTA models as backbones**
Reviewers will ask: why SleepFM? OSF's weights are public. However, OSF was pre-trained on SHHS and MrOS — our test sets — making a direct comparison contaminated. This must be stated clearly in the paper.

**3.3 Mamba as a temporal head** (SleepMaMi motivates this)
SleepMaMi's macro-encoder uses Mamba. Reviewers familiar with SSMs will ask why we didn't test Mamba as a head given it's designed for long sequences. This is a reasonable implementation ask but not blocking.

**3.4 Data efficiency / label efficiency** (both OSF and SleepMaMi test few-shot)
Both papers show performance at low N. We have variable-N tasks already; a training-size ablation on one representative task is easy to add.

**3.5 Missing channel robustness at inference** (OSF's critical finding)
OSF shows existing models degrade sharply when channels are absent at test time. Do our models (which see 4 modality groups) degrade gracefully or catastrophically when one group is zeroed? This can be tested without re-preprocessing.

---

## 4. Planned Experiments and Ablations

### Priority 1 — High value, moderate effort

#### A. Modality group ablation (training-time zeroing — requires retraining heads)

---

##### A.0 The training vs inference question — why you must retrain

SleepFM produces a 512-dim embedding by concatenating four independent 128-dim encoders:

```
embedding[0:128]   = BAS  — EEG + EOG (brain activity)
embedding[128:256] = RESP — Airflow, Thor, ABD, SpO2, HR, Snore, RespRate
embedding[256:384] = EKG  — cardiac signal
embedding[384:512] = EMG  — chin and leg muscles
```

The downstream head (LSTM/Transformer/MeanPool) takes this full 512-dim vector as input. The ablation works by zeroing specific 128-dim slices **in the data loader** before the embedding reaches the head — no re-extraction of SleepFM features is needed.

**This ablation requires retraining the head**, but does NOT require re-running SleepFM feature extraction. Here is why:

| Approach | What it tests | Requires retraining? |
|----------|--------------|---------------------|
| **Zero at training + inference** (Experiment A) | Peak performance achievable with channel subset X — head adapts its weights to the available modalities | **Yes** (head never saw absent modality; learns to use only what's present) |
| **Zero at inference only** (Experiment G, supplementary) | Robustness of the baseline model to missing channels at deployment | **No** (re-run inference on existing baseline checkpoints with zeroed slices) |

**Experiment A is the primary ablation** for the paper. It answers: *"If we had deployed a system with only cardiorespiratory channels from the start, what is the best achievable performance?"* This is clinically meaningful and directly comparable to SleepFounder's claims.

**Experiment G is a supplementary robustness check** (see §G). It answers: *"If a channel fails at deployment, how much does our pre-trained model degrade?"* It can be run now without retraining and takes 1 day.

**Why inference-only is NOT sufficient for Table 6:** A model trained on all 4 modalities has learned weight patterns that expect non-zero values across all 512 input dimensions. Zeroing a 128-dim slice at inference forces the model to operate out-of-distribution. The resulting AUROC degradation may be severe not because the modality was important, but because the model was never trained to ignore those dimensions. This conflates "modality importance" with "model robustness", and the two are different scientific questions.

---

##### A.1 Ablation conditions

Eight conditions covering all scientifically meaningful combinations:

| Label | Active | Zeroed | Key comparison |
|-------|--------|--------|----------------|
| `full` | BAS+RESP+EKG+EMG | — | Baseline (already done) |
| `no_bas` | RESP+EKG+EMG | BAS | No EEG/EOG — zero-burden PSG |
| `no_resp` | BAS+EKG+EMG | RESP | No breathing sensors |
| `no_ekg` | BAS+RESP+EMG | EKG | Standard PSG without ECG |
| `cardio` | RESP+EKG | BAS+EMG | **SleepFounder direct comparison** — cardiorespiratory only |
| `bas_only` | BAS | RESP+EKG+EMG | EEG/EOG only — brain-signal baseline |
| `resp_only` | RESP | BAS+EKG+EMG | Respiratory wearable only |
| `ekg_only` | EKG | BAS+RESP+EMG | Single-lead cardiac monitor |

**Essential conditions for Table 6 (main paper):** `no_bas`, `cardio`, `bas_only` (3 new runs per task; `full` baseline already exists).

**Optional conditions (supplementary):** `no_resp`, `no_ekg`, `resp_only`, `ekg_only` — complete the single-modality and leave-one-out decomposition.

**Scientific hypothesis per condition:**
- `no_bas` vs `full`: Does removing EEG/EOG hurt? Expected: yes for sleep staging, maybe not for apnea (respiratory) or CVD (cardiac).
- `cardio` vs `full`: How much does SleepFounder's cardiorespiratory-only setting sacrifice? SleepFounder achieves OSA AUROC=0.917 with cardio only — can we match or approach this with our frozen-backbone setup?
- `bas_only` vs `full`: Can brain signals alone (EEG/EOG) predict clinical outcomes? Expected: yes for sleep quality, age, sex; no for apnea severity.
- `resp_only` vs `ekg_only`: Which single signal carries more clinical information? RESP expected to dominate for respiratory tasks; EKG for cardiovascular tasks.

---

##### A.2 Task selection

Based on actual Phase 0 v3 results, the following 5 tasks are selected for the channel ablation. Tasks with AUROC near chance (psqi_binary=0.55, sleepiness_binary=0.63) are excluded — the model has not learned anything useful from them, so a channel ablation would produce noise rather than signal.

| Task | LSTM AUROC @120m | Baseline @30s | Δ (context benefit) | N_test | Why include |
|------|-----------------|---------------|---------------------|--------|-------------|
| `sex_binary` | 0.872 | 0.824 | +0.049 | 1433 | Well-powered; all modalities likely contribute; sanity check (sex is a strong signal) |
| `apnea_binary` | 0.832 | 0.758 | +0.074 | 2054 | Primary clinical task; RESP expected to dominate; direct SleepFounder comparison |
| `sleep_efficiency_binary` | 0.780 | 0.697 | +0.083 | 2023 | Highest context benefit; BAS (EEG sleep staging features) expected to matter |
| `age_class` | 0.893 | 0.862 | +0.031 | 1862 | Large N; multi-class; interesting — which modalities encode physiological age? |
| `bmi_binary` | 0.767 | 0.760 | +0.006 | 1856 | Large N; saturates fast (L*=10m); good test of whether metabolic info is modality-specific |

**Optionally include if time allows:**
- `cvd_binary` (AUROC=0.688): EKG hypothesis — cardiac signal should help predict CVD history. AUROC is moderate but the channel comparison is scientifically interesting (only task where EKG should dominate over BAS).
- `depression_extreme_binary` (AUROC=0.770): Included if N≥200 is acceptable; extreme-group design inflates AUROC so interpret carefully.

**Excluded:**
- `sleepiness_binary` (0.628, Δ=0): Model doesn't learn; ablation would show nothing.
- `psqi_binary` (0.553, Δ=0): Near chance; no signal to ablate.
- `osa_binary_apples_postqc` (N=161): Too small for ablation; single cohort limits generalizability.

---

##### A.3 Context length choice

Run all ablation conditions at a **single fixed context per task** equal to the baseline's saturation context:

| Task | Ablation context | Rationale |
|------|-----------------|-----------|
| `sex_binary` | 120m | L*=80m; 120m achieves same AUROC — use 120m for uniformity |
| `apnea_binary` | 120m | L*=120m (peak) — exact match |
| `sleep_efficiency_binary` | 120m | L*=240m but 120m achieves 0.780 (97.7% of peak 0.799); avoids expensive 240m runs for all conditions |
| `age_class` | 120m | L*=40m; 120m is well past saturation — same AUROC, consistent table |
| `bmi_binary` | 40m | L*=10m; using 40m ensures enough windows per subject (120m gives very few windows for this task) |

**Why 120m for most tasks:** It is the peak or near-peak context for the 4 primary tasks, it yields a sufficient number of windows (≥4 per subject for most recordings), and it ensures a fair baseline performance to compare channel conditions against. Using a fixed context (rather than task-specific L*) makes the paper table cleaner and the comparison more direct.

**Note on `bmi_binary`:** This task saturates at 10m (Δ=+0.006 from 30s to best). Running it at 40m still captures near-peak performance while providing more training windows per epoch than 120m. Running at 120m would give very few windows per bmi_binary subject and may destabilize training for certain channel conditions.

---

##### A.4 Run count and effort

**Essential ablation (Table 6 primary):**
- 5 tasks × 3 new conditions (no_bas, cardio, bas_only) × 1 head (lstm) × 1 context = **15 training jobs**
- Inference + analysis for each = 15 more jobs
- Total: **30 new cluster jobs**, each roughly equivalent to one baseline training job

**Optional (supplementary):**
- 5 tasks × 3 more conditions (no_resp, no_ekg, resp_only or ekg_only) × 1 head × 1 context = **15 more training jobs**
- Adding cvd_binary: 6 tasks × 3–6 conditions = up to 18 more training jobs

**Effort estimate:**
- Training time per job: ~same as baseline (same context, same GPU hours)
- Scheduling + monitoring: 1–2 days
- Analysis + table generation: `scripts/make_table6_modality.py` (implemented — see §A.7)

---

##### A.5 Registry structure

Ablation experiments live in a **separate registry** `experiments/v2_ablation_registry.yaml` with a separate config (`configs/phase0_v3_abl_config.yaml`) that points to a dedicated results directory (`phase0_v3_abl/`) and log directory (`logs_v3_abl/`). This guarantees zero overlap with v3 or v3_full results. ✅ **Implemented.**

```yaml
# Excerpt from experiments/v2_ablation_registry.yaml
config: configs/phase0_v3_abl_config.yaml
results_dir: /scratch/boshra95/psg/unified/results/phase0_v3_abl
logs_dir: /home/boshra95/NSRR-tools/logs_v3_abl

sex_binary_lstm_abl_no_bas:
  task: sex_binary
  task_type: seq2label
  num_classes: 2
  head: lstm
  datasets: [apples, shhs]
  contexts: ["120m"]
  lr: 5.0e-5       # context_lr_override matches v3 for 120m
  run_tag: "abl_no_bas"          # results go to sex_binary_lstm_abl_no_bas/context_120m/
  zero_modalities: [BAS]         # gen_commands.py → ZERO_MODALITIES="BAS" env var
  tier: 1

sex_binary_lstm_abl_cardio:
  ...
  run_tag: "abl_cardio"
  zero_modalities: [BAS, EMG]
  tier: 1

sex_binary_lstm_abl_bas_only:
  ...
  run_tag: "abl_bas_only"
  zero_modalities: [RESP, EKG, EMG]
  tier: 1
# ... 15 entries total (5 tasks × 3 conditions)
```

The `zero_modalities` field lists which groups are zeroed (the complement is what the head sees). `run_tag` ensures outputs go to a separate subfolder from the baseline. All 15 entries are in the registry; use `REG="--registry experiments/v2_ablation_registry.yaml"` with all gen_commands.py calls.

---

##### A.6 What Table 6 will show

| Task | Context | full AUROC | no_bas AUROC | cardio AUROC | bas_only AUROC |
|------|---------|------------|--------------|--------------|----------------|
| Sleep apnea (AHI≥15) | 120m | 0.832 | ? | ? | ? |
| Sex | 120m | 0.872 | ? | ? | ? |
| Sleep efficiency | 120m | 0.780 | ? | ? | ? |
| Age group | 120m | 0.893 | ? | ? | ? |
| BMI (obese) | 40m | 0.767 | ? | ? | ? |

**Expected pattern:**
- apnea_binary: full > cardio ≈ no_bas >> bas_only (RESP carries the OSA signal; EEG alone can't detect apnea)
- sex_binary: full > bas_only ≈ no_bas > cardio (EEG/EOG and RESP both encode sex; EKG alone loses some signal)
- sleep_efficiency_binary: full > bas_only ≈ no_bas > cardio (EEG sleep staging features crucial; but RESP also important since sleep efficiency correlates with apnea)
- age_class: full ≈ no_bas (physiological aging present in all modalities; removing one modality loses less)
- bmi_binary: full ≈ all conditions (BMI encodes into all modalities; saturates fast → channel differences may be small)
Implement and compare the results with these expectations

---

##### A.7 Implementation status

1. ✅ **`src/nsrr_tools/datasets/context_window_dataset.py`** — `zero_modality_indices` parameter added.  
   Zeroing is applied to the `(N, 4, 128)` float32 copy before reshaping to `(N, 512)`, so the `.npy` memory-map is never written:
   ```python
   w = window.astype(np.float32)      # (N, 4, 128) — always a new copy (float16→float32)
   for mi in self._zero_modality_indices:   # modality index: 0=BAS, 1=RESP, 2=EKG, 3=EMG
       w[:, mi, :] = 0.0
   x = w.reshape(N, FLAT_DIM)         # (N, 512) fed to head
   ```
   Applied identically at training time (all three window methods) and inference time.

2. ✅ **`scripts/train_context_sweep.py`** — `--zero-modalities BAS RESP EKG EMG` flag added.  
   `scripts/infer_subject_windows.py` — same flag, same zeroing.  
   `jobs/train_context_sweep_gpu.sh` and `jobs/infer_subject_windows_gpu.sh` — `ZERO_MODALITIES` env var forwarded to both.

3. ✅ **`scripts/gen_commands.py`** — reads `zero_modalities` from registry entry, emits `ZERO_MODALITIES="BAS"` etc. in the generated sbatch command.

4. ✅ **`experiments/v2_ablation_registry.yaml`** — 15 entries (5 tasks × 3 conditions).  
   ✅ **`configs/phase0_v3_abl_config.yaml`** — separate config pointing to `phase0_v3_abl/` results dir.

5. ✅ **`scripts/make_table6_modality.py`** — reads `results/collected/phase0_v3/analysis.csv` (Full baseline) and `results/collected/phase0_v3_abl/analysis.csv` (the three ablation conditions, keyed by `run_tag`), joins them into the task × condition AUROC table with deltas, and saves CSV/markdown/LaTeX to `results/tables/table6_modality.*`. See `docs/EXPERIMENTS_GUIDE.md` §Modality ablation — Step 5 for usage.

6. ⚠️ **Bug found and fixed while wiring up Table 6:** `scripts/collect_results_v2.py`'s `parse_exp_dir()` originally matched experiment folders only by suffix (`_lstm`, `_transformer`, `_mean_pool`), so it silently dropped every `run_tag`-suffixed folder (`{task}_{head}_abl_no_bas`, etc. — and also the pre-existing `sleep_staging_lstm_with_stages`). Fixed to find the head as a substring and capture the trailing run_tag separately; `run_tag` was added to the train/analysis dedup keys so the three ablation conditions per task don't collide under the same key. Backward compatible — old CSVs without a `run_tag` column are read as `run_tag=""`.

---

#### B. More channels per modality group — ✅ THIS IS THE v3_full RUN (already in progress)

> **This experiment is not new work.** It is the full-channel run (`phase0_v3_full`) that was
> implemented and is currently running. Nothing here needs to be scheduled or coded.
> See `docs/EXPERIMENTS_GUIDE.md` → **"Full-channel run (channel expansion)"** for the
> step-by-step playbook, current status, and all commands.

**What v3_full does (= what this section originally described as future work):**

| Originally planned | v3_full implementation |
|---|---|
| Expand EKG from 1 channel to ECG-L, ECG-R (use 2-slot budget) | `preprocessing_params_full.yaml` — EKG≤2, priority: [EKG, ECG-L, ECG-R] |
| Expand EMG from 1 channel to CHIN, LLEG, RLEG (use 4-slot budget) | `preprocessing_params_full.yaml` — EMG≤4, priority: [CHIN, LLEG, RLEG, EMG] |
| Use full BAS priority list (up to 10 channels) | `preprocessing_params_full.yaml` — BAS≤10, all EEG + EOG leads |
| Re-run preprocessing + embedding extraction | Done in v3_full Step 0–1 (IN PROGRESS) |
| Re-run training on all Tier 1 tasks | Done in v3_full Step 2–3 |

**What it answers:** Fast-channel (v3) vs full-channel (v3_full) AUROC comparison shows the
marginal gain from using more PSG channels per modality group. This is Table 5 (or equivalent)
in the paper — the primary fast→full comparison.

**Status:** IN PROGRESS — preprocessing and embedding extraction running on cluster.
**No new code or jobs needed here.**

---

### Priority 2 — Moderate value, low effort

#### C. Data efficiency curve
**What:** Train on {10%, 25%, 50%, 75%, 100%} of training subjects. Run for 2 tasks (sex_binary_lstm, bmi_binary_lstm) at 120m context (best single context in general). Plot AUROC vs. training N.

**Implementation:** Add `--train-fraction 0.5` argument to training script that randomly samples the training subject list before building the DataLoader. Seed-fixed for reproducibility.

**Paper contribution:** Shows SleepFM representations are data-efficient (few labeled subjects needed for strong performance). Addresses SleepMaMi and OSF's few-shot claims. Also useful for clinicians: "you only need N subjects to reach 95% of peak performance."

**Effort:** 1 day (add argument, run ~10 jobs per task).

---

#### D. Mamba head
**What:** Add a 4th head using selective state space model (Mamba) as the temporal aggregation architecture. Competes directly with LSTM and Transformer at long contexts.

**Why relevant:** SleepMaMi's macro-encoder is bidirectional Mamba over full-night sequences. Our long-context experiments (80m–240m) test the same regime. Adding Mamba lets us directly test SleepMaMi's claim.

**Expected result:**
- At short contexts (30s, 10m): LSTM ≈ Mamba (short sequences, advantage of SSM not apparent)
- At long contexts (120m, 240m): Mamba may outperform LSTM (selective state space handles long-range dependencies better)
- Transformer with Flash attention (our current setup) may already match Mamba since Flash attention's O(N) memory is similar to Mamba's complexity

**Implementation:** ~100–150 lines — implement `MambaHead` class in the head module using `mamba-ssm` library. Requires CUDA and mamba_ssm installation. Run on 2 tasks first before committing to full sweep.

**Effort:** 3–5 days (implementation + testing + 2 representative tasks). Full sweep adds 2 weeks of job scheduling.

**Decision point:** If LSTM and Transformer already converge at 240m, Mamba won't add a story. If they diverge, Mamba is worth the full sweep.

---

#### E. Cross-dataset generalization breakdown
**What:** Report per-dataset AUROC breakdown in addition to pooled metrics. For multi-dataset tasks: train on all 4 cohorts, report test AUROC broken down by cohort (SHHS, MrOS, APPLES, STAGES).

**Implementation:** Zero code change — add dataset column to analysis output and group-by in the plotting scripts.

**Paper contribution:** Directly comparable to OSF's in-domain (SHHS) vs. out-of-domain (MrOS) evaluation. Shows whether context-length benefits are cohort-specific or universal.

**Effort:** 0.5 day (modify analysis scripts).

---

### Priority 3 — Lower priority, do if time allows

#### F. Bidirectional LSTM
**What:** Add BiLSTM head (processes sequence in both forward and backward directions). Trivially implementable: add `bidirectional=True` to the LSTM constructor and adjust the output projection layer.

**Why relevant:** Examines whether knowing future context (later in the night) helps predict the anchor epoch — relevant for staging. For seq2label (clinical prediction), the model already sees the full context window, so BiLSTM vs. LSTM difference is about whether the order-sensitive processing direction matters.

**Effort:** 0.5 day. Run on sleep_staging_lstm only (most informative for directionality).

---

#### G. Missing channel robustness at inference (supplementary — no retraining needed)

**Relationship to §A:** Experiment §A answers *"what is the peak capability of each channel subset?"* by retraining the head. Experiment G answers *"how does the baseline model degrade when channels go missing at deployment?"* without any retraining. Both questions matter, but for different audiences. Experiment G can be run immediately on the existing baseline checkpoints.

**What:** Use the already-trained full-channel baseline models. At inference only, zero out one modality group at a time. Measure AUROC degradation vs the original baseline. Tests robustness: does the model degrade gracefully (partial capability retained) or catastrophically (AUROC near chance)?

**Implementation:** Modify the inference script to accept `--zero-modalities` flag applied only at inference time (same zeroing logic as §A but applied to a model that was not trained with the zeroing). No retraining or new training jobs required. Can be implemented and run in 1 day.

**Tasks:** Run on sex_binary, apnea_binary, sleep_efficiency_binary (the three tasks where we expect different degradation patterns based on modality-task alignment).

**Paper contribution:** Placed in supplementary material. Directly addresses OSF's channel masking finding. If the model degrades gracefully → SleepFM's pre-training produces modality-robust representations. If catastrophically → motivates channel masking during SleepFM pretraining as future work, and confirms that §A's training-time ablation was necessary to get fair capability estimates.

**Effort:** 1 day. Three tasks × three conditions (no_bas, cardio, bas_only) × inference-only runs.

---

#### H. OSF as backbone (future work / external collaboration)
**What:** Replace SleepFM embedding extractor with OSF's public encoder. Run the same context-length sweep with OSF embeddings.

**Why interesting:** Direct backbone comparison under identical downstream experimental conditions.

**Why deferred:**
1. OSF was pre-trained on SHHS and MrOS — our test cohorts — making AUROC comparison contaminated. Must be disclosed.
2. OSF uses different channel configurations; mapping our 4 NSRR cohorts to OSF's input format requires careful preprocessing.
3. Running OSF encoder over 16k subjects × full-night recordings = 3–5 days of cluster compute.
4. Our paper's contribution is the context-length framework, not the backbone. Swapping backbones is a separate paper or ablation in a journal extension.

**Effort:** 2–3 weeks. **Do not pursue before main runs are complete. Mention as future work.**

---

## 5. Recommended Run Order (after main v3 sweep)

| Order | Experiment | Effort | Blocking? | Requires retraining? |
|---|---|---|---|---|
| 1 | **Modality ablation — training-time (§A, essential 3 conditions)** | 2–3 days | No | **Yes** — 15 new training jobs |
| 2 | Inference-time robustness check (Experiment G, supplementary) | 1 day | No | No — run on existing checkpoints |
| 3 | Cross-dataset generalization breakdown (Experiment E) | 0.5 day | No | No |
| 4 | Data efficiency curve (Experiment C) | 1 day | No | No |
| 5 | Modality ablation — optional 3 conditions (§A, supplementary) | 2–3 days | No | Yes — 15 more training jobs |
| 6 | Mamba head on 2 tasks (Experiment D — pilot) | 3–5 days | No | Yes |
| 7 | Full Mamba sweep (Experiment D — full) | 2 weeks | Depends on pilot | Yes |
| 8 | Bidirectional LSTM (Experiment F) | 0.5 day | No | Yes |
| 9 | More channels per modality group (Experiment B) | 1–2 weeks | Wait until main sweep done | Yes |
| 10 | OSF backbone comparison (Experiment H) | 2–3 weeks | Future work / journal ext. | N/A |

---

## 6. Paper Writing Reference Notes

### Related Work framing
- **Sleep foundation models paragraph:** Start from SleepFM (our backbone) → OSF (scaling/channel invariance) → SleepMaMi (macro-micro hierarchy) → SleepFounder (zero-burden). Position our work as the first systematic study of *context length* as an independent variable in clinical downstream prediction, orthogonal to the pre-training advances these models represent.
- **Context length paragraph:** SleepMaMi's macro-micro distinction implicitly acknowledges that full-night temporal structure matters. Our work quantifies this explicitly: we sweep 6 context lengths (30s to 240m) and measure where performance saturates for each clinical task, providing H1–H4 evidence that these models lack.
- **Channel ablation paragraph:** Cite OSF's channel masking finding and SleepFounder's zero-burden claim as motivation. OSF shows the problem (catastrophic failure on channel removal); SleepFounder shows an extreme solution (train without EEG). Our ablation (Experiment A) tests the intermediate question: which modalities matter for which downstream tasks under a fixed foundation model?

### Metric notes for comparison
- When comparing sleep staging κ: SleepFounder (κ=0.65), OSF does not report κ (uses AUC), SleepMaMi (81.9% acc ≠ κ). Use κ as primary metric (standard in field) and note incompatibility.
- When comparing clinical prediction AUROC: All three papers use different task definitions, cohorts, and evaluation protocols (fine-tuning vs. linear probe vs. head-only). Do not compare our AUROC numbers directly to theirs without explicitly noting this. Write: "Under our evaluation protocol (frozen backbone, lightweight head), we achieve [X]; direct comparison with [paper] is confounded by [difference]."
- SleepFounder's disease prediction AUROCs (0.88–0.94) are from a fine-tuned 800K-hour model. Our numbers are from a frozen SleepFM head — a significantly more constrained setting. This is a strength (data efficiency) not a weakness.

### Our differentiators vs. the three papers
1. **Context-length framework:** None of the three papers sweep context length as a variable. This is our primary contribution. SleepMaMi implicitly acknowledges this is important (Macro vs. Micro) but doesn't quantify it.
2. **Task breadth:** We evaluate 15 clinical tasks across 4 cohorts, including psychiatric (depression, anxiety), sleep quality (PSQI, insomnia, rested morning), cardiovascular (CVD), and metabolic (BMI). None of the three papers comes close to this breadth.
3. **Evaluation rigour:** Stratified splits, consistent K-window strategy, iso-compute analysis. Our analysis framework is more systematic than any of the three papers.
4. **Frozen backbone:** Training only the head is 10–100× cheaper and more deployment-realistic. OSF requires 10 hours of H100 training just for pre-training.

### Potential weaknesses to preempt
- "Why SleepFM and not a more recent backbone?" → Answer: backbone-agnostic framework; swapping is one-line change; OSF comparison is contaminated by training data overlap; left for future work.
- "Why limited channels in your current experiments?" → Answer: we use the intersection of channels available across all 4 cohorts, prioritised by availability. Channel ablation (Experiment A/B) addresses this directly.
- "Did you test Mamba?" → Answer: SleepMaMi's macro-encoder uses Mamba; we run it as a 4th head in our ablation (cite Experiment D if completed, else "left for future work").
- "How does your work compare to OSF's scaling law?" → Answer: we study a different axis — context length in downstream evaluation rather than model scale in pre-training. These are complementary findings.
