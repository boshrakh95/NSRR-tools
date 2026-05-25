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

#### A. Modality group ablation at inference (no preprocessing rerun needed)
**What:** Zero-out specific 128-dim slices of the existing 512-dim SleepFM embeddings at training and inference time. The 512-dim SleepFM embedding = `[BAS=0:128 | RESP=128:256 | EKG=256:384 | EMG=384:512]`.

**Conditions:**
| Condition | Active modalities | Zeros | Comparison to |
|---|---|---|---|
| All 4 (baseline) | BAS+RESP+EKG+EMG | none | — |
| Cardiorespiratory only | RESP+EKG | BAS, EMG | SleepFounder's zero-burden claim |
| EEG only | BAS | RESP, EKG, EMG | Brain-signal-only baseline |
| No EEG | RESP+EKG+EMG | BAS | Zero-burden from full PSG |
| Single: RESP | RESP | BAS, EKG, EMG | Minimal viable |
| Single: EKG | EKG | BAS, RESP, EMG | Cardiac only |

**Implementation:** Add `--zero-modalities BAS RESP` flag to the dataset loader (zero the relevant 128-dim slice before the head sees the embedding). No re-extraction of SleepFM features needed.

**Tasks to run:** bmi_binary_lstm, apnea_binary_lstm, sleep_staging_lstm at 2 context lengths (10m, 120m). ~12 additional training jobs.

**Paper contribution:** Directly addresses OSF's channel masking finding. Shows which modalities carry the clinical signal for each task. If cardiorespiratory-only matches full performance → supports SleepFounder's zero-burden claim. If EEG is critical for staging but not for apnea → task-specific modality importance.

**Effort:** 2–3 days (add flag to dataset, run jobs, add figure).

---

#### B. More channels per modality group (preprocessing rerun required)
**What:** The current config uses a limited number of channels per SleepFM modality group:
- BAS (EEG + EOG): priority list = C3-M2, C4-M1, LOC, ROC, O1-M2, O2-M1, F3-M2, F4-M1, A1, A2 (up to 10)
- RESP: Airflow, Thor, ABD, SpO2, HR, Snore, RespRate (up to 7)
- EKG: priority = [EKG] only (max 2 slots)
- EMG: priority = [EMG] only (max 4 slots)

Currently EKG uses only 1 channel (EKG) and EMG uses only 1 channel (EMG chin lead), with the remaining slots empty. Adding ECG-L, ECG-R for EKG and CHIN, LLEG, RLEG for EMG would use the max_channels budget.

**What it answers:** Does adding more channels within each group improve performance? What is the marginal gain from more EEG leads?

**Implementation:**
1. Update `channel_priority` in `configs/phase0_v3_config.yaml` (add channels)
2. Re-run `scripts/extract_nsrr_channels.py` for all 4 cohorts (cluster jobs)
3. Re-run SleepFM embedding extraction for all subjects (expensive — multi-day cluster job)
4. Then re-run training and inference as normal

**Tasks to run:** All Tier 1 tasks at representative context lengths.

**Effort:** 1–2 weeks (dominated by embedding re-extraction). **Plan for after all main runs are complete.**

**Paper contribution:** Shows sensitivity to channel configuration. Justifies our current channel choices (or motivates future work with richer channels).

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

#### G. Missing channel robustness at inference (no retraining)
**What:** Train model normally on all 4 modality groups. At inference only, zero out one modality group. Measure AUROC degradation. Tests robustness: does our model degrade gracefully or catastrophically when a modality is missing at test time?

**Implementation:** Modify inference script to accept `--zero-modalities-infer` flag. No retraining.

**Paper contribution:** Directly addresses OSF's finding. If our model degrades gracefully → SleepFM's representations are robust. If catastrophically → motivates channel masking during SleepFM pretraining as future work.

**Effort:** 1 day. Run on 2 tasks (bmi_binary, sleep_staging).

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

| Order | Experiment | Effort | Blocking? |
|---|---|---|---|
| 1 | Modality group ablation (zero-out at inference, Experiment A) | 2–3 days | No |
| 2 | Cross-dataset generalization breakdown (Experiment E) | 0.5 day | No |
| 3 | Data efficiency curve (Experiment C) | 1 day | No |
| 4 | Missing channel robustness at inference (Experiment G) | 1 day | No |
| 5 | Mamba head on 2 tasks (Experiment D — pilot) | 3–5 days | No |
| 6 | Full Mamba sweep (Experiment D — full) | 2 weeks | Depends on pilot |
| 7 | Bidirectional LSTM (Experiment F) | 0.5 day | No |
| 8 | More channels per modality group (Experiment B) | 1–2 weeks | Wait until main sweep done |
| 9 | OSF backbone comparison (Experiment H) | 2–3 weeks | Future work / journal ext. |

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
