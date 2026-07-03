# New Plot Ideas — PSG Context-Length Study

**Data available:**
- `analysis.csv` — aggregate metrics per (task, head, context, K): AUROC, balanced_acc, AUC-PR, CIs
- `training.csv` — per-epoch curves: loss, val_auroc, bal_acc, FLOPs
- `test_windows.parquet` — per-window predictions per (task, head, context): subject_id, dataset, window_idx, true_label, pred_label, prob_class0...N
- `table6_modality.csv` — ablation ΔAUROC per (task, condition)
- `analysis.csv` (v3_full) — full-channel aggregate metrics

Rounds: **v3** (fast-channel, main), **v3_full** (full-channel), **v3_abl** (modality ablation).

---

## Group 1 — Context Efficiency & ROI

---

### 1. "Bang-per-Minute" Curve
**What:** Marginal AUROC gain per additional recording minute, plotted as a function
of total context consumed (L×K). Not per window (which we have), but in **time units**,
making the clinical trade-off immediately legible.

**Data:** `analysis.csv`, column `total_compute_min` (= ctx_min × K).

**Why new:** Current marginal-gain plot shows gain per additional *window K*, which is
an abstract unit. Clinicians think in minutes, not window counts.

**Library:** matplotlib with twin y-axis or seaborn `lineplot`.

```
AUROC gain
per extra min
 │
 │ ████░░░░░░░░░░░░░░░░░░
 │   ████░░░░░░░░░░░░░░░
0.5│     ████░░░░░░░░░░
 │         ████░░░░░░
 │             ████░
 │─────────────────────── Recording minutes (log scale)
    5   20  80  320  1200

One line per task, colored by task. Shaded region = "clinical budget zone" (≤4 hrs).
```

---

### 2. Clinical-Threshold Unlock Map
**What:** For each (task, head), at what context length does performance **first cross**
a series of clinical usefulness thresholds (0.75, 0.80, 0.85, 0.90)? Shown as a
threshold × task grid where each cell is colour-coded by the L that "unlocks" it.
Gray cells = threshold never reached at any context.

**Data:** `analysis.csv`, K=all rows.

**Why new:** Reframes saturation as "how much recording do I need to reach clinically
useful performance?" rather than "what's the AUROC ceiling?". Much more actionable.

**Library:** `seaborn.heatmap` or `plotly.imshow`.

```
Target AUROC │ Sleep Eff │ Apnea │  Sex  │  Age  │  BMI
─────────────┼───────────┼───────┼───────┼───────┼──────
    ≥ 0.75   │   10min   │  30s  │  30s  │  30s  │  30s
    ≥ 0.80   │  240min   │  40m  │  80m  │  80m  │  ----
    ≥ 0.85   │  >240m?   │ 120m  │ 240m  │ 120m  │  ----
    ≥ 0.90   │    ---    │  ---  │  ---  │  ---  │  ----

Color: green=short L, yellow=long L, red=unreachable
```

---

### 3. Iso-Compute Efficiency Surface (3D)
**What:** A 3D surface plot: x = context length L (log), y = K, z = AUROC. Iso-compute
contours become visible as diagonal ridges. Far more intuitive than a 2D heatmap for
showing the "ridge" along long-L / small-K.

**Data:** dense heatmap DataFrames (`heatmap_df_test.csv` from `build_heatmap_df.py`).

**Why new:** The 2D heatmap is already in the paper. A 3D view of the same data lets
you see the shape of the efficiency landscape — is it a gentle slope or a sharp ridge?

**Library:** `plotly.graph_objects.Surface` (interactive) or
`matplotlib` `plot_surface` (static PDF). Show sex_binary Transformer as the example.

```
        AUROC
         0.91 ┐
              │        /‾‾‾‾‾‾‾‾
         0.85 ┤       /
              │      /
         0.79 ┤     /  ← iso-compute ridge
              │ ___/
         0.72 ┘
              30s  10m  40m  80m  120m  240m      (L)
         K: 1 ...... 5 ...... 20 ...... all   (into page)
```

---

### 4. Deployment Scenario Heatmap (Practical Decision Guide)
**What:** Two-axis grid: rows = **required AUROC target**, columns = **available
recording budget** (total minutes: 30, 60, 120, 240, 480). Each cell shows the
**optimal (L, K) strategy** for that scenario, color-coded by the AUROC achieved.
Tasks shown as separate panels or selectable.

**Data:** `analysis.csv` (dense K), K×L grid for each task.

**Why new:** Answers the exact deployment question: "I have 2 hours of PSG and need
80% AUROC — what should I do?" No existing figure answers this directly.

**Library:** `plotly` (interactive with task selector) or `matplotlib` annotated grid.

```
Budget (total min) →
            30   60   120   240   480
Required  ┌────┬────┬─────┬─────┬─────┐
AUROC     │    │    │     │     │     │
  0.70    │30s,│10m,│ 10m,│ 40m,│ 80m,│
          │K=6 │K=6 │ K=12│ K=6 │ K=6 │
  0.80    │MISS│MISS│ 40m,│120m,│120m,│
          │    │    │ K=3 │ K=2 │ K=4 │
  0.85    │MISS│MISS│ MISS│240m,│240m,│
          │    │    │     │ K=1 │ K=2 │
  0.90    │MISS│MISS│ MISS│ MISS│ MISS│
          └────┴────┴─────┴─────┴─────┘
Color: green (high AUROC), yellow, red, gray (miss)
```

---

## Group 2 — Head Architecture & Temporal Advantage

---

### 5. Head Advantage Growth Curve
**What:** x = context length (log), y = AUROC gap between heads. Two curves:
**Transformer − LSTM** and **LSTM − MeanPool**, one panel per task (or all tasks
as faint lines + bold mean). Shows whether the temporal advantage grows monotonically
with context, and at what context it becomes meaningful.

**Data:** `analysis.csv`, K=all, filtered by task×head.

**Why new:** We report the gap at L* only (Table IV). This shows the **trajectory**
of how that gap opens up — the key claim of H4 is that it grows with context.

**Library:** seaborn `lineplot` with `hue="gap_type"`, faceted by task.

```
AUROC gap
(pp)
  5 ┤                                          ●─● Transformer−LSTM
    │                                   ●─●  ●
  3 ┤                             ●─●  ●
    │                       ●─●  ●
  1 ┤        ●─●─●─●─●─●  ●            ○─○─○ LSTM−MeanPool
    │   ●─●─○
  0 ┤ ○─○────────────────────────────────────
 -1 ┤
    ├──┬────┬───┬────┬────┬────┬
      30s  10m  40m  80m  120m  240m
```

---

### 6. Temporal vs. Frequency: Head × Channel Contribution Matrix
**What:** 2D grid: rows = 4 modality groups (BAS, RESP, EKG, EMG), columns = 3 heads
(MeanPool, LSTM, Transformer). Each cell = ΔAUROC from the modality ablation **but
hypothetically** shows how each *head type* benefits differently from each modality.
Current ablation only uses LSTM — so this is half-synthetic unless we run ablation
for all 3 heads (which could be a future experiment idea).

**Alternative that works now:** For the LSTM ablation data we have, show modality
importance as a radar chart with 4 modality axes, one polygon per task.

**Data (immediate version):** `table6_modality.csv`, columns No BAS Δ, No RESP Δ,
No EKG Δ (proxy for modality importance; invert sign).

**Library:** `matplotlib` polar chart or `plotly` radar.

```
        BAS importance
          ●
         /|\
        / | \
EMG────/──┼──\────RESP
       \  |  /
        \ | /
          ●
        EKG importance

One polygon per task, 5 tasks overlaid.
Sleep efficiency: large BAS vertex, small RESP/EKG.
Apnea: large RESP vertex.
Sex: balanced BAS+EKG, small RESP.
```

---

### 7. Head Convergence Context Map
**What:** For each task, identify the context length where the Transformer and LSTM
first come within ε=0.005 AUROC of each other (convergence from below) — or diverge.
Show as a divergence-vs-convergence lollipop aligned with the existing L* lollipop.

**Data:** `analysis.csv`, K=all.

**Why new:** We know Transformer > LSTM always, but does the gap ever *close*?
For BMI (weak task) it nearly does. This answers: "when is the more expensive
Transformer architecture actually worth using?"

**Library:** matplotlib horizontal lollipop (2-row: L* and "Transformer worth it at L≥X").

---

## Group 3 — Subject-Level Analysis (Parquet-Based)

---

### 8. Night Fingerprint Heatmap (Case Studies)
**What:** Select 4–6 representative subjects (one always-correct, one always-wrong,
two context-sensitive). For each, show a 2D heatmap:
- x-axis: window position in the night (0 → end of recording)
- y-axis: context length (30s → 240m, 6 rows)
- color: `prob_class1` (prediction probability)

Shows how the model's certainty changes as a function of BOTH where in the night the
window falls AND how long a context it sees. The "fingerprint" of how the model sees
each subject.

**Data:** parquets, filtered to specific subject IDs.

**Library:** `seaborn.heatmap` (one figure per subject, 4 subfigures).

```
               Window position in night →
         0%    25%    50%    75%   100%
  30s   │░░░░░░░▓▓▓▓░░░░░░░░░░░░░░░░░░│ ← noisy, low confidence
  10m   │░░░░▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░│
  40m   │░░░▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░│
  80m   │░░▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░│
 120m   │░▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░│
 240m   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░│ ← high confidence

Color: white=p≈0 (negative prediction), dark=p≈1 (positive)
True label = 1 in this example (positive subject).
```

---

### 9. Cross-Task Hard-Subject Overlap (UpSet Plot)
**What:** For each task, find the set of subjects that are **never correctly classified**
(0/6 contexts correct). Show the intersection of these "hard subject" sets across tasks
using an UpSet plot (the multi-set generalisation of a Venn diagram).

**Data:** parquets for all tasks joined on `subject_id`. Subjects must appear in ≥2 tasks.

**Why new:** Hard-subject analysis exists per task (S-Fig 11). But are they the *same*
subjects who are hard across multiple tasks? If the hard-subject sets have large overlap,
it suggests a shared latent factor (e.g., recording quality, unusual physiology).

**Library:** `upsetplot` (Python package) — gives the characteristic sorted bar chart.

```
Task hard-subjects │  Sex  │ Apnea │ Sleep │  Age  │  BMI
───────────────────┼───────┼───────┼───────┼───────┼──────
Sex∩Apnea          │   ●   │   ●   │       │       │
Sex∩Age            │   ●   │       │       │   ●   │
All five           │   ●   │   ●   │   ●   │   ●   │  ●
...

Bar heights above = number of subjects in each intersection.
```

---

### 10. Per-Subject Confidence Trajectory
**What:** x = context length (log), y = `prob_class1` at K=1 (single central window).
One faint line per test subject, colored by true_label (pos/neg). Bold lines = medians.
Shows the distribution of how individual subjects' predictions evolve with context.

**Data:** parquets (take window_idx = floor(n_windows/2) = the central window).

**Why new:** All current plots are population-level (AUROC). This shows the *distribution*
of individual trajectories — some subjects spike early and stay confident, others oscillate.

**Library:** seaborn `lineplot` with `units=subject_id` and `estimator=None`, or
matplotlib spaghetti with seaborn smooth overlay.

```
prob_class1
  1.0 ┤         ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ← true positives
      │    ___╱
  0.7 ┤   ╱  ═══════════════════════ ← median positive
      │  ╱  (many faint lines)
  0.5 ┤─────────────────────────────
      │  \   ═══════════════════════ ← median negative
  0.3 ┤   \___
      │        ╲____________________ ← true negatives
  0.0 ┤
      ├──┬────┬───┬────┬────┬────┬
        30s  10m  40m  80m  120m  240m
```

---

### 11. Pairwise Task Prediction Correlation
**What:** For every pair of tasks that share subjects (e.g., sex + apnea share SHHS/APPLES
subjects), compute per-subject prediction confidence (prob_class1 at L=120m, K=all) and
scatter one task's confidence against the other's. Color by which quadrant they fall into
(both high, both low, discordant).

**Data:** parquets for each task pair, subject-level aggregated, joined on subject_id.

**Why new:** Does being a "clear positive for apnea" correlate with being confidently
classified for sex? These are physiologically related (OSA is male-dominated). The
correlation pattern reveals latent structure captured by the SleepFM embeddings.

**Library:** `seaborn.pairplot` across all task pairs, or a single scatter for a
specific clinically motivated pair (apnea × sex, sleep_eff × apnea).

```
prob(apnea=1) at 120m
  1.0 ┤                   ● ●  ●●
      │              ●● ●● ●●●●  ← both positive
  0.7 ┤         ●●●●●●●●●●
      │    ● ●●●●●●●●●
  0.4 ┤●●●●●●●●●● ●●
      │  ●●●●   discordant
  0.1 ┤
      ├──────────────────────
       0.1   0.4   0.7   1.0
              prob(sex=F) at 120m
Color: blue=true pos/pos, red=true neg/neg, gray=discordant
```

---

### 12. Subject Prediction Stability Grid
**What:** For each subject, compute prediction **entropy** across the 6 context lengths
(how variable is the prediction across contexts?). Sort subjects by true label, then by
entropy. Show as a sorted heatmap: rows = subjects (sorted), columns = context lengths,
color = prob_class1.

**Data:** parquets, subject-level aggregated (K=all), across contexts.

**Why new:** Shows the full distribution of subject behaviors: always-certain-correct
(dark everywhere), entropy-high-correct (varies but right), always-wrong (wrong color),
etc. — much richer than the bar chart version (S-Fig 11).

**Library:** `seaborn.heatmap` with sorted rows.

```
Subjects        30s   10m   40m   80m  120m  240m
(sorted by  ┌──────────────────────────────────────┐
true label) │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ true neg
& entropy   │░░░░░░░░░░░░░▒▒▒▒░░░░░░░░░░░░░░░░░░░░│
            │░░░░░░░░░▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░│
            ├──────────────────────────────────────┤
            │▓▓▒▒░░▒▒░░▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ context-sensitive
            │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ true pos
            └──────────────────────────────────────┘
```

---

## Group 4 — Multi-Task Comparison & Clustering

---

### 13. Task Profile Radar Chart
**What:** One radar chart per head (3 panels). Each task is a polygon with 6 axes:
AUROC@30s, AUROC@240m, ΔAUROC, L* (normalised), AUC-PR@240m, Temporal_advantage.
Lets you see the "personality" of each task holistically.

**Data:** `analysis.csv` (K=all) + `table5_heads_fast.csv`.

**Library:** `matplotlib` polar axes or `plotly` scatterpolar with `fill='toself'`.

```
              AUROC@30s
              ●
             /|\
           /  |  \
  L*     ●───┼───●   ΔAUROC
           \  |  /
             \|/
              ●
           AUROC@240m

Each task = one colored polygon. 5 tasks × 3 panels (one per head).
```

---

### 14. Task Similarity Dendrogram / Clustermap
**What:** Cluster tasks based on their **saturation curve shape** (AUROC at each of 6
context lengths as a 6-D feature vector). Hierarchical clustering + dendrogram reveals
which tasks are "physiologically similar" in their context requirements.

**Data:** `analysis.csv`, K=all, LSTM, pivot: rows = tasks, columns = context lengths.

**Library:** `seaborn.clustermap` (combines heatmap + dendrogram automatically).

```
Tasks clustered by saturation curve shape:

                           Sex │░░▒▒▒▓▓
Context-sensitive cluster:  Apnea│░░░▒▒▓▓
                         Sleep eff│░░░░▒▒▓
                            ─────────────
Flat cluster:              BMI │▒▒▒▒▒▓▓
                       Depression│▒▓▒▒▓▒▒

Color = AUROC (white=low, dark=high). Left dendrogram groups similar tasks.
```

---

### 15. Parallel Coordinates — Task Properties
**What:** Each task is a line across multiple axes:
1. AUROC@30s  2. L* (min)  3. ΔAUROC  4. Full-ch gain  5. Top modality (ordinal)
6. N_test (log)  7. Temporal advantage (LSTM − MeanPool)

Coloured by a categorical grouping (physiological/demographic/metabolic/mental).

**Data:** Various collected tables + `table6_modality.csv`.

**Library:** `plotly.graph_objects.Parcoords` (interactive, very effective for this).

```
AUROC@30s  L*(min)  ΔAUROC  Full-ch  Temp.adv.
  0.87 ──────────────────────────────────── Sex (blue)
       ╲      ╱ ╲
  0.76 ─╲────╱───╲──────────────────────── BMI (red)
          ╲  ╱     ╲
  0.70 ────╲╱───────╲─────────────────── Sleep eff (green)
            240m     0.091
```

---

### 16. "Context Payoff" Scatter — All Experiments in One View
**What:** Scatter where every point is one (task, head, context) experiment:
- x = total compute (L×K minutes at K=5)
- y = AUROC
- color = task
- shape = head
- size = N_test

Gives a bird's-eye view of the entire experimental space: where are the efficient
operating points? Which task×head combinations give the most AUROC per minute?

**Data:** `analysis.csv`, K=5 rows.

**Library:** `plotly.express.scatter` (interactive hover with experiment ID).

```
AUROC
 0.91 ┤              ▲ (apnea, Transformer, 240m)
      │         ●  ● ▲
 0.85 ┤    ●  ● ▲  ▲▲
      │ ○  ●  ▲
 0.79 ┤ ○○ ▲
      │○
 0.73 ┤
      ├──┬────┬────┬────┬──────┤
        5   20   80  320  1280
             Total compute (min)
● LSTM   ▲ Transformer   ○ MeanPool
```

---

## Group 5 — Channel & Modality Analysis

---

### 17. Modality Contribution Chord Diagram
**What:** A circular chord diagram where:
- Outer segments = modality groups (BAS, RESP, EKG, EMG)
- Inner chords = connect modality pairs whose *joint* presence explains performance
  (from the interaction pattern in the ablation: cardio_only − no_ekg ≠ RESP alone)

Reads as: thick chord BAS↔EKG for sex = these two jointly carry sex information.
Thinner chords for pairs that are more independent.

**Data:** `table6_modality.csv` — interaction terms derived algebraically.

**Library:** `mpl_chord_diagram` or `plotly` chord diagram (requires some setup).

```
          BAS
         ╱╲ ╲
        ╱  ╲──╲── (thick for sex/age)
       ╱    ╲   ╲
    EMG──────────EKG
       ╲    ╱   ╱
        ╲  ╱──╱── (thick for apnea)
         ╲╱ ╱
          RESP
```

---

### 18. Channel Count vs. AUROC Scatter (Fast vs Full, All Tasks)
**What:** Simple but missing from the paper: x = number of PSG channels used (fast-ch ≈
7.5 mean, full-ch varies by dataset up to 23), y = peak AUROC at best L. One point per
(task, head, round). Separates fast and full with different marker shapes.

Add a trend line per head. Shows whether "more channels → better" is a linear relationship,
logarithmic, or task-dependent.

**Data:** `analysis.csv` from v3 and v3_full (K=all, best L per task/head).

**Library:** `seaborn.scatterplot` with `hue=task`, `style=head`.

```
Peak AUROC
  0.91 ┤              ▲ (apnea, full-ch, Transformer)
       │         ● ▲
  0.87 ┤    ● ▲ ●
       │ ●
  0.83 ┤
       │              ▲ (bmi, full-ch)
  0.78 ┤         ●
       ├──────────────────
         7.5           23
         Fast-ch    Full-ch
       (channels per recording)
```

---

### 19. Modality Ablation × Task Heatmap (The Table as a Visual)
**What:** Convert `table6_modality.csv` into a diverging heatmap where red = harmful
removal (large |Δ|), white = neutral, blue = harmful but small. With hierarchical
clustering on both rows (tasks) and columns (conditions) to surface natural groupings.

This is Table V in the paper, but as a clustermap it reveals structure the table hides:
"sleep efficiency" clusters with "age" (both BAS-dominant); "apnea" clusters with its
unique RESP-dominant signature.

**Data:** `table6_modality.csv`.

**Library:** `seaborn.clustermap` with `cmap="RdBu_r"`, `center=0`.

```
           no_bas  no_resp  no_ekg  cardio  bas_only
Sleep eff  ███     ·       ·       ████    ·       ← BAS-dominant
Age        ██      ·       ·       ██      █
─────────────────────────────────────────────
Sex        ██      ·       ██      ██      ██      ← BAS+EKG
─────────────────────────────────────────────
Apnea      █       ███     ·       ██      ████    ← RESP-dominant
BMI        ·       ·       ·       ████    ·       ← interaction-dominant

Red = harmful removal (negative Δ), white = neutral.
```

---

### 20. Channel Gain vs Modality Importance
**What:** For each task, scatter:
- x = ΔAUROC from v3→v3_full (full-channel gain, Transformer, K=all)
- y = importance of the most gained modality (from ablation, e.g., for apnea x=RESP gain)

Color by task. Tests the hypothesis: "tasks that benefit most from extra channels are
the ones whose key modality has more signal in the full-channel setup."

**Data:** `table6_modality.csv` (ablation) + comparison of v3 vs v3_full `analysis.csv`.

**Library:** `seaborn.scatterplot` with task labels annotated.

```
Full-ch gain (ΔAUROC)
  +0.044 ┤  apnea
  +0.039 ┤                  bmi
  +0.033 ┤        sex
  +0.022 ┤                      sleep_eff
  +0.008 ┤    age
  -0.006 ┤
         ├─────────────────────────────────
            low        high
       Importance of top modality (−ΔNo_X)
```

---

## Group 6 — Training Dynamics

---

### 21. Convergence Speed vs Context Length
**What:** From `training.csv` (all epochs, not just best), for each (task, head): how
many epochs until the model first reaches 95% of its best val_auroc? Plot this
"time-to-convergence" (epochs) against context length. Tests whether longer contexts
train slower (they should: fewer gradient steps per epoch, longer sequences).

**Data:** `training.csv`, all rows (is_best_epoch=False included).

**Why new:** U-shape plots show the curve shape; this collapses it to one number per
experiment and reveals the training cost scaling law.

**Library:** seaborn `lineplot` faceted by task.

```
Epochs to 95% peak
  35 ┤                          ●─●  Transformer
     │                     ●─●
  20 ┤               ●─●
     │          ●─●  
  10 ┤ ●─●─●─●
   5 ┤
     ├──┬────┬───┬────┬────┬────┬
       30s  10m  40m  80m  120m  240m
```

---

### 22. Early-Stopping Epoch Distribution (Violin)
**What:** For all experiments, show the distribution of the best epoch (epoch where
val_auroc peaks) as violin plots: one violin per context length, colored by head.
Shows whether longer contexts need more training (violin shifts right) or less (model
converges faster from a better initial signal).

**Data:** `training.csv`, filter `is_best_epoch == True`, group by context × head.

**Library:** `seaborn.violinplot` or `seaborn.boxenplot`.

```
Best epoch
  40 ┤     ╭──╮       ╭──╮
     │    │    │      │    │
  25 ┤ ╭──╮    ╰──╮  ╭╯    ╰╮  Transformer
     ││    │      ││ │       │
  10 ┤│    ╰──────╯╰─╯       │  LSTM / MeanPool
     ││                       │
   0 ┤╰───────────────────────╯
      30s  10m   40m  80m   120m  240m
```

---

### 23. "Return on Training" Curve
**What:** For each epoch, what is the val_auroc (already in training.csv)? Plot the
**cumulative training cost** (total FLOPs spent so far = epoch × steps_per_epoch ×
FLOPs_per_step) on the x-axis. Shows how efficiently training budget translates to
performance for different context lengths.

**Data:** `training.csv`, columns `steps_per_epoch`, `seq_len`, + compute formula.

**Why new:** The existing compute scaling plot (S-Fig 8) uses the BEST epoch only.
This plots the full trajectory — showing that some context lengths reach high AUROC
much faster per FLOP than others.

**Library:** matplotlib multi-line with `hue=context_length`.

---

## Group 7 — Cross-Round / Comparison Plots

---

### 24. Full vs Fast Channel Saturation Shift Map
**What:** For each (task, head): plot L* from v3 (x-axis) against L* from v3_full
(y-axis). Points BELOW the diagonal → full-channel saturates earlier (richer input
achieves goal with less context). Points ABOVE → full-channel needs MORE context.

**Data:** Table III equivalent for v3_full (compute from `analysis.csv` v3_full).

**Why new:** We report Δ peak AUROC for channel comparison. But does adding channels
change the *context requirement*? This directly answers that question.

**Library:** matplotlib scatter with diagonal reference line, annotated task labels.

```
L* (full-ch)
  240m ┤   ●sleep_eff
       │      ●sex  (below diagonal = full-ch needs less context)
  120m ┤         ●apnea
       │   ●age
   80m ┤        ●bmi (far below diagonal: richer channels need less time)
   40m ┤
       ├──────────────────────────
          40m    80m  120m   240m
                 L* (fast-ch)
Diagonal: same L* for both. Below = full-ch is more efficient.
```

---

### 25. SOTA Comparison Bubble Chart
**What:** Place our results alongside SleepFounder, OSF, SleepMaMi on shared tasks.
x = method (or pre-training hours on log scale), y = AUROC on key tasks (apnea, sex,
age, sleep staging κ). Bubble size = training data hours. Color = whether method
uses EEG.

**Data:** Numbers from SOTA_COMPARISON_AND_ABLATIONS.md + our results.

**Why new / important:** No figure in the paper currently positions us against SOTA.
A visual comparison is often more readable than in-text numbers and helps reviewers
quickly see where we stand. Use with care — our evaluation differs.

**Library:** `plotly.express.scatter` with bubble size.

```
AUROC
  0.94 ┤                      ◉ SleepFounder (OSA)
       │         ◉ Our Transformer (OSA)
  0.88 ┤               ◉ OSF (upstream)
       │    ◉ SleepMaMi
  0.82 ┤      ◉ Our LSTM (apnea)
       │
  0.76 ┤  ◉ SleepFounder (sex) ◉ Our Transformer (sex)
       ├───────────────────────────────────
         1K   10K  100K  800K (pre-training hours, log)
Bubble size = N test subjects. Color = EEG included (blue) / not (orange).
Note: different evaluation protocols — show with ⚠️ markers.
```

---

### 26. Ablation Interaction Plot (no_X vs combined)
**What:** For each task, plot AUROC for these conditions on a single axis:
Full | No BAS | No RESP | No EKG | Cardio | BAS-only
But additionally add **predicted-if-independent** points: if RESP and BAS were
independent, `predicted(no_bas+no_resp) = full − ΔBas − ΔRESP`. Compare to actual
joint-ablation values (which we don't have, but we can show the prediction).

**Data:** `table6_modality.csv` + simple algebra.

**Why new:** Reveals modality *interactions*. For BMI, cardio-only (RESP+EKG) drops
much more than removing RESP alone or EKG alone — suggesting a BAS×EMG interaction.
This figure makes that visible.

**Library:** `matplotlib` dot plot or `seaborn` `pointplot`.

---

## Group 8 — Clinical & Conceptual

---

### 27. Night Coverage Schematic (Explanatory Figure)
**What:** A conceptual illustration (not a data figure) showing:
- An 8-hour night as a horizontal bar
- Three rows: L=30s (many tiny windows), L=40m (medium), L=240m (few long windows)
- K=5 windows highlighted with solid color, rest gray
- Annotation showing "K×L = same total signal"

**Why:** The iso-compute concept is abstract. This figure makes it immediately visual:
"a 40m model with K=5 covers the same total time as a 30s model with K=40, but one
is trained on longer context."

**Library:** matplotlib `patches.Rectangle` or a Figma-style illustration.

```
8-hour night
┌────────────────────────────────────────────────────────────────┐
│L=30s:│░│░│░│░│░│░│░│░│░│░│░│░│░│░│░│░│░│░│░│░│░│░│░│░│░│░│░│░│
│     K=5 selected:│▓│         │▓│        │▓│        │▓│    │▓│   │
├────────────────────────────────────────────────────────────────┤
│L=40m:│░░░░░│░░░░░│░░░░░│░░░░░│░░░░░│░░░░░│░░░░░│░░░░░│░░░░░│░░│
│     K=5 selected:│▓▓▓▓▓│    │▓▓▓▓▓│         │▓▓▓▓▓│   │▓▓▓▓▓│  │
├────────────────────────────────────────────────────────────────┤
│L=240m:│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│      │
│     K=2 (only 2 fit, ≈ same K×L budget as 30s K=40)           │
└────────────────────────────────────────────────────────────────┘
```

---

### 28. Bootstrap Uncertainty Bands with Significance Markers
**What:** Re-plot the saturation curves (AUROC vs context) but with 95% CI bands AND
explicit markers showing where adjacent context lengths are **statistically distinguishable**
(non-overlapping CIs → ** marker). Shows not just the curve but the statistical certainty
of each step up.

**Data:** `analysis.csv` columns `mean_prob_auroc_ci_lo` and `_hi` (populated after
bootstrap run), K=all.

**Why new:** The current saturation curves have CI bands but no significance markers.
Reviewers often ask: "is the improvement from 80m to 120m real?" This answers it visually.

**Library:** matplotlib with `fill_between` + `*` / `ns` annotation above each step.

```
AUROC
  0.87 ┤            ○●═══════●═══════════●─── ← L*
       │        ○●════════●
  0.83 ┤    ○●═══════●     **  **   ns
       │  **  **
  0.79 ┤●
       ├──┬────┬───┬────┬────┬────┬
         30s  10m  40m  80m  120m  240m

○ = point estimate, ═══ = 95% CI band
** = significantly better than previous context (non-overlapping CI)
ns = not significant
```

---

### 29. Expected Clinical Impact (Reclassification Rate)
**What:** Given population prevalence and the AUROC improvement from 30s to L*, compute
the number of **additional patients correctly classified per 1,000 screened**, using
the NRI (Net Reclassification Improvement) formula. Show as a bar chart per task.

Makes the abstract AUROC improvement concrete: "+0.074 AUROC for apnea = ~18 extra
correctly classified per 1,000 screened at a standard threshold."

**Data:** Our AUROC numbers + published prevalence estimates (apnea ~30%, OSA severe
~10%, sleep eff poor ~20%, etc.).

**Why new:** Zero current figures connect AUROC to clinical impact. A reviewers/editors
from clinical journals often respond much more to "N extra patients" than "0.074 AUROC".

**Library:** seaborn `barplot` or `plotly.express.bar`.

---

### 30. "Free Lunch" Decomposition Waterfall
**What:** For each task, show a waterfall chart decomposing AUROC at 240m K=5 into
additive contributions:
- Start: AUROC@30s, K=1 (single short window, minimal effort)
- +1: Effect of aggregating K=1→K=5 at 30s (aggregation gain, free at inference)
- +2: Effect of extending L 30s→240m at K=1 (context gain, requires longer recording)
- +3: Effect of switching head from MeanPool→Transformer (architecture gain)
- =: Final AUROC@240m, K=5, Transformer

Not strictly additive (interactions exist) but gives an order-of-magnitude decomposition.

**Data:** `analysis.csv`, specific cells: (30s,K=1,MeanPool), (30s,K=5,MeanPool),
(240m,K=5,MeanPool), (240m,K=5,Transformer).

**Library:** `plotly.waterfall` or `matplotlib` stacked bar arrows.

```
AUROC
  0.91 ┤                              ┌─┐
       │                          ┌─┐ │↑│ +Head
  0.87 ┤                      ┌─┐ │↑│ │ │
       │                  ┌─┐ │↑│ │ │ │ │
  0.82 ┤              ┌─┐ │↑│ │ │ │ │ │ │
       │          ┌─┐ │↑│ │ │ │ │ │ │ │ │
  0.77 ┤      ┌─┐ │↑│ │ │ │ │ │ │ │ │ │ │
       │  ┌─┐ │ │ │ │ │ │ │ │ │ │ │ │ │ │
  0.69 ┤  │30s│ │K│ │L │ │Head   │ Final │
       │  │K=1│ │→5│ │→240m│     │       │
       └──────────────────────────────────
             +Agg   +Context  +Architecture
```

---

## Quick Summary Table

| # | Name | Data needed | Library | Effort | Novel angle |
|---|---|---|---|---|---|
| 1 | Bang-per-Minute | analysis.csv | matplotlib | Low | Time-unit marginal gain |
| 2 | Clinical Threshold Unlock Map | analysis.csv | seaborn heatmap | Low | L* per target AUROC |
| 3 | Iso-Compute Surface (3D) | heatmap_df.csv | plotly Surface | Medium | 3D efficiency landscape |
| 4 | Deployment Scenario Heatmap | analysis.csv dense | plotly | Medium | Decision guide grid |
| 5 | Head Advantage Growth Curve | analysis.csv | seaborn | Low | H4 trajectory |
| 6 | Modality Radar per Task | table6_modality | plotly radar | Low | Modality fingerprint |
| 7 | Head Convergence Context Map | analysis.csv | matplotlib | Low | When is Transformer worth it? |
| 8 | Night Fingerprint Heatmap | parquets | seaborn | Medium | Per-subject temporal signature |
| 9 | Cross-Task Hard-Subject Overlap | parquets (all tasks) | upsetplot | High | Subject-level multi-task |
| 10 | Per-Subject Confidence Trajectory | parquets | matplotlib | Medium | Individual K=1 trajectories |
| 11 | Pairwise Task Prediction Correlation | parquets (2 tasks) | seaborn | Medium | Inter-task embedding structure |
| 12 | Subject Prediction Stability Grid | parquets | seaborn heatmap | Medium | Sorted subject×context heatmap |
| 13 | Task Profile Radar | analysis.csv + tables | plotly | Low | Multi-dim task personality |
| 14 | Task Similarity Dendrogram | analysis.csv | seaborn clustermap | Low | Task clustering |
| 15 | Parallel Coordinates | analysis.csv + tables | plotly parcoords | Low | All properties at once |
| 16 | Context Payoff Scatter | analysis.csv | plotly | Low | All exps in one view |
| 17 | Modality Chord Diagram | table6_modality | mpl_chord_diagram | High | Modality interaction |
| 18 | Channel Count vs AUROC Scatter | analysis.csv v3+v3_full | seaborn | Low | Channels vs performance |
| 19 | Modality Ablation Clustermap | table6_modality | seaborn clustermap | Low | Table V as visual |
| 20 | Channel Gain vs Modality Importance | table6+analysis | seaborn | Low | Cross-round correlation |
| 21 | Convergence Speed vs Context | training.csv | seaborn | Low | Training cost scaling |
| 22 | Early-Stopping Epoch Violin | training.csv | seaborn | Low | Training distribution |
| 23 | Return on Training Curve | training.csv | matplotlib | Medium | Full trajectory vs FLOPs |
| 24 | Full vs Fast Channel L* Shift | analysis v3_full | matplotlib | Low | Does full-ch shorten L*? |
| 25 | SOTA Comparison Bubble | paper numbers + ours | plotly | Low | Positioning vs SleepFounder/OSF |
| 26 | Ablation Interaction Plot | table6_modality | matplotlib | Low | Modality independence test |
| 27 | Night Coverage Schematic | synthetic | matplotlib patches | Low | Explanatory/conceptual |
| 28 | Bootstrap Significance Markers | analysis.csv (w/ CI) | matplotlib | Low | Statistical validity |
| 29 | Clinical Impact Bar | analysis.csv + prevalence | seaborn | Low | AUROC → N patients |
| 30 | Waterfall Decomposition | analysis.csv (specific cells) | plotly waterfall | Low | Gain attribution |

**Highest value for the paper:** #2, #4, #6, #8, #12, #14, #19, #25, #28, #30  
**Most novel / surprising:** #9, #11, #24, #26, #29  
**Easiest to add to supplementary:** #1, #5, #6, #13, #15, #18, #21, #22  
