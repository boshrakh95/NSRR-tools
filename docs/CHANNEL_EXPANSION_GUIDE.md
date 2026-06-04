# Channel Expansion Guide: Fast (7) → SleepFM Full (23)

**Goal:** Re-preprocess all NSRR datasets with the full SleepFM channel set (up to 23 channels),
then retrain the entire v2 experiment pipeline.

---

## 1. Current vs Target State

### What is in the HDF5 files right now (`strategy: "fast"`)

| Dataset | Channels stored | BAS | RESP | EKG | EMG |
|---------|----------------|-----|------|-----|-----|
| STAGES  | C3-M2, C4-M1, LOC, ROC, EKG, CHIN, LLEG, Airflow | 4 | 1 | 1 | 2 |
| SHHS    | EEG, LOC, ROC, EKG, EMG, Airflow, Thor | 3 | 2 | 1 | 1 |
| APPLES  | C3-M2, C4-M1, LOC, ROC, EKG, EMG, LLEG, Airflow | 4 | 1 | 1 | 2 |
| MrOS    | C3-M2, C4-M1, LOC, ROC, EKG, CHIN, LLEG, Airflow | 4 | 1 | 1 | 2 |

### What SleepFM can accept (`strategy: "sleepfm_full"`)

| Modality | Max | Priority channels (stored name in HDF5) |
|----------|-----|-----------------------------------------|
| BAS      | 10  | C3-M2, C4-M1, LOC, ROC, O1-M2, O2-M1, F3-M2, F4-M1, A1, A2 |
| RESP     | 7   | Airflow, Thor, ABD, SpO2, HR, Snore, RespRate |
| EKG      | 2   | EKG, ECG-L |
| EMG      | 4   | CHIN, LLEG, RLEG, EMG |

**Not every dataset will have every channel** — preprocessing silently skips missing ones.
The SleepFM encoder zero-pads to the max, so having fewer channels in a given file is fine.

---

## 2. Pre-flight Fixes (Required Before Reprocessing)

Two channel names written by preprocessing are **not** in
`sleepfm-clinical/sleepfm/configs/channel_groups.json`,
so SleepFM would silently ignore them at load time.
Fix both before reprocessing.

### Fix 1 — Add `RespRate` and `ECG-L` / `ECG-R` to channel_groups.json

```bash
python3 - <<'EOF'
import json, copy
path = "/home/boshra95/sleepfm-clinical/sleepfm/configs/channel_groups.json"
with open(path) as f:
    d = json.load(f)

# RespRate is not in RESP; add it
if "RespRate" not in d["RESP"]:
    d["RESP"].append("RespRate")
    print("Added RespRate to RESP")

# ECG-L / ECG-R are not in EKG (EKG-L and ECG L are, but not ECG-L)
for ch in ["ECG-L", "ECG-R"]:
    if ch not in d["EKG"]:
        d["EKG"].append(ch)
        print(f"Added {ch} to EKG")

with open(path, "w") as f:
    json.dump(d, f, indent=2)
print("channel_groups.json updated.")
EOF
```

### Fix 2 — Change strategy in preprocessing_params.yaml

Edit `configs/preprocessing_params.yaml`, line ~35:

```yaml
# BEFORE:
strategy: "fast"

# AFTER:
strategy: "sleepfm_full"
```

Or run:
```bash
sed -i 's/strategy: "fast"/strategy: "sleepfm_full"/' \
    /home/boshra95/NSRR-tools/configs/preprocessing_params.yaml

# Verify:
grep "strategy:" /home/boshra95/NSRR-tools/configs/preprocessing_params.yaml
```

---

## 3. Preprocessing — Re-run All Datasets

Re-processing overwrites existing HDF5 files with the expanded channel set.
Run all 4 datasets **in parallel** (each is an independent job):

```bash
cd /home/boshra95/NSRR-tools

sbatch jobs/preprocess_signals_parallel.sh stages --no-skip-existing
sbatch jobs/preprocess_signals_parallel.sh shhs   --no-skip-existing
sbatch jobs/preprocess_signals_parallel.sh apples --no-skip-existing
sbatch jobs/preprocess_signals_parallel.sh mros   --no-skip-existing
```

**Resources per job:**
- Time: 26 h  ·  CPUs: 8  ·  RAM: 30 GB  ·  No GPU
- All 4 can run simultaneously if the CPU partition allows it.

**Monitor:**
```bash
squeue -u boshra95
tail -f logs/preprocess_stages_*.out   # live log for STAGES job
```

### Optional: test on 5 subjects first

Before running the full reprocess, verify the new channel set looks right:

```bash
sbatch jobs/preprocess_signals_parallel.sh stages 5 --no-skip-existing --log-level DEBUG
```

Then inspect:
```bash
python3 - <<'EOF'
import h5py, glob
files = sorted(glob.glob("/scratch/boshra95/psg/stages/derived/hdf5_signals/*.h5"))
for f in files[:3]:
    with h5py.File(f) as h:
        print(f.split("/")[-1], "->", sorted(h.keys()))
EOF
```

Expected result (STAGES has occipital and frontal leads):
```
STNF00032.h5 -> ['ABD', 'Airflow', 'C3-M2', 'C4-M1', 'CHIN', 'EKG', 'F3-M2', 'F4-M1',
                  'LLEG', 'LOC', 'O1-M2', 'O2-M1', 'ROC', 'SpO2', 'Thor']
```

---

## 4. Verify Output After Preprocessing

Once all 4 jobs finish, confirm the expanded channels:

```bash
python3 - <<'EOF'
import h5py, glob, json
from collections import Counter

channel_groups = json.load(
    open("/home/boshra95/sleepfm-clinical/sleepfm/configs/channel_groups.json"))
all_known = {ch: mod for mod, chs in channel_groups.items() for ch in chs}

datasets = {
    "stages": "/scratch/boshra95/psg/stages/derived/hdf5_signals/",
    "shhs":   "/scratch/boshra95/psg/shhs/derived/hdf5_signals/",
    "apples": "/scratch/boshra95/psg/apples/derived/hdf5_signals/",
    "mros":   "/scratch/boshra95/psg/mros/derived/hdf5_signals/",
}
for name, path in datasets.items():
    files = sorted(glob.glob(path + "*.h5"))[:10]
    ch_freq = Counter()
    for f in files:
        with h5py.File(f) as h:
            for ch in h.keys():
                ch_freq[ch] += 1
    print(f"\n{name}  ({len(files)} sampled):")
    for ch, cnt in sorted(ch_freq.items()):
        mod = all_known.get(ch, "UNKNOWN ⚠️")
        mark = "✓" if mod != "UNKNOWN ⚠️" else "⚠️"
        print(f"  {mark} {ch:<20} {mod:<8}  {cnt}/{len(files)} files")
EOF
```

**What to look for:**
- No `UNKNOWN ⚠️` entries — all stored channel names must be in channel_groups.json.
- STAGES should now have ~14–15 channels per file.
- SHHS will have fewer (generic `EEG` instead of C3-M2; only 2–3 RESP channels typically).
- Any `UNKNOWN` channel means you need to add it to channel_groups.json and re-run that dataset.

### Count files to make sure nothing was skipped

```bash
for ds in stages shhs apples mros; do
    n=$(ls /scratch/boshra95/psg/${ds}/derived/hdf5_signals/*.h5 2>/dev/null | wc -l)
    echo "$ds: $n HDF5 files"
done
# Expected: stages=1513, shhs=8444, apples=1104, mros=3933
```

---

## 5. No Config Changes Needed for Training

`configs/phase0_v2_config.yaml` already specifies the full channel limits:

```yaml
data:
  max_channels:
    BAS: 10
    RESP: 7
    EKG: 2
    EMG: 4
  channel_priority:
    BAS:  [C3-M2, C4-M1, LOC, ROC, O1-M2, O2-M1, F3-M2, F4-M1, A1, A2]
    RESP: [Airflow, Thor, ABD, SpO2, HR, Snore, RespRate]
    EKG:  [EKG, ECG-L, ECG-R]
    EMG:  [CHIN, LLEG, RLEG, EMG]
```

The SleepFM encoder will automatically pick up the new channels from HDF5
and zero-pad whichever modalities have fewer than the maximum.
**No edits required.**

---

## 6. Retrain All Experiments

Because the SleepFM encoder now sees more real channels (less zero-padding),
the embeddings it produces will differ from the fast-strategy run.
**All existing trained models must be retrained from scratch.**

### Option A — retrain everything at once (recommended)

Submit one training job per experiment entry from `experiments/v2_registry.yaml`.
Use `gen_commands.py` to generate the commands:

```bash
cd /home/boshra95/NSRR-tools

# Tier 1 — large N, all three heads
python scripts/gen_commands.py train sex_binary_lstm         | bash
python scripts/gen_commands.py train sex_binary_transformer  | bash
python scripts/gen_commands.py train sex_binary_mean_pool    | bash

python scripts/gen_commands.py train sleep_efficiency_binary_lstm        | bash
python scripts/gen_commands.py train sleep_efficiency_binary_transformer | bash
python scripts/gen_commands.py train sleep_efficiency_binary_mean_pool   | bash

python scripts/gen_commands.py train bmi_binary_lstm         | bash
python scripts/gen_commands.py train bmi_binary_transformer  | bash
python scripts/gen_commands.py train bmi_binary_mean_pool    | bash

python scripts/gen_commands.py train age_class_lstm          | bash
python scripts/gen_commands.py train age_class_transformer   | bash
python scripts/gen_commands.py train age_class_mean_pool     | bash

# Tier 2 — smaller N
python scripts/gen_commands.py train psqi_binary_lstm               | bash
python scripts/gen_commands.py train depression_extreme_binary_lstm | bash
python scripts/gen_commands.py train osa_binary_apples_postqc_lstm  | bash
python scripts/gen_commands.py train osa_severity_apples_lstm       | bash
```

Each `train` call submits **one job per context length** (e.g., 6 jobs × 6 contexts = 36 jobs
for `sex_binary_lstm`). They all run in parallel if GPU slots are available.

### Option B — one task at a time, verify before moving on

```bash
# Start with sex_binary (already familiar from v1)
python scripts/gen_commands.py train sex_binary_lstm | bash
# Wait for completion, check results, then proceed
python scripts/gen_commands.py status sex_binary_lstm
```

### Monitoring training

```bash
python scripts/gen_commands.py runs               # all running jobs
python scripts/gen_commands.py runs sex_binary_lstm -v   # verbose history
squeue -u boshra95                               # SLURM queue
```

---

## 7. Re-run Inference and Analysis

After training completes for each experiment:

```bash
# Inference
python scripts/gen_commands.py infer sex_binary_lstm | bash
python scripts/gen_commands.py infer sex_binary_transformer | bash
# ... etc

# Standard analysis (sparse K sweep + plots)
python scripts/gen_commands.py analyze sex_binary_lstm --plot | bash

# Iso-compute analysis (dense K sweep → 7 plots)
python scripts/gen_commands.py analyze sex_binary_lstm --k-dense | bash
python scripts/gen_commands.py build-heatmap sex_binary_lstm     | bash
python scripts/gen_commands.py iso-plots sex_binary_lstm         | bash

# Saturation curve (once all 3 heads for a task are done)
python scripts/gen_commands.py saturation sex_binary \
    --heads lstm transformer mean_pool | bash
```

---

## 8. Expected Channel Counts per Dataset After Reprocessing

These are **typical** counts — individual subjects may have fewer if their EDF
lacked a particular channel. The SleepFM encoder zero-pads any missing slots.

| Dataset | Typical BAS | Typical RESP | EKG | EMG | Total typical |
|---------|------------|-------------|-----|-----|---------------|
| STAGES  | 8–10 (C3,C4,O1,O2,F3,F4,LOC,ROC + A1,A2 if available) | 3–5 (Airflow,Thor,ABD,SpO2,HR) | 1–2 | 2–4 | 14–21 |
| SHHS    | 3 (EEG,LOC,ROC — no electrode-specific EEG labels) | 2–3 (Airflow,Thor occasionally ABD) | 1 | 1 | 7–8 |
| APPLES  | 6–8 (C3,C4,LOC,ROC + occipital/frontal if present) | 3–5 | 1–2 | 3–4 | 13–19 |
| MrOS    | 6–8 | 3–5 | 1 | 3–4 | 13–18 |

**Note on SHHS:** SHHS uses generic `EEG`, `LOC`, `ROC` channel names in its EDFs
(no per-electrode labels). It will benefit most from the additional RESP channels
(Thor, ABD, SpO2) rather than additional EEG derivations.

---

## 9. Summary Checklist

```
Pre-flight (do once, before any jobs):
  ☐  Edit preprocessing_params.yaml: strategy "fast" → "sleepfm_full"
  ☐  Run channel_groups.json fix script (adds RespRate, ECG-L, ECG-R)
  ☐  Optional: test on 5 STAGES subjects and verify HDF5 output

Preprocessing (4 parallel jobs, ~26 h):
  ☐  sbatch stages --no-skip-existing
  ☐  sbatch shhs   --no-skip-existing
  ☐  sbatch apples --no-skip-existing
  ☐  sbatch mros   --no-skip-existing
  ☐  Verify HDF5 channel counts and no UNKNOWN channels

Training (many parallel GPU jobs):
  ☐  Tier 1: sex_binary, sleep_efficiency, bmi_binary, age_class (3 heads each)
  ☐  Tier 2: psqi, depression, osa tasks (lstm only)
  ☐  Monitor via gen_commands.py runs / squeue

Inference + Analysis (after each task's training is done):
  ☐  infer → analyze --plot → analyze --k-dense → build-heatmap → iso-plots
  ☐  saturation (once all 3 heads done per task)
```
