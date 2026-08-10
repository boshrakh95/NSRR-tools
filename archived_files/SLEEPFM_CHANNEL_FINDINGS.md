# SleepFM Channel Requirements - Findings

## Key Findings from sleepfm-clinical Analysis

### 1. **Referencing: YES, SleepFM accepts referenced channels**
- SleepFM's `channel_groups.json` **explicitly includes** referenced formats:
  - `"C3-M2"`, `"C4-M1"`, `"O1-M2"`, `"O2-M1"` ✅
  - Also accepts: `"C3-A2"`, `"C4-A1"`, etc.
  - Also accepts raw: `"C3"`, `"C4"`, `"O1"`, `"O2"`
- **Conclusion**: Keep referenced channel names (C3-M2, C4-M1, etc.) - they're compatible!

### 2. **RESP Modality: FLOW channel**
User concern: "I don't want to use FLOW channel because it's mostly just noise"

SleepFM's RESP group includes:
- Airflow: `"Airflow"`, `"Flow"`, `"Nasal Pressure"`, `"NASAL P"`, `"Nasal"`, `"NasalP"`, `"Therm"`, `"Thermistor"`
- Effort belts: `"Thor"`, `"Thorax"`, `"THOR"`, `"Chest"`, `"Abd"`, `"ABD"`, `"ABDM"`, `"Abdomen"`
- Other: `"Snore"`, `"SpO2"`, `"Pulse"`, `"PPG"`

**Decision**: Remove FLOW from required channels. Use only **Thor** and **ABD** (effort belts).
- Respiratory effort belts are more reliable
- FLOW/nasal pressure is often noisy and artifact-prone
- SleepFM accepts RESP modality with just belts

### 3. **Channel Definitions Scope**
Currently `channel_definitions.yaml` includes variants from:
- STAGES dataset (confirmed via nocturn configs)
- Partially SHHS (C3-A2, C4-A1 patterns)
- General NSRR patterns from SleepFM's channel_groups.json

**Action**: Verify against APPLES and MrOS channel names (to be added later)

### 4. **Modality Grouping Verification**
Current grouping is correct:
- BAS (Brain Activity Signals) = EEG + EOG ✅
- RESP = Respiratory channels ✅
- EKG = Cardiac ✅
- EMG = Muscle ✅

All 4 modalities required for SleepFM (at least 1 channel per modality).

## Changes to Implement

1. **Remove FLOW from channel_definitions.yaml**
2. **Remove FLOW from modality_groups.yaml RESP channels**
3. **Update minimum_requirements** to not require FLOW
4. **Keep referenced channel names** (C3-M2, not C3)
5. **Add note** about FLOW being optional/excluded due to noise
