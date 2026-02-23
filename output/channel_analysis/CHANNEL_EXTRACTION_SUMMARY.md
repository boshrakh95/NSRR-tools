# NSRR Channel Extraction Results

**Date:** February 23, 2026  
**Datasets Analyzed:** STAGES (5 files), SHHS (10 files), APPLES (10 files), MrOS (10 files)  
**Total Unique Channels Found:** 100

## Summary by Dataset

| Dataset | Files | Avg Channels | Unique Channels | Sampling Rate |
|---------|-------|--------------|-----------------|---------------|
| STAGES  | 5     | 41.2         | 49             | 1000 Hz       |
| SHHS    | 10    | 15.3         | 19             | 125 Hz        |
| APPLES  | 10    | 15.3         | 16             | 100-200 Hz    |
| MrOS    | 10    | 22.0         | 22             | 512 Hz        |

## Channel Distribution by Modality

| Modality | Unique Channels | Notes |
|----------|-----------------|-------|
| **EEG**  | 24 | C3/C4/O1/O2 with M2/M1 reference, plus EEG_* variants with A1/A2 |
| **EOG**  | 7  | LOC/ROC standard, plus EOG(L)/EOG(R) and EOG_LOC-A2 variants |
| **ECG**  | 6  | Generic ECG, ECG L/R, ECG I/II/IIHF |
| **EMG**  | 11 | Chin, Leg (bilateral), Aux channels |
| **RESP** | 17 | Effort (thorax/abdomen), Airflow, SpO2, Pulse, Snore |
| **OTHER** | 5 | Position, Light, Sound, Plethysmography |
| **UNKNOWN** | 30 | xPAP/CPAP monitoring, Technical channels |

## Top 20 Most Common Channels

1. **SaO2** (20 files) - Oxygen saturation
2. **ECG** (20 files) - Electrocardiogram
3. **EMG** (20 files) - Electromyogram
4. **LOC** (20 files) - Left outer canthus (EOG)
5. **ROC** (20 files) - Right outer canthus (EOG)
6. **SpO2** (15 files) - Pulse oximetry
7. **H.R.** (10 files) - Heart rate
8. **EOG(L)** (10 files) - Left EOG
9. **EOG(R)** (10 files) - Right EOG
10. **EEG** (10 files) - Generic EEG
11. **THOR RES** (10 files) - Thoracic effort
12. **ABDO RES** (10 files) - Abdominal effort
13. **POSITION** (10 files) - Body position
14. **LIGHT** (10 files) - Light sensor
15. **snore** (10 files) - Snore signal
16. **C3_M2** (10 files) - Central EEG referenced to mastoid
17. **C4_M1** (10 files) - Central EEG referenced to mastoid
18. **O1_M2** (10 files) - Occipital EEG referenced to mastoid
19. **O2_M1** (10 files) - Occipital EEG referenced to mastoid
20. **AIRFLOW** (10 files) - Airflow signal

## Channels Missing from Config (69 total)

### Critical EEG Channels (17 missing)
These are commonly used referenced EEG channels that MUST be added:
- `C3_M2`, `C4_M1`, `O1_M2`, `O2_M1` - M1/M2 referenced (MrOS)
- `EEG_C3-A2`, `EEG_C4-A1`, `EEG_O1-A2`, `EEG_O2-A1` - A1/A2 referenced (STAGES)
- `EEG_F3-A2`, `EEG_F4-A1` - Frontal with A references
- `EEG_T3-A2`, `EEG_T4-A1` - Temporal with A references
- `EEG_P3-A2`, `EEG_P4-A1` - Parietal with A references
- `EEG_A1-A2` - Reference channels linked
- `EEG(sec)` - SHHS generic secondary EEG

### Critical EOG Channels (2 missing)
- `EOG_LOC-A2`, `EOG_ROC-A2` - Referenced EOG (STAGES)

### Critical ECG Channels (3 missing)
- `ECG_I`, `ECG_II` - Standard ECG leads (STAGES)
- `ECG_IIHF` - High-frequency filtered ECG

### Critical EMG Channels (6 missing)
- `EMG_Chin` - Chin EMG (STAGES)
- `LEG`, `Leg_1`, `Leg_2` - Leg movement channels
- `EMG_Aux1`, `EMG_Aux2` - Auxiliary EMG

### Critical RESP Channels (11 missing)
- `AIRFLOW`, `Airflow` - Nasal airflow
- `Cannula Flow` - Nasal cannula pressure
- `nasal_pres` - Nasal pressure
- `Effort_ABD`, `abdomen` - Abdominal effort
- `Effort_THO`, `thorax` - Thoracic effort
- `Flow_Patient`, `Flow_Patient2`, `Flow_Patient3` - CPAP device flow
- `snore` - Snore microphone
- `pulse` - Pulse signal

### Critical OTHER Channels (4 missing)
- `POSITION`, `Position` - Body position
- `LIGHT` - Light sensor
- `SOUND` - Sound/ambient noise

### Device/Technical Channels (18 UNKNOWN)
CPAP/xPAP monitoring (mostly from STAGES):
- `xPAP_CPAP`, `xPAP_IPAP`, `xPAP_EPAP` - Pressure levels
- `xPAP_IPAPMax`, `xPAP_IPAPMin` - IPAP limits
- `xPAP_EPAPMax`, `xPAP_EPAPMin` - EPAP limits
- `xPAP_PSMax`, `xPAP_PSMin` - Pressure support limits
- `xPAP_MaxPress` - Maximum pressure
- `Leak_Total`, `Press_Patient`, `PressCheck` - Device monitoring
- `BreathRate`, `RR` - Respiratory rate
- `Body` - Body movement
- `Technical` - Technical marker
- `DHR` - Heart rate derivative
- `thermistor` - Temperature sensor
- `NEW AIR`, `NEWAIR` - Unknown (possibly airflow variant)
- `OX stat`, `STAT`, `SUM`, `AUX` - Status/auxiliary channels

## Dataset-Specific Observations

### STAGES
- **Most comprehensive**: 49 unique channels including CPAP monitoring
- **High sampling rate**: 1000 Hz (requires downsampling to 128 Hz)
- **Referenced EEG**: Uses A1/A2 references (e.g., EEG_C3-A2)
- **CPAP focus**: Extensive xPAP monitoring channels
- **Notable**: Includes CO2 end-tidal, multiple flow channels

### SHHS (Sleep Heart Health Study)
- **Minimal channels**: 15-16 channels per file
- **Simple naming**: Generic labels (EEG, ECG, EMG, EOG(L), EOG(R))
- **Consistent**: Very uniform across all 10 files
- **Sampling rate**: 125 Hz (close to SleepFM target of 128 Hz)
- **Environmental**: Includes LIGHT, SOUND, POSITION sensors

### APPLES
- **Moderate complexity**: 15-16 channels
- **Mixed sampling**: 100 Hz and 200 Hz (needs standardization)
- **Referenced EEG**: Uses M1/M2 mastoid references
- **Respiratory focus**: Multiple flow/effort channels

### MrOS (Osteoporotic Fractures in Men)
- **Bilateral EMG**: Separate L/R channels (Leg L/R, L Chin/R Chin, ECG L/R)
- **Reference channels**: Includes separate A1, A2 channels
- **Moderate rate**: 512 Hz (needs downsampling)
- **Unusual channels**: SUM, STAT, DHR (status/derivative signals)
- **Very consistent**: All 10 files have identical 22 channels

## Recommendations

### 1. Update channel_definitions.yaml
Add all 69 missing channels to the configuration, organized by modality:
- Priority 1: Standard EEG/EOG/ECG/EMG/RESP channels (41 channels)
- Priority 2: Environmental/position sensors (4 channels)  
- Priority 3: Device/technical monitoring (18 channels)
- Consider: xPAP channels might be excluded if not needed for sleep staging

### 2. Channel Categorization
Some channels are currently UNKNOWN and need proper categorization:
- `H.R.`, `HR`, `DHR` → should be categorized as ECG-derived
- `SaO2`, `SpO2`, `OX stat` → currently EEG, should be RESP or separate modality
- `Effort_THO`, `thorax` → should be RESP
- `thermistor` → likely RESP (airflow sensor)
- Reference channels `A1`, `A2` → could be EEG or separate category

### 3. Sampling Rate Handling
Different datasets require different preprocessing:
- **STAGES (1000 Hz)**: Downsample to 128 Hz (factor of ~7.8)
- **MrOS (512 Hz)**: Downsample to 128 Hz (factor of 4)
- **APPLES (100-200 Hz)**: Mixed rates - need careful handling
- **SHHS (125 Hz)**: Very close to 128 Hz - minimal resampling

### 4. Reference Channel Handling
Three different referencing schemes found:
- **M1/M2 mastoid**: C3_M2, C4_M1, O1_M2, O2_M1 (MrOS, APPLES)
- **A1/A2 auricular**: EEG_C3-A2, EEG_C4-A1 (STAGES)
- **Linked ears**: EEG_A1-A2 (STAGES)

SleepFM accepts referenced channels, so these can be used directly without re-referencing.

### 5. Next Steps
1. ✅ Extract sample EDFs from all datasets
2. ✅ Run channel extraction script
3. ✅ Generate comprehensive analysis
4. ⏳ Update channel_definitions.yaml with missing channels
5. ⏳ Run full extraction on complete datasets (not just samples)
6. ⏳ Validate channel mapping with ChannelMapper
7. ⏳ Test preprocessing pipeline on sample data

## Files Generated

- `output/channel_analysis/stages_channels.csv` - STAGES channel inventory
- `output/channel_analysis/shhs_channels.csv` - SHHS channel inventory
- `output/channel_analysis/apples_channels.csv` - APPLES channel inventory
- `output/channel_analysis/mros_channels.csv` - MrOS channel inventory
- `output/channel_analysis/all_unique_channels.txt` - Combined unique channels
- `output/channel_analysis/channel_frequency.json` - Channel frequency analysis

## Validation Notes

- **Config coverage**: 634 channels in config, 100 found in data, 69 missing
- **Config variants**: 603 channels in config not found → these are alternative names/variants
- **SleepFM compatibility**: All found channels can be mapped to SleepFM's 4 modality groups
- **No FLOW channel**: As requested, FLOW was excluded from RESP modality
