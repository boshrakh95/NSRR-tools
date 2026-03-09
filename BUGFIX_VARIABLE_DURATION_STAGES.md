# CRITICAL BUG FIX: Variable-Duration Stage Handling in SHHS/MrOS

## Issue Summary
**Date**: March 9, 2026  
**Severity**: CRITICAL - Affects ALL SHHS and MrOS subjects  
**Impact**: False synchronization issues, unnecessary padding, incorrect metadata

## The Problem

### What You Observed
For subject `shhs2-204173`:
- **XML shows**: Annotations perfectly aligned (0s to 41340s, no gaps)
- **Preprocessing shows**: sync_difference_sec: 9869.996s (~2.7 hours!), status: "padded"

### Root Cause

Both SHHS and MrOS adapters had a hardcoded assumption that **all sleep stage annotations are 30 seconds long**:

**SHHS adapter** ([shhs_adapter.py:276](src/nsrr_tools/datasets/shhs_adapter.py#L276)):
```python
total_duration = stages[-1]['start'] + 30 if stages else 0  # WRONG!
```

**MrOS adapter** ([mros_adapter.py:235](src/nsrr_tools/datasets/mros_adapter.py#L235)):
```python
total_duration = stages[-1]['start'] + 30 if stages else 0  # WRONG!
```

### Why This Is Wrong

SHHS and MrOS XML files contain **variable-duration stage annotations**:

```xml
<EventConcept>Wake|0</EventConcept>
<Start>0.0</Start>
<Duration>1500.0</Duration>  <!-- 50 epochs, not 1! -->

<EventConcept>Stage 2 sleep|2</EventConcept>
<Start>1530.0</Start>
<Duration>270.0</Duration>  <!-- 9 epochs -->

<EventConcept>Wake|0</EventConcept>
<Start>35100.0</Start>
<Duration>6240.0</Duration>  <!-- 208 epochs! -->
```

### Impact on Subject shhs2-204173

| Metric | Correct Value | Buggy Calculation | Error |
|--------|---------------|-------------------|-------|
| Last stage start | 35100s | 35100s | ✓ |
| Last stage duration | **6240s** (208 epochs) | **30s** (hardcoded) | ❌ |
| Annotation end time | 41340s | 35130s | -6210s |
| Signal duration | 41340s | 41340s | ✓ |
| **Sync difference** | **0s** (perfect!) | **6210s** (wrong!) | **-6210s** |

The preprocessing code detected a 6210s "gap" that didn't exist, then padded with unscored epochs!

## The Fix

### Changed Code

**SHHS adapter** ([shhs_adapter.py:276-283](src/nsrr_tools/datasets/shhs_adapter.py#L276-L283)):
```python
# OLD (WRONG):
total_duration = stages[-1]['start'] + 30 if stages else 0

# NEW (CORRECT):
if stages:
    last_stage = stages[-1]
    total_duration = last_stage['start'] + last_stage['duration']
else:
    total_duration = 0
```

**MrOS adapter** - Same fix applied.

### What Changed
- ✅ Now uses **actual duration** from XML `<Duration>` element
- ✅ Handles variable-duration stages correctly (30s, 270s, 6240s, etc.)
- ✅ Calculates accurate annotation end time
- ✅ Prevents false synchronization issues

## Impact Assessment

### Affected Subjects
- **ALL SHHS subjects** (both shhs1 and shhs2)
- **ALL MrOS subjects** (both visit1 and visit2)

### Severity by Last Stage Duration
- If last stage = 30s: **No impact** (calculation was correct by luck)
- If last stage > 30s: **Undershoots annotation end** → False padding
- If last stage < 30s: Unlikely (SHHS uses 30s epochs)

### Expected Changes After Fix

For `shhs2-204173`:
```
BEFORE (buggy):
  annotation_duration: ~35130s (wrong!)
  sync_status: padded
  sync_difference_sec: 9869.996s
  sync_adjustment_epochs: 328

AFTER (fixed):
  annotation_duration: 41340s (correct!)
  sync_status: synchronized
  sync_difference_sec: 0.0s
  sync_adjustment_epochs: 0
```

## How to Verify the Fix

### 1. Check XML directly
```bash
cd /home/boshra95/NSRR-tools
python3 scripts/xml_to_csv_simple.py \
  /path/to/shhs2-204173-nsrr.xml --stages-only
```

Look for:
- ✓ Recording duration vs annotation duration
- ✓ Gap warnings at start/end

### 2. Rerun preprocessing
```bash
cd /home/boshra95/NSRR-tools
source .venv/bin/activate

# Reprocess a single subject to test
python scripts/preprocess_signals.py \
  --dataset shhs \
  --subject-id 204173_v2 \
  --output-dir /scratch/psg/shhs/derived/hdf5_signals \
  --skip-existing false
```

Check the output for:
- Sync status should be "synchronized" or "truncated" (not "padded")
- sync_difference_sec should be small

### 3. Check preprocessing summary
```bash
grep "204173_v2" /home/boshra95/scratch/psg/shhs/derived/logs/preprocessing_summary_shhs.csv
```

Expected changes:
- `sync_status`: `padded` → `synchronized`
- `sync_difference_sec`: `9869.996` → `0.0` (or small value)
- `sync_adjustment_epochs`: `328` → `0`

## Next Steps

### Required Actions
1. ✅ **Bug fixed** in both adapters
2. ⚠️ **Rerun preprocessing** for SHHS and MrOS datasets with fixed adapters
3. ⚠️ **Verify** synchronization statistics improve significantly

### Files Changed
- [src/nsrr_tools/datasets/shhs_adapter.py](src/nsrr_tools/datasets/shhs_adapter.py) (lines 274-283)
- [src/nsrr_tools/datasets/mros_adapter.py](src/nsrr_tools/datasets/mros_adapter.py) (lines 233-242)

### Testing
Run on a few subjects first to verify:
```bash
# Test subjects with known variable-duration stages
shhs2-204173  # Last stage: 6240s (this bug report)
shhs1-200001  # Various durations
mros-visit2-aa2201  # MrOS equivalent
```

## Related Issues

### Already Fixed
- ✅ Duration field parsing (was ignoring Duration, treating each annotation as 1 epoch)
- ✅ Stage text matching (was missing Wake and REM due to 'Stage' in text filter)

### This Fix
- ✅ **Variable-duration stage handling** (was assuming 30s for last stage)

## Conclusion

This was a critical bug that caused **false synchronization issues** for all SHHS/MrOS subjects where the last sleep stage was not exactly 30 seconds. The xml_to_csv tool revealed the issue by showing that annotations were actually perfectly aligned with recordings.

**The padding was unnecessary** - it was added because the adapter calculated the wrong annotation end time!
