# XML to CSV Converter for SHHS/MrOS Annotations

## Purpose
This script converts SHHS and MrOS annotation XML files to CSV format for easier inspection and analysis. It helps visualize synchronization issues between signal recordings and sleep stage annotations.

## Location
`/home/boshra95/NSRR-tools/scripts/xml_to_csv_simple.py`

## Usage

### Basic conversion (all events)
```bash
python3 scripts/xml_to_csv_simple.py path/to/annotation.xml
```

### Stages only (exclude respiratory events, arousals, etc.)
```bash
python3 scripts/xml_to_csv_simple.py path/to/annotation.xml --stages-only
```

### Custom output path
```bash
python3 scripts/xml_to_csv_simple.py path/to/annotation.xml --output my_output.csv
```

## Examples

### SHHS annotation
```bash
cd /home/boshra95/NSRR-tools
python3 scripts/xml_to_csv_simple.py \
  /home/boshra95/scratch/nsrr_downloads/shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-200001-nsrr.xml \
  --stages-only
```

### MrOS annotation
```bash
cd /home/boshra95/NSRR-tools
python3 scripts/xml_to_csv_simple.py \
  /home/boshra95/scratch/nsrr_downloads/mros/polysomnography/annotations-events-nsrr/visit2/mros-visit2-aa2201-nsrr.xml \
  --stages-only
```

## Output Format

The CSV includes these columns:
- **Event**: Sleep stage (Wake, Stage1, Stage2, Stage3, Stage4, REM, Unscored) or event type
- **Stage_Number**: Numeric stage code (0=Wake, 1-4=NREM stages, 5=REM, -1=Unscored)
- **Stage_Sequence**: Sequential numbering of sleep stages (1, 2, 3, ...)
- **Start_Seconds**: Event start time in seconds from recording start
- **Start_Time**: Event start time in HH:MM:SS format
- **Duration_Seconds**: Event duration in seconds
- **End_Seconds**: Event end time in seconds
- **End_Time**: Event end time in HH:MM:SS format
- **Event_Type**: XML event type (e.g., "Stages|Stages", "Respiratory|Respiratory")
- **Clock_Time**: Original clock time from XML (if available)
- **Notes**: Additional information (e.g., SpO2 nadir/baseline for desaturation events)

## Key Features

1. **Gap Detection**: Automatically detects and reports gaps between:
   - Recording start and first annotation
   - Last annotation and recording end

2. **Stage Distribution**: Shows count of each sleep stage

3. **Time Formats**: Provides both seconds and HH:MM:SS for easy reading

4. **Filtering**: Can extract only sleep stages or include all events (respiratory, arousal, etc.)

## Understanding Synchronization Issues

The script output shows:
- **Recording duration**: Total signal recording length
- **Annotation duration**: Total time covered by sleep stage annotations
- **Gap at start**: Unscore d time before first annotation
- **Gap at end**: Unscored time after last annotation

If you see large gaps, it indicates that annotations don't cover the full recording duration, which explains why padding is needed during preprocessing.

## Example Output

```
Converting: shhs1-202622-nsrr.xml
Output to: shhs1-202622-nsrr.csv
Mode: Stages only

✓ Conversion complete!
  Total events: 61
  Recording starts at: 0.0s (00:00:00)
  Recording duration: 31590.0s (8.78 hours)

  Sleep stage events: 61
  First stage at: 0.0s (00:00:00)
  Last stage ends at: 31590.0s (08:46:30)
  Annotation duration: 31590.0s (8.78 hours)

  Stage distribution:
    Stage2: 30 epochs
    Wake: 31 epochs

✓ Saved to: shhs1-202622-nsrr.csv
```

## Notes

- No dependencies except Python standard library (no pandas, no numpy)
- Works with both SHHS (shhs1/shhs2) and MrOS (visit1/visit2) annotation formats
- Output CSV can be opened in Excel, LibreOffice, or any spreadsheet software
- Stage durations in XML are typically in multiples of 30 seconds (epoch length)
