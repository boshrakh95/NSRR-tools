"""
Extract actual channel names from all NSRR dataset EDF files.

This script scans all available EDF files across STAGES, SHHS, APPLES, and MrOS
datasets to extract actual channel names used in the data. This helps verify
and update our channel definitions based on real data.

Usage:
    uv run python scripts/extract_nsrr_channels.py [--datasets STAGES SHHS] [--max-files 100]
    
Output:
    - CSV file per dataset: output/channel_analysis/<dataset>_channels.csv
    - Summary report: output/channel_analysis/channel_summary.txt
    - Combined unique channels: output/channel_analysis/all_unique_channels.txt
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    import mne
except ImportError:
    print("Error: MNE package is required.")
    print("Install with: uv pip install mne")
    sys.exit(1)

from nsrr_tools.utils.config import Config
from nsrr_tools.core.channel_mapper import ChannelMapper


def find_edf_files_stages(raw_path: Path, max_files: int = None):
    """Find STAGES EDF files in various structures."""
    edf_files = []
    
    if not raw_path.exists():
        print(f"  Path does not exist: {raw_path}")
        return []
    
    # Try multiple patterns for STAGES
    patterns = [
        'original/*/usable/*.edf',  # Original structure
        '**/GSSA/*.edf',             # Sample extraction structure
        '**/*.edf'                   # Generic fallback
    ]
    
    for pattern in patterns:
        found = list(raw_path.glob(pattern))
        if found:
            print(f"  Found {len(found)} EDFs using pattern: {pattern}")
            for edf_path in found:
                # Extract subject ID from filename
                subject_id = edf_path.stem
                edf_files.append((subject_id, edf_path))
                if max_files and len(edf_files) >= max_files:
                    break
            break
    
    return edf_files[:max_files] if max_files else edf_files


def find_edf_files_generic(raw_path: Path, dataset_name: str, max_files: int = None):
    """Find EDF files with generic search patterns."""
    edf_files = []
    
    if not raw_path.exists():
        print(f"  Path does not exist: {raw_path}")
        return []
    
    # Try common patterns - ** for recursive search
    patterns = [
        '**/*.edf',  # Recursive search for all EDFs
    ]
    
    for pattern in patterns:
        found = list(raw_path.glob(pattern))
        if found:
            print(f"  Found {len(found)} EDFs using pattern: {pattern}")
            for edf_path in found:
                # Extract subject ID from filename (e.g., apples-160272.edf -> apples-160272)
                subject_id = edf_path.stem
                edf_files.append((subject_id, edf_path))
                if max_files and len(edf_files) >= max_files:
                    break
            break
    
    return edf_files[:max_files] if max_files else edf_files


def extract_channels_from_edf(edf_path: Path):
    """Extract channel names from EDF file without loading data."""
    try:
        # Read EDF header only
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        channels = raw.info['ch_names']
        sampling_freq = raw.info['sfreq']
        return channels, sampling_freq, None
    except Exception as e:
        return None, None, str(e)


def categorize_channel(channel_name: str):
    """Categorize channel by modality."""
    ch = channel_name.upper()
    
    # EEG
    if any(x in ch for x in ['C3', 'C4', 'O1', 'O2', 'F3', 'F4', 'FP1', 'FP2', 
                              'T3', 'T4', 'T5', 'T6', 'F7', 'F8', 'P3', 'P4',
                              'FZ', 'CZ', 'PZ', 'OZ', 'EEG']):
        return 'EEG'
    
    # EOG
    if any(x in ch for x in ['EOG', 'LOC', 'ROC', 'LEOG', 'REOG', 'E1', 'E2']):
        return 'EOG'
    
    # ECG/EKG
    if any(x in ch for x in ['ECG', 'EKG']):
        return 'ECG'
    
    # EMG
    if any(x in ch for x in ['CHIN', 'EMG', 'LEG', 'LLEG', 'RLEG', 
                              'ARM', 'MASSETER', 'MASS']):
        return 'EMG'
    
    # Respiratory
    if any(x in ch for x in ['THOR', 'CHEST', 'ABD', 'ABDO', 'ABDM',
                              'NASAL', 'FLOW', 'AIRFLOW', 'CANNULA',
                              'SNORE', 'SPO2', 'SAO2', 'PULSE', 'PPG']):
        return 'RESP'
    
    # Other
    if any(x in ch for x in ['POSITION', 'LIGHT', 'SOUND', 'TEMP', 
                              'PLETH', 'ACTIVITY']):
        return 'OTHER'
    
    return 'UNKNOWN'


def process_dataset(dataset_name: str, config: Config, max_files: int = None):
    """Process one dataset and extract all channel names."""
    print(f"\n{'='*80}")
    print(f"Processing {dataset_name.upper()} dataset")
    print(f"{'='*80}")
    
    dataset_paths = config.get_dataset_paths(dataset_name.lower())
    
    # Try sample directory first (for testing), then original
    raw_path = None
    for path_key in ['sample', 'original', 'raw']:
        if path_key in dataset_paths and dataset_paths[path_key].exists():
            raw_path = dataset_paths[path_key]
            print(f"  Using: {path_key} directory")
            break
    
    if not raw_path:
        print(f"  No data directory found for {dataset_name}")
        return None
    
    print(f"  Searching in: {raw_path}")
    
    # Find EDF files
    if dataset_name.lower() == 'stages':
        edf_files = find_edf_files_stages(raw_path, max_files)
    else:
        edf_files = find_edf_files_generic(raw_path, dataset_name, max_files)
    
    if not edf_files:
        print(f"  No EDF files found")
        return None
    
    print(f"  Found {len(edf_files)} EDF files")
    
    # Extract channels
    results = []
    channel_counts = Counter()
    modality_counts = defaultdict(Counter)
    errors = []
    
    for subject_id, edf_path in tqdm(edf_files, desc=f"  Extracting {dataset_name}"):
        channels, sfreq, error = extract_channels_from_edf(edf_path)
        
        if error:
            errors.append({'subject_id': subject_id, 'error': error})
            continue
        
        if channels:
            # Store per-subject info
            results.append({
                'dataset': dataset_name.upper(),
                'subject_id': subject_id,
                'num_channels': len(channels),
                'channels': ','.join(channels),
                'sampling_freq': sfreq,
                'edf_path': str(edf_path)
            })
            
            # Count channel occurrences
            for ch in channels:
                channel_counts[ch] += 1
                modality = categorize_channel(ch)
                modality_counts[modality][ch] += 1
    
    if errors:
        print(f"\n  Errors encountered: {len(errors)}")
        for err in errors[:5]:  # Show first 5
            print(f"    {err['subject_id']}: {err['error']}")
    
    if not results:
        print(f"  No channels extracted")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Summary statistics
    print(f"\n  Summary:")
    print(f"    Total subjects: {len(df)}")
    print(f"    Total unique channels: {len(channel_counts)}")
    print(f"    Avg channels per subject: {df['num_channels'].mean():.1f}")
    print(f"    Sampling frequencies: {df['sampling_freq'].unique()}")
    
    print(f"\n  Channel distribution by modality:")
    for modality in ['EEG', 'EOG', 'ECG', 'EMG', 'RESP', 'OTHER', 'UNKNOWN']:
        if modality in modality_counts:
            unique_channels = len(modality_counts[modality])
            print(f"    {modality}: {unique_channels} unique channels")
    
    return {
        'dataframe': df,
        'channel_counts': channel_counts,
        'modality_counts': modality_counts
    }


def compare_with_config(all_channels: set, config: Config, mapper: ChannelMapper):
    """Compare extracted channels with our configuration."""
    print(f"\n{'='*80}")
    print("Comparing with channel_definitions.yaml")
    print(f"{'='*80}")
    
    # Get all configured channel alternatives
    configured_channels = set()
    try:
        for channel_name, alternatives in config.channel_defs['channel_alternatives'].items():
            if isinstance(alternatives, list):
                configured_channels.update(alternatives)
    except Exception as e:
        print(f"Error loading configured channels: {e}")
        return
    
    print(f"  Channels in config: {len(configured_channels)}")
    print(f"  Channels found in data: {len(all_channels)}")
    
    # Find missing channels
    missing_in_config = all_channels - configured_channels
    print(f"  Missing in config: {len(missing_in_config)}")
    
    if missing_in_config:
        print(f"\n  Top 20 missing channels (to add to config):")
        # Categorize missing channels
        missing_by_modality = defaultdict(list)
        for ch in missing_in_config:
            modality = categorize_channel(ch)
            missing_by_modality[modality].append(ch)
        
        for modality in ['EEG', 'EOG', 'ECG', 'EMG', 'RESP', 'OTHER', 'UNKNOWN']:
            if modality in missing_by_modality:
                channels = sorted(missing_by_modality[modality])[:20]
                if channels:
                    print(f"\n    {modality}:")
                    for ch in channels:
                        print(f"      - {ch}")
    
    # Channels in config but not found
    not_found = configured_channels - all_channels
    print(f"\n  In config but not found in data: {len(not_found)} (these are variants/alternatives)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract channel names from NSRR EDF files"
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['stages', 'shhs', 'apples', 'mros'],
        help='Datasets to process (default: all)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files per dataset (for testing)'
    )
    parser.add_argument(
        '--output-dir',
        default='output/channel_analysis',
        help='Output directory for results'
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("NSRR Channel Extraction Tool")
    print(f"{'='*80}")
    print(f"Datasets: {', '.join(args.datasets)}")
    if args.max_files:
        print(f"Max files per dataset: {args.max_files}")
    print(f"Output directory: {output_dir}")
    
    # Load configuration
    print("\nLoading configuration...")
    config = Config()
    mapper = ChannelMapper(config)
    print("✓ Configuration loaded")
    
    # Process each dataset
    all_results = {}
    all_channels = set()
    
    for dataset in args.datasets:
        result = process_dataset(dataset, config, args.max_files)
        if result:
            all_results[dataset] = result
            
            # Save per-dataset CSV
            csv_file = output_dir / f"{dataset}_channels.csv"
            result['dataframe'].to_csv(csv_file, index=False)
            print(f"  ✓ Saved to {csv_file}")
            
            # Collect all channels
            for ch in result['channel_counts'].keys():
                all_channels.add(ch)
    
    if not all_results:
        print("\nNo data processed. Exiting.")
        return
    
    # Save combined summary
    print(f"\n{'='*80}")
    print("Generating summary reports")
    print(f"{'='*80}")
    
    # All unique channels
    unique_file = output_dir / 'all_unique_channels.txt'
    with open(unique_file, 'w') as f:
        f.write(f"Total unique channels across all datasets: {len(all_channels)}\n\n")
        for modality in ['EEG', 'EOG', 'ECG', 'EMG', 'RESP', 'OTHER', 'UNKNOWN']:
            channels = sorted([ch for ch in all_channels if categorize_channel(ch) == modality])
            if channels:
                f.write(f"\n{modality} ({len(channels)} channels):\n")
                for ch in channels:
                    f.write(f"  {ch}\n")
    print(f"  ✓ Saved to {unique_file}")
    
    # Channel frequency analysis
    freq_file = output_dir / 'channel_frequency.json'
    combined_counts = Counter()
    for dataset, result in all_results.items():
        combined_counts.update(result['channel_counts'])
    
    with open(freq_file, 'w') as f:
        json.dump({
            'total_unique': len(combined_counts),
            'by_frequency': [
                {'channel': ch, 'count': count, 'modality': categorize_channel(ch)}
                for ch, count in combined_counts.most_common()
            ]
        }, f, indent=2)
    print(f"  ✓ Saved to {freq_file}")
    
    # Compare with configuration
    compare_with_config(all_channels, config, mapper)
    
    print(f"\n{'='*80}")
    print("✓ Channel extraction complete!")
    print(f"{'='*80}")
    print(f"Results saved in: {output_dir}")
    print(f"  - Per-dataset CSVs: <dataset>_channels.csv")
    print(f"  - All unique channels: all_unique_channels.txt")
    print(f"  - Frequency analysis: channel_frequency.json")


if __name__ == '__main__':
    main()
