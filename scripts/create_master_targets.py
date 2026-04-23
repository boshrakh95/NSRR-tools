#!/usr/bin/env python3
"""
Create Master Targets Parquet

Merges per-dataset target CSVs into a single unified master file.

Per-dataset CSVs use mixed multiclass / binary columns depending on dataset:
  APPLES : apnea_class (4), depression_class (4), sleepiness_class (3)
  SHHS   : apnea_class (4), sleepiness_class (3), cvd_binary
  MrOS   : apnea_class (4), sleepiness_class (3), insomnia_binary, cvd_binary, rested_morning
  STAGES : apnea_class (4), depression_binary, anxiety_binary, insomnia_binary, sleepiness_binary

This script harmonises everything to binary labels in the master file:
  apnea_class     → apnea_binary    (class >= 2 → AHI ≥ 15 → 1)
  sleepiness_class→ sleepiness_binary (class >= 1 → ESS ≥ 11 → 1)
  depression_class→ depression_binary (class >= 1 → BDI ≥ 11 → 1)

Score column naming:
  bdi_score / phq9_score  → depression_score
  All other score columns keep their names.

Missing values:
  binary targets   → -1
  continuous scores→ NaN

Usage:
    python scripts/create_master_targets.py --config configs/target_extraction.yaml
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nsrr_tools.targets.extraction_utils import load_config_file
from nsrr_tools.utils.mount_utils import ensure_sshfs_mounted


# ---------------------------------------------------------------------------
# Multiclass → binary mapping rules
# class_col  : name of the multiclass column in the per-dataset CSV
# binary_col : name of the binary column to create in master
# min_class  : class index that maps to binary 1 (everything >= min_class → 1)
# ---------------------------------------------------------------------------
_MULTICLASS_TO_BINARY = [
    # apnea_class: 0=Normal(<5), 1=Mild(5-15), 2=Moderate(15-30), 3=Severe(≥30)
    # → binary: AHI ≥ 15  → moderate+ → class ≥ 2
    dict(class_col='apnea_class',      binary_col='apnea_binary',      min_class=2),
    # sleepiness_class: 0=Normal(0-10), 1=Mild-Mod(11-15), 2=Severe(16-24)
    # → binary: ESS ≥ 11 → class ≥ 1
    dict(class_col='sleepiness_class', binary_col='sleepiness_binary', min_class=1),
    # depression_class: 0=Normal(BDI 1-10), 1=Mild(11-16), 2=Borderline(17-20), 3=Mod-Severe(21+)
    # → binary: any depression (BDI ≥ 11) → class ≥ 1
    dict(class_col='depression_class', binary_col='depression_binary', min_class=1),
]

# Master output columns (in order)
_MASTER_COLUMNS = [
    'unified_id', 'dataset', 'subject_id', 'visit',
    'apnea_binary', 'ahi_score',
    'depression_binary', 'depression_score',
    'sleepiness_binary', 'ess_score',
    'anxiety_binary', 'gad7_score',
    'insomnia_binary', 'isi_score',
    'fatigue_binary', 'fss_score',
    'cvd_binary',
    'rested_morning',
    # v2 additions
    'sex_binary',
    'sleep_efficiency_binary', 'eff_score',
    'psqi_binary', 'psqi_score',
    'depression_extreme_binary',
    'osa_binary_apples_postqc',
    'age_value',
    'bmi_value', 'bmi_binary',
    'extraction_date',
]

# Columns storing float regression targets (not binary, not score)
_REGRESSION_VALUE_COLUMNS = {'age_value', 'bmi_value'}

MISSING_BINARY = -1
DATASET_FILES = {
    'apples': 'apples_targets.csv',
    'shhs':   'shhs_targets.csv',
    'mros':   'mros_targets.csv',
    'stages': 'stages_targets.csv',
}


def setup_logging(log_file: Path) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, level="DEBUG", mode="w")


def _multiclass_to_binary(val: str, min_class: int) -> int:
    """Convert a string class label to binary. Returns MISSING_BINARY if empty/missing."""
    if val == '' or pd.isna(val):
        return MISSING_BINARY
    try:
        return 1 if int(val) >= min_class else 0
    except (ValueError, TypeError):
        return MISSING_BINARY


def _binary_str_to_int(val) -> int:
    """Convert '0'/'1'/'' string binary to int. Empty/NaN → MISSING_BINARY."""
    if val == '' or pd.isna(val):
        return MISSING_BINARY
    try:
        return int(val)
    except (ValueError, TypeError):
        return MISSING_BINARY


def _score_to_float(val) -> float:
    """Convert score string to float. Empty/NaN → NaN."""
    if val == '' or pd.isna(val):
        return float('nan')
    try:
        return float(val)
    except (ValueError, TypeError):
        return float('nan')


def load_dataset(csv_path: Path, dataset_name: str) -> pd.DataFrame:
    """Load one per-dataset CSV and normalise to master schema."""
    logger.info(f"  Loading {dataset_name}: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Per-dataset CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str).fillna('')
    logger.info(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    out = pd.DataFrame()
    out['subject_id'] = df['subject_id']
    out['dataset']    = df['dataset']
    out['visit']      = pd.to_numeric(df['visit'], errors='coerce').astype('Int64')

    # --- Convert multiclass → binary ---
    for rule in _MULTICLASS_TO_BINARY:
        cc, bc, mc = rule['class_col'], rule['binary_col'], rule['min_class']
        if cc in df.columns:
            out[bc] = df[cc].apply(lambda v: _multiclass_to_binary(v, mc))
            logger.info(f"    {cc} → {bc} (class >= {mc})")

    # --- Pass-through binary columns ---
    for col in ['apnea_binary', 'depression_binary', 'sleepiness_binary',
                'anxiety_binary', 'insomnia_binary', 'fatigue_binary',
                'cvd_binary', 'rested_morning',
                # v2 binary
                'sex_binary', 'sleep_efficiency_binary', 'psqi_binary',
                'depression_extreme_binary', 'osa_binary_apples_postqc',
                'bmi_binary']:
        if col in df.columns and col not in out.columns:
            out[col] = df[col].apply(_binary_str_to_int)

    # --- Score columns ---
    # depression_score: coalesce bdi_score (APPLES) and phq9_score (STAGES)
    if 'bdi_score' in df.columns:
        out['depression_score'] = df['bdi_score'].apply(_score_to_float)
    elif 'phq9_score' in df.columns:
        out['depression_score'] = df['phq9_score'].apply(_score_to_float)

    for src, dst in [('ahi_score', 'ahi_score'), ('ess_score', 'ess_score'),
                     ('gad7_score', 'gad7_score'), ('isi_score', 'isi_score'),
                     ('fss_score', 'fss_score'), ('rested_score', 'rested_score'),
                     # v2 scores
                     ('eff_score', 'eff_score'), ('psqi_score', 'psqi_score')]:
        if src in df.columns and dst not in out.columns:
            out[dst] = df[src].apply(_score_to_float)

    # --- Regression value columns (float, NaN if missing) ---
    for col in ['age_value', 'bmi_value']:
        if col in df.columns:
            out[col] = df[col].apply(_score_to_float)

    return out


def build_master(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate all datasets and finalise master schema."""
    master = pd.concat(dfs, ignore_index=True, sort=False)

    # unified_id
    master['unified_id'] = (
        master['dataset'].astype(str) + '_' +
        master['subject_id'].astype(str) + '_v' +
        master['visit'].astype(str)
    )

    # extraction_date
    master['extraction_date'] = str(date.today())

    # Fill missing columns with appropriate sentinel
    for col in _MASTER_COLUMNS:
        if col not in master.columns:
            if col in _REGRESSION_VALUE_COLUMNS:
                master[col] = float('nan')
            elif col.endswith('_binary') or col in ('rested_morning', 'cvd_binary'):
                master[col] = MISSING_BINARY
            elif col.endswith('_score') or col == 'depression_score':
                master[col] = float('nan')
            else:
                master[col] = ''

    # Enforce binary sentinel for any remaining NaN in binary columns
    binary_cols = [c for c in _MASTER_COLUMNS
                   if (c.endswith('_binary') or c in ('rested_morning', 'cvd_binary'))
                   and c not in _REGRESSION_VALUE_COLUMNS]
    for col in binary_cols:
        master[col] = master[col].fillna(MISSING_BINARY).astype(int)

    return master[_MASTER_COLUMNS]


def log_statistics(master: pd.DataFrame) -> None:
    logger.info("\n" + "=" * 80)
    logger.info("Master Target Statistics")
    logger.info("=" * 80)
    logger.info(f"Total records: {len(master):,}")
    logger.info(f"Datasets: {master['dataset'].value_counts().to_dict()}")

    binary_cols = [c for c in _MASTER_COLUMNS
                   if (c.endswith('_binary') or c in ('rested_morning', 'cvd_binary'))
                   and c not in _REGRESSION_VALUE_COLUMNS]
    for col in binary_cols:
        if col not in master.columns:
            continue
        valid = master[col][master[col] != MISSING_BINARY]
        if len(valid) == 0:
            logger.info(f"  {col}: all missing")
            continue
        n1 = (valid == 1).sum()
        n0 = (valid == 0).sum()
        n_miss = (master[col] == MISSING_BINARY).sum()
        logger.info(
            f"  {col}: N={len(valid):,}  pos={n1:,} ({n1/len(valid):.1%})  "
            f"neg={n0:,} ({n0/len(valid):.1%})  missing={n_miss:,}"
        )

    # Regression value columns
    for col in _REGRESSION_VALUE_COLUMNS:
        if col not in master.columns:
            continue
        valid = master[col].dropna()
        if len(valid) == 0:
            logger.info(f"  {col}: all missing")
            continue
        logger.info(
            f"  {col}: N={len(valid):,}  mean={valid.mean():.1f}  "
            f"std={valid.std():.1f}  [min={valid.min():.1f}, max={valid.max():.1f}]  "
            f"missing={master[col].isna().sum():,}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-dataset target CSVs into master parquet"
    )
    parser.add_argument(
        '--config', type=Path,
        default=Path('configs/target_extraction.yaml'),
    )
    parser.add_argument(
        '--output-dir', type=Path,
        help='Override output directory (default: targets_output from config)'
    )
    parser.add_argument(
        '--datasets', nargs='+',
        default=list(DATASET_FILES.keys()),
        choices=list(DATASET_FILES.keys()),
        help='Which datasets to include (default: all)'
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    config = load_config_file(args.config)

    # Ensure SSHFS scratch mount is alive
    scratch_root = Path(config['paths']['raw_data']).parent
    ensure_sshfs_mounted(
        mount_point=scratch_root,
        remote="boshra95@fir.alliancecan.ca:/home/boshra95/scratch/",
        options=["auto_cache", "reconnect", "compression=yes"],
    )

    targets_dir = args.output_dir or Path(config['paths']['targets_output'])
    targets_dir.mkdir(parents=True, exist_ok=True)

    log_file = targets_dir / "create_master_targets.log"
    setup_logging(log_file)

    logger.info("=" * 80)
    logger.info("Creating Master Targets")
    logger.info("=" * 80)
    logger.info(f"Targets directory: {targets_dir}")
    logger.info(f"Datasets: {args.datasets}")

    dfs = []
    for dataset in args.datasets:
        csv_path = targets_dir / DATASET_FILES[dataset]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Dataset: {dataset.upper()}")
        logger.info(f"{'=' * 60}")
        df = load_dataset(csv_path, dataset)
        dfs.append(df)
        logger.info(f"  → {len(df):,} records after normalisation")

    logger.info(f"\n{'=' * 60}")
    logger.info("Building master DataFrame")
    logger.info(f"{'=' * 60}")
    master = build_master(dfs)

    log_statistics(master)

    # Save
    parquet_path = targets_dir / "master_targets.parquet"
    csv_path     = targets_dir / "master_targets.csv"

    master.to_parquet(parquet_path, index=False)
    master.to_csv(csv_path, index=False)

    logger.info(f"\nSaved parquet: {parquet_path}")
    logger.info(f"Saved CSV:     {csv_path}")
    logger.info(f"Total records: {len(master):,}")
    logger.info("Done.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
