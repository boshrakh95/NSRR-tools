#!/usr/bin/env python3
"""
Extract Classification Targets for APPLES Dataset

Extracts Tier 1 tasks:
- apnea_class (4 classes: Normal <5, Mild 5-15, Moderate 15-30, Severe ≥30)
- depression_class (4 classes: Normal 1-10, Mild 11-16, Borderline 17-20, Mod-Severe 21+)
- sleepiness_class (3 classes: Normal 0-10, Mild-Moderate 11-15, Severe 16-24)

Usage:
    python scripts/extract_targets_apples.py --config configs/target_extraction.yaml
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nsrr_tools.targets.extraction_utils import (
    apply_multiclass_threshold,
    compute_task_statistics,
    load_config_file,
    save_dataset_targets,
    validate_score_range,
)


def setup_logging(log_file: Path) -> None:
    """Configure logging."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, level="DEBUG", rotation="10 MB")


def extract_apples_targets(config: dict) -> pd.DataFrame:
    """
    Extract all targets for APPLES dataset.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        DataFrame with extracted targets
    """
    dataset = 'apples'
    logger.info("="*80)
    logger.info("APPLES Target Extraction")
    logger.info("="*80)
    
    # Get paths and configuration
    raw_data_path = Path(config['paths']['raw_data'])
    dataset_path = raw_data_path / dataset / "datasets"
    apples_config = config['tasks'][dataset]
    thresholds = config['thresholds']
    
    # Note: APPLES uses different subject IDs in main vs harmonized files
    # Main CSV: appleid
    # Harmonized CSV: nsrrid
    # We need to map between them
    subject_id_col = apples_config['subject_id_column']  # appleid
    
    # ----------------------------
    # Load required CSV files
    # ----------------------------
    
    # Main dataset file (for depression, sleepiness)
    main_file = dataset_path / "apples-dataset-0.1.0.csv"
    logger.info(f"Loading main dataset: {main_file}")
    df_main = pd.read_csv(main_file)
    logger.info(f"  Loaded {len(df_main)} records")
    
    # Harmonized dataset (for apnea)
    harmonized_file = dataset_path / "apples-harmonized-dataset-0.1.0.csv"
    logger.info(f"Loading harmonized dataset: {harmonized_file}")
    df_harmonized = pd.read_csv(harmonized_file)
    logger.info(f"  Loaded {len(df_harmonized)} records")
    
    # ----------------------------
    # Extract Task 1: Apnea Class (Visit 3 - DX visit)
    # 4 classes: 0=<5, 1=5-15, 2=15-30, 3=≥30
    # ----------------------------
    logger.info("\n" + "-"*60)
    logger.info("Task 1: Apnea Classification (4 classes)")
    logger.info("-"*60)
    
    apnea_config = apples_config['apnea_class']
    ahi_col = apnea_config['column']
    visit_filter = apnea_config['visit_filter']
    visit_col = apnea_config['visit_column']
    
    # Filter to DX visit (harmonized file uses nsrrid)
    df_dx = df_harmonized[df_harmonized[visit_col] == visit_filter].copy()
    logger.info(f"  Filtered to visit {visit_filter}: {len(df_dx)} subjects")
    
    # Replace -9 with NaN for missing data
    df_dx[ahi_col] = df_dx[ahi_col].replace(-9, pd.NA)
    
    # Validate AHI scores
    validate_score_range(
        df_dx, ahi_col, 
        config['validation']['apnea_binary']['ahi_range'],
        dataset, 'apnea_class'
    )
    
    # Apply multi-class thresholds
    apnea_thresholds = thresholds['apnea_class']['thresholds']
    df_dx['apnea_class'] = df_dx[ahi_col].apply(
        lambda x: apply_multiclass_threshold(x, apnea_thresholds)
    )
    df_dx['ahi_score'] = df_dx[ahi_col].astype(str).replace(['nan', '<NA>'], '')
    
    # Log class distribution
    class_dist = df_dx['apnea_class'][df_dx['apnea_class'] != ''].value_counts().sort_index()
    logger.info(f"  Class distribution: {dict(class_dist)}")
    for class_label, count in class_dist.items():
        class_name = thresholds['apnea_class']['class_labels'][int(class_label)]
        logger.info(f"    Class {class_label} ({class_name}): {count}")
    
    # Keep relevant columns (rename nsrrid to subject_id for consistency)
    apnea_targets = df_dx[['nsrrid', 'apnea_class', 'ahi_score']].copy()
    apnea_targets.rename(columns={'nsrrid': 'subject_id'}, inplace=True)
    
    # ----------------------------
    # Extract Task 2: Depression Class (Visit 1 - Baseline)
    # 4 classes: 0=1-10, 1=11-16, 2=17-20, 3=21-40
    # ----------------------------
    logger.info("\n" + "-"*60)
    logger.info("Task 2: Depression Classification (4 classes, BDI-II)")
    logger.info("-"*60)
    
    depression_config = apples_config['depression_class']
    bdi_col = depression_config['column']
    visit_filter = depression_config['visit_filter']
    visit_col = depression_config['visit_column']
    scale = depression_config['scale']  # 'bdi'
    
    # Filter to baseline visit (main file uses appleid)
    df_bl = df_main[df_main[visit_col] == visit_filter].copy()
    logger.info(f"  Filtered to visit {visit_filter}: {len(df_bl)} subjects")
    
    # Replace -9 with NaN for missing data
    df_bl[bdi_col] = df_bl[bdi_col].replace(-9, pd.NA)
    
    # Validate BDI scores
    validate_score_range(
        df_bl, bdi_col,
        config['validation']['depression_binary']['bdi_range'],
        dataset, 'depression_class'
    )
    
    # Apply multi-class thresholds (BDI scale)
    depression_thresholds = thresholds['depression_class']['scales'][scale]['thresholds']
    df_bl['depression_class'] = df_bl[bdi_col].apply(
        lambda x: apply_multiclass_threshold(x, depression_thresholds)
    )
    df_bl['bdi_score'] = df_bl[bdi_col].astype(str).replace(['nan', '<NA>'], '')
    
    # Log class distribution
    class_dist = df_bl['depression_class'][df_bl['depression_class'] != ''].value_counts().sort_index()
    logger.info(f"  Class distribution: {dict(class_dist)}")
    for class_label, count in class_dist.items():
        class_name = thresholds['depression_class']['scales'][scale]['class_labels'][int(class_label)]
        logger.info(f"    Class {class_label} ({class_name}): {count}")
    
    # Keep relevant columns (rename appleid to subject_id)
    depression_targets = df_bl[[subject_id_col, 'depression_class', 'bdi_score']].copy()
    depression_targets.rename(columns={subject_id_col: 'subject_id'}, inplace=True)
    
    # ----------------------------
    # Extract Task 3: Sleepiness Class (Visit 3 - DX visit)
    # 3 classes: 0=0-10, 1=11-15, 2=16-24
    # ----------------------------
    logger.info("\n" + "-"*60)
    logger.info("Task 3: Sleepiness Classification (3 classes, ESS)")
    logger.info("-"*60)
    
    sleepiness_config = apples_config['sleepiness_class']
    ess_col = sleepiness_config['column']
    visit_filter = sleepiness_config['visit_filter']
    visit_col = sleepiness_config['visit_column']
    
    # Filter to DX visit (main file uses appleid)
    df_dx_sleep = df_main[df_main[visit_col] == visit_filter].copy()
    logger.info(f"  Filtered to visit {visit_filter}: {len(df_dx_sleep)} subjects")
    
    # Replace -9 with NaN for missing data
    df_dx_sleep[ess_col] = df_dx_sleep[ess_col].replace(-9, pd.NA)
    
    # Validate ESS scores
    validate_score_range(
        df_dx_sleep, ess_col,
        config['validation']['sleepiness_binary']['ess_range'],
        dataset, 'sleepiness_class'
    )
    
    # Apply multi-class thresholds
    sleepiness_thresholds = thresholds['sleepiness_class']['thresholds']
    df_dx_sleep['sleepiness_class'] = df_dx_sleep[ess_col].apply(
        lambda x: apply_multiclass_threshold(x, sleepiness_thresholds)
    )
    df_dx_sleep['ess_score'] = df_dx_sleep[ess_col].astype(str).replace(['nan', '<NA>'], '')
    
    # Log class distribution
    class_dist = df_dx_sleep['sleepiness_class'][df_dx_sleep['sleepiness_class'] != ''].value_counts().sort_index()
    logger.info(f"  Class distribution: {dict(class_dist)}")
    for class_label, count in class_dist.items():
        class_name = thresholds['sleepiness_class']['class_labels'][int(class_label)]
        logger.info(f"    Class {class_label} ({class_name}): {count}")
    
    # Keep relevant columns (rename appleid to subject_id)
    sleepiness_targets = df_dx_sleep[[subject_id_col, 'sleepiness_class', 'ess_score']].copy()
    sleepiness_targets.rename(columns={subject_id_col: 'subject_id'}, inplace=True)
    
    # ----------------------------
    # Merge all targets
    # ----------------------------
    logger.info("\n" + "-"*60)
    logger.info("Merging Targets")
    logger.info("-"*60)
    
    # All targets now use 'subject_id', so merge on that
    # Start with apnea (Visit 3) and add sleepiness (also Visit 3)
    targets = apnea_targets.merge(
        sleepiness_targets,
        on='subject_id',
        how='outer'
    )
    
    # Add depression (Visit 1)
    targets = targets.merge(
        depression_targets,
        on='subject_id',
        how='outer'
    )
    
    # Add metadata columns
    targets['dataset'] = dataset
    targets['visit'] = 0  # Collapsed across visits - single record per subject
    
    # Fill NaN with empty string for consistency
    for col in targets.columns:
        if col not in ['subject_id', 'dataset', 'visit']:
            targets[col] = targets[col].fillna('')
    
    logger.info(f"Total subjects in merged targets: {len(targets)}")
    
    # Compute statistics
    logger.info("\n" + "-"*60)
    logger.info("Target Statistics")
    logger.info("-"*60)
    
    # Mark multi-class columns
    is_multiclass = {
        'apnea_class': True,
        'depression_class': True,
        'sleepiness_class': True
    }
    
    stats = compute_task_statistics(
        targets,
        ['apnea_class', 'depression_class', 'sleepiness_class'],
        dataset,
        is_multiclass=is_multiclass
    )
    
    return targets


def main():
    parser = argparse.ArgumentParser(
        description="Extract classification targets for APPLES dataset"
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/target_extraction.yaml'),
        help='Path to target extraction config file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Override output path (default: from config)'
    )
    args = parser.parse_args()
    
    # Load configuration
    if not args.config.exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config_file(args.config)
    
    # Setup logging
    log_dir = Path(config['paths']['targets_output'])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "extract_apples.log"
    setup_logging(log_file)
    
    logger.info("="*80)
    logger.info("APPLES Target Extraction Script")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Log file: {log_file}")
    
    # Extract targets
    try:
        targets_df = extract_apples_targets(config)
        
        # Save results
        output_path = args.output or (log_dir / "apples_targets.csv")
        
        # Define column order
        columns_order = [
            'subject_id', 'dataset', 'visit',
            'apnea_class', 'ahi_score',
            'depression_class', 'bdi_score',
            'sleepiness_class', 'ess_score'
        ]
        
        save_dataset_targets(targets_df, output_path, 'apples', columns_order)
        
        logger.info("\n" + "="*80)
        logger.info("✅ APPLES extraction completed successfully!")
        logger.info(f"   Output: {output_path}")
        logger.info(f"   Total subjects: {len(targets_df)}")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.exception(f"❌ Error during extraction: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
