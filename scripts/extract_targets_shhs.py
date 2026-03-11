#!/usr/bin/env python3
"""
Extract Classification Targets for SHHS Dataset

Extracts Tier 1 and Tier 2 tasks:
- apnea_class (4 classes: Normal <5, Mild 5-15, Moderate 15-30, Severe ≥30) - from rdi3p
- cvd_binary (binary: 0=No CVD, 1=CVD) - from any_cvd (subject-level)

Each visit (1 and 2) is treated as a separate subject record.

Usage:
    python scripts/extract_targets_shhs.py --config configs/target_extraction.yaml
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
    apply_threshold,
    compute_task_statistics,
    load_config_file,
    save_dataset_targets,
)


def setup_logging(log_file: Path) -> None:
    """Configure logging."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, level="DEBUG", rotation="10 MB")


def extract_shhs_targets(config: dict) -> pd.DataFrame:
    """
    Extract all targets for SHHS dataset.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        DataFrame with extracted targets
    """
    dataset = 'shhs'
    shhs_config = config['tasks'][dataset]
    data_dir = Path(config['paths']['raw_data']) / dataset / 'datasets'
    
    subject_id_col = shhs_config['subject_id_column']  # nsrrid
    
    logger.info(f"=== Extracting Targets for {dataset.upper()} ===")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Subject ID column: {subject_id_col}")
    
    # ===================================================================
    # LOAD DATA FILES
    # ===================================================================
    
    # Load Visit 1 data
    visit1_file = data_dir / 'shhs1-dataset-0.21.0.csv'
    logger.info(f"Loading Visit 1 data: {visit1_file}")
    df_v1 = pd.read_csv(visit1_file, low_memory=False, encoding='latin-1')
    logger.info(f"Visit 1 records: {len(df_v1)}")
    
    # Load Visit 2 data
    visit2_file = data_dir / 'shhs2-dataset-0.21.0.csv'
    logger.info(f"Loading Visit 2 data: {visit2_file}")
    df_v2 = pd.read_csv(visit2_file, low_memory=False, encoding='latin-1')
    logger.info(f"Visit 2 records: {len(df_v2)}")
    
    # Load CVD data (subject-level, not visit-specific)
    cvd_file = data_dir / 'shhs-cvd-summary-dataset-0.21.0.csv'
    logger.info(f"Loading CVD data: {cvd_file}")
    df_cvd = pd.read_csv(cvd_file, low_memory=False, encoding='latin-1')
    logger.info(f"CVD records: {len(df_cvd)}")
    
    # ===================================================================
    # TASK 1: APNEA SEVERITY (MULTI-CLASS) - Visit 1 and Visit 2
    # ===================================================================
    
    logger.info("\n=== Task: apnea_class (Tier 1 - Multi-class) ===")
    
    task_config = shhs_config['tasks']['apnea_class']
    apnea_col = task_config['column']  # rdi3p
    apnea_thresholds = config['thresholds']['apnea_class']['thresholds']
    class_labels = config['thresholds']['apnea_class']['class_labels']
    
    logger.info(f"Column: {apnea_col}")
    logger.info(f"Thresholds: {apnea_thresholds}")
    logger.info(f"Class labels: {class_labels}")
    
    # --- Process Visit 1 ---
    logger.info("\nProcessing Visit 1...")
    df_v1_apnea = df_v1[[subject_id_col, apnea_col]].copy()
    
    # Handle missing data (replace -9 with NaN)
    df_v1_apnea[apnea_col] = df_v1_apnea[apnea_col].replace(-9, pd.NA)
    
    # Apply multi-class threshold
    df_v1_apnea['apnea_class'] = df_v1_apnea[apnea_col].apply(
        lambda x: apply_multiclass_threshold(x, apnea_thresholds)
    )
    
    # Log statistics for Visit 1
    valid_v1 = df_v1_apnea['apnea_class'] != ''
    class_dist_v1 = df_v1_apnea['apnea_class'][valid_v1].value_counts().sort_index()
    logger.info(f"Visit 1 - Valid apnea classifications: {valid_v1.sum()}")
    for class_idx, count in class_dist_v1.items():
        class_name = class_labels[int(class_idx)]
        logger.info(f"  Class {class_idx} ({class_name}): {count} ({count/valid_v1.sum()*100:.1f}%)")
    
    # Create subject_id with visit suffix
    df_v1_apnea['subject_id'] = df_v1_apnea[subject_id_col].astype(str) + '_v1'
    df_v1_apnea['visit'] = 1
    df_v1_apnea.rename(columns={apnea_col: 'rdi_score'}, inplace=True)
    apnea_v1 = df_v1_apnea[['subject_id', 'visit', 'apnea_class', 'rdi_score']].copy()
    
    # --- Process Visit 2 ---
    logger.info("\nProcessing Visit 2...")
    df_v2_apnea = df_v2[[subject_id_col, apnea_col]].copy()
    
    # Handle missing data
    df_v2_apnea[apnea_col] = df_v2_apnea[apnea_col].replace(-9, pd.NA)
    
    # Apply multi-class threshold
    df_v2_apnea['apnea_class'] = df_v2_apnea[apnea_col].apply(
        lambda x: apply_multiclass_threshold(x, apnea_thresholds)
    )
    
    # Log statistics for Visit 2
    valid_v2 = df_v2_apnea['apnea_class'] != ''
    class_dist_v2 = df_v2_apnea['apnea_class'][valid_v2].value_counts().sort_index()
    logger.info(f"Visit 2 - Valid apnea classifications: {valid_v2.sum()}")
    for class_idx, count in class_dist_v2.items():
        class_name = class_labels[int(class_idx)]
        logger.info(f"  Class {class_idx} ({class_name}): {count} ({count/valid_v2.sum()*100:.1f}%)")
    
    # Create subject_id with visit suffix
    df_v2_apnea['subject_id'] = df_v2_apnea[subject_id_col].astype(str) + '_v2'
    df_v2_apnea['visit'] = 2
    df_v2_apnea.rename(columns={apnea_col: 'rdi_score'}, inplace=True)
    apnea_v2 = df_v2_apnea[['subject_id', 'visit', 'apnea_class', 'rdi_score']].copy()
    
    # Combine Visit 1 and Visit 2 apnea data
    apnea_targets = pd.concat([apnea_v1, apnea_v2], ignore_index=True)
    logger.info(f"\nTotal apnea records (both visits): {len(apnea_targets)}")
    logger.info(f"Valid apnea classifications (both visits): {(apnea_targets['apnea_class'] != '').sum()}")
    
    # ===================================================================
    # TASK 2: CVD (BINARY) - Subject-level (applies to both visits)
    # ===================================================================
    
    logger.info("\n=== Task: cvd_binary (Tier 2 - Binary) ===")
    
    task_config = shhs_config['tasks']['cvd_binary']
    cvd_col = task_config['column']  # any_cvd
    
    logger.info(f"Column: {cvd_col}")
    logger.info(f"Note: CVD is subject-level (not visit-specific)")
    
    # Process CVD data
    df_cvd_proc = df_cvd[[subject_id_col, cvd_col]].copy()
    
    # Handle missing data
    df_cvd_proc[cvd_col] = df_cvd_proc[cvd_col].replace(-9, pd.NA)
    
    # Apply binary threshold (any_cvd is already 0/1, so use threshold of 0.5)
    df_cvd_proc['cvd_binary'] = df_cvd_proc[cvd_col].apply(
        lambda x: apply_threshold(x, threshold=0.5)
    )
    
    # Log statistics
    valid_cvd = df_cvd_proc['cvd_binary'] != ''
    cvd_dist = df_cvd_proc['cvd_binary'][valid_cvd].value_counts().sort_index()
    logger.info(f"Valid CVD classifications: {valid_cvd.sum()}")
    for cls, count in cvd_dist.items():
        label = "No CVD" if cls == '0' else "CVD"
        logger.info(f"  Class {cls} ({label}): {count} ({count/valid_cvd.sum()*100:.1f}%)")
    
    # Rename columns for merging
    df_cvd_proc.rename(columns={subject_id_col: 'nsrrid_orig'}, inplace=True)
    cvd_targets = df_cvd_proc[['nsrrid_orig', 'cvd_binary']].copy()
    
    # ===================================================================
    # MERGE ALL TASKS
    # ===================================================================
    
    logger.info("\n=== Merging All Tasks ===")
    
    # Start with apnea data (has subject_id with visit suffix)
    targets = apnea_targets.copy()
    
    # Extract original nsrrid from subject_id for CVD merge
    targets['nsrrid_orig'] = targets['subject_id'].str.replace('_v[12]$', '', regex=True).astype(int)
    
    # Merge CVD (left join - CVD available for subset of subjects)
    targets = targets.merge(
        cvd_targets,
        on='nsrrid_orig',
        how='left'
    )
    
    # Drop temporary merge key
    targets.drop(columns=['nsrrid_orig'], inplace=True)
    
    # Add dataset column
    targets['dataset'] = dataset
    
    # Fill missing CVD with empty string (consistent with other tasks)
    targets['cvd_binary'] = targets['cvd_binary'].fillna('')
    
    # ===================================================================
    # FINAL FORMATTING AND VALIDATION
    # ===================================================================
    
    # Reorder columns: subject_id, dataset, visit, then task columns
    task_columns = []
    score_columns = []
    for col in targets.columns:
        if col.endswith('_class') or col.endswith('_binary'):
            task_columns.append(col)
        elif col.endswith('_score'):
            score_columns.append(col)
    
    # Interleave task and score columns
    interleaved_cols = []
    for task_col in task_columns:
        interleaved_cols.append(task_col)
        # Find corresponding score column
        task_name = task_col.rsplit('_', 1)[0]  # Remove _class or _binary suffix
        score_col = f"{task_name}_score" if task_name == 'rdi' else None
        if score_col and score_col in score_columns:
            interleaved_cols.append(score_col)
    
    # Add any remaining score columns not matched
    for score_col in score_columns:
        if score_col not in interleaved_cols:
            interleaved_cols.append(score_col)
    
    column_order = ['subject_id', 'dataset', 'visit'] + interleaved_cols
    targets = targets[column_order]
    
    # Log final statistics
    logger.info(f"\n=== Final Dataset Statistics ===")
    logger.info(f"Total records: {len(targets)}")
    logger.info(f"Visit 1 records: {(targets['visit'] == 1).sum()}")
    logger.info(f"Visit 2 records: {(targets['visit'] == 2).sum()}")
    
    # Mark multi-class columns
    is_multiclass = {
        'apnea_class': True,
        'cvd_binary': False
    }
    
    # Compute statistics for all tasks
    stats = compute_task_statistics(
        targets,
        task_columns,
        dataset,
        is_multiclass=is_multiclass
    )
    
    return targets


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Extract classification targets for SHHS dataset'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/target_extraction.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent.parent / args.config
    config = load_config_file(config_path)
    
    # Setup logging
    log_dir = Path(config['paths']['targets_output'])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'extract_shhs.log'
    setup_logging(log_file)
    
    logger.info("Starting SHHS target extraction")
    logger.info(f"Configuration file: {config_path}")
    
    try:
        # Extract targets
        targets = extract_shhs_targets(config)
        
        # Save targets
        output_file = log_dir / 'shhs_targets.csv'
        
        # Define column order (consistent with APPLES)
        column_order = ['subject_id', 'dataset', 'visit', 'apnea_class', 'rdi_score', 'cvd_binary']
        save_dataset_targets(targets, output_file, 'shhs', column_order)
        
        logger.info("\n" + "="*80)
        logger.info("✅ SHHS extraction completed successfully!")
        logger.info(f"   Output: {output_file}")
        logger.info(f"   Total subjects: {len(targets)}")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.exception(f"❌ Error during SHHS target extraction: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
