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
from nsrr_tools.utils.mount_utils import ensure_sshfs_mounted


def setup_logging(log_file: Path) -> None:
    """Configure logging."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, level="DEBUG", mode="w")


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
    
    # Load harmonized data (for apnea - contains both visits)
    harmonized_file = data_dir / 'shhs-harmonized-dataset-0.21.0.csv'
    logger.info(f"Loading harmonized data: {harmonized_file}")
    df_harmonized = pd.read_csv(harmonized_file, low_memory=False, encoding='latin-1')
    logger.info(f"Harmonized records: {len(df_harmonized)}")
    
    # Load Visit 1 data (for ESS)
    visit1_file = data_dir / 'shhs1-dataset-0.21.0.csv'
    logger.info(f"Loading Visit 1 data (for ESS): {visit1_file}")
    df_v1 = pd.read_csv(visit1_file, low_memory=False, encoding='latin-1')
    logger.info(f"Visit 1 records: {len(df_v1)}")
    
    # Load Visit 2 data (for ESS)
    visit2_file = data_dir / 'shhs2-dataset-0.21.0.csv'
    logger.info(f"Loading Visit 2 data (for ESS): {visit2_file}")
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
    apnea_col = task_config['column']  # nsrr_ahi_hp3r_aasm15
    visit_col = task_config['visit_column']  # visitnumber
    apnea_thresholds = config['thresholds']['apnea_class']['thresholds']
    class_labels = config['thresholds']['apnea_class']['class_labels']
    
    logger.info(f"Source: harmonized file")
    logger.info(f"Column: {apnea_col}")
    logger.info(f"Visit column: {visit_col}")
    logger.info(f"Thresholds: {apnea_thresholds}")
    logger.info(f"Class labels: {class_labels}")
    
    # --- Process Visit 1 ---
    logger.info("\nProcessing Visit 1...")
    df_v1_apnea = df_harmonized[df_harmonized[visit_col] == 1][[subject_id_col, apnea_col]].copy()
    logger.info(f"Visit 1 harmonized records: {len(df_v1_apnea)}")
    
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
    df_v1_apnea.rename(columns={apnea_col: 'ahi_score'}, inplace=True)
    apnea_v1 = df_v1_apnea[['subject_id', 'visit', 'apnea_class', 'ahi_score']].copy()
    
    # --- Process Visit 2 ---
    logger.info("\nProcessing Visit 2...")
    df_v2_apnea = df_harmonized[df_harmonized[visit_col] == 2][[subject_id_col, apnea_col]].copy()
    logger.info(f"Visit 2 harmonized records: {len(df_v2_apnea)}")
    
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
    df_v2_apnea.rename(columns={apnea_col: 'ahi_score'}, inplace=True)
    apnea_v2 = df_v2_apnea[['subject_id', 'visit', 'apnea_class', 'ahi_score']].copy()
    
    # Combine Visit 1 and Visit 2 apnea data
    apnea_targets = pd.concat([apnea_v1, apnea_v2], ignore_index=True)
    logger.info(f"\nTotal apnea records (both visits): {len(apnea_targets)}")
    logger.info(f"Valid apnea classifications (both visits): {(apnea_targets['apnea_class'] != '').sum()}")
    
    # ===================================================================
    # TASK 2: SLEEPINESS (MULTI-CLASS) - Visit 1 and Visit 2
    # ===================================================================
    
    logger.info("\n=== Task: sleepiness_class (Tier 1 - Multi-class) ===")
    
    task_config = shhs_config['tasks']['sleepiness_class']
    ess_col_v1 = task_config['columns']['visit1']  # ess_s1
    ess_col_v2 = task_config['columns']['visit2']  # ess_s2
    sleepiness_thresholds = config['thresholds']['sleepiness_class']['thresholds']
    sleepiness_labels = config['thresholds']['sleepiness_class']['class_labels']
    
    logger.info(f"Visit 1 column: {ess_col_v1}")
    logger.info(f"Visit 2 column: {ess_col_v2}")
    logger.info(f"Thresholds: {sleepiness_thresholds}")
    logger.info(f"Class labels: {sleepiness_labels}")
    
    # --- Process Visit 1 ---
    logger.info("\nProcessing Visit 1 ESS...")
    df_v1_ess = df_v1[[subject_id_col, ess_col_v1]].copy()
    
    # Handle missing data
    df_v1_ess[ess_col_v1] = df_v1_ess[ess_col_v1].replace(-9, pd.NA)
    
    # Apply multi-class threshold
    df_v1_ess['sleepiness_class'] = df_v1_ess[ess_col_v1].apply(
        lambda x: apply_multiclass_threshold(x, sleepiness_thresholds)
    )
    
    # Log statistics for Visit 1
    valid_v1_ess = df_v1_ess['sleepiness_class'] != ''
    class_dist_v1_ess = df_v1_ess['sleepiness_class'][valid_v1_ess].value_counts().sort_index()
    logger.info(f"Visit 1 - Valid sleepiness classifications: {valid_v1_ess.sum()}")
    for class_idx, count in class_dist_v1_ess.items():
        class_name = sleepiness_labels[int(class_idx)]
        logger.info(f"  Class {class_idx} ({class_name}): {count} ({count/valid_v1_ess.sum()*100:.1f}%)")
    
    # Create records with visit suffix
    df_v1_ess['subject_id'] = df_v1_ess[subject_id_col].astype(str) + '_v1'
    df_v1_ess['visit'] = 1
    df_v1_ess.rename(columns={ess_col_v1: 'ess_score'}, inplace=True)
    ess_v1 = df_v1_ess[['subject_id', 'visit', 'sleepiness_class', 'ess_score']].copy()
    
    # --- Process Visit 2 ---
    logger.info("\nProcessing Visit 2 ESS...")
    df_v2_ess = df_v2[[subject_id_col, ess_col_v2]].copy()
    
    # Handle missing data
    df_v2_ess[ess_col_v2] = df_v2_ess[ess_col_v2].replace(-9, pd.NA)
    
    # Apply multi-class threshold
    df_v2_ess['sleepiness_class'] = df_v2_ess[ess_col_v2].apply(
        lambda x: apply_multiclass_threshold(x, sleepiness_thresholds)
    )
    
    # Log statistics for Visit 2
    valid_v2_ess = df_v2_ess['sleepiness_class'] != ''
    class_dist_v2_ess = df_v2_ess['sleepiness_class'][valid_v2_ess].value_counts().sort_index()
    logger.info(f"Visit 2 - Valid sleepiness classifications: {valid_v2_ess.sum()}")
    for class_idx, count in class_dist_v2_ess.items():
        class_name = sleepiness_labels[int(class_idx)]
        logger.info(f"  Class {class_idx} ({class_name}): {count} ({count/valid_v2_ess.sum()*100:.1f}%)")
    
    # Create records with visit suffix
    df_v2_ess['subject_id'] = df_v2_ess[subject_id_col].astype(str) + '_v2'
    df_v2_ess['visit'] = 2
    df_v2_ess.rename(columns={ess_col_v2: 'ess_score'}, inplace=True)
    ess_v2 = df_v2_ess[['subject_id', 'visit', 'sleepiness_class', 'ess_score']].copy()
    
    # Combine Visit 1 and Visit 2 ESS data
    ess_targets = pd.concat([ess_v1, ess_v2], ignore_index=True)
    logger.info(f"\nTotal ESS records (both visits): {len(ess_targets)}")
    logger.info(f"Valid ESS classifications (both visits): {(ess_targets['sleepiness_class'] != '').sum()}")
    
    # ===================================================================
    # TASK 3: CVD (BINARY) - Subject-level (applies to both visits)
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
    
    # Merge ESS data (should match on subject_id and visit)
    targets = targets.merge(
        ess_targets,
        on=['subject_id', 'visit'],
        how='left'
    )
    
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

    # Fill missing values with empty string (consistent with other tasks)
    targets['sleepiness_class'] = targets['sleepiness_class'].fillna('')
    targets['ess_score'] = targets['ess_score'].fillna('')
    targets['cvd_binary'] = targets['cvd_binary'].fillna('')

    # ===================================================================
    # V2 TASKS (only executed if enabled in config)
    # ===================================================================

    visit_col = shhs_config['tasks']['apnea_class']['visit_column']  # visitnumber

    # --- sleep_efficiency_binary ---
    task_cfg = shhs_config['tasks'].get('sleep_efficiency_binary', {})
    if task_cfg.get('enabled', False):
        eff_col = task_cfg['column']
        eff_threshold = config['thresholds']['sleep_efficiency_binary']['threshold']
        logger.info(f"\n=== V2 Task: sleep_efficiency_binary (threshold < {eff_threshold}%) ===")

        def _eff_to_binary(val, thresh):
            if pd.isna(val):
                return ''
            return '1' if float(val) < thresh else '0'

        eff_frames = []
        for vis, vis_label in [(1, 'v1'), (2, 'v2')]:
            df_vis = df_harmonized[df_harmonized[visit_col] == vis][[subject_id_col, eff_col]].copy()
            df_vis[eff_col] = pd.to_numeric(df_vis[eff_col], errors='coerce').replace(-9, pd.NA)
            df_vis['sleep_efficiency_binary'] = df_vis[eff_col].apply(
                lambda x: _eff_to_binary(x, eff_threshold)
            )
            df_vis['eff_score'] = df_vis[eff_col].astype(str).replace(['nan', '<NA>'], '')
            df_vis['subject_id'] = df_vis[subject_id_col].astype(str) + f'_{vis_label}'
            df_vis['visit'] = vis
            valid = (df_vis['sleep_efficiency_binary'] != '').sum()
            pos = (df_vis['sleep_efficiency_binary'] == '1').sum()
            logger.info(f"  Visit {vis}: N={valid}, low_eff(1)={pos} ({pos/max(valid,1):.1%})")
            eff_frames.append(df_vis[['subject_id', 'visit', 'sleep_efficiency_binary', 'eff_score']])

        eff_targets = pd.concat(eff_frames, ignore_index=True)
        targets = targets.merge(eff_targets, on=['subject_id', 'visit'], how='left')
        targets['sleep_efficiency_binary'] = targets['sleep_efficiency_binary'].fillna('')
        targets['eff_score'] = targets['eff_score'].fillna('')

    # --- sex_binary ---
    task_cfg = shhs_config['tasks'].get('sex_binary', {})
    if task_cfg.get('enabled', False):
        sex_col = task_cfg['column']
        logger.info(f"\n=== V2 Task: sex_binary (column: {sex_col}) ===")

        def _sex_to_binary(val):
            v = str(val).strip().lower() if not pd.isna(val) else ''
            return '1' if v == 'female' else ('0' if v == 'male' else '')

        sex_frames = []
        for vis, vis_label in [(1, 'v1'), (2, 'v2')]:
            df_vis = df_harmonized[df_harmonized[visit_col] == vis][[subject_id_col, sex_col]].copy()
            df_vis['sex_binary'] = df_vis[sex_col].apply(_sex_to_binary)
            df_vis['subject_id'] = df_vis[subject_id_col].astype(str) + f'_{vis_label}'
            df_vis['visit'] = vis
            valid = (df_vis['sex_binary'] != '').sum()
            pos = (df_vis['sex_binary'] == '1').sum()
            logger.info(f"  Visit {vis}: N={valid}, female(1)={pos} ({pos/max(valid,1):.1%})")
            sex_frames.append(df_vis[['subject_id', 'visit', 'sex_binary']])

        sex_targets = pd.concat(sex_frames, ignore_index=True)
        targets = targets.merge(sex_targets, on=['subject_id', 'visit'], how='left')
        targets['sex_binary'] = targets['sex_binary'].fillna('')

    # --- age_regression ---
    task_cfg = shhs_config['tasks'].get('age_regression', {})
    if task_cfg.get('enabled', False):
        age_col = task_cfg['column']
        exclude_col = task_cfg.get('exclude_censored_col', 'nsrr_age_gt89')
        logger.info(f"\n=== V2 Task: age_regression (column: {age_col}) ===")

        age_frames = []
        for vis, vis_label in [(1, 'v1'), (2, 'v2')]:
            cols_needed = [c for c in [subject_id_col, age_col, exclude_col]
                          if c in df_harmonized.columns]
            df_vis = df_harmonized[df_harmonized[visit_col] == vis][cols_needed].copy()
            df_vis[age_col] = pd.to_numeric(df_vis[age_col], errors='coerce').replace(-9, pd.NA)
            if exclude_col in df_vis.columns:
                censored = df_vis[exclude_col].astype(str).str.strip().str.lower() == 'yes'
                df_vis.loc[censored, age_col] = pd.NA
                logger.info(f"  Visit {vis}: {censored.sum()} censored ages (>89) set to NaN")
            df_vis['age_value'] = df_vis[age_col].astype(str).replace(['nan', '<NA>'], '')
            df_vis['subject_id'] = df_vis[subject_id_col].astype(str) + f'_{vis_label}'
            df_vis['visit'] = vis
            valid = (df_vis['age_value'] != '').sum()
            logger.info(f"  Visit {vis}: N={valid} with valid age")
            age_frames.append(df_vis[['subject_id', 'visit', 'age_value']])

        age_targets = pd.concat(age_frames, ignore_index=True)
        targets = targets.merge(age_targets, on=['subject_id', 'visit'], how='left')
        targets['age_value'] = targets['age_value'].fillna('')

    # --- bmi_regression ---
    task_cfg = shhs_config['tasks'].get('bmi_regression', {})
    if task_cfg.get('enabled', False):
        bmi_col = task_cfg['column']
        logger.info(f"\n=== V2 Task: bmi_regression (column: {bmi_col}) ===")

        bmi_frames = []
        for vis, vis_label in [(1, 'v1'), (2, 'v2')]:
            df_vis = df_harmonized[df_harmonized[visit_col] == vis][[subject_id_col, bmi_col]].copy()
            df_vis[bmi_col] = pd.to_numeric(df_vis[bmi_col], errors='coerce').replace(-9, pd.NA)
            df_vis['bmi_value'] = df_vis[bmi_col].astype(str).replace(['nan', '<NA>'], '')
            df_vis['subject_id'] = df_vis[subject_id_col].astype(str) + f'_{vis_label}'
            df_vis['visit'] = vis
            valid = (df_vis['bmi_value'] != '').sum()
            logger.info(f"  Visit {vis}: N={valid} with valid BMI")
            bmi_frames.append(df_vis[['subject_id', 'visit', 'bmi_value']])

        bmi_targets = pd.concat(bmi_frames, ignore_index=True)
        targets = targets.merge(bmi_targets, on=['subject_id', 'visit'], how='left')
        targets['bmi_value'] = targets['bmi_value'].fillna('')

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
        # Map task names to score column names
        score_map = {
            'apnea': 'ahi_score',
            'sleepiness': 'ess_score'
        }
        score_col = score_map.get(task_name)
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
        'sleepiness_class': True,
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

    # Ensure SSHFS scratch mount is alive before touching any paths
    scratch_root = Path(config['paths']['raw_data']).parent  # cc_scratch/
    ensure_sshfs_mounted(
        mount_point=scratch_root,
        remote="boshra95@fir.alliancecan.ca:/home/boshra95/scratch/",
        options=["auto_cache", "reconnect", "compression=yes"],
    )

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
        
        # Build column order dynamically (v2 columns only included if present)
        _desired = [
            'subject_id', 'dataset', 'visit',
            'apnea_class', 'ahi_score',
            'sleepiness_class', 'ess_score',
            'cvd_binary',
            'sleep_efficiency_binary', 'eff_score',
            'sex_binary',
            'age_value',
            'bmi_value',
        ]
        column_order = [c for c in _desired if c in targets.columns]
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
