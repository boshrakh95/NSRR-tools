#!/usr/bin/env python3
"""
Extract Classification Targets for MrOS Dataset

Extracts Tier 1, Tier 2, and Tier 3 tasks:
- apnea_class (4 classes: Normal <5, Mild 5-15, Moderate 15-30, Severe ≥30)
- sleepiness_class (3 classes: Normal 0-10, Mild-Moderate 11-15, Severe 16-24)
- insomnia_binary (binary: 0=No insomnia, 1=Moderate+ insomnia, ISI ≥15)
- cvd_binary (binary: 0=No CVD, 1=CVD from cvchd)
- rested_morning (binary: derived from poxqual3 scale 1-5)

Each visit (1 and 2) is treated as a separate subject record.

Usage:
    python scripts/extract_targets_mros.py --config configs/target_extraction.yaml
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


def extract_mros_targets(config: dict) -> pd.DataFrame:
    """
    Extract all targets for MrOS dataset.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        DataFrame with extracted targets
    """
    dataset = 'mros'
    mros_config = config['tasks'][dataset]
    data_dir = Path(config['paths']['raw_data']) / dataset / 'datasets'
    
    subject_id_col = mros_config['subject_id_column']  # nsrrid
    
    logger.info(f"=== Extracting Targets for {dataset.upper()} ===")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Subject ID column: {subject_id_col}")
    
    # ===================================================================
    # LOAD DATA FILES
    # ===================================================================
    
    # Load harmonized data for Visit 1 only (for apnea)
    # Note: MrOS only has harmonized file for Visit 1, not Visit 2
    harmonized_v1_file = data_dir / 'mros-visit1-harmonized-0.6.0.csv'
    logger.info(f"Loading Visit 1 harmonized data: {harmonized_v1_file}")
    df_harm_v1 = pd.read_csv(harmonized_v1_file, low_memory=False)
    logger.info(f"Visit 1 harmonized records: {len(df_harm_v1)}")
    logger.info(f"Note: Visit 2 harmonized file does not exist - will leave AHI empty for V2")
    
    # Load main dataset files for both visits (for ESS, ISI, CVD, rested)
    main_v1_file = data_dir / 'mros-visit1-dataset-0.6.0.csv'
    logger.info(f"Loading Visit 1 main data: {main_v1_file}")
    df_main_v1 = pd.read_csv(main_v1_file, low_memory=False)
    logger.info(f"Visit 1 main records: {len(df_main_v1)}")
    
    main_v2_file = data_dir / 'mros-visit2-dataset-0.6.0.csv'
    logger.info(f"Loading Visit 2 main data: {main_v2_file}")
    df_main_v2 = pd.read_csv(main_v2_file, low_memory=False)
    logger.info(f"Visit 2 main records: {len(df_main_v2)}")
    
    # ===================================================================
    # TASK 1: APNEA SEVERITY (MULTI-CLASS) - Visit 1 and Visit 2
    # ===================================================================
    
    logger.info("\n=== Task: apnea_class (Tier 1 - Multi-class) ===")
    
    task_config = mros_config['tasks']['apnea_class']
    apnea_col = task_config['column']  # nsrr_ahi_hp3r_aasm15
    apnea_thresholds = config['thresholds']['apnea_class']['thresholds']
    class_labels = config['thresholds']['apnea_class']['class_labels']
    
    logger.info(f"Column: {apnea_col}")
    logger.info(f"Thresholds: {apnea_thresholds}")
    logger.info(f"Class labels: {class_labels}")
    
    # --- Process Visit 1 ---
    logger.info("\nProcessing Visit 1...")
    df_v1_apnea = df_harm_v1[[subject_id_col, apnea_col]].copy()
    
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
        pct = 100 * count / valid_v1.sum()
        logger.info(f"  Class {class_idx} ({class_labels[int(class_idx)]}): {count} ({pct:.1f}%)")
    
    # Create subject_id with visit suffix
    df_v1_apnea['subject_id'] = df_v1_apnea[subject_id_col].astype(str) + '_v1'
    df_v1_apnea['visit'] = 1
    df_v1_apnea.rename(columns={apnea_col: 'ahi_score'}, inplace=True)
    apnea_v1 = df_v1_apnea[['subject_id', 'visit', 'apnea_class', 'ahi_score']].copy()
    
    # --- Process Visit 2 ---
    logger.info("\nProcessing Visit 2...")
    logger.info("Note: No harmonized data for Visit 2 - creating records with empty AHI")
    
    # Get all subject IDs from Visit 2 main dataset
    df_v2_apnea = df_main_v2[[subject_id_col]].copy()
    
    # Create empty apnea classification and score for Visit 2
    df_v2_apnea['apnea_class'] = ''
    df_v2_apnea['ahi_score'] = ''
    
    logger.info(f"Visit 2 - Total records (AHI will be empty): {len(df_v2_apnea)}")
    
    # Create subject_id with visit suffix
    df_v2_apnea['subject_id'] = df_v2_apnea[subject_id_col].astype(str) + '_v2'
    df_v2_apnea['visit'] = 2
    apnea_v2 = df_v2_apnea[['subject_id', 'visit', 'apnea_class', 'ahi_score']].copy()
    
    # Combine Visit 1 and Visit 2 apnea data
    apnea_targets = pd.concat([apnea_v1, apnea_v2], ignore_index=True)
    logger.info(f"\nTotal apnea records (both visits): {len(apnea_targets)}")
    logger.info(f"Valid apnea classifications (both visits): {(apnea_targets['apnea_class'] != '').sum()}")
    
    # ===================================================================
    # TASK 2: SLEEPINESS (MULTI-CLASS) - Visit 1 and Visit 2
    # ===================================================================
    
    logger.info("\n=== Task: sleepiness_class (Tier 1 - Multi-class) ===")
    
    task_config = mros_config['tasks']['sleepiness_class']
    ess_col = task_config['column']  # epepwort
    sleepiness_thresholds = config['thresholds']['sleepiness_class']['thresholds']
    sleepiness_labels = config['thresholds']['sleepiness_class']['class_labels']
    
    logger.info(f"Column: {ess_col}")
    logger.info(f"Thresholds: {sleepiness_thresholds}")
    logger.info(f"Class labels: {sleepiness_labels}")
    
    # --- Process Visit 1 ---
    logger.info("\nProcessing Visit 1 ESS...")
    df_v1_ess = df_main_v1[[subject_id_col, ess_col]].copy()
    
    # Handle missing data and convert to numeric
    df_v1_ess[ess_col] = pd.to_numeric(df_v1_ess[ess_col], errors='coerce')
    df_v1_ess[ess_col] = df_v1_ess[ess_col].replace(-9, pd.NA)
    
    # Apply multi-class threshold
    df_v1_ess['sleepiness_class'] = df_v1_ess[ess_col].apply(
        lambda x: apply_multiclass_threshold(x, sleepiness_thresholds)
    )
    
    # Log statistics for Visit 1
    valid_v1_ess = df_v1_ess['sleepiness_class'] != ''
    class_dist_v1_ess = df_v1_ess['sleepiness_class'][valid_v1_ess].value_counts().sort_index()
    logger.info(f"Visit 1 - Valid sleepiness classifications: {valid_v1_ess.sum()}")
    for class_idx, count in class_dist_v1_ess.items():
        pct = 100 * count / valid_v1_ess.sum()
        logger.info(f"  Class {class_idx} ({sleepiness_labels[int(class_idx)]}): {count} ({pct:.1f}%)")
    
    # Create records with visit suffix
    df_v1_ess['subject_id'] = df_v1_ess[subject_id_col].astype(str) + '_v1'
    df_v1_ess['visit'] = 1
    df_v1_ess.rename(columns={ess_col: 'ess_score'}, inplace=True)
    ess_v1 = df_v1_ess[['subject_id', 'visit', 'sleepiness_class', 'ess_score']].copy()
    
    # --- Process Visit 2 ---
    logger.info("\nProcessing Visit 2 ESS...")
    df_v2_ess = df_main_v2[[subject_id_col, ess_col]].copy()
    
    # Handle missing data and convert to numeric
    df_v2_ess[ess_col] = pd.to_numeric(df_v2_ess[ess_col], errors='coerce')
    df_v2_ess[ess_col] = df_v2_ess[ess_col].replace(-9, pd.NA)
    
    # Apply multi-class threshold
    df_v2_ess['sleepiness_class'] = df_v2_ess[ess_col].apply(
        lambda x: apply_multiclass_threshold(x, sleepiness_thresholds)
    )
    
    # Log statistics for Visit 2
    valid_v2_ess = df_v2_ess['sleepiness_class'] != ''
    class_dist_v2_ess = df_v2_ess['sleepiness_class'][valid_v2_ess].value_counts().sort_index()
    logger.info(f"Visit 2 - Valid sleepiness classifications: {valid_v2_ess.sum()}")
    for class_idx, count in class_dist_v2_ess.items():
        pct = 100 * count / valid_v2_ess.sum()
        logger.info(f"  Class {class_idx} ({sleepiness_labels[int(class_idx)]}): {count} ({pct:.1f}%)")
    
    # Create records with visit suffix
    df_v2_ess['subject_id'] = df_v2_ess[subject_id_col].astype(str) + '_v2'
    df_v2_ess['visit'] = 2
    df_v2_ess.rename(columns={ess_col: 'ess_score'}, inplace=True)
    ess_v2 = df_v2_ess[['subject_id', 'visit', 'sleepiness_class', 'ess_score']].copy()
    
    # Combine Visit 1 and Visit 2 ESS data
    ess_targets = pd.concat([ess_v1, ess_v2], ignore_index=True)
    logger.info(f"\nTotal ESS records (both visits): {len(ess_targets)}")
    logger.info(f"Valid ESS classifications (both visits): {(ess_targets['sleepiness_class'] != '').sum()}")
    
    # ===================================================================
    # TASK 3: INSOMNIA (BINARY) - Visit 1 and Visit 2
    # ===================================================================
    
    # Check if insomnia task is enabled
    insomnia_enabled = mros_config['tasks'].get('insomnia_binary', {}).get('enabled', False)
    
    if insomnia_enabled:
        logger.info("\n=== Task: insomnia_binary (Tier 2 - Binary) ===")
        
        task_config = mros_config['tasks']['insomnia_binary']
        isi_col = task_config['column']  # slisiscr
        insomnia_threshold = config['thresholds']['insomnia_binary']['threshold']
        
        logger.info(f"Column: {isi_col}")
        logger.info(f"Threshold: >= {insomnia_threshold}")
        
        # --- Process Visit 1 ---
        logger.info("\nProcessing Visit 1 insomnia...")
        df_v1_ins = df_main_v1[[subject_id_col, isi_col]].copy()
        
        # Handle missing data
        df_v1_ins[isi_col] = df_v1_ins[isi_col].replace(-9, pd.NA)
        
        # Apply binary threshold
        df_v1_ins['insomnia_binary'] = df_v1_ins[isi_col].apply(
            lambda x: apply_threshold(x, threshold=insomnia_threshold)
        )
        
        # Log statistics for Visit 1
        valid_v1_ins = df_v1_ins['insomnia_binary'] != ''
        ins_dist_v1 = df_v1_ins['insomnia_binary'][valid_v1_ins].value_counts().sort_index()
        logger.info(f"Visit 1 - Valid insomnia classifications: {valid_v1_ins.sum()}")
        for cls, count in ins_dist_v1.items():
            pct = 100 * count / valid_v1_ins.sum()
            label = "No insomnia" if cls == '0' else "Moderate+ insomnia"
            logger.info(f"  {label}: {count} ({pct:.1f}%)")
        
        # Create records with visit suffix
        df_v1_ins['subject_id'] = df_v1_ins[subject_id_col].astype(str) + '_v1'
        df_v1_ins['visit'] = 1
        df_v1_ins.rename(columns={isi_col: 'isi_score'}, inplace=True)
        ins_v1 = df_v1_ins[['subject_id', 'visit', 'insomnia_binary', 'isi_score']].copy()
        
        # --- Process Visit 2 ---
        logger.info("\nProcessing Visit 2 insomnia...")
        df_v2_ins = df_main_v2[[subject_id_col, isi_col]].copy()
        
        # Handle missing data
        df_v2_ins[isi_col] = df_v2_ins[isi_col].replace(-9, pd.NA)
        
        # Apply binary threshold
        df_v2_ins['insomnia_binary'] = df_v2_ins[isi_col].apply(
            lambda x: apply_threshold(x, threshold=insomnia_threshold)
        )
        
        # Log statistics for Visit 2
        valid_v2_ins = df_v2_ins['insomnia_binary'] != ''
        ins_dist_v2 = df_v2_ins['insomnia_binary'][valid_v2_ins].value_counts().sort_index()
        logger.info(f"Visit 2 - Valid insomnia classifications: {valid_v2_ins.sum()}")
        for cls, count in ins_dist_v2.items():
            pct = 100 * count / valid_v2_ins.sum()
            label = "No insomnia" if cls == '0' else "Moderate+ insomnia"
            logger.info(f"  {label}: {count} ({pct:.1f}%)")
        
        # Create records with visit suffix
        df_v2_ins['subject_id'] = df_v2_ins[subject_id_col].astype(str) + '_v2'
        df_v2_ins['visit'] = 2
        df_v2_ins.rename(columns={isi_col: 'isi_score'}, inplace=True)
        ins_v2 = df_v2_ins[['subject_id', 'visit', 'insomnia_binary', 'isi_score']].copy()
        
        # Combine Visit 1 and Visit 2 insomnia data
        insomnia_targets = pd.concat([ins_v1, ins_v2], ignore_index=True)
        logger.info(f"\nTotal insomnia records (both visits): {len(insomnia_targets)}")
        logger.info(f"Valid insomnia classifications (both visits): {(insomnia_targets['insomnia_binary'] != '').sum()}")
    else:
        logger.info("\n=== Task: insomnia_binary (SKIPPED - Disabled in config) ===")
        # Create empty insomnia targets for merging
        all_subject_ids = pd.concat([
            df_main_v1[[subject_id_col]].assign(visit=1),
            df_main_v2[[subject_id_col]].assign(visit=2)
        ])
        all_subject_ids['subject_id'] = all_subject_ids[subject_id_col].astype(str) + '_v' + all_subject_ids['visit'].astype(str)
        insomnia_targets = all_subject_ids[['subject_id', 'visit']].copy()
        insomnia_targets['insomnia_binary'] = ''
        insomnia_targets['isi_score'] = ''
    
    # ===================================================================
    # TASK 4: CVD (BINARY) - Visit 1 and Visit 2
    # ===================================================================
    
    logger.info("\n=== Task: cvd_binary (Tier 2 - Binary) ===")
    
    task_config = mros_config['tasks']['cvd_binary']
    cvd_col = task_config['column']  # cvchd
    
    logger.info(f"Column: {cvd_col}")
    logger.info(f"Note: CVD from coronary heart disease history")
    
    # --- Process Visit 1 ---
    logger.info("\nProcessing Visit 1 CVD...")
    df_v1_cvd = df_main_v1[[subject_id_col, cvd_col]].copy()
    
    # Handle missing data and convert to numeric
    df_v1_cvd[cvd_col] = pd.to_numeric(df_v1_cvd[cvd_col], errors='coerce')
    df_v1_cvd[cvd_col] = df_v1_cvd[cvd_col].replace(9, pd.NA)  # 9 = unknown
    
    # Apply binary threshold (cvchd is already 0/1, threshold at 0.5)
    df_v1_cvd['cvd_binary'] = df_v1_cvd[cvd_col].apply(
        lambda x: apply_threshold(x, threshold=0.5)
    )
    
    # Log statistics
    valid_v1_cvd = df_v1_cvd['cvd_binary'] != ''
    cvd_dist_v1 = df_v1_cvd['cvd_binary'][valid_v1_cvd].value_counts().sort_index()
    logger.info(f"Visit 1 - Valid CVD classifications: {valid_v1_cvd.sum()}")
    for cls, count in cvd_dist_v1.items():
        pct = 100 * count / valid_v1_cvd.sum()
        label = "No CVD" if cls == '0' else "CVD"
        logger.info(f"  {label}: {count} ({pct:.1f}%)")
    
    # Create records with visit suffix
    df_v1_cvd['subject_id'] = df_v1_cvd[subject_id_col].astype(str) + '_v1'
    df_v1_cvd['visit'] = 1
    cvd_v1 = df_v1_cvd[['subject_id', 'visit', 'cvd_binary']].copy()
    
    # --- Process Visit 2 ---
    logger.info("\nProcessing Visit 2 CVD...")
    df_v2_cvd = df_main_v2[[subject_id_col, cvd_col]].copy()
    
    # Handle missing data and convert to numeric
    df_v2_cvd[cvd_col] = pd.to_numeric(df_v2_cvd[cvd_col], errors='coerce')
    df_v2_cvd[cvd_col] = df_v2_cvd[cvd_col].replace(9, pd.NA)
    
    # Apply binary threshold
    df_v2_cvd['cvd_binary'] = df_v2_cvd[cvd_col].apply(
        lambda x: apply_threshold(x, threshold=0.5)
    )
    
    # Log statistics
    valid_v2_cvd = df_v2_cvd['cvd_binary'] != ''
    cvd_dist_v2 = df_v2_cvd['cvd_binary'][valid_v2_cvd].value_counts().sort_index()
    logger.info(f"Visit 2 - Valid CVD classifications: {valid_v2_cvd.sum()}")
    for cls, count in cvd_dist_v2.items():
        pct = 100 * count / valid_v2_cvd.sum()
        label = "No CVD" if cls == '0' else "CVD"
        logger.info(f"  {label}: {count} ({pct:.1f}%)")
    
    # Create records with visit suffix
    df_v2_cvd['subject_id'] = df_v2_cvd[subject_id_col].astype(str) + '_v2'
    df_v2_cvd['visit'] = 2
    cvd_v2 = df_v2_cvd[['subject_id', 'visit', 'cvd_binary']].copy()
    
    # Combine Visit 1 and Visit 2 CVD data
    cvd_targets = pd.concat([cvd_v1, cvd_v2], ignore_index=True)
    logger.info(f"\nTotal CVD records (both visits): {len(cvd_targets)}")
    logger.info(f"Valid CVD classifications (both visits): {(cvd_targets['cvd_binary'] != '').sum()}")
    
    # ===================================================================
    # TASK 5: RESTED MORNING (BINARY from 1-5 scale) - Visit 1 and Visit 2
    # ===================================================================
    
    logger.info("\n=== Task: rested_morning (Tier 3 - Binary from scale) ===")
    
    task_config = mros_config['tasks']['rested_morning']
    rested_col = task_config['column']  # poxqual3
    
    logger.info(f"Column: {rested_col}")
    logger.info(f"Scale: 1-5 (≥4 = well-rested, ≤3 = not well-rested)")
    
    # --- Process Visit 1 ---
    logger.info("\nProcessing Visit 1 rested morning...")
    df_v1_rest = df_main_v1[[subject_id_col, rested_col]].copy()
    
    # Handle missing data and convert to numeric (column may be read as string)
    df_v1_rest[rested_col] = pd.to_numeric(df_v1_rest[rested_col], errors='coerce')
    df_v1_rest[rested_col] = df_v1_rest[rested_col].replace(-9, pd.NA)
    
    # Convert scale to binary: ≥4 = 1 (well-rested), ≤3 = 0 (not well-rested)
    # All 5 answers are classified; only NaN is treated as missing.
    def rested_to_binary(val):
        if pd.isna(val):
            return ''
        if val >= 4:
            return '1'
        else:  # val <= 3 (includes neutral answer 3)
            return '0'
    
    df_v1_rest['rested_morning'] = df_v1_rest[rested_col].apply(rested_to_binary)
    
    # Log statistics
    valid_v1_rest = df_v1_rest['rested_morning'] != ''
    rest_dist_v1 = df_v1_rest['rested_morning'][valid_v1_rest].value_counts().sort_index()
    logger.info(f"Visit 1 - Valid rested morning classifications: {valid_v1_rest.sum()}")
    for cls, count in rest_dist_v1.items():
        pct = 100 * count / valid_v1_rest.sum()
        label = "Poorly rested" if cls == '0' else "Well rested"
        logger.info(f"  {label}: {count} ({pct:.1f}%)")
    
    # Create records with visit suffix
    df_v1_rest['subject_id'] = df_v1_rest[subject_id_col].astype(str) + '_v1'
    df_v1_rest['visit'] = 1
    df_v1_rest.rename(columns={rested_col: 'rested_score'}, inplace=True)
    rest_v1 = df_v1_rest[['subject_id', 'visit', 'rested_morning', 'rested_score']].copy()
    
    # --- Process Visit 2 ---
    logger.info("\nProcessing Visit 2 rested morning...")
    df_v2_rest = df_main_v2[[subject_id_col, rested_col]].copy()
    
    # Handle missing data and convert to numeric
    df_v2_rest[rested_col] = pd.to_numeric(df_v2_rest[rested_col], errors='coerce')
    df_v2_rest[rested_col] = df_v2_rest[rested_col].replace(-9, pd.NA)
    
    # Convert scale to binary
    df_v2_rest['rested_morning'] = df_v2_rest[rested_col].apply(rested_to_binary)
    
    # Log statistics
    valid_v2_rest = df_v2_rest['rested_morning'] != ''
    rest_dist_v2 = df_v2_rest['rested_morning'][valid_v2_rest].value_counts().sort_index()
    logger.info(f"Visit 2 - Valid rested morning classifications: {valid_v2_rest.sum()}")
    for cls, count in rest_dist_v2.items():
        pct = 100 * count / valid_v2_rest.sum()
        label = "Poorly rested" if cls == '0' else "Well rested"
        logger.info(f"  {label}: {count} ({pct:.1f}%)")
    
    # Create records with visit suffix
    df_v2_rest['subject_id'] = df_v2_rest[subject_id_col].astype(str) + '_v2'
    df_v2_rest['visit'] = 2
    df_v2_rest.rename(columns={rested_col: 'rested_score'}, inplace=True)
    rest_v2 = df_v2_rest[['subject_id', 'visit', 'rested_morning', 'rested_score']].copy()
    
    # Combine Visit 1 and Visit 2 rested data
    rested_targets = pd.concat([rest_v1, rest_v2], ignore_index=True)
    logger.info(f"\nTotal rested morning records (both visits): {len(rested_targets)}")
    logger.info(f"Valid rested morning classifications (both visits): {(rested_targets['rested_morning'] != '').sum()}")
    
    # ===================================================================
    # MERGE ALL TASKS
    # ===================================================================
    
    logger.info("\n=== Merging All Targets ===")
    
    # Start with apnea (has subject_id with visit suffix)
    targets = apnea_targets.copy()
    
    # Merge ESS (on subject_id and visit)
    targets = targets.merge(ess_targets, on=['subject_id', 'visit'], how='left')
    logger.info(f"After merging ESS: {len(targets)} records")
    
    # Merge insomnia (on subject_id and visit)
    targets = targets.merge(insomnia_targets, on=['subject_id', 'visit'], how='left')
    logger.info(f"After merging insomnia: {len(targets)} records")
    
    # Merge CVD (on subject_id and visit)
    targets = targets.merge(cvd_targets, on=['subject_id', 'visit'], how='left')
    logger.info(f"After merging CVD: {len(targets)} records")
    
    # Merge rested morning (on subject_id and visit)
    targets = targets.merge(rested_targets, on=['subject_id', 'visit'], how='left')
    logger.info(f"After merging rested morning: {len(targets)} records")
    
    # Add dataset column
    targets['dataset'] = dataset

    # Fill missing values with empty string
    for col in targets.columns:
        if col not in ['subject_id', 'dataset', 'visit']:
            targets[col] = targets[col].fillna('')

    logger.info(f"\n✅ Total subjects in final targets: {len(targets)}")

    # ===================================================================
    # V2 TASKS (only executed if enabled in config)
    # ===================================================================

    # --- sleep_efficiency_binary (main v1 + v2, poslpeff < 85 → 1) ---
    task_cfg = mros_config['tasks'].get('sleep_efficiency_binary', {})
    if task_cfg.get('enabled', False):
        eff_col = task_cfg['column']   # poslpeff
        eff_threshold = config['thresholds']['sleep_efficiency_binary']['threshold']
        logger.info(f"\n=== V2 Task: sleep_efficiency_binary (column: {eff_col}, threshold < {eff_threshold}%) ===")

        eff_frames = []
        for df_vis, vis, vis_label in [
            (df_main_v1, 1, 'v1'),
            (df_main_v2, 2, 'v2'),
        ]:
            if eff_col not in df_vis.columns:
                logger.warning(f"  Visit {vis}: column '{eff_col}' not found — skipping")
                continue
            tmp = df_vis[[subject_id_col, eff_col]].copy()
            tmp[eff_col] = pd.to_numeric(tmp[eff_col], errors='coerce').replace(-9, pd.NA)
            tmp['sleep_efficiency_binary'] = tmp[eff_col].apply(
                lambda x: '' if pd.isna(x) else ('1' if x < eff_threshold else '0')
            )
            tmp['eff_score'] = tmp[eff_col].astype(str).replace(['nan', '<NA>'], '')
            tmp['subject_id'] = tmp[subject_id_col].astype(str) + f'_{vis_label}'
            tmp['visit'] = vis
            valid = (tmp['sleep_efficiency_binary'] != '').sum()
            pos = (tmp['sleep_efficiency_binary'] == '1').sum()
            logger.info(f"  Visit {vis}: N={valid}, low_eff(1)={pos} ({pos/max(valid,1):.1%})")
            eff_frames.append(tmp[['subject_id', 'visit', 'sleep_efficiency_binary', 'eff_score']])

        if eff_frames:
            eff_targets = pd.concat(eff_frames, ignore_index=True)
            targets = targets.merge(eff_targets, on=['subject_id', 'visit'], how='left')
            targets['sleep_efficiency_binary'] = targets['sleep_efficiency_binary'].fillna('')
            targets['eff_score'] = targets['eff_score'].fillna('')

    # --- psqi_binary (main v1 + v2, pqpsqi > 5 → 1) ---
    task_cfg = mros_config['tasks'].get('psqi_binary', {})
    if task_cfg.get('enabled', False):
        psqi_col = task_cfg['column']   # pqpsqi
        psqi_threshold = config['thresholds']['psqi_binary']['threshold']
        logger.info(f"\n=== V2 Task: psqi_binary (column: {psqi_col}, threshold > {psqi_threshold}) ===")

        psqi_frames = []
        for df_vis, vis, vis_label in [
            (df_main_v1, 1, 'v1'),
            (df_main_v2, 2, 'v2'),
        ]:
            if psqi_col not in df_vis.columns:
                logger.warning(f"  Visit {vis}: column '{psqi_col}' not found — skipping")
                continue
            tmp = df_vis[[subject_id_col, psqi_col]].copy()
            tmp[psqi_col] = pd.to_numeric(tmp[psqi_col], errors='coerce').replace(-9, pd.NA)
            tmp['psqi_binary'] = tmp[psqi_col].apply(
                lambda x: '' if pd.isna(x) else ('1' if x > psqi_threshold else '0')
            )
            tmp['psqi_score'] = tmp[psqi_col].astype(str).replace(['nan', '<NA>'], '')
            tmp['subject_id'] = tmp[subject_id_col].astype(str) + f'_{vis_label}'
            tmp['visit'] = vis
            valid = (tmp['psqi_binary'] != '').sum()
            pos = (tmp['psqi_binary'] == '1').sum()
            logger.info(f"  Visit {vis}: N={valid}, poor_sleep(1)={pos} ({pos/max(valid,1):.1%})")
            psqi_frames.append(tmp[['subject_id', 'visit', 'psqi_binary', 'psqi_score']])

        if psqi_frames:
            psqi_targets = pd.concat(psqi_frames, ignore_index=True)
            targets = targets.merge(psqi_targets, on=['subject_id', 'visit'], how='left')
            targets['psqi_binary'] = targets['psqi_binary'].fillna('')
            targets['psqi_score'] = targets['psqi_score'].fillna('')

    # --- age_regression (harmonized v1 only — visit 2 left empty) ---
    task_cfg = mros_config['tasks'].get('age_regression', {})
    if task_cfg.get('enabled', False):
        age_col = task_cfg['column']
        logger.info(f"\n=== V2 Task: age_regression (column: {age_col}, visit 1 only) ===")

        df_harm_v1_age = df_harm_v1[[subject_id_col, age_col]].copy()
        df_harm_v1_age[age_col] = pd.to_numeric(df_harm_v1_age[age_col], errors='coerce').replace(-9, pd.NA)
        df_harm_v1_age['age_value'] = df_harm_v1_age[age_col].astype(str).replace(['nan', '<NA>'], '')
        logger.info(f"  N={( df_harm_v1_age['age_value'] != '').sum()} subjects with valid age (visit 1 only; visit 2 will be empty)")

        age_targets = df_harm_v1_age.copy()
        age_targets['subject_id'] = age_targets[subject_id_col].astype(str) + '_v1'
        age_targets['visit'] = 1
        age_targets = age_targets[['subject_id', 'visit', 'age_value']]
        targets = targets.merge(age_targets, on=['subject_id', 'visit'], how='left')
        targets['age_value'] = targets['age_value'].fillna('')

    # --- bmi_regression (harmonized v1 only — visit 2 left empty) ---
    task_cfg = mros_config['tasks'].get('bmi_regression', {})
    if task_cfg.get('enabled', False):
        bmi_col = task_cfg['column']
        logger.info(f"\n=== V2 Task: bmi_regression (column: {bmi_col}, visit 1 only) ===")

        df_harm_v1_bmi = df_harm_v1[[subject_id_col, bmi_col]].copy()
        df_harm_v1_bmi[bmi_col] = pd.to_numeric(df_harm_v1_bmi[bmi_col], errors='coerce').replace(-9, pd.NA)
        df_harm_v1_bmi['bmi_value'] = df_harm_v1_bmi[bmi_col].astype(str).replace(['nan', '<NA>'], '')
        logger.info(f"  N={( df_harm_v1_bmi['bmi_value'] != '').sum()} subjects with valid BMI (visit 1 only; visit 2 will be empty)")

        bmi_targets = df_harm_v1_bmi.copy()
        bmi_targets['subject_id'] = bmi_targets[subject_id_col].astype(str) + '_v1'
        bmi_targets['visit'] = 1
        bmi_targets = bmi_targets[['subject_id', 'visit', 'bmi_value']]
        targets = targets.merge(bmi_targets, on=['subject_id', 'visit'], how='left')
        targets['bmi_value'] = targets['bmi_value'].fillna('')

    # --- age_class (3-class derived from age_value already in targets) ---
    task_cfg = mros_config['tasks'].get('age_class', {})
    if task_cfg.get('enabled', False):
        age_thresholds = config['thresholds']['age_class']['thresholds']
        logger.info(f"\n=== V2 Task: age_class (thresholds: {age_thresholds}) ===")
        logger.info("  Note: MrOS is 65+ — expect all subjects in class 2")

        if 'age_value' not in targets.columns:
            logger.warning("  age_value not in targets — age_regression must be enabled. Skipping.")
        else:
            def _age_to_class(val_str, thresh):
                if val_str == '':
                    return ''
                try:
                    age = float(val_str)
                    if age < thresh[0]:
                        return '0'
                    elif age < thresh[1]:
                        return '1'
                    else:
                        return '2'
                except ValueError:
                    return ''

            targets['age_class'] = targets['age_value'].apply(
                lambda x: _age_to_class(x, age_thresholds)
            )
            for vis in [1, 2]:
                sub = targets[targets['visit'] == vis]
                dist = sub['age_class'][sub['age_class'] != ''].value_counts().sort_index()
                logger.info(f"  Visit {vis}: class_dist={dict(dist)}")

    # --- bmi_binary (derived from bmi_value already in targets) ---
    task_cfg = mros_config['tasks'].get('bmi_binary', {})
    if task_cfg.get('enabled', False):
        bmi_threshold = config['thresholds']['bmi_binary']['threshold']
        logger.info(f"\n=== V2 Task: bmi_binary (threshold >= {bmi_threshold}) ===")

        if 'bmi_value' not in targets.columns:
            logger.warning("  bmi_value not in targets — bmi_regression must be enabled. Skipping.")
        else:
            def _bmi_to_binary(val_str, thresh):
                if val_str == '':
                    return ''
                try:
                    return '1' if float(val_str) >= thresh else '0'
                except ValueError:
                    return ''

            targets['bmi_binary'] = targets['bmi_value'].apply(
                lambda x: _bmi_to_binary(x, bmi_threshold)
            )
            for vis in [1, 2]:
                sub = targets[targets['visit'] == vis]
                valid = sub['bmi_binary'][sub['bmi_binary'] != '']
                pos = (valid == '1').sum()
                logger.info(f"  Visit {vis}: N={len(valid)}, obese(1)={pos} ({pos/max(len(valid),1):.1%})")
    
    # ===================================================================
    # COMPUTE STATISTICS
    # ===================================================================
    
    logger.info("\n=== Target Statistics ===")
    
    # Define task columns and types
    task_columns = {
        'apnea_class': True,  # multi-class
        'sleepiness_class': True,  # multi-class
        'insomnia_binary': False,  # binary
        'cvd_binary': False,  # binary
        'rested_morning': False,  # binary
    }
    
    is_multiclass = {col: is_multi for col, is_multi in task_columns.items()}
    
    stats = compute_task_statistics(
        targets,
        list(task_columns.keys()),
        dataset,
        is_multiclass=is_multiclass
    )
    
    # Define score columns mapping
    score_mapping = {
        'apnea': 'ahi_score',
        'sleepiness': 'ess_score',
        'insomnia': 'isi_score',
        'rested': 'rested_score'
    }
    
    return targets


def main():
    parser = argparse.ArgumentParser(
        description="Extract classification targets for MrOS dataset"
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
    log_file = log_dir / "extract_mros.log"
    setup_logging(log_file)
    
    logger.info("="*80)
    logger.info("MrOS Target Extraction Script")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Log file: {log_file}")
    
    # Extract targets
    try:
        targets_df = extract_mros_targets(config)
        
        # Save results
        output_path = args.output or (log_dir / "mros_targets.csv")
        
        # Build column order dynamically (v2 columns only included if present)
        _desired = [
            'subject_id', 'dataset', 'visit',
            'apnea_class', 'ahi_score',
            'sleepiness_class', 'ess_score',
            'insomnia_binary', 'isi_score',
            'cvd_binary',
            'rested_morning', 'rested_score',
            'sleep_efficiency_binary', 'eff_score',
            'psqi_binary', 'psqi_score',
            'age_value', 'age_class',
            'bmi_value', 'bmi_binary',
        ]
        column_order = [c for c in _desired if c in targets_df.columns]
        save_dataset_targets(targets_df, output_path, 'mros', column_order)
        
        logger.info("\n" + "="*80)
        logger.info("✅ MrOS extraction completed successfully!")
        logger.info(f"Output file: {output_path}")
        logger.info(f"Total subjects: {len(targets_df)}")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error during extraction: {e}")
        logger.exception(e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
