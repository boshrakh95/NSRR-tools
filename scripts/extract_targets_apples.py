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
from nsrr_tools.utils.mount_utils import ensure_sshfs_mounted


def setup_logging(log_file: Path) -> None:
    """Configure logging."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, level="DEBUG", mode="w")


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

    # ===================================================================
    # V2 TASKS (only executed if enabled in config)
    # ===================================================================

    apples_cfg = config['tasks'][dataset]
    visit_col = 'visitn'  # common visit column in both main and harmonized files

    # Convenience: visit-1 slices (demographics are static — only populated at v1)
    df_harm_v1 = df_harmonized[df_harmonized[visit_col] == 1].copy()
    df_dx_harm = df_harmonized[df_harmonized[visit_col] == 3].copy()   # DX visit
    df_bl_main = df_main[df_main[visit_col] == 1].copy()               # baseline

    def _merge_left_on_nsrrid(targets_df, extra_df, extra_col):
        """Left-join extra_df (keyed on nsrrid renamed to subject_id) into targets."""
        extra_df = extra_df[['nsrrid', extra_col]].copy()
        extra_df = extra_df.rename(columns={'nsrrid': 'subject_id'})
        extra_df['subject_id'] = extra_df['subject_id'].astype(str)
        return targets_df.merge(extra_df, on='subject_id', how='left')

    def _merge_left_on_appleid(targets_df, extra_df, cols):
        """Left-join extra_df (keyed on appleid renamed to subject_id) into targets."""
        extra_df = extra_df[[subject_id_col] + cols].copy()
        extra_df = extra_df.rename(columns={subject_id_col: 'subject_id'})
        extra_df['subject_id'] = extra_df['subject_id'].astype(str)
        return targets_df.merge(extra_df, on='subject_id', how='left')

    # --- sleep_efficiency_binary (harmonized DX visit) ---
    task_cfg = apples_cfg.get('sleep_efficiency_binary', {})
    if task_cfg.get('enabled', False):
        eff_col = task_cfg['column']
        eff_threshold = config['thresholds']['sleep_efficiency_binary']['threshold']
        logger.info(f"\n=== V2 Task: sleep_efficiency_binary (threshold < {eff_threshold}%) ===")

        df_dx_harm[eff_col] = pd.to_numeric(df_dx_harm[eff_col], errors='coerce').replace(-9, pd.NA)
        df_dx_harm['sleep_efficiency_binary'] = df_dx_harm[eff_col].apply(
            lambda x: '' if pd.isna(x) else ('1' if x < eff_threshold else '0')
        )
        df_dx_harm['eff_score'] = df_dx_harm[eff_col].astype(str).replace(['nan', '<NA>'], '')
        targets = _merge_left_on_nsrrid(targets, df_dx_harm, 'sleep_efficiency_binary')
        targets = _merge_left_on_nsrrid(targets, df_dx_harm, 'eff_score')
        targets['sleep_efficiency_binary'] = targets['sleep_efficiency_binary'].fillna('')
        targets['eff_score'] = targets['eff_score'].fillna('')
        valid = (targets['sleep_efficiency_binary'] != '').sum()
        pos = (targets['sleep_efficiency_binary'] == '1').sum()
        logger.info(f"  N={valid}, low_eff(1)={pos} ({pos/max(valid,1):.1%})")

    # --- osa_severity_apples (main baseline visit, string parse) ---
    task_cfg = apples_cfg.get('osa_severity_apples', {})
    if task_cfg.get('enabled', False):
        osa_col = task_cfg['column']
        logger.info(f"\n=== V2 Task: osa_severity_apples (column: {osa_col}) ===")

        def _parse_osa_severity(val):
            """Extract integer prefix from strings like '1) mild'."""
            if pd.isna(val) or str(val).strip() == '':
                return ''
            try:
                return str(int(str(val).split(')')[0].strip()))
            except (ValueError, IndexError):
                return ''

        def _osa_to_binary(cls_str):
            """Non-rand(0)+Mild(1) → 0, Moderate(2)+Severe(3) → 1."""
            if cls_str == '':
                return ''
            try:
                c = int(cls_str)
                return '1' if c >= 2 else '0'
            except ValueError:
                return ''

        df_bl_main['osa_severity_apples'] = df_bl_main[osa_col].apply(_parse_osa_severity)
        df_bl_main['osa_binary_apples_postqc'] = df_bl_main['osa_severity_apples'].apply(_osa_to_binary)
        targets = _merge_left_on_appleid(
            targets, df_bl_main, ['osa_severity_apples', 'osa_binary_apples_postqc']
        )
        targets['osa_severity_apples'] = targets['osa_severity_apples'].fillna('')
        targets['osa_binary_apples_postqc'] = targets['osa_binary_apples_postqc'].fillna('')
        dist = targets['osa_severity_apples'][targets['osa_severity_apples'] != ''].value_counts().sort_index()
        pos = (targets['osa_binary_apples_postqc'] == '1').sum()
        neg = (targets['osa_binary_apples_postqc'] == '0').sum()
        logger.info(f"  4-class distribution: {dict(dist)}")
        logger.info(f"  binary (mod+sev=1): pos={pos}, neg={neg}")

    # --- depression_extreme_binary (main baseline, BDI ≤9 vs ≥20) ---
    task_cfg = apples_cfg.get('depression_extreme_binary', {})
    if task_cfg.get('enabled', False):
        bdi_col = task_cfg['column']
        ext_cfg = config['thresholds']['depression_extreme_binary']['apples']
        low_max = ext_cfg['low_max']
        high_min = ext_cfg['high_min']
        logger.info(f"\n=== V2 Task: depression_extreme_binary (BDI ≤{low_max}=0, ≥{high_min}=1) ===")

        def _extreme_binary(val, lmax, hmin):
            if pd.isna(val):
                return ''
            v = float(val)
            if v <= lmax:
                return '0'
            if v >= hmin:
                return '1'
            return ''   # middle group — dropped

        df_bl_main[bdi_col] = pd.to_numeric(df_bl_main[bdi_col], errors='coerce').replace(-9, pd.NA)
        df_bl_main['depression_extreme_binary'] = df_bl_main[bdi_col].apply(
            lambda x: _extreme_binary(x, low_max, high_min)
        )
        targets = _merge_left_on_appleid(targets, df_bl_main, ['depression_extreme_binary'])
        targets['depression_extreme_binary'] = targets['depression_extreme_binary'].fillna('')
        valid = (targets['depression_extreme_binary'] != '').sum()
        pos = (targets['depression_extreme_binary'] == '1').sum()
        neg = (targets['depression_extreme_binary'] == '0').sum()
        drop = len(targets) - valid
        logger.info(f"  N={valid} (dropped middle={drop}), pos(1)={pos}, neg(0)={neg}")

    # --- sex_binary (harmonized visit 1) ---
    task_cfg = apples_cfg.get('sex_binary', {})
    if task_cfg.get('enabled', False):
        sex_col = task_cfg['column']
        logger.info(f"\n=== V2 Task: sex_binary (column: {sex_col}) ===")

        def _sex_to_binary(val):
            v = str(val).strip().lower() if not pd.isna(val) else ''
            return '1' if v == 'female' else ('0' if v == 'male' else '')

        df_harm_v1['sex_binary'] = df_harm_v1[sex_col].apply(_sex_to_binary)
        targets = _merge_left_on_nsrrid(targets, df_harm_v1, 'sex_binary')
        targets['sex_binary'] = targets['sex_binary'].fillna('')
        valid = (targets['sex_binary'] != '').sum()
        pos = (targets['sex_binary'] == '1').sum()
        logger.info(f"  N={valid}, female(1)={pos} ({pos/max(valid,1):.1%})")

    # --- age_regression (harmonized visit 1) ---
    task_cfg = apples_cfg.get('age_regression', {})
    if task_cfg.get('enabled', False):
        age_col = task_cfg['column']
        logger.info(f"\n=== V2 Task: age_regression (column: {age_col}) ===")
        df_harm_v1[age_col] = pd.to_numeric(df_harm_v1[age_col], errors='coerce').replace(-9, pd.NA)
        df_harm_v1['age_value'] = df_harm_v1[age_col].astype(str).replace(['nan', '<NA>'], '')
        targets = _merge_left_on_nsrrid(targets, df_harm_v1, 'age_value')
        targets['age_value'] = targets['age_value'].fillna('')
        valid = (targets['age_value'] != '').sum()
        logger.info(f"  N={valid} with valid age")

    # --- bmi_regression (harmonized visit 1) ---
    task_cfg = apples_cfg.get('bmi_regression', {})
    if task_cfg.get('enabled', False):
        bmi_col = task_cfg['column']
        logger.info(f"\n=== V2 Task: bmi_regression (column: {bmi_col}) ===")
        df_harm_v1[bmi_col] = pd.to_numeric(df_harm_v1[bmi_col], errors='coerce').replace(-9, pd.NA)
        df_harm_v1['bmi_value'] = df_harm_v1[bmi_col].astype(str).replace(['nan', '<NA>'], '')
        targets = _merge_left_on_nsrrid(targets, df_harm_v1, 'bmi_value')
        targets['bmi_value'] = targets['bmi_value'].fillna('')
        valid = (targets['bmi_value'] != '').sum()
        logger.info(f"  N={valid} with valid BMI")

    # --- age_class (3-class derived from age_value already in targets) ---
    task_cfg = apples_cfg.get('age_class', {})
    if task_cfg.get('enabled', False):
        age_thresholds = config['thresholds']['age_class']['thresholds']
        logger.info(f"\n=== V2 Task: age_class (thresholds: {age_thresholds}) ===")

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
            dist = targets['age_class'][targets['age_class'] != ''].value_counts().sort_index()
            logger.info(f"  Class distribution: {dict(dist)}")

    # --- bmi_binary (derived from bmi_value already in targets) ---
    task_cfg = apples_cfg.get('bmi_binary', {})
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
            valid = targets['bmi_binary'][targets['bmi_binary'] != '']
            pos = (valid == '1').sum()
            logger.info(f"  N={len(valid)}, obese(1)={pos} ({pos/max(len(valid),1):.1%})")
    
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
        
        # Build column order dynamically (v2 columns only included if present)
        _desired = [
            'subject_id', 'dataset', 'visit',
            'apnea_class', 'ahi_score',
            'depression_class', 'bdi_score',
            'sleepiness_class', 'ess_score',
            'sleep_efficiency_binary', 'eff_score',
            'osa_severity_apples',
            'osa_binary_apples_postqc',
            'depression_extreme_binary',
            'sex_binary',
            'age_value', 'age_class',
            'bmi_value', 'bmi_binary',
        ]
        columns_order = [c for c in _desired if c in targets_df.columns]
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
