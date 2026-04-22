#!/usr/bin/env python3
"""
Extract Classification Targets for STAGES Dataset

STAGES is a single-visit dataset. Subject ID column is `subject_code`
(NOT nsrrid — see STAGES_DATA_NOTES.md and stages_adapter.py).

Extracts:
- apnea_class        : AHI 4-class (STAGESPSGKeySRBDVariables XLSX) [<5, 5-15, 15-30, >=30]
- depression_binary  : PHQ-9 (phq_1000) >= 10
- anxiety_binary     : GAD-7 (gad_0800) >= 10
- insomnia_binary    : ISI  (isi_score)  >= 15  [confirmed in main CSV]
- sleepiness_binary  : ESS  (ess_0900)   >= 11  [confirmed in main CSV]
- fatigue_binary     : FSS  (fss_1000)   >= 36  [disabled by default]

Usage:
    python scripts/extract_targets_stages.py --config configs/target_extraction.yaml
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nsrr_tools.targets.extraction_utils import (
    apply_multiclass_threshold,
    apply_threshold,
    compute_task_statistics,
    load_config_file,
    save_dataset_targets,
    validate_score_range,
)
from nsrr_tools.utils.mount_utils import ensure_sshfs_mounted


def setup_logging(log_file: Path) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, level="DEBUG", mode="w")


def extract_stages_targets(config: dict) -> pd.DataFrame:
    """
    Extract all targets for STAGES dataset.

    STAGES specifics:
    - Single visit per subject (visit column = 0)
    - Subject ID column: subject_code (not nsrrid)
    - Questionnaire columns in stages-dataset-0.3.0.csv
    - AHI from STAGESPSGKeySRBDVariables XLSX (merged on subject_code)
    """
    dataset = 'stages'
    stages_config = config['tasks'][dataset]
    data_dir = Path(config['paths']['raw_data']) / dataset / 'datasets'

    subject_id_col = stages_config['subject_id_column']  # subject_code

    logger.info("=" * 80)
    logger.info("STAGES Target Extraction")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Subject ID column: {subject_id_col}")
    logger.info("Note: STAGES is single-visit; visit column will be 0")

    # ===================================================================
    # LOAD MAIN CSV
    # ===================================================================
    main_file = data_dir / 'stages-dataset-0.3.0.csv'
    logger.info(f"\nLoading main dataset: {main_file}")
    if not main_file.exists():
        raise FileNotFoundError(f"Main STAGES CSV not found: {main_file}")
    df = pd.read_csv(main_file, low_memory=False)
    logger.info(f"Loaded {len(df)} records")

    # Verify subject ID column exists
    if subject_id_col not in df.columns:
        raise ValueError(
            f"Expected subject_id_col '{subject_id_col}' not in STAGES CSV. "
            f"Available columns (first 20): {list(df.columns[:20])}"
        )

    # STAGES CSVs can have duplicate subject_codes — keep last (most complete)
    n_before = len(df)
    df = df.drop_duplicates(subset=[subject_id_col], keep='last')
    if len(df) < n_before:
        logger.info(f"Removed {n_before - len(df)} duplicate subject_codes (kept last)")
    logger.info(f"Subject IDs after dedup: {df[subject_id_col].notna().sum()} non-null")

    # Replace -9 sentinel with NaN across all numeric columns (NSRR convention)
    df = df.replace(-9, pd.NA)

    # ===================================================================
    # TASK 1: APNEA CLASS (4-class) — AHI from PSG key XLSX
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Task: apnea_class (AHI: <5, 5-15, 15-30, >=30)")
    logger.info("=" * 60)

    apnea_cfg = stages_config['apnea_class']
    ahi_col = apnea_cfg['column']               # ahi
    xlsx_id_col = apnea_cfg.get('xlsx_subject_id_column', subject_id_col)
    psg_key_file = data_dir / apnea_cfg['source_file']

    logger.info(f"PSG key file: {psg_key_file}")
    logger.info(f"AHI column: {ahi_col},  subject ID column in XLSX: {xlsx_id_col}")

    if not psg_key_file.exists():
        raise FileNotFoundError(
            f"PSG key XLSX not found: {psg_key_file}\n"
            f"Expected: {apnea_cfg['source_file']} in {data_dir}"
        )

    df_psg = pd.read_excel(psg_key_file)
    logger.info(f"PSG key loaded: {len(df_psg)} rows, columns: {list(df_psg.columns[:15])}")

    if xlsx_id_col not in df_psg.columns:
        raise ValueError(
            f"Subject ID column '{xlsx_id_col}' not found in PSG key XLSX. "
            f"Available columns: {list(df_psg.columns)}"
        )
    if ahi_col not in df_psg.columns:
        raise ValueError(
            f"AHI column '{ahi_col}' not found in PSG key XLSX. "
            f"Available columns: {list(df_psg.columns)}"
        )

    # Deduplicate PSG key on subject ID
    df_psg = df_psg.drop_duplicates(subset=[xlsx_id_col], keep='last')

    # Merge AHI into main df (left join: keep all subjects, NaN for those without PSG)
    df_psg_slim = df_psg[[xlsx_id_col, ahi_col]].copy()
    if xlsx_id_col != subject_id_col:
        df_psg_slim = df_psg_slim.rename(columns={xlsx_id_col: subject_id_col})
    df = df.merge(df_psg_slim, on=subject_id_col, how='left')
    logger.info(f"After merge: {df[ahi_col].notna().sum()} subjects have AHI")

    df[ahi_col] = pd.to_numeric(df[ahi_col], errors='coerce')
    validate_score_range(df, ahi_col, config['validation']['apnea_binary']['ahi_range'],
                         dataset, 'apnea_class')

    apnea_thresholds = config['thresholds']['apnea_class']['thresholds']  # [5, 15, 30]
    df['apnea_class'] = df[ahi_col].apply(
        lambda x: apply_multiclass_threshold(x, apnea_thresholds)
    )
    df['ahi_score'] = df[ahi_col].astype(str).replace(['nan', '<NA>'], '')

    class_dist = df['apnea_class'][df['apnea_class'] != ''].value_counts().sort_index()
    logger.info(f"  Class distribution: {dict(class_dist)}")
    for cls, count in class_dist.items():
        cls_name = config['thresholds']['apnea_class']['class_labels'][int(cls)]
        logger.info(f"    Class {cls} ({cls_name}): {count}")
    n_miss = (df['apnea_class'] == '').sum()
    logger.info(f"  Missing AHI: {n_miss}")

    # ===================================================================
    # TASK 2: DEPRESSION (BINARY) — PHQ-9 >= 10
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Task: depression_binary (PHQ-9 >= 10)")
    logger.info("=" * 60)

    task_cfg = stages_config['depression_binary']
    phq_col = task_cfg['column']      # phq_1000
    dep_threshold = task_cfg['threshold_override']  # 10.0

    logger.info(f"Column: {phq_col},  threshold: >= {dep_threshold}")

    if phq_col not in df.columns:
        logger.warning(f"Column '{phq_col}' not found — depression_binary will be empty")
        df['depression_binary'] = ''
        df['phq9_score'] = ''
    else:
        df[phq_col] = pd.to_numeric(df[phq_col], errors='coerce')
        validate_score_range(df, phq_col, config['validation']['depression_binary']['phq9_range'],
                             dataset, 'depression_binary')
        df['depression_binary'] = df[phq_col].apply(
            lambda x: apply_threshold(x, dep_threshold)
        )
        df['phq9_score'] = df[phq_col].astype(str).replace(['nan', '<NA>'], '')
        _log_binary_dist(df, 'depression_binary', "No depression / Depression")

    # ===================================================================
    # TASK 3: ANXIETY (BINARY) — GAD-7 >= 10
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Task: anxiety_binary (GAD-7 >= 10)")
    logger.info("=" * 60)

    task_cfg = stages_config['anxiety_binary']
    gad_col = task_cfg['column']      # gad_0800
    anx_threshold = config['thresholds']['anxiety_binary']['threshold']  # 10.0

    logger.info(f"Column: {gad_col},  threshold: >= {anx_threshold}")

    if gad_col not in df.columns:
        logger.warning(f"Column '{gad_col}' not found — anxiety_binary will be empty")
        df['anxiety_binary'] = ''
        df['gad7_score'] = ''
    else:
        df[gad_col] = pd.to_numeric(df[gad_col], errors='coerce')
        validate_score_range(df, gad_col, config['validation']['anxiety_binary']['gad7_range'],
                             dataset, 'anxiety_binary')
        df['anxiety_binary'] = df[gad_col].apply(
            lambda x: apply_threshold(x, anx_threshold)
        )
        df['gad7_score'] = df[gad_col].astype(str).replace(['nan', '<NA>'], '')
        _log_binary_dist(df, 'anxiety_binary', "No anxiety / Anxiety")

    # ===================================================================
    # TASK 4: INSOMNIA (BINARY) — ISI >= 15
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Task: insomnia_binary (ISI >= 15)")
    logger.info("=" * 60)

    task_cfg = stages_config['insomnia_binary']
    ins_col = task_cfg['column']       # isi_score  (confirmed in main CSV)
    ins_threshold = config['thresholds']['insomnia_binary']['threshold']  # 15.0

    logger.info(f"Column: {ins_col},  threshold: >= {ins_threshold}")
    logger.info(f"Source: {task_cfg['source_file']} (main CSV, not external XLSX)")

    if ins_col not in df.columns:
        logger.warning(f"Column '{ins_col}' not found — insomnia_binary will be empty")
        df['insomnia_binary'] = ''
        df['isi_score'] = ''
    else:
        df[ins_col] = pd.to_numeric(df[ins_col], errors='coerce')
        validate_score_range(df, ins_col, config['validation']['insomnia_binary']['isi_range'],
                             dataset, 'insomnia_binary')
        df['insomnia_binary'] = df[ins_col].apply(
            lambda x: apply_threshold(x, ins_threshold)
        )
        df['isi_score'] = df[ins_col].astype(str).replace(['nan', '<NA>'], '')
        _log_binary_dist(df, 'insomnia_binary', "No insomnia / Insomnia")

    # ===================================================================
    # TASK 5: SLEEPINESS (BINARY) — ESS >= 11
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Task: sleepiness_binary (ESS >= 11)")
    logger.info("=" * 60)

    task_cfg = stages_config['sleepiness_binary']
    ess_col = task_cfg['column']       # ess_0900
    sle_threshold = task_cfg['threshold_override']  # 11.0

    logger.info(f"Column: {ess_col},  threshold: >= {sle_threshold}")

    if ess_col not in df.columns:
        logger.warning(f"Column '{ess_col}' not found — sleepiness_binary will be empty")
        df['sleepiness_binary'] = ''
        df['ess_score'] = ''
    else:
        df[ess_col] = pd.to_numeric(df[ess_col], errors='coerce')
        validate_score_range(df, ess_col, config['validation']['sleepiness_binary']['ess_range'],
                             dataset, 'sleepiness_binary')
        df['sleepiness_binary'] = df[ess_col].apply(
            lambda x: apply_threshold(x, sle_threshold)
        )
        df['ess_score'] = df[ess_col].astype(str).replace(['nan', '<NA>'], '')
        _log_binary_dist(df, 'sleepiness_binary', "Normal sleepiness / Excessive sleepiness")

    # ===================================================================
    # TASK 6: FATIGUE (BINARY) — FSS >= 36 — disabled by default
    # ===================================================================
    task_cfg = stages_config.get('fatigue_binary', {})
    fatigue_enabled = task_cfg.get('enabled', False)

    if fatigue_enabled:
        logger.info("\n" + "=" * 60)
        logger.info("Task: fatigue_binary (FSS >= 36)")
        logger.info("=" * 60)
        fss_col = task_cfg['column']   # fss_1000
        fat_threshold = config['thresholds']['fatigue_binary']['threshold']  # 36.0
        logger.info(f"Column: {fss_col},  threshold: >= {fat_threshold}")

        if fss_col not in df.columns:
            logger.warning(f"Column '{fss_col}' not found — fatigue_binary will be empty")
            df['fatigue_binary'] = ''
            df['fss_score'] = ''
        else:
            df[fss_col] = pd.to_numeric(df[fss_col], errors='coerce')
            validate_score_range(df, fss_col, config['validation']['fatigue_binary']['fss_range'],
                                 dataset, 'fatigue_binary')
            df['fatigue_binary'] = df[fss_col].apply(
                lambda x: apply_threshold(x, fat_threshold)
            )
            df['fss_score'] = df[fss_col].astype(str).replace(['nan', '<NA>'], '')
            _log_binary_dist(df, 'fatigue_binary', "No fatigue / Fatigue")
    else:
        logger.info("\n=== Task: fatigue_binary (SKIPPED — disabled in config) ===")

    # ===================================================================
    # ASSEMBLE OUTPUT
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Assembling output")
    logger.info("=" * 60)

    # Rename subject_code to subject_id for schema compatibility
    targets = df[[subject_id_col]].copy()
    targets.rename(columns={subject_id_col: 'subject_id'}, inplace=True)
    targets['dataset'] = dataset
    targets['visit'] = 0  # single-visit dataset

    # Active tasks only — no empty placeholder columns
    active_cols = ['apnea_class', 'ahi_score',
                   'depression_binary', 'phq9_score',
                   'sleepiness_binary', 'ess_score',
                   'anxiety_binary', 'gad7_score',
                   'insomnia_binary', 'isi_score']
    if fatigue_enabled:
        active_cols += ['fatigue_binary', 'fss_score']

    for col in active_cols:
        targets[col] = df[col].values if col in df.columns else ''

    # Fill any remaining NaN with empty string
    for col in targets.columns:
        if col not in ['subject_id', 'dataset', 'visit']:
            targets[col] = targets[col].fillna('')

    logger.info(f"Total subjects in output: {len(targets)}")

    # ===================================================================
    # V2 TASKS (only executed if enabled in config)
    # ===================================================================

    # --- depression_extreme_binary (main CSV, PHQ-9 ≤4=0 vs ≥15=1) ---
    task_cfg = stages_config.get('depression_extreme_binary', {})
    if task_cfg.get('enabled', False):
        phq_col = task_cfg['column']   # phq_1000
        ext_cfg = config['thresholds']['depression_extreme_binary']['stages']
        low_max = ext_cfg['low_max']
        high_min = ext_cfg['high_min']
        logger.info(f"\n=== V2 Task: depression_extreme_binary (PHQ-9 ≤{low_max}=0, ≥{high_min}=1) ===")

        def _extreme_binary(val, lmax, hmin):
            if pd.isna(val):
                return ''
            v = float(val)
            if v <= lmax:
                return '0'
            if v >= hmin:
                return '1'
            return ''   # middle group — dropped

        phq_series = pd.to_numeric(df[phq_col], errors='coerce') if phq_col in df.columns else pd.Series(
            pd.NA, index=df.index
        )
        targets['depression_extreme_binary'] = phq_series.apply(
            lambda x: _extreme_binary(x, low_max, high_min)
        ).values
        valid = (targets['depression_extreme_binary'] != '').sum()
        pos = (targets['depression_extreme_binary'] == '1').sum()
        neg = (targets['depression_extreme_binary'] == '0').sum()
        drop = len(targets) - valid
        logger.info(f"  N={valid} (dropped middle={drop}), pos(1)={pos}, neg(0)={neg}")

    # --- Load harmonized file for demographics (sex, age, bmi) ---
    _need_harmonized = any(
        stages_config.get(t, {}).get('enabled', False)
        for t in ['sex_binary', 'age_regression', 'bmi_regression']
    )
    df_harm = None
    if _need_harmonized:
        harm_fname = stages_config.get(
            'sex_binary',
            stages_config.get('age_regression', stages_config.get('bmi_regression', {}))
        ).get('source_file', 'stages-harmonized-dataset-0.3.0.csv')
        harm_id_col = stages_config.get(
            'sex_binary',
            stages_config.get('age_regression', {'subject_id_column': subject_id_col})
        ).get('subject_id_column', subject_id_col)
        harm_file = data_dir / harm_fname
        logger.info(f"\nLoading harmonized file for demographics: {harm_file}")
        if harm_file.exists():
            df_harm = pd.read_csv(harm_file, low_memory=False)
            df_harm = df_harm.replace(-9, pd.NA)
            logger.info(f"  Loaded {len(df_harm)} rows")
            # deduplicate on subject ID
            if harm_id_col in df_harm.columns:
                df_harm = df_harm.drop_duplicates(subset=[harm_id_col], keep='last')
        else:
            logger.warning(f"  Harmonized file not found: {harm_file} — demographics tasks will be empty")

    def _merge_harmonized(targets_df, df_harm_local, harm_id, new_col):
        """Left-join a single column from harmonized into targets on subject_id."""
        if df_harm_local is None or new_col not in df_harm_local.columns:
            targets_df[new_col] = ''
            return targets_df
        slim = df_harm_local[[harm_id, new_col]].copy()
        slim = slim.rename(columns={harm_id: 'subject_id'})
        slim['subject_id'] = slim['subject_id'].astype(str)
        return targets_df.merge(slim, on='subject_id', how='left')

    # --- sex_binary ---
    task_cfg = stages_config.get('sex_binary', {})
    if task_cfg.get('enabled', False):
        sex_col = task_cfg['column']
        harm_id_col = task_cfg.get('subject_id_column', subject_id_col)
        logger.info(f"\n=== V2 Task: sex_binary (column: {sex_col}) ===")

        def _sex_to_binary(val):
            v = str(val).strip().lower() if not pd.isna(val) else ''
            return '1' if v == 'female' else ('0' if v == 'male' else '')

        if df_harm is not None and sex_col in df_harm.columns:
            df_harm['sex_binary'] = df_harm[sex_col].apply(_sex_to_binary)
            targets = _merge_harmonized(targets, df_harm, harm_id_col, 'sex_binary')
            targets['sex_binary'] = targets['sex_binary'].fillna('')
            valid = (targets['sex_binary'] != '').sum()
            pos = (targets['sex_binary'] == '1').sum()
            logger.info(f"  N={valid}, female(1)={pos} ({pos/max(valid,1):.1%})")
        else:
            targets['sex_binary'] = ''
            logger.warning(f"  Column '{sex_col}' not found in harmonized — sex_binary will be empty")

    # --- age_regression ---
    task_cfg = stages_config.get('age_regression', {})
    if task_cfg.get('enabled', False):
        age_col = task_cfg['column']
        harm_id_col = task_cfg.get('subject_id_column', subject_id_col)
        logger.info(f"\n=== V2 Task: age_regression (column: {age_col}) ===")

        if df_harm is not None and age_col in df_harm.columns:
            df_harm[age_col] = pd.to_numeric(df_harm[age_col], errors='coerce')
            df_harm['age_value'] = df_harm[age_col].astype(str).replace(['nan', '<NA>'], '')
            targets = _merge_harmonized(targets, df_harm, harm_id_col, 'age_value')
            targets['age_value'] = targets['age_value'].fillna('')
            valid = (targets['age_value'] != '').sum()
            logger.info(f"  N={valid} with valid age")
        else:
            targets['age_value'] = ''
            logger.warning(f"  Column '{age_col}' not found in harmonized — age_value will be empty")

    # --- bmi_regression ---
    task_cfg = stages_config.get('bmi_regression', {})
    if task_cfg.get('enabled', False):
        bmi_col = task_cfg['column']
        harm_id_col = task_cfg.get('subject_id_column', subject_id_col)
        logger.info(f"\n=== V2 Task: bmi_regression (column: {bmi_col}) ===")

        if df_harm is not None and bmi_col in df_harm.columns:
            df_harm[bmi_col] = pd.to_numeric(df_harm[bmi_col], errors='coerce')
            df_harm['bmi_value'] = df_harm[bmi_col].astype(str).replace(['nan', '<NA>'], '')
            targets = _merge_harmonized(targets, df_harm, harm_id_col, 'bmi_value')
            targets['bmi_value'] = targets['bmi_value'].fillna('')
            valid = (targets['bmi_value'] != '').sum()
            logger.info(f"  N={valid} with valid BMI")
        else:
            targets['bmi_value'] = ''
            logger.warning(f"  Column '{bmi_col}' not found in harmonized — bmi_value will be empty")

    # ===================================================================
    # STATISTICS
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Target Statistics")
    logger.info("=" * 60)

    binary_cols = [c for c in ['depression_binary', 'anxiety_binary',
                                'insomnia_binary', 'sleepiness_binary',
                                'fatigue_binary'] if c in targets.columns]
    compute_task_statistics(targets, binary_cols, dataset,
                            is_multiclass={'apnea_class': True})

    return targets


def _log_binary_dist(df: pd.DataFrame, col: str, label_str: str) -> None:
    """Log binary class distribution."""
    valid = df[col][df[col].isin(['0', '1'])]
    if len(valid) == 0:
        logger.warning(f"  No valid labels found in '{col}'")
        return
    n_pos = (valid == '1').sum()
    n_neg = (valid == '0').sum()
    n_miss = len(df) - len(valid)
    labels = label_str.split(' / ')
    neg_label = labels[0] if len(labels) == 2 else '0'
    pos_label = labels[1] if len(labels) == 2 else '1'
    logger.info(f"  N={len(valid):,}  "
                f"{neg_label}={n_neg:,} ({n_neg/len(valid):.1%})  "
                f"{pos_label}={n_pos:,} ({n_pos/len(valid):.1%})  "
                f"missing={n_miss:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract classification targets for STAGES dataset"
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

    if not args.config.exists():
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    config = load_config_file(args.config)

    # Ensure SSHFS scratch mount is alive before touching any paths
    scratch_root = Path(config['paths']['raw_data']).parent  # cc_scratch/
    ensure_sshfs_mounted(
        mount_point=scratch_root,
        remote="boshra95@fir.alliancecan.ca:/home/boshra95/scratch/",
        options=["auto_cache", "reconnect", "compression=yes"],
    )

    log_dir = Path(config['paths']['targets_output'])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "extract_stages.log"
    setup_logging(log_file)

    logger.info("=" * 80)
    logger.info("STAGES Target Extraction Script")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Log file: {log_file}")

    try:
        targets_df = extract_stages_targets(config)

        output_path = args.output or (log_dir / "stages_targets.csv")

        _desired = [
            'subject_id', 'dataset', 'visit',
            'apnea_class', 'ahi_score',
            'depression_binary', 'phq9_score',
            'sleepiness_binary', 'ess_score',
            'anxiety_binary', 'gad7_score',
            'insomnia_binary', 'isi_score',
            'fatigue_binary', 'fss_score',
            'depression_extreme_binary',
            'sex_binary',
            'age_value',
            'bmi_value',
        ]
        column_order = [c for c in _desired if c in targets_df.columns]
        save_dataset_targets(targets_df, output_path, 'stages', column_order)

        logger.info("\n" + "=" * 80)
        logger.info("STAGES extraction completed successfully!")
        logger.info(f"   Output: {output_path}")
        logger.info(f"   Total subjects: {len(targets_df)}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.exception(f"Error during extraction: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
