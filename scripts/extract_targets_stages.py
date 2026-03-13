#!/usr/bin/env python3
"""
Extract Classification Targets for STAGES Dataset

STAGES is a single-visit dataset. Subject ID column is `subject_code`
(NOT nsrrid — see STAGES_DATA_NOTES.md and stages_adapter.py).

Extracts (all from stages-dataset-0.3.0.csv unless noted):
- depression_binary  : PHQ-9 (phq_1000) >= 10
- anxiety_binary     : GAD-7 (gad_0800) >= 10
- insomnia_binary    : ISI  (isi_score)  >= 15  [confirmed in main CSV]
- sleepiness_binary  : ESS  (ess_0900)   >= 11  [confirmed in main CSV]
- fatigue_binary     : FSS  (fss_1000)   >= 36  [disabled by default]
- apnea_binary       : SKIPPED (requires XML parsing)

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
    - All questionnaire columns in stages-dataset-0.3.0.csv
    - AHI not available from CSV (disabled)
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
    logger.info(f"Subject IDs found: {df[subject_id_col].notna().sum()} non-null")

    # Replace -9 sentinel with NaN across all numeric columns (NSRR convention)
    df = df.replace(-9, pd.NA)

    # ===================================================================
    # TASK 1: DEPRESSION (BINARY) — PHQ-9 >= 10
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
    # TASK 2: ANXIETY (BINARY) — GAD-7 >= 10
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
    # TASK 3: INSOMNIA (BINARY) — ISI >= 15
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
        # Rename raw column to score column (avoid collision with binary column name)
        df['isi_score'] = df[ins_col].astype(str).replace(['nan', '<NA>'], '')
        _log_binary_dist(df, 'insomnia_binary', "No insomnia / Insomnia")

    # ===================================================================
    # TASK 4: SLEEPINESS (BINARY) — ESS >= 11
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
    # TASK 5: FATIGUE (BINARY) — FSS >= 36 — disabled by default
    # ===================================================================
    task_cfg = stages_config['fatigue_binary']
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
        df['fatigue_binary'] = ''
        df['fss_score'] = ''

    # ===================================================================
    # APNEA — ALWAYS SKIPPED
    # ===================================================================
    logger.info("\n=== Task: apnea_binary (SKIPPED — AHI not in CSV, requires XML) ===")
    df['apnea_binary'] = ''
    df['ahi_score'] = ''

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

    for col in ['apnea_binary', 'ahi_score',
                'depression_binary', 'phq9_score',
                'sleepiness_binary', 'ess_score',
                'anxiety_binary', 'gad7_score',
                'insomnia_binary', 'isi_score',
                'fatigue_binary', 'fss_score']:
        targets[col] = df[col].values if col in df.columns else ''

    # Fill any remaining NaN with empty string
    for col in targets.columns:
        if col not in ['subject_id', 'dataset', 'visit']:
            targets[col] = targets[col].fillna('')

    logger.info(f"Total subjects in output: {len(targets)}")

    # ===================================================================
    # STATISTICS
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Target Statistics")
    logger.info("=" * 60)

    binary_cols = [c for c in ['depression_binary', 'anxiety_binary',
                                'insomnia_binary', 'sleepiness_binary',
                                'fatigue_binary'] if c in targets.columns]
    compute_task_statistics(targets, binary_cols, dataset, is_multiclass={})

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

        column_order = [
            'subject_id', 'dataset', 'visit',
            'apnea_binary', 'ahi_score',
            'depression_binary', 'phq9_score',
            'sleepiness_binary', 'ess_score',
            'anxiety_binary', 'gad7_score',
            'insomnia_binary', 'isi_score',
            'fatigue_binary', 'fss_score',
        ]

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
