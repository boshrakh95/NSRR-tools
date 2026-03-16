#!/usr/bin/env python3
"""
Create Per-Task Subject Lists

Produces one CSV per task containing only subjects with a valid label.
Supports both binary and multiclass tasks:

  Binary tasks   — read from master_targets.parquet (all datasets merged)
  Multiclass tasks — read from per-dataset CSVs (only datasets that have the column)

Output directory: <targets_output>/task_subjects/
  {task}_subjects.csv     columns: unified_id, dataset, subject_id, visit, label
  task_subject_summary.tsv  one-row-per-task summary of N / class balance / datasets

Usage:
    python scripts/create_task_subject_lists.py \\
        --config configs/target_extraction.yaml

Optional flags:
    --tasks apnea_binary apnea_class ...   (default: all tasks)
    --output-dir /path/override
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nsrr_tools.targets.extraction_utils import load_config_file
from nsrr_tools.utils.mount_utils import ensure_sshfs_mounted


MISSING_BINARY = -1   # sentinel in master_targets.parquet

# ── Binary tasks ──────────────────────────────────────────────────────────────
# Source: master_targets.parquet (all datasets, harmonised)
BINARY_TASKS = [
    'apnea_binary',
    'depression_binary',
    'sleepiness_binary',
    'anxiety_binary',
    'insomnia_binary',
    'fatigue_binary',
    'cvd_binary',
    'rested_morning',
]

# ── Multiclass tasks ──────────────────────────────────────────────────────────
# Source: per-dataset CSVs (only datasets that actually have the column)
# empty_sentinel: value stored when label is missing in the per-dataset CSV
MULTICLASS_TASKS = {
    'apnea_class': {
        'col': 'apnea_class',
        'datasets': ['apples', 'shhs', 'mros', 'stages'],  # all have AHI
        'num_classes': 4,
        'empty_sentinel': '',
    },
    'sleepiness_class': {
        'col': 'sleepiness_class',
        'datasets': ['apples', 'shhs', 'mros'],  # STAGES only has sleepiness_binary
        'num_classes': 3,
        'empty_sentinel': '',
    },
    'depression_class': {
        'col': 'depression_class',
        'datasets': ['apples'],  # only APPLES uses BDI multiclass
        'num_classes': 4,
        'empty_sentinel': '',
    },
}

DATASET_FILES = {
    'apples': 'apples_targets.csv',
    'shhs':   'shhs_targets.csv',
    'mros':   'mros_targets.csv',
    'stages': 'stages_targets.csv',
}

ALL_TASKS = BINARY_TASKS + list(MULTICLASS_TASKS.keys())


def setup_logging(log_file: Path) -> None:
    logger.remove()
    logger.add(sys.stderr, level='INFO')
    logger.add(log_file, level='DEBUG', mode='w')


# ── Binary ────────────────────────────────────────────────────────────────────

def build_binary_task_list(master: pd.DataFrame, task: str) -> pd.DataFrame:
    """Filter master to rows with valid (0/1) label for *task*."""
    if task not in master.columns:
        logger.warning(f"  Column '{task}' not in master — skipping")
        return pd.DataFrame()

    valid = master[master[task] != MISSING_BINARY].copy()
    valid = valid[['unified_id', 'dataset', 'subject_id', 'visit', task]].rename(
        columns={task: 'label'}
    )
    valid['label'] = valid['label'].astype(int)
    return valid.reset_index(drop=True)


# ── Multiclass ────────────────────────────────────────────────────────────────

def build_multiclass_task_list(
    task: str,
    task_def: dict,
    targets_dir: Path,
) -> pd.DataFrame:
    """Merge per-dataset CSVs for a multiclass task, keeping only labelled rows."""
    col      = task_def['col']
    sentinel = task_def['empty_sentinel']
    frames   = []

    for dataset in task_def['datasets']:
        csv_path = targets_dir / DATASET_FILES[dataset]
        if not csv_path.exists():
            logger.warning(f"  [{dataset}] CSV not found: {csv_path} — skipping")
            continue

        df = pd.read_csv(csv_path, dtype=str).fillna('')
        if col not in df.columns:
            logger.warning(f"  [{dataset}] column '{col}' not found — skipping")
            continue

        valid = df[df[col] != sentinel].copy()
        valid = valid[['subject_id', 'dataset', 'visit', col]].rename(
            columns={col: 'label'}
        )
        valid['label'] = pd.to_numeric(valid['label'], errors='coerce')
        valid = valid.dropna(subset=['label'])
        valid['label'] = valid['label'].astype(int)

        # re-create unified_id in the same format as master
        valid['unified_id'] = (
            valid['dataset'].astype(str) + '_' +
            valid['subject_id'].astype(str) + '_v' +
            valid['visit'].astype(str)
        )
        valid = valid[['unified_id', 'dataset', 'subject_id', 'visit', 'label']]
        logger.info(f"  [{dataset}] {len(valid):,} valid rows for '{col}'")
        frames.append(valid)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ── Stats ─────────────────────────────────────────────────────────────────────

def log_task_stats(task: str, df: pd.DataFrame, num_classes: int = 2) -> dict:
    if df.empty:
        logger.info(f"  {task}: NO valid subjects")
        return {}

    n_total  = len(df)
    datasets = df['dataset'].value_counts().to_dict()

    if num_classes == 2:
        n_pos   = (df['label'] == 1).sum()
        n_neg   = (df['label'] == 0).sum()
        pct_pos = n_pos / n_total
        logger.info(
            f"  {task}: N={n_total:,}  pos={n_pos:,} ({pct_pos:.1%})  "
            f"neg={n_neg:,}  datasets={datasets}"
        )
        balance_str = f"pos={n_pos},neg={n_neg}"
    else:
        counts = df['label'].value_counts().sort_index().to_dict()
        logger.info(
            f"  {task}: N={n_total:,}  class_counts={counts}  datasets={datasets}"
        )
        balance_str = ','.join(f"c{k}={v}" for k, v in counts.items())

    # per-dataset breakdown
    for ds, grp in df.groupby('dataset'):
        n_ds    = len(grp)
        counts_ds = grp['label'].value_counts().sort_index().to_dict()
        logger.info(f"    [{ds}] N={n_ds:,}  {counts_ds}")

    return {
        'task':        task,
        'n_total':     n_total,
        'num_classes': num_classes,
        'balance':     balance_str,
        'datasets':    ','.join(f"{k}:{v}" for k, v in datasets.items()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Create per-task subject lists (binary and multiclass)'
    )
    parser.add_argument(
        '--config', type=Path,
        default=Path('configs/target_extraction.yaml'),
    )
    parser.add_argument(
        '--output-dir', type=Path,
        help='Override output directory (default: <targets_output>/task_subjects/)'
    )
    parser.add_argument(
        '--tasks', nargs='+', default=ALL_TASKS, choices=ALL_TASKS,
        metavar='TASK',
        help=f'Tasks to process (default: all). Choices: {ALL_TASKS}'
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    config = load_config_file(args.config)

    scratch_root = Path(config['paths']['raw_data']).parent
    ensure_sshfs_mounted(
        mount_point=scratch_root,
        remote='boshra95@fir.alliancecan.ca:/home/boshra95/scratch/',
        options=['auto_cache', 'reconnect', 'compression=yes'],
    )

    targets_dir = Path(config['paths']['targets_output'])
    out_dir = args.output_dir or (targets_dir / 'task_subjects')
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = targets_dir / 'create_task_subject_lists.log'
    setup_logging(log_file)

    logger.info('=' * 80)
    logger.info('Create Per-Task Subject Lists')
    logger.info('=' * 80)

    # Load master once (needed for binary tasks)
    master_path = targets_dir / 'master_targets.parquet'
    master = None
    binary_tasks_requested = [t for t in args.tasks if t in BINARY_TASKS]
    if binary_tasks_requested:
        if not master_path.exists():
            logger.error(f'master_targets.parquet not found: {master_path}')
            logger.error('Run create_master_targets.py first.')
            return 1
        logger.info(f'Loading master: {master_path}')
        master = pd.read_parquet(master_path)
        logger.info(f'Loaded {len(master):,} records  datasets={master["dataset"].value_counts().to_dict()}')

    summary_rows = []

    # ── Binary tasks ─────────────────────────────────────────────────────────
    if binary_tasks_requested:
        logger.info('\n' + '=' * 80)
        logger.info('Binary tasks  (source: master_targets.parquet)')
        logger.info('=' * 80)

    for task in binary_tasks_requested:
        logger.info(f'\n--- {task} ---')
        df = build_binary_task_list(master, task)
        stats = log_task_stats(task, df, num_classes=2)
        if df.empty:
            continue
        out_path = out_dir / f'{task}_subjects.csv'
        df.to_csv(out_path, index=False)
        logger.info(f'  Saved: {out_path}')
        if stats:
            summary_rows.append(stats)

    # ── Multiclass tasks ──────────────────────────────────────────────────────
    multiclass_tasks_requested = [t for t in args.tasks if t in MULTICLASS_TASKS]
    if multiclass_tasks_requested:
        logger.info('\n' + '=' * 80)
        logger.info('Multiclass tasks  (source: per-dataset CSVs)')
        logger.info('=' * 80)

    for task in multiclass_tasks_requested:
        task_def = MULTICLASS_TASKS[task]
        logger.info(f'\n--- {task}  ({task_def["num_classes"]} classes, '
                    f'datasets: {task_def["datasets"]}) ---')
        df = build_multiclass_task_list(task, task_def, targets_dir)
        stats = log_task_stats(task, df, num_classes=task_def['num_classes'])
        if df.empty:
            continue
        out_path = out_dir / f'{task}_subjects.csv'
        df.to_csv(out_path, index=False)
        logger.info(f'  Saved: {out_path}')
        if stats:
            summary_rows.append(stats)

    # ── Summary ───────────────────────────────────────────────────────────────
    if summary_rows:
        summary_path = targets_dir / 'task_subject_summary.tsv'
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_path, sep='\t', index=False)
        logger.info(f'\nSummary saved: {summary_path}')
        logger.info('\n' + '=' * 80)
        logger.info('Summary')
        logger.info('=' * 80)
        logger.info('\n' + summary_df.to_string(index=False))

    logger.info('\nDone.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
