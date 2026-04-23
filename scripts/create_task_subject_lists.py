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
    # v2 additions
    'sex_binary',
    'sleep_efficiency_binary',
    'psqi_binary',
    'depression_extreme_binary',
    'osa_binary_apples_postqc',
    'bmi_binary',
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
    # v2 additions
    'osa_severity_apples': {
        'col': 'osa_severity_apples',
        'datasets': ['apples'],
        'num_classes': 4,
        'empty_sentinel': '',
    },
    'age_class': {
        'col': 'age_class',
        'datasets': ['apples', 'shhs', 'mros', 'stages'],
        'num_classes': 3,
        'empty_sentinel': '',
    },
}

# ── Regression tasks ──────────────────────────────────────────────────────────
# Source: master_targets.parquet (float columns, NaN if missing)
REGRESSION_TASKS = {
    'age_regression': {
        'col': 'age_value',
        'datasets': ['apples', 'shhs', 'mros', 'stages'],
    },
    'bmi_regression': {
        'col': 'bmi_value',
        'datasets': ['apples', 'shhs', 'mros', 'stages'],
    },
}

DATASET_FILES = {
    'apples': 'apples_targets.csv',
    'shhs':   'shhs_targets.csv',
    'mros':   'mros_targets.csv',
    'stages': 'stages_targets.csv',
}

# ── Sleep staging ─────────────────────────────────────────────────────────────
# Source: per-subject *_stages.npy in psg/{dataset}/derived/annotations/
# Labels are per-30-sec-epoch arrays — NOT a single label per subject.
# Output CSV has annotation_path + n_epochs instead of a label column.
# Class values in files: 0=Wake, 1=N1, 2=N2, 3=N3, 5=REM
#   Remap 5→4 happens in the DataLoader, not here.
# Filename conventions:
#   SHHS (multi-visit):  {subject_id}_v{visit}_stages.npy
#   All others:          {subject_id}_stages.npy  (visit inferred from master_targets)
STAGING_DATASETS = ['apples', 'shhs', 'mros', 'stages']

ALL_TASKS = (
    BINARY_TASKS +
    list(MULTICLASS_TASKS.keys()) +
    list(REGRESSION_TASKS.keys()) +
    ['sleep_staging']
)


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

    valid = master[master[task].notna() & (master[task] != MISSING_BINARY)].copy()
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


# ── Regression ───────────────────────────────────────────────────────────────

def build_regression_task_list(master: pd.DataFrame, task: str, col: str) -> pd.DataFrame:
    """Filter master to rows with non-NaN float value for a regression task."""
    if col not in master.columns:
        logger.warning(f"  Column '{col}' not in master — skipping")
        return pd.DataFrame()

    valid = master[master[col].notna()].copy()
    valid = valid[['unified_id', 'dataset', 'subject_id', 'visit', col]].rename(
        columns={col: 'label'}
    )
    valid['label'] = valid['label'].astype(float)
    return valid.reset_index(drop=True)


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


# ── Sleep staging ─────────────────────────────────────────────────────────────

def build_sleep_staging_subject_list(
    derived_root: Path,
    master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Scan annotations dirs for *_stages.npy files and cross-reference with master.

    Returns DataFrame with columns:
        unified_id, dataset, subject_id, visit, annotation_path, n_epochs
    annotation_path is absolute.
    Subjects present in annotations but missing from master are warned and skipped.
    """
    import re
    import numpy as np

    rows = []

    # Build a lookup from master: (dataset, subject_id, visit) → unified_id
    master_index = {
        (row['dataset'], str(row['subject_id']), int(row['visit'])): row['unified_id']
        for _, row in master.iterrows()
    }

    for dataset in STAGING_DATASETS:
        ann_dir = derived_root / dataset / 'derived' / 'annotations'
        if not ann_dir.exists():
            logger.warning(f'  [{dataset}] annotations dir not found: {ann_dir}')
            continue

        npy_files = sorted(
            p for p in ann_dir.glob('*_stages.npy')
            if not p.name.startswith('._')
        )
        logger.info(f'  [{dataset}] found {len(npy_files):,} staging files')

        matched = skipped = 0
        for npy_path in npy_files:
            stem = npy_path.stem  # e.g. "200001_v1_stages" or "AA0001_stages"

            # Parse subject_id and visit from filename
            # Pattern 1: {subject_id}_v{visit}_stages  (SHHS multi-visit)
            m = re.match(r'^(.+)_v(\d+)_stages$', stem)
            if m:
                subject_id = m.group(1)
                visit = int(m.group(2))
            else:
                # Pattern 2: {subject_id}_stages  (single-visit or visit not in filename)
                m2 = re.match(r'^(.+)_stages$', stem)
                if not m2:
                    logger.warning(f'  [{dataset}] unexpected filename: {npy_path.name} — skipping')
                    skipped += 1
                    continue
                subject_id = m2.group(1)
                # Infer visit from master: find any matching (dataset, subject_id) row
                candidates = [
                    v for (ds, sid, v) in master_index if ds == dataset and sid == subject_id
                ]
                if not candidates:
                    logger.debug(f'  [{dataset}] {subject_id}: not in master — skipping')
                    skipped += 1
                    continue
                visit = candidates[0]  # take the first (usually only one)

            unified_id = master_index.get((dataset, subject_id, visit))
            if unified_id is None:
                # Fallback: try composite subject_id (MrOS stores subject_id as AA0001_v1)
                composite_id = f"{subject_id}_v{visit}"
                unified_id = master_index.get((dataset, composite_id, visit))
            if unified_id is None:
                logger.debug(f'  [{dataset}] ({subject_id}, v{visit}) not in master — skipping')
                skipped += 1
                continue

            try:
                arr = np.load(npy_path, mmap_mode='r')
                n_epochs = int(arr.shape[0])
            except Exception as e:
                logger.warning(f'  [{dataset}] failed to read {npy_path.name}: {e}')
                skipped += 1
                continue

            rows.append({
                'unified_id':       unified_id,
                'dataset':          dataset,
                'subject_id':       subject_id,
                'visit':            visit,
                'annotation_path':  str(npy_path),
                'n_epochs':         n_epochs,
            })
            matched += 1

        logger.info(f'  [{dataset}] matched={matched:,}  skipped={skipped:,}')

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    total_epochs = df['n_epochs'].sum()
    logger.info(
        f'  sleep_staging: N={len(df):,} subjects  '
        f'total_epochs={total_epochs:,} (~{total_epochs*30/3600:.0f} h)'
    )
    return df.reset_index(drop=True)


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

    # ── Regression tasks ──────────────────────────────────────────────────────
    regression_tasks_requested = [t for t in args.tasks if t in REGRESSION_TASKS]
    if regression_tasks_requested:
        logger.info('\n' + '=' * 80)
        logger.info('Regression tasks  (source: master_targets.parquet)')
        logger.info('=' * 80)

        if master is None:
            if not master_path.exists():
                logger.error(f'master_targets.parquet not found: {master_path}')
                return 1
            master = pd.read_parquet(master_path)

    for task in regression_tasks_requested:
        task_def = REGRESSION_TASKS[task]
        col = task_def['col']
        logger.info(f'\n--- {task}  (column: {col}) ---')
        df = build_regression_task_list(master, task, col)
        if df.empty:
            logger.info(f'  {task}: NO valid subjects')
            continue
        n_total = len(df)
        datasets = df['dataset'].value_counts().to_dict()
        mean_val = df['label'].mean()
        std_val = df['label'].std()
        logger.info(
            f'  {task}: N={n_total:,}  mean={mean_val:.1f}  std={std_val:.1f}  datasets={datasets}'
        )
        for ds, grp in df.groupby('dataset'):
            logger.info(f'    [{ds}] N={len(grp):,}  mean={grp["label"].mean():.1f}')
        out_path = out_dir / f'{task}_subjects.csv'
        df.to_csv(out_path, index=False)
        logger.info(f'  Saved: {out_path}')
        summary_rows.append({
            'task':        task,
            'n_total':     n_total,
            'num_classes': 0,  # 0 signals regression
            'balance':     f'mean={mean_val:.1f},std={std_val:.1f}',
            'datasets':    ','.join(f'{k}:{v}' for k, v in datasets.items()),
        })

    # ── Sleep staging ────────────────────────────────────────────────────────
    if 'sleep_staging' in args.tasks:
        logger.info('\n' + '=' * 80)
        logger.info('Sleep staging  (source: psg/*/derived/annotations/*_stages.npy)')
        logger.info('=' * 80)

        # master must be loaded for unified_id lookup
        if master is None:
            if not master_path.exists():
                logger.error(f'master_targets.parquet not found: {master_path}')
                return 1
            master = pd.read_parquet(master_path)

        derived_root = Path(config['paths']['derived_data'])
        df_staging = build_sleep_staging_subject_list(derived_root, master)

        if not df_staging.empty:
            out_path = out_dir / 'sleep_staging_subjects.csv'
            df_staging.to_csv(out_path, index=False)
            logger.info(f'  Saved: {out_path}')
            summary_rows.append({
                'task':        'sleep_staging',
                'n_total':     len(df_staging),
                'num_classes': 5,
                'balance':     'per-epoch (see annotation_path)',
                'datasets':    ','.join(
                    f"{k}:{v}" for k, v in
                    df_staging['dataset'].value_counts().to_dict().items()
                ),
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    if summary_rows:
        summary_path = targets_dir / 'task_subject_summary.tsv'
        new_df = pd.DataFrame(summary_rows)
        if summary_path.exists():
            existing = pd.read_csv(summary_path, sep='\t')
            # Replace rows for tasks we just ran, keep the rest
            existing = existing[~existing['task'].isin(new_df['task'])]
            summary_df = pd.concat([existing, new_df], ignore_index=True)
        else:
            summary_df = new_df
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
