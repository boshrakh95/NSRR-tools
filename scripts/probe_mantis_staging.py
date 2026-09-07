#!/usr/bin/env python3
"""
probe_mantis_staging.py — Mantis baseline, Phase 1 Step 5

A THROWAWAY ENGINEERING INSTRUMENT, not a paper deliverable. Scores a set
of per-epoch Mantis embeddings against real per-epoch sleep-stage labels
via a plain multinomial logistic regression, subject-wise held-out split.
Exists ONLY to help decide between windowing options (Option D vs
D-interp vs B, plan §13.1) and output-token/layer options (plan §13.2/
§3.3) BEFORE committing to one for the real Stage 1/Stage 2 sweep.

⚠️ SLEEP STAGING IS NOT ONE OF THIS PROJECT'S COMPARISON TASKS. It is
explicitly out of scope for all three TSFM baselines (plan §5.8) — the 5
real tasks are sex_binary/sleep_efficiency_binary/bmi_binary/age_class/
apnea_binary, unchanged. This script's output never appears in any
results table; it only ever informs which `embedding.windowing`/
`pe_mode`/`return_transf_layer`/`output_token` config values are used to
extract the REAL embeddings those 5 tasks are trained on.

WHY THIS INSTRUMENT, NOT 30s sex_binary VAL AUROC (plan §13.4)
─────────────────────────────────────────────────────────────
~20,000 held-out LABELLED EPOCHS (subject-wise split) vs ~165 held-out
SUBJECTS for a sex_binary pilot — detects a real difference of ~0.005
instead of ~0.04, at a fraction of the extraction cost, and is directly
comparable to published Mantis-on-NSRR sleep-staging numbers (Gnassounou
et al. 2025, weighted F1 75-89 on NSRR-family cohorts, plan
§13.4/`docs/TSFM_THIRD_MODEL_DECISION.md` §2).

WHAT COUNTS AS "PASSING" FOR THIS STEP (1.5)
─────────────────────────────────────────────
This step is about the SCRIPT being correct — data loading, subject-wise
split, NaN/degenerate-embedding checks, F1/kappa computation — not about
reaching a real conclusion. Checklist 1.6 is the step that actually runs
this across all 12 windowing/layer/token variants at real scale (~100
subjects) to make the Pilot 1/2 decision.

USAGE
─────
  python scripts/probe_mantis_staging.py \\
      --embedding-dir /scratch/boshra95/psg/unified/embeddings/mantis_30sec \\
      --datasets apples shhs
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Same remapping convention as osf_context_window_dataset.py's _remap_stages
# (REM=5 in the raw annotation -> 4, giving 5 contiguous classes 0..4).
REM_ORIGINAL = 5
REM_REMAPPED = 4


def _remap_stages(arr: np.ndarray) -> np.ndarray:
    out = arr.copy().astype(np.int64)
    out[out == REM_ORIGINAL] = REM_REMAPPED
    return out


def load_subject_epochs(embedding_path: Path, stage_path: Path):
    """Loads one subject's [T, 6, D] embeddings + [T_ann] stage labels,
    truncates both to T_eff = min(T, T_ann) (same convention
    osf_context_window_dataset.py uses for embedding/annotation length
    mismatches), and flattens embeddings to [T_eff, 6*D] — the same
    reshape the real sequence head does (x.reshape(N, FLAT_DIM)).

    Returns (X, y) or None if either file is missing or T_eff == 0.
    """
    if not embedding_path.exists() or not stage_path.exists():
        return None
    emb = np.load(embedding_path).astype(np.float32)  # [T, 6, D]
    stages = _remap_stages(np.load(stage_path))  # [T_ann]
    t_eff = min(emb.shape[0], len(stages))
    if t_eff == 0:
        return None
    emb = emb[:t_eff].reshape(t_eff, -1)  # [t_eff, 6*D]
    return emb, stages[:t_eff]


def main():
    parser = argparse.ArgumentParser(
        description="Single-epoch sleep-staging probe — a Pilot 1/2 engineering "
                    "instrument, NOT a paper task (plan §13.4)."
    )
    parser.add_argument("--embedding-dir", required=True,
                        help="Root dir of {dataset}/{subject_id}.npy Mantis embeddings")
    parser.add_argument("--stage-dir", default="/scratch/boshra95/psg",
                        help="Root of {dataset}/derived/annotations/{subject_id}_stages.npy")
    parser.add_argument("--datasets", nargs="+", default=["apples", "shhs"])
    parser.add_argument("--limit-per-dataset", type=int, default=None,
                        help="Debug: cap subjects per dataset")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    embedding_root = Path(args.embedding_dir)
    stage_root = Path(args.stage_dir)

    subjects = []  # (dataset, subject_id)
    for ds in args.datasets:
        ds_dir = embedding_root / ds
        if not ds_dir.exists():
            print(f"WARNING: {ds_dir} does not exist, skipping {ds}")
            continue
        files = sorted(p.stem for p in ds_dir.glob("*.npy"))
        if args.limit_per_dataset:
            files = files[: args.limit_per_dataset]
        subjects.extend((ds, sid) for sid in files)

    if len(subjects) < 4:
        raise SystemExit(
            f"Only {len(subjects)} subjects found under {embedding_root} for "
            f"{args.datasets} — need more to run a meaningful subject-wise split."
        )
    print(f"Found {len(subjects)} candidate subjects across {args.datasets}")

    # Subject-wise split (same rng convention as the rest of this project —
    # np.random.default_rng(split_seed).shuffle() — so this is reproducible
    # and comparable across variant runs in checklist 1.6).
    rng = np.random.default_rng(args.split_seed)
    order = np.arange(len(subjects))
    rng.shuffle(order)
    n_test = max(1, int(round(len(subjects) * args.test_fraction)))
    test_positions = set(order[:n_test].tolist())

    X_train, y_train, X_test, y_test = [], [], [], []
    n_train_subj = n_test_subj = n_skipped = 0
    flat_dim_seen = None

    for pos, (ds, sid) in enumerate(subjects):
        emb_path = embedding_root / ds / f"{sid}.npy"
        stage_path = stage_root / ds / "derived" / "annotations" / f"{sid}_stages.npy"
        result = load_subject_epochs(emb_path, stage_path)
        if result is None:
            n_skipped += 1
            continue
        emb, stages = result

        # NaN / degenerate-embedding assertions (subsumes the old standalone
        # sanity-check step — a collapsed/broken embedding can't reach a
        # plausible staging score anyway, so a sane probe result is stronger
        # evidence than a NaN/variance check alone, plan §13.4).
        if np.isnan(emb).any() or np.isinf(emb).any():
            print(f"  SKIP {ds}/{sid}: NaN/Inf in embedding")
            n_skipped += 1
            continue
        if (emb.std(axis=0) < 1e-6).all():
            print(f"  SKIP {ds}/{sid}: degenerate (near-zero variance) embedding")
            n_skipped += 1
            continue

        if flat_dim_seen is None:
            flat_dim_seen = emb.shape[1]
        elif emb.shape[1] != flat_dim_seen:
            raise ValueError(
                f"{ds}/{sid} has flattened dim {emb.shape[1]}, expected "
                f"{flat_dim_seen} — mixed embedding configs under {embedding_root}?"
            )

        if pos in test_positions:
            X_test.append(emb); y_test.append(stages)
            n_test_subj += 1
        else:
            X_train.append(emb); y_train.append(stages)
            n_train_subj += 1

    if not X_train or not X_test:
        raise SystemExit(
            f"Empty train or test split after loading (train_subj={n_train_subj}, "
            f"test_subj={n_test_subj}, skipped={n_skipped}) — need more subjects "
            f"or a smaller --test-fraction."
        )

    X_train = np.concatenate(X_train); y_train = np.concatenate(y_train)
    X_test = np.concatenate(X_test); y_test = np.concatenate(y_test)

    print(
        f"Loaded {n_train_subj + n_test_subj} subjects ({n_skipped} skipped) — "
        f"train: {len(y_train)} epochs from {n_train_subj} subjects, "
        f"test: {len(y_test)} epochs from {n_test_subj} subjects, "
        f"flat_dim={flat_dim_seen}"
    )
    if n_test_subj < 5:
        print(
            f"⚠️  Only {n_test_subj} held-out test subject(s) — fine for a code "
            f"smoke test (checklist 1.5), NOT enough for a real Pilot 1/2 "
            f"decision (checklist 1.6 needs ~100 subjects, plan §13.4)."
        )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)

    print(f"\nweighted F1: {weighted_f1:.4f}")
    print(f"Cohen's kappa: {kappa:.4f}")
    print(
        "(Reference: Gnassounou et al. 2025 report Mantis at 75-89 weighted F1 "
        "on real NSRR-family sleep-staging datasets, plan §13.4 — this smoke "
        "test's subject count is far too small to expect that range; checklist "
        "1.6's ~100-subject run is what actually informs the Pilot 1/2 decision.)"
    )


if __name__ == "__main__":
    main()
