"""
repo_sync.py — Shared helpers for mirroring analysis outputs (figures, window
analysis tables, threshold tuning) from scratch into the NSRR-tools git repo,
round-tagged, so a `git pull` on a local machine gives access to all
CSVs/figures/markdown tables without checkpoints or parquets.

Repo layout (mirrors the scratch layout under <results_dir>/, round-tagged by
the results_dir's basename, e.g. "phase0_v3", "phase0_v3_full", "phase0_v3_abl"):

    results/figures/{round}/<same relative path as under results_dir/figures/>/
    results/inference/{round}/{task}_{head}{_tag}/
        window_analysis.md
        window_analysis_{split}.csv
        threshold_tuning.csv

PDFs are never mirrored — results/figures/**/*.pdf is gitignored (large; PNGs
are sufficient for quick viewing and paper drafting). Checkpoints and
*.parquet files are never touched by this module.

Usage (figures):
    from repo_sync import configure_repo_figures, save_figure
    configure_repo_figures(args.results_dir, args.repo_figures_dir)
    ...
    save_figure(fig, out_dir, stem)   # saves scratch png+pdf, mirrors png to repo

Usage (CSV/MD tables, e.g. analyze_windows.py, apply_threshold_tuning.py):
    from repo_sync import mirror_file, default_repo_inference_dir
    mirror_file(out_csv, repo_dir / out_csv.name)
"""
from pathlib import Path
import shutil

REPO_ROOT = Path(__file__).parent.parent

_results_dir: Path | None = None
_repo_figures_dir: Path | None = None


def round_name(results_dir: Path) -> str:
    """'/scratch/.../results/phase0_v3_full' -> 'phase0_v3_full'."""
    return Path(results_dir).name


def default_repo_figures_dir(results_dir: Path) -> Path:
    return REPO_ROOT / "results" / "figures" / round_name(results_dir)


def default_repo_inference_dir(results_dir: Path) -> Path:
    return REPO_ROOT / "results" / "inference" / round_name(results_dir)


def configure_repo_figures(results_dir: Path, repo_figures_dir: Path | None) -> None:
    """Call once from a plotting script's main(). repo_figures_dir=None disables
    repo mirroring (e.g. for ad-hoc local runs that shouldn't touch the repo)."""
    global _results_dir, _repo_figures_dir
    _results_dir = Path(results_dir)
    _repo_figures_dir = Path(repo_figures_dir) if repo_figures_dir else None


def save_figure(fig, out_dir: Path, stem: str, exts=("png", "pdf")) -> None:
    """Save fig to out_dir/{stem}.{ext} for each ext (scratch, as before), then
    mirror the PNG into the repo if configure_repo_figures() set a target dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in exts:
        fig.savefig(out_dir / f"{stem}.{ext}",
                    dpi=150 if ext == "png" else None, bbox_inches="tight")

    repo_note = ""
    if _repo_figures_dir is not None and _results_dir is not None and "png" in exts:
        try:
            rel = out_dir.relative_to(Path(_results_dir) / "figures")
        except ValueError:
            rel = Path(out_dir.name)  # out_dir wasn't under results_dir/figures; fall back flat
        repo_dest_dir = _repo_figures_dir / rel
        repo_dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_dir / f"{stem}.png", repo_dest_dir / f"{stem}.png")
        repo_note = " (+ repo copy)"

    print(f"  Saved: {out_dir}/{stem}.{{{','.join(exts)}}}{repo_note}")


def mirror_file(src: Path, repo_dir: Path | None) -> None:
    """Copy a single already-written scratch file (CSV/MD) into repo_dir, if given."""
    if repo_dir is None:
        return
    repo_dir = Path(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(src), repo_dir / Path(src).name)
