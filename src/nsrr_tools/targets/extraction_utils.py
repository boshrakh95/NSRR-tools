"""
Target Extraction Utilities

Common functions for extracting and processing classification targets from CSV files.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger


def apply_threshold(
    value: Any,
    threshold: float,
    missing_value: str = ""
) -> str:
    """
    Apply threshold to convert continuous score to binary classification.
    
    Args:
        value: Continuous score value
        threshold: Threshold for binary classification (>= threshold → 1)
        missing_value: Value to use for missing/NaN entries (default: empty string)
    
    Returns:
        Binary classification as string: '1', '0', or missing_value
    """
    if pd.isna(value):
        return missing_value
    try:
        return '1' if float(value) >= threshold else '0'
    except (ValueError, TypeError):
        return missing_value


def apply_multiclass_threshold(
    value: Any,
    thresholds: List[float],
    missing_value: str = ""
) -> str:
    """
    Apply multiple thresholds to convert continuous score to multi-class classification.
    
    Args:
        value: Continuous score value
        thresholds: List of threshold boundaries. For n+1 classes, provide n thresholds.
                   Example: [5, 15, 30] creates 4 classes: <5, 5-15, 15-30, >=30
        missing_value: Value to use for missing/NaN entries (default: empty string)
    
    Returns:
        Class label as string: '0', '1', '2', ... or missing_value
        
    Example:
        apply_multiclass_threshold(25, [5, 15, 30]) → '2' (falls in 15-30 range)
        apply_multiclass_threshold(35, [5, 15, 30]) → '3' (>= 30)
    """
    if pd.isna(value):
        return missing_value
    try:
        score = float(value)
        for class_idx, threshold in enumerate(thresholds):
            if score < threshold:
                return str(class_idx)
        # If score >= all thresholds, assign to highest class
        return str(len(thresholds))
    except (ValueError, TypeError):
        return missing_value


def apply_rested_threshold(
    value: Any,
    good_threshold: int = 4,
    poor_threshold: int = 2,
    missing_value: str = ""
) -> str:
    """
    Apply threshold for morning restedness (1-5 scale).
    
    Args:
        value: Self-report score (typically 1-5)
        good_threshold: Score >= this is "well-rested" (1)
        poor_threshold: Score <= this is "poorly-rested" (0)
        missing_value: Value for ambiguous scores (e.g., 3) or NaN
    
    Returns:
        Binary classification: '1' (good), '0' (poor), or missing_value
    """
    if pd.isna(value):
        return missing_value
    try:
        score = int(value)
        if score >= good_threshold:
            return '1'
        elif score <= poor_threshold:
            return '0'
        else:
            return missing_value  # Ambiguous (e.g., 3)
    except (ValueError, TypeError):
        return missing_value


def validate_score_range(
    df: pd.DataFrame,
    column: str,
    expected_range: Tuple[float, float],
    dataset: str,
    task: str
) -> bool:
    """
    Validate that scores are within expected range.
    
    Args:
        df: DataFrame containing the scores
        column: Column name to validate
        expected_range: (min, max) tuple for valid range
        dataset: Dataset name for logging
        task: Task name for logging
    
    Returns:
        True if validation passes, False otherwise
    """
    if column not in df.columns:
        logger.error(f"{dataset}.{task}: Column '{column}' not found")
        return False
    
    # Get non-null values
    scores = df[column].dropna()
    
    if len(scores) == 0:
        logger.warning(f"{dataset}.{task}: No valid scores in '{column}'")
        return False
    
    min_val, max_val = expected_range
    out_of_range = scores[(scores < min_val) | (scores > max_val)]
    
    if len(out_of_range) > 0:
        logger.error(
            f"{dataset}.{task}: {len(out_of_range)} scores out of range "
            f"[{min_val}, {max_val}] in '{column}'"
        )
        logger.error(f"  Out of range values: {out_of_range.values[:10]}")
        return False
    
    logger.info(
        f"{dataset}.{task}: Score range validation passed for '{column}' "
        f"({len(scores)} valid scores)"
    )
    return True


def validate_prevalence(
    df: pd.DataFrame,
    column: str,
    expected_range: Tuple[float, float],
    dataset: str,
    task: str
) -> bool:
    """
    Validate that prevalence of positive class is within expected range.
    
    Args:
        df: DataFrame containing binary classification column
        column: Column name (should contain '0', '1', or '')
        expected_range: (min, max) tuple for expected prevalence
        dataset: Dataset name for logging
        task: Task name for logging
    
    Returns:
        True if validation passes (with warning if out of range)
    """
    if column not in df.columns:
        logger.error(f"{dataset}.{task}: Column '{column}' not found")
        return False
    
    # Count valid labels (exclude missing)
    valid_labels = df[column][df[column].isin(['0', '1'])]
    
    if len(valid_labels) == 0:
        logger.warning(f"{dataset}.{task}: No valid labels in '{column}'")
        return False
    
    positive_count = (valid_labels == '1').sum()
    prevalence = positive_count / len(valid_labels)
    
    min_prev, max_prev = expected_range
    
    if min_prev <= prevalence <= max_prev:
        logger.info(
            f"{dataset}.{task}: Prevalence {prevalence:.2%} "
            f"within expected range [{min_prev:.1%}, {max_prev:.1%}] "
            f"({positive_count}/{len(valid_labels)})"
        )
        return True
    else:
        logger.warning(
            f"{dataset}.{task}: Prevalence {prevalence:.2%} "
            f"outside expected range [{min_prev:.1%}, {max_prev:.1%}] "
            f"({positive_count}/{len(valid_labels)})"
        )
        return True  # Warning, not error


def merge_visit_data(
    df_list: List[pd.DataFrame],
    visit_numbers: List[int],
    subject_id_col: str,
    dataset: str
) -> pd.DataFrame:
    """
    Merge data from multiple visits, keeping each visit as a separate row.
    
    Args:
        df_list: List of DataFrames, one per visit
        visit_numbers: List of visit numbers corresponding to each DataFrame
        subject_id_col: Column name for subject ID
        dataset: Dataset name for logging
    
    Returns:
        Merged DataFrame with 'visit' column added
    """
    merged_dfs = []
    
    for df, visit_num in zip(df_list, visit_numbers):
        df = df.copy()
        df['visit'] = visit_num
        merged_dfs.append(df)
        logger.info(f"{dataset}: Visit {visit_num} - {len(df)} subjects")
    
    result = pd.concat(merged_dfs, ignore_index=True)
    logger.info(f"{dataset}: Total records after merging visits: {len(result)}")
    
    return result


def create_source_files_dict(file_paths: Dict[str, str]) -> str:
    """
    Create a JSON string representation of source files for documentation.
    
    Args:
        file_paths: Dictionary of {task: filename}
    
    Returns:
        JSON string of source files
    """
    import json
    return json.dumps(file_paths, indent=2)


def compute_task_statistics(
    df: pd.DataFrame,
    task_columns: List[str],
    dataset: str,
    is_multiclass: Optional[Dict[str, bool]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compute statistics for all task columns (binary or multi-class).
    
    Args:
        df: DataFrame with task columns
        task_columns: List of task column names
        dataset: Dataset name
        is_multiclass: Dict mapping column names to True/False for multi-class handling
    
    Returns:
        Dictionary with statistics per task
    """
    stats = {}
    is_multiclass = is_multiclass or {}
    
    for col in task_columns:
        if col not in df.columns:
            continue
        
        # Get valid entries (non-empty strings)
        valid = df[col][df[col] != '']
        
        if len(valid) == 0:
            stats[col] = {
                'dataset': dataset,
                'total': 0,
                'missing': len(df),
                'distribution': {}
            }
        else:
            missing = len(df) - len(valid)
            
            # Count class distribution
            class_counts = valid.value_counts().to_dict()
            class_dist = {k: int(v) for k, v in class_counts.items()}
            
            stats[col] = {
                'dataset': dataset,
                'total': len(valid),
                'missing': int(missing),
                'distribution': class_dist
            }
            
            # Log statistics
            if is_multiclass.get(col, False):
                # Multi-class
                dist_str = ", ".join([f"class{k}={v} ({v/len(valid):.1%})" 
                                     for k, v in sorted(class_dist.items())])
                logger.info(
                    f"{dataset}.{col}: "
                    f"N={len(valid):,}, "
                    f"{dist_str}, "
                    f"missing={missing:,}"
                )
            else:
                # Binary (legacy support)
                if '1' in class_dist and '0' in class_dist:
                    positive = class_dist.get('1', 0)
                    negative = class_dist.get('0', 0)
                    logger.info(
                        f"{dataset}.{col}: "
                        f"N={len(valid):,}, "
                        f"pos={positive:,} ({positive/len(valid):.1%}), "
                        f"neg={negative:,}, "
                        f"missing={missing:,}"
                    )
                else:
                    # Unknown format, just show distribution
                    dist_str = ", ".join([f"{k}={v}" for k, v in sorted(class_dist.items())])
                    logger.info(
                        f"{dataset}.{col}: "
                        f"N={len(valid):,}, "
                        f"{dist_str}, "
                        f"missing={missing:,}"
                    )
    
    return stats


def save_dataset_targets(
    df: pd.DataFrame,
    output_path: Path,
    dataset: str,
    columns_order: Optional[List[str]] = None
) -> None:
    """
    Save per-dataset target file as CSV.
    
    Args:
        df: DataFrame with extracted targets
        output_path: Path to save CSV file
        dataset: Dataset name
        columns_order: Preferred column ordering (optional)
    """
    # Reorder columns if specified
    if columns_order:
        # Keep only columns that exist
        cols = [c for c in columns_order if c in df.columns]
        # Add any extra columns not in the order list
        extra_cols = [c for c in df.columns if c not in columns_order]
        df = df[cols + extra_cols]
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"{dataset}: Saved {len(df)} records to {output_path}")


def load_config_file(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
