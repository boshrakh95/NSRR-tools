#!/usr/bin/env python3
"""
CSV File and Column Verification Script

Verifies that all required CSV files exist and contain the expected columns
for target extraction.

Usage:
    python scripts/verify_csv_files.py --config configs/target_extraction.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from loguru import logger


def setup_logging(log_file: Path) -> None:
    """Configure logging with loguru."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, level="DEBUG", mode="w")


def load_config(config_path: Path) -> Dict:
    """Load target extraction configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def verify_file_exists(file_path: Path, dataset: str, file_desc: str) -> bool:
    """Check if a file exists and is readable."""
    if not file_path.exists():
        logger.error(f"❌ {dataset}: {file_desc} not found: {file_path}")
        return False
    
    if not file_path.is_file():
        logger.error(f"❌ {dataset}: {file_desc} is not a file: {file_path}")
        return False
    
    logger.info(f"✅ {dataset}: {file_desc} found")
    return True


def verify_columns(file_path: Path, required_columns: List[str], 
                   dataset: str, file_desc: str) -> Tuple[bool, List[str]]:
    """Verify that required columns exist in the CSV/Excel file."""
    try:
        # Determine file type and read accordingly
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path, nrows=0)  # Just read headers
        elif file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, nrows=0)
        else:
            logger.error(f"❌ {dataset}: Unsupported file type: {file_path.suffix}")
            return False, []
        
        available_columns = df.columns.tolist()
        missing_columns = [col for col in required_columns if col not in available_columns]
        
        if missing_columns:
            logger.warning(
                f"⚠️  {dataset}: {file_desc} missing columns: {missing_columns}"
            )
            logger.debug(f"    Available columns: {available_columns[:20]}...")  # Show first 20
            return False, missing_columns
        else:
            logger.info(f"✅ {dataset}: All required columns present in {file_desc}")
            return True, []
            
    except Exception as e:
        logger.error(f"❌ {dataset}: Error reading {file_desc}: {e}")
        return False, []


def verify_dataset(dataset_name: str, dataset_config: Dict, 
                   raw_data_path: Path) -> Dict:
    """Verify all files and columns for a single dataset."""
    results = {
        'dataset': dataset_name,
        'files_ok': True,
        'columns_ok': True,
        'missing_files': [],
        'missing_columns': {},
        'tasks_verified': []
    }
    
    dataset_path = raw_data_path / dataset_name / "datasets"
    logger.info(f"\n{'='*60}")
    logger.info(f"Verifying {dataset_name.upper()} dataset")
    logger.info(f"{'='*60}")
    
    # Iterate through each task for this dataset
    for task_name, task_config in dataset_config.items():
        if task_name in ['dataset_id', 'subject_id_column']:
            continue
        
        if not task_config.get('enabled', True):
            logger.info(f"⏭️  {dataset_name}: {task_name} disabled, skipping")
            continue
        
        logger.info(f"\n--- Task: {task_name} (Priority: {task_config.get('priority', 'N/A')}) ---")
        
        # Handle single source file or multiple
        source_files = task_config.get('source_file') or task_config.get('source_files', {})
        
        if isinstance(source_files, str):
            # Single file
            source_files = {'default': source_files}
        elif isinstance(source_files, dict):
            # Multiple files (e.g., visit1/visit2)
            pass
        else:
            logger.warning(f"⚠️  {dataset_name}.{task_name}: No source files defined")
            continue
        
        # Verify each source file
        for file_key, filename in source_files.items():
            file_path = dataset_path / filename
            file_desc = f"{task_name} ({file_key})"
            
            # Check file exists
            if not verify_file_exists(file_path, dataset_name, file_desc):
                results['files_ok'] = False
                results['missing_files'].append(str(file_path))
                continue
            
            # Determine required columns
            required_cols = [dataset_config['subject_id_column']]
            
            if isinstance(task_config.get('column'), str):
                required_cols.append(task_config['column'])
            elif isinstance(task_config.get('columns'), dict):
                # Visit-specific columns
                if file_key in task_config['columns']:
                    required_cols.append(task_config['columns'][file_key])
            
            # Add visit column if specified
            if 'visit_column' in task_config:
                required_cols.append(task_config['visit_column'])
            
            # Verify columns exist
            cols_ok, missing = verify_columns(file_path, required_cols, 
                                             dataset_name, file_desc)
            if not cols_ok:
                results['columns_ok'] = False
                results['missing_columns'][f"{task_name}_{file_key}"] = missing
            else:
                results['tasks_verified'].append(f"{task_name}_{file_key}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Verify CSV files and columns for target extraction"
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/target_extraction.yaml'),
        help='Path to target extraction config file'
    )
    args = parser.parse_args()
    
    # Load configuration
    if not args.config.exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Setup logging
    log_dir = Path(config['paths']['targets_output'])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "csv_verification.log"
    setup_logging(log_file)
    
    logger.info("="*80)
    logger.info("CSV File and Column Verification")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Raw data path: {config['paths']['raw_data']}")
    logger.info(f"Log file: {log_file}")
    
    raw_data_path = Path(config['paths']['raw_data'])
    
    # Verify each dataset
    all_results = {}
    for dataset_name in ['apples', 'shhs', 'mros', 'stages']:
        if dataset_name not in config['tasks']:
            logger.warning(f"⚠️  {dataset_name} not found in config tasks")
            continue
        
        dataset_config = config['tasks'][dataset_name]
        results = verify_dataset(dataset_name, dataset_config, raw_data_path)
        all_results[dataset_name] = results
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*80)
    
    all_ok = True
    for dataset_name, results in all_results.items():
        status = "✅ PASS" if (results['files_ok'] and results['columns_ok']) else "❌ FAIL"
        logger.info(f"\n{dataset_name.upper()}: {status}")
        logger.info(f"  Tasks verified: {len(results['tasks_verified'])}")
        
        if results['missing_files']:
            all_ok = False
            logger.error(f"  Missing files ({len(results['missing_files'])}):")
            for f in results['missing_files']:
                logger.error(f"    - {f}")
        
        if results['missing_columns']:
            all_ok = False
            logger.error(f"  Missing columns in {len(results['missing_columns'])} files:")
            for task, cols in results['missing_columns'].items():
                logger.error(f"    - {task}: {cols}")
    
    logger.info("\n" + "="*80)
    if all_ok:
        logger.info("✅ All verifications passed! Ready for target extraction.")
        return 0
    else:
        logger.error("❌ Verification failed. Please fix issues before extraction.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
