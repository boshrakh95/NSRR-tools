#!/usr/bin/env python3
"""
Preprocess NSRR EDF Files to SleepFM Format
==========================================

Main script for converting raw EDF files to HDF5 with sleep stage annotations.

Usage:
    python scripts/preprocess_signals.py --dataset stages --max-subjects 10
    python scripts/preprocess_signals.py --dataset all --skip-existing

Author: NSRR Preprocessing Pipeline
Date: February 2026
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm
import yaml
import gc

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring disabled")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nsrr_tools.core.signal_processor import SignalProcessor
from nsrr_tools.core.annotation_processor import AnnotationProcessor
from nsrr_tools.utils.config import Config
from nsrr_tools.datasets.stages_adapter import STAGESAdapter
from nsrr_tools.datasets.shhs_adapter import SHHSAdapter
from nsrr_tools.datasets.apples_adapter import APPLESAdapter
from nsrr_tools.datasets.mros_adapter import MrOSAdapter


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 2)
    return 0.0


class PreprocessingPipeline:
    """Main preprocessing pipeline for NSRR data."""
    
    def __init__(self, config_path: Optional[Path] = None, mros_visit: Optional[int] = None):
        """Initialize pipeline.

        Args:
            config_path: Path to preprocessing config YAML (optional)
            mros_visit: Which MrOS visit to process (1 or 2), or None for all.
        """
        # Load configurations
        self.config = Config()
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                self.preprocess_config = yaml.safe_load(f)
        else:
            # Use default config
            default_config_path = Path(__file__).parent.parent / 'configs' / 'preprocessing_params.yaml'
            with open(default_config_path, 'r') as f:
                self.preprocess_config = yaml.safe_load(f)
        
        # Initialize processors
        self.signal_processor = SignalProcessor(self.config)
        
        # Initialize adapters (MrOS visit is configurable via --mros-visit)
        self.adapters = {
            'stages': STAGESAdapter(self.config),
            'shhs': SHHSAdapter(self.config),
            'apples': APPLESAdapter(self.config),
            'mros': MrOSAdapter(self.config, visit=mros_visit),
        }
        
        logger.info("Preprocessing pipeline initialized")
    
    def process_dataset(
        self,
        dataset_name: str,
        max_subjects: Optional[int] = None,
        skip_existing: bool = True,
        reprocess_annotations: bool = False
    ):
        """Process all subjects in a dataset.
        
        Args:
            dataset_name: Dataset name (stages, shhs, apples, mros)
            max_subjects: Maximum number of subjects to process (for testing)
            skip_existing: Skip subjects with existing output files
            reprocess_annotations: Reprocess annotations even if they exist (keeps HDF5)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing dataset: {dataset_name.upper()}")
        logger.info(f"{'='*80}\n")
        
        # Get adapter
        if dataset_name not in self.adapters:
            logger.error(f"Unknown dataset: {dataset_name}")
            logger.info(f"Available datasets: {list(self.adapters.keys())}")
            return
        
        adapter = self.adapters[dataset_name]
        annotation_processor = AnnotationProcessor(adapter)
        
        # Load metadata from Phase 1
        # Try unified location first, fall back to psg_metadata for backward compatibility
        unified_metadata_path = Path(self.config.paths['unified']['metadata']) / 'unified_metadata.parquet'
        legacy_metadata_path = Path('/scratch/boshra95/psg_metadata/unified_metadata.parquet')
        
        if unified_metadata_path.exists():
            metadata_path = unified_metadata_path
        elif legacy_metadata_path.exists():
            metadata_path = legacy_metadata_path
            logger.warning(f"Using legacy metadata location: {metadata_path}")
        else:
            logger.error(f"Metadata file not found at:")
            logger.error(f"  {unified_metadata_path}")
            logger.error(f"  {legacy_metadata_path}")
            logger.info("Please run Phase 1 metadata extraction first")
            return
        
        logger.info(f"Loading metadata from {metadata_path}...")
        metadata_df = pd.read_parquet(metadata_path)
        
        # Filter to dataset (case-insensitive match)
        dataset_name_upper = dataset_name.upper()
        dataset_df = metadata_df[metadata_df['dataset'].str.upper() == dataset_name_upper].copy()
        
        if len(dataset_df) == 0:
            logger.warning(f"No subjects found for dataset {dataset_name}")
            logger.debug(f"Available datasets in metadata: {metadata_df['dataset'].unique()}")
            return
        
        logger.info(f"Found {len(dataset_df)} subjects in metadata")
        
        # Filter to subjects with EDF files
        dataset_df = dataset_df[dataset_df['has_edf'] == True].copy()
        logger.info(f"  {len(dataset_df)} subjects have EDF files")
        
        # Limit number if specified
        if max_subjects:
            dataset_df = dataset_df.head(max_subjects)
            logger.info(f"  Processing first {max_subjects} subjects")
        
        # Get output paths
        base_output = Path(self.preprocess_config['paths']['base_output'])
        hdf5_dir = base_output / self.preprocess_config['paths'][dataset_name]['hdf5_signals']
        annot_dir = base_output / self.preprocess_config['paths'][dataset_name]['annotations']
        log_dir = base_output / self.preprocess_config['paths'][dataset_name]['logs']
        
        # Create directories
        hdf5_dir.mkdir(parents=True, exist_ok=True)
        annot_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directories:")
        logger.info(f"  HDF5: {hdf5_dir}")
        logger.info(f"  Annotations: {annot_dir}")
        logger.info(f"  Logs: {log_dir}")
        
        # Process subjects
        results = []
        
        pbar = tqdm(dataset_df.iterrows(), total=len(dataset_df), desc="Processing subjects")
        
        for idx, row in pbar:
            subject_id = row['subject_id']
            pbar.set_description(f"Processing {subject_id}")
            
            # Define output paths
            hdf5_path = hdf5_dir / f"{subject_id}.h5"
            annot_path = annot_dir / f"{subject_id}_stages.npy"
            
            # Skip logic
            if skip_existing:
                # If reprocessing annotations, only check if HDF5 exists
                if reprocess_annotations:
                    if hdf5_path.exists():
                        logger.debug(f"Skipping signal processing for {subject_id} (HDF5 exists), will check annotations")
                        skip_signal = True
                    else:
                        skip_signal = False
                    skip_annotation = False  # Always reprocess annotations
                else:
                    # Normal skip: both must exist
                    if hdf5_path.exists() and annot_path.exists():
                        logger.debug(f"Skipping {subject_id} (outputs exist)")
                        results.append({
                            'subject_id': subject_id,
                            'dataset': dataset_name,
                            'status': 'skipped',
                            'reason': 'exists'
                        })
                        continue
                    skip_signal = hdf5_path.exists()
                    skip_annotation = annot_path.exists()
            else:
                skip_signal = False
                skip_annotation = False
            
            try:
                # Get file paths from metadata
                edf_path_val = row.get('edf_path', None)
                if edf_path_val is None or (not isinstance(edf_path_val, str) and pd.isna(edf_path_val)):
                    logger.warning(f"No EDF path in metadata for {subject_id}, skipping")
                    results.append({'subject_id': subject_id, 'dataset': dataset_name,
                                    'status': 'failed', 'reason': 'no_edf_path_in_metadata'})
                    continue
                edf_path = Path(edf_path_val)
                annotation_path_raw = row.get('annotation_path', None)
                # Treat float NaN and the string 'nan' (from astype(str) conversion) as missing
                if annotation_path_raw is None or \
                   (isinstance(annotation_path_raw, float) and pd.isna(annotation_path_raw)) or \
                   annotation_path_raw == 'nan':
                    annotation_path_str = None
                else:
                    annotation_path_str = annotation_path_raw
                
                if not edf_path.exists():
                    logger.warning(f"EDF not found for {subject_id}: {edf_path}")
                    results.append({
                        'subject_id': subject_id,
                        'dataset': dataset_name,
                        'status': 'failed',
                        'reason': 'edf_not_found'
                    })
                    continue
                
                # Process signal (or load existing)
                if not skip_signal:
                    signal_result = self.signal_processor.process_edf(
                        edf_path=edf_path,
                        output_path=hdf5_path
                    )
                else:
                    # Load existing HDF5 metadata for reporting
                    logger.debug(f"Using existing HDF5 for {subject_id}")
                    signal_result = {
                        'success': True,
                        'channels_processed': 0,  # Unknown from existing file
                        'duration_hours': 0,      # Unknown from existing file
                        'output_size_mb': hdf5_path.stat().st_size / (1024**2) if hdf5_path.exists() else 0,
                        'note': 'existing_file'
                    }
                
                # Process annotation if available
                annotation_result = {'success': False, 'reason': 'no_annotation'}
                
                if not skip_annotation and annotation_path_str and annotation_path_str != 'None':
                    annotation_file = Path(annotation_path_str)
                    
                    if annotation_file.exists():
                        annotation_result = annotation_processor.process_annotations(
                            annotation_path=annotation_file,
                            edf_path=edf_path,
                            output_path=annot_path,
                            validate_sync=True
                        )
                    else:
                        logger.debug(f"Annotation file not found for {subject_id}")
                elif skip_annotation and annot_path.exists():
                    # Load existing annotation metadata
                    logger.debug(f"Using existing annotations for {subject_id}")
                    annot_array = np.load(annot_path)
                    scored = np.sum(annot_array >= 0)
                    unscored = np.sum(annot_array < 0)
                    annotation_result = {
                        'success': True,
                        'num_epochs': len(annot_array),
                        'scored_epochs': int(scored),
                        'unscored_epochs': int(unscored),
                        'scoring_coverage': float(scored / len(annot_array)) if len(annot_array) > 0 else 0,
                        'duration_hours': len(annot_array) * 30 / 3600,
                        'note': 'existing_file'
                    }
                
                # Combine results
                result = {
                    'subject_id': subject_id,
                    'dataset': dataset_name,
                    'status': 'success' if signal_result['success'] else 'failed',
                    'signal_channels': signal_result.get('channels_processed', 0),
                    'signal_duration_hours': signal_result.get('duration_hours', 0),
                    'annotation_epochs': annotation_result.get('num_epochs', 0),
                    'annotation_scored_epochs': annotation_result.get('scored_epochs', 0),
                    'annotation_unscored_epochs': annotation_result.get('unscored_epochs', 0),
                    'annotation_scoring_coverage': annotation_result.get('scoring_coverage', 0.0),
                    'annotation_duration_hours': annotation_result.get('duration_hours', 0),
                    'hdf5_size_mb': signal_result.get('output_size_mb', 0),
                    'has_annotations': annotation_result.get('success', False),
                    # Synchronization info
                    'sync_status': annotation_result.get('sync_status', 'no_annotation'),
                    'sync_difference_sec': annotation_result.get('difference_sec', None),
                    'sync_adjustment_epochs': annotation_result.get('adjustment_epochs', 0)
                }
                
                if signal_result['success']:
                    logger.success(f"✓ {subject_id}: {result['signal_channels']} channels, {result['annotation_epochs']} epochs")
                else:
                    logger.error(f"✗ {subject_id}: {signal_result.get('error', 'Unknown error')}")
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {subject_id}: {e}")
                results.append({
                    'subject_id': subject_id,
                    'dataset': dataset_name,
                    'status': 'failed',
                    'error': str(e)
                })
            
            # Explicitly free memory after each subject
            gc.collect()
            
            # Log memory usage periodically
            if PSUTIL_AVAILABLE and (len(results) % 10 == 0):
                mem_mb = get_memory_usage_mb()
                logger.info(f"Memory usage after {len(results)} subjects: {mem_mb:.1f} MB")
        
        # Save summary
        summary_df = pd.DataFrame(results)
        summary_path = log_dir / f'preprocessing_summary_{dataset_name}.csv'
        summary_df.to_csv(summary_path, index=False)
        
        # Print summary statistics
        logger.info(f"\n{'='*80}")
        logger.info(f"PREPROCESSING SUMMARY - {dataset_name.upper()}")
        logger.info(f"{'='*80}")
        logger.info(f"Total subjects processed: {len(results)}")
        logger.info(f"  Successful: {len(summary_df[summary_df['status'] == 'success'])}")
        logger.info(f"  Failed: {len(summary_df[summary_df['status'] == 'failed'])}")
        logger.info(f"  Skipped: {len(summary_df[summary_df['status'] == 'skipped'])}")
        
        if 'signal_channels' in summary_df.columns:
            successful = summary_df[summary_df['status'] == 'success']
            if len(successful) > 0:
                logger.info(f"\nSuccessful subjects:")
                logger.info(f"  Average channels: {successful['signal_channels'].mean():.1f}")
                logger.info(f"  Average signal duration: {successful['signal_duration_hours'].mean():.1f} hours")
                logger.info(f"  With annotations: {successful['has_annotations'].sum()}")
                logger.info(f"  Total HDF5 size: {successful['hdf5_size_mb'].sum():.1f} MB")
                
                # Synchronization summary
                if 'sync_status' in successful.columns:
                    annotated = successful[successful['has_annotations'] == True]
                    if len(annotated) > 0:
                        logger.info(f"\nSynchronization summary (annotated subjects):")
                        sync_counts = annotated['sync_status'].value_counts()
                        for status, count in sync_counts.items():
                            logger.info(f"  {status}: {count} subjects")
                        
                        # Show average difference for those that needed adjustment
                        adjusted = annotated[annotated['sync_status'].isin(['truncated', 'padded'])]
                        if len(adjusted) > 0:
                            avg_diff = adjusted['sync_difference_sec'].mean()
                            avg_adj = adjusted['sync_adjustment_epochs'].mean()
                            logger.info(f"  Average difference (adjusted): {avg_diff:.1f}s ({avg_adj:.1f} epochs)")
                        
                        # Scoring coverage statistics
                        if 'annotation_scoring_coverage' in annotated.columns:
                            avg_coverage = annotated['annotation_scoring_coverage'].mean()
                            min_coverage = annotated['annotation_scoring_coverage'].min()
                            max_coverage = annotated['annotation_scoring_coverage'].max()
                            logger.info(f"\nAnnotation scoring coverage:")
                            logger.info(f"  Average: {avg_coverage:.1%}")
                            logger.info(f"  Range: {min_coverage:.1%} - {max_coverage:.1%}")
                            
                            # Flag low-coverage subjects
                            low_coverage = annotated[annotated['annotation_scoring_coverage'] < 0.5]
                            if len(low_coverage) > 0:
                                logger.warning(f"  {len(low_coverage)} subjects with <50% scoring coverage")
        
        logger.info(f"\nSummary saved to: {summary_path}")
        logger.info(f"{'='*80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess NSRR EDF files to SleepFM HDF5 format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['stages', 'shhs', 'apples', 'mros', 'all'],
        help='Dataset to process'
    )
    
    parser.add_argument(
        '--max-subjects',
        type=int,
        default=None,
        help='Maximum number of subjects to process (for testing)'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip subjects with existing output files'
    )
    
    parser.add_argument(
        '--reprocess-annotations',
        action='store_true',
        default=False,
        help='Reprocess annotations even if they exist (keeps existing HDF5 signals)'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to preprocessing config YAML'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    parser.add_argument(
        '--mros-visit',
        type=int,
        default=None,
        choices=[1, 2],
        help='For MrOS only: which visit to preprocess (1 or 2). '
             'If omitted, processes all visits found in unified_metadata.'
    )

    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=args.log_level
    )
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline(config_path=args.config, mros_visit=args.mros_visit)
    
    # Process datasets
    if args.dataset == 'all':
        for dataset in ['stages', 'shhs', 'apples', 'mros']:
            pipeline.process_dataset(
                dataset_name=dataset,
                max_subjects=args.max_subjects,
                skip_existing=args.skip_existing,
                reprocess_annotations=args.reprocess_annotations
            )
    else:
        pipeline.process_dataset(
            dataset_name=args.dataset,
            max_subjects=args.max_subjects,
            skip_existing=args.skip_existing,
            reprocess_annotations=args.reprocess_annotations
        )


if __name__ == '__main__':
    main()
