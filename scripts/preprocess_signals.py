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
from loguru import logger
from tqdm import tqdm
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nsrr_tools.core.signal_processor import SignalProcessor
from nsrr_tools.core.annotation_processor import AnnotationProcessor
from nsrr_tools.utils.config import Config
from nsrr_tools.datasets.stages_adapter import STAGESAdapter
from nsrr_tools.datasets.shhs_adapter import SHHSAdapter
from nsrr_tools.datasets.apples_adapter import APPLESAdapter
from nsrr_tools.datasets.mros_adapter import MrOSAdapter


class PreprocessingPipeline:
    """Main preprocessing pipeline for NSRR data."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize pipeline.
        
        Args:
            config_path: Path to preprocessing config YAML (optional)
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
        
        # Initialize adapters
        self.adapters = {
            'stages': STAGESAdapter(self.config),
            'shhs': SHHSAdapter(self.config),
            'apples': APPLESAdapter(self.config),
            'mros': MrOSAdapter(self.config)
        }
        
        logger.info("Preprocessing pipeline initialized")
    
    def process_dataset(
        self,
        dataset_name: str,
        max_subjects: Optional[int] = None,
        skip_existing: bool = True
    ):
        """Process all subjects in a dataset.
        
        Args:
            dataset_name: Dataset name (stages, shhs, apples, mros)
            max_subjects: Maximum number of subjects to process (for testing)
            skip_existing: Skip subjects with existing output files
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
        
        # Filter to dataset
        dataset_df = metadata_df[metadata_df['dataset'] == dataset_name].copy()
        
        if len(dataset_df) == 0:
            logger.warning(f"No subjects found for dataset {dataset_name}")
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
            
            # Skip if exists
            if skip_existing and hdf5_path.exists() and annot_path.exists():
                logger.debug(f"Skipping {subject_id} (outputs exist)")
                results.append({
                    'subject_id': subject_id,
                    'dataset': dataset_name,
                    'status': 'skipped',
                    'reason': 'exists'
                })
                continue
            
            try:
                # Get file paths from metadata
                edf_path = Path(row['edf_path'])
                annotation_path_str = row.get('annotation_path', None)
                
                if not edf_path.exists():
                    logger.warning(f"EDF not found for {subject_id}: {edf_path}")
                    results.append({
                        'subject_id': subject_id,
                        'dataset': dataset_name,
                        'status': 'failed',
                        'reason': 'edf_not_found'
                    })
                    continue
                
                # Process signal
                signal_result = self.signal_processor.process_edf(
                    edf_path=edf_path,
                    output_path=hdf5_path
                )
                
                # Process annotation if available
                annotation_result = {'success': False, 'reason': 'no_annotation'}
                
                if annotation_path_str and annotation_path_str != 'None':
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
                
                # Combine results
                result = {
                    'subject_id': subject_id,
                    'dataset': dataset_name,
                    'status': 'success' if signal_result['success'] else 'failed',
                    'signal_channels': signal_result.get('channels_processed', 0),
                    'annotation_epochs': annotation_result.get('num_epochs', 0),
                    'duration_hours': signal_result.get('duration_hours', 0),
                    'hdf5_size_mb': signal_result.get('output_size_mb', 0),
                    'has_annotations': annotation_result.get('success', False)
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
                logger.info(f"  Average duration: {successful['duration_hours'].mean():.1f} hours")
                logger.info(f"  With annotations: {successful['has_annotations'].sum()}")
                logger.info(f"  Total HDF5 size: {successful['hdf5_size_mb'].sum():.1f} MB")
        
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
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=args.log_level
    )
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline(config_path=args.config)
    
    # Process datasets
    if args.dataset == 'all':
        for dataset in ['stages', 'shhs', 'apples', 'mros']:
            pipeline.process_dataset(
                dataset_name=dataset,
                max_subjects=args.max_subjects,
                skip_existing=args.skip_existing
            )
    else:
        pipeline.process_dataset(
            dataset_name=args.dataset,
            max_subjects=args.max_subjects,
            skip_existing=args.skip_existing
        )


if __name__ == '__main__':
    main()
