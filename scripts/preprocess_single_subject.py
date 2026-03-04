#!/usr/bin/env python3
"""Process a single subject's EDF file to HDF5 format.

This script is designed to be called by parallel processing jobs.
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from nsrr_tools.utils.config import Config
from nsrr_tools.core.signal_processor import SignalProcessor
from nsrr_tools.core.annotation_processor import AnnotationProcessor
from nsrr_tools.datasets.stages_adapter import STAGESAdapter
from nsrr_tools.datasets.shhs_adapter import SHHSAdapter
from nsrr_tools.datasets.apples_adapter import APPLESAdapter
from nsrr_tools.datasets.mros_adapter import MrOSAdapter


def get_adapter(config, dataset_name):
    """Get dataset adapter."""
    adapters = {
        'stages': STAGESAdapter,
        'shhs': SHHSAdapter,
        'apples': APPLESAdapter,
        'mros': MrOSAdapter
    }
    return adapters[dataset_name.lower()](config)


def main():
    parser = argparse.ArgumentParser(description='Process single subject EDF to HDF5')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--subject-id', required=True, help='Subject ID')
    parser.add_argument('--edf-path', required=True, help='Path to EDF file')
    parser.add_argument('--skip-existing', action='store_true', help='Skip if output exists')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    args = parser.parse_args()
    
    # Configure logging (minimal for parallel jobs)
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:SS}</green> | <level>{level: <8}</level> | {message}",
        level=args.log_level,
        filter=lambda record: record["level"].name in ["SUCCESS", "ERROR", "WARNING"]
    )
    
    try:
        # Initialize components
        config = Config()
        adapter = get_adapter(config, args.dataset)
        signal_processor = SignalProcessor(config)
        annotation_processor = AnnotationProcessor(adapter)
        
        # Get paths config
        preprocess_config = config.preprocessing_params
        base_output = Path(preprocess_config['paths']['base_output'])
        
        # Construct output paths
        hdf5_dir = base_output / preprocess_config['paths'][args.dataset]['hdf5_signals']
        annot_dir = base_output / preprocess_config['paths'][args.dataset]['annotations']
        hdf5_dir.mkdir(parents=True, exist_ok=True)
        annot_dir.mkdir(parents=True, exist_ok=True)
        
        # Output file paths
        edf_path = Path(args.edf_path)
        hdf5_path = hdf5_dir / f"{edf_path.stem}.h5"
        annot_path = annot_dir / f"{edf_path.stem}_annotations.json"
        
        # Skip if exists
        if args.skip_existing and hdf5_path.exists() and annot_path.exists():
            logger.debug(f"Skipping {args.subject_id}: output exists")
            sys.exit(0)
        
        # Process signals
        signal_result = signal_processor.process_edf(edf_path, hdf5_path)
        
        if not signal_result['success']:
            logger.error(f"Failed to process signals for {args.subject_id}: {signal_result.get('error', 'Unknown')}")
            sys.exit(1)
        
        # Process annotations
        try:
            annot_file = adapter.find_annotation_file(args.subject_id)
            if annot_file:
                annot_data = annotation_processor.process_annotations(annot_file, annot_path)
                logger.success(f"✓ {args.subject_id}: {signal_result['channels_processed']} channels, {len(annot_data.get('stages', []))} stages")
            else:
                logger.warning(f"⚠ {args.subject_id}: No annotations found")
        except Exception as e:
            logger.warning(f"⚠ {args.subject_id}: Annotation processing failed: {e}")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"✗ {args.subject_id}: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
