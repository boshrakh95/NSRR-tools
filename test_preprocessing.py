#!/usr/bin/env python3
"""
Quick test of Phase 2 preprocessing on 1 STAGES subject.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from loguru import logger
from nsrr_tools.core.signal_processor import SignalProcessor
from nsrr_tools.core.annotation_processor import AnnotationProcessor
from nsrr_tools.utils.config import Config
from nsrr_tools.datasets.stages_adapter import STAGESAdapter

# Configure logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")

def main():
    logger.info("Testing Phase 2 preprocessing...")
    
    # Initialize
    config = Config()
    signal_processor = SignalProcessor(config)
    stages_adapter = STAGESAdapter(config)
    annotation_processor = AnnotationProcessor(stages_adapter)
    
    # Test with one STAGES subject
    # Use a subject we know has data
    test_edf = Path("/scratch/boshra95/nsrr_downloads/stages/original/STAGES PSGs/GSSA/GSSA00001.edf")
    test_annot = Path("/scratch/boshra95/nsrr_downloads/stages/original/STAGES PSGs/GSSA/GSSA00001.csv")
    
    if not test_edf.exists():
        logger.error(f"Test EDF not found: {test_edf}")
        logger.info("Please update the path to a valid STAGES EDF file")
        return 1
    
    # Output paths
    output_dir = Path("/tmp/test_preprocessing")
    output_dir.mkdir(exist_ok=True)
    
    hdf5_path = output_dir / "test_subject.h5"
    annot_path = output_dir / "test_subject_stages.npy"
    
    logger.info(f"Input EDF: {test_edf}")
    logger.info(f"Input annotation: {test_annot}")
    logger.info(f"Output HDF5: {hdf5_path}")
    logger.info(f"Output stages: {annot_path}")
    
    # Process signal
    logger.info("\n--- Processing Signal ---")
    signal_result = signal_processor.process_edf(
        edf_path=test_edf,
        output_path=hdf5_path
    )
    
    if signal_result['success']:
        logger.success(f"✓ Signal processing successful!")
        logger.info(f"  Channels: {signal_result['num_channels']}")
        logger.info(f"  Duration: {signal_result['duration_seconds']:.1f} sec")
        logger.info(f"  File size: {hdf5_path.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        logger.error(f"✗ Signal processing failed: {signal_result.get('error', 'Unknown')}")
        return 1
    
    # Process annotation
    if test_annot.exists():
        logger.info("\n--- Processing Annotation ---")
        annot_result = annotation_processor.process_annotations(
            annotation_path=test_annot,
            edf_path=test_edf,
            output_path=annot_path,
            validate_sync=True
        )
        
        if annot_result['success']:
            logger.success(f"✓ Annotation processing successful!")
            logger.info(f"  Epochs: {annot_result['num_epochs']}")
            logger.info(f"  Mismatch: {annot_result.get('mismatch_epochs', 0)} epochs")
        else:
            logger.warning(f"⚠ Annotation processing failed: {annot_result.get('error', 'Unknown')}")
    else:
        logger.warning(f"Annotation file not found: {test_annot}")
    
    logger.success("\n✓ Test completed!")
    logger.info(f"Outputs in: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
