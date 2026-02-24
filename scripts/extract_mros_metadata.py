#!/usr/bin/env python3
"""Extract unified metadata for MrOS dataset.

This script:
1. Scans all MrOS EDF files to extract channel information
2. Extracts sampling rates per modality
3. Merges with phenotypic data (demographics, AHI, cognitive scores)
4. Saves unified metadata to parquet file

Usage:
    python scripts/extract_mros_metadata.py [--subset N]
    
Options:
    --subset N : Process only first N subjects (for testing)
    --force    : Force rebuild even if cache exists
"""

import argparse
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from loguru import logger
from nsrr_tools.utils.config import Config
from nsrr_tools.core.metadata_builder import MetadataBuilder


def main():
    parser = argparse.ArgumentParser(description='Extract MrOS metadata')
    parser.add_argument(
        '--subset',
        type=int,
        default=None,
        help='Process only first N subjects (for testing)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rebuild even if cache exists'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/scratch/boshra95/psg_metadata/mros',
        help='Output directory for metadata'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Do not use cached dataset metadata'
    )
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'metadata_extraction.log'
    logger.add(log_file, level="DEBUG")
    
    logger.info("="*80)
    logger.info("MrOS Metadata Extraction")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Force rebuild: {args.force}")
    logger.info(f"Use cache: {not args.no_cache}")
    if args.subset:
        logger.info(f"Processing subset: {args.subset} subjects")
    
    # Load configuration
    logger.info(f"Loading config...")
    config = Config()  # Will auto-detect configs directory
    
    # Initialize metadata builder
    builder = MetadataBuilder(config, output_dir=output_dir)
    
    # Build metadata for MrOS only
    try:
        logger.info("\nStarting metadata extraction...")
        metadata_df = builder.build_metadata(
            datasets=['mros'],
            force_rebuild=args.force,
            use_cache=not args.no_cache
        )
        
        # Apply subset if requested
        if args.subset and args.subset < len(metadata_df):
            logger.info(f"\nApplying subset: {args.subset} subjects")
            metadata_df = metadata_df.head(args.subset)
            subset_path = output_dir / f'unified_metadata_subset{args.subset}.parquet'
            metadata_df.to_parquet(subset_path, index=False)
            logger.info(f"Saved subset to: {subset_path}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("Metadata extraction complete!")
        logger.info("="*80)
        builder.print_summary(metadata_df)
        
        # Save detailed summary
        summary = builder.get_summary_statistics(metadata_df)
        summary_file = output_dir / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write("MrOS Metadata Summary\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total subjects: {summary['total_subjects']}\n\n")
            
            f.write("Dataset breakdown:\n")
            for dataset, info in summary['datasets'].items():
                f.write(f"  {dataset}: {info['n_subjects']} subjects "
                       f"({info['with_edf']} with EDF)\n")
            
            f.write("\nChannel coverage:\n")
            for modality, count in summary['channel_coverage'].items():
                pct = 100 * count / summary['total_subjects']
                f.write(f"  {modality}: {count} ({pct:.1f}%)\n")
            
            if 'usability' in summary:
                f.write(f"\nUsable subjects: {summary['usability'].get('usable_subjects', 0)} "
                       f"({summary['usability'].get('usable_percent', 0):.1f}%)\n")
        
        logger.success(f"\nSummary saved to: {summary_file}")
        logger.success(f"Metadata saved to: {output_dir / 'unified_metadata.parquet'}")
        
    except Exception as e:
        logger.exception(f"Error during metadata extraction: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
