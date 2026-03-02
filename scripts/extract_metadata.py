#!/usr/bin/env python3
"""Extract unified metadata for NSRR datasets.

Works for any combination of: mros, shhs, stages, apples
"""

import argparse
from pathlib import Path
import sys

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from loguru import logger
from nsrr_tools.utils.config import Config
from nsrr_tools.core.metadata_builder import MetadataBuilder


def main():
    parser = argparse.ArgumentParser(description='Extract NSRR metadata')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['mros'],
        choices=['mros', 'shhs', 'stages', 'apples'],
        help='Datasets to process (default: mros)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit to first N subjects per dataset (for testing)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output base directory (default: from config unified.metadata path)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rebuild even if cached'
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
    
    # Initialize config first
    config = Config()
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
        builder = MetadataBuilder(config, output_dir=output_dir)
    else:
        # Use config-based path (same as preprocess_signals.py expects)
        builder = MetadataBuilder(config)
        output_dir = builder.unified_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'metadata_extraction.log'
    logger.add(log_file, level="DEBUG")
    
    logger.info("="*80)
    logger.info("NSRR Metadata Extraction")
    logger.info("="*80)
    logger.info(f"Datasets: {', '.join(args.datasets)}")
    logger.info(f"Output: {builder.unified_path}")
    if args.limit:
        logger.info(f"Limit: {args.limit} subjects per dataset")
    logger.info(f"Force rebuild: {args.force}")
    logger.info(f"Use cache: {not args.no_cache}")
    
    # Build metadata
    try:
        logger.info("\nStarting metadata extraction...")
        metadata_df = builder.build_metadata(
            datasets=args.datasets,
            force_rebuild=args.force,
            use_cache=not args.no_cache,
            limit=args.limit
        )
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("Extraction Complete!")
        logger.info("="*80)
        builder.print_summary(metadata_df)
        
        logger.success(f"\n✓ Metadata saved to: {output_dir / 'unified_metadata.parquet'}")
        logger.success(f"✓ Total subjects: {len(metadata_df)}")
        
    except Exception as e:
        logger.exception(f"Error during metadata extraction: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
