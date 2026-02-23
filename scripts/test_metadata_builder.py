"""Test metadata builder on sample data."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nsrr_tools.utils.config import Config
from nsrr_tools.core.metadata_builder import MetadataBuilder
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")


def main():
    """Test metadata builder."""
    print("\n" + "="*80)
    print("TESTING METADATA BUILDER")
    print("="*80 + "\n")
    
    # Initialize
    config = Config()
    builder = MetadataBuilder(config)
    
    # Build metadata for STAGES only (since it's the only adapter implemented)
    print("Building metadata for STAGES (sample data)...")
    try:
        metadata_df = builder.build_metadata(
            datasets=['stages'],
            force_rebuild=True,
            use_cache=False
        )
        
        print(f"\n✓ Metadata built successfully!")
        print(f"  Total subjects: {len(metadata_df)}")
        print(f"  Columns: {len(metadata_df.columns)}")
        
        # Show sample
        print("\nFirst few rows:")
        print(metadata_df.head())
        
        # Show summary
        print()
        builder.print_summary(metadata_df)
        
        # Show channel distribution
        if 'channels' in metadata_df.columns:
            print("\nSample channel lists:")
            for idx, row in metadata_df.head(3).iterrows():
                if row.get('has_edf', False):
                    print(f"  {row['subject_code']}: {row.get('num_channels', 0)} channels")
                    if 'channels' in row:
                        channels = str(row['channels']).split(',')[:5]
                        print(f"    {', '.join(channels)}...")
        
    except Exception as e:
        logger.error(f"Failed to build metadata: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
