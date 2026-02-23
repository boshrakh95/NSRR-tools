"""Test script for APPLES adapter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nsrr_tools.utils.config import Config
from nsrr_tools.datasets.apples_adapter import APPLESAdapter


def main():
    """Test APPLES adapter."""
    print("=" * 80)
    print("TESTING APPLES ADAPTER")
    print("=" * 80)
    
    config = Config()
    
    print("\n1. Creating APPLES adapter...")
    adapter = APPLESAdapter(config)
    
    print("\n2. Finding EDF files...")
    edf_files = adapter.find_edf_files()
    print(f"   Found {len(edf_files)} EDF files")
    if edf_files:
        print(f"   First 5 subjects: {[sid for sid, _ in edf_files[:5]]}")
    
    print("\n3. Loading metadata...")
    metadata = adapter.load_metadata()
    print(f"   Loaded metadata: {len(metadata)} subjects, {len(metadata.columns)} columns")
    print(f"   Subject ID column: {adapter.get_subject_id_column()}")
    
    if not metadata.empty:
        print(f"\n   First few rows:")
        print(metadata.head())
        
        print(f"\n   Expected phenotype columns:")
        for col in adapter.phenotype_cols:
            status = "✓" if col in metadata.columns else "✗ MISSING"
            print(f"      {status} {col}")
    
    if edf_files:
        print(f"\n4. Testing annotation file finding...")
        test_subject = edf_files[0][0]
        annot_path = adapter.find_annotation_file(test_subject)
        if annot_path:
            print(f"   ✓ Found annotation for {test_subject}: {annot_path.name}")
            
            print(f"\n5. Testing annotation parsing...")
            annotations = adapter.parse_annotations(annot_path)
            print(f"   Format: {annotations['format']}")
            print(f"   Epochs: {annotations['num_epochs']}")
            print(f"   Duration: {annotations['duration']:.1f}s")
        else:
            print(f"   ✗ No annotation found for {test_subject}")
    
    print("\n" + "=" * 80)
    print("APPLES ADAPTER TEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
