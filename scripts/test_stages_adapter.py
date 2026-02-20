"""Test STAGES adapter with actual data."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nsrr_tools.utils.config import Config
from nsrr_tools.datasets.stages_adapter import STAGESAdapter

def main():
    print("=" * 80)
    print("STAGES Adapter Test")
    print("=" * 80)
    
    # Load config
    print("\n1. Loading configuration...")
    config = Config()
    print("✓ Config loaded")
    
    # Initialize adapter
    print("\n2. Initializing STAGES adapter...")
    adapter = STAGESAdapter(config)
    print(f"✓ Adapter: {adapter}")
    print(f"  Dataset paths: {adapter.dataset_paths}")
    
    # Validate file structure
    print("\n3. Validating file structure...")
    validation = adapter.validate_file_structure()
    print(f"  Valid: {validation['valid']}")
    print(f"  Paths exist: {validation['paths_exist']}")
    print(f"  Number of EDFs: {validation['num_edfs']}")
    print(f"  Has metadata: {validation['has_metadata']}")
    
    if validation['errors']:
        print("\n  Errors:")
        for error in validation['errors']:
            print(f"    - {error}")
    
    # Find EDF files
    print("\n4. Finding EDF files...")
    edf_files = adapter.find_edf_files()
    print(f"  Found {len(edf_files)} EDF files")
    
    if edf_files:
        print(f"\n  First 5 subjects:")
        for subject_id, edf_path in edf_files[:5]:
            print(f"    {subject_id}: {edf_path.name}")
    
    # Load metadata
    print("\n5. Loading metadata...")
    metadata_df = adapter.load_metadata()
    print(f"  Loaded {len(metadata_df)} subjects")
    
    if not metadata_df.empty:
        print(f"  Columns: {list(metadata_df.columns)[:10]}...")
    
    # Test with one subject
    if edf_files and not metadata_df.empty:
        print("\n6. Testing with first subject...")
        test_subject_id, test_edf_path = edf_files[0]
        print(f"  Subject: {test_subject_id}")
        print(f"  EDF: {test_edf_path}")
        
        # Find annotation
        annot_path = adapter.find_annotation_file(test_subject_id)
        if annot_path:
            print(f"  Annotation: {annot_path}")
            
            # Parse annotation
            print("\n  Parsing annotations...")
            annotations = adapter.parse_annotations(annot_path)
            print(f"    Format: {annotations.get('format')}")
            print(f"    Duration: {annotations.get('duration')} seconds")
            print(f"    Number of epochs: {annotations.get('num_epochs')}")
            print(f"    Number of events: {len(annotations.get('events', []))}")
            
            if annotations.get('stages'):
                stages = annotations['stages']
                print(f"\n    First 5 stages:")
                for stage in stages[:5]:
                    print(f"      {stage['start']:.1f}s: Stage {stage['stage']} ({stage['label']})")
        else:
            print(f"  WARNING: No annotation found")
        
        # Extract subject metadata
        print("\n  Extracting subject metadata...")
        subject_meta = adapter.extract_subject_metadata(test_subject_id, metadata_df)
        print(f"    Found: {subject_meta.get('found')}")
        
        if subject_meta.get('found'):
            print(f"    Sex: {subject_meta.get('sex')}")
            print(f"    AHI: {subject_meta.get('ahi')}")
            print(f"    ISI score: {subject_meta.get('isi_score')}")
            
            if subject_meta.get('labels'):
                print(f"    Labels: {subject_meta['labels']}")
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()
