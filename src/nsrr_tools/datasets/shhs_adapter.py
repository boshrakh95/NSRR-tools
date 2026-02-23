"""SHHS dataset adapter.

SHHS (Sleep Heart Health Study) is a multi-site prospective cohort study:
- Visit 1 (SHHS1): ~5800 subjects, 1995-1998
- Visit 2 (SHHS2): ~4000 subjects, 2001-2003
- Multi-site with extensive cardiovascular follow-up

Data structure:
- EDFs: Multiple directory structures per visit
- Annotations: NSRR XML format in separate archives
- Metadata: Multiple CSV files (harmonized, CVD summary, HRV, etc.)

Metadata files (from nocturn):
- harmonized: nsrr_age, nsrr_sex, nsrr_race, nsrr_bmi, PSG metrics
- shhs1/shhs2-dataset: Visit-specific questionnaires, AHI (rdi3p)
- cvd-summary: Cardiovascular outcomes
- hrv-summary: Heart rate variability (optional)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import xml.etree.ElementTree as ET
from loguru import logger

from .base_adapter import BaseNSRRAdapter


class SHHSAdapter(BaseNSRRAdapter):
    """Adapter for SHHS dataset."""
    
    def __init__(self, config, visit: int = 1):
        """Initialize SHHS adapter.
        
        Args:
            config: Config object
            visit: Visit number (1 or 2)
        """
        super().__init__(config, 'shhs')
        self.visit = visit
        
        # Metadata files (from nocturn)
        self.metadata_files = {
            'harmonized': 'shhs-harmonized-dataset-0.21.0.csv',
            'main': f'shhs{visit}-dataset-0.21.0.csv',
            'cvd': 'shhs-cvd-summary-dataset-0.21.0.csv',
            # 'hrv': f'shhs{visit}-hrv-summary-0.21.0.csv',  # optional
        }
        
        # Subject ID column (SHHS uses nsrrid)
        self.subject_id_col = 'nsrrid'
        
        # Phenotype columns (from multiple files)
        # harmonized: nsrr_age, nsrr_sex, nsrr_race, nsrr_bmi, PSG metrics
        # main: rdi3p (AHI), ess_s2, rest10/ms204c (rested morning)
        # cvd: any_cvd, cvd_death
        self.phenotype_cols = [
            'nsrrid', 'visitnumber',
            # From harmonized
            'nsrr_age', 'nsrr_sex', 'nsrr_race', 'nsrr_bmi', 'nsrr_current_smoker',
            'nsrr_ttleffsp_f1', 'nsrr_phrnumar_f1', 'nsrr_pctdursp_sr', 'nsrr_pctdursp_s3',
            # From main dataset
            'rdi3p',  # AHI
            'ess_s2' if visit == 2 else 'rest10',
        ]
    
    def find_edf_files(self) -> List[Tuple[str, Path]]:
        """Find all SHHS EDF files.
        
        SHHS structure varies by visit and extraction.
        Subject ID extracted from EDF filename.
        
        Returns:
            List of (subject_id, edf_path) tuples
        """
        edf_files = []
        
        # Try sample directory first (for testing)
        if 'sample' in self.dataset_paths:
            sample_path = self.dataset_paths['sample']
            if sample_path.exists():
                logger.info(f"Checking SHHS visit {self.visit} sample directory")
                sample_edfs = list(sample_path.rglob('*.edf'))
                
                for edf_path in sample_edfs:
                    subject_id = self._extract_subject_id_from_filename(edf_path.stem)
                    if subject_id:
                        edf_files.append((subject_id, edf_path))
                
                if edf_files:
                    # Filter duplicates (prefer base file over _1, _2 versions)
                    edf_files = self._filter_duplicate_edfs(edf_files)
                    logger.info(f"Found {len(edf_files)} SHHS{self.visit} EDF files in sample")
                    return edf_files
        
        # Original structure
        original_path = self.dataset_paths.get('original')
        if not original_path or not original_path.exists():
            logger.warning(f"SHHS original path does not exist: {original_path}")
            return []
        
        # Search for EDFs
        for edf_path in original_path.rglob('*.edf'):
            subject_id = self._extract_subject_id_from_filename(edf_path.stem)
            if subject_id:
                edf_files.append((subject_id, edf_path))
        
        # Filter duplicates
        edf_files = self._filter_duplicate_edfs(edf_files)
        logger.info(f"Found {len(edf_files)} SHHS{self.visit} EDF files")
        return edf_files
    
    def _extract_subject_id_from_filename(self, filename: str) -> Optional[str]:
        """Extract subject ID from SHHS EDF filename.
        
        SHHS filenames: shhs1-200001.edf, shhs2-200001.edf
        Or: 200001.edf (just the ID)
        
        Args:
            filename: EDF filename without extension
        
        Returns:
            Subject ID (nsrrid) or None
        """
        # Remove _1, _2 suffix if present
        base_name = filename.split('_')[0]
        
        # Try patterns
        if base_name.startswith(f'shhs{self.visit}-'):
            return base_name.replace(f'shhs{self.visit}-', '')
        elif base_name.startswith('shhs'):
            # Extract number after 'shhs'
            parts = base_name.split('-')
            if len(parts) > 1:
                return parts[1]
        elif base_name.isdigit():
            return base_name
        
        return None
    
    def _filter_duplicate_edfs(self, edf_files: List[Tuple[str, Path]]) -> List[Tuple[str, Path]]:
        """Filter duplicate EDFs, preferring base file over numbered versions.
        
        Priority: X.edf > X_1.edf > X_2.edf
        
        Args:
            edf_files: List of (subject_id, edf_path) tuples
        
        Returns:
            Filtered list with one file per subject
        """
        subject_files = {}
        
        for subject_id, edf_path in edf_files:
            stem = edf_path.stem
            
            # Determine priority (lower is better)
            if '_' not in stem:
                priority = 0  # Base file (X.edf)
            else:
                try:
                    suffix = stem.split('_')[-1]
                    priority = int(suffix)  # X_1.edf → priority 1
                except (ValueError, IndexError):
                    priority = 99  # Unknown suffix
            
            # Keep file with lowest priority
            if subject_id not in subject_files or priority < subject_files[subject_id][1]:
                subject_files[subject_id] = (edf_path, priority)
        
        # Return sorted list
        result = [(sid, path) for sid, (path, _) in subject_files.items()]
        return sorted(result, key=lambda x: x[0])
    
    def find_annotation_file(self, subject_id: str) -> Optional[Path]:
        """Find SHHS annotation file (NSRR XML format).
        
        Args:
            subject_id: Subject identifier (nsrrid)
        
        Returns:
            Path to XML annotation file, or None
        """
        annotations_path = self.dataset_paths.get('annotations')
        if not annotations_path or not annotations_path.exists():
            return None
        
        # SHHS XML pattern: shhs1-200001-nsrr.xml
        xml_pattern = f'shhs{self.visit}-{subject_id}-nsrr.xml'
        
        for xml_path in annotations_path.rglob(xml_pattern):
            return xml_path
        
        return None
    
    def parse_annotations(self, annotation_path: Path) -> Dict[str, Any]:
        """Parse NSRR XML annotations for SHHS.
        
        Args:
            annotation_path: Path to XML file
        
        Returns:
            Dictionary with stages, events, duration, format
        """
        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
            return {'stages': [], 'events': [], 'duration': 0, 'format': 'error'}
        
        stages = []
        events = []
        
        for scored_event in root.findall('.//ScoredEvent'):
            event_type = scored_event.find('EventType')
            event_concept = scored_event.find('EventConcept')
            start = scored_event.find('Start')
            duration_elem = scored_event.find('Duration')
            
            # Sleep stages
            if event_concept is not None and 'Stage' in event_concept.text:
                start_time = float(start.text) if start is not None else 0
                stage_label = event_concept.text
                
                # Map to standard stage numbers
                stage_map = {
                    'Stage 1 sleep|1': 1,
                    'Stage 2 sleep|2': 2,
                    'Stage 3 sleep|3': 3,
                    'Stage 4 sleep|4': 4,
                    'REM sleep|5': 5,
                    'Wake|0': 0,
                    'Unscored|9': -1
                }
                
                stage_num = stage_map.get(stage_label, -1)
                stages.append({
                    'start': start_time,
                    'stage': stage_num,
                    'label': stage_label
                })
            else:
                # Other events
                if event_concept is not None and start is not None and duration_elem is not None:
                    events.append({
                        'type': event_type.text if event_type is not None else 'Unknown',
                        'concept': event_concept.text,
                        'start': float(start.text),
                        'duration': float(duration_elem.text)
                    })
        
        stages.sort(key=lambda x: x['start'])
        
        total_duration = stages[-1]['start'] + 30 if stages else 0
        
        return {
            'stages': stages,
            'events': events,
            'duration': total_duration,
            'format': 'nsrr-xml',
            'num_epochs': len(stages)
        }
    
    def load_metadata(self) -> pd.DataFrame:
        """Load and merge SHHS metadata from multiple CSV files.
        
        Returns:
            DataFrame with merged subject metadata
        """
        datasets_path = self.dataset_paths['datasets']
        dfs_to_merge = []
        
        # Load each available file
        for file_key, filename in self.metadata_files.items():
            file_path = datasets_path / filename
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    logger.info(f"Loaded SHHS {file_key}: {len(df)} subjects, {len(df.columns)} columns")
                    dfs_to_merge.append(df)
                except Exception as e:
                    logger.error(f"Error loading {file_key}: {e}")
            else:
                logger.warning(f"{file_key} not found: {file_path}")
        
        # Merge dataframes
        if not dfs_to_merge:
            logger.error("No metadata files found")
            return pd.DataFrame()
        
        if len(dfs_to_merge) == 1:
            df = dfs_to_merge[0]
        else:
            df = dfs_to_merge[0]
            for df_next in dfs_to_merge[1:]:
                merge_key = self.subject_id_col
                
                df = df.merge(df_next, on=merge_key, how='outer', suffixes=('', '_dup'))
                
                # Remove duplicate columns
                dup_cols = [col for col in df.columns if col.endswith('_dup')]
                if dup_cols:
                    df = df.drop(columns=dup_cols)
        
        logger.info(f"Merged SHHS metadata: {len(df)} subjects, {len(df.columns)} columns")
        
        # Check for expected columns
        missing_cols = [col for col in self.phenotype_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing expected columns: {missing_cols}")
        
        return df
    
    def get_subject_id_column(self) -> str:
        """Get the name of the subject ID column.
        
        Returns:
            'nsrrid'
        """
        return self.subject_id_col
    
    def extract_subject_metadata(self, subject_id: str, 
                                metadata_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract metadata for a specific SHHS subject.
        
        Args:
            subject_id: Subject identifier (nsrrid)
            metadata_df: Full metadata DataFrame
        
        Returns:
            Dictionary with subject metadata
        """
        subject_df = metadata_df[metadata_df[self.subject_id_col] == subject_id]
        
        if subject_df.empty:
            logger.warning(f"No metadata found for subject {subject_id}")
            return {}
        
        return subject_df.iloc[0].to_dict()
