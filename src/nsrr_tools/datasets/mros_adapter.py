"""MrOS dataset adapter.

MrOS (Osteoporotic Fractures in Men Study):
- Large cohort study of older men
- ~3,000 subjects with PSG
- Multiple visits

Data structure:
- EDFs: Visit-specific directories
- Annotations: Check format
- Metadata: Visit-specific CSV files

Metadata files (from nocturn):
- harmonized: nsrr_age, nsrr_sex, nsrr_race, nsrr_bmi, nsrr_ahi_hp3r_aasm15, PSG metrics
- mros-visit1/2-dataset: Questionnaires, medical history, PSG metrics
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import xml.etree.ElementTree as ET
from loguru import logger

from .base_adapter import BaseNSRRAdapter


class MrOSAdapter(BaseNSRRAdapter):
    """Adapter for MrOS dataset."""
    
    def __init__(self, config, visit: int = 1):
        """Initialize MrOS adapter.
        
        Args:
            config: Config object
            visit: Visit number (1 or 2)
        """
        super().__init__(config, 'mros')
        self.visit = visit
        
        # Metadata files (from nocturn)
        self.metadata_files = {
            'harmonized': f'mros-visit{visit}-harmonized-0.6.0.csv',
            'main': f'mros-visit{visit}-dataset-0.6.0.csv',
        }
        
        # Subject ID column
        self.subject_id_col = 'nsrrid'
        
        # Phenotype columns (from multiple files)
        self.phenotype_cols = [
            'nsrrid',
            # From harmonized
            'nsrr_age', 'nsrr_sex', 'nsrr_race', 'nsrr_bmi', 'nsrr_current_smoker',
            'nsrr_ahi_hp3r_aasm15',  # AHI
            'nsrr_phrnumar_f1',  # Arousal index
            # From main dataset
            'epepwort',  # ESS
            'pqpsqi',  # PSQI
        ]
    
    def find_edf_files(self) -> List[Tuple[str, Path]]:
        """Find all MrOS EDF files.
        
        Returns:
            List of (subject_id, edf_path) tuples
        """
        edf_files = []
        
        # Try sample directory first (for testing)
        if 'sample' in self.dataset_paths:
            sample_path = self.dataset_paths['sample']
            if sample_path.exists():
                logger.info(f"Checking MrOS visit {self.visit} sample directory")
                sample_edfs = list(sample_path.rglob('*.edf'))
                
                for edf_path in sample_edfs:
                    subject_id = self._extract_base_subject_id(edf_path.stem)
                    edf_files.append((subject_id, edf_path))
                
                if edf_files:
                    edf_files = self._filter_duplicate_edfs(edf_files)
                    logger.info(f"Found {len(edf_files)} MrOS visit{self.visit} EDF files in sample")
                    return edf_files
        
        # Original structure
        original_path = self.dataset_paths.get('original')
        if not original_path or not original_path.exists():
            logger.warning(f"MrOS original path does not exist: {original_path}")
            return []
        
        # Search for EDFs
        for edf_path in original_path.rglob('*.edf'):
            subject_id = self._extract_base_subject_id(edf_path.stem)
            edf_files.append((subject_id, edf_path))
        
        # Filter duplicates
        edf_files = self._filter_duplicate_edfs(edf_files)
        logger.info(f"Found {len(edf_files)} MrOS visit{self.visit} EDF files")
        return edf_files
    
    def _extract_base_subject_id(self, filename: str) -> str:
        """Extract base subject ID from EDF filename.
        
        Removes _1, _2 suffixes from filenames.
        
        Args:
            filename: EDF filename without extension
        
        Returns:
            Base subject ID
        """
        return filename.split('_')[0]
    
    def _filter_duplicate_edfs(self, edf_files: List[Tuple[str, Path]]) -> List[Tuple[str, Path]]:
        """Filter duplicate EDFs, preferring base file over numbered versions.
        
        Priority: X.edf > X_1.edf > X_2.edf
        """
        subject_files = {}
        
        for subject_id, edf_path in edf_files:
            stem = edf_path.stem
            
            if '_' not in stem:
                priority = 0
            else:
                try:
                    suffix = stem.split('_')[-1]
                    priority = int(suffix)
                except (ValueError, IndexError):
                    priority = 99
            
            if subject_id not in subject_files or priority < subject_files[subject_id][1]:
                subject_files[subject_id] = (edf_path, priority)
        
        result = [(sid, path) for sid, (path, _) in subject_files.items()]
        return sorted(result, key=lambda x: x[0])
    
    def find_annotation_file(self, subject_id: str) -> Optional[Path]:
        """Find MrOS annotation file.
        
        Args:
            subject_id: Subject identifier (nsrrid)
        
        Returns:
            Path to annotation file, or None
        """
        annotations_path = self.dataset_paths.get('annotations')
        if not annotations_path or not annotations_path.exists():
            return None
        
        # MrOS annotations are in visit-specific subdirectories
        # File format: mros-visit{N}-{nsrrid}-nsrr.xml
        visit_dir = annotations_path / f'visit{self.visit}'
        if not visit_dir.exists():
            return None
        
        # Look for XML file matching subject_id
        xml_file = visit_dir / f'{subject_id}-nsrr.xml'
        if xml_file.exists():
            return xml_file
        
        return None
    
    def parse_annotations(self, annotation_path: Path) -> Dict[str, Any]:
        """Parse MrOS annotations.
        
        Args:
            annotation_path: Path to annotation file
        
        Returns:
            Dictionary with stages, events, duration, format
        """
        # Assuming NSRR XML format
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
            
            if event_concept is not None and 'Stage' in event_concept.text:
                start_time = float(start.text) if start is not None else 0
                stage_label = event_concept.text
                
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
        """Load and merge MrOS metadata from multiple CSV files.
        
        Returns:
            DataFrame with merged subject metadata
        """
        datasets_path = self.dataset_paths['datasets']
        dfs_to_merge = []
        
        for file_key, filename in self.metadata_files.items():
            file_path = datasets_path / filename
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    logger.info(f"Loaded MrOS {file_key}: {len(df)} subjects, {len(df.columns)} columns")
                    dfs_to_merge.append(df)
                except Exception as e:
                    logger.error(f"Error loading {file_key}: {e}")
            else:
                logger.warning(f"{file_key} not found: {file_path}")
        
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
                dup_cols = [col for col in df.columns if col.endswith('_dup')]
                if dup_cols:
                    df = df.drop(columns=dup_cols)
        
        logger.info(f"Merged MrOS metadata: {len(df)} subjects, {len(df.columns)} columns")
        
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
        """Extract metadata for a specific MrOS subject.
        
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
