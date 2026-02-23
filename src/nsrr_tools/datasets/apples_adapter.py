"""APPLES dataset adapter.

APPLES (Apnea Positive Pressure Long-term Efficacy Study):
- Multi-center RCT of CPAP treatment
- ~1,100 subjects with OSA
- Multiple visits: BL (1), DX (3), CPAP (4)

Data structure:
- EDFs: Relatively flat structure
- Annotations: Check format (XML or other)
- Metadata: Multiple CSV files

Metadata files (from nocturn):
- harmonized: nsrr_age, nsrr_sex, nsrr_race, nsrr_bmi, nsrr_ahi_chicago1999, PSG metrics
- apples-dataset: Questionnaires (BDI, ESS, MMSE), medical history
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import xml.etree.ElementTree as ET
from loguru import logger

from .base_adapter import BaseNSRRAdapter


class APPLESAdapter(BaseNSRRAdapter):
    """Adapter for APPLES dataset."""
    
    def __init__(self, config):
        super().__init__(config, 'apples')
        
        # Metadata files (from nocturn)
        self.metadata_files = {
            'harmonized': 'apples-harmonized-dataset-0.1.0.csv',
            'main': 'apples-dataset-0.1.0.csv',
        }
        
        # Subject ID column
        self.subject_id_col = 'nsrrid'
        
        # Phenotype columns (from multiple files)
        # harmonized: demographics + AHI + PSG metrics
        # main: BDI, ESS, MMSE, medical history
        self.phenotype_cols = [
            'nsrrid', 'visitn',
            # From harmonized
            'nsrr_age', 'nsrr_sex', 'nsrr_race', 'nsrr_bmi', 'nsrr_current_smoker',
            'nsrr_ahi_chicago1999',  # AHI
            'nsrr_ttleffsp_f1', 'nsrr_phrnumar_f1', 'nsrr_pctdursp_sr', 'nsrr_pctdursp_s3',
            # From main dataset
            'bditotalscore', 'esstotalscoreqc', 'mmsetotalscore',
        ]
    
    def find_edf_files(self) -> List[Tuple[str, Path]]:
        """Find all APPLES EDF files.
        
        Returns:
            List of (subject_id, edf_path) tuples
        """
        edf_files = []
        
        # Try sample directory first (for testing)
        if 'sample' in self.dataset_paths:
            sample_path = self.dataset_paths['sample']
            if sample_path.exists():
                logger.info("Checking APPLES sample directory")
                sample_edfs = list(sample_path.rglob('*.edf'))
                
                for edf_path in sample_edfs:
                    subject_id = self._extract_base_subject_id(edf_path.stem)
                    edf_files.append((subject_id, edf_path))
                
                if edf_files:
                    edf_files = self._filter_duplicate_edfs(edf_files)
                    logger.info(f"Found {len(edf_files)} APPLES EDF files in sample")
                    return edf_files
        
        # Original structure
        original_path = self.dataset_paths.get('original')
        if not original_path or not original_path.exists():
            logger.warning(f"APPLES original path does not exist: {original_path}")
            return []
        
        # Search for EDFs
        for edf_path in original_path.rglob('*.edf'):
            subject_id = self._extract_base_subject_id(edf_path.stem)
            edf_files.append((subject_id, edf_path))
        
        # Filter duplicates
        edf_files = self._filter_duplicate_edfs(edf_files)
        logger.info(f"Found {len(edf_files)} APPLES EDF files")
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
    
    def find_annotation_file(self, subject_id: str, edf_path: Optional[Path] = None) -> Optional[Path]:
        """Find APPLES annotation file.
        
        APPLES annotations are .annot files next to EDF files.
        
        Args:
            subject_id: Subject identifier (fileid like apples-270377)
            edf_path: Optional path to EDF file (helps locate annotation)
        
        Returns:
            Path to annotation file, or None
        """
        # If we have the EDF path, look for .annot in same directory
        if edf_path and edf_path.exists():
            annot_path = edf_path.with_suffix('.annot')
            if annot_path.exists():
                logger.debug(f"Found annotation for {subject_id}: {annot_path.name}")
                return annot_path
        
        # Otherwise search in polysomnography directory
        original_path = self.dataset_paths.get('original')
        if not original_path or not original_path.exists():
            return None
        
        # Search for .annot file matching subject_id
        for annot_path in original_path.rglob(f'*{subject_id}*.annot'):
            logger.debug(f"Found annotation for {subject_id}: {annot_path.name}")
            return annot_path
        
        return None
    
    def parse_annotations(self, annotation_path: Path) -> Dict[str, Any]:
        """Parse APPLES .annot annotation file.
        
        APPLES annotations are tab-separated files with format:
        class   instance        channel start   stop    meta
        
        Args:
            annotation_path: Path to .annot file
        
        Returns:
            Dictionary with stages, events, duration, format
        """
        try:
            import pandas as pd
            df = pd.read_csv(annotation_path, sep='\t')
        except Exception as e:
            logger.error(f"Error parsing annot file: {e}")
            return {'stages': [], 'events': [], 'duration': 0, 'format': 'error', 'num_epochs': 0}
        
        stages = []
        events = []
        
        # Map stage labels to standard values
        stage_map = {
            'W': 0,      # Wake
            'N1': 1,     # Stage 1
            'N2': 2,     # Stage 2
            'N3': 3,     # Stage 3
            'N4': 3,     # Stage 4 (merge with 3)
            'R': 5,      # REM
            'REM': 5,    # REM alternative
            '?': -1,     # Unscored
            'U': -1      # Unscored alternative
        }
        
        for idx, row in df.iterrows():
            stage_class = row.get('class', '')
            start_time = row.get('start', '')
            stop_time = row.get('stop', '')
            
            # Check if this is a stage annotation
            if stage_class in stage_map:
                # Parse time (format: HH:MM:SS)
                try:
                    if ':' in str(start_time):
                        parts = str(start_time).split(':')
                        hours, mins, secs = int(parts[0]), int(parts[1]), int(parts[2])
                        start_seconds = hours * 3600 + mins * 60 + secs
                    else:
                        start_seconds = idx * 30  # Assume 30s epochs
                    
                    if ':' in str(stop_time):
                        parts = str(stop_time).split(':')
                        hours, mins, secs = int(parts[0]), int(parts[1]), int(parts[2])
                        stop_seconds = hours * 3600 + mins * 60 + secs
                        duration = stop_seconds - start_seconds
                    else:
                        duration = 30  # Default epoch duration
                except:
                    start_seconds = idx * 30
                    duration = 30
                
                stages.append({
                    'start': start_seconds,
                    'stage': stage_map[stage_class],
                    'label': stage_class,
                    'duration': duration
                })
            else:
                # Other events (arousals, apneas, etc.)
                events.append({
                    'type': stage_class,
                    'start_time': start_time,
                    'stop_time': stop_time
                })
        
        # Calculate total duration
        if stages:
            last_stage = stages[-1]
            total_duration = last_stage['start'] + last_stage.get('duration', 30)
        else:
            total_duration = 0
        
        return {
            'stages': stages,
            'events': events,
            'duration': total_duration,
            'format': 'apples-annot',
            'num_epochs': len(stages)
        }
    
    def load_metadata(self) -> pd.DataFrame:
        """Load and merge APPLES metadata from multiple CSV files.
        
        Returns:
            DataFrame with merged subject metadata
        """
        datasets_path = self.dataset_paths['datasets']
        dfs_to_merge = []
        
        for file_key, filename in self.metadata_files.items():
            file_path = datasets_path / filename
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    logger.info(f"Loaded APPLES {file_key}: {len(df)} subjects, {len(df.columns)} columns")
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
                # APPLES main dataset uses 'appleid' instead of 'nsrrid' - rename for consistency
                if 'appleid' in df_next.columns and 'nsrrid' not in df_next.columns:
                    df_next = df_next.rename(columns={'appleid': 'nsrrid'})
                
                # Determine merge keys (nsrrid is primary, visitn if present in both)
                merge_keys = [self.subject_id_col]
                if 'visitn' in df.columns and 'visitn' in df_next.columns:
                    merge_keys.append('visitn')
                
                df = df.merge(df_next, on=merge_keys, how='outer', suffixes=('', '_dup'))
                dup_cols = [col for col in df.columns if col.endswith('_dup')]
                if dup_cols:
                    df = df.drop(columns=dup_cols)
        
        logger.info(f"Merged APPLES metadata: {len(df)} subjects, {len(df.columns)} columns")
        
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
        """Extract metadata for a specific APPLES subject.
        
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
