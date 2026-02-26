"""STAGES dataset adapter.

STAGES (Stress and Sleep Study) is a cohort study focusing on:
- Sleep, stress, and health relationships
- Single visit per subject
- n=795 subjects

Data structure:
- EDFs: raw/stages/original/*/usable/*.edf
- Annotations: XML files in same directory as EDFs
- Metadata: CSV files with nsrrid as subject_id_col

Phenotype columns (from nocturn):
- nsrrid: subject identifier
- ahi: Apnea-Hypopnea Index
- isi_score: Insomnia Severity Index score
- phq_1000: PHQ-8 depression score
- gad_0800: GAD-7 anxiety score
- age_category: Age category
- sex: Sex (male/female)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import xml.etree.ElementTree as ET
from loguru import logger

from .base_adapter import BaseNSRRAdapter


class STAGESAdapter(BaseNSRRAdapter):
    """Adapter for STAGES dataset."""
    
    def __init__(self, config):
        super().__init__(config, 'stages')
        
        # Nocturn config shows these metadata files
        self.metadata_files = {
            'main': 'stages-dataset-0.3.0.csv',
            'harmonized': 'stages-harmonized-dataset-0.3.0.csv',
        }
        
        # Subject ID column (STAGES uses subject_code, not nsrrid)
        self.subject_id_col = 'subject_code'
        
        # Phenotype columns of interest (from multiple files)
        # harmonized: nsrr_age, nsrr_sex, nsrr_race, nsrr_bmi, nsrr_current_smoker
        # main: phq_1000, gad_0800, isi_score, ess_0900, fss_1000, fosq_1100, etc.
        # Note: AHI comes from PSG annotations (XML), not CSV metadata
        self.phenotype_cols = [
            'subject_code',
            # From harmonized
            'nsrr_age', 'nsrr_sex', 'nsrr_race', 'nsrr_bmi', 'nsrr_current_smoker',
            # From main dataset  
            'phq_1000', 'gad_0800', 'isi_score', 'ess_0900', 'fss_1000'
        ]
    
    def find_edf_files(self) -> List[Tuple[str, Path]]:
        """Find all STAGES EDF files.
        
        STAGES structure: original/*/usable/*.edf
        Also checks sample_extraction for testing.
        Subject ID is extracted from EDF filename (site code like GSSA00001).
        
        Note: EDF filenames use site codes (GSSA00001) which differ from 
        metadata subject_codes (BOGN00002). Direct mapping not available.
        
        Returns:
            List of (subject_id, edf_path) tuples
        """
        edf_files = []
        
        # Look in original/STAGES PSGs directory structure
        original_path = self.dataset_paths['original']
        
        if not original_path.exists():
            logger.warning(f"STAGES original path does not exist: {original_path}")
            return []
        
        # STAGES structure: original/STAGES PSGs/<SITE>/<subject_id>.edf
        stages_psg_path = original_path / 'STAGES PSGs'
        if stages_psg_path.exists():
            logger.info(f"Checking {stages_psg_path}")
            all_edfs = list(stages_psg_path.rglob('*.edf'))
            
            # Build list of (subject_id, path) tuples, then filter duplicates
            for edf_path in all_edfs:
                subject_id = self._extract_base_subject_id(edf_path.stem)
                edf_files.append((subject_id, edf_path))
            
            # Now filter duplicates
            edf_files = self._filter_duplicate_edfs(edf_files)
            
            logger.info(f"Found {len(edf_files)} STAGES EDF files")
            return edf_files
        
        logger.warning(f"STAGES PSGs directory not found in {original_path}")
        return []
    
    def _extract_base_subject_id(self, filename: str) -> str:
        """Extract base subject ID from EDF filename.
        
        Removes _1, _2 suffixes from filenames.
        GSSA00001_1 → GSSA00001
        GSSA00001 → GSSA00001
        
        Args:
            filename: EDF filename without extension
        
        Returns:
            Base subject ID
        """
        return filename.split('_')[0]
    
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
        """Find annotation XML file for a STAGES subject.
        
        Annotations are in the same usable/ directory as the EDF.
        Typically named: <subject_id>-nsrr.xml
        
        Args:
            subject_id: Subject identifier
        
        Returns:
            Path to annotation XML, or None if not found
        """
        original_path = self.dataset_paths['original']
        
        # Expected location: <subject_id>/usable/<subject_id>-nsrr.xml
        usable_dir = original_path / subject_id / 'usable'
    def find_annotation_file(self, subject_id: str, edf_path: Optional[Path] = None) -> Optional[Path]:
        """Find annotation CSV file for a subject.
        
        For STAGES, annotations are CSV files next to the EDF files.
        Format: <subject_id>.csv in same directory as <subject_id>.edf
        
        Args:
            subject_id: Subject identifier
            edf_path: Optional path to EDF file (helps locate annotation)
        
        Returns:
            Path to annotation CSV, or None if not found
        """
        # If we have the EDF path, look for CSV in same directory
        if edf_path and edf_path.exists():
            csv_path = edf_path.with_suffix('.csv')
            if csv_path.exists():
                logger.debug(f"Found annotation CSV for {subject_id}: {csv_path.name}")
                return csv_path
        
        # Otherwise search in STAGES PSGs directories
        original_path = self.dataset_paths['original']
        stages_psg_path = original_path / 'STAGES PSGs'
        
        if not stages_psg_path.exists():
            logger.debug(f"STAGES PSGs directory not found")
            return None
        
        # Search all site directories for the subject's CSV
        for site_dir in stages_psg_path.iterdir():
            if site_dir.is_dir():
                csv_path = site_dir / f"{subject_id}.csv"
                if csv_path.exists():
                    logger.debug(f"Found annotation CSV for {subject_id} in {site_dir.name}")
                    return csv_path
        
        logger.debug(f"No annotation CSV found for {subject_id}")
        return None
    
    def parse_annotations(self, annotation_path: Path) -> Dict[str, Any]:
        """Parse STAGES CSV annotation file.
        
        STAGES CSV format contains:
        Start Time,Duration (seconds),Event
        Each row has a timestamp, duration, and event label
        
        Args:
            annotation_path: Path to CSV file
        
        Returns:
            Dictionary with stages, events, duration, format
        """
        try:
            import pandas as pd
            df = pd.read_csv(annotation_path)
        except Exception as e:
            logger.error(f"Error parsing CSV {annotation_path}: {e}")
            return {
                'stages': [],
                'events': [],
                'duration': 0,
                'format': 'stages-csv',
                'parse_error': str(e)
            }
        
        stages = []
        events = []
        
        # Map stage labels to standard values
        stage_map = {
            'Wake': 0,
            'Stage1': 1,
            'Stage2': 2,
            'Stage3': 3,
            'Stage4': 3,  # Merge stage 4 into stage 3
            'REM': 5,
            'UnknownStage': -1,
            'Unscored': -1
        }
        
        # Parse CSV rows
        for _, row in df.iterrows():
            event = row.get('Event', '')
            start_time_str = row.get('Start Time', '')
            duration_sec = row.get('Duration (seconds)', 0)
            
            # Try to extract stage label
            stage_label = None
            for label in stage_map.keys():
                if label in event:
                    stage_label = label
                    break
            
            if stage_label:
                # Parse start time (format like "21:34:38")
                try:
                    time_parts = start_time_str.split(':')
                    if len(time_parts) == 3:
                        hours, mins, secs = map(int, time_parts)
                        start_seconds = hours * 3600 + mins * 60 + secs
                    else:
                        start_seconds = 0
                except:
                    start_seconds = 0
                
                stages.append({
                    'start': start_seconds,
                    'stage': stage_map[stage_label],
                    'label': stage_label,
                    'duration': float(duration_sec) if duration_sec else 30
                })
            else:
                # Other events (desaturations, calibration, etc.)
                events.append({
                    'type': event,
                    'start_time': start_time_str,
                    'duration': float(duration_sec) if duration_sec else 0
                })
        
        # Sort stages by start time
        stages.sort(key=lambda x: x['start'])
        
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
            'format': 'stages-csv',
            'num_epochs': len(stages)
        }
    
    def load_metadata(self) -> pd.DataFrame:
        """Load STAGES metadata from multiple CSV files and merge.
        
        Following nocturn project structure:
        - harmonized file: nsrr_age, nsrr_sex, nsrr_race, nsrr_bmi, ahi, etc.
        - main file: phq_1000, gad_0800, isi_score, ess_0900, etc.
        
        Returns:
            DataFrame with merged subject metadata
        """
        datasets_path = self.dataset_paths['datasets']
        
        # Load harmonized file (has demographics and AHI)
        harmonized_path = datasets_path / self.metadata_files['harmonized']
        main_path = datasets_path / self.metadata_files['main']
        
        dfs_to_merge = []
        
        # Load harmonized dataset
        if harmonized_path.exists():
            try:
                df_harmonized = pd.read_csv(harmonized_path)
                logger.info(f"Loaded STAGES harmonized: {len(df_harmonized)} subjects, {len(df_harmonized.columns)} columns")
                
                # STAGES CSV files have duplicate subject_codes - keep last occurrence
                # (analysis shows last row has most complete data in 70% of cases)
                if 'subject_code' in df_harmonized.columns:
                    n_before = len(df_harmonized)
                    df_harmonized = df_harmonized.drop_duplicates(subset=['subject_code'], keep='last')
                    n_after = len(df_harmonized)
                    if n_before != n_after:
                        logger.info(f"  Removed {n_before - n_after} duplicate subject_codes (keeping last occurrence)")
                
                dfs_to_merge.append(df_harmonized)
            except Exception as e:
                logger.error(f"Error loading harmonized file: {e}")
        else:
            logger.warning(f"Harmonized file not found: {harmonized_path}")
        
        # Load main dataset
        if main_path.exists():
            try:
                df_main = pd.read_csv(main_path)
                logger.info(f"Loaded STAGES main: {len(df_main)} subjects, {len(df_main.columns)} columns")
                
                # STAGES CSV files have duplicate subject_codes - keep last occurrence
                # (analysis shows last row has most complete data in 70% of cases)
                if 'subject_code' in df_main.columns:
                    n_before = len(df_main)
                    df_main = df_main.drop_duplicates(subset=['subject_code'], keep='last')
                    n_after = len(df_main)
                    if n_before != n_after:
                        logger.info(f"  Removed {n_before - n_after} duplicate subject_codes (keeping last occurrence)")
                
                dfs_to_merge.append(df_main)
            except Exception as e:
                logger.error(f"Error loading main file: {e}")
        else:
            logger.warning(f"Main file not found: {main_path}")
        
        # Merge dataframes
        if not dfs_to_merge:
            logger.error("No metadata files found")
            return pd.DataFrame()
        
        if len(dfs_to_merge) == 1:
            df = dfs_to_merge[0]
        else:
            # Merge on subject_code (after deduplication, should be 1-to-1 merge)
            df = dfs_to_merge[0]
            for df_next in dfs_to_merge[1:]:
                # Determine merge key
                merge_key = 'subject_code'
                
                # Use inner merge to keep only subjects present in both files
                df = df.merge(df_next, on=merge_key, how='inner', suffixes=('', '_dup'))
                
                # Remove duplicate columns from merge
                dup_cols = [col for col in df.columns if col.endswith('_dup')]
                if dup_cols:
                    df = df.drop(columns=dup_cols)
        
        logger.info(f"Merged STAGES metadata: {len(df)} subjects, {len(df.columns)} columns")
        
        # Check for expected columns
        missing_cols = [col for col in self.phenotype_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing expected columns: {missing_cols}")
        
        return df
    
    def get_subject_id_column(self) -> str:
        """Get the name of the subject ID column in STAGES metadata.
        
        Returns:
            Column name: 'subject_code' (not 'nsrrid' for STAGES)
        """
        return self.subject_id_col
    
    def extract_subject_metadata(self, subject_id: str, 
                                metadata_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract metadata for a specific STAGES subject.
        
        Args:
            subject_id: Subject identifier (nsrrid)
            metadata_df: Full metadata DataFrame
        
        Returns:
            Dictionary with subject metadata
        """
        # Find subject row
        subject_row = metadata_df[metadata_df[self.subject_id_col] == subject_id]
        
        if len(subject_row) == 0:
            logger.info(f"Subject {subject_id} not found in metadata (EDF exists but no corresponding metadata entry - likely excluded from study)")
            return {'subject_id': subject_id, 'found': False}
        
        if len(subject_row) > 1:
            logger.warning(f"Subject {subject_id} has multiple rows, using first")
        
        row = subject_row.iloc[0]
        
        # Extract relevant fields
        metadata = {
            'subject_id': subject_id,
            'found': True,
            'dataset': 'stages',
            'visit': None,  # STAGES has single visit
        }
        
        # Add phenotype columns if they exist
        for col in self.phenotype_cols:
            if col in row.index:
                value = row[col]
                # Handle NaN
                if pd.isna(value):
                    value = None
                metadata[col] = value
        
        # Age handling (might be category or numeric)
        if 'age_category' in metadata:
            metadata['age_cat'] = metadata['age_category']
        
        # Clinical labels (from nocturn config)
        metadata['labels'] = {}
        if 'ahi' in metadata and metadata['ahi'] is not None:
            try:
                metadata['labels']['apnea_binary'] = float(metadata['ahi']) >= 15
            except:
                pass
        
        if 'isi_score' in metadata and metadata['isi_score'] is not None:
            try:
                metadata['labels']['insomnia_binary'] = float(metadata['isi_score']) >= 15
            except:
                pass
        
        if 'phq_1000' in metadata and metadata['phq_1000'] is not None:
            try:
                metadata['labels']['depression_binary'] = float(metadata['phq_1000']) >= 10
            except:
                pass
        
        if 'gad_0800' in metadata and metadata['gad_0800'] is not None:
            try:
                metadata['labels']['anxiety_binary'] = float(metadata['gad_0800']) >= 10
            except:
                pass
        
        return metadata
