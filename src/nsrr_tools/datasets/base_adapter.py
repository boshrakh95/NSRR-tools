"""Base adapter class for dataset-specific implementations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from loguru import logger


class BaseNSRRAdapter(ABC):
    """Abstract base class for dataset-specific adapters.
    
    Each dataset (STAGES, SHHS, APPLES, MrOS) has its own file structure,
    metadata format, and annotation format. The adapter encapsulates
    all dataset-specific logic.
    """
    
    def __init__(self, config, dataset_name: str):
        """Initialize adapter.
        
        Args:
            config: Config object
            dataset_name: Name of dataset (stages, shhs, apples, mros)
        """
        self.config = config
        self.dataset_name = dataset_name
        self.dataset_paths = config.get_dataset_paths(dataset_name)
        
        # Set up derived path
        self.derived_path = self.dataset_paths.get('derived')
        if self.derived_path:
            self.derived_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {dataset_name.upper()} adapter")
        logger.debug(f"Dataset paths: {self.dataset_paths}")
    
    @abstractmethod
    def find_edf_files(self) -> List[Tuple[str, Path]]:
        """Find all EDF files for this dataset.
        
        Returns:
            List of tuples: (subject_id, edf_path)
        """
        pass
    
    @abstractmethod
    def find_annotation_file(self, subject_id: str) -> Optional[Path]:
        """Find annotation file for a subject.
        
        Args:
            subject_id: Subject identifier
        
        Returns:
            Path to annotation file, or None if not found
        """
        pass
    
    @abstractmethod
    def parse_annotations(self, annotation_path: Path) -> Dict[str, Any]:
        """Parse sleep staging and events from annotation file.
        
        Args:
            annotation_path: Path to annotation file
        
        Returns:
            Dictionary with:
                - 'stages': array of sleep stages (30s epochs)
                - 'events': list of events (optional)
                - 'duration': total duration in seconds
                - 'format': annotation format type
        """
        pass
    
    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        """Load phenotype/demographic metadata for dataset.
        
        Returns:
            DataFrame with subject-level metadata
        """
        pass
    
    @abstractmethod
    def get_subject_id_column(self) -> str:
        """Get the name of the subject ID column in metadata.
        
        Returns:
            Column name (e.g., 'nsrrid', 'subject_code', 'pptid')
        """
        pass
    
    @abstractmethod
    def extract_subject_metadata(self, subject_id: str, 
                                metadata_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract metadata for a specific subject.
        
        Args:
            subject_id: Subject identifier
            metadata_df: Full metadata DataFrame
        
        Returns:
            Dictionary with subject metadata (age, sex, BMI, etc.)
        """
        pass
    
    def get_subject_list(self) -> List[str]:
        """Get list of all subject IDs with EDF files.
        
        Returns:
            List of subject IDs
        """
        edf_files = self.find_edf_files()
        return [subject_id for subject_id, _ in edf_files]
    
    def get_edf_path(self, subject_id: str) -> Optional[Path]:
        """Get EDF file path for a subject.
        
        Args:
            subject_id: Subject identifier
        
        Returns:
            Path to EDF file, or None if not found
        """
        edf_files = self.find_edf_files()
        for sid, path in edf_files:
            if sid == subject_id:
                return path
        return None
    
    def validate_file_structure(self) -> Dict[str, Any]:
        """Validate that expected directories and files exist.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'dataset': self.dataset_name,
            'paths_exist': {},
            'num_edfs': 0,
            'has_metadata': False,
            'errors': []
        }
        
        # Check if paths exist
        for path_name, path in self.dataset_paths.items():
            exists = path.exists()
            results['paths_exist'][path_name] = exists
            if not exists:
                results['errors'].append(f"Path not found: {path_name} = {path}")
        
        # Try to count EDFs
        try:
            edf_files = self.find_edf_files()
            results['num_edfs'] = len(edf_files)
        except Exception as e:
            results['errors'].append(f"Error finding EDFs: {e}")
        
        # Try to load metadata
        try:
            metadata = self.load_metadata()
            results['has_metadata'] = len(metadata) > 0
            results['num_subjects_in_metadata'] = len(metadata)
        except Exception as e:
            results['errors'].append(f"Error loading metadata: {e}")
        
        results['valid'] = len(results['errors']) == 0
        
        return results
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset_name})"
