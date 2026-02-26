"""Channel mapping utilities for robust channel detection across datasets."""

from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import pyedflib
from loguru import logger


class ChannelMapper:
    """Maps dataset-specific channel names to standardized names."""
    
    def __init__(self, config):
        """Initialize channel mapper.
        
        Args:
            config: Config object with channel definitions
        """
        self.config = config
        self.channel_alternatives = config.channel_defs['channel_alternatives']
        self.sleepfm_naming = config.channel_defs['sleepfm_naming']
        self.channel_priority = config.channel_defs['channel_priority']
        
        # Build reverse mapping: alternative_name -> standard_name
        self._reverse_mapping = self._build_reverse_mapping()
    
    def _build_reverse_mapping(self) -> Dict[str, str]:
        """Build mapping from alternative names to standard names.
        
        Returns:
            Dictionary mapping alternative -> standard name
        """
        reverse_map = {}
        for standard_name, alternatives in self.channel_alternatives.items():
            for alt_name in alternatives:
                # Store in lowercase for case-insensitive matching
                reverse_map[alt_name.lower()] = standard_name
        return reverse_map
    
    def detect_channels_in_edf(self, edf_path: Path) -> Dict[str, str]:
        """Detect available channels in an EDF file.
        
        Args:
            edf_path: Path to EDF file
            
        Returns:
            Dictionary mapping standard_name -> found_name
            Example: {'C3-M2': 'C3M2', 'LOC': 'E1-M2', ...}
        """
        try:
            # Read EDF header only (fast with pyedflib)
            with pyedflib.EdfReader(str(edf_path)) as edf:
                available_channels = edf.getSignalLabels()
            
            return self.detect_channels_from_list(available_channels)
        
        except Exception as e:
            logger.error(f"Error reading EDF {edf_path}: {e}")
            return {}
    
    def detect_channels_from_list(self, available_channels: List[str]) -> Dict[str, str]:
        """Detect standard channels from a list of channel names.
        
        Args:
            available_channels: List of channel names from EDF
            
        Returns:
            Dictionary mapping standard_name -> found_name
        """
        detected = {}
        
        # Convert available channels to lowercase for matching
        available_lower = {ch.lower(): ch for ch in available_channels}
        
        # For each standard channel, check if any alternative is present
        for standard_name, alternatives in self.channel_alternatives.items():
            found_name = None
            
            # Check alternatives in priority order if priority defined
            search_order = self.channel_priority.get(standard_name, alternatives)
            
            for alt_name in search_order:
                if alt_name.lower() in available_lower:
                    found_name = available_lower[alt_name.lower()]
                    break
            
            if found_name:
                detected[standard_name] = found_name
        
        return detected
    
    def standardize_channel_name(self, channel_name: str) -> Optional[str]:
        """Convert a channel name to its standard form.
        
        Args:
            channel_name: Original channel name
            
        Returns:
            Standard channel name, or None if not recognized
        """
        return self._reverse_mapping.get(channel_name.lower())
    
    def to_sleepfm_name(self, standard_name: str) -> str:
        """Convert standard name to SleepFM-compatible name.
        
        Args:
            standard_name: Standard channel name
            
        Returns:
            SleepFM-compatible name
        """
        return self.sleepfm_naming.get(standard_name, standard_name)
    
    def get_channel_mapping(self, detected_channels: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """Get full mapping: standard -> {found, sleepfm}.
        
        Args:
            detected_channels: Dict from detect_channels_in_edf
            
        Returns:
            Dictionary with complete mapping info
            Example: {
                'C3-M2': {'found': 'C3M2', 'sleepfm': 'C3-M2'},
                'LOC': {'found': 'E1-M2', 'sleepfm': 'EOG(L)'},
            }
        """
        mapping = {}
        for standard_name, found_name in detected_channels.items():
            mapping[standard_name] = {
                'found': found_name,
                'sleepfm': self.to_sleepfm_name(standard_name)
            }
        return mapping
    
    def filter_by_modality(self, detected_channels: Dict[str, str], 
                          modality: str) -> Dict[str, str]:
        """Filter detected channels by modality.
        
        Args:
            detected_channels: Dict from detect_channels_in_edf
            modality: Modality name (EEG, EOG, ECG, EMG, RESP)
            
        Returns:
            Subset of detected_channels for this modality
        """
        modality_channels = set(self.config.get_modality_channels(modality))
        return {k: v for k, v in detected_channels.items() 
                if k in modality_channels}
    
    def get_modality_coverage(self, detected_channels: Dict[str, str]) -> Dict[str, int]:
        """Get number of channels available per modality.
        
        Args:
            detected_channels: Dict from detect_channels_in_edf
            
        Returns:
            Dictionary: {modality: num_channels}
        """
        coverage = {}
        modalities = self.config.modality_groups['modalities'].keys()
        
        for modality in modalities:
            modality_chans = self.filter_by_modality(detected_channels, modality)
            coverage[modality] = len(modality_chans)
        
        return coverage
    
    def check_minimum_requirements(self, detected_channels: Dict[str, str],
                                   requirement: str = "option_2") -> Tuple[bool, str]:
        """Check if detected channels meet minimum requirements.
        
        Args:
            detected_channels: Dict from detect_channels_in_edf
            requirement: Requirement option from config
            
        Returns:
            Tuple of (passes, message)
        """
        coverage = self.get_modality_coverage(detected_channels)
        reqs = self.config.modality_groups['minimum_requirements'][requirement]
        
        # Check required modalities
        required_mods = reqs.get('required_modalities', [])
        min_channels = reqs.get('min_channels_per_modality', {})
        
        for modality in required_mods:
            min_count = min_channels.get(modality, 1)
            actual_count = coverage.get(modality, 0)
            
            if actual_count < min_count:
                return False, f"Insufficient {modality} channels: {actual_count} < {min_count}"
        
        # Check minimum other modalities if specified
        min_other = reqs.get('min_other_modalities', 0)
        if min_other > 0:
            other_count = sum(1 for m, c in coverage.items() 
                            if m not in required_mods and c > 0)
            if other_count < min_other:
                return False, f"Insufficient other modalities: {other_count} < {min_other}"
        
        return True, "Requirements met"
