"""Modality detection and grouping utilities."""

from typing import Dict, List, Set
from loguru import logger


class ModalityDetector:
    """Detects and groups channels by modality families."""
    
    def __init__(self, config):
        """Initialize modality detector.
        
        Args:
            config: Config object with modality definitions
        """
        self.config = config
        self.modalities = config.modality_groups['modalities']
        self.sleepfm_groups = config.modality_groups['sleepfm_modalities']
    
    def group_channels_by_modality(self, detected_channels: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """Group detected channels by their modality.
        
        Args:
            detected_channels: Dict from ChannelMapper.detect_channels_in_edf()
                              {standard_name: found_name}
        
        Returns:
            Dictionary grouped by modality
            Example: {
                'EEG': {'C3-M2': 'C3M2', 'C4-M1': 'C4M1'},
                'EOG': {'LOC': 'E1-M2', 'ROC': 'E2-M2'},
                'ECG': {'EKG': 'EKG'},
                ...
            }
        """
        grouped = {mod: {} for mod in self.modalities.keys()}
        
        for standard_name, found_name in detected_channels.items():
            # Find which modality this channel belongs to
            for modality, mod_info in self.modalities.items():
                if standard_name in mod_info['channels']:
                    grouped[modality][standard_name] = found_name
                    break
        
        # Remove empty modalities
        grouped = {k: v for k, v in grouped.items() if v}
        
        return grouped
    
    def get_sleepfm_groups(self, modality_grouped: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        """Convert modality-grouped channels to SleepFM modality groups.
        
        SleepFM uses 4 groups: BAS, RESP, EKG, EMG
        - BAS includes EEG + EOG
        - Others map directly
        
        Args:
            modality_grouped: Dict from group_channels_by_modality()
        
        Returns:
            Dictionary grouped by SleepFM modalities
            Example: {
                'BAS': {'C3-M2': 'C3M2', 'LOC': 'E1-M2', ...},  # EEG + EOG
                'EKG': {'EKG': 'EKG'},
                'RESP': {'Flow': 'Airflow', 'Thor': 'THOR'},
                'EMG': {'CHIN': 'Chin1', 'LLEG': 'L Leg'}
            }
        """
        sleepfm_grouped = {}
        
        # Map each modality to its SleepFM group
        for modality, channels in modality_grouped.items():
            sleepfm_group = self.modalities[modality]['sleepfm_group']
            
            if sleepfm_group not in sleepfm_grouped:
                sleepfm_grouped[sleepfm_group] = {}
            
            # Merge channels into the SleepFM group
            sleepfm_grouped[sleepfm_group].update(channels)
        
        return sleepfm_grouped
    
    def get_modality_availability(self, detected_channels: Dict[str, str]) -> Dict[str, bool]:
        """Get boolean flags for modality availability.
        
        Args:
            detected_channels: Dict from ChannelMapper.detect_channels_in_edf()
        
        Returns:
            Dictionary of modality availability flags
            Example: {'EEG': True, 'EOG': True, 'ECG': True, 'EMG': False, 'RESP': True}
        """
        grouped = self.group_channels_by_modality(detected_channels)
        return {modality: len(channels) > 0 
                for modality, channels in grouped.items()}
    
    def get_modality_counts(self, detected_channels: Dict[str, str]) -> Dict[str, int]:
        """Get number of channels per modality.
        
        Args:
            detected_channels: Dict from ChannelMapper.detect_channels_in_edf()
        
        Returns:
            Dictionary of channel counts per modality
            Example: {'EEG': 4, 'EOG': 2, 'ECG': 1, 'EMG': 0, 'RESP': 3}
        """
        grouped = self.group_channels_by_modality(detected_channels)
        
        # Initialize all modalities with 0
        counts = {modality: 0 for modality in self.modalities.keys()}
        
        # Update with actual counts
        for modality, channels in grouped.items():
            counts[modality] = len(channels)
        
        return counts
    
    def create_modality_mask(self, detected_channels: Dict[str, str], 
                           modality_order: List[str] = None) -> List[int]:
        """Create binary mask for modality availability.
        
        Args:
            detected_channels: Dict from ChannelMapper.detect_channels_in_edf()
            modality_order: Order of modalities in mask. 
                          Default: ['EEG', 'EOG', 'ECG', 'EMG', 'RESP']
        
        Returns:
            Binary list (1=available, 0=missing)
        """
        if modality_order is None:
            modality_order = ['EEG', 'EOG', 'ECG', 'EMG', 'RESP']
        
        availability = self.get_modality_availability(detected_channels)
        return [1 if availability.get(mod, False) else 0 
                for mod in modality_order]
    
    def get_missing_modalities(self, detected_channels: Dict[str, str]) -> List[str]:
        """Get list of modalities that are completely missing.
        
        Args:
            detected_channels: Dict from ChannelMapper.detect_channels_in_edf()
        
        Returns:
            List of missing modality names
        """
        availability = self.get_modality_availability(detected_channels)
        return [mod for mod, avail in availability.items() if not avail]
    
    def get_available_modalities(self, detected_channels: Dict[str, str]) -> List[str]:
        """Get list of modalities that are available.
        
        Args:
            detected_channels: Dict from ChannelMapper.detect_channels_in_edf()
        
        Returns:
            List of available modality names
        """
        availability = self.get_modality_availability(detected_channels)
        return [mod for mod, avail in availability.items() if avail]
    
    def check_multimodal_coverage(self, detected_channels: Dict[str, str], 
                                  min_modalities: int = 3) -> bool:
        """Check if recording has sufficient multimodal coverage.
        
        Args:
            detected_channels: Dict from ChannelMapper.detect_channels_in_edf()
            min_modalities: Minimum number of modalities required
        
        Returns:
            True if sufficient multimodal coverage
        """
        available = self.get_available_modalities(detected_channels)
        return len(available) >= min_modalities
    
    def get_channel_summary(self, detected_channels: Dict[str, str]) -> Dict:
        """Get comprehensive summary of channel detection.
        
        Args:
            detected_channels: Dict from ChannelMapper.detect_channels_in_edf()
        
        Returns:
            Dictionary with complete summary
        """
        grouped = self.group_channels_by_modality(detected_channels)
        sleepfm_grouped = self.get_sleepfm_groups(grouped)
        counts = self.get_modality_counts(detected_channels)
        availability = self.get_modality_availability(detected_channels)
        
        return {
            'total_channels': len(detected_channels),
            'modality_counts': counts,
            'modality_availability': availability,
            'channels_by_modality': grouped,
            'channels_by_sleepfm_group': sleepfm_grouped,
            'available_modalities': self.get_available_modalities(detected_channels),
            'missing_modalities': self.get_missing_modalities(detected_channels),
            'modality_mask': self.create_modality_mask(detected_channels),
        }
