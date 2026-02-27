"""
Annotation Processor for NSRR Data
==================================

Parses sleep stage annotations from various formats and synchronizes them with EDF signals.

Handles:
- STAGES: CSV format
- SHHS/MrOS: NSRR XML format
- APPLES: Tab-separated .annot files

Key features:
- Format detection and unified parsing
- Synchronization validation (annotations vs signal length)
- Epoch-level sleep stage arrays
- Handling of format variations within datasets

Author: NSRR Preprocessing Pipeline
Date: February 2026
"""

import numpy as np
import mne
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger

from nsrr_tools.datasets.base_adapter import BaseNSRRAdapter


class AnnotationProcessor:
    """Process and synchronize sleep stage annotations."""
    
    # Standard epoch duration (seconds)
    EPOCH_DURATION = 30
    
    # Standard stage mapping for output
    STAGE_MAP = {
        'wake': 0,
        'n1': 1,
        'n2': 2,
        'n3': 3,
        'n4': 3,  # Merge N4 into N3
        'rem': 5,
        'unknown': -1
    }
    
    def __init__(self, adapter: BaseNSRRAdapter):
        """Initialize annotation processor.
        
        Args:
            adapter: Dataset-specific adapter (has parse_annotations method)
        """
        self.adapter = adapter
        logger.info(f"AnnotationProcessor initialized for {adapter.dataset_name}")
    
    def process_annotations(
        self,
        annotation_path: Path,
        edf_path: Path,
        output_path: Path,
        validate_sync: bool = True
    ) -> Dict[str, Any]:
        """Process sleep stage annotations and save as numpy array.
        
        Args:
            annotation_path: Path to annotation file
            edf_path: Path to EDF file (for validation)
            output_path: Path to output .npy file
            validate_sync: Whether to validate synchronization with EDF
        
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing annotations from {annotation_path.name}...")
        
        try:
            # Parse annotations using dataset-specific adapter
            annot_data = self.adapter.parse_annotations(annotation_path)
            
            if not annot_data or 'stages' not in annot_data:
                logger.warning(f"No stages found in {annotation_path.name}")
                return {
                    'success': False,
                    'error': 'No stages found',
                    'num_epochs': 0
                }
            
            stages = annot_data['stages']
            
            if not stages:
                logger.warning(f"Empty stages list in {annotation_path.name}")
                return {
                    'success': False,
                    'error': 'Empty stages',
                    'num_epochs': 0
                }
            
            logger.info(f"  Found {len(stages)} stage epochs")
            logger.info(f"  Format: {annot_data.get('format', 'unknown')}")
            
            # Convert to epoch array
            stage_array = self._stages_to_array(stages)
            
            # Validate synchronization with EDF if requested
            if validate_sync and edf_path.exists():
                sync_result = self._validate_synchronization(
                    stage_array, edf_path, annotation_path
                )
                
                if sync_result['needs_adjustment']:
                    logger.warning(f"  Synchronization issue detected:")
                    logger.warning(f"    Signal duration: {sync_result['signal_duration']:.1f}s")
                    logger.warning(f"    Annotation duration: {sync_result['annotation_duration']:.1f}s")
                    logger.warning(f"    Difference: {sync_result['difference']:.1f}s")
                    
                    # Adjust if possible
                    stage_array = self._adjust_synchronization(
                        stage_array, sync_result
                    )
            
            # Save as numpy array
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, stage_array)
            
            logger.success(f"  Saved {len(stage_array)} epochs to {output_path.name}")
            
            return {
                'success': True,
                'num_epochs': len(stage_array),
                'duration_hours': len(stage_array) * self.EPOCH_DURATION / 3600,
                'stage_distribution': self._get_stage_distribution(stage_array),
                'output_size_kb': output_path.stat().st_size / 1024 if output_path.exists() else 0
            }
            
        except Exception as e:
            logger.error(f"Error processing annotations {annotation_path.name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'num_epochs': 0
            }
    
    def _stages_to_array(self, stages: List[Dict[str, Any]]) -> np.ndarray:
        """Convert list of stage dictionaries to epoch array.
        
        Args:
            stages: List of stage dictionaries with 'start', 'stage', 'duration'
        
        Returns:
            Numpy array of stage labels (one per epoch)
        """
        if not stages:
            return np.array([], dtype=np.int8)
        
        # Sort by start time
        stages_sorted = sorted(stages, key=lambda x: x.get('start', 0))
        
        # Calculate number of epochs
        last_stage = stages_sorted[-1]
        last_start = last_stage.get('start', 0)
        last_duration = last_stage.get('duration', self.EPOCH_DURATION)
        total_duration = last_start + last_duration
        num_epochs = int(np.ceil(total_duration / self.EPOCH_DURATION))
        
        # Initialize array with unknown (-1)
        stage_array = np.full(num_epochs, -1, dtype=np.int8)
        
        # Fill in stages
        for stage_info in stages_sorted:
            start_time = stage_info.get('start', 0)
            duration = stage_info.get('duration', self.EPOCH_DURATION)
            stage_value = stage_info.get('stage', -1)
            
            # Calculate epoch indices
            start_epoch = int(start_time / self.EPOCH_DURATION)
            num_stage_epochs = int(np.ceil(duration / self.EPOCH_DURATION))
            end_epoch = start_epoch + num_stage_epochs
            
            # Ensure within bounds
            end_epoch = min(end_epoch, num_epochs)
            
            # Fill array
            if start_epoch < num_epochs:
                stage_array[start_epoch:end_epoch] = stage_value
        
        return stage_array
    
    def _validate_synchronization(
        self,
        stage_array: np.ndarray,
        edf_path: Path,
        annotation_path: Path
    ) -> Dict[str, Any]:
        """Validate that annotations match signal duration.
        
        Args:
            stage_array: Array of sleep stages
            edf_path: Path to EDF file
            annotation_path: Path to annotation file (for logging)
        
        Returns:
            Dictionary with synchronization information
        """
        try:
            # Load EDF header only (fast)
            raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose='ERROR')
            signal_duration = raw.times[-1]  # Duration in seconds
            
            # Calculate annotation duration
            annotation_duration = len(stage_array) * self.EPOCH_DURATION
            
            # Calculate difference
            difference = abs(signal_duration - annotation_duration)
            
            # Allow small tolerance (1-2 epochs)
            tolerance = 2 * self.EPOCH_DURATION
            needs_adjustment = difference > tolerance
            
            return {
                'signal_duration': signal_duration,
                'annotation_duration': annotation_duration,
                'difference': difference,
                'difference_epochs': difference / self.EPOCH_DURATION,
                'needs_adjustment': needs_adjustment,
                'signal_epochs': int(signal_duration / self.EPOCH_DURATION),
                'annotation_epochs': len(stage_array)
            }
            
        except Exception as e:
            logger.warning(f"Could not validate synchronization for {edf_path.name}: {e}")
            return {
                'signal_duration': 0,
                'annotation_duration': len(stage_array) * self.EPOCH_DURATION,
                'difference': 0,
                'needs_adjustment': False,
                'error': str(e)
            }
    
    def _adjust_synchronization(
        self,
        stage_array: np.ndarray,
        sync_result: Dict[str, Any]
    ) -> np.ndarray:
        """Adjust stage array to match signal duration.
        
        Strategies:
        - If annotations are longer: truncate
        - If annotations are shorter: pad with unknown (-1)
        - If close but misaligned: align to signal epochs
        
        Args:
            stage_array: Original stage array
            sync_result: Synchronization validation result
        
        Returns:
            Adjusted stage array
        """
        signal_epochs = sync_result.get('signal_epochs', len(stage_array))
        annotation_epochs = len(stage_array)
        
        if signal_epochs == annotation_epochs:
            return stage_array
        
        elif signal_epochs < annotation_epochs:
            # Annotations are longer - truncate
            logger.info(f"    Truncating from {annotation_epochs} to {signal_epochs} epochs")
            return stage_array[:signal_epochs]
        
        else:
            # Annotations are shorter - pad with unknown
            logger.info(f"    Padding from {annotation_epochs} to {signal_epochs} epochs")
            padding = np.full(signal_epochs - annotation_epochs, -1, dtype=np.int8)
            return np.concatenate([stage_array, padding])
    
    def _get_stage_distribution(self, stage_array: np.ndarray) -> Dict[str, int]:
        """Get distribution of sleep stages.
        
        Args:
            stage_array: Array of sleep stages
        
        Returns:
            Dictionary with counts per stage
        """
        unique, counts = np.unique(stage_array, return_counts=True)
        
        stage_names = {
            -1: 'unknown',
            0: 'wake',
            1: 'n1',
            2: 'n2',
            3: 'n3',
            5: 'rem'
        }
        
        distribution = {}
        for stage_value, count in zip(unique, counts):
            stage_name = stage_names.get(stage_value, f'stage_{stage_value}')
            distribution[stage_name] = int(count)
        
        return distribution
    
    def get_epoch_times(
        self,
        stage_array: np.ndarray,
        sampling_rate: int = 128
    ) -> np.ndarray:
        """Get start sample indices for each epoch.
        
        Useful for aligning signals with annotations.
        
        Args:
            stage_array: Array of sleep stages
            sampling_rate: Signal sampling rate (Hz)
        
        Returns:
            Array of sample indices for each epoch start
        """
        num_epochs = len(stage_array)
        epoch_samples = int(self.EPOCH_DURATION * sampling_rate)
        
        epoch_times = np.arange(num_epochs) * epoch_samples
        
        return epoch_times
