"""
Signal Processor for NSRR Data
==============================

Converts raw EDF signals to SleepFM-compatible HDF5 format.

Key operations:
- Load EDF files using MNE
- Detect and map channels to standard names
- Apply modality-specific bandpass filtering
- Resample all channels to 128 Hz (SleepFM requirement)
- Per-channel z-score normalization
- Save as compressed HDF5 (float16, gzip)

Author: NSRR Preprocessing Pipeline
Date: February 2026
"""

import numpy as np
import h5py
import mne
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

from nsrr_tools.core.channel_mapper import ChannelMapper
from nsrr_tools.core.modality_detector import ModalityDetector
from nsrr_tools.utils.config import Config


class SignalProcessor:
    """Process raw EDF signals to standardized HDF5 format."""
    
    # Modality-specific filtering parameters (Hz)
    FILTER_PARAMS = {
        'EEG': {'low': 0.3, 'high': 35.0, 'order': 4},
        'EOG': {'low': 0.3, 'high': 35.0, 'order': 4},
        'ECG': {'low': 0.5, 'high': 45.0, 'order': 4},
        'EMG': {'low': 10.0, 'high': 100.0, 'order': 4},
        'RESP': {'low': 0.05, 'high': 2.0, 'order': 4},
    }
    
    # SleepFM requirements
    TARGET_SR = 128  # Hz
    OUTPUT_DTYPE = np.float16
    COMPRESSION = 'gzip'
    COMPRESSION_LEVEL = 4
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize signal processor.
        
        Args:
            config: Configuration object (optional, will create if not provided)
        """
        self.config = config or Config()
        self.channel_mapper = ChannelMapper(self.config)
        self.modality_detector = ModalityDetector(self.config)
        
        logger.info("SignalProcessor initialized")
        logger.info(f"  Target sampling rate: {self.TARGET_SR} Hz")
        logger.info(f"  Output dtype: {self.OUTPUT_DTYPE}")
        logger.info(f"  Compression: {self.COMPRESSION} (level {self.COMPRESSION_LEVEL})")
    
    def process_edf(
        self,
        edf_path: Path,
        output_path: Path,
        channel_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Process a single EDF file to HDF5.
        
        Args:
            edf_path: Path to input EDF file
            output_path: Path to output HDF5 file
            channel_mapping: Optional pre-computed channel mapping
        
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing {edf_path.name}...")
        
        try:
            # Load EDF
            raw = self._load_edf(edf_path)
            
            # Detect and map channels
            if channel_mapping is None:
                channel_mapping = self.channel_mapper.detect_channels_from_list(raw.ch_names)
            
            if not channel_mapping:
                logger.warning(f"No recognized channels in {edf_path.name}")
                return {
                    'success': False,
                    'error': 'No recognized channels',
                    'channels_found': 0
                }
            
            # Group by modality
            modality_groups = self.modality_detector.group_channels_by_modality(channel_mapping)
            
            logger.info(f"  Found {len(channel_mapping)} channels across {len(modality_groups)} modalities")
            
            # Apply SleepFM channel limits and prioritization
            channel_mapping = self._apply_sleepfm_limits(channel_mapping, modality_groups)
            
            if not channel_mapping:
                logger.warning(f"No channels remaining after applying SleepFM limits")
                return {
                    'success': False,
                    'error': 'No channels after filtering',
                    'channels_found': 0
                }
            
            logger.info(f"  Selected {len(channel_mapping)} channels after applying SleepFM limits")
            
            # Process each channel
            processed_channels = {}
            normalization_stats = {}
            
            for std_name, raw_name in channel_mapping.items():
                try:
                    # Get modality for this channel
                    modality = self._get_channel_modality(std_name, modality_groups)
                    
                    # Process signal
                    processed_signal, stats = self._process_channel(
                        raw, raw_name, modality
                    )
                    
                    processed_channels[std_name] = processed_signal
                    normalization_stats[std_name] = stats
                    
                except Exception as e:
                    logger.warning(f"  Failed to process {std_name} ({raw_name}): {e}")
                    continue
            
            if not processed_channels:
                logger.warning(f"No channels successfully processed for {edf_path.name}")
                return {
                    'success': False,
                    'error': 'No channels processed',
                    'channels_found': len(channel_mapping)
                }
            
            # Save to HDF5
            self._save_hdf5(output_path, processed_channels, normalization_stats, raw)
            
            logger.success(f"  Saved {len(processed_channels)} channels to {output_path.name}")
            
            return {
                'success': True,
                'channels_processed': len(processed_channels),
                'channels_found': len(channel_mapping),
                'duration_hours': len(processed_channels[list(processed_channels.keys())[0]]) / self.TARGET_SR / 3600,
                'output_size_mb': output_path.stat().st_size / (1024**2) if output_path.exists() else 0
            }
            
        except Exception as e:
            logger.error(f"Error processing {edf_path.name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'channels_found': 0
            }
    
    def _load_edf(self, edf_path: Path) -> mne.io.Raw:
        """Load EDF file using MNE.
        
        Args:
            edf_path: Path to EDF file
        
        Returns:
            MNE Raw object
        """
        # Load with MNE (preload=False for memory efficiency, we'll load channels one by one)
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose='ERROR')
        return raw
    
    def _apply_sleepfm_limits(
        self,
        channel_mapping: Dict[str, str],
        modality_groups: Dict[str, Dict[str, str]]
    ) -> Dict[str, str]:
        """Apply SleepFM channel limits and prioritization.
        
        Per SleepFM paper: BAS=10, ECG=2, EMG=4, RESP=7 channels max.
        BAS combines EEG+EOG channels.
        
        Args:
            channel_mapping: Dict of {std_name: raw_name}
            modality_groups: Dict of {modality: {std_name: raw_name}}
        
        Returns:
            Filtered channel_mapping respecting SleepFM limits
        """
        # Get SleepFM modality specs from config
        sleepfm_modalities = self.config.modality_groups['sleepfm_modalities']
        
        # Group channels by SleepFM modality (BAS, EKG, EMG, RESP)
        sleepfm_groups = {
            'BAS': [],
            'EKG': [],
            'EMG': [],
            'RESP': []
        }
        
        for std_name in channel_mapping.keys():
            # Determine which SleepFM modality this channel belongs to
            if std_name in modality_groups.get('EEG', {}) or std_name in modality_groups.get('EOG', {}):
                sleepfm_groups['BAS'].append(std_name)
            elif std_name in modality_groups.get('ECG', {}):
                sleepfm_groups['EKG'].append(std_name)
            elif std_name in modality_groups.get('EMG', {}):
                sleepfm_groups['EMG'].append(std_name)
            elif std_name in modality_groups.get('RESP', {}):
                sleepfm_groups['RESP'].append(std_name)
        
        # Apply limits with prioritization
        filtered_channels = {}
        
        for sleepfm_mod, channels in sleepfm_groups.items():
            max_channels = sleepfm_modalities[sleepfm_mod]['max_channels']
            
            if len(channels) > max_channels:
                # Use priority order to select top N channels
                priority_order = sleepfm_modalities[sleepfm_mod].get('priority_order', [])
                
                # Sort channels by priority
                prioritized = []
                for priority_ch in priority_order:
                    if priority_ch in channels:
                        prioritized.append(priority_ch)
                
                # Add any remaining channels not in priority list
                for ch in channels:
                    if ch not in prioritized:
                        prioritized.append(ch)
                
                # Keep only top N
                selected = prioritized[:max_channels]
                logger.info(f"  {sleepfm_mod}: Limiting {len(channels)} → {max_channels} channels")
                logger.debug(f"    Selected: {selected}")
                logger.debug(f"    Dropped: {prioritized[max_channels:]}")
                
                channels = selected
            
            # Add selected channels to filtered mapping
            for ch in channels:
                filtered_channels[ch] = channel_mapping[ch]
        
        return filtered_channels
    
    def _get_channel_modality(
        self,
        std_name: str,
        modality_groups: Dict[str, Dict[str, str]]
    ) -> str:
        """Determine modality for a channel.
        
        Args:
            std_name: Standardized channel name
            modality_groups: Modality groupings from detector
        
        Returns:
            Modality name (EEG, EOG, ECG, EMG, RESP)
        """
        for modality, channels in modality_groups.items():
            if std_name in channels:
                return modality
        
        # Default fallback based on channel name
        if 'EEG' in std_name or any(eeg in std_name for eeg in ['C3', 'C4', 'O1', 'O2', 'F3', 'F4']):
            return 'EEG'
        elif 'EOG' in std_name or 'LOC' in std_name or 'ROC' in std_name:
            return 'EOG'
        elif 'ECG' in std_name or 'EKG' in std_name:
            return 'ECG'
        elif 'EMG' in std_name or 'CHIN' in std_name or 'LEG' in std_name:
            return 'EMG'
        elif any(resp in std_name for resp in ['Flow', 'Thor', 'ABD', 'RESP']):
            return 'RESP'
        else:
            logger.warning(f"Could not determine modality for {std_name}, using EEG defaults")
            return 'EEG'
    
    def _process_channel(
        self,
        raw: mne.io.Raw,
        channel_name: str,
        modality: str
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Process a single channel: filter, resample, normalize.
        
        Args:
            raw: MNE Raw object
            channel_name: Name of channel in raw data
            modality: Modality type (EEG, EOG, ECG, EMG, RESP)
        
        Returns:
            Tuple of (processed_signal, normalization_stats)
        """
        # Get raw signal and sampling rate
        ch_idx = raw.ch_names.index(channel_name)
        signal_data, _ = raw[ch_idx, :]
        signal_data = signal_data.flatten()
        original_sr = raw.info['sfreq']
        
        # Apply bandpass filter
        if modality in self.FILTER_PARAMS:
            filter_params = self.FILTER_PARAMS[modality]
            signal_data = self._bandpass_filter(
                signal_data,
                original_sr,
                filter_params['low'],
                filter_params['high'],
                filter_params['order']
            )
        
        # Resample to target sampling rate
        if original_sr != self.TARGET_SR:
            signal_data = self._resample_signal(signal_data, original_sr, self.TARGET_SR)
        
        # Z-score normalization
        signal_data, norm_stats = self._normalize_signal(signal_data)
        
        # Convert to output dtype
        signal_data = signal_data.astype(self.OUTPUT_DTYPE)
        
        return signal_data, norm_stats
    
    def _bandpass_filter(
        self,
        signal_data: np.ndarray,
        sr: float,
        low: float,
        high: float,
        order: int = 4
    ) -> np.ndarray:
        """Apply Butterworth bandpass filter.
        
        Args:
            signal_data: Input signal
            sr: Sampling rate
            low: Low cutoff frequency (Hz)
            high: High cutoff frequency (Hz)
            order: Filter order
        
        Returns:
            Filtered signal
        """
        nyquist = sr / 2.0
        low_norm = low / nyquist
        high_norm = high / nyquist
        
        # Ensure cutoff frequencies are valid
        low_norm = max(0.001, min(low_norm, 0.999))
        high_norm = max(0.001, min(high_norm, 0.999))
        
        if low_norm >= high_norm:
            logger.warning(f"Invalid filter range: [{low}, {high}] Hz at {sr} Hz SR")
            return signal_data
        
        try:
            b, a = scipy_signal.butter(order, [low_norm, high_norm], btype='band')
            filtered = scipy_signal.filtfilt(b, a, signal_data)
            return filtered
        except Exception as e:
            logger.warning(f"Filter failed: {e}, returning unfiltered signal")
            return signal_data
    
    def _resample_signal(
        self,
        signal_data: np.ndarray,
        original_sr: float,
        target_sr: float
    ) -> np.ndarray:
        """Resample signal to target sampling rate.
        
        Uses linear interpolation for efficiency and quality.
        
        Args:
            signal_data: Input signal
            original_sr: Original sampling rate
            target_sr: Target sampling rate
        
        Returns:
            Resampled signal
        """
        original_length = len(signal_data)
        duration = original_length / original_sr
        
        # Calculate new length
        target_length = int(duration * target_sr)
        
        # Create time arrays
        original_time = np.arange(original_length) / original_sr
        target_time = np.arange(target_length) / target_sr
        
        # Linear interpolation (faster than scipy.signal.resample, good quality)
        interpolator = interp1d(
            original_time,
            signal_data,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        resampled = interpolator(target_time)
        
        return resampled
    
    def _normalize_signal(
        self,
        signal_data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Z-score normalize signal.
        
        Args:
            signal_data: Input signal
        
        Returns:
            Tuple of (normalized_signal, stats_dict)
        """
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        
        # Handle zero std (flat signal)
        if std == 0 or np.isnan(std) or np.isinf(std):
            logger.warning("Zero or invalid std detected, using mean-centering only")
            normalized = signal_data - mean
            std = 1.0
        else:
            normalized = (signal_data - mean) / std
        
        # Check for NaN/Inf
        if np.isnan(normalized).any() or np.isinf(normalized).any():
            logger.warning("NaN/Inf detected after normalization, clipping...")
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=10.0, neginf=-10.0)
        
        stats = {
            'mean': float(mean),
            'std': float(std),
            'min': float(np.min(signal_data)),
            'max': float(np.max(signal_data))
        }
        
        return normalized, stats
    
    def _save_hdf5(
        self,
        output_path: Path,
        channels: Dict[str, np.ndarray],
        norm_stats: Dict[str, Dict[str, float]],
        raw: mne.io.Raw
    ):
        """Save processed signals to HDF5.
        
        Args:
            output_path: Output HDF5 file path
            channels: Dictionary of {channel_name: signal_array}
            norm_stats: Normalization statistics per channel
            raw: Original MNE Raw object (for metadata)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as hf:
            # Save each channel as a dataset
            for channel_name, signal_data in channels.items():
                # Calculate chunk size (5 minutes of data)
                chunk_size = min(5 * 60 * self.TARGET_SR, len(signal_data))
                
                hf.create_dataset(
                    channel_name,
                    data=signal_data,
                    dtype=self.OUTPUT_DTYPE,
                    chunks=(chunk_size,),
                    compression=self.COMPRESSION,
                    compression_opts=self.COMPRESSION_LEVEL
                )
            
            # Save metadata as attributes
            hf.attrs['sampling_rate'] = self.TARGET_SR
            hf.attrs['duration_seconds'] = len(signal_data) / self.TARGET_SR
            hf.attrs['num_channels'] = len(channels)
            hf.attrs['original_sfreq'] = raw.info['sfreq']
            
            # Save normalization stats as JSON string
            import json
            hf.attrs['normalization_stats'] = json.dumps(norm_stats)
            hf.attrs['channel_names'] = json.dumps(list(channels.keys()))
