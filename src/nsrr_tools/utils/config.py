"""Configuration utilities for NSRR preprocessing pipeline."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from string import Template


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def expand_env_vars(path_str: str) -> str:
    """Expand environment variables in path strings.
    
    Supports both $VAR and ${VAR} syntax.
    
    Args:
        path_str: Path string with environment variables
        
    Returns:
        Expanded path string
    """
    # Use Template for safe substitution
    template = Template(path_str)
    try:
        # Get all environment variables
        expanded = template.substitute(os.environ)
    except KeyError as e:
        raise ValueError(f"Environment variable {e} not set. "
                        f"Please set it before running the pipeline.")
    return expanded


def expand_paths_in_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively expand environment variables in all path strings.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with expanded paths
    """
    expanded = {}
    for key, value in config.items():
        if isinstance(value, dict):
            expanded[key] = expand_paths_in_dict(value)
        elif isinstance(value, str):
            expanded[key] = expand_env_vars(value)
        else:
            expanded[key] = value
    return expanded


class Config:
    """Central configuration manager for NSRR preprocessing pipeline."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Path to configuration directory.
                       If None, uses NSRR-tools/configs
        """
        if config_dir is None:
            # Find config dir relative to this file
            # This file is at: NSRR-tools/src/nsrr_tools/utils/config.py
            # Package root is: NSRR-tools/src/nsrr_tools
            # Project root is: NSRR-tools
            # Configs are at: NSRR-tools/configs
            this_file = Path(__file__)  # .../src/nsrr_tools/utils/config.py
            project_root = this_file.parent.parent.parent.parent  # Go up 4 levels
            config_dir = project_root / "configs"
        
        self.config_dir = Path(config_dir)
        
        # Load all configuration files
        self.channel_defs = self._load_config("channel_definitions.yaml")
        self.modality_groups = self._load_config("modality_groups.yaml")
        self.preprocessing_params = self._load_config("preprocessing_params.yaml")
        self.paths = self._load_config("paths.yaml")
        
        # Expand environment variables in paths
        self.paths = expand_paths_in_dict(self.paths)
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load a configuration file."""
        config_path = self.config_dir / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        return load_yaml_config(config_path)
    
    def get_channel_alternatives(self, channel_name: str) -> list:
        """Get alternative names for a channel.
        
        Args:
            channel_name: Standard channel name
            
        Returns:
            List of alternative names
        """
        return self.channel_defs['channel_alternatives'].get(channel_name, [])
    
    def get_channel_priority(self, channel_name: str) -> list:
        """Get priority order for channel alternatives.
        
        Args:
            channel_name: Standard channel name
            
        Returns:
            List of channel names in priority order
        """
        return self.channel_defs['channel_priority'].get(channel_name, [])
    
    def get_sleepfm_name(self, channel_name: str) -> str:
        """Get SleepFM-compatible name for channel.
        
        Standard names ARE SleepFM-compatible, so return as-is.
        
        Args:
            channel_name: Standard channel name
            
        Returns:
            SleepFM-compatible name (same as standard)
        """
        return channel_name
    
    def get_modality_channels(self, modality: str) -> list:
        """Get list of channels for a modality.
        
        Args:
            modality: Modality name (EEG, EOG, ECG, EMG, RESP)
            
        Returns:
            List of channel names
        """
        return self.modality_groups['modalities'][modality]['channels']
    
    def get_processing_params(self, modality: str) -> Dict[str, Any]:
        """Get preprocessing parameters for a modality.
        
        Args:
            modality: Modality name
            
        Returns:
            Dictionary of processing parameters
        """
        param_key = self.modality_groups['modalities'][modality]['processing_params']
        return self.preprocessing_params['processing'][param_key]
    
    def get_dataset_paths(self, dataset: str) -> Dict[str, Path]:
        """Get paths for a dataset.
        
        Args:
            dataset: Dataset name (stages, shhs, apples, mros)
            
        Returns:
            Dictionary of paths
        """
        paths = self.paths['datasets'][dataset]
        # Convert to Path objects
        return {k: Path(v) for k, v in paths.items()}
    
    def get_unified_paths(self) -> Dict[str, Path]:
        """Get unified metadata/output paths.
        
        Returns:
            Dictionary of paths
        """
        paths = self.paths['unified']
        return {k: Path(v) for k, v in paths.items()}
    
    def create_directories(self, dataset: Optional[str] = None):
        """Create necessary directories for processing.
        
        Args:
            dataset: If specified, create dirs for this dataset.
                    If None, create unified dirs.
        """
        if dataset:
            # Create dataset-specific directories
            ds_paths = self.get_dataset_paths(dataset)
            derived_base = ds_paths['derived']
            
            # Create subdirectories within derived
            for subdir in self.paths['derived_structure'].values():
                (derived_base / subdir).mkdir(parents=True, exist_ok=True)
        else:
            # Create unified directories
            unified_paths = self.get_unified_paths()
            for path in unified_paths.values():
                Path(path).mkdir(parents=True, exist_ok=True)
    
    def __repr__(self) -> str:
        return f"Config(config_dir={self.config_dir})"
