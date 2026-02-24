"""Metadata builder for unified NSRR dataset catalog."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import mne
from loguru import logger
from tqdm import tqdm

from nsrr_tools.utils.config import Config
from nsrr_tools.core.channel_mapper import ChannelMapper
from nsrr_tools.core.modality_detector import ModalityDetector


class MetadataBuilder:
    """Build unified metadata catalog across NSRR datasets.
    
    Scans all datasets and creates a comprehensive metadata table containing:
    - Subject identifiers
    - Available channels and modalities
    - Phenotypic data (age, sex, AHI, cognitive scores, etc.)
    - File paths and data availability
    - Quality metrics
    
    Output: unified_metadata.parquet with one row per subject
    """
    
    def __init__(self, config: Config, output_dir: Optional[Path] = None):
        """Initialize metadata builder.
        
        Args:
            config: Configuration object
            output_dir: Custom output directory (default: from config)
        """
        self.config = config
        self.channel_mapper = ChannelMapper(config)
        self.modality_detector = ModalityDetector(config)
        
        if output_dir:
            self.unified_path = Path(output_dir) / 'unified_metadata.parquet'
        else:
            unified_base = config.paths['paths']['unified_base']
            self.unified_path = Path(unified_base) / 'unified_metadata.parquet'
        
        self.unified_path.parent.mkdir(parents=True, exist_ok=True)
    
    def build_metadata(
        self,
        datasets: List[str] = None,
        force_rebuild: bool = False,
        use_cache: bool = True,
        limit: int = None
    ) -> pd.DataFrame:
        """Build unified metadata across datasets.
        
        Args:
            datasets: List of dataset names to process (None = all)
            force_rebuild: Rebuild even if cached metadata exists
            use_cache: Use cached per-dataset metadata if available
            limit: Limit to first N subjects per dataset (for testing)
            
        Returns:
            DataFrame with unified metadata
        """
        if datasets is None:
            datasets = ['stages', 'shhs', 'apples', 'mros']
        
        # Check if unified metadata already exists (skip if using limit for testing)
        if self.unified_path.exists() and not force_rebuild and not limit:
            logger.info(f"Loading existing metadata from {self.unified_path}")
            return pd.read_parquet(self.unified_path)
        
        logger.info(f"Building metadata for datasets: {', '.join(datasets)}")
        
        # Process each dataset
        all_metadata = []
        for dataset_name in datasets:
            logger.info(f"Processing {dataset_name.upper()}...")
            
            try:
                dataset_meta = self._process_dataset(dataset_name, use_cache, limit)
                if dataset_meta is not None and len(dataset_meta) > 0:
                    all_metadata.append(dataset_meta)
                    logger.info(f"  ✓ {len(dataset_meta)} subjects from {dataset_name}")
                else:
                    logger.warning(f"  ⚠ No metadata from {dataset_name}")
                    
            except Exception as e:
                logger.error(f"  ✗ Error processing {dataset_name}: {e}")
                continue
        
        if not all_metadata:
            raise ValueError("No metadata collected from any dataset")
        
        # Combine all datasets
        unified_df = pd.concat(all_metadata, ignore_index=True)
        logger.info(f"Combined metadata: {len(unified_df)} total subjects")
        
        # Add derived columns
        unified_df = self._add_derived_columns(unified_df)
        
        # Save unified metadata
        unified_df.to_parquet(self.unified_path, index=False)
        logger.success(f"Saved unified metadata to {self.unified_path}")
        
        return unified_df
    
    def _process_dataset(
        self,
        dataset_name: str,
        use_cache: bool = True,
        limit: int = None
    ) -> Optional[pd.DataFrame]:
        """Process one dataset and extract metadata.
        
        Args:
            dataset_name: Name of dataset (stages, shhs, apples, mros)
            use_cache: Use cached metadata if available
            limit: Limit to first N subjects (for testing)
            
        Returns:
            DataFrame with dataset metadata
        """
        # Get dataset adapter
        adapter = self._get_dataset_adapter(dataset_name)
        if adapter is None:
            logger.error(f"No adapter found for {dataset_name}")
            return None
        
        # Check for cached metadata (skip if using limit for testing)
        cache_file = adapter.derived_path / 'metadata_cache.parquet'
        if use_cache and cache_file.exists() and not limit:
            logger.info(f"  Using cached metadata from {cache_file}")
            return pd.read_parquet(cache_file)
        
        # Load phenotype data
        try:
            pheno_df = adapter.load_metadata()
            logger.info(f"  Loaded {len(pheno_df)} subjects from metadata CSV")
        except Exception as e:
            logger.error(f"  Failed to load metadata: {e}")
            return None
        
        # Scan EDF files for available channels
        logger.info("  Scanning EDF files for channels...")
        edf_files = adapter.find_edf_files()
        
        if not edf_files:
            logger.warning("  No EDF files found")
            # Return phenotype data without channel info
            pheno_df['dataset'] = dataset_name.upper()
            pheno_df['has_edf'] = False
            return pheno_df
        
        logger.info(f"  Found {len(edf_files)} EDF files")
        
        # Apply limit if specified
        if limit and limit < len(edf_files):
            logger.info(f"  Limiting to first {limit} subjects")
            edf_files = edf_files[:limit]
        
        # Extract channel information for each subject
        channel_info = {}
        for subject_id, edf_path in tqdm(edf_files, desc=f"  Scanning {dataset_name}"):
            try:
                # Normalize subject ID to match phenotype format
                # EDF may have "mros-visit1-aa0001", phenotype has "AA0001"
                # Extract last part after last dash and uppercase
                normalized_id = subject_id.split('-')[-1].upper()
                
                # Read EDF header with MNE (lightweight, doesn't load signal data)
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose='ERROR')
                
                # Get all channel names and sampling rates
                ch_names = raw.ch_names
                sfreqs = {ch: raw.info['sfreq'] for ch in ch_names}
                duration = raw.times[-1]
                
                # Detect standardized channels
                detected = self.channel_mapper.detect_channels_from_list(ch_names)
                
                # Group by modality
                modality_groups = self.modality_detector.group_channels_by_modality(detected)
                
                # Get sampling rates per modality (max rate in each modality)
                modality_sfreqs = {}
                for mod, channels in modality_groups.items():
                    if channels:
                        rates = [sfreqs.get(orig_ch, 0) 
                                for orig_ch in ch_names 
                                for std_ch in channels 
                                if detected.get(orig_ch) == std_ch]
                        modality_sfreqs[f'{mod}_sfreq'] = max(rates) if rates else 0
                
                # Store info (use normalized ID as key for merging)
                channel_info[normalized_id] = {
                    'edf_path': str(edf_path),
                    'num_channels': len(ch_names),
                    'duration_sec': duration,
                    'has_edf': True,
                    **{f'has_{mod}': len(channels) > 0 
                       for mod, channels in modality_groups.items()},
                    **{f'n_{mod}': len(channels) 
                       for mod, channels in modality_groups.items()},
                    **modality_sfreqs,
                    'channels': ','.join(sorted(detected.values())),
                    'raw_channels': ','.join(ch_names)
                }
                
                raw.close()
                
            except Exception as e:
                logger.warning(f"  Error processing {subject_id}: {e}")
                channel_info[normalized_id] = {
                    'edf_path': str(edf_path),
                    'has_edf': False,
                    'error': str(e)
                }
        
        # Merge with phenotype data
        channel_df = pd.DataFrame.from_dict(channel_info, orient='index')
        channel_df['subject_id'] = channel_df.index
        
        # Merge on subject_id (handle different ID column names)
        id_col = adapter.get_subject_id_column()
        
        # If limiting, only keep subjects we scanned (inner join)
        # Otherwise keep all phenotype subjects (left join)
        merge_how = 'inner' if limit else 'left'
        
        merged_df = pheno_df.merge(
            channel_df,
            left_on=id_col,
            right_on='subject_id',
            how=merge_how
        )
        
        # Fill missing EDF info (only relevant for left join)
        if not limit:
            merged_df['has_edf'] = merged_df['has_edf'].fillna(False)
        merged_df['dataset'] = dataset_name.upper()
        
        # Add label availability flags
        merged_df = self._add_label_availability(merged_df, adapter, dataset_name)
        
        # Cache the results (but not when testing with limit)
        if not limit:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            merged_df.to_parquet(cache_file, index=False)
            logger.info(f"  Cached metadata to {cache_file}")
        
        return merged_df
    
    def _get_dataset_adapter(self, dataset_name: str):
        """Get the appropriate dataset adapter.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            Dataset adapter instance
        """
        from nsrr_tools.datasets.stages_adapter import STAGESAdapter
        from nsrr_tools.datasets.shhs_adapter import SHHSAdapter
        from nsrr_tools.datasets.apples_adapter import APPLESAdapter
        from nsrr_tools.datasets.mros_adapter import MrOSAdapter
        
        if dataset_name.lower() == 'stages':
            return STAGESAdapter(self.config)
        elif dataset_name.lower() == 'shhs':
            return SHHSAdapter(self.config)
        elif dataset_name.lower() == 'apples':
            return APPLESAdapter(self.config)
        elif dataset_name.lower() == 'mros':
            return MrOSAdapter(self.config)
        else:
            logger.warning(f"No adapter implemented for {dataset_name}")
            return None
    
    def _add_label_availability(
        self, 
        df: pd.DataFrame, 
        adapter, 
        dataset_name: str
    ) -> pd.DataFrame:
        """Add flags for label availability (staging, AHI, cognitive scores).
        
        Args:
            df: Dataframe with metadata
            adapter: Dataset adapter
            dataset_name: Name of dataset
            
        Returns:
            DataFrame with label availability flags
        """
        # Check for sleep staging annotations
        df['has_staging'] = df['has_edf'].copy()  # Assume if EDF exists, staging exists
        
        # Check for AHI (apnea-hypopnea index)
        ahi_cols = [c for c in df.columns if 'ahi' in c.lower()]
        if ahi_cols:
            df['has_ahi'] = df[ahi_cols[0]].notna()
        else:
            df['has_ahi'] = False
        
        # Check for cognitive scores (dataset-specific)
        if dataset_name.lower() == 'mros':
            # MrOS has cognitive scores: 3ms (modified mini-mental state)
            cog_cols = [c for c in df.columns if '3ms' in c.lower() or 'cogni' in c.lower()]
            if cog_cols:
                df['has_cognitive'] = df[cog_cols].notna().any(axis=1)
            else:
                df['has_cognitive'] = False
        elif dataset_name.lower() == 'stages':
            # STAGES has cognitive outcomes
            cog_cols = [c for c in df.columns if 'cogni' in c.lower() or 'memory' in c.lower()]
            if cog_cols:
                df['has_cognitive'] = df[cog_cols].notna().any(axis=1)
            else:
                df['has_cognitive'] = False
        else:
            df['has_cognitive'] = False
        
        # Check for demographics (age, sex, BMI)
        age_cols = [c for c in df.columns if c.lower() in ['age', 'age_at_visit', 'vsage1']]
        sex_cols = [c for c in df.columns if c.lower() in ['sex', 'gender', 'male']]
        bmi_cols = [c for c in df.columns if 'bmi' in c.lower()]
        
        if age_cols:
            df['has_age'] = df[age_cols[0]].notna()
        else:
            df['has_age'] = False
            
        if sex_cols:
            df['has_sex'] = df[sex_cols[0]].notna()
        else:
            df['has_sex'] = False
            
        if bmi_cols:
            df['has_bmi'] = df[bmi_cols[0]].notna()
        else:
            df['has_bmi'] = False
        
        return df
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived/computed columns to metadata.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with additional columns
        """
        # Add completeness score (0-1)
        modality_cols = [c for c in df.columns if c.startswith('has_')]
        if modality_cols:
            df['completeness'] = df[modality_cols].sum(axis=1) / len(modality_cols)
        
        # Add usability flag (has minimum required channels)
        if 'has_BAS' in df.columns and 'has_RESP' in df.columns:
            df['is_usable'] = df['has_BAS'] & df['has_RESP']
        
        # Sort by dataset and subject_id
        if 'dataset' in df.columns:
            df = df.sort_values(['dataset', df.columns[0]])
        
        return df
    
    def get_summary_statistics(self, metadata_df: pd.DataFrame = None) -> Dict:
        """Generate summary statistics from metadata.
        
        Args:
            metadata_df: Metadata dataframe (loads from file if None)
            
        Returns:
            Dictionary with summary statistics
        """
        if metadata_df is None:
            if not self.unified_path.exists():
                raise FileNotFoundError(f"Metadata not found: {self.unified_path}")
            metadata_df = pd.read_parquet(self.unified_path)
        
        summary = {
            'total_subjects': len(metadata_df),
            'datasets': {},
            'channel_coverage': {},
            'usability': {}
        }
        
        # Per-dataset stats
        for dataset in metadata_df['dataset'].unique():
            dataset_df = metadata_df[metadata_df['dataset'] == dataset]
            summary['datasets'][dataset] = {
                'n_subjects': len(dataset_df),
                'with_edf': dataset_df['has_edf'].sum() if 'has_edf' in dataset_df else 0
            }
        
        # Channel coverage
        modality_cols = [c for c in metadata_df.columns if c.startswith('has_')]
        for col in modality_cols:
            modality = col.replace('has_', '')
            summary['channel_coverage'][modality] = metadata_df[col].sum()
        
        # Usability
        if 'is_usable' in metadata_df.columns:
            summary['usability']['usable_subjects'] = metadata_df['is_usable'].sum()
            summary['usability']['usable_percent'] = (
                100 * metadata_df['is_usable'].mean()
            )
        
        return summary
    
    def print_summary(self, metadata_df: pd.DataFrame = None):
        """Print a formatted summary of the metadata.
        
        Args:
            metadata_df: Metadata dataframe (loads from file if None)
        """
        stats = self.get_summary_statistics(metadata_df)
        
        print("\n" + "="*80)
        print("UNIFIED METADATA SUMMARY")
        print("="*80)
        print(f"\nTotal subjects: {stats['total_subjects']}")
        
        print("\nPer-dataset breakdown:")
        for dataset, info in stats['datasets'].items():
            print(f"  {dataset:10s}: {info['n_subjects']:5d} subjects "
                  f"({info['with_edf']:5d} with EDF)")
        
        print("\nChannel coverage (subjects with modality):")
        for modality, count in stats['channel_coverage'].items():
            pct = 100 * count / stats['total_subjects']
            print(f"  {modality:10s}: {count:5d} ({pct:5.1f}%)")
        
        if 'usability' in stats and 'usable_subjects' in stats['usability']:
            print(f"\nUsable subjects (BAS + RESP): {stats['usability']['usable_subjects']} "
                  f"({stats['usability']['usable_percent']:.1f}%)")
        
        print("="*80 + "\n")
