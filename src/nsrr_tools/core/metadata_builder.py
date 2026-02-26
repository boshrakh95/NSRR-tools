"""Metadata builder for unified NSRR dataset catalog."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import pyedflib
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
        
        # Normalize mixed-type columns before saving (e.g., visit numbers)
        # Convert columns with mixed int/str/nan to string for consistency
        for col in unified_df.columns:
            if unified_df[col].dtype == 'object':
                # Check if column has mixed numeric and string types
                sample = unified_df[col].dropna().head(100)
                if len(sample) > 0:
                    has_int = any(isinstance(x, (int, float)) for x in sample)
                    has_str = any(isinstance(x, str) for x in sample)
                    if has_int and has_str:
                        # Mixed types - convert all to string
                        unified_df[col] = unified_df[col].astype(str)
                        unified_df[col] = unified_df[col].replace('nan', None)
        
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
                # Normalize subject ID to match phenotype format (dataset-specific)
                if dataset_name.lower() == 'mros':
                    # MrOS: "mros-visit1-aa0001" -> "AA0001"
                    normalized_id = subject_id.split('-')[-1].upper()
                else:
                    # APPLES/SHHS/STAGES: use subject_id as-is (already extracted by adapter)
                    normalized_id = subject_id
                
                # Read EDF header with pyedflib (fast, only reads header)
                with pyedflib.EdfReader(str(edf_path)) as edf:
                    # Get all channel names and sampling rates
                    ch_names = edf.getSignalLabels()
                    sfreqs = {ch_names[i]: edf.getSampleFrequency(i) 
                             for i in range(edf.signals_in_file)}
                    duration = edf.getFileDuration()
                
                # Detect standardized channels
                detected = self.channel_mapper.detect_channels_from_list(ch_names)
                
                # Group by modality
                modality_groups = self.modality_detector.group_channels_by_modality(detected)
                
                # Get sampling rates per modality (max rate in each modality)
                modality_sfreqs = {}
                for mod, channels in modality_groups.items():
                    if channels:
                        # channels is {standard_name: found_name}, get rates of found channels
                        rates = [sfreqs.get(found_ch, 0) for std_ch, found_ch in channels.items()]
                        modality_sfreqs[f'{mod}_sfreq'] = max(rates) if rates else 0
                
                # For SHHS: Include visit in key to avoid overwriting multi-visit subjects
                # For others: Use normalized_id directly
                if dataset_name.lower() == 'shhs':
                    # Extract visit from EDF path (shhs1 or shhs2)
                    visit = 2 if 'shhs2' in str(edf_path).lower() else 1
                    # Use composite key: nsrrid_visit (to be consistent with merge logic)
                    dict_key = (normalized_id, visit)
                else:
                    dict_key = normalized_id
                
                # Store info (use dict_key for deduplication)
                channel_info[dict_key] = {
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
                
            except Exception as e:
                logger.warning(f"  Error processing {subject_id}: {e}")
                # Use same key logic as success case
                if dataset_name.lower() == 'shhs':
                    visit = 2 if 'shhs2' in str(edf_path).lower() else 1
                    dict_key = (normalized_id, visit)
                else:
                    dict_key = normalized_id
                    
                channel_info[dict_key] = {
                    'edf_path': str(edf_path),
                    'has_edf': False,
                    'error': str(e)
                }
        
        # Merge with phenotype data
        channel_df = pd.DataFrame.from_dict(channel_info, orient='index')
        
        # For SHHS: Extract merge_key and visit from tuple index
        # For others: Index is already the merge_key
        if dataset_name.lower() == 'shhs':
            # Index is tuple (nsrrid, visit)
            channel_df['merge_key'] = [idx[0] for idx in channel_df.index]
            channel_df['psg_visit'] = [idx[1] for idx in channel_df.index]
        else:
            channel_df['merge_key'] = channel_df.index
        
        # Determine merge strategy based on dataset
        id_col = adapter.get_subject_id_column()
        
        if dataset_name.lower() == 'mros':
            # MrOS: Simple merge on subject_id (normalized from EDF filename)
            merged_df = pheno_df.merge(
                channel_df,
                left_on=id_col,
                right_on='merge_key',
                how='inner' if limit else 'left'
            ).copy()
            
            # Use original ID as subject_id (single visit per subject)
            merged_df['subject_id'] = merged_df[id_col].astype(str)
        
        elif dataset_name.lower() == 'apples':
            # APPLES: Multi-step merge strategy with visit 1 baseline data
            # Strategy: Filter to visit 3 (DX/PSG visit) first, then overlay visit 1 (BL) demographics
            # Background: APPLES has 7 visits per subject, but only visit 3 has PSG (fileid)
            #   - Visit 1 (BL): Demographics (age, sex, BMI) and baseline measures
            #   - Visit 3 (DX): PSG recording (has fileid) but demographics are NaN
            #   - We need: ONE row per subject, with visit 1 demographics + visit 3 PSG data
            if 'fileid' in pheno_df.columns and 'visitn' in pheno_df.columns:
                id_col = adapter.get_subject_id_column()
                
                # Step 1: Filter phenotype data to visit 3 (DX/PSG visit) only
                # This is where the EDF recordings are from
                visit3_data = pheno_df[pheno_df['visitn'] == 3].copy()
                logger.info(f"  Loaded {len(pheno_df)} total rows from metadata CSVs")
                logger.info(f"  Filtered to visit 3 (DX/PSG): {len(visit3_data)} subjects")
                
                # Step 2: Create visit 1 (BL/baseline) demographics dataframe
                visit1_data = pheno_df[pheno_df['visitn'] == 1].copy()
                
                # Select demographic and baseline columns from visit 1
                demo_cols = []
                for col in visit1_data.columns:
                    # Demographics and baseline measures (NOT PSG-specific like AHI)
                    if any(term in col.lower() for term in ['age', 'sex', 'race', 'bmi', 'education',
                                                             'mmse', 'cogn', 'phq', 'gad', 'bdi', 'ess',
                                                             'medical', 'history', 'smoker', 'current_smoker']) \
                       and not any(term in col.lower() for term in ['ahi', 'ttleffsp', 'phrnumar', 'pctdursp']) \
                       and col not in [id_col, 'fileid', 'visitn']:
                        demo_cols.append(col)
                
                if demo_cols:
                    visit1_demo = visit1_data[[id_col] + demo_cols].copy()
                    # Rename to indicate baseline origin
                    demo_rename = {col: f'{col}_bl' for col in demo_cols}
                    visit1_demo = visit1_demo.rename(columns=demo_rename)
                    
                    # Step 3: Overlay baseline demographics onto visit 3 data
                    visit3_data = visit3_data.merge(
                        visit1_demo,
                        on=id_col,
                        how='left'
                    )
                    
                    # Replace NaN demographics in visit 3 with baseline values
                    for col in demo_cols:
                        bl_col = f'{col}_bl'
                        if bl_col in visit3_data.columns:
                            if col in visit3_data.columns:
                                # Use baseline value when visit 3 is missing
                                visit3_data[col] = visit3_data[col].fillna(visit3_data[bl_col])
                            else:
                                # Column doesn't exist in visit 3, use baseline
                                visit3_data[col] = visit3_data[bl_col]
                            # Drop the _bl column
                            visit3_data = visit3_data.drop(columns=[bl_col])
                    
                    logger.info(f"  Overlaid visit 1 baseline data for {len(demo_cols)} demographic variables")
                
                # Step 4: Merge visit 3 data (with baseline overlay) with channel data
                merged_df = visit3_data.merge(
                    channel_df,
                    left_on='fileid',
                    right_on='merge_key',
                    how='inner' if limit else 'left'
                ).copy()
                
                logger.info(f"  Result: {len(merged_df)} subjects after merge (one row per subject)")
                edf_count = int(merged_df['has_edf'].sum() if 'has_edf' in merged_df else 0)
                no_edf_count = len(merged_df) - edf_count
                logger.info(f"    - {edf_count} with EDF files")
                logger.info(f"    - {no_edf_count} without EDF (metadata only)")
                logger.info(f"  Demographics from visit 1 (baseline), PSG from visit 3 (diagnosis)")
                
                # Use nsrrid as subject_id (one row per subject)
                merged_df['subject_id'] = merged_df[id_col].astype(str)
            elif 'fileid' in pheno_df.columns:
                # No visitn column, simple merge on fileid (one EDF per subject)
                id_col = adapter.get_subject_id_column()
                merged_df = pheno_df.merge(
                    channel_df,
                    left_on='fileid',
                    right_on='merge_key',
                    how='inner' if limit else 'left'
                ).copy()
                merged_df['subject_id'] = merged_df[id_col].astype(str)
                logger.info(f"  Merged {len(merged_df)} subjects on fileid")
            else:
                # Fallback: merge on subject_id
                id_col = adapter.get_subject_id_column()
                logger.warning(f"  No 'fileid' in APPLES metadata, using subject_id merge")
                merged_df = pheno_df.merge(
                    channel_df,
                    left_on=id_col,
                    right_on='merge_key',
                    how='inner' if limit else 'left'
                ).copy()
                merged_df['subject_id'] = merged_df[id_col].astype(str)
        
        elif dataset_name.lower() in ['stages', 'shhs']:
            # STAGES/SHHS merge
            # Convert types if needed (SHHS has int nsrrid, EDF filenames are strings)
            if id_col in pheno_df.columns:
                # Convert pheno_df ID to string for consistent merging
                pheno_df[id_col] = pheno_df[id_col].astype(str)
            
            # For SHHS: Extract visit from phenotype data to handle multi-visit subjects
            # Note: channel_df already has psg_visit from earlier processing
            if dataset_name.lower() == 'shhs':
                # Extract visit from phenotype data (visitnumber column if present)
                if 'visitnumber' in pheno_df.columns:
                    pheno_df['psg_visit'] = pheno_df['visitnumber'].fillna(1).astype(int)
                else:
                    # If no visitnumber, assume visit 1 for all
                    pheno_df['psg_visit'] = 1
                
                # Merge on both nsrrid AND visit
                merged_df = pheno_df.merge(
                    channel_df,
                    left_on=[id_col, 'psg_visit'],
                    right_on=['merge_key', 'psg_visit'],
                    how='inner' if limit else 'left'
                ).copy()
                
                # Create composite subject_id to treat each visit as a distinct subject
                # Format: {nsrrid}_v{visit} (e.g., 200001_v1, 200001_v2)
                merged_df['subject_id'] = merged_df[id_col].astype(str) + '_v' + merged_df['psg_visit'].astype(str)
                
                logger.info(f"  SHHS visits: {merged_df['psg_visit'].value_counts().sort_index().to_dict()}")
                logger.info(f"  Created {len(merged_df)} unique visit records with composite subject_ids")
            else:
                # STAGES: Simple merge on subject_id only
                merged_df = pheno_df.merge(
                    channel_df,
                    left_on=id_col,
                    right_on='merge_key',
                    how='inner' if limit else 'left'
                ).copy()
                
                # Use original ID as subject_id (single visit per subject)
                merged_df['subject_id'] = merged_df[id_col].astype(str)
        
        else:
            # Unknown dataset: try subject_id merge
            logger.warning(f"Unknown dataset {dataset_name}, using default subject_id merge")
            merged_df = pheno_df.merge(
                channel_df,
                left_on=id_col,
                right_on='merge_key',
                how='inner' if limit else 'left'
            ).copy()
            
            # Use original ID as subject_id
            merged_df['subject_id'] = merged_df[id_col].astype(str)
        
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
        
        # Check for cognitive/psychiatric scores (dataset-specific, from nocturn ontology)
        if dataset_name.lower() == 'apples':
            # APPLES: mmsetotalscore (MMSE - cognitive function, visit 1)
            if 'mmsetotalscore' in df.columns:
                df['has_cognitive'] = df['mmsetotalscore'].notna()
            else:
                df['has_cognitive'] = False
        
        elif dataset_name.lower() == 'stages':
            # STAGES: Cognitive scores from external files (ISI, PHQ-9, GAD-7)
            # NOTE: These require complex Excel file loading - placeholder for now
            # Will implement after base metadata extraction is complete
            # Expected columns: isi_score, phq_1000, gad_0800, fss_1000
            df['has_cognitive'] = False
            # TODO: Load and merge STAGES cognitive data from:
            # - STAGES ASQ ISI to DIET 20200513 Final deidentified.xlsx (isi_score)
            # - stages-dataset-0.3.0.csv (phq_1000, gad_0800, fss_1000)
        
        else:
            # MrOS, SHHS: No cognitive/psychiatric scores in standard CSVs
            df['has_cognitive'] = False
        
        # Check for demographics (age, sex, BMI)
        # Look for columns containing these keywords (e.g., nsrr_age, age_at_visit, vsage1)
        age_cols = [c for c in df.columns if 'age' in c.lower() and 'gt89' not in c.lower()]
        sex_cols = [c for c in df.columns if 'sex' in c.lower() or 'gender' in c.lower() or c.lower() == 'male']
        bmi_cols = [c for c in df.columns if 'bmi' in c.lower()]
        
        if age_cols:
            # Use first age column found (prefer nsrr_age over others)
            age_col = next((c for c in age_cols if 'nsrr' in c.lower()), age_cols[0])
            df['has_age'] = df[age_col].notna()
        else:
            df['has_age'] = False
            
        if sex_cols:
            # Use first sex column found (prefer nsrr_sex over others)
            sex_col = next((c for c in sex_cols if 'nsrr' in c.lower()), sex_cols[0])
            df['has_sex'] = df[sex_col].notna()
        else:
            df['has_sex'] = False
            
        if bmi_cols:
            # Use first BMI column found (prefer nsrr_bmi over others)
            bmi_col = next((c for c in bmi_cols if 'nsrr' in c.lower()), bmi_cols[0])
            df['has_bmi'] = df[bmi_col].notna()
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
