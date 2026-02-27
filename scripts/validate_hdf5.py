#!/usr/bin/env python3
"""
Validate Preprocessed HDF5 Files
================================

Validates HDF5 signal files and annotation files for correctness.

Checks:
- HDF5 file structure and metadata
- Channel data quality (NaN, Inf, outliers)
- Normalization correctness (mean ≈ 0, std ≈ 1)
- Annotation file existence and format
- Signal/annotation synchronization

Usage:
    python scripts/validate_hdf5.py --file /path/to/subject.h5
    python scripts/validate_hdf5.py --dataset stages --num-samples 10

Author: NSRR Preprocessing Pipeline
Date: February 2026
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import h5py
import numpy as np
from loguru import logger
import pandas as pd


class HDF5Validator:
    """Validate preprocessed HDF5 files."""
    
    # Expected sampling rate
    EXPECTED_SR = 128  # Hz
    
    # Normalization tolerances
    MEAN_TOLERANCE = 0.1
    STD_TOLERANCE = 0.2
    
    def __init__(self):
        """Initialize validator."""
        logger.info("HDF5 Validator initialized")
    
    def validate_file(
        self,
        hdf5_path: Path,
        annotation_path: Optional[Path] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Validate a single HDF5 file.
        
        Args:
            hdf5_path: Path to HDF5 file
            annotation_path: Optional path to annotation file
            verbose: Print detailed validation results
        
        Returns:
            Dictionary with validation results
        """
        if verbose:
            logger.info(f"\nValidating: {hdf5_path.name}")
            logger.info(f"{'='*80}")
        
        results = {
            'file': str(hdf5_path),
            'exists': hdf5_path.exists(),
            'valid': False,
            'checks': []
        }
        
        if not hdf5_path.exists():
            logger.error(f"File not found: {hdf5_path}")
            return results
        
        try:
            with h5py.File(hdf5_path, 'r') as hf:
                # Check 1: File structure
                check_structure = self._check_structure(hf, verbose)
                results['checks'].append(check_structure)
                
                # Check 2: Channel data quality
                check_quality = self._check_data_quality(hf, verbose)
                results['checks'].append(check_quality)
                
                # Check 3: Normalization
                check_norm = self._check_normalization(hf, verbose)
                results['checks'].append(check_norm)
                
                # Check 4: Sampling rate
                check_sr = self._check_sampling_rate(hf, verbose)
                results['checks'].append(check_sr)
                
                # Check 5: Annotations (if provided)
                if annotation_path and annotation_path.exists():
                    check_annot = self._check_annotations(
                        hf, annotation_path, verbose
                    )
                    results['checks'].append(check_annot)
            
            # Overall validation status
            results['valid'] = all(check['passed'] for check in results['checks'])
            
            if verbose:
                if results['valid']:
                    logger.success("✓ All validation checks passed")
                else:
                    logger.warning("✗ Some validation checks failed")
                logger.info(f"{'='*80}\n")
            
            return results
            
        except Exception as e:
            logger.error(f"Error validating file: {e}")
            results['error'] = str(e)
            return results
    
    def _check_structure(self, hf: h5py.File, verbose: bool) -> Dict[str, Any]:
        """Check HDF5 file structure."""
        try:
            # Get all datasets (channels)
            channels = list(hf.keys())
            
            # Check for required attributes
            has_sr = 'sampling_rate' in hf.attrs
            has_duration = 'duration_seconds' in hf.attrs
            has_num_channels = 'num_channels' in hf.attrs
            
            passed = len(channels) > 0 and has_sr
            
            if verbose:
                logger.info(f"Structure Check:")
                logger.info(f"  Channels: {len(channels)}")
                logger.info(f"  Channel names: {', '.join(channels[:5])}{'...' if len(channels) > 5 else ''}")
                logger.info(f"  Has sampling_rate attr: {has_sr}")
                logger.info(f"  Has duration_seconds attr: {has_duration}")
                
            return {
                'name': 'structure',
                'passed': passed,
                'num_channels': len(channels),
                'channels': channels,
                'has_metadata': has_sr and has_duration
            }
            
        except Exception as e:
            return {
                'name': 'structure',
                'passed': False,
                'error': str(e)
            }
    
    def _check_data_quality(self, hf: h5py.File, verbose: bool) -> Dict[str, Any]:
        """Check for NaN, Inf, and outliers."""
        try:
            channels = list(hf.keys())
            issues = []
            
            for channel in channels:
                data = hf[channel][:]
                
                # Check for NaN
                if np.isnan(data).any():
                    issues.append(f"{channel}: Contains NaN values")
                
                # Check for Inf
                if np.isinf(data).any():
                    issues.append(f"{channel}: Contains Inf values")
                
                # Check for extreme outliers (> 10 std from mean)
                if not np.isnan(data).all():
                    mean = np.nanmean(data)
                    std = np.nanstd(data)
                    if std > 0:
                        outliers = np.abs(data - mean) > 10 * std
                        if outliers.any():
                            issues.append(f"{channel}: Contains extreme outliers ({outliers.sum()} samples)")
            
            passed = len(issues) == 0
            
            if verbose:
                logger.info(f"Data Quality Check:")
                if passed:
                    logger.info(f"  ✓ No NaN/Inf/outliers detected")
                else:
                    logger.warning(f"  ✗ {len(issues)} issues found:")
                    for issue in issues[:5]:
                        logger.warning(f"    - {issue}")
                    if len(issues) > 5:
                        logger.warning(f"    ... and {len(issues) - 5} more")
            
            return {
                'name': 'data_quality',
                'passed': passed,
                'issues': issues
            }
            
        except Exception as e:
            return {
                'name': 'data_quality',
                'passed': False,
                'error': str(e)
            }
    
    def _check_normalization(self, hf: h5py.File, verbose: bool) -> Dict[str, Any]:
        """Check if data is properly normalized."""
        try:
            channels = list(hf.keys())
            norm_stats = {}
            issues = []
            
            for channel in channels:
                data = hf[channel][:]
                
                mean = np.mean(data)
                std = np.std(data)
                
                norm_stats[channel] = {'mean': mean, 'std': std}
                
                # Check if close to 0 mean and 1 std
                if abs(mean) > self.MEAN_TOLERANCE:
                    issues.append(f"{channel}: mean = {mean:.3f} (expected ≈ 0)")
                
                if abs(std - 1.0) > self.STD_TOLERANCE:
                    issues.append(f"{channel}: std = {std:.3f} (expected ≈ 1)")
            
            passed = len(issues) == 0
            
            if verbose:
                logger.info(f"Normalization Check:")
                avg_mean = np.mean([s['mean'] for s in norm_stats.values()])
                avg_std = np.mean([s['std'] for s in norm_stats.values()])
                logger.info(f"  Average mean: {avg_mean:.4f} (expected ≈ 0)")
                logger.info(f"  Average std: {avg_std:.4f} (expected ≈ 1)")
                
                if passed:
                    logger.info(f"  ✓ All channels properly normalized")
                else:
                    logger.warning(f"  ✗ {len(issues)} channels with normalization issues")
            
            return {
                'name': 'normalization',
                'passed': passed,
                'issues': issues,
                'stats': norm_stats
            }
            
        except Exception as e:
            return {
                'name': 'normalization',
                'passed': False,
                'error': str(e)
            }
    
    def _check_sampling_rate(self, hf: h5py.File, verbose: bool) -> Dict[str, Any]:
        """Check sampling rate consistency."""
        try:
            sr = hf.attrs.get('sampling_rate', None)
            duration = hf.attrs.get('duration_seconds', None)
            
            if sr is None:
                return {
                    'name': 'sampling_rate',
                    'passed': False,
                    'error': 'sampling_rate attribute missing'
                }
            
            # Check if matches expected
            passed = sr == self.EXPECTED_SR
            
            # Verify with actual data
            if duration is not None:
                channels = list(hf.keys())
                if channels:
                    actual_length = len(hf[channels[0]])
                    expected_length = int(duration * sr)
                    length_match = abs(actual_length - expected_length) < sr  # Allow 1s tolerance
                    
                    if verbose:
                        logger.info(f"Sampling Rate Check:")
                        logger.info(f"  Sampling rate: {sr} Hz (expected {self.EXPECTED_SR} Hz)")
                        logger.info(f"  Duration: {duration:.1f} seconds")
                        logger.info(f"  Expected samples: {expected_length}")
                        logger.info(f"  Actual samples: {actual_length}")
                        logger.info(f"  ✓ Sampling rate correct" if passed and length_match else f"  ✗ Sampling rate mismatch")
                    
                    passed = passed and length_match
            
            return {
                'name': 'sampling_rate',
                'passed': passed,
                'sampling_rate': sr,
                'duration': duration
            }
            
        except Exception as e:
            return {
                'name': 'sampling_rate',
                'passed': False,
                'error': str(e)
            }
    
    def _check_annotations(
        self,
        hf: h5py.File,
        annotation_path: Path,
        verbose: bool
    ) -> Dict[str, Any]:
        """Check annotation file and synchronization."""
        try:
            # Load annotations
            stages = np.load(annotation_path)
            
            # Get signal duration
            duration = hf.attrs.get('duration_seconds', None)
            sr = hf.attrs.get('sampling_rate', self.EXPECTED_SR)
            
            if duration is None:
                channels = list(hf.keys())
                if channels:
                    duration = len(hf[channels[0]]) / sr
            
            # Calculate expected epochs
            expected_epochs = int(duration / 30)  # 30-sec epochs
            actual_epochs = len(stages)
            
            # Check synchronization (allow small tolerance)
            diff = abs(expected_epochs - actual_epochs)
            synchronized = diff <= 2  # Allow 2 epochs (60s) difference
            
            if verbose:
                logger.info(f"Annotation Check:")
                logger.info(f"  Annotation file: {annotation_path.name}")
                logger.info(f"  Number of epochs: {actual_epochs}")
                logger.info(f"  Expected epochs: {expected_epochs}")
                logger.info(f"  Difference: {diff} epochs ({diff * 30}s)")
                
                # Stage distribution
                unique, counts = np.unique(stages, return_counts=True)
                logger.info(f"  Stage distribution:")
                stage_names = {-1: 'Unknown', 0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 5: 'REM'}
                for stage, count in zip(unique, counts):
                    name = stage_names.get(stage, f'Stage {stage}')
                    logger.info(f"    {name}: {count} ({100*count/len(stages):.1f}%)")
                
                if synchronized:
                    logger.info(f"  ✓ Annotations synchronized with signal")
                else:
                    logger.warning(f"  ✗ Annotations not synchronized (diff: {diff} epochs)")
            
            return {
                'name': 'annotations',
                'passed': synchronized,
                'actual_epochs': actual_epochs,
                'expected_epochs': expected_epochs,
                'difference': diff,
                'synchronized': synchronized
            }
            
        except Exception as e:
            return {
                'name': 'annotations',
                'passed': False,
                'error': str(e)
            }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate preprocessed HDF5 files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--file',
        type=Path,
        help='Path to single HDF5 file to validate'
    )
    
    parser.add_argument(
        '--annotation',
        type=Path,
        help='Path to annotation file (for validation with --file)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['stages', 'shhs', 'apples', 'mros'],
        help='Validate random samples from dataset'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of random samples to validate (with --dataset)'
    )
    
    parser.add_argument(
        '--base-path',
        type=Path,
        default=Path('/scratch/boshra95/psg'),
        help='Base output path'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    validator = HDF5Validator()
    
    if args.file:
        # Validate single file
        results = validator.validate_file(
            hdf5_path=args.file,
            annotation_path=args.annotation,
            verbose=True
        )
        
        sys.exit(0 if results['valid'] else 1)
    
    elif args.dataset:
        # Validate random samples from dataset
        hdf5_dir = args.base_path / args.dataset / 'derived' / 'hdf5_signals'
        annot_dir = args.base_path / args.dataset / 'derived' / 'annotations'
        
        if not hdf5_dir.exists():
            logger.error(f"HDF5 directory not found: {hdf5_dir}")
            sys.exit(1)
        
        # Get all HDF5 files
        hdf5_files = list(hdf5_dir.glob('*.h5'))
        
        if len(hdf5_files) == 0:
            logger.error(f"No HDF5 files found in {hdf5_dir}")
            sys.exit(1)
        
        logger.info(f"Found {len(hdf5_files)} HDF5 files in {args.dataset}")
        logger.info(f"Validating {min(args.num_samples, len(hdf5_files))} random samples...\n")
        
        # Random sample
        import random
        sample_files = random.sample(hdf5_files, min(args.num_samples, len(hdf5_files)))
        
        all_valid = True
        for hdf5_file in sample_files:
            # Find corresponding annotation
            annot_file = annot_dir / f"{hdf5_file.stem}_stages.npy"
            if not annot_file.exists():
                annot_file = None
            
            results = validator.validate_file(
                hdf5_path=hdf5_file,
                annotation_path=annot_file,
                verbose=True
            )
            
            if not results['valid']:
                all_valid = False
        
        logger.info(f"\n{'='*80}")
        if all_valid:
            logger.success(f"✓ All {len(sample_files)} samples passed validation")
        else:
            logger.warning(f"✗ Some samples failed validation")
        logger.info(f"{'='*80}\n")
        
        sys.exit(0 if all_valid else 1)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
