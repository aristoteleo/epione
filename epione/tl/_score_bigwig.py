#!/usr/bin/env python

"""
ScoreBigwig: Calculate footprint scores from corrected cutsites
Adapted from TOBIAS ScoreBigwig functionality

@author: Zehua Zeng
@contact: starlitnightly@gmail.com
@license: GPL 3.0

This module calculates footprint scores from corrected ATAC-seq cutsite signals
"""

import os
import sys
import numpy as np
import pandas as pd
import pyBigWig
from typing import Union, List, Dict, Optional, Tuple, Any
import logging
import multiprocessing as mp
from scipy import stats, ndimage
import warnings

# Try to import Cython optimized functions
try:
    from ..utils._footprint_cython import (
        tobias_footprint_array, 
        fos_score_array,
        fast_rolling_mean,
        fast_rolling_max
    )
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    # Only show warning once and make it less prominent
    import os
    if not os.environ.get('EPIONE_CYTHON_WARNING_SHOWN'):
        warnings.warn(
            "Cython optimizations not available. Install Cython and recompile for better performance. "
            "Using Python fallback implementations.",
            UserWarning,
            stacklevel=2
        )
        os.environ['EPIONE_CYTHON_WARNING_SHOWN'] = '1'


class FootprintScorer:
    """Class for calculating footprint scores from cutsite signals"""
    
    def __init__(
        self,
        method: str = 'tobias',
        flank_min: int = 5,
        flank_max: int = 30,
        fp_min: int = 8,
        fp_max: int = 40,
        smooth: int = 1,
        abs_values: bool = False,
        min_limit: Optional[float] = None,
        max_limit: Optional[float] = None,
        region_flank: int = 100,
        verbose: bool = True
    ):
        """
        Initialize FootprintScorer
        
        Parameters
        ----------
        method : str
            Scoring method ('tobias' or 'fos')
        flank_min : int
            Minimum flank window size
        flank_max : int
            Maximum flank window size
        fp_min : int
            Minimum footprint window size
        fp_max : int
            Maximum footprint window size
        smooth : int
            Smoothing window size
        abs_values : bool
            Use absolute values
        min_limit : Optional[float]
            Minimum signal value limit
        max_limit : Optional[float]
            Maximum signal value limit
        region_flank : int
            Flanking region around scored regions
        verbose : bool
            Verbose output
        """
        
        self.method = method
        self.flank_min = flank_min
        self.flank_max = flank_max
        self.fp_min = fp_min
        self.fp_max = fp_max
        self.smooth = smooth
        self.abs_values = abs_values
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.region_flank = region_flank
        self.verbose = verbose
        
        # Validate parameters
        if method not in ['tobias', 'fos']:
            raise ValueError(f"Unknown method: {method}. Must be 'tobias' or 'fos'")
        
        if flank_min < 1 or flank_max < flank_min:
            raise ValueError("Invalid flank window sizes")
        
        if fp_min < 1 or fp_max < fp_min:
            raise ValueError("Invalid footprint window sizes")
    
    def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Preprocess signal array
        
        Parameters
        ----------
        signal : np.ndarray
            Raw signal array
            
        Returns
        -------
        np.ndarray
            Processed signal array
        """
        
        # Handle NaN values
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply absolute values if requested
        if self.abs_values:
            signal = np.abs(signal)
        
        # Apply limits
        if self.min_limit is not None:
            signal = np.maximum(signal, self.min_limit)
        if self.max_limit is not None:
            signal = np.minimum(signal, self.max_limit)
        
        # Apply smoothing
        if self.smooth > 1:
            if CYTHON_AVAILABLE:
                signal = fast_rolling_mean(signal.astype(np.float64), self.smooth)
            else:
                # Use scipy for smoothing
                window = np.ones(self.smooth) / self.smooth
                signal = np.convolve(signal, window, mode='same')
        
        return signal.astype(np.float64)
    
    def calculate_scores_python(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate footprint scores using pure Python (fallback)
        
        Parameters
        ----------
        signal : np.ndarray
            Preprocessed signal array
            
        Returns
        -------
        np.ndarray
            Footprint scores
        """
        
        signal_len = len(signal)
        scores = np.zeros(signal_len, dtype=np.float64)
        
        for i in range(signal_len):
            if self.method == 'tobias':
                max_score = -np.inf
                
                for flank_size in range(self.flank_min, self.flank_max + 1):
                    for fp_size in range(self.fp_min, self.fp_max + 1):
                        # Define windows
                        left_start = max(0, i - flank_size - fp_size // 2)
                        left_end = max(0, i - fp_size // 2)
                        fp_start = max(0, i - fp_size // 2)
                        fp_end = min(signal_len, i + fp_size // 2 + 1)
                        right_start = min(signal_len, i + fp_size // 2 + 1)
                        right_end = min(signal_len, i + flank_size + fp_size // 2 + 1)
                        
                        # Check window validity
                        if (left_end - left_start < 1 or 
                            fp_end - fp_start < 1 or 
                            right_end - right_start < 1):
                            continue
                        
                        # Calculate means
                        left_mean = np.mean(signal[left_start:left_end])
                        fp_mean = np.mean(signal[fp_start:fp_end])
                        right_mean = np.mean(signal[right_start:right_end])
                        
                        # Calculate TOBIAS score
                        flank_mean = (left_mean + right_mean) / 2
                        score = flank_mean - fp_mean
                        
                        if score > max_score:
                            max_score = score
                
                scores[i] = max_score if max_score != -np.inf else 0.0
                
            elif self.method == 'fos':
                min_score = np.inf
                
                for flank_size in range(self.flank_min, self.flank_max + 1):
                    for fp_size in range(self.fp_min, self.fp_max + 1):
                        # Define windows
                        left_start = max(0, i - flank_size - fp_size // 2)
                        left_end = max(0, i - fp_size // 2)
                        fp_start = max(0, i - fp_size // 2)
                        fp_end = min(signal_len, i + fp_size // 2 + 1)
                        right_start = min(signal_len, i + fp_size // 2 + 1)
                        right_end = min(signal_len, i + flank_size + fp_size // 2 + 1)
                        
                        # Check window validity
                        if (left_end - left_start < 1 or 
                            fp_end - fp_start < 1 or 
                            right_end - right_start < 1):
                            continue
                        
                        # Calculate means
                        left_mean = np.mean(signal[left_start:left_end])
                        fp_mean = np.mean(signal[fp_start:fp_end])
                        right_mean = np.mean(signal[right_start:right_end])
                        
                        # Calculate FOS score
                        if (left_mean > 0 and right_mean > 0 and 
                            fp_mean < left_mean and fp_mean < right_mean):
                            score = (fp_mean + 1) / left_mean + (fp_mean + 1) / right_mean
                        else:
                            score = np.inf
                        
                        if score < min_score:
                            min_score = score
                
                scores[i] = min_score if min_score != np.inf else 0.0
        
        return scores
    
    def calculate_scores(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate footprint scores for signal array
        
        Parameters
        ----------
        signal : np.ndarray
            Signal array
            
        Returns
        -------
        np.ndarray
            Footprint scores
        """
        
        # Preprocess signal
        processed_signal = self.preprocess_signal(signal)
        
        # Use Cython implementation if available
        if CYTHON_AVAILABLE:
            if self.method == 'tobias':
                scores = tobias_footprint_array(
                    processed_signal, 
                    self.flank_min, self.flank_max,
                    self.fp_min, self.fp_max
                )
            else:  # fos
                scores = fos_score_array(
                    processed_signal,
                    self.flank_min, self.flank_max,
                    self.fp_min, self.fp_max
                )
        else:
            # Fallback to Python implementation
            scores = self.calculate_scores_python(processed_signal)
        
        return scores
    
    def score_region(
        self,
        bigwig_file: str,
        region: Tuple[str, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score a single genomic region
        
        Parameters
        ----------
        bigwig_file : str
            Path to BigWig file
        region : Tuple[str, int, int]
            Genomic region (chrom, start, end)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (signal, scores) arrays
        """
        
        chrom, start, end = region
        
        # Extend region for flanking
        ext_start = max(0, start - self.region_flank)
        ext_end = end + self.region_flank
        
        # Read signal from BigWig
        with pyBigWig.open(bigwig_file) as bw:
            try:
                signal_values = bw.values(chrom, ext_start, ext_end)
                if signal_values is None:
                    signal_array = np.zeros(ext_end - ext_start)
                else:
                    signal_array = np.array(signal_values, dtype=float)
            except:
                signal_array = np.zeros(ext_end - ext_start)
        
        # Calculate scores
        scores = self.calculate_scores(signal_array)
        
        # Extract central region (without flanks)
        central_start = self.region_flank
        central_end = central_start + (end - start)
        
        if central_end <= len(scores):
            central_signal = signal_array[central_start:central_end]
            central_scores = scores[central_start:central_end]
        else:
            # Handle edge case
            central_signal = signal_array[:end-start]
            central_scores = scores[:end-start]
            if len(central_scores) < end - start:
                # Pad with zeros
                padding = end - start - len(central_scores)
                central_scores = np.concatenate([central_scores, np.zeros(padding)])
                central_signal = np.concatenate([central_signal, np.zeros(padding)])
        
        return central_signal, central_scores


# Global function for multiprocessing - needs to be at module level
def _score_single_region_worker(args):
    """Worker function for parallel region scoring"""
    region, signal_file, scorer_params = args
    
    try:
        # Create scorer instance
        scorer = FootprintScorer(**scorer_params)
        
        # Score the region
        signal, scores = scorer.score_region(signal_file, region)
        region_key = f"{region[0]}:{region[1]}-{region[2]}"
        return region_key, signal, scores
    except Exception as e:
        if scorer_params.get('verbose', False):
            print(f"Error processing region {region}: {e}")
        return None, None, None


def score_bigwig_core(
    signal_file: str,
    regions: List[Tuple[str, int, int]],
    method: str = 'tobias',
    output_file: Optional[str] = None,
    n_cores: int = 1,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Core function to calculate footprint scores from BigWig file
    
    Parameters
    ----------
    signal_file : str
        Path to BigWig signal file
    regions : List[Tuple[str, int, int]]
        List of regions to score
    method : str
        Scoring method ('tobias' or 'fos')
    output_file : Optional[str]
        Output BigWig file path
    n_cores : int
        Number of CPU cores to use
    **kwargs
        Additional parameters for FootprintScorer
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of region -> scores
    """
    
    # Initialize scorer parameters
    scorer_params = {'method': method, **kwargs}
    scorer = FootprintScorer(**scorer_params)
    
    if scorer.verbose:
        print(f"Calculating {method.upper()} footprint scores...")
        print(f"- Signal file: {signal_file}")
        print(f"- Regions: {len(regions)}")
        print(f"- Using {n_cores} cores")
    
    # Process regions
    results = {}
    signals = {}
    
    if n_cores > 1:
        # Parallel processing - prepare arguments
        worker_args = [(region, signal_file, scorer_params) for region in regions]
        
        try:
            from tqdm import tqdm
            with mp.Pool(n_cores) as pool:
                pool_results = list(tqdm(
                    pool.imap(_score_single_region_worker, worker_args),
                    total=len(worker_args),
                    desc="Processing regions (parallel)",
                    disable=not scorer.verbose
                ))
        except ImportError:
            # Fallback without progress bar
            with mp.Pool(n_cores) as pool:
                pool_results = pool.map(_score_single_region_worker, worker_args)
        
        for region_key, signal, scores in pool_results:
            if region_key is not None:
                results[region_key] = scores
                signals[region_key] = signal
    else:
        # Sequential processing
        from tqdm import tqdm
        for region in tqdm(regions, desc="Processing regions", disable=not scorer.verbose):
            
            try:
                signal, scores = scorer.score_region(signal_file, region)
                region_key = f"{region[0]}:{region[1]}-{region[2]}"
                results[region_key] = scores
                signals[region_key] = signal
            except Exception as e:
                if scorer.verbose:
                    print(f"Error processing region {region}: {e}")
                continue
    
    # Save to BigWig if requested
    if output_file:
        save_scores_to_bigwig(results, signal_file, output_file)
    
    if scorer.verbose:
        print(f"Completed scoring {len(results)} regions")
    
    return results


def save_scores_to_bigwig(
    scores_dict: Dict[str, np.ndarray],
    reference_bigwig: str,
    output_file: str
):
    """
    Save footprint scores to BigWig file
    
    Parameters
    ----------
    scores_dict : Dict[str, np.ndarray]
        Dictionary of region_key -> scores
    reference_bigwig : str
        Reference BigWig for chromosome information
    output_file : str
        Output BigWig file path
    """
    
    # Get chromosome sizes from reference BigWig
    with pyBigWig.open(reference_bigwig) as ref_bw:
        chrom_sizes = ref_bw.chroms()
    
    # Create output BigWig
    with pyBigWig.open(output_file, "w") as out_bw:
        # Add header
        chrom_list = [(chrom, size) for chrom, size in chrom_sizes.items()]
        out_bw.addHeader(chrom_list)
        
        # Add entries
        for region_key, scores in scores_dict.items():
            try:
                chrom, coords = region_key.split(':')
                start, end = map(int, coords.split('-'))
                
                if len(scores) == end - start:
                    positions = list(range(start, end))
                    values = scores.tolist()
                    out_bw.addEntries(chrom, positions, values=values, span=1)
            except Exception as e:
                print(f"Warning: Could not write region {region_key}: {e}")
    
    print(f"Saved scores to {output_file}")


def calculate_aggregate_scores(
    scores_dict: Dict[str, np.ndarray],
    method: str = 'mean'
) -> Dict[str, Any]:
    """
    Calculate aggregate statistics from footprint scores
    
    Parameters
    ----------
    scores_dict : Dict[str, np.ndarray]
        Dictionary of scores by region
    method : str
        Aggregation method ('mean', 'median', 'max', 'sum')
        
    Returns
    -------
    Dict[str, Any]
        Aggregate statistics
    """
    
    if not scores_dict:
        return {}
    
    # Collect all scores
    all_scores = np.concatenate(list(scores_dict.values()))
    
    # Remove infinite and NaN values
    valid_scores = all_scores[np.isfinite(all_scores)]
    
    if len(valid_scores) == 0:
        return {'error': 'No valid scores found'}
    
    # Calculate statistics
    stats = {
        'n_regions': len(scores_dict),
        'n_positions': len(all_scores),
        'n_valid': len(valid_scores),
        'mean': np.mean(valid_scores),
        'median': np.median(valid_scores),
        'std': np.std(valid_scores),
        'min': np.min(valid_scores),
        'max': np.max(valid_scores),
        'q25': np.percentile(valid_scores, 25),
        'q75': np.percentile(valid_scores, 75)
    }
    
    # Method-specific aggregation
    if method == 'mean':
        stats['aggregate_score'] = stats['mean']
    elif method == 'median':
        stats['aggregate_score'] = stats['median']
    elif method == 'max':
        stats['aggregate_score'] = stats['max']
    elif method == 'sum':
        stats['aggregate_score'] = np.sum(valid_scores)
    else:
        stats['aggregate_score'] = stats['mean']
    
    return stats


# Convenience function
def score_bigwig(
    signal_file: str,
    regions_bed: str,
    output_file: str,
    method: str = 'tobias',
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Convenient wrapper for footprint scoring with BED input
    
    Parameters
    ----------
    signal_file : str
        Input BigWig signal file
    regions_bed : str
        BED file with regions to score
    output_file : str
        Output BigWig file
    method : str
        Scoring method
    **kwargs
        Additional parameters
        
    Returns
    -------
    Dict[str, np.ndarray]
        Footprint scores by region
    """
    
    # Read regions from BED file
    regions = []
    with open(regions_bed, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    chrom, start, end = parts[0], int(parts[1]), int(parts[2])
                    regions.append((chrom, start, end))
    
    # Calculate scores
    results = score_bigwig_core(
        signal_file=signal_file,
        regions=regions,
        method=method,
        output_file=output_file,
        **kwargs
    )
    
    # Calculate and print summary statistics
    stats = calculate_aggregate_scores(results)
    if 'error' not in stats:
        print(f"\nSummary Statistics:")
        print(f"- Regions processed: {stats['n_regions']}")
        print(f"- Total positions: {stats['n_positions']}")
        print(f"- Valid scores: {stats['n_valid']}")
        print(f"- Mean score: {stats['mean']:.3f}")
        print(f"- Score range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    return results


def compare_methods(
    signal_file: str,
    regions: List[Tuple[str, int, int]],
    methods: List[str] = ['tobias', 'fos'],
    **kwargs
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compare different footprint scoring methods
    
    Parameters
    ----------
    signal_file : str
        Input BigWig signal file
    regions : List[Tuple[str, int, int]]
        Regions to score
    methods : List[str]
        Methods to compare
    **kwargs
        Additional parameters
        
    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Results by method and region
    """
    
    results = {}
    
    for method in methods:
        print(f"\nCalculating scores using {method.upper()} method...")
        method_results = score_bigwig_core(
            signal_file=signal_file,
            regions=regions,
            method=method,
            **kwargs
        )
        results[method] = method_results
        
        # Print statistics
        stats = calculate_aggregate_scores(method_results)
        if 'error' not in stats:
            print(f"{method.upper()} - Mean score: {stats['mean']:.3f}, "
                  f"Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    return results