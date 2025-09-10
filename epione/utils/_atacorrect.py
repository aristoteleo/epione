#!/usr/bin/env python

"""
ATACorrect: Tn5 bias correction for ATAC-seq data
Adapted from TOBIAS ATACorrect functionality

@author: Zehua Zeng
@contact: starlitnightly@gmail.com  
@license: GPL 3.0

This module provides Tn5 transposase bias correction for ATAC-seq data
"""

import os
import sys
import numpy as np
import pandas as pd
import pysam
import pyBigWig
from typing import Union, List, Dict, Optional, Tuple, Any
import pickle
import logging
from scipy.stats import pearsonr
import multiprocessing as mp
from collections import defaultdict

# Sequence encoding
NUCL_DICT = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


class GenomicSequence:
    """Class for handling genomic sequences efficiently"""
    
    def __init__(self, region: Tuple[str, int, int]):
        """
        Initialize genomic sequence
        
        Parameters
        ----------
        region : Tuple[str, int, int]
            Genomic region (chrom, start, end)
        """
        self.chrom, self.start, self.end = region
        self.length = self.end - self.start
        self.sequence = None
        self.revcomp = None
    
    def from_fasta(self, fasta_file: str):
        """
        Load sequence from FASTA file
        
        Parameters
        ----------
        fasta_file : str
            Path to genome FASTA file
        """
        with pysam.FastaFile(fasta_file) as fasta:
            seq_str = fasta.fetch(self.chrom, self.start, self.end).upper()
            
            # Convert to numeric array
            self.sequence = np.array([NUCL_DICT.get(base, 4) for base in seq_str], dtype=np.int8)
            
            # Create reverse complement
            rev_seq_str = "".join([COMPLEMENT.get(base, "N") for base in seq_str[::-1]])
            self.revcomp = np.array([NUCL_DICT.get(base, 4) for base in rev_seq_str], dtype=np.int8)
    
    def get_kmer(self, position: int, k_flank: int = 12, strand: str = "+") -> np.ndarray:
        """
        Get k-mer sequence around position
        
        Parameters
        ----------
        position : int
            Position relative to sequence start
        k_flank : int
            Flanking length on each side
        strand : str
            Strand ('+' or '-')
            
        Returns
        -------
        np.ndarray
            K-mer sequence as numeric array
        """
        total_len = 2 * k_flank + 1
        start_pos = position - k_flank
        end_pos = position + k_flank + 1
        
        if start_pos < 0 or end_pos > self.length:
            return np.full(total_len, 4, dtype=np.int8)  # N's for out-of-bounds
        
        if strand == "+":
            return self.sequence[start_pos:end_pos].copy()
        else:
            return self.revcomp[start_pos:end_pos].copy()


class SequenceMatrix:
    """Base class for sequence bias matrices"""
    
    def __init__(self, length: int):
        """
        Initialize sequence matrix
        
        Parameters
        ----------
        length : int
            K-mer length
        """
        self.length = length
        self.no_bias = 0
        self.no_bg = 0
        
    def add_sequence(self, sequence: np.ndarray, is_background: bool = False):
        """Add sequence to matrix"""
        raise NotImplementedError
    
    def prepare_mat(self):
        """Prepare final scoring matrix"""
        raise NotImplementedError
    
    def score_sequence(self, sequence: np.ndarray) -> float:
        """Score a sequence"""
        raise NotImplementedError


class PWMMatrix(SequenceMatrix):
    """Position Weight Matrix for sequence bias"""
    
    def __init__(self, length: int):
        super().__init__(length)
        self.counts = np.zeros((5, length), dtype=np.float64)  # ATCGN
        self.bg_counts = np.zeros((5, length), dtype=np.float64)
        self.pssm = None
        
    def add_sequence(self, sequence: np.ndarray, is_background: bool = False):
        """
        Add sequence to PWM counts
        
        Parameters
        ----------
        sequence : np.ndarray
            Numeric sequence array
        is_background : bool
            Whether this is background sequence
        """
        if len(sequence) != self.length:
            return
        
        if is_background:
            for i, base in enumerate(sequence):
                if base < 5:
                    self.bg_counts[base, i] += 1
            self.no_bg += 1
        else:
            for i, base in enumerate(sequence):
                if base < 5:
                    self.counts[base, i] += 1
            self.no_bias += 1
    
    def prepare_mat(self):
        """Prepare PSSM from counts"""
        if self.no_bias == 0 or self.no_bg == 0:
            self.pssm = np.zeros((5, self.length))
            return
        
        # Calculate PWMs with pseudocounts
        bias_pwm = (self.counts + 0.25) / (self.no_bias + 1.0)
        bg_pwm = (self.bg_counts + 0.25) / (self.no_bg + 1.0)
        
        # Calculate PSSM (log odds ratio)
        self.pssm = np.log2(bias_pwm / bg_pwm)
        
        # Set N scores to 0
        self.pssm[4, :] = 0.0
    
    def score_sequence(self, sequence: np.ndarray) -> float:
        """
        Score sequence using PSSM
        
        Parameters
        ----------
        sequence : np.ndarray
            Sequence to score
            
        Returns
        -------
        float
            PSSM score
        """
        if self.pssm is None:
            return 0.0
        
        if len(sequence) != self.length:
            return 0.0
        
        score = 0.0
        for i, base in enumerate(sequence):
            if base < 5:
                score += self.pssm[base, i]
        
        return score


class DWMMatrix(SequenceMatrix):
    """Dinucleotide Weight Matrix for sequence bias"""
    
    def __init__(self, length: int):
        super().__init__(length)
        # 5x5 matrix for each position pair
        self.counts = np.zeros((5, 5, length-1), dtype=np.float64)
        self.bg_counts = np.zeros((5, 5, length-1), dtype=np.float64)
        # Single nucleotide counts for first position
        self.first_counts = np.zeros(5, dtype=np.float64)
        self.first_bg_counts = np.zeros(5, dtype=np.float64)
        self.dwm = None
        
    def add_sequence(self, sequence: np.ndarray, is_background: bool = False):
        """Add sequence to DWM counts"""
        if len(sequence) != self.length:
            return
        
        if is_background:
            # First nucleotide
            if sequence[0] < 5:
                self.first_bg_counts[sequence[0]] += 1
            
            # Dinucleotides
            for i in range(len(sequence) - 1):
                if sequence[i] < 5 and sequence[i+1] < 5:
                    self.bg_counts[sequence[i], sequence[i+1], i] += 1
            self.no_bg += 1
        else:
            # First nucleotide
            if sequence[0] < 5:
                self.first_counts[sequence[0]] += 1
            
            # Dinucleotides
            for i in range(len(sequence) - 1):
                if sequence[i] < 5 and sequence[i+1] < 5:
                    self.counts[sequence[i], sequence[i+1], i] += 1
            self.no_bias += 1
    
    def prepare_mat(self):
        """Prepare DWM from counts"""
        if self.no_bias == 0 or self.no_bg == 0:
            return
        
        # Calculate conditional probabilities
        self.first_pwm = (self.first_counts + 0.25) / (self.no_bias + 1.0)
        self.first_bg_pwm = (self.first_bg_counts + 0.25) / (self.no_bg + 1.0)
        
        # DWM scores
        self.dwm = np.zeros((5, 5, self.length-1))
        
        for i in range(self.length - 1):
            for curr_base in range(5):
                for next_base in range(5):
                    # Calculate conditional probabilities
                    bias_joint = (self.counts[curr_base, next_base, i] + 0.01) / (self.no_bias + 0.25)
                    bg_joint = (self.bg_counts[curr_base, next_base, i] + 0.01) / (self.no_bg + 0.25)
                    
                    bias_curr = (self.first_counts[curr_base] + 0.01) / (self.no_bias + 0.05)
                    bg_curr = (self.first_bg_counts[curr_base] + 0.01) / (self.no_bg + 0.05)
                    
                    # P(next|curr) = P(next,curr) / P(curr)
                    bias_cond = bias_joint / bias_curr if bias_curr > 0 else 0
                    bg_cond = bg_joint / bg_curr if bg_curr > 0 else 0
                    
                    # Log odds ratio
                    if bg_cond > 0:
                        self.dwm[curr_base, next_base, i] = np.log2(bias_cond / bg_cond)
                    
        # Set N scores to 0
        self.dwm[4, :, :] = 0.0
        self.dwm[:, 4, :] = 0.0
    
    def score_sequence(self, sequence: np.ndarray) -> float:
        """Score sequence using DWM"""
        if self.dwm is None or len(sequence) != self.length:
            return 0.0
        
        score = 0.0
        for i in range(len(sequence) - 1):
            if sequence[i] < 5 and sequence[i+1] < 5:
                score += self.dwm[sequence[i], sequence[i+1], i]
        
        return score


class AtacBias:
    """Main class for ATAC bias estimation and correction"""
    
    def __init__(self, k_flank: int = 12, score_mat: str = "DWM"):
        """
        Initialize ATAC bias object
        
        Parameters
        ----------
        k_flank : int
            Flanking length around cutsite
        score_mat : str
            Scoring matrix type ('PWM' or 'DWM')
        """
        self.k_flank = k_flank
        self.total_length = 2 * k_flank + 1
        self.score_mat = score_mat
        
        # Create matrices for each strand
        if score_mat == "PWM":
            self.forward = PWMMatrix(self.total_length)
            self.reverse = PWMMatrix(self.total_length)
        elif score_mat == "DWM":
            self.forward = DWMMatrix(self.total_length)
            self.reverse = DWMMatrix(self.total_length)
        else:
            raise ValueError(f"Unknown score_mat type: {score_mat}")
        
        self.no_reads = 0
        self.correction_factor = 1.0
        self.prepared = False
    
    def add_read(self, cutsite_pos: int, sequence_obj: GenomicSequence, 
                 strand: str, is_background: bool = False):
        """
        Add read cutsite to bias estimation
        
        Parameters
        ----------
        cutsite_pos : int
            Cutsite position relative to sequence start
        sequence_obj : GenomicSequence
            Genomic sequence object
        strand : str
            Read strand ('+' or '-')
        is_background : bool
            Whether this is background sequence
        """
        # Get k-mer sequence around cutsite
        kmer = sequence_obj.get_kmer(cutsite_pos, self.k_flank, strand)
        
        # Add to appropriate matrix
        if strand == "+":
            self.forward.add_sequence(kmer, is_background)
        else:
            self.reverse.add_sequence(kmer, is_background)
        
        if not is_background:
            self.no_reads += 1
    
    def prepare_model(self):
        """Prepare bias model from accumulated counts"""
        self.forward.prepare_mat()
        self.reverse.prepare_mat()
        self.prepared = True
    
    def predict_bias(self, cutsite_pos: int, sequence_obj: GenomicSequence, 
                     strand: str) -> float:
        """
        Predict bias score for a cutsite
        
        Parameters
        ----------
        cutsite_pos : int
            Cutsite position
        sequence_obj : GenomicSequence
            Sequence object
        strand : str
            Strand
            
        Returns
        -------
        float
            Bias score
        """
        if not self.prepared:
            return 0.0
        
        kmer = sequence_obj.get_kmer(cutsite_pos, self.k_flank, strand)
        
        if strand == "+":
            return self.forward.score_sequence(kmer)
        else:
            return self.reverse.score_sequence(kmer)
    
    def save(self, filename: str):
        """Save bias model to pickle file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filename: str):
        """Load bias model from pickle file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)


def relu(x: np.ndarray, params: Tuple[float, float]) -> np.ndarray:
    """
    Apply ReLU function with linear scaling
    
    Parameters
    ----------
    x : np.ndarray
        Input scores
    params : Tuple[float, float]
        (slope, intercept) parameters
        
    Returns
    -------
    np.ndarray
        ReLU output
    """
    slope, intercept = params
    return np.maximum(0, slope * x + intercept)


def estimate_bias_parameters(bias_scores: np.ndarray, signal: np.ndarray) -> Tuple[float, float]:
    """
    Estimate ReLU parameters from bias scores and signal
    
    Parameters
    ----------
    bias_scores : np.ndarray
        Predicted bias scores
    signal : np.ndarray
        Observed signal
        
    Returns
    -------
    Tuple[float, float]
        (slope, intercept) parameters
    """
    # Simple linear regression
    if len(bias_scores) < 10 or np.std(bias_scores) < 1e-10:
        return 0.0, 0.0
    
    try:
        slope = np.cov(bias_scores, signal)[0, 1] / np.var(bias_scores)
        intercept = np.mean(signal) - slope * np.mean(bias_scores)
        return slope, intercept
    except:
        return 0.0, 0.0


def atacorrect_core(
    bam_file: str,
    fasta_file: str,
    peak_regions: List[Tuple[str, int, int]],
    bias_regions: Optional[List[Tuple[str, int, int]]] = None,
    output_regions: Optional[List[Tuple[str, int, int]]] = None,
    k_flank: int = 12,
    score_mat: str = "DWM",
    read_shift: Tuple[int, int] = (4, -5),
    bg_shift: int = 100,
    window_size: int = 100,
    extend: int = 100,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Core ATACorrect function for Tn5 bias correction
    
    Parameters
    ----------
    bam_file : str
        Path to ATAC-seq BAM file
    fasta_file : str
        Path to genome FASTA file
    peak_regions : List[Tuple[str, int, int]]
        List of peak regions (chrom, start, end)
    bias_regions : Optional[List[Tuple[str, int, int]]]
        Regions for bias estimation (default: non-peak regions)
    output_regions : Optional[List[Tuple[str, int, int]]]
        Regions for output (default: peak regions)
    k_flank : int
        Flanking length around cutsite
    score_mat : str
        Scoring matrix type ('PWM' or 'DWM')
    read_shift : Tuple[int, int]
        Read shift for forward and reverse strands
    bg_shift : int
        Background shift for bias estimation
    window_size : int
        Window size for correction
    extend : int
        Extension around output regions
    verbose : bool
        Verbose output
        
    Returns
    -------
    Dict[str, Any]
        Results dictionary containing:
        - bias_model: AtacBias object
        - corrected_signals: Dict of corrected signals by region
        - uncorrected_signals: Dict of uncorrected signals by region
        - bias_predictions: Dict of bias predictions by region
        - stats: Correction statistics
    """
    
    if verbose:
        print("Starting ATACorrect bias correction...")
        print(f"- BAM file: {bam_file}")
        print(f"- FASTA file: {fasta_file}")
        print(f"- Peak regions: {len(peak_regions)}")
    
    # Set default regions
    if output_regions is None:
        output_regions = [(chrom, max(0, start - extend), end + extend) 
                         for chrom, start, end in peak_regions]
    
    # Initialize bias model
    bias_model = AtacBias(k_flank=k_flank, score_mat=score_mat)
    
    # Step 1: Bias estimation
    if verbose:
        print("Step 1: Estimating Tn5 bias...")
    
    forward_shift, reverse_shift = read_shift
    
    with pysam.AlignmentFile(bam_file, 'rb') as bam:
        
        # Use bias regions or create from non-peak regions  
        if bias_regions is None:
            # Create bias regions from genome gaps (simplified)
            bias_regions = []
            for chrom, start, end in peak_regions[:10]:  # Limit for demo
                # Create regions shifted by bg_shift
                bias_start = max(0, start - bg_shift - 10000)
                bias_end = start - bg_shift
                if bias_end > bias_start + 1000:
                    bias_regions.append((chrom, bias_start, bias_end))
        
        # Process bias regions
        from tqdm import tqdm
        for region_idx, (chrom, start, end) in tqdm(enumerate(bias_regions), 
        total=len(bias_regions), desc="Processing bias regions", disable=not verbose):
            
            # Load sequence
            sequence_obj = GenomicSequence((chrom, start, end))
            try:
                sequence_obj.from_fasta(fasta_file)
            except:
                continue
            
            # Process reads in region
            try:
                for read in bam.fetch(chrom, start, end):
                    if read.is_unmapped or read.is_duplicate:
                        continue
                    
                    # Calculate cutsite position
                    if read.is_reverse:
                        cutsite = read.reference_end + reverse_shift
                        strand = "-"
                    else:
                        cutsite = read.reference_start + forward_shift
                        strand = "+"
                    
                    # Convert to relative position
                    rel_cutsite = cutsite - start
                    if 0 <= rel_cutsite < sequence_obj.length:
                        # Add as background (for bias estimation)
                        bias_model.add_read(rel_cutsite, sequence_obj, strand, is_background=True)
                        
                        # Add background shifted version
                        bg_rel_cutsite = rel_cutsite + bg_shift
                        if 0 <= bg_rel_cutsite < sequence_obj.length:
                            bias_model.add_read(bg_rel_cutsite, sequence_obj, strand, is_background=False)
            except:
                continue
    
    # Prepare bias model
    bias_model.prepare_model()
    
    if verbose:
        print(f"  Bias model trained on {bias_model.no_reads} reads")
    
    # Step 2: Apply bias correction to output regions
    if verbose:
        print("Step 2: Applying bias correction...")
    
    results = {
        'bias_model': bias_model,
        'corrected_signals': {},
        'uncorrected_signals': {},
        'bias_predictions': {},
        'stats': {}
    }
    
    total_uncorrected = 0
    total_corrected = 0
    
    with pysam.AlignmentFile(bam_file, 'rb') as bam:
        
        from tqdm import tqdm
        for chrom, start, end in tqdm(output_regions, desc="Processing output regions", disable=not verbose):
            
            region_length = end - start
            region_key = f"{chrom}:{start}-{end}"
            
            # Initialize signal arrays
            uncorrected_signal = np.zeros(region_length, dtype=np.float64)
            bias_scores = np.zeros(region_length, dtype=np.float64)
            
            # Load sequence
            sequence_obj = GenomicSequence((chrom, start, end))
            try:
                sequence_obj.from_fasta(fasta_file)
            except:
                continue
            
            # Collect reads and calculate signals
            try:
                for read in bam.fetch(chrom, start, end):
                    if read.is_unmapped or read.is_duplicate:
                        continue
                    
                    # Calculate cutsite
                    if read.is_reverse:
                        cutsite = read.reference_end + reverse_shift
                        strand = "-"
                    else:
                        cutsite = read.reference_start + forward_shift
                        strand = "+"
                    
                    rel_cutsite = cutsite - start
                    if 0 <= rel_cutsite < region_length:
                        # Add to uncorrected signal
                        uncorrected_signal[rel_cutsite] += 1
                        
                        # Calculate bias prediction
                        bias_score = bias_model.predict_bias(rel_cutsite, sequence_obj, strand)
                        bias_scores[rel_cutsite] = bias_score
            except:
                continue
            
            # Apply bias correction
            if np.sum(uncorrected_signal) > 0:
                # Estimate ReLU parameters
                relu_params = estimate_bias_parameters(bias_scores, uncorrected_signal)
                
                # Apply ReLU function
                bias_prediction = relu(bias_scores, relu_params)
                
                # Rolling window correction
                corrected_signal = np.zeros_like(uncorrected_signal)
                
                for pos in range(region_length):
                    window_start = max(0, pos - window_size // 2)
                    window_end = min(region_length, pos + window_size // 2 + 1)
                    
                    window_signal = np.sum(uncorrected_signal[window_start:window_end])
                    window_bias = np.sum(bias_prediction[window_start:window_end])
                    
                    if window_bias > 0 and bias_prediction[pos] > 0:
                        expected_signal = window_signal * (bias_prediction[pos] / window_bias)
                        corrected_signal[pos] = max(0, uncorrected_signal[pos] - expected_signal)
                    else:
                        corrected_signal[pos] = uncorrected_signal[pos]
                
                # Rescale to maintain total signal
                total_original = np.sum(uncorrected_signal)
                total_corrected_pos = np.sum(corrected_signal)
                
                if total_corrected_pos > 0:
                    scaling_factor = total_original / total_corrected_pos
                    corrected_signal *= scaling_factor
            else:
                corrected_signal = uncorrected_signal.copy()
                bias_prediction = bias_scores.copy()
            
            # Store results
            results['uncorrected_signals'][region_key] = uncorrected_signal
            results['corrected_signals'][region_key] = corrected_signal
            results['bias_predictions'][region_key] = bias_prediction
            
            total_uncorrected += np.sum(uncorrected_signal)
            total_corrected += np.sum(corrected_signal)
    
    # Calculate summary statistics
    results['stats'] = {
        'total_regions_processed': len(results['corrected_signals']),
        'total_uncorrected_signal': total_uncorrected,
        'total_corrected_signal': total_corrected,
        'correction_ratio': total_corrected / total_uncorrected if total_uncorrected > 0 else 1.0,
        'bias_model_reads': bias_model.no_reads
    }
    
    if verbose:
        print("ATACorrect completed!")
        print(f"- Regions processed: {results['stats']['total_regions_processed']}")
        print(f"- Correction ratio: {results['stats']['correction_ratio']:.3f}")
    
    return results


def save_atacorrect_results_to_bigwig(
    results: Dict[str, Any],
    output_prefix: str,
    chrom_sizes: Dict[str, int]
):
    """
    Save ATACorrect results to BigWig files
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results from atacorrect_core
    output_prefix : str
        Output file prefix
    chrom_sizes : Dict[str, int]
        Chromosome sizes
    """
    
    signal_types = ['uncorrected_signals', 'corrected_signals', 'bias_predictions']
    suffixes = ['uncorrected', 'corrected', 'bias']
    
    for signal_type, suffix in zip(signal_types, suffixes):
        if signal_type not in results:
            continue
        
        output_file = f"{output_prefix}_{suffix}.bw"
        
        with pyBigWig.open(output_file, "w") as bw:
            # Add header
            chrom_list = [(chrom, size) for chrom, size in chrom_sizes.items()]
            bw.addHeader(chrom_list)
            
            # Add entries
            for region_key, signal in results[signal_type].items():
                chrom, coords = region_key.split(':')
                start, end = map(int, coords.split('-'))
                
                if len(signal) == end - start:
                    positions = list(range(start, end))
                    values = signal.tolist()
                    bw.addEntries(chrom, positions, values=values, span=1)
        
        print(f"Saved {output_file}")


# Convenience function with default parameters
def atacorrect(
    bam_file: str,
    fasta_file: str, 
    peak_bed_file: str,
    output_prefix: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenient wrapper for ATACorrect with BED file input
    
    Parameters
    ----------
    bam_file : str
        BAM file path
    fasta_file : str
        Genome FASTA file path
    peak_bed_file : str
        Peak regions BED file
    output_prefix : str
        Output file prefix
    **kwargs
        Additional parameters for atacorrect_core
        
    Returns
    -------
    Dict[str, Any]
        ATACorrect results
    """
    
    # Read peak regions from BED file
    peak_regions = []
    with open(peak_bed_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    chrom, start, end = parts[0], int(parts[1]), int(parts[2])
                    peak_regions.append((chrom, start, end))
    
    # Run ATACorrect
    results = atacorrect_core(
        bam_file=bam_file,
        fasta_file=fasta_file,
        peak_regions=peak_regions,
        **kwargs
    )
    
    # Save bias model
    bias_model_file = f"{output_prefix}_bias_model.pkl"
    results['bias_model'].save(bias_model_file)
    
    # Get chromosome sizes from FASTA
    chrom_sizes = {}
    with pysam.FastaFile(fasta_file) as fasta:
        for chrom in fasta.references:
            chrom_sizes[chrom] = fasta.get_reference_length(chrom)
    
    # Save BigWig files
    save_atacorrect_results_to_bigwig(results, output_prefix, chrom_sizes)
    
    return results