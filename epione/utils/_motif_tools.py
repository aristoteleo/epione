#!/usr/bin/env python

"""
Motif processing tools for epione
Includes FormatMotifs and ClusterMotifs functionality

@author: Zehua Zeng
@contact: starlitnightly@gmail.com
@license: GPL 3.0

This module provides tools for motif format conversion and clustering
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any, IO
import logging
import warnings
from collections import defaultdict
import json
import tempfile
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import BioPython
try:
    from Bio import motifs
    from Bio.motifs import matrix
    BIO_AVAILABLE = True
except ImportError:
    BIO_AVAILABLE = False
    warnings.warn("BioPython not available. Some motif functionality may be limited.")


class MotifMatrix:
    """Class representing a position weight matrix"""
    
    def __init__(
        self,
        name: str,
        matrix: np.ndarray,
        alphabet: str = "ACGT"
    ):
        """
        Initialize motif matrix
        
        Parameters
        ----------
        name : str
            Motif name/ID
        matrix : np.ndarray
            Position weight matrix (4 x length)
        alphabet : str
            Nucleotide alphabet
        """
        
        self.name = name
        self.matrix = matrix
        self.alphabet = alphabet
        self.length = matrix.shape[1]
        
        # Validate matrix
        if matrix.shape[0] != len(alphabet):
            raise ValueError(f"Matrix must have {len(alphabet)} rows for alphabet {alphabet}")
    
    @classmethod
    def from_counts(
        cls,
        name: str,
        counts: np.ndarray,
        pseudocount: float = 1.0,
        alphabet: str = "ACGT"
    ):
        """
        Create motif from count matrix
        
        Parameters
        ----------
        name : str
            Motif name
        counts : np.ndarray
            Count matrix
        pseudocount : float
            Pseudocount for normalization
        alphabet : str
            Nucleotide alphabet
            
        Returns
        -------
        MotifMatrix
            Motif object
        """
        
        # Add pseudocounts and normalize
        counts_pseudo = counts + pseudocount
        matrix = counts_pseudo / np.sum(counts_pseudo, axis=0, keepdims=True)
        
        return cls(name, matrix, alphabet)
    
    def to_pwm(self, background: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert to position weight matrix (log odds)
        
        Parameters
        ----------
        background : Optional[np.ndarray]
            Background frequencies (default: uniform)
            
        Returns
        -------
        np.ndarray
            Log odds matrix
        """
        
        if background is None:
            background = np.ones(len(self.alphabet)) / len(self.alphabet)
        
        # Calculate log odds
        pwm = np.log2(self.matrix / background.reshape(-1, 1))
        
        return pwm
    
    def consensus_sequence(self, threshold: float = 0.5) -> str:
        """
        Get consensus sequence
        
        Parameters
        ----------
        threshold : float
            Threshold for consensus calling
            
        Returns
        -------
        str
            Consensus sequence
        """
        
        consensus = ""
        for pos in range(self.length):
            max_idx = np.argmax(self.matrix[:, pos])
            if self.matrix[max_idx, pos] >= threshold:
                consensus += self.alphabet[max_idx]
            else:
                consensus += "N"
        
        return consensus
    
    def information_content(self) -> np.ndarray:
        """
        Calculate information content at each position
        
        Returns
        -------
        np.ndarray
            Information content per position
        """
        
        max_entropy = np.log2(len(self.alphabet))
        ic = np.zeros(self.length)
        
        for pos in range(self.length):
            # Calculate entropy
            probs = self.matrix[:, pos]
            probs = probs[probs > 0]  # Remove zeros for entropy calculation
            pos_entropy = -np.sum(probs * np.log2(probs))
            ic[pos] = max_entropy - pos_entropy
        
        return ic
    
    def reverse_complement(self) -> 'MotifMatrix':
        """
        Get reverse complement of motif
        
        Returns
        -------
        MotifMatrix
            Reverse complement motif
        """
        
        if self.alphabet != "ACGT":
            raise ValueError("Reverse complement only supported for DNA motifs")
        
        # Reverse complement mapping: A<->T, C<->G
        rc_mapping = np.array([3, 2, 1, 0])  # ACGT -> TGCA
        
        # Reverse the matrix and apply complement mapping
        rc_matrix = self.matrix[rc_mapping, ::-1]
        
        return MotifMatrix(f"{self.name}_rc", rc_matrix, self.alphabet)


class FormatMotifs:
    """Class for motif format conversion and manipulation"""
    
    def __init__(self):
        """Initialize format converter"""
        pass
    
    def read_jaspar(self, file_path: str) -> List[MotifMatrix]:
        """
        Read motifs from JASPAR format file
        
        Parameters
        ----------
        file_path : str
            Path to JASPAR file
            
        Returns
        -------
        List[MotifMatrix]
            List of motif objects
        """
        
        motifs_list = []
        
        if BIO_AVAILABLE:
            try:
                with open(file_path, 'r') as f:
                    motif_records = motifs.parse(f, 'jaspar')
                    
                    for motif in motif_records:
                        # Convert to numpy array
                        matrix = np.array([
                            motif.counts['A'],
                            motif.counts['C'], 
                            motif.counts['G'],
                            motif.counts['T']
                        ], dtype=float)
                        
                        motif_obj = MotifMatrix.from_counts(motif.name, matrix)
                        motifs_list.append(motif_obj)
                        
            except Exception as e:
                warnings.warn(f"Error reading JASPAR file with BioPython: {e}")
                return self._read_jaspar_simple(file_path)
        else:
            return self._read_jaspar_simple(file_path)
        
        return motifs_list
    
    def _read_jaspar_simple(self, file_path: str) -> List[MotifMatrix]:
        """Simple JASPAR reader without BioPython"""
        
        motifs_list = []
        current_motif = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # New motif header
                    motif_id = line[1:].split()[0]
                    current_motif = {'name': motif_id, 'counts': defaultdict(list)}
                elif line and current_motif:
                    # Parse count line
                    if line.startswith('A'):
                        counts = [float(x) for x in line.split('[')[1].split(']')[0].split()]
                        current_motif['counts']['A'] = counts
                    elif line.startswith('C'):
                        counts = [float(x) for x in line.split('[')[1].split(']')[0].split()]
                        current_motif['counts']['C'] = counts
                    elif line.startswith('G'):
                        counts = [float(x) for x in line.split('[')[1].split(']')[0].split()]
                        current_motif['counts']['G'] = counts
                    elif line.startswith('T'):
                        counts = [float(x) for x in line.split('[')[1].split(']')[0].split()]
                        current_motif['counts']['T'] = counts
                        
                        # End of motif, create MotifMatrix
                        if len(current_motif['counts']) == 4:
                            matrix = np.array([
                                current_motif['counts']['A'],
                                current_motif['counts']['C'],
                                current_motif['counts']['G'],
                                current_motif['counts']['T']
                            ], dtype=float)
                            
                            motif_obj = MotifMatrix.from_counts(current_motif['name'], matrix)
                            motifs_list.append(motif_obj)
                            current_motif = None
        
        return motifs_list
    
    def read_meme(self, file_path: str) -> List[MotifMatrix]:
        """
        Read motifs from MEME format file
        
        Parameters
        ----------
        file_path : str
            Path to MEME file
            
        Returns
        -------
        List[MotifMatrix]
            List of motif objects
        """
        
        motifs_list = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('MOTIF'):
                # Parse motif header
                parts = line.split()
                motif_name = parts[1] if len(parts) > 1 else f"MOTIF_{len(motifs_list)+1}"
                
                # Look for letter-probability matrix
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('letter-probability matrix'):
                    i += 1
                
                if i >= len(lines):
                    break
                
                # Parse matrix dimensions
                matrix_line = lines[i].strip()
                # Extract w= parameter
                w_start = matrix_line.find('w=') + 2
                w_end = matrix_line.find(' ', w_start)
                if w_end == -1:
                    w_end = len(matrix_line)
                width = int(matrix_line[w_start:w_end])
                
                # Read matrix values
                matrix = []
                i += 1
                for pos in range(width):
                    if i >= len(lines):
                        break
                    values = [float(x) for x in lines[i].strip().split()]
                    if len(values) >= 4:
                        matrix.append(values[:4])
                    i += 1
                
                if len(matrix) == width:
                    matrix = np.array(matrix).T  # Transpose to get 4 x width
                    motif_obj = MotifMatrix(motif_name, matrix)
                    motifs_list.append(motif_obj)
            
            i += 1
        
        return motifs_list
    
    def write_jaspar(self, motifs: List[MotifMatrix], output_file: str):
        """
        Write motifs to JASPAR format
        
        Parameters
        ----------
        motifs : List[MotifMatrix]
            List of motif objects
        output_file : str
            Output file path
        """
        
        with open(output_file, 'w') as f:
            for motif in motifs:
                # Write header
                f.write(f">{motif.name}\n")
                
                # Convert back to counts (approximate)
                counts = motif.matrix * 100  # Scale up for counts
                
                # Write count lines
                f.write(f"A [ {' '.join([f'{c:.0f}' for c in counts[0, :]])} ]\n")
                f.write(f"C [ {' '.join([f'{c:.0f}' for c in counts[1, :]])} ]\n")
                f.write(f"G [ {' '.join([f'{c:.0f}' for c in counts[2, :]])} ]\n")
                f.write(f"T [ {' '.join([f'{c:.0f}' for c in counts[3, :]])} ]\n")
                f.write("\n")
    
    def write_meme(self, motifs: List[MotifMatrix], output_file: str):
        """
        Write motifs to MEME format
        
        Parameters
        ----------
        motifs : List[MotifMatrix]
            List of motif objects  
        output_file : str
            Output file path
        """
        
        with open(output_file, 'w') as f:
            # Write header
            f.write("MEME version 4\n\n")
            f.write("ALPHABET= ACGT\n\n")
            f.write("strands: + -\n\n")
            f.write("Background letter frequencies\n")
            f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
            
            # Write motifs
            for i, motif in enumerate(motifs):
                f.write(f"MOTIF {motif.name}\n\n")
                f.write(f"letter-probability matrix: alength= 4 w= {motif.length} nsites= 100 E= 0\n")
                
                # Write matrix (transpose back)
                for pos in range(motif.length):
                    values = motif.matrix[:, pos]
                    f.write(f" {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}\n")
                f.write("\n")
    
    def convert_format(
        self,
        input_file: str,
        output_file: str,
        input_format: str,
        output_format: str
    ):
        """
        Convert motif file between formats
        
        Parameters
        ----------
        input_file : str
            Input file path
        output_file : str
            Output file path
        input_format : str
            Input format ('jaspar' or 'meme')
        output_format : str
            Output format ('jaspar' or 'meme')
        """
        
        # Read motifs
        if input_format.lower() == 'jaspar':
            motifs_list = self.read_jaspar(input_file)
        elif input_format.lower() == 'meme':
            motifs_list = self.read_meme(input_file)
        else:
            raise ValueError(f"Unsupported input format: {input_format}")
        
        # Write motifs
        if output_format.lower() == 'jaspar':
            self.write_jaspar(motifs_list, output_file)
        elif output_format.lower() == 'meme':
            self.write_meme(motifs_list, output_file)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        print(f"Converted {len(motifs_list)} motifs from {input_format} to {output_format}")
    
    def filter_motifs(
        self,
        motifs: List[MotifMatrix],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_ic: Optional[float] = None,
        names_include: Optional[List[str]] = None,
        names_exclude: Optional[List[str]] = None
    ) -> List[MotifMatrix]:
        """
        Filter motifs based on criteria
        
        Parameters
        ----------
        motifs : List[MotifMatrix]
            Input motifs
        min_length : Optional[int]
            Minimum motif length
        max_length : Optional[int]
            Maximum motif length
        min_ic : Optional[float]
            Minimum total information content
        names_include : Optional[List[str]]
            Names to include
        names_exclude : Optional[List[str]]
            Names to exclude
            
        Returns
        -------
        List[MotifMatrix]
            Filtered motifs
        """
        
        filtered = []
        
        for motif in motifs:
            # Length filters
            if min_length and motif.length < min_length:
                continue
            if max_length and motif.length > max_length:
                continue
            
            # Information content filter
            if min_ic:
                total_ic = np.sum(motif.information_content())
                if total_ic < min_ic:
                    continue
            
            # Name filters
            if names_include and motif.name not in names_include:
                continue
            if names_exclude and motif.name in names_exclude:
                continue
            
            filtered.append(motif)
        
        return filtered


class ClusterMotifs:
    """Class for motif clustering and consensus generation"""
    
    def __init__(self):
        """Initialize motif clusterer"""
        pass
    
    def calculate_similarity(
        self,
        motif1: MotifMatrix,
        motif2: MotifMatrix,
        method: str = 'pearson'
    ) -> float:
        """
        Calculate similarity between two motifs
        
        Parameters
        ----------
        motif1 : MotifMatrix
            First motif
        motif2 : MotifMatrix
            Second motif
        method : str
            Similarity method ('pearson', 'euclidean', 'kl_divergence')
            
        Returns
        -------
        float
            Similarity score
        """
        
        # Align motifs to same length (simple padding)
        max_len = max(motif1.length, motif2.length)
        
        # Pad shorter motif with background (0.25 for each nucleotide)
        matrix1 = motif1.matrix.copy()
        matrix2 = motif2.matrix.copy()
        
        if motif1.length < max_len:
            padding = np.full((4, max_len - motif1.length), 0.25)
            matrix1 = np.concatenate([matrix1, padding], axis=1)
        
        if motif2.length < max_len:
            padding = np.full((4, max_len - motif2.length), 0.25)
            matrix2 = np.concatenate([matrix2, padding], axis=1)
        
        # Calculate similarity
        if method == 'pearson':
            # Pearson correlation coefficient
            flat1 = matrix1.flatten()
            flat2 = matrix2.flatten()
            corr = np.corrcoef(flat1, flat2)[0, 1]
            return corr if not np.isnan(corr) else 0.0
            
        elif method == 'euclidean':
            # Euclidean distance (converted to similarity)
            distance = np.sqrt(np.sum((matrix1 - matrix2) ** 2))
            return 1.0 / (1.0 + distance)
            
        elif method == 'kl_divergence':
            # Kullback-Leibler divergence (symmetrized)
            kl_div = 0.0
            for pos in range(max_len):
                p = matrix1[:, pos] + 1e-10  # Add small pseudocount
                q = matrix2[:, pos] + 1e-10
                
                # Symmetrized KL divergence
                kl_div += entropy(p, q) + entropy(q, p)
            
            return 1.0 / (1.0 + kl_div)
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def cluster_motifs(
        self,
        motifs: List[MotifMatrix],
        method: str = 'pearson',
        linkage_method: str = 'average',
        n_clusters: Optional[int] = None,
        distance_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Cluster motifs based on similarity
        
        Parameters
        ----------
        motifs : List[MotifMatrix]
            Motifs to cluster
        method : str
            Similarity method
        linkage_method : str
            Hierarchical clustering linkage method
        n_clusters : Optional[int]
            Number of clusters (if specified)
        distance_threshold : Optional[float]
            Distance threshold for clustering
            
        Returns
        -------
        Dict[str, Any]
            Clustering results
        """
        
        n_motifs = len(motifs)
        
        if n_motifs < 2:
            return {
                'motifs': motifs,
                'clusters': [0] * n_motifs,
                'similarity_matrix': np.ones((n_motifs, n_motifs)),
                'linkage_matrix': None
            }
        
        # Calculate pairwise similarities
        similarity_matrix = np.zeros((n_motifs, n_motifs))
        
        for i in range(n_motifs):
            for j in range(i, n_motifs):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self.calculate_similarity(motifs[i], motifs[j], method)
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # Convert similarity to distance
        distance_matrix = 1.0 - similarity_matrix
        
        # Perform hierarchical clustering
        condensed_distances = pdist(distance_matrix, metric='precomputed')
        linkage_matrix = linkage(condensed_distances, method=linkage_method)
        
        # Determine clusters
        if n_clusters is not None:
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        elif distance_threshold is not None:
            cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
        else:
            # Default: use distance threshold of 0.5
            cluster_labels = fcluster(linkage_matrix, 0.5, criterion='distance')
        
        return {
            'motifs': motifs,
            'clusters': cluster_labels,
            'similarity_matrix': similarity_matrix,
            'linkage_matrix': linkage_matrix,
            'n_clusters': len(set(cluster_labels))
        }
    
    def create_consensus_motif(
        self,
        motifs: List[MotifMatrix],
        method: str = 'average'
    ) -> MotifMatrix:
        """
        Create consensus motif from multiple motifs
        
        Parameters
        ----------
        motifs : List[MotifMatrix]
            Input motifs
        method : str
            Consensus method ('average', 'weighted_average')
            
        Returns
        -------
        MotifMatrix
            Consensus motif
        """
        
        if not motifs:
            raise ValueError("No motifs provided")
        
        if len(motifs) == 1:
            return motifs[0]
        
        # Determine consensus length (use maximum)
        max_length = max(motif.length for motif in motifs)
        
        # Pad all motifs to same length
        padded_matrices = []
        for motif in motifs:
            matrix = motif.matrix.copy()
            if motif.length < max_length:
                padding = np.full((4, max_length - motif.length), 0.25)
                matrix = np.concatenate([matrix, padding], axis=1)
            padded_matrices.append(matrix)
        
        # Calculate consensus
        if method == 'average':
            consensus_matrix = np.mean(padded_matrices, axis=0)
        elif method == 'weighted_average':
            # Weight by information content
            weights = []
            for motif in motifs:
                total_ic = np.sum(motif.information_content())
                weights.append(total_ic)
            
            weights = np.array(weights)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(motifs)) / len(motifs)
            
            consensus_matrix = np.average(padded_matrices, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown consensus method: {method}")
        
        # Create consensus motif
        consensus_name = f"CONSENSUS_{len(motifs)}_motifs"
        consensus_motif = MotifMatrix(consensus_name, consensus_matrix)
        
        return consensus_motif
    
    def plot_clustering_dendrogram(
        self,
        clustering_results: Dict[str, Any],
        output_file: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 8)
    ) -> plt.Figure:
        """
        Plot clustering dendrogram
        
        Parameters
        ----------
        clustering_results : Dict[str, Any]
            Results from cluster_motifs
        output_file : Optional[str]
            Output file path
        figsize : Tuple[float, float]
            Figure size
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        
        if clustering_results['linkage_matrix'] is None:
            raise ValueError("No linkage matrix available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get motif names for labels
        labels = [motif.name for motif in clustering_results['motifs']]
        
        # Create dendrogram
        dendrogram(
            clustering_results['linkage_matrix'],
            labels=labels,
            orientation='top',
            leaf_rotation=45,
            ax=ax
        )
        
        ax.set_title(f"Motif Clustering Dendrogram ({clustering_results['n_clusters']} clusters)")
        ax.set_xlabel("Motifs")
        ax.set_ylabel("Distance")
        
        plt.tight_layout()
        
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved dendrogram to {output_file}")
        
        return fig


# Convenience functions
def format_motifs(
    input_file: str,
    output_file: str,
    input_format: str,
    output_format: str
):
    """
    Convert motif file format
    
    Parameters
    ----------
    input_file : str
        Input file
    output_file : str
        Output file
    input_format : str
        Input format
    output_format : str
        Output format
    """
    
    formatter = FormatMotifs()
    formatter.convert_format(input_file, output_file, input_format, output_format)


def cluster_motifs(
    motif_file: str,
    output_dir: str,
    input_format: str = 'jaspar',
    **kwargs
) -> Dict[str, Any]:
    """
    Cluster motifs and generate consensus
    
    Parameters
    ----------
    motif_file : str
        Input motif file
    output_dir : str
        Output directory
    input_format : str
        Input format
    **kwargs
        Additional clustering parameters
        
    Returns
    -------
    Dict[str, Any]
        Clustering results
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read motifs
    formatter = FormatMotifs()
    if input_format == 'jaspar':
        motifs = formatter.read_jaspar(motif_file)
    elif input_format == 'meme':
        motifs = formatter.read_meme(motif_file)
    else:
        raise ValueError(f"Unsupported format: {input_format}")
    
    # Perform clustering
    clusterer = ClusterMotifs()
    results = clusterer.cluster_motifs(motifs, **kwargs)
    
    # Generate consensus motifs for each cluster
    cluster_consensus = {}
    for cluster_id in range(1, results['n_clusters'] + 1):
        cluster_motifs = [motifs[i] for i, c in enumerate(results['clusters']) if c == cluster_id]
        if cluster_motifs:
            consensus = clusterer.create_consensus_motif(cluster_motifs)
            cluster_consensus[cluster_id] = consensus
    
    # Save results
    # Save clustering dendrogram
    dendrogram_file = os.path.join(output_dir, 'clustering_dendrogram.png')
    clusterer.plot_clustering_dendrogram(results, dendrogram_file)
    
    # Save consensus motifs
    consensus_motifs = list(cluster_consensus.values())
    consensus_file = os.path.join(output_dir, 'consensus_motifs.jaspar')
    formatter.write_jaspar(consensus_motifs, consensus_file)
    
    # Save cluster assignments
    cluster_df = pd.DataFrame({
        'motif_name': [motif.name for motif in motifs],
        'cluster': results['clusters']
    })
    cluster_file = os.path.join(output_dir, 'cluster_assignments.csv')
    cluster_df.to_csv(cluster_file, index=False)
    
    print(f"Clustering completed. Results saved to {output_dir}")
    print(f"- {len(motifs)} motifs clustered into {results['n_clusters']} clusters")
    print(f"- Consensus motifs: {len(consensus_motifs)}")
    
    return {
        'clustering_results': results,
        'consensus_motifs': cluster_consensus,
        'output_directory': output_dir
    }