#!/usr/bin/env python

"""
Network construction and fragment filtering tools for epione
Includes CreateNetwork and FilterFragments functionality

@author: Zehua Zeng
@contact: starlitnightly@gmail.com
@license: GPL 3.0

This module provides tools for TF network construction and BAM fragment filtering
"""

import os
import sys
import numpy as np
import pandas as pd
import pysam
from typing import Union, List, Dict, Optional, Tuple, Any
import logging
import warnings
from collections import defaultdict, Counter
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score
import tempfile


class FragmentFilter:
    """Class for filtering BAM fragments based on regions"""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize fragment filter
        
        Parameters
        ----------
        verbose : bool
            Verbose output
        """
        self.verbose = verbose
    
    def filter_bam_by_regions(
        self,
        input_bam: str,
        regions_bed: str,
        output_bam: str,
        mode: str = 'include',
        extend: int = 0,
        quality_threshold: int = 30,
        remove_duplicates: bool = True,
        remove_unmapped: bool = True
    ) -> Dict[str, int]:
        """
        Filter BAM file based on genomic regions
        
        Parameters
        ----------
        input_bam : str
            Input BAM file path
        regions_bed : str
            BED file with regions to include/exclude
        output_bam : str
            Output BAM file path
        mode : str
            Filter mode ('include' or 'exclude')
        extend : int
            Extension around regions
        quality_threshold : int
            Minimum mapping quality
        remove_duplicates : bool
            Remove duplicate reads
        remove_unmapped : bool
            Remove unmapped reads
            
        Returns
        -------
        Dict[str, int]
            Filtering statistics
        """
        
        # Read regions from BED file
        regions = {}
        with open(regions_bed, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        chrom, start, end = parts[0], int(parts[1]), int(parts[2])
                        
                        # Apply extension
                        start = max(0, start - extend)
                        end = end + extend
                        
                        if chrom not in regions:
                            regions[chrom] = []
                        regions[chrom].append((start, end))
        
        # Sort regions for efficient overlap checking
        for chrom in regions:
            regions[chrom].sort()
        
        # Filter BAM file
        stats = {
            'total_reads': 0,
            'filtered_reads': 0,
            'quality_filtered': 0,
            'duplicate_filtered': 0,
            'unmapped_filtered': 0,
            'region_filtered': 0
        }
        
        with pysam.AlignmentFile(input_bam, 'rb') as infile:
            with pysam.AlignmentFile(output_bam, 'wb', template=infile) as outfile:
                
                for read in infile:
                    stats['total_reads'] += 1
                    
                    # Filter unmapped reads
                    if remove_unmapped and read.is_unmapped:
                        stats['unmapped_filtered'] += 1
                        continue
                    
                    # Filter duplicates
                    if remove_duplicates and read.is_duplicate:
                        stats['duplicate_filtered'] += 1
                        continue
                    
                    # Filter by quality
                    if read.mapping_quality < quality_threshold:
                        stats['quality_filtered'] += 1
                        continue
                    
                    # Check region overlap
                    read_chrom = read.reference_name
                    read_start = read.reference_start
                    read_end = read.reference_end
                    
                    if read_chrom is None or read_start is None or read_end is None:
                        continue
                    
                    # Check if read overlaps with any region
                    overlaps = False
                    if read_chrom in regions:
                        for region_start, region_end in regions[read_chrom]:
                            if not (read_end <= region_start or read_start >= region_end):
                                overlaps = True
                                break
                    
                    # Apply filtering based on mode
                    if mode == 'include':
                        if overlaps:
                            outfile.write(read)
                            stats['filtered_reads'] += 1
                        else:
                            stats['region_filtered'] += 1
                    elif mode == 'exclude':
                        if not overlaps:
                            outfile.write(read)
                            stats['filtered_reads'] += 1
                        else:
                            stats['region_filtered'] += 1
        
        # Index output BAM
        try:
            pysam.index(output_bam)
        except:
            warnings.warn(f"Could not index {output_bam}")
        
        if self.verbose:
            print(f"BAM filtering completed:")
            print(f"- Total reads: {stats['total_reads']:,}")
            print(f"- Filtered reads: {stats['filtered_reads']:,}")
            print(f"- Quality filtered: {stats['quality_filtered']:,}")
            print(f"- Duplicate filtered: {stats['duplicate_filtered']:,}")
            print(f"- Unmapped filtered: {stats['unmapped_filtered']:,}")
            print(f"- Region filtered: {stats['region_filtered']:,}")
        
        return stats
    
    def extract_fragments_from_regions(
        self,
        bam_file: str,
        regions: List[Tuple[str, int, int]],
        output_bed: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract fragment information from specific regions
        
        Parameters
        ----------
        bam_file : str
            Input BAM file
        regions : List[Tuple[str, int, int]]
            Regions to extract from
        output_bed : Optional[str]
            Output BED file for fragments
            
        Returns
        -------
        pd.DataFrame
            Fragment information
        """
        
        fragments = []
        
        with pysam.AlignmentFile(bam_file, 'rb') as bamfile:
            for chrom, start, end in regions:
                try:
                    for read in bamfile.fetch(chrom, start, end):
                        if read.is_unmapped or read.is_duplicate:
                            continue
                        
                        # Get fragment coordinates
                        if read.is_proper_pair and read.template_length > 0:
                            # Use template length for paired reads
                            frag_start = read.reference_start
                            frag_end = read.reference_start + abs(read.template_length)
                        else:
                            # Use read coordinates for single reads
                            frag_start = read.reference_start
                            frag_end = read.reference_end
                        
                        fragments.append({
                            'chrom': chrom,
                            'start': frag_start,
                            'end': frag_end,
                            'length': frag_end - frag_start,
                            'quality': read.mapping_quality,
                            'is_paired': read.is_proper_pair,
                            'template_length': read.template_length
                        })
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not process region {chrom}:{start}-{end}: {e}")
                    continue
        
        fragments_df = pd.DataFrame(fragments)
        
        # Save to BED file if requested
        if output_bed and not fragments_df.empty:
            bed_data = fragments_df[['chrom', 'start', 'end']].copy()
            bed_data.to_csv(output_bed, sep='\t', header=False, index=False)
        
        return fragments_df


class NetworkBuilder:
    """Class for constructing transcription factor networks"""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize network builder
        
        Parameters
        ----------
        verbose : bool
            Verbose output
        """
        self.verbose = verbose
    
    def build_tfbs_network(
        self,
        tfbs_data: pd.DataFrame,
        correlation_threshold: float = 0.5,
        min_shared_regions: int = 10,
        correlation_method: str = 'pearson'
    ) -> nx.Graph:
        """
        Build TF-TF interaction network from TFBS data
        
        Parameters
        ----------
        tfbs_data : pd.DataFrame
            TFBS data with columns: ['tf', 'chrom', 'start', 'end', 'score']
        correlation_threshold : float
            Minimum correlation for edge creation
        min_shared_regions : int
            Minimum number of shared regions
        correlation_method : str
            Correlation method ('pearson' or 'spearman')
            
        Returns
        -------
        nx.Graph
            TF interaction network
        """
        
        # Create TF-region matrix
        tf_names = tfbs_data['tf'].unique()
        
        # Create region identifiers
        tfbs_data['region_id'] = (tfbs_data['chrom'].astype(str) + ':' + 
                                 tfbs_data['start'].astype(str) + '-' + 
                                 tfbs_data['end'].astype(str))
        
        # Pivot to create TF-region matrix
        tf_region_matrix = tfbs_data.pivot_table(
            index='region_id',
            columns='tf',
            values='score',
            fill_value=0,
            aggfunc='max'  # Take maximum score if multiple TFBS in same region
        )
        
        # Calculate pairwise correlations
        correlation_matrix = tf_region_matrix.corr(method=correlation_method)
        
        # Create network
        G = nx.Graph()
        
        # Add nodes (TFs)
        for tf in tf_names:
            G.add_node(tf, type='TF')
        
        # Add edges based on correlations
        for i, tf1 in enumerate(tf_names):
            for j, tf2 in enumerate(tf_names):
                if i < j:  # Avoid duplicate edges
                    correlation = correlation_matrix.loc[tf1, tf2]
                    
                    # Count shared regions
                    tf1_regions = set(tfbs_data[tfbs_data['tf'] == tf1]['region_id'])
                    tf2_regions = set(tfbs_data[tfbs_data['tf'] == tf2]['region_id'])
                    shared_regions = len(tf1_regions & tf2_regions)
                    
                    # Add edge if correlation and shared regions meet thresholds
                    if (abs(correlation) >= correlation_threshold and 
                        shared_regions >= min_shared_regions and
                        not np.isnan(correlation)):
                        
                        G.add_edge(tf1, tf2, 
                                  weight=abs(correlation),
                                  correlation=correlation,
                                  shared_regions=shared_regions)
        
        if self.verbose:
            print(f"Network constructed:")
            print(f"- Nodes (TFs): {G.number_of_nodes()}")
            print(f"- Edges: {G.number_of_edges()}")
            print(f"- Density: {nx.density(G):.3f}")
        
        return G
    
    def calculate_network_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Calculate network topology metrics
        
        Parameters
        ----------
        G : nx.Graph
            Input network
            
        Returns
        -------
        Dict[str, Any]
            Network metrics
        """
        
        metrics = {}
        
        # Basic metrics
        metrics['n_nodes'] = G.number_of_nodes()
        metrics['n_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
        
        # Connectivity metrics
        if G.number_of_nodes() > 0:
            metrics['is_connected'] = nx.is_connected(G)
            metrics['n_components'] = nx.number_connected_components(G)
            
            if nx.is_connected(G):
                metrics['diameter'] = nx.diameter(G)
                metrics['average_path_length'] = nx.average_shortest_path_length(G)
            
            # Node metrics
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
            
            metrics['degree_centrality'] = degree_centrality
            metrics['betweenness_centrality'] = betweenness_centrality  
            metrics['closeness_centrality'] = closeness_centrality
            
            # Average metrics
            metrics['average_degree'] = np.mean(list(dict(G.degree()).values()))
            metrics['average_clustering'] = nx.average_clustering(G)
        
        return metrics
    
    def detect_communities(
        self,
        G: nx.Graph,
        method: str = 'louvain'
    ) -> Dict[str, int]:
        """
        Detect communities in the network
        
        Parameters
        ----------
        G : nx.Graph
            Input network
        method : str
            Community detection method ('louvain', 'greedy_modularity')
            
        Returns
        -------
        Dict[str, int]
            Node to community mapping
        """
        
        try:
            if method == 'louvain':
                import community as community_louvain
                partition = community_louvain.best_partition(G)
            elif method == 'greedy_modularity':
                communities = nx.community.greedy_modularity_communities(G)
                partition = {}
                for i, community in enumerate(communities):
                    for node in community:
                        partition[node] = i
            else:
                raise ValueError(f"Unknown community detection method: {method}")
        except ImportError:
            warnings.warn("python-louvain not available, using greedy modularity")
            communities = nx.community.greedy_modularity_communities(G)
            partition = {}
            for i, community in enumerate(communities):
                for node in community:
                    partition[node] = i
        
        if self.verbose:
            n_communities = len(set(partition.values()))
            print(f"Detected {n_communities} communities using {method}")
        
        return partition
    
    def plot_network(
        self,
        G: nx.Graph,
        communities: Optional[Dict[str, int]] = None,
        output_file: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 10),
        layout: str = 'spring',
        node_size_attr: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot network visualization
        
        Parameters
        ----------
        G : nx.Graph
            Network to plot
        communities : Optional[Dict[str, int]]
            Community assignments for coloring
        output_file : Optional[str]
            Output file path
        figsize : Tuple[float, float]
            Figure size
        layout : str
            Network layout ('spring', 'circular', 'kamada_kawai')
        node_size_attr : Optional[str]
            Node attribute for sizing
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Node colors based on communities
        if communities:
            community_colors = plt.cm.Set3(np.linspace(0, 1, len(set(communities.values()))))
            node_colors = [community_colors[communities[node]] for node in G.nodes()]
        else:
            node_colors = 'lightblue'
        
        # Node sizes
        if node_size_attr and all(node_size_attr in G.nodes[node] for node in G.nodes()):
            node_sizes = [G.nodes[node][node_size_attr] * 300 for node in G.nodes()]
        else:
            # Use degree centrality for node sizes
            degree_cent = nx.degree_centrality(G)
            node_sizes = [degree_cent[node] * 1000 + 100 for node in G.nodes()]
        
        # Edge weights
        edge_weights = [G[u][v].get('weight', 1) * 2 for u, v in G.edges()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                              alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        ax.set_title(f"TF Interaction Network\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        ax.axis('off')
        
        plt.tight_layout()
        
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved network plot to {output_file}")
        
        return fig
    
    def export_network(
        self,
        G: nx.Graph,
        output_prefix: str,
        communities: Optional[Dict[str, int]] = None
    ):
        """
        Export network in multiple formats
        
        Parameters
        ----------
        G : nx.Graph
            Network to export
        output_prefix : str
            Output file prefix
        communities : Optional[Dict[str, int]]
            Community assignments
        """
        
        # Export as GraphML
        graphml_file = f"{output_prefix}.graphml"
        nx.write_graphml(G, graphml_file)
        
        # Export as edge list
        edgelist_file = f"{output_prefix}_edges.csv"
        edges_data = []
        for u, v, data in G.edges(data=True):
            edge_info = {'source': u, 'target': v}
            edge_info.update(data)
            edges_data.append(edge_info)
        
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv(edgelist_file, index=False)
        
        # Export node attributes
        nodes_file = f"{output_prefix}_nodes.csv"
        nodes_data = []
        for node, data in G.nodes(data=True):
            node_info = {'node': node}
            node_info.update(data)
            if communities:
                node_info['community'] = communities.get(node, -1)
            nodes_data.append(node_info)
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_csv(nodes_file, index=False)
        
        print(f"Network exported:")
        print(f"- GraphML: {graphml_file}")
        print(f"- Edges: {edgelist_file}")
        print(f"- Nodes: {nodes_file}")


# Convenience functions
def filter_fragments(
    input_bam: str,
    regions_bed: str,
    output_bam: str,
    **kwargs
) -> Dict[str, int]:
    """
    Convenient wrapper for BAM fragment filtering
    
    Parameters
    ----------
    input_bam : str
        Input BAM file
    regions_bed : str
        Regions BED file
    output_bam : str
        Output BAM file
    **kwargs
        Additional parameters
        
    Returns
    -------
    Dict[str, int]
        Filtering statistics
    """
    
    filter_tool = FragmentFilter()
    return filter_tool.filter_bam_by_regions(
        input_bam=input_bam,
        regions_bed=regions_bed,
        output_bam=output_bam,
        **kwargs
    )


def create_network(
    tfbs_file: str,
    output_prefix: str,
    tf_column: str = 'tf',
    region_columns: List[str] = ['chrom', 'start', 'end'],
    score_column: str = 'score',
    **kwargs
) -> nx.Graph:
    """
    Convenient wrapper for TF network creation
    
    Parameters
    ----------
    tfbs_file : str
        TFBS data file (CSV format)
    output_prefix : str
        Output file prefix
    tf_column : str
        Column name for TF names
    region_columns : List[str]
        Column names for genomic regions
    score_column : str
        Column name for TFBS scores
    **kwargs
        Additional parameters for network building
        
    Returns
    -------
    nx.Graph
        Constructed network
    """
    
    # Read TFBS data
    tfbs_data = pd.read_csv(tfbs_file)
    
    # Standardize column names
    required_columns = ['tf', 'chrom', 'start', 'end', 'score']
    column_mapping = {
        tf_column: 'tf',
        score_column: 'score'
    }
    
    for i, col in enumerate(region_columns[:3]):
        column_mapping[col] = required_columns[i+1]
    
    tfbs_data = tfbs_data.rename(columns=column_mapping)
    
    # Build network
    network_builder = NetworkBuilder()
    G = network_builder.build_tfbs_network(tfbs_data, **kwargs)
    
    # Detect communities
    communities = network_builder.detect_communities(G)
    
    # Calculate metrics
    metrics = network_builder.calculate_network_metrics(G)
    
    # Create visualization
    plot_file = f"{output_prefix}_network.png"
    network_builder.plot_network(G, communities, plot_file)
    
    # Export network
    network_builder.export_network(G, output_prefix, communities)
    
    # Save metrics
    metrics_file = f"{output_prefix}_metrics.json"
    import json
    with open(metrics_file, 'w') as f:
        # Convert numpy types for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                serializable_metrics[key] = {str(k): float(v) for k, v in value.items()}
            elif isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = float(value)
            else:
                serializable_metrics[key] = value
        
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"Network analysis completed. Results saved with prefix: {output_prefix}")
    
    return G