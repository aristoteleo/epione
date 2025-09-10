#!/usr/bin/env python

"""
Plotting functions for TOBIAS-style footprint visualization
Includes PlotAggregate, PlotHeatmap, and PlotTracks functionality

@author: Zehua Zeng
@contact: starlitnightly@gmail.com
@license: GPL 3.0

This module provides comprehensive visualization tools for footprint analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import pyBigWig
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Union, List, Dict, Optional, Tuple, Any
import warnings


class FootprintPlotter:
    """Base class for footprint visualization"""
    
    def __init__(
        self,
        figsize: Tuple[float, float] = (12, 8),
        dpi: int = 300,
        style: str = 'whitegrid'
    ):
        """
        Initialize plotter
        
        Parameters
        ----------
        figsize : Tuple[float, float]
            Figure size (width, height)
        dpi : int
            Resolution for saved figures
        style : str
            Seaborn style
        """
        
        self.figsize = figsize
        self.dpi = dpi
        
        # Set style
        try:
            sns.set_style(style)
            plt.rcParams['figure.dpi'] = dpi
            plt.rcParams['savefig.dpi'] = dpi
        except:
            pass
    
    def _read_signal_from_bigwig(
        self,
        bigwig_file: str,
        region: Tuple[str, int, int]
    ) -> np.ndarray:
        """Read signal from BigWig file for a region"""
        
        chrom, start, end = region
        
        try:
            with pyBigWig.open(bigwig_file) as bw:
                values = bw.values(chrom, start, end)
                if values is None:
                    return np.zeros(end - start)
                return np.array(values, dtype=float)
        except Exception as e:
            warnings.warn(f"Could not read signal from {bigwig_file}: {e}")
            return np.zeros(end - start)
    
    def _read_regions_from_bed(self, bed_file: str) -> List[Tuple]:
        """Read regions from BED file with full BED format support"""
        
        regions = []
        with open(bed_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        # Build region tuple with all available columns
                        region = [parts[0], int(parts[1]), int(parts[2])]  # chr, start, end
                        if len(parts) > 3:
                            region.append(parts[3])  # name
                        if len(parts) > 4:
                            region.append(parts[4])  # score
                        if len(parts) > 5:
                            region.append(parts[5])  # strand
                        
                        regions.append(tuple(region))
        
        return regions


class PlotAggregate(FootprintPlotter):
    """Class for creating aggregate footprint plots"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def plot_aggregate_profile(
        self,
        bigwig_files: List[str],
        regions: Union[str, List[Tuple[str, int, int]]],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        smooth: int = 1,
        normalize: bool = False,
        center_regions: bool = True,
        region_size: Optional[int] = None,
        output_file: Optional[str] = None,
        title: str = "Aggregated signals",
        remove_outliers: float = 1.0,
        log_transform: bool = False,
        flank: int = 60,
        signal_on_x: bool = False,
        share_y: str = "none",
        plot_boundaries: bool = False
    ) -> plt.Figure:
        """
        Create aggregate footprint profile plot - EXACT TOBIAS IMPLEMENTATION
        
        Parameters
        ----------
        bigwig_files : List[str]
            List of BigWig files (signals) to plot
        regions : Union[str, List[Tuple[str, int, int]]]
            BED file path or list of regions (TFBS)
        labels : Optional[List[str]]
            Labels for each BigWig file (signal_labels)
        colors : Optional[List[str]]
            Colors for each profile
        smooth : int
            Smoothing window size  
        normalize : bool
            Whether to normalize between 0-1 using minmax_scale
        center_regions : bool
            Whether to center regions
        region_size : Optional[int]
            Fixed size for regions (if centering)
        output_file : Optional[str]
            Output file path
        title : str
            Title of plot
        remove_outliers : float
            Value between 0-1 indicating percentile of regions to include (1.0 = keep all)
        log_transform : bool
            Log transform the signals before aggregation
        flank : int
            Flanking basepairs (+/-) to show in plot (counted from middle of TFBS)
        signal_on_x : bool
            Show signals on x-axis and TFBSs on y-axis
        share_y : str
            Share y-axis range across plots ("none"/"signals"/"sites"/"both")
        plot_boundaries : bool
            Plot TFBS boundaries
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        
        # Use EXACT TOBIAS logic - import the TOBIAS implementation
        from ._plotting_tobias import plot_aggregate_tobias
        
        # Convert regions to TFBS files format
        if isinstance(regions, str):
            TFBS_files = [regions]  
            TFBS_labels = [os.path.splitext(os.path.basename(regions))[0]]
        else:
            # If regions is a list, we need to create a temporary bed file
            import tempfile
            temp_bed = tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False)
            for region in regions:
                if len(region) >= 3:
                    temp_bed.write(f"{region[0]}\t{region[1]}\t{region[2]}")
                    if len(region) > 3:
                        temp_bed.write(f"\t{region[3]}")
                    if len(region) > 4:
                        temp_bed.write(f"\t{region[4]}")
                    if len(region) > 5:
                        temp_bed.write(f"\t{region[5]}")
                    temp_bed.write("\n")
            temp_bed.close()
            TFBS_files = [temp_bed.name]
            TFBS_labels = ["regions"]
        
        # Set default signal labels
        if labels is None:
            signal_labels = [os.path.splitext(os.path.basename(f))[0] for f in bigwig_files]
        else:
            signal_labels = labels
        
        # Call exact TOBIAS implementation
        try:
            fig = plot_aggregate_tobias(
                TFBS_files=TFBS_files,
                signal_files=bigwig_files,
                output_file=output_file or "temp_aggregate.pdf",
                TFBS_labels=TFBS_labels,
                signal_labels=signal_labels,
                flank=flank,
                remove_outliers=remove_outliers,
                log_transform=log_transform,
                normalize=normalize,
                smooth=smooth,
                title=title,
                signal_on_x=signal_on_x,
                share_y=share_y,
                plot_boundaries=plot_boundaries,
                verbosity=1
            )
        finally:
            # Clean up temporary file if created
            if not isinstance(regions, str) and 'temp_bed' in locals():
                try:
                    os.unlink(temp_bed.name)
                except:
                    pass
        
        return fig
    
    def plot_aggregate_heatmap(
        self,
        bigwig_file: str,
        regions: Union[str, List[Tuple[str, int, int]]],
        output_file: Optional[str] = None,
        center_regions: bool = True,
        region_size: int = 2000,
        sort_by: str = 'max',
        colormap: str = 'RdBu_r',
        title: str = "Aggregate Heatmap"
    ) -> plt.Figure:
        """
        Create aggregate heatmap
        
        Parameters
        ----------
        bigwig_file : str
            BigWig file path
        regions : Union[str, List[Tuple[str, int, int]]]
            Regions to plot
        output_file : Optional[str]
            Output file path
        center_regions : bool
            Whether to center regions
        region_size : int
            Size of centered regions
        sort_by : str
            How to sort regions ('max', 'mean', 'sum')
        colormap : str
            Colormap name
        title : str
            Plot title
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        
        # Parse regions
        if isinstance(regions, str):
            regions_list = self._read_regions_from_bed(regions)
        else:
            regions_list = regions
        
        # Collect signals
        signals_matrix = []
        region_labels = []
        
        for region in regions_list:
            chrom, start, end = region
            
            # Center and resize if requested
            if center_regions:
                center = (start + end) // 2
                new_start = center - region_size // 2
                new_end = center + region_size // 2
                region = (chrom, max(0, new_start), new_end)
                start, end = new_start, new_end
            
            # Read signal
            signal = self._read_signal_from_bigwig(bigwig_file, region)
            
            # Resize to fixed length if needed
            if len(signal) != region_size:
                # Interpolate to fixed size
                from scipy.interpolate import interp1d
                if len(signal) > 1:
                    f = interp1d(np.arange(len(signal)), signal, kind='linear', bounds_error=False, fill_value=0)
                    signal = f(np.linspace(0, len(signal)-1, region_size))
                else:
                    signal = np.zeros(region_size)
            
            signals_matrix.append(signal)
            region_labels.append(f"{chrom}:{start}-{end}")
        
        signals_matrix = np.array(signals_matrix)
        
        # Sort regions
        if sort_by == 'max':
            sort_indices = np.argsort(np.max(signals_matrix, axis=1))[::-1]
        elif sort_by == 'mean':
            sort_indices = np.argsort(np.mean(signals_matrix, axis=1))[::-1]
        elif sort_by == 'sum':
            sort_indices = np.argsort(np.sum(signals_matrix, axis=1))[::-1]
        else:
            sort_indices = np.arange(len(signals_matrix))
        
        sorted_matrix = signals_matrix[sort_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.imshow(
            sorted_matrix,
            cmap=colormap,
            aspect='auto',
            interpolation='bilinear'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Signal Intensity')
        
        # Formatting
        ax.set_xlabel('Position (bp)')
        ax.set_ylabel(f'Regions (n={len(regions_list)})')
        ax.set_title(title)
        
        # Set x-axis ticks
        n_ticks = 5
        tick_positions = np.linspace(0, region_size-1, n_ticks)
        tick_labels = [f"{int(pos - region_size//2)}" for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        
        plt.tight_layout()
        
        # Save if requested
        if output_file:
            fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved heatmap to {output_file}")
        
        return fig


class PlotTracks(FootprintPlotter):
    """Class for creating IGV-style track plots"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def plot_genomic_tracks(
        self,
        region: Tuple[str, int, int],
        bigwig_files: List[str],
        track_labels: Optional[List[str]] = None,
        track_colors: Optional[List[str]] = None,
        bed_files: Optional[List[str]] = None,
        bed_labels: Optional[List[str]] = None,
        y_limits: Optional[List[Tuple[float, float]]] = None,
        output_file: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Create IGV-style genomic track plot
        
        Parameters
        ----------
        region : Tuple[str, int, int]
            Genomic region to plot
        bigwig_files : List[str]
            BigWig files for tracks
        track_labels : Optional[List[str]]
            Labels for BigWig tracks
        track_colors : Optional[List[str]]
            Colors for tracks
        bed_files : Optional[List[str]]
            BED files for annotation tracks
        bed_labels : Optional[List[str]]
            Labels for BED tracks
        y_limits : Optional[List[Tuple[float, float]]]
            Y-axis limits for each track
        output_file : Optional[str]
            Output file path
        title : Optional[str]
            Plot title
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        
        chrom, start, end = region
        region_size = end - start
        
        # Set defaults
        if track_labels is None:
            track_labels = [f"Track {i+1}" for i in range(len(bigwig_files))]
        
        if track_colors is None:
            track_colors = plt.cm.Set1(np.linspace(0, 1, len(bigwig_files)))
        
        # Calculate subplot layout
        n_bigwig_tracks = len(bigwig_files)
        n_bed_tracks = len(bed_files) if bed_files else 0
        n_total_tracks = n_bigwig_tracks + n_bed_tracks
        
        # Create figure with subplots
        fig, axes = plt.subplots(
            n_total_tracks, 1,
            figsize=(self.figsize[0], self.figsize[1] * n_total_tracks / 4),
            sharex=True
        )
        
        if n_total_tracks == 1:
            axes = [axes]
        
        # Plot BigWig tracks
        for i, (bigwig_file, label, color) in enumerate(zip(bigwig_files, track_labels, track_colors)):
            ax = axes[i]
            
            # Read signal
            signal = self._read_signal_from_bigwig(bigwig_file, region)
            
            # Create x coordinates
            x = np.linspace(start, end, len(signal))
            
            # Plot signal
            ax.fill_between(x, 0, signal, color=color, alpha=0.7, linewidth=0)
            ax.plot(x, signal, color=color, linewidth=1)
            
            # Formatting
            ax.set_ylabel(label, rotation=90, ha='right', va='center')
            ax.grid(True, alpha=0.3)
            
            # Set y limits if provided
            if y_limits and i < len(y_limits):
                ax.set_ylim(y_limits[i])
            
            # Remove x-axis labels except for last track
            if i < n_bigwig_tracks - 1 and n_bed_tracks == 0:
                ax.set_xticklabels([])
        
        # Plot BED tracks
        if bed_files:
            for i, bed_file in enumerate(bed_files):
                ax_idx = n_bigwig_tracks + i
                ax = axes[ax_idx]
                
                # Read BED regions
                bed_regions = self._read_regions_from_bed(bed_file)
                
                # Filter regions that overlap with plotting region
                overlapping_regions = []
                for bed_region in bed_regions:
                    bed_chrom, bed_start, bed_end = bed_region
                    if (bed_chrom == chrom and 
                        not (bed_end <= start or bed_start >= end)):
                        overlapping_regions.append(bed_region)
                
                # Plot regions as rectangles
                y_center = 0.5
                height = 0.8
                
                for bed_chrom, bed_start, bed_end in overlapping_regions:
                    # Clip to plotting region
                    plot_start = max(bed_start, start)
                    plot_end = min(bed_end, end)
                    
                    if plot_start < plot_end:
                        rect = patches.Rectangle(
                            (plot_start, y_center - height/2),
                            plot_end - plot_start,
                            height,
                            linewidth=1,
                            edgecolor='black',
                            facecolor='blue',
                            alpha=0.7
                        )
                        ax.add_patch(rect)
                
                # Formatting
                label = bed_labels[i] if bed_labels and i < len(bed_labels) else f"BED {i+1}"
                ax.set_ylabel(label, rotation=90, ha='right', va='center')
                ax.set_ylim(0, 1)
                ax.set_yticks([])
                
                # Remove grid for BED tracks
                ax.grid(False)
        
        # Format x-axis for bottom subplot
        bottom_ax = axes[-1]
        bottom_ax.set_xlabel(f'Genomic Position ({chrom})')
        
        # Format x-axis ticks
        n_ticks = 6
        tick_positions = np.linspace(start, end, n_ticks)
        tick_labels = [f"{int(pos):,}" for pos in tick_positions]
        bottom_ax.set_xticks(tick_positions)
        bottom_ax.set_xticklabels(tick_labels, rotation=45)
        
        # Add title
        if title is None:
            title = f"Genomic Tracks: {chrom}:{start:,}-{end:,}"
        fig.suptitle(title, fontsize=14, y=0.95)
        
        plt.tight_layout()
        
        # Save if requested
        if output_file:
            fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved tracks plot to {output_file}")
        
        return fig


# Convenience functions
def plot_aggregate(
    bigwig_files: List[str],
    regions_bed: str,
    output_file: str,
    labels: Optional[List[str]] = None,
    remove_outliers: float = 1.0,
    log_transform: bool = False,
    smooth: int = 1,
    normalize: bool = False,
    flank: int = 60,
    signal_on_x: bool = False,
    share_y: str = "none",
    plot_boundaries: bool = False,
    title: str = "Aggregated signals",
    **kwargs
) -> plt.Figure:
    """
    Convenient wrapper for TOBIAS-style aggregate plotting
    
    Parameters
    ----------
    bigwig_files : List[str]
        BigWig files (signals) to plot
    regions_bed : str
        BED file with regions (TFBS)
    output_file : str
        Output file path
    labels : Optional[List[str]]
        Labels for signals
    remove_outliers : float
        Value between 0-1 indicating percentile of regions to include (1.0 = keep all)
    log_transform : bool
        Log transform the signals before aggregation
    smooth : int
        Smoothing window size
    normalize : bool
        Normalize the aggregate signal(s) to be between 0-1
    flank : int
        Flanking basepairs (+/-) to show in plot (counted from middle of TFBS)
    signal_on_x : bool
        Show signals on x-axis and TFBSs on y-axis
    share_y : str
        Share y-axis range across plots ("none"/"signals"/"sites"/"both")
    plot_boundaries : bool
        Plot TFBS boundaries
    title : str
        Title of plot
    **kwargs
        Additional parameters
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    
    plotter = PlotAggregate()
    return plotter.plot_aggregate_profile(
        bigwig_files=bigwig_files,
        regions=regions_bed,
        output_file=output_file,
        labels=labels,
        remove_outliers=remove_outliers,
        log_transform=log_transform,
        smooth=smooth,
        normalize=normalize,
        flank=flank,
        signal_on_x=signal_on_x,
        share_y=share_y,
        plot_boundaries=plot_boundaries,
        title=title,
        **kwargs
    )


def plot_heatmap(
    bigwig_file: str,
    regions_bed: str,
    output_file: str,
    **kwargs
) -> plt.Figure:
    """
    Convenient wrapper for heatmap plotting
    
    Parameters
    ----------
    bigwig_file : str
        BigWig file
    regions_bed : str
        BED file with regions
    output_file : str
        Output file path
    **kwargs
        Additional parameters
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    
    plotter = PlotAggregate()
    return plotter.plot_aggregate_heatmap(
        bigwig_file=bigwig_file,
        regions=regions_bed,
        output_file=output_file,
        **kwargs
    )


def plot_tracks(
    region: Tuple[str, int, int],
    bigwig_files: List[str],
    output_file: str,
    **kwargs
) -> plt.Figure:
    """
    Convenient wrapper for track plotting
    
    Parameters
    ----------
    region : Tuple[str, int, int]
        Genomic region
    bigwig_files : List[str]
        BigWig files
    output_file : str
        Output file
    **kwargs
        Additional parameters
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    
    plotter = PlotTracks()
    return plotter.plot_genomic_tracks(
        region=region,
        bigwig_files=bigwig_files,
        output_file=output_file,
        **kwargs
    )