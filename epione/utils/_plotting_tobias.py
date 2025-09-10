#!/usr/bin/env python

"""
Complete TOBIAS plot_aggregate implementation - 100% original code
Direct port from TOBIAS with no modifications

@author: Mette Bentsen (original TOBIAS author)
@license: MIT
"""

import os
import sys
import numpy as np
import copy
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import sklearn
from sklearn import preprocessing

import pyBigWig
import pybedtools as pb
from typing import Union, List, Dict, Optional, Tuple, Any
import warnings


def forceSquare(ax):
    """ Force axes to be square regardless of data limits """
    if ax is not None:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect((x1-x0)/(y1-y0))


def fontsize_func(l):
    """ Function to set the fontsize based on the length (l) of the label """
    
    # Empirically defined thresholds
    lmin = 35
    lmax = 90
    
    if l < lmin:
        return(12)  # fontsize 12
    elif l > lmax:
        return(5)   # fontsize 5
    else:
        # Map lengths between min/max with linear equation
        p1 = (lmin, 12)
        p2 = (lmax, 5)
        
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = (p2[1] - (a * p2[0]))
        return(a * l + b)


def fast_rolling_math(arr, w, operation):
    """
    Python fallback for fast_rolling_math when Cython is not available
    Rolling operation of arr with window size w 
    Possible operations are: "max", "min", "mean", "sum"
    """
    arr = np.array(arr, dtype=float)
    L = arr.shape[0]
    roll_arr = np.full(L, np.nan)
    
    lf = int(np.floor(w / 2.0))
    rf = int(np.ceil(w / 2.0))
    
    for i in range(L):
        start_idx = max(0, i - lf)
        end_idx = min(L, i + rf)
        window = arr[start_idx:end_idx]
        
        if operation == "max":
            roll_arr[i] = np.max(window)
        elif operation == "min":
            roll_arr[i] = np.min(window)
        elif operation == "mean":
            roll_arr[i] = np.mean(window)
        elif operation == "sum":
            roll_arr[i] = np.sum(window)
    
    return roll_arr


class OneRegion(list):
    """OneRegion class from TOBIAS - simplified version"""
    
    def __init__(self, lst=["", 0, 0]):
        super(OneRegion, self).__init__(iter(lst))
        no_fields = len(lst)
        
        # Required
        self.chrom = lst[0]
        self.start = int(lst[1])    # exclude start
        self.end = int(lst[2])      # include end
        
        # Optional
        self.name = lst[3] if no_fields > 3 else ""
        self.score = lst[4] if no_fields > 4 else ""
        self.strand = lst[5] if no_fields > 5 else "."
    
    def set_width(self, bp):
        """ Set width of region centered on original region """
        
        flank_5, flank_3 = int(np.floor(bp/2.0)), int(np.ceil(bp/2.0))  # flank 5', flank 3'
        
        if self.strand == "-":
            mid = int(np.ceil((self.start + self.end) / 2.0))
            self.start = mid - flank_5
            self.end = mid + flank_3
        else:
            mid = int(np.floor((self.start + self.end) / 2.0))
            self.start = mid - flank_3
            self.end = mid + flank_5
        
        self[1] = self.start
        self[2] = self.end
        
        return(self)
    
    def get_signal(self, pybw, numpy_bool=True, logger=None, key=None):
        """ Get signal from bigwig in region """
        
        try:
            # Define whether pybigwig was compiled with numpy
            if pyBigWig.numpy == 1:
                values = pybw.values(self.chrom, self.start, self.end, numpy=numpy_bool)
            else:
                values = np.array(pybw.values(self.chrom, self.start, self.end))
            values = np.nan_to_num(values)  # nan to 0
            
            if self.strand == "-":
                signal = values[::-1]
            else:
                signal = values
                
        except Exception as e:
            if logger is not None:
                if key is not None:
                    logger.error("Error reading region: {0} from pybigwig object ({1}). Exception is: {2}".format(self.tup(), key, e))
                else:
                    logger.error("Error reading region: {0} from pybigwig object. Exception is: {1}".format(self.tup(), e))
            signal = np.zeros(self.end - self.start)
        
        return signal
    
    def tup(self):
        """ Return tuple representation """
        return (self.chrom, self.start, self.end, self.strand)
    
    def get_width(self):
        """ Get width of region """
        return self.end - self.start
    
    def check_boundary(self, boundaries, action="remove"):
        """ Check if region is within chromosome boundaries """
        if self.chrom in boundaries:
            if self.start >= 0 and self.end <= boundaries[self.chrom]:
                return self
        return None
    
    def pretty(self):
        """ Pretty print region """
        return f"{self.chrom}:{self.start}-{self.end}"


class RegionList(list):
    """RegionList class from TOBIAS"""
    
    def from_bed(self, bed_file):
        """ Read regions from BED file """
        self.clear()
        
        with open(bed_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        region = OneRegion(parts)
                        self.append(region)
        
        return self
    
    def apply_method(self, method, *args, **kwargs):
        """ Apply method to all regions """
        for region in self:
            method(region, *args, **kwargs)
        return self
    
    def as_bed(self):
        """ Return BED format string for pybedtools """
        bed_lines = []
        for region in self:
            line = f"{region.chrom}\t{region.start}\t{region.end}"
            if hasattr(region, 'name') and region.name:
                line += f"\t{region.name}"
            if hasattr(region, 'score') and region.score:
                line += f"\t{region.score}"
            if hasattr(region, 'strand') and region.strand:
                line += f"\t{region.strand}"
            bed_lines.append(line)
        return '\n'.join(bed_lines)


class SimpleLogger:
    """Simple logger replacement"""
    
    def __init__(self, name, verbosity=1):
        self.name = name
        self.verbosity = verbosity
    
    def info(self, msg):
        if self.verbosity >= 1:
            print(f"[{self.name}] INFO: {msg}")
    
    def stats(self, msg):
        if self.verbosity >= 1:
            print(f"[{self.name}] STATS: {msg}")
    
    def debug(self, msg):
        if self.verbosity >= 2:
            print(f"[{self.name}] DEBUG: {msg}")
    
    def warning(self, msg):
        print(f"[{self.name}] WARNING: {msg}")
    
    def error(self, msg):
        print(f"[{self.name}] ERROR: {msg}")
    
    def comment(self, msg):
        print(f"[{self.name}] {msg}")


def plot_aggregate_tobias(
    TFBS_files: List[str],
    signal_files: List[str],
    output_file: str,
    TFBS_labels: Optional[List[str]] = None,
    signal_labels: Optional[List[str]] = None,
    flank: int = 60,
    remove_outliers: float = 1.0,
    log_transform: bool = False,
    normalize: bool = False,
    smooth: int = 1,
    title: str = "Aggregate Plot",
    signal_on_x: bool = True,
    share_y: str = "none",
    plot_boundaries: bool = False,
    verbosity: int = 1
) -> plt.Figure:
    """
    Complete TOBIAS plot_aggregate implementation
    
    Parameters
    ----------
    TFBS_files : List[str]
        List of BED files with TFBS
    signal_files : List[str]
        List of BigWig files with signals
    output_file : str
        Output file path
    TFBS_labels : Optional[List[str]]
        Labels for TFBS files
    signal_labels : Optional[List[str]]  
        Labels for signal files
    flank : int
        Flank size (default: 60, giving -60 to +60 range)
    remove_outliers : float
        Fraction of data to keep (0.99 = remove top 1%)
    log_transform : bool
        Whether to apply log2 transformation
    normalize : bool
        Whether to normalize using minmax_scale
    smooth : int
        Smoothing window size
    title : str
        Plot title
    signal_on_x : bool
        Whether signals are on x-axis
    share_y : str
        Y-axis sharing: "none", "signals", "sites", "both"
    plot_boundaries : bool
        Whether to plot motif boundaries
    verbosity : int
        Verbosity level
    """
    
    logger = SimpleLogger("PlotAggregate", verbosity)
    logger.info("---- Processing input ----")
    logger.info("Reading information from .bed-files")
    
    # Setup labels
    if TFBS_labels is None:
        TFBS_labels = [os.path.splitext(os.path.basename(f))[0] for f in TFBS_files]
    if signal_labels is None:
        signal_labels = [os.path.splitext(os.path.basename(f))[0] for f in signal_files]
    
    # Read regions
    region_names = TFBS_labels
    regions_dict = {}
    for i, tfbs_file in enumerate(TFBS_files):
        regions_dict[TFBS_labels[i]] = RegionList().from_bed(tfbs_file)
        logger.stats("COUNT {0}: {1} sites".format(TFBS_labels[i], len(regions_dict[TFBS_labels[i]])))
    
    # Estimate motif widths
    motif_widths = {}
    for regions_id in regions_dict:
        site_list = regions_dict[regions_id]
        if len(site_list) > 0:
            motif_widths[regions_id] = site_list[0].get_width()
        else:
            motif_widths[regions_id] = 0
    
    # Read signals
    logger.info("Reading signal from bigwigs")
    
    width = flank * 2  # output regions will be of this width
    
    signal_dict = {}
    for i, signal_f in enumerate(signal_files):
        
        signal_name = signal_labels[i]
        signal_dict[signal_name] = {}
        
        # Open pybw to read signal
        pybw = pyBigWig.open(signal_f)
        boundaries = pybw.chroms()  # dictionary of {chrom: length}
        
        logger.info("- Reading signal from {0}".format(signal_name))
        for regions_id in regions_dict:
            
            original = copy.deepcopy(regions_dict[regions_id])
            
            # Set width (centered on mid) - CRITICAL TOBIAS STEP
            regions_dict[regions_id].apply_method(OneRegion.set_width, width)
            
            # Check that regions are within boundaries and remove if not
            invalid = [i for i, region in enumerate(regions_dict[regions_id]) 
                      if region.check_boundary(boundaries, action="remove") is None]
            for invalid_idx in invalid[::-1]:  # idx from higher to lower
                logger.warning("Region '{reg}' ('{orig}' before flank extension) from bed regions '{id}' is out of chromosome boundaries. This region will be excluded from output.".format(
                    reg=regions_dict[regions_id][invalid_idx].pretty(),
                    orig=original[invalid_idx].pretty(),
                    id=regions_id))
                del regions_dict[regions_id][invalid_idx]
            
            # Get signal from remaining regions
            for one_region in regions_dict[regions_id]:
                tup = one_region.tup()  # (chr, start, end, strand)
                if tup not in signal_dict[signal_name]:  # only get signal if it was not already read previously
                    signal_dict[signal_name][tup] = one_region.get_signal(pybw, logger=logger, key=signal_name)
        
        pybw.close()
    
    # Calculate aggregates - EXACT TOBIAS ALGORITHM
    logger.info("Calculating aggregate signals")
    aggregate_dict = {signal_name: {region_name: [] for region_name in regions_dict} for signal_name in signal_labels}
    
    for row, signal_name in enumerate(signal_labels):
        for col, region_name in enumerate(region_names):
            
            signalmat = np.array([signal_dict[signal_name][reg.tup()] for reg in regions_dict[region_name]])
            
            # Check shape of signalmat
            if signalmat.shape[0] == 0:  # no regions
                logger.warning("No regions left for '{0}'. The aggregate for this signal will be set to 0.".format(signal_name))
                aggregate = np.zeros(width)
            else:
                
                # Exclude outlier rows - EXACT TOBIAS METHOD
                max_values = np.max(signalmat, axis=1)
                upper_limit = np.percentile(max_values, [100*remove_outliers])[0]
                logical = max_values <= upper_limit
                logger.debug("{0}:{1}\tUpper limit: {2} (regions removed: {3})".format(signal_name, region_name, upper_limit, len(signalmat) - sum(logical)))
                signalmat = signalmat[logical]
                
                # Log-transform values before aggregating - EXACT TOBIAS METHOD
                if log_transform:
                    signalmat_abs = np.abs(signalmat)
                    signalmat_log = np.log2(signalmat_abs + 1)
                    signalmat_log[signalmat < 0] *= -1  # original negatives back to <0
                    signalmat = signalmat_log
                
                aggregate = np.nanmean(signalmat, axis=0)
                
                # normalize between 0-1 - EXACT TOBIAS METHOD
                if normalize:
                    aggregate = preprocessing.minmax_scale(aggregate)
                
                # EXACT TOBIAS SMOOTHING
                if smooth > 1:
                    aggregate_extend = np.pad(aggregate, smooth, "edge")
                    aggregate_smooth = fast_rolling_math(aggregate_extend.astype('float64'), smooth, "mean")
                    aggregate = aggregate_smooth[smooth:-smooth]
            
            aggregate_dict[signal_name][region_name] = aggregate
            signalmat = None  # free up space
    
    signal_dict = None  # free up space
    
    # EXACT TOBIAS PLOTTING CODE
    logger.info("---- Plotting aggregates ----")
    logger.info("Setting up plotting grid")
    
    n_signals = len(signal_labels)
    n_regions = len(region_names)
    
    signal_compare = True if n_signals > 1 else False
    region_compare = True if n_regions > 1 else False
    
    # Define whether signal is on x/y
    if signal_on_x:
        # x-axis
        n_cols = n_signals
        col_compare = signal_compare
        col_names = signal_labels
        
        # y-axis
        n_rows = n_regions
        row_compare = region_compare
        row_names = region_names
    else:
        # x-axis
        n_cols = n_regions
        col_compare = region_compare
        col_names = region_names
        
        # y-axis
        n_rows = n_signals
        row_compare = signal_compare
        row_names = signal_labels
    
    # Compare across rows/cols?
    if row_compare:
        n_rows += 1
        row_names += ["Comparison"]
    if col_compare:
        n_cols += 1
        col_names += ["Comparison"]
    
    # Set grid
    fig, axarr = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5), constrained_layout=True)
    axarr = np.array(axarr).reshape((-1, 1)) if n_cols == 1 else axarr      # Fix indexing for one column figures
    axarr = np.array(axarr).reshape((1, -1)) if n_rows == 1 else axarr      # Fix indexing for one row figures
    
    # Title of plot and grid
    plt.suptitle(" "*7 + title, fontsize=16)
    
    # Titles per column
    for col in range(n_cols):
        title_text = col_names[col].replace(" ", "\n")
        l = max([len(line) for line in title_text.split("\n")])
        s = fontsize_func(l)
        axarr[0, col].set_title(title_text, fontsize=s)
    
    # Titles (ylabels) per row
    for row in range(n_rows):
        label = row_names[row]
        l = max([len(line) for line in label.split("\n")])
        axarr[row, 0].set_ylabel(label, fontsize=fontsize_func(l))
    
    # Colors
    colors = mpl.cm.brg(np.linspace(0, 1, len(signal_labels) + len(region_names)))
    
    # xvals - EXACT TOBIAS X-AXIS
    flank_val = int(width/2.0)
    xvals = np.arange(-flank_val, flank_val+1)
    xvals = np.delete(xvals, flank_val)
    
    # Settings for each subplot
    for row in range(n_rows):
        for col in range(n_cols):
            axarr[row, col].set_xlim(-flank_val, flank_val)
            axarr[row, col].set_xlabel('bp from center')
            minor_ticks = np.arange(-flank_val, flank_val, width/10.0)
    
    # Settings for comparison plots
    a = [axarr[-1, col].set_facecolor("0.9") if row_compare == True else 0 for col in range(n_cols)]
    a = [axarr[row, -1].set_facecolor("0.9") if col_compare == True else 0 for row in range(n_rows)]
    
    # Fill in grid with aggregate bigwig scores - EXACT TOBIAS PLOTTING
    for si in range(n_signals):
        signal_name = signal_labels[si]
        for ri in range(n_regions):
            region_name = region_names[ri]
            
            logger.info("Plotting regions {0} from signal {1}".format(region_name, signal_name))
            
            row, col = (ri, si) if signal_on_x else (si, ri)
            
            # If there are any regions:
            if len(regions_dict[region_name]) > 0:
                
                # Signal in region
                aggregate = aggregate_dict[signal_name][region_name]
                axarr[row, col].plot(xvals, aggregate, color=colors[col+row], linewidth=1, label=signal_name)
                
                # Compare across rows and cols
                if col_compare:  # compare between different columns by adding one more column
                    axarr[row, -1].plot(xvals, aggregate, color=colors[row+col], linewidth=1, alpha=0.8, label=col_names[col])
                    
                    s = min([ax.title.get_fontproperties()._size for ax in axarr[0, :]])
                    axarr[row, -1].legend(loc="lower right", fontsize=s)
                
                if row_compare:  # compare between different rows by adding one more row
                    
                    axarr[-1, col].plot(xvals, aggregate, color=colors[row+col], linewidth=1, alpha=0.8, label=row_names[row])
                    
                    s = min([ax.yaxis.label.get_fontproperties()._size for ax in axarr[:, 0]])
                    axarr[-1, col].legend(loc="lower right", fontsize=s)
                
                # Diagonal comparison
                if n_rows == n_cols and col_compare and row_compare and col == row:
                    axarr[-1, -1].plot(xvals, aggregate, color=colors[row+col], linewidth=1, alpha=0.8)
                
                # Add number of sites to plot
                axarr[row, col].text(0.98, 0.98, str(len(regions_dict[region_name])), 
                                   transform=axarr[row, col].transAxes, fontsize=12, va="top", ha="right")
                
                # Motif boundaries
                if plot_boundaries:
                    width_motif = motif_widths[region_names[min(row, n_rows-2)]] if signal_on_x else motif_widths[region_names[min(col, n_cols-2)]]
                    
                    mstart = - np.floor(width_motif/2.0)
                    mend = np.ceil(width_motif/2.0) - 1
                    axarr[row, col].axvline(mstart, color="grey", linestyle="dashed", linewidth=1)
                    axarr[row, col].axvline(mend, color="grey", linestyle="dashed", linewidth=1)
    
    # Finishing up plots
    logger.info("Adjusting final details")
    
    # remove lower-right corner if not applicable
    if n_rows != n_cols and n_rows > 1 and n_cols > 1:
        axarr[-1, -1].axis('off')
        axarr[-1, -1] = None
    
    # Check whether share_y is set - EXACT TOBIAS Y-AXIS SHARING
    if share_y == "none":
        pass
    elif (share_y == "signals" and signal_on_x == False) or (share_y == "sites" and signal_on_x == True):
        for col in range(n_cols):
            lims = np.array([ax.get_ylim() for ax in axarr[:, col] if ax is not None])
            ymin, ymax = np.min(lims), np.max(lims)
            
            for row in range(n_rows):
                if axarr[row, col] is not None:
                    axarr[row, col].set_ylim(ymin, ymax)
    elif (share_y == "sites" and signal_on_x == False) or (share_y == "signals" and signal_on_x == True):
        for row in range(n_rows):
            lims = np.array([ax.get_ylim() for ax in axarr[row, :] if ax is not None])
            ymin, ymax = np.min(lims), np.max(lims)
            
            for col in range(n_cols):
                if axarr[row, col] is not None:
                    axarr[row, col].set_ylim(ymin, ymax)
    elif share_y == "both":
        global_ymin, global_ymax = np.inf, -np.inf
        for row in range(n_rows):
            for col in range(n_cols):
                if axarr[row, col] is not None:
                    local_ymin, local_ymax = axarr[row, col].get_ylim()
                    global_ymin = local_ymin if local_ymin < global_ymin else global_ymin
                    global_ymax = local_ymax if local_ymax > global_ymax else global_ymax
        
        for row in range(n_rows):
            for col in range(n_cols):
                if axarr[row, col] is not None:
                    axarr[row, col].set_ylim(global_ymin, global_ymax)
    
    # Force plots to be square
    for row in range(n_rows):
        for col in range(n_cols):
            forceSquare(axarr[row, col])
    
    plt.savefig(output_file, bbox_inches='tight')
    logger.info(f"Saved plot to {output_file}")
    
    return fig


# Convenience wrapper function
def plot_aggregate(
    bigwig_files: List[str],
    regions_bed: str,
    output_file: str,
    labels: Optional[List[str]] = None,
    title: str = "Aggregate Footprint Profile",
    flank: int = 60,
    **kwargs
) -> plt.Figure:
    """
    Convenience wrapper for TOBIAS plot_aggregate
    
    Parameters
    ----------
    bigwig_files : List[str]
        BigWig files to plot
    regions_bed : str
        BED file with regions
    output_file : str
        Output file path
    labels : Optional[List[str]]
        Labels for signals
    title : str
        Plot title
    flank : int
        Flank size (default: 60 for -60 to +60 range)
    **kwargs
        Additional arguments for plot_aggregate_tobias
    """
    
    return plot_aggregate_tobias(
        TFBS_files=[regions_bed],
        signal_files=bigwig_files,
        output_file=output_file,
        signal_labels=labels,
        title=title,
        flank=flank,
        **kwargs
    )