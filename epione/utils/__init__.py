from ._genome import Genome, register_datasets
from .genome import GRCh37, GRCh38, hg19, hg38

from ._read import read_ATAC_10x, read_gtf, read_features
from ._findgenes import find_genes, Annotation

from ._call_peaks import merge_peaks

# TOBIAS-inspired footprint analysis functionality
from ._atacorrect import (
    AtacBias, atacorrect_core, atacorrect, 
    save_atacorrect_results_to_bigwig
)
from ._score_bigwig import (
    FootprintScorer, score_bigwig_core, score_bigwig,
    calculate_aggregate_scores, compare_methods
)
from ._bindetect import bindetect
from ._plotting import (
    FootprintPlotter, PlotTracks,
    plot_heatmap, plot_tracks
)
from ._plotting_tobias import (
    plot_aggregate, plot_aggregate_tobias
)
from ._motif_tools import (
    MotifMatrix, FormatMotifs, ClusterMotifs,
    format_motifs, cluster_motifs
)
from ._network_tools import (
    FragmentFilter, NetworkBuilder,
    filter_fragments, create_network
)


__all__ = [
    # Original functionality
    'Genome',
    'register_datasets',
    'read_ATAC_10x',
    'read_gtf',
    'read_features',
    'find_genes',
    'Annotation',
    'merge_peaks',
    'GRCh37',
    'GRCh38',
    'hg19',
    'hg38',
    
    # TOBIAS-inspired footprint analysis
    # ATACorrect
    'AtacBias',
    'atacorrect_core',
    'atacorrect',
    'save_atacorrect_results_to_bigwig',
    
    # ScoreBigwig
    'FootprintScorer',
    'score_bigwig_core', 
    'score_bigwig',
    'calculate_aggregate_scores',
    'compare_methods',
    
    # BINDetect
    'bindetect',
    
    # Plotting
    'FootprintPlotter',
    'PlotTracks',
    'plot_aggregate',
    'plot_aggregate_tobias',
    'plot_heatmap',
    'plot_tracks',
    
    # Motif tools
    'MotifMatrix',
    'FormatMotifs',
    'ClusterMotifs',
    'format_motifs',
    'cluster_motifs',
    
    # Network tools
    'FragmentFilter',
    'NetworkBuilder',
    'filter_fragments',
    'create_network',
]
