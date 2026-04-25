#TODO: add motif analysis

from typing import Union

from anndata import AnnData
from mudata import MuData


def add_dna_sequence(
    adata: AnnData,
    genome_name: str = "GRCh38",
    genome_dir: str = "data",
    chr_var_key: str = "chr",
    start_var_key: str = "start",
    end_var_key: str = "end",
):
    """
    Add DNA sequence to the input data.

    Parameters
    ----------
    adata : AnnData
        The input data.
    genome_name : str, optional
        The name of the genome, by default "GRCh38".
    genome_dir : str, optional
        The directory of the genome, by default "data".
    chr_var_key : str, optional
        The key of the chromosome in the input data, by default "chr".
    start_var_key : str, optional
        The key of the start position in the input data, by default "start".
    end_var_key : str, optional
        The key of the end position in the input data, by default "end".
    """
    import scvi
    scvi.data.add_dna_sequence(
        adata,
        genome_name=genome_name,
        genome_dir=genome_dir,
        chr_var_key=chr_var_key,
        start_var_key=start_var_key,
        end_var_key=end_var_key,
    )

def match_motif(
    data: Union[AnnData, MuData],
    methods: list[str] = ["chromVAR","scBasset"],
    genome_file: str = None,
    delimiter=":|-",
    motif_db: str = "JASPAR2024",
    chunk_size: int = 15000,
):
    """
    Perform motif enrichment analysis on the input data.

    Parameters
    ----------
    data : Union[AnnData, MuData]
        AnnData or MuData object with peak counts or 'atac' modality.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError(
            "Expected AnnData or MuData object with 'atac' modality")

    from external.pychromvar import match_motif, get_bg_peaks, add_peak_seq, add_gc_bias, compute_deviations

    
    

    if "chromVAR" in methods:
        #add_peak_seq(adata, genome_file=genome_file, delimiter=delimiter)
        add_gc_bias(adata)
        get_bg_peaks(adata)

        from pyjaspar import jaspardb
        # get motifs
        jdb_obj = jaspardb(release=motif_db)
        motifs = jdb_obj.fetch_motifs(
            collection = 'CORE',
            tax_group = ['vertebrates'])

        match_motif(adata, motifs=motifs)
        


    if "scBasset" in methods:
        import scvi
        split_interval = adata.var["gene_ids"].str.split(":", expand=True)
        adata.var["chr"] = split_interval[0]
        split_start_end = split_interval[1].str.split("-", expand=True)
        adata.var["start"] = split_start_end[0].astype(int)
        adata.var["end"] = split_start_end[1].astype(int)
        # Filter out non-chromosomal regions
        mask = adata.var["chr"].str.startswith("chr")
        adata = adata[:, mask].copy()

        bdata = adata.transpose()
        bdata.layers["binary"] = (bdata.X.copy() > 0).astype(float)
        scvi.external.SCBASSET.setup_anndata(bdata, layer="binary", dna_code_key="dna_code")

        bas = scvi.external.SCBASSET(bdata)
        bas.train(precision=16)

        latent = bas.get_latent_representation()
        adata.obsm["X_scbasset"] = latent

        print(latent.shape)
        return bas

        