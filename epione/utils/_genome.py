#copy from snapatac2.genome.py


from __future__ import annotations
from collections.abc import Callable

#from snapatac2.datasets import register_datasets
from pathlib import Path
from pooch import Decompress
import pooch

# This is a global variable used to store all datasets. It is initialized only once
# when the data is requested.
_datasets = None

def register_datasets():
    global _datasets
    if _datasets is None:
        _datasets = pooch.create(
            path=pooch.os_cache("snapatac2"),
            base_url="http://renlab.sdsc.edu/kai/public_datasets/",
            env="EPIONE_DATA_DIR",  # The user can overwrite the storage path by setting this environment variable.
            # The registry specifies the files that can be fetched
            registry={
                "atac_pbmc_500_fastqs.tar": "sha256:5897a4790d2841eff69c85b4bef2825166dd0cc2587da91f42eeed09000c5f47",
                "atac_pbmc_500.bam": "sha256:2fac56ca45186943a1daf9da71aed42263ad43a9428f2388fa5f3bcf6d2754ff",
                "atac_pbmc_500.tsv.gz": "sha256:196c5d7ee0169957417e9f4d5502abf1667ef99453328f8d290d4a7f3b205c6c",
                "atac_pbmc_500_downsample.tsv.gz": "sha256:6053cf4578a140bfd8ce34964602769dc5f5ec6b25ba4f2db23cdbd4681b0e2f",

                "atac_pbmc_5k.tsv.gz": "sha256:5fe44c0f8f76ce1534c1ae418cf0707ca5ef712004eee77c3d98d2d4b35ceaec",
                "atac_pbmc_5k.h5ad": "sha256:92ae7f185cdec26517fd8d5acb60b2ce92c71e0ace824de35589c6d7942cab06",
                "atac_pbmc_5k_annotated.h5ad": "sha256:592f1551c27d0cfe4d81e7febad624d6b7d3ebf977b0c3ea64e06b3f3d76f078",

                "colon_transverse.tar": "sha256:18c56bf405ec0ef8e0e2ea31c63bf2299f21bcb82c67f46e8f70f8d71c65ae0e",
                "HEA_cCRE.bed.gz": "sha256:d69ae94649201cd46ffdc634852acfccc317196637c1786aba82068618001408",

                "10x-Multiome-Pbmc10k-ATAC.h5ad": "sha256:24d030fb7f90453a0303b71a1e3e4e7551857d1e70072752d7fff9c918f77217",
                "10x-Multiome-Pbmc10k-RNA.h5ad": "sha256:a25327acff48b20b295c12221a84fd00f8f3f486ff3e7bd090fdef241b996a22",
                "pbmc_10k_atac.tsv.gz": "md5:a959ef83dfb9cae6ff73ab0147d547d1",

                # TF motifs
                "cisBP_human.meme": "sha256:8bf995450258e61cb1c535d5bf9656d580eb68ba68893fa36b77d17ee0730579",
                "Meuleman_2020.meme": "sha256:400dd60ca61dc8388aa0942b42c95920aad7f6bedf5324005cee7e84bcf5b6d0",

                # Genome files
                "gencode_v41_GRCh37.gff3.gz": "sha256:df96d3f0845127127cc87c729747ae39bc1f4c98de6180b112e71dda13592673",
                "gencode_v41_GRCh37.fa.gz": "sha256:ac73947d38df63ccb00724520a5c31d880c1ca423702ca7ccb7e6c2182a362d9",
                #"gencode_v41_GRCh37.fa.gz": "sha256:94330d402e53cf39a1fef6c132e2500121909c2dfdce95cc31d541404c0ed39e",
                "gencode_v41_GRCh38.gff3.gz": "sha256:b82a655bdb736ca0e463a8f5d00242bedf10fa88ce9d651a017f135c7c4e9285",
                "gencode_v41_GRCh38.fa.gz": "sha256:4fac949d7021cbe11117ddab8ec1960004df423d672446cadfbc8cca8007e228",
                "gencode_vM25_GRCm38.gff3.gz": "sha256:e8ed48bef6a44fdf0db7c10a551d4398aa341318d00fbd9efd69530593106846",
                "gencode_vM25_GRCm38.fa.gz": "sha256:617b10dc7ef90354c3b6af986e45d6d9621242b64ed3a94c9abeac3e45f18c17",
                "gencode_vM30_GRCm39.gff3.gz": "sha256:6f433e2676e26569a678ce78b37e94a64ddd50a09479e433ad6f75e37dc82e48",
                "gencode_vM30_GRCm39.fa.gz": "sha256:3b923c06a0d291fe646af6bf7beaed7492bf0f6dd5309d4f5904623cab41b0aa",
            },
            urls={
                "atac_pbmc_500_fastqs.tar": "https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_500_nextgem/atac_pbmc_500_nextgem_fastqs.tar",
                "atac_pbmc_500.tsv.gz": "https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_500_nextgem/atac_pbmc_500_nextgem_fragments.tsv.gz",
                "atac_pbmc_500.bam": "https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_500_nextgem/atac_pbmc_500_nextgem_possorted_bam.bam",
                "atac_pbmc_500_downsample.tsv.gz": "https://data.mendeley.com/api/datasets/dr2z4jbcx3/draft/files/b0e7e9e8-9ffb-4710-8619-73f7e5cbd10b?a=758c37e5-4832-4c91-af89-9a1a83a051b3",

                "atac_pbmc_5k.tsv.gz": "https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_5k_nextgem/atac_pbmc_5k_nextgem_fragments.tsv.gz", 
                "atac_pbmc_5k.h5ad": "https://osf.io/download/rj9nc/",
                "atac_pbmc_5k_annotated.h5ad": "https://osf.io/download/e9vc3/",

                "10x-Multiome-Pbmc10k-ATAC.h5ad": "https://data.mendeley.com/api/datasets/dr2z4jbcx3/draft/files/165dfb5c-c557-42a0-bd21-1276d4d7b23e?a=758c37e5-4832-4c91-af89-9a1a83a051b3",
                "10x-Multiome-Pbmc10k-RNA.h5ad": "https://data.mendeley.com/api/datasets/dr2z4jbcx3/draft/files/d079a087-2913-4e29-979e-638e5932bd8c?a=758c37e5-4832-4c91-af89-9a1a83a051b3", 
                "pbmc_10k_atac.tsv.gz": "https://cf.10xgenomics.com/samples/cell-arc/1.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz",

                "colon_transverse.tar": "https://data.mendeley.com/api/datasets/dr2z4jbcx3/draft/files/eaa46151-a73f-4ef5-8b05-9648c8d1efda?a=758c37e5-4832-4c91-af89-9a1a83a051b3", 
                "HEA_cCRE.bed.gz": "https://data.mendeley.com/api/datasets/dr2z4jbcx3/draft/files/91f93222-1a24-49a5-92e3-d9105ec53f91?a=758c37e5-4832-4c91-af89-9a1a83a051b3",

                "cisBP_human.meme": "https://osf.io/download/uk6vn",
                "Meuleman_2020.meme": "https://osf.io/download/6uet5/",

                "gencode_v41_GRCh37.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/GRCh37_mapping/gencode.v41lift37.basic.annotation.gff3.gz",
                "gencode_v41_GRCh37.fa.gz": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz",
                #"gencode_v41_GRCh37.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/GRCh37_mapping/GRCh37.primary_assembly.genome.fa.gz",
                "gencode_v41_GRCh38.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.basic.annotation.gff3.gz",
                "gencode_v41_GRCh38.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/GRCh38.primary_assembly.genome.fa.gz",
                "gencode_vM25_GRCm38.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.basic.annotation.gff3.gz",
                "gencode_vM25_GRCm38.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/GRCm38.primary_assembly.genome.fa.gz",
                "gencode_vM30_GRCm39.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M30/gencode.vM30.basic.annotation.gff3.gz",
                "gencode_vM30_GRCm39.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M30/GRCm39.primary_assembly.genome.fa.gz",
            },
        )
    return _datasets

class Genome:
    """
    A class that encapsulates information about a genome, including its FASTA sequence,
    its annotation, and chromosome sizes.

    Attributes
    ----------
    fasta
        The path to the FASTA file.
    annotation
        The path to the annotation file.
    chrom_sizes
        A dictionary containing chromosome names and sizes.

    Raises
    ------
    ValueError
        If `fasta` or `annotation` are not a Path, a string, or a callable.
    """

    def __init__(
        self,
        *,
        fasta: Path | Callable[[], Path], 
        annotation: Path | Callable[[], Path],
        chrom_sizes: dict[str, int] | None = None,
    ):
        """
        Initializes the Genome object with paths or callables for FASTA and annotation files,
        and optionally, chromosome sizes.

        Parameters
        ----------
        fasta
            A Path or callable that returns a Path to the FASTA file.
        annotation
            A Path or callable that returns a Path to the annotation file.
        chrom_sizes
            Optional chromosome sizes. If not provided, chromosome sizes will
            be inferred from the FASTA file.
        """
        if callable(fasta):
            self._fetch_fasta = fasta
            self._fasta = None
        elif isinstance(fasta, Path) or isinstance(fasta, str):
            self._fasta = Path(fasta)
            self._fetch_fasta = None
        else:
            raise ValueError("fasta must be a Path or Callable")

        if callable(annotation):
            self._fetch_annotation = annotation
            self._annotation = None
        elif isinstance(annotation, Path) or isinstance(annotation, str):
            self._annotation = Path(annotation)
            self._fetch_annotation = None
        else:
            raise ValueError("annotation must be a Path or Callable")

        self._chrom_sizes = chrom_sizes

    @property
    def fasta(self):
        """
        The Path to the FASTA file. 

        Returns
        -------
        Path
            The path to the FASTA file.
        """
        if self._fasta is None:
            self._fasta = Path(self._fetch_fasta())
        return self._fasta

    @property
    def annotation(self):
        """
        The Path to the annotation file.

        Returns
        -------
        Path
            The path to the annotation file.
        """
        if self._annotation is None:
            self._annotation = Path(self._fetch_annotation())
        return self._annotation

    @property
    def chrom_sizes(self):
        """
        A dictionary with chromosome names as keys and their lengths as values.

        Returns
        -------
        dict[str, int]
            A dictionary of chromosome sizes.
        """
        if self._chrom_sizes is None:
            from pyfaidx import Fasta
            fasta = Fasta(self.fasta)
            self._chrom_sizes = {chr: len(fasta[chr]) for chr in fasta.keys()}
        return self._chrom_sizes
        
