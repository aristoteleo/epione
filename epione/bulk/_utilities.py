

def getTempFileName(suffix=''):
    """
    Return a temporary file name. The calling function is responsible for
    deleting this upon completion.
    """
    import tempfile
    _tempFile = tempfile.NamedTemporaryFile(prefix="_deeptools_",
                                            suffix=suffix,
                                            delete=False)

    memFileName = _tempFile.name
    _tempFile.close()
    return memFileName

def gtfOptions(allArgs=None):
    """
    This is used a couple places to setup arguments to mapReduce
    """
    transcriptID = "transcript"
    exonID = "exon"
    transcript_id_designator = "transcript_id"
    keepExons = False
    if allArgs is not None:
        allArgs = vars(allArgs)
        transcriptID = allArgs.get("transcriptID", transcriptID)
        exonID = allArgs.get("exonID", exonID)
        transcript_id_designator = allArgs.get("transcript_id_designator", transcript_id_designator)
        keepExons = allArgs.get("keepExons", keepExons)
    return transcriptID, exonID, transcript_id_designator, keepExons