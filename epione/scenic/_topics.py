"""cisTopic topic modelling — a port of pycisTopic's ``run_cgs_models`` (SCENIC+ 1.0).

``run_topic_models`` reproduces ``pycisTopic.lda_models.run_cgs_models``:
latent Dirichlet allocation fitted with the collapsed Gibbs sampler of the
``lda`` package (Griffiths & Steyvers 2004; ``lda`` 3.0.0, MPL-2.0), followed
by the tmtoolkit model-quality metrics (Arun 2010, Cao Juan 2009, Mimno 2011)
and pycisTopic's log-likelihood. ``evaluate_topic_models`` reproduces
``evaluate_models`` (automatic model selection).

The sampler is a numba re-implementation of ``lda/_lda.pyx`` that consumes
the same random stream (``numpy.random.RandomState(seed).rand(131072)``,
re-shuffled with a fresh ``RandomState(seed)`` before every iteration) and
evaluates the per-topic weights with the same floating-point operations, so
a given seed returns the same topic assignments, bit for bit. The speed-ups
leave every value unchanged: topic-word counts are stored word-major (the
inner loop over topics reads one cache line instead of K), the count row of
an upcoming token is prefetched, the new topic is found by a branch-free
count instead of a bisection, and models run as threads instead of Ray
workers.

The metrics reproduce numpy 1.26 bit for bit on any CPU (see :mod:`._np126`
and the ``compat`` argument): tie order of ``np.argsort``, SVML ``np.log`` and
the buffered pairwise ``np.sum``.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._np126 import argpartition_np126, argsort_compat, log_compat, rowsum_compat

try:
    from numba import njit, types
    from numba.core import cgutils
    from numba.extending import intrinsic
    from llvmlite import ir

    @intrinsic
    def _prefetch_row(typingctx, arr, row, byte_offset):
        """Hint the CPU to load ``arr[row]`` + ``byte_offset`` (``llvm.prefetch``).

        A pure performance hint: it changes no value.
        """
        sig = types.void(arr, row, byte_offset)

        def codegen(context, builder, signature, args):
            aryty = signature.args[0]
            ary = context.make_array(aryty)(context, builder, args[0])
            strides = cgutils.unpack_tuple(builder, ary.strides, aryty.ndim)
            row_i = context.cast(builder, args[1], signature.args[1], types.intp)
            off_i = context.cast(builder, args[2], signature.args[2], types.intp)
            offset = builder.add(builder.mul(row_i, strides[0]), off_i)
            base = builder.ptrtoint(ary.data, ir.IntType(64))
            addr = builder.inttoptr(builder.add(base, offset), ir.IntType(8).as_pointer())
            fnty = ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer(), ir.IntType(32),
                                                   ir.IntType(32), ir.IntType(32)])
            fn = cgutils.get_or_insert_function(builder.module, fnty, "llvm.prefetch.p0i8")
            i32 = ir.IntType(32)
            builder.call(fn, [addr, ir.Constant(i32, 0), ir.Constant(i32, 3), ir.Constant(i32, 1)])
            return context.get_dummy_value()

        return sig, codegen
except ImportError:  # pragma: no cover - numba ships with scanpy
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f


_N_RANDS = 1024 ** 2 // 8  # ``lda.LDA`` draws 1 MiB of uniforms once
_PREFETCH = 32             # tokens ahead whose topic-count row is prefetched


def _console(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"  └─ [scenic.topics] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Collapsed Gibbs sampler (lda/_lda.pyx)
# ---------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def _initialize(WS, DS, n_topics, nwz, ndz, nz):
    """``LDA._initialize``: token ``i`` starts in topic ``i % n_topics``."""
    N = WS.shape[0]
    ZS = np.empty(N, dtype=np.int32)
    for i in range(N):
        z = i % n_topics
        ZS[i] = z
        ndz[DS[i], z] += 1
        nwz[WS[i], z] += 1
        nz[z] += 1
    return ZS


@njit(cache=True, nogil=True)
def _sample_topics(WS, DS, ZS, nwz, ndz, nz, alpha, eta, eta_sum, rands, dist_sum):
    """One sweep of ``lda._lda._sample_topics``.

    ``nwz`` is the transpose of ``lda``'s ``nzw`` (words × topics). Each weight
    is ``(n_wk + eta) / (n_k + eta_sum) * (n_dk + alpha)`` evaluated left to
    right in double precision and the running sum is accumulated in topic
    order, as in ``lda``.
    """
    N = WS.shape[0]
    n_rand = rands.shape[0]
    n_topics = nz.shape[0]
    j = 0
    for i in range(N):
        if i + _PREFETCH < N:
            _prefetch_row(nwz, WS[i + _PREFETCH], 0)
            if n_topics > 16:
                _prefetch_row(nwz, WS[i + _PREFETCH], 64)
            if n_topics > 32:
                _prefetch_row(nwz, WS[i + _PREFETCH], 128)
        w = WS[i]
        d = DS[i]
        z = ZS[i]
        nwz[w, z] -= 1
        ndz[d, z] -= 1
        nz[z] -= 1

        # per-topic weights first (independent, so LLVM can vectorise the
        # divisions), then the running sum in topic order
        for k in range(n_topics):
            dist_sum[k] = (nwz[w, k] + eta) / (nz[k] + eta_sum) * (ndz[d, k] + alpha)
        dist_cum = 0.0
        for k in range(n_topics):
            dist_cum += dist_sum[k]
            dist_sum[k] = dist_cum

        r = rands[j] * dist_cum            # rands[i % n_rand]
        j += 1
        if j == n_rand:
            j = 0
        # lda's bisection returns the first k with dist_sum[k] >= r; the
        # running sums increase strictly, so that is the number of sums below r
        # (counted without branches)
        z_new = 0
        for k in range(n_topics):
            z_new += dist_sum[k] < r

        ZS[i] = z_new
        nwz[w, z_new] += 1
        ndz[d, z_new] += 1
        nz[z_new] += 1


@njit(cache=True, nogil=True)
def _sequential_sum(value, n):
    """``for i in range(n): s += value`` (the ``eta_sum`` loop of ``lda``)."""
    s = 0.0
    for _ in range(n):
        s += value
    return s


def _matrix_to_lists(doc_word: sp.csr_matrix):
    """``lda.utils.matrix_to_lists``: tokens in row-major order, columns sorted."""
    doc_word = sp.csr_matrix(doc_word)
    doc_word.sort_indices()
    counts = doc_word.data.astype(np.int64)
    if np.any(counts < 0):
        raise ValueError("negative counts in the document-term matrix")
    rows = np.repeat(np.arange(doc_word.shape[0]), np.diff(doc_word.indptr))
    keep = counts > 0
    DS = np.repeat(rows[keep], counts[keep]).astype(np.int32)
    WS = np.repeat(doc_word.indices[keep], counts[keep]).astype(np.int32)
    return WS, DS


def _lda_rands(random_state: int) -> np.ndarray:
    return np.random.RandomState(random_state).rand(_N_RANDS)


def _fit_cgs(doc_word: sp.csr_matrix, n_topics: int, n_iter: int, alpha: float,
             eta: float, random_state: int):
    """``lda.LDA(n_topics, n_iter, alpha, eta, random_state).fit(doc_word)``.

    Returns the count matrices ``nzw`` (topics × words), ``ndz`` (docs × topics)
    and ``nz`` in ``lda``'s layout.
    """
    D, W = doc_word.shape
    rands = _lda_rands(random_state)        # LDA.__init__
    rng = np.random.RandomState(random_state)  # LDA._fit
    rands = rands.copy()
    WS, DS = _matrix_to_lists(doc_word)
    nwz = np.zeros((W, n_topics), dtype=np.int32)
    ndz = np.zeros((D, n_topics), dtype=np.int32)
    nz = np.zeros(n_topics, dtype=np.int32)
    ZS = _initialize(WS, DS, n_topics, nwz, ndz, nz)
    eta_sum = _sequential_sum(float(eta), W)
    dist_sum = np.empty(n_topics, dtype=np.float64)
    for _ in range(n_iter):
        rng.shuffle(rands)
        _sample_topics(WS, DS, ZS, nwz, ndz, nz, float(alpha), float(eta), eta_sum,
                       rands, dist_sum)
    return np.ascontiguousarray(nwz.T), ndz, nz


# ---------------------------------------------------------------------------
# Model-quality metrics (tmtoolkit 0.12 / pycisTopic)
# ---------------------------------------------------------------------------

def _loglikelihood(nzw: np.ndarray, ndz: np.ndarray, alpha: float, eta: float) -> float:
    """``pycisTopic.utils.loglikelihood``, reproduced as written.

    The upstream function assigns (``=``) instead of accumulating inside both
    loops, so only the last non-zero count of every topic and of the last
    document survive. It is reproduced as is, because the value feeds the
    automatic model selection.
    """
    D, n_topics = ndz.shape
    vocab_size = nzw.shape[1]
    const_prior = (n_topics * math.lgamma(alpha) - math.lgamma(alpha * n_topics)) * D
    const_ll = (vocab_size * math.lgamma(eta) - math.lgamma(eta * vocab_size)) * n_topics

    topic_ll = 0
    for k in range(n_topics):
        row = nzw[k]
        nzi = np.flatnonzero(row > 0)
        if nzi.size:
            topic_ll = math.lgamma(int(row[nzi[-1]]) + eta)
            # ``sum += nzw[k, w]`` in a Python float, in word order
            s = float(np.cumsum(np.concatenate(([eta * vocab_size],
                                                 row[nzi].astype(np.float64))))[-1])
        else:
            s = eta * vocab_size
        topic_ll -= math.lgamma(s)

    doc_ll = 0
    for d in range(D):
        row = ndz[d]
        s = alpha * n_topics
        for k in range(n_topics):
            if row[k] > 0:
                doc_ll = math.lgamma(int(row[k]) + alpha)
                s += int(row[k])
        doc_ll -= math.lgamma(s)
    return doc_ll - const_prior + topic_ll - const_ll


def _single_thread_blas():
    """One BLAS thread: LAPACK ``dgesdd`` results depend on the thread count."""
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=1, user_api="blas")
    except ImportError:  # pragma: no cover - threadpoolctl ships with scikit-learn
        import contextlib

        return contextlib.nullcontext()


def _metric_arun_2010(topic_word, doc_topic, doc_lengths, compat="numpy126") -> float:
    with _single_thread_blas():
        cm1 = np.linalg.svd(topic_word, compute_uv=False)
        doc_lengths = np.asarray(doc_lengths).flatten()
        cm2 = doc_lengths @ doc_topic
    cm1 /= np.sum(cm1)
    cm2 = -np.sort(-cm2)
    cm2 /= np.sum(cm2)
    return float(np.sum(cm1 * (log_compat(cm1 / cm2, compat)))
                 + np.sum(cm2 * (log_compat(cm2 / cm1, compat))))


def _metric_cao_juan_2009(topic_word) -> float:
    from scipy.spatial.distance import pdist

    return float(np.mean(1 - pdist(topic_word, metric="cosine")))


def _top_words(topic_word: np.ndarray, top_n: int, compat: str = "numpy126") -> list:
    """``tmtoolkit.topicmod.model_stats.top_words_for_topics`` (word indices)."""
    n_vocab = topic_word.shape[1]
    sl = slice(None, -(top_n + 1), -1) if top_n < n_vocab else slice(None)
    return [argsort_compat(t, compat)[sl] for t in topic_word]


def _metric_coherence_mimno_2011(topic_word, dtm_csc: sp.csc_matrix, top_n=20, eps=1e-12,
                                 normalize=True, compat="numpy126") -> np.ndarray:
    """``metric_coherence_mimno_2011(..., include_prob=False, return_mean=False)``."""
    top_words = _top_words(topic_word, top_n, compat)
    coh = []
    for v in top_words:
        top = (dtm_csc[:, v] >= 1).astype(np.int64)
        df = np.asarray(top.sum(axis=0)).ravel()
        codf = (top.T @ top).toarray()
        ratios = np.array([(codf[m, l] + eps) / df[l]
                           for m in range(1, top_n) for l in range(m)])
        c_t = 0
        for v_log in log_compat(ratios, compat):  # accumulated in pair order
            c_t += v_log
        coh.append(c_t)
    coh = np.array(coh)
    if normalize:
        coh *= 2 / (top_n * (top_n - 1))
    return coh


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class TopicModel:
    """A fitted cisTopic LDA model (mirror of ``pycisTopic.CistopicLDAModel``).

    Attributes
    ----------
    n_topic
        Number of topics.
    cell_topic
        Topics × cells probabilities (``Topic1`` … rows).
    topic_region
        Regions × topics probabilities.
    metrics
        One-row table with ``Arun_2010``, ``Cao_Juan_2009``, ``Mimno_2011`` and
        ``loglikelihood``.
    coherence, marg_topic, topic_ass
        Per-topic Mimno 2011 coherence, marginal topic distribution and the
        number of tokens assigned to each topic.
    parameters
        The fitting parameters (``package``, ``n_topics``, ``n_iter``, ...).
    extra
        Final Gibbs count matrices: ``nzw`` (topics × regions) and ``ndz``
        (cells × topics).
    """

    n_topic: int
    cell_topic: pd.DataFrame
    topic_region: pd.DataFrame
    metrics: pd.DataFrame
    coherence: pd.DataFrame
    marg_topic: pd.DataFrame
    topic_ass: pd.DataFrame
    parameters: pd.DataFrame
    extra: dict = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        return (f"TopicModel(n_topic={self.n_topic}, cells={self.cell_topic.shape[1]}, "
                f"regions={self.topic_region.shape[0]})")


def _binary_cells_by_regions(adata, layer: Optional[str]) -> sp.csr_matrix:
    X = adata.X if layer is None else adata.layers[layer]
    X = sp.csr_matrix(X)
    X = (X > 0).astype(np.int32)
    X.eliminate_zeros()
    return sp.csr_matrix(X)


def _fit_one(binary: sp.csr_matrix, n_topics: int, cell_names, region_names, n_iter: int,
             random_state: int, alpha: float, alpha_by_topic: bool, eta: float,
             eta_by_topic: bool, top_topics_coh: int, compat: str,
             verbose: bool) -> TopicModel:
    a = alpha / n_topics if alpha_by_topic else alpha
    e = eta / n_topics if eta_by_topic else eta
    _console(f"running model with {n_topics} topics", verbose)
    t0 = time.time()
    nzw, ndz, nz = _fit_cgs(binary, n_topics, n_iter, a, e, random_state)
    elapsed = time.time() - t0
    model = _build_model(binary, nzw, ndz, nz, cell_names, region_names, n_iter, random_state,
                         alpha, alpha_by_topic, eta, eta_by_topic, top_topics_coh, compat, elapsed)
    _console(f"model with {n_topics} topics done ({elapsed:.1f} s)", verbose)
    return model


def _build_model(binary, nzw, ndz, nz, cell_names, region_names, n_iter, random_state, alpha,
                 alpha_by_topic, eta, eta_by_topic, top_topics_coh, compat, elapsed) -> TopicModel:
    """Distributions, metrics and tables of a fitted model (``run_cgs_model`` after ``fit``)."""
    n_topics = nz.shape[0]
    a = alpha / n_topics if alpha_by_topic else alpha
    e = eta / n_topics if eta_by_topic else eta
    topic_word = (nzw + e).astype(float)
    topic_word /= rowsum_compat(topic_word, compat)[:, np.newaxis]
    doc_topic = (ndz + a).astype(float)
    doc_topic /= np.sum(doc_topic, axis=1)[:, np.newaxis]

    doc_lengths = np.asarray(binary.sum(axis=1)).astype(float)
    arun = _metric_arun_2010(topic_word, doc_topic, doc_lengths, compat)
    cao = _metric_cao_juan_2009(topic_word)
    mimno = _metric_coherence_mimno_2011(topic_word, binary.tocsc(), top_n=20, eps=1e-12,
                                         normalize=True, compat=compat)
    # upstream passes the undivided alpha / eta to its log-likelihood
    ll = _loglikelihood(nzw, ndz, alpha, eta)
    if len(mimno) <= top_topics_coh:
        mimno_model = np.mean(mimno)
    else:
        part = (argpartition_np126(mimno, len(mimno) - top_topics_coh) if compat == "numpy126"
                else np.argpartition(mimno, -top_topics_coh))
        mimno_model = np.mean(mimno[part[-top_topics_coh:]])
    metrics = pd.DataFrame([arun, cao, mimno_model, ll],
                           index=["Arun_2010", "Cao_Juan_2009", "Mimno_2011", "loglikelihood"],
                           columns=["Metric"]).transpose()
    topics = range(1, n_topics + 1)
    coherence = pd.DataFrame([topics, mimno], index=["Topic", "Mimno_2011"]).transpose()
    # tmtoolkit.marginal_topic_distrib with an np.matrix of document lengths
    with _single_thread_blas():
        unnorm = np.asarray(doc_topic.T @ doc_lengths).ravel()
    marg = unnorm / unnorm.sum()
    marg_topic = pd.DataFrame([topics, marg.tolist()], index=["Topic", "Marg_Topic"]).transpose()
    topic_ass = pd.DataFrame.from_records([topics, nz], index=["Topic", "Assignments"]).transpose()
    names = ["Topic" + str(i) for i in topics]
    cell_topic = pd.DataFrame(doc_topic, index=cell_names, columns=names).transpose()
    topic_region = pd.DataFrame(topic_word, index=names, columns=region_names).transpose()
    parameters = pd.DataFrame(
        ["lda", n_topics, n_iter, random_state, alpha, alpha_by_topic, eta, eta_by_topic,
         top_topics_coh, elapsed],
        index=["package", "n_topics", "n_iter", "random_state", "alpha", "alpha_by_topic",
               "eta", "eta_by_topic", "top_topics_coh", "time"],
        columns=["Parameter"],
    )
    return TopicModel(n_topics, cell_topic, topic_region, metrics, coherence, marg_topic,
                      topic_ass, parameters, extra={"nzw": nzw, "ndz": ndz})


def run_topic_models(
    adata,
    n_topics: Sequence[int] = (10, 20, 30, 40),
    *,
    n_iter: int = 150,
    random_state: int = 555,
    alpha: float = 50,
    alpha_by_topic: bool = True,
    eta: float = 0.1,
    eta_by_topic: bool = False,
    top_topics_coh: int = 5,
    layer: Optional[str] = None,
    n_jobs: Optional[int] = None,
    compat: str = "numpy126",
    verbose: bool = True,
) -> list:
    """Fit cisTopic LDA models with collapsed Gibbs sampling.

    Port of ``pycisTopic.lda_models.run_cgs_models``. Every model is fitted on
    the binarized cell × region matrix (``adata.X`` or ``adata.layers[layer]``,
    any value > 0 counts as accessible); with the same ``random_state`` the
    result is identical to pycisTopic's.

    Parameters
    ----------
    adata
        Cells × regions AnnData (e.g. from :func:`cistopic_matrix`).
    n_topics
        Number of topics of each model.
    n_iter
        Gibbs sweeps per model.
    random_state
        Seed of the ``lda`` random stream.
    alpha, alpha_by_topic
        Dirichlet prior on topic proportions; divided by the number of topics
        when ``alpha_by_topic`` (pycisTopic default 50 / K).
    eta, eta_by_topic
        Dirichlet prior on topic-region distributions.
    top_topics_coh
        Number of most coherent topics averaged into the model coherence.
    layer
        Layer to use instead of ``adata.X``.
    n_jobs
        Models fitted concurrently (threads); defaults to one per model.
    compat
        pycisTopic's metrics call ``np.argsort`` (which ranks tied region
        probabilities in a build-dependent order) and ``np.log`` (whose last
        bit differs between numpy builds). ``"numpy126"`` (default) reproduces
        numpy 1.26 on AVX-512 CPUs, the SCENIC+ reference stack, on any
        machine; ``"native"`` uses the installed numpy.
    verbose
        Print progress.

    Returns
    -------
    list of :class:`TopicModel`, in the order of ``n_topics``.
    """
    n_topics = [int(k) for k in n_topics]
    binary = _binary_cells_by_regions(adata, layer)
    cell_names = list(map(str, adata.obs_names))
    region_names = list(map(str, adata.var_names))
    args = dict(cell_names=cell_names, region_names=region_names, n_iter=n_iter,
                random_state=random_state, alpha=alpha, alpha_by_topic=alpha_by_topic,
                eta=eta, eta_by_topic=eta_by_topic, top_topics_coh=top_topics_coh,
                compat=compat, verbose=verbose)
    n_jobs = len(n_topics) if n_jobs is None else max(1, int(n_jobs))
    if n_jobs == 1 or len(n_topics) == 1:
        return [_fit_one(binary, k, **args) for k in n_topics]
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_fit_one, binary, k, **args) for k in n_topics]
        return [f.result() for f in futures]


def evaluate_topic_models(
    models: Sequence[TopicModel],
    select_model: Optional[int] = None,
    metrics: Sequence[str] = ("Minmo_2011", "loglikelihood", "Cao_Juan_2009", "Arun_2010"),
    min_topics_coh: int = 5,
    return_metrics: bool = False,
):
    """Select a topic model; port of ``pycisTopic.lda_models.evaluate_models``.

    Each metric is rescaled to [0, 1] across models (``Arun_2010`` and
    ``Cao_Juan_2009`` inverted) and the model with the highest sum wins. The
    metric name ``"Minmo_2011"`` keeps pycisTopic's spelling.

    Returns
    -------
    The selected :class:`TopicModel`, or ``(model, table)`` with the rescaled
    metrics per number of topics when ``return_metrics``.
    """
    models = [models[i] for i in np.argsort([m.n_topic for m in models])]
    all_topics = sorted(m.n_topic for m in models)
    metrics_dict = {}
    if "Minmo_2011" in metrics:
        in_index = [i for i in range(len(all_topics)) if all_topics[i] >= min_topics_coh]

    def _sub(values, idx):
        return [values[i] for i in idx]

    def _get(name):
        return [models[i].metrics.loc["Metric", name] for i in range(len(all_topics))]

    for name in ("Arun_2010", "Cao_Juan_2009"):
        if name in metrics:
            neg = [-x for x in _get(name)]
            rescale = (neg - min(neg)) / (max(neg) - min(neg))
            metrics_dict[name] = (np.array(_sub(rescale, in_index))
                                  if "Minmo_2011" in metrics else rescale)
    if "Minmo_2011" in metrics:
        mimno = _sub(_get("Mimno_2011"), in_index)
        mimno_topics = _sub(all_topics, in_index)
        metrics_dict["Minmo_2011"] = np.array((mimno - min(mimno)) / (max(mimno) - min(mimno)))
    if "loglikelihood" in metrics:
        ll = _get("loglikelihood")
        rescale = (ll - min(ll)) / (max(ll) - min(ll))
        metrics_dict["loglikelihood"] = (np.array(_sub(rescale, in_index))
                                         if "Minmo_2011" in metrics else rescale)
    if select_model is None:
        combined = sum(metrics_dict.values())
        pool = mimno_topics if "Minmo_2011" in metrics else all_topics
        best = pool[combined.tolist().index(max(combined))]
    else:
        combined = None
        best = select_model
    model = models[all_topics.index(best)]
    if not return_metrics:
        return model
    index = mimno_topics if "Minmo_2011" in metrics else all_topics
    table = pd.DataFrame({k: np.asarray(v) for k, v in metrics_dict.items()}, index=index)
    table.index.name = "n_topics"
    if combined is not None:
        table["combined"] = np.asarray(combined)
    return model, table


def add_topic_model(adata, model: TopicModel, key: str = "topic") -> None:
    """Store a topic model in ``adata`` (cells × regions).

    Writes ``adata.obsm[f"X_{key}"]`` (cells × topics), ``adata.varm[key]``
    (regions × topics) and the metrics / parameters to ``adata.uns[key]``.
    Cells and regions are aligned by name.
    """
    ct = model.cell_topic.T.reindex(list(map(str, adata.obs_names)))
    tr = model.topic_region.reindex(list(map(str, adata.var_names)))
    adata.obsm[f"X_{key}"] = ct.to_numpy()
    adata.varm[key] = tr.to_numpy()
    adata.uns[key] = {
        "topics": list(model.cell_topic.index),
        "metrics": model.metrics.iloc[0].to_dict(),
        "coherence": model.coherence["Mimno_2011"].to_numpy(),
        "assignments": model.topic_ass["Assignments"].to_numpy().astype(np.int64),
        "parameters": {k: str(v) for k, v in model.parameters["Parameter"].items()},
    }
