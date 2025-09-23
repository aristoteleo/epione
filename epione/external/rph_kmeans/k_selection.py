"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import os

import numpy as np
import scipy
from kneed import KneeLocator
from multiprocessing import Pool, cpu_count
from sklearn.cluster import KMeans
import warnings
from tqdm import tqdm
from typing import Optional, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed


def _derive_seed(base: Optional[int], a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> Optional[int]:
    if base is None:
        return None
    # Simple, deterministic combination within 32-bit range
    mod = 2**31 - 1
    val = (int(base) + 1009 * int(a) + 9176 * int(b) + 13331 * int(c) + 29791 * int(d)) % mod
    return int(val)

def get_point_reducer(point_reducer_version='cy'):
	if point_reducer_version == 'cy':
		try:
			from .point_reducer_cy import RPPointReducerCy
			RPPointReducer = RPPointReducerCy
		except:
			warnings.warn('The cython version of rph-kmeans is not installed properly. Use python version instead.')
			from .point_reducer_py import RPPointReducerPy
			RPPointReducer = RPPointReducerPy
	else:
		from .point_reducer_py import RPPointReducerPy
		RPPointReducer = RPPointReducerPy
	return RPPointReducer


def cal_inertia(X, y_pred, centers, weight):
    """Vectorized weighted inertia.
    Args:
        X (ndarray): (n_samples, n_features)
        y_pred (ndarray): (n_samples,)
        centers (ndarray): (n_clusters, n_features)
        weight (ndarray): (n_samples,)
    Returns:
        float: weighted sum of squared distances to assigned centers
    """
    assigned_centers = centers[y_pred]
    diff = X - assigned_centers
    # sum over features → (n_samples,)
    sq = np.einsum('ij,ij->i', diff, diff)
    return float(np.dot(sq, weight))


def cal_cluster_variance(X, y_pred, centers, weight):
    denom = (X.shape[0] - centers.shape[0]) * X.shape[1]
    # Invalid/degenerate: not enough points per parameters
    if denom <= 0:
        return np.nan
    return cal_inertia(X, y_pred, centers, weight) / denom


def cal_log_likelihood(X, y_pred, centers, weight, eps=1e-100):
    """Vectorized log-likelihood for spherical k-means model with weights."""
    n_clusters = centers.shape[0]
    variance = cal_cluster_variance(X, y_pred, centers, weight)
    if not np.isfinite(variance) or variance <= 0:
        return float(-np.inf)
    total_weight = weight.sum()
    # weighted cluster sizes (m_i)
    # Use bincount with weights for speed and numerical stability
    m = np.bincount(y_pred, weights=weight, minlength=n_clusters)
    # Guard zeros to avoid log(0)
    m_safe = np.maximum(m, eps)
    D = X.shape[1]
    term1 = np.sum(m_safe * (np.log(m_safe) - np.log(total_weight)))
    term2 = 0.5 * D * np.sum(m * np.log(2.0 * np.pi * variance))
    term3 = 0.5 * D * np.sum(m - 1.0)
    return float(term1 - term2 - term3)


def cal_bic(X, y_pred, centers, weight=None, weight_norm=False):
	"""Bayesian information criterion;
		Ref: Pelleg, Dan et.al., "X-means: Extending k-means with efficient estimation of the number of clusters.", 2000.
		Ref: 《Notes on Bayesian Information Criterion Calculation for X-Means Clustering》
		Ref: https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
		Ref: https://en.wikipedia.org/wiki/Bayesian_information_criterion#cite_note-Priestley-5
	Args:
		X (np.ndarray): (n_samples, n_features)
		y_pred (np.ndarray): (n_samples,)
		centers (np.ndarray): (k, n_features)
	Returns:
		float: BIC score (the higher the better)
	"""
	if weight is None:
		weight = np.ones(X.shape[0], dtype=X.dtype)
	if weight_norm:
		weight = weight * (weight.shape[0] / weight.sum())
	para_num = centers.shape[0] * (X.shape[1] + 1)
	return cal_log_likelihood(X, y_pred, centers, weight) - 0.5 * para_num * np.log(X.shape[0])


def cal_aic(X, y_pred, centers, weight=None):
	"""Ref: https://en.wikipedia.org/wiki/Akaike_information_criterion
	"""
	if weight is None:
		weight = np.ones(X.shape[0], dtype=X.dtype)
	para_num = centers.shape[0] * (X.shape[1] + 1)
	return cal_log_likelihood(X, y_pred, centers, weight) - para_num


def get_bic_optimal_k_wrapper(args):
	"""
	Args:
		args (list):
			X (np.ndarray or scipy.sparse.csr_matrix): (n_samples, n_features)
			k_range (list)
			k_repeat (int): Times to run kmeans for each k to calculate averaged bic score
			point_reducer_initializer (RPPointReducerBase)
			point_reducer_kwargs (dict): kwargs for point_reducer_initializer
	Returns:
		list: bic scores; length == len(k_range)
	"""
	#scipy.random.seed()
	# Backward/forward compatible unpacking
	# Formats:
	# (X, k_range, k_repeat, init, kwargs)
	# (X, k_range, k_repeat, init, kwargs, inner_n_jobs)
	# (X, k_range, k_repeat, init, kwargs, inner_n_jobs, base_seed, ske_idx)
	if len(args) == 5:
		X, k_range, k_repeat, point_reducer_initializer, point_reducer_kwargs = args
		inner_n_jobs = 1; base_seed = None; ske_idx = 0
	elif len(args) == 6:
		X, k_range, k_repeat, point_reducer_initializer, point_reducer_kwargs, inner_n_jobs = args
		base_seed = None; ske_idx = 0
	elif len(args) == 8:
		X, k_range, k_repeat, point_reducer_initializer, point_reducer_kwargs, inner_n_jobs, base_seed, ske_idx = args
	else:
		raise ValueError('Unexpected args length for get_bic_optimal_k_wrapper')
	pr = point_reducer_initializer(**point_reducer_kwargs)
	# Set deterministic random_state for point reducer if provided
	if hasattr(pr, 'random_state') and pr.random_state is None:
		try:
			pr.random_state = _derive_seed(base_seed, ske_idx, 0, 0, 0)
			# reinitialize rng if available
			pr._rng = np.random.default_rng(pr.random_state)
		except Exception:
			pass
	X, weight = pr.fit_transform(X)[:-2]    # get skeleton

	def bic_for_k(k: int) -> float:
		bic_k_list = []
		for repeat_id in range(k_repeat):
			while True:
				try:
					seed = _derive_seed(base_seed, ske_idx, k, repeat_id)
					clt = KMeans(n_clusters=k, random_state=seed)
					y_pred = clt.fit_predict(X, sample_weight=weight)
				except IndexError:
					# rare sklearn bug; retry
					continue
				else:
					break
			bic_k_list.append(cal_bic(X, y_pred, clt.cluster_centers_, weight))
		return float(np.mean(bic_k_list))

	# Optional inner threading across ks to speed up per-skeleton evaluation
	if inner_n_jobs and inner_n_jobs > 1:
		results = [None] * len(k_range)
		with ThreadPoolExecutor(max_workers=inner_n_jobs) as ex:
			future_to_idx = {ex.submit(bic_for_k, k): i for i, k in enumerate(k_range)}
			for fut in as_completed(future_to_idx):
				idx = future_to_idx[fut]
				results[idx] = fut.result()
		return results
	else:
		return [bic_for_k(k) for k in k_range]


def select_k_with_bic(
		X,
		kmax,
		kmin=2,
		ske_repeat=30,
		k_repeat=5,
		kneedle_s=3.0,
		point_reducer_version='cy',
		point_reducer_kwargs=None,
		n_jobs=-1,
		inner_n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
):
	"""The bic score will be calculated for each k in range(kmin, kmax+1). And the optimal k will be the knee point of k-bic curve.
	Args:
		X (np.ndarray or scipy.sparse.csr_matrix):
		kmax (int)
		kmin (int): default: 2
		ske_repeat (int): default: 40
			Running times to generate skeleton
		k_repeat (int): default: 5
			Running times to calculate bic for each k in range(kmin, kmax+1) for every ske_repeat.
		kneedle_s (float): default: 5.0
			Sensitivity of Kneedle algorithm, S.
			See Satopaa, Ville, et al., "Finding a "kneedle" in a haystack: Detecting knee points in system behavior", 2011.
		point_reducer_version (str): {'cy', 'py'}, default: 'cy'
			Version of point reducer module. 'cy' is for cython version and 'py' is for python version.
			If cython version is failed to import, python version will be used instead.
		point_reducer_kwargs (dict or None): default: None
			kwargs for point reducer.
		cpu_use (int): default: -1
			The number of jobs to use for the computation. This works by computing each of the ske_repeat runs in parallel.
			-1 means using all processors.
		inner_n_jobs (int or None): optional inner threading across different k values per skeleton. Defaults to min(len(k_range), 4).
		random_state (int or None): base seed for reproducible skeleton generation and KMeans runs. Different seeds are derived per ske_repeat/k/repeat.
	Returns:
		int: optimal k
		list: [bic_list(1), ..., bic_list(ske_repeat)]; bic_list(i) = [bic_score(kmin), bic_score(kmin+1), ..., bic_score(kmax)];
			bic_score(k) = np.mean([bic_score for k in range(k_repeat)]).
		list: [kmin, kmin+1, ..., kmax]
	"""
	cpu_use = cpu_count() if n_jobs == -1 else n_jobs
	point_reducer_initializer = get_point_reducer(point_reducer_version)
	k_range = list(range(kmin, kmax+1))
	point_reducer_kwargs = point_reducer_kwargs or {}
	# inner_n_jobs: if None, choose a reasonable default (min(len(k_range), 4)) to avoid oversubscription.
	if inner_n_jobs is None:
		inner_n_jobs_eff = min(len(k_range), 4)
	else:
		inner_n_jobs_eff = max(1, inner_n_jobs)
	with Pool(cpu_use) as pool:
		args_list = [
			(X, k_range, k_repeat, point_reducer_initializer, point_reducer_kwargs, inner_n_jobs_eff, random_state, i)
			for i in range(ske_repeat)
		]
		bic_lists = []
		for bic_list in tqdm(pool.imap_unordered(get_bic_optimal_k_wrapper, args_list), total=len(args_list), leave=False):
			bic_lists.append(bic_list)
	k_list = []
	s_range = ([] if int(kneedle_s) == kneedle_s else [kneedle_s]) + list(range(int(kneedle_s), 0, -1))
	for bic_list in bic_lists:
		predict_k = None
		for s in s_range:
			kl = KneeLocator(k_range, bic_list, curve='concave', direction='increasing', S=s)
			if kl.knee is not None:
				predict_k = kl.knee
				break
		assert predict_k is not None
		k_list.append(predict_k)
	optimal_k = int(round(np.mean(k_list)))
	return optimal_k, bic_lists, k_range


def plot_bic(bic_lists: Sequence[Sequence[float]], k_range: Sequence[int], optimal_k: Optional[int] = None,
		show_mean: bool = True, ax=None, figsize=(6, 4), alpha=0.25, linewidth=1.2,
		color='C0', mean_color='C1', title='BIC vs k', savepath: Optional[str] = None):
	"""Visualize BIC curves over k, across skeleton repeats.

	Args:
		bic_lists: list of length ske_repeat; each is list of BIC over k_range.
		k_range: iterable of k values corresponding to bic_lists entries.
		optimal_k: if provided, draws a vertical line at this k.
		show_mean: also plot the mean BIC curve across repeats.
		ax: optional matplotlib Axes to draw on; if None, creates one.
		figsize: figure size if creating new figure.
		alpha: alpha for per-repeat curves.
		linewidth: line width for curves.
		color: color for per-repeat curves.
		mean_color: color for mean curve.
		title: plot title.
		savepath: optional path to save the figure.
	Returns:
		matplotlib Axes with the plot.
	"""
	import matplotlib.pyplot as plt
	if ax is None:
		fig, ax = plt.subplots(1, 1, figsize=figsize)
	else:
		fig = ax.get_figure()
	# Plot each repeat
	for bic in bic_lists:
		ax.plot(k_range, bic, color=color, alpha=alpha, linewidth=linewidth)
        # Plot mean
	if show_mean:
		mean_bic = np.mean(np.asarray(bic_lists), axis=0)
		ax.plot(k_range, mean_bic, color=mean_color, linewidth=2.0, label='mean BIC')
	# Optimal k marker
	if optimal_k is not None:
		ax.axvline(optimal_k, color='k', linestyle='--', linewidth=1.2, label=f'k* = {optimal_k}')
	ax.set_xlabel('k')
	ax.set_ylabel('BIC (higher is better)')
	ax.set_title(title)
	ax.grid(True, alpha=0.3)
	if show_mean or optimal_k is not None:
		ax.legend(frameon=False)
	if savepath:
		fig.savefig(savepath, bbox_inches='tight', dpi=150)
	return ax
