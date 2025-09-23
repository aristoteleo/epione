"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import warnings
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.utils import check_array

class RPHKMeans(object):
	def __init__(self, n_clusters=8, n_init=1, point_reducer_version='cy',
			w=None, max_point=2000, proj_num=5, max_iter=1000, sample_dist_num=1000,
			bkt_improve=None, radius_divide=None, bkt_size_keepr=1.0, center_dist_keepr=1.0,
			reduced_kmeans_kwargs=None, final_kmeans_kwargs=None, verbose=1,
			min_point=None, allow_fewer_clusters=False, merge_centers_tol=0.0, min_cluster_size=None,
			random_state=None):
		"""
		Args:
			n_clusters (int): default: 8
				The number of clusters to form as well as the number of centroids to generate.
			n_init (int): default: 1
				Number of time the rp k-means algorithm will be run. The final results will be
				the best output of n_init consecutive runs in terms of inertia.
			point_reducer_version (str): {'cy', 'py'}, default: 'cy'
				Version of point reducer module. 'cy' is for cython version and 'py' is for python version.
				If cython version is failed to import, python version will be used instead.
			w (float): default: None
				Width of bucket for random projection process. If set to None, w will be the half of
				the median of distances of paired samples from input X.
			max_point (int): default: 2000
				Algorithm will stop when reduced point number is less than max_point. If max_point is larger than
				or equal to sample number of X, the process of point reduction won't run.
			proj_num (int): default: 5
				Number of vector for random projection process.
			max_iter (int): default: 1000
				Maximum number of iterations of the algorithm to run.
			sample_dist_num (int): default: 1000
				Number of paired samples chosen to decide default w. It will be ignored when w is set by user.
			bkt_improve (str or None): {None, 'radius', 'min_bkt_size', 'min_center_dist'}
				Methods of improving bucket quality.
			radius_divide (float): default: None
				Radius for 'radius' bucket-improving method. If set to None, no bucket-improving process
				will be ran even though bkt_improve is set to 'radius'. See paper or code for details.
			bkt_size_keepr (float): default: 0.8
				Keep ratio for 'min_bkt_size' bucket-improving method. If set to 1.0, no bucket-improving
				process will be ran even though bkt_improve is set to 'min_bkt_size'. See paper or code for details.
			center_dist_keepr (float): default: 0.8
				Keep ratio for 'min_center_dist' bucket-improving method. If set to 1.0, no bucket-improving
				process will be ran even though bkt_improve is set to 'min_center_dist'. See paper or code for details.
			min_point (int or None): default: None
				Optional lower bound on the number of reduced points (skeleton size). If the next
				reduction step would produce fewer than this number, reduction stops before applying.
			reduced_kmeans_kwargs (dict): default: None
				kwargs of kmeans to find centers of reduced point. If set to None, default kwargs will be used.
			final_kmeans_kwargs (dict): default: None
				kwargs of kmeans after center initialization. If set to None, default kwargs will be used.
			verbose (int): {0, 1, 2}, default: 1
				Controls the verbosity. Print nothing when set to 0 and print most details when set to 2.

		Attributes:
			cluster_centers_ (np.ndarray): (n_clusters, n_features)
				Coordinates of cluster centers.
			labels_ (np.ndarray): (n_samples)
				Labels of each point
			inertia_ (float):
				Sum of squared distances of samples to their closest cluster center.
			n_iter_ (int):
				Number of kmeans's iterations run after center initialization.

			reduced_X_ (np.ndarray): (n_reduced_point, n_features)
				Reduced points.
			reduced_X_weight_ (np.ndarray): (n_reduced_point,)
				Weight of reduced points. reduced_X_weight_[i] represents the number of original points merged into
				reduced_X_[i].
			rp_labels_ (np.ndarray): (n_samples,)
				Labels of each original point indicating which reduced point it belongs to.
				Specifically， X[i] belongs to reduced_X_[rp_labels_[i]].
			init_centers_ (np.ndarray): (n_clusters, n_features)
				Initial centers.
			rp_iter_ (int):
				Number of iteration of point reduction process.

			allow_fewer_clusters (bool): default: False
				If True, when skeleton size < n_clusters, use the smaller number instead of raising,
				so the final output cluster count can be less than requested n_clusters.
			merge_centers_tol (float): default: 0.0 (disabled)
				If > 0, merge near-duplicate initial centers (Euclidean distance <= tol)
				before final KMeans, potentially reducing the effective number of clusters.
			min_cluster_size (int or None): default: None
				If set, enforce minimal cluster size after final KMeans by merging
				smaller clusters into their nearest neighbor clusters.
			random_state (int or None): default: None
				Global random seed to make projection, sampling and KMeans reproducible.
		"""
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
		self.point_reducer = RPPointReducer(w, max_point, proj_num, max_iter, sample_dist_num,
			bkt_improve, radius_divide, bkt_size_keepr, center_dist_keepr, verbose, min_point, random_state=random_state)
		self.n_clusters = n_clusters
		self.n_init = n_init
		self.reduced_kmeans_kwargs = deepcopy(reduced_kmeans_kwargs) if reduced_kmeans_kwargs is not None else {}
		if 'n_clusters' not in self.reduced_kmeans_kwargs:
			self.reduced_kmeans_kwargs['n_clusters'] = n_clusters
		self.final_kmeans_kwargs = deepcopy(final_kmeans_kwargs) if final_kmeans_kwargs is not None else {}
		if 'n_clusters' not in self.final_kmeans_kwargs:
			self.final_kmeans_kwargs['n_clusters'] = n_clusters
		self.final_kmeans_kwargs['n_init'] = 1

		self.reduced_X_ = None
		self.reduced_X_weight_ = None
		self.rp_labels_ = None
		self.rp_iter_ = None
		self.init_centers_ = None
		self.rp_inertia_ = None

		self.final_kmeans_clt = None
		self.inertia_ = None
		self.cluster_centers_ = None
		self.labels_ = None
		self.n_iter_ = None
		self.n_clusters_effective_ = None
		self.allow_fewer_clusters = allow_fewer_clusters
		self.merge_centers_tol = merge_centers_tol
		self.min_cluster_size = min_cluster_size
		self.random_state = random_state


	def init_centers(self, X):
		reduced_X, group_weight, labels, rp_iter = self.point_reducer.fit_transform(X)
		assert len(reduced_X) == len(group_weight)
		if len(reduced_X) < self.n_clusters:
			if not self.allow_fewer_clusters:
				raise RuntimeError('Number of reduced points is too small, please try smaller w or larger proj_num')
			k_reduced = max(1, len(reduced_X))
		else:
			k_reduced = self.n_clusters
		reduced_kmeans_kwargs = deepcopy(self.reduced_kmeans_kwargs)
		reduced_kmeans_kwargs['n_clusters'] = k_reduced
		if 'random_state' not in reduced_kmeans_kwargs and self.random_state is not None:
			reduced_kmeans_kwargs['random_state'] = self.random_state
		reduced_clt = KMeans(**reduced_kmeans_kwargs)
		while True:
			try:
				y_pred = reduced_clt.fit_predict(reduced_X, sample_weight=group_weight)
			except IndexError:
				print('Index error raised, re-run kmeans for skeleton to initialize centers. This may due to bugs in sklearn.')
			else:
				break
		return reduced_clt.cluster_centers_, reduced_X, group_weight, labels, rp_iter, reduced_clt.inertia_, y_pred


	def fit_predict(self, X):
		"""
		Args:
			X (numpy.ndarray or scipy.sparse.csr_matrix): (n_samples, n_features)
		Returns:
			np.ndarray: (n_samples,)
		"""
		self.fit(X)
		return self.labels_


	def fit(self, X):
		"""
		Args:
			X (numpy.ndarray or scipy.sparse.csr_matrix): (n_samples, n_features)
			Training instances to cluster. It must be noted that the data will
			be converted to C ordering, which will cause a memory copy
			if the given data is not C-contiguous.
		"""
		self.inertia_ = np.inf
		pbar = tqdm(total=self.n_init, leave=True, desc='RPH n_init', disable=(getattr(self, 'verbose', 1) <= 0 or self.n_init <= 1))
		for i in range(self.n_init):
			init_centers_, reduced_X_, reduced_X_weight_, rp_labels_, rp_iter_, rp_inertia_, rp_y_pred_ = self.init_centers(X)
			centers_to_use = init_centers_
			# Optional merging of near-duplicate centers
			if self.allow_fewer_clusters and self.merge_centers_tol > 0.0 and centers_to_use.shape[0] > 1:
				keep = []
				for j in range(centers_to_use.shape[0]):
					c = centers_to_use[j]
					if len(keep) == 0:
						keep.append(c)
						continue
					dist = np.linalg.norm(np.stack(keep) - c, axis=1)
					if np.any(dist <= self.merge_centers_tol):
						continue
					keep.append(c)
				centers_to_use = np.stack(keep)
			final_k = max(1, centers_to_use.shape[0])
			final_kwargs = deepcopy(self.final_kmeans_kwargs)
			final_kwargs['n_clusters'] = final_k
			if 'random_state' not in final_kwargs and self.random_state is not None:
				final_kwargs['random_state'] = self.random_state
			clt = KMeans(init=centers_to_use, **final_kwargs)
			clt.fit(X)
			if clt.inertia_ < self.inertia_:
				self.inertia_ = clt.inertia_
				self.final_kmeans_clt = clt
				self.init_centers_, self.reduced_X_, self.reduced_X_weight_, self.rp_labels_, self.rp_iter_, self.rp_inertia_, self.rp_y_pred_ = \
					init_centers_, reduced_X_, reduced_X_weight_, rp_labels_, rp_iter_, rp_inertia_, rp_y_pred_
			if pbar is not None:
				pbar.update(1)
		self.cluster_centers_ = self.final_kmeans_clt.cluster_centers_
		self.labels_, self.n_iter_ = self.final_kmeans_clt.labels_, self.final_kmeans_clt.n_iter_
		self.n_clusters_effective_ = self.cluster_centers_.shape[0]
		if pbar is not None:
			pbar.close()

		# Enforce minimum cluster size if requested
		if self.min_cluster_size is not None and self.min_cluster_size > 1:
			self._enforce_min_cluster_size(X)

		return self

	def _compute_center(self, X_subset):
		if sp.issparse(X_subset):
			c = X_subset.mean(axis=0)
			try:
				return c.A1
			except AttributeError:
				return np.asarray(c).ravel()
		else:
			return X_subset.mean(axis=0)

	def _compute_inertia(self, X, labels, centers):
		inertia = 0.0
		k = centers.shape[0]
		for i in range(k):
			idx = np.where(labels == i)[0]
			if idx.size == 0:
				continue
			Xi = X[idx]
			c = centers[i]
			if sp.issparse(X):
				row_norms = Xi.multiply(Xi).sum(axis=1).A1
				c_norm2 = float(np.dot(c, c))
				dot = Xi @ c
				dot = np.asarray(dot).ravel() if hasattr(dot, 'A1') else np.array(dot).ravel()
				d2 = row_norms + c_norm2 - 2.0 * dot
			else:
				diff = Xi - c
				d2 = np.einsum('ij,ij->i', diff, diff)
			inertia += float(d2.sum())
		return inertia

	def _enforce_min_cluster_size(self, X):
		labels = self.labels_.copy()
		centers = self.cluster_centers_.copy()
		k = centers.shape[0]
		changed = False
		while True:
			counts = np.bincount(labels, minlength=k)
			small = np.where(counts < self.min_cluster_size)[0]
			if k <= 1 or small.size == 0:
				break
			# smallest violating cluster id
			s = small[np.argmin(counts[small])]
			# nearest neighbor cluster by center distance
			others = [idx for idx in range(k) if idx != s]
			dists = np.linalg.norm(centers[others] - centers[s], axis=1)
			target = others[int(np.argmin(dists))]
			# move all points from s to target
			labels[labels == s] = target
			changed = True
			# remove center s and renumber labels > s
			mask_keep = np.ones(k, dtype=bool)
			mask_keep[s] = False
			centers = centers[mask_keep]
			labels[labels > s] -= 1
			k -= 1
			# recompute centers after this merge
			for idx in range(k):
				pts_idx = np.where(labels == idx)[0]
				if pts_idx.size > 0:
					centers[idx] = self._compute_center(X[pts_idx])
		if changed:
			self.labels_ = labels
			self.cluster_centers_ = centers
			self.n_clusters_effective_ = centers.shape[0]
			self.inertia_ = self._compute_inertia(X, labels, centers)


	def predict(self, X):
		"""
		Args:
			X (numpy.ndarray or scipy.sparse.csr_matrix): (n_samples, n_features)
		Returns:
			np.ndarray: (n_samples,)
		"""
		X = check_array(X, accept_sparse="csr", order='C', dtype=[np.float64, np.float32])
		# If no size constraints changed the centers count, delegate to sklearn
		if (self.final_kmeans_clt is not None and
			self.final_kmeans_clt.cluster_centers_.shape[0] == self.cluster_centers_.shape[0] and
			self.min_cluster_size in (None, 0, 1)):
			return self.final_kmeans_clt.predict(X)
		# Otherwise, assign by nearest final centers we hold
		centers = self.cluster_centers_
		if sp.issparse(X):
			row_norms = X.multiply(X).sum(axis=1).A1
			center_norms = np.einsum('ij,ij->i', centers, centers)
			dots = X @ centers.T
			dots = dots.toarray() if hasattr(dots, 'toarray') else np.asarray(dots)
			d2 = row_norms[:, None] + center_norms[None, :] - 2.0 * dots
		else:
			row_norms = np.einsum('ij,ij->i', X, X)
			center_norms = np.einsum('ij,ij->i', centers, centers)
			dots = X @ centers.T
			d2 = row_norms[:, None] + center_norms[None, :] - 2.0 * dots
		return np.argmin(d2, axis=1)
