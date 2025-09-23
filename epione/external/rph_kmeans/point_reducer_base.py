"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import warnings
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import paired_distances
from sklearn.utils import check_array

from .utils import check_return




class RPPointReducerBase(object):
	def __init__(self, w=None, max_point=2000, proj_num=5, max_iter=1000, sample_dist_num=1000,
				bkt_improve=None, radius_divide=None, bkt_size_keepr=1.0, center_dist_keepr=1.0, verbose=False,
				min_point=None, random_state=None):
		self.w = w
		self.max_point = max_point
		self.min_point = min_point
		self.random_state = random_state
		self.proj_num = proj_num
		self.max_iter = max_iter
		self.sample_dist_num = int(sample_dist_num)
		self.bkt_improve = bkt_improve
		self.radius2 = radius_divide**2 if radius_divide is not None else None
		self.bkt_size_keepr=bkt_size_keepr
		self.center_dist_keepr = center_dist_keepr
		self.verbose=verbose
		self.sparse_x = None
		# Dedicated RNG for reproducibility
		try:
			self._rng = np.random.default_rng(random_state)
		except Exception:
			self._rng = np.random


	def fit_transform(self, X):
		raise NotImplementedError


	@check_return('sample_dist')
	def get_sample_dist(self, X):
		idx1 = self._rng.choice(X.shape[0], self.sample_dist_num)
		idx2 = self._rng.choice(X.shape[0], self.sample_dist_num)
		return paired_distances(
			X[idx1],
			X[idx2],
			metric='euclidean'
		)


	@check_return('w')
	def get_w(self, X):
		sample_dist = self.get_sample_dist(X)
		w = np.median(sample_dist) * 0.5
		return w


	def gen_proj(self, dim, w):
		b = self._rng.uniform(0, 1, size=(self.proj_num,))
		proj_vecs = self._rng.normal(0, 1.0/w, size=(dim, self.proj_num))  # (feature_size, projection_num)
		return proj_vecs, b


	def random_projection(self, features, proj_vecs, b):
		return (features.dot(proj_vecs) + b).astype(np.int32) # (sample_num, projection_num)


	def check_input_X(self, X):
		X = check_array(X, accept_sparse="csr", order='C', dtype=[np.float64, np.float32])
		self.sparse_x = sp.issparse(X)
		return X


	def check_max_point(self, max_point, X):
		if max_point >= X.shape[0]:
			warnings.warn("max_point is larger than sample number of input X. The process of point reduction won't run")
		if self.min_point is not None:
			if self.min_point < 1:
				warnings.warn("min_point < 1 is invalid; ignoring.")
				self.min_point = None
			elif self.min_point > max_point:
				warnings.warn("min_point is greater than max_point; reduction will stop earlier to respect min_point.")


	def split_group_orphan(self, buckets):
		"""
		Returns:
			list: group_buckets; [[point_idx, ...], ...]
			list: orphan_points; [point_idx, ...]
		"""
		group_buckets, orphan_points = [], []
		for bkt in buckets:
			if len(bkt) == 1:
				orphan_points.append(bkt[0])
			else:
				group_buckets.append(bkt)
		return group_buckets, orphan_points
