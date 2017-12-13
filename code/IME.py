import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils.graph import graph_shortest_path
from sklearn.decomposition import KernelPCA

class IME(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5, n_components=2, eigen_solver='auto',
                 tol=0, max_iter=None, path_method='auto',
                 neighbors_algorithm='auto', n_jobs=1):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.path_method = path_method
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs

    def fit_transform_xj(self, X, features_in,original_weight):
        # compute cov matrix
        self.feature_M=features_in
        M_cov = np.dot(features_in, features_in.T)
        M_cov=2-2*M_cov
        # t-distribution
        M_cov = 1 + M_cov
        M_cov = 1.0 / M_cov
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.neighbors_algorithm
                                      )
        self.nbrs_.fit(X)
        self.training_data_ = self.nbrs_._fit_X
        self.kernel_pca_ = KernelPCA(n_components=self.n_components,
                                     kernel="precomputed",
                                     eigen_solver=self.eigen_solver,
                                     tol=self.tol, max_iter=self.max_iter
                                     )
        kng = kneighbors_graph(self.nbrs_, self.n_neighbors,
                               mode='distance')
        # # A^2
        kng = np.dot(kng, kng)
        self.dist_matrix_ = graph_shortest_path(kng,
                                                method=self.path_method,
                                                directed=False)
        # similarity=1/(1+dist^2)    t-distribution n=1
        G = 1 + self.dist_matrix_ ** 2
        G = 1.0 / G
        G=G+M_cov*original_weight
        self.embedding_ = self.kernel_pca_.fit_transform(G)
        return self.embedding_


