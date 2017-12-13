#coding:utf-8 允许中文注释
import numpy as np
import time
from IME import IME
from sklearn.preprocessing import normalize as sknormalize

def compute_IME(features=None, copy=False, params=None):
    # Normalize
    features = L2_normalize(features, copy=copy)
    # 多级Isomap as transform matrix  as  IME layer
    if params:
        transform_matrx = params['Transform_matrx']
        start = time.time()
        features = np.dot(features, transform_matrx)
        end = time.time()
        print end - start
    else:
        start = time.time()
        features_in_original = features
        features_in = features
        # k1
        isomap1 = IME(n_neighbors=5, n_components=2048, eigen_solver='auto', tol=0, max_iter=None,
                                  path_method='auto', neighbors_algorithm='auto', n_jobs=5)
        features = isomap1.fit_transform_xj(X=features,features_in=features_in,original_weight=2.0)
        features = L2_normalize(features, copy=copy)
        # k2
        features_in = features
        isomap2 = IME(n_neighbors=5, n_components=2048, eigen_solver='auto', tol=0, max_iter=None,
                                  path_method='auto', neighbors_algorithm='auto', n_jobs=5)
        features = isomap2.fit_transform_xj(X=features,features_in=features_in,original_weight=2.0)
        features = L2_normalize(features, copy=copy)
        # IME layer
        norm = 1.0 * np.eye(features_in_original.shape[1], features_in_original.shape[1])
        transform_matrx = np.dot(np.dot(np.linalg.inv(np.dot(features_in_original.T, features_in_original) + norm),
                                        features_in_original.T), features)
        params = {'Transform_matrx': transform_matrx}
        end = time.time()
        print end - start
    # Normalize
    features = L2_normalize(features, copy=copy)
    return features, params

def use_original_feature(X,region_scale_num=0):
    if len(X.shape)==1:
        X=X.reshape((1,-1))
    if region_scale_num==0:
        features = [X[0]]
    else:
        features = X
        features = [features.sum(0)]
    return features

def L2_normalize(x, copy=False):
    """
    A helper function that wraps the function of the same name in sklearn.
    This helper handles the case of a single column vector.
    """
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1,-1), copy=copy))
    else:
        return sknormalize(x, copy=copy)

