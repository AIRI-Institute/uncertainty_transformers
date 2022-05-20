from tqdm import tqdm
import numpy as np
import time

import logging

log = logging.getLogger()

def compute_centroids(train_features, train_labels, class_cond=True):
    if class_cond:
        centroids = []
        for label in np.sort(np.unique(train_labels)):
            centroids.append(train_features[train_labels == label].mean(axis=0))
        return np.asarray(centroids)
    else:
        return train_features.mean(axis=0)



def compute_covariance(centroids, train_features, train_labels, class_cond=True):
    cov = np.zeros((train_features.shape[1], train_features.shape[1]))
    if class_cond:
        for c, mu_c in tqdm(enumerate(centroids)):
            for x in train_features[train_labels == c]:
                d = (x - mu_c)[:, None]
                cov += d @ d.T
    else:
        for x in train_features:
            d = (x - mu_0)[:, None]
            cov += d @ d.T
    cov /= train_features.shape[0]
    
    try:
        sigma_inv = np.linalg.inv(cov)
    except:
        sigma_inv = np.linalg.pinv(cov)
        log.info("Compute pseudo-inverse matrix")
        
    return sigma_inv


def mahalanobis_distance(train_features, train_labels, eval_features, centroids=None, covariance=None, return_full=False):
    if centroids is None:
        centroids = compute_centroids(train_features, train_labels)
    if covariance is None:
        covariance = compute_covariance(centroids, train_features, train_labels)
    diff = eval_features[:, None, :] - centroids[None, :, :]
    start = time.time()
    dists = np.matmul(np.matmul(diff, covariance), diff.transpose(0, 2, 1))
    dists = np.asarray([np.diag(dist) for dist in dists])
    end = time.time()
    if return_full:
        return dists, end-start
    else:
        return np.min(dists, axis=1), end-start


def mahalanobis_distance_marginal(train_features, eval_features, centroids=None, covariance=None):
    if centroids is None:
        centroids = compute_centroids(train_features, train_labels, class_cond=False)
    if covariance is None:
        covariance = compute_covariance(centroids, train_features, train_labels, class_cond=False)
        
    diff = eval_features - mu_0[None, :]
    dists = np.matmul(np.matmul(diff, covariance), diff.T)
    return np.diag(dists)


def mahalanobis_distance_relative(train_features, train_labels, eval_features, 
                                  centroids=None, covariance=None,
                                  train_centroid=None, train_covariance=None):
    if centroids is None:
        centroids = compute_centroids(train_features, train_labels)
    if covariance is None:
        covariance = compute_covariance(centroids, train_features, train_labels)
    diff = eval_features[:, None, :] - centroids[None, :, :]
    dists = np.matmul(np.matmul(diff, covariance), diff.transpose(0, 2, 1))
    dists = np.asarray([np.diag(dist) for dist in dists])

    md_marginal = mahalanobis_distance_marginal(train_features, eval_features, train_centroid, train_covariance)
    return np.min(dists - md_marginal[:, None], axis=1)