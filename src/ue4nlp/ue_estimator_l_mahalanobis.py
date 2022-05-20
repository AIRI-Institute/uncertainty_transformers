import torch
import numpy as np
from tqdm import tqdm

from utils.utils_heads import (
    ElectraClassificationHeadIdentityPooler,
    BertClassificationHeadIdentityPooler,
    ElectraNERHeadIdentityPooler,
)
from utils.utils_inference import (
    is_custom_head,
    unpad_features,
    pad_scores
)
from ue4nlp.mahalanobis_distance import (
    mahalanobis_distance,
    mahalanobis_distance_relative,
    mahalanobis_distance_marginal,
    compute_centroids,
    compute_covariance
)

import logging
import time

log = logging.getLogger()


class UeEstimatorLMahalanobis:
    def __init__(self, cls, ue_args, config, train_dataset):
        self.cls = cls
        self.ue_args = ue_args
        self.config = config
        self.train_dataset = train_dataset
        
    def __call__(self, X, y):
        return self._predict_with_fitted_cov(X, y)
    
    def fit_ue(self, X, y=None, X_test=None):
        cls = self.cls
        model = self.cls._auto_model
        
        log.info("****************Start fitting covariance and centroids **************")
        
        if y is None:
            y = self._exctract_labels(X)
            
        self._replace_model_head()
        X_features, X_hidden_states = self._exctract_features(X)
            
        self.class_cond_centroids = self._fit_centroids(X_features, y)
        self.class_cond_covariance = self._fit_covariance(X_features, y)
        
        self.class_cond_centroids_hiddens = []
        self.class_cond_covariance_hiddens = []
        
        for X_hidden_state in X_hidden_states:
            self.class_cond_centroids_hiddens.append(self._fit_centroids(X_hidden_state, y))
            self.class_cond_covariance_hiddens.append(self._fit_covariance(X_hidden_state, y, centroids=self.class_cond_centroids_hiddens[-1]))
                
        log.info("**************Done.**********************")
        
    def _fit_covariance(self, X, y, class_cond=True, centroids=None):
        if class_cond:
            if centroids is None:
                centroids = self.class_cond_centroids
            return compute_covariance(centroids, X, y, class_cond)
        if centroids is None:
            centroids = self.train_centroid
        return compute_covariance(centroids, X, y, class_cond)
        
    def _fit_centroids(self, X, y, class_cond=True):
        return compute_centroids(X, y, class_cond)
      
    def _replace_model_head(self): 
        log.info("Change classifier to Identity Pooler")
        cls = self.cls
        model = self.cls._auto_model

        if is_custom_head(model):
            model.classifier = ElectraClassificationHeadIdentityPooler(model.classifier)
        else:
            model.classifier = BertClassificationHeadIdentityPooler(model.classifier)
        
    def _exctract_labels(self, X):    
        return np.asarray([example["label"] for example in X])
    
    def _exctract_features(self, X):
        cls = self.cls
        model = self.cls._auto_model
        
        try:
            X = X.remove_columns("label")
        except:
            X.dataset = X.dataset.remove_columns("label")
            
        features = self.cls.predict(
            X, apply_softmax=False, return_preds=False
        )
        
        X_hidden_states = [np.tanh(state) for state in features[1][1]]
        X_features = features[0]
        
        return X_features, X_hidden_states

        
    def _predict_with_fitted_cov(self, X, y):
        cls = self.cls
        model = self.cls._auto_model
        
        log.info("****************Compute MD with fitted covariance and centroids **************")

        start = time.time()
        if y is None:
            y = self._exctract_labels(X)
        X_features, X_hidden_states = self._exctract_features(X)
        end = time.time()
        
        eval_results = {}
    
        md_n, inf_time = mahalanobis_distance(None, None, X_features, 
                                             self.class_cond_centroids, self.class_cond_covariance)
        sum_inf_time = inf_time + (end - start)        

        md_l = []
        for i, X_hidden_state in enumerate(X_hidden_states):
            md_i, inf_time = mahalanobis_distance(None, None, X_hidden_state,
                                                  self.class_cond_centroids_hiddens[i],
                                                  self.class_cond_covariance_hiddens[i])
            md_l.append(md_i.copy())
            sum_inf_time += inf_time
            
        md_l = np.asarray(md_l)
        eval_results["mahalanobis_distance"] = (md_n + md_l.sum(axis=0)).tolist()
        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")
        
        log.info("**************Done.**********************")
        return eval_results