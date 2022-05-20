import torch
import numpy as np

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

import logging
import sys
import ray
import time

log = logging.getLogger()

try:
    from nuq import NuqClassifier
except:
    log.info('There is no NUQ module!')
    

class UeEstimatorNUQ:
    def __init__(self, cls, ue_args, config, train_dataset, calibration_dataset):
        self.cls = cls
        self.ue_args = ue_args
        self.config = config
        self.train_dataset = train_dataset
        self.calibration_dataset = calibration_dataset

    def __call__(self, X, y=None):
        return self._predict_with_fitted_nuq(X, y)
    
    def fit_ue(self, X, y=None, X_test=None):
        cls = self.cls
        model = self.cls._auto_model

        if y is None:
            y = self._exctract_labels(X)
            
        self._replace_model_head()
        X_features = self._exctract_features(X)
            
        self._fit_nuq(X_features, y)        
    
    def _predict_with_fitted_nuq(self, X, y):
        cls = self.cls
        model = self.cls._auto_model
        
        log.info("****************Compute NUQ uncertainty with fitted NuqClassifier**************")

        start = time.time()
        if y is None:
            y = self._exctract_labels(X)
        X_features = self._exctract_features(X)
        
        eval_results = {}
        
        nuq_probs, log_epistemic_uncs = self.nuq_classifier.predict_proba(np.asarray(X_features), return_uncertainty="epistemic")
        _, log_aleatoric_uncs = self.nuq_classifier.predict_proba(np.asarray(X_features), return_uncertainty="aleatoric")
        
        end = time.time()
        sum_inf_time = (end - start)

        eval_results["aleatoric"] = log_epistemic_uncs.tolist()
        eval_results["epistemic"] = log_aleatoric_uncs.tolist()
        eval_results["total"] = (log_epistemic_uncs+log_aleatoric_uncs).tolist()
        eval_results["nuq_probabilities"] = nuq_probs.todense().tolist()
        
        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")
    
        log.info("**************Done.**********************")
        return eval_results
    
    def _fit_nuq(self, X, y):
        log.info("****************Start fitting NuqClassifier**************")
        tune_bandwidth = self.ue_args.nuq.tune_bandwidth
        tune_bandwidth = None if tune_bandwidth=='None' else tune_bandwidth
        
        try:
            self.nuq_classifier = NuqClassifier(
                tune_bandwidth=tune_bandwidth,
                n_neighbors=self.ue_args.nuq.n_neighbors,
                log_pN=self.ue_args.nuq.log_pN,
            )
            self.nuq_classifier.fit(X=np.asarray(X), 
                                    y=np.asarray(y))
        except:
            tune_bandwidth = None 
            self.nuq_classifier = NuqClassifier(
                tune_bandwidth=tune_bandwidth,
                n_neighbors=self.ue_args.nuq.n_neighbors,
                log_pN=self.ue_args.nuq.log_pN,
            )
            self.nuq_classifier.fit(X=np.asarray(X), 
                                    y=np.asarray(y))
        if tune_bandwidth is None:
            _, squared_dists = ray.get(
                self.nuq_classifier.index_.knn_query.remote(self.nuq_classifier.X_ref_, return_dist=True)
            )
            # bandwidth = np.max(np.sqrt(squared_dists))
            dists = np.sqrt(squared_dists)[:, 1:]
            min_dists = dists[:, 0]
            left, right = min_dists[min_dists != 0].min(), dists.max()
            bandwidth = np.sqrt(left*right)

            log.info(f'NUQ bandwidth {bandwidth}')
            self.nuq_classifier.bandwidth_ref_ = ray.put(np.array(bandwidth))
        
        log.info("**************Done.**********************")

    
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
            
        X_features = cls.predict(X, apply_softmax=False, return_preds=False)[0]
        return X_features
    
    
class UeEstimatorNUQNer:
    def __init__(self, cls, ue_args, config, train_dataset, calibration_dataset):
        self.cls = cls
        self.ue_args = ue_args
        self.config = config
        self.train_dataset = train_dataset
        self.calibration_dataset = calibration_dataset

    def __call__(self, X, y=None):
        return self._predict_with_fitted_nuq(X, y)
    
    def fit_ue(self, X, y=None, X_test=None):
        cls = self.cls
        model = self.cls._auto_model

        if y is None:
            y, y_shape = self._exctract_labels(X)
            
        self._replace_model_head()
        X_features = self._exctract_features(X)
        X_features, y = unpad_features(X_features, y)
            
        self._fit_nuq(X_features, y)        
    
    def _predict_with_fitted_nuq(self, X, y):
        cls = self.cls
        model = self.cls._auto_model
        
        log.info("****************Compute NUQ uncertainty with fitted NuqClassifier**************")

        start = time.time()

        y_pad, y_shape = self._exctract_labels(X)
        X_features = self._exctract_features(X)
        X_features, y = unpad_features(X_features, y_pad)
        
        eval_results = {}
        
        nuq_probs, log_epistemic_uncs = self.nuq_classifier.predict_proba(np.asarray(X_features), return_uncertainty="epistemic")
        _, log_aleatoric_uncs = self.nuq_classifier.predict_proba(np.asarray(X_features), return_uncertainty="aleatoric")
        
        end = time.time()
        sum_inf_time = (end - start)

        log_epistemic_uncs = pad_scores(log_epistemic_uncs, np.asarray(y_pad).reshape(y_shape), y_pad)
        log_aleatoric_uncs = pad_scores(log_aleatoric_uncs, np.asarray(y_pad).reshape(y_shape), y_pad)
        
        eval_results["aleatoric"] = log_epistemic_uncs.tolist()
        eval_results["epistemic"] = log_aleatoric_uncs.tolist()
        eval_results["total"] = (log_epistemic_uncs+log_aleatoric_uncs).tolist()
        eval_results["nuq_probabilities"] = nuq_probs.todense().tolist()
        
        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")
    
        log.info("**************Done.**********************")
        return eval_results
    
    def _fit_nuq(self, X, y):
        log.info("****************Start fitting NuqClassifier**************")
        tune_bandwidth = self.ue_args.nuq.tune_bandwidth
        tune_bandwidth = None if tune_bandwidth=='None' else tune_bandwidth
        
        try:
            self.nuq_classifier = NuqClassifier(
                tune_bandwidth=tune_bandwidth,
                n_neighbors=self.ue_args.nuq.n_neighbors,
                log_pN=self.ue_args.nuq.log_pN,
            )
            self.nuq_classifier.fit(X=np.asarray(X), 
                                    y=np.asarray(y))
        except:
            tune_bandwidth = None 
            self.nuq_classifier = NuqClassifier(
                tune_bandwidth=tune_bandwidth,
                n_neighbors=self.ue_args.nuq.n_neighbors,
                log_pN=self.ue_args.nuq.log_pN,
            )
            self.nuq_classifier.fit(X=np.asarray(X), 
                                    y=np.asarray(y))
        if tune_bandwidth is None:
            _, squared_dists = ray.get(
                self.nuq_classifier.index_.knn_query.remote(self.nuq_classifier.X_ref_, return_dist=True)
            )
            # bandwidth = np.max(np.sqrt(squared_dists))
            dists = np.sqrt(squared_dists)[:, 1:]
            min_dists = dists[:, 0]
            left, right = min_dists[min_dists != 0].min(), dists.max()
            bandwidth = np.sqrt(left*right)

            log.info(f'NUQ bandwidth {bandwidth}')
            self.nuq_classifier.bandwidth_ref_ = ray.put(np.array(bandwidth))
        
        log.info("**************Done.**********************")

    
    def _replace_model_head(self): 
        log.info("Change classifier to Identity Pooler")
        cls = self.cls
        model = self.cls._auto_model
        
        if is_custom_head(model):
            model.classifier = ElectraNERHeadIdentityPooler(model.classifier)
        else:
            model.classifier = BertClassificationHeadIdentityPooler(model.classifier)
        
    def _exctract_labels(self, X):    
        y = np.asarray([example["labels"] for example in X])
        y_shape = y.shape
        
        return y.reshape(-1), y_shape
    
    def _exctract_features(self, X):
        cls = self.cls
        model = self.cls._auto_model
        
        try:
            X = X.remove_columns("labels")
        except:
            X.dataset = X.dataset.remove_columns("labels")
            
        X_features = cls.predict(X, apply_softmax=False, return_preds=False)[0]
        X_features = X_features.reshape(-1, X_features.shape[-1])
        
        return X_features