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

import time
import logging
import sys
import os
import random
import ray

log = logging.getLogger()

try:
    from nuq import NuqClassifier
except:
    log.info('There is no NUQ module!')
    
    
def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class UeEstimatorLNUQ:
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
        seed_everything(self.config.training.seed)

        if y is None:
            y = self._exctract_labels(X)
            
        self._replace_model_head()
        X_features, X_hidden_states = self._exctract_features(X)
            
        self.nuq_classifier = self._fit_nuq(X_features, y)   
        self.hidden_nuq_classifiers = []
        
        for i, X_hidden_state in enumerate(X_hidden_states):
            self.hidden_nuq_classifiers.append(self._fit_nuq(X_hidden_state, y, i))
    
    def _predict_with_fitted_nuq(self, X, y):
        cls = self.cls
        model = self.cls._auto_model
        seed_everything(self.config.training.seed)
        
        log.info("****************Compute NUQ uncertainty with fitted NuqClassifier**************")

        start = time.time()
        if y is None:
            y = self._exctract_labels(X)
        
        X_features, X_hidden_states = self._exctract_features(X)
    
        eval_results = {}
        
        nuq_probs, log_epistemic_uncs = self.nuq_classifier.predict_proba(np.asarray(X_features), return_uncertainty="epistemic")
        _, log_aleatoric_uncs = self.nuq_classifier.predict_proba(np.asarray(X_features), return_uncertainty="aleatoric")
        
        end = time.time()
        sum_inf_time = (end - start)
        
        for i, X_hidden_state in enumerate(X_hidden_states):
            start = time.time()
            
            nuq_classifier_i = self.hidden_nuq_classifiers[i]
            
            if nuq_classifier_i is None:
                continue
                
            _, log_epistemic_uncs_l = nuq_classifier_i.predict_proba(np.asarray(X_hidden_state),
                                                                     return_uncertainty="epistemic")
            _, log_aleatoric_uncs_l = nuq_classifier_i.predict_proba(np.asarray(X_hidden_state),
                                                                     return_uncertainty="aleatoric")
            log_epistemic_uncs += log_epistemic_uncs_l
            log_aleatoric_uncs += log_aleatoric_uncs_l
   
            end = time.time()
            sum_inf_time += (end - start)        

        eval_results["aleatoric"] = log_epistemic_uncs.tolist()
        eval_results["epistemic"] = log_aleatoric_uncs.tolist()
        eval_results["total"] = (log_epistemic_uncs+log_aleatoric_uncs).tolist()
        eval_results["nuq_probabilities"] = nuq_probs.todense().tolist()
        
        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")
    
        log.info("**************Done.**********************")
        return eval_results
    
    def _fit_nuq(self, X, y, i=None):
        log.info("****************Start fitting NuqClassifier**************")
        tune_bandwidth = self.ue_args.nuq.tune_bandwidth
        tune_bandwidth = None if tune_bandwidth=='None' else tune_bandwidth
        
        try:
            nuq_classifier = NuqClassifier(
                tune_bandwidth=tune_bandwidth,
                n_neighbors=self.ue_args.nuq.n_neighbors,
                log_pN=self.ue_args.nuq.log_pN,
            )
            nuq_classifier.fit(X=np.asarray(X), 
                               y=np.asarray(y))
        except:
            tune_bandwidth = None 
            nuq_classifier = NuqClassifier(
                tune_bandwidth=tune_bandwidth,
                n_neighbors=self.ue_args.nuq.n_neighbors,
                log_pN=self.ue_args.nuq.log_pN,
            )
            nuq_classifier.fit(X=np.asarray(X), 
                               y=np.asarray(y))
            
        if tune_bandwidth is None:
            try:
                _, squared_dists = ray.get(
                    self.nuq_classifier.index_.knn_query.remote(self.nuq_classifier.X_ref_, return_dist=True)
                )
                dists = np.sqrt(squared_dists)[:, 1:]
                min_dists = dists[:, 0]
                left, right = min_dists[min_dists != 0].min(), dists.max()
                bandwidth = np.sqrt(left*right)

                log.info(f'NUQ bandwidth {bandwidth}')
                nuq_classifier.bandwidth_ref_ = ray.put(np.array(bandwidth))
            except Exception as e: 
                log.info(f"Again error while fitting L-NUQ {i}, skip")
                log.info(f"Error: {e}")
                return None

        log.info("**************Done.**********************")
        return nuq_classifier

    
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