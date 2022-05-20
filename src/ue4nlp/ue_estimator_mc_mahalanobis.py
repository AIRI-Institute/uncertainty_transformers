from tqdm import tqdm
import numpy as np

from ue4nlp.dropconnect_mc import (
    LinearDropConnectMC,
    activate_mc_dropconnect,
    convert_to_mc_dropconnect,
    hide_dropout,
)
from ue4nlp.dropout_mc import DropoutMC, activate_mc_dropout, convert_to_mc_dropout
from utils.utils_dropout import set_last_dropout, get_last_dropout, set_last_dropconnect

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
import time
import logging

log = logging.getLogger()
from ue4nlp.dropconnect_mc import (
    LinearDropConnectMC,
    activate_mc_dropconnect,
    convert_to_mc_dropconnect,
    hide_dropout,
)
from ue4nlp.dropout_mc import DropoutMC, activate_mc_dropout, convert_to_mc_dropout
from utils.utils_dropout import set_last_dropout, get_last_dropout, set_last_dropconnect

from tqdm import tqdm
import time

import logging

log = logging.getLogger()


def convert_dropouts(model, ue_args):
    if ue_args.dropout_type == "MC":
        dropout_ctor = lambda p, activate: DropoutMC(
            p=ue_args.inference_prob, activate=False
        )

    elif ue_args.dropout_type == "DC_MC":
        dropout_ctor = lambda linear, activate: LinearDropConnectMC(
            linear=linear, p_dropconnect=ue_args.inference_prob, activate=activate
        )

    else:
        raise ValueError(f"Wrong dropout type: {ue_args.dropout_type}")

    if (ue_args.dropout_subs == "all") and (ue_args.dropout_type == "DC_MC"):
        convert_to_mc_dropconnect(
            model.electra.encoder, {"Linear": dropout_ctor}
        )  # TODO: check encoder or all dropouts ?
        hide_dropout(model.electra.encoder)

    elif (ue_args.dropout_subs == "last") and (ue_args.dropout_type == "DC_MC"):
        set_last_dropconnect(model, dropout_ctor)
        hide_dropout(model.classifier)

    elif ue_args.dropout_subs == "last":
        set_last_dropout(model, dropout_ctor(p=ue_args.inference_prob, activate=False))

    elif ue_args.dropout_subs == "all":
        convert_to_mc_dropout(model, {"Dropout": dropout_ctor, "StableDropout": dropout_ctor})

    else:
        raise ValueError(f"Wrong ue args {ue_args.dropout_subs}")


class UeEstimatorMcMahalanobis:
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
        
        log.info("****************Start fitting covariance and centroids**************")
        
        if y is None:
            y = self._exctract_labels(X)
                
        if self.ue_args.use_cache:
            log.info("Caching enabled.")
            model.enable_cache()
            
        self._replace_model_head()
                
        X_features = self._exctract_features(X)
        self.class_cond_centroids = self._fit_centroids(X_features, y)
        self.class_cond_covariance = self._fit_covariance(X_features, y)
            
        model = self._activate_dropouts(model)
        X_features_stoch = self._exctract_features(X)
        self.class_cond_centroids_stoch = self._fit_centroids(X_features_stoch, y)
        self.class_cond_covariance_stoch = self._fit_covariance(X_features_stoch, y, centroids=self.class_cond_centroids_stoch)
        
        model = self._deactivate_dropouts(model)
        
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
    
    def _exctract_features(self, X, remove_col=True):
        cls = self.cls
        model = self.cls._auto_model
        
        if remove_col:
            try:
                X = X.remove_columns("label")
            except:
                X.dataset = X.dataset.remove_columns("label")
            
        X_features = cls.predict(X, apply_softmax=False, return_preds=False)[0]
        return X_features
    
    def _activate_dropouts(self, model):
        ue_args = self.ue_args
        log.info("******Perform stochastic inference...*******")

        if ue_args.dropout_type == "DC_MC":
            activate_mc_dropconnect(model, activate=True, random=ue_args.inference_prob)
        else:
            convert_dropouts(model, ue_args)
            activate_mc_dropout(model, activate=True, random=ue_args.inference_prob)

        if ue_args.use_cache:
            log.info("Caching enabled.")
            model.enable_cache()
        return model

    def _deactivate_dropouts(self, model):
        activate_mc_dropout(model, activate=False)
        activate_mc_dropconnect(model, activate=False)
        return model

        
    def _predict_with_fitted_cov(self, X, y):
        cls = self.cls
        model = self.cls._auto_model
        
        log.info("****************Compute MD with fitted covariance and centroids **************")

        start = time.time()
        if y is None:
            y = self._exctract_labels(X)
        X_features = self._exctract_features(X)
        end = time.time()
        
        eval_results = {}
        
        md, inf_time = mahalanobis_distance(None, None, X_features, 
                                            self.class_cond_centroids, self.class_cond_covariance)
        
        sum_inf_time = inf_time + (end - start)
        eval_results["mahalanobis_distance"] = md.tolist()
        eval_results["sampled_mahalanobis_distance"] = []
        
        model = self._activate_dropouts(model)
        
        for i in tqdm(range(self.ue_args.committee_size)):
            start_loop_maha = time.time()
            X_features_stoch = self._exctract_features(X)
            end_loop_maha = time.time()
            md, inf_time = mahalanobis_distance(None, None, X_features_stoch, 
                                                self.class_cond_centroids_stoch, self.class_cond_covariance_stoch)
            
            sum_inf_time += inf_time + (end_loop_maha - start_loop_maha)
            eval_results["sampled_mahalanobis_distance"].append(md.tolist())

        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")

    
        log.info("**************Done.**********************")
        return eval_results

    
    
class UeEstimatorMcMahalanobisNer:
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
        
        log.info("****************Start fitting covariance and centroids**************")
        
        if y is None:
            y = self._exctract_labels(X)
                
        if self.ue_args.use_cache:
            log.info("Caching enabled.")
            model.enable_cache()
            
        self._replace_model_head()
                
        y_pad, y_shape = self._exctract_labels(X)
        X_features = self._exctract_features(X)
        X_features, y = unpad_features(X_features, y_pad)
        self.class_cond_centroids = self._fit_centroids(X_features, y)
        self.class_cond_covariance = self._fit_covariance(X_features, y)
            
        model = self._activate_dropouts(model)
        X_features_stoch = self._exctract_features(X, remove_col=False)
        X_features_stoch, y = unpad_features(X_features_stoch, y_pad)
        self.class_cond_centroids_stoch = self._fit_centroids(X_features_stoch, y)
        self.class_cond_covariance_stoch = self._fit_covariance(X_features_stoch, y, centroids=self.class_cond_centroids_stoch)
        
        model = self._deactivate_dropouts(model)
        
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
            model.classifier = ElectraNERHeadIdentityPooler(model.classifier)
        else:
            model.classifier = BertClassificationHeadIdentityPooler(model.classifier)
            
    def _exctract_labels(self, X):    
        y = np.asarray([example["labels"] for example in X])
        y_shape = y.shape
        
        return y.reshape(-1), y_shape
    
    def _exctract_features(self, X, remove_col=True):
        cls = self.cls
        model = self.cls._auto_model
        
        if remove_col:
            try:
                X = X.remove_columns("labels")
            except:
                X.dataset = X.dataset.remove_columns("labels")

        X_features = cls.predict(X, apply_softmax=False, return_preds=False)[0]
        X_features = X_features.reshape(-1, X_features.shape[-1])
        
        return X_features

    def _activate_dropouts(self, model):
        ue_args = self.ue_args
        log.info("******Perform stochastic inference...*******")

        if ue_args.dropout_type == "DC_MC":
            activate_mc_dropconnect(model, activate=True, random=ue_args.inference_prob)
        else:
            convert_dropouts(model, ue_args)
            activate_mc_dropout(model, activate=True, random=ue_args.inference_prob)

        if ue_args.use_cache:
            log.info("Caching enabled.")
            model.enable_cache()
        return model

    def _deactivate_dropouts(self, model):
        activate_mc_dropout(model, activate=False)
        activate_mc_dropconnect(model, activate=False)
        return model

        
    def _predict_with_fitted_cov(self, X, y):
        cls = self.cls
        model = self.cls._auto_model
        
        log.info("****************Compute MD with fitted covariance and centroids **************")

        start = time.time()
        
        y_pad, y_shape = self._exctract_labels(X)
        X_features = self._exctract_features(X)
        X_features, y = unpad_features(X_features, y_pad)
        
        end = time.time()
        
        eval_results = {}
        
        md, inf_time = mahalanobis_distance(None, None, X_features, 
                                            self.class_cond_centroids, self.class_cond_covariance)
        
        sum_inf_time = inf_time + (end - start)
        
        md = pad_scores(md, np.asarray(y_pad).reshape(y_shape), y_pad)
        eval_results["mahalanobis_distance"] = md.tolist()
        eval_results["sampled_mahalanobis_distance"] = []
        
        model = self._activate_dropouts(model)
        
        for i in tqdm(range(self.ue_args.committee_size)):
            start_loop_maha = time.time()
            X_features_stoch = self._exctract_features(X)
            X_features_stoch, y = unpad_features(X_features_stoch, y_pad)

            end_loop_maha = time.time()
            md, inf_time = mahalanobis_distance(None, None, X_features_stoch, 
                                                self.class_cond_centroids_stoch, self.class_cond_covariance_stoch)
            
            md = pad_scores(md, np.asarray(y_pad).reshape(y_shape), y_pad)
            sum_inf_time += inf_time + (end_loop_maha - start_loop_maha)
            eval_results["sampled_mahalanobis_distance"].append(md.tolist())

        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")

    
        log.info("**************Done.**********************")
        return eval_results