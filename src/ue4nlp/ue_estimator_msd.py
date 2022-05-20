import torch
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
    XLNetClassificationHeadIdentityPooler,
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

import numpy as np
import copy
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
        convert_to_mc_dropout(model, {"Dropout": dropout_ctor})

    else:
        raise ValueError(f"Wrong ue args {ue_args.dropout_subs}")


class UeEstimatorMSD:
    def __init__(
        self, cls, config, ue_args, eval_metric, calibration_dataset, train_dataset
    ):
        self.cls = cls
        self.ue_args = ue_args
        self.calibration_dataset = calibration_dataset
        self.eval_metric = eval_metric
        self.train_dataset = train_dataset
        self.config = config
    
    def fit_ue(self, X, y=None, X_test=None):
        cls = self.cls
        model = self.cls._auto_model
        
        log.info("****************Start fitting covariance and centroids **************")
        
        if y is None:
            y = self._exctract_labels(X)
            
        self._replace_model_head()
        X_features = self._exctract_features(X)
            
        self.class_cond_centroids = self._fit_centroids(X_features, y)
        self.class_cond_covarince = self._fit_covariance(X_features, y)

        self._restore_model_head()
        log.info("**************Done.**********************")
        
    def _fit_covariance(self, X, y, class_cond=True):
        if class_cond:
            return compute_covariance(self.class_cond_centroids, X, y, class_cond)
        return compute_covariance(self.train_centroid, X, y, class_cond)
        
    def _fit_centroids(self, X, y, class_cond=True):
        return compute_centroids(X, y, class_cond)
      
    def _replace_model_head(self): 
        cls = self.cls
        model = self.cls._auto_model
        self.old_classifier = copy.deepcopy(model.classifier)
        use_paper_version = self.ue_args.get("use_paper_version", False)
        use_activation = not use_paper_version
        
        if is_custom_head(model):
            model.classifier = ElectraClassificationHeadIdentityPooler(model.classifier, use_activation)
        elif "xlnet" in self.config.model.model_name_or_path:
            # so XLNet hasn't classifier, we replace sequence_summary and logits_proj
            self.cls.model.logits_proj = XLNetClassificationHeadIdentityPooler()
        else:
            model.classifier = BertClassificationHeadIdentityPooler(model.classifier)

    def _restore_model_head(self):
        model = self.cls._auto_model
        model.classifier = self.old_classifier

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

    def _calc_distinctivness_score(self, full_mahalanobis_distance, eval_labels, eval_results):
        start_unc = time.time()
        min_mahalanobis_distance = np.min(full_mahalanobis_distance, axis=-1)
        # calc penalty
        penalty = self.config.mixup.margin * np.where(
            eval_labels == np.argmin(full_mahalanobis_distance, axis=-1), 0, 1
        )
        dist_score = np.log10(
            self.config.mixup.beta1 * penalty
            + self.config.mixup.beta2 * min_mahalanobis_distance
        )
        # after calc uncertainty score
        max_probs = np.max(
            np.mean(np.asarray(eval_results["sampled_probabilities"]), axis=0), axis=-1
        )
        uncertainty_score = (
            self.config.mixup.gamma1 / max_probs + self.config.mixup.gamma2 * dist_score
        )
        end_unc = time.time()
        eval_results["uncertainty_score"] = uncertainty_score.tolist()
        return eval_results, end_unc - start_unc

    def _predict_with_fitted_cov(self, X, y, eval_results):
        cls = self.cls
        model = self.cls._auto_model
        
        self._replace_model_head()
        log.info("****************Compute MD with fitted covariance and centroids **************")

        start = time.time()
        if y is None:
            y = self._exctract_labels(X)
        X_features = self._exctract_features(X)
        end = time.time()
        
        md, inf_time = mahalanobis_distance(None, None, X_features, 
                                            self.class_cond_centroids, self.class_cond_covarince, True)

        sum_inf_time = inf_time + (end - start)
        eval_results["mahalanobis_distance"] = md.tolist()
        self._restore_model_head()
        log.info("**************Done.**********************")
        return eval_results, md, sum_inf_time

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

    def _predict_mc(self, X, y):
        ue_args = self.ue_args
        eval_metric = self.eval_metric
        model = self.cls._auto_model

        start = time.time()
        model = self._activate_dropouts(model)

        eval_results = {}
        eval_results["sampled_probabilities"] = []
        eval_results["sampled_answers"] = []

        log.info("****************Start runs**************")

        for i in tqdm(range(ue_args.committee_size)):
            preds, probs = self.cls.predict(X)[:2]

            eval_results["sampled_probabilities"].append(probs.tolist())
            eval_results["sampled_answers"].append(preds.tolist())

            if ue_args.eval_passes:
                eval_score = eval_metric.compute(
                    predictions=preds, references=true_labels
                )
                log.info(f"Eval score: {eval_score}")
        end = time.time()

        log.info("**************Done.********************")
        model = self._deactivate_dropouts(model)
        return eval_results, end - start

    def _predict_msd(self, X, y):
        ue_args = self.ue_args
        model = self.cls._auto_model

        if y is None:
            y = self._exctract_labels(X)
        eval_results, mc_time = self._predict_mc(X, y)
        eval_results, full_mahalanobis_distance, md_time = self._predict_with_fitted_cov(X, y, eval_results)
        eval_results["eval_labels"] = y

        # so now we have sampled probs and mahalanobis distances in eval_preds
        # we have to calc distinctivness score and uncertainty scores
        eval_results, unc_time = self._calc_distinctivness_score(full_mahalanobis_distance, y, eval_results)

        sum_inf_time = mc_time + md_time + unc_time
        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")
        return eval_results

    def __call__(self, X, y):
        return self._predict_msd(X, y)


class UeEstimatorMSDNer:
    def __init__(
        self, cls, config, ue_args, eval_metric, calibration_dataset, train_dataset
    ):
        self.cls = cls
        self.ue_args = ue_args
        self.calibration_dataset = calibration_dataset
        self.eval_metric = eval_metric
        self.train_dataset = train_dataset
        self.config = config

    def fit_ue(self, X, y=None, X_test=None):
        cls = self.cls
        model = self.cls._auto_model
        
        log.info("****************Start fitting covariance and centroids **************")
        
        if y is None:
            y, y_shape = self._exctract_labels(X)
            
        self._replace_model_head()
        X_features = self._exctract_features(X)
        
        self.class_cond_centroids = self._fit_centroids(X_features, y)
        self.class_cond_covarince = self._fit_covariance(X_features, y)

        self._restore_model_head()
        log.info("**************Done.**********************")

    def _fit_covariance(self, X, y, class_cond=True):
        if class_cond:
            return compute_covariance(self.class_cond_centroids, X, y, class_cond)
        return compute_covariance(self.train_centroid, X, y, class_cond)
        
    def _fit_centroids(self, X, y, class_cond=True):
        return compute_centroids(X, y, class_cond)
      
    def _replace_model_head(self): 
        cls = self.cls
        model = self.cls._auto_model
        self.old_classifier = copy.deepcopy(model.classifier)
        
        use_paper_version = self.ue_args.get("use_paper_version", False)
        use_activation = not use_paper_version
        if is_custom_head(model):
            model.classifier = ElectraNERHeadIdentityPooler(model.classifier, use_activation)
        else:
            model.classifier = BertClassificationHeadIdentityPooler(model.classifier)

    def _restore_model_head(self):
        model = self.cls._auto_model
        model.classifier = self.old_classifier

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

    def _calc_distinctivness_score(self, full_mahalanobis_distance, eval_labels, eval_shape, eval_results):
        start_unc = time.time()
        min_mahalanobis_distance = np.min(full_mahalanobis_distance, axis=-1).reshape(
            eval_shape
        )
        # calc penalty
        penalty = self.config.mixup.margin * np.where(
            eval_labels
            == np.argmin(full_mahalanobis_distance, axis=-1).reshape(eval_shape),
            0,
            1,
        )
        dist_score = np.log10(
            self.config.mixup.beta1 * penalty
            + self.config.mixup.beta2 * min_mahalanobis_distance
        )
        # after calc uncertainty score
        max_probs = np.max(
            np.mean(np.asarray(eval_results["sampled_probabilities"]), axis=0), axis=-1
        )
        uncertainty_score = (
            self.config.mixup.gamma1 / max_probs + self.config.mixup.gamma2 * dist_score
        )
        end_unc = time.time()
        eval_results["uncertainty_score"] = uncertainty_score.tolist()
        return eval_results, end_unc - start_unc

    def _predict_with_fitted_cov(self, X, y, eval_results):
        cls = self.cls
        model = self.cls._auto_model
        
        self._replace_model_head()
        log.info("****************Compute MD with fitted covariance and centroids **************")

        start = time.time()
        
        y_pad, y_shape = self._exctract_labels(X)
        X_features = self._exctract_features(X)

        end = time.time()
        
        md, inf_time = mahalanobis_distance(None, None, X_features, 
                                            self.class_cond_centroids, self.class_cond_covarince, True)

        sum_inf_time = inf_time + (end - start)
        eval_results["mahalanobis_distance"] = md.tolist()
        self._restore_model_head()

        log.info("**************Done.**********************")
        return eval_results, md, sum_inf_time

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

    def _predict_mc(self, X, y):
        ue_args = self.ue_args
        eval_metric = self.eval_metric
        model = self.cls._auto_model

        start = time.time()
        model = self._activate_dropouts(model)

        eval_results = {}
        eval_results["sampled_probabilities"] = []
        eval_results["sampled_answers"] = []

        log.info("****************Start runs**************")

        for i in tqdm(range(ue_args.committee_size)):
            preds, probs = self.cls.predict(X)[:2]

            eval_results["sampled_probabilities"].append(probs.tolist())
            eval_results["sampled_answers"].append(preds.tolist())

            if ue_args.eval_passes:
                eval_score = eval_metric.compute(
                    predictions=preds, references=true_labels
                )
                log.info(f"Eval score: {eval_score}")
        end = time.time()

        log.info("**************Done.********************")
        model = self._deactivate_dropouts(model)
        return eval_results, end - start

    def _predict_msd(self, X, y):
        ue_args = self.ue_args
        model = self.cls._auto_model

        y_pad, y_shape = self._exctract_labels(X)
            
        eval_results, mc_time = self._predict_mc(X, y)
        eval_results, full_mahalanobis_distance, md_time = self._predict_with_fitted_cov(X, y, eval_results)
        eval_results["eval_labels"] = y

        # so now we have sampled probs and mahalanobis distances in eval_preds
        # we have to calc distinctivness score and uncertainty scores
        eval_results, unc_time = self._calc_distinctivness_score(full_mahalanobis_distance, y_pad, y_shape, eval_results)

        sum_inf_time = mc_time + md_time + unc_time
        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")
        return eval_results

    def __call__(self, X, y):
        return self._predict_msd(X, y)
