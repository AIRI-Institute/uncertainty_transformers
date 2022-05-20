from transformers import (
    ElectraForSequenceClassification,
    ElectraForTokenClassification,
    DistilBertForTokenClassification,
    DistilBertForSequenceClassification,
    DebertaForSequenceClassification,
    DebertaForTokenClassification,
    RobertaForSequenceClassification
)

from ue4nlp.dropout_mc import activate_mc_dropout
from ue4nlp.dropout_dpp import DropoutDPP_v3
from utils.utils_dropout import set_last_dropout
from utils.utils_heads import (ElectraClassificationHeadCustom, 
                               ElectraNERHeadCustom,
                               ElectraClassificationHeadSN,
                               ElectraNERHeadSN
                              )

import time
from tqdm import tqdm
import numpy as np

import logging

log = logging.getLogger()



def is_last_dropout_in_classifier(model):
    return (isinstance(model, ElectraForSequenceClassification) or 
            isinstance(model, ElectraForTokenClassification) or 
            isinstance(model, RobertaForSequenceClassification) or
            isinstance(model, DistilBertForTokenClassification))


def is_custom_classifier(model):
    return (isinstance(model.classifier, ElectraClassificationHeadCustom) or 
            isinstance(model.classifier, ElectraNERHeadCustom) or
            isinstance(model.classifier, ElectraClassificationHeadSN) or 
            isinstance(model.classifier, ElectraNERHeadSN))


def get_replaced_dropout(model):
    if is_last_dropout_in_classifier(model):
        if is_custom_classifier(model):
            return model.classifier.dropout2
        else:
            return model.classifier.dropout
    else:
        return model.dropout
    
def probability_variance(sampled_probabilities):
    mean_probabilities = np.mean(sampled_probabilities, axis=1)
    mean_probabilities = np.expand_dims(mean_probabilities, axis=1)
    return ((sampled_probabilities - mean_probabilities) ** 2).mean(1).mean(-1)


def select_diverse_masks(dpp_dropout, sampled_probabilities, committee_size):
    ood_probs = np.asarray(sampled_probabilities)
    ood_probs = ood_probs.reshape(ood_probs.shape[0], -1, ood_probs.shape[-1])
    ood_probs_variance = probability_variance(ood_probs)

    diverse_idx = np.argsort(ood_probs_variance)[
        -committee_size:
    ]    
    dpp_dropout.diverse_masks = dpp_dropout.diverse_masks[:, diverse_idx]
    dpp_dropout.used_mask_id = 0


def convert_dropouts(model, ue_args):
    def dropout_ctor(p, activate):
        return DropoutDPP_v3(
            p=p,
            activate=activate,
            max_n=ue_args.dropout.max_n,
            max_frac=ue_args.dropout.max_frac,
            is_reused_mask=ue_args.dropout.is_reused_mask,
            mask_name_for_mask=ue_args.dropout.mask_name_for_mask,
        )

    set_last_dropout(model, dropout_ctor(p=ue_args.inference_prob, activate=False))
    
    
def freeze_all_dpp_dropouts(model, freeze):
    for layer in model.children():
        if isinstance(layer, DropoutDPP_v3):
            if freeze:
                layer.mask.freeze(dry_run=True)
            else:
                layer.mask.unfreeze(dry_run=True)
        else:
            freeze_all_dpp_dropouts(model=layer, freeze=freeze)
    
    
class UeEstimatorMcDdpp:
    def __init__(
        self,
        cls,
        ue_args,
        eval_metric,
        calibration_dataset,
        dry_run_dataset,
        ood_dataset=None,
        ner=False
    ):
        self.cls = cls
        self.ue_args = ue_args
        self.calibration_dataset = calibration_dataset
        self.eval_metric = eval_metric
        self.dry_run_dataset = dry_run_dataset
        self.ood_dataset = ood_dataset
        self.use_paper_version = ue_args.get("use_paper_version", False)
        self.ner = ner

    def __call__(self, X, y):
        if self.ue_args.dropout.is_reused_mask:
            return self._predict_with_selected_masks(X, y)
        else:
            return self._predict_with_dpp_masks(X, y)
    
    def fit_ue(self, X, y=None, X_test=None):
        cls = self.cls
        model = self.cls._auto_model
        
        if not(self.use_paper_version and self.ner):
            cls.temperature = 1.0
        convert_dropouts(model, self.ue_args)
        activate_mc_dropout(model, activate=True, random=self.ue_args.inference_prob)

        if self.ue_args.use_cache:
            model.enable_cache()

        log.info("**************Construct DPP kernel ********************")
        freeze_all_dpp_dropouts(model, freeze=True)
        dry_run_dataset = (
            X_test if self.dry_run_dataset == "eval" else self.dry_run_dataset
        )
        cls.predict(dry_run_dataset)
        freeze_all_dpp_dropouts(model, freeze=False)
        log.info("**************Done.***********************************")

        X_contruct = X_test if X_test is not None else X
        if self.ue_args.dropout.is_reused_mask:
            if self.ue_args.dropout.use_ood_sampling:
                self._construct_dpp_ood_masks(X_contruct, y)
            else:
                self._construct_dpp2_masks(X_contruct, y)
            if self.ue_args.calibrate:
                self._calibrate_constructed_dpp_masks(self.calibration_dataset)
                
        if self.ue_args.calibrate and not self.ue_args.dropout.is_reused_mask:
            log.info('**************** Calibrating UE masks ****************')
            cls.predict(self.calibration_dataset, calibrate=True)
            log.info(f"Calibration temperature = {cls.temperature}")
            dpp_dropout = get_replaced_dropout()
            dpp_dropout.calib_temp = cls.temperature # TODO:
            cls.temperature = 1.
            log.info('*****************Done. ******************************')

        
    def _calibrate_constructed_dpp_masks(self, X):
        log.info("*************Calibrating constructed masks...***********************")
        cls = self.cls
        model = self.cls._auto_model
        dpp_dropout = get_replaced_dropout(model)
        dpp_dropout.calib_temps = [1.]
        
        for i in tqdm(range(dpp_dropout.diverse_masks.shape[1]-1)):
            dpp_dropout.change_mask(on_calibration=self.use_paper_version)
            cls.predict(X, calibrate=True)
            log.info(f"Calibration temperature = {cls.temperature}")
            dpp_dropout.calib_temps.append(cls.temperature)

        dpp_dropout.used_mask_id = 0
        log.info("************Done.************************************")
    
    def _construct_dpp2_masks(self, X, y):
        log.info("****************Construct DPP mask pool**************")
        dpp_start = time.time()
        
        self._predict_with_dpp_masks(X, y, committee_size=self.ue_args.committee_size)
        cls = self.cls
        model = cls._auto_model
        dpp_dropout = get_replaced_dropout(model)
        dpp_dropout.inference_step = True
        dpp_dropout.construct_pool_of_masks(sampling=True)
        
        dpp_end = time.time()
        log.info("************Done.************************************")
        
    def _construct_dpp_ood_masks(self, X, y):
        log.info("****************Select DPP mask pool on OOD dataset**************")
        dpp_start = time.time()
        
        self._predict_with_dpp_masks(X, y, committee_size=self.ue_args.committee_size)
        cls = self.cls
        model = cls._auto_model
        dpp_dropout = get_replaced_dropout(model)
        dpp_dropout.inference_step = True
        dpp_dropout.construct_pool_of_masks(sampling=False)
        dpp_dropout = get_replaced_dropout(model)
        committee_size = dpp_dropout.diverse_masks.shape[1]
        if self.use_paper_version:
            committee_size -= 1

        sampled_probabilities = []
        for i in tqdm(range(committee_size)):
            dpp_dropout.change_mask(on_calibration=self.use_paper_version)
            preds, probs = cls.predict(self.ood_dataset, calibrate=False)[:2]
            sampled_probabilities.append(probs.tolist())

        select_diverse_masks(dpp_dropout, sampled_probabilities, self.ue_args.dropout.committee_size)
        
        dpp_end = time.time()
        log.info("************Done.************************************")
    
    def _predict_with_dpp_masks(self, X, y=None, committee_size=None):
        committee_size = self.ue_args.committee_size if committee_size is None else committee_size
            
        log.info("****************Start predict**************")
        cls = self.cls
        model = cls._auto_model
        eval_results = {}
        eval_results["sampled_probabilities"] = []
        eval_results["sampled_answers"] = []
        dpp_dropout = get_replaced_dropout(model)

        for i in tqdm(range(committee_size)):
            if not(self.use_paper_version and self.ner):
                cls.temperature = dpp_dropout.get_calib_temp()
            preds, probs = cls.predict(X, calibrate=False)[:2]

            eval_results["sampled_probabilities"].append(probs.tolist())
            eval_results["sampled_answers"].append(preds.tolist())

            if self.ue_args.eval_passes:
                eval_score = eval_metric.compute(
                    predictions=preds, references=y
                )
                log.info(f"Eval score: {eval_score}")
        
        log.info("**************Done.**********************")
        return eval_results
    
    def _predict_with_selected_masks(self, X, y=None):
        log.info("****************Start predict with selected masks **************")
        cls = self.cls
        model = cls._auto_model
        eval_results = {}
        eval_results["sampled_probabilities"] = []
        eval_results["sampled_answers"] = []
        
        dpp_dropout = get_replaced_dropout(model)
        diverse_committee_size = dpp_dropout.diverse_masks.shape[1] - 1
        actual_committee_size = min(
            self.ue_args.dropout.committee_size,
            diverse_committee_size,
        )
        log.info(f'Actual committee size: {actual_committee_size}')
        dpp_temps = dpp_dropout.get_calib_temp()
        for i in tqdm(range(actual_committee_size)):
            dpp_dropout.change_mask(on_calibration=False)
            cls.temperature = dpp_dropout.get_calib_temp()
            preds, probs = cls.predict(X, calibrate=False)[:2]

            eval_results["sampled_probabilities"].append(probs.tolist())
            eval_results["sampled_answers"].append(preds.tolist())
            
            if self.ue_args.eval_passes:
                eval_score = eval_metric.compute(
                    predictions=preds, references=y
                )
                log.info(f"Eval score: {eval_score}")
                
        log.info("*********************Done. *********************************")
        
        return eval_results