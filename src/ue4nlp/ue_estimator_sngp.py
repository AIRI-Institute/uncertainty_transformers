import torch
import numpy as np
from tqdm import tqdm
import logging
import time
from scipy.special import softmax

log = logging.getLogger()


class UeEstimatorSngp:
    def __init__(self, cls, ue_args, eval_metric):
        self.cls = cls
        self.ue_args = ue_args
        self.eval_metric = eval_metric

    def __call__(self, X, y=None):
        return self._predict_sngp(X, y)
    
    def _predict_sngp(self, X, y):
        log.info("****************Compute logits and variance with SNGP**************")

        start = time.time()
        res = self.cls.predict(X)
        _, _, logits, stds, _, _ = res
        end = time.time()
        sum_inf_time = end - start
        
        eval_results = {}
        eval_results["logits"] = logits.tolist()
        eval_results["stds"] = stds.tolist()
        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")
        
        sngp_logits_adjusted = logits / np.sqrt(1. + (np.pi / 8.) * stds)
        sngp_probs = softmax(sngp_logits_adjusted, axis=-1)
        eval_results["stds_paper"] = sngp_probs.tolist()
        eval_results = self._randn_preds(X, y, eval_results)
        log.info("**************Done.********************")
        
        return eval_results
        
    def fit_ue(self, X, y=None, X_test=None):
        log.info("**************Fitting...********************")
        log.info("**************Done.********************")
        
    def _randn_preds(self, X, y, eval_results={}):
        
        eval_results["sampled_probabilities"] = []
        eval_results["sampled_answers"] = []

        for i in tqdm(range(self.ue_args.committee_size)):
            logits = torch.normal(
                mean=torch.Tensor(eval_results["logits"]),
                std=torch.Tensor(eval_results["stds"]),
            )
            probs = torch.nn.functional.softmax(logits, dim=1).numpy()
            preds = np.argmax(probs, axis=1)
            eval_results["sampled_probabilities"].append(probs.tolist())
            eval_results["sampled_answers"].append(preds.tolist())

            if self.ue_args.eval_passes and true_labels is not None:
                eval_score = eval_metric.compute(
                    predictions=preds, references=true_labels
                )
                log.info(f"Eval score: {eval_score}")
        
        return eval_results