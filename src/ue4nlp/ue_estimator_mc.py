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


class UeEstimatorMc:
    def __init__(self, cls, ue_args, eval_metric, calibration_dataset, train_dataset):
        self.cls = cls
        self.ue_args = ue_args
        self.calibration_dataset = calibration_dataset
        self.eval_metric = eval_metric
        self.train_dataset = train_dataset

    def fit_ue(self, X, X_test):
        log.info("**************Fitting...********************")
        log.info("**************Done.********************")

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

    def __call__(self, X, y=None):
        return self._predict(X, y)

    def _predict(self, X, y):
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
            if ue_args.calibrate:  # TODO: what is the purpose of calibration here?
                self.cls.predict(self.calibration_dataset, calibrate=True)
                log.info(f"Calibration temperature = {self.cls.temperature}")

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
        log.info(f"UE time: {end - start}")
        eval_results["ue_time"] = end - start
        model = self._deactivate_dropouts(model)
        return eval_results
