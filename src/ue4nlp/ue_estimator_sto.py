from tqdm import tqdm
import time

import logging

log = logging.getLogger()



class UeEstimatorSTO:
    def __init__(self, cls, ue_args, eval_metric, calibration_dataset, train_dataset):
        self.cls = cls
        self.ue_args = ue_args
        self.calibration_dataset = calibration_dataset
        self.eval_metric = eval_metric
        self.train_dataset = train_dataset

    def __call__(self, eval_dataset, true_labels=None):
        ue_args = self.ue_args
        eval_metric = self.eval_metric
        model = self.cls._auto_model

        start = time.time()
        log.info("******Perform stochastic inference...*******")

        if ue_args.use_cache:
            log.info("Caching enabled.")
            model.enable_cache()

        eval_results = {}
        eval_results["sampled_probabilities"] = []
        eval_results["sampled_answers"] = []

        log.info("****************Start runs**************")

        for i in tqdm(range(ue_args.committee_size)):
            if ue_args.calibrate:  # TODO: what is the purpose of calibration here?
                self.cls.predict(self.calibration_dataset, calibrate=True)
                log.info(f"Calibration temperature = {self.cls.temperature}")

            preds, probs = self.cls.predict(eval_dataset)[:2]

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

        return eval_results
