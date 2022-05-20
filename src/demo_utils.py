""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
from dataclasses import dataclass, field
from typing import Optional
import json
import numpy as np
from pathlib import Path
import random
import torch
import hydra
import pickle
import yaml
from copy import deepcopy

from utils.utils_wandb import init_wandb, wandb

from ue4nlp.dropconnect_mc import (
    LinearDropConnectMC,
    activate_mc_dropconnect,
    convert_to_mc_dropconnect,
    hide_dropout,
)
from ue4nlp.dropout_mc import activate_mc_dropout, convert_to_mc_dropout
from ue4nlp.text_classifier import TextClassifier

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    set_seed,
    ElectraForSequenceClassification,
)
from datasets import load_metric, load_dataset
from sklearn.model_selection import train_test_split

from ue4nlp.ue_estimator_mc import UeEstimatorMc, convert_dropouts
from ue4nlp.ue_estimator_sngp import UeEstimatorSngp
from ue4nlp.ue_estimator_mcdpp import UeEstimatorMcDpp
from ue4nlp.ue_estimator_nuq import UeEstimatorNUQ
from ue4nlp.ue_estimator_mahalanobis import UeEstimatorMahalanobis
from ue4nlp.ue_estimator_mc_mahalanobis import UeEstimatorMcMahalanobis
from ue4nlp.ue_estimator_msd import UeEstimatorMSD

from utils.utils_data import preprocess_function
from utils.utils_models import create_model

from utils.hyperparameter_search import get_optimal_hyperparameters
from utils.utils_train import TrainingArgsWithLossCoefs
from utils.utils_train import get_trainer
from utils.utils_sngp import SNGPTrainer

import mlflow

from ue4nlp.transformers_regularized import (
    SelectiveTrainer, SelectiveSNGPTrainer
)

from ue4nlp.transformers_regularized import SelectiveTrainer
from utils.utils_tasks import get_config
import logging
from analyze_results import (
    extract_result,
    aggregate_runs,
    from_model_outputs_calc_rcc_auc,
    from_model_outptus_calc_rejection_table,
    format_arc_table_results,
    mean_std_str,
)
from analyze_results import (
    format_results2,
    from_model_outputs_calc_pr_auc,
    from_model_outputs_calc_rpp,
    aggregate_runs_rejection_table,
)
from ue4nlp.ue_scores import *
import pandas as pd

log = logging.getLogger(__name__)


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(task_to_keys.keys())
        },
    )
    max_seq_length: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in "
                    + ",".join(task_to_keys.keys())
                )
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


def calculate_dropouts(model):
    res = 0
    for i, layer in enumerate(list(model.children())):
        # module_name = list(model._modules.items())[i][0]
        layer_name = layer._get_name()
        if layer_name == "Dropout":
            res += 1
        else:
            res += calculate_dropouts(model=layer)

    return res


def compute_metrics(is_regression, metric, accuracy_metric, p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    result = metric.compute(predictions=preds, references=p.label_ids)
    accuracy = accuracy_metric.compute(predictions=preds, references=p.label_ids)
    result.update(accuracy)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()

    return result


def reset_params(model: torch.nn.Module):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        else:
            reset_params(model=layer)


def load_ood_dataset(dataset_path, max_seq_length, tokenizer, cache_dir=None):
    log.info("Load out-of-domain dataset.")
    datasets_ood = load_dataset(
        dataset_path, ignore_verifications=True, cache_dir=cache_dir
    )
    log.info("Done with loading the dataset.")

    log.info("Preprocessing the dataset...")
    sentence1_key, sentence2_key = ("text", None)

    f_preprocess = lambda examples: preprocess_function(
        None, sentence1_key, sentence2_key, tokenizer, max_seq_length, examples
    )

    datasets_ood = datasets_ood.map(
        f_preprocess,
        batched=True,
        load_from_cache_file=True,  # TODO: add config
    )

    ood_dataset = datasets_ood["test"].select(
        list(range(1000))
    )  # TODO: What is this ???
    # TODO: Why to take test dataset, we can take train dataset
    log.info("Done with preprocessing the dataset.")

    return ood_dataset


def create_ue_estimator(
    model,
    ue_args,
    eval_metric,
    calibration_dataset,
    train_dataset,
    cache_dir,
    config=None,
):
    if ue_args.ue_type == "sngp":
        return UeEstimatorSngp(model, ue_args, eval_metric)

    elif ue_args.ue_type == "mc" or ue_args.ue_type == "mc-dc":
        return UeEstimatorMc(
            model, ue_args, eval_metric, calibration_dataset, train_dataset
        )

    elif ue_args.ue_type == "mc-dpp":
        if ue_args.dropout.dry_run_dataset == "eval":
            dry_run_dataset = "eval"
        elif ue_args.dropout.dry_run_dataset == "train":
            dry_run_dataset = train_dataset
        elif ue_args.dropout.dry_run_dataset == "val":
            dry_run_dataset = calibration_dataset
        else:
            raise ValueError()

        ood_dataset = None
        if ue_args.dropout.use_ood_sampling:
            ood_dataset = load_ood_dataset(
                "imdb", model._max_len, model.tokenizer, cache_dir
            )  # TODO: configure

        return UeEstimatorMcDpp(
            model,
            ue_args,
            eval_metric,
            calibration_dataset,
            dry_run_dataset,
            ood_dataset=ood_dataset,
        )
    elif ue_args.ue_type == "nuq":
        return UeEstimatorNUQ(
            model, ue_args, config, train_dataset, calibration_dataset
        )
    elif ue_args.ue_type == "maha":
        return UeEstimatorMahalanobis(model, ue_args, config, train_dataset)
    elif ue_args.ue_type == "mc_maha":
        return UeEstimatorMcMahalanobis(model, config, ue_args, train_dataset)
    elif ue_args.ue_type == "msd":
        return UeEstimatorMSD(
            model, config, ue_args, eval_metric, calibration_dataset, train_dataset
        )
    else:
        raise ValueError()


def do_predict_eval(
    model,
    tokenizer,
    trainer,
    eval_dataset,
    train_dataset,
    calibration_dataset,
    eval_metric,
    config,
    work_dir,
    model_dir,
    metric_fn,
    max_len,
):
    eval_results = {}

    true_labels = [example["label"] for example in eval_dataset]
    eval_results["true_labels"] = true_labels

    cls = TextClassifier(
        model,
        tokenizer,
        training_args=config.training,
        trainer=trainer,
        max_len=max_len,
    )

    if config.do_eval:
        if config.ue.calibrate:
            cls.predict(calibration_dataset, calibrate=True)
            log.info(f"Calibration temperature = {cls.temperature}")

        log.info("*** Evaluate ***")

        res = cls.predict(eval_dataset)
        preds, probs = res[:2]

        eval_score = eval_metric.compute(predictions=preds, references=true_labels)

        log.info(f"Eval score: {eval_score}")
        eval_results["eval_score"] = eval_score
        eval_results["probabilities"] = probs.tolist()
        eval_results["answers"] = preds.tolist()

    if config.do_ue_estimate:
        dry_run_dataset = None

        ue_estimator = create_ue_estimator(
            cls,
            config.ue,
            eval_metric,
            calibration_dataset=calibration_dataset,
            train_dataset=train_dataset,
            cache_dir=config.cache_dir,
            config=config,
        )

        ue_results = ue_estimator(eval_dataset, true_labels)
        eval_results.update(ue_results)

    with open(Path(work_dir) / "dev_inference.json", "w") as res:
        json.dump(eval_results, res)

    if wandb.run is not None:
        wandb.save(str(Path(work_dir) / "dev_inference.json"))


def do_eval(
    model,
    tokenizer,
    trainer,
    eval_dataset,
    train_dataset,
    calibration_dataset,
    eval_metric,
    config,
    work_dir,
    model_dir,
    metric_fn,
    max_len,
):
    eval_results = {}

    true_labels = [example["label"] for example in eval_dataset]
    eval_results["true_labels"] = true_labels

    cls = TextClassifier(
        model,
        tokenizer,
        training_args=config.training,
        trainer=trainer,
        max_len=max_len,
    )

    if config.do_eval:
        if config.ue.calibrate:
            cls.predict(calibration_dataset, calibrate=True)
            log.info(f"Calibration temperature = {cls.temperature}")

        log.info("*** Evaluate ***")

        res = cls.predict(eval_dataset)
        preds, probs = res[:2]

        eval_score = eval_metric.compute(predictions=preds, references=true_labels)

        log.info(f"Eval score: {eval_score}")
        eval_results["eval_score"] = eval_score
        eval_results["probabilities"] = probs.tolist()
        eval_results["answers"] = preds.tolist()
    return cls, eval_metric, calibration_dataset, train_dataset, eval_dataset, eval_results, work_dir


def train_eval_glue_model(config, training_args, data_args, work_dir, eval_before_ue=False):
    ue_args = config.ue
    model_args = config.model

    log.info(f"Seed: {config.seed}")
    set_seed(config.seed)
    random.seed(config.seed)
    training_args.seed = config.seed

    ############### Loading dataset ######################

    log.info("Load dataset.")
    datasets = load_dataset("glue", config.data.task_name, cache_dir=config.cache_dir)
    log.info("Done with loading the dataset.")

    # Labels
    if data_args.task_name is not None:
        label_list = datasets["train"].features["label"].names
    else:
        label_list = datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism

    num_labels = len(label_list)
    log.info(f"Number of labels: {num_labels}")

    ################ Loading model #######################

    model, tokenizer = create_model(num_labels, model_args, data_args, ue_args, config)

    ################ Preprocessing the dataset ###########

    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
            }
        else:
            log.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    f_preprocess = lambda examples: preprocess_function(
        label_to_id, sentence1_key, sentence2_key, tokenizer, max_seq_length, examples
    )

    datasets = datasets.map(
        f_preprocess,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    datasets = datasets.remove_columns("idx")

    ################### Training ####################################
    if config.reset_params:
        reset_params(model)

    if ue_args.dropout_type == "DC_MC":
        convert_dropouts(model, ue_args)

    train_dataset = datasets["train"]
    train_indexes = list(range(len(train_dataset)))
    calibration_dataset = None
    if (
        config.do_train
        or config.ue.calibrate
        or config.ue.ue_type == "maha"
        or (
            config.ue.dropout_type == "DPP"
            and config.ue.dropout.dry_run_dataset == "train"
        )
    ):
        # train_dataset = datasets["train"]
        # train_indexes = list(range(len(train_dataset)))

        if config.data.subsample_perc > 0:
            train_indexes = random.sample(
                train_indexes, int(len(train_indexes) * config.data.subsample_perc)
            )

        if config.data.validation_subsample > 0:
            train_indexes, calibration_indexes = train_test_split(
                train_indexes,
                test_size=config.data.validation_subsample,
                random_state=config.data.validation_seed,
            )
        else:
            calibration_indexes = train_indexes

        calibration_dataset = torch.utils.data.Subset(
            train_dataset, calibration_indexes
        )
        train_dataset = torch.utils.data.Subset(train_dataset, train_indexes)

        with open(Path(work_dir) / "calibration_indexes.pkl", "wb") as f:
            pickle.dump(calibration_indexes, f)
        with open(Path(work_dir) / "training_indexes.pkl", "wb") as f:
            pickle.dump(train_indexes, f)

        log.info(f"Training dataset size: {len(train_dataset)}")
        log.info(f"Calibration dataset size: {len(calibration_dataset)}")

    elif (
        config.ue.dropout_type == "DPP" and config.ue.dropout.dry_run_dataset != "eval"
    ):
        training_indexes_path = (
            Path(config.model.model_name_or_path) / "training_indexes.pkl"
        )
        with open(training_indexes_path, "rb") as f:
            train_indexes = pickle.load(f)

        train_dataset = torch.utils.data.Subset(train_dataset, train_indexes)
        log.info(f"Training dataset size: {len(train_dataset)}")

    eval_dataset = (
        datasets["validation"] if config.do_eval or config.do_ue_estimate else None
    )

    metric = load_metric(
        "glue", data_args.task_name, keep_in_memory=True, cache_dir=config.cache_dir
    )
    accuracy_metric = load_metric("accuracy", keep_in_memory=True, cache_dir=config.cache_dir)
    is_regression = False
    metric_fn = lambda p: compute_metrics(is_regression, metric, accuracy_metric, p)

    training_args.save_steps = 0
    if config.do_train:
        training_args.warmup_steps = int(
            training_args.warmup_ratio  # TODO:
            * len(train_dataset)
            * training_args.num_train_epochs
            / training_args.train_batch_size
        )
        log.info(f"Warmup steps: {training_args.warmup_steps}")
        training_args.logging_steps = training_args.warmup_steps
        training_args.weight_decay_rate = training_args.weight_decay

    use_sngp = ue_args.ue_type == "sngp"
    use_selective = "use_selective" in ue_args.keys() and ue_args.use_selective
    #################### Training ##########################
    trainer = get_trainer(
        "cls",
        use_selective,
        use_sngp,
        model,
        training_args,
        train_dataset,
        eval_dataset,
        metric_fn
    )
    if config.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        # Rewrite the optimal hyperparam data if we want the evaluation metrics of the final trainer
        if config.do_eval:
            evaluation_metrics = trainer.evaluate()
        trainer.save_model(work_dir)
        tokenizer.save_pretrained(work_dir)

    #################### Predicting ##########################

    if eval_before_ue:
        if config.do_eval or config.do_ue_estimate:
            return do_eval(
                       model,
                       tokenizer,
                       trainer,
                       eval_dataset,
                       train_dataset,
                       calibration_dataset,
                       metric,
                       config,
                       work_dir,
                       model_args.model_name_or_path,
                       metric_fn,
                       max_seq_length,
                   )
    else:
        if config.do_eval or config.do_ue_estimate:
            do_predict_eval(
                model,
                tokenizer,
                trainer,
                eval_dataset,
                train_dataset,
                calibration_dataset,
                metric,
                config,
                work_dir,
                model_args.model_name_or_path,
                metric_fn,
                max_seq_length,
            )


def update_config(cfg_old, cfg_new):
    for k, v in cfg_new.items():
        if k in cfg_old.__dict__:
            setattr(cfg_old, k, v)

    return cfg_old


def fix_config(config):
    if config.ue.dropout_subs == "all":
        config.ue.use_cache = False

    if config.ue.ue_type == "mc-dpp":
        config.ue.dropout_type = "DPP"

    if config.ue.ue_type == "mc-dc":
        config.ue.dropout_type = "DC_MC"

    if config.ue.reg_type == "metric":
        config.ue.use_cache = False


def setup_dirs(config):
    """Set dirs from config, because now we doesn't use hydra. Only for demo notebook"""
    # model_save_dir
    model_save_dir = os.path.join(os.path.abspath(config.model_dir),
                                  config.model.model_name_or_path,
                                  config.data.task_name)
    # output_dir
    output_dir = os.path.join(os.path.abspath(config.output_dir), config.data.task_name)
    # cache_dir
    cache_dir = os.path.abspath(config.cache_dir)
    config.model_dir = model_save_dir
    config.output_dir = output_dir
    config.cache_dir = cache_dir
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.cache_dir, exist_ok=True)
    return config


def preproc_config(config, init=False):
    # disable wandb
    os.environ["WANDB_WATCH"] = "False"
    os.environ["WANDB_DISABLED"] = "true"
    auto_generated_dir = os.getcwd()
    if init:
        # setup work dirs
        config = setup_dirs(config)
    #model_dir = os.path.join(auto_generated_dir, config.model_dir)
    # assume that we rewrite model if exists
    #os.makedirs(model_dir, exist_ok=True)
    #os.chdir(model_dir)
    # common training procedure
    fix_config(config)
    args_train = TrainingArgsWithLossCoefs(
        output_dir=auto_generated_dir,
        reg_type=config.ue.get("reg_type", "reg-curr"),
        lamb=config.ue.get("lamb", 0.01),
        margin=config.ue.get("margin", 0.05),
        lamb_intra=config.ue.get("lamb_intra", 0.01),
        unc_threshold=config.ue.get("unc_threshold", 0.5),
    )
    args_train = update_config(args_train, config.training)

    args_data = DataTrainingArguments(task_name=config.data.task_name)
    args_data = update_config(args_data, config.data)
    return config, args_train, args_data


def train_model(config):
    config, args_train, args_data = preproc_config(config)
    model_postfix = config.get("model_postfix", "")
    model_save_dir = os.path.join(config.model_dir, model_postfix, str(config.seed))
    os.makedirs(model_save_dir, exist_ok=True)
    train_eval_glue_model(config, args_train, args_data, model_save_dir)


def train_model_ensemble(config, seeds, num_models):
    """Like train model, but for ensemble"""
    for idx in range(num_models):
        seeds = [idx + seed for seed in seeds]
        for seed in seeds:
            config.seed = seed
            config, args_train, args_data = preproc_config(config)
            model_save_dir = os.path.join(config.model_dir, 'ensemble', str(idx), str(config.seed))
            os.makedirs(model_save_dir, exist_ok=True)
            train_eval_glue_model(config, args_train, args_data, model_save_dir)


def eval_model(config, ue_type, args_train, args_data):
    model_postfix = config.get("model_postfix", "")
    model_save_dir = os.path.join(config.model_dir, model_postfix, str(config.seed))
    config.model.model_name_or_path = model_save_dir
    # also add method prefix for results saving
    model_postfix = config.get("model_postfix", "")
    mc_save_dir = os.path.join(config.output_dir, model_postfix, ue_type, str(config.seed))
    os.makedirs(mc_save_dir, exist_ok=True)
    return train_eval_glue_model(config, args_train, args_data, mc_save_dir, eval_before_ue=True)


def calc_mc_dropout(config):
    config, args_train, args_data = preproc_config(config)
    # set config params for MC dropout
    config.ue.ue_type = 'mc'
    config.ue.dropout_type = 'MC'
    config.ue.inference_prob = 0.1
    config.ue.committee_size = 20
    config.ue.dropout_subs = 'all'
    config.ue.use_cache = True
    config.ue.eval_passes = False
    config.ue.calibrate = False
    config.ue.use_selective = False
    # also change model_path to path to saved model
    model_save_dir = os.path.join(config.model_dir, str(config.seed))
    config.model.model_name_or_path = model_save_dir
    # also add method prefix for results saving
    mc_save_dir = os.path.join(config.output_dir, 'mc', str(config.seed))
    os.makedirs(mc_save_dir, exist_ok=True)
    # and now calc ue for model
    train_eval_glue_model(config, args_train, args_data, mc_save_dir)


def calc_mahalanobis(config):
    config, args_train, args_data = preproc_config(config)
    # set config params for Mahalanobis
    config.ue.ue_type = 'maha'
    config.ue.dropout_type = ''
    config.ue.dropout_subs = ''
    config.ue.use_cache = True
    config.ue.eval_passes = False
    config.ue.calibrate = False
    config.ue.use_selective = False
    # also change model_path to path to saved model
    model_save_dir = os.path.join(config.model_dir, str(config.seed))
    config.model.model_name_or_path = model_save_dir
    maha_save_dir = os.path.join(config.output_dir, 'maha', str(config.seed))
    os.makedirs(maha_save_dir, exist_ok=True)
    # and now calc ue for model
    train_eval_glue_model(config, args_train, args_data, maha_save_dir)


def accumulate_results(results_dir, final_dir):
    final_result = {
        "true_labels": [],
        "probabilities": [],
        "answers": [],
        "sampled_probabilities": [],
        "sampled_answers": [],
    }

    for seed in os.listdir(results_dir):
        results_file_path = Path(results_dir) / seed / "dev_inference.json"
        with open(results_file_path) as f:
            result = json.load(f)

        final_result["sampled_probabilities"].append(result["probabilities"])
        final_result["sampled_answers"].append(result["answers"])

    final_result["true_labels"] = result["true_labels"]
    final_result["answers"] = result["answers"]
    final_result["probabilities"] = result["probabilities"]

    with open(Path(final_dir) / "dev_inference.json", "w") as f:
        json.dump(final_result, f)


def calc_ensemble(config, seeds):
    # here we simply stack model predictions
    config, args_train, args_data = preproc_config(config)
    # we assume that we have several trained and evaluated models
    # so we just copy saved models output and stack it
    # get dir with models by seeds
    for idx in os.listdir(os.path.join(config.model_dir, 'ensemble')):
        results_dir = os.path.join(config.model_dir, 'ensemble', idx)
        ensemble_save_dir = os.path.join(config.output_dir, 'ensemble', idx)
        os.makedirs(ensemble_save_dir, exist_ok=True)
        accumulate_results(results_dir, ensemble_save_dir)


def choose_metric(metric_type):
    if metric_type in ["rejection-curve-auc", "roc-auc"]:
        return metric_type

    elif metric_type == "rcc-auc":
        return from_model_outputs_calc_rcc_auc

    elif metric_type == "pr-auc":
        return from_model_outputs_calc_pr_auc

    elif metric_type == "rpp":
        return from_model_outputs_calc_rpp

    elif metric_type == "table_accuracy":
        return from_model_outptus_calc_rejection_table

    elif metric_type == "table_f1_macro":
        return partial(from_model_outptus_calc_rejection_table,
                       metric=partial(f1_score, average="macro"))
    elif metric_type == "table_f1_micro":
        return partial(from_model_outptus_calc_rejection_table,
                       metric=partial(f1_score, average="micro"))
    else:
        raise ValueError("Wrong metric type!")


def simple_aggregate_runs(data_path, methods, metric, task_type="classification", oos=False, avg_type='sum'):
    results = []
    model_results = []
    level = None
    for model_seed in os.listdir(data_path):
        try:
            model_seed_int = int(model_seed)
        except:
            if model_seed == "results":
                pass
            else:
                continue

        model_path = Path(data_path) / model_seed

        model_results = []

        #if "ensemble" in str(model_path):
        #    model_path = model_path / "models"
        #    model_path = list(model_path.iterdir())[0]

        run_dir = model_path
        try:
            if task_type == "classification":
                model_results.append(
                    extract_result(run_dir, methods=methods, metric=metric, oos=oos)
                )
            else:
                level = task_type.split("-")[1]
                model_results.append(
                    extract_result_ner(
                        run_dir, methods=methods, metric=metric, level=level, avg_type=avg_type
                    )
                )
        except FileNotFoundError:
            pass
        except:
            continue

        log.info(f"N runs: {len(model_results)}")
        # print(pd.DataFrame.from_dict(model_results, orient='columns'))
        # print("N runs:", len(model_results))
        model_avg_res = pd.DataFrame.from_dict(
            model_results, orient="columns"
        )  # .mean(axis=0)
        # results.append(model_avg_res.to_dict())
        results.append(model_avg_res)

    results = pd.concat(results, axis=0)
    if level is not None:
        # ner case
        # TODO: changed df structure - now we calc mean by all exps, not by all models. Fix or add switch
        results = results.reset_index(drop=True)
    return results


def improvement_over_baseline(
    results,
    baseline_col,
    baseline=None,
    metric="roc-auc",
    subtract=False,
    percents=False,
):
    if baseline is None:
        baseline = results[baseline_col]
        if subtract:
            diff_res = results.drop(columns=baseline_col).subtract(
                baseline, axis="rows"
            )
        else:
            diff_res = results.drop(columns=baseline_col)
    else:
        baseline_raw = baseline[metric]
        baseline = results[baseline_col]
        if subtract:
            diff_res = results.drop(columns=baseline_col) - baseline_raw.values[0]
        else:
            diff_res = results.drop(columns=baseline_col)

    ndp = 2
    formatted_results = format_results2(diff_res, percents=percents, ndp=ndp)

    if percents:
        baseline_percent = baseline * 100
    else:
        baseline_percent = baseline
    formatted_results.loc["baseline (max_prob)"] = mean_std_str(
        [baseline_percent.mean(), baseline_percent.std()], ndp
    )
    formatted_results.loc["count"] = baseline_percent.shape[0]
    return formatted_results


def get_table(config, results_path):
    metrics = ['rpp', 'rcc-auc']
    results = {key: [] for key in metrics}
    config, args_train, args_data = preproc_config(config)
    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    #os.chdir(hydra.utils.get_original_cwd())

    default_methods = {
        "bald": bald,
        "sampled_max_prob": sampled_max_prob,
        "variance": probability_variance,
        #"sampled_entropy": mean_entropy,
        #"var_ratio": var_ratio,
    }
    # TODO: switch if use maha
    if "maha" in results_path or "mixup" in results_path:
        maha_dist = lambda x: np.squeeze(np.squeeze(x, axis=-1), axis=-1)
        maha_dist = lambda x: np.squeeze(x[:, 0], axis=-1)
        default_methods = {"mahalanobis_distance": maha_dist}
        if "mixup" in results_path:
            default_methods = {"mixup": maha_dist}
        if "maha_mc" in results_path or "maha_sn_mc" in results_path:
            # Maha MC case
            sm_maha_dist = lambda x: np.squeeze(x[:, 1:], axis=-1).max(1)
            default_methods["sampled_mahalanobis_distance"] = sm_maha_dist
    # TODO: same for NUQ
    if "nuq" in results_path:
        nuq_aleatoric = lambda x: np.squeeze(x[0], axis=-1)
        nuq_epistemic = lambda x: np.squeeze(x[1], axis=-1)
        nuq_total = lambda x: np.squeeze(x[2], axis=-1)
        default_methods = {
            "nuq_aleatoric": nuq_aleatoric,
            "nuq_epistemic": nuq_epistemic,
            "nuq_total": nuq_total,
        }
    for metric_type in metrics:
        log.info(f"Metric: {metric_type}")

        metric = choose_metric(metric_type=metric_type)

        agg_res = simple_aggregate_runs(
            results_path, methods=default_methods, metric=metric
        )

        agg_res = agg_res.reset_index(drop=True)
        metric_path = Path(auto_generated_dir) / f"metrics_{metric_type}.json"
        #with open(metric_path, "w") as f:
        #    f.write(agg_res.to_json())

        if agg_res.empty:
            log.info("Broken\n")
            continue

        if metric_type == "rcc-auc":
            final_score = improvement_over_baseline(
                agg_res, baseline_col="max_prob", subtract=False, percents=False
            )
        elif metric_type == "rpp":
            final_score = improvement_over_baseline(
                agg_res, baseline_col="max_prob", subtract=False, percents=True
            )
        else:
            final_score = improvement_over_baseline(
                agg_res, baseline_col="max_prob", subtract=True, percents=True
            )

        results[metric_type] = final_score
        log.info("\n" + str(final_score))
        log.info("\n")
    return results
