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
from utils.utils_sngp import SNGPTrainer

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
    DataCollatorForTokenClassification,
)
from datasets import load_metric, load_dataset
from sklearn.model_selection import train_test_split

from utils.utils_ue_estimator import create_ue_estimator
from types import SimpleNamespace

from utils.utils_data import preprocess_function
from utils.utils_models import create_model
from analyze_results import calc_rcc_aucs

from utils.hyperparameter_search import get_optimal_hyperparameters
from utils.utils_train import TrainingArgsWithLossCoefs, TrainingArgsWithMSDCoefs
from utils.utils_train import get_trainer
from utils.utils_data import split_dataset
import mlflow

from ue4nlp.transformers_regularized import (
    SelectiveTrainer, SelectiveSNGPTrainer
)
from utils.utils_tasks import get_config
from utils.utils_heads import change_attention_params
import logging

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

        ue_estimator.fit_ue(X=train_dataset, X_test=eval_dataset)
        
        ue_results = ue_estimator(eval_dataset, true_labels)
        eval_results.update(ue_results)

    with open(Path(work_dir) / "dev_inference.json", "w") as res:
        json.dump(eval_results, res)

    if wandb.run is not None:
        wandb.save(str(Path(work_dir) / "dev_inference.json"))

def train_eval_glue_model(config, training_args, data_args, work_dir):
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

    # always load eval_dataset because we use it for metrics calc
    validation_name = 'validation' if data_args.task_name!='mnli' else config.data.validation_name
    eval_dataset = (
        datasets[validation_name]
    )

    use_sngp = ue_args.ue_type == "sngp"
    use_selective = "use_selective" in ue_args.keys() and ue_args.use_selective

    metric = load_metric(
        "glue", data_args.task_name, keep_in_memory=True, cache_dir=config.cache_dir
    )
    if data_args.task_name == "cola":
        # load metric from another subtask, because cola metrics doesn't contain accuracy
        metric = load_metric(
            "glue", "mrpc", keep_in_memory=True, cache_dir=config.cache_dir
        )
    is_regression = False

    metric_glue = deepcopy(metric)
    metric = config.metric if 'metric' in config.keys() else metric
    metric_fn = lambda p: compute_metrics(is_regression, metric, accuracy_metric, p)

    #################### Hyperparameter Search ##########################
    # TODO: add train_size as a config param
    if config.do_train:
        hp_train_dataset, hp_eval_dataset = split_dataset(
            train_dataset, train_size=0.8, shuffle=True, seed=config.seed
        )
        training_args.seed=config.seed
        trainer = get_trainer(
            "cls",
            use_selective,
            use_sngp,
            model,
            training_args,
            hp_train_dataset,
            hp_eval_dataset,
            metric_fn
        )
        # TODO: add config.hyperparameter_search: bool
        # Model init fn is required for hyperperameter search
        if ue_args.ue_type == 'sngp':
            model_state_dict = deepcopy(model.state_dict())
            model_copy, _ = create_model(num_labels, model_args, data_args, ue_args, config)
            model_copy.load_state_dict(model_state_dict)
        else:
            model_copy = deepcopy(model)
        if ue_args.ue_type == "msd":
            # set another model init
            def update_model(trial, model_copy):
                mixup = SimpleNamespace(**trial.params)
                model_copy.post_init(mixup)
                return deepcopy(model_copy)

            model_init = lambda trial: update_model(trial, model_copy)
        elif ue_args.ue_type == "sto":
            def update_model(trial, model_copy):
                params = SimpleNamespace(**trial.params)
                model_copy = change_attention_params(model_copy, params, ue_args.sto_layer)
                return deepcopy(model_copy)
            model_init = lambda trial: update_model(trial, model_copy)
        elif ue_args.ue_type == 'sngp':
            def model_init(x):
                model_copy.load_state_dict(model_state_dict)
                return model_copy
        else:
            model_init = lambda x: deepcopy(model_copy)
        # Get optimal hyperparams
        optimal_hyperparams = get_optimal_hyperparameters(
            trainer, model_init, "cls", False, metric, use_sngp
        )
        # Dump them
        with open(Path(work_dir) / "optimal_hyperparameters.yaml", "w") as f:
            yaml.dump(optimal_hyperparams, f)
        # Update training args wrt optimal hyperparams
        for k, v in optimal_hyperparams.items():
            if k != "objective":
                setattr(training_args, k, v)
        trainer.args = training_args
        # End run
        mlflow.end_run()

    #################### Training ##########################
    trainer = get_trainer(
        "cls",
        use_selective,
        use_sngp,
        model,
        training_args,
        train_dataset,
        eval_dataset,
        metric_fn,
    )
    if config.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        # Rewrite the optimal hyperparam data if we want the evaluation metrics of the final trainer
        evaluation_metrics = trainer.evaluate()
        optimal_hyperparams.update(evaluation_metrics)
        with open(Path(work_dir) / "optimal_hyperparameters_eval.yaml", "w") as f:
            yaml.dump(optimal_hyperparams, f)
        trainer.save_model(work_dir)
        tokenizer.save_pretrained(work_dir)

    #################### Predicting ##########################

    if config.do_eval or config.do_ue_estimate:
        do_predict_eval(
            model,
            tokenizer,
            trainer,
            eval_dataset,
            train_dataset,
            calibration_dataset,
            metric_glue,
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


@hydra.main(
    config_path=get_config()[0],
    config_name=get_config()[1],
)
def main(config):
    os.environ["WANDB_WATCH"] = "False"  # To disable Huggingface logging
    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    os.chdir(hydra.utils.get_original_cwd())

    init_wandb(auto_generated_dir, config)

    fix_config(config)
    args_train = TrainingArgsWithLossCoefs(
        output_dir=auto_generated_dir,
        reg_type=config.ue.get("reg_type", "reg-curr"),
        lamb=config.ue.get("lamb", 0.01),
        margin=config.ue.get("margin", 0.5),
        lamb_intra=config.ue.get("lamb_intra", 0.5),
        unc_threshold=config.ue.get("unc_threshold", 0.5),
    )
    if "mixup" in config.keys() and config.mixup.use_mixup:
        # create modified args_train
        args_train = TrainingArgsWithMSDCoefs(
            output_dir=auto_generated_dir,
            mixup=config.mixup.get("mixup", True),
            self_ensembling=config.mixup.get("self_ensembling", True),
            omega=config.mixup.get("omega", 1.0),
            lam1=config.mixup.get("lam1", 1.0),
            lam2=config.mixup.get("lam2", 0.01),
        )
    args_train = update_config(args_train, config.training)

    args_data = DataTrainingArguments(task_name=config.data.task_name)
    args_data = update_config(args_data, config.data)

    if not os.path.exists(Path(auto_generated_dir) / "dev_inference.json"):
        train_eval_glue_model(config, args_train, args_data, auto_generated_dir)
    else:
        log.info(
            f"Result file: {auto_generated_dir}/dev_inference.json already exists \n"
        )

if __name__ == "__main__":
    main()
