""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import sys
import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path
import random
import torch
import hydra
import pickle

from utils.utils_wandb import init_wandb, wandb

from ue4nlp.transformers_cached import (
    ElectraForSequenceClassificationCached,
    BertForSequenceClassificationCached,
    ElectraForTokenClassificationCached,
)
from ue4nlp.dropconnect_mc import (
    LinearDropConnectMC,
    activate_mc_dropconnect,
    convert_to_mc_dropconnect,
    hide_dropout,
)
from ue4nlp.dropout_mc import DropoutMC, activate_mc_dropout, convert_to_mc_dropout
from ue4nlp.dropout_dpp import DropoutDPP, DropoutDPP_v2
from ue4nlp.sequence_tagger import SequenceTagger
from utils.utils_heads import ElectraNERHeadCustom

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    PretrainedConfig,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
    ElectraForTokenClassification,
)

from ue4nlp.ue_estimator_mc import UeEstimatorMc, convert_dropouts
from ue4nlp.ue_estimator_sngp import UeEstimatorSngp
from ue4nlp.ue_estimator_mcdpp import UeEstimatorMcDpp
from ue4nlp.ue_estimator_nuq import UeEstimatorNUQ
from ue4nlp.ue_estimator_mahalanobis import UeEstimatorMahalanobis

from datasets import load_metric, load_dataset, concatenate_datasets

from utils.utils_dropout import set_last_dropout, get_last_dropout, set_last_dropconnect
import ue4nlp.alpaca_calibrator as calibrator

import logging

log = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(
        default="ner", metadata={"help": "The name of the task (ner, pos...)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    dataset_name: Optional[str] = field(
        default="conll2003",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a csv or JSON file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to predict on (a csv or JSON file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
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
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={
            "help": "Whether to return all the entity levels during evaluation or just the overall ones."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


def compute_metrics(p, metric, return_entity_level_metrics=False):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    labels = labels.reshape(predictions.shape)
    label_list = [
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-MISC",
        "I-MISC",
    ]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


def create_model(num_labels, model_args, data_args, ue_args, config):

    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=config.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=config.cache_dir,
        use_fast=True,
    )

    if ue_args.use_cache:
        if "electra" in model_args.model_name_or_path:  # TODO:
            model = ElectraForTokenClassificationCached.from_pretrained(
                model_args.model_name_or_path,
                from_tf=False,
                config=model_config,
                cache_dir=config.cache_dir,
            )
            model.use_cache = True
            model.classifier = ElectraNERHeadCustom(model)
            log.info("Replaced ELECTRA's head")
        else:
            raise ValueError(
                f"{model_args.model_name_or_path} does not have a cached option."
            )

    else:
        if "electra" in model_args.model_name_or_path:
            model = ElectraForTokenClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=False,
                config=model_config,
                cache_dir=config.cache_dir,
            )
            model.classifier = ElectraNERHeadCustom(model)
            log.info("Replaced ELECTRA's head")

    return model, tokenizer


def load_ood_dataset(dataset_path, data_args, tokenizer, cache_dir=None):
    log.info("Load out-of-domain dataset.")
    datasets_ood = load_dataset(
        dataset_path, ignore_verifications=True, cache_dir=cache_dir
    )
    log.info("Done with loading the dataset.")

    log.info("Preprocessing the dataset...")

    text_column_name, label_column_name = "tokens", "ner_tags"
    label_to_id = {0: 0}
    f_preprocess = lambda examples: tokenize_and_align_labels(
        tokenizer,
        examples,
        text_column_name,
        label_column_name,
        data_args=data_args,
        label_to_id=label_to_id,
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
    ood_dataset = ood_dataset.remove_columns(["text", "label"])
    log.info("Done with preprocessing the dataset.")

    return ood_dataset


def load_ood_dataset_test(dataset_path, data_args, tokenizer, cache_dir=None):
    log.info("Load out-of-domain dataset.")
    datasets_ood = load_dataset(
        dataset_path, "en", ignore_verifications=True, cache_dir=cache_dir
    )
    log.info("Done with loading the dataset.")

    log.info("Preprocessing the dataset...")

    column_names = datasets_ood["train"].column_names
    features = datasets_ood["train"].features

    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = "ner_tags" if "ner_tags" in column_names else column_names[1]

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    label_list = features[label_column_name].feature.names
    label_to_id = {i: 0 for i in range(len(label_list))}
    num_labels = len(label_list)

    f_preprocess = lambda examples: tokenize_and_align_labels(
        tokenizer,
        examples,
        text_column_name,
        label_column_name,
        data_args=data_args,
        label_to_id=label_to_id,
    )

    datasets_ood = datasets_ood.map(
        f_preprocess,
        batched=True,
        load_from_cache_file=True,  # TODO: add config
    )

    ood_dataset = datasets_ood["train"].select(list(range(3000)))
    ood_dataset = ood_dataset.remove_columns(["langs", "spans"])
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
    data_args=None,
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
                "imdb", data_args, model._bpe_tokenizer, cache_dir
            )

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
    else:
        raise ValueError()


def do_predict_eval(
    model,
    tokenizer,
    trainer,
    eval_dataset,
    validation_dataset,
    train_dataset,
    metric,
    config,
    data_args,
    work_dir,
    model_dir,
    metric_fn,
):
    if config.ue.use_cache:
        model.enable_cache()

    tagger = SequenceTagger(
        model, tokenizer, training_args=config.training, trainer=trainer
    )
    eval_results = {}

    ood_dataset = load_ood_dataset_test(
        "wikiann", data_args, tokenizer, config.cache_dir
    )

    eval_dataset = eval_dataset.remove_columns(["id", "pos_tags", "chunk_tags"])
    validation_dataset = validation_dataset.remove_columns(
        ["id", "pos_tags", "chunk_tags"]
    )

    eval_results["ood_labels"] = [0] * len(eval_dataset) + [1] * len(ood_dataset)

    ood_dataset.cast(eval_dataset.features)
    eval_dataset = concatenate_datasets([eval_dataset, ood_dataset])

    true_labels = [example["labels"] for example in eval_dataset]
    eval_results["true_labels"] = true_labels

    if config.do_eval:
        if config.ue.calibrate:
            tagger.predict(validation_dataset, calibrate=True)
            log.info(f"Calibration temperature = {tagger.temperature}")

        log.info("*** Evaluate ***")

        res = tagger.predict(eval_dataset)
        preds, probs = res[:2]

        eval_score = metric_fn([probs, np.asarray(true_labels)])

        log.info(f"Eval score: {eval_score}")
        eval_results["eval_score"] = eval_score
        eval_results["probabilities"] = probs.tolist()
        eval_results["answers"] = preds.tolist()

    if config.do_ue_estimate:
        dry_run_dataset = None

        ue_estimator = create_ue_estimator(
            tagger,
            config.ue,
            metric,
            calibration_dataset=validation_dataset,
            train_dataset=train_dataset,
            cache_dir=config.cache_dir,
            config=config,
            data_args=data_args,
        )

        ue_results = ue_estimator(eval_dataset, true_labels)
        eval_results.update(ue_results)

    with open(Path(work_dir) / "dev_inference.json", "w") as res:
        json.dump(eval_results, res)

    if wandb.run is not None:
        wandb.save(str(Path(work_dir) / "dev_inference.json"))


def tokenize_and_align_labels(
    tokenizer,
    examples,
    text_column_name,
    label_column_name,
    data_args,
    label_to_id,
    padding="max_length",
):
    if text_column_name not in examples:
        examples[text_column_name] = [exp.split(" ") for exp in examples["text"]]
        examples[label_column_name] = [
            [0] * len(exp.split(" ")) for exp in examples["text"]
        ]

    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=padding,
        max_length=data_args.max_seq_length,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(
                    label_to_id[label[word_idx]] if data_args.label_all_tokens else -100
                )

            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def train_eval_conll2003_model(config, training_args, data_args, work_dir):
    ue_args = config.ue
    model_args = config.model

    log.info(f"Seed: {config.seed}")
    set_seed(config.seed)
    random.seed(config.seed)

    log.info("Load dataset.")
    datasets = load_dataset(config.data.task_name, cache_dir=config.cache_dir)
    log.info("Done with loading the dataset.")

    if config.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features

    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = "ner_tags" if "ner_tags" in column_names else column_names[1]

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    label_list = features[label_column_name].feature.names
    label_to_id = {i: i for i in range(len(label_list))}
    num_labels = len(label_list)

    model, tokenizer = create_model(num_labels, model_args, data_args, ue_args, config)

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    f_preprocess = lambda examples: tokenize_and_align_labels(
        tokenizer, examples, text_column_name, label_column_name, data_args, label_to_id
    )
    datasets = datasets.map(
        f_preprocess,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    columns_to_return = ["input_ids", "labels", "attention_mask"]
    datasets.set_format(columns=columns_to_return)

    train_dataset = None
    if config.do_train or (
        config.ue.dropout_type == "DPP" and config.ue.dropout.dry_run_dataset != "eval"
    ):
        train_dataset = datasets["train"]

    train_indexes = None
    if config.do_train:
        train_indexes = list(range(len(train_dataset)))

        if config.data.subsample_perc > 0:
            train_indexes = random.sample(
                train_indexes, int(len(train_dataset) * config.data.subsample_perc)
            )
            train_dataset = torch.utils.data.Subset(train_dataset, train_indexes)

        with open(Path(work_dir) / "training_indexes.pkl", "wb") as f:
            pickle.dump(train_indexes, f)

        log.info(f"Training dataset size: {len(train_dataset)}")

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

    validation_dataset = datasets["validation"]
    eval_dataset = datasets["test"] if config.do_eval else None

    training_args.save_steps = 0
    if config.do_train:
        training_args.warmup_steps = int(
            training_args.warmup_ratio
            * len(train_dataset)
            * training_args.num_train_epochs
            / training_args.train_batch_size
        )
        log.info(f"Warmup steps: {training_args.warmup_steps}")

    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8, max_length=data_args.max_seq_length
    )

    metric = load_metric("seqeval", keep_in_memory=True, cache_dir=config.cache_dir)
    metric_fn = lambda p: compute_metrics(
        p, metric, data_args.return_entity_level_metrics
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metric_fn,
        data_collator=data_collator,
    )

    if config.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.save_model(work_dir)
        tokenizer.save_pretrained(work_dir)

    if config.do_eval:
        do_predict_eval(
            model,
            tokenizer,
            trainer,
            eval_dataset,
            validation_dataset,
            train_dataset,
            metric,
            config,
            data_args,
            work_dir,
            model_args.model_name_or_path,
            metric_fn,
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


@hydra.main(
    config_path=os.path.dirname(os.environ["HYDRA_CONFIG_PATH"]),
    config_name=os.path.basename(os.environ["HYDRA_CONFIG_PATH"]),
)
def main(config):
    os.environ["WANDB_WATCH"] = "False"  # To disable Huggingface logging

    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    os.chdir(hydra.utils.get_original_cwd())

    wandb_run = init_wandb(auto_generated_dir, config)

    fix_config(config)

    args_train = TrainingArguments(output_dir=auto_generated_dir)
    args_train = update_config(args_train, config.training)

    args_data = DataTrainingArguments(task_name=config.data.task_name)
    args_data = update_config(args_data, config.data)

    if not os.path.exists(Path(auto_generated_dir) / "dev_inference.json"):
        train_eval_conll2003_model(config, args_train, args_data, auto_generated_dir)
    else:
        log.info(
            f"Result file: {auto_generated_dir}/dev_inference.json already exists \n"
        )


if __name__ == "__main__":
    main()
