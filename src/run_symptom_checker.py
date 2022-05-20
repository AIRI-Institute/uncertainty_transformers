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
import pandas as pd
import hydra
import pickle
import joblib
import yaml

from utils.utils_wandb import init_wandb, wandb

from ue4nlp.transformers_cached import (
    ElectraForSequenceClassificationCached,
    BertForSequenceClassificationCached,
)
from ue4nlp.dropconnect_mc import (
    LinearDropConnectMC,
    activate_mc_dropconnect,
    convert_to_mc_dropconnect,
    hide_dropout,
)
from ue4nlp.dropout_mc import DropoutMC, activate_mc_dropout, convert_to_mc_dropout
from ue4nlp.dropout_dpp import DropoutDPP, DropoutDPP_v2
from ue4nlp.text_classifier import TextClassifier
from ue4nlp.electra_duq_model import ElectraForSequenceClassificationDUQ
from ue4nlp.bert_sngp_model import (
    SNGPBertForSequenceClassificationCached,
    SNGPElectraForSequenceClassificationCached,
)
from utils.utils_heads import ElectraClassificationHeadCustom

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
from datasets import load_metric, load_dataset, Dataset

from utils.utils_dropout import set_last_dropout, get_last_dropout, set_last_dropconnect
import alpaca_calibrator as calibrator
from utils.utils_sngp import SNGPTrainer
import sbermed.experimental.ainesterov.symptom_checker.src.top3 as models

import logging

log = logging.getLogger(__name__)


task_to_keys = {"symptoms": ("body", None)}


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
        default=128,
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


def convert_dropouts(model, ue_args):
    if ue_args.dropout_type == "MC":
        dropout_ctor = lambda p, activate: DropoutMC(
            p=ue_args.inference_prob, activate=False
        )
    elif ue_args.dropout_type == "DPP":

        def dropout_ctor(p, activate):
            return DropoutDPP_v2(
                p=p,
                activate=activate,
                max_n=ue_args.dropout.max_n,
                max_frac=ue_args.dropout.max_frac,
                is_reused_mask=ue_args.dropout.is_reused_mask,
                mask_name_for_mask=ue_args.dropout.mask_name_for_mask,
            )

    elif ue_args.dropout_type == "DC_MC":
        dropout_ctor = lambda linear, activate: LinearDropConnectMC(
            linear=linear, p_dropconnect=ue_args.inference_prob, activate=activate
        )
    else:
        raise ValueError(f"Wrong dropout type: {ue_args.dropout_type}")

    if (ue_args.dropout_subs == "all") and (ue_args.dropout_type == "DC_MC"):
        convert_to_mc_dropconnect(model.longformer.encoder, {"Linear": dropout_ctor})
        hide_dropout(model.longformer.encoder)
    elif (ue_args.dropout_subs == "last") and (ue_args.dropout_type == "DC_MC"):
        set_last_dropconnect(model, dropout_ctor)
        # hide_dropout(model.electra.encoder)
        hide_dropout(model.classifier)

    elif ue_args.dropout_subs == "last":
        set_last_dropout(model, dropout_ctor(p=ue_args.inference_prob, activate=False))
    #         model.classifier.dropout1 = DropoutDPP(p=0.1,
    #                               activate=False,
    #                               max_n=100,
    #                               max_frac=0.1,
    #                               mask_name='ht_dpp')
    elif ue_args.dropout_subs == "all":
        # convert_to_mc_dropout(model, {'Dropout': dropout_ctor})
        convert_to_mc_dropout(model.longformer.encoder, {"Dropout": dropout_ctor})

    else:
        raise ValueError(f"Wrong ue args {ue_args.dropout_subs}")


def calculate_dropouts(model):
    res = 0
    for i, layer in enumerate(list(model.children())):
        module_name = list(model._modules.items())[i][0]
        layer_name = layer._get_name()
        if layer_name == "Dropout":
            res += 1
        else:
            res += calculate_dropouts(model=layer)

    return res


def freeze_all_dpp_dropouts(model, freeze):
    for layer in model.children():
        if isinstance(layer, DropoutDPP):
            if freeze:
                layer.mask.freeze(dry_run=True)
            else:
                layer.mask.unfreeze(dry_run=True)
        else:
            freeze_all_dpp_dropouts(model=layer, freeze=freeze)


def compute_metrics(is_regression, metric, p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()

    return result


def reset_params(model: torch.nn.Module):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        else:
            reset_params(model=layer)


def set_dpp_step_inference(model):
    if isinstance(model, ElectraForSequenceClassification):
        if isinstance(model.classifier, ElectraClassificationHeadCustom):
            model.classifier.dropout2.inference_step = 1
        else:
            model.classifier.dropout.inference_step = 1
    else:
        model.dropout.inference_step = 1


def change_mask(model):
    if isinstance(model, ElectraForSequenceClassification):
        if isinstance(model.classifier, ElectraClassificationHeadCustom):
            model.classifier.dropout2.change_mask = 1
        else:
            model.classifier.dropout.change_mask = 1
    else:
        model.dropout.change_mask = 1


def do_predict_eval(
    model,
    tokenizer,
    trainer,
    eval_dataset,
    train_dataset,
    metric,
    config,
    work_dir,
    model_dir,
    metric_fn,
    ood_dataset,
):
    if config.ue.use_cache:
        model.enable_cache()

    log.info("*** Evaluate ***")

    training_args = config.training

    true_labels = eval_dataset["label"]

    ue_args = config.ue
    use_sngp = "use_sngp" in ue_args.keys() and ue_args.use_sngp
    use_duq = "use_duq" in ue_args.keys() and ue_args.use_duq

    tagger = TextClassifier(
        model,
        tokenizer,
        training_args=training_args,
        trainer=trainer,
        use_sngp=use_sngp,
    )
    if use_sngp:
        if ue_args.calibrate:
            _, _, _, _ = tagger.predict(train_dataset, calibrate=True)
            preds, probs, logits, stds = tagger.predict(eval_dataset, calibrate=False)
            log.info(f"Calibration temperature = {tagger.temperature}")
            tagger.temperature = 1
        else:
            preds, probs, logits, stds = tagger.predict(eval_dataset)

        eval_score = metric.compute(predictions=preds, references=true_labels)
        log.info(f"Eval score: {eval_score}")

        preds, probs, logits, stds = tagger.predict(eval_dataset)

        eval_results = {}
        eval_results["true_labels"] = true_labels
        eval_results["probabilities"] = probs.tolist()
        eval_results["answers"] = preds.tolist()
        eval_results["logits"] = logits.tolist()
        eval_results["stds"] = stds.tolist()
        eval_results["sampled_probabilities"] = []
        eval_results["sampled_answers"] = []

        set_seed(config.seed)
        random.seed(config.seed)
        for i in tqdm(range(ue_args.committee_size)):
            logits = torch.normal(
                mean=torch.Tensor(eval_results["logits"]),
                std=torch.Tensor(eval_results["stds"]),
            )
            probs = torch.nn.functional.softmax(logits, dim=1).numpy()
            preds = np.argmax(probs, axis=1)
            eval_results["sampled_probabilities"].append(probs.tolist())
            eval_results["sampled_answers"].append(preds.tolist())

            if ue_args.eval_passes:
                eval_score = eval_metric.compute(
                    predictions=preds, references=true_labels
                )
                log.info(f"Eval score: {eval_score}")

        log.info("Done.")

    else:
        if ue_args.calibrate:
            _, _ = tagger.predict(train_dataset, calibrate=True)
            preds, probs = tagger.predict(eval_dataset, calibrate=False)
            log.info(f"Calibration temperature = {tagger.temperature}")
            tagger.temperature = 1
        else:
            preds, probs = tagger.predict(eval_dataset)

        eval_results = {}
        eval_results["true_labels"] = true_labels
        eval_results["probabilities"] = probs.tolist()
        eval_results["answers"] = preds.tolist()

        if not use_duq:

            eval_results["sampled_probabilities"] = []
            eval_results["sampled_answers"] = []

            log.info("******Perform stochastic inference...*******")

            # if ue_args.dropout_type != "DC_MC":
            #         log.info("Model before dropout replacement:")
            #         log.info(str(model))
            convert_dropouts(model, ue_args)
            #         log.info("Model after dropout replacement:")
            #         log.info(str(model))

            if ue_args.dropout_type == "DC_MC":
                activate_mc_dropconnect(
                    model, activate=True, random=ue_args.inference_prob
                )
            else:
                activate_mc_dropout(model, activate=True, random=ue_args.inference_prob)

            if ue_args.dropout_type == "DPP" or ue_args.dropout_type == "DPP_masks":
                log.info("**************Dry run********************")

                freeze_all_dpp_dropouts(model, freeze=True)

                dry_run_dataset = (
                    eval_dataset
                    if ue_args.dropout.dry_run_dataset == "eval"
                    else train_dataset
                )
                tagger.predict(dry_run_dataset)

                freeze_all_dpp_dropouts(model, freeze=False)

                log.info("Done.")

            log.info("****************Start runs**************")
            eval_metric = metric

            set_seed(config.seed)
            random.seed(config.seed)
            for i in tqdm(range(ue_args.committee_size)):
                if ue_args.calibrate:
                    _, _ = tagger.predict(train_dataset, calibrate=True)
                    preds, probs = tagger.predict(eval_dataset, calibrate=False)
                    log.info(f"Calibration temperature = {tagger.temperature}")
                    tagger.temperature = 1
                else:
                    preds, probs = tagger.predict(eval_dataset)

                eval_results["sampled_probabilities"].append(probs.tolist())
                eval_results["sampled_answers"].append(preds.tolist())

                if ue_args.eval_passes:
                    eval_score = eval_metric.compute(
                        predictions=preds, references=true_labels
                    )
                    log.info(f"Eval score: {eval_score}")

                log.info("Done.")

            if ue_args.use_ood_sampling and ue_args.dropout.is_reused_mask:
                log.info(
                    "****************Start runs with reused dpp masks on odd**************"
                )
                set_dpp_step_inference(model)
                eval_results["sampled_probabilities"] = []
                eval_results["sampled_answers"] = []

                model.classifier.dropout2.dry_run(sampling=False)
                set_seed(config.seed)
                random.seed(config.seed)

                for i in tqdm(range(model.classifier.dropout2.count_diverse_masks)):

                    change_mask(model)
                    if ue_args.calibrate:
                        _, _ = tagger.predict(train_dataset, calibrate=True)
                        preds, probs = tagger.predict(ood_dataset, calibrate=False)
                        log.info(f"Calibration temperature = {tagger.temperature}")
                        tagger.temperature = 1
                    else:
                        preds, probs = tagger.predict(ood_dataset, calibrate=False)

                    eval_results["sampled_probabilities"].append(probs.tolist())
                    eval_results["sampled_answers"].append(preds.tolist())

            elif ue_args.dropout.is_reused_mask:
                log.info(
                    "****************Start runs with reused dpp masks**************"
                )
                set_dpp_step_inference(model)
                eval_results["sampled_probabilities"] = []
                eval_results["sampled_answers"] = []

                model.classifier.dropout2.dry_run()
                set_seed(config.seed)
                random.seed(config.seed)

                for i in tqdm(range(model.classifier.dropout2.count_diverse_masks)):
                    change_mask(model)
                    if ue_args.calibrate:
                        _, _ = tagger.predict(train_dataset, calibrate=True)
                        preds, probs = tagger.predict(eval_dataset, calibrate=False)
                        log.info(f"Calibration temperature = {tagger.temperature}")
                        tagger.temperature = 1
                    else:
                        preds, probs = tagger.predict(eval_dataset)

                    eval_results["sampled_probabilities"].append(probs.tolist())
                    eval_results["sampled_answers"].append(preds.tolist())

                    if ue_args.eval_passes:
                        eval_score = eval_metric.compute(
                            predictions=preds, references=true_labels
                        )
                        log.info(f"Eval score: {eval_score}")

                log.info("Done.")

            if ue_args.use_ood_sampling:

                def probability_variance(sampled_probabilities):
                    mean_probabilities = np.mean(sampled_probabilities, axis=1)
                    mean_probabilities = np.expand_dims(mean_probabilities, axis=1)
                    return ((sampled_probabilities - mean_probabilities) ** 2).mean(1)

                ood_probs = np.asarray(eval_results["sampled_probabilities"])[:, :, 1]
                ood_probs_variance = probability_variance(ood_probs)
                diverse_idx = np.argsort(ood_probs_variance)[
                    -ue_args.dropout.committee_size :
                ]
                model.classifier.dropout2.diverse_masks = (
                    model.classifier.dropout2.diverse_masks[:, diverse_idx]
                )
                model.classifier.dropout2.count_diverse_masks = (
                    model.classifier.dropout2.diverse_masks.shape[1] - 1
                )
                model.classifier.dropout2.used_mask_id = 0
                print(f"\n\n diverse_idx {diverse_idx}\n\n")

                log.info(
                    "****************Start runs with diverse dpp masks**************"
                )
                set_dpp_step_inference(model)
                eval_results["sampled_probabilities"] = []
                eval_results["sampled_answers"] = []

                set_seed(config.seed)
                random.seed(config.seed)

                for i in tqdm(range(model.classifier.dropout2.count_diverse_masks)):
                    change_mask(model)
                    if ue_args.calibrate:
                        _, _ = tagger.predict(train_dataset, calibrate=True)
                        preds, probs = tagger.predict(eval_dataset, calibrate=False)
                        log.info(f"Calibration temperature = {tagger.temperature}")
                        tagger.temperature = 1
                    else:
                        preds, probs = tagger.predict(eval_dataset)

                    eval_results["sampled_probabilities"].append(probs.tolist())
                    eval_results["sampled_answers"].append(preds.tolist())

                    if ue_args.eval_passes:
                        eval_score = eval_metric.compute(
                            predictions=preds, references=true_labels
                        )
                        log.info(f"Eval score: {eval_score}")

                log.info("Done.")

            activate_mc_dropout(model, activate=False)
            activate_mc_dropconnect(model, activate=False)

    with open(Path(work_dir) / "dev_inference.json", "w") as res:
        json.dump(eval_results, res)

    if wandb.run is not None:
        wandb.save(str(Path(work_dir) / "dev_inference.json"))


def create_model(num_labels, model_args, data_args, ue_args, config):
    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=config.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=config.cache_dir,
    )
    use_sngp = "use_sngp" in ue_args.keys() and ue_args.use_sngp
    use_duq = "use_duq" in ue_args.keys() and ue_args.use_duq

    if ue_args.use_cache:
        if use_sngp:
            if "electra" in model_args.model_name_or_path:
                model = SNGPElectraForSequenceClassificationCached.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=False,
                    config=model_config,
                    cache_dir=config.cache_dir,
                    ue_config=ue_args.sngp,
                )
                log.info("Loaded ELECTRA with SNGP")
            else:  # load bert
                model = SNGPBertForSequenceClassificationCached.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=False,
                    config=model_config,
                    cache_dir=config.cache_dir,
                    ue_config=ue_args.sngp,
                )
        elif "electra" in model_args.model_name_or_path:  # TODO:
            model = ElectraForSequenceClassificationCached.from_pretrained(
                model_args.model_name_or_path,
                from_tf=False,
                config=model_config,
                cache_dir=config.cache_dir,
            )
            model.use_cache = True
            model.classifier = ElectraClassificationHeadCustom(model.classifier)
            log.info("Replaced ELECTRA's head")

        elif "bert" in model_args.model_name_or_path:
            model = BertForSequenceClassificationCached.from_pretrained(
                model_args.model_name_or_path,
                from_tf=False,
                config=model_config,
                cache_dir=config.cache_dir,
            )

        else:
            raise ValueError(
                f"Cannot use cache with this type of model: {model_args.model_name_or_path}"
            )

        model.disable_cache()

    else:
        if use_duq:
            log.info("Using ELECTRA DUQ model")
            model = ElectraForSequenceClassificationDUQ.from_pretrained(
                model_args.model_name_or_path,
                from_tf=False,
                config=model_config,
                cache_dir=config.cache_dir,
            )

            model.make_duq(
                output_dir=config.cache_dir,
                batch_size=config.training.per_device_train_batch_size,
                duq_params=config.ue.duq_params,
            )

        if "electra" in model_args.model_name_or_path:
            model = ElectraForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=False,
                config=model_config,
                cache_dir=config.cache_dir,
            )

            model.classifier = ElectraClassificationHeadCustom(model.classifier)
            log.info("Replaced ELECTRA's head")
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=False,
                config=model_config,
                cache_dir=config.cache_dir,
            )

    return model, tokenizer


def train_eval_glue_model(config, training_args, data_args, work_dir):
    ue_args = config.ue
    model_args = config.model

    log.info(f"Seed: {config.seed}")
    set_seed(config.seed)
    random.seed(config.seed)
    training_args.seed = config.training.seed
    log.info("Load dataset.")

    base_df = pd.read_csv(config.data.test_data_dir)
    base_df.body.fillna("", inplace=True)
    keep_mask = ~base_df.duplicated(subset=["body"]).values
    df_test = base_df.iloc[keep_mask, :]

    train_dataset = None  # Dataset.from_pandas(df_train)
    eval_dataset = Dataset.from_pandas(
        df_test.sample(frac=config.data.subsample_perc_test, random_state=config.seed)
    )

    log.info("Done with loading the dataset.")

    # Labels
    label_list = np.unique(eval_dataset["mkb_code"])
    label_list.sort()  # Let's sort it for determinism

    num_labels = len(label_list)

    label_to_id = joblib.load(model_args.label_to_id_dir)
    with open(model_args.config_dir, "r") as f:
        conf = yaml.safe_load(f)

    conf = update_config_symptom_checker(conf)

    estimator = models.Estimator(conf["top3"], conf["device"])
    model, tokenizer = estimator.evaluator, estimator.tokenizer

    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    padding = "max_length"
    max_seq_length = estimator.max_seq_len

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = {"input_ids": [], "token_type_ids": [], "attention_mask": []}

        for text in examples[sentence1_key]:
            encoded = estimator._encode(text)
            result["input_ids"].append(encoded[0].numpy().tolist())
            result["attention_mask"].append(encoded[1].numpy().tolist())
            result["token_type_ids"].append([0] * max_seq_length)

        label_column_name = (
            "original_mkb_code"
            if "original_mkb_code" in examples.keys()
            else "mkb_code"
        )
        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and label_column_name in examples:
            result["label"] = [
                (label_to_id[l] if l != -1 else -1) for l in examples[label_column_name]
            ]
        return result

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        num_proc=1,
    )

    # train_dataset = train_dataset.map(
    #    preprocess_function,
    #    batched=True,
    #    load_from_cache_file=not data_args.overwrite_cache,
    #    num_proc=16
    # )

    eval_dataset.set_format(
        columns=["attention_mask", "input_ids", "token_type_ids", "label"]
    )
    # train_dataset.set_format(columns=['attention_mask', 'input_ids', 'token_type_ids', 'label'])

    if config.reset_params:
        reset_params(model)

    train_dataset = None
    train_indexes = None

    if config.do_train or (
        config.ue.dropout_type == "DPP" and config.ue.dropout.dry_run_dataset != "eval"
    ):
        train_dataset = datasets["train"]
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

    log.info("Load out-of-domain dataset.")
    datasets_ood = load_dataset(
        "imdb", ignore_verifications=True, cache_dir=config.cache_dir
    )
    log.info("Done with loading the dataset.")

    label_list_ood = datasets_ood["train"].features["label"].names
    num_labels = len(label_list_ood)

    label_to_id_ood = {v: i for i, v in enumerate(label_list_ood)}
    sentence1_key, sentence2_key = ("text", None)

    ood_dataset = (
        datasets_ood["test"]
        .select(list(range(1000)))
        .map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    )

    metric = load_metric("accuracy", keep_in_memory=True, cache_dir=config.cache_dir)

    is_regression = False
    metric_fn = lambda p: compute_metrics(is_regression, metric, p)

    training_args.save_steps = 0
    if config.do_train:
        training_args.warmup_steps = int(
            0.1
            * len(train_dataset)
            * training_args.num_train_epochs
            / training_args.train_batch_size
        )
        log.info(f"Warmup steps: {training_args.warmup_steps}")

    training_args.logging_steps = 50  # DEBUG

    use_sngp = "use_sngp" in config.ue.keys() and config.ue.use_sngp
    if use_sngp:
        trainer = SNGPTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=metric_fn,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=metric_fn,
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
            train_dataset,
            metric,
            config,
            work_dir,
            model_args.model_name_or_path,
            metric_fn,
            ood_dataset,
        )


def update_config(cfg_old, cfg_new):
    for k, v in cfg_new.items():
        if k in cfg_old.__dict__:
            setattr(cfg_old, k, v)

    return cfg_old


def update_config_symptom_checker(conf, device="cuda"):
    for key in conf.keys():
        if type(conf[key]) is dict:
            for sub_key in conf[key]:
                if "path" in sub_key:
                    conf[key][sub_key] = (
                        "/home/avazhentsev/data/tmp" + conf[key][sub_key][2:]
                    )
        elif "path" in key:
            conf[key] = "/home/avazhentsev/data/tmp" + conf[key][2:]
    conf["device"] = device
    return conf


@hydra.main(
    config_path=os.environ["HYDRA_CONFIG_PATH"],
    config_name=os.environ["HYDRA_CONFIG_NAME"],
)
def main(config):
    os.environ["WANDB_WATCH"] = "False"  # To disable Huggingface logging
    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    os.chdir(hydra.utils.get_original_cwd())

    wandb_run = init_wandb(auto_generated_dir, config)

    args_train = TrainingArguments(output_dir=auto_generated_dir)
    args_train = update_config(args_train, config.training)

    args_data = DataTrainingArguments(
        task_name=config.data.task_name  # , data_dir=config.data.data_dir
    )
    args_data = update_config(args_data, config.data)

    train_eval_glue_model(config, args_train, args_data, auto_generated_dir)


if __name__ == "__main__":
    main()
