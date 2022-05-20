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
)
from datasets import load_metric, load_dataset, Dataset
from sklearn.model_selection import train_test_split

from utils.utils_ue_estimator import create_ue_estimator
from utils.utils_dropout import set_last_dropout, get_last_dropout, set_last_dropconnect
from utils.utils_data import preprocess_function, load_data_ood
from utils.utils_models import create_model, create_tokenizer
from utils.hyperparameter_search import get_optimal_hyperparameters
from utils.utils_train import TrainingArgsWithLossCoefs
from utils.utils_train import get_trainer

from ue4nlp.transformers_regularized import (
    SelectiveTrainer, SelectiveSNGPTrainer
)

import logging

log = logging.getLogger(__name__)

task_to_keys = {
    "clinc_oos": ("text", None),
    "rostd": ("text", None),
    "snips": ("text", None),
    "rostd_coarse": ("text", None),
    "sst2": ("text", None),
    "20newsgroups": ("text", None),
    "amazon": ("text", None),
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
        convert_to_mc_dropconnect(model.electra.encoder, {"Linear": dropout_ctor})
        hide_dropout(model.electra.encoder)
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
        convert_to_mc_dropout(model.electra.encoder, {"Dropout": dropout_ctor})

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
    split,
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

        eval_dataset_without_labels = eval_dataset.remove_columns("label")
        res = cls.predict(eval_dataset_without_labels)
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

        if "mc" in config.ue.ue_type or "sngp" in config.ue.ue_type:
            X_test = eval_dataset_without_labels
        else:
            X_test = eval_dataset
            
        ue_estimator.fit_ue(X=train_dataset, X_test=X_test)
        ue_results = ue_estimator(eval_dataset, true_labels)
        eval_results.update(ue_results)

    with open(Path(work_dir) / f"dev_inference{split}.json", "w") as res:
        json.dump(eval_results, res)

    if wandb.run is not None:
        wandb.save(str(Path(work_dir) / f"dev_inference{split}.json"))


def train_eval_oos_model(config, training_args, data_args, work_dir, split=""):
    ue_args = config.ue
    model_args = config.model

    log.info(f"Seed: {config.seed}")
    set_seed(config.seed)
    random.seed(config.seed)
    training_args.seed = config.training.seed

    #load tokenizer for creating ood datasets correctly
    #it is necessary when we use different datasets with different task_to_keys options
    if config.data.task_name not in ['rostd', 'snips', 'clinc', 'rostd_coarse']:
        tokenizer = create_tokenizer(model_args, config)
    else:
        tokenizer = None
        
    log.info("Load dataset.")
    datasets = load_data_ood(
        config.data.task_name, config, data_args, config.data.task_subset, split, tokenizer
    )

    label_list = datasets["train"].unique("label")
    label_list.sort()  # Let's sort it for determinism

    num_labels = len(label_list)

    model, tokenizer = create_model(num_labels, model_args, data_args, ue_args, config)

    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    padding = "max_length"
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
        None, sentence1_key, sentence2_key, tokenizer, max_seq_length, examples
    )

    if 'input_ids' not in datasets.column_names['train']:
        datasets = datasets.map(
            f_preprocess,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if config.reset_params:
        reset_params(model)

    if ue_args.dropout_type == "DC_MC":
        convert_dropouts(model, ue_args)

    train_dataset = None
    train_indexes = None
    if (
        config.do_train
        or config.ue.calibrate
        or config.ue.ue_type in ["maha", "nuq", "l-nuq", "l-maha", "mc_maha", "msd", "ddu", "decomposing_md"]
        or (
        config.ue.dropout_type == "DPP" and config.ue.dropout.dry_run_dataset != "eval"
        )
    ):
        train_dataset = datasets["train"]
        train_indexes = list(range(len(train_dataset)))

        if config.data.subsample_perc > 0:
            train_indexes = random.sample(
                train_indexes, int(len(train_dataset) * config.data.subsample_perc)
            )
            
        ################### Replace Labels with noise ####################################
        noise_perc = 0 if 'noise_perc' not in config.data.keys() else config.data.noise_perc
        if noise_perc > 0: 
            labels = np.asarray(train_dataset['label'])
            all_labels = np.sort(np.unique(labels))
            n_items = len(train_indexes)
            noisy_idx = np.random.choice(train_indexes, int(n_items*noise_perc))
            new_labels = np.array([np.random.choice(np.delete(all_labels, l), 1) for l in labels[noisy_idx]]).flatten()
            labels[noisy_idx] = new_labels
            train_dataset_dict = {}
            for k in train_dataset.features:
                if k == 'label':
                    train_dataset_dict[k] = list(labels)
                else:
                    train_dataset_dict[k] = train_dataset[k]
            train_dataset = Dataset.from_dict(train_dataset_dict)
            log.info(f"Replace labels for {len(noisy_idx)} items.")
            
        train_dataset = train_dataset.select(
            train_indexes
        ).flatten_indices()  # torch.utils.data.Subset(train_dataset, train_indexes)

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

        train_dataset = train_dataset.select(
            train_indexes
        ).flatten_indices()  # torch.utils.data.Subset(train_dataset, train_indexes)
        log.info(f"Training dataset size: {len(train_dataset)}")

    eval_dataset = datasets[config.data.task_test] if config.do_eval else None

    #######
    if config.data.task_name == 'clinc_oos':
        eval_dataset_id = datasets["validation"].filter(lambda x: x["label"] != 42)

        def map_classes(examples):
            examples["label"] = (
                examples["label"] if (examples["label"] < 42) else examples["label"] - 1
            )
            return examples

        eval_dataset_id = eval_dataset_id.map(
            map_classes,
            batched=False,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        if 'validation' in datasets.column_names.keys():
            ood_label = len(np.unique(datasets['validation']['label']))
            eval_dataset_id = datasets["validation"].filter(lambda x: x["label"] != ood_label)
        else:
            eval_dataset_id = datasets["train"]
           
    ######

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

    use_sngp = config.ue.ue_type == "sngp"
    use_selective = "use_selective" in ue_args.keys() and ue_args.use_selective

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
        trainer.save_model(work_dir)
        tokenizer.save_pretrained(work_dir)

    if config.do_eval:
        do_predict_eval(
            trainer.model,
            tokenizer,
            trainer,
            eval_dataset,
            train_dataset,
            eval_dataset_id,
            metric,
            config,
            work_dir,
            model_args.model_name_or_path,
            metric_fn,
            max_seq_length,
            split,
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
    
    if config.ue.ue_type == "l-maha" or config.ue.ue_type == "l-nuq":
        config.ue.use_cache = False


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

    args_train = TrainingArgsWithLossCoefs(
        output_dir=auto_generated_dir,
        reg_type=config.ue.get("reg_type", "reg-curr"),
        lamb=config.ue.get("lamb", 0.01),
        margin=config.ue.get("margin", 0.05),
        lamb_intra=config.ue.get("lamb_intra", 0.01),
        unc_threshold=config.ue.get("unc_threshold", 0.5),
    )
    args_train = update_config(args_train, config.training)

    args_data = DataTrainingArguments(
        task_name=config.data.task_name  # , data_dir=config.data.data_dir
    )

    args_data = update_config(args_data, config.data)

    if config.do_train and not config.do_eval:
        filename = "pytorch_model.bin"
    else:
        filename = "dev_inference.json" 
        
    if config.data.task_name == "snips":
        for split in range(config.data.n_splits):
            if not os.path.exists(
                Path(auto_generated_dir) / f"dev_inference{split}.json"
            ):
                train_eval_oos_model(
                    config, args_train, args_data, auto_generated_dir, split
                )
            else:
                log.info(
                    f"Result file: {auto_generated_dir}/dev_inference{split}.json already exists \n"
                )
    else:
        if not os.path.exists(Path(auto_generated_dir) / filename):
            train_eval_oos_model(config, args_train, args_data, auto_generated_dir)
        else:
            log.info(
                f"Result file: {auto_generated_dir}/{filename} already exists \n"
            )


if __name__ == "__main__":
    main()
