from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional

from transformers.file_utils import add_start_docstrings
from transformers import Trainer
from ue4nlp.transformers_regularized import (
    SelectiveTrainer,
    SNGPTrainer,
    SelectiveSNGPTrainer,
)


def get_trainer(
    task: str,  # "cls" or "ner"
    use_selective: bool,
    use_sngp: bool,
    model,
    training_args,
    train_dataset,
    eval_dataset,
    metric_fn,
    data_collator=None,
) -> "Trainer":
    training_args.save_total_limit = 1
    training_args.save_steps = 1e5
    training_args.task = task
    if not use_selective and not use_sngp:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=metric_fn,
            data_collator=data_collator,
        )
    elif use_sngp:
        if use_selective:
            trainer = SelectiveSNGPTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=metric_fn,
            )
        else:
            trainer = SNGPTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=metric_fn,
                data_collator=data_collator,
            )
    elif use_selective:
        trainer = SelectiveTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=metric_fn,
        )
    return trainer


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class TrainingArgsWithLossCoefs(TrainingArguments):
    """
    reg_type (:obj:`str`, `optional`, defaults to :obj:`reg-curr`):
        Type of regularization.
    lamb (:obj:`float`, `optional`, defaults to :obj:`0.01`):
        lambda value for regularization.
    margin (:obj:`float`, `optional`, defaults to :obj:`0.01`):
        margin value for metric loss.
    """
    reg_type: Optional[str] = field(
        default="reg-curr", metadata={"help": "Type of regularization."}
    )
    lamb: Optional[float] = field(
        default=0.01, metadata={"help": "lambda value for regularization."}
    )
    margin: Optional[float] = field(
        default=0.05, metadata={"help": "margin value for metric loss."}
    )
    lamb_intra: Optional[float] = field(
        default=0.05, metadata={"help": "lambda intra value for metric loss."}
    )
    unc_threshold: Optional[float] = field(
        default=0.5, metadata={"help": "unc_threshold value for RAU loss."}
    )


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class TrainingArgsWithMSDCoefs(TrainingArguments):
    """
    mixup (:obj:`bool`, `optional`, defaults to :obj:`True`):
        Use mixup or not.
    self_ensembling (:obj:`bool`, `optional`, defaults to :obj:`True`):
        Use self-ensembling or not.
    omega (:obj:`float`, `optional`, defaults to :obj:`1.0`):
        mixup sampling coefficient.
    lam1 (:obj:`float`, `optional`, defaults to :obj:`1.0`):
        lambda_1 value for regularization.
    lam2 (:obj:`float`, `optional`, defaults to :obj:`0.01`):
        lambda_2 value for regularization.
    """

    mixup: Optional[bool] = field(default=True, metadata={"help": "Use mixup or not."})
    self_ensembling: Optional[bool] = field(
        default=True, metadata={"help": "Use self-ensembling or not."}
    )
    omega: Optional[float] = field(
        default=1.0, metadata={"help": "mixup sampling coefficient."}
    )
    lam1: Optional[float] = field(
        default=1.0, metadata={"help": "lambda_1 value for regularization."}
    )
    lam2: Optional[float] = field(
        default=0.01, metadata={"help": "lambda_2 value for regularization."}
    )
